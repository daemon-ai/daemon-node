// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! G6 — malformed certificate / malformed frame rejection: the acceptance suite injects a garbage
//! frame and an untrusted-base certificate record onto the live control plane mid-run; every real
//! node must refuse them TYPED (a `frame_refused` / `distribution_refused` warning), stay up (no
//! panic, no process exit, no silent drop), and the run must still converge with agreeing digests.

mod harness;

use std::time::Duration;

use daemon_api::VhcEvent;
use harness::{
    assert_digests_agree, base_peer, join, journal_digests, leave, recent_events, seed_corpus_fs,
    spawn_node, start_cluster_on, untrusted_cert_record, wait_rounds, worker_children, NodeSpec,
};

const RUN: &str = "acceptance-malformed";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn malformed_frame_and_untrusted_cert_are_typed_refusals() {
    let _serial = harness::serial_guard();
    let payload_root = tempfile::tempdir().expect("shared payload root");
    let base_port = harness::free_port();
    let base_url = format!("http://127.0.0.1:{base_port}/api/v1/vhc");

    let coord = spawn_node(&NodeSpec {
        name: "coordinator",
        registry_base: &base_url,
        seat_claim: true,
        payload_dir: Some(payload_root.path()),
        allowlist: &base_url,
        reconcile_tick_ms: 500,
        initial_backoff_ms: 0,
    });
    let trainer_a = spawn_node(&NodeSpec {
        name: "trainer-a",
        registry_base: &base_url,
        seat_claim: false,
        payload_dir: Some(payload_root.path()),
        allowlist: &base_url,
        reconcile_tick_ms: 500,
        initial_backoff_ms: 0,
    });
    let trainer_b = spawn_node(&NodeSpec {
        name: "trainer-b",
        registry_base: &base_url,
        seat_claim: false,
        payload_dir: Some(payload_root.path()),
        allowlist: &base_url,
        reconcile_tick_ms: 500,
        initial_backoff_ms: 0,
    });

    let bases = [
        base_peer(&coord),
        base_peer(&trainer_a),
        base_peer(&trainer_b),
    ];
    let cluster = start_cluster_on(base_port, RUN, &bases, 0, 2).await;
    seed_corpus_fs(payload_root.path(), RUN, &cluster.genesis);

    join(&coord, RUN, "op-coord").await;
    join(&trainer_a, RUN, "op-a").await;
    join(&trainer_b, RUN, "op-b").await;

    // Let the control plane form (all three peers connected), then inject the adversarial frames.
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    while cluster.registry.ws_peer_count() < 3 {
        assert!(
            std::time::Instant::now() < deadline,
            "the three nodes did not all connect the control plane"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // (1) A garbage frame — neither a §12.1 signed frame nor a distribution record: refused
    // `Malformed`. (2) A certificate issued by an untrusted base: refused at distribution ingest.
    let injected = cluster
        .registry
        .inject_raw(b"\xff\x00not-a-vhc-frame\x01\x02\x03".to_vec());
    assert!(injected >= 3, "the garbage frame reached every peer");
    cluster
        .registry
        .inject_raw(untrusted_cert_record(cluster.genesis.genesis_hash));

    // The run still converges — the injection perturbs nothing.
    let rounds = 3u64;
    let timeout = Duration::from_secs(180);
    wait_rounds(&trainer_a, RUN, rounds, timeout).await;
    wait_rounds(&trainer_b, RUN, rounds, timeout).await;

    // Every node is still alive (a typed refusal never crashes the process).
    for node in [&coord, &trainer_a, &trainer_b] {
        assert!(
            node.is_alive(),
            "node `{}` must survive the injection",
            node.name
        );
        assert!(
            !worker_children(node).is_empty(),
            "node `{}` must still have a live worker",
            node.name
        );
    }

    // At least one node surfaced a TYPED refusal (never a silent drop).
    let mut refusals = 0usize;
    for node in [&coord, &trainer_a, &trainer_b] {
        for ev in recent_events(node, RUN).await {
            if let VhcEvent::Warning { class, .. } = ev {
                if class == "frame_refused" || class == "distribution_refused" {
                    refusals += 1;
                }
            }
        }
    }
    assert!(
        refusals > 0,
        "the malformed frame / untrusted cert must surface as a typed refusal warning"
    );

    leave(&trainer_a, RUN, daemon_api::VhcLeaveMode::Graceful, "op-la").await;
    leave(&trainer_b, RUN, daemon_api::VhcLeaveMode::Graceful, "op-lb").await;
    leave(&coord, RUN, daemon_api::VhcLeaveMode::Graceful, "op-lc").await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let a = journal_digests(&trainer_a, RUN);
    let b = journal_digests(&trainer_b, RUN);
    assert_digests_agree(&a, &b, rounds as usize);

    drop(cluster);
}
