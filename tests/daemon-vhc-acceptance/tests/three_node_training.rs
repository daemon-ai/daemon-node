// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! G1 — the baseline decentralized round-trip: a seat-claiming coordinator-role node and two
//! trainer nodes, all real `daemon` processes over Unix sockets, WS control plane (the promoted
//! relay-grade coordinator the seat holder publishes on) + the shared filesystem payload plane.
//! The run trains N barrier rounds; the two trainers' durable journals must agree on the tag-4
//! det digest for every shared round (the offline product-path digest oracle).

mod harness;

use std::time::Duration;

use harness::{
    assert_digests_agree, base_peer, join, journal_digests, leave, seed_corpus_fs, spawn_node,
    start_cluster_on, wait_rounds, NodeSpec,
};

const RUN: &str = "acceptance-baseline";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_node_training_converges_with_agreeing_digests() {
    let _serial = harness::serial_guard();
    // A shared filesystem payload root every node serves content-addressed objects over (the
    // local stand-in for a shared object store).
    let payload_root = tempfile::tempdir().expect("shared payload root");

    // The registry binds on a pre-chosen port, so every node is configured with the base URL
    // before the fixture comes up. Discovery/seat are lazy (first exercised at `join`, below,
    // after the cluster is serving), so booting the nodes first is safe.
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

    // The coordinator node claims the seat + launches the coordinator role; the trainers join.
    join(&coord, RUN, "op-coord").await;
    join(&trainer_a, RUN, "op-a").await;
    join(&trainer_b, RUN, "op-b").await;

    // Drive to a handful of rounds on both trainers.
    let rounds = 3u64;
    let timeout = Duration::from_secs(180);
    wait_rounds(&trainer_a, RUN, rounds, timeout).await;
    wait_rounds(&trainer_b, RUN, rounds, timeout).await;

    // Graceful leave (drains + settles the journals).
    leave(&trainer_a, RUN, daemon_api::VhcLeaveMode::Graceful, "op-la").await;
    leave(&trainer_b, RUN, daemon_api::VhcLeaveMode::Graceful, "op-lb").await;
    leave(&coord, RUN, daemon_api::VhcLeaveMode::Graceful, "op-lc").await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // The offline product-path oracle: both trainers' durable journals agree on the tag-4 det
    // digest for every shared round.
    let a = journal_digests(&trainer_a, RUN);
    let b = journal_digests(&trainer_b, RUN);
    assert_digests_agree(&a, &b, rounds as usize);

    drop(cluster);
}
