// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! G3 — training convergence over the iroh/DualPlane transport: the same three-real-process
//! topology as the baseline gate (a seat-claiming coordinator-role node + two trainer nodes over
//! Unix sockets), with `[vhc.iroh] enabled = true` on every node, so each node
//!
//! 1. publishes its signed iroh roster record (endpoint id + pinned loopback `ip:port`) to the
//!    registry fixture,
//! 2. fetches + node-side-verifies the run's roster (signature, certificate chain to the
//!    genesis-trusted bases, freshness precedence), and
//! 3. authors dual-plane credentials — the worker composes `DualPlane(WS, IrohGossip)` with
//!    direct loopback dialing (relay mode disabled; the registry-served roster IS discovery).
//!
//! Convergence alone is NECESSARY but not SUFFICIENT (`DualPlane::publish` succeeds when either
//! plane accepts), so the gate also asserts the iroh mesh-formation marker per node: the
//! `NeighborUp` log line is emitted only when a real iroh QUIC gossip connection formed — the
//! black-box proof the second transport actually carries.

mod harness;

use std::time::Duration;

use harness::{
    assert_digests_agree, base_peer, join, journal_digests, leave, node_log_contains,
    seed_corpus_fs, spawn_node_with, start_cluster_on, wait_rounds, NodeSpec,
};

const RUN: &str = "acceptance-iroh-dual-plane";
/// Enable the iroh plane with the loopback defaults (node-picked free bind port, loopback
/// advertisement, no relays — direct dialing from the verified roster addresses).
const IROH_TOML: &str = "[vhc.iroh]\nenabled = true\n";
/// The mesh-formation marker `daemon-vhc-net`'s gossip plane logs on a formed QUIC connection.
const NEIGHBOR_MARKER: &str = "iroh gossip neighbor up";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_node_training_converges_over_the_dual_plane() {
    let _serial = harness::serial_guard();
    let payload_root = tempfile::tempdir().expect("shared payload root");

    let base_port = harness::free_port();
    let base_url = format!("http://127.0.0.1:{base_port}/api/v1/vhc");

    let coord = spawn_node_with(
        &NodeSpec {
            name: "iroh-coordinator",
            registry_base: &base_url,
            seat_claim: true,
            payload_dir: Some(payload_root.path()),
            allowlist: &base_url,
            reconcile_tick_ms: 500,
            initial_backoff_ms: 0,
        },
        IROH_TOML,
    );
    let trainer_a = spawn_node_with(
        &NodeSpec {
            name: "iroh-trainer-a",
            registry_base: &base_url,
            seat_claim: false,
            payload_dir: Some(payload_root.path()),
            allowlist: &base_url,
            reconcile_tick_ms: 500,
            initial_backoff_ms: 0,
        },
        IROH_TOML,
    );
    let trainer_b = spawn_node_with(
        &NodeSpec {
            name: "iroh-trainer-b",
            registry_base: &base_url,
            seat_claim: false,
            payload_dir: Some(payload_root.path()),
            allowlist: &base_url,
            reconcile_tick_ms: 500,
            initial_backoff_ms: 0,
        },
        IROH_TOML,
    );

    let bases = [
        base_peer(&coord),
        base_peer(&trainer_a),
        base_peer(&trainer_b),
    ];
    let cluster = start_cluster_on(base_port, RUN, &bases, 0, 2).await;
    seed_corpus_fs(payload_root.path(), RUN, &cluster.genesis);

    // Sequential joins (the baseline shape): each node publishes its roster record at join
    // authoring, so later joiners see every earlier record and dial INTO the mesh — the earlier
    // peers need no update (HyParView membership propagates the join).
    join(&coord, RUN, "op-coord").await;
    join(&trainer_a, RUN, "op-a").await;
    join(&trainer_b, RUN, "op-b").await;

    // Drive to a handful of rounds on both trainers over the dual plane.
    let rounds = 3u64;
    let timeout = Duration::from_secs(180);
    wait_rounds(&trainer_a, RUN, rounds, timeout).await;
    wait_rounds(&trainer_b, RUN, rounds, timeout).await;

    // The SUFFICIENCY signal: every node's worker formed at least one real iroh QUIC gossip
    // connection (the NeighborUp marker) — convergence with WS still up cannot prove the second
    // plane by itself.
    for node in ["iroh-coordinator", "iroh-trainer-a", "iroh-trainer-b"] {
        assert!(
            node_log_contains(node, NEIGHBOR_MARKER),
            "node `{node}` never logged `{NEIGHBOR_MARKER}` — the iroh mesh did not form"
        );
    }

    // Graceful leave (drains + settles the journals), then the offline product-path oracle.
    leave(&trainer_a, RUN, daemon_api::VhcLeaveMode::Graceful, "op-la").await;
    leave(&trainer_b, RUN, daemon_api::VhcLeaveMode::Graceful, "op-lb").await;
    leave(&coord, RUN, daemon_api::VhcLeaveMode::Graceful, "op-lc").await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let a = journal_digests(&trainer_a, RUN);
    let b = journal_digests(&trainer_b, RUN);
    assert_digests_agree(&a, &b, rounds as usize);

    drop(cluster);
}
