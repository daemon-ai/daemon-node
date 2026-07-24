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
    seed_corpus_fs, spawn_node_with, spawn_node_with_budget, start_cluster_membership,
    start_cluster_on, wait_rounds, NodeSpec,
};

const RUN: &str = "acceptance-iroh-dual-plane";
/// The fleet-shaped colocation run: a coordinator+trainer box meshing with a remote trainer.
const RUN_COLO: &str = "acceptance-iroh-colocated-fleet";
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

/// The ceremony fleet's real Strix shape: a **co-located coordinator+trainer box with a PINNED
/// bind port** meshing over iroh with a **remote** trainer box (its own pinned port).
///
/// The gate above never colocates roles, so it could not see the co-located bind collision; the
/// single-host colocation gate (`single_node_coordinator_trainer.rs`) pins the port but has no
/// remote peer, so it cannot show that the WS-only co-located sibling still trains ALONGSIDE a
/// real iroh peer. This one asserts both halves of the resolution end-to-end:
///
/// - the seat instance owns the box's single iroh endpoint on its pinned port and forms a real QUIC
///   gossip connection with the remote node (`NeighborUp` on both boxes),
/// - the co-located trainer, which shares that endpoint and attaches WS-only, still completes
///   rounds and agrees with the remote trainer's per-round det digests byte-for-byte.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_co_located_box_meshes_with_a_remote_peer_on_one_pinned_port_each() {
    let _serial = harness::serial_guard();
    let payload_root = tempfile::tempdir().expect("shared payload root");

    let base_port = harness::free_port();
    let base_url = format!("http://127.0.0.1:{base_port}/api/v1/vhc");

    // One pinned UDP port PER BOX (the runbook §9.1 per-box pin), not a node-picked ephemeral one.
    let colo_port = harness::free_udp_port();
    let remote_port = harness::free_udp_port();
    let colo_toml =
        format!("coordinator_trains = true\n[vhc.iroh]\nenabled = true\nbind_port = {colo_port}\n");
    let remote_toml = format!("[vhc.iroh]\nenabled = true\nbind_port = {remote_port}\n");

    // The co-located box runs the FINITE default duty ledger (coordinator 0% + trainer 100%) —
    // the same arbitration the live boxes ran.
    let colo = spawn_node_with_budget(
        &NodeSpec {
            name: "iroh-colo-box",
            registry_base: &base_url,
            seat_claim: true,
            payload_dir: Some(payload_root.path()),
            allowlist: &base_url,
            reconcile_tick_ms: 500,
            initial_backoff_ms: 0,
        },
        &colo_toml,
        "duty_pct = 100\nhost_ram_mb = 131072\n\n[vhc.owner_budget.device_memory_mb]\n\"gpu:0\" = 131072\n",
    );
    let remote = spawn_node_with(
        &NodeSpec {
            name: "iroh-remote-trainer",
            registry_base: &base_url,
            seat_claim: false,
            payload_dir: Some(payload_root.path()),
            allowlist: &base_url,
            reconcile_tick_ms: 500,
            initial_backoff_ms: 0,
        },
        &remote_toml,
    );

    let bases = [base_peer(&colo), base_peer(&remote)];
    // Two trainers meet the floor: the co-located one and the remote one.
    let cluster = start_cluster_membership(base_port, RUN_COLO, &bases, 0, 2, 2, 2).await;
    seed_corpus_fs(payload_root.path(), RUN_COLO, &cluster.genesis);

    // The seat box joins first (it must hold the seat before its sibling trains).
    join(&colo, RUN_COLO, "op-colo").await;
    join(&remote, RUN_COLO, "op-remote").await;

    let rounds = 2u64;
    let timeout = Duration::from_secs(300);
    wait_rounds(&remote, RUN_COLO, rounds, timeout).await;
    wait_rounds(&colo, RUN_COLO, rounds, timeout).await;

    // The mesh really formed between the two boxes (the sufficiency signal — WS alone converges).
    for node in ["iroh-colo-box", "iroh-remote-trainer"] {
        assert!(
            node_log_contains(node, NEIGHBOR_MARKER),
            "node `{node}` never logged `{NEIGHBOR_MARKER}` — the iroh mesh did not form"
        );
    }
    // The co-located sibling shared the box's single endpoint instead of re-binding its port.
    assert!(
        node_log_contains(
            "iroh-colo-box",
            "co-located role-instance shares this node's single iroh endpoint"
        ),
        "the co-located trainer must share the box's endpoint"
    );
    for marker in ["endpoint bind failed", "Failed to bind sockets"] {
        assert!(
            !node_log_contains("iroh-colo-box", marker),
            "an iroh bind failure (`{marker}`) surfaced on the co-located box"
        );
    }

    leave(
        &remote,
        RUN_COLO,
        daemon_api::VhcLeaveMode::Graceful,
        "op-lr",
    )
    .await;
    leave(&colo, RUN_COLO, daemon_api::VhcLeaveMode::Graceful, "op-lc").await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // The WS-only co-located trainer and the dual-plane remote trainer agree round for round.
    let colocated = journal_digests(&colo, RUN_COLO);
    let remote_digests = journal_digests(&remote, RUN_COLO);
    assert_digests_agree(&colocated, &remote_digests, rounds as usize);

    drop(cluster);
}
