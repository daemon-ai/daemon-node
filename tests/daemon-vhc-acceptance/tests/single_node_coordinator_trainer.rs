// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! Defect D — a single-peer self-coordinated run must actually train. ONE real `daemon` process
//! that is BOTH the coordinator seat AND the run's only trainer (`seat_claim = true` +
//! `coordinator_trains = true`, `min_peers = max_peers = 1`). Pre-fix, a seat-claiming node
//! brought up ONLY the coordinator role, so its own trainer never joined: the run parked at
//! `peers = 0 < min_peers = 1` with 0 rounds. With the co-located trainer role-instance the box
//! meets its own membership floor and completes real rounds, whose per-round det digests are
//! observable on the PRODUCT path (wire v44 `last_round_digest`) and byte-equal to the offline
//! journal oracle.
//!
//! This is the ceremony's Strix shape in miniature (trainer + coordinator seat) and the exact gap
//! the three-node baseline never caught — there the coordinator node is coordinator-ONLY (its
//! two SEPARATE trainer nodes meet the floor).

mod harness;

use std::time::Duration;

use harness::{
    assert_api_digests_cover_and_match_journal, base_peer, collect_api_digests, join,
    journal_digests, leave, seed_corpus_fs, spawn_node_with_budget, start_cluster_membership,
    wait_rounds, NodeSpec,
};

const RUN: &str = "acceptance-single-peer";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn single_node_coordinator_and_trainer_completes_rounds() {
    let _serial = harness::serial_guard();
    // The shared filesystem payload plane (the single-host stand-in for a shared object store);
    // the one node serves + fetches its own content-addressed objects over it.
    let payload_root = tempfile::tempdir().expect("shared payload root");

    let base_port = harness::free_port();
    let base_url = format!("http://127.0.0.1:{base_port}/api/v1/vhc");

    // ONE box: coordinator seat + its own trainer. `coordinator_trains` opts a seat-holding node
    // into ALSO running a trainer role-instance for the same run (defect D).
    //
    // Crucially this gate runs against a FINITE owner budget with exactly ONE full accelerator-duty
    // (`duty_pct = 100`) — the SAME ledger the shipped node derives by default (`from_config`),
    // NOT the harness's permissive `unbounded = true`. That is what makes this gate exercise the
    // real owner-arbitration path the live M4/Windows boxes ran: the seat-holding node must admit
    // BOTH its coordinator seat instance (which claims zero accelerator duty — consensus only) AND
    // its co-located trainer (full duty) under a single 100% duty ledger. Pre-fix the coordinator
    // claimed the full 100% duty and starved its own trainer (`duty cycle exhausted: requested
    // 100%, remaining 0%` → peers=0, 0 rounds), so this gate would fail; the device/host ledgers
    // are sized large so DUTY is the exercised constraint (the arbitration honesty gap the
    // fleet-smoke exposed — the gate previously ran unbounded and never touched the duty ledger).
    let node = spawn_node_with_budget(
        &NodeSpec {
            name: "single-peer",
            registry_base: &base_url,
            seat_claim: true,
            payload_dir: Some(payload_root.path()),
            allowlist: &base_url,
            reconcile_tick_ms: 500,
            initial_backoff_ms: 0,
        },
        "coordinator_trains = true\n",
        "duty_pct = 100\nhost_ram_mb = 131072\n\n[vhc.owner_budget.device_memory_mb]\n\"gpu:0\" = 131072\n",
    );

    let bases = [base_peer(&node)];
    // min = max = 1: the box is its own coordinator and its sole trainer.
    let cluster = start_cluster_membership(base_port, RUN, &bases, 0, 2, 1, 1).await;
    seed_corpus_fs(payload_root.path(), RUN, &cluster.genesis);

    // A single API join: the node claims the coordinator seat, launches the coordinator role, AND
    // (the fix) brings up its co-located trainer, which joins the coordinator like any peer.
    join(&node, RUN, "op-single").await;

    // The self-coordinated run must actually progress: rounds complete end-to-end (pre-fix: 0).
    let rounds = 2u64;
    let timeout = Duration::from_secs(240);
    let reached = wait_rounds(&node, RUN, rounds, timeout).await;
    assert!(
        reached >= rounds,
        "the co-located trainer must complete ≥{rounds} rounds end-to-end (got {reached})"
    );

    // The G-2 evidence on the PRODUCT path: every completed round's det digest is surfaced via
    // `vhc_run_detail` and is byte-equal to the offline journal oracle.
    let api = collect_api_digests(&node, RUN, rounds, timeout).await;
    assert_api_digests_cover_and_match_journal(&api, &journal_digests(&node, RUN), rounds);

    leave(&node, RUN, daemon_api::VhcLeaveMode::Graceful, "op-leave").await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    drop(cluster);
}
