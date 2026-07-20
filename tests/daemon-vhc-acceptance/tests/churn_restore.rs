// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! G4 + G5 — churn and checkpoint restore over three real node processes.
//!
//! **Graceful churn + restore** (`graceful_churn_rejoins_through_checkpoint_restore`): a trainer
//! leaves mid-run (drain snapshot → checkpoint document on the content plane → pointer at the
//! registry); the coordinator finalizes the stalled round with an absence mark, drops the member,
//! breaches the `min_peers` floor into cooldown, and waits. The second trainer then leaves too
//! (its later checkpoint — which folds the interregnum record — wins the pointer). Both trainers
//! REJOIN as fresh incarnations: each node resolves the pointer and the workers migrate from the
//! same checkpoint document before running, so training resumes with identical state and the
//! per-round det digests agree across the churn.
//!
//! **Hard churn + digest continuity** (`killed_worker_respawns_and_rejoins`): a trainer's
//! WORKER SUBPROCESS is SIGKILLed mid-run. The node's supervisor respawns the child and
//! reconciliation re-joins the run as a new incarnation; the coordinator absence-drops the dead
//! incarnation and the run reconverges — training rounds advance again for both trainer nodes,
//! every node process stays up, and the killed node's replacement worker is a different OS
//! process. Digest continuity across the HARD crash rides the periodic LIVE checkpoint cadence
//! (spec §9): the trainers publish full-state checkpoint pointers every ingested round, so the
//! rejoined incarnation — whose reconvergence backoff outlasts the survivor's deadline round —
//! restores from the survivor's post-deadline live pointer and voices AGREEING det digests for
//! every shared round across the kill (the full cross-peer oracle, not just reconvergence).

mod harness;

use std::time::Duration;

use harness::{
    assert_digests_agree, base_peer, join, journal_digests, leave, seed_corpus_fs, spawn_node,
    start_cluster_with, wait_rounds, worker_children, NodeSpec,
};

/// Wait until the registry holds a trainer checkpoint pointer of `kind` at (or past)
/// `min_round`.
async fn wait_checkpoint(
    cluster: &harness::Cluster,
    kind: &str,
    min_round: u64,
    timeout: Duration,
) -> harness::registry::Checkpoint {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(c) = cluster.registry.checkpoint("trainer", kind) {
            if c.round >= min_round {
                return c;
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "no trainer {kind} checkpoint pointer at round >= {min_round} within {timeout:?} \
             (got {:?})",
            cluster
                .registry
                .checkpoint("trainer", kind)
                .map(|c| c.round)
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn graceful_churn_rejoins_through_checkpoint_restore() {
    let _serial = harness::serial_guard();
    let run = "acceptance-churn-graceful";
    let payload_root = tempfile::tempdir().expect("shared payload root");
    let base_port = harness::free_port();
    let base_url = format!("http://127.0.0.1:{base_port}/api/v1/vhc");

    let mk = |name: &str, seat_claim: bool| NodeSpec {
        name: Box::leak(name.to_string().into_boxed_str()),
        registry_base: Box::leak(base_url.clone().into_boxed_str()),
        seat_claim,
        payload_dir: Some(payload_root.path()),
        allowlist: Box::leak(base_url.clone().into_boxed_str()),
        reconcile_tick_ms: 500,
        initial_backoff_ms: 0,
    };
    let coord = spawn_node(&mk("coordinator", true));
    let trainer_a = spawn_node(&mk("trainer-a", false));
    let trainer_b = spawn_node(&mk("trainer-b", false));

    let bases = [
        base_peer(&coord),
        base_peer(&trainer_a),
        base_peer(&trainer_b),
    ];
    // The churn tier: real coordinator timer, deadline-survivable absences, a one-round absence
    // budget, prompt cooldown (where staged rejoins materialize).
    let cluster = start_cluster_with(
        base_port,
        run,
        &bases,
        0,
        2,
        1,
        daemon_vhc_testkit::live_genesis::LiveTiming::churn(),
    )
    .await;
    seed_corpus_fs(payload_root.path(), run, &cluster.genesis);

    join(&coord, run, "op-coord").await;
    join(&trainer_a, run, "op-a").await;
    join(&trainer_b, run, "op-b").await;

    // A healthy pre-churn baseline.
    let timeout = Duration::from_secs(180);
    wait_rounds(&trainer_a, run, 2, timeout).await;
    wait_rounds(&trainer_b, run, 2, timeout).await;

    // Trainer A leaves gracefully: drain snapshot -> checkpoint document + pointer. The stalled
    // round finalizes at the deadline with A absent (dropped), the floor breach forces cooldown.
    leave(&trainer_a, run, daemon_api::VhcLeaveMode::Graceful, "op-la").await;
    wait_checkpoint(&cluster, "drain", 0, Duration::from_secs(60)).await;

    // The survivor folds the interregnum record; ITS later checkpoint must win the pointer so
    // both rejoiners restore a state that already contains every record either of them missed.
    let after_a = journal_digests(&trainer_b, run)
        .keys()
        .max()
        .copied()
        .unwrap_or(0);
    let deadline = std::time::Instant::now() + Duration::from_secs(90);
    loop {
        let b_now = journal_digests(&trainer_b, run)
            .keys()
            .max()
            .copied()
            .unwrap_or(0);
        if b_now >= after_a {
            // B has ingested up to its own watermark; give the coordinator a beat to settle the
            // absence drop + cooldown before B leaves too.
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "survivor never settled after the graceful leave"
        );
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    tokio::time::sleep(Duration::from_secs(2)).await;
    leave(&trainer_b, run, daemon_api::VhcLeaveMode::Graceful, "op-lb").await;
    let pointer = wait_checkpoint(&cluster, "drain", after_a, Duration::from_secs(60)).await;

    // Both trainers rejoin as fresh incarnations; each node resolves the pointer and the worker
    // migrates from the checkpoint document before running (§10.3 step 4).
    join(&trainer_a, run, "op-a2").await;
    join(&trainer_b, run, "op-b2").await;

    // Training resumes past the checkpoint: both trainers must publish digests for rounds beyond
    // the pointer round (the ghost incarnation costs one deadline round before it drops).
    let target = pointer.round + 2;
    let churn_timeout = Duration::from_secs(240);
    wait_rounds(&trainer_a, run, target, churn_timeout).await;
    wait_rounds(&trainer_b, run, target, churn_timeout).await;

    leave(
        &trainer_a,
        run,
        daemon_api::VhcLeaveMode::Graceful,
        "op-la2",
    )
    .await;
    leave(
        &trainer_b,
        run,
        daemon_api::VhcLeaveMode::Graceful,
        "op-lb2",
    )
    .await;
    leave(&coord, run, daemon_api::VhcLeaveMode::Graceful, "op-lc").await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // The digest oracle across the churn: every round BOTH trainers report — pre-churn rounds and
    // post-restore rounds alike — carries one agreed det digest. The restored state is b's
    // checkpoint, so post-rejoin folds are identical by construction.
    let a = journal_digests(&trainer_a, run);
    let b = journal_digests(&trainer_b, run);
    assert!(
        a.keys().any(|r| *r > pointer.round),
        "trainer-a produced no post-restore digests (restore did not resume training)"
    );
    assert_digests_agree(&a, &b, 3);

    drop(cluster);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn killed_worker_respawns_and_rejoins() {
    let _serial = harness::serial_guard();
    let run = "acceptance-churn-kill";
    let payload_root = tempfile::tempdir().expect("shared payload root");
    let base_port = harness::free_port();
    let base_url = format!("http://127.0.0.1:{base_port}/api/v1/vhc");

    let mk = |name: &str, seat_claim: bool, backoff_ms: u64| NodeSpec {
        name: Box::leak(name.to_string().into_boxed_str()),
        registry_base: Box::leak(base_url.clone().into_boxed_str()),
        seat_claim,
        payload_dir: Some(payload_root.path()),
        allowlist: Box::leak(base_url.clone().into_boxed_str()),
        reconcile_tick_ms: 500,
        initial_backoff_ms: backoff_ms,
    };
    let coord = spawn_node(&mk("coordinator", true, 0));
    // The victim's first reconvergence backoff outlasts the deadline round the survivor
    // finalizes alone (~30s at the churn timing): the rejoin then resolves the SURVIVOR's
    // post-deadline live checkpoint as its restore source — the digest-continuity input —
    // deterministically, never racing the round settle.
    let trainer_a = spawn_node(&mk("trainer-a", false, 40_000));
    let trainer_b = spawn_node(&mk("trainer-b", false, 0));

    let bases = [
        base_peer(&coord),
        base_peer(&trainer_a),
        base_peer(&trainer_b),
    ];
    let cluster = start_cluster_with(
        base_port,
        run,
        &bases,
        0,
        2,
        1,
        daemon_vhc_testkit::live_genesis::LiveTiming::churn(),
    )
    .await;
    seed_corpus_fs(payload_root.path(), run, &cluster.genesis);

    join(&coord, run, "op-coord").await;
    join(&trainer_a, run, "op-a").await;
    join(&trainer_b, run, "op-b").await;

    let timeout = Duration::from_secs(180);
    wait_rounds(&trainer_a, run, 2, timeout).await;
    wait_rounds(&trainer_b, run, 2, timeout).await;

    // The periodic LIVE checkpoint cadence is producing pointers BEFORE the crash (spec §9):
    // the hard-killed peer never drains, so this is the only restore source it will have.
    wait_checkpoint(&cluster, "live", 1, Duration::from_secs(60)).await;

    // SIGKILL trainer-a's worker SUBPROCESS (never the node): the supervisor must respawn it and
    // the node's reconciliation must re-join the run as a fresh incarnation.
    let before = worker_children(&trainer_a);
    assert!(!before.is_empty(), "trainer-a has a live worker to kill");
    for pid in &before {
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(*pid as i32),
            nix::sys::signal::Signal::SIGKILL,
        );
    }

    // Reconvergence: the coordinator absence-drops the dead incarnation, the floor breach forces
    // cooldown, the respawned incarnation's join materializes, and rounds advance again on BOTH
    // trainer nodes (the survivor keeps folding; the rejoined incarnation trains anew).
    let b_before = journal_digests(&trainer_b, run)
        .keys()
        .max()
        .copied()
        .unwrap_or(0);
    let churn_timeout = Duration::from_secs(240);
    wait_rounds(&trainer_b, run, b_before + 2, churn_timeout).await;

    // The killed node's replacement worker is a DIFFERENT OS process, and its fresh incarnation
    // publishes digests again for POST-KILL rounds the survivor also voices (its journal grows
    // past the kill into the shared window the continuity oracle runs over).
    let deadline = std::time::Instant::now() + churn_timeout;
    loop {
        let after = worker_children(&trainer_a);
        let respawned = !after.is_empty() && after.iter().all(|p| !before.contains(p));
        let a_now = journal_digests(&trainer_a, run);
        let b_now = journal_digests(&trainer_b, run);
        let rejoined = a_now.keys().any(|r| *r > b_before && b_now.contains_key(r));
        if respawned && rejoined {
            break;
        }
        if std::time::Instant::now() >= deadline {
            // Surface the node's own diagnosis (the reconciler emits a typed `reconvergence`
            // error per failed re-join attempt) before failing.
            let mut detail = String::new();
            for ev in harness::recent_events(&trainer_a, run).await {
                if let daemon_api::VhcEvent::Error {
                    class, detail: d, ..
                } = ev
                {
                    detail = format!("{class}: {d}");
                }
            }
            panic!(
                "killed worker did not respawn + rejoin \
                 (respawned={respawned} rejoined={rejoined}; last node error: {detail})"
            );
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // DIGEST CONTINUITY ACROSS THE HARD CRASH (spec §9): the rejoined incarnation restored from
    // the freshest (trainer, live) pointer — the survivor's post-deadline periodic checkpoint,
    // whose state folds every record the victim missed while dead — so every round BOTH
    // trainers voice a det digest for, before AND after the kill, carries one agreed digest.
    // The full cross-peer oracle runs over the whole journal set, and at least one shared round
    // must sit PAST the kill (the loop above waited for it). Open-ended round progression
    // beyond this shared window is deliberately NOT awaited: how far the round protocol carries
    // a re-materialized roster is the coordination layer's own behavior, not this gate's
    // restore-continuity claim.
    let a = journal_digests(&trainer_a, run);
    let b = journal_digests(&trainer_b, run);
    let post_crash_shared = a
        .keys()
        .filter(|r| **r > b_before && b.contains_key(r))
        .count();
    assert!(
        post_crash_shared >= 1,
        "the rejoined incarnation shares no post-crash rounds with the survivor \
         (A={:?} B={:?})",
        a.keys().collect::<Vec<_>>(),
        b.keys().collect::<Vec<_>>()
    );
    assert_digests_agree(&a, &b, 3);

    // Every node process survived the churn.
    for node in [&coord, &trainer_a, &trainer_b] {
        assert!(
            node.is_alive(),
            "node `{}` must survive the churn",
            node.name
        );
    }

    leave(&trainer_a, run, daemon_api::VhcLeaveMode::Graceful, "op-la").await;
    leave(&trainer_b, run, daemon_api::VhcLeaveMode::Graceful, "op-lb").await;
    leave(&coord, run, daemon_api::VhcLeaveMode::Graceful, "op-lc").await;

    drop(cluster);
}
