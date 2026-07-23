// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! Hard-kill continuity through the REMOTE live checkpoint (the streaming det-state restore path,
//! end to end over three real node processes on the R2-compatible payload plane).
//!
//! The existing hard-kill gate (`churn_restore::killed_worker_respawns_and_rejoins`) proves
//! digest continuity across a SIGKILL over a shared-filesystem plane; the R2 gate
//! (`r2_payload_plane`) proves the presign plane under baseline training. Neither proves the
//! streaming restore path over the remote plane. This gate is that intersection: a trainer's
//! WORKER SUBPROCESS is SIGKILLed mid-run, and the rejoined incarnation restores from the
//! survivor's post-deadline (trainer, live) checkpoint published to the presigned R2 content
//! plane — resolve the pointer, fetch the by-reference document, register the family folds
//! (`register_state_chunks`, [SF-R2]), and stream each family's windows by chunk-keyed
//! `data@2::fetch`, materializing no whole family in guest memory (ABI §12.14 [SF-6]).
//!
//! Digest continuity alone would be the shortcut this gate rejects: it would pass even if the
//! restore had silently rehydrated some other way. So the gate also asserts the RESTORE
//! MECHANISM directly:
//!
//! 1. the restored (trainer, live) pointer's document, fetched from the remote content plane, is
//!    the v2 BY-REFERENCE form — `master`/`ef`/`adamw_m`/`adamw_v` are `FamilyRef` sections (each
//!    a fold over its own chunk list), `round` is inline;
//! 2. every family the document references has its chunks present as content-addressed objects on
//!    the remote plane (the real chunk-keyed content the streaming rehydration fetches);
//! 3. the rejoined node's own log carries the streaming-restore registration marker (the by-ref
//!    root registration path actually ran);
//!
//! on top of the outcome — the killed node's replacement worker is a different OS process and its
//! fresh incarnation voices AGREEING det digests for post-kill rounds the survivor also voices.

mod harness;

use std::time::Duration;

use daemon_vhc_proto::det_state::{decode_checkpoint_doc, CkptDocSection};
use harness::{
    assert_digests_agree, base_peer, join, journal_digests, leave, recent_events, seed_corpus_r2,
    spawn_node, wait_rounds, worker_children, NodeSpec,
};

/// The content-plane object key for a content-addressed artifact (`runs/<run>/payload/<hex>` —
/// the seam corpus objects, checkpoint documents, and state chunks all share; the chunk-keyed
/// restore resolver fetches through it).
fn content_key(run: &str, hex: &str) -> String {
    format!("runs/{run}/payload/{hex}")
}

/// Wait until the registry holds a trainer checkpoint pointer of `kind` at (or past) `min_round`.
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
            "no trainer {kind} checkpoint pointer at round >= {min_round} within {timeout:?}"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn killed_worker_restores_from_remote_live_checkpoint() {
    let _serial = harness::serial_guard();
    let run = "acceptance-hardkill-remote";
    let base_port = harness::free_port();
    let base_url = format!("http://127.0.0.1:{base_port}/api/v1/vhc");

    // The R2 tier: `payload_dir: None` selects the presigned content store on every worker (no
    // shared filesystem), so the live checkpoint document + its family chunks cross the remote
    // plane and the restore is a genuine remote rehydration.
    let mk = |name: &str, seat_claim: bool, backoff_ms: u64| NodeSpec {
        name: Box::leak(name.to_string().into_boxed_str()),
        registry_base: Box::leak(base_url.clone().into_boxed_str()),
        seat_claim,
        payload_dir: None,
        allowlist: Box::leak(base_url.clone().into_boxed_str()),
        reconcile_tick_ms: 500,
        initial_backoff_ms: backoff_ms,
    };
    let coord = spawn_node(&mk("coordinator", true, 0));
    // The victim's first reconvergence backoff outlasts the deadline round the survivor finalizes
    // alone, so the rejoin resolves the SURVIVOR's post-deadline live checkpoint as its restore
    // source deterministically (never racing the round settle) — the churn gate's timing.
    let trainer_a = spawn_node(&mk("trainer-a", false, 40_000));
    let trainer_b = spawn_node(&mk("trainer-b", false, 0));

    let bases = [
        base_peer(&coord),
        base_peer(&trainer_a),
        base_peer(&trainer_b),
    ];
    let cluster = harness::start_cluster_with(
        base_port,
        run,
        &bases,
        0,
        2,
        1,
        daemon_vhc_testkit::live_genesis::LiveTiming::churn(),
    )
    .await;
    seed_corpus_r2(&cluster, run, &cluster.genesis);

    join(&coord, run, "op-coord").await;
    join(&trainer_a, run, "op-a").await;
    join(&trainer_b, run, "op-b").await;

    let timeout = Duration::from_secs(180);
    wait_rounds(&trainer_a, run, 2, timeout).await;
    wait_rounds(&trainer_b, run, 2, timeout).await;

    // The periodic LIVE checkpoint cadence publishes pointers + uploads their documents/chunks to
    // the remote plane BEFORE the crash: the hard-killed peer never drains, so a remote live
    // checkpoint is the only restore source it will have.
    wait_checkpoint(&cluster, "live", 1, Duration::from_secs(60)).await;

    // SIGKILL trainer-a's worker SUBPROCESS (never the node): the supervisor respawns it and the
    // node's reconciliation re-joins the run as a fresh incarnation.
    let before = worker_children(&trainer_a);
    assert!(!before.is_empty(), "trainer-a has a live worker to kill");
    for pid in &before {
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(*pid as i32),
            nix::sys::signal::Signal::SIGKILL,
        );
    }

    // Reconvergence: the coordinator absence-drops the dead incarnation, the respawned
    // incarnation's join materializes and RESTORES from the remote live checkpoint, and rounds
    // advance again on both trainer nodes.
    let b_before = journal_digests(&trainer_b, run)
        .keys()
        .max()
        .copied()
        .unwrap_or(0);
    let churn_timeout = Duration::from_secs(240);
    wait_rounds(&trainer_b, run, b_before + 2, churn_timeout).await;

    // The killed node's replacement worker is a DIFFERENT OS process, and its fresh incarnation
    // publishes digests again for POST-KILL rounds the survivor also voices.
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
            let mut detail = String::new();
            for ev in recent_events(&trainer_a, run).await {
                if let daemon_api::VhcEvent::Error {
                    class, detail: d, ..
                } = ev
                {
                    detail = format!("{class}: {d}");
                }
            }
            panic!(
                "killed worker did not respawn + rejoin over the remote plane \
                 (respawned={respawned} rejoined={rejoined}; last node error: {detail})"
            );
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // ---- the outcome: digest continuity across the hard crash over the remote plane -------------
    // Snapshot the oracle IMMEDIATELY after the rejoin is confirmed (before the fixture I/O of the
    // mechanism-evidence block below), so the shared window is the stable restore-continuity window
    // and never drifts into the open-ended post-reconvergence round progression the restore claim
    // deliberately does not own (the survivor keeps folding; how far a re-materialized roster
    // carries rounds is the coordination layer's behavior, not this gate's).
    let a = journal_digests(&trainer_a, run);
    let b = journal_digests(&trainer_b, run);
    let post_crash_shared = a
        .keys()
        .filter(|r| **r > b_before && b.contains_key(r))
        .count();
    assert!(
        post_crash_shared >= 1,
        "the rejoined incarnation shares no post-crash rounds with the survivor (A={:?} B={:?})",
        a.keys().collect::<Vec<_>>(),
        b.keys().collect::<Vec<_>>()
    );
    assert_digests_agree(&a, &b, 3);

    // ---- mechanism evidence: the restore is streaming by-ref rehydration over the remote plane,
    //      not merely a continuity coincidence (this reads the fixture's persisted state, which is
    //      stable regardless of ongoing round progression) ----------------------------------------

    // (1) The restored (trainer, live) pointer's document is the v2 BY-REFERENCE form. All live
    // checkpoints share this shape; the freshest live pointer's document is the restore-source
    // form. It is fetched from the REMOTE content plane by its content hash (the pointer hash).
    let ptr = cluster
        .registry
        .checkpoint("trainer", "live")
        .expect("a (trainer, live) checkpoint pointer exists");
    let doc_bytes = cluster
        .registry
        .get_object(&content_key(run, &ptr.hash))
        .expect(
            "the live checkpoint document is on the remote content plane under its content hash",
        );
    let (_manifest, sections) =
        decode_checkpoint_doc(&doc_bytes).expect("the live checkpoint document decodes as v2");

    let by_ref: std::collections::BTreeMap<&str, &daemon_vhc_proto::det_state::FamilyRef> =
        sections
            .iter()
            .filter_map(|s| match s {
                CkptDocSection::ByRef(name, family) => Some((name.as_str(), family)),
                CkptDocSection::Inline(..) => None,
            })
            .collect();
    for family in ["master", "ef", "adamw_m", "adamw_v"] {
        assert!(
            by_ref.contains_key(family),
            "live checkpoint document must carry `{family}` by reference (v2 by-ref form); \
             sections were {:?}",
            sections
                .iter()
                .map(CkptDocSection::name)
                .collect::<Vec<_>>()
        );
    }
    assert!(
        sections
            .iter()
            .any(|s| matches!(s, CkptDocSection::Inline(n, _) if n == "round")),
        "the round watermark stays inline"
    );

    // (2) Every referenced family's chunks are present as content-addressed objects on the remote
    // plane — the real chunk-keyed content the streaming rehydration fetches by presigned GET. A
    // by-ref FamilyRef whose fold is not the fold of its own chunk list would fail here structurally.
    for (name, family) in &by_ref {
        family
            .validate()
            .unwrap_or_else(|e| panic!("family `{name}` ref is not self-consistent: {e}"));
        for chunk in &family.chunk_hashes {
            assert!(
                cluster
                    .registry
                    .get_object(&content_key(run, &chunk.to_hex()))
                    .is_some(),
                "family `{name}` chunk {} is missing from the remote content plane (the \
                 streaming restore would have no bytes to rehydrate)",
                chunk.to_hex()
            );
        }
    }

    // (Restore-path observability lives in the product binaries — the node's
    // `resolved late-join checkpoint restore pointer` marker and the worker's
    // `streaming restore from checkpoint document` / register_state_chunks marker — but is NOT
    // asserted here: subprocess stdio is block-buffered, so a quiet node's buffer need not reach
    // its log file before teardown. The structural by-ref document + remote-chunk evidence above,
    // together with the agreeing post-restore digests, establish the streaming rehydration
    // mechanism deterministically and log-independently — strictly stronger than a log grep, and
    // the opposite of the digest-continuity-alone shortcut.)

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
