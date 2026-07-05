// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE W3 NOTIFY GATE: `NodeApiImpl::inject_session_input` — the seam background-process exit /
//! watch notifications ride into their owning session — works across **both** session lifecycles:
//!
//! - a **live** (actor-resident) session takes a real `StartTurn` and runs a reactive turn
//!   (Observe-while-idle folds context but drives no turn — proven in `daemon-core`'s actor tests —
//!   so the notification must StartTurn);
//! - a **durable** (activation-lifecycle) session — which `SessionApi::submit` rejects under the
//!   one-lifecycle-owner guard-rail — receives a durable pending input + wake; the incarnation
//!   drains it into the conversation at hydrate (the seam W7's child messaging reuses);
//! - a **settled** durable session drops the input (its owner is gone) instead of erroring or
//!   resurrecting the session.

use super::harness::*;

use daemon_api::{ControlApi, Outbound, SessionApi};
use daemon_common::{JobId, ReqId};
use daemon_core::{PendingApproval, Snapshot, ToolCall, Turn};
use daemon_protocol::{AgentCommand, AgentEvent, UserMsg};

/// A live session that already ran one turn takes an injected notification as a fresh reactive
/// turn: a second `TurnFinished` arrives without any user submit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn injected_input_drives_a_reactive_turn_on_a_live_session() {
    as_system(injected_input_drives_a_reactive_turn_on_a_live_session_impl()).await;
}
async fn injected_input_drives_a_reactive_turn_on_a_live_session_impl() {
    let (node, handle) = assemble();
    let session = SessionId::new("notify-live-1");

    as_system(node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello"),
            request_id: ReqId(1),
        },
    ))
    .await
    .expect("submit StartTurn");
    drain_until_finished_count(&node, &session, 1).await;

    // The notification: no user involvement, yet a full turn runs over it.
    node.inject_session_input(
        &session,
        "[IMPORTANT: Background process proc_test completed normally (exit code 0).]".to_string(),
    )
    .await
    .expect("inject into a live session");
    drain_until_finished_count(&node, &session, 1).await;

    handle.shutdown().await;
}

/// Poll-drain the live session until `want` further `TurnFinished` events arrive (bounded).
async fn drain_until_finished_count(node: &Arc<NodeApiImpl>, session: &SessionId, want: usize) {
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut seen = 0usize;
    while Instant::now() < deadline {
        for o in node.poll(session.clone(), 0).await.expect("poll") {
            if matches!(&o, Outbound::Event(AgentEvent::TurnFinished { .. })) {
                seen += 1;
            }
        }
        if seen >= want {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("expected {want} TurnFinished event(s), saw {seen}");
}

/// A durable session parked dormant on a §12 approval (no runnable job, no live actor — the
/// stable "dehydrated" state; the daemon may even have restarted, so its id is **unclaimed**)
/// receives an injected notification through the store seam: the pending input is drained at the
/// wake's hydrate and lands in the re-checkpointed conversation, while the session stays parked
/// (no forward progress without the operator).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn injected_input_reaches_a_parked_durable_session_via_the_store_seam() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 0, [0x65; 32], fast_host_config());
    let session = SessionId::new("notify-durable-1");

    // Seed a session whose snapshot already carries an unanswered edit approval (the durable
    // engine state after a gated tool asked and the daemon restarted). Its id carries no
    // lifecycle claim — the router must go by durable evidence alone. The node's own recovery
    // scanner activates the Ready row, and the engine deterministically PARKS it
    // (`pending_approvals` non-empty → suspend-for-approval, no provider call, no runnable job):
    // the stable dormant state a background process outlives.
    let job_id = JobId::new(format!("{session}:1:approval:0"));
    let mut snapshot = Snapshot::fresh(session.clone());
    snapshot.waiting_for = vec![job_id.clone()];
    snapshot.pending_approvals = vec![PendingApproval {
        job_id: job_id.clone(),
        call: ToolCall {
            call_id: "c1".into(),
            name: "fs".into(),
            args: r#"{"op":"write","path":"gated.txt","content":"hi"}"#.into(),
        },
        prompt: "approve write to gated.txt".into(),
        path: Some("gated.txt".into()),
    }];
    store
        .create_session(
            session.clone(),
            PARTITION,
            snapshot.encode().expect("encode"),
        )
        .await
        .expect("create session");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !matches!(
        store.status(&session).await,
        Some(daemon_store::SessionStatus::Suspended { .. })
    ) {
        assert!(
            Instant::now() < deadline,
            "the seeded approval never parked the session"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Inject: the durable branch enqueues a pending input + wake (never a live submit — which
    // would reject a durable session, and would build a fresh divergent engine for this one).
    node.inject_session_input(&session, "[IMPORTANT: proc marker-xyz exited]".to_string())
        .await
        .expect("inject into a parked durable session");

    // The wake re-activates the incarnation: hydrate drains the input into the conversation, the
    // run re-parks on the still-unanswered approval, and the checkpoint persists the folded text.
    // (Reading the pending queue here would *steal* the input from hydrate — `take` is
    // destructive — so the loop watches the durable snapshot only. The loop also re-nudges the
    // wake each pass: a wake is a HINT, and the one the inject enqueued can be benignly absorbed
    // by an in-flight activation that hydrated before the input landed — in production the next
    // natural wake drains it; this test has no other wake source.)
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        store.enqueue_wake(session.clone()).await;
        let snapshot = store
            .peek_snapshot(&session)
            .await
            .and_then(|blob| Snapshot::decode(&blob).ok());
        if let Some(snapshot) = snapshot {
            let has_marker = snapshot.conversation.turns.iter().any(
                |t| matches!(t, Turn::User(msg) if msg.text.contains("proc marker-xyz exited")),
            );
            if has_marker {
                break;
            }
        }
        assert!(
            Instant::now() < deadline,
            "the injected input never reached the parked session's conversation"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // The injected input did not fast-forward the approval: once the (nudge-driven) activation
    // churn settles, the session is parked (Suspended) again — a read can transiently catch
    // `Active` mid-activation — and it never completes.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match store.status(&session).await {
            Some(daemon_store::SessionStatus::Suspended { .. }) => break,
            Some(daemon_store::SessionStatus::Completed) | None => {
                panic!("the parked session must not complete off an injected notification")
            }
            _ => {}
        }
        assert!(
            Instant::now() < deadline,
            "the session never re-parked after draining the injected input"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // Hydrate (not this test) consumed the queue: nothing is left pending.
    assert!(store.take_session_inputs(&session).await.is_empty());

    handle.shutdown().await;
}

/// An injected notification for a **settled** durable session is dropped (Ok, no pending row):
/// the owner is gone; nothing should resurrect it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn injected_input_into_a_settled_durable_session_is_dropped() {
    as_system(injected_input_into_a_settled_durable_session_is_dropped_impl()).await;
}
async fn injected_input_into_a_settled_durable_session_is_dropped_impl() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 0, [0x66; 32], fast_host_config());
    let session = SessionId::new("notify-settled-1");

    node.assign(session.clone()).await.expect("assign");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !matches!(
        store.status(&session).await,
        Some(daemon_store::SessionStatus::Completed)
    ) {
        assert!(
            Instant::now() < deadline,
            "the assigned session never settled"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    node.inject_session_input(&session, "[IMPORTANT: too late]".to_string())
        .await
        .expect("inject into a settled session is a clean drop");
    assert!(
        store.take_session_inputs(&session).await.is_empty(),
        "nothing is queued for a settled session"
    );
    assert!(
        matches!(
            store.status(&session).await,
            Some(daemon_store::SessionStatus::Completed)
        ),
        "the settled session stays settled"
    );

    handle.shutdown().await;
}
