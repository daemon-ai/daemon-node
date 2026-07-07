// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE F4 OPERATOR-STEER GATE (DEL-4/DEL-5/DEL-6): the wire `Submit { session, command }` op can
//! address ANY session id — including a delegated child created by the durable job worker — and a
//! parked durable child wakes through the operator `Assign` op. Nothing about the submit path is
//! parent-only; the one-lifecycle-owner guard only rejects ids *claimed* by the durable control
//! surface (`assign`), and delegated children are materialized store-side without a wire claim.
//!
//! SCOPE BOUNDARY (read before extending): an operator `Submit` to a delegated child that is
//! **not** parked-durable (e.g. a `Completed` child) opens a **fresh live incarnation** over the
//! child's session id (`LiveSessions::ensure` builds a new engine; it does NOT hydrate the child's
//! durable delegated conversation) — the turn runs beside, not inside, the durable transcript the
//! delegation produced (proven by `operator_steers_a_delegated_child_by_session_id`). F4 durable
//! resume closes the gap for a **parked-durable** child: a `Submit { StartTurn | Steer }` at a
//! durable session that is live-but-dormant (`Active|Suspended|Ready`) now rides the
//! `enqueue_session_input` + wake rail so the operator's text folds INTO the durable transcript
//! (proven by `submit_steer_lands_in_the_parked_durable_transcript`), rather than opening a
//! divergent incarnation.

use super::harness::*;

use daemon_api::{Outbound, SessionApi};
use daemon_common::{JobId, ReqId};
use daemon_core::{PendingApproval, Snapshot, ToolCall};
use daemon_protocol::{AgentCommand, AgentEvent, ConvTurnView, UserMsg};
use daemon_store::SessionStatus;

/// Poll the durable store until `id` reaches `want` (bounded).
async fn wait_status(store: &Arc<dyn SessionStore>, id: &SessionId, want: SessionStatus) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while store.status(id).await != Some(want.clone()) {
        assert!(
            Instant::now() < deadline,
            "session {id} never reached {want:?}, got {:?}",
            store.status(id).await
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Poll-drain the live session until `want` further `TurnFinished` events arrive (bounded).
async fn drain_finished(node: &Arc<NodeApiImpl>, session: &SessionId, want: usize) {
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

/// Poll-drain the live session until BOTH the `Steered` ack for `request_id` and a subsequent
/// `TurnFinished` arrive, returning the ack's `accepted`. One combined drain because `poll` is
/// destructive: the ack and the steer turn's events can land in one batch, so a caller that
/// returned on the ack alone would silently discard the rest of that batch.
async fn drain_steered_and_finished(
    node: &Arc<NodeApiImpl>,
    session: &SessionId,
    request_id: ReqId,
) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut accepted: Option<bool> = None;
    let mut finished = false;
    while Instant::now() < deadline {
        for o in node.poll(session.clone(), 0).await.expect("poll") {
            match o {
                Outbound::Event(AgentEvent::Steered {
                    request_id: rid,
                    accepted: ack,
                    ..
                }) if rid == request_id => accepted = Some(ack),
                Outbound::Event(AgentEvent::TurnFinished { .. }) => finished = true,
                _ => {}
            }
        }
        if let Some(ack) = accepted {
            if finished {
                return ack;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("no Steered ack + finished steer turn for {request_id:?} (ack: {accepted:?}, finished: {finished})");
}

/// Snapshot a live session's conversation turns (submits `Snapshot`, polls for the reply).
async fn snapshot_turns(node: &Arc<NodeApiImpl>, session: &SessionId) -> Vec<ConvTurnView> {
    node.submit(
        session.clone(),
        AgentCommand::Snapshot {
            request_id: ReqId(9),
        },
    )
    .await
    .expect("submit snapshot");
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        for o in node.poll(session.clone(), 0).await.expect("poll") {
            if let Outbound::Event(AgentEvent::Snapshot { view, .. }) = o {
                return view.turns;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("no snapshot reply");
}

/// OPERATOR ADDRESSES A DELEGATED CHILD BY SESSION ID: after a durable parent's delegation settles,
/// an operator `Submit { session: <child>, StartTurn }` is accepted (no lifecycle conflict — the
/// child was never wire-claimed), a turn runs on the child id, and the operator's text lands in the
/// child's conversation; a follow-up `Steer` is acked `accepted` and its `[steer]` marker lands
/// too. Note the fresh-live-incarnation caveat in the module docs: this proves the operator can
/// reach and drive the child *session id*, not that the durable delegated transcript is resumed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn operator_steers_a_delegated_child_by_session_id() {
    as_system(operator_steers_a_delegated_child_by_session_id_impl()).await;
}
async fn operator_steers_a_delegated_child_by_session_id_impl() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 0, [0x71; 32], fast_host_config());

    // A durable parent delegates once (the gate's orchestrator provider) and completes; the job
    // worker materialized its delegated durable child (`{parent}/c{epoch}`), which also settled.
    let parent = SessionId::new("steer-parent");
    node.assign(parent.clone()).await.expect("assign parent");
    wait_status(&store, &parent, SessionStatus::Completed).await;
    let child = store
        .children_of(&parent)
        .await
        .first()
        .cloned()
        .expect("the delegation materialized a durable child");
    wait_status(&store, &child, SessionStatus::Completed).await;

    // OPERATOR STEER 1 — StartTurn addressed at the CHILD session id. The submit must be accepted
    // (the child is not durable-claimed) and must run a turn on the child.
    node.submit(
        child.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("OPERATOR-STEER-PING"),
            request_id: ReqId(41),
        },
    )
    .await
    .expect("operator Submit must address a delegated child session id");
    drain_finished(&node, &child, 1).await;

    let turns = snapshot_turns(&node, &child).await;
    assert!(
        turns.iter().any(|t| t.text.contains("OPERATOR-STEER-PING")),
        "the operator's StartTurn text must land in the child's conversation: {turns:?}"
    );

    // OPERATOR STEER 2 — Steer addressed at the (now live-resident) child: acked `accepted`, the
    // steer turn runs, and the `[steer]` marker lands in the conversation.
    node.submit(
        child.clone(),
        AgentCommand::Steer {
            text: "OPERATOR-STEER-NUDGE".into(),
            request_id: ReqId(42),
        },
    )
    .await
    .expect("operator Steer must address a delegated child session id");
    assert!(
        drain_steered_and_finished(&node, &child, ReqId(42)).await,
        "the child must accept the operator steer"
    );
    let turns = snapshot_turns(&node, &child).await;
    assert!(
        turns
            .iter()
            .any(|t| t.text.contains("OPERATOR-STEER-NUDGE")),
        "the operator's steer marker must land in the child's conversation: {turns:?}"
    );

    handle.shutdown().await;
}

/// A PARKED DURABLE CHILD WAKES VIA THE OPERATOR `Assign` OP: a durable child dormant on an
/// unanswered approval receives a durably-enqueued input (the store seam `inject_session_input`
/// rides for exactly this shape — a message to a delegated child), and the operator's wire
/// `Assign { session: <child> }` wakes it: the hydrate drains the pending input into the child's
/// re-checkpointed conversation while the child stays parked on its approval. Assign is re-nudged
/// each pass to recover from a wake a concurrent recovery-scanner activation benignly absorbed —
/// the same discipline the detached-delegation parked gate uses.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn operator_assign_wakes_a_parked_durable_child() {
    as_system(operator_assign_wakes_a_parked_durable_child_impl()).await;
}
async fn operator_assign_wakes_a_parked_durable_child_impl() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 0, [0x72; 32], fast_host_config());

    // Seed a durable child-shaped row parked on an unanswered edit approval (the stable dormant
    // durable state), the same way the detached-delegation suite seeds a parked parent — in
    // production the job worker creates this row at `{parent}/cN`. The recovery scanner activates
    // the Ready row and the engine deterministically PARKS it.
    let child = SessionId::new("steer-wake/c1");
    let job_id = JobId::new(format!("{child}:1:approval:0"));
    let mut snapshot = Snapshot::fresh(child.clone());
    snapshot.waiting_for = vec![job_id.clone()];
    snapshot.pending_approvals = vec![PendingApproval {
        job_id,
        call: ToolCall {
            call_id: "c1".into(),
            name: "fs".into(),
            args: r#"{"op":"write","path":"gated.txt","content":"hi"}"#.into(),
        },
        prompt: "approve write to gated.txt".into(),
        path: Some("gated.txt".into()),
        fingerprint: None,
    }];
    store
        .create_session(child.clone(), PARTITION, snapshot.encode().expect("encode"))
        .await
        .expect("create parked child");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !matches!(
        store.status(&child).await,
        Some(SessionStatus::Suspended { .. })
    ) {
        assert!(Instant::now() < deadline, "the child never parked");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // The operator-ish input reaches the parked child durably (pending input, FIFO): this is the
    // store half of `inject_session_input` — the seam a first-class operator durable-steer op
    // would ride (recorded follow-up).
    store
        .enqueue_session_input(&child, UserMsg::new("OPERATOR-WAKE-PING").encode())
        .await;

    // The operator wakes the parked child through the wire `Assign` op; the woken hydrate drains
    // the pending input into the conversation. Generous deadline + a measured re-nudge cadence:
    // under full-workspace test load the activation/wake ticks are starved, and a wake absorbed by
    // a concurrent recovery-scanner activation (which hydrated before the input landed) is benign
    // — the next Assign recovers it.
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        node.assign(child.clone())
            .await
            .expect("operator Assign must wake a parked durable child");
        let has_marker = store
            .peek_snapshot(&child)
            .await
            .and_then(|blob| Snapshot::decode(&blob).ok())
            .map(|s| {
                s.conversation.turns.iter().any(|t| {
                    matches!(t, daemon_core::Turn::User(msg) if msg.text.contains("OPERATOR-WAKE-PING"))
                })
            })
            .unwrap_or(false);
        if has_marker {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the pending input never reached the parked child via Assign"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    // The child stayed parked: the wake folded the input in but did NOT fast-forward the approval.
    assert!(matches!(
        store.status(&child).await,
        Some(SessionStatus::Suspended { .. } | SessionStatus::Active)
    ));

    handle.shutdown().await;
}

/// THE F4 DURABLE-RESUME GATE: a wire `Submit { StartTurn | Steer }` addressed at a PARKED-DURABLE
/// session rides the durable pending-input rail — the operator's text folds INTO the durable
/// transcript on the next wake — instead of opening a divergent fresh live incarnation over the
/// durable state. Seeded exactly like `operator_assign_wakes_a_parked_durable_child`: a durable
/// child parked (`Suspended`) on an unanswered approval. Two operator submits (a `StartTurn` and a
/// `Steer`) enqueue their messages; the wake drains them into the child's re-checkpointed
/// conversation while it stays parked on its approval. Assign is re-nudged each pass to recover a
/// wake a concurrent recovery-scanner activation benignly absorbed (same discipline as the parked
/// gate above).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn submit_steer_lands_in_the_parked_durable_transcript() {
    as_system(submit_steer_lands_in_the_parked_durable_transcript_impl()).await;
}
async fn submit_steer_lands_in_the_parked_durable_transcript_impl() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 0, [0x73; 32], fast_host_config());

    // Seed a durable child parked on an unanswered edit approval (the stable dormant durable
    // state), the same shape the parked-assign gate uses.
    let child = SessionId::new("f4-steer/c1");
    let job_id = JobId::new(format!("{child}:1:approval:0"));
    let mut snapshot = Snapshot::fresh(child.clone());
    snapshot.waiting_for = vec![job_id.clone()];
    snapshot.pending_approvals = vec![PendingApproval {
        job_id,
        call: ToolCall {
            call_id: "c1".into(),
            name: "fs".into(),
            args: r#"{"op":"write","path":"gated.txt","content":"hi"}"#.into(),
        },
        prompt: "approve write to gated.txt".into(),
        path: Some("gated.txt".into()),
        fingerprint: None,
    }];
    store
        .create_session(child.clone(), PARTITION, snapshot.encode().expect("encode"))
        .await
        .expect("create parked child");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !matches!(
        store.status(&child).await,
        Some(SessionStatus::Suspended { .. })
    ) {
        assert!(Instant::now() < deadline, "the child never parked");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // F4: the operator submits a StartTurn then a Steer AT the parked-durable child. Neither opens
    // a fresh live incarnation (no lifecycle conflict, no divergent engine) — each rides the
    // durable pending-input rail (enqueue + wake).
    node.submit(
        child.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("F4-STARTTURN-PING"),
            request_id: ReqId(51),
        },
    )
    .await
    .expect("Submit StartTurn at a parked-durable session must be accepted (durable-resume)");
    node.submit(
        child.clone(),
        AgentCommand::Steer {
            text: "F4-STEER-NUDGE".into(),
            request_id: ReqId(52),
        },
    )
    .await
    .expect("Submit Steer at a parked-durable session must be accepted (durable-resume)");

    // Submit's own durable-rail wake drives the drain: the woken incarnation folds both messages
    // into the durable conversation. (No competing `assign` nudge here — that would race submit's
    // enqueued wake and can overwrite the folded checkpoint.)
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let both = store
            .peek_snapshot(&child)
            .await
            .and_then(|blob| Snapshot::decode(&blob).ok())
            .map(|s| {
                let has = |needle: &str| {
                    s.conversation.turns.iter().any(
                        |t| matches!(t, daemon_core::Turn::User(msg) if msg.text.contains(needle)),
                    )
                };
                has("F4-STARTTURN-PING") && has("F4-STEER-NUDGE")
            })
            .unwrap_or(false);
        if both {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the operator's Submit StartTurn+Steer never landed in the durable transcript"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    // The child stayed parked: the resume folded the input in but did NOT fast-forward the approval
    // (F4 is a durable resume, not an operator approval).
    assert!(matches!(
        store.status(&child).await,
        Some(SessionStatus::Suspended { .. } | SessionStatus::Active)
    ));

    handle.shutdown().await;
}
