// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::harness::*;

/// THE THREAD-C GATE: reconnect + scroll-back through durable, verified history. After an
/// interactive turn seals into the unified verifiable journal, the session's history is read
/// back through the (non-destructive) `session_history` surface — independent of the live drain,
/// exactly as a reconnecting client sees it. The coalesced assistant message is present, the
/// whole sealed chain verifies under the node's published verifying key, and the read is
/// non-destructive (a second read returns the same page).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reconnect_reads_back_verified_session_history() {
    as_system(reconnect_reads_back_verified_session_history_impl()).await;
}
async fn reconnect_reads_back_verified_session_history_impl() {
    use daemon_api::{ControlApi, JournalRecordPayload, Outbound, SessionApi};
    use daemon_common::ReqId;
    use daemon_protocol::{AgentCommand, AgentEvent, TranscriptBlock, TranscriptRole, UserMsg};

    let (node, handle) = assemble();
    let session = SessionId::new("history-1");

    // Drive an interactive turn to TurnFinished (the live path journals + seals per turn).
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit StartTurn");
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut finished = false;
    while Instant::now() < deadline {
        let drained = node.poll(session.clone(), 0).await.expect("poll");
        if drained
            .iter()
            .any(|o| matches!(o, Outbound::Event(AgentEvent::TurnFinished { .. })))
        {
            finished = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(finished, "the interactive turn never reached TurnFinished");

    // Scroll back through durable history — non-destructive and independent of the live drain
    // (the seal may land just after TurnFinished drains, so retry until the page appears).
    let mut page = None;
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        let p = node.session_history(session.clone(), 0, None, 0).await;
        if !p.entries.is_empty() {
            page = Some(p);
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let page = page.expect("durable history should appear after the turn seals");

    // The whole sealed chain verifies, and the coalesced assistant message is present.
    assert!(
        page.entries.iter().all(|e| e.verified),
        "every sealed entry must verify under the node key: {page:?}"
    );
    // BOTH halves of the conversation are durable blocks: the journal is the conversation, so a
    // cold-rendering client (no live drain, no local echo) reconstructs user AND assistant turns
    // from `SessionHistory` alone. The user block carries the submitted text verbatim.
    assert!(
        page.entries.iter().any(|e| matches!(
            &e.payload,
            JournalRecordPayload::Block {
                block: TranscriptBlock::Message { role: TranscriptRole::User, text }
            } if text == "hello"
        )),
        "expected the user's turn-opening message as a durable block, got {page:?}"
    );
    assert!(
        page.entries.iter().any(|e| matches!(
            &e.payload,
            JournalRecordPayload::Block {
                block: TranscriptBlock::Message {
                    role: TranscriptRole::Assistant,
                    ..
                }
            }
        )),
        "expected a coalesced assistant message block, got {page:?}"
    );

    // Non-destructive: a repeat read from the same cursor returns the same entries.
    let again = node.session_history(session.clone(), 0, None, 0).await;
    assert_eq!(
        again.entries, page.entries,
        "history read must be non-destructive"
    );

    // The node publishes its verifying key so an auditor can verify the chain offline.
    let key = node.verifying_key().await;
    assert!(
        key.map(|k| !k.is_empty()).unwrap_or(false),
        "the node must publish a journal verifying key"
    );

    handle.shutdown().await;
}

/// Wire page bound (v24): a session journal holding more than WIRE_PAGE_MAX records is served in
/// <= 64-entry pages through `SessionHistory` — `max == 0` no longer returns the entire journal —
/// and the `after_cursor` loop (via `next_cursor`/`head_cursor`) reads it to completion with no
/// truncation and no duplicates.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_history_pages_past_the_wire_bound() {
    as_system(session_history_pages_past_the_wire_bound_impl()).await;
}
async fn session_history_pages_past_the_wire_bound_impl() {
    use daemon_api::{SessionApi, WIRE_PAGE_MAX};
    use daemon_common::JournalStreamId;
    use daemon_host::JournalSink;

    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode {
        node,
        handle,
        signer,
        ..
    } = assemble_over(store.clone(), 0, [0x66; 32], fast_host_config());
    let session = SessionId::new("history-paged");

    // Write 70 sealed records straight into the session's journal stream (the same sink the host
    // writes through), so the durable history holds more than one wire page.
    let sink = JournalSink::new(
        store.clone(),
        signer.clone(),
        JournalStreamId::session(&session),
    );
    for n in 0..70 {
        sink.record_management("test.rec", format!("record {n}"))
            .await
            .expect("record");
    }
    sink.seal().await.expect("seal");

    let mut cursor = 0u64;
    let mut seen = 0usize;
    let mut pages = 0usize;
    loop {
        let page = node.session_history(session.clone(), cursor, None, 0).await;
        assert!(
            page.entries.len() <= WIRE_PAGE_MAX,
            "a history page must never exceed the wire bound, got {}",
            page.entries.len()
        );
        if page.entries.is_empty() {
            break;
        }
        seen += page.entries.len();
        cursor = page.next_cursor;
        pages += 1;
        assert!(pages <= 4, "runaway pagination");
        if cursor >= page.head_cursor {
            break;
        }
    }
    assert_eq!(seen, 70, "the cursor loop must read the whole journal");
    assert!(pages >= 2, "70 records must span more than one page");

    handle.shutdown().await;
}

/// Conversation rewind (conversation-rewind spec) end-to-end over the node surface: a
/// `daemon-core` session is rewindable, `RewindTo` emits `Rewound`, the durable journal records
/// the seal (`JournalPageView::sealed_after`), and a follow-up `StartTurn` re-runs from the anchor.
#[tokio::test]
async fn rewind_to_seals_history_and_reruns_over_node() {
    as_system(rewind_to_seals_history_and_reruns_over_node_impl()).await;
}
async fn rewind_to_seals_history_and_reruns_over_node_impl() {
    use daemon_api::{ControlApi, SessionApi};
    use daemon_common::ReqId;
    use daemon_protocol::{AgentCommand, AgentEvent, Outbound, RewindAnchor, UserMsg};

    async fn drive_to_finished(node: &Arc<NodeApiImpl>, session: &SessionId) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            let drained = node.poll(session.clone(), 0).await.expect("poll");
            if drained
                .iter()
                .any(|o| matches!(o, Outbound::Event(AgentEvent::TurnFinished { .. })))
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("the turn never reached TurnFinished");
    }

    let (node, handle) = assemble();
    let session = SessionId::new("rewind-1");

    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit StartTurn");
    drive_to_finished(&node, &session).await;

    // A daemon-core session advertises itself as rewindable (durable store sessions are all
    // daemon-core-backed). A purely-live session may not be in the durable list yet; when it is,
    // it must report `rewindable = true`.
    if let Some(info) = node
        .sessions()
        .await
        .into_iter()
        .find(|s| s.session == session)
    {
        assert!(info.rewindable, "daemon-core sessions must be rewindable");
    }

    // Rewind to the first user turn; the engine emits `Rewound { to_cursor: 0 }`.
    node.submit(
        session.clone(),
        AgentCommand::RewindTo {
            anchor: RewindAnchor::UserTurn { ordinal: 0 },
            request_id: ReqId(2),
        },
    )
    .await
    .expect("submit RewindTo");

    let mut rewound = None;
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        let drained = node.poll(session.clone(), 0).await.expect("poll");
        if let Some(ev) = drained.iter().find_map(|o| match o {
            Outbound::Event(AgentEvent::Rewound {
                to_cursor, epoch, ..
            }) => Some((*to_cursor, *epoch)),
            _ => None,
        }) {
            rewound = Some(ev);
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let (to_cursor, _epoch) = rewound.expect("Rewound event observed");
    assert_eq!(to_cursor, 0, "rewound to the first user turn");

    // The durable journal records the seal so a reconnecting client sees the boundary.
    let mut sealed = None;
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        let page = node.session_history(session.clone(), 0, None, 0).await;
        if page.sealed_after.is_some() {
            sealed = page.sealed_after;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        sealed.is_some(),
        "session_history must flag the rewind seal"
    );

    // A follow-up StartTurn replays from the rewound point (the engine is idle and accepts it).
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("again"),
            request_id: ReqId(3),
        },
    )
    .await
    .expect("submit StartTurn after rewind");
    drive_to_finished(&node, &session).await;

    handle.shutdown().await;
}

/// The wire-level rewind verb (`ApiRequest::Rewind` → `ControlApi::rewind`) end-to-end, on BOTH
/// store backends: the edit/retry contract a thin client composes is `Rewind` (truncate-only,
/// reply `Ok`) followed by a separate `StartTurn` carrying the replacement text. Asserts the
/// durable journal records the truncation seal (`sealed_after`) and the follow-up turn runs to
/// completion on the truncated history. (`RewindTo`-over-submit is covered above; this is the
/// `Call`-dispatched verb the GUI/TUI actually send.)
#[tokio::test]
async fn rewind_verb_seals_history_and_accepts_a_follow_up_turn() {
    for store in [
        Arc::new(InMemoryStore::new()) as Arc<dyn SessionStore>,
        Arc::new(SqliteStore::open_in_memory().expect("open sqlite store"))
            as Arc<dyn SessionStore>,
    ] {
        as_system(rewind_verb_seals_history_and_accepts_a_follow_up_turn_impl(
            store,
        ))
        .await;
    }
}
async fn rewind_verb_seals_history_and_accepts_a_follow_up_turn_impl(store: Arc<dyn SessionStore>) {
    use daemon_api::{dispatch, ApiRequest, ApiResponse, RewindPoint, SessionApi};
    use daemon_common::ReqId;
    use daemon_protocol::{AgentCommand, AgentEvent, Outbound, RewindAnchor, UserMsg};

    async fn drive_to_finished(node: &Arc<NodeApiImpl>, session: &SessionId) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            let drained = node.poll(session.clone(), 0).await.expect("poll");
            if drained
                .iter()
                .any(|o| matches!(o, Outbound::Event(AgentEvent::TurnFinished { .. })))
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("the turn never reached TurnFinished");
    }

    let AssembledNode { node, handle, .. } =
        assemble_over(store, 0, [0x6a; 32], fast_host_config());
    let session = SessionId::new("rewind-verb-1");

    // Two exchanges, so the rewind genuinely discards history (the second one).
    for (n, text) in [(1u64, "first question"), (2, "second question")] {
        node.submit(
            session.clone(),
            AgentCommand::StartTurn {
                input: UserMsg::new(text),
                request_id: ReqId(n),
            },
        )
        .await
        .expect("submit StartTurn");
        drive_to_finished(&node, &session).await;
    }

    // The wire verb: truncate back to the FIRST user turn. Truncate-only — the reply is a bare
    // `Ok`; any replacement text is the client's follow-up `StartTurn`.
    let res = dispatch(
        node.as_ref(),
        ApiRequest::Rewind {
            session: session.clone(),
            point: RewindPoint {
                anchor: RewindAnchor::UserTurn { ordinal: 0 },
                restore_workspace: false,
            },
        },
    )
    .await;
    assert!(
        matches!(res, ApiResponse::Ok),
        "Rewind replies a bare Ok, got {res:?}"
    );

    // The durable journal records the truncation seal for reconnecting clients.
    let mut sealed = None;
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        let page = node.session_history(session.clone(), 0, None, 0).await;
        if page.sealed_after.is_some() {
            sealed = page.sealed_after;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        sealed.is_some(),
        "session_history must flag the rewind seal"
    );

    // The follow-up StartTurn (the edited/replayed text) runs on the truncated history.
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("first question, edited"),
            request_id: ReqId(3),
        },
    )
    .await
    .expect("submit StartTurn after the wire rewind");
    drive_to_finished(&node, &session).await;

    handle.shutdown().await;
}
