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
    use daemon_api::{ControlApi, JournalRecordPayload, Outbound, SessionApi};
    use daemon_common::ReqId;
    use daemon_protocol::{AgentCommand, AgentEvent, TranscriptBlock, UserMsg};

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
        let p = node.session_history(session.clone(), 0, 0).await;
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
    assert!(
        page.entries.iter().any(|e| matches!(
            &e.payload,
            JournalRecordPayload::Block {
                block: TranscriptBlock::Message { .. }
            }
        )),
        "expected a coalesced assistant message block, got {page:?}"
    );

    // Non-destructive: a repeat read from the same cursor returns the same entries.
    let again = node.session_history(session.clone(), 0, 0).await;
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

/// Conversation rewind (conversation-rewind spec) end-to-end over the node surface: a
/// `daemon-core` session is rewindable, `RewindTo` emits `Rewound`, the durable journal records
/// the seal (`JournalPageView::sealed_after`), and a follow-up `StartTurn` re-runs from the anchor.
#[tokio::test]
async fn rewind_to_seals_history_and_reruns_over_node() {
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
        let page = node.session_history(session.clone(), 0, 0).await;
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
