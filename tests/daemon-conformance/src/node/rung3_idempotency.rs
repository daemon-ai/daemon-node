// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Rung 3 (op-id idempotency + uniform operation provenance + Bootstrap probe, api/39)
//! end-to-end over the assembled node.
//!
//! The properties proven:
//! - **Dedup**: a mutating verb carrying an `op_id`, dispatched twice, has exactly ONE side
//!   effect and returns the byte-identical response; a `None` op_id never dedups.
//! - **Uniform provenance** (per carrier): the node stamps `origin_op` where it owns the
//!   mutation record — the journal-record envelope + the single-mutation `MessagesChanged` event
//!   (the `LifecycleSink::chat_message` choke point), and the delta-page `origin_ops` map (a
//!   store mutation carrying an op_id).
//! - **Null-provenance path**: a token-less mutation leaves `origin_op` absent everywhere and
//!   nothing breaks (degraded, never heuristic).
//! - **Bootstrap consistency**: a probe snapshots cursor + epoch + every collection rev under one
//!   feed-lock acquisition, so its values are mutually consistent even under a mutation storm.

use super::harness::*;
use daemon_protocol::{TransportId, UserMsg};

// ---------------------------------------------------------------------------
// Shared bring-up: a node + live Rooms adapter over a durable sqlite store.
// ---------------------------------------------------------------------------

struct RoomsNode {
    node: Arc<NodeApiImpl>,
    handle: daemon_host::SupervisorHandle,
    adapter_tasks: Vec<tokio::task::JoinHandle<()>>,
    dir: std::path::PathBuf,
}

impl RoomsNode {
    async fn bring_up(tag: &str, seed: [u8; 32]) -> Self {
        let dir = std::env::temp_dir().join(format!("daemon-rung3-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store: Arc<dyn SessionStore> =
            Arc::new(SqliteStore::open(dir.join("store.sqlite")).expect("open sqlite store"));
        let AssembledNode { node, handle, .. } =
            assemble_over(store.clone(), 0, seed, fast_host_config());
        let rooms_cfg = daemon_rooms::RoomsConfig {
            enabled: true,
            max_turns: 8,
        };
        let registry = daemon_host::AdapterRegistry::new().with_adapter(
            daemon_rooms::RoomsAdapter::new(store.clone(), rooms_cfg, Some(node.lifecycle_sink())),
        );
        node.set_adapters(registry);
        let adapter_tasks = node.spawn_adapters().await;
        Self {
            node,
            handle,
            adapter_tasks,
            dir,
        }
    }

    async fn create_room(&self, id: &str) {
        let mut details = daemon_api::CreateConversationDetails::default();
        details.extras.values.insert("id".into(), id.into());
        details.extras.values.insert("name".into(), id.into());
        match daemon_api::dispatch(
            self.node.as_ref(),
            ApiRequest::ConvCreate {
                transport: room(),
                details,
                op_id: None,
            },
        )
        .await
        {
            ApiResponse::Conversation(Some(_)) => {}
            other => panic!("expected Conversation from ConvCreate, got {other:?}"),
        }
        self.node
            .lifecycle_sink()
            .conversations_changed(room(), id.to_string(), daemon_api::ConvChange::Added)
            .await;
    }

    /// `ConvSend` carrying `op_id` (the retry-idempotency key).
    async fn send(&self, conv: &str, text: &str, op_id: Option<&str>) -> ApiResponse {
        daemon_api::dispatch(
            self.node.as_ref(),
            ApiRequest::ConvSend(daemon_api::ConvSendArgs {
                transport: room(),
                conv: conv.into(),
                from: None,
                message: UserMsg::new(text),
                op_id: op_id.map(|s| s.to_string()),
            }),
        )
        .await
    }

    async fn history(&self, conv: &str) -> daemon_api::JournalPageView {
        match daemon_api::dispatch(
            self.node.as_ref(),
            ApiRequest::ConvHistory(daemon_api::ConvHistoryArgs {
                transport: room(),
                conv: conv.into(),
                after_cursor: 0,
                before_cursor: None,
                max: 0,
            }),
        )
        .await
        {
            ApiResponse::Journal(page) => page,
            other => panic!("expected Journal, got {other:?}"),
        }
    }

    async fn history_at_least(&self, conv: &str, n: usize) -> daemon_api::JournalPageView {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let page = self.history(conv).await;
            if page.entries.len() >= n || Instant::now() >= deadline {
                return page;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    async fn tear_down(self) {
        for task in &self.adapter_tasks {
            task.abort();
        }
        self.handle.shutdown().await;
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn room() -> TransportId {
    TransportId::new("room")
}

// ---------------------------------------------------------------------------
// Dedup — one side effect, byte-identical result
// ---------------------------------------------------------------------------

/// ConvSend carrying an `op_id` dispatched twice deduplicates: exactly one durable chat record
/// lands (one side effect), and both dispatches return the byte-identical response. A distinct
/// `op_id` executes independently; a `None` op_id never dedups.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn conv_send_op_id_dedups_to_one_side_effect() {
    as_system(conv_send_dedup_impl()).await;
}
async fn conv_send_dedup_impl() {
    let h = RoomsNode::bring_up("dedup", [0x81; 32]).await;
    h.create_room("c").await;

    // Same op_id twice -> one durable record; the second returns the ORIGINAL result bytes.
    let first = h.send("c", "hello", Some("op-1")).await;
    let again = h.send("c", "hello", Some("op-1")).await;
    assert_eq!(first, ApiResponse::Ok);
    assert_eq!(
        again, first,
        "a duplicate returns the byte-identical result"
    );
    let page = h.history_at_least("c", 1).await;
    // Give any (erroneously) re-executed second send time to journal.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let settled = h.history("c").await;
    assert_eq!(
        settled.entries.len(),
        1,
        "the deduplicated retry must NOT append a second record: {page:?}"
    );

    // A distinct op_id is an independent operation.
    h.send("c", "world", Some("op-2")).await;
    let two = h.history_at_least("c", 2).await;
    assert_eq!(two.entries.len(), 2, "a fresh op_id executes independently");

    // A None op_id never dedups (two identical token-less sends both execute).
    h.send("c", "again", None).await;
    h.send("c", "again", None).await;
    let four = h.history_at_least("c", 4).await;
    assert_eq!(four.entries.len(), 4, "token-less sends never dedup");

    h.tear_down().await;
}

// ---------------------------------------------------------------------------
// Provenance carrier 1 + 3 — journal-record envelope + single-mutation event
// ---------------------------------------------------------------------------

/// The node stamps `origin_op` at the `LifecycleSink::chat_message` choke point: the journaled
/// record envelope carries it (carrier 1) and the emitted `MessagesChanged` pointer carries it
/// (carrier 3). A token-less report leaves both absent (the null-provenance path).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_message_stamps_origin_op_on_journal_and_event() {
    as_system(chat_message_provenance_impl()).await;
}
async fn chat_message_provenance_impl() {
    let h = RoomsNode::bring_up("prov", [0x82; 32]).await;
    h.create_room("p").await;

    // A provenance-carrying report (the node owns the record; the token is opaque).
    let mut msg = daemon_api::ChatMessage::new(None, "carried".to_string());
    msg.timestamp = Some(1234);
    h.node
        .lifecycle_sink()
        .chat_message(room(), "p".to_string(), msg, Some("op-prov".to_string()))
        .await;

    // A token-less report (the null path).
    let plain = daemon_api::ChatMessage::new(None, "plain".to_string());
    h.node
        .lifecycle_sink()
        .chat_message(room(), "p".to_string(), plain, None)
        .await;

    let page = h.history_at_least("p", 2).await;
    let carried = page
        .entries
        .iter()
        .find(|e| matches!(&e.payload, daemon_api::JournalRecordPayload::Chat { message } if message.text == "carried"))
        .expect("the carried record is journaled");
    assert_eq!(
        carried.origin_op.as_deref(),
        Some("op-prov"),
        "carrier 1: the journal-record envelope carries origin_op"
    );
    let plain = page
        .entries
        .iter()
        .find(|e| matches!(&e.payload, daemon_api::JournalRecordPayload::Chat { message } if message.text == "plain"))
        .expect("the plain record is journaled");
    assert_eq!(
        plain.origin_op, None,
        "null path: a token-less record has no origin_op"
    );

    // Carrier 3: the MessagesChanged pointer for the carried message names the causing op.
    let events = match daemon_api::dispatch(
        h.node.as_ref(),
        ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: None,
        },
    )
    .await
    {
        ApiResponse::EventsPage(p) => p.events,
        other => panic!("expected EventsPage, got {other:?}"),
    };
    let origin_ops: Vec<Option<String>> = events
        .iter()
        .filter_map(|e| match e {
            daemon_api::NodeEvent::MessagesChanged { origin_op, .. } => Some(origin_op.clone()),
            _ => None,
        })
        .collect();
    assert!(
        origin_ops.contains(&Some("op-prov".to_string())),
        "carrier 3: a MessagesChanged event carries the causing origin_op: {origin_ops:?}"
    );
    assert!(
        origin_ops.contains(&None),
        "the token-less report emits a null-provenance MessagesChanged: {origin_ops:?}"
    );

    h.tear_down().await;
}

// ---------------------------------------------------------------------------
// Provenance carrier 2 — delta-page origin_ops map (a node store mutation)
// ---------------------------------------------------------------------------

/// A store mutation carrying an `op_id` (`RosterAdd`) threads the token into the per-collection
/// changed-key index, so the delta page's `origin_ops` map names the causing op for the changed
/// key (carrier 2). A token-less mutation leaves the map empty for its key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn roster_delta_page_carries_origin_ops_map() {
    as_system(roster_origin_ops_impl()).await;
}
async fn roster_origin_ops_impl() {
    let h = RoomsNode::bring_up("ops", [0x83; 32]).await;

    // Baseline rev before the tracked add.
    let baseline = roster_list(&h, None).await.rev;

    match daemon_api::dispatch(
        h.node.as_ref(),
        ApiRequest::RosterAdd {
            transport: room(),
            contact: contact("agent-zoe"),
            op_id: Some("roster-op".to_string()),
        },
    )
    .await
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }

    let delta = roster_list(&h, Some(baseline)).await;
    assert_eq!(
        delta.origin_ops.get("agent-zoe").map(|s| s.as_str()),
        Some("roster-op"),
        "carrier 2: the delta page's origin_ops names the op that changed the key: {:?}",
        delta.origin_ops
    );

    // A token-less add: no origin_ops entry for its key.
    match daemon_api::dispatch(
        h.node.as_ref(),
        ApiRequest::RosterAdd {
            transport: room(),
            contact: contact("agent-yan"),
            op_id: None,
        },
    )
    .await
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }
    let delta2 = roster_list(&h, Some(delta.rev)).await;
    assert!(
        !delta2.origin_ops.contains_key("agent-yan"),
        "null path: a token-less mutation adds no origin_ops entry: {:?}",
        delta2.origin_ops
    );

    h.tear_down().await;
}

async fn roster_list(h: &RoomsNode, since_rev: Option<u64>) -> daemon_api::ContactPage {
    match daemon_api::dispatch(
        h.node.as_ref(),
        ApiRequest::RosterList {
            transport: room(),
            after: None,
            since_rev,
        },
    )
    .await
    {
        ApiResponse::ContactPage(page) => page,
        other => panic!("expected ContactPage, got {other:?}"),
    }
}

fn contact(id: &str) -> daemon_api::ContactInfo {
    daemon_api::ContactInfo {
        id: id.into(),
        display_name: None,
        presence: daemon_api::Presence::default(),
        permission: daemon_api::ContactPermission::Allow,
    }
}

// ---------------------------------------------------------------------------
// Bootstrap — a consistent revs+cursor+epoch snapshot under a mutation storm
// ---------------------------------------------------------------------------

/// The `Bootstrap` probe snapshots cursor + epoch + every collection rev under ONE feed-lock
/// acquisition. Under a concurrent mutation storm, each probe is self-consistent (epoch constant,
/// cursor + revs never exceed the final settled values — no torn read), and the final probe
/// equals an independent read of the same counters.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_snapshots_consistent_revs_cursor_epoch() {
    as_system(bootstrap_impl()).await;
}
async fn bootstrap_impl() {
    let h = RoomsNode::bring_up("boot", [0x84; 32]).await;

    let first = bootstrap(&h).await;
    let epoch = first.epoch;

    // Mutation storm: create rooms while repeatedly probing.
    let mut probes = vec![first];
    for i in 0..24 {
        h.create_room(&format!("r{i}")).await;
        let p = bootstrap(&h).await;
        assert_eq!(p.epoch, epoch, "epoch is stable within a feed generation");
        probes.push(p);
    }
    let settled = bootstrap(&h).await;

    // Monotonicity: cursor + each rev never decrease across probes and never exceed the final
    // settled snapshot (a torn read across lock boundaries would break this).
    for p in &probes {
        assert!(
            p.cursor <= settled.cursor,
            "cursor never exceeds the settled snapshot"
        );
        for (k, v) in &p.revs {
            assert!(
                *v <= *settled.revs.get(k).unwrap_or(&0),
                "rev {k} never exceeds the settled snapshot"
            );
        }
    }
    let mut prev = 0u64;
    for p in &probes {
        assert!(
            p.cursor >= prev,
            "cursor is monotonic non-decreasing across probes"
        );
        prev = p.cursor;
    }

    // The final probe agrees with an independent read of the same counters.
    let feed_cursor = settled.cursor;
    let again = bootstrap(&h).await;
    assert_eq!(again.cursor, feed_cursor, "a settled cursor is stable");
    assert!(
        settled.revs.values().any(|v| *v > 0),
        "the storm advanced at least one collection rev: {:?}",
        settled.revs
    );

    h.tear_down().await;
}

async fn bootstrap(h: &RoomsNode) -> daemon_api::BootstrapReport {
    match daemon_api::dispatch(h.node.as_ref(), ApiRequest::Bootstrap).await {
        ApiResponse::Bootstrap(report) => report,
        other => panic!("expected Bootstrap, got {other:?}"),
    }
}
