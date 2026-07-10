// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Rung 2 (delta reads + generalized backward windows, api/39) end-to-end over the assembled
//! node — the SessionsQuery template cloned onto ConvList / RosterList / PersonList, and the
//! `before_cursor` backward window over the durable journal reads (ConvHistory / SessionHistory /
//! UnitHistory).
//!
//! The delta properties proven per collection:
//! - equivalence: a baseline full read + an applied delta (upsert `items`, prune `removed`)
//!   reconstructs exactly a fresh full read;
//! - tombstone delivery: a removal after `since_rev` rides `removed`;
//! - unservable fallback: an ahead-of-rev `since_rev` (the post-restart signature — in-memory
//!   counters reset) degrades to a full page with the correct current rev and no removals.
//!
//! The backward-window properties proven per read:
//! - newest-anchored: `before_cursor = u64::MAX` returns the newest `max` entries in one
//!   round-trip, ascending;
//! - continuation: `next_cursor` (the oldest returned cursor) chains contiguous pages with no
//!   duplicate and no skip, down to an empty page;
//! - stable anchoring: records landing mid-walk never disturb pages below a served anchor (they
//!   surface through the forward read past the old head instead).

use super::harness::*;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Shared bring-up: a node + live Rooms adapter over a durable sqlite store
// (rooms persist; InMemoryStore's `room_*` are no-ops), driven in-process.
// ---------------------------------------------------------------------------

struct RoomsNode {
    node: Arc<NodeApiImpl>,
    handle: daemon_host::SupervisorHandle,
    adapter_tasks: Vec<tokio::task::JoinHandle<()>>,
    dir: std::path::PathBuf,
}

impl RoomsNode {
    async fn bring_up(tag: &str, seed: [u8; 32]) -> Self {
        let dir = std::env::temp_dir().join(format!("daemon-rung2-{tag}-{}", std::process::id()));
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

    /// Create a members-less room `id` and report the set change through the public
    /// [`LifecycleSink`] seam — exactly what a fully-wired messaging adapter does (the delta
    /// index is node bookkeeping keyed off these emissions; the rooms reference adapter does not
    /// emit them itself yet).
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

    /// Delete room `id` and report the removal through the sink (see [`Self::create_room`]).
    async fn delete_room(&self, id: &str) {
        match daemon_api::dispatch(
            self.node.as_ref(),
            ApiRequest::ConvDelete {
                transport: room(),
                conv: id.to_string(),
            },
        )
        .await
        {
            ApiResponse::Ok => {}
            other => panic!("expected Ok from ConvDelete, got {other:?}"),
        }
        self.node
            .lifecycle_sink()
            .conversations_changed(room(), id.to_string(), daemon_api::ConvChange::Removed)
            .await;
    }

    /// One `ConvList` page (in-process dispatch).
    async fn conv_list(&self, since_rev: Option<u64>) -> daemon_api::ConvPage {
        match daemon_api::dispatch(
            self.node.as_ref(),
            ApiRequest::ConvList {
                transport: room(),
                after: None,
                since_rev,
            },
        )
        .await
        {
            ApiResponse::Conversations(page) => page,
            other => panic!("expected Conversations, got {other:?}"),
        }
    }

    /// `ConvSend` to `conv` (the journal append lands on the adapter's serve loop).
    async fn send(&self, conv: &str, text: &str) {
        use daemon_protocol::UserMsg;
        match daemon_api::dispatch(
            self.node.as_ref(),
            ApiRequest::ConvSend(daemon_api::ConvSendArgs {
                transport: room(),
                conv: conv.into(),
                from: None,
                message: UserMsg::new(text),
                op_id: None,
            }),
        )
        .await
        {
            ApiResponse::Ok => {}
            other => panic!("expected Ok from ConvSend, got {other:?}"),
        }
    }

    /// One `ConvHistory` page.
    async fn history(
        &self,
        conv: &str,
        after_cursor: u64,
        before_cursor: Option<u64>,
        max: u32,
    ) -> daemon_api::JournalPageView {
        match daemon_api::dispatch(
            self.node.as_ref(),
            ApiRequest::ConvHistory(daemon_api::ConvHistoryArgs {
                transport: room(),
                conv: conv.into(),
                after_cursor,
                before_cursor,
                max,
            }),
        )
        .await
        {
            ApiResponse::Journal(page) => page,
            other => panic!("expected Journal, got {other:?}"),
        }
    }

    /// Poll the forward read until `conv` holds at least `n` journaled entries (the send path
    /// journals on the adapter's async serve loop), returning the settled full page.
    async fn history_at_least(&self, conv: &str, n: usize) -> daemon_api::JournalPageView {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let page = self.history(conv, 0, None, 0).await;
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

fn room() -> daemon_protocol::TransportId {
    daemon_protocol::TransportId::new("room")
}

/// Apply a delta page onto a keyed baseline: prune `removed`, then upsert `items` — the client's
/// replace semantics for a delta read (a full page instead replaces the whole map).
fn apply_conv_delta(
    baseline: &mut BTreeMap<String, daemon_api::ConversationInfo>,
    page: &daemon_api::ConvPage,
) {
    for id in &page.removed {
        baseline.remove(id);
    }
    for item in &page.items {
        baseline.insert(item.id.clone(), item.clone());
    }
}

// ---------------------------------------------------------------------------
// Delta reads: ConvList
// ---------------------------------------------------------------------------

/// ConvList delta (rung 2): a servable `since_rev` returns exactly the conversations changed
/// after it plus `removed` tombstones; baseline + delta ≡ a fresh full read; a delta anchored at
/// the current rev is empty (the cheap "nothing changed" round-trip).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn conv_list_delta_serves_changes_tombstones_and_equivalence() {
    as_system(conv_list_delta_impl()).await;
}
async fn conv_list_delta_impl() {
    let h = RoomsNode::bring_up("convdelta", [0x71; 32]).await;

    // Baseline: two rooms, taken as a FULL page (the client's initial sync).
    h.create_room("d1").await;
    h.create_room("d2").await;
    let full = h.conv_list(None).await;
    assert!(full.removed.is_empty(), "a full page carries no removals");
    let baseline_rev = full.rev;
    assert!(baseline_rev >= 2, "two Added emissions bumped the rev");
    let mut baseline: BTreeMap<String, daemon_api::ConversationInfo> =
        full.items.into_iter().map(|c| (c.id.clone(), c)).collect();
    assert_eq!(baseline.len(), 2);

    // Mutate past the baseline: one add + one remove.
    h.create_room("d3").await;
    h.delete_room("d1").await;

    // The delta read: only the changed conversation rides `items`; the removal is a tombstone.
    let delta = h.conv_list(Some(baseline_rev)).await;
    assert!(
        delta.rev > baseline_rev,
        "the delta reflects a later revision"
    );
    let ids: Vec<&str> = delta.items.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["d3"],
        "exactly the conversations changed after since_rev ride the delta page"
    );
    assert_eq!(
        delta.removed,
        vec!["d1".to_string()],
        "the removal after since_rev rides `removed`"
    );

    // Equivalence: baseline + applied delta == a fresh full read.
    apply_conv_delta(&mut baseline, &delta);
    let fresh = h.conv_list(None).await;
    let fresh_ids: Vec<&str> = fresh.items.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(
        baseline.keys().map(|k| k.as_str()).collect::<Vec<_>>(),
        fresh_ids,
        "baseline + delta must reconstruct the fresh full read"
    );

    // Anchored at the current rev: the empty delta (skip-if-unchanged over the read path).
    let empty = h.conv_list(Some(delta.rev)).await;
    assert_eq!(empty.rev, delta.rev);
    assert!(empty.items.is_empty(), "nothing changed since the head rev");
    assert!(empty.removed.is_empty());

    h.tear_down().await;
}

/// ConvList fallback (rung 2): an unservable `since_rev` — ahead of the node's rev, the exact
/// signature of a client holding a pre-restart revision (in-memory counters reset on restart) —
/// degrades to a FULL page: every conversation, no removals, the correct current rev.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn conv_list_unservable_since_rev_falls_back_to_full_page() {
    as_system(conv_list_unservable_impl()).await;
}
async fn conv_list_unservable_impl() {
    let h = RoomsNode::bring_up("convfall", [0x72; 32]).await;
    h.create_room("f1").await;
    h.create_room("f2").await;
    let current = h.conv_list(None).await.rev;

    let page = h.conv_list(Some(1_000_000)).await;
    let ids: Vec<&str> = page.items.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["f1", "f2"],
        "an unservable since_rev must serve the full page"
    );
    assert!(page.removed.is_empty(), "a full page carries no removals");
    assert_eq!(page.rev, current, "the fallback page reflects the real rev");

    h.tear_down().await;
}

// ---------------------------------------------------------------------------
// Delta reads: RosterList (server-side contacts; the real rooms SupportsRoster)
// ---------------------------------------------------------------------------

async fn roster_add(h: &RoomsNode, id: &str, name: &str) {
    match daemon_api::dispatch(
        h.node.as_ref(),
        ApiRequest::RosterAdd {
            transport: room(),
            contact: contact(id, Some(name)),
            op_id: None,
        },
    )
    .await
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok from RosterAdd, got {other:?}"),
    }
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

fn contact(id: &str, name: Option<&str>) -> daemon_api::ContactInfo {
    daemon_api::ContactInfo {
        id: id.into(),
        display_name: name.map(|s| s.into()),
        presence: daemon_api::Presence::default(),
        permission: daemon_api::ContactPermission::Allow,
    }
}

/// RosterList delta (rung 2): changed contacts + removal tombstones past `since_rev`, the
/// baseline+delta equivalence, and the unservable fallback — through the REAL node-mediated
/// roster mutation paths (`RosterAdd`/`RosterUpdate`/`RosterRemove` emit `ContactsChanged`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn roster_list_delta_serves_changes_tombstones_and_fallback() {
    as_system(roster_list_delta_impl()).await;
}
async fn roster_list_delta_impl() {
    let h = RoomsNode::bring_up("rostdelta", [0x73; 32]).await;

    // Baseline: two contacts, taken as a full page.
    roster_add(&h, "agent-alice", "Alice").await;
    roster_add(&h, "agent-bob", "Bob").await;
    let full = roster_list(&h, None).await;
    let baseline_rev = full.rev;
    assert_eq!(baseline_rev, 2, "two adds bumped the roster rev twice");
    let mut baseline: BTreeMap<String, daemon_api::ContactInfo> =
        full.items.into_iter().map(|c| (c.id.clone(), c)).collect();

    // Mutate past the baseline: update bob (a change), add carol, remove alice (a tombstone).
    match daemon_api::dispatch(
        h.node.as_ref(),
        ApiRequest::RosterUpdate {
            transport: room(),
            contact: contact("agent-bob", Some("Bobby")),
            op_id: None,
        },
    )
    .await
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok from RosterUpdate, got {other:?}"),
    }
    roster_add(&h, "agent-carol", "Carol").await;
    match daemon_api::dispatch(
        h.node.as_ref(),
        ApiRequest::RosterRemove {
            transport: room(),
            contact: contact("agent-alice", None),
            op_id: None,
        },
    )
    .await
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok from RosterRemove, got {other:?}"),
    }

    // The delta: exactly {bob (updated), carol (added)} + tombstone {alice}.
    let delta = roster_list(&h, Some(baseline_rev)).await;
    assert_eq!(delta.rev, baseline_rev + 3, "three mutations, three bumps");
    let ids: Vec<&str> = delta.items.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["agent-bob", "agent-carol"],
        "the changed + still-present contacts, id order"
    );
    assert_eq!(
        delta
            .items
            .iter()
            .find(|c| c.id == "agent-bob")
            .and_then(|c| c.display_name.as_deref()),
        Some("Bobby"),
        "the delta carries the contact's NEW state"
    );
    assert_eq!(delta.removed, vec!["agent-alice".to_string()]);

    // Equivalence: baseline + delta == fresh full read.
    for id in &delta.removed {
        baseline.remove(id);
    }
    for item in &delta.items {
        baseline.insert(item.id.clone(), item.clone());
    }
    let fresh = roster_list(&h, None).await;
    let fresh_ids: Vec<&str> = fresh.items.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(
        baseline.keys().map(|k| k.as_str()).collect::<Vec<_>>(),
        fresh_ids
    );

    // Unservable fallback: an ahead-of-rev anchor serves the full page (no removals, real rev).
    let fallback = roster_list(&h, Some(999_999)).await;
    assert_eq!(
        fallback.items.len(),
        fresh.items.len(),
        "the fallback is a full page"
    );
    assert!(fallback.removed.is_empty());
    assert_eq!(fallback.rev, fresh.rev);

    h.tear_down().await;
}

// ---------------------------------------------------------------------------
// Delta reads: PersonList (the node person registry; no adapter needed)
// ---------------------------------------------------------------------------

async fn person_list(
    node: &Arc<NodeApiImpl>,
    since_rev: Option<u64>,
) -> daemon_api::RevDeltaList<daemon_api::Person> {
    match daemon_api::dispatch(node.as_ref(), ApiRequest::PersonList { since_rev }).await {
        ApiResponse::Persons(list) => list,
        other => panic!("expected Persons, got {other:?}"),
    }
}

fn person(id: &str) -> daemon_api::Person {
    daemon_api::Person {
        id: id.into(),
        alias: None,
        avatar: None,
        endpoints: Vec::new(),
    }
}

/// PersonList delta (rung 2): changed persons (add + endpoint association) and removal
/// tombstones past `since_rev`, baseline+delta equivalence, the empty at-head delta, and the
/// unservable fallback — through the real person-registry emit paths.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn person_list_delta_serves_changes_tombstones_and_fallback() {
    as_system(person_list_delta_impl()).await;
}
async fn person_list_delta_impl() {
    use daemon_protocol::TransportId;

    let (node, handle) = assemble();

    // Baseline: two persons, full read.
    node.person_add(person("p-ada"));
    node.person_add(person("p-bob"));
    let full = person_list(&node, None).await;
    let baseline_rev = full.rev;
    assert_eq!(baseline_rev, 2);
    assert!(full.removed.is_empty(), "a full list carries no removals");
    let mut baseline: BTreeMap<String, daemon_api::Person> =
        full.items.into_iter().map(|p| (p.id.clone(), p)).collect();

    // Mutate: associate an endpoint with ada (a change), add carol, remove bob (a tombstone).
    assert!(node.person_associate(
        "p-ada",
        daemon_api::PersonEndpoint::new(
            TransportId::new("matrix/@me:hs.org"),
            contact("@ada:hs.org", Some("Ada")),
        ),
    ));
    node.person_add(person("p-carol"));
    assert!(node.person_remove("p-bob", false));

    // The delta: {ada (endpoint change), carol (added)} + tombstone {bob}.
    let delta = person_list(&node, Some(baseline_rev)).await;
    assert_eq!(delta.rev, baseline_rev + 3);
    let mut ids: Vec<&str> = delta.items.iter().map(|p| p.id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec!["p-ada", "p-carol"]);
    assert_eq!(
        delta
            .items
            .iter()
            .find(|p| p.id == "p-ada")
            .map(|p| p.endpoints.len()),
        Some(1),
        "the delta carries the person's NEW state (the associated endpoint)"
    );
    assert_eq!(delta.removed, vec!["p-bob".to_string()]);

    // Equivalence: baseline + delta == fresh full read (as key sets + endpoint state).
    for id in &delta.removed {
        baseline.remove(id);
    }
    for item in &delta.items {
        baseline.insert(item.id.clone(), item.clone());
    }
    let fresh = person_list(&node, None).await;
    let mut fresh_ids: Vec<&str> = fresh.items.iter().map(|p| p.id.as_str()).collect();
    fresh_ids.sort_unstable();
    let mut baseline_ids: Vec<&str> = baseline.keys().map(|k| k.as_str()).collect();
    baseline_ids.sort_unstable();
    assert_eq!(baseline_ids, fresh_ids);

    // At-head delta: empty items + removals, same rev.
    let empty = person_list(&node, Some(delta.rev)).await;
    assert_eq!(empty.rev, delta.rev);
    assert!(empty.items.is_empty() && empty.removed.is_empty());

    // Unservable fallback: full list, no removals, real rev.
    let fallback = person_list(&node, Some(1_000_000)).await;
    assert_eq!(fallback.items.len(), fresh.items.len());
    assert!(fallback.removed.is_empty());
    assert_eq!(fallback.rev, fresh.rev);

    handle.shutdown().await;
}

// ---------------------------------------------------------------------------
// Backward windows: ConvHistory (the durable conv journal, via the live send path)
// ---------------------------------------------------------------------------

/// ConvHistory backward windows (rung 2): `before_cursor = u64::MAX` anchors the newest window in
/// one round-trip; `next_cursor` chains contiguous pages (no dupes, no skips) down to empty; and
/// records landing mid-walk never disturb pages below a served anchor — they surface through the
/// forward read past the old head instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn conv_history_backward_windows_page_and_anchor_stably() {
    as_system(conv_history_backward_impl()).await;
}
async fn conv_history_backward_impl() {
    let h = RoomsNode::bring_up("convbwd", [0x74; 32]).await;
    h.create_room("bw").await;

    let total = 7usize;
    for i in 0..total {
        h.send("bw", &format!("m{i}")).await;
    }
    let full = h.history_at_least("bw", total).await;
    assert_eq!(full.entries.len(), total, "all sends journaled: {full:?}");
    let cursors: Vec<u64> = full.entries.iter().map(|e| e.cursor).collect();
    let head = full.head_cursor;
    assert_eq!(cursors.last().copied(), Some(head));

    // The newest window in one round-trip, ascending, with the backward continuation.
    let page1 = h.history("bw", 0, Some(u64::MAX), 3).await;
    let got1: Vec<u64> = page1.entries.iter().map(|e| e.cursor).collect();
    assert_eq!(got1, cursors[4..7], "the 3 newest entries, ascending");
    assert_eq!(page1.head_cursor, head);
    assert_eq!(
        page1.next_cursor, cursors[4],
        "next_cursor = the oldest returned cursor (the next before_cursor)"
    );

    // Interleaved writes land mid-walk; the continuation below the anchor is untouched.
    for i in total..total + 2 {
        h.send("bw", &format!("m{i}")).await;
    }
    h.history_at_least("bw", total + 2).await;

    let page2 = h.history("bw", 0, Some(page1.next_cursor), 3).await;
    let got2: Vec<u64> = page2.entries.iter().map(|e| e.cursor).collect();
    assert_eq!(
        got2,
        cursors[1..4],
        "pages below a served anchor must not shift under interleaved appends"
    );
    let page3 = h.history("bw", 0, Some(page2.next_cursor), 3).await;
    let got3: Vec<u64> = page3.entries.iter().map(|e| e.cursor).collect();
    assert_eq!(got3, cursors[0..1], "the final, short page");

    // Termination: an anchor at the oldest cursor yields the empty page echoing the anchor.
    let done = h.history("bw", 0, Some(page3.next_cursor), 3).await;
    assert!(done.entries.is_empty(), "nothing below the oldest cursor");
    assert_eq!(done.next_cursor, page3.next_cursor);

    // No dupes / no skips: the backward union is exactly the pre-append stream, and the two
    // interleaved records ride the forward read from the old head.
    let mut union: Vec<u64> = page1
        .entries
        .iter()
        .chain(&page2.entries)
        .chain(&page3.entries)
        .map(|e| e.cursor)
        .collect();
    union.sort_unstable();
    union.dedup();
    assert_eq!(union, cursors, "backward union = the anchored stream");
    let tail = h.history("bw", head, None, 0).await;
    assert_eq!(
        tail.entries.len(),
        2,
        "the interleaved appends surface past the old head (forward)"
    );

    // Ordering inside every page is strictly ascending (the client renders windows in order).
    for page in [&page1, &page2, &page3] {
        assert!(
            page.entries.windows(2).all(|w| w[0].cursor < w[1].cursor),
            "backward pages are served in ascending cursor order"
        );
    }

    h.tear_down().await;
}

// ---------------------------------------------------------------------------
// Backward windows: SessionHistory + UnitHistory (seeded via the host's JournalSink,
// exactly like history.rs's paging test)
// ---------------------------------------------------------------------------

/// SessionHistory backward windows (rung 2): the newest-anchored window over a seeded session
/// journal pages 4/4/2 with contiguous continuation and the same entries the forward read serves.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_history_backward_windows_page_newest_first() {
    as_system(session_history_backward_impl()).await;
}
async fn session_history_backward_impl() {
    use daemon_api::SessionApi;
    use daemon_common::JournalStreamId;
    use daemon_host::JournalSink;

    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode {
        node,
        handle,
        signer,
        ..
    } = assemble_over(store.clone(), 0, [0x75; 32], fast_host_config());
    let session = SessionId::new("bwd-session");

    let sink = JournalSink::new(
        store.clone(),
        signer.clone(),
        JournalStreamId::session(&session),
    );
    for n in 0..10 {
        sink.record_management("test.rec", format!("record {n}"))
            .await
            .expect("record");
    }
    sink.seal().await.expect("seal");

    let full = node.session_history(session.clone(), 0, None, 0).await;
    assert_eq!(full.entries.len(), 10, "seed sanity");
    let cursors: Vec<u64> = full.entries.iter().map(|e| e.cursor).collect();

    // Backward walk: 4 / 4 / 2, newest-anchored, contiguous, then empty.
    let p1 = node
        .session_history(session.clone(), 0, Some(u64::MAX), 4)
        .await;
    assert_eq!(
        p1.entries.iter().map(|e| e.cursor).collect::<Vec<_>>(),
        cursors[6..10]
    );
    assert_eq!(p1.head_cursor, full.head_cursor);
    let p2 = node
        .session_history(session.clone(), 0, Some(p1.next_cursor), 4)
        .await;
    assert_eq!(
        p2.entries.iter().map(|e| e.cursor).collect::<Vec<_>>(),
        cursors[2..6]
    );
    let p3 = node
        .session_history(session.clone(), 0, Some(p2.next_cursor), 4)
        .await;
    assert_eq!(
        p3.entries.iter().map(|e| e.cursor).collect::<Vec<_>>(),
        cursors[0..2],
        "the final short page"
    );
    let done = node
        .session_history(session.clone(), 0, Some(p3.next_cursor), 4)
        .await;
    assert!(done.entries.is_empty());

    // The backward union is the forward read, entry-identical (decode + verify agree).
    let union: Vec<daemon_api::JournalRecord> = p3
        .entries
        .iter()
        .chain(&p2.entries)
        .chain(&p1.entries)
        .cloned()
        .collect();
    assert_eq!(union, full.entries, "backward union == forward read");

    handle.shutdown().await;
}

/// UnitHistory backward windows (rung 2): the same newest-anchored walk over a seeded unit
/// journal stream (the fleet transcript read).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unit_history_backward_windows_page_newest_first() {
    as_system(unit_history_backward_impl()).await;
}
async fn unit_history_backward_impl() {
    use daemon_api::ControlApi;
    use daemon_common::{JournalStreamId, UnitId};
    use daemon_host::JournalSink;

    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode {
        node,
        handle,
        signer,
        ..
    } = assemble_over(store.clone(), 0, [0x76; 32], fast_host_config());
    let unit = UnitId::new("bwd-unit");

    let sink = JournalSink::new(store.clone(), signer.clone(), JournalStreamId::unit(&unit));
    for n in 0..5 {
        sink.record_management("test.rec", format!("record {n}"))
            .await
            .expect("record");
    }
    sink.seal().await.expect("seal");

    let full = node.unit_history(unit.clone(), 0, None, 0).await;
    assert_eq!(full.entries.len(), 5, "seed sanity");
    let cursors: Vec<u64> = full.entries.iter().map(|e| e.cursor).collect();

    let p1 = node.unit_history(unit.clone(), 0, Some(u64::MAX), 2).await;
    assert_eq!(
        p1.entries.iter().map(|e| e.cursor).collect::<Vec<_>>(),
        cursors[3..5],
        "the newest window"
    );
    let p2 = node
        .unit_history(unit.clone(), 0, Some(p1.next_cursor), 2)
        .await;
    assert_eq!(
        p2.entries.iter().map(|e| e.cursor).collect::<Vec<_>>(),
        cursors[1..3]
    );
    let p3 = node
        .unit_history(unit.clone(), 0, Some(p2.next_cursor), 2)
        .await;
    assert_eq!(
        p3.entries.iter().map(|e| e.cursor).collect::<Vec<_>>(),
        cursors[0..1]
    );
    let done = node
        .unit_history(unit.clone(), 0, Some(p3.next_cursor), 2)
        .await;
    assert!(done.entries.is_empty());

    handle.shutdown().await;
}
