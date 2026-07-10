// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `RoomsAdapter` — the internal Rooms transport presented as a libpurple-style
//! [`MessagingProtocol`](daemon_api::MessagingProtocol).
//!
//! This is the first consumer of the messaging-adapter interface (daemon-messaging-adapter-spec.md
//! §10.1). A **Room** is a conversation within the single `"room"` transport: management addresses it
//! as `(transport = "room", conv = <room id>)`, while `room/<id>` stays an internal delivery-routing
//! detail of the loopback transport. The adapter:
//!
//! - persists rooms + membership to the durable [`SessionStore`] (`room_set` / `room_member_set`),
//! - shares the in-memory [`Membership`] table with the live [`RoomRuntime`] (built in [`serve`]), and
//! - forwards a `ConvSend` to that runtime over an `mpsc` command channel so the floor-gated fan-out,
//!   the chat-journal report (via the node's [`LifecycleSink`], wire v38), and the agent-reply
//!   re-injection all run on the loop that owns the node `api` — the adapter struct never holds an
//!   `Arc<dyn NodeApi>`, so there is no registry<->adapter reference cycle.
//!
//! [`serve`]: RoomsAdapter::serve

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use daemon_api::{
    from_cbor, to_cbor, AccountSettingsSchema, AdapterCapabilities, AdapterInfo, ApiError,
    ChannelJoinDetails, ChatMessage, ConnectionState, ContactInfo, ContactPermission, ConvSendArgs,
    ConversationInfo, ConversationMember, ConversationOps, ConversationType,
    CreateConversationDetails, FileTransfer, FileTransferOps, LifecycleSink, MemberInviteArgs,
    MemberRemoveArgs, MemberRole, MembershipOps, MessagingProtocol, NodeApi, Participant, Presence,
    PresenceState, RosterOps, SupportsConversations, SupportsFileTransfer, SupportsMembership,
    SupportsRoster, TransportAdapter, TransportInstanceInfo, TypingState,
};
use daemon_common::SessionId;
use daemon_host::{with_request_context, BlobStore, RequestContext};
use daemon_ingest::Ingestor;
use daemon_protocol::{
    AgentEvent, RoomId, RoomMember, RoomPolicy, SenderId, SessionPayload, TransportId,
};
use daemon_store::{Room, RoomMember as StoreRoomMember, SessionStore};

use crate::{FloorControl, Membership, RoomInbound, RoomsConfig};

/// The transport family this adapter answers to (the management-addressable `transport`).
pub const FAMILY: &str = "room";

/// The typed Room metadata the adapter CBOR-encodes into the store's protocol-free
/// [`Room::descriptor`] opaque column (the store gives `id`/`name`/`policy` typed columns; everything
/// else round-trips through here).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct RoomDescriptor {
    kind: ConversationType,
    title: Option<String>,
    topic: Option<String>,
    description: Option<String>,
    policy: RoomPolicy,
}

/// Decode a stored room's typed descriptor (lenient: a malformed/absent descriptor defaults).
fn decode_descriptor(room: &Room) -> RoomDescriptor {
    from_cbor(&room.descriptor).unwrap_or_default()
}

/// A command from an adapter op to the live [`serve`](RoomsAdapter::serve) loop (which owns the node
/// `api`). Membership/metadata mutations are applied to the store + the shared table directly; only
/// the external post (which needs the running runtime's `api`, floor state, and journal seam) defers.
enum RoomCommand {
    /// Fan a `from`-attributed external post out to a room's members (floor-gated; journaled).
    Post {
        room: RoomId,
        /// The immutable sender identity (an agent/contact handle, or [`SenderId::local_loopback`]
        /// for an operator post) — never re-derived from display text.
        sender: SenderId,
        /// The structured author for the journal record (`None` = the account/operator), carried
        /// alongside `sender` so history keeps the full `Participant`, not a flattened handle.
        author: Option<Participant>,
        text: String,
    },
}

/// The live room loop's shared state: the membership table (shared with the owning [`RoomsAdapter`]),
/// the inbound fan-out, the ingest busy-gate, the set of subscribed member sessions, per-room
/// [`FloorControl`] (cursor + cascade budget), and the node-owned [`LifecycleSink`] every post is
/// journaled through. Built in [`RoomsAdapter::serve`]; holds a `Weak` self-reference so an outbound
/// subscription task can call back into it.
pub(crate) struct RoomRuntime {
    me: Weak<RoomRuntime>,
    api: Arc<dyn NodeApi>,
    store: Arc<dyn SessionStore>,
    sink: Option<Arc<dyn LifecycleSink>>,
    membership: Arc<Mutex<Membership>>,
    inbound: Arc<RoomInbound>,
    ingestor: Arc<Ingestor>,
    subscribed: Mutex<HashSet<SessionId>>,
    floors: Mutex<HashMap<RoomId, FloorControl>>,
    max_turns: u32,
}

/// Constructor inputs for [`RoomRuntime::new`], grouped so `serve` passes one value instead of six
/// positional arguments.
struct RoomRuntimeParts {
    api: Arc<dyn NodeApi>,
    store: Arc<dyn SessionStore>,
    sink: Option<Arc<dyn LifecycleSink>>,
    membership: Arc<Mutex<Membership>>,
    ingest_policy: daemon_ingest::IngestPolicy,
    max_turns: u32,
}

/// Inputs for [`RoomRuntime::post`]: the post to journal + fan out, and whether it starts a fresh
/// cascade (`reset_budget`).
struct RoomPost {
    room: RoomId,
    sender: SenderId,
    /// The structured author the journal record carries (`None` = the account/operator).
    author: Option<Participant>,
    text: String,
    reset_budget: bool,
}

impl RoomRuntime {
    fn new(parts: RoomRuntimeParts) -> Arc<Self> {
        let RoomRuntimeParts {
            api,
            store,
            sink,
            membership,
            ingest_policy,
            max_turns,
        } = parts;
        Arc::new_cyclic(|me| Self {
            me: me.clone(),
            inbound: Arc::new(RoomInbound::new(api.clone())),
            ingestor: Arc::new(Ingestor::with_policy(api.clone(), ingest_policy)),
            api,
            store,
            sink,
            membership,
            subscribed: Mutex::new(HashSet::new()),
            floors: Mutex::new(HashMap::new()),
            max_turns,
        })
    }

    /// Seed the shared membership table from the durable store (`room_list` + `room_members`).
    async fn load(&self) {
        // Await every store read before taking the std `Mutex` — the guard is `!Send` and must not be
        // held across an await, or `serve`'s future would be un-spawnable.
        let rooms = self.store.room_list().await;
        let mut fetched = Vec::new();
        for room in &rooms {
            let members = self.store.room_members(&room.id).await;
            fetched.push((RoomId::new(room.id.clone()), members));
        }
        let mut table = self.membership.lock().unwrap();
        for (rid, members) in fetched {
            for m in members {
                table.upsert(
                    rid.clone(),
                    RoomMember::new(m.member, m.profile, m.session_id),
                );
            }
        }
    }

    /// Subscribe a member `session`'s merged log (idempotent) so its `TurnFinished` re-injects and its
    /// `TurnStarted`/`TurnFinished` drive the busy gate. Member sessions are created lazily on the
    /// first post, so the loop subscribes them at fan-out time rather than enumerating up front.
    fn ensure_subscribed(&self, session: SessionId) {
        if !self.subscribed.lock().unwrap().insert(session.clone()) {
            return;
        }
        let Some(this) = self.me.upgrade() else {
            return;
        };
        // Bind the in-process `internal` principal for the detached subscription task: a spawned task
        // inherits no request context, so the now-ownership-gated `subscribe` and the ingestor's
        // `submit`/`submit_routed` (via `note_turn_finished` / `reinject_reply`) would run with `None`
        // (deny). `internal` is the trusted embedded-caller identity for these fan-out sessions.
        tokio::spawn(with_request_context(
            RequestContext::internal(),
            async move {
                let mut stream = match this.api.subscribe(session.clone(), 0).await {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(error = %e, session = %session.as_str(), "rooms: subscribe failed");
                        this.subscribed.lock().unwrap().remove(&session);
                        return;
                    }
                };
                while let Some(item) = stream.next().await {
                    // Best-effort-skip a lossy lag (turn-lifecycle notes may be missed; the durable
                    // conv journal remains the record). Re-baseline is future work.
                    let entry = match item {
                        daemon_api::LogStreamItem::Entry(e) => e,
                        daemon_api::LogStreamItem::Lagged => continue,
                    };
                    match &entry.payload {
                        SessionPayload::Event(AgentEvent::TurnStarted { .. }) => {
                            this.ingestor.note_turn_started(&session);
                        }
                        SessionPayload::Event(AgentEvent::TurnFinished { summary, .. }) => {
                            if let Some(text) = &summary.final_text {
                                if !text.is_empty() {
                                    this.reinject_reply(&session, text.clone()).await;
                                }
                            }
                            if let Err(e) = this.ingestor.note_turn_finished(&session).await {
                                tracing::warn!(error = %e, "rooms: gate flush failed");
                            }
                        }
                        _ => {}
                    }
                }
            },
        ));
    }

    /// The room's floor-control policy (from its stored descriptor; default if absent).
    async fn room_policy(&self, room: &RoomId) -> RoomPolicy {
        self.store
            .room_get(room.as_str())
            .await
            .map(|r| decode_descriptor(&r).policy)
            .unwrap_or_default()
    }

    /// Journal the post as one `ChatMessage` on the room's conversation history, then floor-gate it
    /// and fan it out (StartTurn to admitted members, Observe to the rest; the sender is skipped).
    /// `reset_budget` starts a fresh cascade (an external/operator post); a re-injected reply
    /// continues the cascade.
    async fn post(&self, args: RoomPost) {
        let RoomPost {
            room,
            sender,
            author,
            text,
            reset_budget,
        } = args;
        // 1. Durable conversation history (wire v38): report the message through the node's
        //    LifecycleSink seam, which appends one verified `JournalRecordPayload::Chat` onto
        //    `conv:room:<id>` (the stream `conv_history` pages) and emits `MessagesChanged`. The
        //    record carries the RAW text + structured author — attribution never rides the body.
        if let Some(sink) = &self.sink {
            let mut message = ChatMessage::new(author, text.clone());
            message.timestamp = Some(now_unix_secs());
            // rung 3 (api vNEXT): the rooms reference adapter routes sends through an internal
            // command channel (a serve-loop boundary a task-local op token cannot cross), so it is
            // a token-incapable adapter for now — `origin_op` is null (the degraded path, never
            // heuristic). Confirmation resolves via the accepted-state policy (09§6.6).
            sink.chat_message(
                TransportId::new(FAMILY),
                room.as_str().to_string(),
                message,
                None,
            )
            .await;
        }

        // 2. Snapshot members + policy (await before locking the !Send floor map).
        let members: Vec<RoomMember> = self.membership.lock().unwrap().members(&room).to_vec();
        if members.is_empty() {
            return;
        }
        let policy = self.room_policy(&room).await;

        // 3. Floor decision against the per-room cursor/budget state.
        let admitted = {
            let mut floors = self.floors.lock().unwrap();
            let fc = floors
                .entry(room.clone())
                .or_insert_with(|| FloorControl::new(policy, self.max_turns));
            if reset_budget {
                fc.begin_post();
            }
            fc.decide(&members, sender.as_str(), &text)
        };

        // 4. Fan out (creates member sessions on first `StartTurn`).
        self.inbound
            .fan_out(&room, &sender, &text, &members, |m| {
                admitted.contains(&m.member)
            })
            .await;

        // 5. Subscribe each member session *after* fan-out, so it exists; the subscription replays
        //    from cursor 0, catching the reply that advances the cascade even if a turn finished
        //    before we attached. Idempotent + retried (on a not-yet-created session) on the next post.
        for m in &members {
            self.ensure_subscribed(m.session.clone());
        }
    }

    /// An external/operator `ConvSend` post (starts a fresh cascade).
    async fn external_post(
        &self,
        room: RoomId,
        sender: SenderId,
        author: Option<Participant>,
        text: String,
    ) {
        self.post(RoomPost {
            room,
            sender,
            author,
            text,
            reset_budget: true,
        })
        .await;
    }

    /// Re-inject a member session's finished-turn reply back into its Room (continues the cascade).
    async fn reinject_reply(&self, session: &SessionId, text: String) {
        let resolved = self.membership.lock().unwrap().find_by_session(session);
        if let Some((room, member)) = resolved {
            // The member handle is a structured identity (from the membership table), not text; the
            // journal author keeps the full participant shape (agent when the profile binding is
            // known, a bare contact handle for a legacy profile-less row).
            let author = Some(match &member.profile {
                Some(profile) => Participant::Agent {
                    profile: profile.clone(),
                    member: member.member.clone(),
                },
                None => Participant::Contact(ContactInfo {
                    id: member.member.clone(),
                    ..ContactInfo::default()
                }),
            });
            self.post(RoomPost {
                room,
                sender: SenderId::new(member.member),
                author,
                text,
                reset_budget: false,
            })
            .await;
        }
    }
}

/// Unix seconds now — the node-side clock the journal record's `timestamp` is stamped with.
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The Rooms transport adapter. Holds the durable store, the node-owned lifecycle sink (the
/// wire-v38 chat-journal seam every post is reported through), the resolved config, the
/// membership table it shares with the live runtime, and the command channel into
/// [`serve`](Self::serve).
pub struct RoomsAdapter {
    store: Arc<dyn SessionStore>,
    sink: Option<Arc<dyn LifecycleSink>>,
    cfg: RoomsConfig,
    membership: Arc<Mutex<Membership>>,
    cmd_tx: mpsc::UnboundedSender<RoomCommand>,
    cmd_rx: Mutex<Option<mpsc::UnboundedReceiver<RoomCommand>>>,
    /// The in-memory, per-transport server-side contact roster ([`SupportsRoster`], wire v34): a map
    /// of `transport -> (contact id -> contact)`. The Rooms transport has no external directory, so
    /// the roster is purely local process state (unlike rooms/membership, which persist to the store);
    /// the host paginates + emits `ContactsChanged` centrally.
    roster: Mutex<HashMap<TransportId, HashMap<String, ContactInfo>>>,
    /// The node content store, for the loopback [`SupportsFileTransfer`] (W2-H). When present, file
    /// transfer round-trips bytes through the content-addressed blob store (send verifies the blob
    /// resolves; receive fetches it — a same-node loopback). `None` ⟹ the feature is absent
    /// (`file_transfer()` returns `None`), keeping `assert_ops_match_behavior` honest.
    blobs: Option<Arc<dyn BlobStore>>,
}

impl RoomsAdapter {
    /// Construct the adapter over the durable `store`, the node's [`LifecycleSink`] (wire v38:
    /// the chat-journal + `MessagesChanged` seam; pass `None` where the node is not wired — unit
    /// tests), and the resolved `cfg`. The returned `Arc` is what the host registry holds and what
    /// `serve` consumes; ops borrow `&self` through it.
    pub fn new(
        store: Arc<dyn SessionStore>,
        cfg: RoomsConfig,
        sink: Option<Arc<dyn LifecycleSink>>,
    ) -> Arc<Self> {
        Self::build(store, cfg, sink, None)
    }

    /// Like [`new`](Self::new), but wires the node content store so the loopback
    /// [`SupportsFileTransfer`] (W2-H) is advertised + operable.
    pub fn with_blobs(
        store: Arc<dyn SessionStore>,
        cfg: RoomsConfig,
        sink: Option<Arc<dyn LifecycleSink>>,
        blobs: Arc<dyn BlobStore>,
    ) -> Arc<Self> {
        Self::build(store, cfg, sink, Some(blobs))
    }

    fn build(
        store: Arc<dyn SessionStore>,
        cfg: RoomsConfig,
        sink: Option<Arc<dyn LifecycleSink>>,
        blobs: Option<Arc<dyn BlobStore>>,
    ) -> Arc<Self> {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        Arc::new(Self {
            store,
            sink,
            cfg,
            membership: Arc::new(Mutex::new(Membership::new())),
            cmd_tx,
            cmd_rx: Mutex::new(Some(cmd_rx)),
            roster: Mutex::new(HashMap::new()),
            blobs,
        })
    }

    /// Project a stored room (+ its membership rows) into the wire [`ConversationInfo`].
    async fn room_to_info(&self, transport: &TransportId, room: &Room) -> ConversationInfo {
        let desc = decode_descriptor(room);
        let members = self
            .store
            .room_members(&room.id)
            .await
            .into_iter()
            .map(|m| ConversationMember {
                contact: ContactInfo {
                    id: m.member,
                    display_name: None,
                    presence: Presence::default(),
                    permission: ContactPermission::Unset,
                },
                alias: None,
                nickname: None,
                typing: TypingState::None,
                role: MemberRole::None,
                session: Some(m.session_id),
            })
            .collect();
        ConversationInfo {
            transport: transport.clone(),
            id: room.id.clone(),
            kind: desc.kind,
            title: room.name.clone().or(desc.title),
            topic: desc.topic,
            description: desc.description,
            members,
            // The Rooms adapter models flat agent rooms — no space/server hierarchy (wire v38).
            parent: None,
        }
    }

    /// Read-modify-write a room's typed descriptor.
    async fn mutate_descriptor<F>(&self, conv: &str, f: F) -> Result<(), ApiError>
    where
        F: FnOnce(&mut Room, &mut RoomDescriptor),
    {
        let mut room = self
            .store
            .room_get(conv)
            .await
            .ok_or_else(|| ApiError::Other(format!("room {conv} not found")))?;
        let mut desc = decode_descriptor(&room);
        f(&mut room, &mut desc);
        room.policy = policy_tag(&desc.policy);
        room.descriptor = to_cbor(&desc);
        self.store
            .room_set(room)
            .await
            .map_err(|e| ApiError::Other(format!("store: {e}")))
    }
}

/// The store's `policy` typed column tag (column-level listing; the descriptor stays authoritative).
fn policy_tag(policy: &RoomPolicy) -> String {
    match policy {
        RoomPolicy::AddressedOnly => "addressed_only",
        RoomPolicy::FreeForAll => "free_for_all",
        RoomPolicy::RoundRobin => "round_robin",
        RoomPolicy::Moderator { .. } => "moderator",
        _ => "addressed_only",
    }
    .to_string()
}

/// Parse a floor-policy tag from create/extras input (default: the [`RoomPolicy`] default). The
/// `moderator:<member>` form selects a moderator-arbitrated room.
fn parse_policy(tag: Option<&str>) -> RoomPolicy {
    match tag {
        Some("free_for_all") => RoomPolicy::FreeForAll,
        Some("round_robin") => RoomPolicy::RoundRobin,
        Some("addressed_only") => RoomPolicy::AddressedOnly,
        Some(other) if other.starts_with("moderator:") => RoomPolicy::Moderator {
            profile: other.trim_start_matches("moderator:").to_string(),
        },
        _ => RoomPolicy::default(),
    }
}

/// Parse a conversation kind from create/extras input (default: a group DM).
fn parse_kind(tag: Option<&str>) -> ConversationType {
    match tag {
        Some("Channel") | Some("channel") => ConversationType::Channel,
        Some("Dm") | Some("dm") => ConversationType::Dm,
        Some("Thread") | Some("thread") => ConversationType::Thread,
        _ => ConversationType::GroupDm,
    }
}

#[async_trait]
impl TransportAdapter for RoomsAdapter {
    fn family(&self) -> &str {
        FAMILY
    }

    fn info(&self) -> AdapterInfo {
        AdapterInfo {
            family: FAMILY.to_string(),
            display_name: "Rooms (internal)".to_string(),
            capabilities: AdapterCapabilities {
                rooms: true,
                direct_messages: true,
                presence: false,
                room_enumeration: true,
                file_transfer: false,
                interactive_auth: false,
            },
            account_schema: AccountSettingsSchema::default(),
            // The internal loopback transport has no operator-facing policies (wire v30).
            policies: Vec::new(),
            // Per-verb ops (wire v33) are enriched centrally in the host `transport_adapters()` from
            // the feature-trait `supported()` probes; the adapter leaves them at default here.
            ..Default::default()
        }
    }

    async fn instances(&self) -> Vec<TransportInstanceInfo> {
        vec![TransportInstanceInfo {
            transport: TransportId::new(FAMILY),
            family: FAMILY.to_string(),
            display_name: "Rooms (internal)".to_string(),
            connection: ConnectionState::Connected,
            presence: PresenceState::default(),
            bound_profile: None,
            // The loopback transport is always connected; no disconnect provenance (wire v30).
            reason: None,
            message: None,
            fatal: false,
            // Wire v35: enabled/label are node-overlaid from the store; report inert default.
            enabled: true,
            label: None,
        }]
    }

    async fn serve(self: Arc<Self>, api: Arc<dyn NodeApi>) {
        if !self.cfg.enabled {
            return;
        }

        let runtime = RoomRuntime::new(RoomRuntimeParts {
            api: api.clone(),
            store: self.store.clone(),
            sink: self.sink.clone(),
            membership: self.membership.clone(),
            ingest_policy: self.cfg.ingest_policy(),
            max_turns: self.cfg.max_turns,
        });
        runtime.load().await;
        // Subscribe any members that already existed at boot (durable rooms), so their replies
        // re-inject even before the first new post. New members get subscribed on the first post.
        for session in runtime.membership.lock().unwrap().all_member_sessions() {
            runtime.ensure_subscribed(session);
        }

        let mut rx = match self.cmd_rx.lock().unwrap().take() {
            Some(rx) => rx,
            None => {
                tracing::warn!("rooms: serve called twice; the command channel was already taken");
                return;
            }
        };

        while let Some(cmd) = rx.recv().await {
            match cmd {
                RoomCommand::Post {
                    room,
                    sender,
                    author,
                    text,
                } => {
                    // The serve loop runs with no request context; an external post fans out via the
                    // ownership-gated `submit_from`, so bind the trusted `internal` embedded-caller
                    // identity for the fan-out.
                    with_request_context(
                        RequestContext::internal(),
                        runtime.external_post(room, sender, author, text),
                    )
                    .await;
                }
            }
        }
    }

    fn messaging(self: Arc<Self>) -> Option<Arc<dyn MessagingProtocol>> {
        Some(self)
    }
}

#[async_trait]
impl MessagingProtocol for RoomsAdapter {
    fn conversations(self: Arc<Self>) -> Option<Arc<dyn SupportsConversations>> {
        Some(self)
    }

    fn membership(self: Arc<Self>) -> Option<Arc<dyn SupportsMembership>> {
        Some(self)
    }

    fn roster(self: Arc<Self>) -> Option<Arc<dyn SupportsRoster>> {
        Some(self)
    }

    fn file_transfer(self: Arc<Self>) -> Option<Arc<dyn SupportsFileTransfer>> {
        // The feature exists only when the node content store is wired (W2-H). Absent ⟹ `None`, so
        // the ops-vs-behavior invariant sees no advertised (yet unimplemented) file-transfer verbs.
        if self.blobs.is_some() {
            Some(self)
        } else {
            None
        }
    }
}

#[async_trait]
impl SupportsConversations for RoomsAdapter {
    fn supported(&self) -> ConversationOps {
        ConversationOps {
            create: true,
            join_channel: true,
            leave: true,
            delete: true,
            send: true,
            set_topic: true,
            set_title: true,
            set_description: true,
        }
    }

    async fn list(&self, transport: TransportId) -> Vec<ConversationInfo> {
        let rooms = self.store.room_list().await;
        let mut out = Vec::with_capacity(rooms.len());
        for room in &rooms {
            out.push(self.room_to_info(&transport, room).await);
        }
        out
    }

    async fn get(&self, transport: TransportId, conv: String) -> Option<ConversationInfo> {
        let room = self.store.room_get(&conv).await?;
        Some(self.room_to_info(&transport, &room).await)
    }

    async fn create(
        &self,
        transport: TransportId,
        details: CreateConversationDetails,
    ) -> Result<ConversationInfo, ApiError> {
        let v = &details.extras.values;
        let id = v
            .get("id")
            .cloned()
            .or_else(|| v.get("name").cloned())
            .unwrap_or_else(|| {
                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or_default();
                format!("room-{nanos}")
            });
        let name = v.get("name").cloned();
        let desc = RoomDescriptor {
            kind: parse_kind(v.get("kind").map(String::as_str)),
            title: name.clone(),
            topic: None,
            description: None,
            policy: parse_policy(v.get("policy").map(String::as_str)),
        };
        let room = Room {
            id: id.clone(),
            name,
            policy: policy_tag(&desc.policy),
            descriptor: to_cbor(&desc),
        };
        self.store
            .room_set(room.clone())
            .await
            .map_err(|e| ApiError::Other(format!("store: {e}")))?;
        Ok(self.room_to_info(&transport, &room).await)
    }

    async fn join_channel(
        &self,
        transport: TransportId,
        details: ChannelJoinDetails,
    ) -> Result<ConversationInfo, ApiError> {
        let id = details
            .name
            .clone()
            .or_else(|| details.extras.values.get("id").cloned())
            .ok_or_else(|| ApiError::Other("rooms join requires a channel name".to_string()))?;
        if let Some(room) = self.store.room_get(&id).await {
            return Ok(self.room_to_info(&transport, &room).await);
        }
        let mut create = CreateConversationDetails {
            extras: details.extras,
            ..CreateConversationDetails::default()
        };
        create
            .extras
            .values
            .entry("id".to_string())
            .or_insert_with(|| id.clone());
        create.extras.values.entry("name".to_string()).or_insert(id);
        self.create(transport, create).await
    }

    async fn leave(&self, _transport: TransportId, _conv: String) -> Result<(), ApiError> {
        // The Rooms "account" is the node itself; it is never a leaving occupant. Removing an agent
        // member is `SupportsMembership::remove`; destroying the room is `delete`.
        Ok(())
    }

    async fn delete(&self, _transport: TransportId, conv: String) -> Result<(), ApiError> {
        self.store
            .room_remove(&conv)
            .await
            .map_err(|e| ApiError::Other(format!("store: {e}")))?;
        // Drop the live membership so the running loop stops fanning posts to the gone room (its
        // delivery subscription, floor state, and transcript sink become inert with no members).
        self.membership
            .lock()
            .unwrap()
            .remove_room(&RoomId::new(conv));
        Ok(())
    }

    async fn send(&self, args: ConvSendArgs) -> Result<(), ApiError> {
        let ConvSendArgs {
            transport: _transport,
            conv,
            from,
            message,
            // rung 3 (api vNEXT): token-incapable adapter (routes through an internal command
            // channel); `origin_op` resolves null via the accepted-state policy (09§6.6).
            op_id: _,
        } = args;
        if self.store.room_get(&conv).await.is_none() {
            return Err(ApiError::Other(format!("room {conv} not found")));
        }
        let sender = match &from {
            Some(Participant::Agent { member, .. }) => SenderId::new(member.clone()),
            Some(Participant::Contact(c)) => SenderId::new(c.id.clone()),
            // No external participant: a node/operator loopback post — a typed, documented identity
            // rather than a re-derivable "operator" string.
            None => SenderId::local_loopback(),
        };
        self.cmd_tx
            .send(RoomCommand::Post {
                room: RoomId::new(conv),
                sender,
                // The full participant rides through to the journal record's author (`None` = the
                // account/operator), matching the `ChatMessage::author` convention.
                author: from,
                text: message.text,
            })
            .map_err(|_| ApiError::Other("rooms serve loop is not running".to_string()))
    }

    async fn set_topic(
        &self,
        _transport: TransportId,
        conv: String,
        topic: Option<String>,
    ) -> Result<(), ApiError> {
        self.mutate_descriptor(&conv, |_room, desc| desc.topic = topic)
            .await
    }

    async fn set_title(
        &self,
        _transport: TransportId,
        conv: String,
        title: Option<String>,
    ) -> Result<(), ApiError> {
        self.mutate_descriptor(&conv, |room, desc| {
            room.name = title.clone();
            desc.title = title;
        })
        .await
    }

    async fn set_description(
        &self,
        _transport: TransportId,
        conv: String,
        description: Option<String>,
    ) -> Result<(), ApiError> {
        self.mutate_descriptor(&conv, |_room, desc| desc.description = description)
            .await
    }
}

#[async_trait]
impl SupportsMembership for RoomsAdapter {
    fn supported(&self) -> MembershipOps {
        // invite/remove are Rooms' membership administration; ban/set_role are off — floor policy is a
        // separate Rooms concern (daemon-messaging-adapter-spec.md §10.1).
        MembershipOps {
            invite: true,
            remove: true,
            ban: false,
            set_role: false,
        }
    }

    async fn invite(&self, args: MemberInviteArgs) -> Result<(), ApiError> {
        let MemberInviteArgs {
            transport: _transport,
            conv,
            who,
            message: _message,
            op_id: _,
        } = args;
        let (profile, member) = match who {
            Participant::Agent { profile, member } => (profile, member),
            Participant::Contact(_) => {
                return Err(ApiError::Unsupported(
                    "rooms invite binds an agent participant (Participant::Agent)".to_string(),
                ))
            }
        };
        if self.store.room_get(&conv).await.is_none() {
            return Err(ApiError::Other(format!("room {conv} not found")));
        }
        // The member's engine incarnation is a deterministic per-(room, member) session.
        let session = SessionId::new(format!("room:{conv}:{member}"));
        self.store
            .room_member_set(StoreRoomMember {
                room_id: conv.clone(),
                member: member.clone(),
                profile: Some(profile.clone()),
                session_id: session.clone(),
            })
            .await
            .map_err(|e| ApiError::Other(format!("store: {e}")))?;
        self.membership.lock().unwrap().upsert(
            RoomId::new(conv),
            RoomMember::new(member, Some(profile), session),
        );
        Ok(())
    }

    async fn remove(&self, args: MemberRemoveArgs) -> Result<(), ApiError> {
        let MemberRemoveArgs {
            transport: _transport,
            conv,
            who,
            reason: _reason,
            op_id: _,
        } = args;
        let member = match who {
            Participant::Agent { member, .. } => member,
            Participant::Contact(c) => c.id,
        };
        self.store
            .room_member_remove(&conv, &member)
            .await
            .map_err(|e| ApiError::Other(format!("store: {e}")))?;
        self.membership
            .lock()
            .unwrap()
            .remove(&RoomId::new(conv), &member);
        Ok(())
    }
}

#[async_trait]
impl SupportsFileTransfer for RoomsAdapter {
    fn supported(&self) -> FileTransferOps {
        // Reachable only when `blobs` is wired (see `file_transfer()`); both verbs are then live.
        FileTransferOps {
            send: self.blobs.is_some(),
            receive: self.blobs.is_some(),
        }
    }

    async fn send(&self, _transport: TransportId, transfer: FileTransfer) -> Result<(), ApiError> {
        // Loopback send: the content is content-addressed and already resident in the node store, so
        // "sending" verifies the blob resolves (a full, integrity-checked read).
        let blobs = self
            .blobs
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("file_transfer_send".into()))?;
        blobs
            .get(&transfer.blob.hash, None)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("rooms file transfer send: {e}")))
    }

    async fn receive(
        &self,
        _transport: TransportId,
        transfer: FileTransfer,
    ) -> Result<(), ApiError> {
        // Loopback receive: fetch the sender's blob from the same node store (a same-node transfer).
        let blobs = self
            .blobs
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("file_transfer_receive".into()))?;
        blobs
            .get(&transfer.blob.hash, None)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("rooms file transfer receive: {e}")))
    }
}

#[async_trait]
impl SupportsRoster for RoomsAdapter {
    fn supported(&self) -> RosterOps {
        // The Rooms transport keeps a full in-memory contact list; all four verbs are live.
        RosterOps {
            list: true,
            add: true,
            update: true,
            remove: true,
        }
    }

    async fn list(&self, transport: TransportId) -> Vec<ContactInfo> {
        // Unpaged + adapter-ordered: the host sorts by contact id and pages centrally.
        self.roster
            .lock()
            .unwrap()
            .get(&transport)
            .map(|contacts| contacts.values().cloned().collect())
            .unwrap_or_default()
    }

    async fn add(&self, transport: TransportId, contact: ContactInfo) -> Result<(), ApiError> {
        let mut roster = self.roster.lock().unwrap();
        let contacts = roster.entry(transport).or_default();
        if contacts.contains_key(&contact.id) {
            return Err(ApiError::Other(format!(
                "contact {} already on the roster",
                contact.id
            )));
        }
        contacts.insert(contact.id.clone(), contact);
        Ok(())
    }

    async fn update(&self, transport: TransportId, contact: ContactInfo) -> Result<(), ApiError> {
        let mut roster = self.roster.lock().unwrap();
        match roster
            .get_mut(&transport)
            .and_then(|c| c.get_mut(&contact.id))
        {
            Some(slot) => {
                *slot = contact;
                Ok(())
            }
            None => Err(ApiError::Other(format!("contact {} not found", contact.id))),
        }
    }

    async fn remove(&self, transport: TransportId, contact: ContactInfo) -> Result<(), ApiError> {
        let mut roster = self.roster.lock().unwrap();
        let removed = roster
            .get_mut(&transport)
            .is_some_and(|contacts| contacts.remove(&contact.id).is_some());
        if removed {
            Ok(())
        } else {
            Err(ApiError::Other(format!("contact {} not found", contact.id)))
        }
    }
}
