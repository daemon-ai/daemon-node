# Daemon Messaging-Adapter Interface — a faithful port of libpurple 3

Status: landed. Defines the daemon analogue of libpurple 3's protocol interface: one typed,
capability-probed **messaging-adapter** surface that every chat-like transport (the internal
`daemon-rooms` loopback, `daemon-matrix`, and future Slack/XMPP/IRC/…) implements selectively. This
is primarily a **port**, not a fresh design: libpurple has done the hard contract work over two
decades, so we copy its decomposition, translate idioms (GObject → Rust), and delineate the few
daemon-specific extensions. Where the in-progress libpurple-3 snapshot is incomplete (notably
membership administration) we restore the stable contract from libpurple 2 / Adium / Kopete (§2.3.1).
The interface family, the `Conv*`/`Member*`/`Contact*`/`Directory*` wire ops, and the `daemon-rooms`
and `daemon-matrix` implementors have landed (`WireVersion` 18–20); the remaining deferrals are
itemized in §12.1.

Companion to `daemon-event-io-spec.md` (§5 the IO edge, §5.9 routing), `daemon-rooms-spec.md` (the
internal loopback transport, a reference implementor), and `daemon-matrix-transport-spec.md` (the
reference external chat transport). **Specializes** (does not supersede) the transport-adapter
framework in `daemon-transport-adapter-spec.md`: that spec's `TransportAdapter` base + `AdapterInfo`/
`AdapterCapabilities`/`TransportInstanceInfo` + host `AdapterRegistry` remain the live foundation, and
this spec layers the libpurple-3-style typed feature traits on top of it via the
`TransportAdapter::messaging()` accessor — see [§12.3](#123-relationship-to-the-transport-adapter-framework).

> Source authority: the libpurple 3 tree at
> `/home/j/experiments/multiprotocol-instant-messengers/pidgin-496de266ac6c/libpurple/`. Every ported
> interface and DTO below cites the `purple*.h` file it is taken from. Any uncertainty during
> implementation is resolved by reading that source, not by improvising.

---

## 1. Motivation & principle

Daemon needs to *manage* chat transports over the wire — create a room, post into it, set a topic,
list conversations, manage members, report connection state — and to do so uniformly across many
protocols rather than special-casing each. Three mature multi-protocol messengers (libpurple/Pidgin,
Kopete, Adium) independently converged on the same decomposition for exactly this problem, which is
strong evidence it is the right one. So instead of inventing a generic action bus, we port libpurple
3's typed interface family.

Principles the implementation follows:

1. **Faithful port.** Port libpurple 3's method names, argument shapes, the per-verb capability probe
   (`implements_*`), and the typed "details" builder objects as libpurple defines them.
2. **The host never special-cases a family.** It resolves the owning adapter from a registry and
   forwards by *capability* (which feature interface the adapter implements), never by matching a
   transport-family string.
3. **Translate idioms minimally.** GObject derivable types/interfaces → Rust structs/traits;
   `_async`/`_finish` vfunc pairs → `async fn`; `GError**` → `Result<_, ApiError>`;
   `GListModel` of `X` → `Vec<X>`; `implements_*()` probes → a `supported()` struct of bools plus an
   `Option<Arc<dyn Feature>>` accessor.
4. **Delineate daemon extensions.** Where daemon's internal loopback genuinely needs what no
   federated chat protocol models (agent-as-participant, floor control),
   put it in a clearly separate section ([§8](#8-daemon-extensions-delineated)), never folded into the
   ported interface.
5. **Reuse the hard half daemon already owns.** `daemon-ingest` (inbound gate) and `daemon-delivery`
   (outbound `Projector`) are the send/receive *mechanics*; this interface adds only the *management*
   verbs and the descriptive/lifecycle layer on top. They are not reimplemented.

---

## 2. Provenance: libpurple 3, extracted

libpurple 3 is a **base type + selectively-implemented optional feature interfaces + per-verb
capability probes + typed "details" builders + a normalized data model**. The recent GObject rewrite
split the old monolithic `PurplePluginProtocolInfo` into a base `PurpleProtocol` plus optional feature
interfaces a protocol implements à la carte. Extracted:

### 2.1 Base + feature interfaces

- `PurpleProtocol` (`purpleprotocol.h`) — identity (`id`/`name`/`description`/`icon`/`tags`),
  `get_default_account_settings`, `validate_account`, `can_connect_async/finish`,
  `create_connection`, `generate_account_name`, `delete_account`.
- `PurpleProtocolConversation` (`purpleprotocolconversation.h`) —
  `get_create_conversation_details` + `create_conversation`, `leave_conversation`, `send_message`,
  `set_topic`, `get_channel_join_details` + `join_channel`, `set_avatar`, `send_typing`, `refresh`,
  `set_title`, `set_description`. Each guarded by an `implements_*` probe. There is **no `invite` /
  `add_member` verb on the conversation interface**; membership administration is a separate concern
  (§2.3.1 / §3.2.1).
- `PurpleProtocolRoster` (`purpleprotocolroster.h`) — `add`/`update`/`remove` a `PurpleContact` on the
  **server-side account contact list** (the buddy list), not conversation membership.
- `PurpleProtocolContacts` (`purpleprotocolcontacts.h`) — `get_profile`, `get_action_menu`,
  `set_alias`.
- `PurpleProtocolDirectory` (`purpleprotocoldirectory.h`) — `search_contacts`.
- `PurpleProtocolFileTransfer` (`purpleprotocolfiletransfer.h`) — `send`/`receive`.
- `PurpleProtocolWhiteboard` (`purpleprotocolwhiteboard.h`) — collaborative whiteboard (niche; not
  ported now).

### 2.2 Data model

- `PurpleConversationType` (`purpleconversation.h`): `UNSET`, `DM`, `GROUP_DM`, `CHANNEL`, `THREAD`.
- `PurpleConversationMembers`/`PurpleConversationMember` (`purpleconversationmembers.h`/
  `purpleconversationmember.h`): the **occupant model** — membership state the protocol *populates
  from sync* (`add_member`/`remove_member` there are "intended to be called by a protocol plugin to
  directly manage the membership state", with `announce`/`message` for display), each member wrapping a
  `PurpleContactInfo` and carrying alias/nickname/typing/badges/tags.
- `PurpleCreateConversationDetails` (`purplecreateconversationdetails.h`): `max_participants` +
  `participants` (a list of `PurpleContact`) + `is_valid`. The UI fetches it from the protocol, fills
  it, and passes it back.
- `PurpleChannelJoinDetails` (`purplechanneljoindetails.h`): `name`(+`name_max_length`),
  `nickname`(+`nickname_supported`/`nickname_max_length`), `password`(+`password_supported`/
  `password_max_length`), `merge`.
- `PurpleContactInfo` (`purplecontactinfo.h`): `id`, `display_name`, `presence`,
  `permission` (`UNSET`/`ALLOW`/`DENY`), badges, tags; `PurplePerson` (`purpleperson.h`) is the
  cross-protocol MetaContact grouping many `ContactInfo` endpoints.
- `PurplePresence` (`purplepresence.h`) with `PurplePresencePrimitive`: `OFFLINE`, `AVAILABLE`,
  `IDLE`, `INVISIBLE`, `AWAY`, `DO_NOT_DISTURB`, `STREAMING`, `OUT_OF_OFFICE`; plus message/emoji/
  mobile/idle/login-time.
- `PurpleConnectionState` (`purpleconnection.h`): `DISCONNECTED`, `DISCONNECTING`, `CONNECTED`,
  `CONNECTING`; plus `PurpleConnectionError`.
- `PurpleTypingState` (`purpletyping.h`): `NONE`, `TYPING`, `PAUSED`.
- `PurpleAccountSettings` + `purpleaccountsetting{boolean,int,string,stringlist}.h`: the **typed
  account-setup schema** (the UI renders a form from these).
- `PurpleProtocolManager` (`purpleprotocolmanager.h`): the registry of protocols.

### 2.3 Kopete / Adium convergence

Kopete (`Kopete::Protocol` + `Capability` flags + `ChatSession`/`Contact`/`MetaContact`) and Adium
(`AIService` + per-service bool flags + `AIChat`/`AIListContact`/`AIMetaContact`) are the same six
concepts with coarser capability flags instead of split interfaces. libpurple's split-interface form
is the most granular and is what we port.

### 2.3.1 Membership administration across the ecosystem

Inviting/removing participants and per-member roles are first-class in the mature systems even though
the in-progress libpurple-3 snapshot has not re-ported them:

| Operation | libpurple 2 | libpurple 3 (this snapshot) | Adium | Kopete |
| --- | --- | --- | --- | --- |
| invite / add | `chat_invite` + `serv_chat_invite` | deferred to `PurpleCommand` (`/quote INVITE`) | `AIAccount inviteContact:toChat:withMessage:` | `ChatSession::inviteContact` + `mayInvite()` |
| remove / kick | prpl-specific | inbound KICK handler; outbound via raw `/quote` | IRC raw `KICK` (`ESIRCAccount`) | local `removeContact` only (no wire kick) |
| ban | prpl-specific | inbound; `/quote MODE +b` | IRC raw `MODE +b` | not modeled |
| set topic | `set_chat_topic` | `set_topic_async` (first-class) | `setTopic:forChat:` | message subject only |
| roles / flags | per-prpl | `PurpleBadge`s on `ConversationMember` | `AIGroupChatFlags` (voice/half-op/op/founder) | not modeled |

So **invite is the convergent first-class membership verb**; remove/kick and ban are real but
protocol-specific; roles are first-class as observed flags. daemon ports these as a
[`SupportsMembership`](#321-supportsmembership--membership-administration) capability grounded in this
contract, documenting the deliberate divergence from the in-progress libpurple-3 snapshot (§7).

### 2.4 What daemon already owns (so we do not re-port it)

| libpurple concept | daemon equivalent (existing) |
| --- | --- |
| send/receive message mechanics | `daemon-ingest` (`Ingestor::receive` → `submit_routed`) + `daemon-delivery` (`serve_delivery` + `Projector`) |
| `PurpleConversationType` | `OriginScope{Dm, Group, Api, Internal}` (`daemon-protocol`) — bridged, see §5 |
| typed account settings | `AccountSettingsSchema` / `AuthParamField` (`daemon-api`) |
| `PurpleProtocolManager` | `AdapterRegistry` (`daemon-host`) |
| connection/presence DTOs | `ConnectionState` / `PresenceState` / `TransportInstanceInfo` (`daemon-api`) — reconciled to the libpurple primitives, see §5 |
| interactive login | `AuthApi` (`auth_begin`/`auth_complete`) + `AccountProvisioning` (`daemon-host`) |
| the `TransportAdapter` seam + capability model | the live base `TransportAdapter` + `AdapterCapabilities` in `daemon-api` — this spec specializes it with the typed feature traits |

---

## 3. The interface family

All types live in `daemon-api` (co-located with the existing `TransportAdapter` base and the
capability DTOs an adapter crate already depends on — no new crate). The libpurple `PurpleProtocol`
maps to a **`MessagingProtocol` specialization of the generic `TransportAdapter`** (§3.1.1), **not** to
`TransportAdapter` itself — so the generic adapter seam (shared by non-chat transports like a webhook
ingress or a scheduled-trigger source) never carries messaging concepts. GObject interfaces become
Rust traits; the `implements_*` two-level probe becomes (a) an `Option<Arc<dyn Feature>>` accessor on
`MessagingProtocol` — the "does it implement this interface at all" probe — and (b) a `supported()`
struct of per-verb bools returned by the feature trait — the `implements_<verb>` probe.

### 3.1 Base: `TransportAdapter` (generic events-IO seam; pre-existing)

`TransportAdapter` is daemon's **existing generic** adapter base — it is *not* the libpurple
`PurpleProtocol`. It applies to **any** events-IO transport, messaging or not: identity, the
account-setup descriptor, the run loop, instance enumeration, and a `messaging()` probe. The libpurple
port is a specialization of it (§3.1.1).

```rust
#[async_trait]
pub trait TransportAdapter: Send + Sync {
    /// Stable family id: "room", "matrix", "webhook", ...
    fn family(&self) -> &str;

    /// Descriptor the GUI reads for the "Add channel" picker: display name, coarse capability
    /// summary, and the typed account-setup schema.
    fn info(&self) -> AdapterInfo;

    /// Drive the transport until shutdown (registry-spawned, §6; wires daemon-ingest/daemon-delivery).
    async fn serve(self: Arc<Self>, api: Arc<dyn NodeApi>);

    /// Configured instances (accounts) with live connection/presence state (the daemon analogue of
    /// `PurpleAccountManager` enumeration). Default: empty.
    async fn instances(&self) -> Vec<TransportInstanceInfo> { Vec::new() }

    /// Is this transport a messaging protocol? Returns its libpurple-`PurpleProtocol` view when so;
    /// generic (non-chat) transports return None. The one-level-up analogue of libpurple's
    /// `PURPLE_IS_PROTOCOL_*` type checks.
    fn messaging(self: Arc<Self>) -> Option<Arc<dyn MessagingProtocol>> { None }
}
```

### 3.1.1 `MessagingProtocol: TransportAdapter` (← `purpleprotocol.h`)

The faithful port of libpurple's `PurpleProtocol`. A messaging protocol is a `TransportAdapter` that
*additionally* validates accounts and exposes the optional conversation/roster/contacts/directory/
file-transfer feature interfaces. Non-messaging transports never implement it (`messaging()` returns
`None`), keeping the generic seam free of chat concepts.

```rust
#[async_trait]
pub trait MessagingProtocol: TransportAdapter {
    /// Validate proposed account settings (← `validate_account`). Default: Ok.
    async fn validate_account(&self, _settings: &AccountSettingsValues) -> Result<(), ApiError> { Ok(()) }

    // -- optional feature interfaces (libpurple's split-interface probe: accessor returns Some) --
    fn conversations(self: Arc<Self>) -> Option<Arc<dyn SupportsConversations>> { None }
    fn membership(self: Arc<Self>)    -> Option<Arc<dyn SupportsMembership>>    { None }
    fn roster(self: Arc<Self>)        -> Option<Arc<dyn SupportsRoster>>        { None }
    fn contacts(self: Arc<Self>)      -> Option<Arc<dyn SupportsContacts>>      { None }
    fn directory(self: Arc<Self>)     -> Option<Arc<dyn SupportsDirectory>>     { None }
    fn file_transfer(self: Arc<Self>) -> Option<Arc<dyn SupportsFileTransfer>>  { None }
}
```

The host reaches messaging capabilities through `TransportAdapter::messaging()` →
`MessagingProtocol::conversations()` (etc.). `serve` and `info()`/account-setup stay on the generic
base (even non-messaging transports run and have account config); only `validate_account` + the
feature accessors are messaging-specific. `messaging()` with `self: Arc<Self>` receivers is
object-safe, so the registry can hold `Arc<dyn TransportAdapter>` and recover `Arc<dyn MessagingProtocol>`.

### 3.2 `SupportsConversations` (← `purpleprotocolconversation.h`)

Ported verbatim; every method defaults to `Err(ApiError::Unsupported)` and is gated by `supported()`.
There is deliberately **no** `invite`/`add_member` verb (libpurple has none — see §7).

```rust
#[async_trait]
pub trait SupportsConversations: Send + Sync {
    /// Per-verb probe (← the `implements_*` family).
    fn supported(&self) -> ConversationOps;

    // enumerate (per-adapter; the host fans out, §6.2 — a deliberate divergence from libpurple's
    // central PurpleConversationManager). Rooms reads the store; Matrix reads its synced room list.
    async fn list(&self, transport: TransportId) -> Vec<ConversationInfo>;
    async fn get(&self, transport: TransportId, conv: String) -> Option<ConversationInfo>;

    // create (dm / group dm)
    async fn create_details(&self, transport: TransportId) -> CreateConversationDetails;
    async fn create(&self, transport: TransportId, details: CreateConversationDetails)
        -> Result<ConversationInfo, ApiError>;

    // join (channel)
    async fn channel_join_details(&self, transport: TransportId) -> ChannelJoinDetails;
    async fn join_channel(&self, transport: TransportId, details: ChannelJoinDetails)
        -> Result<ConversationInfo, ApiError>;

    async fn leave(&self, transport: TransportId, conv: String) -> Result<(), ApiError>;
    async fn send(&self, transport: TransportId, conv: String, from: Option<Participant>, message: UserMsg) -> Result<(), ApiError>;
    async fn set_topic(&self, transport: TransportId, conv: String, topic: Option<String>) -> Result<(), ApiError>;
    async fn set_title(&self, transport: TransportId, conv: String, title: Option<String>) -> Result<(), ApiError>;
    async fn set_description(&self, transport: TransportId, conv: String, description: Option<String>) -> Result<(), ApiError>;
    async fn set_avatar(&self, transport: TransportId, conv: String, avatar: Option<Image>) -> Result<(), ApiError>;

    fn send_typing(&self, transport: TransportId, conv: String, state: TypingState);
    fn refresh(&self, transport: TransportId, conv: String);
}

/// The `implements_*` projection: which conversation verbs this adapter supports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConversationOps {
    pub create: bool, pub join_channel: bool, pub leave: bool, pub send: bool,
    pub set_topic: bool, pub set_title: bool, pub set_description: bool,
    pub set_avatar: bool, pub send_typing: bool, pub refresh: bool,
}
```

> Conversation *enumeration* (`list`/`get`) is a **per-adapter** method on `SupportsConversations` here
> (the host fans out, §6.2) — a deliberate divergence from libpurple's central
> `PurpleConversationManager` (chosen for simplicity; no central index). Rooms reads the store; Matrix
> reads its synced room list.

### 3.2.1 `SupportsMembership` — membership administration

Outbound administration of an *existing* conversation's participants. libpurple 3's snapshot defers
these to the slash-command layer, but invite is first-class cross-protocol in libpurple 2
(`chat_invite`), Adium (`AIAccount inviteContact:toChat:withMessage:`), and Kopete
(`ChatSession::inviteContact` + `mayInvite`) — see §2.3.1 and §7. Kick/ban are protocol-specific (Adium
issues raw IRC `KICK`/`MODE +b`); roles mirror libpurple `PurpleBadge`s / XMPP affiliations. A split
interface like the rest, gated by `supported()`; methods default to `Err(ApiError::Unsupported)`.

```rust
#[async_trait]
pub trait SupportsMembership: Send + Sync {
    fn supported(&self) -> MembershipOps;
    /// Invite/add a participant (← libpurple2 `chat_invite` / Adium `inviteContact:toChat:`). `who` is a
    /// human `Contact` (Matrix/IRC) or an `Agent` (Rooms binds `profile -> session`) — see `Participant`.
    async fn invite(&self, transport: TransportId, conv: String, who: Participant, message: Option<String>) -> Result<(), ApiError>;
    /// Remove/kick a participant (Matrix kick; IRC `KICK`; XMPP role=none).
    async fn remove(&self, transport: TransportId, conv: String, who: Participant, reason: Option<String>) -> Result<(), ApiError>;
    /// Ban a participant (Matrix ban; IRC `MODE +b`; XMPP affiliation=outcast). Optional.
    async fn ban(&self, transport: TransportId, conv: String, who: Participant, reason: Option<String>) -> Result<(), ApiError>;
    /// Set a participant's role/affiliation (← Adium `AIGroupChatFlags` / XMPP affiliation). Optional.
    async fn set_role(&self, transport: TransportId, conv: String, who: Participant, role: MemberRole) -> Result<(), ApiError>;
}

/// The `implements_*` projection for membership administration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MembershipOps { pub invite: bool, pub remove: bool, pub ban: bool, pub set_role: bool }
```

> Observed occupants remain the `ConversationMembers` state the protocol populates from sync (§5);
> `SupportsMembership` is the *outbound* admin counterpart, distinct from that observation.

### 3.3 `SupportsRoster` (← `purpleprotocolroster.h`)

The **account-level server-side contact list** (buddy list), keyed by account + `ContactInfo`. Not
conversation membership.

```rust
#[async_trait]
pub trait SupportsRoster: Send + Sync {
    fn supported(&self) -> RosterOps; // { add, update, remove }
    async fn add(&self, transport: TransportId, contact: ContactInfo) -> Result<(), ApiError>;
    async fn update(&self, transport: TransportId, contact: ContactInfo) -> Result<(), ApiError>;
    async fn remove(&self, transport: TransportId, contact: ContactInfo) -> Result<(), ApiError>;
}
```

### 3.4 `SupportsContacts` (← `purpleprotocolcontacts.h`)

```rust
#[async_trait]
pub trait SupportsContacts: Send + Sync {
    fn supported(&self) -> ContactsOps; // { get_profile, action_menu, set_alias }
    async fn get_profile(&self, transport: TransportId, contact: ContactInfo) -> Result<String, ApiError>;
    fn action_menu(&self, transport: TransportId, contact: ContactInfo) -> Option<ActionMenu>;
    async fn set_alias(&self, transport: TransportId, contact: ContactInfo, alias: Option<String>) -> Result<(), ApiError>;
}
```

### 3.5 `SupportsDirectory` (← `purpleprotocoldirectory.h`)

Also libpurple's `roomlist` successor (room/channel *discovery*); complements
`SupportsConversations::list` (joined conversations) with discoverable ones.

```rust
#[async_trait]
pub trait SupportsDirectory: Send + Sync {
    fn supported(&self) -> bool;
    async fn search_contacts(&self, transport: TransportId, query: Option<String>) -> Result<Vec<ContactInfo>, ApiError>;
}
```

### 3.6 `SupportsFileTransfer` (← `purpleprotocolfiletransfer.h`)

```rust
#[async_trait]
pub trait SupportsFileTransfer: Send + Sync {
    fn supported(&self) -> FileTransferOps; // { send, receive }
    async fn send(&self, transfer: FileTransfer) -> Result<(), ApiError>;
    async fn receive(&self, transfer: FileTransfer) -> Result<(), ApiError>;
}
```

### 3.7 Two-level capability gating

1. **Interface presence** — `adapter.conversations()` is `Some` (≈ `PURPLE_IS_PROTOCOL_CONVERSATION`).
2. **Per-verb** — `SupportsConversations::supported().set_topic` (≈
   `purple_protocol_conversation_implements_set_topic`).

The coarse `AdapterInfo.capabilities` bools remain only for the "Add channel" picker; fine
affordance-gating in a GUI uses `supported()`.

---

## 4. Object-safety & the capability-recovery chain

Every trait here is object-safe (no by-value `Self`, no generic methods; `supported()` returns a
concrete struct), so the registry holds `Arc<dyn TransportAdapter>` and the host recovers capabilities
through a chain of `Option`-returning accessors with `self: Arc<Self>` receivers:
`TransportAdapter::messaging()` → `MessagingProtocol::conversations()`/`membership()`/… → the feature
trait. A `None` at any link is the daemon analogue of libpurple's `PURPLE_IS_PROTOCOL_*` returning
false, and yields `ApiError::Unsupported` at the wire.

---

## 5. Data model

Ported from libpurple, translated to Rust/serde. New types live in `daemon-api`.

```rust
/// ← PurpleConversationType (purpleconversation.h). `Unset` is retained for faithful round-trips.
pub enum ConversationType { Unset, Dm, GroupDm, Channel, Thread }

/// A conversation as the host/GUI sees it (the `list`/`get` projection; ≈ the conversation object
/// plus its members). `id` is the adapter-opaque handle within `transport`.
pub struct ConversationInfo {
    pub transport: TransportId,
    pub id: String,
    pub kind: ConversationType,
    pub title: Option<String>,
    pub topic: Option<String>,
    pub description: Option<String>,
    pub members: Vec<ConversationMember>,
}

/// ← PurpleConversationMember. Observed occupant state the adapter populates from sync.
pub struct ConversationMember {
    pub contact: ContactInfo,
    pub alias: Option<String>,
    pub nickname: Option<String>,
    pub typing: TypingState,
    pub role: MemberRole, // observed role/affiliation, populated from sync (← libpurple badges / XMPP affiliation)
    // daemon extension (§8): the engine incarnation this participant drives, when it is an agent.
    pub session: Option<SessionId>,
}

/// ← Adium `AIGroupChatFlags` / libpurple `PurpleBadge`s / XMPP affiliations+roles. The observed
/// per-participant role; `SupportsMembership::set_role` (§3.2.1) is the optional outbound counterpart.
pub enum MemberRole { None, Voice, HalfOp, Op, Founder }

/// ← PurpleContactInfo.
pub struct ContactInfo {
    pub id: String,
    pub display_name: Option<String>,
    pub presence: Presence,
    pub permission: ContactPermission,
}
/// ← PurpleContactInfoPermission.
pub enum ContactPermission { Unset, Allow, Deny }

/// Who an `invite`/`remove`/`ban`/`set_role` targets, and the `send` author. `Contact` is the faithful
/// libpurple identity (a human/remote contact); `Agent` is the delineated daemon extension (§8) — an
/// agent bound as a participant: `member` is its in-conversation @handle, `profile` resolves to a session.
pub enum Participant {
    Contact(ContactInfo),
    Agent { profile: ProfileRef, member: String },
}

/// ← PurplePresence + PurplePresencePrimitive (faithful 8-value set).
pub struct Presence {
    pub primitive: PresencePrimitive,
    pub message: Option<String>,
    pub emoji: Option<String>,
    pub mobile: bool,
    pub idle_since: Option<u64>, // unix seconds; None = not idle
}
pub enum PresencePrimitive { Offline, Available, Idle, Invisible, Away, DoNotDisturb, Streaming, OutOfOffice }

/// ← PurpleConnectionState.
pub enum ConnectionState { Disconnected, Disconnecting, Connected, Connecting }

/// ← PurpleTypingState.
pub enum TypingState { None, Typing, Paused }

/// ← PurpleCreateConversationDetails. Typed common core + adapter-described protocol-specific extras.
pub struct CreateConversationDetails {
    pub max_participants: u32,            // 0 = unlimited
    pub participants: Vec<ContactInfo>,   // the INITIAL participants, set at create time; live add/remove
                                          // on an existing conversation is SupportsMembership (§3.2.1)
    pub extras_schema: AccountSettingsSchema, // adapter-provided form for protocol-specific fields
    pub extras: AccountSettingsValues,        // the filled values (Rooms puts its floor policy here)
}

/// ← PurpleChannelJoinDetails.
pub struct ChannelJoinDetails {
    pub name: Option<String>,            pub name_max_length: u32,
    pub nickname: Option<String>,        pub nickname_supported: bool, pub nickname_max_length: u32,
    pub password: Option<String>,        pub password_supported: bool, pub password_max_length: u32,
    pub extras_schema: AccountSettingsSchema, pub extras: AccountSettingsValues,
}
```

Reuse / reconcile with existing `daemon-api` types:

- `Message` — reuse `UserMsg` (and the merged session log) for `send`; the spec does not introduce a
  parallel message type. (libpurple `PurpleMessage` ≈ the `UserMsg` + log entry pair.)
- `AccountSettingsSchema` / `AuthParamField` — already present (the libpurple typed-settings analogue);
  `AccountSettingsValues` is the filled-form companion (add if absent).
- `ConnectionState` — already present; reconcile its variants to the libpurple four
  (`Disconnected`/`Disconnecting`/`Connected`/`Connecting`).
- `PresenceState` — the existing daemon enum is reconciled/replaced by the faithful 8-value
  `PresencePrimitive` (the implementation either renames `PresenceState` or maps onto it).
- `Image`, `ActionMenu`, `FileTransfer` — minimal carriers introduced for the deferred-impl
  interfaces (avatars, contact action menus, transfers); shapes ported from
  `purpleimage.h`/`birb` action menu/`purplefiletransfer.h` when those interfaces are implemented.
- `Person` (← `purpleperson.h`, the MetaContact) — **implemented** (W3-J `port-person`):
  `crates/contracts/daemon-api/src/person.rs` (+ host `PersonManager` in
  `crates/substrate/daemon-host/src/person.rs`, `PersonList` wire op; see §10/§12).

---

## 6. Registry, host forwarding, and the wire vocabulary

### 6.1 Registry + lifecycle (← `PurpleProtocolManager`)

`AdapterRegistry` (`daemon-host`) holds the registered `Arc<dyn TransportAdapter>`s and gains:

- `adapter_for_family(family) -> Option<Arc<dyn TransportAdapter>>`,
  `adapter_for_transport(&TransportId) -> Option<…>` (maps `room`, `matrix/<account>`, … to a family via
  `family()`). A Room is addressed as a *conversation* (`transport="room"`, `conv=<room id>`), not a
  transport per room; the per-room `room/<id>` loopback `TransportId` is an internal Rooms delivery
  detail (fan-out/subscribe), not management-addressable (§7).
- `instances()` fan-out over adapters.
- `spawn_all(api) -> Vec<JoinHandle<()>>` driving `adapter.clone().serve(api.clone())` — the
  registry-owned lifecycle that retires the bespoke `daemon_matrix::serve`/`daemon_rooms::serve` spawn
  blocks in `bins/daemon/src/main.rs`.

### 6.2 Host forwarding (no family switch)

`NodeApiImpl` implements each management op by resolving the owning adapter from the registry,
calling `messaging()` then the relevant feature accessor, checking `supported()`, and forwarding —
returning `ApiError::Unsupported` when the transport is not a messaging protocol, the feature accessor
is `None`, or the verb is off. Conversation enumeration (`ConvList`/`ConvGet`) forwards to the adapter's
`SupportsConversations::list`/`get` (the per-adapter model, §3.2). Mutating ops are audit-journaled (§9).

### 6.3 Wire vocabulary (CBOR/CDDL)

Typed `ApiRequest`/`ApiResponse` variants keyed by `transport` (+ `conversation` where relevant),
shared by every messaging adapter — the daemon analogue of the libpurple UI calling the protocol
vtable:

- Conversations: `ConvCreateDetails`/`ConvCreate`, `ConvJoinDetails`/`ConvJoin`, `ConvLeave`,
  `ConvList`/`ConvGet`, `ConvSend`, `ConvSetTopic`/`ConvSetTitle`/`ConvSetDescription`,
  `ConvSetAvatar`, `ConvSendTyping`.
- Membership: `MemberInvite`/`MemberRemove`/`MemberBan`/`MemberSetRole`.
- Roster: `RosterAdd`/`RosterUpdate`/`RosterRemove`.
- Contacts: `ContactGetProfile`/`ContactSetAlias`/`ContactActionMenu`.
- Directory: `DirectorySearch`.
- FileTransfer: `FileTransferSend`/`FileTransferReceive`.
- Enumeration: `TransportInstances` (+ existing `TransportAdapters`).

CDDL group defs are added to `daemon-api.cddl` for every new DTO and op; `WireVersion::CURRENT` bumps
to **18** (changelog in `daemon-common`; the CDDL `current = N` comment updated). The stubbed
`room_*` `ApiRequest`/`ApiResponse`/`ControlApi`/`dispatch` ops are **retired** — Room management
rides the generic `Conv*`/membership vocabulary. The host does not runtime-validate CBOR against the
CDDL; the owning adapter decodes into its own typed structs (serde is the validator), exactly as every
other wire op already works.

---

## 7. Resolved by libpurple 3 source study

- **Membership administration lives in `SupportsMembership`, not `SupportsConversations`.** A grep for
  `invite|kick|occupant|add_user|remove_user|ban` across the libpurple-3 headers in this snapshot
  returns nothing — but that reflects an in-progress rewrite, not the stable contract. Invite is
  first-class in libpurple 2 (`chat_invite` + `serv_chat_invite`), Adium
  (`AIAccount inviteContact:toChat:withMessage:`), and Kopete (`ChatSession::inviteContact` +
  `mayInvite`); this snapshot merely deferred it to the `PurpleCommand` slash-command layer
  (`/quote INVITE`). Kick/ban are real but protocol-specific (Adium issues raw IRC `KICK`/`MODE +b`);
  per-member roles are observed flags (libpurple `PurpleBadge`s / Adium `AIGroupChatFlags` / XMPP
  affiliations). daemon therefore ports these as the
  [`SupportsMembership`](#321-supportsmembership--membership-administration) capability (§2.3.1),
  keeping `SupportsConversations` free of them and documenting the divergence from the snapshot.
  `SupportsRoster` remains the *account* contact list, separate from conversation membership.
- **Participants at create time vs. live invite.** `CreateConversationDetails.participants` is the
  initial set at create time (item type `Contact`, with `NO_PARTICIPANTS`/`TOO_MANY_PARTICIPANTS`
  validation; ← `purplecreateconversationdetails.h`). Adding/removing a participant on an *existing*
  conversation is `SupportsMembership::invite`/`remove`. Beyond those, occupants also change as
  observed `ConversationMembers` state the protocol syncs (e.g. someone self-joins or parts).
- **`send` vs delivery.** `SupportsConversations::send` is libpurple's `send_message` — an explicit
  send into a conversation. It is distinct from `daemon-delivery`'s automatic agent→world reply
  projector. Rooms `send` fans out; Matrix `send` is `room.send`.
- **Conversation addressing & Room identity.** `transport: TransportId` + opaque `conversation: String`
  handle, mirroring libpurple's per-account conversation identity (`PurpleConversation` keyed by
  account + type + id; see `PurpleConversationManager::find`). A Room is therefore a *conversation*
  within the single `"room"` transport (`conv` = room id), **not** a transport per room; the per-room
  `room/<id>` loopback `TransportId` stays an internal Rooms delivery detail, not management-addressable.
- **`Unset` enum members.** libpurple keeps `*_UNSET` variants (conversation type, permission);
  ported faithfully for round-trip fidelity.

---

## 8. Daemon extensions (delineated)

Recorded separately from the faithful port; these have no libpurple counterpart.

> Membership administration (add/remove a participant on a live conversation) is **not** a daemon
> extension: it is the ported [`SupportsMembership`](#321-supportsmembership--membership-administration)
> capability (§2.3.1/§3.2.1), grounded in the libpurple-2/Adium/Kopete contract, which Rooms implements
> (`invite` = bind a participant + persist + reconcile; `remove` = unbind). What remains genuinely
> daemon-specific:

- **Agent-as-participant.** `ConversationMember.session: Option<SessionId>` binds a member to a daemon
  engine incarnation. libpurple's member wraps a human `ContactInfo`; the session binding is purely a
  daemon concept.
- **Floor-control policy.** `RoomPolicy` (AddressedOnly/FreeForAll/RoundRobin/Moderator) rides in
  `CreateConversationDetails.extras` under the Rooms adapter's `extras_schema`. A Rooms concern with no
  libpurple counterpart.

---

## 9. dCBOR management auditing

Management mutations are recorded on the verifiable journal as dCBOR Gordian Envelopes, mirroring the
existing **credential-audit** precedent (`daemon-host/src/journal.rs`:
`JournalSink::record_management` + `seal`, the `drain_credential_audit` record-then-seal shape, sealed
under the node's seed-derived `TraceSigner`; credential audit uses a per-node
`JournalStreamId::unit("node-credentials")` stream wired in `bins/daemon/src/main.rs`).

- `NodeApiImpl` gains a `management_journal: Option<Arc<JournalSink>>` built in `with_journal` (it
  already receives `store` + `signer`; the signer is also held as `verifier`) over
  `JournalStreamId::unit("node-management")`.
- At the host forwarding choke point, every mutating op (`ConvCreate`/`ConvLeave`/`ConvSet*`/
  `ConvSend`/`Member*` (invite/remove/ban/set_role)/`Roster*`) records one `mgmt.*` entry (actor/origin +
  transport + conversation + params summary) then `seal()`s. Uniform across all adapters, zero per-adapter code.
- `ConvSend` *content* is already journaled on each member session's stream (the delivery
  `JournalFeeder`); the management entry adds the missing **admin attribution** (who created/changed/
  sent-as).

---

## 10. Reference adapters

### 10.1 Rooms (`daemon-rooms`) — Conversations + Membership

Implements `TransportAdapter` (`family() == "room"`, `info()`, `instances()` reporting one `room`
instance `Connected`, `messaging()` returning `Some(self)`), `MessagingProtocol`,
`SupportsConversations`, and `SupportsMembership`:

- `create` = create + persist a room (`store.room_set`) and reconcile the live router; `kind` is
  `GroupDm`/`Channel`; `extras` carries the floor policy.
- `send(from, …)` injects an operator/`from`-attributed post via the floor-gated `RoomInbound::fan_out`
  (`FloorControl`, working for `AddressedOnly`/`FreeForAll`); the agent-reply loop is delivery (the
  projector re-injecting `TurnFinished`), not `send`.
- `set_topic`/`set_title`/`set_description` update stored room metadata; `leave`/`join_channel`
  supported; `set_avatar`/`send_typing`/`refresh` reported off in `supported()`.
- `serve(self, api)` owns the live `RoomRouter`, runs `RoomRouter::load()` reading the **store** (fixes
  today's empty-`api.room_list()` early-return), subscribes delivery, and selects over delivery events
  + an `mpsc` command channel.
- `SupportsMembership`: `invite(Participant::Agent { profile, member })` binds the agent (resolve
  `profile -> session`, `store.room_member_set`, reconcile into the live `Membership`); `remove` unbinds
  (`store.room_member_remove`); `supported()` = `{ invite, remove }` (`ban`/`set_role` off — floor policy
  is a separate Rooms concern, §8).

### 10.2 Matrix (`daemon-matrix`) — proves a different support set

Implements `TransportAdapter` (`family() == "matrix"`, `instances()` enumerating bound accounts +
`ConnectionState`, `serve` wrapping the existing `serve(api, provisioning, cfg)` body, `messaging()`
returning `Some(self)`), `MessagingProtocol`, a `SupportsConversations` subset (`send` via matrix-sdk `room.send` and `set_topic`
exist today; `list`/`get` from synced rooms) — and (new work) `SupportsMembership`
(`invite(Participant::Contact)`/`remove`/`ban` via `m.room.member` invite/leave/ban; `set_role` via power
levels, or off initially). Demonstrates two adapters on one interface with different `supported()` sets —
Matrix's membership ops are richer than Rooms' — and no host changes.

### 10.3 Mechanics unchanged

`daemon-ingest`/`daemon-delivery` are not touched: inbound messages still flow `Ingestor::receive →
submit_routed`, and agent replies still flow through the delivery `Projector`. This interface adds only
the management verbs and the descriptive/lifecycle layer.

---

## 11. libpurple 3 surface inventory (port-now vs defer)

Each row cites its source header. "Now" = part of the first implementation; "Defer" = ported as a
defined-but-unimplemented contract (interface/DTO present, `supported()` reports off / no adapter
returns the accessor).

Protocol interfaces:

- `purpleprotocol.h` (base) — **Now** (as `MessagingProtocol`, the specialization of the generic
  pre-existing `TransportAdapter`).
- `purpleprotocolconversation.h` — **Now** (`SupportsConversations`; Rooms full, Matrix subset).
- Membership administration — **Now** as `SupportsMembership` (`invite`/`remove`: Rooms + Matrix;
  `ban`/`set_role`: Matrix, off in Rooms). No single libpurple-3 header; grounded in libpurple-2
  `chat_invite` + Adium `AIAccount`/`AIGroupChatFlags` + Kopete `inviteContact`/`mayInvite` (§2.3.1).
- `purpleprotocolroster.h` — **Now** as a defined interface; **Defer** real impl (no adapter
  implements account-contact management yet).
- `purpleprotocolcontacts.h`, `purpleprotocoldirectory.h`, `purpleprotocolfiletransfer.h` — **Now**
  defined; **Defer** impl.
- `purpleprotocolwhiteboard.h` — **Defer** entirely (niche).

Data model:

- `purpleconversation.h` (type enum + conversation), `purpleconversationmember(s).h`,
  `purplecreateconversationdetails.h`, `purplechanneljoindetails.h`, `purpletyping.h`,
  `purpleconnection.h` (state), `purplepresence.h` (primitive), `purplecontactinfo.h`
  (+ permission), `purpleaccountsetting*.h` (typed settings) — **Now**.
- `purplemessage.h` — **Now** via reuse of `UserMsg`/the merged log (no new type).
- `MemberRole` (observed per-member role; ← `purplebadges.h` / Adium `AIGroupChatFlags` / XMPP
  affiliations) — **Now** (observed field on `ConversationMember`); `set_role` outbound optional per-adapter.
- `purpleperson.h` (MetaContact) — **Now** (W3-J `port-person`: `daemon_api::person` +
  the host `PersonManager`, `PersonList` wire op).
- `purplebadges.h`, `purpleimage.h`, `purplefiletransfer.h`,
  `purpleauthorizationrequest.h`, `purpleaddcontactrequest.h`, `purpletags.h`, `purplecontact.h`
  (the roster-side contact wrapper) — **Defer** (ported as the dependent interfaces land).
- `purpleaccount.h`/account manager — partially **Now** via the existing `TransportInstanceInfo` +
  `AccountProvisioning`; full port **Defer**.

This inventory is the checklist the implementation follows; anything not listed is resolved by
reading the source at implementation time.

---

## 12. Phasing, acceptance criteria, and relationship to the transport-adapter framework

### 12.1 Phasing

- **Landed:** the generic `TransportAdapter` base + the `MessagingProtocol` specialization +
  `SupportsConversations` + `SupportsMembership` (invite/remove); Rooms full (incl. all four floor
  policies and `RoomProjector` re-injection + sealed transcript); Matrix subset (incl. membership
  invite/remove/ban, and `SupportsContacts`/`SupportsDirectory` — `get_profile` + directory search);
  the wire vocabulary + CDDL + `WireVersion` 18–20; registry-driven lifecycle (`AdapterRegistry::spawn_all`
  via `node.spawn_adapters()`, with `daemon-rooms`/`daemon-matrix` registered in `bins/daemon`); dCBOR
  management audit; CLI; conformance test. `SupportsMembership` `ban`/`set_role` implemented by Matrix
  (power levels) and off in Rooms.
- **Deferred:** real `SupportsRoster`/`SupportsFileTransfer` (defined but no implementor); Matrix admin
  beyond the landed `send`/`set_topic`/membership; the daemon-app GUI client. (`Person`/MetaContact
  unification is no longer deferred — implemented by W3-J `port-person`: `daemon_api::person` +
  host `PersonManager`.)

### 12.2 Acceptance criteria (for the implementation)

- Management rides the typed `Conv*`/`Member*` vocabulary (no generic action bus); `xtask cddl` parity
  passes with the new variants and the `room_*` ops removed from both Rust and CDDL.
- Over the unix socket (daemon-conformance `node_interface`): `ConvCreate("room", …)` then
  `ConvList("room")` returns it; `ConvSetTopic` reflects in `ConvGet`; `ConvSend` to an addressed
  member opens a turn on that member's session; `MemberInvite` reflects in `ConvGet.members` and
  `MemberRemove` removes them (for Rooms the invited member's `session` binding drives fan-out, so a
  subsequent `ConvSend` opens a turn on it);
  `TransportInstances` returns rooms + matrix with the matrix `ConversationOps` reporting the right
  subset; a mutating op produces a sealed dCBOR `mgmt.*` entry on the `node-management` stream
  (assert via `store.load_trace_segment`, mirroring `drain_records_audit_into_the_journal`).
- CBOR round-trip unit tests in `daemon-api` for the new DTOs and ops.

### 12.3 Relationship to the transport-adapter framework

- `daemon-transport-adapter-spec.md`: this spec **specializes**, it does not supersede, that framework.
  Its `TransportAdapter` base, host `AdapterRegistry`, `AdapterInfo`/`AdapterCapabilities`,
  `TransportInstanceInfo`, and presence/connection DTOs remain the live foundation (this spec only
  reconciles the presence primitive to libpurple's 8-value set). The one place it diverges: that spec's
  **§3.1** *marker* capability traits (`SupportsRooms`/`SupportsPresence`/`SupportsRoomEnumeration`/
  `SupportsFileTransfer`/`SupportsAuth`) were a sketch that was never built — coarse capability is
  carried by the `AdapterCapabilities` bool struct, and the fine-grained per-verb capability by this
  spec's typed feature traits (`SupportsConversations`/`SupportsMembership`/… each with a `supported()`
  probe), reached via `TransportAdapter::messaging()`. So §3.1's marker traits and §7's P1 are
  *realized by* this spec's feature-trait family, not discarded.
- `daemon-rooms-spec.md`: Room management is realized via this messaging-adapter interface
  (`SupportsConversations` + `SupportsMembership`), not bespoke `room_*` ops (which were retired).
- `daemon-matrix-transport-spec.md`: Matrix is a reference implementor of this interface.
