# Matrix adapter port reference: matrix-bifrost / purple-matrix → daemon-matrix

Status: implementation reference. This is the source audit that grounds the `daemon-matrix` messaging
adapter implementation. It maps the two reference Matrix clients in the libpurple ecosystem —
`matrix-bifrost` (the modern TypeScript bridge) and `purple-matrix` (the classic libpurple 2 C prpl
plugin) — onto the daemon `MessagingProtocol` trait family, with file/line citations, the
daemon-specific deltas, and the deliberate gaps the daemon adapter supersedes via `matrix-sdk`.

Companion to `daemon-messaging-adapter-spec.md` (the trait contract — the spec's authority is the
libpurple 3 header tree) and `daemon-matrix-transport-spec.md` (the Matrix transport's lifecycle).
This document is the *port* reference: how a working Matrix client wires the prpl/bridge logic, and
where ours diverges because it is built on `matrix-sdk` rather than raw HTTP + libolm.

> Reference source trees (read-only clones, not vendored):
> - `matrix-bifrost`: `/home/j/experiments/multiprotocol-instant-messengers/matrix-bifrost` (TypeScript)
> - `purple-matrix`: `/home/j/experiments/multiprotocol-instant-messengers/purple-matrix` (C, libpurple 2)
> - libpurple 3 headers (the trait authority): `…/pidgin-496de266ac6c/libpurple/`

---

## 1. The two references, contrasted

| | matrix-bifrost | purple-matrix | daemon-matrix |
| --- | --- | --- | --- |
| Role | Matrix ↔ remote-protocol **bridge** (Matrix is the *near* side, libpurple the *far* side) | libpurple **prpl plugin** (Matrix is the protocol libpurple drives) | `MessagingProtocol` adapter (Matrix is a transport the daemon drives as a client) |
| Matrix access | appservice (server-side, plaintext) | raw HTTP client-server API | `matrix-sdk` 0.18 client |
| E2EE | **none** (classic appservice) | decrypt-only (libolm; no outbound encryption) | automatic via `matrix-sdk` `e2e-encryption` (encrypt-on-send + decrypt-on-sync) |
| Login | n/a (appservice token) | password + stored-token; **no SSO** | SSO (`sso-login`) + token write-back |
| Backend contract | `IBifrostInstance` + `IBifrostAccount` (`src/bifrost/`) | `PurplePluginProtocolInfo` vtable (`libmatrix.c`) | the `Supports*` trait family (`daemon-api`) |

The important structural lesson is that **bifrost already abstracted libpurple behind a protocol
interface** (`IBifrostInstance`/`IBifrostAccount`), and that interface is the same shape as the
daemon trait family. purple-matrix is the concrete Matrix wire knowledge (which endpoints, which sync
fields) that `matrix-sdk` now encapsulates for us. So the port is: take bifrost's *interface
decomposition*, take purple-matrix's *Matrix semantics*, and realize both through `matrix-sdk`.

---

## 2. matrix-bifrost: the backend abstraction (what we are porting)

bifrost reaches libpurple only inside `src/purple/`; everything else is backend-agnostic and talks to
the two interfaces below. node-purple is imported only in `src/purple/PurpleInstance.ts` and
`src/purple/PurpleAccount.ts`.

### 2.1 `IBifrostInstance` (`src/bifrost/Instance.ts:23-60`)

The instance-level surface (≈ daemon `TransportAdapter` + `MessagingProtocol`):

- `start()` / `close()` — backend lifecycle. libpurple init + the 300 ms `helper.pollEvents()` loop
  lives in `src/purple/PurpleInstance.ts:63-94`.
- `getAccount(username, protocolId, mxid?)` / `getProtocol(id)` / `getProtocols()` — account + protocol
  registry (`src/purple/PurpleInstance.ts:96-107`).
- `createBifrostAccount(username, protocol)` — account factory.
- `on(eventName, cb)` typed overloads (`:36-52`) — the inbound event subscription contract.
- `needsDedupe()` / `needsAccountLock()` — both `true` for libpurple.

### 2.2 `IBifrostAccount` (`src/bifrost/Account.ts:19-48`)

The per-account surface (≈ daemon `SupportsConversations` + `SupportsMembership`, scoped to one
account):

| bifrost method | `src/purple/PurpleAccount.ts` | daemon trait method |
| --- | --- | --- |
| `sendIM(recipient, body)` | `:69-76` → `messaging.sendIM` | `SupportsConversations::send` (DM room) |
| `sendChat(chatName, body)` | `:83-84` → `messaging.sendChat` | `SupportsConversations::send` (group room) |
| `sendIMTyping(recipient, isTyping)` | `:79-80` → `messaging.setIMTypingState` | `SupportsConversations::send_typing` (deferred) |
| `joinChat(components)` | `:107-133` → `messaging.joinChat` | `SupportsConversations::join_channel` |
| `rejectChat(components)` | `:136-137` → `messaging.rejectChat` | `SupportsConversations::leave` |
| `getChatParamsForProtocol()` | `:152` → `messaging.chatParams` | `SupportsConversations::channel_join_details` |
| `setEnabled(enable)` | `:64-66` → `accounts.set_enabled` | adapter `serve()` bring-up (lifecycle) |
| `createNew(password, cfg)` | `:59-61` → `accounts.new`+`configure` | `AccountProvisioning` + `auth`/`login` (SSO) |
| `setStatus(statusId, active)` | `:177-178` → `accounts.set_status` | presence (deferred) |
| `getBuddy(user)` / `getUserInfo` | `:87-88` / `:156` | `SupportsContacts::get_profile` (deferred) |

### 2.3 Inbound events (backend → bridge → Matrix)

Polled from libpurple via `helper.pollEvents()` (`src/purple/PurpleInstance.ts:169`) and re-emitted by
name (`:197`). The Matrix-side handlers live in `src/MatrixRoomHandler.ts`:

| bifrost event | handler | daemon equivalent |
| --- | --- | --- |
| `received-im-msg` | `MatrixRoomHandler.ts:392-461` | `inbound::on_room_message` → `Ingestor::receive` (DM) |
| `received-chat-msg` | `MatrixRoomHandler.ts:464-536` | `inbound::on_room_message` → `Ingestor::receive` (group) |
| `chat-joined` / `chat-joined-new` | `MatrixRoomHandler.ts:54-78` | room appears in synced room list (`list`/`get`) |
| `chat-invite` | `MatrixRoomHandler.ts:539-573` | inbound invite (observed; reaction deferred) |
| `chat-user-joined/left/kick` | `MatrixRoomHandler.ts:575-636` | `ConversationMember` sync (occupant state) |
| `chat-topic` | `MatrixRoomHandler.ts:639-666` | `ConversationInfo.topic` from synced state |
| `im-typing` / `chat-typing` | `MatrixRoomHandler.ts:668-700` | `ConversationMember.typing` (deferred) |
| `account-signed-on/off`, `account-connection-error` | `Program.ts:294-324` | `ConnectionState` in `instances()` |

> Key takeaway: bifrost's inbound events are exactly the daemon split between **the ingest edge**
> (messages → `Ingestor`) and **observed conversation state** (members/topic/typing surfaced by
> `list`/`get`). `matrix-sdk`'s sync + state store gives us the latter for free; we only wire the
> former (already done in `inbound.rs`).

---

## 3. purple-matrix: the Matrix wire semantics (what matrix-sdk now owns)

purple-matrix is the libpurple 2 prpl (`PRPL_ID = "prpl-matrix"`). Its value here is the concrete
Matrix client-server semantics, all of which `matrix-sdk` encapsulates.

### 3.1 prpl vtable (`libmatrix.c:272-358`)

| vtable field | C function | line | daemon trait method |
| --- | --- | --- | --- |
| `login` | `matrixprpl_login` | `296` | adapter `serve()` + `auth`/`login` (SSO) |
| `close` | `matrixprpl_close` | `297` | `serve()` task shutdown |
| `chat_info` / `chat_info_defaults` | `matrixprpl_chat_info` | `294-295` | `channel_join_details` |
| `join_chat` | `matrixprpl_join_chat` | `314` | `join_channel` |
| `reject_chat` / `chat_leave` | `matrixprpl_reject_chat` / `_chat_leave` | `315/318` | `leave` |
| `chat_send` | `matrixprpl_chat_send` | `320` | `send` |
| `chat_invite` | `matrixprpl_chat_invite` | `317` | `SupportsMembership::invite` |
| `get_cb_real_name` | `matrixprpl_get_cb_real_name` | `333` | `ConversationMember.contact` mapping |

Notable NULLs (gaps purple-matrix never implemented): `set_chat_topic` (topic is display-only),
`send_im` (no DM model), `roomlist_get_list` (no directory), and no kick/ban/create-room. daemon
supersedes all of these via `matrix-sdk`.

### 3.2 Connection & sync (`matrix-connection.c`)

- Login: `matrix_connection_start_login` (`:359-384`) → stored-token via `/account/whoami` or password
  via `matrix_api_password_login`. **No SSO.** daemon uses `matrix-sdk` SSO (`auth.rs`).
- Sync loop: `_start_next_sync` (`:146-151`) → `matrix_api_sync(timeout=30000, full_state, …)` →
  `_sync_complete` (`:110-143`) sets `PURPLE_CONNECTED`, calls `matrix_sync_parse`, chains the next
  sync on `next_batch`. daemon uses `Client::sync(SyncSettings)` (`lib.rs:209`).
- `/sync` parse: `matrix_sync_parse` (`matrix-sync.c:267-352`) — two-pass (state before timeline),
  `to_device` decrypt, key-count replenish. `matrix-sdk` owns all of this.

### 3.3 Send / membership / state mapping

| purple-matrix path | line | matrix-sdk equivalent |
| --- | --- | --- |
| `matrix_room_send_message` → `matrix_api_send` (PUT `/rooms/{id}/send/m.room.message/{txn}`) | `matrix-room.c:1447`, `matrix-api.c:707` | `room.send(RoomMessageEventContent::text_plain(..))` |
| `matrix_api_invite_user` (POST `/invite`) | `matrix-api.c:753` | `room.invite_user_by_id(user)` |
| `matrix_api_leave_room` (POST `/leave`) | `matrix-api.c:873` | `room.leave()` |
| `matrix_api_join_room` (POST `/join/{idOrAlias}`) | `matrix-api.c:795` | `client.join_room_by_id_or_alias(..)` |
| membership state `m.room.member` → `matrix_roommembers_update_member` | `matrix-roommembers.c:175` | `room.members(RoomMemberships::ACTIVE)` → `ConversationMember` |
| topic display `m.room.topic` → `_on_topic_change` | `matrix-room.c:183` | `room.topic()` (read) / `room.set_room_topic()` (write — new in daemon) |

### 3.4 E2EE (`matrix-e2e.c`)

purple-matrix implements Olm/Megolm **decrypt-only** via libolm + sqlite (`matrix_e2e_decrypt_room`
`:1594`, `decrypt_olm` `:1410`), with no outbound encryption, no `/keys/query`, no device
verification. daemon delegates the entire crypto stack to `matrix-sdk` (`e2e-encryption` feature +
the on-disk sqlite crypto store created per account in `account::build_client`), which encrypts
outbound automatically in encrypted rooms and decrypts inbound before handing events to our
`m.room.message` handler. Device verification / cross-signing / key backup remain deferred.

---

## 4. Mapping to the daemon trait family

The daemon adapter is the union of bifrost's interface decomposition and purple-matrix's Matrix
semantics, realized on `matrix-sdk`. All trait/DTO definitions are in
`daemon-node/crates/contracts/daemon-api/src/lib.rs`.

### 4.1 `SupportsConversations` (`daemon-api/src/lib.rs:2480`)

| trait method | matrix-sdk realization | bifrost / purple-matrix origin |
| --- | --- | --- |
| `list(transport)` | `client.rooms()` → `room_to_info` | `chat-joined` / synced rooms |
| `get(transport, conv)` | `client.get_room(RoomId::parse(conv))` | — |
| `send(transport, conv, _from, msg)` | `room.send(RoomMessageEventContent::text_plain(..))` | `sendChat` / `matrix_room_send_message` |
| `set_topic(.., topic)` | `room.set_room_topic(&topic)` | new (purple-matrix is read-only) |
| `set_title(.., title)` | `room.set_name(title)` (`m.room.name`) | new |
| `create(.., details)` | `client.create_room(request)` from `extras` + `participants` | new (neither reference creates rooms) |
| `channel_join_details` / `join_channel(.., details)` | `client.join_room_by_id_or_alias(details.name)` | `getChatParamsForProtocol` / `joinChat` / `matrixprpl_join_chat` |
| `leave(.., conv)` | `room.leave()` | `rejectChat` / `matrix_api_leave_room` |
| `set_description` / `delete` | unsupported (Matrix has no native description or room-destroy) | — |

### 4.2 `SupportsMembership` (`daemon-api/src/lib.rs:2533`)

Targets `Participant::Contact(ContactInfo { id })` where `id` is the MXID (`@user:hs`).
`Participant::Agent` returns `Unsupported` (agent-as-participant is a Rooms-only daemon extension).

| trait method | matrix-sdk realization | origin |
| --- | --- | --- |
| `invite(.., who, _msg)` | `room.invite_user_by_id(user)` | `chat_invite` / `matrix_api_invite_user` |
| `remove(.., who, reason)` | `room.kick_user(user, reason)` | new (purple-matrix has no kick) |
| `ban(.., who, reason)` | `room.ban_user(user, reason)` | new |
| `set_role(.., who, role)` | power-levels update (`Founder=100`, `Op=50`, `HalfOp=25`, `Voice/None=0`) | observed badges in references; outbound is new |

### 4.3 DTO mapping (matrix-sdk → `daemon-api`)

- `room_to_info(room) -> ConversationInfo` (`daemon-api/src/lib.rs:2769`): `id = room.room_id()`,
  `kind` from `room.is_direct()`/member count (`Dm`/`GroupDm`/`Channel`), `title` from
  `room.display_name()`/`room.name()`, `topic = room.topic()`, `members` from
  `room.members(RoomMemberships::ACTIVE)`.
- `member_to_member(m) -> ConversationMember` (`:2747`): `contact = ContactInfo { id: m.user_id(),
  display_name: m.display_name(), presence: default, permission: Unset }`, `role` from the inverse
  power-level map, `session: None` (Matrix members are humans).

---

## 5. Daemon-specific deltas (no libpurple/bifrost counterpart)

- **Origin / ingest edge.** Inbound messages are normalized into an `Origin` and handed to the
  reusable `daemon-ingest` gate (`inbound::on_room_message`), not pushed straight at the UI as bifrost
  does. Addressing classification (mention/DM/`!command`) is the only transport-specific piece.
- **Delivery Projector.** Agent replies flow back through a `daemon-delivery` `Projector`
  (`outbound::MatrixProjector`), which also drives the gate's busy state from `TurnStarted`/
  `TurnFinished`. bifrost has no agent loop; this is the daemon's reason for existing.
- **Registry forwarding + capability probe.** The host resolves `adapter_for_transport → messaging()
  → conversations()/membership()` (`daemon-host/src/node_api.rs:490-525`) and forwards by capability,
  never by a family string — the typed analogue of bifrost's `instance.on(eventName)` dispatch.
- **dCBOR management audit.** Every mutating verb is journaled + sealed on the `node-management` stream
  (`node_api.rs:529`), a verifiable-journal concern with no messenger counterpart.
- **Per-account isolation.** Each Matrix account is its own `TransportId` (`matrix/@user:hs`) with its
  own `Client`, sqlite state + crypto store, and sync loop (`account.rs`, `lib.rs`) — analogous to
  bifrost's per-account `PurpleAccount`, but the management trait methods must resolve the right
  `Client` from a `serve()`-populated registry (the architectural seam this implementation adds).

---

## 6. Deliberate gaps the daemon adapter supersedes or defers

| Concern | bifrost | purple-matrix | daemon-matrix |
| --- | --- | --- | --- |
| Outbound E2EE | n/a | not implemented | automatic (matrix-sdk) |
| SSO login | n/a | not implemented | implemented (`auth.rs`/`login.rs`) |
| Create room | not implemented | not implemented | implemented (`create`) |
| Kick / ban | leave only | not implemented | implemented (`remove`/`ban`) |
| Set topic (outbound) | gateway only | read-only | implemented (`set_topic`) |
| Power levels / roles | observed only | observed only | implemented (`set_role`) |
| Presence | not wired to libpurple | status only | **deferred** |
| File transfer | media proxy | upload only | **deferred** |
| Device verification / cross-signing / key backup | n/a | not implemented | **deferred** |
| Roster / contacts / directory | partial | not implemented | **deferred** |

These deferred items remain defined-but-off in the adapter's `supported()` projections, exactly as the
spec's §11 port-now/defer inventory prescribes.
