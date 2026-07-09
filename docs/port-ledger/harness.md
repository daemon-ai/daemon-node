# Port ledger — W1-A `port-harness` (MessagingProtocol conformance)

Executable spec for the libpurple → daemon-api `MessagingProtocol` conformance harness.
Every libpurple `g_test` case in scope maps to a planned Rust test (or is skipped with a
reason). "Done" = every row is green or explicitly skipped-with-reason.

## Deliverables

- New dev-only crate `crates/contracts/daemon-api-testkit` (`publish = false`), auto-included
  by the root `members = ["crates/*/*", ...]` glob. Also declared in `[workspace.dependencies]`.
  - `EmptyProtocol` — a `MessagingProtocol` that exposes **every** `Supports*` feature-trait
    handle but reports **no** verb supported (`supported()` all-false) and leaves every verb at
    its trait default (→ `ApiError::Unsupported`). This is the faithful daemon analogue of
    libpurple's "Empty" fixtures (interface implemented with an empty `iface_init`, so
    `implements_*` is false and each verb warns/errors).
  - `FakeProtocol` — an in-memory reference impl of **every** `Supports*` feature trait, with a
    per-verb failure-switch map (`FailSwitches`) mirroring libpurple's per-fixture `should_error`
    boolean. Advertises all verbs supported; each verb returns `Ok` normally and
    `Err(ApiError::Other("fake:<verb> error"))` when its switch is on (a *non-`Unsupported`*
    error, so it never collides with the capability sentinel).
  - `assert_ops_match_behavior(Arc<dyn MessagingProtocol>)` — the cross-adapter invariant.

### Error variant used for "unsupported"

`daemon_api::ApiError::Unsupported(String)` (defined in `daemon-api/src/wire.rs`). Each trait
default method carries a **capability sentinel** string (e.g. `SupportsConversations::send` →
`Unsupported("conv_send")`). The harness keys the invariant on those exact sentinels.

### `assert_ops_match_behavior` — scope & the ⟺ decision

libpurple's biconditional is "`implements_X()==false` ⟺ the verb warns/errors". Translated to
daemon-api it is "`supported().<verb>==false` ⟺ the verb returns `Unsupported(<sentinel>)`".

- **Forward half (universal, enforced against every adapter):** for every optional verb, if
  `supported()` reports it **off**, calling it returns exactly `Err(ApiError::Unsupported(<the
  capability sentinel>))` (and non-`Result` accessors return their empty default — `action_menu`
  → `None`). This only ever invokes trait-**default** method bodies, so it performs **no** network
  or store I/O and is safe against a hermetically-constructed (unconnected) real adapter.
- **Reverse half (enforced against the reference `FakeProtocol` + connected `Fake`):** for every
  verb `supported()` reports **on**, calling it does **not** return the capability sentinel (it
  returns `Ok`, or a *different* error such as a transient `Unsupported("… not connected")`).
  `assert_ops_match_behavior` checks this half too, keyed on the sentinel string, so it is safe
  against real adapters: a real adapter that advertises `send` but is unconnected returns
  `Unsupported("<family> … not connected")` ≠ the `"conv_send"` sentinel and therefore passes.

  Divergence note: the reverse half is **not** "advertised ⟹ Ok" — a correctly-advertised verb
  legitimately fails transiently when the account is offline (which is the state of every
  hermetic adapter fixture). The sentinel keying is what makes the biconditional a safe,
  universal runtime invariant instead of a connection-dependent one.

## Data-model divergences (daemon-api vs libpurple `PurpleProtocolConversation`)

The daemon `SupportsConversations` verb set is `{create, join_channel, leave, delete, send,
set_topic, set_title, set_description}`. libpurple additionally has `set_avatar`, `send_typing`,
and `refresh`, and its detail getters return `NULL` when unimplemented. Consequences:

- `set_avatar` / `send_typing` / `refresh` cases → **skipped**: no such verb in the daemon
  `MessagingProtocol` model (out of scope for W1-A; not part of the wire contract).
- `get_create_conversation_details` / `get_channel_join_details`: daemon models these as
  infallible getters returning a value (`CreateConversationDetails` / `ChannelJoinDetails`),
  not `Option`. The Empty case therefore returns `Default::default()` rather than libpurple's
  `NULL` + warning. Ported with that documented divergence (assert the default value).
- daemon has a `delete` verb libpurple's conversation suite does not test → covered only by
  `assert_ops_match_behavior` (no dedicated libpurple row).

## Test locations

- `crates/contracts/daemon-api/tests/protocol_conformance.rs` — the ported cases + the Empty/Fake
  invariant runs (dev-dep on `daemon-api-testkit`; cargo permits the dev-dep cycle).
- `crates/adapters/<adapter>/tests/ops_invariant.rs` — `assert_ops_match_behavior` against each
  real adapter (forward half; reverse half passes via sentinel keying).

---

## `test_protocol_conversation.c` (46 cases)

Rust tests in `protocol_conformance.rs`, module `conversation`. Verb sentinels:
`conv_create / conv_join / conv_leave / conv_delete / conv_send / conv_set_topic /
conv_set_title / conv_set_description`.

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 1 | empty/implements-create-conversation | `conv_empty_implements_create` | green |
| 2 | empty/get-create-conversation-details | `conv_empty_create_details_default` | green (divergence: default not NULL) |
| 3 | empty/create-conversation-async | `conv_empty_create_unsupported` | green |
| 4 | empty/implements-leave-conversation | `conv_empty_implements_leave` | green |
| 5 | empty/leave-conversation-async | `conv_empty_leave_unsupported` | green |
| 6 | empty/implements-send-message | `conv_empty_implements_send` | green |
| 7 | empty/send-message-async | `conv_empty_send_unsupported` | green |
| 8 | empty/implements-set-topic | `conv_empty_implements_set_topic` | green |
| 9 | empty/set-topic-async | `conv_empty_set_topic_unsupported` | green |
| 10 | empty/implements-set-avatar | — | skipped: no `set_avatar` verb in daemon model |
| 11 | empty/set-avatar-async | — | skipped: no `set_avatar` verb in daemon model |
| 12 | empty/get-channel-join-details | `conv_empty_channel_join_details_default` | green (divergence: default not NULL) |
| 13 | empty/join_channel_async | `conv_empty_join_channel_unsupported` | green |
| 14 | empty/implements-set-title | `conv_empty_implements_set_title` | green |
| 15 | empty/set-title-async | `conv_empty_set_title_unsupported` | green |
| 16 | empty/implements-set-description | `conv_empty_implements_set_description` | green |
| 17 | empty/set-description-async | `conv_empty_set_description_unsupported` | green |
| 18 | empty/implements-send-typing | — | skipped: no `send_typing` verb in daemon model |
| 19 | empty/send-typing | — | skipped: no `send_typing` verb in daemon model |
| 20 | empty/implements-refresh | — | skipped: no `refresh` verb in daemon model |
| 21 | empty/refresh | — | skipped: no `refresh` verb in daemon model |
| 22 | normal/implements-create-conversation | `conv_fake_implements_create` | green |
| 23 | normal/get-create-conversation-details-normal | `conv_fake_create_details_value` | green (mirrors `_new(10)`) |
| 24 | normal/create-conversation-normal | `conv_fake_create_ok` | green |
| 25 | normal/create-conversation-error | `conv_fake_create_error` | green |
| 26 | normal/implements-leave-conversation | `conv_fake_implements_leave` | green |
| 27 | normal/leave-conversation-normal | `conv_fake_leave_ok` | green |
| 28 | normal/leave-conversation-error | `conv_fake_leave_error` | green |
| 29 | normal/implements-send-message | `conv_fake_implements_send` | green |
| 30 | normal/send-message-normal | `conv_fake_send_ok` | green |
| 31 | normal/send-message-error | `conv_fake_send_error` | green |
| 32 | normal/get-channel-join-details | `conv_fake_channel_join_details_value` | green (mirrors `_new(16,T,16,T,0)`) |
| 33 | normal/join-channel-normal | `conv_fake_join_channel_ok` | green |
| 34 | normal/join-channel-error | `conv_fake_join_channel_error` | green |
| 35 | normal/set-topic-normal | `conv_fake_set_topic_ok` | green |
| 36 | normal/set-topic-error | `conv_fake_set_topic_error` | green |
| 37 | normal/set-avatar-normal | — | skipped: no `set_avatar` verb in daemon model |
| 38 | normal/set-avatar-error | — | skipped: no `set_avatar` verb in daemon model |
| 39 | normal/implements-send-typing | — | skipped: no `send_typing` verb in daemon model |
| 40 | normal/send-typing | — | skipped: no `send_typing` verb in daemon model |
| 41 | normal/implements-refresh | — | skipped: no `refresh` verb in daemon model |
| 42 | normal/refresh | — | skipped: no `refresh` verb in daemon model |
| 43 | normal/set-title-normal | `conv_fake_set_title_ok` | green |
| 44 | normal/set-title-error | `conv_fake_set_title_error` | green |
| 45 | normal/set-description-normal | `conv_fake_set_description_ok` | green |
| 46 | normal/set-description-error | `conv_fake_set_description_error` | green |

Conversation: 34 ported, 12 skipped (`set_avatar`/`send_typing`/`refresh` — not in daemon model).

## `test_protocol_contacts.c` (14 cases)

Module `contacts`. Sentinels: `contact_get_profile / contact_set_alias`; `action_menu` → `Option`.

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 1 | empty/implements-get-profile | `contacts_empty_implements_get_profile` | green |
| 2 | empty/get-profile | `contacts_empty_get_profile_unsupported` | green |
| 3 | empty/implements-get-action-menu | `contacts_empty_implements_action_menu` | green |
| 4 | empty/get-action-menu | `contacts_empty_action_menu_none` | green |
| 5 | empty/implements-set-alias | `contacts_empty_implements_set_alias` | green |
| 6 | empty/set-alias | `contacts_empty_set_alias_unsupported` | green |
| 7 | normal/implements-get-profile | `contacts_fake_implements_get_profile` | green |
| 8 | normal/get-profile-normal | `contacts_fake_get_profile_ok` | green (returns `"profile data"`) |
| 9 | normal/get-profile-error | `contacts_fake_get_profile_error` | green |
| 10 | normal/implements-get-action-menu | `contacts_fake_implements_action_menu` | green |
| 11 | normal/get-action-menu | `contacts_fake_action_menu_some` | green |
| 12 | normal/implements-set-alias | `contacts_fake_implements_set_alias` | green |
| 13 | normal/set-alias-normal | `contacts_fake_set_alias_ok` | green |
| 14 | normal/set-alias-error-normal | `contacts_fake_set_alias_error` | green |

Contacts: 14 ported, 0 skipped.

## `test_protocol_roster.c` (9 cases)

Module `roster`. Sentinels: `roster_add / roster_update / roster_remove`. (`RosterOps.list`
has no libpurple counterpart in this suite; covered by `assert_ops_match_behavior` only.)

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 1 | empty/add | `roster_empty_add_unsupported` | green |
| 2 | empty/update | `roster_empty_update_unsupported` | green |
| 3 | empty/remove | `roster_empty_remove_unsupported` | green |
| 4 | add | `roster_fake_add_ok` | green |
| 5 | add-error | `roster_fake_add_error` | green |
| 6 | update | `roster_fake_update_ok` | green |
| 7 | update-error | `roster_fake_update_error` | green |
| 8 | remove | `roster_fake_remove_ok` | green |
| 9 | remove-error | `roster_fake_remove_error` | green |

Roster: 9 ported, 0 skipped.

## `test_protocol_directory.c` (2 cases)

Module `directory`. Sentinel: `directory_search`. (libpurple has no "empty" directory case; the
Empty-side unsupported behavior is still covered by `assert_ops_match_behavior`.)

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 1 | normal/search-async-normal | `directory_fake_search_ok` | green |
| 2 | normal/search-async-error | `directory_fake_search_error` | green |

Directory: 2 ported, 0 skipped. Empty-side covered by the invariant run.

## `test_protocol_file_transfer.c` (6 cases — EMPTY side only per scope)

Module `file_transfer`. Sentinels: `file_transfer_send / file_transfer_receive`.

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 1 | empty/send | `ft_empty_implements_and_send_unsupported` | green |
| 2 | empty/receive | `ft_empty_implements_and_receive_unsupported` | green |
| 3 | normal/send-normal | — | skipped: W2-H (file-transfer impl is a later package) |
| 4 | normal/send-error | — | skipped: W2-H |
| 5 | normal/receive-normal | — | skipped: W2-H |
| 6 | normal/receive-error | — | skipped: W2-H |

File transfer: 2 ported (empty), 4 skipped (W2-H).

## `test_protocol.c` (9 cases — validate/account scope)

Module `protocol`. The daemon `MessagingProtocol` only owns `validate_account`; the remaining
`PurpleProtocol` account-manager lifecycle verbs are node-owned (ControlApi / account manager),
out of W1-A scope. `properties`/`get-default-account-settings` map to the adapter descriptor
(`TransportAdapter::family` / `info().display_name` / `info().account_schema`).

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 1 | properties | `protocol_fake_descriptor_identity` | green (maps to `family()` + `info().display_name`) |
| 2 | can-connect/error | — | skipped: no `can_connect` verb in daemon MessagingProtocol (node-owned) |
| 3 | can-connect/false | — | skipped: node-owned connection lifecycle |
| 4 | can-connect/true | — | skipped: node-owned connection lifecycle |
| 5 | generate-account-name/override | — | skipped: no account-name generation verb (node-owned) |
| 6 | generate-account-name/default | — | skipped: no account-name generation verb (node-owned) |
| 7 | get-default-account-settings/override | `protocol_fake_default_account_settings` | green (maps to `info().account_schema`) |
| 8 | validate-account | `protocol_fake_validate_account_ok` | green |
| 9 | delete-account | — | skipped: no `delete_account` verb in daemon MessagingProtocol (node-owned) |

Also ported (daemon-specific, no direct libpurple row — exercises `MessagingProtocol::validate_account`'s
`Result` surface via the Fake failure switch and the Empty default):
- `protocol_empty_validate_account_ok` — the default `validate_account` returns `Ok`.
- `protocol_fake_validate_account_error` — Fake's `validate` switch → `Err`.

Protocol: 3 ported, 6 skipped (node-owned account lifecycle).

## Invariant runs (harness self-tests, in `protocol_conformance.rs`, module `invariant`)

| Rust test | asserts |
|---|---|
| `invariant_empty_matches_behavior` | `assert_ops_match_behavior(EmptyProtocol)` — every verb off ⟹ sentinel `Unsupported` |
| `invariant_fake_matches_behavior` | `assert_ops_match_behavior(FakeProtocol)` — every verb on ⟹ never the sentinel |
| `invariant_fake_all_failing_matches_behavior` | Fake with all switches on still passes (errors are `Other`, not the sentinel) |

## Adapter `ops_invariant.rs` (deliverable 3)

Each runs `assert_ops_match_behavior` against the real adapter (forward half; reverse via sentinel).

| adapter | constructor (hermetic) | status |
|---|---|---|
| daemon-matrix | `MatrixAdapter::new(mock_provisioning, MatrixConfig::default(), None)` | green |
| daemon-rooms | `RoomsAdapter::new(SqliteStore::open_in_memory, TraceSigner::generate, RoomsConfig::default())` | green |
| daemon-discord | `DiscordAdapter::new(mock_provisioning, DiscordConfig::default())` | green |
| daemon-telegram | `TelegramAdapter::new(mock_provisioning, TelegramConfig::default())` | green |
| daemon-slack | `SlackAdapter::new(mock_provisioning, SlackConfig::default())` | green |
| daemon-line | `LineAdapter::new(mock_provisioning, LineConfig::default())` | green |
| daemon-wechat | `WeChatAdapter::new(mock_provisioning, WeChatConfig::default())` | green |
| daemon-whatsapp | `WhatsappAdapter::new(mock_provisioning, WhatsappConfig::default())` | green |

Each ops_invariant.rs defines a tiny local `MockProvisioning` (the 3-method
`daemon_host::AccountProvisioning`) since the adapters' own test-module mocks are not exported.

## Totals

- libpurple cases in scope: **86** (conv 46, contacts 14, roster 9, directory 2, FT 6, protocol 9).
- Ported (green): **64**. Skipped-with-reason: **22** (12 conv verbs absent from daemon model,
  4 FT normal → W2-H, 6 protocol account-lifecycle → node-owned).
- Plus harness self-tests (3 invariant runs, 2 extra validate_account rows) and 8 adapter
  ops-invariant runs.
