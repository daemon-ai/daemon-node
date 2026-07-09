# Port ledger — consolidated parity report (libpurple → daemon full-parity port)

This is the aggregate coverage report for the ten-package libpurple full-parity port, merged into
`port/integration` and sealed with the one-time `WireVersion` bump to **v37** (see
`crates/contracts/daemon-common/src/lib.rs`, `WireVersion::CURRENT = 37`).

Each row's counts are transcribed from that package's own ledger summary (linked below) — they are
**not** re-derived here. Terminology is normalised to three buckets:

- **Ported** — a libpurple `g_test` case reproduced green in Rust (`P` / `PORT` / `green` /
  `green-parity`).
- **Derived** — a daemon-native row with no direct libpurple `g_test` (`D` / `+` / `DERIVED` /
  daemon-native / EXTRA), derived from the C implementation or the new wire surface.
- **Skipped** — a case skipped-with-reason (`S` / `SKIP`), almost always a GObject/GLib toolkit
  artifact (signal-emission counters, `GListModel` item-type/`items-changed`, refcount finalisation,
  property-bag introspection) or a surface owned by another package / not modeled in the daemon.

## Per-package summary

| Wave | Package | Ledger | libpurple test files (cases) | Ported | Derived | Skipped |
|---|---|---|---|---|---|---|
| W1-A | harness (MessagingProtocol conformance) | [harness.md](harness.md) | `test_protocol_conversation.c` (46), `test_protocol_contacts.c` (14), `test_protocol_roster.c` (9), `test_protocol_directory.c` (2), `test_protocol_file_transfer.c` (6, empty-side), `test_protocol.c` (9) | 64 | 5 self-tests + 8 adapter ops-invariant runs | 22 |
| W1-B | dto-logic (conversation/account DTO logic) | [dto-logic.md](dto-logic.md) | `test_create_conversation_details.c` (7), `test_channel_join_details.c` (3), `test_conversation.c` (23, title/type/tag subset), `test_account_setting.c` (8), `test_account_settings.c` (12), `test_presence.c` (3), `test_connection.c` (5) | 40 | 7 | 26 |
| W1-C | matching (display/matching/ordering) | [matching.md](matching.md) | `test_contact_info.c`, `test_conversation_member.c`, `test_conversation_members.c`, `test_conversation_manager.c` (find-dm) — 41 in-scope | 38 | 1 | 3 |
| W1-D | registries (adapter/cron/command/credential managers) | [registries.md](registries.md) | `test_protocol_manager.c` (2), `test_scheduled_task.c` (7), `test_command_manager.c` (7), `test_credential_manager.c` (11) + `test_credential_provider_{normal,empty}.c` (5+5 folded) — 32 in-scope | 24 | 29 new tests, mostly daemon-native rows | 8 |
| W2-E | message (`ChatMessage`/`Tags`/markup) | [message.md](message.md) | `test_message.c` (5), `test_tags.c` (27), `test_markup.c` (1 g_test / 10-row table) | 33 | EXTRA (message `is_empty`/`compare`/wire round-trip, tags conv-type, markup entity/numeric/malformed) | 0 |
| W2-F | presence (`SavedPresence` + `PresenceManager`) | [presence.md](presence.md) | `test_saved_presence.c` (18), `test_presence_manager.c` (2), `test_presence_manager_backend_normal.c` (3); `*_backend_empty.c` / `*_seagull_backend.c` whole-file skips | 23 | 5 | 3 (backend GObject-async / backend variants) |
| W2-G | notify (notifications + `NotificationManager`) | [notify.md](notify.md) | `test_authorization_request.c` (8), `test_notification.c` (2), `test_notification_add_contact.c` (3), `test_notification_authorization_request.c` (3), `test_notification_link.c` (3), `test_notification_manager.c` (8); `purpleaddcontactrequest.c` / `purplenotificationconnectionerror.c` (no test file) | 27 | 7 (5 derived + 2 extra) | signal-count mechanics folded (0 discrete) |
| W2-H | filetransfer (`FileTransfer` + `SupportsFileTransfer`) | [filetransfer.md](filetransfer.md) | `test_file_transfer.c` (3), `test_protocol_file_transfer.c` (6), `test_file_transfer_manager.c` (2) | 11 | 10 (state machine ×6 + wire/host/authz/ownership ×4) + adapter impls | send/finish call-count folded (0 discrete) |
| W2-I | request (typed request-field model) | [request.md](request.md) | `test_request_field.c` (5), `test_request_field_choice.c` (5), `test_request_field_account.c` (4), `test_request_field_image.c` (4), `test_request_field_list.c` (5), `test_request_group.c` (1), `test_request_page.c` (4) | 28 | derived-helper tests (typed getters, lookup, choice reset) | GObject sub-assertions in 15 cases folded (0 discrete) |
| W3-J | person (`Person`/MetaContact) | [person.md](person.md) | `test_person.c` (16), `test_contact_manager.c` (2, person cases), `test_contact_info.c` (3, re-activated) — 21 in-scope | 17 | 8 | 4 (+ GObject signal/back-ref halves) |

### Headline totals

- **libpurple `g_test` cases in scope across all ten packages: ~363.**
- **Ported (green): ~305** (harness 64, dto-logic 40, matching 38, registries 24, message 33,
  presence 23, notify 27, filetransfer 11, request 28, person 17).
- **Skipped-with-reason: ~66 discrete cases** (harness 22, dto-logic 26, matching 3, registries 8,
  presence 3, person 4) — plus the many GObject signal-emission / `GListModel` / property-bag
  *sub-assertions* that several ledgers fold into a ported case rather than count as a discrete
  skip (message, notify, filetransfer, request).
- **Derived / daemon-native rows: ~40+** (dto-logic 7, matching 1, presence 5, notify 7,
  filetransfer 10, person 8) plus registries' 29 new-test set (mostly daemon-native), message/request
  extras, and the harness self-tests + 8 adapter ops-invariant runs.

Counting notes (transcribed, not reconciled): a few ledgers' internal `ported + skipped` exceed their
stated in-scope count because they split multi-assertion cases or fold sub-aspects — e.g. dto-logic
reports 40 P + 26 S against 61 in-scope. **Overlap:** the 3 person-dependent `test_contact_info.c`
cases (`get_name_for_display/person_with_alias`, `compare/person__no_person`,
`compare/no_person__person`) were re-activated by W3-J `port-person` and are counted in **both**
`matching.md` and `person.md`; the distinct-case count is ~3 lower than the raw sum.

## Recorded divergences

Intentional, documented differences between the daemon port and libpurple (collected from the ten
ledgers). None is a regression; each is a faithful mapping of a GObject/GLib idiom onto the
daemon's wire/DTO model or an out-of-scope-for-the-tested-behavior note.

- **harness (W1-A):** the biconditional's reverse half is **sentinel-keyed** — "advertised verb does
  not return the capability sentinel", *not* "advertised ⟹ `Ok`" — so an unconnected real adapter
  that returns `Unsupported("<family> … not connected")` still passes. Daemon `MessagingProtocol`
  omits `set_avatar`/`send_typing`/`refresh`; detail getters are infallible (return
  `Default::default()`, not libpurple's `NULL` + warning).
- **dto-logic (W1-B):** `set_delivered`/`set_edited` take a node-authoritative `now` (a pure wire DTO
  has no internal clock); `OutOfOffice` is **not** "online" (falls to the C `switch` default arm);
  the `federated` channel flag and the Pango `attributes` run-list are not modeled.
- **matching (W1-C):** `g_utf8_collate` is approximated by codepoint order after Unicode casefold
  (identical for all tested ASCII inputs; full ICU collation is out of scope); the
  `name_for_display` chain is `display_name → id` (no `alias`/`person` on `ContactInfo`) until the
  W3-J person layer inserts `person-alias`.
- **registries (W1-D):** a duplicate adapter `family` is **retained** in an ordered `Vec` (first
  wins) rather than rejected as libpurple's manager does; a **past one-shot** cron yields no fire
  (auto-deleted) instead of erroring, while a past recurring schedule fast-forwards; the command
  registry is **first-wins build-once** (no runtime remove / remove-all / find-all / per-conversation
  tag filter — those cases are the documented skips); credentials have **no active-provider** concept
  (mutating a never-acquired profile is a clean no-op; an unknown profile falls back — not an error)
  and **no lock/unlock** (v1 store is plaintext-at-rest, 0600, always "unlocked").
- **message (W2-E):** the `notify::delivered`/`notify::edited` signal-emission counters are folded
  into state assertions; `PurpleMessage:attributes` (Pango run-list) is not wire state.
- **presence (W2-F):** the GObject async backend (save/delete/load call-count harness) maps onto the
  durable `SessionStore` seam; the "empty" and "seagull" GSettings backends are libpurple-specific
  plumbing variants covered by the one durable-store backend; `use_count`/`last_used` bookkeeping on
  `set_active` is daemon-native.
- **notify (W2-G):** a double-add returns an `AddOutcome::DuplicateRejected` enum rather than
  libpurple's `g_warning`-aborts-subprocess precondition; `set_read` asserts `unread_count` +
  a `ReadChange` transition instead of GObject `notify::unread-count` emission counts.
- **filetransfer (W2-H):** the port follows the **real 5-value** `PurpleFileTransferState`
  (`Unknown|Negotiating|Started|Finished|Failed`) — the work order's illustrative
  `accepted/cancelled` states do not exist in libpurple; cancellation is modeled as `Failed` + an
  `error` message. `account`/`local-file`/`cancellable` GObject props are not ported (the account is
  the transport instance, the local file is a content-addressed `BlobRef`, cancellation is a node
  concern). `send_async`/`send_finish` call-count assertions are GObject dispatch mechanics (the
  single `await` returns the result directly).
- **request (W2-I):** no GObject plumbing (type-identity, refcount finalisation, property
  round-trip, `items-changed`/`notify::…` emission counts are skipped-with-reason); group/page
  validity is a pure recomputed predicate with a cached last-value to detect flips, vs libpurple's
  `invalid_fields` set. The full `RequestField`/`RequestGroup`/`RequestPage` surface is kept
  **node-internal** (no serde/CDDL/ops/`WireVersion` bump) — tagged as a candidate for a future wire
  version beyond v37, not part of this bump.
- **person (W3-J):** `preferred_endpoint()` ports the **presence-only** comparator
  (`purpleperson.c`'s `purple_person_contact_compare` == `purple_presence_compare`); the richer
  "(open conversation) > best presence > (account/transport) priority" chain the brief mentions does
  not exist on the C `PurplePerson` and is **deferred** as out-of-C-scope (TDD: no untested behavior
  invented). `color`/`tags` are not modeled; contact-level `avatar` is absent; GObject property-notify
  counters and the `PurpleContactInfo ↔ PurplePerson` back-reference are skipped, with the state half
  ported.

## Known pre-existing flakes (non-blocking; recommend human triage)

These are **not** introduced by the port and were treated as non-blocking when they were the sole
failure of a workspace run.

- **`node::detached_delegation`** — `detached_fanout_materializes_distinct_children` and
  `detached_notice_reaches_a_parked_durable_parent` are nondeterministic on `master` (fail ~50% of
  runs even in isolation). The test file is **untouched by every port branch**, verified by
  `git log master..HEAD -- tests/daemon-conformance/src/node/detached_delegation.rs` returning
  **empty** (master `c9c3236`, integration HEAD `2dcb7d9`).
- **`node::process_notify::injected_input_reaches_a_parked_durable_session_via_the_store_seam`** —
  same nondeterministic profile; `tests/daemon-conformance/src/node/process_notify.rs` is likewise
  untouched since `master` (`git log master..HEAD -- …` empty).
- **`daemon-cli` `sigint_host_launch`** — a signal/launch-timing flake observed **once** during the
  gate; a re-run was green.

**Recommendation:** these belong to a pre-existing nondeterminism in the detached-delegation /
parked-durable-session store seam (and a CLI SIGINT launch-timing race), independent of the libpurple
parity surface. A human should triage/deflake them separately; per the program's flake policy, an
otherwise-green run with only these failing is treated as a pass after a single isolated re-run.
