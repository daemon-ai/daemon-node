# Port ledger — W2-H `port-filetransfer`

The FileTransfer model + real `SupportsFileTransfer` implementations, ported from
libpurple's file-transfer subsystem:

- `libpurple/purplefiletransfer.c` (`PurpleFileTransfer` DTO + state)
- `libpurple/purpleprotocolfiletransfer.c` (`PurpleProtocolFileTransfer` interface)
- `libpurple/purplefiletransfermanager.c` (`PurpleFileTransferManager`)
- test fixtures: `tests/test_file_transfer.c`, `tests/test_protocol_file_transfer.c`,
  `tests/test_file_transfer_manager.c`

"Done" = every row below is green or explicitly skipped-with-reason.

Legend: **P** = ported (green), **S** = skipped (reason), **+** = extra Rust
case with no direct libpurple `g_test` (derived from the `.c` implementation or a
daemon-native decision).

---

## Wire model decision (recorded per package instructions)

The existing `FileTransfer { name, blob }` wire shape is kept **intact** and
extended **additively** (every new field is `#[serde(default)]`), so pre-existing
CBOR decodes unchanged. New fields port `PurpleFileTransfer`'s properties:

- `direction: FileTransferDirection` (`Send` | `Receive`) — daemon-native framing
  of libpurple's initiator-vs-remote asymmetry (`new_send` initiator = account,
  `new_receive` initiator = remote).
- `state: FileTransferState` (`Unknown|Negotiating|Started|Finished|Failed`) ←
  `PurpleFileTransferState` (`purplefiletransfer.h`). The illustrative
  `new→accepted→…/cancelled` list in the work order does **not** match the actual
  libpurple enum (there is no accepted/cancelled state — cancellation is a
  `GCancellable` + error); this port follows the real 5-value enum and models
  cancellation as `Failed` + an `error` message.
- `remote`, `initiator: Option<ContactInfo>` ← the `remote`/`initiator`
  `PurpleContactInfo` properties.
- `file_size: u64` ← `file-size` (kept independent of `blob.size`, as in C).
- `transferred: u64` — progress bytes (daemon-native; the UI progress the node
  owns rather than a client re-derives).
- `content_type: Option<String>` ← `content-type`.
- `message: Option<String>` ← `message`.
- `error: Option<String>` ← the `error` `GError` property (rendered as text).
- `source: Option<String>` — the remote content locator a `receive` fetches from
  (e.g. a Matrix `mxc://` URI); daemon-native, protocol-opaque.

`account`/`local-file`/`cancellable` GObject props are **not** ported: the
daemon has no `PurpleAccount`/`GFile`/`GCancellable` (the account is the
transport instance; the "local file" is a content-addressed `BlobRef`;
cancellation is a node concern, not a per-DTO handle).

Wire ops added (additive, appended at the END of `ApiRequest`): `FtSend` /
`FtReceive`, each `{ transport, transfer }`. Both answer `ApiResponse::Ok`
(reused, exactly like `ConvSend`) — no new response arm. The
`SupportsFileTransfer` trait verbs gain a `transport: TransportId` argument
(mirroring every other feature trait, e.g. `ConvSendArgs.transport`), so an
adapter can resolve its per-account client.

`WireVersion::CURRENT` is bumped once by integration (to 37); new wire items
are tagged "wire v37".

---

## 1. `purplefiletransfer.c` — `tests/test_file_transfer.c` (3 g_test)

| # | libpurple case | Rust test (`file_transfer::tests`) | status |
|---|---|---|---|
| 1 | `/file-transfer/new/send` | `ft_new_send_initiator_is_account_name_and_size` | P |
| 2 | `/file-transfer/new/receive` | `ft_new_receive_initiator_is_remote` | P |
| 3 | `/file-transfer/properties` | `ft_properties_roundtrip` | P |

Semantics:
- `new_send(account, remote, blob)`: `direction = Send`, `initiator = account`,
  `remote = remote`, `name`/`file_size` derived from the `BlobRef`
  (daemon analogue of libpurple deriving them from the `GFile`).
- `new_receive(remote, name, file_size)`: `direction = Receive`,
  `initiator = remote`, `remote = remote`.
- `/properties`: in C a GObject get/set round-trip of every property; ported as a
  construct-via-setters-then-read-getters round-trip over the (public-field) DTO.

## 2. state machine (`purplefiletransfer.c` set_state/error + `purplefiletransfer.h`)

| # | case | Rust test | status |
|---|---|---|---|
| 4 | `+` state enum default is `Unknown` | `ft_state_default_unknown` | + |
| 5 | `+` `set_state` transitions | `ft_set_state` | + |
| 6 | `+` `fail(msg)` → `Failed` + error text | `ft_fail_sets_failed_and_error` | + |
| 7 | `+` guarded `advance` lifecycle | `ft_advance_lifecycle_and_rejects_backwards` | + |
| 8 | `+` `record_progress` clamps at `file_size` | `ft_record_progress_clamps` | + |
| 9 | `+` `is_send`/`is_receive`/`is_terminal`/`is_complete` | `ft_predicates` | + |

The guarded lifecycle (`Unknown→Negotiating→Started→Finished`; any non-terminal
→`Failed`) is daemon-native: libpurple's `set_state` is an unguarded setter (the
protocol plugin owns ordering). The unguarded `set_state` is ported too (#5) for
faithfulness; `advance` is the node-authoritative validated transition.

## 3. `purplefiletransfermanager.c` — `tests/test_file_transfer_manager.c` (2 g_test)

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 10 | `/file-transfer-manager/add-remove` | `manager_add_remove_emits_signals` | P |
| 11 | `/file-transfer-manager/propagates-notify` | `manager_propagates_notify` | P |

`PurpleFileTransferManager` is a `GListModel` with `added`/`removed` and
`transfer-changed[::detail]` signals. Ported as an in-memory `FileTransferManager`
(`Vec<FileTransfer>` + `added`/`removed` counters + a generic + per-field
change counter). `add`/`remove` bump the add/remove counters and item count;
`set_transfer_state` bumps generic + `state` counters; `set_transfer_message`
bumps only the generic counter (mirrors the `::state`-detailed signal semantics).

## 4. `purpleprotocolfiletransfer.c` — `tests/test_protocol_file_transfer.c` (6 g_test)

Ported into `daemon-api/tests/protocol_conformance.rs` `mod file_transfer`
against `EmptyProtocol`/`FakeProtocol`. The EMPTY rows landed in Wave 1; this
package adds the normal/error rows (the "W2-H" skips).

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 12 | `/protocol-file-transfer/empty/send` | `ft_empty_implements_and_send_unsupported` | P (Wave 1) |
| 13 | `/protocol-file-transfer/empty/receive` | `ft_empty_implements_and_receive_unsupported` | P (Wave 1) |
| 14 | `/protocol-file-transfer/normal/send-normal` | `ft_fake_implements_and_send_ok` | P |
| 15 | `/protocol-file-transfer/normal/send-error` | `ft_fake_send_error` | P |
| 16 | `/protocol-file-transfer/normal/receive-normal` | `ft_fake_implements_and_receive_ok` | P |
| 17 | `/protocol-file-transfer/normal/receive-error` | `ft_fake_receive_error` | P |

libpurple's `implements_send/receive` (both iface fn-ptrs non-NULL) → daemon's
`supported()` ops flags. `send_async`+`send_finish` success/error → the async
`send`/`receive` `Result`. `should_error` → `FakeProtocol::failing()` (a
non-`Unsupported` `Other`, so it never collides with the capability sentinel).
The `send_async`/`send_finish` call-count assertions are GObject-dispatch
mechanics with no daemon analogue — **S** (the single `await` returns the result
directly).

## 5. `SupportsFileTransfer` implementations (per package scope)

| # | implementation | Rust test | status |
|---|---|---|---|
| 18 | testkit `FakeProtocol` in-memory send/receive + fail switch | conformance #14–17 + `fake_records_transfers` | P |
| 19 | testkit `EmptyProtocol` all-unsupported | conformance #12–13 | P (Wave 1) |
| 20 | `daemon-rooms` loopback via node blob store (send) | `rooms::file_transfer_send_roundtrips_via_blob_store` | P |
| 21 | `daemon-rooms` loopback via node blob store (receive) | `rooms::file_transfer_receive_roundtrips_via_blob_store` | P |
| 22 | `daemon-rooms` ops honest (no blob store ⟹ feature absent) | `ops_invariant` (existing) + `rooms::file_transfer_ops_reflect_blob_store` | P |
| 23 | `daemon-matrix` media upload (send) | `matrix::file_transfer_send_uploads_media` | P |
| 24 | `daemon-matrix` media download (receive) | `matrix::file_transfer_receive_downloads_media` | P |
| 25 | `daemon-matrix`/other adapters ops honesty | each `tests/ops_invariant.rs` (existing) | P |

Rooms: the loopback transfer round-trips bytes through the node
`daemon_host::BlobStore` (send verifies the content-addressed blob resolves;
receive fetches it). No blob store ⟹ `file_transfer()` returns `None` (the
feature is absent, ops stay `false`) — keeps `assert_ops_match_behavior` honest.

Matrix: `send` reads the blob's bytes from the node blob store and uploads them to
the Matrix content repository (`client.media().upload`); `receive` downloads the
`source` `mxc://` content (`client.media().get_media_content`) and stores it back
into the node blob store. Tested against the existing `MatrixMockServer` harness
(`mock_upload`, `mock_authed_media_download`).

Adapters that do **not** implement file transfer (discord, slack, telegram,
line, wechat, whatsapp) keep `file_transfer()` = `None` / advertise `false`; their
`tests/ops_invariant.rs` stays green unchanged.

## 6. Wire dispatch / host / authz / ownership

| # | item | Rust test | status |
|---|---|---|---|
| 26 | `ControlApi::ft_send`/`ft_receive` defaults (unsupported) | `lib.rs` ControlApi default tests (existing bare-impl test) | + |
| 27 | dispatch `FtSend`/`FtReceive` → `ft_send`/`ft_receive` | `wire`/`dispatch` roundtrip fixture | + |
| 28 | authz: `FtSend`/`FtReceive` → `MessagingWrite` | `authz` classification table | + |
| 29 | ownership: `FtSend`/`FtReceive` → `NotSessionTouching` | `ownership_matrix` exhaustive match | + |
| 30 | CDDL `file-transfer` + `request-ft-send`/`-receive` | `conformance` + `--features arbitrary` proptest + `verify-codec` | P |
| 31 | CBOR fixtures `request-ft-send`/`request-ft-receive` | `xtask api-fixtures` + `verify-codec` | P |

---

## Totals

- libpurple `g_test` cases enumerated across the 3 fixture files: **11**
  (file_transfer 3, protocol_file_transfer 6, file_transfer_manager 2).
- Ported (P): **11** (send/receive/normal/error protocol rows, DTO new_send/
  new_receive/properties, manager add-remove/propagates-notify).
- Skipped (S): the `send_async`/`send_finish` **call-count** assertions inside the
  protocol normal/error fixtures (GObject dispatch mechanics; the single `await`
  returns the result directly). Recorded above under §4.
- Extra derived/daemon-native (+): state-machine + predicates (#4–9),
  wire/host/authz/ownership plumbing (#26–29).
