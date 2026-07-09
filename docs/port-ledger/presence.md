# Port ledger — W2-F `port-presence`

`SavedPresence` DTO + host-side `PresenceManager` with durable persistence, plus the minimal
list/save/delete/set-active wire surface, ported from libpurple.

- **Target modules:**
  - DTO + behavior: `crates/contracts/daemon-api/src/saved_presence.rs`
    (wired via `mod saved_presence;` + re-exports in `lib.rs`).
  - Host manager: `crates/substrate/daemon-host/src/presence.rs` (`PresenceManager`).
  - Persistence: `crates/substrate/daemon-store` — `StoredSavedPresence` + `SessionStore`
    `saved_presence_*` methods (SQLite migration `M13` + in-memory analogue).
  - Wire: `ApiRequest::{PresenceList,PresenceSave,PresenceDelete,PresenceSetActive}` +
    `ApiResponse::SavedPresences`, `daemon-api.cddl`, CBOR fixtures, `dispatch::serve_control`,
    `authz::required_capability`, conformance `ownership_matrix::classify`.
- **Reference C sources (read-only):** `libpurple/purplesavedpresence.c`,
  `libpurple/purplepresencemanager.c`; tests under `libpurple/tests/`.

## Semantics established from the C sources

- **`PurpleSavedPresence`** (`purplesavedpresence.c`): `id` (auto-UUID if unset via `constructed`),
  `name`, `primitive` (`PurplePresencePrimitive`, default `OFFLINE`), `message`, `emoji`,
  `last_used` (`GDateTime`), `use_count` (`guint64`). Modeled with the existing wire
  `PresencePrimitive`; `last_used` as `Option<u64>` (unix seconds, mirroring `Presence::idle_since`);
  the string fields as `Option<String>`.
- **`purple_saved_presence_equal`** — NULL/NULL → true; one NULL → false; then compares
  `last_used` (both-null equal, one-null unequal, both-set `g_date_time_equal`), `use_count`, `name`
  (`birb_str_equal`), `primitive`, `message`, `emoji`. **`id` is intentionally NOT compared** (two
  freshly-constructed presences with distinct random ids are `equal`). Ported as
  `SavedPresence::equal` + the NULL-aware free fn `saved_presence_equal(Option, Option)`.
  `Option<u64>`/`Option<String>` equality reproduces the birb NULL rules exactly. (Derived
  `PartialEq`/`Eq` — needed by the wire enums — DOES include `id`; the domain `equal()` is separate.)
- **`purple_saved_presence_matches(needle)`** — empty needle (`birb_str_is_empty`) → true; then
  caseless **subsequence** match (`birb_str_matches`, reused from `matching.rs::str_matches`) against
  `name`, then `message`; then **exact** equality (`birb_str_equal`) against `emoji`. Ported faithfully
  (emoji is exact-match, not subsequence — matches the C).
- **`purple_saved_presence_new(primitive)`** / `constructed` — a new presence gets a random id.
  Ported as `SavedPresence::new(primitive)` (id minted from time+counter; no new `uuid` dep) and
  `ensure_id()`; the "generates-id" test asserts a non-empty, non-"" id.
- **`PurplePresenceManager`** (`purplepresencemanager.c`): an ordered `GListModel` of saved
  presences that always guarantees a default **Offline** (`00000000-…`) and **Available**
  (`ffffffff-…`) presence; `add` (reject duplicate id → FALSE, else append + persist via backend +
  emit `added`), `find_with_id`, `remove(id)` (→ bool, persist delete + emit `removed`),
  `set_active_from_id`/`set_active_from_index`, `get_active`. The libpurple **backend** is an async
  GObject (save/delete/load) bound to GSettings for the active id. **Daemon mapping:** the backend
  seam is the `daemon-store` `SessionStore` — `add`→`saved_presence_set`, `remove`→
  `saved_presence_remove`, `new`/`load`→`saved_presence_list` + default-seeding; the active id is a
  single-row store setting. `use_count`+`last_used` are bumped on `set_active` (daemon-native
  activation bookkeeping the C leaves to callers).

## Case ledger

Status: PORT = ported & green · SKIP = out of model/scope (reason) · DERIVED = requested by scope but
no direct libpurple g_test.

### `test_saved_presence.c` (18 cases) → `daemon-api/src/saved_presence.rs` tests

| # | libpurple g_test | Rust test | status |
|---|---|---|---|
| 1 | `/saved-presence/properties` | `sp_properties_roundtrip` | PORT |
| 2 | `/saved-presence/generates-id` | `sp_generates_id` | PORT |
| 3 | `/saved-presence/equal/null_null` | `sp_equal_null_null` | PORT |
| 4 | `/saved-presence/equal/null_a` | `sp_equal_null_a` | PORT |
| 5 | `/saved-presence/equal/null_b` | `sp_equal_null_b` | PORT |
| 6 | `/saved-presence/equal/default` | `sp_equal_default` | PORT |
| 7 | `/saved-presence/equal/last-used` | `sp_equal_last_used` | PORT |
| 8 | `/saved-presence/equal/use-count` | `sp_equal_use_count` | PORT |
| 9 | `/saved-presence/equal/name` | `sp_equal_name` | PORT |
| 10 | `/saved-presence/equal/primitive` | `sp_equal_primitive` | PORT |
| 11 | `/saved-presence/equal/message` | `sp_equal_message` | PORT |
| 12 | `/saved-presence/equal/emoji` | `sp_equal_emoji` | PORT |
| 13 | `/saved-presence/matches/accepts_null` | `sp_matches_accepts_null` | PORT |
| 14 | `/saved-presence/matches/empty_string` | `sp_matches_empty_string` | PORT |
| 15 | `/saved-presence/matches/name` | `sp_matches_name` | PORT |
| 16 | `/saved-presence/matches/message` | `sp_matches_message` | PORT |
| 17 | `/saved-presence/matches/emoji` | `sp_matches_emoji` | PORT |
| 18 | `/saved-presence/matches/none` | `sp_matches_none` | PORT |

### `test_presence_manager.c` (2 cases) → `daemon-host/src/presence.rs` tests

| # | libpurple g_test | Rust test | status |
|---|---|---|---|
| 1 | `/presence-manager/new` (2 default presences) | `mgr_new_has_two_defaults` | PORT |
| 2 | `/presence-manager/add-remove` | `mgr_add_remove` | PORT |

### `test_presence_manager_backend_normal.c` (3 cases) → `daemon-host/src/presence.rs` tests

The libpurple backend is a GObject async interface (call-count assertions on save/delete/load). The
daemon backend seam is the `SessionStore`; the async-plumbing/call-count harness is GObject-specific
and out of model, so these are ported as **store-persistence** behavior tests (the meaningful port).

| # | libpurple g_test | Rust test | status |
|---|---|---|---|
| 1 | `/presence-manager-backend-normal/save-saved-presence` | `mgr_add_persists_to_store` | PORT (store seam) |
| 2 | `/presence-manager-backend-normal/delete-saved-presence` | `mgr_remove_deletes_from_store` | PORT (store seam) |
| 3 | `/presence-manager-backend-normal/load-saved-presences` | `mgr_new_loads_from_store` | PORT (store seam) |
| — | GObject async call-count assertions (`*_async`/`*_finish` counters) | — | SKIP (GObject async harness; not modeled — the daemon store call IS the seam) |

### `test_presence_manager_backend_empty.c` / `test_presence_manager_seagull_backend.c`

SKIP — the "empty" backend (no-op GObject) and the "seagull" GSettings-backed backend are
libpurple-specific `GObject`/`GSettings` plumbing variants. The daemon has one backend (the durable
`SessionStore`), covered by the backend-normal ports above.

### Derived (daemon-native, requested by scope; no direct g_test)

| item | Rust test | status |
|---|---|---|
| `matches(Option<&str>)` filter over the manager list | `sp_matches_*` (DTO) | DERIVED |
| lookup by name | `mgr_find_by_name` | DERIVED |
| set-active bumps use-count + last-used | `mgr_set_active_bumps_use_count_and_last_used` | DERIVED |
| set-active persists across a reload | `mgr_active_persists_across_reload` | DERIVED |
| store round-trip (list/set/remove) | `saved_presence_store_round_trips` (daemon-store) | DERIVED |

## Wire additions (all tagged `wire vNEXT`; `WireVersion::CURRENT` NOT bumped — integration owns it)

- **Types:** `SavedPresence` (new serde wire DTO; `Arbitrary` feature-gated).
- **Requests:** `PresenceList`, `PresenceSave { presence }`, `PresenceDelete { id }`,
  `PresenceSetActive { id }` — appended at the END of `ApiRequest`.
- **Responses:** `SavedPresences(Vec<SavedPresence>)` — appended at the END of `ApiResponse`.
- **CDDL arms:** `saved-presence`, `request-presence-{list,save,delete,set-active}`,
  `response-saved-presences`, appended at the end of the `api-request`/`api-response` unions.
- **Fixtures:** `request-presence-list.cbor`, `request-presence-save.cbor`,
  `request-presence-delete.cbor`, `request-presence-set-active.cbor`,
  `response-saved-presences.cbor`.
- **Access-control:** `PresenceList → ControlRead`; `PresenceSave/PresenceDelete/PresenceSetActive →
  ControlWrite` (saved presences are node-wide shared config, like the gateway/telemetry-consent
  toggles — operator-tier writes, viewer-readable). Classified `NotSessionTouching` in the ownership
  matrix (no per-owner session state).
- **NodeEvent:** SKIPPED — the existing presence surface has no saved-presence `NodeEvent` pointer to
  mirror (per scope, skip + note here).

## Persistence / schema

- SQLite `M13`: `saved_presences (rowseq INTEGER PK AUTOINCREMENT, id TEXT UNIQUE, payload BLOB)`
  (insertion-ordered; `payload` is opaque CBOR of the wire `SavedPresence`, store stays
  protocol-free like `cron_jobs.spec`) + `saved_presence_active (id INTEGER PK CHECK(id=0),
  active TEXT)` (single-row active id). Golden schema refreshed via `DAEMON_UPDATE_SCHEMA=1`; the
  `fresh DB is stamped to the latest migration` version assertion bumped 12 → 13.
