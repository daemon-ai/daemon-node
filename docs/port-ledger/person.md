# Port ledger — W3-J `port-person`

The `PurplePerson`/MetaContact model — the one concept the transport-adapter spec explicitly
deferred, now un-deferred. Adds the `Person` DTO + `preferred_endpoint`/`matches`, the person-aware
display layering seam left in `matching.rs`, a host `PersonManager`, and a minimal wire read surface
(`PersonList` → `Persons`, `PersonsChanged` pointer).

- **New DTO module:** `crates/contracts/daemon-api/src/person.rs` (`lib.rs` gains `mod person;` + a
  re-export).
- **Display layering:** additions to `crates/contracts/daemon-api/src/matching.rs` (person-aware
  helpers over `ContactInfo`, layered on the existing `display_name → id` chain).
- **Host manager:** `crates/substrate/daemon-host/src/person.rs` (`PersonManager`), following the
  conventions of `presence.rs` / `notifications.rs`.
- **Reference C (read-only):** `libpurple/purpleperson.c` + `tests/test_person.c` (~16 g_test),
  the person cases of `tests/test_contact_manager.c` (2 g_test), and the 3 person-dependent cases of
  `tests/test_contact_info.c` (re-activated from `matching.md`). Impl provenance:
  `purplecontactinfo.c` (`get_name_for_display` / `compare`), `purplepresence.c`
  (`purple_presence_compare`, already ported as `Presence::compare` in `src/details.rs`).

This package **touches the wire**: `Person`/`PersonEndpoint` cross the wire (reachable from
`ApiResponse::Persons`), so they are serde types mirrored in `daemon-api.cddl`, derive feature-gated
`Arbitrary`, and gain CBOR fixtures; a `PersonList` op + a payload-free `NodeEvent::PersonsChanged`
pointer are appended (wire v37 — the integration branch bumps `WireVersion` to 37, not this one).

"Done" = every row below is green or explicitly skipped-with-reason.

Legend: **P** = ported (green), **S** = skipped (reason given), **D** = derived (no libpurple
`g_test` — derived from the impl `.c` / daemon-native wire surface).

---

## Model mapping (how libpurple's `PurplePerson` maps to the daemon)

- `PurplePerson` holds a `GPtrArray *contacts` of `PurpleContactInfo`. The daemon `Person` holds
  `endpoints: Vec<PersonEndpoint>`, each binding a `TransportId` (the account/transport the contact
  lives on — libpurple's per-`PurpleContact` `PurpleAccount`) to a `ContactInfo` (the contact
  id/handle + its presence). This is the daemon-idiomatic "the same human across transports"
  association the transport-adapter spec had deferred.
- `PurplePerson` fields: `id` (auto UUID), `alias`, `avatar` (`PurpleImage`), `color`, `tags`. The
  daemon `Person` models `id` (auto-minted when empty, like `SavedPresence`), `alias`, and `avatar`
  (`Option<Image>`, the existing `Image`/`BlobRef` carrier). **`color` and `tags` are NOT modeled**
  (no daemon field; out of this package's scope) → the `color-for-display`/`tags` C cases are **S**.
- **Priority-contact algorithm.** `purple_person_get_priority_contact_info` returns index 0 after
  sorting `contacts` with `purple_person_contact_compare`, which is *exactly* `purple_presence_compare`
  (`purpleperson.c:88-99`). The daemon `preferred_endpoint()` computes the same on demand: the
  endpoint whose `contact.presence` is minimum under `Presence::compare` (from `src/details.rs`),
  ties resolved to the first (insertion order), mirroring the C stable sort + index-0.
  **Divergence / scope note:** the package brief names a richer chain "(open conversation) > best
  presence > (account/transport) priority". purple-3's `purpleperson.c` comparator is presence-only;
  neither an open-conversation flag nor an account-priority field exists on the C `PurplePerson`
  contacts nor is exercised by any `test_person.c` case. To stay faithful to the tests (TDD: never
  invent untested behavior) `preferred_endpoint` ports the presence-only comparator; the extra layers
  are recorded here as out-of-C-scope for a later package once `Person` gains conversation/priority
  context. The single-contact and multiple-with-change matrices below cover the tested behavior.
- **GObject artifacts.** libpurple asserts property-notify signal counters (`avatar`,
  `avatar-for-display`, `name-for-display`, `priority-contact-info`, `n-items`, list-model
  items-changed) and the `PurpleContactInfo ↔ PurplePerson` back-reference (`set_person`). The daemon
  DTO has no signals and no back-pointer; those counter/back-ref halves are **S** (GObject artifacts),
  with the state half ported.

---

## 1. `purpleperson.c` — `test_person.c` (16 g_test)

Module `daemon_api::person` (`Person`, `PersonEndpoint`).

| # | libpurple case | Rust test (`person::tests`) | status |
|---|---|---|---|
| 1 | `/person/new` | `person_new` | P |
| 2 | `/person/properties` | `person_properties` | P (id/alias/avatar/avatar-for-display/name-for-display subset) / S (`color`, `color-for-display`, `tags` — not modeled) |
| 3 | `/person/avatar-for-display/person` | `person_avatar_for_display_person` | P (person avatar overrides) / S (notify counters) |
| 4 | `/person/avatar-for-display/contact` | — | S: daemon `ContactInfo` has no `avatar` field (contact-level avatar not modeled) |
| 5 | `/person/color-for-display/person` | — | S: no `color` on daemon `Person` |
| 6 | `/person/color-for-display/contact` | — | S: no `color` on daemon `Person`/`ContactInfo` |
| 7 | `/person/name-for-display/person` | `person_name_for_display_person` | P |
| 8 | `/person/name-for-display/contact` | `person_name_for_display_contact` | P |
| 9 | `/person/contacts/single` | `person_contacts_single` | P (membership) / S (items-changed counter + `set_person` back-ref) |
| 10 | `/person/contacts/multiple` | `person_contacts_multiple` | P (5-add/5-remove) / S (counters) |
| 11 | `/person/priority/single` | `person_priority_single` | P |
| 12 | `/person/priority/multiple-with-change` | `person_priority_multiple_with_change` | P |
| 13 | `/person/matches/accepts_null` | `person_matches_accepts_null` | P |
| 14 | `/person/matches/empty_string` | `person_matches_empty_string` | P |
| 15 | `/person/matches/alias` | `person_matches_alias` | P |
| 16 | `/person/matches/contact_info` | `person_matches_contact_info` | P |

## 2. `test_contact_manager.c` — person cases (2 g_test)

Module `daemon_host::person` (`PersonManager`).

| # | libpurple case | Rust test (`person::tests`) | status |
|---|---|---|---|
| 17 | `/contact-manager/person/add-remove` | `manager_person_add_remove` | P (add/remove; `person-added`/`person-removed` signal counters → S) |
| 18 | `/contact-manager/person/add-via-contact-remove-person-with-contacts` | `manager_person_remove_with_contacts` | P (person + its endpoints removed on `remove_contacts=true`) |

## 3. `test_contact_info.c` — re-activated person-dependent cases (3 g_test)

Module `daemon_api::matching` (previously skipped in `matching.md` as "person precedence is Wave-3").
Now ported via the person-aware layering helpers; the corresponding rows in `matching.md` are
annotated as re-activated.

| # | libpurple case | Rust test (`matching::tests`) | status |
|---|---|---|---|
| 19 | `/contact-info/get_name_for_display/person_with_alias` | `contact_info_name_for_display_person_alias` | P |
| 20 | `/contact-info/compare/person__no_person` | `contact_info_compare_person_no_person` | P |
| 21 | `/contact-info/compare/no_person__person` | `contact_info_compare_no_person_person` | P |

`/contact-info/get_name_for_display/contact_with_alias` stays **S** (no `alias` on daemon
`ContactInfo`; the person layer inserts `person-alias` between the absent contact-alias and
`display_name`, giving the chain `person-alias → display_name → id`).

## 4. Derived / daemon-native (no direct g_test)

| # | derived case | Rust test | status |
|---|---|---|---|
| D1 | `Person::new`/`ensure_id` auto-mint id (mirrors `SavedPresence`; `purple_person_set_id` UUID) | `person_generates_id` | D/P |
| D2 | `preferred_endpoint` over an empty person → `None` (`purple_person_get_priority_contact_info` NULL) | `person_priority_empty_none` | D/P |
| D3 | `add_endpoint` double-add / `remove_endpoint` double-remove edges | `person_endpoint_double_edges` | D/P |
| D4 | `Person` CBOR round-trip (daemon-native wire) | `person_cbor_round_trips` | D/P |
| D5 | `PersonManager::{add_person double-add, remove_person double-remove}` | `manager_person_double_add`, `manager_person_double_remove` | D/P |
| D6 | `PersonManager::{associate/dissociate}` double edges | `manager_associate_dissociate_edges` | D/P |
| D7 | `PersonManager::{find_person, find_by_endpoint}` lookups | `manager_lookup_by_id_and_endpoint` | D/P |
| D8 | `PersonManager::add_person` mints a missing id (`ensure_id` discipline) | `manager_add_mints_missing_id` | D/P |

---

## Totals

- libpurple `g_test` cases enumerated in scope: **21** (16 person + 2 contact-manager-person + 3
  re-activated contact-info).
- Ported (P): **16** (13 person + 2 manager + … see below) — precisely: person 12 rows P (2, 3 are
  partial-P), manager 2 P, contact-info 3 P = **17 rows with a P** (rows 2 and 3 also carry an S half).
- Skipped (S): **4 full skips** (`avatar-for-display/contact`, both `color-for-display`, plus the
  `contact_with_alias` name row already S in `matching.md`) + the GObject signal/back-ref halves of
  rows 2/3/9/10/17.
- Derived (D): **8** daemon-native rows (id mint, empty-priority, endpoint edges, CBOR round-trip,
  manager double add/remove, associate/dissociate edges, lookups, add-mints-id).

## Wire additions (wire v37 — integration bumps `WireVersion` to 37)

- Types: `Person`, `PersonEndpoint` — `Serialize`/`Deserialize` + feature-gated `Arbitrary`. `Image`
  becomes wire-reachable (via `Person::avatar`) and gains its first CDDL rule.
- Op: `ApiRequest::PersonList` → `ApiResponse::Persons(Vec<Person>)` (`ControlApi::person_list`,
  default empty; classified `ControlRead` in `authz.rs`, `NotSessionTouching` in the ownership
  matrix) — EXACTLY the `NotificationList` pattern.
- Event: `NodeEvent::PersonsChanged` (payload-free pointer; clients re-list) — added because the host
  `PersonManager` has create/remove/associate/dissociate mutation seams, so a change pointer mirrors
  `NotificationsChanged`; emitted by `NodeApiImpl::emit_persons_changed`.
- CDDL: `image`, `person-endpoint`, `person`, `request-person-list`, `response-persons`,
  `node-event-persons-changed` (appended to the respective unions).
- Fixtures: `request-person-list.cbor`, `response-persons.cbor`; the `PersonsChanged` event added to
  `response-events-page.cbor`.

## Spec-doc updates

- `crates/engine/daemon-core/docs/daemon-transport-adapter-spec.md` §6: Person/MetaContact
  DEFERRED → implemented, citing `daemon_api::person` + `daemon_host::person`.
- Any `rg -i deferred` cross-references in `crates/engine/daemon-core/docs/` that name Person are
  updated with a short "implemented by" note (surgical, status-line only).
