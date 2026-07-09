# Port ledger — W1-C `port-matching`

Display-name, matching, ordering, and member-collection logic ported from libpurple.

- **Target module:** `crates/contracts/daemon-api/src/matching.rs` (wired via `mod matching;` + re-exports in `lib.rs`).
- **Reference C sources (read-only):** `libpurple/purplecontactinfo.c`, `purpleconversationmember.c`,
  `purpleconversationmembers.c`, `purpleconversationmanager.c`, `purplebadges.c`, `util.c`; tests under
  `libpurple/tests/`.

## Semantics established from the C sources

- **`birb_str_matches(pattern, str)`** — caseless **subsequence** match: returns TRUE if `pattern`
  occurs in sequential order within `str`, caseless, ignoring characters in between
  (docs.imfreedom.org/birb: `Br` matches `biRb`). Ported as `str_matches` (casefold both via Unicode
  lowercase, then subsequence scan).
- **`birb_str_is_empty(s)`** — TRUE when `s` is NULL or `""`. Modeled as `None`/`Some("")`/`""`.
- **`purple_utf8_strcasecmp(a, b)`** — NULL: `!a&&b→-1`, `!b&&a→1`, `!a&&!b→0`; else
  `g_utf8_collate(g_utf8_casefold(a), g_utf8_casefold(b))`. Ported as Unicode-lowercase casefold then
  codepoint (`str::cmp`) ordering. **Divergence:** `g_utf8_collate` is full locale collation; we use
  codepoint order after casefolding — identical for all tested (ASCII) inputs; full ICU collation is
  out of scope (no test exercises it).
- **`ContactInfo::name_for_display` chain** — C is `alias → person-alias → display_name → id`. The
  daemon `ContactInfo` has neither an `alias` nor a `person` field (alias/nickname live on
  `ConversationMember`), so the ported chain is `display_name → id`. **W3-J update:** the
  person-aware layer now exists (`name_for_display_with_person` / `contact_info_compare_with_person`,
  taking `Option<&Person>`) — see `docs/port-ledger/person.md`.
- **`ConversationMember::name_for_display` chain** — `alias → nickname → contact.name_for_display`
  (faithful; `purpleconversationmember.c:582`).
- **Member equality for collections** — `check_member_equal` compares the members' **contact infos**
  via `purple_contact_info_compare(...) == 0` (i.e. name-for-display caseless), *not* full member
  compare. Ported as `contact_info_equal`.
- **`purple_conversation_member_compare`** — badges first, then name-for-display caseless. The daemon
  `ConversationMember` has no badges; it has `role: MemberRole`. **Mapping:** badges→role — a member
  with a higher role sorts first (mirrors "more/higher badges sorts first" in `purple_badges_compare`).
- **`find_dm`** — per (account, is-DM, `has_member(contact)`); account→`TransportId`.

## Case ledger

Status: PORT = ported & green · SKIP = out of model/scope (reason given) · DERIVED = requested by
scope but no direct libpurple g_test.

### `test_contact_info.c` (in-scope: name-for-display / compare / equal / matches)

| libpurple g_test | Rust test (`matching.rs`) | Status |
|---|---|---|
| `/contact-info/get_name_for_display/person_with_alias` | `contact_info_name_for_display_person_alias` | PORT (re-activated by W3-J `port-person` via `ContactInfo::name_for_display_with_person`; was SKIP: no `person` on daemon `ContactInfo`) |
| `/contact-info/get_name_for_display/contact_with_alias` | — | SKIP: no `alias` field on daemon `ContactInfo` (alias lives on `ConversationMember`, covered there) |
| `/contact-info/get_name_for_display/contact_with_display_name` | `contact_info_name_for_display_display_name` | PORT |
| `/contact-info/get_name_for_display/id_fallback` | `contact_info_name_for_display_id_fallback` | PORT |
| `/contact-info/compare/not_null__null` | `contact_info_compare_not_null_null` | PORT |
| `/contact-info/compare/null__not_null` | `contact_info_compare_null_not_null` | PORT |
| `/contact-info/compare/null__null` | `contact_info_compare_null_null` | PORT |
| `/contact-info/compare/person__no_person` | `contact_info_compare_person_no_person` | PORT (re-activated by W3-J `port-person` via `contact_info_compare_with_person`) |
| `/contact-info/compare/no_person__person` | `contact_info_compare_no_person_person` | PORT (re-activated by W3-J `port-person` via `contact_info_compare_with_person`) |
| `/contact-info/compare/name__name` | `contact_info_compare_name_name` | PORT |
| `/contact-info/equal/not_null__not_null` | `contact_info_equal_not_null_not_null` | PORT |
| `/contact-info/equal/not_null__null` | `contact_info_equal_not_null_null` | PORT |
| `/contact-info/equal/null__not_null` | `contact_info_equal_null_not_null` | PORT |
| `/contact-info/equal/null__null` | `contact_info_equal_null_null` | PORT |
| `/contact-info/matches/accepts_null` | `contact_info_matches_accepts_null` | PORT |
| `/contact-info/matches/emptry_string` | `contact_info_matches_empty_string` | PORT |
| `/contact-info/matches/alias` | — | SKIP: no `alias` field (covered at member level) |
| `/contact-info/matches/display_name` | `contact_info_matches_display_name` | PORT |
| `/contact-info/matches/none` | `contact_info_matches_none` | PORT (drop alias set) |

Out of scope (other W1 packages): `/new`, `/properties`, `/get-avatar-for-display`,
`/presence-changed-signal`, `/get_menu`.

### `test_conversation_member.c` (in-scope: name-for-display / matches / compare)

| libpurple g_test | Rust test | Status |
|---|---|---|
| `/conversation-member/name-for-display` | `member_name_for_display_precedence` | PORT |
| `/conversation-member/matches/accepts_null` | `member_matches_accepts_null` | PORT |
| `/conversation-member/matches/empty_string` | `member_matches_empty_string` | PORT |
| `/conversation-member/matches/alias` | `member_matches_alias` | PORT |
| `/conversation-member/matches/nickname` | `member_matches_nickname` | PORT (C sets `alias`; faithful — C test name is a misnomer) |
| `/conversation-member/matches/contact_info` | `member_matches_contact_info` | PORT (contact alias→`display_name`) |
| `/conversation-member/compare/not_null__null` | `member_compare_not_null_null` | PORT |
| `/conversation-member/compare/null__not_null` | `member_compare_null_not_null` | PORT |
| `/conversation-member/compare/null__null` | `member_compare_null_null` | PORT |
| `/conversation-member/compare/same` | `member_compare_same` | PORT |
| `/conversation-member/compare/nickname__nickname` | `member_compare_nickname_nickname` | PORT |
| `/conversation-member/compare/badges__nickname` | `member_compare_role_nickname` | PORT (badges→role) |

Out of scope: `/new`, `/properties`, `/typing-state/timeout`, `/badges-changed-signal`, `/is-account`.

### `test_conversation_members.c` (in-scope: membership collection semantics)

| libpurple g_test | Rust test | Status |
|---|---|---|
| `/conversation-members/add-remove` | `members_add_remove` | PORT (membership semantics; GObject signal counters out of scope) |
| `/conversation-members/remove-all` | `members_remove_all` | PORT |
| `/conversation-members/find-or-add-member` | `members_find_or_add_member` | PORT |
| `/conversation-members/items-changed` | — | SKIP: `GListModel` items-changed signal; N/A to a plain `Vec` |
| `/conversation-members/active-typers` | `members_active_typers` | PORT (filter) |
| `/conversation-members/extend` | `members_extend` | PORT (append + clear source; signal positions out of scope) |
| `/conversation-members/find-first-other` | `members_find_first_other` | PORT |

### `test_conversation_manager.c` (in-scope: find-dm only)

| libpurple g_test | Rust test | Status |
|---|---|---|
| `/conversation-manager/find-dm/empty` | `find_dm_empty` | PORT |
| `/conversation-manager/find-dm/exists` | `find_dm_exists` | PORT |
| `/conversation-manager/find-dm/does-not-exist` | `find_dm_does_not_exist` | PORT |

Out of scope: `/add-remove`, `/signals/*`.

### Derived (scope item 5 — "same set of participants regardless of order")

| Behavior | Rust test | Status |
|---|---|---|
| order-independent member-set equality (`same_member_set`) | `same_member_set_order_independent` | DERIVED (no direct g_test; requested by W1-C scope) |

## Summary

- **In-scope g_test cases enumerated:** 41 (23 contact-info of which 18 in-scope + 5 person/alias
  skips; 12 member; 7 members of which 6 in-scope + 1 skip; 3 find-dm).
- **Ported:** 35 · **Skipped:** 6 (5 person/alias model gaps + 1 GListModel signal) · **Derived:** 1.
- **W3-J update:** 3 of the person-dependent skips (`get_name_for_display/person_with_alias`,
  `compare/person__no_person`, `compare/no_person__person`) are now PORTED by `port-person`
  (see `docs/port-ledger/person.md`), leaving 2 skips (`contact_with_alias` name + `matches/alias` —
  no `alias` field on daemon `ContactInfo`) + the GListModel signal row → totals now
  **Ported: 38 · Skipped: 3 · Derived: 1**.
