# Port ledger — W2-E `port-message`

Message state (`ChatMessage`), the `Tags` container, and HTML-markup stripping, ported from
libpurple.

- **Target modules:**
  - `crates/contracts/daemon-api/src/message.rs` — `ChatMessage` DTO (+ `MessageAttachment`), wired
    via `mod message;` + re-exports in `lib.rs`; a **wire type** (reachable from `ApiResponse::Journal`
    through the new `JournalRecordPayload::Chat` arm), mirrored in `daemon-api.cddl`.
  - `crates/contracts/daemon-api/src/tags.rs` — `Tags` container, wired via `mod tags;` + re-exports.
    A **non-wire** domain type (like `details.rs`'s typed `AccountSettings`); no serde/CDDL. Placed in
    `daemon-api` so it can align with `ConversationType::tag_value` (W1-B, `details.rs`) — the
    conversation-type tag derivation is **reused, not duplicated** (`Tags::set_conversation_type`).
  - `crates/contracts/daemon-common/src/markup.rs` — `strip_html` (+ `unescape_entity`). No existing
    html-strip logic was found in `daemon-common` or the adapters (`daemon-matrix` sends
    `text_plain` / reads `Text.body`; it has no formatted-body→plaintext path), so this is a fresh,
    consolidated util rather than a duplicate.

- **Reference C sources (read-only):** `libpurple/purplemessage.c`, `libpurple/purpletags.c`,
  `libpurple/purplemarkup.c`, `libpurple/purpleattachment.c`; tests `tests/test_message.c`,
  `tests/test_tags.c`, `tests/test_markup.c`.

## Semantics established from the C sources

### `PurpleMessage` (`purplemessage.c`)
- `delivered` is **derived**: `purple_message_get_delivered` returns `delivered_at != NULL`. There is
  no stored `delivered` bool — the timestamp is the source of truth. Same for `edited`/`edited_at`.
- `purple_message_set_delivered(TRUE)` stamps `delivered_at` with "now"; `set_delivered(FALSE)` clears
  it to NULL. `purple_message_set_delivered_at(dt)` sets the stamp directly; a non-NULL `dt` therefore
  makes `get_delivered()` TRUE, and NULL makes it FALSE. Same truth table for edited.
  - **Divergence:** libpurple stamps with an internal `g_date_time_new_now_utc()`. A pure wire DTO has
    no clock, so `set_delivered(bool, now)` / `set_edited(bool, now)` take the node-authoritative `now`
    (unix seconds, matching the DTO's `Option<u64>` timestamp convention used elsewhere, e.g.
    `Presence::idle_since`). The state truth table is identical.
  - **Skip w/ reason:** the `notify::delivered` / `notify::delivered-at` **signal-emission counters**
    (`g_assert_cmpuint(counter, ==, 1)`) are GObject-specific; a plain DTO emits no signals. The
    observable STATE each counter guards (getter flips, stamp set/cleared) IS ported.
- `timestamp`, `id`, `title`, `highlight_color`, `replying_to` are plain nullable fields; `action`,
  `event`, `notice`, `system`, `highlighted` are plain bools; `error` is modeled as `Option<String>`
  (libpurple `GError`, reduced to its message). `attachments` is an (initially empty) collection
  (`PurpleAttachments`). `attributes` (a Pango `PangoAttrList` of rich-text runs) is **not modeled** —
  it is a GUI-toolkit formatting structure, not wire state.

### `PurpleTags` (`purpletags.c`)
- A tag is a string; `purple_tag_split` splits on the **first** `:` into `(name, value)`. No colon →
  `value = None`. A leading/sole `:` yields `("", Some(""))` etc. (see the `/tag/split` table).
- `add` = `real_add`: **remove any exactly-equal existing tag first, then append** (so a duplicate add
  moves the tag to the end; length stays constant). `add_with_value(name, value)` builds `"name:value"`
  (or `"name"` when `value == None`) then `real_add`.
- `exists(tag)` = exact full-string match; an empty tag is never present.
- `lookup(name)` walks tags; for a tag that has `name` as a prefix, the char after the prefix decides:
  `'\0'` → bare tag, returns `(None, found=true)`; `':'` → returns `(Some(value_after_colon), true)`.
  A partial name match (e.g. `"pur"` vs tag `"purple"`) does **not** match (`found=false`). `get` =
  `lookup` ignoring `found`.
- `get_all_with_name(name)` returns the sub-collection of tags whose name is exactly `name` (prefix
  then `'\0'` or `':'`); empty `name` → empty.
- `to_string(sep)` joins tag strings; `sep=None` → no separator (concatenation).
- `contains(needle)` — every tag in `needle` must `exists` in self.
- **Skip w/ reason:** GObject `added`/`removed` signal counters and the `GListModel` item-type /
  `items-changed` machinery are toolkit-specific; ported the observable state (`len`, membership,
  order, `remove` return bool) that those signals guard.

### `purple_markup_strip_html` (`purplemarkup.c`)
- Faithful byte-level port of the single-pass scanner. Key behaviors exercised by the matrix:
  `<script>`/`<style>` start CDATA that is dropped until the matching close tag; `</td>` then `<td>`
  becomes a tab; `<p>`/`<tr>`/`<hr>`/`<li>`/`<div>` map to `\n` **only when output is non-empty**
  (leading ones are suppressed); `<br>` and `</table>` map to `\n` unconditionally; `<a href=…>` saves
  the href and, at `</a>`, appends ` (href)` **unless** the visible link text already equals the href
  (modulo a leading `http://`). Entities are unescaped via `unescape_entity`.
- `unescape_entity` ports the named set (`&amp; &lt; &gt; &nbsp; &copy; &quot; &reg; &apos;`) plus
  numeric `&#dec;` / `&#xhex;` (rejecting `0`, `> i32::MAX`, or a missing `;`), emitting UTF-8.

## Case ledger

Status: PORT = ported & green · SKIP = out of model/scope (reason given) · EXTRA = added coverage
beyond the libpurple g_test (requested by scope: "port the full test matrix — entities, nested,
malformed").

### `test_message.c` → `message.rs` (5 g_test cases)

| libpurple g_test | Rust test | Status |
|---|---|---|
| `/message/properties` | `message_properties_roundtrip` | PORT (modeled fields; `attributes` skipped — GUI Pango run-list; signal-free) |
| `/message/delivered-sets-delivered-at` | `message_set_delivered_stamps_delivered_at` | PORT (state; signal counters skipped) |
| `/message/delivered-at-sets-delivered` | `message_set_delivered_at_implies_delivered` | PORT |
| `/message/edited-sets-edited-at` | `message_set_edited_stamps_edited_at` | PORT |
| `/message/edited-at-sets-edited` | `message_set_edited_at_implies_edited` | PORT |
| (extra) `is_empty` (`purple_message_is_empty`) | `message_is_empty` | EXTRA |
| (extra) `compare_timestamp` (`purple_message_compare_timestamp`) | `message_compare_timestamp` | EXTRA |
| (extra) wire round-trip through `JournalRecordPayload::Chat` | `chat_message_journal_payload_round_trips` | EXTRA (wire lockstep) |

### `test_tags.c` → `tags.rs` (27 g_test cases)

| libpurple g_test | Rust test | Status |
|---|---|---|
| `/tags/exists` | `tags_exists` | PORT |
| `/tags/lookup-exists` | `tags_lookup_exists` | PORT (signal counters skipped) |
| `/tags/lookup-non-existent` | `tags_lookup_non_existent` | PORT |
| `/tags/add-remove-bare` | `tags_add_remove_bare` | PORT |
| `/tags/add-duplicate-bare` | `tags_add_duplicate_bare` | PORT (dup re-add keeps len 1) |
| `/tags/remove-non-existent-bare` | `tags_remove_non_existent_bare` | PORT |
| `/tags/add-with-value` | `tags_add_with_value` | PORT |
| `/tags/add-with-value-null` | `tags_add_with_value_null` | PORT |
| `/tags/add-remove` | `tags_add_remove` | PORT |
| `/tags/add-remove-with-null-value` | `tags_add_remove_with_null_value` | PORT |
| `/tags/add-remove-with-value` | `tags_add_remove_with_value` | PORT |
| `/tags/add-duplicate-with-value` | `tags_add_duplicate_with_value` | PORT |
| `/tags/remove-non-existent-with-value` | `tags_remove_non_existent_with_value` | PORT |
| `/tags/remove-all-empty` | `tags_remove_all_empty` | PORT |
| `/tags/remove-all-single` | `tags_remove_all_single` | PORT |
| `/tags/remove-all-multiple` | `tags_remove_all_multiple` | PORT |
| `/tags/get-single` | `tags_get_single` | PORT |
| `/tags/get-multiple` | `tags_get_multiple` | PORT |
| `/tags/get-all` | `tags_get_all` | PORT |
| `/tags/get-all-with-name` | `tags_get_all_with_name` | PORT |
| `/tags/to-string-single` | `tags_to_string_single` | PORT |
| `/tags/to-string-multiple-with-separator` | `tags_to_string_multiple_with_separator` | PORT |
| `/tags/to-string-multiple-with-null-separator` | `tags_to_string_multiple_with_null_separator` | PORT |
| `/tag/split` | `tag_split_table` | PORT (full data table incl. `🐦` unicode rows) |
| `/tag/contains/full` | `tags_contains_full` | PORT |
| `/tag/contains/partial` | `tags_contains_partial` | PORT |
| `/tag/contains/none` | `tags_contains_none` | PORT |
| (extra) reuse `ConversationType::tag_value` | `tags_set_conversation_type` | EXTRA (alignment w/ W1-B) |

### `test_markup.c` → `markup.rs` (1 g_test case, 10 data rows)

`/util/markup/strip-html` is one g_test with a 10-row data table; ported as `strip_html_libpurple_matrix`
covering every row:

| data row | expected | Status |
|---|---|---|
| `""` | `""` | PORT |
| `<a href="…example.com/">…example.com/</a>` | `https://example.com/` (href == text → no suffix) | PORT |
| `<a href="…example.com/">example.com</a>` | `example.com (https://example.com/)` | PORT |
| `<script>…</script>` | `""` | PORT |
| `<style>…</style>` | `""` | PORT |
| table (2×2) | `1\t2\n3\t4\n` | PORT |
| `<p>foo</p><p>bar</p><p>baz</p>` | `foo\nbar\nbaz` | PORT |
| `<div><p>foo</p><p>bar</p></div>` | `foo\nbar` | PORT |
| `<hr>` | `""` | PORT |
| `<br>` | `\n` | PORT |
| (extra) named entities `&amp;/&lt;/&gt;/&quot;/&apos;/&nbsp;/&copy;/&reg;` | decoded | EXTRA |
| (extra) numeric `&#65;` / `&#x41;` and invalid `&#0;`/`&#foo;` | decoded / literal | EXTRA |
| (extra) nested + malformed/unclosed tag | stripped safely | EXTRA |

## Summary

- **libpurple g_test cases enumerated in scope:** 33 (5 message + 27 tags + 1 markup g_test whose
  10-row data table is ported in full), all PORT — zero skipped cases.
- **Skipped (documented sub-aspects, not counted as cases):** GObject signal-emission counters and
  `GListModel` machinery (message `notify::*`, tags `added`/`removed` + item-type/items-changed), and
  `PurpleMessage:attributes` (Pango run-list). All are toolkit constructs with no wire/DTO meaning;
  the state they guard is ported.
- **EXTRA coverage added:** message `is_empty` / `compare_timestamp` / wire round-trip; tags
  conversation-type alignment; markup entity + numeric-entity + malformed rows.

## Evidence (TDD)

- **Rust test totals:** 41 tests — 8 `message.rs` + 28 `tags.rs` + 5 `markup.rs`.
- **RED (commit `3dad397`, stubs only):** 35 of 41 failed at assertion level — markup 5/5,
  tags 23/28, message 7/8. The 6 that passed red are vacuously-true-on-empty edges
  (`tags_lookup_non_existent`, `tags_remove_non_existent_bare`, `tags_remove_non_existent_with_value`,
  `tags_remove_all_empty`, `tags_contains_none`) and the serde-only wire round-trip
  (`chat_message_journal_payload_round_trips`), which exercises no stubbed behavior.
- **GREEN:** all 41 pass; CDDL conformance (`fixtures_validate_against_cddl`,
  `invalid_payloads_are_rejected`) and the arbitrary proptests
  (`arbitrary_api_request_matches_cddl`, `arbitrary_api_response_matches_cddl`) green, proving the
  `chat-message` / `message-attachment` / `journal-record-payload-chat` CDDL rules are in lockstep
  with the Rust types (incl. the `Box<ChatMessage>` variant payload, which encodes identically).
