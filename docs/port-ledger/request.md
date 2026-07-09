# Port ledger — W2-I `port-request`

The libpurple **request-field model**: typed fields, groups, pages, per-field filled/valid semantics,
and group/page validity aggregation.

- **Target module:** `crates/contracts/daemon-api/src/request.rs` (wired via `mod request;` + `pub use
  request::*;` appended at the END of the module list in `lib.rs`, mirroring how `details.rs` /
  `matching.rs` were added in Wave 1 — new file + two appended lines, to minimise integration-merge
  conflicts).
- **Reference C sources (read-only):** `libpurple/request/purplerequestfield.c` (abstract base),
  `purplerequestfieldstring.c`, `purplerequestfieldint.c`, `purplerequestfieldbool.c`,
  `purplerequestfieldchoice.c`, `purplerequestfieldlist.c`, `purplerequestfieldimage.c`,
  `purplerequestfieldaccount.c`, `purplerequestfieldlabel.c`, `purplerequestgroup.c`,
  `purplerequestpage.c`; tests under `libpurple/tests/test_request_field*.c`, `test_request_group.c`,
  `test_request_page.c`.

## Relationship to the existing `AuthParamField` (do NOT break it)

`crate::AuthParamField` (`{ key, label, required }`) is the **wire** discovery shape for the
interactive-auth `params` form (`AuthApi`, `AuthProviderInfo::params_schema`, `AccountSettingsSchema`).
It STAYS exactly as-is — it is the minimal, serialised contract a thin client renders for auth
discovery and is untouched by this package. The new `RequestField` enum is the *fuller, node-internal*
generalisation of that idea: it carries per-variant typed data (string/int/bool/choice/list/image/
account/label) plus filled/valid semantics that `AuthParamField` deliberately omits. Think of
`AuthParamField` as the on-the-wire projection of a would-be `RequestField::String { required, .. }`;
`RequestField` is the node-side authority that computes validity. See the "wire exposure" note below.

## Semantics established from the C sources

- **`birb_str_is_empty(s)`** — TRUE iff `s` is NULL or `""`. Whitespace is NOT empty (consistent with
  Wave-1 `details.rs`/`matching.rs`). A string field's value is modelled as `Option<String>`; *filled*
  = `Some(s)` with `s != ""`. A whitespace-only value is therefore **filled** (documented, matches C).
- **`g_set_str(&dst, src)`** — replaces `dst` and returns TRUE iff the string changed; NULL and `""`
  are *distinct* (NULL→"" is a change). Modelled by `set_str` over `Option<String>`; drives the
  string field's "filled changed" transition (the `notify::filled` the C test counts).
- **Base field `purple_request_field_is_valid(field, &errmsg)`** — ordered gate:
  1. the subclass validator (`klass->is_valid`; only *int* has one — the bounds check),
  2. then, iff still valid, the custom validator closure,
  3. then, iff still valid AND `required` AND `!is_filled`, fail with `"Required field is not
     filled."`.
  Ported as `RequestField::is_valid() -> Result<(), String>` with the same short-circuit order
  (`?`-chaining), so e.g. an out-of-bounds int reports the bounds error and never runs the custom
  validator (the `valid-custom` test's `called` stays false).
- **`is_filled`** — only strings compute it (`!birb_str_is_empty(value)`); every other variant is
  always filled (`filled = TRUE`), so a non-string never blocks validity via the required-check.
- **Int bounds** — `value < lower → "Int value {v} exceeds lower bound {lo}"`,
  `value > upper → "Int value {v} exceeds upper bound {hi}"`, else ok.
- **Choice** (single-select typed options): ordered `items: Vec<LocalizedString>`, a `selected` index
  (default 0). `get_selected` → `None` (C `G_MAXUINT`) iff empty, else the stored index.
  `set_selected(i)` updates iff `i != current && i < len` (out-of-bounds ignored). `selected_item` →
  item at `selected` iff `len>0 && selected<len`. `remove(pos)`: on removing the selected position,
  `selected` resets to 0. `remove_by_id`/`remove_item` delegate to first-id match. No de-dup on add.
- **List** (multi-select): ordered `items`, `multi_select` flag, and a `selected` sub-list.
  `select_item(id)`: already-selected → `false`; in single-select with a non-empty selection, clear it
  first; then append the matching item → `true` (or `false` if `id` unknown). No de-dup on add;
  `remove_by_id` removes the first id match.
- **Image / Account / Label** — thin data holders: image = `Option<ImageRef>`; account =
  `Option<TransportId>` selected + `model: Vec<TransportId>` (accounts are instance-qualified
  transport ids in this daemon — see `BoundAccount::transport_instance`); label = id+label only.
  All nullable setters round-trip (`…-supports-null` tests).
- **Group** — ordered fields; `is_valid` = every field valid (empty ⇒ valid). Validity is **recomputed
  when a member field changes** (C keeps an `invalid_fields` set updated via each field's
  `notify::valid`; we recompute the pure predicate and cache the last value only to detect *flips*).
- **Page** — ordered groups; `is_valid` = every group valid (empty ⇒ valid); field lookup by id across
  all groups (`get_field`/`exists` + typed `get_string`/`get_integer`/`get_bool`/`get_choice`/
  `get_account`, each returning the type-mismatch fallback the C gettersuse); `close()` emits a
  one-shot close (modelled as an emission counter + `is_closed()`).

## Modelling divergences (faithful, documented)

- **No GObject plumbing.** `PurpleRequestField` is a GObject hierarchy (abstract base + `G_DEFINE_*`
  subclasses, `GListModel`, properties, signals). The daemon model is a plain `RequestField` enum +
  per-variant structs. The following GObject-only assertions are therefore **skipped-with-reason**
  (they test the toolkit, not domain behaviour): `birb_assert_type` / `G_IS_LIST_MODEL` type-identity;
  `g_assert_finalize_object` refcount finalisation; `g_object_get`/`g_object_new` property plumbing;
  and the `items-changed` / `notify::…` **emission-count** integers (`counter == 1/2`). Where a
  counter encodes *domain* meaning (single-select *replaces* → selection size stays 1; multi-select
  *adds* → size grows to 2; a validity *flip* occurred) that meaning IS ported via the resulting state
  and a flip-count.
- **Custom validators** are `Arc<dyn Fn(&RequestField) -> Result<(), String> + Send + Sync>` (the C
  `PurpleRequestFieldValidator` closure). The `is_even` / `"valid"` test validators port directly; the
  `called`-was-invoked flag ports via a captured `Arc<AtomicBool>`.
- **`notify::filled` / `notify::valid` transitions** — the string field's `set_value` returns whether
  *filled* toggled; group/page expose `revalidate()` returning whether validity *flipped*. Test
  counters accumulate these booleans, reproducing the C `called` counts exactly.

## Wire exposure decision — KEEP NODE-INTERNAL (candidate for a future wire version)

**No wire change in this package.** The wire carries *data*, not *validators*, and there is no
concrete consuming surface today that should carry a full `RequestPage`: the auth-discovery surface is
already served by the untouched `AuthParamField`/`AuthProviderInfo` shapes, and no adapter action
surface currently upgrades to a request-form. So `RequestField`/`RequestGroup`/`RequestPage` are
node-internal (NO serde, NO `Arbitrary`, NO CDDL rules, NO new ops, NO `WireVersion` bump), exactly as
`details.rs`/`matching.rs` are. The module doc-comment tags the future request-UI surface as a
**candidate for a future wire version** (beyond the v37 libpurple-parity bump): when a client needs
to render an interactive request form (e.g. an
`AuthChallenge::Form` upgrade or a protocol-driven prompt), lift the *data* projection of these types
onto the wire then (append-only, feature-gated `Arbitrary`, CDDL lockstep in the same commit) — the
validators stay node-side by the "node decides, apps render" invariant.

## Case ledger

Planned Rust test names live in `request.rs`'s `#[cfg(test)] mod tests`. Status: ✅ ported green,
⏭️ skipped-with-reason (GObject plumbing, covered elsewhere in the same case).

### `test_request_field.c` — `/request-field/*`

| C g_test case            | Rust test                          | Notes |
|--------------------------|------------------------------------|-------|
| `/filled-string`         | `field_filled_string`              | ✅ NULL≡"" empty; filled-transition count via `set_value` return |
| `/filled-nonstring`      | `field_filled_nonstring`           | ✅ int always filled, value changes never toggle filled |
| `/valid-int`             | `field_valid_int`                  | ✅ bounds + exact error strings; errmsg-less path is just `.is_err()` |
| `/valid-custom`          | `field_valid_custom`               | ✅ bounds checked before custom; `called` via `Arc<AtomicBool>` |
| `/required-validity`     | `field_required_validity`          | ✅ required+empty ⇒ "Required field is not filled." |

### `test_request_field_choice.c` — `/request/field/choice/*`

| C g_test case      | Rust test                       | Notes |
|--------------------|---------------------------------|-------|
| `/new`             | `choice_new`                    | ✅ ctor + item-type notion; ⏭️ `G_IS_LIST_MODEL`/finalize |
| `/properties`      | `choice_properties`             | ✅ id/label/n_items=0/selected=None/selected_item=None; ⏭️ GType |
| `/add-remove`      | `choice_add_remove`             | ✅ n_items after each op, remove returns; ⏭️ items-changed counts |
| `/selected`        | `choice_selected`               | ✅ empty→None, set in/out of bounds |
| `/selected-item`   | `choice_selected_item`          | ✅ selected-item id tracking, out-of-bounds ignored |

### `test_request_field_account.c` — `/request/field/account/*`

| C g_test case              | Rust test                          | Notes |
|----------------------------|------------------------------------|-------|
| `/new/without-model`       | `account_new_without_model`        | ✅ ctor, empty model; ⏭️ type/finalize |
| `/new/with-model`          | `account_new_with_model`           | ✅ ctor with model |
| `/properties`              | `account_properties`               | ✅ account + model round-trip; ⏭️ `g_object_get` |
| `/account-supports-null`   | `account_supports_null`            | ✅ default None, set/clear round-trip |

### `test_request_field_image.c` — `/request/field/image/*`

| C g_test case            | Rust test                    | Notes |
|--------------------------|------------------------------|-------|
| `/new/normal`            | `image_new_normal`           | ✅ ctor with image; ⏭️ type/finalize |
| `/new/null`              | `image_new_null`             | ✅ ctor with None image |
| `/properties`            | `image_properties`           | ✅ id/label/image round-trip |
| `/image-supports-null`   | `image_supports_null`        | ✅ set image None round-trip |

### `test_request_field_list.c` — `/request/field/list/*`

| C g_test case          | Rust test                    | Notes |
|------------------------|------------------------------|-------|
| `/new/normal`          | `list_new_normal`            | ✅ ctor; ⏭️ `G_IS_LIST_MODEL`/finalize |
| `/properties`          | `list_properties`            | ✅ id/label/multi_select/n_items=0; ⏭️ GType |
| `/add-remove`          | `list_add_remove`            | ✅ dup allowed (n_items grows), remove_by_id, remove(None)→false, clear |
| `/single-selection`    | `list_single_selection`      | ✅ select replaces (size stays 1), re-select→false, clear_selected |
| `/multi-selection`     | `list_multi_selection`       | ✅ select adds (size grows), re-select→false |

### `test_request_group.c` — `/request-group/*`

| C g_test case | Rust test      | Notes |
|---------------|----------------|-------|
| `/valid`      | `group_valid`  | ✅ empty valid, add/change field flips validity, flip-count parity |

### `test_request_page.c` — `/request/page/*`

| C g_test case   | Rust test          | Notes |
|-----------------|--------------------|-------|
| `/new`          | `page_new`         | ✅ ctor; ⏭️ type/finalize |
| `/properties`   | `page_properties`  | ✅ title/subtitle round-trip; ⏭️ `g_object_get` |
| `/valid`        | `page_valid`       | ✅ empty valid, group validity aggregation, flip-count parity |
| `/close`        | `page_close`       | ✅ close emits once (emission counter) |

**Totals:** 28 C g_test cases → 28 Rust tests (all ✅). GObject-only sub-assertions inside 15 of those
cases are ⏭️ skipped-with-reason as noted (toolkit plumbing, not domain behaviour). Plus derived-helper
tests for typed page getters, group/page field lookup, and choice selected-reset-on-removal.
