# Port ledger — W1-B `port-dto-logic`

Conversation/account **DTO behavior logic** ported from libpurple into
`crates/contracts/daemon-api/src/details.rs` (new module; `lib.rs` gains only
`mod details;` + a re-export).

Scope (per work package): `CreateConversationDetails::is_valid`,
`ChannelJoinDetails::merge`, `ConversationInfo` title logic + conversation-type
predicates/tag derivation, typed account-settings accessors, and
`Presence`/`PresencePrimitive` predicate/ordering + `DisconnectReason` variants.

Wave-1 rule: **no wire-contract changes**. Every ported behavior is a new
method/function/enum defined over existing wire DTOs, or a new **non-wire**
in-memory model (the typed account-settings model) that projects back onto the
existing string-keyed `AccountSettingsValues`. The sibling package owns
name/display/matching helpers on `ContactInfo`/`ConversationMember`
(`src/matching.rs`); this package does not implement those.

"Done" = every row below is green or explicitly skipped-with-reason.

Legend: **P** = ported (green), **S** = skipped (reason given), **+** = extra
Rust case with no direct libpurple `g_test` (derived from the `.c`
implementation / daemon-native DTO).

---

## 1. `purplecreateconversationdetails.c` — `test_create_conversation_details.c` (7 g_test)

| # | libpurple case | Rust test (`details::tests`) | status |
|---|---|---|---|
| 1 | `/create-conversation-details/new` | `ccd_new_sets_max_participants` | P |
| 2 | `/create-conversation-details/properties` | `ccd_properties_roundtrip` | P |
| 3 | `/create-conversation-details/is-valid/null` | `ccd_is_valid_null_no_participants` | P |
| 4 | `/create-conversation-details/is-valid/empty` | `ccd_is_valid_empty_no_participants` | P |
| 5 | `/create-conversation-details/is-valid/too-many` | `ccd_is_valid_too_many` | P |
| 6 | `/create-conversation-details/is-valid/limited` | `ccd_is_valid_limited_ok` | P |
| 7 | `/create-conversation-details/is-valid/unlimited` | `ccd_is_valid_unlimited_ok` | P |

Semantics (`purple_create_conversation_details_is_valid`): 0 participants →
`NoParticipants`; `max>0 && n>max` → `TooManyParticipants`; `max==0` = unlimited;
else Ok. Ported as `is_valid(&self) -> Result<(), CreateConversationDetailsError>`.

## 2. `purplechanneljoindetails.c` — `test_channel_join_details.c` (3 g_test)

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 8 | `/channel-join-details/new` | `cjd_new_defaults` | P |
| 9 | `/channel-join-details/properties` | `cjd_properties_roundtrip` | P |
| 10 | `/channel-join-details/merge` | `cjd_merge_copies_source_fields` | P |

Semantics (`purple_channel_join_details_merge(source, destination)`): copies
`name`, `nickname_supported`, `nickname`, `password_supported`, `password` from
source into destination; **does not** copy the three `*_max_length` fields.
Ported as `dest.merge(&source)`.

## 3. `purpleconversation.c` — `test_conversation.c` (23 g_test; this package owns the title/type/tag subset)

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 11 | `/conversation/is-dm` | `conv_type_is_dm` | P |
| 12 | `/conversation/is-group-dm` | `conv_type_is_group_dm` | P |
| 13 | `/conversation/is-channel` | `conv_type_is_channel` | P |
| 14 | `/conversation/is-thread` | `conv_type_is_thread` | P |
| 15 | `/conversation/title-for-display` | `conv_title_for_display_precedence` | P |
| 16 | `/conversation/generate-title/empty` | `conv_generate_title_empty_none` | P |
| 17 | `/conversation/generate-title/dm` | `conv_generate_title_dm` | P |
| 18 | `/conversation/generate-title/group-dm` | `conv_generate_title_group_dm` | P |
| 19 | `/conversation/tags/unset` | `conv_type_tag_unset_none` | P (type-tag portion) |
| 20 | `/conversation/tags/dm` | `conv_type_tag_dm` | P (type-tag portion) |
| 21 | `/conversation/tags/group-dm` | `conv_type_tag_group_dm` | P (type-tag portion) |
| 22 | `/conversation/tags/channel` | `conv_type_tag_channel` | P (type-tag portion) |
| 23 | `/conversation/tags/thread` | `conv_type_tag_thread` | P (type-tag portion) |
| 24 | `/conversation/tags/federated-channel` | — | S: `federated` is not modeled on the `ConversationInfo` wire DTO; the `type=channel` portion is covered by #22. |
| — | `/conversation/properties` | — | S: GObject property bag; not domain behavior (and touches Image/Badges/Tags/dates out of scope). |
| — | `/conversation/equal` | — | S: identity/equality over live `PurpleConversation` (account+type+id); belongs to conversation-manager scope, not DTO logic. |
| — | `/conversation/set-topic-full` | — | S: setter + GDateTime bookkeeping; no DTO decision logic. |
| — | `/conversation/message/write-one` | — | S: message-list model (other package). |
| — | `/conversation/signals/present` | — | S: GObject signal emission. |
| — | `/conversation/signals/displayed` | — | S: GObject signal emission. |
| — | `/conversations/tags/changed-signal` | — | S: `PurpleTags` signal plumbing (other package). |
| — | `/conversation/new-message-signal` | — | S: signal + member model (other package). |
| — | `/conversation/member-propagation-signals` | — | S: member add/remove signals (other package). |

Notes:
- `title_for_display`: `alias` (non-empty) → `title` (non-empty) → `id`
  (`purple_conversation_get_title_for_display`). Ported as
  `ConversationInfo::title_for_display(&self, alias: Option<&str>)` + free
  `title_for_display(alias, title, id)`.
- `generate_title`: only DM/GroupDM; skip the account's own member; per remaining
  member use its display name (empty skipped); join with `", "`; `Some(title)`
  iff ≥1 name, else `None` (title unchanged). Member display name resolution
  uses the DTO's `display_name`-else-`id` (the tested behavior); the fuller
  `name_for_display` precedence lives in the sibling `matching.rs`.
- Type-tag derivation (`purple_conversation_set_conversation_type`):
  `Unset→None`, `Dm→"dm"`, `GroupDm→"group-dm"`, `Channel→"channel"`,
  `Thread→"thread"`. Ported as `ConversationType::tag_value()`.

## 4. `purpleaccountsetting.c` — `test_account_setting.c` (8 g_test)

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 25 | `/account-setting/properties` | — | S: GObject metadata property bag (advanced/developer-mode/hint/weight); not behavior. id/label are covered by the constructors used below. |
| 26 | `/account-setting/changed` | — | S: GObject signal emission. |
| 27 | `/account-setting/boolean` | `setting_boolean_get_set_value` | P |
| 28 | `/account-setting/int` | `setting_int_get_set_value` | P |
| 29 | `/account-setting/string` | `setting_string_get_set_value` | P |
| 30 | `/account-setting/string-list/new` | `setting_string_list_new_empty` | P |
| 31 | `/account-setting/string-list/add` | `setting_string_list_add_dedup` | P |
| 32 | `/account-setting/string-list/set-active` | `setting_string_list_set_active` | P |

## 5. `purpleaccountsettings.c` — `test_account_settings.c` (12 g_test)

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 33 | `/account-settings/new` | `settings_new_empty` | P |
| 34 | `/account-settings/properties` | — | S: GObject `item-type`/`n-items`; `n-items==0` covered by #33. |
| 35 | `/account-settings/add-remove` | `settings_add_remove` | P |
| 36 | `/account-settings/double-add` | `settings_double_add_rejected` | P (modeled as `add_setting` returning `false`; libpurple's `g_return`/CRITICAL is a C precondition, not portable) |
| 37 | `/account-settings/add-again` | `settings_add_again_wrong_type_rejected` | P (same-id different-type add rejected) |
| 38 | `/account-settings/propagate-changed-signal` | — | S: GObject signal emission. |
| 39 | `/account-settings/get-set-boolean` | `settings_get_set_boolean` | P |
| 40 | `/account-settings/get-set-int` | `settings_get_set_int` | P |
| 41 | `/account-settings/get-set-string` | `settings_get_set_string` | P |
| 42 | `/account-settings/get-set-string-list` | `settings_get_set_string_list` | P |
| 43 | `/account-settings/remove-all` | `settings_remove_all` | P |
| 44 | `/account-settings/update` | `settings_update` | P |

Typed-accessor semantics: `get_bool/int/string/string_list(id, fallback)` return
the value only when a setting exists **and has the matching type**, else the
fallback (the "wrong type → fallback" cases). `set_*` are no-ops unless the
setting exists with the matching type. `update` copies same-type values, adds
missing settings, and skips type-mismatched ones. Implemented as a new non-wire
`AccountSettings` model; `AccountSettings::to_values()` projects onto the wire
`AccountSettingsValues`.

## 6. `purplepresence.c` — `test_presence.c` (3 g_test) + `DisconnectReason` (daemon-native)

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 45 | `/presence/new` | — | S: GObject constructor; `Presence::default()` primitive is `Offline` — covered by #48. |
| 46 | `/presence/properties` | `presence_properties_roundtrip` | P |
| 47 | `/presence/primitive-primitive-changed-signal` | — | S: GObject signal emission. |
| 48 | `purple_presence_is_available` (impl) | `presence_is_available` | + |
| 49 | `purple_presence_is_online` (impl) | `presence_is_online_all_primitives` | + |
| 50 | `purple_presence_is_idle` (impl) | `presence_is_idle` | + |
| 51 | `purple_presence_compare` (impl) | `presence_compare_ordering` | + |
| 52 | `purple_presence_compare` NULL-arms (impl) | `presence_compare_options` | + |
| 53 | `DisconnectReason` fatal policy (daemon) | `disconnect_reason_is_fatal_variants` | + |

Presence semantics (`purplepresence.c`):
- `is_available` = primitive == `Available`.
- `is_online` = one of `Available/Idle/Invisible/Away/DoNotDisturb/Streaming`;
  `Offline` and (notably) `OutOfOffice` → false (OutOfOffice falls to the
  `default` arm in the C `switch`).
- `is_idle` = `is_online() && idle_since.is_some()`.
- `compare`: non-offline sorts before offline; otherwise compare idle times with
  `birb_date_time_compare` semantics (`None` sorts before `Some`, i.e.
  `Option<u64>::cmp`). NULL-aware `presence_compare(Option, Option)` matches the
  C pointer-null arms.
- `DisconnectReason::is_fatal`: `AuthenticationFailed | InvalidSettings |
  CertificateError` → fatal (mirrors the node's existing
  `daemon-host::…::reason_is_fatal`; the node — not a thin client — owns this).

## `test_connection.c` (5 g_test) — all out of scope

`properties`, `set_presence_default`, `get-action-menu/valid`,
`implements-set-display-name`, `set-display-name` are GObject/async connection
mechanics with no `DisconnectReason`/DTO decision logic. **S** (all 5). The
disconnect-reason DTO behavior is covered by #53.

---

## Totals

- libpurple `g_test` cases enumerated across the 7 scope files: **61**
  (CCD 7, CJD 3, conversation 23, account-setting 8, account-settings 12,
  presence 3, connection 5).
- Ported (P): **35** (CCD 7, CJD 3, conversation 13, account-setting 6,
  account-settings 10, presence 1).
- Extra derived/daemon-native cases (+): **6** (presence predicates/ordering ×5,
  disconnect-reason ×1).
- Skipped (S): **26**, each with a reason above — GObject property-bag/signal
  mechanics, or member/message/tag-plumbing owned by other packages, or the
  `federated` conversation flag absent from the wire DTO.

Rust test count in `details::tests`: **41** (35 P + 6 +).

## Helper APIs added (for later packages)

- `CreateConversationDetails::is_valid() -> Result<(), CreateConversationDetailsError>`
- `ChannelJoinDetails::merge(&mut self, source: &ChannelJoinDetails)`
- `ConversationType::{is_dm,is_group_dm,is_channel,is_thread,tag_value}` and
  `ConversationInfo::{is_dm,…,generate_title,title_for_display}`
- Free `title_for_display(alias, title, id)`
- Non-wire typed settings: `AccountSetting`, `AccountSettingStringList`,
  `AccountSettings` (+ `AccountSettings::to_values()` → wire `AccountSettingsValues`)
- `Presence::{is_available,is_online,is_idle,compare}`,
  `PresencePrimitive::is_online`, free `presence_compare(Option, Option)`
- `DisconnectReason::is_fatal()`
