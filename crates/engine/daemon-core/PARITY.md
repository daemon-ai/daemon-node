# daemon-core parity test tracking

Ported test coverage from the Python **Hermes** agent
(`/home/j/experiments/daemon-hermes/hermes-agent`, tests under `tests/`) into
`daemon-core`. The purpose is to uncover missing or broken logic in the Rust port, document each
gap with a failing `parity_gap_` test (the red audit trail), then close it by porting the real
Hermes implementation (green). Ports of behavior that already works get normal names and pass
immediately.

## Baseline (commit `a40caac`, tip of `prompt/integration`)

`nix develop --command cargo test -p daemon-core`:

```
test result: ok. 190 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Green baseline — no pre-existing failures.

## Current state (this branch)

```
test result: ok. 250 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

All Wave 1 P0 gaps are closed (green). No `gap-open` (deliberately-left-red) tests remain.

## Status legend

- `ported-pass` — behavior already correct in Rust; test added and passes.
- `gap-closed` — desired behavior was missing/broken; a red `parity_gap_` test documented it, then
  the Hermes implementation was ported and the test renamed + made green. Cites red + green SHAs.
- `already-covered` — an existing Rust test already covers this; not duplicated.
- `gap-open` — a documented red backlog item with an implementation sketch (none remain).
- `unportable-no-API` — no reachable runtime surface to exercise the behavior.
- `out-of-scope` — belongs to another crate / wave / subsystem, or fights the Rust architecture.

---

## Wave 1 (P0)

### 1. Tool-argument JSON repair — `src/repair/tool_arg.rs`

Source: `tests/run_agent/test_repair_tool_call_arguments.py`;
impl `agent/message_sanitization.py:185` (`_repair_tool_call_arguments`).

`repair_tool_args` was rewritten as a faithful port of the Python pipeline (empty/whitespace/None →
`{}`; lenient control-char parse; trailing-comma strip; close unclosed; bounded excess-closer trim;
control-char escape fallback; unrepairable → `{}`). Red matrix: `0f564b6` (prior agent).
Green: `ba60bf3`.

Two pre-existing Rust unit tests that encoded the *old* passthrough/string-closing philosophy were
updated to Hermes semantics in the green commit (`truncated_mid_string_falls_back_to_empty_object`,
`unrepairable_falls_back_to_empty_object`). `guardrail::hash_canonical_json` was adjusted so distinct
non-JSON result bodies still hash distinctly now that repair no longer passes garbage through.

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `test_empty_string_returns_empty_object` | gap-closed (red `0f564b6`, green `ba60bf3`) | `empty_string_returns_empty_object` | empty → `{}` |
| `test_whitespace_only_returns_empty_object` | gap-closed (`0f564b6`→`ba60bf3`) | `whitespace_only_returns_empty_object` | whitespace-only → `{}` |
| `test_none_type_returns_empty_object` | unportable-no-API | — | `&str` param; a `None` value is not representable |
| `test_python_none_literal` | gap-closed (`0f564b6`→`ba60bf3`) | `python_none_literal_returns_empty_object` | Python `None` → `{}` |
| `test_python_none_with_whitespace` | gap-closed (`0f564b6`→`ba60bf3`) | `python_none_with_whitespace_returns_empty_object` | ` None ` → `{}` |
| `test_trailing_comma_in_object` | already-covered | (existing `repairs_trailing_comma`) | trailing comma already stripped |
| `test_trailing_comma_in_array` | ported-pass | `trailing_comma_in_array` | array trailing comma |
| `test_multiple_trailing_commas` | ported-pass | `multiple_trailing_commas` | handled |
| `test_unclosed_brace` | already-covered | (existing tests) | close-unclosed handled |
| `test_unclosed_bracket_and_brace` | ported-pass | `unclosed_bracket_and_brace_yields_valid_json` | handled |
| `test_extra_closing_brace` | gap-closed (`0f564b6`→`ba60bf3`) | `extra_closing_brace_is_trimmed` | excess `}}` trimmed |
| `test_extra_closing_bracket` | gap-closed (`0f564b6`→`ba60bf3`) | `extra_closing_bracket_yields_valid_json` | excess `]]` handled |
| `test_unrepairable_garbage_returns_empty_object` | gap-closed (`0f564b6`→`ba60bf3`) | `unrepairable_garbage_returns_empty_object` | unrepairable → `{}` |
| `test_unrepairable_partial_returns_empty_object` | gap-closed (`0f564b6`→`ba60bf3`) | `unrepairable_partial_returns_empty_object` | truncated-mid-string → `{}` (over-repair fixed) |
| `test_already_valid_json_passes_through` | already-covered | (existing `valid_json_is_canonicalized`) | passthrough |
| `test_trailing_comma_plus_unclosed_brace` | ported-pass | `trailing_comma_plus_unclosed_brace_yields_valid_json` | handled |
| `test_real_world_glm_truncation` | gap-closed (`0f564b6`→`ba60bf3`) | `glm_truncation_yields_valid_json` | truncation after `"key":` → `{}` |
| `test_literal_newline_inside_string_value` | gap-closed (`0f564b6`→`ba60bf3`) | `literal_newline_inside_string_value` | lenient parse of raw control chars |
| `test_literal_tab_inside_string_value` | gap-closed (`0f564b6`→`ba60bf3`) | `literal_tab_inside_string_value` | literal tab in string |
| `test_literal_control_char_reserialised_to_wire_form` | already-covered | (newline/tab gaps) | not duplicated |
| `test_control_chars_with_trailing_comma` | gap-closed (`0f564b6`→`ba60bf3`) | `control_chars_with_trailing_comma` | control-char escape fallback |

Adaptation: Python preserves object insertion order; the Rust port sorts object keys for
determinism. Rust keeps a Rust-only markdown-fence-strip pass (no Hermes equivalent) that the
existing `strips_markdown_fence` test relies on.

### 2. Tool-name repair — `src/repair/tool_name.rs`

Source: `tests/run_agent/test_repair_tool_call_name.py`;
impl `agent/agent_runtime_helpers.py:1925` (`repair_tool_call`). Added CamelCase→snake_case,
class-like `_tool`/`-tool`/`tool` suffix stripping (twice), and VolcEngine XML-attribute trimming
on top of the existing namespace/quote/case/separator normalization + fuzzy fallback. Red matrix:
`7e064b5` (prior agent). Green: `7316edee89b1be06466850aa0d0ca95f7164e07e`. (Hermes returns `None`;
the Rust port returns `Err`.)

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `test_lowercase_already_matches` | already-covered | (existing `exact_match_passes_through`) | exact match |
| `test_uppercase_simple` | ported-pass | `uppercase_simple` | case-fold |
| `test_dash_to_underscore` | already-covered | (existing separator test) | separator normalize |
| `test_space_to_underscore` | already-covered | (existing separator test) | separator normalize |
| `test_fuzzy_near_miss` | ported-pass | `fuzzy_near_miss` | fuzzy |
| `test_unknown_returns_none` | ported-pass | `unknown_returns_none` | rejects far name |
| `test_camel_case_no_suffix` | already-covered | (fuzzy/CamelCase) | not duplicated |
| `test_camel_case_with_underscore_tool_suffix` | gap-closed (red `7e064b5`, green `7316edee89b1be06466850aa0d0ca95f7164e07e`) | `camel_case_with_underscore_tool_suffix` | `_tool` suffix |
| `test_camel_case_with_Tool_class_suffix` | gap-closed (`7e064b5`→`7316edee89b1be06466850aa0d0ca95f7164e07e`) | `camel_case_with_tool_class_suffix` | `PatchTool` |
| `test_double_tacked_class_and_snake_suffix` | gap-closed (`7e064b5`→`7316edee89b1be06466850aa0d0ca95f7164e07e`) | `double_tacked_class_and_snake_suffix` | `TodoTool_tool` |
| `test_simple_name_with_tool_suffix` | gap-closed (`7e064b5`→`7316edee89b1be06466850aa0d0ca95f7164e07e`) | `simple_name_with_tool_suffix` | `Patch_tool` |
| `test_simple_name_with_dash_tool_suffix` | gap-closed (`7e064b5`→`7316edee89b1be06466850aa0d0ca95f7164e07e`) | `simple_name_with_dash_tool_suffix` | `patch-tool` |
| `test_camel_case_preserves_multi_word_match` | gap-closed (`7e064b5`→`7316edee89b1be06466850aa0d0ca95f7164e07e`) | `camel_case_preserves_multi_word_match` | `WriteFileTool` → `write_file` |
| `test_mixed_separators_and_suffix` | gap-closed (`7e064b5`→`7316edee89b1be06466850aa0d0ca95f7164e07e`) | `mixed_separators_and_suffix` | `write-file_Tool` |
| `test_empty_string` | ported-pass | `empty_string_returns_none` | empty → None |
| `test_only_tool_suffix` | ported-pass | `only_tool_suffix_returns_none` | `_tool` → None |
| `test_none_passed_as_name` | unportable-no-API | — | `&str` param; `None` not representable |
| `test_very_long_name_does_not_match_by_accident` | ported-pass | `very_long_name_returns_none` | no accidental match |
| `test_terminal_with_xml_attribute_pollution` | gap-closed (`7e064b5`→`7316edee89b1be06466850aa0d0ca95f7164e07e`) | `terminal_with_xml_attribute_pollution` | XML attr trimmed |
| `test_execute_code_with_xml_attribute_pollution` | gap-closed (`7e064b5`→`7316edee89b1be06466850aa0d0ca95f7164e07e`) | `execute_code_with_xml_attribute_pollution` | XML attr trimmed |
| `test_session_search_with_xml_attribute_pollution` | already-covered | (same root cause) | not duplicated |
| `test_camel_case_tool_with_xml_pollution` | gap-closed (`7e064b5`→`7316edee89b1be06466850aa0d0ca95f7164e07e`) | `camel_case_tool_with_xml_pollution` | XML + CamelCase |
| `test_tool_name_with_trailing_quote_only` | ported-pass | `trailing_quote_only_is_trimmed` | quote strip |
| `test_tool_name_with_angle_bracket_pollution` | gap-closed (`7e064b5`→`7316edee89b1be06466850aa0d0ca95f7164e07e`) | `tool_name_with_angle_bracket_pollution` | `<` trimmed |
| `test_tool_name_with_single_quote_pollution` | gap-closed (`7e064b5`→`7316edee89b1be06466850aa0d0ca95f7164e07e`) | `tool_name_with_single_quote_pollution` | inner `'` trimmed |
| `test_clean_tool_name_unaffected_by_sanitizer` | already-covered | (existing `exact_match_passes_through`) | passthrough |
| `test_space_separated_name_still_normalizes` | already-covered | (existing separator test) | space normalize |
| `test_pollution_with_unknown_tool_root_still_fails` | ported-pass | `polluted_unknown_root_returns_none` | garbage → None |
| `test_leading_quote_falls_through_to_fuzzy_match` | ported-pass | `leading_and_trailing_quotes_resolve` | `"terminal"` → terminal |

Adaptation: fuzzy match uses `strsim` (normalized Levenshtein ≥ 0.82) where Hermes uses
`difflib.get_close_matches` (cutoff 0.7); they agree on the accept/reject cases in the matrix.

### 3. Tool-error sanitization — `src/repair/tool_error.rs`

Source: `tests/test_sanitize_tool_error.py`; impl `model_tools.py:599` (`_sanitize_tool_error`).
Added whitelisted XML role-tag stripping (case-insensitive), CDATA stripping, and markdown
code-fence stripping (hand-rolled — no `regex` dependency) on top of the existing ANSI/control-byte
stripping. Red: `99156f6`. Green: `2831f20`.

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `test_strips_tool_call_tags` | gap-closed (red `99156f6`, green `2831f20`) | `strips_tool_call_tags` | `<tool_call>` stripped |
| `test_strips_function_call_tags` | gap-closed (`99156f6`→`2831f20`) | `strips_function_call_tags` | `<function_call>` stripped |
| `test_strips_role_tags` | gap-closed (`99156f6`→`2831f20`) | `strips_role_tags` | role tags stripped |
| `test_role_tag_strip_is_case_insensitive` | gap-closed (`99156f6`→`2831f20`) | `role_tag_strip_is_case_insensitive` | case-insensitive |
| `test_unrelated_xml_kept` | ported-pass | `unrelated_xml_kept` | non-role XML kept |
| `test_strips_cdata` | gap-closed (`99156f6`→`2831f20`) | `strips_cdata` | CDATA stripped |
| `test_strips_multiline_cdata` | gap-closed (`99156f6`→`2831f20`) | `strips_multiline_cdata` | multiline CDATA |
| `test_strips_leading_fence_with_lang` | gap-closed (`99156f6`→`2831f20`) | `strips_leading_fence_with_lang` | ` ```json ` fence |
| `test_strips_trailing_fence` | gap-closed (`99156f6`→`2831f20`) | `strips_trailing_fence` | trailing fence |
| `test_strips_bare_fence` | gap-closed (`99156f6`→`2831f20`) | `strips_bare_fence` | bare fence |
| `test_caps_long_input` | out-of-scope | — | Python `[TOOL_ERROR] `+2000 envelope; Rust caps at 4096 (see `bounds_long_errors`) |
| `test_does_not_truncate_short_input` | ported-pass | `does_not_truncate_short_input` | short passthrough |
| `test_wraps_with_prefix` | out-of-scope | — | Rust has no `[TOOL_ERROR] ` prefix (uses `wrap_untrusted_tool_result`) |
| `test_empty_input` | ported-pass (adapted) | `empty_input_returns_empty` | Rust: `""` (no prefix envelope) |
| `test_preserves_normal_error_text` | ported-pass | `preserves_normal_error_text` | normal text preserved |
| `TestHandleFunctionCallIntegration::*` | out-of-scope | — | dispatcher integration + `[TOOL_ERROR]` envelope |

Adaptation: Hermes wraps in a `[TOOL_ERROR] ` envelope capped at 2000 chars; the Rust port keeps its
own envelope (no prefix, 4096-byte cap, separate `wrap_untrusted_tool_result` fence), so parity is on
the *stripping* behavior, not the envelope.

### 4. Message-sequence repair — `src/repair/message_sequence.rs`

Source: `tests/run_agent/test_message_sequence_repair.py`;
impl `agent/agent_runtime_helpers.py:347` (`repair_message_sequence`). Added consecutive-user-message
merging (`\n\n`-joined, plain-text only) on top of the existing stray-tool drop / orphan back-fill /
empty-drop. Red: `6887b44`. Green: `f39ccfe`.

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `test_repair_merges_consecutive_user_messages` | gap-closed (red `6887b44`, green `f39ccfe`) | `merges_consecutive_user_messages` | consecutive users merged |
| `test_repair_preserves_user_content_when_one_side_empty` | ported-pass | `preserves_user_content_when_one_side_empty` | empty side dropped |
| `test_repair_preserves_multimodal_user_content` | ported-pass | `preserves_multimodal_user_content` | image user not merged |
| `test_repair_does_not_rewind_ongoing_dialog_tool_pair` | ported-pass | `does_not_rewind_ongoing_dialog_tool_pair` | valid tool+user kept |
| `test_repair_drops_stray_tool_with_unknown_tool_call_id` | ported-pass | `drops_stray_tool_with_unknown_tool_call_id` | orphan tool dropped |
| `test_repair_leaves_valid_conversation_unchanged` | ported-pass | `leaves_valid_conversation_unchanged` | no-op on valid |
| `test_repair_empty_messages_returns_zero` | ported-pass | `empty_messages_is_noop` | empty input |
| `test_repair_preserves_system_messages` | out-of-scope | — | system is a separate wire field in the Rust engine; leading non-`user` is trimmed by design |
| `_drop_trailing_empty_response_scaffolding` (`~L21`) | out-of-scope | — | needs a per-message `_empty_terminal_sentinel` flag the typed model does not carry; part of the run_agent empty-recovery loop |
| `repair_message_sequence_with_cursor` (`~L209`) | out-of-scope | — | operates on the session-DB flush cursor (`_last_flushed_db_idx`); persistence, not daemon-core |

Adaptation: Hermes represents multimodal user content as a *list* and skips merging a list side; the
Rust `RequestMsg` carries text in `content` + images in `images`, so it skips merging when either
side has non-empty `images`. The green commit updated the `empty_system_gives_all_four_slots_to_messages`
cache test fixture to alternate user/assistant, since `build_context` now merges consecutive users.

### 5. Engine end-to-end — `src/engine/tests.rs`

#### 5a. Deduplicate identical parallel tool calls

Source: `tests/run_agent/test_agent_guardrails.py::TestDeduplicateToolCalls`;
impl `run_agent.py:3395` (`_deduplicate_tool_calls`), wired at `agent/conversation_loop.py:3866`.
Added `deduplicate_tool_calls` at the decode/dispatch boundary in `engine.rs` (collapse duplicate
`(name, args)` calls within one assistant message, keeping the first). Red: `ec885bb`. Green: `3078303`.

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `test_duplicate_pair_deduplicated` (+ matrix) | gap-closed (red `ec885bb`, green `3078303`) | `deduplicates_identical_parallel_tool_calls` | identical parallel call runs once |

The green commit gave `parallel_tool_batch_runs_concurrently` distinct args (two identical calls
would now dedup to one).

#### 5b. Guardrail escalation (warn → block → halt) through `run_turn`

Source: `tests/run_agent/test_tool_call_guardrail_runtime.py`;
impl `agent/tool_guardrails.py` (Rust `src/guardrail.rs`, already wired into `execute_tool_batch`).
Behavior was already implemented; added two end-to-end `run_turn` tests. Ported-pass: `594ce38`.

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `test_config_enabled_hard_stop_run_conversation_returns_controlled_guardrail_halt_without_top_level_error` | ported-pass | `guardrail_hard_stop_halts_repeated_failure_through_run_turn` | hard-stop → `NoProgress`, not a top-level error |
| `test_default_run_conversation_warns_without_guardrail_halt` | ported-pass | `guardrail_warn_only_does_not_halt_through_run_turn` | warn-only runs to budget (`BudgetExhausted`) |
| per-call warn/block/halt matrix | already-covered | (`src/guardrail.rs` unit tests) | the escalation axes are exhaustively unit-tested |

#### 5c. Payload-too-large → compact → retry + infinite-compaction guard

Source: `tests/run_agent/test_413_compression.py`, `tests/run_agent/test_infinite_compaction_loop.py`.

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `test_413_triggers_compression` / context-overflow → compact → retry | already-covered | `context_overflow_compacts_then_retries` | 413/overflow compacts once then retries |
| infinite-compaction loop guard (`TestAntiThrashing`, `test_413_cannot_compress_further`) | already-covered | `unrecoverable_overflow_aborts` | compaction attempted at most once (`compacted` flag + `compact_context` returning false → abort); `unrecoverable_overflow_aborts` proves no infinite loop |
| hard-cap when engine compaction frees nothing | already-covered | `hard_cap_truncates_when_engine_compaction_frees_nothing` | C6 hard cap |
| anti-thrash (skip marginal compaction) | already-covered | `context::anti_thrash_skips_marginal_compaction` | context-engine anti-thrash guard |

The infinite-compaction loop guard and the 413/compact/retry path are already implemented and tested
in the Rust engine; no new work was required.

#### 5d. Streaming partial tool-call assembly/repair

Source: `tests/run_agent/test_streaming_tool_call_repair.py::TestStreamingAssemblyRepair`,
partial-tool sections of `tests/run_agent/test_streaming.py`.

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `TestStreamingAssemblyRepair::*` (arg repair on assembled streaming args) | already-covered | (`repair::tool_arg::parity::*`) | these Python tests call `_repair_tool_call_arguments` directly; the same function is now ported (section 1) and is applied to tool args in `tool_pipeline` |
| partial-delta assembly (concatenating streamed `tool_call` deltas) | out-of-scope | — | the Rust `drive_model_call` consumes a fully-assembled `ModelOutput` (`StreamEvent::Done`); assembling partial tool-call deltas is the provider transport layer (`daemon-providers`), not daemon-core |

---

## Summary

- **gap-closed: 5 behavior areas** (33 individual source-test rows) — tool-argument repair, tool-name
  repair, tool-error sanitization, message-sequence user-merge, tool-call dedup.
- **ported-pass:** tool-arg (5), tool-name (9), tool-error (2), message-sequence (6), guardrail
  end-to-end (2).
- **already-covered:** tool-arg (3), tool-name (5), 413/compaction (4), streaming arg repair (1),
  plus various existing separator/exact-match tests.
- **out-of-scope:** tool-error Python envelope + dispatcher integration (3), message-sequence system
  message / scaffolding stripper / flush-cursor (3), streaming partial-delta assembly (1).
- **unportable-no-API:** `None`-typed inputs for both repair functions (2).
- **gap-open: 0** — no deliberately-left-red tests remain.

### Regressions hit while implementing (all resolved)

- `guardrail::hash_canonical_json` relied on `repair_tool_args` passing non-JSON through unchanged;
  the new `{}` fallback broke distinct-result hashing. Fixed by hashing the raw string when the body
  is not JSON (in the tool-arg green commit `ba60bf3`).
- `provider::cache_tests::empty_system_gives_all_four_slots_to_messages` used 6 consecutive user
  messages; `build_context`'s new user-merge collapsed them. Fixed by alternating user/assistant in
  the fixture (message-sequence green commit `f39ccfe`).
- `parallel_tool_batch_runs_concurrently` used two identical `(para, {})` calls; the new tool-call
  dedup collapsed them. Fixed by giving them distinct args (dedup green commit `3078303`).

### Scope not reached

None — all Wave 1 (P0) items 1–5 were addressed (implemented, ported, or explicitly recorded as
already-covered / out-of-scope above).
