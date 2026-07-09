# daemon-core parity test tracking

Ported test coverage from the Python **Hermes** agent
(`/home/j/experiments/daemon-hermes/hermes-agent`, tests under `tests/`) into
`daemon-core`. The purpose of these tests is to **uncover missing or broken logic**
in the Rust port. Tests named `parity_gap_*` assert the *desired* behavior per the
Python source and are **expected to fail** — a red `parity_gap_` test is a documented
gap, not a bug in the test. Production logic is never modified to make a test pass.

## Baseline (commit `a40caac`, tip of `prompt/integration`)

`nix develop --command cargo test -p daemon-core`:

```
test result: ok. 190 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Green baseline — no pre-existing failures.

## Status legend

- `ported-pass` — behavior already correct in Rust; test added and passes.
- `ported-fail (gap)` — desired behavior missing/broken; `parity_gap_` test fails.
- `already-covered` — an existing Rust test already covers this; not duplicated.
- `unportable-no-API` — no reachable runtime surface to exercise the behavior.
- `out-of-scope` — belongs to another crate / wave / subsystem.

---

## Wave 1 (P0)

### 1. Tool-argument JSON repair — `src/repair/tool_arg.rs`

Source: `tests/run_agent/test_repair_tool_call_arguments.py`;
impl `agent/message_sanitization.py:185` (`_repair_tool_call_arguments`).

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `test_empty_string_returns_empty_object` | ported-fail (gap) | `parity_gap_empty_string_returns_empty_object` | empty/whitespace should collapse to `{}`; Rust passes original through |
| `test_whitespace_only_returns_empty_object` | ported-fail (gap) | `parity_gap_whitespace_only_returns_empty_object` | same — whitespace-only → `{}` |
| `test_none_type_returns_empty_object` | unportable-no-API | — | `repair_tool_args` takes `&str`; a `None` value is not representable (covered in intent by empty-string gap) |
| `test_python_none_literal` | ported-fail (gap) | `parity_gap_python_none_literal_returns_empty_object` | Python literal `None` → `{}` not handled |
| `test_python_none_with_whitespace` | ported-fail (gap) | `parity_gap_python_none_with_whitespace_returns_empty_object` | `  None  ` → `{}` not handled |
| `test_trailing_comma_in_object` | already-covered | (existing `repairs_trailing_comma`) | trailing comma already stripped |
| `test_trailing_comma_in_array` | ported-pass | `trailing_comma_in_array` | array trailing comma handled |
| `test_multiple_trailing_commas` | ported-pass | `multiple_trailing_commas` | handled |
| `test_unclosed_brace` | already-covered | (existing `repairs_truncated_object`) | close-unclosed handled |
| `test_unclosed_bracket_and_brace` | ported-pass | `unclosed_bracket_and_brace_yields_valid_json` | handled |
| `test_extra_closing_brace` | ported-fail (gap) | `parity_gap_extra_closing_brace_is_trimmed` | excess `}}` not trimmed |
| `test_extra_closing_bracket` | ported-fail (gap) | `parity_gap_extra_closing_bracket_yields_valid_json` | excess `]]` not trimmed |
| `test_unrepairable_garbage_returns_empty_object` | ported-fail (gap) | `parity_gap_unrepairable_garbage_returns_empty_object` | unrepairable → `{}` fallback missing (Rust returns original) |
| `test_unrepairable_partial_returns_empty_object` | ported-fail (gap) | `parity_gap_unrepairable_partial_returns_empty_object` | truncated-mid-string → `{}`; Rust over-repairs by closing the string |
| `test_already_valid_json_passes_through` | already-covered | (existing `valid_json_is_canonicalized`) | passthrough handled |
| `test_trailing_comma_plus_unclosed_brace` | ported-pass | `trailing_comma_plus_unclosed_brace_yields_valid_json` | handled |
| `test_real_world_glm_truncation` | ported-fail (gap) | `parity_gap_glm_truncation_yields_valid_json` | truncation after `"key":` → `{}` fallback missing |
| `test_literal_newline_inside_string_value` | ported-fail (gap) | `parity_gap_literal_newline_inside_string_value` | lenient parse of raw control chars in strings missing |
| `test_literal_tab_inside_string_value` | ported-fail (gap) | `parity_gap_literal_tab_inside_string_value` | same — literal tab in string |
| `test_literal_control_char_reserialised_to_wire_form` | already-covered | (same behavior as newline/tab gaps) | not duplicated |
| `test_control_chars_with_trailing_comma` | ported-fail (gap) | `parity_gap_control_chars_with_trailing_comma` | control-char escape fallback after comma-strip missing |

### 2. Tool-name repair — `src/repair/tool_name.rs`

Source: `tests/run_agent/test_repair_tool_call_name.py`;
impl `AIAgent._repair_tool_call`. The Rust port handles case/separator/quote-strip/
namespace/fuzzy; it lacks CamelCase→snake_case, `_tool`/`Tool` suffix stripping, and
VolcEngine XML-attribute trimming. hermes returns `None` for an unresolved name; the
Rust port returns `Err` (treated as the parity equivalent).

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `test_lowercase_already_matches` | already-covered | (existing `exact_match_passes_through`) | exact match |
| `test_uppercase_simple` | ported-pass | `uppercase_simple` | case-fold handled |
| `test_dash_to_underscore` | already-covered | (existing `normalizes_case_prefix_and_separators`) | separator normalize |
| `test_space_to_underscore` | already-covered | (existing `normalizes_case_prefix_and_separators`) | separator normalize |
| `test_fuzzy_near_miss` | ported-pass | `fuzzy_near_miss` | fuzzy handled |
| `test_unknown_returns_none` | ported-pass | `unknown_returns_none` | rejects far name |
| `test_camel_case_no_suffix` | already-covered | (fuzzy already recovers `BrowserClick`) | not duplicated (passes via fuzzy) |
| `test_camel_case_with_underscore_tool_suffix` | ported-fail (gap) | `parity_gap_camel_case_with_underscore_tool_suffix` | `_tool` suffix not stripped |
| `test_camel_case_with_Tool_class_suffix` | ported-fail (gap) | `parity_gap_camel_case_with_tool_class_suffix` | `PatchTool` class suffix not stripped |
| `test_double_tacked_class_and_snake_suffix` | ported-fail (gap) | `parity_gap_double_tacked_class_and_snake_suffix` | `TodoTool_tool` double suffix |
| `test_simple_name_with_tool_suffix` | ported-fail (gap) | `parity_gap_simple_name_with_tool_suffix` | `Patch_tool` |
| `test_simple_name_with_dash_tool_suffix` | ported-fail (gap) | `parity_gap_simple_name_with_dash_tool_suffix` | `patch-tool` |
| `test_camel_case_preserves_multi_word_match` | ported-fail (gap) | `parity_gap_camel_case_preserves_multi_word_match` | `WriteFileTool` → `write_file` |
| `test_mixed_separators_and_suffix` | ported-fail (gap) | `parity_gap_mixed_separators_and_suffix` | `write-file_Tool` |
| `test_empty_string` | ported-pass | `empty_string_returns_none` | empty → None |
| `test_only_tool_suffix` | ported-pass | `only_tool_suffix_returns_none` | `_tool` → None |
| `test_none_passed_as_name` | unportable-no-API | — | `&str` param; `None` not representable |
| `test_very_long_name_does_not_match_by_accident` | ported-pass | `very_long_name_returns_none` | no accidental fuzzy match |
| `test_terminal_with_xml_attribute_pollution` | ported-fail (gap) | `parity_gap_terminal_with_xml_attribute_pollution` | XML attr pollution not trimmed |
| `test_execute_code_with_xml_attribute_pollution` | ported-fail (gap) | `parity_gap_execute_code_with_xml_attribute_pollution` | same |
| `test_session_search_with_xml_attribute_pollution` | already-covered | (same root cause as terminal/execute_code gaps) | not duplicated |
| `test_camel_case_tool_with_xml_pollution` | ported-fail (gap) | `parity_gap_camel_case_tool_with_xml_pollution` | XML + CamelCase suffix |
| `test_tool_name_with_trailing_quote_only` | ported-pass | `trailing_quote_only_is_trimmed` | surrounding-quote strip handles it |
| `test_tool_name_with_angle_bracket_pollution` | ported-fail (gap) | `parity_gap_tool_name_with_angle_bracket_pollution` | `<` pollution not trimmed |
| `test_tool_name_with_single_quote_pollution` | ported-fail (gap) | `parity_gap_tool_name_with_single_quote_pollution` | inner single-quote attr not trimmed |
| `test_clean_tool_name_unaffected_by_sanitizer` | already-covered | (existing `exact_match_passes_through`) | passthrough |
| `test_space_separated_name_still_normalizes` | already-covered | (existing separator test) | space normalize |
| `test_pollution_with_unknown_tool_root_still_fails` | ported-pass | `polluted_unknown_root_returns_none` | garbage → None |
| `test_leading_quote_falls_through_to_fuzzy_match` | ported-pass | `leading_and_trailing_quotes_resolve` | `"terminal"` → terminal |

---

## `parity_gap_` summary (the red list)

### Missing feature

**Tool-argument repair (`tool_arg.rs`)**
- `parity_gap_empty_string_returns_empty_object` — empty input → `{}`.
- `parity_gap_whitespace_only_returns_empty_object` — whitespace-only → `{}`.
- `parity_gap_python_none_literal_returns_empty_object` — Python `None` literal → `{}`.
- `parity_gap_python_none_with_whitespace_returns_empty_object` — ` None ` → `{}`.
- `parity_gap_extra_closing_brace_is_trimmed` — trim excess `}}`.
- `parity_gap_extra_closing_bracket_yields_valid_json` — trim excess `]]`.
- `parity_gap_unrepairable_garbage_returns_empty_object` — unrepairable → `{}` fallback.
- `parity_gap_glm_truncation_yields_valid_json` — truncation after `"key":` → `{}` fallback.
- `parity_gap_literal_newline_inside_string_value` — lenient parse of raw control chars.
- `parity_gap_literal_tab_inside_string_value` — literal tab in string.
- `parity_gap_control_chars_with_trailing_comma` — control-char escape fallback.

**Tool-name repair (`tool_name.rs`)** — CamelCase→snake_case + class-like suffix stripping
- `parity_gap_camel_case_with_underscore_tool_suffix`
- `parity_gap_camel_case_with_tool_class_suffix`
- `parity_gap_double_tacked_class_and_snake_suffix`
- `parity_gap_simple_name_with_tool_suffix`
- `parity_gap_simple_name_with_dash_tool_suffix`
- `parity_gap_camel_case_preserves_multi_word_match`
- `parity_gap_mixed_separators_and_suffix`

**Tool-name repair (`tool_name.rs`)** — VolcEngine XML-attribute pollution trimming
- `parity_gap_terminal_with_xml_attribute_pollution`
- `parity_gap_execute_code_with_xml_attribute_pollution`
- `parity_gap_camel_case_tool_with_xml_pollution`
- `parity_gap_tool_name_with_angle_bracket_pollution`
- `parity_gap_tool_name_with_single_quote_pollution`

### Suspected broken logic

- `parity_gap_unrepairable_partial_returns_empty_object` — Rust over-repairs a
  truncated-mid-string payload (closes the string) where hermes discards it as `{}`.

---

## Scope not yet reached

- P0.3 Tool-error sanitization (`tool_error.rs`)
- P0.4 Message-sequence repair (`message_sequence.rs`)
- P0.5 Engine end-to-end (dedup, guardrail escalation, 413/compaction loop, streaming)
