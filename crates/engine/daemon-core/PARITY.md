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
test result: ok. 252 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

All Wave 1 (P0) and Wave 2 (P1) gaps are closed (green). No `gap-open` (deliberately-left-red)
tests remain.

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

## Wave 2 (P1)

Priorities 1–7 from the wave-2 scope. Each pointer was verified against the actual Hermes source
before porting. Where a behavior turned out to live in the provider transport / CLI / node-config
layer rather than the engine (`daemon-core`), it is recorded `out-of-scope` with a pointer instead
of crossing the crate boundary (the superproject invariant: the node decides transport/config, the
engine drives the turn).

### 1. Provider fallback chains — `src/engine.rs` (`call_model` `RecoveryStep::Fallback`)

Source: `tests/run_agent/test_provider_fallback.py`; impl `run_agent.py:4243` (`_try_activate_fallback`
→ `agent/chat_completion_helpers.py:1045` `try_activate_fallback`).

The engine-side intent — *primary profile errors → hop to the fallback profile and continue the same
turn with the conversation intact; the fallback serves the final response* — is already implemented
(`call_model`'s `RecoveryStep::Fallback` swaps `self.profile` to `fallback_profile` with a fresh
retry budget) and covered by `fallback_profile_hops_credential_profile` (acquires primary, then
fallback; completes with the fallback's response; the user turn is untouched). The rest of the Python
file is provider-client resolution, not engine logic.

| Source test | Status | Reason |
|---|---|---|
| `TestFallbackChainAdvancement::*` (hop → continue → complete on fallback) | already-covered | `fallback_profile_hops_credential_profile` proves the hop + same-turn continuation + which profile served the final response |
| `TestFallbackChainInit::*` (list-vs-dict `fallback_model`, filter invalid entries) | out-of-scope | `_fallback_chain` config parsing lives in `run_agent.__init__`; the Rust engine models a single `EngineProfile::with_fallback_profile` (chain length = 1). Chain config is a node/config concern |
| `TestFallbackChainDedup::*` (skip self-matching provider/model/base_url) | out-of-scope | provider-client identity dedup (`resolve_provider_client`, base_url compare) is `daemon-providers`/config, not the profile-hop engine |
| `TestPoolRotationRoom::*` (`_pool_may_recover_from_rate_limit`) | out-of-scope | credential-pool rotation-room heuristic; the Rust `Recovery::Rotate`→`Recovery::Fallback` budget escalation (`recovery.rs::decide`) is the engine analog and is unit-tested (`decide_bounds_by_budget`) |
| `try_activate_fallback` client swap (api_mode detection, Anthropic/OpenAI client build, pool clear) | out-of-scope | transport-layer client construction (`daemon-providers`) |
| user-facing "trying fallback…" status | out-of-scope | Hermes buffers a CLI `vprint` status; the daemon-core wire signal is the completed `TurnFinished`, and the profile hop is silent by design (no per-hop `AgentEvent`) |

### 2. Steering — `src/engine.rs` (`boundary`, `push_steer_marker`)

Source: `tests/run_agent/test_steer.py`; impl `agent/agent_runtime_helpers.py`
(`apply_pending_steer_to_tool_results`) + `run_agent.py` (`steer` / `_drain_pending_steer` /
`clear_interrupt`).

The engine already drains queued steers at each phase boundary, appends an out-of-band `[steer]`
marker into the conversation, and acks each with `AgentEvent::Steered` (covered by
`steer_drained_appends_marker_and_acks`). The one behavior that was **missing**: an interrupt must
supersede a pending steer.

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `TestSteerClearedOnInterrupt::test_clear_interrupt_drops_pending_steer` | gap-closed (red `223a672`, green `0cad275`) | `interrupt_supersedes_pending_steer` | a steer queued while the turn is cancelling is dropped (not appended) and acked `accepted=false` |
| `TestSteerInjection::*` (append to last tool result), `TestPreApiCallSteerDrain::*` | ported-pass (adapted) | `steer_drained_appends_marker_and_acks` | Rust appends an out-of-band `[steer]` **user marker** at the boundary rather than mutating the last tool-result content — same intent (cache-safe out-of-band injection the model trusts), different carrier |
| `TestSteerAcceptance::*` (reject empty/whitespace/None, strip, concatenate) | out-of-scope | steer *text validation* (empty/whitespace rejection, `\n`-join of repeated steers) is the actor/control-layer submit path; the engine consumes already-validated `SteerReq`s off `TurnControl`. Repeated steers queue as separate markers (merged by the §-repair user-merge) rather than one `\n`-joined buffer |
| `TestSteerThreadSafety`, `TestSteerMarkerContract`, `TestSteerCommandRegistry` | out-of-scope | `_pending_steer_lock` threading, `STEER_MARKER`/system-prompt-note contract, and CLI `/steer` command registration are CLI/gateway concerns, not the engine turn loop |

Adaptation: Hermes stores one mutable `_pending_steer` string cleared by `clear_interrupt`; the Rust
engine keeps steers on the shared `TurnControl` queue and the *boundary* drops them when cancelling —
the same "interrupt beats a queued steer" invariant, expressed against the typed control surface.

### 3. Stream-interrupt retry — `src/engine.rs` (`call_model` recovery loop)

Source: `tests/run_agent/test_stream_interrupt_retry.py`; impl `run_agent.py`
(`_interruptible_streaming_api_call` — `_interrupt_requested` check at the top of the retry loop).

`drive_model_call` (recovery.rs) already aborts a *single* stream on cancel (biased `select!` on the
cancel token — `cancel_aborts_stream`). The gap was one level up: the §8 recovery **loop** in
`call_model` retried after a transient failure without re-checking cancellation, so a `/stop` that
arrived while a transient error was being handled still acquired a fresh credential and served out
the backoff before the next attempt observed the cancel.

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `TestStreamInterruptBeforeRetry::test_interrupt_prevents_stream_retry` | gap-closed (red `3ac2e6a`, green `d61ab4e`) | `interrupt_aborts_model_retry_loop` | a cancel observed between attempts aborts before the fresh credential acquire / provider re-invocation |
| `test_interrupt_before_first_attempt` | already-covered | (`interrupt_at_boundary_finalizes_interrupted`) | a pre-cancelled turn never reaches `call_model` (the opening `boundary` finalizes interrupted) |
| `test_normal_retry_still_works_without_interrupt` | already-covered | (`rate_limit_retries_with_backoff_then_completes`) | transient retries still succeed when not cancelled |
| partial-delta stream assembly (concatenating streamed tool-call deltas) | out-of-scope | (as recorded in Wave 1 §5d) the Rust `drive_model_call` consumes a fully-assembled `ModelOutput`; delta assembly is the `daemon-providers` transport layer |

The green commit also made the backoff sleep itself interruptible (`select!` on the cancel token),
so a long `Retry-After` wait aborts immediately rather than serving out the full delay — the faithful
port of Hermes' "exit immediately instead of retrying/serving the read-timeout."

### 4. Anthropic prompt-cache policy — `src/provider.rs` (`mark_cache_breakpoints`)

Source: `tests/run_agent/test_anthropic_prompt_cache_policy.py`; impl
`agent/prompt_caching.py:49` (`apply_anthropic_cache_control`) + `run_agent`
(`_anthropic_prompt_cache_policy`).

The **placement** policy (the daemon-core part) — hermes' `system_and_3`: up to 4 breakpoints,
tools+system prefix + last 3 messages, marked after the composed system is folded, all-to-messages
when the system is empty — is fully implemented in `mark_cache_breakpoints` and already covered.

| Source behavior | Status | Rust test | Reason |
|---|---|---|---|
| `system_and_3` placement: system + last 3 messages, 4-breakpoint budget | already-covered | `provider::cache_tests::{system_and_3_marks_system_plus_last_three, system_and_3_caps_at_four_breakpoints_total, empty_system_gives_all_four_slots_to_messages}` + engine `assembled_request_carries_post_fold_breakpoints_and_ttl` | the placement rules the audit lists |
| system-prompt byte-stability (breakpoints marked post-fold) | already-covered | engine `request_system_is_byte_stable_across_turns`, `assembled_request_carries_post_fold_breakpoints_and_ttl` | `mark_cache_breakpoints` runs after the composed system is assembled |
| `_anthropic_prompt_cache_policy` matrix — `(should_cache, use_native_layout)` by provider / base_url / api_mode / model (OpenRouter envelope vs native, MiniMax/Qwen/opencode allowlists, third-party gateways, OpenAI-wire blocklist, switch/fallback overrides) | out-of-scope | the *whether/which-layout* decision is provider-transport identity (`daemon-providers`): daemon-core places breakpoints; the networked provider decides if the wire honors them and in which layout |

### 5. Unicode / adversarial payload sanitization

Source (audit pointer `tests/test_message_sanitization.py` does not exist; the real files are)
`tests/run_agent/test_unicode_ascii_codec.py` + `tests/cli/test_surrogate_sanitization.py`; impl
`run_agent.py` (`_strip_non_ascii`, `_sanitize_messages_non_ascii`, `_sanitize_messages_surrogates`,
`_sanitize_structure_non_ascii`, `_sanitize_tools_non_ascii`).

Verified against source: this is **transport-encoding recovery**, not decode-boundary adversarial
sanitization. `_sanitize_messages_*` strip non-ASCII / replace surrogates on the request
payload+headers only when httpx raises `UnicodeEncodeError` on an ASCII locale (`LANG=C`, issue
#6843) — it runs in the request-send/retry path and mutates the OpenAI/Anthropic client's `api_key`,
`api_messages`, `api_kwargs`. There is no Hermes rule that strips zero-width/bidi/control characters
at a daemon-core decode boundary; the daemon-core repair modules (`repair/tool_arg.rs`,
`repair/tool_error.rs`, `repair/content.rs`) already handle the control-char/ANSI/`<think>` cases the
model output actually carries (Wave 1).

| Source behavior | Status | Reason |
|---|---|---|
| `_strip_non_ascii` / `_sanitize_messages_non_ascii` / `_sanitize_tools_non_ascii` / `_sanitize_structure_non_ascii` | out-of-scope | ASCII-locale `UnicodeEncodeError` recovery in the request-send path — `daemon-providers` transport, keyed on the SDK client + headers the engine never sees |
| `_sanitize_messages_surrogates` (`\ud800` → `\ufffd`) | out-of-scope | surrogate scrub on the outgoing payload, same transport-encoding path |
| control-char / ANSI / role-tag / `<think>` scrubbing on model output | already-covered | Wave 1 `repair/tool_arg.rs`, `repair/tool_error.rs`, `repair/content.rs` |

### 6. Budget-pressure notices — `src/engine.rs` (`finish_budget_exhausted`)

Source: `tests/run_agent/test_iteration_budget_race.py`; impl `run_agent.py` (`IterationBudget`,
`_budget_reminder_text`).

The engine-side intent — *the iteration budget is a hard stop, preceded by one final toolless
"grace" summary call, then the turn ends `BudgetExhausted`* — is implemented in
`finish_budget_exhausted` and covered.

| Source behavior | Status | Rust test | Reason |
|---|---|---|---|
| iteration budget hard stop + one grace (toolless summary) call | already-covered | `iteration_budget_exhaustion_ends_with_summary` | budget exhaustion runs the tool `max_iterations` times, then one summary round, then ends `BudgetExhausted` |
| `IterationBudget` consume/refund/remaining/used + thread safety | out-of-scope | Hermes' `IterationBudget` is a shared, lock-guarded counter across worker threads; the Rust engine uses a single-owner local `rounds_left: u32` in `run_turn` (no shared budget object, no lock — no data race to test) |
| user-visible `_budget_reminder_text` ("budget exhausted… send `continue`") | out-of-scope | CLI/gateway reminder prose; the daemon-core wire signal is the `TurnFinished { end_reason: BudgetExhausted }` the client renders |

### 7. Toolset composition

Source: `tests/test_toolsets.py`; impl `toolsets.py` (`resolve_toolset`, `resolve_multiple_toolsets`,
`get_toolset`, cycle detection, `create_custom_toolset`, `hermes-*` platform base toolsets).

`daemon-core`'s tool layer is a flat `ToolRegistry` (register-by-name, last-writer-wins, offered set
gated by `tool_search_threshold_bytes`). Toolset *composition* — named toolsets, `includes`
resolution with cycle detection, platform base-toolset inheritance, enable/disable lists, `all`/`*`
aliases, registry-snapshot membership — has no daemon-core surface: the node/config layer builds the
`ToolRegistry` an engine (or a constrained subsystem child) is constructed with. This matches
`EngineProfile::with_registry` ("constrain a background-review child to a skills-only / memory-only
toolset") — composition is decided outside the engine.

| Source test | Status | Reason |
|---|---|---|
| `TestResolveToolset` / `TestResolveMultipleToolsets` (includes, cycle detection, union/dedup, `all`/`*`) | out-of-scope | toolset graph resolution is node/config; the engine receives an already-resolved `ToolRegistry` |
| `TestGetToolset` / `TestRegistryOwnedToolsets` / `TestPluginToolsets` (live registry membership, plugin toolsets) | out-of-scope | `tools.registry` snapshot + plugin discovery is the host tool-plumbing layer |
| `TestToolsetConsistency::test_hermes_platforms_share_core_tools` (platform base toolset) | out-of-scope | `hermes-*` platform toolset definitions live in `toolsets.py`; the node picks the platform registry per surface |
| within-registry dedup (register same name twice → one tool) | already-covered | `tools.rs` `ToolRegistry` unit tests (register-by-name map) |

---

## Summary

### Wave 1 (P0)

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

### Wave 2 (P1)

- **gap-closed: 2 behaviors** — (2) interrupt supersedes a pending steer (`223a672`→`0cad275`);
  (3) interrupt aborts the model-call recovery loop (`3ac2e6a`→`d61ab4e`).
- **already-covered:** (1) fallback profile hop + same-turn continuation; (3) pre-cancel + normal
  retry; (4) `system_and_3` breakpoint placement + system byte-stability; (6) budget grace call +
  hard stop; (7) within-registry name dedup.
- **out-of-scope (provider transport / CLI / node-config layer):** (1) `_fallback_chain` config,
  skip-self dedup, `resolve_provider_client`, pool rotation-room, CLI fallback status; (2) steer
  text validation + `/steer` command registry + marker contract + thread-safety; (3) partial-delta
  stream assembly; (4) `_anthropic_prompt_cache_policy` should-cache/native-layout matrix;
  (5) ASCII-locale `UnicodeEncodeError` / surrogate payload sanitization (transport-encoding
  recovery, not a decode-boundary rule); (6) `IterationBudget` shared-counter thread-safety +
  `_budget_reminder_text`; (7) toolset graph resolution / includes / platform base toolsets.
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

Wave 2 introduced no regressions: the two engine changes (`boundary` steer-on-cancel;
`call_model` cancel-check + interruptible backoff) are cancel-path-only, so the existing 250 tests
stayed green throughout (250 → 251 → 252 as each gap closed).

### Scope not reached

None — all Wave 1 (P0) items 1–5 and Wave 2 (P1) items 1–7 were addressed (implemented, ported, or
explicitly recorded as already-covered / out-of-scope above).
