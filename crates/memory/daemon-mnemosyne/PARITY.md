# Mnemosyne parity ledger — wave 1 (P0)

Port of test coverage from the Python Mnemosyne project (`daemon-hermes/Mnemosyne`) into
`daemon-mnemosyne`, under the TDD-hybrid protocol: each behavioral gap got a red `parity_gap_`
test commit asserting the Python contract, then a green commit porting the Python implementation
and renaming the test.

## Baseline

At `a40caac` (tip of `prompt/integration`): `cargo test -p daemon-mnemosyne` (default features)
**green** — 236 lib tests + 4 integration tests (`tests/recall_modes.rs`), 0 failed.

End state after wave 1: **281 lib tests + 4 integration tests, 0 failed**; clippy
(`--all-targets -- -D warnings`) clean; no `gap-open` red tests remain.

## Statuses

- `ported-pass` — behavior was already correct in Rust; the ported test passed immediately.
- `already-covered` — an existing Rust test or a structural property covers the assertion intent.
- `gap-closed` — red `parity_gap_` commit + green implementation commit (SHAs listed).
- `gap-open` — deliberately-left red test (none in wave 1).
- `out-of-scope` — not portable/meaningful for the Rust architecture (reason given).

## Gap-closed summary

| Gap | Red | Green |
|---|---|---|
| Forget cascade must roll back atomically on failure | `3e89be3` | `6f004a8` |
| `mnemosyne_validate` must implement the provider attestation contract (attest/update/invalidate/delete, ring buffer, banks, errors) | `d202d48` | `d86a081` |
| `consolidate_fact` / `resolve_conflict` must serialize writers (BEGIN IMMEDIATE, first-writer-wins, fact-pair guard) | `4050b02` | `203953d` |
| `remember_batch` API + tool with per-row enrichment parity | `7f98ec1` | `0c06576` |

All gap tests were verified red before their green commit; the full suite was re-run green after
each green commit.

## Area 1 — forget / invalidate cascades (`tests/test_e6a_followup_gaps.py`)

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `TestForgetCascadeToAnnotations::test_forget_deletes_annotations_for_memory_id` (:47) | ported-pass | `engine::tests::forget_cascades_annotations_and_embeddings` | Rust `forget` already cascades annotations + embeddings. |
| `TestForgetCascadeToAnnotations::test_forget_doesnt_touch_other_memories_annotations` (:66) | ported-pass | `engine::tests::forget_leaves_other_memories_annotations_intact` | Cascade is scoped to the forgotten id. |
| `TestForgetCascadeToAnnotations::test_beam_forget_working_directly_cascades` (:82) | already-covered | `engine::tests::forget_cascades_annotations_and_embeddings` | `Engine::forget` IS `forget_working`; no separate facade layer in Rust. |
| `TestForgetCascadeToAnnotations::test_forget_after_export_leaves_no_leaked_annotations` (:94) | already-covered | `engine::tests::forget_cascades_annotations_and_embeddings` | Rust `export` carries no annotations payload; the leak intent is asserted directly on the table. |
| `TestForgetCrossSessionDoesNotLeakAnnotations::test_wrong_session_forget_does_not_touch_annotations` (:131) | ported-pass | `engine::tests::cross_session_forget_is_denied_and_preserves_annotations` | The session-scoped DELETE is the authorization boundary. |
| `TestForgetCrossSessionDoesNotLeakAnnotations::test_correct_session_forget_still_works` (:148) | ported-pass | `engine::tests::cross_session_forget_is_denied_and_preserves_annotations` | Same test, owning-session half. |
| `TestForgetCascadeIsAtomic::test_failed_cascade_rolls_back_working_memory_delete` (:178) | gap-closed (`3e89be3` → `6f004a8`) | `engine::tests::forget_cascade_failure_rolls_back_row_delete` | Rust ran the cascade in autocommit; the row delete survived a failed annotation delete. Fixed with a transaction mirroring Python's try/commit/rollback. |

## Area 2 — validation workflow (`tests/test_hermes_memory_provider_validation.py`)

All gap-closed rows: red `d202d48` → green `d86a081`. The green adds
`Engine::validate_action` (port of `_handle_validate`'s SQL, atomic in one transaction) and
rewrites the `mnemosyne_validate` dispatch (arg/action/bank validation, response shapes, surface
bank routing, `validate_<action>` audit events). `beam.py`'s own confirm/correct/reject
`Engine::validate` surface is untouched — Python has both layers too.

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `test_validator_columns_exist_after_init` (:78) | already-covered | `store::tests::schema_matches_golden` | Columns are pinned by the schema golden; every validate test reads them. |
| `test_memory_validations_table_exists` (:88) | already-covered | `store::tests::schema_matches_golden` | Same. |
| `test_trim_trigger_exists` (:96) | already-covered | `provider::tests::tool_validate_ring_buffer_keeps_last_three_while_count_grows` | The trigger is asserted behaviorally, not by name. |
| `test_validate_attest_preserves_author_and_records_validator` (:106) | gap-closed | `provider::tests::tool_validate_attest_records_validator_preserving_author` | Old dispatch had no attest action/response contract. |
| `test_validate_attest_falls_back_to_agent_identity` (:129) | gap-closed | `provider::tests::tool_validate_attest_falls_back_to_agent_identity` | Adaptation: the configured `author_id` (MNEMOSYNE_AUTHOR_ID) is the Rust analog of `_agent_identity`. |
| `test_validate_update_replaces_content_and_keeps_author` (:143) | gap-closed | `provider::tests::tool_validate_update_replaces_content_and_requires_new_content` | `update` previously did not replace content. Adaptation beyond Python: the content rewrite also drops the stale dense embedding (the crate's `Engine::update` invariant). |
| `test_validate_update_requires_new_content` (:161) | gap-closed | `provider::tests::tool_validate_update_replaces_content_and_requires_new_content` | Same test. |
| `test_validate_invalidate_sets_valid_until` (:175) | gap-closed | `provider::tests::tool_validate_invalidate_sets_valid_until` | `invalidate` previously unreachable through the tool's action vocabulary. |
| `test_validate_delete_removes_row` (:195) | gap-closed | `provider::tests::tool_validate_delete_removes_row` | `delete` previously unreachable. |
| `test_validate_works_on_shared_surface` (:211) | gap-closed | `provider::tests::tool_validate_works_on_shared_surface` | The tool previously had no `bank` routing. |
| `test_ring_buffer_keeps_only_last_three_validations` (:230) | gap-closed | `provider::tests::tool_validate_ring_buffer_keeps_last_three_while_count_grows` | Trigger existed but was unreachable through the tool contract. |
| `test_validation_count_grows_unbounded` (:247) | gap-closed | `provider::tests::tool_validate_ring_buffer_keeps_last_three_while_count_grows` | Same test. |
| `test_validate_unknown_memory_returns_error` (:265) | gap-closed | `provider::tests::tool_validate_rejects_bad_requests` | `memory_not_found` error shape. |
| `test_validate_unknown_action_rejected` (:274) | gap-closed | `provider::tests::tool_validate_rejects_bad_requests` | Old dispatch accepted any action string and appended a ring-buffer row for it. |
| `test_validate_unknown_bank_rejected` (:284) | gap-closed | `provider::tests::tool_validate_rejects_bad_requests` | Bank allowlist. |
| `test_validate_missing_memory_id_rejected` (:295) | gap-closed | `provider::tests::tool_validate_rejects_bad_requests` | Required-arg error shape. |
| `test_collaborative_attestation_chain` (:303) | gap-closed | `provider::tests::tool_validate_collaborative_attestation_chain` | Cross-agent author-preservation chain. |

## Area 3 — concurrent consolidation

Gap-closed rows: red `4050b02` → green `203953d`. The green adds
`veracity::serialized_write` (the `_serialized_write` port: BEGIN IMMEDIATE, participates in a
caller-owned transaction, rolls back on error), wraps `consolidate_fact` +
`run_consolidation_pass` in it, and hardens `Engine::resolve_conflict` with first-writer-wins +
a fact-pair membership guard.

### `tests/test_consolidate_fact_concurrency.py`

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `test_two_threads_same_spo_produce_one_row_count_2` (:85) | gap-closed | `knowledge::veracity::tests::concurrent_same_spo_yields_one_row_with_all_mentions` | Folded into the 8-thread variant. |
| `test_eight_threads_same_spo_produce_one_row_count_8` (:123) | gap-closed | `knowledge::veracity::tests::concurrent_same_spo_yields_one_row_with_all_mentions` | Pre-fix: 7/8 threads died on `UNIQUE constraint failed: consolidated_facts.id` — the exact silent-loss race. |
| `test_eight_threads_distinct_spos_produce_eight_rows` (:157) | ported-pass | `knowledge::veracity::tests::concurrent_distinct_spos_all_stored` | Distinct PKs + busy_timeout already survived contention. |
| `test_consolidate_fact_nested_in_outer_transaction` (:192) | gap-closed | `knowledge::veracity::tests::consolidate_fact_participates_in_outer_transaction` | The skip-BEGIN-when-nested behavior only exists post-fix. |
| `test_consolidate_fact_rolls_back_own_transaction_on_error` (:221) | already-covered | `knowledge::veracity::tests::serialized_write_owns_commit_and_rolls_back_on_error` | Python injects the failure via monkeypatch; Rust asserts the helper's rollback directly. |
| `test_concurrent_updates_compound_confidence_correctly` (:261) | gap-closed | `knowledge::veracity::tests::concurrent_updates_compound_confidence_and_count` | Pre-fix: read-compute-write lost updates (mention_count 2 of 5). |
| `TestReviewHardening::test_record_conflict_does_not_commit_when_nested` (:333) | already-covered | — | Conflict rows are inserted inside `consolidate_fact`'s serialized-write transaction; Python's `commit=` kwarg has no Rust analog. |
| `TestReviewHardening::test_record_conflict_default_still_commits` (:355) | out-of-scope | — | Python-API-specific back-compat kwarg. |
| `TestReviewHardening::test_begin_immediate_failure_raises_does_not_silently_proceed` (:370) | out-of-scope | — | No silent-fallthrough path exists: `Transaction::new_unchecked` errors propagate via `?` by construction. |
| `TestReviewHardening::test_consolidator_sets_wal_and_busy_timeout` (:399) | ported-pass | `store::tests::store_sets_wal_and_busy_timeout` | `Store::init` applies both pragmas. |
| `TestReviewHardening::test_race_window_widening_demonstrates_serialization` (:475) | out-of-scope | — | Requires monkeypatch fault injection; the barrier-synchronized thread tests cover the contention shape. |

### `tests/test_consolidate_fact_sibling_races.py`

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `test_concurrent_resolve_conflict_different_winners_deterministic` (:56) | gap-closed | `engine::tests::resolve_conflict_first_writer_wins` | Ported as a deterministic double-resolution (the Rust engine's `Mutex<Connection>` already serializes same-instance threads); pre-fix both facts ended superseded. |
| `test_resolve_conflict_happy_path_unchanged` (:102) | already-covered | `engine::tests::resolve_conflict_first_writer_wins` | The first resolution's supersede + stamp is asserted there. |
| `test_resolve_conflict_by_facts_happy_path_unchanged` (:133) | already-covered | — | Rust folds resolve-by-facts into `resolve_conflict(confirmed, winner, loser)`. |
| `test_concurrent_resolve_conflict_by_facts_idempotent` (:157) | already-covered | `engine::tests::resolve_conflict_first_writer_wins` | Repeat resolutions are no-ops under the first-writer-wins guard. |
| `test_run_consolidation_pass_resolves_obvious_conflicts` (:198) | already-covered | `knowledge::veracity::tests::consolidation_pass_supersedes_lower_confidence` | Pre-existing test; stays green under the serialized pass. |
| `test_run_consolidation_pass_nested_resolve_does_not_crash` (:224) | already-covered | `knowledge::veracity::tests::consolidation_pass_supersedes_lower_confidence` | The Rust pass performs its own UPDATE statements inside one `serialized_write` scope; no nested helper call exists. |
| `test_serialized_write_begins_immediate_when_not_in_tx` (:246) | ported-pass | `knowledge::veracity::tests::serialized_write_owns_commit_and_rolls_back_on_error` | Written with the green (helper did not exist before). |
| `test_serialized_write_rolls_back_on_exception` (:272) | ported-pass | `knowledge::veracity::tests::serialized_write_owns_commit_and_rolls_back_on_error` | Same test, error half. |
| `test_serialized_write_participates_in_outer_transaction` (:299) | ported-pass | `knowledge::veracity::tests::consolidate_fact_participates_in_outer_transaction` | Asserted through the public `consolidate_fact` surface. |
| `TestReviewHardening::test_consolidator_sets_wal_and_busy_timeout` (:342) | ported-pass | `store::tests::store_sets_wal_and_busy_timeout` | Shared with the concurrency file. |
| `TestReviewHardening::test_same_connection_writers_serialize_via_rlock` (:366) | already-covered | — | The store's `Mutex<Connection>` is the RLock analog: same-instance writers serialize by construction. |
| `TestReviewHardening::test_first_writer_wins_logs_warning` (:436) | already-covered | `engine::tests::resolve_conflict_first_writer_wins` | The no-op behavior is asserted; the `tracing::warn!` is emitted but log capture is not part of the crate's test harness. |
| `TestReviewHardening::test_helper_captures_conn_at_entry` (:471) | out-of-scope | — | Rust borrows preclude swapping the connection mid-scope. |
| `TestReviewHardening::test_consolidate_fact_same_connection_serializes_via_rlock` (:511) | already-covered | — | Same `Mutex<Connection>` argument as :366. |

### `tests/test_consolidate_fact_id_collision.py`

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `test_compute_fact_id_is_deterministic_for_same_spo` (:53) | already-covered | `knowledge::veracity::tests::fact_id_is_stable_and_prefixed` | Pre-existing. |
| `test_compute_fact_id_distinguishes_distinct_spos` (:61) | ported-pass | `knowledge::veracity::tests::fact_id_distinguishes_distinct_spos` | |
| `test_compute_fact_id_format_is_stable` (:70) | already-covered | `knowledge::veracity::tests::fact_id_is_stable_and_prefixed` | `cf_` + 24 chars pinned. |
| `test_compute_fact_id_long_content_does_not_collide` (:82) | ported-pass | `knowledge::veracity::tests::fact_id_long_content_does_not_collide` | |
| `test_compute_fact_id_separator_prevents_smuggling` (:104) | ported-pass | `knowledge::veracity::tests::fact_id_length_prefix_prevents_separator_smuggling` | |
| `test_consolidate_fact_stores_hash_based_id` (:123) | ported-pass | `knowledge::veracity::tests::consolidate_stores_the_computed_hash_id` | |
| `test_consolidate_fact_dedup_by_spo_still_works` (:137) | already-covered | `knowledge::veracity::tests::repeated_mention_bumps_confidence_and_count` | Pre-existing. |
| `test_consolidate_fact_distinct_long_content_both_stored` (:154) | ported-pass | `knowledge::veracity::tests::distinct_long_content_facts_both_stored` | |
| `test_mixed_format_db_dedup_still_finds_old_rows` (:185) | ported-pass | `knowledge::veracity::tests::legacy_format_rows_dedup_by_spo_and_keep_their_id` | |
| `test_polyphonic_fact_voice_id_matches_stored` (:227) | already-covered | — | `poly_fact_voice` maps the stored `fact.id` from `get_consolidated_facts`; no recompute path exists. |
| `test_resolve_conflict_with_compute_fact_id` (:258) | already-covered | `engine::tests::resolve_conflict_first_writer_wins` | `pending_conflicts` ids ARE the computed ids. |
| `TestReviewHardening::test_separator_smuggling_does_not_collide` (:292) | ported-pass | `knowledge::veracity::tests::fact_id_length_prefix_prevents_separator_smuggling` | |
| `TestReviewHardening::test_unicode_nfc_and_nfd_hash_identically` (:304) | ported-pass | `knowledge::veracity::tests::fact_id_nfc_and_nfd_hash_identically` | Fixture reworded (protégé) to satisfy the typos hook. |
| `TestReviewHardening::test_input_validation_rejects_empty_strings` (:318) | out-of-scope | — | Defensive validation with no reachable Rust caller: every `consolidate_fact` path filters empty SPO components first; changing `compute_fact_id`'s infallible signature is not warranted. |
| `TestReviewHardening::test_input_validation_rejects_non_string` (:328) | out-of-scope | — | The type system prevents non-`&str` inputs. |
| `TestReviewHardening::test_hash_uses_sha256_codebase_consistency` (:337) | ported-pass | `knowledge::veracity::tests::fact_id_pins_sha256_of_length_prefixed_nfc` | |
| `TestReviewHardening::test_fact_voice_uses_stored_id_for_legacy_rows` (:358) | already-covered | — | Same structural argument as :227. |
| `TestReviewHardening::test_resolve_conflict_rejects_ambiguous_winning_id` (:389) | gap-closed | `engine::tests::resolve_conflict_rejects_foreign_fact_ids` | Pre-fix a foreign winner id silently superseded the loser and stamped the conflict. |
| `TestReviewHardening::test_consolidated_fact_dataclass_carries_id` (:424) | already-covered | `knowledge::veracity::tests::consolidate_stores_the_computed_hash_id` | `ConsolidatedFact.id` is asserted against the stored row. |

## Area 4 — configurable scoring (`tests/test_configurable_scoring.py`)

The Rust port already had the full surface: `scoring::normalize_weights`, config-level
`recall_weights` defaults, and per-call `RecallFilters::{vec,fts,importance}_weight` overrides
(forwarded by the `mnemosyne_recall` tool). Every ported test passed immediately.

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `TestNormalizeWeights::*` (:38-:107, 10 tests) | already-covered | `recall::scoring::tests::weights_normalize_to_one` + `engine::tests::recall_weight_override_edge_cases_never_break_recall` | Pure-fn defaults/normalization pre-existing; zero-fallback + negative-clamp asserted end-to-end. Env-var layering lives in the node's figment config, not this crate. |
| `TestRecallConfigurableWeights::test_recall_accepts_weight_params` (:126) | already-covered | `engine::tests::recall_importance_weight_override_reorders_results` | Acceptance is implied by the ranking tests. |
| `test_recall_without_weight_params_is_backward_compatible` (:137) | already-covered | `engine::tests::remember_then_recall` | Pre-existing default-recall coverage. |
| `test_high_importance_weight_boosts_high_importance_memories` (:147) | ported-pass | `engine::tests::recall_importance_weight_override_reorders_results` | Overrides change end-to-end RANKING, not just normalization. |
| `test_results_include_score_breakdown` (:171) | ported-pass | `engine::tests::recall_weight_overrides_report_breakdown_and_compose_with_temporal` | |
| `test_env_vars_affect_scoring` (:185) | ported-pass | `engine::tests::configured_recall_weights_change_ranking_end_to_end` | Env vars map to injected `MnemosyneConfig::recall_weights`. |
| `test_explicit_params_override_env_in_recall` (:202) | ported-pass | `engine::tests::recall_explicit_weight_overrides_beat_configured_defaults` | Per-call overrides beat configured defaults. |
| `test_weight_params_dont_break_temporal_scoring` (:217) | ported-pass | `engine::tests::recall_weight_overrides_report_breakdown_and_compose_with_temporal` | |
| `test_zero_all_weights_uses_defaults_in_recall` (:227) | ported-pass | `engine::tests::recall_weight_override_edge_cases_never_break_recall` | |
| `TestPublicRecallConfigurableWeights::*` (:241, :257) | ported-pass | `provider::tests::tool_recall_forwards_weight_overrides` | The tool wire forwards all three overrides and the ranking flips. |
| `TestEdgeCases::test_very_high_vec_weight` (:290) | already-covered | `engine::tests::recall_weight_override_edge_cases_never_break_recall` | No-crash contract; vector-only weighting without embeddings degrades gracefully. |
| `TestEdgeCases::test_very_high_fts_weight` (:298) | ported-pass | `engine::tests::recall_weight_override_edge_cases_never_break_recall` | |
| `TestEdgeCases::test_invalid_negative_param_clamped` (:307) | ported-pass | `engine::tests::recall_weight_override_edge_cases_never_break_recall` | |

## Area 5 — remember_batch (`tests/test_e2_remember_batch_enrichment.py`)

The Rust engine had NO batch API. Gap-closed rows: red `7f98ec1` (through the tool-dispatch
surface — the only reachable runtime surface for a missing API) → green `0c06576`, which ports
`beam.py remember_batch` L3047-L3310 as `Engine::remember_batch` (`BatchItem` /
`RememberBatchArgs`) plus the `mnemosyne_remember_batch` tool. Python exposes batch ingest only
at the library level; the tool wrapper is the Rust node's dispatch analog, embedding per item and
running opt-in LLM extraction at the async seam. Noted adaptations: the per-row `MEMORY_ADDED`
event carries the row's real importance (Python hardcodes 0.5); batch rows keep Python's
column-default `global` scope and skip dedup, exactly as in Python.

| Source test | Status | Rust test | Reason |
|---|---|---|---|
| `test_remember_batch_writes_temporal_annotations_for_every_row` (:82) | gap-closed | `provider::tests::tool_remember_batch_enriches_every_row` + `engine::tests::remember_batch_enriches_every_row_with_annotations_gists_and_facts` | |
| `test_remember_batch_writes_has_source_when_source_is_non_default` (:101) | gap-closed | `provider::tests::tool_remember_batch_enriches_every_row` | Conversational sources skip `has_source`. |
| `test_remember_batch_extracts_gists_and_consolidated_facts` (:122) | gap-closed | `engine::tests::remember_batch_enriches_every_row_with_annotations_gists_and_facts` | |
| `test_per_row_veracity_threads_into_consolidated_facts` (:151) | gap-closed | `provider::tests::tool_remember_batch_enriches_every_row` + `engine::tests::remember_batch_threads_per_row_source_and_veracity` | stated vs inferred confidences must differ. |
| `test_per_row_source_flows_to_has_source_annotation` (:188) | gap-closed | `engine::tests::remember_batch_threads_per_row_source_and_veracity` | |
| `test_extract_entities_off_by_default` (:213) | gap-closed | `engine::tests::remember_batch_entity_extraction_is_opt_in` | |
| `test_extract_entities_true_populates_mentions` (:227) | gap-closed | `engine::tests::remember_batch_entity_extraction_is_opt_in` | |
| `test_extract_false_does_not_call_llm` (:244) | gap-closed | `provider::tests::tool_remember_batch_extract_flag_gates_llm_enrichment` | |
| `test_extract_true_calls_llm_fact_extractor_per_row` (:258) | gap-closed | `provider::tests::tool_remember_batch_extract_flag_gates_llm_enrichment` | Asserted by LLM-triple effect (MockProvider), not call counting. |
| `test_remember_batch_parity_with_remember_for_annotations` (:284) | gap-closed | `engine::tests::remember_batch_matches_single_remember_enrichment` | |
| `test_remember_batch_parity_with_remember_for_gists` (:314) | gap-closed | `engine::tests::remember_batch_matches_single_remember_enrichment` | |
| `test_enrichment_exception_does_not_break_batch` (:339) | gap-closed | `engine::tests::remember_batch_survives_per_row_enrichment_failure` | Failure injected by dropping the `gists` table instead of monkeypatching. |
| `TestReviewHardening::test_enrichment_loop_uses_single_deferred_commit` (:416) | out-of-scope | — | No commit-count seam in rusqlite; the port's bulk insert IS one transaction by construction. |
| `TestReviewHardening::test_extract_facts_caps_long_content` (:469) | ported-pass | `knowledge::episodic_graph::tests::extract_facts_truncates_pathological_long_content` | Structural (truncation window + fact cap) instead of a wall-clock assert. |
| `TestReviewHardening::test_remember_batch_emits_memory_added_event_per_row` (:493) | gap-closed | `engine::tests::remember_batch_emits_memory_added_event_per_row` | Stream + `memory_events` log both see batch rows. |
| `TestReviewHardening::test_meta_by_id_dict_survives_python_o` (:531) | gap-closed | `engine::tests::remember_batch_threads_per_row_source_and_veracity` | Per-row keying correctness. |
| `TestReviewHardening::test_deferred_commits_rollback_on_exception` (:560) | out-of-scope | — | `_deferred_commits` is Python-internal; the Rust analog (transaction rollback) is covered by `serialized_write_owns_commit_and_rolls_back_on_error` and the forget-cascade test. |
| `remember_batch` `force_veracity` / `trust_tier` kwargs (beam.py:3047-3080) | gap-closed | `engine::tests::remember_batch_force_veracity_and_imported_trust_tier` | Spec'd in the method contract rather than a dedicated Python test. |

## Gap-open items

None. Every wave-1 gap was closed red→green in this pass.

## Notes for wave 2

- Out of scope per the brief (untouched): CLI (`test_cli_*.py`), MCP server + SSE auth,
  importers, local-LLM sleep, LLM backends registry, stats dashboard, migration scripts,
  benchmark perf scripts, sync (already covered in Rust), Hermes-plugin lifecycle beyond
  `src/provider.rs`.
- `Engine::validate` (the `beam.py` confirm/correct/reject surface) remains untested by any
  wave-1 Python source file; it is now bypassed by the `mnemosyne_validate` tool (which ports
  the provider contract) but kept as public API.
- The typos pre-commit hook rejects the literal `café` NFC fixture; the NFC/NFD test uses
  `protégé` instead.

---

# Mnemosyne parity ledger — wave 2 (P1)

Port of the P1 themes. Same TDD-hybrid protocol as wave 1. Baseline at wave-1 tip `df65fd1`:
281 lib + 4 integration tests, green. End state: **320 lib + 1 (`multilingual_recall.rs`) + 4
(`recall_modes.rs`) integration tests, 0 failed**; clippy (`--all-targets -- -D warnings`) clean;
no `gap-open` red tests remain.

Statuses as in wave 1 (`ported-pass` / `already-covered` / `gap-closed` / `gap-open` /
`out-of-scope`).

## Gap-closed summary (wave 2)

| Gap | Red | Green |
|---|---|---|
| Polyphonic voice A/B toggles (`MNEMOSYNE_VOICE_{VECTOR,GRAPH,FACT,TEMPORAL}`) had no effect — voices ran unconditionally | `8b2e8d0` | `a499059` |
| Annotation store `export_all` / `import_all` round-trip (id-carrying, idempotent reimport) was missing | `855d067` | `9ca2543` |

## Per-theme ledger

### 1. Cyrillic / non-Latin recall E2E (`tests/test_cyrillic_fts.py`, `tests/test_multilingual_local_recall.py`)

The pure Cyrillic layer (`has_cyrillic`, `_ngrams`, `cyrillic_score`) and the engine's FTS→LIKE
fallback routing were already ported. Added the missing DB-routing + public-surface E2E coverage.
Note: single-token *inflection-only* recall does NOT surface through `recall()` in either Python or
Rust — the 1-token lexical gate (`min_relevance = 0.15`) drops it unless a dense vector carries it;
the Cyrillic fallback is a `_fts_search*` routing feature. The E2E test therefore uses exact-token
Cyrillic overlap; inflection routing is asserted at the `fts_search_working`/`_episodic` layer.

| Python source | Rust test | Status | Notes |
|---|---|---|---|
| `TestFtsSearchRoutesCyrillic::test_working_memory_fallback` | `engine::recall::tests::cyrillic_fts_working_routes_inflected_query_to_fallback` | ported-pass | `8b38eb5` |
| `TestFtsSearchRoutesCyrillic::test_episodic_fallback` | `engine::recall::tests::cyrillic_fts_episodic_routes_inflected_query_to_fallback` | ported-pass | |
| `TestCyrillicLikeSearchWorking::test_returns_empty_for_latin_query` | `engine::recall::tests::latin_query_does_not_engage_cyrillic_fallback` | ported-pass | |
| `test_cyrillic_fts.py` (E2E adaptation) | `tests/multilingual_recall.rs::cyrillic_query_recalls_matching_memory_end_to_end` | ported-pass | exact-token Cyrillic through the public `remember`/`recall` surface |
| `TestHasCyrillic`, `TestNgrams`, `TestCyrillicScore` | `recall::lexical::tests::cyrillic_trigram_scoring_matches_inflections` | already-covered | pure functions covered pre-wave-2 |

### 2. Cross-tier dedup / polyphonic (`tests/test_e3a3_cross_tier_dedup.py`)

`dedup_cross_tier_summary_links` + its wiring into base & polyphonic recall pre-existed. Ported the
helper's unit matrix (tie→episodic, per-cluster, order-preservation, empty summary_of). All pass.

| Python source | Rust test | Status | Notes |
|---|---|---|---|
| `TestDedupHelperUnit::test_no_episodic_rows_returns_input_unchanged` | `engine::recall::tests::dedup_no_episodic_rows_returns_input_unchanged` | ported-pass | `85c3e8c` |
| `::test_wm_wins_drops_episodic` | `::dedup_wm_wins_drops_episodic` | ported-pass | |
| `::test_episodic_wins_drops_wm` | `::dedup_episodic_wins_drops_wm` | ported-pass | |
| `::test_ties_keep_episodic` | `::dedup_ties_keep_episodic` | ported-pass | tie policy = keep episodic |
| `::test_summary_covers_multiple_wms_partial_overlap_per_cluster` | `::dedup_per_cluster_keeps_all_sources_when_summary_loses` | ported-pass | |
| `::test_summary_covers_multiple_wms_all_in_results` | `::dedup_summary_beats_all_sources_drops_them` | ported-pass | |
| `::test_preserves_order_on_retained_rows` | `::dedup_preserves_input_order_on_retained_rows` | ported-pass | |
| `::test_empty_summary_of_string_handled` | `::dedup_empty_summary_of_string_is_a_noop` | ported-pass | |
| `::test_only_one_side_in_results_keeps_it` | `::dedup_only_one_side_in_results_keeps_it` | ported-pass | |
| `TestLinearRecallPathIntegration::*` | `engine::tests::episodic_recall_after_consolidation_dedups_cross_tier` | already-covered | E2E dedup pre-existed |

### 3. Temporal boost E2E (`tests/test_temporal_recall.py`)

`temporal_boost`/`parse_query_time` + the recall `t_boost` plumbing pre-existed (tested only for
"composes/doesn't crash"). Added the missing E2E ranking assertions.

| Python source | Rust test | Status | Notes |
|---|---|---|---|
| `TestTemporalRecallEndToEnd::test_temporal_boost_recent_vs_old` | `engine::tests::temporal_boost_ranks_recent_over_old_end_to_end` | ported-pass | `d256981` |
| `TestTemporalRecallEndToEnd::test_temporal_halflife_override` | `engine::tests::temporal_halflife_override_changes_boost_end_to_end` | ported-pass | isolates the halflife knob |
| `TestTemporalBoostFunction::*`, `TestParseQueryTime::*` | (recall.rs `temporal_boost`/`parse_query_time` internal tests) | already-covered | pure helpers |

### 4. A/B toggle matrix (`tests/test_ab_toggles.py`)

Five scoring toggles (`veracity_multiplier`, `graph_bonus`, `fact_bonus`, `binary_bonus`,
`cross_tier_dedup`) were wired as `MnemosyneConfig` bools — ported behavioral tests, all pass. The
four polyphonic **voice** toggles were absent (voices ran unconditionally) — gap-closed by adding
`voice_{vector,graph,fact,temporal}` config flags + gating the voice calls in `recall_polyphonic`.
Env-var parsing (`_env_disabled`) is the node's figment layer, out of this crate.

| Python source | Rust test | Status | Notes |
|---|---|---|---|
| `TestLinearBonusToggles::test_graph_bonus_disabled_does_not_apply` | `engine::tests::graph_bonus_toggle_alters_episodic_recall_score` | ported-pass | `8ef116e` |
| `TestLinearBonusToggles::test_fact_bonus_disabled_does_not_apply` | `engine::tests::fact_bonus_toggle_alters_episodic_recall_score` | ported-pass | |
| `TestVeracityMultiplierToggle::test_disabled_*` + `test_enabled_*` | `engine::tests::veracity_multiplier_toggle_controls_stated_vs_unknown_ranking` | ported-pass | |
| `TestCrossTierDedupToggle::*` | `engine::tests::cross_tier_dedup_toggle_controls_summary_source_collapse` | ported-pass | |
| `TestPolyphonicVoiceToggles::test_vector_voice_disabled_returns_empty` | `engine::tests::voice_vector_toggle_gates_the_vector_voice` | gap-closed (`8b2e8d0`→`a499059`) | added `voice_vector` flag + gate |
| `TestPolyphonicVoiceToggles::test_temporal_voice_disabled_returns_empty` | `engine::tests::voice_temporal_toggle_gates_the_temporal_voice` | gap-closed (`8b2e8d0`→`a499059`) | `voice_graph`/`voice_fact` gated symmetrically |
| `TestLinearBonusToggles::test_binary_bonus_toggle_structural` | — | already-covered | `binary_bonus` gate consulted at `recall.rs:594` |
| `TestEnvDisabledHelper::*`, `TestToggleCoverageMap::*` | — | out-of-scope | env-var parsing lives in the node figment config, not this crate |

### 5. Unified private + surface recall (`tests/test_hermes_memory_provider_unified_recall.py`)

`mnemosyne_recall`'s surface merge (`shared_surface_read`, `bank` tags, top-k truncation, private
fallback) pre-existed in `tools.rs`. Ported the behavioral tests. All pass.

| Python source | Rust test | Status | Notes |
|---|---|---|---|
| `test_recall_default_returns_private_only` + `test_recall_default_tags_results_as_private` | `provider::tests::recall_default_returns_private_only_and_tags_private` | ported-pass | `34cfea2` |
| `test_recall_merges_results_from_both_banks` + `test_recall_tags_surface_results_with_bank_surface` | `provider::tests::recall_with_surface_read_merges_both_banks` | ported-pass | |
| `test_recall_truncates_to_top_k_after_merge` | `provider::tests::recall_truncates_to_top_k_after_merge` | ported-pass | |
| `test_recall_surface_init_failure_falls_back_to_private` | — | already-covered | the `if let Some(surface)` guard structurally guarantees the private fallback; the in-memory surface cannot be forced to fail as Python monkeypatches |
| `test_shared_surface_read_in_config_schema` / `_reads_from_config_yaml` / `_kwarg_overrides_config` | — | out-of-scope | provider `__init__.py` config-schema/YAML wiring is the node layer |

### 6. Session isolation, recall-side (`tests/test_hermes_memory_provider_thread_isolation.py`)

Engine-level isolation was covered by `session_scoping_over_shared_bank`; added the provider-tool
recall-surface variant (two providers over one on-disk bank, distinct session ids).

| Python source | Rust test | Status | Notes |
|---|---|---|---|
| `test_gateway_session_key_isolates_session_memories` | `provider::tests::recall_isolates_session_memories_across_providers` | ported-pass | `905818d` |
| `test_no_gateway_session_key_falls_back_to_session_id`, `_sanitized_*`, `_empty_*` | — | out-of-scope | `gateway_session_key`→`session_id` derivation is the node composition layer (`MnemosyneBanks`) |
| `test_prefetch_scopes_to_thread` | `engine::tests::session_scoping_over_shared_bank` | already-covered | same `(session_id = ? OR scope = 'global')` scope branch drives prefetch |

### 7. Identity inject / capture (`tests/test_prefetch_identity_always_inject.py`, `tests/test_identity_memory.py`)

Always-inject identity prefetch (`identity_rows` + `render_identity_block` dedup) and signal capture
(`capture_identity_signals`) pre-existed. Ported the E2E behaviors.

| Python source | Rust test | Status | Notes |
|---|---|---|---|
| `test_identity_surfaces_on_non_matching_generic_query` | `provider::tests::identity_always_injects_on_non_matching_generic_query` | ported-pass | `0f38a42` |
| `test_identity_does_not_leak_across_sessions` | `provider::tests::identity_does_not_leak_across_sessions` | ported-pass | |
| `test_no_identity_rows_is_a_noop` | `provider::tests::no_identity_rows_yields_no_identity_block` | ported-pass | |
| `test_identity_memory.py` (capture) | `provider::tests::identity_signal_capture_persists_and_injects` | ported-pass | signal-phrase capture → source='identity' → always-inject |
| `test_no_duplicate_when_query_matches_identity` | `recall::prefetch::tests` (render_identity_block dedup) | already-covered | dedup against the bank block is unit-tested |

### 8. Session-end drain (`tests/test_hermes_memory_provider.py::on_session_end`)

`on_session_switch(End|Handoff)` runs a forced sleep (drain WM→episodic). Added the drain-vs-no-drain
contrast test. The Python bounded-daemon-thread / join-timeout / shutdown-drain mechanics are
Python-threading specifics (the Rust engine is synchronous) — out-of-scope.

| Python source | Rust test | Status | Notes |
|---|---|---|---|
| `test_on_session_end_completes_when_sleep_is_fast` | `provider::tests::session_end_drains_pending_working_memory` | ported-pass | `e53ed18` (also asserts Start does NOT drain) |
| `session_end_promotes_working_memory_to_episodic` | (pre-existing) | already-covered | |
| `test_on_session_end_returns_within_timeout_*`, `_logs_warning_on_timeout`, `test_shutdown_drains_*` | — | out-of-scope | bounded daemon-thread / join-cap is Python threading; Rust `run_sleep` is synchronous |

### 9. Proactive content-similarity linking (`tests/test_proactive_linking.py`)

Fully implemented in `ingest.rs::proactively_link` (gated by `config.proactive_linking`): FTS content
similarity → `related_to`, entity co-occurrence → `references`, edge dedup. Ported the behavioral
matrix; all pass.

| Python source | Rust test | Status | Notes |
|---|---|---|---|
| `TestProactiveContentLinking::test_similar_content_creates_edges` + `test_self_not_linked` | `engine::tests::proactive_linking_links_similar_content_and_never_itself` | ported-pass | `9b63cd2` |
| `TestProactiveContentLinking::test_unrelated_content_no_edges` | `engine::tests::proactive_linking_skips_unrelated_content` | ported-pass | |
| `TestProactiveLinkingGating::test_disabled_by_default` | `engine::tests::proactive_linking_disabled_by_default_creates_no_cross_memory_edges` | ported-pass | |
| `TestEdgeDeduplication::test_repeat_remember_doesnt_duplicate_edges` | `engine::tests::proactive_linking_dedups_edges_on_repeat_remember` | ported-pass | |
| `TestEdgeTypesAndWeights::test_entity_edge_type` / `TestProactiveEntityLinking::test_shared_subject_creates_edge` | `engine::tests::proactive_linking_creates_references_edge_on_shared_entity` | ported-pass | |
| `TestNonBlocking::*` | — | already-covered | linking failures are `tracing::debug!`-swallowed (non-fatal) by construction |

### 10. Annotations export / canonical isolation (`tests/test_annotations.py`, `tests/test_canonical.py`)

Canonical **owner isolation** + upsert/supersede/forget pre-existed; annotation **multi-value
preservation** (append-only E6) pre-existed. The annotation store's **`export_all`/`import_all`** was
absent — gap-closed. The canonical store's `export_all`/`import_all`/`history`/`search`/`list` remain
out-of-scope (see below).

| Python source | Rust test | Status | Notes |
|---|---|---|---|
| `TestOwnerIsolation::test_same_slot_different_owners_coexist` + `test_list_is_owner_scoped` | `knowledge::canonical::tests::canonical_slots_are_owner_isolated` | ported-pass | `4c6a99c` |
| `TestUpsertSemantics::*` + `TestForget::*` | `knowledge::canonical::tests::versioned_remember_and_forget` | already-covered | created/unchanged/updated + version + forget + reborn |
| `TestAnnotationStoreMultiValuePreservation::test_multiple_mentions_for_one_memory_preserved` | `knowledge::annotations::tests::multiple_values_for_one_memory_kind_are_preserved` | ported-pass | |
| `::test_add_returns_row_id` | `knowledge::annotations::tests::add_returns_distinct_row_ids` | ported-pass | |
| `TestAnnotationStoreQueries::test_query_by_memory_with_kind_filter` | `knowledge::annotations::tests::query_by_memory_filters_by_kind` | ported-pass | |
| `TestAnnotationStoreExportImport::test_export_import_round_trip` + `test_import_idempotent_on_existing_ids` | `knowledge::annotations::tests::annotation_export_import_round_trips_and_is_idempotent` | gap-closed (`855d067`→`9ca2543`) | added `AnnotationExport` + `export_all`/`import_all` (id-carrying, `INSERT OR IGNORE` idempotent reimport) |
| `test_canonical.py` `CanonicalStore.export_all`/`import_all`/`history`/`search`/`list`; `TestProviderTools` search/history modes | — | out-of-scope | the Rust canonical port is deliberately the current-recall subset the provider tools use (`remember`/`forget`/`current`). Full history/search/list/export is a larger unbuilt subsystem; cross-store transfer in this crate flows through the event-based `src/sync/` layer, not per-store export. Recorded for a future wave. |

## Gap-open items (wave 2)

None. Both wave-2 gaps were closed red→green.

## Out-of-scope notes (wave 2)

- Env-var toggle parsing (`_env_disabled`, `TestToggleCoverageMap`), provider `config.yaml`/schema
  wiring, `gateway_session_key`→`session_id` derivation, and the bounded-daemon-thread
  session-end/shutdown mechanics all live in the node/figment/threading layers above this
  synchronous crate.
- Canonical store `history`/`search`/`list`/`export_all`/`import_all` (and the provider tool's
  search/history recall modes) are unbuilt in the Rust port — flagged above for a future wave rather
  than forced in as an oversized addition.
