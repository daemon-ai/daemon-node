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
