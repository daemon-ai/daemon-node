# LCM parity audit — `daemon-context-lcm`

TDD-style parity port from the Python `hermes-lcm` plugin
(`/home/j/experiments/daemon-hermes/hermes-lcm`) into this Rust crate.

## Baseline

At branch base commit `a40caac` (tip of `prompt/integration`):

```
cargo test -p daemon-context-lcm
test result: ok. 183 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

No prior commits in this worktree (`a40caac..HEAD` empty), clean tree.

## Architecture adaptation notes

- **One engine per session.** The Python `LCMEngine` is a long-lived, multi-session
  singleton that reconciles the ingest cursor per `on_session_start`. The Rust
  `LcmContextEngine` is constructed per session (`open_for_session`); reconciliation
  runs once per incarnation on the first `ingest_current`.
- **System prompt is not a row.** Python treats `{"role":"system",...}` as an ordinary
  message and stores it. The Rust `Conversation` keeps the system prompt in
  `conv.system` (out of the turn stream), so Rust store rows never include a `system`
  row. Row-count / role-sequence assertions ported from Python are adjusted to the
  turn stream (`user`/`assistant`/`tool`).
- **Frontier vs tail-identity reconcile.** Python infers the ingest cursor by matching
  the replayed prefix against the durable store tail (`_reconcile_ingest_cursor_from_store`,
  never deleting durable rows). The Rust reconcile deletes the volatile tail
  (`store_id > frontier`) and re-ingests from turn 0 so `turn_store_ids` (the
  turn→store-row index compaction consumes) is rebuilt. This adaptation is correct for
  a full transcript replay and for a compacted-session restart, but it is **wrong for a
  delta-only replay** (see the reconcile gap rows below).

## Scope status table

Status legend: `ported-pass` (behavior already worked, test passes immediately) ·
`already-covered` (an existing Rust test already asserts it) · `gap-closed` (red then
green) · `gap-open` (documented red backlog) · `out-of-scope`.

### Area 1 — restart-reconciliation matrix (`tests/test_lcm_engine.py` L1264–2542)

Reconcile gap-closed cluster: red `3f0a48a`, green `20e38d6`.

| Python test | status | Rust test | note |
|---|---|---|---|
| `test_existing_session_restart_reconciles_cursor_before_ingest` (L1264) | ported-pass | `restart_full_transcript_replay_persists_only_new_tail` | frontier=0 delete-all + re-ingest of the full replay yields the same observable rows |
| `test_existing_session_restart_persists_delta_message_matching_store_tail` (L1688) | gap-closed | `restart_delta_matching_store_tail_is_preserved` | an ambiguous delta repeating the tail now appends instead of wiping the durable transcript |
| `test_existing_session_restart_persists_single_delta_message_matching_store_tail_with_followup` (L1762) | gap-closed | `restart_single_delta_matching_tail_with_followup_is_preserved` | ambiguous delta + follow-up appended |
| `test_existing_session_restart_does_not_skip_repeated_non_tail_messages` (L1553) | gap-closed | `restart_does_not_skip_repeated_non_tail_messages` | short LCM-scaffolded replay repeating an early pair is appended, not treated as replay/stale |
| `test_existing_session_restart_skips_stale_short_no_overlap_snapshot` (L2133) | gap-closed | `restart_skips_stale_short_no_overlap_snapshot` | stale head-prefix snapshot with a plain system prompt is skipped (system-anchor adaptation) |
| `test_existing_session_restart_persists_one_message_no_overlap_delta` (L2240) | gap-closed | `restart_persists_one_message_no_overlap_delta` | singleton no-overlap delta stays ambiguous and is appended |
| `test_existing_session_restart_scaffold_prefix_does_not_skip_unrelated_new_rows` (L2282) | gap-closed | `restart_scaffold_prefix_does_not_skip_unrelated_new_rows` | scaffold-only prefix skipped, new rows appended |

**Reconcile implementation** (`src/provider.rs`): `ingest_current` now routes the once-per-incarnation
reconcile three ways — fresh session (`session_count==0`, ingest from top); LCM-summary-scaffold-led
replay (`leading_scaffold_count>0`, the original delete-volatile-tail + re-ingest path, retained for
compaction restart); and any other restart, which runs `reconcile_turn_cursor` (a core turn-level
port of `_reconcile_ingest_cursor_from_store` / `_find_reconciled_cursor_for_store_tail`) and advances
the cursor past the proven replay prefix **without deleting durable rows**, then rebuilds
`turn_store_ids` from the durable tail so a later compaction still maps replayed turns to real rows.

| `test_existing_compacted_session_restart_skips_synthetic_context_but_persists_new_tool` (L1315) | ported-pass | `compacted_restart_skips_synthetic_context_but_persists_new_tool` | scaffold-led delete-path reconcile skips the synthetic context, persists the new tool turn |
| `test_existing_session_restart_reconciles_full_replay_without_system_prompt` (L1606) | ported-pass | `restart_full_replay_without_system_anchor_appends_only_new_row` | raw multi-row full replay accepted without a system anchor |
| `test_existing_session_restart_reconciles_complete_replay_without_system_prompt` (L1651) | ported-pass | `restart_complete_replay_without_new_rows_is_noop` | complete replay with nothing new is a no-op |
| `test_existing_session_restart_persists_scaffolded_delta_message_matching_store_tail` (L1800) | ported-pass | `restart_scaffolded_delta_matching_tail_is_preserved` | LCM-note system prompt + singleton tail-matching delta stays ambiguous, appended |
| `..._with_followup` (L1845) | ported-pass | `restart_scaffolded_delta_with_followup_is_preserved` | same + follow-up |
| `test_restart_reconciliation_filtered_singleton_tail_stays_ambiguous` (L2397) | ported-pass | `restart_filtered_singleton_tail_stays_ambiguous` | ignore-filtered durable rows leave a singleton visible tail; delta preserved |
| `test_existing_large_session_restart_reconciles_beyond_short_tail_window` (L1510) | ported-pass | `restart_large_session_reconciles_beyond_short_tail_window` | 5000-row session, full replay + new tool turn, no duplication |
| `test_existing_session_restart_persists_repeated_prefix_after_scaffold_only_prefix` (L2338) | ported-pass | `restart_repeated_head_after_scaffold_only_prefix_is_preserved` | scaffold-only prefix skipped, repeated durable-head pair appended |
| `test_existing_session_restart_persists_cleanup_sensitive_scaffolded_repeated_tail` (L1891) | ported-pass | `restart_literal_json_assistant_tail_delta_is_preserved` | literal-JSON assistant tail delta appended (Rust replay is not collapsed for this shape) |
| `test_existing_session_restart_skips_exact_lcm_system_scaffold` (L2095) | ported-pass | `restart_system_only_replay_leaves_store_untouched` | zero-turn replay (system prompt is off the turn stream) never ingests/deletes |
| `test_restart_reconciliation_filtered_prefix_does_not_create_stale_proof` (L2487) | gap-closed | `restart_filtered_prefix_does_not_create_stale_proof` | red `4ea0d51`, green `e588990`; stale proof now compares the RAW durable prefix |
| `test_existing_session_restart_skips_stale_short_snapshot_with_externalized_head_payload` (L2185) | gap-closed | `restart_stale_snapshot_with_externalized_head_payload_is_skipped` | same pair; stored-row identities restore §8.2 ingest spills (`restore_ingest_placeholders` accepts the Rust family kinds — Python writes the umbrella `ingest_payload` kind) |
| `test_existing_compacted_session_restart_ignores_preserved_objective_anchor` (L1375) | gap-closed | `restart_ignores_preserved_objective_anchor` | red `b020357`, green `dcf8bf5`; the anchor search now runs over the pre-drain conversation (Python `anchor_source_messages`) |
| `test_lcm_status_reports_ingest_reconciliation_diagnostics` (L2542) | gap-closed | `status_reports_ingest_reconciliation_diagnostics` | same pair; `_record_ingest_reconciliation` ported (turn-based counts), surfaced by `lcm_status` with the Python "not run" default |
| `..._cleanup_sensitive_scaffolded_repeated_tail_with_followup` (L1948) | already-covered | `restart_literal_json_assistant_tail_delta_is_preserved` + `restart_scaffolded_delta_with_followup_is_preserved` | the follow-up variant exercises the same ambiguous-delta append mechanics |
| `test_gateway_session_without_system_does_not_replay_old_first_user_as_anchor` (L1438) | already-covered | `compaction.rs::anchor_*` tests + `restart_ignores_preserved_objective_anchor` | anchor emission/skip logic unit-covered; the engine-level anchor path is now covered by the L1375 port |
| `test_existing_session_restart_persists_new_system_message` (L2007) / `..._that_mentions_lcm` (L2051) | out-of-scope | — | system messages are not rows in Rust (`Conversation.system` is out of the turn stream); nothing to persist or skip |
| `test_existing_session_restart_persists_prefix_repeated_without_system_anchor` (L2440) | out-of-scope | — | Python keys the stale-vs-delta distinction on whether the replay *includes a system row*; the Rust conversation always carries a system prompt, so the exact differentiator does not exist. The Rust adaptation keys on the plain-vs-LCM-note system prompt (see `restart_skips_stale_short_no_overlap_snapshot`), and a head-repeating multi-row batch under a plain prompt is treated as stale |
| rebind cleanup-equivalence family (L9896–L10360, outside the L1264–2542 block) | out-of-scope | — | models Python's active-context assistant cleanup (`_clean_active_assistant_message`) which the daemon does not perform on replay; no sanitized-collapse divergence exists to reconcile |

Reconcile gap-open rows (not attempted this pass) are grouped in the backlog section below.

### Area 2 — engine-level compaction behaviors

| Python test | status | Rust test | note |
|---|---|---|---|
| `test_dynamic_leaf_chunk_sizing_compacts_only_oldest_bounded_raw_chunk` (L4029) | ported-pass | `dynamic_leaf_chunk_compacts_only_oldest_bounded_chunk` | 1 node covering only the bounded oldest chunk; remainder kept raw; store intact |
| `test_adaptive_leaf_rescue_retries_with_smaller_oldest_chunk` (L4085) | ported-pass | `adaptive_leaf_rescue_retries_with_smaller_oldest_chunk` | retry-worthy aux failure shrinks to the oldest single turn (breaker threshold raised in-test — the Python mock has no breaker) |
| `test_unlimited_depth_condenses_beyond_ten` (L9029) | ported-pass | `unlimited_condensation_depth_reaches_d12` | `incremental_max_depth = -1` builds d12 from seeded d11 nodes through `compact` |
| `test_dynamic_leaf_chunk_sizing_runs_bounded_catchup_passes_when_pressure_remains_high` (L4144) | already-covered | `bounded_catchup_reduces_then_clears_debt` | the bounded multi-pass catch-up loop is exercised end-to-end by the Area-3 debt port (multiple passes per compact, node count grows per pass) |
| `test_dynamic_leaf_chunk_pressure_uses_current_working_window_after_each_pass` (L4185) | out-of-scope | — | pure monkeypatch instrumentation (records `_working_leaf_chunk_tokens` inputs per pass); the Rust loop recomputes `remaining_raw` per iteration by construction and offers no hook seam |
| `test_adaptive_leaf_rescue_stops_after_bounded_retry_worthy_failures` (L4230) | already-covered | `compaction.rs::rescue_ladder_shrinks_75_then_50_then_drop_last` + `adaptive_leaf_rescue_retries_with_smaller_oldest_chunk` | the 3-attempt bound and the L3-truncation terminal step are unit-covered; the engine path is covered by the rescue port |
| `test_cache_friendly_gating_does_not_block_forced_overflow_condensation` (L4463) | already-covered | `critical_pressure_bypasses_cache_friendly_suppression` + `CondensationGate` unit tests | `gate.allows` returns `Ok` for `force_overflow` by the same branch the critical bypass exercises (unit-tested in `compaction.rs`) |
| `test_cache_friendly_gating_suppresses_follow_on_condensation_for_single_fanin_group` (L4306) | gap-closed | `cache_friendly_gating_suppresses_single_fanin_group` | red `efe45ee`, green `7bc5b31`; node-level gating worked, the `condensation_suppressed_reason` status surface was missing |
| `test_critical_budget_pressure_bypasses_cache_friendly_single_group_suppression` (L4359) | gap-closed | `critical_pressure_bypasses_cache_friendly_suppression` | same pair |
| `test_cache_friendly_gating_allows_condensation_when_debt_reaches_two_groups` (L4411) | gap-closed | `cache_friendly_gating_allows_two_debt_groups` | same pair |

### Area 3 — deferred maintenance debt lifecycle (`tests/test_lcm_engine.py` L8840–8990)

| Python test | status | Rust test | note |
|---|---|---|---|
| `test_debt_persists_when_bounded_leaf_passes_leave_raw_backlog` (L8849) | ported-pass | `debt_persists_when_bounded_passes_leave_backlog` | bounded pass leaves backlog; lifecycle row records `raw_backlog` debt |
| `test_bounded_catchup_reduces_then_clears_debt_only_after_backlog_shrinks` (L8877) | ported-pass | `bounded_catchup_reduces_then_clears_debt` | catch-up passes reduce then clear (single fixed `max_passes=2` engine — Rust config is per-instance, Python monkeypatches it per call) |
| `test_status_and_lcm_status_surface_debt_state` (L8912) | ported-pass | `status_surfaces_debt_state` | `lcm_status.lifecycle.debt_kind` + config block |
| `test_critical_budget_pressure_drains_under_threshold_deferred_debt` (L8938) | ported-pass | `critical_pressure_drains_under_threshold_debt` | critical pressure bypasses the leaf floor and drains under-threshold debt |
| debt preflight advertising (`should_compress_preflight` asserts) | already-covered | `deferred_maintenance_debt_advertises_catchup_pressure_under_threshold` | pre-existing test |
| `test_critical_budget_pressure_continues_dynamic_catchup_after_first_pass` (L8969) | already-covered | `bounded_catchup_reduces_then_clears_debt` + `critical_pressure_drains_under_threshold_debt` | the multi-pass continuation + critical-drain mechanics are the same code paths |

### Area 4 — doctor/maintenance commands (`tests/test_lcm_command.py` L440–1091)

| Python test | status | Rust test | note |
|---|---|---|---|
| `/lcm doctor source` scan (L440) | gap-closed | `doctor_source_scans_legacy_blank_rows` | red `053b890`, green `179200d`; new `Store::source_normalization_plan` |
| `/lcm doctor source apply` (L451) | gap-closed | `doctor_source_apply_normalizes_legacy_blank_rows` | same pair; backup-first `Store::normalize_legacy_blank_sources`, no-op batch skips the backup |
| `test_lcm_doctor_retention_reports_old_heavy_sessions` (L744) | gap-closed | `doctor_retention_scopes_analysis_to_the_active_session` | red `d14302b`, green `61293ae`; active-session-scoped footprint/age analysis |
| `test_lcm_doctor_retention_counts_summary_only_sessions` (L786) | gap-closed | `doctor_retention_reports_nothing_without_active_session_rows` | same pair |
| `test_lcm_doctor_retention_keeps_stale_sessions_visible_when_list_is_truncated` (L816) | already-covered | `doctor_retention_reports_nothing_without_active_session_rows` | scoping makes the truncation case identical to the empty case (Python asserts the same "no stored sessions" output) |
| `test_lcm_doctor_clean_reports_pattern_matched_junk_candidates` (L840) | gap-closed | `doctor_clean_reports_pattern_matched_junk_candidates` | same pair; new `Store::session_footprints` scan |
| `test_lcm_doctor_clean_prefers_ignore_over_stateless_when_both_match` (L859) | gap-closed | `doctor_clean_prefers_ignored_class_over_stateless` | same pair |
| `test_lcm_doctor_clean_apply_is_backup_first_and_deletes_safe_candidates` (L910) | gap-closed | `doctor_clean_apply_backup_first_deletes_safe_candidates` | same pair; `Store::delete_sessions_atomically` (single-tx messages+nodes+lifecycle) |
| `test_lcm_doctor_clean_apply_denied_by_default` (L1026) | gap-closed | `doctor_clean_apply_denied_by_default` | same pair; gated on the pre-existing `doctor_clean_apply_enabled` config |
| `test_lcm_doctor_clean_lifecycle_reports_empty_candidates` (L1042) | gap-closed | `doctor_clean_lifecycle_reports_empty_rows` | same pair; `Store::empty_lifecycle_stats` |
| `test_lcm_doctor_clean_lifecycle_apply_is_backup_first_and_deletes_safe_candidates` (L1063) | gap-closed | `doctor_clean_lifecycle_apply_deletes_empty_rows` | same pair; operator apply prunes regardless of row age (the automatic bind-time GC keeps its age guard) |
| `test_lcm_doctor_clean_lifecycle_apply_denied_by_default` (L1091) | gap-closed | `doctor_clean_lifecycle_apply_denied_by_default` | same pair |
| `test_lcm_doctor_clean_returns_error_on_schema_problem` (L875) / `test_lcm_backup_returns_error_when_sqlite_backup_fails` (L897) | out-of-scope | — | driven by Python connection monkeypatching (`_FakeConn` / `sqlite3.connect` boom); the Rust store owns its connection and offers no equivalent fault-injection seam. Error branches exist and mirror command.py wording |
| `test_lcm_doctor_clean_apply_aborts_if_backup_fails` (L950) / `..._rolls_back_if_delete_fails_after_backup` (L976) | out-of-scope | — | same monkeypatch-driven fault injection; the Rust apply is backup-first and single-transaction (rollback on error) by construction |

## Out of scope (recorded per task brief)

- Packaging/install tests (`test_packaging_install.py`), benchmarking + stress CLI
  (`test_benchmarking_*.py`, `test_stress_release_check.py`), `import_lossless_claw`.
- Auxiliary child-session lineage and foreground-vs-cron side-channel session views —
  architectural divergence (no cron side-channel; one engine per session).
- Preset apply dry-run (wave-2 decision), host-capability probing.

## Wave-1 result summary

Final state: `cargo test -p daemon-context-lcm` → **225 passed, 0 failed** (baseline 183);
`cargo clippy -p daemon-context-lcm --all-targets -- -D warnings` clean; rustfmt clean.

Counts (wave-1 scope; 42 new tests total, 183 → 225):

- **ported-pass**: 18 tests (passed against the existing implementation immediately)
- **gap-closed**: 24 tests across 6 red/green commit pairs
- **already-covered**: 8 rows (existing Rust tests or the same code path)
- **out-of-scope**: 6 table rows (architectural divergence or Python monkeypatch seams, reasons
  inline) plus the deliberately deferred suites below
- **gap-open (red tests left in tree)**: 0 — every red test written this wave was closed green

Red/green pairs:

| behavior | red | green |
|---|---|---|
| restart reconcile preserves durable rows on non-full replay (6 tests) | `3f0a48a` | `20e38d6` |
| `/lcm doctor source [apply]` | `053b890` | `179200d` |
| stale proof uses raw prefix; stored identities restore ingest spills (2 tests) | `4ea0d51` | `e588990` |
| preserved-objective anchor pre-drain search; status reconcile record (2 tests) | `b020357` | `dcf8bf5` |
| `condensation_suppressed_reason` in `lcm_status` (3 tests) | `efe45ee` | `7bc5b31` |
| `/lcm doctor retention` + `clean [lifecycle] [apply]` (9 tests) | `d14302b` | `61293ae` |

## Wave-2 backlog (not attempted this pass — no red tests left behind)

Grouped by kind, each with an implementation sketch:

**Missing feature**

- *Reconcile sanitized-active-cleanup equivalence* (`_active_cleanup_replay_identity`,
  `has_raw_cleanup_replay`, `sanitized_tail_collapsed` — `LCM:engine.py:2793-2848/3020-3054`, tests
  at L9896–L10360): Python tolerates a host that strips/omits assistant rows from active replay
  when matching the durable tail. The daemon does not clean assistant rows out of replay, so no
  test in the L1264–2542 block needs it; if the daemon ever grows active-context cleanup, port
  `_clean_active_assistant_message` into `ReplayId` normalization (a second, "cleaned" identity per
  stored assistant row, matched as a fallback) and add the `fresh_tail_count`-gated
  `has_raw_cleanup_replay` proof.
- *Quarantined-assistant identity matching in reconcile* (`_is_quarantined_assistant_replay_identity`,
  `has_quarantined_singleton_replay`): needed only when a quarantined singleton is the whole
  session. Sketch: regex-match the quarantine placeholder shape in `ReplayId::content` (the Rust
  placeholder differs — `[Externalized quarantined assistant output: …]`) and accept a singleton
  full replay when both sides are quarantine placeholders.
- *`/lcm doctor clean` platform-key classification*: Python builds match keys with the session's
  platform (`platform:id`); the daemon has no platform notion, so `build_session_match_keys("", id)`
  is used. If platforms land in daemon-core, extend the scan keys.

**Suspected broken logic**

- none currently known — the wave-1 reds all closed, and the remaining Python-only branches above
  are unreachable through the daemon's replay surface today.

**Deliberately deferred (per task brief)**

- preset apply dry-run (wave-2 decision), host-capability probing, packaging/benchmarking/stress
  suites, `import_lossless_claw`, auxiliary child-session lineage and foreground-vs-cron
  side-channel session views.
