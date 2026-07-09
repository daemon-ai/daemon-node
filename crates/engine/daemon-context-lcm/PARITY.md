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
| `test_existing_compacted_session_restart_ignores_preserved_objective_anchor` (L1375) | gap-closed | `restart_ignores_preserved_objective_anchor` | red `b020357`, green `<anchor green commit>`; the anchor search now runs over the pre-drain conversation (Python `anchor_source_messages`) |
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
| `test_dynamic_leaf_chunk_sizing_runs_bounded_catchup_passes_when_pressure_remains_high` (L4144) | already-covered | `compaction.rs` multi-pass unit tests | the bounded catch-up loop is unit-covered; the single-pass engine path is exercised by the L4029 port |
| `test_cache_friendly_gating_suppresses_follow_on_condensation_for_single_fanin_group` (L4306) | gap-closed | `cache_friendly_gating_suppresses_single_fanin_group` | red `efe45ee`, green `<area2 green commit>`; node-level gating worked, the `condensation_suppressed_reason` status surface was missing |
| `test_critical_budget_pressure_bypasses_cache_friendly_single_group_suppression` (L4359) | gap-closed | `critical_pressure_bypasses_cache_friendly_suppression` | same pair |
| `test_cache_friendly_gating_allows_condensation_when_debt_reaches_two_groups` (L4411) | gap-closed | `cache_friendly_gating_allows_two_debt_groups` | same pair |

### Area 3 — deferred maintenance debt lifecycle

(pending)

### Area 4 — doctor/maintenance commands (`tests/test_lcm_command.py` L440–1091)

| Python test | status | Rust test | note |
|---|---|---|---|
| `/lcm doctor source` scan (L440) | gap-closed | `doctor_source_scans_legacy_blank_rows` | red `053b890`, green `179200d`; new `Store::source_normalization_plan` |
| `/lcm doctor source apply` (L451) | gap-closed | `doctor_source_apply_normalizes_legacy_blank_rows` | same pair; backup-first `Store::normalize_legacy_blank_sources`, no-op batch skips the backup |

## Out of scope (recorded per task brief)

- Packaging/install tests (`test_packaging_install.py`), benchmarking + stress CLI
  (`test_benchmarking_*.py`, `test_stress_release_check.py`), `import_lossless_claw`.
- Auxiliary child-session lineage and foreground-vs-cron side-channel session views —
  architectural divergence (no cron side-channel; one engine per session).
- Preset apply dry-run (wave-2 decision), host-capability probing.

## Remaining gap-open backlog

(summary appended at the end of the pass)
