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

| Python test | status | Rust test | note |
|---|---|---|---|
| `test_existing_session_restart_reconciles_cursor_before_ingest` (L1264) | ported-pass | `restart_full_transcript_replay_persists_only_new_tail` | frontier=0 delete-all + re-ingest of the full replay yields the same observable rows |

(more rows appended as work proceeds)

### Area 2 — engine-level compaction behaviors

(pending)

### Area 3 — deferred maintenance debt lifecycle

(pending)

### Area 4 — doctor/maintenance commands (`tests/test_lcm_command.py` L440–1091)

(pending)

## Out of scope (recorded per task brief)

- Packaging/install tests (`test_packaging_install.py`), benchmarking + stress CLI
  (`test_benchmarking_*.py`, `test_stress_release_check.py`), `import_lossless_claw`.
- Auxiliary child-session lineage and foreground-vs-cron side-channel session views —
  architectural divergence (no cron side-channel; one engine per session).
- Preset apply dry-run (wave-2 decision), host-capability probing.

## Remaining gap-open backlog

(summary appended at the end of the pass)
