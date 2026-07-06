# HARDENING-PLAN — Phase 5 support / node-side `allow_permanent` capability

Track: `hardening/allow-permanent` · worktree `/home/j/experiments/daemon-worktrees/allow-permanent`
Base: `hardening/integration` (contains ALL daemon-node hardening incl. the v28 wire bundle and the
Phase 2 exec-approval `CommandFingerprint`). daemon-node only. **v28 is NOT promoted yet** → extend the
v28 contract *in place* (no v29). Do NOT run `just update-codec` (coordinator re-regens the codec at the
superproject after merge).

> STATUS: PLAN ONLY — awaiting review. No source touched. Implement tests-first only after approval.

## 0. Goal

Give the node a real, least-privilege "Allow permanently" so the Phase 5 daemon-app render-honesty track
can stop treating the button as a cosmetic no-op. Semantics (the *secure* interpretation):

> "Allow permanently" adds the **approved command's `CommandFingerprint`** (the Phase 2 exec-approval
> hash of the fully-resolved `(surface, abs-binary, argv, env-delta, cwd)` tuple) to a **per-session
> allow-list**. That EXACT command auto-approves for the rest of the session. It is **not** a blanket
> approval-mode flip to auto-allow-everything (trust this one command, not all commands).

Fail-safe: when the resolved fingerprint is absent/unavailable (fs edits, `execute_code`, a vanished
binary, legacy snapshots), `allow_permanent` **degrades to a single allow** and remembers nothing — it
can never *broaden*. The node offers permanence only where a fingerprint exists to key the list on.

## 1. Two approval wire surfaces (both confirmed in `daemon-api.cddl`)

- **INLINE** (the surface the transcript "Allow permanently" button drives; live/synchronous HITL):
  `host-request-kind-approval = { "Approval": { "prompt": tstr } }` (cddl:149) →
  `host-response-body-approved = { "Approved": bool }` (cddl:159), consumed in
  `daemon-core::turn::ask_host`. The app's decision arrives node-side as
  `ApiRequest::Respond { response: HostResponse { body: Approved{..} } }` (wire.rs:45, cddl:1010) routed
  back to the parked `ask_host`.
- **DURABLE** inbox (headless/parked HITL):
  `request-approval-decide = { "ApprovalDecide": { session, request_id, allow: bool } }` (cddl:1047) →
  `ControlApi::approval_decide` → `SessionStore::answer_approval` (payload `b"allow"`/`b"deny"` into
  `completion_inbox`) → engine `resolve_approvals` (`decision.starts_with("allow")`). The app renders the
  inbox from `ApprovalInfo` (which already carries the v28 `fingerprint: Option<String>`).

Both approval gates funnel through `ask_host` (turn.rs:191): a live host answers `Approved`, a headless
host answers `Deferred(job)` (park), later resolved by `resolve_approvals` (engine.rs:1473).

## 2. Where things live (file:line inventory, current tree)

### Engine / gate (daemon-core)
- `turn.rs`
  - `TurnCx` (23–61) — ambient per-turn handles; `pre_approved: bool` (51) already means "skip the gate,
    run it". `child_for_call` (67–84) copies every field.
  - `Gate { Proceed | Reject(String) | Defer(JobId) }` (132–142).
  - `approve_path` (148–157, fs), `approve_command` (160–169, shell dangerous-argv + execute_code),
    `approve_shell_command` (177–186, shell background/pty — always-ask), all → `ask_host` (191–205).
  - `Effect` enum (95–128): `Persist | Delegate | Spawn | AwaitDecision`.
- `snapshot.rs`
  - `Snapshot` (…–86); `pending_approvals: Vec<PendingApproval>` (78–79); `fresh` (113–126).
  - `PendingApproval { job_id, call, prompt, path, #[serde(default)] fingerprint }` (90–109).
- `engine.rs`
  - `Engine { snapshot, .. }` (110–151). cx built for the react loop (989–1002, `pre_approved: false`).
  - Park stamp: `a.fingerprint = tool.resolved_fingerprint(&a.call, &cx)` (1071–1075), then suspend.
  - `resolve_approvals` (1473–1560): matches completion→`PendingApproval` by `job_id`;
    `allow = decision.starts_with("allow")` (1501); on allow builds cx `pre_approved:true` (1503–1516);
    **Cluster B fingerprint gate** verifies `resolved_fingerprint == approval.fingerprint` (1523–1534);
    on verify runs the tool, else refuses; `replace_awaiting_result` (1554).
- `engine/views.rs`
  - `PartitionedEffects { persists, delegated, spawns, awaiting }` (30–40) + `partition_tool_effects`
    (45–79) — **exhaustive** `match effect` (no `_` arm); AwaitDecision→PendingApproval (55–70).
- `tools.rs`
  - `Tool::resolved_fingerprint(&call, &cx) -> Option<CommandFingerprint>` default `None` (186–192);
    only `shell` overrides.
- `exec/mod.rs`
  - `CommandFingerprint(String)` (202) + `compute(surface, program_abs, argv, env_delta, cwd)` (209–246);
    `resolve_program_abs` (280–…). Re-exported from `lib.rs` (78).

### Shell tool
- `tools/daemon-tool-shell/src/lib.rs`
  - `run` (243–356): resolves the tuple (`resolve_exec`, 391–435), computes `resolved.fingerprint()`
    (317), builds `honest_prompt` (318), gates via `approve_shell_command` (background/pty, 320) or
    `approve_command` (dangerous foreground argv, 322); `Gate::{Proceed|Reject|Defer}` (324–349).
  - `resolved_fingerprint` override (358–382).

### Durable chain
- `daemon-api/src/wire.rs` — `ApiRequest::ApprovalDecide { session, request_id, allow }` (479–486);
  `Respond { session, response: HostResponse }` (45–50).
- `daemon-api/src/dispatch.rs` — destructures `ApprovalDecide` → `api.approval_decide(...)` (132–136).
- `daemon-api/src/lib.rs` — `ControlApi::approval_decide(session, request_id, allow)` default-Unsupported
  (536–543); `ApprovalInfo { …, #[serde(default)] fingerprint: Option<String> }` (1944–1961).
- `daemon-host/src/node_api/control.rs` — impl `approval_decide` (355–380) → `store.answer_approval` +
  `manager.wake`.
- `daemon-host/src/node_api/builtins.rs` — `/approve` builtin → `self.approval_decide(session, args, allow)`
  (189).
- `daemon-store/src/lib.rs` — `SessionStore::answer_approval(session, job_id, allow)` default (849–856);
  MemoryStore impl builds payload `b"allow"/b"deny"` (1599–1633).
- `daemon-store/src/sqlite.rs` — sqlite impl builds payload `b"allow"/b"deny"` (995–1045); `decision`
  column already has `fingerprint TEXT` sibling (schema.golden.sql:129).

### Wire types (daemon-protocol)
- `HostRequestKind::Approval { prompt }` (642–645) — construct: turn.rs:196, acp:645, process_agent test.
- `HostResponseBody::Approved(bool)` (716) — read only in `ask_host` (turn.rs:200–201); ~20 construct/
  match sites elsewhere are node-internal auto/deny hosts.

## 3. Wire changes (CDDL + Rust, all additive / `#[serde(default)]`)

All three are additive. v28-not-promoted lets us change `Approved`'s value shape in place.

### 3.1 INLINE offer — `host-request-kind-approval`
CDDL (cddl:149):
```
host-request-kind-approval = { "Approval": { "prompt": tstr, ? "allow_permanent_offered": bool } }
```
Rust (`daemon-protocol` `HostRequestKind::Approval`):
```rust
Approval {
    prompt: String,
    /// The node offers a durable per-session allow when it has a fingerprint to key the allow-list
    /// on (a command surface). The app renders "Allow permanently" ONLY when this is true.
    #[serde(default)]
    allow_permanent_offered: bool,
},
```

### 3.2 INLINE decision — `host-response-body-approved`
`Approved(bool)` is an externally-tagged newtype (`{"Approved": bool}`); serde cannot add a sibling key,
so promote it to a **struct variant** (the inner object gains the optional field — this is exactly
"add `? allow_permanent` to `host-response-body-approved`"):
CDDL (cddl:159):
```
host-response-body-approved = { "Approved": { "approved": bool, ? "allow_permanent": bool } }
```
Rust (`daemon-protocol` `HostResponseBody::Approved`):
```rust
Approved {
    approved: bool,
    /// The operator chose "Allow permanently": on the inline live path the engine remembers the
    /// approved command's fingerprint for the rest of the session. Honored only if the node offered
    /// it (a fingerprint exists); otherwise degrades to a single allow.
    #[serde(default)]
    allow_permanent: bool,
},
```
> Shape change (bare bool → 1-key map) is acceptable because v28 is unpromoted; `#[serde(default)]` keeps
> `{"Approved":{"approved":true}}` decodable. ~20 node-internal construct/match sites become
> `Approved { approved: X, allow_permanent: false }` / `Approved { approved: true, .. }` (mechanical;
> §6). The app-supplied `allow_permanent` rides in via the existing `Respond` decode.

### 3.3 DURABLE decision — `request-approval-decide`
CDDL (cddl:1047):
```
request-approval-decide = { "ApprovalDecide": { "session": session-id, "request_id": tstr, "allow": bool, ? "allow_permanent": bool } }
```
Rust (`daemon-api` `ApiRequest::ApprovalDecide`): add `#[serde(default)] allow_permanent: bool`.

### 3.4 DURABLE offer
No new field. The durable inbox app already keys the offer off `ApprovalInfo.fingerprint.is_some()`
(v28). Documented, not changed.

## 4. Data model — the per-session fingerprint allow-list

Add to `daemon-core` `Snapshot` (durable → survives restart, matching "rest of the session"):
```rust
/// Command fingerprints the operator approved **permanently** this session (Cluster B / allow_permanent):
/// an exact-tuple match auto-approves the gate without re-prompting. Least-privilege — this trusts the
/// specific resolved command, never a blanket approval-mode flip. Only ever grows within a session;
/// `#[serde(default)]` keeps pre-existing snapshots decodable (empty).
#[serde(default)]
pub session_allow_fingerprints: Vec<CommandFingerprint>,
```
Set to `Vec::new()` in `Snapshot::fresh`. `CommandFingerprint` already derives Ser/De + `Eq`. It stays
**engine-internal (snapshot only)** — it is never put on the wire (the wire carries only the two `bool`s;
the existing v28 `ApprovalInfo.fingerprint` display field is the only fingerprint the app ever sees).

Single source of truth = the snapshot. No mirror, no interior mutability.

## 5. Behavior

### 5.1 Short-circuit — wired into `ask_host` (covers BOTH surfaces)
`ask_host` short-circuit prevents *both* the inline prompt and the durable park (a park only happens if
we actually ask). To compare the current call's fingerprint it needs (a) the fingerprint and (b) a
read of the allow-list:
- Add a read-only view to `TurnCx`: `pub session_allow: &'a [CommandFingerprint]` (seeded per round from
  the snapshot — an owned clone, so no borrow conflict with `&mut self.snapshot`; `child_for_call` copies
  the slice ref). Tests pass `session_allow: &[]` (feature-off = never short-circuit = fail-safe).
- Thread the fingerprint into the gate → `ask_host`:
  - `approve_command(cx, prompt, fingerprint: Option<&CommandFingerprint>)`
  - `approve_shell_command(cx, prompt, fingerprint: Option<&CommandFingerprint>)`
  - `approve_path` unchanged (fs has no fingerprint → passes `None` internally → never offers/remembers).
- `ask_host(cx, prompt, fingerprint)`:

```rust
async fn ask_host(cx: &TurnCx<'_>, prompt: String, fingerprint: Option<&CommandFingerprint>) -> Gate {
    // Session allow-list short-circuit: an EXACT fingerprint match auto-approves without contacting the
    // host — this is what makes an identical in-session re-request skip the gate on BOTH the inline and
    // the durable surface (a park only happens if we ask).
    if let Some(fp) = fingerprint {
        if cx.session_allow.contains(fp) {
            return Gate::Proceed { permanent: false };
        }
    }
    let resp = cx.host.request(HostRequest {
        request_id: ReqId(0),
        // Offer permanence ONLY where a fingerprint exists to key the allow-list on.
        kind: HostRequestKind::Approval { prompt, allow_permanent_offered: fingerprint.is_some() },
    }).await;
    match resp.body {
        HostResponseBody::Approved { approved: true, allow_permanent } =>
            // Honor permanence only if we actually offered it (fingerprint present): defense in depth.
            Gate::Proceed { permanent: allow_permanent && fingerprint.is_some() },
        HostResponseBody::Approved { approved: false, .. } => Gate::Reject("denied by operator".into()),
        HostResponseBody::Deferred(job_id) => Gate::Defer(job_id),
        _ => Gate::Reject("approval not granted".into()),
    }
}
```

`Gate::Proceed` gains a `permanent: bool` (the one enum-shape change): `Gate::Proceed { permanent: bool }`.
fs/execute_code match `Gate::Proceed { .. } => …` (ignore it); only shell reads it.

### 5.2 Populate — inline (via a new `Effect`, applied by the single-owner applier)
`ask_host` cannot mutate the snapshot (`&TurnCx`). The shell tool surfaces the grant as an effect the
engine applies to the snapshot — keeping the snapshot the single source of truth (mirrors how
`AwaitDecision`/`Delegate`/`Spawn` already flow):
- New `Effect::RememberApproval(CommandFingerprint)` in `turn.rs`.
- Shell `run`: capture `permanent` from `Gate::Proceed { permanent }`; when true, append
  `Effect::RememberApproval(resolved.fingerprint())` to the returned `ToolOutcome.effects` (shell already
  holds `resolved.fingerprint()`).
- `partition_tool_effects` (views.rs): add a `remember: Vec<CommandFingerprint>` bucket + the new arm.
- Engine (right after `for turn in persists`, ~1050): dedup-insert each into
  `self.snapshot.session_allow_fingerprints`.

### 5.3 Populate — durable (in `resolve_approvals`)
- Carry permanence on the durable completion. `answer_approval(session, job_id, allow, allow_permanent)`
  encodes the payload:
  - deny → `b"deny"`; allow (single) → `b"allow"`; allow+permanent → `b"allow_permanent"`.
  - Backward compatible: `resolve_approvals` keeps `allow = decision.starts_with("allow")` and adds
    `let permanent = &*decision == "allow_permanent";`. The `decision` SQL column stays `allow as i64`
    (no schema migration — permanence rides only in the completion payload).
- In `resolve_approvals`, after the **verified** fingerprint re-run succeeds (`verified == true`), and only
  when `permanent && approval.fingerprint.is_some()`, dedup-insert `approval.fingerprint` into
  `self.snapshot.session_allow_fingerprints`. (`verified` is already computed against the freshly-resolved
  tuple, so we only ever remember a fingerprint that still resolves to the approved command.)

### 5.4 Fail-safe degrade (never broadens)
- No fingerprint (fs edit via `approve_path`; `execute_code` whose `resolved_fingerprint` is `None`; a
  binary that no longer resolves; a legacy snapshot approval with `fingerprint: None`) →
  `allow_permanent_offered = false`, permanence is dropped, **single allow only**, nothing remembered.
- Inline: `permanent = allow_permanent && fingerprint.is_some()` — a client that sets `allow_permanent`
  without an offer is ignored.
- Durable: populate is gated on `verified && approval.fingerprint.is_some()`; a mismatch already refuses
  the run entirely (Phase 2 gate), so a swapped command is never remembered.
- The builtin `/approve` (builtins.rs) and ACP/auto hosts pass `allow_permanent = false`.

### 5.5 Offer (item 3)
- Inline: `allow_permanent_offered = fingerprint.is_some()` in `ask_host` — true only for shell command
  surfaces (fs/`execute_code`/choice/input never offer).
- Durable: app derives the offer from `ApprovalInfo.fingerprint.is_some()` (existing v28, unchanged).

## 6. Churn inventory (all edit sites, for hunk-minimality review)

| File | Change |
|---|---|
| `crates/contracts/daemon-api/daemon-api.cddl` | 3 rules: cddl:149, :159, :1047 (§3) |
| `crates/contracts/daemon-protocol/src/lib.rs` | `HostRequestKind::Approval` +field; `HostResponseBody::Approved` tuple→struct variant |
| `crates/contracts/daemon-api/src/wire.rs` | `ApprovalDecide` +`#[serde(default)] allow_permanent` |
| `crates/contracts/daemon-api/src/dispatch.rs` | destructure + pass `allow_permanent` |
| `crates/contracts/daemon-api/src/lib.rs` | `ControlApi::approval_decide` +param (default impl) |
| `crates/substrate/daemon-host/src/node_api/control.rs` | impl +param → `answer_approval(.., allow_permanent)` |
| `crates/substrate/daemon-host/src/node_api/builtins.rs` | pass `false` (single-allow builtin) |
| `crates/substrate/daemon-store/src/lib.rs` | `answer_approval` trait default +param; MemoryStore impl +param + payload encode |
| `crates/substrate/daemon-store/src/sqlite.rs` | sqlite impl +param + payload encode |
| `crates/engine/daemon-core/src/snapshot.rs` | `Snapshot.session_allow_fingerprints` +field, +`fresh` |
| `crates/engine/daemon-core/src/turn.rs` | `TurnCx.session_allow`; `Gate::Proceed{permanent}`; gate fns +`fingerprint`; `ask_host` short-circuit+offer+decision; `Effect::RememberApproval`; `child_for_call` copy |
| `crates/engine/daemon-core/src/engine/views.rs` | `PartitionedEffects.remember` + partition arm |
| `crates/engine/daemon-core/src/engine.rs` | seed `session_allow` in both cx (989, 1503, owned clone/round); apply `remember` after partition; `resolve_approvals` parse+populate |
| `tools/daemon-tool-shell/src/lib.rs` | pass `Some(&fp)` to gates; capture `permanent`; emit `RememberApproval`; `Gate::Proceed{..}` |
| `tools/daemon-tool-fs/src/lib.rs` | `Gate::Proceed { .. }` (1 arm) |
| `tools/daemon-tool-execute-code/src/lib.rs` | `approve_command(cx, prompt, None)`; `Gate::Proceed { .. }` |
| ~24 `TurnCx { .. }` literals (mostly tests, cross-crate) | add `session_allow: &[]` |
| ~20 `HostResponseBody::Approved(..)` construct/match (node-internal) | `Approved { approved: X, allow_permanent: false }` / `Approved { approved: true, .. }` |
| ~4 `HostRequestKind::Approval { prompt }` construct/match | +`allow_permanent_offered: false` / `, ..` |
| `tests/daemon-conformance/src/approval.rs`, `ownership_matrix.rs` | +`allow_permanent` arg/field |
| `xtask/src/main.rs` (`gen_api_fixtures`) | add a `request-approval-decide.cbor` fixture w/ `allow_permanent` |

No changes to: `daemon-store` SQL schema (permanence rides in the completion payload), `ParkedApproval`,
`Effect::AwaitDecision`, `ApprovalInfo` (v28 fingerprint already present), the ManageResponseBody /
daemon-supervision enums (a different `Approved(bool)`).

## 7. Bug-reproducing tests (added FIRST)

Type-additive changes can't literally fail to compile pre-fix; for those I'll confirm teeth by stubbing
the new check to a no-op and observing the failure, then restoring (documented in the commit). I'll reuse
the `FingerprintProbeTool` + `drive_approval_resolution` harness (`engine/tests.rs:1247–1339`), extending
it to take a payload and expose `snapshot.session_allow_fingerprints`.

Engine (`engine/tests.rs`):
1. **Auto-approve identical in-session re-request.** Seed `snapshot.session_allow_fingerprints = [FP]`;
   run a turn whose gated tool resolves to `FP` against a **denying** host; assert the tool RAN (the
   host was never consulted — `ask_host` short-circuited). Teeth: clear the seed → the denying host
   refuses → tool does not run.
2. **A DIFFERENT command still prompts.** Seed `[FP_A]`; tool resolves `FP_B`; denying host → tool does
   NOT run (short-circuit is fingerprint-exact).
3. **Durable permanent populates the list.** `drive_approval_resolution(stored=FP, resolved=FP,
   payload=b"allow_permanent")` → tool runs (verified) AND `session_allow_fingerprints` contains `FP`;
   a follow-up gate for `FP` then short-circuits.
4. **`allow_permanent` with NO fingerprint = single allow only.** Parked approval with
   `fingerprint: None`, payload `b"allow_permanent"` → tool runs once BUT `session_allow_fingerprints`
   stays EMPTY (nothing to key on). Teeth: a second identical request still prompts.
5. **Durable permanent on a MISMATCH remembers nothing.** `stored=FP_A, resolved=FP_B,
   payload=b"allow_permanent"` → refused (Phase 2 gate), tool not run, list empty.

Turn/gate (`turn.rs` unit or shell):
6. **Offer flag only when a fingerprint exists.** A capturing host records
   `HostRequestKind::Approval.allow_permanent_offered`: a shell command gate (`Some(fp)`) → `true`; an
   `approve_path` (fs, `None`) gate → `false`.

Shell (`daemon-tool-shell`):
7. **Inline permanent emits `RememberApproval`.** A host answering `Approved { approved: true,
   allow_permanent: true }` for a background command → the outcome carries
   `Effect::RememberApproval(fp)` whose fp == `resolved_fingerprint`. And `allow_permanent: false` → no
   such effect.
8. **Seeded allow-list short-circuits the shell gate.** `cx.session_allow = &[fp_of_the_command]` +
   denying host → the command runs (gate skipped).

Store (`daemon-store` / conformance):
9. **`answer_approval` payload encodes permanence.** `(allow=true, permanent=true)` → completion payload
   `b"allow_permanent"`; `(true,false)` → `b"allow"`; `(false,_)` → `b"deny"`.

Snapshot (`snapshot.rs`):
10. **Round-trip.** `session_allow_fingerprints: [FP]` survives `encode`/`decode`; a pre-existing blob
    without the field decodes to empty (serde default).

Wire / CDDL (`daemon-api`, gated on `--features arbitrary`):
11. `ApiRequest::ApprovalDecide { .., allow_permanent: true }` ciborium round-trips and validates against
    `daemon-api.cddl` (proptest already covers ApprovalDecide + `Respond`→HostResponseBody +
    transcript-block-request→HostRequestKind); the new `request-approval-decide.cbor` fixture validates.

## 8. Fixture / CDDL / conformance

- Edit `daemon-api.cddl` (§3) so CDDL↔Rust agree.
- `cargo test -p daemon-api --features arbitrary` — the proptest reaches the new fields via `api-request`
  (`ApprovalDecide`, `Respond`→`host-response`) and `api-response` (`transcript-block-request`→
  `host-request-kind`). Any drift fails the build.
- `cargo run -p xtask -- api-fixtures` — regenerate; commit `request-approval-decide.cbor` (new) and any
  changed bytes. `cargo test -p daemon-api` (`fixtures_validate_against_cddl`) must stay green.
- **NOT** running `just update-codec` (coordinator does it at the superproject post-merge).

## 9. Honest residual coverage

- Same-absolute-path content swap (rewriting the approved binary's bytes) is NOT closed — the fingerprint
  pins the resolved absolute path + argv + env-delta + cwd + surface, not file contents (Phase 3 OS
  sandbox / artifact-provenance territory). A permanently-allowed command therefore trusts that exact
  resolved tuple for the session; a content swap is out of scope, consistent with Phase 2.
- Same-turn staleness: the per-round `session_allow` snapshot clone is taken before the round's batch, so
  a permanence granted *within* a round is honored from the *next* round on (persisted immediately to the
  snapshot; only the in-flight clone is stale). Acceptable; documented.
- The allow-list only grows within a session and is per-session (never global); it is cleared naturally
  when the session ends. No revocation UI this track (a follow-on could add `ApprovalForget`).
- fs edits / `execute_code` never gain permanence (no fingerprint) — deliberate least-privilege.

## 10. Gate commands (from worktree root, after approval + tests-first implementation)

```
nix develop --command cargo fmt --all -- --check
nix develop --command cargo clippy --workspace --all-targets -- -D warnings
nix develop --command cargo test --workspace --no-fail-fast
nix develop --command cargo test -p daemon-api --features arbitrary
nix develop --command cargo run -p xtask -- api-fixtures   # + commit fixture changes
```
Phase-4 disallow-lints stay active. Do NOT merge, remove the worktree, or run `update-codec`.

Known flakes to ignore (treat only new/different signatures as real): `detached_delegation` ×2,
`process_notify` store-seam; `bins/daemon/tests/host_launch.rs` under parallel load → re-run isolated.

## 11. Open questions for the reviewer

1. **`HostResponseBody::Approved` tuple→struct variant** (nested `{ "Approved": { "approved", ? "allow_permanent" } }`).
   This is the faithful reading of "add `? allow_permanent` to `host-response-body-approved`" given
   serde's externally-tagged encoding, and is legal because v28 is unpromoted — but it restructures the
   value shape and touches ~20 node-internal construct/match sites. Alternative (rejected): a new
   `ApprovedPermanent` union member (leaves `Approved(bool)` untouched, ~0 churn) — but that adds a
   *union member*, not "a field on `host-response-body-approved`", contradicting the stated CDDL edit.
   Confirm the struct-variant reading.
2. **Inline populate via `Effect::RememberApproval` + `Gate::Proceed { permanent }`** (snapshot stays the
   single source of truth) vs. an interior-mutable `TurnCx` allow-list handle that lets `ask_host` write
   directly (no `Effect`/`Gate` change, but introduces a snapshot/handle mirror to keep in sync). I chose
   the Effect design; confirm.
3. **Durable permanence payload = `b"allow_permanent"`** in the completion (keeps `starts_with("allow")`
   working, no `decision`-column/SQL migration). OK, or prefer a structured completion?
4. **Allow-list durability = on the `Snapshot`** (survives node restart / rehydrate, matching "rest of
   the session"). OK, or should permanence be in-memory only (reset on restart)?
5. **Fixture**: add a single `request-approval-decide.cbor` (with `allow_permanent`) for committed-fixture
   coverage of the new field, in addition to the proptest? (Recommended.)
