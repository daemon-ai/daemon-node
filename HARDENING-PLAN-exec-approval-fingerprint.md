# HARDENING-PLAN — Phase 2 / Cluster B: exec approval fingerprinting

Track: `hardening/exec-approval-fingerprint` · worktree `/home/j/experiments/daemon-worktrees/exec-approval`
Base: `hardening/integration` (Wave 1 merged: egress client, EnvPolicy, symlink guards, auth-4 flip).

> STATUS: APPROVED with decisions (below). Implementing.

## 0. Approved decisions (final)

1. **`execute_code`: DEFER entirely this wave.** Do NOT touch `tools/daemon-tool-execute-code` at all
   (neither `python.rs` nor `lib.rs`). The sibling policy-partition owns `python.rs::resolve_interpreter`
   and lands first; `execute_code` fingerprinting is a tracked follow-on after it merges. My scope: the
   shell tool + the engine gate + the daemon-core abs-resolution helper.
   - *Consequence:* I must NOT add a field to the shared `Effect::AwaitDecision` variant (execute_code
     constructs it). Instead, the engine computes the approved fingerprint **at park time** by calling the
     new `Tool::resolved_fingerprint` on the parked tool — which returns `None` for execute_code (default),
     so execute_code is naturally un-fingerprinted with zero edits. `Effect::AwaitDecision` is unchanged.
2. **Hash the RESOLVED ABSOLUTE BINARY, not the raw `PATH` value.** The abs-binary is the ground truth of
   what `PATH` would have selected at approval time; hashing raw `PATH` would spuriously refuse on benign
   daemon-env changes. Keep the explicit **env-delta** (vars set for the command, e.g. `PYTHONUNBUFFERED`)
   in the hash — just never the ambient `PATH`. Final tuple: `(abs-binary, argv, env-delta, cwd, exec-surface)`.
3. **No structured `ApprovalInfo.fingerprint` wire field this wave.** Enforcement stays on the internal
   CBOR snapshot; embed a short digest in the existing `prompt` string. **Fix the lossy prompt:** enrich it
   to show the resolved absolute binary, argv, cwd, and exec-surface tier (plus the short digest). No
   `daemon-api.cddl` edit; still run `cargo test -p daemon-api --features arbitrary` to prove zero drift.
   A structured wire field is deferred to the Phase 5 app/codec bundle.
4. **Background `sh -c` = distinct ALWAYS-gated surface,** prompting even under `AutoAllow`/`AcceptEdits`
   (background shell is the persistence/exfil vector in this CVE class). Foreground benign argv behaves as
   today. Implemented via a new `approve_shell_command` gate (always asks unless hard `Deny` or a
   `pre_approved` re-run).

**Merge-order / conflict avoidance:** `resolve_program_abs` is a SELF-CONTAINED NEW function in
`daemon-core/src/exec/mod.rs` (purely additive — it does not modify `contain`, `Command`, or the
`ExecutionEnvironment` trait), so it merges cleanly beside policy-partition's trait-method additions.
policy-partition merges first; rebase onto integration if needed.

## 1. Goal (Cluster B scope)

1. **Bind the exec approval decision AND the operator-facing display to a hash of the fully-resolved
   command** — the tuple `(absolute-binary-path, argv, env-delta, cwd, exec-surface)`. Refuse to run if
   the tuple that will actually execute differs from what was approved/displayed. Closes the
   approve-then-swap TOCTOU on the durable HITL path.
2. **Resolve command binaries to ABSOLUTE paths** at approval time, and exec that exact absolute path,
   so PATH resolution at exec time cannot diverge from what was approved.
3. **Gate the background `sh -c` (shell-string) surface as a DISTINCT, higher-friction capability**
   separate from ordinary foreground argv exec.

## 2. Where approval + exec live today (file:line inventory)

### 2.1 The approval gate (generic §12 HITL plumbing) — `crates/engine/daemon-core/`

- `approval.rs`
  - `ApprovalPolicy` enum `Ask|AcceptEdits|AutoAllow|Deny` (lines 22–33).
  - `decide_command(self) -> Decision` (65–71): `Ask`/`AcceptEdits` → `Ask`; `AutoAllow` → `Allow`; `Deny` → `Deny`.
  - `decide_edit(self, path)` (49–61) and `is_sensitive_path` (85–100) — the fs-edit side, out of Cluster B scope.
- `turn.rs`
  - `TurnCx` fields `approval_policy` (47) and `pre_approved: bool` (48–51). `child_for_call` copies all fields (67–84).
  - `Effect::AwaitDecision { job_id, call, prompt, path }` (118–127) — the durable-suspend effect a tool returns.
  - `Gate { Proceed | Reject(String) | Defer(JobId) }` (132–142).
  - `approve_command(cx, prompt) -> Gate` (160–169) and `approve_path` (148–157): `pre_approved` short-circuits to `Proceed`; else `decide_command()` → `ask_host` (174–188) which raises `HostRequestKind::Approval { prompt }` and maps `Deferred(job_id)` → `Gate::Defer`.
- `snapshot.rs`
  - `PendingApproval { job_id, call: ToolCall, prompt: String, path: Option<String> }` (88–102), persisted on `Snapshot.pending_approvals` (78–79). Snapshot is CBOR (`encode`/`decode`, 121–147), tolerant via `#[serde(default)]`. **Not** a daemon-api wire type.
- `engine/views.rs`
  - `Effect::AwaitDecision { .. }` → `crate::snapshot::PendingApproval { job_id, call, prompt, path }` mapping (55–65) — the field-add site.
- `engine.rs`
  - Resume entry: `resolve_approvals(...)` called when `pending_approvals` non-empty on resume (847–848).
  - `resolve_approvals` (1462–1520): for each completion matching a `PendingApproval` by `job_id`, on `"allow"` it **re-runs `approval.call` verbatim** via `run_tool(&approval.call, &registry, &cx)` with `TurnCx.pre_approved = true` (1492–1507), then `replace_awaiting_result` (1514, 1522–1537). **This is the un-checked re-run — the TOCTOU site.**
  - `suspend_for_approval` (1254), `set_approval_policy` (221), `effective_policy` (579).
- `conversation.rs`: `ToolCall { call_id, name, args: String }` (49–56) — `args` is the JSON payload; program/argv/workdir live inside it.
- `tools.rs`: `Tool` trait (110–178) with default methods (`concurrency`, `mutates`, `call_timeout`, …). `ToolRegistry::get(name)` (247–252). `ToolOutcome::with_effects` (72–75).
- `lib.rs` exports: `approve_command, approve_path, Effect, Gate, TurnCx` (101); `ApprovalPolicy, Decision` (52).

### 2.2 The durable store mirror + operator display (wire path)

- `crates/substrate/daemon-host/src/engine_incarnation.rs` (527–543): on approval suspend, builds `ParkedApproval { session_id, job_id, epoch, prompt, path, decision:None }` from `engine.snapshot().pending_approvals`. Comment (524–526) confirms the snapshot keeps the typed `PendingApproval` (with the deferred `ToolCall`); the store rows are just the operator surface.
- `crates/substrate/daemon-store/src/lib.rs`: `ParkedApproval` struct (427–…, carries `prompt`/`path`/`decision`); `pending_approvals_of` (851). `sqlite.rs` (1041–…) has its SQL schema.
- `crates/substrate/daemon-host/src/node_api/control.rs`: `approvals_pending` maps `ParkedApproval → daemon_api::ApprovalInfo { session, request_id, prompt, path }` (280–307); `approval_decide` (309–…) → `store.answer_approval` + `manager.wake`.
- `crates/contracts/daemon-api/src/lib.rs`: `ApprovalInfo { session, request_id, prompt, path }` (1935–1945) — the wire mirror the GUI renders.
- `crates/contracts/daemon-api/src/wire.rs`: `ApiRequest::ApprovalsPending` (470), `ApprovalDecide` (479), `ApiResponse::Approvals(WirePage<ApprovalInfo>)` (971).

**Key consequence:** the enforcement copy of the fingerprint rides in the **snapshot** `PendingApproval`
(read by `resolve_approvals`), NOT in the store row. The operator **display** is the `prompt` string,
which already flows snapshot → `ParkedApproval` → `ApprovalInfo` → GUI unchanged. So enriching `prompt`
content + adding a snapshot-only fingerprint field needs **no daemon-store SQL migration and no
daemon-api/CDDL change** (see §6).

### 2.3 The exec sites (what actually runs)

- **`tools/daemon-tool-shell/src/lib.rs`** — the primary Cluster B target.
  - `HARDLINE` denylist (44–53); `is_hardline(line)` (113–115); `needs_approval(program, line)` (118–124) — heuristic (`sudo`/`dd`/`rm -rf|-fr|-r `).
  - `resolve_cwd(root, sticky, workdir)` (128–137) — contained absolute cwd (workdir contained; else sticky; else root).
  - `run` (161–255): joins `command + args` into `line` (172–176); Tier-1 hardline block (180–186); Tier-2 `needs_approval` → `approve_command(prompt="approve command: {line}")` (191–217) with `Gate::Defer` → `Effect::AwaitDecision { job_id, call, prompt, path:None }` (202–214). Then `cd` builtin (227–229), `resolve_cwd` (231), background branch (242–243), pty-requires-bg (245–252), else `run_foreground` (253).
  - `run_background` (307–374) → `procs.spawn(SpawnRequest{ line, cwd, pty, … })` (333–343).
  - `run_foreground` (378–466): builds `Command::new(parsed.command).args(parsed.args)` (405) + optional `.cwd` (407–409); runs via `cx.exec.run(cmd, &exec_cx)` (423).
- **`crates/engine/daemon-core/src/exec/local.rs`** — `LocalEnvironment::run` (46–108): `tokio::process::Command::new(&cmd.program)` (63) does **PATH resolution at exec time**; `env_clear()` + `.env("PATH", std::env::var_os("PATH"))` (67–68); `current_dir(&dir)` (63–66). This is where relative/PATH resolution can diverge from approval.
- **`crates/engine/daemon-core/src/exec/mod.rs`** — `Command { program, args, cwd }` (24–34), `contain` (118–145). Home for the new absolute-path resolver.
- **`crates/substrate/daemon-processes/src/registry.rs`** — the background shell-string surface.
  - `SpawnRequest { owner, line, cwd, pty, notify_on_complete, watch_patterns }` (139–153).
  - `spawn` (407–435) → `spawn_piped` / `spawn_pty`.
  - `spawn_piped` (437–503): `script = "exec 2>&1\nset +m\n{line}"` (447); `std::process::Command::new("sh").arg("-c").arg(script)` (448–458); `env_clear()` + `PATH` + `PYTHONUNBUFFERED=1` (453–455); `current_dir(cwd)` (452).
  - `spawn_pty` (505–…): `builder.arg(format!("set +m; {line}"))` (527) via a shell.
- **`tools/daemon-tool-execute-code/src/lib.rs`** — a SECOND gated exec site (see §7 coordination). Uses `approve_command` (229) + `Effect::AwaitDecision` (244–249); re-resolves interpreter + argv at exec time in `execute`/`run_staged` (263–341).
- **`tools/daemon-tool-fs/src/lint.rs`** (118): `cx.exec.run(cmd, …)` runs a linter — internal, not an agent-gated command; **out of scope** (noted for completeness).
- Construction sites (no signature change needed): `crates/node/daemon-node/src/profiles/{registry.rs:170, dress.rs:125}` build `ShellTool::with_processes`.

## 3. The fingerprint: definition, computation, check, display

### 3.1 Definition

A new newtype in `daemon-core` (`exec/mod.rs`, re-exported from `lib.rs`):

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandFingerprint(String); // lowercase hex sha256
```

Computed over a canonical, domain-separated encoding of the **resolved** tuple:

- `surface`: an enum tag — `"exec.argv"` (foreground argv) | `"exec.shell"` (background `sh -c`) | `"exec.pty"` (pty shell) — so an argv command and a shell-string with a coincidentally-equal textual line never share a fingerprint, and the *capability tier* is part of identity.
- `program_abs`: the resolved **canonicalized absolute** executable path (foreground). For the shell-string surfaces, the "program" is the resolved absolute `sh` plus the exact script string.
- `argv`: the ordered argument vector (foreground) or the exact `sh -c` script (background/pty).
- `env`: the child's explicit env-delta as a **sorted** `Vec<(String,String)>` (foreground: `PATH=<value>`; background: `PATH`,`PYTHONUNBUFFERED=1`) — the same values the exec backend applies.
- `cwd`: the resolved **absolute** working directory string.

Encoding: ciborium-encode a private `#[derive(Serialize)]` struct with fixed field order and the env
vec pre-sorted (deterministic), then `sha2::Sha256` → hex. (Both `ciborium` and `sha2` are already
available; see §5.)

### 3.2 Where computed / checked / displayed

- **Compute** in the tool that owns the resolution (`daemon-tool-shell`), via a shared helper so the
  live path, the defer effect, and the re-run recompute are byte-identical.
- **Display**: the tool builds an **honest prompt** from the resolved tuple — resolved absolute binary,
  argv (each arg on its own line / quoted so a space-in-arg is unambiguous), cwd, surface tier, and the
  short fingerprint digest. This replaces `format!("approve command: {line}")` (the lossy space-joined
  string). The prompt already flows to the operator (`ApprovalInfo.prompt`) unchanged, so the operator
  sees exactly the tuple that is hashed — the display is *bound to* the hash by construction.
- **Store (enforcement copy)**: add `fingerprint: Option<CommandFingerprint>` to
  `snapshot::PendingApproval` (`#[serde(default)]`). `Effect::AwaitDecision` is **unchanged** (§0.1); the
  engine populates the field **at park time** by calling `tool.resolved_fingerprint(&call, &cx)` on the
  parked tool right before extending `snapshot.pending_approvals` (engine.rs ~1064). `engine/views.rs`
  constructs the `PendingApproval` with `fingerprint: None`; the engine then fills it in for tools that
  return `Some` (shell). execute_code returns `None` → deferred, zero edits.
- **Check** on the durable re-run, in `engine.rs::resolve_approvals`, BEFORE re-running:
  when `approval.fingerprint.is_some()`, resolve the tool (`registry.get(&approval.call.name)`) and call
  a new default-`None` trait method `Tool::resolved_fingerprint(&call, &cx).await`; compare:
  - `Some(actual)` == expected → run as today (`pre_approved = true`).
  - `Some(actual)` != expected → **refuse**: splice a denial result (`"refused: the command to run no longer matches what was approved"`), do NOT run.
  - `None` (tool cannot resolve now, e.g. binary vanished) → **refuse** (fail-closed).
  - `approval.fingerprint == None` (legacy snapshot / non-fingerprinted tool like fs-edit) → run as before (back-compat).
- **Live (inline) path**: approval and exec happen in the same `run()` call — the tool resolves the tuple
  once, displays it, and on `Gate::Proceed` execs that exact resolved absolute path. No cross-call window;
  the guarantee is "what was displayed is what runs" by construction. (Same-abs-path content swaps within
  the single call are a file-level TOCTOU — see residuals §8.)

### 3.3 Why a `Tool` trait method instead of threading through `TurnCx`

Threading an `approved_fingerprint` field through `TurnCx` would touch ~25 `TurnCx { .. }` literals
across every tool's tests (large, cross-crate churn). Instead, add one default method to the `Tool`
trait:

```rust
async fn resolved_fingerprint(&self, _call: &ToolCall, _cx: &TurnCx<'_>) -> Option<CommandFingerprint> { None }
```

Only `daemon-tool-shell` overrides it. Every other tool inherits `None` → zero churn. The engine calls
it in `resolve_approvals` (which already has the tool registry and a `TurnCx`).

## 4. Absolute-path resolution (item 2)

New helper in `crates/engine/daemon-core/src/exec/mod.rs`:

```rust
pub fn resolve_program_abs(program: &str, cwd: &Path, path_env: &OsStr) -> std::io::Result<PathBuf>;
```

- If `program` contains a path separator: candidate = absolute-as-is or `cwd.join(program)`; verify it is
  a regular file and (unix) has an executable bit; `canonicalize()` and return.
- Else (bare name): split `path_env` on the OS path separator; the first entry that joins to an
  executable regular file wins; `canonicalize()` and return.
- No executable found → `ErrorKind::NotFound`.

The shell tool resolves the program with the SAME `PATH` value the exec backend passes
(`std::env::var_os("PATH")`), sets `Command.program` to the resolved absolute path, and execs that. So
`LocalEnvironment::run`'s internal `Command::new` receives an absolute path — no PATH re-resolution at
exec time. std-only (no `unsafe`; unix mode check via `PermissionsExt`).

## 5. Background `sh -c` as a distinct, higher-friction capability (item 3)

Foreground argv today only prompts when `needs_approval` matches a dangerous pattern; a **benign**
background line spawns unattended (no approval) — yet ANY shell string is arbitrary code (pipes,
redirects, subshells, `curl … | sh`). Make the shell-string surface distinct:

- Route background/pty through a **separate gate** in `daemon-tool-shell::run` (a `approve_shell_command`
  call / distinct code branch) that ALWAYS requires an approval decision — it is never eligible for the
  benign "runs unattended" fast path that foreground argv uses. Under the default `Ask` policy this means
  every background spawn now prompts (higher friction); under `AutoAllow` `decide_command()` still returns
  `Allow` (the operator-tier restriction of `AutoAllow` itself is the sibling **policy-partition** track's
  job — §7). Hardline denylist still applies to both surfaces.
- The distinct gate's prompt and fingerprint are computed over the resolved `sh -c` **script** (surface
  `exec.shell`/`exec.pty`) + resolved absolute `sh` + cwd + env-delta, so the durable park of a background
  command is fingerprint-checked on re-run too.

This is self-contained in `daemon-tool-shell` (the gate) + `daemon-processes` only if the fingerprint of
the script must be recomputed at spawn (it is recomputed by the tool's `resolved_fingerprint`, so
`registry.rs` needs no logic change; a doc note on `SpawnRequest`/`spawn` recording that the line is the
approved shell surface is the only registry touch, if any).

## 6. Wire / CDDL / snapshot impact

- **No daemon-api wire type change, no CDDL change, no daemon-store SQL change** in the planned design:
  the fingerprint enforcement lives in the CBOR snapshot (`PendingApproval`, `#[serde(default)]`), and the
  operator display is carried by the existing `prompt` string. Therefore `cargo test -p daemon-api
  --features arbitrary` is **not** triggered by this track.
- Surfaced choice for the coordinator: if a *structured* `ApprovalInfo.fingerprint` wire field is wanted
  for the GUI (vs embedding the digest in `prompt` text), that WOULD be a wire change → CDDL update +
  `-p daemon-api --features arbitrary`. **Deferred** here to keep the track wire-stable and avoid
  colliding with the Phase 4 conformance/CDDL work; the honest prompt already surfaces the tuple.

## 7. CROSS-TRACK COORDINATION (policy-partition sibling) — files/functions I may touch

The sibling **policy-partition** track constrains `execute_code` project-mode venv trust and is scoped to
**`tools/daemon-tool-execute-code/src/python.rs`** (`resolve_interpreter`), plus
`daemon-api/src/profile.rs`, `daemon-host/src/node_api/builtins.rs`, and `tools/daemon-tool-cron`.

`execute_code` is also a fingerprint-relevant exec site (it re-resolves interpreter + argv at exec time,
which is itself an approve-then-swap vector). To deconflict, here is exactly what THIS track would touch
in that crate, and my recommended split:

- **`tools/daemon-tool-execute-code/src/lib.rs`** (functions `run` 207–258, `approval_prompt` 434–447,
  `execute` 263–301, `run_staged` 304–341): to (a) bind the fingerprint over the resolved
  `(interpreter_abs, argv, cwd, surface="exec.python")` and (b) implement `Tool::resolved_fingerprint`.
- **`tools/daemon-tool-execute-code/src/python.rs`**: OWNED BY THE SIBLING. I will **not** modify it.

**DECIDED (§0.1): option (1) — DEFER.** `tools/daemon-tool-execute-code` is NOT touched this wave (no
edit to `lib.rs` or `python.rs`). Because the engine computes the fingerprint at park time via
`Tool::resolved_fingerprint` (default `None`), execute_code's parked approvals simply carry
`fingerprint: None` and run as today — no enforcement, no code change. `execute_code` fingerprinting is a
tracked follow-on after policy-partition merges.

## 8. Bug-reproducing tests (added FIRST) + honest residual coverage

Methodology note: some changes are type-additive (a new field / trait method), so a test cannot literally
fail to *compile* pre-fix. For each such test I will confirm it has teeth by temporarily stubbing the new
check to a no-op and observing the failure, then restoring — documented in the commit. Pure-behavior tests
below fail pre-fix with no new API.

### Pure-behavior reproducers (fail on today's code, no new API)

1. `daemon-tool-shell`: **benign background line is refused by a denying host.**
   `{"command":"echo","args":["hi"],"background":true}` with `FixedHost(false)`.
   - Pre-fix: benign background is not gated → spawns → `ok == true`. FAILS.
   - Post-fix: distinct shell-string gate asks → denied → `ok == false`, content mentions the denial.

### Enforcement reproducers (engine-level, `engine/tests.rs`, using the existing `support.rs` harness + a fake fingerprinted tool)

2. `resolve_approvals` **refuses on fingerprint mismatch.** Seed `snapshot.pending_approvals` with a
   `PendingApproval { fingerprint: Some(EXPECTED), call, .. }` and `self.pending` with an `"allow"`
   completion for that `job_id`; the fake tool's `resolved_fingerprint` returns a DIFFERENT value.
   - Assert: the fake tool's `run` was NOT invoked (a shared counter stays 0) and the spliced result is a
     refusal. (Teeth check: stub the comparison to always-equal → the tool runs → assertion fails.)
3. `resolve_approvals` **runs on fingerprint match.** Same, but `resolved_fingerprint` == `EXPECTED`.
   - Assert: the tool ran and its output was spliced.
4. `resolve_approvals` **refuses when the fingerprint can no longer be resolved** (`resolved_fingerprint`
   returns `None` while `approval.fingerprint == Some`). Assert refusal, tool not run.
5. **Back-compat:** `approval.fingerprint == None` (legacy snapshot) → tool runs as before.

### Tool-contract reproducers (`daemon-tool-shell`) — prove the fingerprint would catch a swap

6. `resolved_fingerprint` **changes when cwd changes** (same call, two sticky cwds → different digests).
7. `resolved_fingerprint` **changes when the resolved binary path changes** (a bare name resolving under a
   PATH dir vs a workspace-planted `./tool` that canonicalizes elsewhere → different digests).
8. `resolve_program_abs` returns an **absolute** path for a real binary and errors (`NotFound`) for a bare
   name that is not on PATH / not executable.
9. **Display honesty:** the defer prompt for a parked command contains the resolved absolute binary, the
   cwd, and the surface tier (not just the space-joined line).

### Snapshot round-trip

10. `snapshot.rs`: a `PendingApproval` with `fingerprint: Some(..)` survives `encode`/`decode`; a
    pre-existing blob without the field decodes with `fingerprint == None` (serde default).

### Residual coverage (stated honestly)

- **Same-absolute-path content swap** (rewriting the bytes of the already-resolved binary between approval
  and exec, or within the live-path await window) is NOT closed — the fingerprint pins the absolute
  path (canonicalized), not the file *contents*. A content hash of the binary is intentionally out of
  scope (heavy, and hostile to legitimately-updated tools); this class is addressed by the Phase 3 OS
  exec sandbox / artifact-provenance items, not here.
- **Intermediate-directory symlink swap** on the cwd/binary path is bounded by the Wave-1 interim
  symlink guard (final component only) and fully closed by the Phase 3 `ContainedRoot`/openat2 work.
- **env-delta strictness:** including the `PATH` *value* in the fingerprint means a durable re-run
  refuses if the daemon's `PATH` changed across the park (fail-closed, consistent with the plan theme).
  If you prefer fewer false-refusals, we can hash only the resolved absolute binary (which already
  captures the security-relevant resolution) and drop the raw `PATH` value — flagged as a choice.
- **`execute_code`** fingerprinting is deferred per §7 (option 1).
- **`AutoAllow`/yolo** still auto-allows the shell-string surface; tightening `AutoAllow` to an
  operator-tier capability is the policy-partition track.

## 9. Planned edits (summary, for hunk-minimality review)

| File | Change |
|---|---|
| `crates/engine/daemon-core/Cargo.toml` | add `sha2 = { workspace = true }` |
| `crates/engine/daemon-core/src/exec/mod.rs` | `CommandFingerprint` newtype + `resolve_program_abs` + canonical hashing helper |
| `crates/engine/daemon-core/src/lib.rs` | re-export `CommandFingerprint`, `resolve_program_abs` |
| `crates/engine/daemon-core/src/tools.rs` | `Tool::resolved_fingerprint` default-`None` method |
| `crates/engine/daemon-core/src/turn.rs` | new `approve_shell_command` gate (always-ask shell-string surface); `Effect::AwaitDecision` UNCHANGED |
| `crates/engine/daemon-core/src/snapshot.rs` | `PendingApproval` gains `#[serde(default)] fingerprint` |
| `crates/engine/daemon-core/src/engine/views.rs` | construct `PendingApproval { fingerprint: None, .. }` (55–65) |
| `crates/engine/daemon-core/src/engine.rs` | stamp fingerprint at park (~1064); refuse-on-mismatch in `resolve_approvals` before re-run |
| `crates/engine/daemon-core/src/engine/tests.rs` | fake fingerprinted tool + enforcement tests |
| `tools/daemon-tool-shell/src/lib.rs` | resolve abs binary + exec it; `resolved_fingerprint`; distinct always-gated shell-string surface; honest prompt; tests |

No changes to: `daemon-store` SQL, `daemon-api` wire/CDDL, `ParkedApproval`, `Effect::AwaitDecision`,
`tools/daemon-tool-execute-code/*` (deferred), `daemon-processes` logic, or the ~25 `TurnCx { .. }` literals.

## 10. Gate commands (from worktree root, after approval + implementation)

```
nix develop --command cargo fmt --all -- --check
nix develop --command cargo clippy --workspace --all-targets -- -D warnings
nix develop --command cargo test --workspace --no-fail-fast
```

`-p daemon-api --features arbitrary` + `daemon-api.cddl` update are **not** expected (no wire type change,
§6); will run them only if the design changes to add a wire field.

Known pre-existing flakes to ignore (treat only new/different signatures as real):
`node::detached_delegation::detached_notice_reaches_a_parked_durable_parent`,
`node::detached_delegation::detached_fanout_materializes_distinct_children`,
`node::process_notify::injected_input_reaches_a_parked_durable_session_via_the_store_seam`.

## 11. Open questions for reviewer

1. `execute_code` fingerprinting: option (1) defer, or (2) wire it this wave editing only `lib.rs`? (§7)
2. Include the raw `PATH` value in the env-delta hash (strict, fail-closed) or hash only the resolved
   absolute binary (fewer false-refusals)? (§8)
3. Structured `ApprovalInfo.fingerprint` wire field for the GUI now (CDDL + arbitrary) or keep the digest
   embedded in the `prompt` text and defer the wire field to Phase 4 conformance? (§6)
4. Background shell-string always prompting under the default `Ask` policy — acceptable friction, or gate
   only "non-trivial" shell strings (and how to define trivial without re-introducing a bypass)?
