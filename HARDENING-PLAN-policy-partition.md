# Cluster E — Policy Partition (Phase 2) — HARDENING PLAN

Worktree: `/home/j/experiments/daemon-worktrees/policy-partition` (branch `hardening/policy-partition`, off `hardening/integration`).
Status: **APPROVED — implementing.** Decisions confirmed by reviewer:
- **Decision 1 (widening-only):** APPROVED — gate only `approval_mode ∈ {AcceptEdits, AutoAllow}` and
  `ToolsOverride::FullToolset`; `Ask`/`Deny`/`Allowlist`/model/provider/workspace stay owner-allowed.
- **Decision 2 (blueprint accept):** APPROVED — a blueprint suggestion carrying `enabled_toolsets`
  requires operator to accept; the four starters set no security field, so common UX is unaffected.
- **Cron coverage (item 3):** CONFIRMED — `accept_suggestion` (cron.rs:362) calls `self.create(spec)`
  at cron.rs:367 under the accepting caller's `RequestContext`, so the `CronOps::create` operator gate
  covers it. No separate gate in `accept_suggestion` is needed. The agent-`cron`-tool unconditional
  reject stays as defense-in-depth.
- **Cross-track (item 4):** this track OWNS `python.rs::resolve_interpreter(mode, ws_root, trusted)`;
  it merges into integration first, and exec-approval-fingerprint composes its absolute-binary
  resolution on top. This track does NOT touch `exec.rs`/`sandbox.rs`.
- **Wire (item 5):** no CDDL change; `cargo test -p daemon-api --features arbitrary` run to prove zero
  wire drift.

## Scope (three defects to close)

1. **SessionOverlay security subset** — a non-operator (own-session `SessionWrite`) principal can
   widen its own approval posture (`approval_mode` → `AcceptEdits`/`AutoAllow`) or tool surface
   (`ToolsOverride::FullToolset`). Require an **operator-tier capability** for those specific
   mutations.
2. **Cron workdir/toolset** — a cron entry's `workdir` (→ `WorkspaceBinding::Bound`, an arbitrary
   in-place directory) and `enabled_toolsets` (→ `ToolsOverride::Allowlist`) are persisted with no
   operator check on either the API path or the agent-tool path. Require operator-tier for those two
   fields; the agent tool may never set them.
3. **execute_code venv trust on untrusted roots** — `project` mode auto-discovers and executes a
   workspace-local `.venv`/`venv` interpreter (and `VIRTUAL_ENV`/`CONDA_PREFIX`) even when the
   workspace root is an operator-bound external directory (`WorkspaceBinding::Bound`) that may carry
   attacker-planted files. Do not auto-trust a venv discovered under an untrusted (Bound) root.

Guiding constraint (from the plan): **reuse the existing capability model** — `daemon_auth::Capability`,
the `SessionSeeAll`/`SessionControlAny` operator overrides, `Role::Operator`, and the synthetic
`RequestContext::system()`/`internal()` principals. No parallel model.

---

## Capability decision (single operator-tier sentinel)

Use **`daemon_auth::Capability::SessionControlAny`** as the "operator-tier" gate for **all three**
security-widening surfaces (overlay approval/tools, cron workdir/toolset).

Why this cap:
- It is granted **only** to `Role::Operator` + `Role::Admin` (see `capability.rs::Role::capabilities`
  → `operator_extra`), and to the synthetic `RequestContext::internal()` (Operator-tier) and
  `system()` (Admin-tier). `Viewer`/`User` never hold it. So `principal.has(SessionControlAny)`
  cleanly means "operator or above."
- It is the existing session-control override the ownership layer already uses
  (`roster.rs::require_session_access`), so widening a session's execution posture reads as a
  session-control-tier act. Cron `workdir`/`toolset` are projected into a **session** overlay at
  hydrate (`cron/worker.rs::overlay_from_spec` → `WorkspaceBinding::Bound` + `ToolsOverride`), so the
  same sentinel is semantically consistent for cron.
- The plan explicitly names `SessionControlAny`/`SessionSeeAll` as the caps to reuse.

Fail-closed: a `None` request principal is denied (`ApiError::Unauthenticated`), matching the Wave-1
ownership posture. A legitimate in-process caller enters `system()`/`internal()` (both hold the cap).

New host helper (one place, mirrors `require_session_access`):

```rust
// daemon-host: node_api/roster.rs (next to require_session_access)
pub(crate) fn require_operator(&self, what: &str) -> Result<(), ApiError> {
    match crate::request_context::current_principal() {
        Some(p) if p.has(daemon_auth::Capability::SessionControlAny) => Ok(()),
        Some(_) => Err(ApiError::Forbidden(format!(
            "{what} requires an operator-tier capability (SessionControlAny)"
        ))),
        None => Err(ApiError::Unauthenticated(
            "no authenticated principal bound to this request".into(),
        )),
    }
}
```

### Decision: gate *widening* directions only (not the whole field)

The scope says "must not **widen** its own approval posture or tool surface." So:
- `approval_mode` → gated **only** for the widening modes `AcceptEdits` and `AutoAllow`. `Ask`
  (default) and `Deny` (strictest) are non-widening and remain available to a session owner — a user
  making their own session *safer* (`/mode ask`, `/mode deny`) must not need operator.
- `tool_allowlist` → gated **only** for `FullToolset` (widen to the entire node toolset). `Inherit`
  (keep profile) and `Allowlist(..)` (a restriction) are non-widening and stay owner-allowed.

(Stricter alternative, for reviewer: gate *every* `approval_mode = Some(_)` change. Rejected here as
user-hostile with no security gain — it would block a user from tightening their own session.)

---

## Defect 1 — SessionOverlay security subset

### Where the hole is
- `crates/contracts/daemon-api/src/profile.rs:382-425` — `SessionOverlay` (fields `approval_mode`,
  `tool_allowlist: ToolsOverride` incl. `FullToolset` at `:366-374`).
- `crates/substrate/daemon-host/src/node_api/session.rs`:
  - `set_session_mode` `:257-275` — persists `approval_mode`, only `require_session_access` (SessionWrite/own).
  - `set_session_overlay` `:277-295` — full-replace of the overlay, only `require_session_access`.
  - `set_session_model` `:233-255` — model/provider only; **not** security → stays ungated.
- `crates/substrate/daemon-host/src/node_api/builtins.rs`:
  - `builtin_mode` `:139-151` (`/mode`,`/yolo`,`/fast`) → `set_session_mode`. `resolve_approval_mode`
    `:220-243` maps `yolo→AutoAllow`, `fast→AcceptEdits`. Gating inside `set_session_mode` covers this
    path too (builtins call the handler directly).
- `crates/substrate/daemon-host/src/authz.rs:56-59` — `SetSessionOverlay`/`SetSessionMode`/
  `SetSessionModel` all map to `C::SessionWrite` (User-tier). **Unchanged** (coarse gate stays; the
  operator check is the *finer* in-handler gate for the security subset only).

### The predicate (added to daemon-api, next to the type — no wire change)
`crates/contracts/daemon-api/src/lib.rs` (`ApprovalMode`, `:98-128`):
```rust
impl ApprovalMode {
    /// Whether this mode *widens* autonomy (auto-approves gated actions) vs. the safe directions
    /// (`Ask` default / `Deny` strictest). Widening is operator-tier on a session overlay.
    pub fn widens_autonomy(self) -> bool {
        matches!(self, ApprovalMode::AcceptEdits | ApprovalMode::AutoAllow)
    }
}
```
`crates/contracts/daemon-api/src/profile.rs` (`SessionOverlay`, `:400-425`):
```rust
impl SessionOverlay {
    /// Whether this overlay *widens* the session's security posture — a security-relevant change that
    /// requires an operator-tier capability (approval-mode autonomy widening, or FullToolset). Model/
    /// provider/workspace/Allowlist/Inherit/Ask/Deny are not widenings.
    pub fn widens_security_posture(&self) -> bool {
        self.approval_mode.is_some_and(ApprovalMode::widens_autonomy)
            || matches!(self.tool_allowlist, ToolsOverride::FullToolset)
    }
}
```
Putting the definition on the types keeps "what counts as widening" next to the wire shape and unit-
testable in daemon-api; **enforcement** lives in the host (where the principal is bound).

### Enforcement (daemon-host `session.rs`)
- `set_session_mode` — after `require_session_access(&session, true)`, if `mode.widens_autonomy()`
  then `self.require_operator("widening the session approval mode")?;`
- `set_session_overlay` — after `require_session_access(&session, true)`, if
  `overlay.widens_security_posture()` then
  `self.require_operator("widening the session approval mode or tool surface")?;`
- `set_session_model` — untouched.

Both handlers already run under the caller's `RequestContext` (`require_session_access` reads
`current_principal()` today), and the builtin `/mode` path reaches `set_session_mode` under the same
context — so one in-handler gate covers wire + builtin uniformly.

---

## Defect 2 — Cron workdir/toolset

`workdir` and `enabled_toolsets` are the two privilege-relevant `CronSpec` fields:
- `cron/worker.rs::overlay_from_spec` `:118-132` maps `workdir → WorkspaceBinding::Bound` and
  `enabled_toolsets → ToolsOverride::Allowlist`, applied as the cron session's overlay at hydrate
  (`worker.rs:281-288`). A `workdir` therefore escapes the isolated per-session sandbox into an
  arbitrary directory; a pinned toolset fixes an unattended run's tool surface.

Two agent/user-reachable authoring paths, gated with **defense in depth**:

### (a) Shared choke point — `crates/substrate/daemon-host/src/cron.rs`
`CronOps::create` `:187-216` and `CronOps::update` `:219-237` (both API `cron_create`/`cron_update`
in `node_api/control.rs:977-989` **and** `accept_suggestion:362-374` funnel through here; both run
under the caller's `RequestContext` — `create` already reads `current_principal()` for the owner
stamp at `:209`. **CONFIRMED:** `accept_suggestion` calls `self.create(spec)` at cron.rs:367, so the
`create` gate covers the accept path with no separate check). Add, at the top of each:
```rust
if spec.workdir.is_some() || spec.enabled_toolsets.is_some() {
    require_operator_for_cron("cron workdir/toolset")?; // current_principal().has(SessionControlAny)
}
```
`CronOps` is in `daemon-host` (already imports `crate::request_context` + `daemon_auth`), so it uses a
free function form of the same predicate (or `crate::request_context::current_principal()` inline).
- **Not** gated: `pause`/`resume` (`:248-266`, re-parses the stored spec, never re-validates fields),
  `delete`, `trigger`, `list`. Node startup only *upserts suggestions* (`seed_catalog:307-328`), never
  creates jobs, so no principal-less internal `create` with workdir/toolset exists to break.
- **Consequence (documented):** the four starter suggestions set neither field
  (`cron_catalog.rs:445-511`) → a `User` may still accept them. A **blueprint** suggestion that
  carries `enabled_toolsets` (`cron_catalog.rs:531`) now requires operator to *accept* — a deliberate,
  correct outcome (an unattended run pinning a toolset is an operator decision).

### (b) Agent tool — `tools/daemon-tool-cron/src/lib.rs`
The agent `cron` tool must **never** author `workdir`/`enabled_toolsets` (it has no operator
identity; belt-and-suspenders that holds regardless of whether a principal is bound during a turn).
Add an unconditional reject in `spec_from` `:128-203` (mirrors the existing script/prompt/cron-session
guards):
```rust
if map.get("workdir").and_then(|v| v.as_str()).is_some_and(|s| !s.trim().is_empty()) {
    return Err("workdir is operator-only and cannot be set from the agent cron tool".into());
}
if str_list("enabled_toolsets").is_some() {
    return Err("enabled_toolsets is operator-only and cannot be set from the agent cron tool".into());
}
```
After this, the tool's `CronSpec` never carries the fields, so it passes the choke-point gate
regardless of the turn's principal binding.

No `daemon-auth` dep is added to `daemon-tool-cron` (the reject is a plain value check).

---

## Defect 3 — execute_code venv trust on untrusted roots

### The threat and the "untrusted root" definition
`python.rs::candidate_paths` (`:33-50`) in `Mode::Project` prepends `VIRTUAL_ENV`/`CONDA_PREFIX` and
the workspace-local `.venv`/`venv` `bin/python[3]` ahead of PATH `python3`/`python`, and
`resolve_interpreter` (`:23-30`) executes the first that probes as Python ≥ 3.8. The workspace root is
`cx.exec.cwd()` (`lib.rs:253`).

**"Untrusted root" = `WorkspaceBinding::Bound(path)`** — an operator/user-specified *external* directory
bound in place (`daemon-common/src/lib.rs:1412-1421`; realized in
`crates/node/daemon-node/src/profiles/resolve.rs::apply_workspace_exec:200-217`). Its contents
(a cloned repo, a PR branch, downloaded code) can carry an attacker-planted `.venv/bin/python`, which
project mode would silently execute — code execution outside the approval path. The default
`WorkspaceBinding::Isolated` per-session sandbox is node-managed → **trusted**.

The single shared `ExecuteCodeTool` instance is built once (`bins/daemon/src/main.rs:1129-1143`) and
shared across all sessions, so trust cannot be baked into settings — it must be read **per-session at
runtime** from the exec environment.

### Mechanism — thread a trust bit through the exec seam (no wire/CDDL change)
1. `crates/engine/daemon-core/src/exec/mod.rs` — add a **defaulted** trait method to
   `ExecutionEnvironment` (`:99-110`), so no other impl changes and object-safety is preserved:
   ```rust
   /// Whether this environment's root is trusted (a node-managed isolated sandbox) vs. an
   /// operator-bound external directory whose contents may be attacker-influenced. Tools use this to
   /// refuse auto-trusting workspace-discovered artifacts (e.g. a planted `.venv` interpreter).
   fn workspace_trusted(&self) -> bool { true }
   ```
2. `crates/engine/daemon-core/src/exec/local.rs` — add `trusted: bool` to `LocalEnvironment`
   (`:18-20`); `new()` (`:25-27`) and `sandbox()` (`:31-37`) keep `trusted: true`; add
   `pub fn with_trust(root, trusted)` (or `new_untrusted`); implement `workspace_trusted()`.
3. `crates/node/daemon-node/src/profiles/resolve.rs::apply_workspace_exec` (`:200-217`) — build the
   `LocalEnvironment` as **untrusted** when the binding is `Bound(_)`, trusted otherwise:
   ```rust
   let root = match &binding { Some(WorkspaceBinding::Bound(p)) => p.clone(), _ => roots.isolated_root(id.as_str()) };
   let trusted = !matches!(binding, Some(WorkspaceBinding::Bound(_)));
   Arc::new(LocalEnvironment::with_trust(root, trusted)) as Arc<dyn ExecutionEnvironment>
   ```
   (`dress.rs::root_profile:88-100` uses `LocalEnvironment::new` → trusted, unchanged.)
4. `tools/daemon-tool-execute-code/src/lib.rs::run` (`:253`) — read
   `let trusted = cx.exec.workspace_trusted();` and pass it into `execute(..)` (`:263-301`) → into
   `python::resolve_interpreter(mode, ws_root, trusted)`.
5. `tools/daemon-tool-execute-code/src/python.rs` — `resolve_interpreter(mode, ws_root, trusted)`
   (`:23-30`) and `candidate_paths(mode, ws_root, trusted)` (`:33-50`): when `!trusted`, **skip every
   venv candidate** (the `VIRTUAL_ENV`/`CONDA_PREFIX` block and the workspace `.venv`/`venv` block),
   leaving only the PATH `python3`/`python` fallback (`:44-48`). On an untrusted root, project mode
   thus resolves the same interpreter set as `strict` mode.

**Tradeoff (documented):** the "work on my repo, use its venv" convenience is intentionally lost on a
Bound root — the operator keeps venv auto-use on the trusted isolated sandbox. This is the plan's
explicit posture ("do not auto-trust/activate a venv discovered under an untrusted workspace root").

---

## CROSS-TRACK COORDINATION (sibling: exec-approval-fingerprint, Cluster B)

The sibling track binds exec approval to the resolved `(abs-binary, argv, env-delta, cwd)` and
"resolves binaries to absolute paths," and is noted to "also touch the execute_code tool … venv
resolution/trust and command execution." **Files where we may collide — deconflict before either
implements:**

| File | My edit (Cluster E) | Collision risk |
|---|---|---|
| `tools/daemon-tool-execute-code/src/python.rs` | `resolve_interpreter`/`candidate_paths` gain a `trusted` param; skip venv when untrusted | **HIGH** — if the fingerprint track resolves the interpreter to an absolute path here, we edit the same two fns. **This is the most likely direct conflict.** |
| `tools/daemon-tool-execute-code/src/lib.rs` | `run()` reads `cx.exec.workspace_trusted()`; `execute()` signature gains `trusted` | **MED** — `run()`/`execute()` is where the fingerprint hash over the resolved argv would also be computed. |
| `crates/engine/daemon-core/src/exec/local.rs` | add `trusted` field + `workspace_trusted()` method | **MED** — the fingerprint track's absolute-path resolution likely edits `LocalEnvironment::run` in the same file. |
| `crates/engine/daemon-core/src/exec/mod.rs` | add defaulted `workspace_trusted()` to the trait | **LOW** — additive; fingerprint may touch `contain`/argv nearby. |
| `crates/node/daemon-node/src/profiles/resolve.rs` | `apply_workspace_exec` builds untrusted env for Bound | **LOW** — fingerprint unlikely to touch profile resolution. |
| `tools/daemon-tool-execute-code/src/exec.rs`, `.../sandbox.rs` | **NOT TOUCHED by me** (argv build / subprocess run) | If the fingerprint track owns command execution, these are theirs. |

**Explicit flag:** if the fingerprint track rewrites `python::resolve_interpreter` (e.g. to return an
absolute interpreter path for its hash), **that is the same function my venv-trust change edits** —
we must agree on a merged signature (`resolve_interpreter(mode, ws_root, trusted) -> Option<PathBuf>`
already returns an absolute path, so both goals compose: I add the `trusted` gate, they consume the
absolute result). Merge order suggestion: land whichever first, the other rebases the shared fn.

I do **not** touch approval/argv/command-execution logic in execute_code, only interpreter *selection*
+ the `cx.exec` trust seam.

---

## Wire / CDDL impact

**None.** No wire type changes:
- `SessionOverlay`/`ApprovalMode` gain only inherent helper methods (no field/serde/shape change).
- `CronSpec` unchanged; gating is behavioral in `CronOps`/the tool.
- venv trust is threaded through the internal `ExecutionEnvironment` trait + `LocalEnvironment`; the
  `WorkspaceBinding` wire type is unchanged (trust is *derived* from it node-side).

Therefore `daemon-api.cddl` is **not** edited. I will still run
`cargo test -p daemon-api --features arbitrary` as a sanity check because I edit `daemon-api` source
(profile.rs/lib.rs), but expect no changes required.

---

## Bug-reproducing tests — ADDED FIRST, must fail pre-fix

### T1 — daemon-api predicate unit tests (`crates/contracts/daemon-api/src/profile.rs` tests, + lib.rs)
- `approval_mode_widening_classification`: `AcceptEdits`/`AutoAllow` ⇒ `widens_autonomy()` true;
  `Ask`/`Deny` ⇒ false.
- `overlay_widens_on_full_toolset_and_autonomy`: overlay with `FullToolset` ⇒ true; with
  `approval_mode=Some(AutoAllow)` ⇒ true; with `Allowlist`+`Ask` ⇒ false; model-only ⇒ false.
  (Pre-fix: the methods don't exist → these are the new contract; they *define* the classification.)

### T2 — daemon-host overlay enforcement (`tests/daemon-conformance/src/node/ownership.rs`, uses `ctx(name, role)`)
- `non_operator_cannot_widen_approval_via_set_session_mode`: alice (`Role::User`) owns a session;
  `node.set_session_mode(s, AutoAllow)` ⇒ `Err(Forbidden)`. **Pre-fix: `Ok`.**
- `non_operator_cannot_full_toolset_via_overlay`: alice (User) `set_session_overlay({tool_allowlist:
  FullToolset})` ⇒ `Err(Forbidden)`. **Pre-fix: `Ok`.**
- `operator_may_widen_approval_and_toolset`: `ctx("op", Role::Operator)` ⇒ both `Ok`.
- `owner_may_narrow_and_switch_model` (regression guard): alice `set_session_mode(Ask)` /
  `set_session_mode(Deny)` / `set_session_model(..)` / `set_session_overlay({tool_allowlist:
  Allowlist(..)})` ⇒ all `Ok`.

### T3 — daemon-host cron enforcement (`tests/daemon-conformance/src/node/ownership.rs`)
- `non_operator_cannot_set_cron_workdir`: alice (User) `cron_create(spec{workdir: Some})` ⇒
  `Err(Forbidden)`. **Pre-fix: `Ok`.** Same for `enabled_toolsets: Some`.
- `operator_may_set_cron_workdir_and_toolset`: `ctx("op", Role::Operator)` ⇒ `Ok`.
- (Existing `cron_session_inherits_cron_creator_owner:387` — a User creating a plain job with no
  workdir/toolset — must still pass: regression guard for the "no security field ⇒ user-allowed" path.)

### T4 — cron tool reject (`tools/daemon-tool-cron/src/lib.rs` tests)
- `spec_from_rejects_workdir_and_toolset`: `spec_from({... "workdir":"/srv"})` ⇒ `Err`;
  `spec_from({... "enabled_toolsets":["fs"]})` ⇒ `Err`. **Pre-fix: `Ok`.**
- Update the existing `spec_maps_full_arg_set:358-384` and `schema_*` expectation: drop `workdir`/
  `enabled_toolsets` from the "maps" assertion (they are now rejected), keeping the rest.

### T5 — execute_code venv trust (`tools/daemon-tool-execute-code/src/python.rs` tests, deterministic)
- `untrusted_root_skips_workspace_venv_candidates`: `candidate_paths(Mode::Project, root, trusted=false)`
  contains **no** path under `root/.venv` or `root/venv`; `trusted=true` **does** include them.
  (Pure path logic — no fork, no probe. **Pre-fix:** `candidate_paths` has no `trusted` param and
  always includes the venv paths.)
- (Optional integration test in `tests/execute_code.rs`: plant an executable `.venv/bin/python3` stub
  in a temp root, run the tool through an untrusted `LocalEnvironment::with_trust(root, false)`, and
  assert the planted interpreter is **not** the one resolved — only if the harness supports injecting
  an untrusted env cheaply; the unit test T5 is the primary reproducer.)

Every test above will be committed and shown red before the corresponding fix, then green after.
Run with `--no-fail-fast`; ignore the three known timing flakes
(`node::detached_delegation::*`, `node::process_notify::injected_input_*`).

---

## Residual coverage / honest gaps

- **Overlay full-replace clearing:** `set_session_overlay` full-replaces the overlay. A non-operator
  *clearing* an operator-set widening (e.g. sending `approval_mode=None`, `Allowlist`) is **allowed** —
  that narrows, not widens, so it is not a hole. Documented in the handler.
- **Non-security overlay fields stay user-writable:** model/provider/workspace and `Allowlist`/`Ask`/
  `Deny` remain owner-allowed by design (T2 regression guards this).
- **Cron `enabled_toolsets` is an `Allowlist` (a restriction), never `FullToolset`** — gating it is
  conservative (per the plan's "workdir/toolset") rather than strictly a widening fix; the real escape
  is `workdir`→Bound. Both gated.
- **venv trust is coarse (binding-level):** trust = `Isolated` vs `Bound`. It does **not** attempt to
  detect a malicious venv *inside* an isolated sandbox (an agent that `pip install`ed into its own
  sandbox in a prior turn is still "trusted"). That is acceptable — the sandbox is node-managed and
  agent-created; the OpenClaw failure mode is auto-trusting *external* (Bound) content. Deeper venv
  provenance is out of scope for Cluster E.
- **`VIRTUAL_ENV`/`CONDA_PREFIX` come from the daemon's own process env**, not the workspace; skipping
  them on untrusted roots (per plan wording) is extra caution, not the primary vector.
- **Coarse authz map unchanged:** `SetSessionOverlay`/`SetSessionMode`/`CronCreate`/`CronUpdate` keep
  their existing `SessionWrite`/`CronWrite` coarse gates; the operator check is a *finer* in-handler
  gate for the security subset only. A Phase-4 conformance addition (out of this cluster) could assert
  the finer gate in the ownership matrix.
- **`/mode yolo` via builtins:** covered because gating is inside `set_session_mode`. I will grep the
  command/builtin conformance tests during implementation for any existing test that drives `/mode
  yolo`/`/mode fast` under a non-operator principal and update it (expected to now require operator).

---

## Files to change (inventory)

Enforcement / logic:
- `crates/contracts/daemon-api/src/lib.rs` — `ApprovalMode::widens_autonomy` (+ tests).
- `crates/contracts/daemon-api/src/profile.rs` — `SessionOverlay::widens_security_posture` (+ tests).
- `crates/substrate/daemon-host/src/node_api/roster.rs` — `require_operator` helper.
- `crates/substrate/daemon-host/src/node_api/session.rs` — gate `set_session_mode`, `set_session_overlay`.
- `crates/substrate/daemon-host/src/cron.rs` — gate `CronOps::create` / `CronOps::update`.
- `tools/daemon-tool-cron/src/lib.rs` — reject `workdir`/`enabled_toolsets` in `spec_from`; update tests.
- `crates/engine/daemon-core/src/exec/mod.rs` — `ExecutionEnvironment::workspace_trusted` (defaulted).
- `crates/engine/daemon-core/src/exec/local.rs` — `LocalEnvironment` trust field + `with_trust` + impl.
- `crates/node/daemon-node/src/profiles/resolve.rs` — untrusted env for `Bound` in `apply_workspace_exec`.
- `tools/daemon-tool-execute-code/src/lib.rs` — read `workspace_trusted()`, thread into `execute`.
- `tools/daemon-tool-execute-code/src/python.rs` — `trusted` param; skip venv candidates when untrusted (+ tests).

Tests:
- `tests/daemon-conformance/src/node/ownership.rs` — T2, T3.
- `daemon-api` profile.rs/lib.rs test modules — T1.
- `tools/daemon-tool-cron/src/lib.rs` tests — T4.
- `tools/daemon-tool-execute-code/src/python.rs` tests — T5.

**Not touched:** `authz.rs` coarse map, `exec.rs`/`sandbox.rs` (execute_code command exec), the
`WorkspaceBinding` wire type, `daemon-api.cddl`.

---

## Gate commands (from worktree root, after implementation)

```
nix develop --command cargo fmt --all -- --check
nix develop --command cargo clippy --workspace --all-targets -- -D warnings
nix develop --command cargo test --workspace --no-fail-fast
nix develop --command cargo test -p daemon-api --features arbitrary   # sanity; no CDDL change expected
```

Definition of done: all four green (bar the three known flakes), each new test shown red pre-fix then
green post-fix, hunks minimal, work committed on `hardening/policy-partition`. **Do not merge; do not
remove the worktree.**
