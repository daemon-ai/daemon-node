# HARDENING-PLAN — Cluster E: finish the `EnvPolicy::apply` migration + turn on the env clippy ban

Track: `hardening/env-ban-migration` (worktree
`/home/j/experiments/daemon-worktrees/env-ban-migration`, branch `hardening/env-ban-migration`
off `hardening/integration` @ `d98e226`).
**STATUS: APPROVED — implementing.** (Phase 1 was plan-only; the coordinator approved all three
decisions in §10, recorded below.)

### Approved decisions (coordinator)
1. **MCP stdio stays `InheritFull`.** Operator-configured node components that legitimately need the
   daemon env (PATH, HOME, provider keys); `Clean` would break real MCP servers. Per-entry trust is a
   separate future feature. Keep the existing declaration + comment. **No change this track.**
2. **`EnvSink` sealed generic `apply` — approved** (§2): generalized over std+tokio `Command` +
   `AsRef<OsStr>` extras. Handles `spawn_piped`'s std `Command` and execute-code's `OsStr` `TMPDIR`
   without a lossy `String` conversion, and keeps the two merged callers (provision, MCP) byte-identical.
3. **Coupling residual — accepted and documented** (§4/§7): a single fs `#[allow(clippy::disallowed_methods)]`
   also silences the env/shell bans in that scope (same residual clippy-disallow already accepted). **No
   `dylint` dependency this pass** — noted as a possible future precision refinement in §7.

### Per-site no-op-vs-scrub finding (verified against current source)
All three in-scope sites **already call `env_clear()`** today, so each already yields *exactly* PATH +
its own declared extras — none inherits more than PATH. Therefore **every migration is a true no-op**
(behavior-identical), not a deliberate scrub, and **no agent-relevant var is dropped** (nothing to flag):

| Site | Current env (verified) | Post-migration | Verdict |
|------|------------------------|----------------|---------|
| `local.rs` L93-94 | `env_clear()` + `PATH` | `Clean{["PATH"]}` | **no-op** |
| `execute-code/exec.rs` L90-103 | `env_clear()` + `PATH`(=`var_os("PATH")`) + `PYTHONDONTWRITEBYTECODE` + opt `TZ` + opt `TMPDIR`(=cwd) | `Clean{["PATH"]}` + those exact extras | **no-op** |
| `registry.rs` `spawn_piped` L461-463 | `env_clear()` + `PATH`(=`var_os("PATH")`) + `PYTHONUNBUFFERED` | `Clean{["PATH"]}` + that extra | **no-op** |

The tests (§6) still assert the **correct post-migration invariant** (child env == PATH + declared
extras only), which — because these already scrub — coincides with today's behavior.

This is the deferred completion of two earlier tracks:
- **child-env-policy** (merged) introduced `daemon_common::env_policy::EnvPolicy` + `apply()` behind the
  `process` feature and wired exactly **3 sites**: provisioner `spawn_framed`, MCP stdio, ACP.
- **clippy-disallow** (merged) shipped the fs / egress / `Command::new` bans and **pre-placed** the
  `#[allow(clippy::disallowed_methods)]` anchor on `EnvPolicy::apply`, but **deferred the env ban**
  because the remaining spawn sites still call raw `.env_clear()`/`.env()` (its §0/§3.4 note).

This track migrates *every remaining* raw-env spawn site onto `EnvPolicy::apply`, then adds the
`disallowed-methods` entries banning raw `Command::env`/`env_clear`/`envs` — making the pre-placed
anchor load-bearing and completing the Phase 4 lint set.

---

## 0. Ground truth (verified against the merged tree, not the stale child-env-policy inventory)

A whole-workspace grep for `\.(env|env_clear|envs)\s*\(` on `*.rs` yields exactly these hits. Each is
classified as **in-scope Command site** (must migrate), **already done**, **not a Command** (ban can't
reach it), or **test carve-out** (already covered).

| # | file:line | What it is | Command type | Status |
|---|-----------|-----------|--------------|--------|
| 1 | `crates/contracts/daemon-common/src/env_policy.rs:55,58,64` | **`EnvPolicy::apply` itself** — the sanctioned home | tokio | anchor (keep; see §2) |
| 2 | `crates/substrate/daemon-provision/src/lib.rs:309` | `InheritFull.apply(&mut command, &spec.env)` | tokio | **already migrated** (child-env-policy) |
| 3 | `crates/adapters/daemon-mcp-client/src/lib.rs:149` | `InheritFull.apply(&mut cmd, env)` | tokio | **already migrated** |
| 4 | `crates/adapters/daemon-acp/src/lib.rs:115,276` | `McpServerStdio::…env(...)` (rmcp builder) | **not a Command** | already declared `policy: EnvPolicy` (site owns no Command → ban never applies) |
| 5 | `crates/node/daemon-node/src/fleet/spawner.rs:196` | `AcpLaunch::env(...)` (our own builder) | **not a Command** | delegates to #4; ban never applies |
| 6 | `crates/engine/daemon-core/src/exec/local.rs:93-94` | agent foreground exec: `env_clear()` + `PATH` | **tokio** | **IN SCOPE → Clean{["PATH"]}** |
| 7 | `tools/daemon-tool-execute-code/src/exec.rs:90-92,98,103` | execute_code interpreter: `env_clear` + `PATH` + `PYTHONDONTWRITEBYTECODE` + opt `TZ` + opt `TMPDIR` | **tokio** | **IN SCOPE → Clean{["PATH"]}** + extras |
| 8 | `crates/substrate/daemon-processes/src/registry.rs:461-463` | `spawn_piped` (the sanctioned `sh -c` gate): `env_clear` + `PATH` + `PYTHONUNBUFFERED` | **std** (`shared_child`) | **IN SCOPE → Clean{["PATH"]}** + extra (needs std support, see §2) |
| 9 | `crates/substrate/daemon-processes/src/registry.rs:537-540` | `spawn_pty`: `env_clear` + `PATH` + `PYTHONUNBUFFERED` + `TERM` | **`portable_pty::CommandBuilder`** | **not a std/tokio Command** → ban can't match it. Residual (§7), behavior unchanged. |
| 10 | `crates/substrate/daemon-sandbox/src/linux.rs:190,356` | `run_sh`/probe helpers | tokio/std | **`#[cfg(test)] mod tests`** (starts L151) → covered by crate `#![cfg_attr(test, allow(...))]` |
| 11 | `bins/daemon/tests/host_launch.rs:25,27,44,52,72,73,74,76` | test harness launcher | std | test file, top-level `#![allow(...)]` (L4) |

**In scope for migration: exactly sites 6, 7, 8** (three production spawns). Everything else is
already migrated, not a `Command`, or a test carve-out that the clippy-disallow track already anchored.

### Silent full-inherit spawns are deliberately NOT in scope for the env-method ban
`python.rs:143` (interpreter probe), `checkpoint.rs` (`git`), `hardware.rs` (`nvidia-smi`),
`quantize.rs` (worker), `cron/seed.rs`, `matrix/login.rs` (`xdg-open`) spawn with **no `.env*` call at
all** — they inherit the full daemon env implicitly. A ban on `Command::env*` cannot see them (there is
no method call to flag); they are already declared, commented `Command::new` anchors from
clippy-disallow. Forcing them to *state* `InheritFull` needs the tier-2 "ban raw spawn / require a
policy to obtain a runnable child" approach the child-env-policy plan explicitly deferred. Out of scope
here; noted as residual (§7).

---

## 1. Policy classification (per in-scope site) + the InheritFull→Clean judgment call

| Site | Policy | Justification |
|------|--------|---------------|
| **local.rs** `LocalEnvironment::run` (agent foreground exec) | **`Clean { allowlist: ["PATH"] }`** | Agent-facing: a tool's subprocess. Already scrubs (`env_clear` + PATH). This is the canonical agent-facing shape the `EnvPolicy` doc cites; migrating it *proves the `Clean` path on a live production surface*. |
| **execute-code/exec.rs** `run_subprocess` | **`Clean { allowlist: ["PATH"] }`** + extras | Agent-facing: runs agent-authored code inside the execute_code sandbox. Already scrubs. |
| **registry.rs** `spawn_piped` (`sh -c` background) | **`Clean { allowlist: ["PATH"] }`** + extra | Agent-facing background shell; the Phase-2-gated high-friction capability. Already scrubs. |

All three are **already effectively `Clean`** — this migration is behavior-preserving, not a scrub-tightening.

### Judgment call surfaced for your decision — MCP stdio is `InheritFull`
The two node-worker `InheritFull` sites are correct by design:
- **provisioner `spawn_framed`** — a placed *cut of the daemon itself*; unquestionably `InheritFull`.
- **ACP** — trusted foreign-engine component; `Clean` is not even representable (rmcp owns the spawn).

The one arguable case is **MCP stdio (`daemon-mcp-client`)**: it is `InheritFull`, i.e. it inherits the
**full daemon env — including provider API keys — into an operator-configured external binary**. Its
call-site comment already flags that "server configs can originate from less-trusted sources." If any
MCP server entry is less-trusted, `InheritFull` leaks host secrets into it.
**My recommendation: leave it `InheritFull` for now** (behavior-preserving; it is a node-trusted
component today, and per-entry trust does not yet exist in the config model) but this is the prime
`InheritFull → Clean` candidate. **Flagging for your call** — say the word and I will flip MCP stdio to
`Clean { allowlist: [...] }` (a scrub, i.e. a deliberate behavior change) in this track instead of a
later one.

---

## 2. The sanctioned-home change: make `apply` cover **both** Command flavors + OsStr values

`apply` today is tokio-only with `extra: &[(String, String)]`. Two in-scope sites don't fit as-is:
- **registry `spawn_piped` uses `std::process::Command`** (via `shared_child::SharedChild::spawn`), not
  tokio. `apply` cannot be called on it.
- **execute-code sets `TMPDIR = cwd` (a `&Path` → `OsStr`)** losslessly today; `String` extras would
  force a lossy `to_string_lossy()` (a theoretical behavior change on a non-UTF-8 workspace path — a
  violation of "preserve exactly").

**Recommended design (minimal churn, keeps the merged sites untouched):** generalize `apply` over a
tiny **sealed `EnvSink` trait** (implemented for the two `Command` types) and over `AsRef<OsStr>` keys
/ values. This keeps **one** sanctioned env-mutation home for every spawn flavor.

```rust
// env_policy.rs — additions, all behind #[cfg(feature = "process")]

use std::ffi::OsStr;

mod sealed {
    pub trait Sealed {}
    impl Sealed for tokio::process::Command {}
    impl Sealed for std::process::Command {}
}

/// The child-command types that can receive a declared [`EnvPolicy`]. Sealed: only the std and tokio
/// `Command` flavors implement it, so `apply` is the single sanctioned env-mutation site for both.
pub trait EnvSink: sealed::Sealed {
    #[doc(hidden)]
    fn clear_env(&mut self);
    #[doc(hidden)]
    fn set_env(&mut self, key: &OsStr, val: &OsStr);
}

// The one sanctioned place raw `.env_clear()` / `.env()` on a `Command` appears (Phase 4 lint anchor).
#[allow(clippy::disallowed_methods)]
impl EnvSink for tokio::process::Command {
    fn clear_env(&mut self) { self.env_clear(); }
    fn set_env(&mut self, key: &OsStr, val: &OsStr) { self.env(key, val); }
}
#[allow(clippy::disallowed_methods)]
impl EnvSink for std::process::Command {
    fn clear_env(&mut self) { self.env_clear(); }
    fn set_env(&mut self, key: &OsStr, val: &OsStr) { self.env(key, val); }
}

impl EnvPolicy {
    /// The ONLY sanctioned way to set a child's environment (both `std` and `tokio` `Command`).
    /// Applies the base inheritance policy, then layers the declared `extra` vars on top (same
    /// precedence as the per-site `.env` loops this replaces). Phase 4 clippy bans raw
    /// `env`/`env_clear`/`envs` everywhere else, so an *undeclared* policy is unrepresentable.
    pub fn apply<'c, C, K, V>(&self, cmd: &'c mut C, extra: &[(K, V)]) -> &'c mut C
    where
        C: EnvSink,
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        match self {
            EnvPolicy::InheritFull => { /* keep the parent env exactly as-is */ }
            EnvPolicy::Clean { allowlist } => {
                cmd.clear_env();
                for name in allowlist {
                    if let Some(value) = std::env::var_os(name) {
                        cmd.set_env(name.as_ref(), &value);
                    }
                }
            }
        }
        for (key, value) in extra {
            cmd.set_env(key.as_ref(), value.as_ref());
        }
        cmd
    }
}
```

Notes:
- **The `#[allow]` moves from `apply` onto the two `EnvSink` impls** (that is where the raw `.env*`
  calls now live). `apply`'s body no longer calls any banned method. Still a single sanctioned location
  in one file — the pre-placed anchor's intent is preserved, just relocated.
- **The two already-merged callers stay byte-identical** — `apply(&mut command, &spec.env)` (provision)
  and `apply(&mut cmd, env)` (mcp) both infer `K=V=String` (`String: AsRef<OsStr>`), `C = tokio…Command`.
  No churn on merged sites.
- **The unit test needs one edit**: its `("…".into(), "…".into())` becomes ambiguous under generic
  `K`/`V`; change to bare `&str` literals (`("DAEMON_ENV_POLICY_EXTRA", "inherit")`) — behavior-identical.
- **Empty extras** at `local.rs` are written `&[] as &[(&str, &str)]` (concrete element type; no
  turbofish).
- `daemon-common/Cargo.toml` already exposes the `process` feature (`dep:tokio`); the `std` impl needs
  no tokio but shares the same gate for simplicity (all consuming crates enable `process` anyway).

> **Lower-churn alternative (offered):** keep `apply` (tokio/`String`) untouched; add a sibling
> `apply_std(&mut std::process::Command, &[(String,String)])` for the registry gate, and for
> execute-code's `TMPDIR` accept `cwd.to_string_lossy()`. Rejected as the default because the lossy
> path is a (theoretical) behavior change, and two near-identical methods are uglier than one sealed
> generic. Pick this if you prefer zero signature change to `apply`.

---

## 3. Concrete edits (Phase 2, after approval) — tests FIRST, minimal hunks

### 3.0 `crates/contracts/daemon-common/src/env_policy.rs` (the home)
- Add the `EnvSink` sealed trait + two impls; generalize `apply` (§2). Move the `#[allow]` onto the
  impls.
- **Extend the existing roundtrip test** (tests FIRST) to prove the new surface:
  - keep the tokio InheritFull-superset + Clean-drops assertions;
  - add an identical Clean-drops / allowlist-carries assertion run against a **`std::process::Command`**
    (proves the std path);
  - add an assertion that an **`OsString`-valued extra** is carried through unchanged (proves the
    `AsRef<OsStr>` path that `TMPDIR` relies on).
  - continue to use only *ambient* parent env vars as markers (no `set_var` — matches the current test's
    parallel-safe design).

### 3.1 `crates/engine/daemon-core/src/exec/local.rs` (site 6, Clean)
- `Cargo.toml`: `daemon-common = { workspace = true, features = ["process"] }`.
- Drop `.env_clear()` + `.env("PATH", …)` from the builder chain (L93-94); after the chain add:
  ```rust
  // EnvPolicy::Clean — agent-facing tool subprocess: nothing inherited but PATH, so no host secret
  // leaks into the child (unchanged behavior; now a declared, lintable policy).
  daemon_common::env_policy::EnvPolicy::Clean { allowlist: vec!["PATH".into()] }
      .apply(&mut command, &[] as &[(&str, &str)]);
  ```
- Keep the existing `#[allow]` on the `Command::new` line (L88) — still required for the spawn ban.

### 3.2 `tools/daemon-tool-execute-code/src/exec.rs` (site 7, Clean + extras)
- `Cargo.toml`: `daemon-common = { workspace = true, features = ["process"] }`.
- Drop `.env_clear()`, `.env("PATH", &path_env)`, `.env("PYTHONDONTWRITEBYTECODE", …)` from the chain,
  and the separate `cmd.env("TZ", …)` / `cmd.env("TMPDIR", cwd)` statements. Replace with one declared
  application built from an `extra` vector (preserving `path_env == parent PATH`, see nuance below):
  ```rust
  let mut extra: Vec<(&str, std::ffi::OsString)> =
      vec![("PYTHONDONTWRITEBYTECODE", "1".into())];
  if let Some(tz) = &tz { extra.push(("TZ", tz.into())); }
  if let Some(_spec) = &confine { extra.push(("TMPDIR", cwd.as_os_str().to_os_string())); }
  // EnvPolicy::Clean — agent-facing execute_code interpreter; PATH is the only inherited var.
  daemon_common::env_policy::EnvPolicy::Clean { allowlist: vec!["PATH".into()] }
      .apply(&mut cmd, &extra);
  ```
  `path_env` is `std::env::var_os("PATH")` (lib.rs:347) — identical to what `Clean{["PATH"]}` reads, so
  PATH is carried via the allowlist rather than as an extra (see the negligible-nuance note below).
- The `confine`/`TMPDIR` ordering vs `daemon_sandbox::confine_command(&mut cmd, spec)` is preserved
  (extras applied before `confine_command`, exactly as today). Keep the `#[allow]` on `Command::new`.

### 3.3 `crates/substrate/daemon-processes/src/registry.rs` `spawn_piped` (site 8, std, Clean + extra)
- `Cargo.toml`: `daemon-common = { workspace = true, features = ["process"] }`.
- Drop `.env_clear()` + `.env("PATH", …)` + `.env("PYTHONUNBUFFERED", "1")` from the `command` chain;
  after it add:
  ```rust
  // EnvPolicy::Clean — the one gated background sh -c capability; scrubbed env (PATH only) + the
  // unbuffered marker, unchanged. std Command via the same sanctioned apply (EnvSink).
  daemon_common::env_policy::EnvPolicy::Clean { allowlist: vec!["PATH".into()] }
      .apply(&mut command, &[("PYTHONUNBUFFERED", "1")]);
  ```
- Keep the sanctioned `sh -c` `#[allow]` on the `Command::new("sh")` line (L455).
- **`spawn_pty` (L533-540) is untouched** — `portable_pty::CommandBuilder` is neither a std nor tokio
  `Command`, so the env ban does not reach it and it cannot call `apply` (routing it would force a
  `portable_pty` dep into the `daemon-common` contracts crate — rejected). Its env is the identical
  fixed `Clean`-style shape; behavior unchanged. Documented as residual (§7).

### 3.4 `clippy.toml` — turn on the env ban (the capstone)
Append to the existing `disallowed-methods` list (after the `Command::new` entries):
```toml
    # --- env: every child-process env mutation must be a declared EnvPolicy via
    #     daemon_common::env_policy::EnvPolicy::apply (the one anchored EnvSink site). ---
    { path = "std::process::Command::env",         reason = "declare child env via EnvPolicy::apply" },
    { path = "std::process::Command::env_clear",   reason = "declare child env via EnvPolicy::apply" },
    { path = "std::process::Command::envs",        reason = "declare child env via EnvPolicy::apply" },
    { path = "tokio::process::Command::env",       reason = "declare child env via EnvPolicy::apply" },
    { path = "tokio::process::Command::env_clear", reason = "declare child env via EnvPolicy::apply" },
    { path = "tokio::process::Command::envs",      reason = "declare child env via EnvPolicy::apply" },
```
No `[workspace.lints]` change is needed — `disallowed_methods = "warn"` is already set (root
`Cargo.toml` L32); `-D warnings` escalates it. No new `#[allow]` anchors are needed beyond the two
`EnvSink` impls (§2) because sites 6/7/8's env calls disappear, and every other `.env*` is either a
test carve-out (already present) or a non-`Command` builder the paths don't match.

---

## 4. How the anchor covers itself, and the honest coupling limit

- **Self-coverage:** the raw `.env_clear()`/`.env()` now live *only* in the two `EnvSink` impls in
  `env_policy.rs`, each carrying `#[allow(clippy::disallowed_methods)]`. `apply` calls the trait methods
  (`clear_env`/`set_env`), which are **not** banned, so `apply` itself needs no allow. One file, two
  small anchored impls = the single sanctioned home.
- **Coupling limit (must be stated).** `disallowed_methods` is **one lint**: an
  `#[allow(clippy::disallowed_methods)]` silences *every* entry in that scope, including the new env
  ones. Because the clippy-disallow track already banned `Command::new` **wholesale**, every spawn site
  already carries such an allow. I verified the three in-scope sites use a **per-line** allow on only the
  `Command::new` *statement* — it does **not** extend over the separate `.env*` statements — so the env
  ban is genuinely effective there today (a stray `.env()` added as its own statement/chain trips it).
  But the trusted-internal crates that got a **crate-/file-level** `#![allow(clippy::disallowed_methods)]`
  from clippy-disallow (models, mnemosyne, matrix, `daemon-host/*`, checkpoint, cron, xtask, …) will also
  silence the env ban *within those scopes*. None of them currently contains raw `Command` env (grep-
  verified), so the gate passes; but a *future* raw `.env()` added inside one of those already-allowed
  scopes would not be caught. This is the **same fs/shell coupling residual** clippy-disallow documented
  (§7 there); the env ban simply joins it. A precise, argument-and-method-aware guard would need a custom
  `dylint` lint — noted, not in scope. **Net effect achieved:** no raw child-env mutation survives at
  gate time, and the agent-facing surfaces (the threat model) are guarded per-line.

---

## 5. Test-code carve-out (already in place — verified, no new work)
- `daemon-common` crate root has `#![cfg_attr(test, allow(clippy::disallowed_methods, clippy::disallowed_types))]`
  (lib.rs:15) → covers the env_policy roundtrip test (which uses `Command::new("env")`; it calls `apply`,
  not raw env, but is covered regardless).
- `daemon-sandbox` crate root has the same `cfg_attr(test, allow(...))` (lib.rs:29) → covers the
  `#[cfg(test)]` `run_sh`/probe env calls (site 10).
- `bins/daemon/tests/host_launch.rs` has a top-level `#![allow(clippy::disallowed_methods, clippy::disallowed_types)]`
  (L4) → covers the harness's `cmd.env(...)` calls (site 11).
The `--all-targets` `--lib` pass keeps cfg(test) OFF, so production env is still guarded in every one of
these crates. No new carve-out is required by this track.

---

## 6. Tests to add / behavior-preservation proof
1. **Policy roundtrip (extended, in `env_policy.rs`)** — the required behavior-preservation proof at the
   policy layer:
   - `InheritFull` on a `Command` yields a **superset** of the parent env (a pre-existing ambient marker
     passes through, declared extra layered on, PATH survives). *(existing assertion, kept)*
   - `Clean{["PATH"]}` yields **exactly** allowlist + declared extras — a non-allowlisted marker is
     **dropped**. *(existing assertion, kept)*
   - the **same** `Clean`/`InheritFull` assertions run against a `std::process::Command` (new — proves the
     `EnvSink` std path used by the registry gate).
   - an **`OsString`-valued extra** is carried through unchanged (new — proves the `AsRef<OsStr>` path
     `TMPDIR` uses).
2. **Site-level regression coverage (existing, exercised by migration):** `local.rs`'s existing exec
   tests and `daemon-tool-execute-code`'s integration tests already spawn real children and assert
   output; they run under the standard gate and confirm the migrated `Clean` sites still behave
   identically. No new site tests are needed beyond (1); if the gate surfaces a gap I will add a focused
   one.
3. **Ban-is-live proof (manual, documented in the implement report, not committed):** temporarily add a
   raw `tokio::process::Command::new("x").env("A","B")` in a non-anchored production spot and confirm
   `-D warnings` fails with `use of a disallowed method`, then revert — mirroring how clippy-disallow
   proved its bans satisfiable-and-live.

Marker discipline: reuse the current test's parallel-safe approach (read an **ambient** parent var; never
`set_var`) so nothing races under the parallel test runner.

---

## 7. Residuals / honest limitations
- **`spawn_pty` (`portable_pty::CommandBuilder`)** — a distinct type the `std/tokio::process::Command`
  ban does not match; left as-is (identical fixed `Clean` env shape; no security delta). Routing it
  through `apply` would need a `portable_pty` impl of `EnvSink`, i.e. a heavy dep in the contracts crate
  — rejected.
- **Silent full-inherit spawns** (python probe, `git`, `nvidia-smi`, quantize, cron seed, `xdg-open`)
  set no env, so the env-*method* ban cannot see them; they remain declared `Command::new` anchors.
  Making them *state* `InheritFull` is the deferred tier-2 "ban raw spawn" work, not this track.
- **fs/shell/env coupling under one lint** (§4) — crate-/file-level fs allows also silence env in those
  trusted-internal scopes; the agent-facing surfaces are guarded per-line. **Accepted** (matches the
  residual clippy-disallow already accepted). A `dylint` custom lint is the only way to make it
  argument/method precise — noted as a **possible future precision refinement**, deliberately **not**
  added this pass (avoids a new dev dependency for marginal gain).
- **MCP stdio `InheritFull`** (§1) — the flagged `InheritFull → Clean` judgment call; left `InheritFull`
  pending your decision.
- **Negligible behavior nuance at the three Clean sites:** today's `.env("PATH", var_os("PATH").unwrap_or_default())`
  sets `PATH=""` if PATH is unset; `Clean{["PATH"]}` instead *skips* PATH when unset (never sets an empty
  PATH). PATH is always present in the daemon env, so this is inert in practice; flagged for completeness
  because it is the one micro-difference from a byte-exact port (and arguably an improvement).

---

## 8. Cross-track / overlap
- I touch: `daemon-common/env_policy.rs` (+ its `Cargo.toml` already has `process`), `daemon-core/exec/local.rs`,
  `daemon-tool-execute-code/exec.rs`, `daemon-processes/registry.rs`, their three `Cargo.toml`s, and
  `clippy.toml`. No overlap with **codec-wire-bundle** (wire types) or **authz-f3f4** (`node_api` authz)
  — different files.
- **exec-os-sandbox** (merged): I checked its execute-code spawn wiring — `sandbox.rs` (bwrap) passes env
  via **argv** (`--setenv`-style / `path_env` pushed into `argv`), not `Command.env`; `python.rs` probe
  spawns with **no** env call. Neither has a raw `Command` env site, so neither needs migration. Only
  `exec.rs` (site 7) does, and it is in scope.
- No wire type changes → **no `daemon-api.cddl` / CDDL conformance impact** (the `--features arbitrary`
  gate is not triggered by this track).

---

## 9. Exact gate (implement phase; run from worktree root in the devShell)
The load-bearing proof is clippy passing **with the env ban active**:
```bash
nix develop --command cargo fmt --all -- --check
nix develop --command cargo clippy --workspace --all-targets -- -D warnings   # ← MUST pass WITH the new env ban
nix develop --command cargo test --workspace --no-fail-fast
```
Machine-load note: if `bins/daemon/tests/host_launch.rs` fails under the parallel run, re-run it
isolated. Known flakes to ignore: `detached_delegation` ×2, `process_notify` store-seam.

Definition of done (Phase 2): tests first, minimal hunks, all three commands green (tails pasted), the
ban proven live-and-satisfiable (§6.3). Commit on `hardening/env-ban-migration`. Do **not** merge; do
**not** remove the worktree.

---

## 10. Decisions — RESOLVED (see the "Approved decisions" block at the top)
1. **MCP stdio** → keep `InheritFull` (operator-configured node component; `Clean` would break real
   servers; per-entry trust is a separate future feature).
2. **`apply` API** → sealed `EnvSink` generic (std+tokio+`AsRef<OsStr>`); zero churn on merged sites.
3. **Coupling residual** → accepted and documented; no `dylint` this pass (future precision refinement,
   §7).
