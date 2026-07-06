# HARDENING-PLAN — Cluster E: Declared `EnvPolicy` on all child spawns

Track: `hardening/child-env-policy` (worktree `/home/j/experiments/daemon-worktrees/child-env-policy`, base master `37d8167`).
Phase 1 deliverable: **plan only** — no source edits, no commit. Implementation waits for coordinator approval.

## Goal (and the non-goal)

Make child-process environment inheritance a **declared, visible, lintable** choice at every spawn site
instead of an implicit default. This is the OpenClaw failure mode: guards that are conventional, not
mandatory — "the Nth spawn ships without the guard."

**Non-goal / do not "fix" by clearing env everywhere.** Long-lived node workers (provisioner, MCP
stdio, ACP) inherit the full daemon env **by design** — they are trusted node components that need
provider keys etc. This track introduces zero behavior change for those workers. The win is purely
that each spawn site now *states* `EnvPolicy::InheritFull` explicitly (with a justification), so
Phase 4 can add a clippy lint that bans *undeclared* inheritance, and MCP/ACP entries sourced from
less-trusted config can be flipped to `Clean` per-entry later.

## The correct existing pattern (reference, cited)

Agent-facing subprocess exec already scrubs the environment — this is the `Clean` shape to mirror:

```55:64:crates/engine/daemon-core/src/exec/local.rs
        // Scrubbed child env: nothing inherited (no host secrets leak into a tool's subprocess).
        let mut command = tokio::process::Command::new(&cmd.program);
        command
            .args(&cmd.args)
            .current_dir(&dir)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .stdin(std::process::Stdio::null())
```

`LocalEnvironment::run` (`crates/engine/daemon-core/src/exec/local.rs:43`) is the canonical
agent-facing `Clean{allowlist:["PATH"]}` site. `daemon-processes` `registry.rs` (the background
`sh -c` surface, lines 448–532) independently uses the same `env_clear()` + `PATH` + `PYTHONUNBUFFERED`
shape. Both are already effectively `Clean`; they are owned by other tracks (exec-approval /
policy-partition) and are **not modified here** — but they demonstrate the target `Clean` semantics.

## The `EnvPolicy` type and its home

### Type

```rust
/// The declared environment-inheritance policy for a child process spawn. Every spawn site states
/// one explicitly so inheritance is an audited, lintable choice — never an implicit default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnvPolicy {
    /// Inherit the parent (daemon) environment as-is. For **trusted node workers** that legitimately
    /// need the daemon's ambient env (provider keys, PATH, locale). Every use MUST carry a call-site
    /// doc comment justifying the trust. This is the audited choice, not a default.
    InheritFull,
    /// Start from an empty environment and carry through only the named variables in `allowlist`
    /// (values read from the parent env). For **agent-facing / less-trusted** children, so no host
    /// secret leaks into the child. Mirrors `LocalEnvironment::run` (allowlist `["PATH"]`).
    Clean { allowlist: Vec<String> },
}
```

### Home: `daemon-common` (new module `env_policy`), tokio helper behind an optional `process` feature

`daemon-common` is the **only** internal crate all three target sites already share
(`daemon-provision` depends on `daemon-common` + `tokio`; `daemon-mcp-client` on `daemon-core` +
`daemon-common` + `tokio`; `daemon-acp` on `daemon-common` + `agent-client-protocol`). `daemon-core`
(the reference `Clean` site) and `daemon-processes` (the `sh -c` site) also depend on it. It is the
root of the crate DAG (`crates/contracts/daemon-common/src/lib.rs:8` — "depends on nothing internal").

- The **enum** is pure data (no tokio) and is *always* compiled, so it is available for declaration
  everywhere and for the Phase 4 lint.
- The **application helper** `EnvPolicy::apply(&self, cmd: &mut tokio::process::Command, extra)` needs
  tokio, so it is gated behind a new optional `process` feature that pulls `tokio` (workspace dep,
  already `features=["full"]` and already present at both tokio spawn sites — no new tree cost).
  `daemon-provision` and `daemon-mcp-client` enable `daemon-common/process`.

`daemon-common/Cargo.toml` additions:

```toml
[dependencies]
tokio = { workspace = true, optional = true }

[features]
process = ["dep:tokio"]

[dev-dependencies]
tokio = { workspace = true }              # for the apply() roundtrip test
```

Not a wire type: `EnvPolicy` never appears in `ApiRequest`/`ApiResponse` or anything reachable from
them, so **no `daemon-api.cddl` change and no CDDL conformance impact.** (The `--features arbitrary`
gate is therefore not required for this track, though the standard gate still runs.)

> Alternative considered: a dedicated `daemon-spawn` crate (enum + helper + the only sanctioned
> `Command` env surface). It gives the strongest module boundary for Phase 4, but adds a whole crate
> for one enum + one fn and forces every site to route `Command` construction through it. Rejected for
> Phase 1 on "simplicity first" grounds — the same lint outcome is achievable with `disallowed-methods`
> + a single `#[allow]` on `apply` regardless of which crate hosts it. Noted as the escalation path if
> the coordinator prefers a hard crate boundary.

### The helper (the mandatory, lint-anchored application path)

```rust
#[cfg(feature = "process")]
impl EnvPolicy {
    /// The ONLY sanctioned way to set a child's environment. Applies the base inheritance policy,
    /// then layers the caller's explicit `extra` vars (declared, deliberate additions — distinct
    /// from ambient inheritance). Phase 4 clippy bans raw `.env_clear`/`.env`/`.envs` outside this fn.
    #[allow(clippy::disallowed_methods)] // the one sanctioned env-mutation site
    pub fn apply<'c>(
        &self,
        cmd: &'c mut tokio::process::Command,
        extra: &[(String, String)],
    ) -> &'c mut tokio::process::Command {
        match self {
            EnvPolicy::InheritFull => { /* keep the parent env as-is */ }
            EnvPolicy::Clean { allowlist } => {
                cmd.env_clear();
                for name in allowlist {
                    if let Some(v) = std::env::var_os(name) {
                        cmd.env(name, v);
                    }
                }
            }
        }
        for (k, v) in extra {
            cmd.env(k, v);
        }
        cmd
    }
}
```

Handling `extra` in the same call means **all** child-env mutation flows through `apply`, so raw
`.env*` can be banned wholesale later. `extra` covers the per-spawn declared vars (`spec.env`,
MCP `env`, ACP recipe `env`) that today are set in ad-hoc loops.

## Spawn-site inventory (whole workspace) with file:line + policy decision

| # | Site | file:line | Owns a `Command`? | Current env behavior | This track |
|---|------|-----------|-------------------|----------------------|------------|
| 1 | **ProcessProvisioner::spawn_framed** (worker spawn) | `crates/substrate/daemon-provision/src/lib.rs:295` (`.env` loop `302–304`) | yes, `tokio::process::Command` | inherit full + `spec.env` extras | **ADOPT `InheritFull`** via `EnvPolicy::InheritFull.apply(&mut command, &spec.env)`; doc comment: trusted node worker needing provider env |
| 2 | **McpClient::connect** (MCP stdio) | `crates/adapters/daemon-mcp-client/src/lib.rs:138` (`.env` loop `140–142`) | yes, `tokio::process::Command` (then wrapped by `TokioChildProcess::new`) | inherit full + config `env` extras | **ADOPT `InheritFull`** via `apply(&mut cmd, env)`; doc note: MCP servers *can* come from less-trusted config, so `Clean` per-entry becomes possible later — default unchanged now |
| 3 | **AcpLaunch::into_agent** (ACP stdio) | `crates/adapters/daemon-acp/src/lib.rs:93` (`McpServerStdio…env(...)`) | **no** — the `agent-client-protocol` lib owns the spawn | inherit full + recipe `env` extras (library default) | **DECLARE `InheritFull`** (documented on `AcpLaunch`); cannot call `apply` — see limitation below |
| 4 | LocalEnvironment::run (agent-facing exec) — **reference** | `crates/engine/daemon-core/src/exec/local.rs:56` (`env_clear` `60–61`) | yes, tokio | `Clean{["PATH"]}` already | out of scope (other track); cite only. *Optional*: could migrate to `EnvPolicy::Clean{["PATH"]}.apply(...)` to prove the Clean path — see "Optional" |
| 5 | daemon-processes `sh -c` background | `crates/substrate/daemon-processes/src/registry.rs:448,529` (`env_clear` `453,529`) | yes, std | `Clean`-style (`PATH`,`PYTHONUNBUFFERED`,`TERM`) | out of scope (exec-approval / policy-partition track); no change |
| 6 | execute-code `exec.rs` | `tools/daemon-tool-execute-code/src/exec.rs:77` (`env_clear` `80`) | yes, std | `Clean`-style (`PATH`,`PYTHONDONTWRITEBYTECODE`,`TZ`) | out of scope (execute-code track); no change |
| 7 | execute-code `sandbox.rs` (bwrap) | `tools/daemon-tool-execute-code/src/sandbox.rs:148` | yes, tokio | bwrap wrapper | out of scope; no change |
| 8 | execute-code `python.rs` | `tools/daemon-tool-execute-code/src/python.rs:127` | yes, tokio | interpreter probe | out of scope; no change |
| 9 | models `quantize.rs` (worker) | `crates/providers/daemon-models/src/quantize.rs:141` | yes, tokio | inherits full (no env set) | out of scope (models track); note as *undeclared inherit* the Phase-4 lint would flag |
| 10 | models `hardware.rs` (`nvidia-smi`) | `crates/providers/daemon-models/src/hardware.rs:53` | yes, std | inherits full (probe) | out of scope; note as undeclared inherit |
| 11 | fleet `spawner.rs` (ACP launch build) | `crates/node/daemon-node/src/fleet/spawner.rs:196` | no (builds `AcpLaunch`) | delegates to site 3 | no change; benefits from site 3's declaration |
| 12 | cron `seed.rs` | `crates/node/daemon-node/src/cron/seed.rs:103` | yes, std | inherits full | out of scope; note as undeclared inherit |
| 13 | engine checkpoint `git` | `crates/engine/daemon-core/src/checkpoint.rs:254,280,296` | yes, std | inherits full (`git`) | out of scope; note as undeclared inherit |
| 14 | matrix `login.rs` (browser opener) | `crates/adapters/daemon-matrix/src/login.rs:30` | yes, std | inherits full (`xdg-open`) | out of scope; note as undeclared inherit |
| 15 | tools `daemon-tool-shell` | `tools/daemon-tool-shell/src/lib.rs:405` | wraps `exec::Command` (routes to site 4) | via ExecutionEnvironment | no change (goes through the `Clean` exec seam) |
| 16 | tools `daemon-tool-fs` lint | `tools/daemon-tool-fs/src/lint.rs:95` | wraps exec Command builder | via exec seam | no change |
| 17 | `xtask`, `daemon-common/build.rs` | `xtask/src/main.rs:95,1048,1195,1210`, `crates/contracts/daemon-common/build.rs:50` | yes, std | build-time only | out of scope (not runtime child spawns) |
| 18 | `bins/daemon/tests/host_launch.rs` | `bins/daemon/tests/host_launch.rs:68` | yes, std (test harness) | already `env_clear` + allowlist | test-only; no change |

**In scope for this track:** sites **1, 2, 3** (plus the optional site 4 migration). Everything else is
inventoried so the plan is explicit about coverage; sites 9/10/12/13/14 are the "undeclared inherit"
population the Phase 4 lint is designed to force into a stated policy, owned by their respective tracks.

## Concrete edits (Phase 2, after approval)

1. **`crates/contracts/daemon-common/`**
   - New `src/env_policy.rs`: the `EnvPolicy` enum (always compiled) + `apply` (feature `process`) +
     the roundtrip unit test (below).
   - `src/lib.rs`: `pub mod env_policy;` with a module doc line.
   - `Cargo.toml`: optional `tokio` dep, `process` feature, `tokio` dev-dependency (as above).

2. **`crates/substrate/daemon-provision/`**
   - `Cargo.toml`: `daemon-common = { workspace = true, features = ["process"] }`.
   - `src/lib.rs` `spawn_framed`: replace the `for (key,value) in &spec.env { command.env(...) }` loop
     (lines 302–304) with:
     ```rust
     // EnvPolicy::InheritFull — this is a trusted node worker (a placed cut of the daemon itself);
     // it needs the daemon's ambient env (provider keys, PATH, locale). Declared, not implicit.
     daemon_common::env_policy::EnvPolicy::InheritFull.apply(&mut command, &spec.env);
     ```

3. **`crates/adapters/daemon-mcp-client/`**
   - `Cargo.toml`: add `features = ["process"]` to the `daemon-common` dep.
   - `src/lib.rs` `connect`: replace the `for (k,v) in env { cmd.env(k,v) }` loop (lines 140–142) with:
     ```rust
     // EnvPolicy::InheritFull — MCP stdio servers are launched as trusted node components today and
     // inherit the daemon env. NOTE: server configs can originate from less-trusted sources; when
     // that lands, flip per-entry to EnvPolicy::Clean{allowlist} — behavior unchanged for now.
     daemon_common::env_policy::EnvPolicy::InheritFull.apply(&mut cmd, env);
     ```

4. **`crates/adapters/daemon-acp/`**
   - `Cargo.toml`: `daemon-common` is already a dep; **no `process` feature** (ACP does not own a
     `Command`, so it cannot call `apply`).
   - `src/lib.rs`: add a `policy: EnvPolicy` field to `AcpLaunch` (defaulting to `InheritFull` in
     `new`), with a doc comment stating ACP agents inherit the full daemon env by design and that
     `Clean` is **not currently representable** for ACP (the `agent-client-protocol` transport owns
     the spawn and offers no `env_clear`). `into_agent` consumes `self.policy` in a `match`:
     `InheritFull => { /* pass env through to McpServerStdio as today */ }`. This gives the site a
     *named, lintable* policy even though enforcement is limited to declaration for now.

## Tests to add

Single combined roundtrip test in `daemon-common/src/env_policy.rs` (feature `process`), spawning a
child that prints its environment (`env` — consistent with `local.rs`'s existing `pwd`/`printf`
tests, which already assume coreutils in the devshell):

```rust
#[cfg(all(test, feature = "process"))]
mod tests {
    use super::EnvPolicy;

    async fn child_env(cmd: &mut tokio::process::Command) -> std::collections::BTreeMap<String, String> {
        let out = cmd.output().await.expect("spawn env");
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| l.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
            .collect()
    }

    #[tokio::test]
    async fn env_policy_variants_apply_expected_child_env() {
        // Mutate the parent env once, in a single test, with a uniquely-named marker (edition 2021:
        // set_var is safe). Ambient PATH is also present in the parent.
        std::env::set_var("DAEMON_ENV_POLICY_MARKER", "present");

        // InheritFull → superset of the parent env: marker + extra both present.
        let mut inherit = tokio::process::Command::new("env");
        EnvPolicy::InheritFull.apply(&mut inherit, &[("EXTRA_A".into(), "1".into())]);
        let e = child_env(&mut inherit).await;
        assert_eq!(e.get("DAEMON_ENV_POLICY_MARKER").map(String::as_str), Some("present"));
        assert_eq!(e.get("EXTRA_A").map(String::as_str), Some("1"));
        assert!(e.contains_key("PATH"));

        // Clean{allowlist:["PATH"]} + extra → exactly {PATH, EXTRA_B}: marker absent.
        let mut clean = tokio::process::Command::new("env");
        EnvPolicy::Clean { allowlist: vec!["PATH".into()] }
            .apply(&mut clean, &[("EXTRA_B".into(), "2".into())]);
        let c = child_env(&mut clean).await;
        assert!(c.contains_key("PATH"));
        assert_eq!(c.get("EXTRA_B").map(String::as_str), Some("2"));
        assert!(!c.contains_key("DAEMON_ENV_POLICY_MARKER"), "Clean must not inherit the marker");
        let keys: std::collections::BTreeSet<&str> = c.keys().map(String::as_str).collect();
        assert_eq!(keys, ["EXTRA_B", "PATH"].into_iter().collect());
    }
}
```

This satisfies the requirement: **InheritFull yields a superset incl. a marker var; Clean yields
exactly the allowlist (+ declared extras).** Under `cargo test --workspace`, Cargo feature unification
turns `process` on for `daemon-common` (because sites 1/2 enable it), so the test compiles and runs in
the standard gate.

*Optional (recommended if the coordinator wants a live end-to-end assertion):* migrate site 4
(`local.rs`) to `EnvPolicy::Clean { allowlist: vec!["PATH".into()] }.apply(&mut command, &[])` — a
one-line, behavior-identical change that (a) exercises `Clean` on a real production path and (b) lets
`local.rs`'s existing tests double as coverage. Left optional because the task framed site 4 as a
*reference to cite*, not a mandated edit.

## How the policy becomes mandatory-and-lintable (Phase 4 hook — not implemented here)

Two enforcement tiers, both anchored on `apply` being the single sanctioned env-mutation site:

1. **Ban raw env mutation.** `clippy.toml` `disallowed-methods` on
   `std::process::Command::{env,env_clear,envs}` and `tokio::process::Command::{env,env_clear,envs}`.
   The only `#[allow(clippy::disallowed_methods)]` is on `EnvPolicy::apply`. Any spawn site that wants
   to touch child env must call `apply`, which forces it to *name a variant* — so "undeclared
   inheritance" fails the lint. (Undeclared inherit sites 9/10/12/13/14 above are exactly what this
   catches.)
2. **Ban raw spawn (stronger, optional).** To also catch a spawn that sets *no* env at all (silent
   inherit), provide a `daemon_common::env_policy::spawn(program, args, policy, extra)` wrapper (or a
   `CommandBuilder::new(policy)` whose only constructor takes an `EnvPolicy`), then `disallowed-methods`
   on `Command::spawn`/`::output`/`::status` outside the wrapper. This makes stating a policy the only
   way to obtain a runnable child. Deferred to Phase 4 because it is more invasive across the ~14 other
   sites; the enum + `apply` designed here are the foundation it builds on.

## Risks / ambiguities

- **ACP cannot enforce `Clean`.** The `agent-client-protocol` library owns the spawn via
  `McpServerStdio`; there is no `env_clear` hook, so ACP is `InheritFull`-only for now. Flipping ACP to
  `Clean` later needs upstream support or a pre-spawn shim. Documented at the site; declaration still
  gives the lint a hook.
- **`daemon-common` gains an (optional) `tokio` dep.** Slight layering smell for a "contracts" crate,
  mitigated by (a) feature-gating so the pure build is unaffected, (b) the enum being pure data, and
  (c) tokio already present at every consuming site. The `daemon-spawn` crate alternative is on the
  table if the coordinator prefers a hard boundary.
- **Feature-unification reliance for the test.** The roundtrip test runs under `cargo test --workspace`
  because sites 1/2 enable `daemon-common/process`. `cargo test -p daemon-common` alone would skip it;
  acceptable, and the gate uses `--workspace`.
- **`std::env::set_var` in the test** mutates process-global env; confined to one test with a uniquely
  named marker (`DAEMON_ENV_POLICY_MARKER`) and safe under edition 2021. No other daemon-common test
  reads that var.
- **`extra` var precedence.** `apply` sets `extra` *after* the base policy, so an explicit extra
  overrides an inherited value of the same name — matches today's per-site loops (they call `.env`
  after inheriting), so no behavior change.
- **`cargo machete`.** The new optional `tokio` in `daemon-common` is used by `apply`; the `process`
  feature makes the linkage visible, so no machete ignore entry is needed (verify in the gate).

## Definition of done (Phase 2)

From the worktree root, all green:
- `nix develop --command cargo fmt` (leave `--check` clean)
- `nix develop --command cargo clippy --workspace --all-targets -- -D warnings`
- `nix develop --command cargo test --workspace`
(No wire type changed, so the `--features arbitrary` CDDL gate is not triggered by this track, though
running the full AGENTS gate is fine.)

Commit on `hardening/child-env-policy`. Do **not** merge; do **not** remove the worktree.
