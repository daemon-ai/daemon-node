# HARDENING-PLAN — Phase 4 / clippy disallow-lists (capstone)

Track: `hardening/clippy-disallow` (worktree `/home/j/experiments/daemon-worktrees/clippy-disallow`,
off `hardening/integration`). **PLAN ONLY — no source touched yet.** This is the last Phase 4 track;
it converts the Phase 1–3 hardening conventions into build breaks so new adapters cannot regress.

Guiding principle inherited from the plan: *make the unsafe form unrepresentable*. Here the mechanism
is a workspace `clippy.toml` (`disallowed-methods` / `disallowed-types`) escalated to errors by the
gate `cargo clippy --workspace --all-targets -- -D warnings`.

---

## 0. Scope recap and the three lints

From the track brief, three invariants become self-enforcing:

1. **fs** — raw `std::fs` / `tokio::fs` *mutating/opening* ops must not appear outside the
   `ContainedRoot` sanctioned home (force attacker-influenced fs through `ContainedRoot`).
2. **egress** — raw `reqwest` client construction must not appear outside the `daemon-egress` crate
   (force outbound HTTP through the SSRF-safe egress client).
3. **shell** — `Command::new("sh")` / `bash -c` shell-string spawning must not appear outside the one
   gated exec path.

The four sanctioned homes that get a scoped `#[allow(...)]` anchor:

| Home | File | Role |
|---|---|---|
| `ContainedRoot` impl | `crates/engine/daemon-core/src/exec/contained.rs` | the fs capability |
| `daemon-egress` | `crates/engine/daemon-egress/src/lib.rs` | the SSRF-safe HTTP client |
| `EnvPolicy::apply` | `crates/contracts/daemon-common/src/env_policy.rs` | declared child env (anchor **already present**, line 46) |
| exec-approval shell gate | `crates/substrate/daemon-processes/src/registry.rs` (≈L448 `sh -c`) | the one gated background shell |

> **Env note (out of my 3-lint scope, deferred):** `EnvPolicy::apply` already carries
> `#[allow(clippy::disallowed_methods)]` (pre-placed by the child-env-policy track). The matching
> *ban* (raw `Command::env` / `env_clear` / `envs` outside `apply`) is **not** enabled here because
> the spawn sites have **not yet been migrated** to `EnvPolicy::apply` — they still call raw
> `.env_clear()`/`.env()` (see §3.4). Enabling the env ban is child-env-policy's remaining migration
> work; turning it on now would fail the gate. The pre-placed anchor becomes live (a harmless no-op
> `#[allow]` until then). Recommendation: **defer the env ban** to a follow-up once those sites route
> through `apply`. Confirm if you want it folded in anyway.

---

## 1. Clippy mechanism facts (these drive every scoping decision)

These are load-bearing constraints — the plan is shaped around them:

- **`clippy.toml` is workspace-global.** Clippy reads one `clippy.toml` at the workspace root for all
  members; there is no reliable per-crate config. So `disallowed-methods`/`disallowed-types` cannot be
  scoped to "only crate X" from config. **"Outside module X" is therefore expressed exclusively via
  per-site / per-module / per-crate `#[allow(...)]` anchors**, never via config scoping.
- **`disallowed_methods` is a single lint.** You cannot `#[allow]` *one* entry of the list; an
  `#[allow(clippy::disallowed_methods)]` silences **every** disallowed method in that scope. This
  couples fs and shell (both are "methods"). It is why **reqwest is placed under `disallowed-types`**
  (the separate `clippy::disallowed_types` lint) — so an fs `#[allow(disallowed_methods)]` never
  silences the egress ban.
- **`disallowed_methods` cannot match on arguments.** `Command::new("sh")` and `Command::new("git")`
  are indistinguishable to clippy. Catching the shell-string form therefore requires banning
  `Command::new` *wholesale* and letting every legitimate spawn carry a commented `#[allow]` (the
  "every spawn is a declared, reviewed site" property — the same philosophy as `EnvPolicy`). See §5.
- **Default levels.** `clippy::disallowed_methods` and `clippy::disallowed_types` are **Warn by
  default** and fire in *every* crate clippy compiles, independent of `[workspace.lints]`. So the two
  members lacking `[lints] workspace = true` (`crates/substrate/daemon-schedule`,
  `tools/daemon-tool-cron`) are still covered; both are already fs/egress/shell-clean, so they need no
  anchors. The `-D warnings` gate turns the warns into hard errors.
- **`--all-targets` runs a `--lib`/`--bin` pass (cfg(test) *off*) and a `--test` pass (cfg(test)
  *on*).** This is what makes the cheap test carve-out safe (§6): a crate-root
  `#![cfg_attr(test, allow(...))]` silences unit-test code in the `--test` pass, while the `--lib`
  pass still lints production code — so a production regression is still caught.

---

## 2. `clippy.toml` + workspace-lint wiring (exact)

Create `clippy.toml` at the worktree root:

```toml
# Phase 4 hardening guardrails (OpenClaw class). See HARDENING-PLAN-clippy-disallow.md.
# These make the Phase 1–3 choke points unbypassable: fs must go through ContainedRoot,
# outbound HTTP through daemon-egress, and every process spawn must be a declared, reviewed site.

disallowed-types = [
    # Raw reqwest clients must only be constructed inside `daemon-egress` (the SSRF-safe,
    # per-hop-revalidated redirect client). Placed under disallowed-TYPES (not -methods) so an fs
    # `#[allow(disallowed_methods)]` never silences the egress ban. Fires on field/param type
    # mentions AND on `reqwest::Client::{new,builder}()` construction paths.
    { path = "reqwest::Client", reason = "construct the SSRF-safe client via daemon-egress::EgressClient" },
    { path = "reqwest::blocking::Client", reason = "blocking reqwest is banned; use daemon-egress" },
]

disallowed-methods = [
    # --- fs: mutating / content-opening ops. Attacker-influenced paths must go through
    #     daemon_core::exec::contained::ContainedRoot (openat2 RESOLVE_BENEATH|NO_SYMLINKS). ---
    { path = "std::fs::write",            reason = "use ContainedRoot::write / open_write_file" },
    { path = "std::fs::read",             reason = "use ContainedRoot::read" },
    { path = "std::fs::read_to_string",   reason = "use ContainedRoot::read" },
    { path = "std::fs::read_dir",         reason = "use ContainedRoot::read_dir" },
    { path = "std::fs::create_dir",       reason = "use ContainedRoot::create_dir_all" },
    { path = "std::fs::create_dir_all",   reason = "use ContainedRoot::create_dir_all" },
    { path = "std::fs::remove_file",      reason = "use ContainedRoot::remove_file" },
    { path = "std::fs::remove_dir",       reason = "use ContainedRoot::remove_dir" },
    { path = "std::fs::remove_dir_all",   reason = "use ContainedRoot::remove_dir_all_sync" },
    { path = "std::fs::rename",           reason = "use ContainedRoot::rename" },
    { path = "std::fs::copy",             reason = "route through ContainedRoot" },
    { path = "std::fs::hard_link",        reason = "route through ContainedRoot" },
    { path = "std::fs::set_permissions",  reason = "use ContainedRoot::set_mode" },
    { path = "std::fs::File",             reason = "open via ContainedRoot (open_write_file / read)" },
    { path = "std::fs::OpenOptions",      reason = "open via ContainedRoot::open_write_file" },
    { path = "tokio::fs::write",          reason = "use ContainedRoot::write" },
    { path = "tokio::fs::read",           reason = "use ContainedRoot::read" },
    { path = "tokio::fs::read_to_string", reason = "use ContainedRoot::read" },
    { path = "tokio::fs::read_dir",       reason = "use ContainedRoot::read_dir" },
    { path = "tokio::fs::create_dir",     reason = "use ContainedRoot::create_dir_all" },
    { path = "tokio::fs::create_dir_all", reason = "use ContainedRoot::create_dir_all" },
    { path = "tokio::fs::remove_file",    reason = "use ContainedRoot::remove_file" },
    { path = "tokio::fs::remove_dir",     reason = "use ContainedRoot::remove_dir" },
    { path = "tokio::fs::remove_dir_all", reason = "use ContainedRoot::remove_dir_all_sync" },
    { path = "tokio::fs::rename",         reason = "use ContainedRoot::rename" },
    { path = "tokio::fs::copy",           reason = "route through ContainedRoot" },
    { path = "tokio::fs::File",           reason = "open via ContainedRoot" },
    { path = "tokio::fs::OpenOptions",    reason = "open via ContainedRoot" },

    # --- shell / process: force every spawn to be a declared, reviewed site; the sh/bash -c
    #     form must live only in the gated exec path (daemon-processes registry). ---
    { path = "std::process::Command::new",   reason = "declare the spawn; sh/bash -c only via the gated exec path" },
    { path = "tokio::process::Command::new", reason = "declare the spawn; sh/bash -c only via the gated exec path" },
]
```

**Deliberately NOT banned** (documented in §7): `metadata`, `symlink_metadata`, `try_exists`, `canonicalize`
— non-mutating stat/existence probes used pervasively for benign size/existence checks; they open no
content handle and banning them multiplies anchors for near-zero security value (`ContainedRoot`
already offers `symlink_metadata` for the cases that matter). `std::os::unix::fs::symlink` is noted as
an optional future addition.

**Workspace-lint wiring** (root `Cargo.toml`, `[workspace.lints.clippy]`): add, for
self-documentation and to guarantee the level regardless of the clippy default:

```toml
disallowed_methods = "warn"
disallowed_types   = "warn"
```

Members already inherit via `[lints] workspace = true`; the two non-opted crates are covered by the
default-Warn behaviour. No new `[lints]` opt-ins required.

---

## 3. Straggler inventory (classified a / b / c)

Classification: **(a)** migrate to the sanctioned API · **(b)** narrowly-scoped, commented `#[allow]`
(genuine straggler that cannot cleanly migrate now) · **(c)** owned by a sibling (ingress-governor:
`socket.rs`/`remote.rs`/`ws.rs`) — do **not** touch; rebase onto their merged result.

All line numbers are against the current `hardening/integration` snapshot and **must be re-verified
after the ingress-governor + conformance-cddl merges + rebase** (they may drift).

### 3.1 egress / reqwest (lint: `disallowed_types`)

| Site | Class | Action |
|---|---|---|
| `crates/engine/daemon-egress/src/lib.rs:164` field `http: reqwest::Client` + `:173` builder | anchor | **sanctioned home** — `#[allow(clippy::disallowed_types)]` on the struct field + the `new()` builder line |
| `tools/daemon-tool-vision/src/lib.rs:104` field + `:117` builder | **b** | scoped allow — vision is the *proven* self-contained SSRF pattern (`Policy::none()` + `next_hop` + `check_url`, `MAX_REDIRECT_HOPS=5`). Comment: "self-contained per-hop check_url; dedupe into daemon-egress is a follow-up". (Alt (a): migrate to `EgressClient`.) |
| `tools/daemon-tool-web/src/tavily.rs:23`+`:33` | **b** | scoped allow — fixed operator-keyed SaaS host (`api.tavily.com`), no agent-controlled URL, no redirect follow. (Alt (a): `EgressClient` + `Redirects::None`.) |
| `tools/daemon-tool-web/src/firecrawl.rs:22`+`:32` | **b** | scoped allow — fixed operator-keyed SaaS host (`api.firecrawl.dev`). (Alt (a): migrate.) |
| `crates/providers/daemon-models/src/hf/client.rs:20`+`:42` | **b** | scoped allow — fixed host (`huggingface.co`) metadata client. (Alt (a): migrate.) |
| `bins/daemon/src/main.rs:317` `reqwest::Client::new().get()` | **a** (pref) / b | single keyless `GET {base}/models` on an operator-configured base. **Recommend migrate** to `EgressClient` (add `daemon-egress` dep to `bins/daemon`); fallback: per-line `#[allow(clippy::disallowed_types)]` with comment. |
| `tests/daemon-conformance/src/node/daemon_cloud_e2e.rs:96` | test | carve-out (§6) |
| `crates/memory/daemon-mnemosyne/src/sync/tests.rs:618` | test | carve-out (§6) |
| `crates/adapters/daemon-http/tests/http_surface.rs:138,159,200,235` | test | carve-out (§6) |

> **Not a straggler:** `crates/adapters/daemon-matrix/src/account.rs:91` `Client::builder()` is
> `matrix_sdk::Client`, **not** reqwest — it will not be flagged. (Its `std::fs::create_dir_all` at
> L89 is a separate fs item, §3.2.)
>
> Note mnemosyne `sync/mod.rs` is already fully migrated to `daemon-egress` (no raw reqwest in
> production) — confirms the tree is the post-egress-client-merge state.

### 3.2 fs (lint: `disallowed_methods`) — production, non-test

The full workspace fs surface is ~575 call sites, but the overwhelming majority are **test code** and
**daemon-internal trusted paths** (config/data/store dirs). The security-relevant *untrusted-path*
surfaces (`daemon-tool-fs`, `daemon-tool-shell`, `daemon-tool-execute-code`,
`daemon-host/workspace_fs.rs`, `daemon-core/exec`) are **already migrated** to `ContainedRoot` in
Phase 1/3 — e.g. the fs tool's only non-test raw fs is two `tokio::fs::try_exists` probes (not
banned), and `workspace_fs.rs` has **zero** production raw fs. That is the whole point: the lint keeps
those already-clean surfaces clean.

**(c) — owned by ingress-governor, do not touch:** `daemon-host/src/socket.rs`, `remote.rs`, `ws.rs`
have **no** fs/reqwest/shell sites in the inventory, so there is no overlap; nothing to anchor there.
Rebase onto their merged result and re-verify.

**Untrusted-path surfaces → per-line/small anchors (keeps the file guarded against NEW raw fs):**

| File | Sites | Class | Action |
|---|---|---|---|
| `crates/engine/daemon-core/src/exec/contained.rs` | ~25 (unix `mod imp` + non-unix `mod imp`) | anchor | **sanctioned home** — `#[allow(clippy::disallowed_methods)]` on `mod imp` (unix) and `mod imp` (non-unix). 2 anchors. |
| `crates/engine/daemon-core/src/exec/local.rs:56` `tokio::fs::create_dir_all(&self.root)` | 1 | b | per-line allow — root bootstrap before `ContainedRoot::open`; comment. |
| `crates/engine/daemon-core/src/exec/mod.rs:324` `std::fs::metadata` | 1 | — | `metadata` not banned → no anchor. |
| `tools/daemon-tool-fs/src/lib.rs:642,888` `tokio::fs::try_exists` | 2 | — | `try_exists` not banned → no anchor. |
| `tools/daemon-tool-browser/src/supervisor.rs:280` `create_dir_all` + `:281` `write` | 2 | **a**/b | download/artifact dir under workspace. **Prefer migrate** to `ContainedRoot`; fallback 2 per-line allows. Flag for review. |
| `crates/substrate/daemon-workspace-index/src/indexer.rs:40` `metadata` + `:60` `std::fs::read` | 1 (read) | **a**/b | walks workspace content (attacker-influenced). **Prefer migrate** the `read` to `ContainedRoot`; fallback per-line allow. Flag. |
| `crates/substrate/daemon-processes/src/registry.rs:408` `std::fs::create_dir_all(&req.cwd)` | 1 | b | per-line allow — same file as the sanctioned shell gate; comment (req.cwd is validated upstream). |
| `tools/daemon-tool-execute-code/src/python.rs:106` `std::fs::metadata` | — | — | `metadata` not banned → no anchor. |

**Trusted-internal production fs → crate-level `#![allow(clippy::disallowed_methods)]`** (low churn;
these crates are not the threat surface — daemon-internal data/config/store paths):

| Crate (root `lib.rs`/`main.rs`) | Representative files |
|---|---|
| `crates/memory/daemon-mnemosyne` | `dr.rs`(19), `banks.rs`(5), `streaming.rs`(4), `diagnose.rs`, `sanitize.rs`, `sync/mod.rs`, `cost_log.rs`, `provider.rs`, `recall/query_cache.rs`, `store/mod.rs` |
| `crates/providers/daemon-models` | `manager.rs`(7), `registry.rs`(4), `acquire.rs`, `cache.rs`, `gguf.rs`, `hash.rs`, `inspect.rs`, `quantize.rs`, `hardware.rs`(spawn) |
| `crates/engine/daemon-context-lcm` | `tools/diagnostics.rs`(4), `store/mod.rs` |
| `crates/coprocessor/daemon-metta` | `state.rs`(3) |
| `crates/skills/daemon-skills` | `usage.rs`(3) |
| `crates/node/daemon-node` | `profiles/resolve.rs`(2), `cron/seed.rs`(spawn) |
| `crates/substrate/daemon-auth` | `store.rs` |
| `crates/substrate/daemon-provision` | `lib.rs`(root create + spawn) |
| `crates/adapters/daemon-matrix` | `account.rs`(crypto store dir) + `login.rs`(spawn) |
| `bins/daemon` | `main.rs`(config/socket/ws_root) — crate allow covers fs; reqwest handled in §3.1 |
| `xtask` | dev-only tooling (`main.rs`) — crate allow (or drop its `[lints] workspace = true`) |

**`daemon-core` and `daemon-host` are split** (they each contain BOTH an untrusted surface that must
stay guarded AND trusted internals), so they get **file-level** allows on the trusted files rather
than a crate-level allow:

- `daemon-core`: file-level `#![allow]` on `checkpoint.rs`(9) and `memory.rs`(1). (`exec/*` and
  `contained.rs` handled above with tighter anchors.)
- `daemon-host`: file-level `#![allow]` on `profiles.rs`(8), `blob_store.rs`(4), `credstore.rs`(4),
  `engine_incarnation.rs`(4), `web.rs`(4), `node_api/roster.rs`(2), `tls.rs`(1). **Not crate-level**,
  so `workspace_fs.rs` stays guarded (its production is already ContainedRoot-clean, and we want a NEW
  raw fs there to fail).

### 3.3 shell / spawn (lint: `disallowed_methods`, `Command::new`)

Because `Command::new` is banned wholesale (clippy is arg-blind), every production spawn becomes a
commented `#[allow]`. Files that already get an fs `#[allow(disallowed_methods)]` (crate- or
file-level) **cover their spawns too** (same lint) — so only spawns in otherwise-unallowed files need
a dedicated anchor.

| File:line | Binary | Class | Anchor needed? |
|---|---|---|---|
| `crates/substrate/daemon-processes/src/registry.rs:448` `Command::new("sh").arg("-c")` | **sh -c** | anchor | **sanctioned gated shell** — `#[allow]` + comment "the one gated background shell". (File already fs-allowed per §3.2.) |
| `crates/substrate/daemon-sandbox/src/linux.rs:187,206,352` (`/bin/sh -c`, probe, py) | sh/infra | b | OS-sandbox infra; per-site allows + comment (no fs in this file → dedicated anchors). |
| `crates/engine/daemon-core/src/exec/local.rs:83` `Command::new(&cmd.program)` | agent exec | b | per-line allow (the sanctioned foreground exec; argv-only, resolved-abs per Phase 2). |
| `tools/daemon-tool-shell/src/lib.rs:590` `Command::new(resolved.program_abs…)` | agent shell tool | b | per-line allow — resolved absolute program, gated per Phase 2; comment. |
| `tools/daemon-tool-execute-code/src/exec.rs:84`, `sandbox.rs:261`(bwrap), `python.rs:140` | interpreter/bwrap | b | per-site allows + comment. |
| `tools/daemon-tool-fs/src/lint.rs:95` `Command::new(program)` | fs-tool linter | b | per-site allow. |
| `crates/adapters/daemon-mcp-client/src/lib.rs:138` `Command::new(command)` | MCP stdio | b | per-site allow (declared InheritFull worker). |
| `crates/adapters/daemon-matrix/src/login.rs:30` `Command::new(opener)` | browser opener | covered | matrix crate-allowed (§3.2). |
| `crates/engine/daemon-core/src/checkpoint.rs:254,280,296` `git` | git | covered | daemon-core `checkpoint.rs` file-allowed (§3.2). |
| `crates/node/daemon-node/src/cron/seed.rs:115` | seed | covered | daemon-node crate-allowed. |
| `crates/providers/daemon-models/src/hardware.rs:53`(nvidia-smi), `quantize.rs:141` | probe/worker | covered | daemon-models crate-allowed. |
| `crates/substrate/daemon-provision/src/lib.rs:296` | provisioner | covered | provision crate-allowed. |
| `xtask/src/main.rs:95,1048(bash),1195,1210` | dev tooling | covered | xtask crate-allowed. |

No shell-string spawn (`sh`/`bash -c`) survives outside the sanctioned gate + documented infra
(`daemon-sandbox`, `xtask`). `execute-code/python.rs:141` `.arg("-c")` is `python -c` (interpreter),
**not** a shell — no action.

### 3.4 env (deferred — see §0 note)

Raw `Command` env mutation still lives at the spawn sites (not yet routed through `EnvPolicy::apply`):
`exec/local.rs:87-88`, `processes/registry.rs:453-455,529-532`, `sandbox/linux.rs:190,356`,
`execute-code/exec.rs:87-100`, `daemon-node/fleet/spawner.rs:196`. Because these are unmigrated, the
env ban is **not enabled** in this track. (`daemon-acp/src/lib.rs:115,276` `.env(...)` is an
`rmcp::McpServerStdio` builder, not `Command` — never in scope.)

---

## 4. Anchor summary (the "how many `#[allow]`" answer)

| Category | Lint | Count (approx) |
|---|---|---|
| Sanctioned homes | types+methods | `contained.rs` 2 (mod), `daemon-egress` 1, `EnvPolicy::apply` 0 (already present) |
| reqwest stragglers | `disallowed_types` | 4 scoped-allow (vision, tavily, firecrawl, hf) + 1 migrate-or-allow (`bins/daemon`) |
| Untrusted-surface fs/shell (per-line) | `disallowed_methods` | ~12 (exec/local, browser*, workspace-index*, processes, sandbox×3, shell tool, execute-code×3, fs-tool lint, mcp-client) |
| Trusted-internal fs | `disallowed_methods` | ~11 crate-level `#![allow]` + ~9 file-level (`daemon-host` 7, `daemon-core` 2) |
| Test carve-out | both | ~15 crate-root `#![cfg_attr(test, allow(...))]` + per-file allows on integration test files + 1 plain `#![allow]` on `daemon-conformance` |

Total production anchors ≈ **35–40**; test carve-outs ≈ **18**. This is a bounded, mechanical,
one-time pass. `*` = flagged for possible migration instead of allow (browser supervisor,
workspace-index indexer) — decide at review.

**Scoping mechanism ("outside module X"):** confirmed — clippy has no per-crate config, so the
boundary is drawn entirely by where the `#[allow]` anchors sit. Tight scopes (per-line / `mod`) are
used for the security-critical untrusted surfaces (so a *new* raw fs there still fails); coarse scopes
(crate-level) are used for daemon-internal trusted crates (low churn, not the threat model). reqwest is
under a *separate* lint so fs allows never widen the egress hole.

---

## 5. Alternatives considered (for reviewer choice)

- **Shell, lighter option:** drop the `Command::new` ban entirely and rely on the Phase 2
  exec-approval gate for runtime enforcement (clippy cannot see the `"sh"` argument anyway, so the ban
  is arg-blind and only yields the "declared spawn" property at the cost of ~15 spawn anchors). I
  **recommend keeping** the ban — the audited-spawn property matches the hardening theme and catches a
  *new* `Command::new("sh")` at review — but it is the most debatable churn; say the word to cut it.
- **fs, lower-churn option:** crate-level `#![allow]` on *every* crate except the ~5 tool/exec crates
  (instead of the hybrid file-level split for `daemon-host`/`daemon-core`). Fewer anchors, but coarser:
  a crate-allow on `daemon-host` would also stop guarding `workspace_fs.rs`. I **recommend the hybrid**
  (file-level for the two mixed crates) to keep the key untrusted surfaces guarded.
- **reqwest as `disallowed_methods`** (constructors only) instead of `disallowed_types`: more precise
  (fires only on construction) but re-couples reqwest with fs allows (`bins/daemon/main.rs` has both).
  `disallowed_types` is chosen for the clean decoupling.

---

## 6. Test interaction / carve-out

Test code legitimately uses raw fs/spawn/reqwest. Strategy, cheapest-first:

1. **Unit tests (`#[cfg(test)] mod tests` inside production crates):** add crate-root
   `#![cfg_attr(test, allow(clippy::disallowed_methods, clippy::disallowed_types))]`. Safe because the
   `--all-targets` `--lib` pass (cfg(test) off) still lints production code — only the `--test` pass is
   relaxed. Needed for crates whose *production* is otherwise tightly anchored (e.g. `daemon-core`,
   `daemon-host`, `daemon-tool-fs`, `daemon-tool-shell`, `daemon-tool-execute-code`). Crates that get a
   crate-level production `#![allow]` already cover their tests (allow is unconditional) — no cfg_attr
   needed there.
2. **Integration test files (`<crate>/tests/*.rs`):** top-of-file `#![allow(clippy::disallowed_methods,
   clippy::disallowed_types)]` (they are test-only; compiled with `--test`). e.g.
   `daemon-tool-execute-code/tests/execute_code.rs`, `daemon-http/tests/http_surface.rs`.
3. **`tests/daemon-conformance` (all-test member crate):** one plain crate-root
   `#![allow(clippy::disallowed_methods, clippy::disallowed_types)]` covers its ~92 fs + 1 reqwest
   sites.

---

## 7. Residual coverage / honest limitations

- **Arg-blind shell ban.** Clippy cannot distinguish `Command::new("sh")` from `Command::new("git")`;
  the ban is on all spawns. Precise "no shell string" enforcement would need a custom lint
  (`dylint`) — noted as a future option, not in this track.
- **fs/shell coupling.** A file/crate that is `#[allow(disallowed_methods)]` for fs also loses the
  `Command::new` guard in that scope. Mitigated by tight scopes on the untrusted surfaces; the
  crate-allowed trusted crates that also spawn (models, node, provision, matrix, xtask) accept this
  (their spawns are non-shell). Documented per site.
- **Stat probes unbanned.** `metadata`/`symlink_metadata`/`try_exists`/`canonicalize` are not banned
  (benign, pervasive, no content handle). `ContainedRoot::symlink_metadata` remains the sanctioned
  path where symlink-aware stat matters.
- **Non-unix `ContainedRoot` stub** uses lexical `contain` + `symlink_metadata` (weaker) — the
  Windows v1 stub lane; unchanged here, only anchored.
- **Env ban deferred** (§0/§3.4) — pending child-env-policy's spawn-site migration to
  `EnvPolicy::apply`.
- **Sibling overlap (`socket.rs`/`remote.rs`/`ws.rs`)** is class (c): no fs/reqwest/shell sites today,
  so no anchors from this track — but re-verify after the ingress-governor merge + rebase in case
  their consolidation introduced any.

---

## 8. Sequencing (expect to be held, then rebased)

I branch from `hardening/integration` now, but **I expect to be held** until the other two Phase 4
tracks — **ingress-governor** (editing `socket.rs`/`remote.rs`/`ws.rs`) and **conformance-cddl** —
merge. After they land I will **rebase `hardening/clippy-disallow` onto the merged result and
re-verify every file:line in §3** (the inventory is only accurate against the final tree). Landing this
capstone earlier would fail every not-yet-merged Phase 4 branch, which is exactly why it goes last.

---

## 9. Exact gate (run from worktree root, in the devShell)

The load-bearing gate is clippy passing **with the new denies active**:

```bash
nix develop --command cargo fmt --all -- --check
nix develop --command cargo clippy --workspace --all-targets -- -D warnings   # ← must pass WITH the new lints
nix develop --command cargo test --workspace --no-fail-fast
```

Machine-load note: if `bins/daemon/tests/host_launch.rs` fails under the full parallel run, re-run it
isolated. Known flakes to ignore (confirmed green in isolation): `detached_delegation` ×2,
`process_notify` store-seam.

Definition of done for the implement phase: minimal hunks, all three gate commands green (tails
pasted), a bug-reproducing test where feasible (e.g. a `trybuild`/compile-fail style check that a raw
`std::fs::write` / `reqwest::Client::new` / `Command::new("sh")` outside a sanctioned home trips
`-D warnings`). Do **not** merge; do **not** remove the worktree.

---

## 10. IMPLEMENTED — final dispositions (rebased onto merged integration)

Rebased cleanly onto `hardening/integration` (7b3796f) — which now contains ingress-governor
(`socket.rs`/`ws.rs`/`tls.rs`/`web.rs`/`remote.rs` + `daemon-common::ingress`) and
conformance-coverage (`control.rs`/`ownership_matrix.rs`). **Re-inventory against the merged tree
introduced no new reqwest/`Command`/fs stragglers** — the sibling files carry none, and the class-(c)
files were not touched. Two production files my Phase-1 inventory had miscounted (an early
`#[cfg(test)]` shadowed later production sites) were caught by clippy itself and anchored:
`daemon-host/src/revision.rs` (durable revision store) and `daemon-common/build.rs` (`git describe`).

### Lints shipped
- `clippy.toml`: `disallowed-types` = `reqwest::Client`, `reqwest::ClientBuilder`,
  `reqwest::blocking::Client` (`allow-invalid = true`, since `blocking` is not enabled in the tree);
  `disallowed-methods` = the fs mutating/opening set (std + tokio) **plus** `std/tokio process
  Command::new`. `[workspace.lints.clippy]` sets both to `warn`; `-D warnings` escalates.

### reqwest (disallowed_types)
| Site | Disposition |
|---|---|
| `daemon-egress/src/lib.rs` | sanctioned home — crate-level `#![allow(clippy::disallowed_types)]` |
| `bins/daemon/src/main.rs:317` (Daemon Cloud `/models`) | **MIGRATED** → `EgressClient` (`Redirects::None`: `base` is operator-config, may be private/self-hosted). `reqwest` dep removed from `bins/daemon`; `daemon-egress` added |
| vision, tavily, firecrawl, hf/client | **scoped allow** (fixed operator/SaaS/Hub endpoints, no agent URL) — per-field + per-ctor `#[allow(clippy::disallowed_types)]` with inline justification |
| `daemon-matrix/account.rs:91` | not reqwest (`matrix_sdk::Client`) — no anchor |

### fs (disallowed_methods)
| Site | Disposition |
|---|---|
| `daemon-core/exec/contained.rs` | sanctioned home — `#[allow]` on both `mod imp` |
| `workspace-index/indexer.rs` | **MIGRATED** — workspace-content read now `ContainedRoot::read_sync` (symlink-hardened; a symlinked file is skipped, fail-closed) |
| `browser/supervisor.rs` (screenshot) | **scoped allow** — fixed daemon temp dir + daemon-generated filename (no workspace/agent path) |
| trusted-internal crates (mnemosyne, models\*, context-lcm, metta, skills, auth) | crate-level `#![allow(clippy::disallowed_methods)]` (\* models is per-file, it also spawns) |
| `daemon-host` (profiles/blob_store/credstore/engine_incarnation/web/roster/revision), `daemon-core` (checkpoint/memory), `bins/daemon`, matrix/account, node/resolve | **file-level** allows (keeps `workspace_fs.rs` + `exec` guarded) |

### shell / spawns (disallowed_methods, arg-blind `Command::new`) — per Decision 2
Every production spawn is an explicit **commented** anchor naming what it spawns; no blanket over a
spawn site (a new `Command::new("sh")` in a new fn/file still fails). Sanctioned `sh -c` gate:
`daemon-processes/registry.rs`. Others (all argv-only, non-shell): matrix opener, MCP stdio, cron
seed, `git` (checkpoint/build.rs), nvidia-smi, quantize worker, provisioner, exec/local foreground
exec, shell-tool exec, execute-code interpreter/bwrap/python-probe, fs-tool linter. `xtask` (dev
tooling) is crate-level allowed. `daemon-sandbox` shell spawns are all `#[cfg(test)]` (test carve-out).

### env ban — DEFERRED (as agreed). No env-mutation entries added; `EnvPolicy::apply`'s pre-placed
anchor remains a harmless no-op until the spawn-site migration lands.

### tests — `#![cfg_attr(test, allow(clippy::disallowed_methods, clippy::disallowed_types))]` on each
crate with inline `#[cfg(test)]` banned usage (the `--lib` pass still guards production); top-of-file
`#![allow]` on integration test files (`tests/*.rs`); plain crate-level `#![allow]` on the all-test
`daemon-conformance`.

### Gate results (from worktree root, in the devShell)
- `cargo fmt --all -- --check` — clean.
- `cargo clippy --workspace --all-targets -- -D warnings` — **clean WITH the denies active**.
  Proven satisfiable AND live: a temporary `std::fs::read` added to a non-anchored production spot
  tripped `error: use of a disallowed method \`std::fs::read\`` (exit 101); reverted.
- `cargo test --workspace --no-fail-fast` — **1623 passed, 0 failed** (migrations behavior-neutral;
  known flakes did not surface).

### Residuals (unchanged from §7)
Arg-blind shell ban; fs/shell coupling under a shared lint (a file/crate fs-allow also permits
`Command::new` in that scope — used only where nothing spawns, or paired with per-spawn comments);
stat probes (`metadata`/`try_exists`/`canonicalize`) unbanned; env ban deferred. A future `dylint`
custom lint could make the shell ban argument-aware.
```
