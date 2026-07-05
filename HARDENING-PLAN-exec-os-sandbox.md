# HARDENING-PLAN — Phase 3 / Cluster B+C: kernel-enforced exec sandbox

Track: `exec-os-sandbox` (branch `hardening/exec-os-sandbox`, off `hardening/integration`).
Status: **APPROVED (with scope addition §11). Implementing the independent Linux/posture parts;
the staging-open containment (§11) is pending `contained-root-type`'s merged `ContainedRoot` API.**

## 1. Problem & scope

`execute_code` runs arbitrary agent-authored Python. Today its OS confinement is **bubblewrap
or nothing**:

- `SandboxPolicy::Auto` → bwrap when `bwrap_usable()` (binary present *and* user namespaces work),
  otherwise a **plain subprocess with zero kernel confinement** (`SandboxKind::Plain`). The only
  thing left is the tool's own lexical CWD containment (which the child, running real code under the
  daemon uid, can trivially ignore — it can `open("/etc/passwd")`, connect a socket, read the
  operator's home, etc.).
- `SandboxPolicy::Bwrap` → requires bwrap, else a setup error.
- `SandboxPolicy::None` → always plain.

So on any host where bwrap/userns is unavailable (hardened kernels with `kernel.unprivileged_userns_clone=0`,
many container/CI environments, non-Linux), `Auto` **silently degrades to unconfined**. That is the
exact OpenClaw failure shape: the guard is skippable and the skip is invisible.

**This track adds a kernel-enforced, in-process sandbox that does not need user namespaces, requires
it for the fallback, and makes "no confinement" an explicit, fail-closed operator decision.**

- **Linux (primary):** Landlock (filesystem scoping) + seccomp-bpf (network/syscall scoping) applied
  in the child via `pre_exec`. This is the fallback for the exact case that degrades to `Plain`
  today — no userns required (Landlock ≥ 5.13, seccomp ≥ 3.5; the dev/CI kernel here is 6.19).
- **macOS (secondary):** Seatbelt via a `sandbox-exec` SBPL profile (deny-default fs, ro system
  paths, rw workdir, network deny/allow) — an argv wrapper, like bwrap.
- **Windows (best-effort, v1 = stub-worker lane):** documented fail-closed. `Require` refuses;
  `Auto` runs plain **with a warning**; no Job Object in v1 (rationale + upgrade path in §5).

Out of scope (explicitly): the daemon-core shell path (`LocalEnvironment::run` in
`crates/engine/daemon-core/src/exec/local.rs`). That is a much larger surface **and it is the
cross-track seam** owned by `contained-root-type` (§8). This track confines only `execute_code`'s
child; shell-tool confinement is future work noted in §7.

## 2. Sandbox abstraction design

A new low-level crate **`crates/substrate/daemon-sandbox`** owns everything that cannot be expressed
as a plain argv prefix (the in-process Landlock+seccomp install, capability probing, and the
posture-decision logic). It has **no `daemon-core` dependency** (avoids a cycle; it is a substrate
primitive) and is the *only* place that carries `unsafe` (the `pre_exec` call). `execute_code` keeps
`#![forbid(unsafe_code)]`.

```
tools/daemon-tool-execute-code  (forbid(unsafe_code))
        │  builds a SandboxSpec (rw=cwd, ro=system+interpreter, network) and asks daemon-sandbox
        │  to (a) resolve the backend for the posture and (b) confine the Command
        ▼
crates/substrate/daemon-sandbox  (localized, documented `unsafe` for pre_exec only)
   ├─ posture + capability probing (bwrap? landlock? seccomp? platform?)
   ├─ linux:  landlock ruleset (built pre-fork) + seccomp BpfProgram (compiled pre-fork),
   │          applied in child via Command::pre_exec (syscalls only, no alloc in child)
   ├─ macos:  build an SBPL profile string → argv prefix `sandbox-exec -p <profile> …`
   └─ windows: fail-closed decision (no confinement handle in v1)
```

Core types (names indicative):

```rust
/// What the child is allowed to touch. Declarative; platform backends translate it.
pub struct SandboxSpec {
    pub rw_paths: Vec<PathBuf>,   // read/write/create allowed roots (the run CWD)
    pub ro_paths: Vec<PathBuf>,   // read+execute allowed roots (interpreter + system libs)
    pub allow_network: bool,      // false ⇒ deny INET/INET6 sockets
    pub tmp_dir: PathBuf,         // child TMPDIR, redirected under rw scope (see §4 fs policy)
}

/// Host capabilities, probed once per process (cached like today's `bwrap_usable`).
pub struct Capabilities { pub bwrap: bool, pub landlock: bool, pub seccomp: bool }
pub fn probe() -> Capabilities;

/// The resolved backend for one run.
pub enum Backend { Bwrap, LandlockSeccomp, SandboxExec, Plain }

/// Posture-driven selection. `Require` returns Err if the strongest usable backend is `Plain`.
pub fn resolve(policy: Posture, caps: &Capabilities) -> io::Result<Backend>;

/// Apply in-process confinement to a not-yet-spawned child. No-op for argv backends
/// (Bwrap/SandboxExec/Plain). For LandlockSeccomp: builds the ruleset + BPF in the PARENT,
/// then installs them in the child via `pre_exec` (fail-closed: the closure returns Err if
/// install fails, so the spawn fails rather than running unconfined).
#[cfg(unix)]
pub fn confine_command(cmd: &mut tokio::process::Command, backend: Backend, spec: &SandboxSpec) -> io::Result<()>;
```

`Posture` mirrors the tool's `SandboxPolicy` (§3); daemon-sandbox takes its own copy so it stays
independent of the tool crate. Argv-prefix backends (bwrap today; macOS sandbox-exec) continue to be
built in `execute_code`'s `sandbox::argv` — daemon-sandbox does not own argv construction, only the
in-process install and the decision logic. This keeps the diff additive and the existing bwrap argv
untouched.

### Why `pre_exec` and why it is allocation-safe in the child

Landlock and seccomp restrict the **calling thread** and are inherited across `execve`. Applied in
the parent they would jail the daemon itself, so they must be installed in the child *after fork,
before exec* — i.e. in `Command::pre_exec`. Post-fork in a multi-threaded process only
async-signal-safe work is sound, so:

- The Landlock `RulesetCreated` (opening the rw/ro path fds) is built **in the parent** and moved
  into the closure; the child only calls `restrict_self()` (prctl + `landlock_restrict_self` syscall).
- The seccomp `BpfProgram` (a `Vec<sock_filter>`) is compiled **in the parent** and moved in; the
  child only calls `seccompiler::apply_filter(&prog)` (a `prctl`; the kernel copies the program in,
  no child-side allocation) after setting `PR_SET_NO_NEW_PRIVS`.

The child closure therefore performs **syscalls only** — no allocation, no locks. (An earlier
alternative — a self-re-exec launcher stub — was rejected as more cross-cutting than a contained
`pre_exec`.)

## 3. `SandboxPolicy` posture — semantics & where chosen

Rename the tool's policy enum to the requested posture, preserving old config via serde aliases so
existing `sandbox = "bwrap"` / `"none"` TOML keeps working (this is tool config, **not** a CDDL wire
type — no conformance impact):

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SandboxPolicy {
    /// Strongest available kernel backend for this platform; if none is usable, run PLAIN
    /// (unconfined) after emitting a warning. The default.
    #[default]
    Auto,                                   // wire: "auto"
    /// Require a kernel backend. If none is usable, FAIL the call (never a silent unconfined run).
    #[serde(alias = "bwrap")]
    Require,                                // wire: "require" (old "bwrap" still parses)
    /// Never sandbox — explicit, high-friction operator choice. Unconfined subprocess.
    #[serde(alias = "none")]
    Plain,                                  // wire: "plain" (old "none" still parses)
}
```

Backend preference per platform under `Auto`/`Require`:

| Platform | order tried                          | `Auto` if none usable | `Require` if none usable |
|----------|--------------------------------------|-----------------------|--------------------------|
| Linux    | bwrap → **landlock+seccomp**         | plain + warn          | **error (fail closed)**  |
| macOS    | sandbox-exec                         | plain + warn          | **error (fail closed)**  |
| Windows  | *(none in v1)*                       | plain + warn          | **error (fail closed)**  |

**Where the posture is chosen:** unchanged from today — the operator sets it in
`[execute_code].sandbox` (`bins/daemon/src/config.rs`, `ExecuteCodeConfig::sandbox`, default `Auto`),
threaded into `ExecuteCodeSettings.sandbox` in `bins/daemon/src/main.rs::build_execute_code_tool`.
The tool evaluates it per-run in `sandbox::resolve` (now delegating the decision to
`daemon_sandbox::resolve(posture, &probe())`). No new config surface; only the variant labels and the
"require any backend / fail closed" semantics change.

## 4. How the `execute_code` plain fallback becomes gated

`ExecuteCodeTool::execute` (`src/lib.rs`) already calls `sandbox::resolve(self.settings.sandbox)`
before staging. The change:

1. `sandbox::resolve` → returns a `daemon_sandbox::Backend` (was `SandboxKind{Bwrap,Plain}`), driven
   by the posture table above. Under Linux `Auto` on a userns-less host it now returns
   `LandlockSeccomp` instead of `Plain`. Under `Require` with no usable backend it returns `Err`,
   which `execute` already maps to a setup-error outcome (no staging dir, no process) — the existing
   `SandboxPolicy::Bwrap` "unavailable" test path.
2. `execute` builds a `SandboxSpec` from the resolved run: `rw = [cwd]`,
   `ro = [interpreter dir + fixed system list]`, `allow_network = (network == Shared)`,
   `tmp_dir = cwd/.tmp`.
3. `run_staged`/`run_subprocess` (`src/exec.rs`) gain an `Option<(Backend, SandboxSpec)>` parameter.
   For `LandlockSeccomp` it calls `daemon_sandbox::confine_command(&mut cmd, backend, &spec)` before
   `spawn()`. For `Bwrap`/`SandboxExec`/`Plain` the argv already carries the wrapper (or nothing) and
   `confine_command` is a no-op.
4. `argv` (`src/sandbox.rs`): the `LandlockSeccomp` backend uses the **plain** `[interp, script]`
   argv (confinement is applied in-process, not via a wrapper binary); a new **macOS** arm emits
   `sandbox-exec -p <profile> interp script`.
5. The result detail gains a `backend: &str` label (`"bwrap"|"landlock"|"sandbox-exec"|"plain"`)
   alongside the existing `sandboxed: bool` (true for any non-plain backend). Detail is opaque JSON
   (rendered by `kind`), **not** a CDDL wire type — no conformance change.

Net: under the default `Auto`, a host that previously ran `execute_code` unconfined now runs it under
Landlock+seccomp; `Require` refuses instead of degrading; `Plain` is the only path to an unconfined
run and is a named, deliberate choice.

### Linux fs/network policy detail

- **Landlock (fs):** default-deny once an access type is handled. Grant read+execute on the same
  system set bwrap binds read-only (`/nix/store`, `/usr`, `/bin`, `/lib`, `/lib64`,
  `/run/current-system/sw`, `/etc/ssl`, `/etc/pki`, `/etc/resolv.conf`, `/etc/nsswitch.conf`,
  `/etc/static`) plus the resolved interpreter's directory; read on `/dev/urandom`, `/dev/random`,
  `/dev/zero`; read+write on `/dev/null`; **read+write+create on the run CWD only**. `TMPDIR` is set
  to a dir under the CWD so Python's `tempfile` stays in scope (Landlock cannot synthesize a private
  tmpfs — see coverage §7). Paths absent on the host are skipped (bwrap parity via `-try`).
- **seccomp (network):** when `allow_network == false`, a targeted filter (default `Allow`) returns
  `EACCES` for `socket(2)`/`socketcall` where domain arg == `AF_INET`/`AF_INET6` (and a small always-
  deny set, e.g. `ptrace`). This is a **targeted denylist, not a minimal-syscall allowlist** — a
  strict allowlist for arbitrary Python is fragile and high-false-positive; this design blocks
  network egress and process poking without breaking legitimate interpreters. Coverage stated
  honestly in §7. When `allow_network == true`, no seccomp network rule is installed.

## 5. Non-Linux / no-kernel-support fallback (must build + degrade explicitly)

The `daemon` binary (hence `daemon-tool-execute-code`) **cross-compiles to `x86_64-pc-windows-gnu`**
(flake `daemon-windows`, stub-worker lane) and builds natively on macOS. Therefore:

- Landlock/seccompiler are pulled **only** under
  `[target.'cfg(target_os = "linux")'.dependencies]` in `daemon-sandbox` — the macOS and Windows
  builds never see them.
- `daemon-sandbox` compiles on every target: non-Linux `confine_command` is a no-op for argv/plain
  backends; macOS adds the `sandbox-exec` argv path (no crate dep — it's an OS binary).
- **Windows (v1): fail-closed, no Job Object.** `resolve` returns no kernel backend on Windows, so
  `Require` errors and `Auto` runs `Plain` with a warning. Rationale: the v1 Windows lane ships
  **stub-worker daemon + CLI only** (no engine workers, so `execute_code` is effectively unused
  there), and a correct Job-Object + restricted-token confinement is substantial `unsafe` FFI
  (`windows-sys`, assign-to-job-before-resume with tokio's spawned child) for a path that does not
  run in v1. Fail-closed fully satisfies the hard invariant ("never silently unconfined under
  `Require`"). A follow-up can add a real Job Object behind
  `[target.'cfg(windows)'.dependencies] windows-sys` without changing the posture surface — flagged
  as the documented upgrade path, **not** implemented in this track unless you direct otherwise.

## 6. Tests — bug-reproducing, written FIRST

The dev/CI kernel is Linux 6.19, so the Landlock+seccomp tests actually execute here (they still
runtime-probe and skip cleanly on a kernel/host without support, matching the existing bwrap-test
pattern). The core security tests live in **`daemon-sandbox`** so they exercise the kernel backend
directly (bypassing `execute_code`'s bwrap-first preference), confining a small child
(`/bin/sh -c …` or a tiny probe) and asserting the escape fails:

1. `landlock_blocks_read_outside_scope` *(Linux, probe-gated)* — confine a child whose `rw`/`ro` set
   excludes a planted secret in a sibling temp dir; the child's `open()`/`cat` of it fails
   (`EACCES`). **Reproduces the bug:** the same child *without* confinement (today's `Plain`) reads
   it successfully — asserted as the control in the same test.
2. `landlock_blocks_write_outside_workspace` *(Linux, probe-gated)* — a write to `../escape` outside
   the rw root is denied; the outside target is untouched.
3. `seccomp_blocks_inet_socket` *(Linux, probe-gated)* — with `allow_network=false`, a child that
   calls `socket(AF_INET, …)` fails; the control (`allow_network=true`) succeeds.
4. `resolve_require_fails_closed_without_backend` *(all platforms, deterministic)* — pure posture
   logic over an injected `Capabilities{bwrap:false, landlock:false, seccomp:false}`:
   `resolve(Require, …)` → `Err`; `resolve(Auto, …)` → `Plain`; with `landlock:true`,
   `resolve(Require)` → `LandlockSeccomp`. No kernel dependency — this is the fail-closed guarantee.
5. `plain_backend_is_a_noop` *(all platforms)* — `confine_command(Plain, …)` leaves the child
   unrestricted (documents that `Plain` is genuinely unconfined).

At the `execute_code` integration level (`tools/daemon-tool-execute-code/tests/execute_code.rs`):

6. Rename existing `SandboxPolicy::Bwrap`→`Require`, `SandboxPolicy::None`→`Plain` (mechanical).
7. `plain_policy_reports_unconfined` — `Plain` → detail `sandboxed==false`, `backend=="plain"`.
8. `auto_on_this_host_is_confined` *(Linux, python+kernel-probe-gated)* — `Auto` → `sandboxed==true`
   with `backend` ∈ {`bwrap`,`landlock`}, and a script that reads a planted out-of-workspace secret
   comes back without the secret (end-to-end confinement, whichever backend won).
9. A `sandbox.rs` `#[cfg(test)] mod tests` unit test calling `pub(crate) resolve` to assert `Require`
   maps a no-backend host to a setup error (the tool-level fail-closed path), independent of #4.

## 7. Residual coverage (honest)

- **Landlock is not a namespace.** The in-process backend provides fs *containment* and network
  *egress deny*, but **not** bwrap's private `/tmp`, private `/proc`, or pid/ipc/uts isolation: the
  child sees the real `/proc` and (unless `TMPDIR` redirection holds) a shared `/tmp`. It is strictly
  stronger than today's unconfined `Plain` and is the documented fallback where userns is
  unavailable; bwrap remains the preferred Linux backend under `Auto`.
- **seccomp is a targeted denylist, not a syscall allowlist.** It blocks INET/INET6 sockets and a
  small dangerous set; it does not minimize the syscall surface (deliberate, to not break Python).
- **Content-swap TOCTOU** on the interpreter binary between resolve and exec is out of scope (left to
  Phase 3 artifact-provenance, as the Cluster B fingerprint note already states).
- **Windows** has no kernel confinement in v1 (fail-closed only) — see §5.
- **Shell tool** (`LocalEnvironment::run`) is unconfined and out of scope (cross-track seam §8);
  extending this sandbox to it is future work.
- **Landlock ABI degradation:** on a kernel below the required Landlock ABI the parent probe reports
  `landlock:false`, so `Require` fails closed and `Auto` falls through — no partial/misleading
  enforcement is silently accepted.

## 8. Cross-track coordination (`contained-root-type` sibling — DECONFLICT)

The `contained-root-type` track owns the **FS-containment** path; this track owns the
**PROCESS-sandbox** path. Seam kept clean by preferring a new crate + additive edits.

**Files this track will NOT touch** (owned by contained-root):
- `crates/engine/daemon-core/src/exec/mod.rs` — `contain`, `open_read_guarded`, `open_write_guarded`,
  `reject_symlink_final*`, and the future `ContainedRoot`. **Zero edits from this track.**
- `crates/engine/daemon-core/src/exec/local.rs` — `LocalEnvironment` and its `run`/`read`/`write`/
  `list`. **Zero edits.**
- `tools/daemon-tool-execute-code/src/python.rs` — `resolve_interpreter`/`candidate_paths` (their
  venv/fs trust path). **Zero edits.**

**Files/functions this track WILL modify** (flagged for deconfliction):
- NEW `crates/substrate/daemon-sandbox/**` — new crate, no overlap.
- `tools/daemon-tool-execute-code/src/sandbox.rs` — `enum SandboxKind` (+ Landlock/SandboxExec),
  `fn resolve`, `fn argv` (macOS arm), new spec/probe glue. *Not touched by contained-root.*
- `tools/daemon-tool-execute-code/src/exec.rs` — `fn run_subprocess` signature (+ confinement param)
  and the pre-spawn confinement call. *Not touched by contained-root.*
- `tools/daemon-tool-execute-code/src/lib.rs` — `enum SandboxPolicy` (variant rename+aliases),
  `struct ExecDetail`/`struct Executed` (add `backend`), **`ExecuteCodeTool::execute`** and
  **`ExecuteCodeTool::run_staged`** (build `SandboxSpec`, thread backend through), `success_outcome`.
  ⚠️ **`execute`/`run_staged` are the only plausible collision points** if contained-root also edits
  `lib.rs` for staging/CWD containment. Recommend: contained-root keeps its `lib.rs` changes to the
  interpreter/venv/`contain`-adjacent lines and leaves the process-spawn portion of
  `execute`/`run_staged` to this track; I will keep my edits to those two fns narrow and clearly
  scoped to backend/spec threading.
- `tools/daemon-tool-execute-code/tests/execute_code.rs` — variant renames + new tests.
- `tools/daemon-tool-execute-code/Cargo.toml` — add `daemon-sandbox` dep.
- `bins/daemon/src/config.rs` — doc comment on `ExecuteCodeConfig::sandbox` variants (default `Auto`
  unchanged; no functional edit). `bins/daemon/src/main.rs` — no change expected (`ec.sandbox`
  already passed through).
- Root `Cargo.toml` — add `daemon-sandbox` path dep and workspace deps `landlock`, `seccompiler`
  (referenced only by daemon-sandbox under `cfg(target_os="linux")`). No `windows-sys` (Windows is
  fail-closed).

Merge order per the plan: contained-root (TA) merges first; this track (TC) is listed as independent
of it — with the seam above there should be no rebase conflict beyond the two flagged `lib.rs` fns.

## 9. Dependencies & `cargo deny`

Added to `[workspace.dependencies]`, consumed by `daemon-sandbox` **only under
`[target.'cfg(target_os = "linux")'.dependencies]`**:

- `landlock = "0.4.5"` — license **MIT OR Apache-2.0** (allowed). Safe wrapper over Landlock
  syscalls; MSRV 1.71 (< workspace 1.93). Linux-only.
- `seccompiler = "0.5.0"` — license **Apache-2.0 OR BSD-3-Clause** (both allowed). `apply_filter`
  copies the program into the kernel (no child-side alloc). Linux-only.
- `libc` (already a workspace dep) — `PR_SET_NO_NEW_PRIVS`, `AF_INET`/`AF_INET6` consts. Linux path.
- `tokio` (already a workspace dep) — `daemon-sandbox` takes `&mut tokio::process::Command` on unix
  to install `pre_exec`.

Both new crates are permissively licensed and on crates.io (no new git/unknown source), so
`cargo deny check` (advisories / licenses / bans / sources) stays green. Transitive deps
(`enumflags2`, `thiserror`, `bitflags`, …) are already permissive; verified at implement time. No
new `deny.toml` `ignore` entries anticipated.

## 10. Exact gate (from worktree root, in the devShell)

```
nix develop --command cargo fmt --all -- --check
nix develop --command cargo clippy --workspace --all-targets -- -D warnings
nix develop --command cargo test --workspace --no-fail-fast
nix develop --command cargo deny check
```

No CDDL/wire type changes → `cargo test -p daemon-api --features arbitrary` not required.
Machine-load note: if `bins/daemon/tests/host_launch.rs` fails under the parallel `--workspace` run,
re-run isolated `cargo test -p daemon --test host_launch -- --test-threads=2` before treating as
real; known pre-existing flakes (`detached_delegation` ×2, `process_notify` store-seam) are not this
track's regressions. Tests first, minimal hunks; do not merge, do not remove the worktree.

## 11. SCOPE ADDITION — execute_code staging-open containment (handed over from contained-root-type)

`execute_code` now owns **all** execute_code edits, including the host-side fs-containment of its own
staging opens (previously ambiguous between tracks; `contained-root-type` explicitly handed it here).

**The gap.** In `ExecuteCodeTool::execute`/`run_staged` (`src/lib.rs` ~290/321) the tool derives a
staging path from the workspace root and opens it **in the daemon process**, not in the child:

```rust
let staging = ws_root.join(".execute_code").join(new_run_id());
tokio::fs::create_dir_all(&staging).await?;      // ~290 — host-side
...
let script = staging.join("script.py");
tokio::fs::write(&script, code).await?;           // ~321 — host-side
```

The child-process Landlock (§4) confines the *spawned interpreter*, not these daemon-side opens. So a
symlink planted at `<ws_root>/.execute_code` (e.g. by earlier attacker-influenced workspace content on
a `Bound` root) would be followed by `create_dir_all`/`write`, letting the staging write escape the
workspace — a traversal vector Landlock does not cover.

**Fix (pending merged API).** Route these opens through the `ContainedRoot` primitive that
`contained-root-type` is landing in daemon-core. Per its published plan, `ContainedRoot` is
constructed from `ws_root` (opens a root fd) and exposes open-relative `create_dir`/`write` operating
under that fd via `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` — so a symlinked `.execute_code`
component is refused rather than followed. `execute`/`run_staged` construct a `ContainedRoot` from
`ws_root` and perform the `.execute_code/<run_id>` `create_dir` + `script.py` `write` through it;
best-effort cleanup (`remove_dir_all`) likewise routes through the root fd.

**Dependency & merge order.** This makes the track **depend on `contained-root-type`'s merged
`ContainedRoot`**, so it **merges after** it. This worktree branched from `integration` before
contained-root lands, so:

- The Linux Landlock/seccomp backend, the `Auto`/`Require`/`Plain` posture, and the `execute`/
  `run_staged` **spawn wiring** are implemented and gated now (they do not depend on `ContainedRoot`).
- The staging-open containment is implemented against the API shape published in contained-root's
  plan; on rebase onto `integration` (after contained-root merges) it compiles against the real type.
- **API sufficiency check:** the shape needed is `ContainedRoot::from_root(&Path) -> io::Result<Self>`
  plus relative `create_dir_all(&self, rel) -> io::Result<()>`, `write(&self, rel, &[u8]) ->
  io::Result<()>`, and `remove_dir_all(&self, rel) -> io::Result<()>`. If the published API lacks a
  recursive `create_dir_all` or a `remove_dir_all` over the root fd, flag to relay to contained-root
  before it finalizes.

**Staging-open test (pending, Linux-gated):** a symlinked `<ws_root>/.execute_code` pointing outside
the workspace is **not** followed — staging fails/refuses and the outside target is untouched (the
same shape as the existing `read_rejects_symlinked_final_component` guard tests). Held until the real
`ContainedRoot` is available so the test exercises the merged API, not a placeholder.

**Files (all within the owned execute_code surface):** `tools/daemon-tool-execute-code/src/lib.rs`
(`execute`/`run_staged` staging opens) and its test file. Still **no** edits to
`crates/engine/daemon-core/src/exec/mod.rs`, `exec/local.rs`, or `tools/.../python.rs` — those remain
contained-root's; this track only *consumes* the exported `ContainedRoot`.
