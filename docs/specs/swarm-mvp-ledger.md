# Swarm-training MVP — program ledger

Wave-0 scaffold coordination record for the daemon swarm-training MVP. This is the single source
of truth for the branch map, lane file-ownership, and the frozen-file rule. Lane agents: read this
before you touch anything.

## Base + branch map

- **Repo:** `daemon-node` (this is the Rust backend submodule; standalone checkout).
- **Base commit:** `0dbd720` (`0dbd7208826cdfafbc7214713ef38e7d2c51d621`,
  `merge(mirror/nv): WireVersion 39 — rungs 1+2+3 sealed (NV)`).
- **Trunk:** `integrations/swarm` — the integration branch. Wave-0 scaffold lands here (the commit
  list below). This is the merge target for every lane.
- **Lanes (branch off the Wave-0 scaffold tip — i.e. the commit that adds THIS file):**
  - `swarm/p1` — **P**rotocol / coordinator / observability lane.
  - `swarm/r1` — **R**untime / transport / node-supervisor lane.
  - `swarm/e1` — **E**ngine / tensor-ABI / guests lane.

All three lanes fork from the same HEAD and integrate back into `integrations/swarm`. Keep lanes on
disjoint file sets (table below) so merges are conflict-free by construction.

## Wave-0 commit list (on `integrations/swarm`, oldest → newest)

| Commit | Subject |
|---|---|
| `cc3df12` | `docs(specs): swarm training architecture + tensor ABI + TDD plan` |
| `de8fd64` | `build(deps): wasmtime + burn + blake3 + xxhash + ed25519 workspace pins` |
| `a621ca5` | `feat(swarm): crate scaffolds for the swarm training stack (spec §10.1)` |
| `26b08a5` | `build(nix): wasm32-unknown-unknown rust-std in devshell` |
| `31170e5` | `feat(xtask): build-guests + guests mini-workspace` |
| `53ddb21` | `build(deps): allow bincode unmaintained advisory (burn transitive)` |
| _(this file)_ | `docs(specs): swarm MVP program ledger` |

## Crate scaffolds (spec §10.1)

Nine empty-but-compiling crates, `crates/*/*`-globbed into the root workspace, each with
`[lints] workspace = true`, a spec-referencing crate doc, and a natural error type (no `todo!()`):

| Crate | Group | Deps (declared) | Lane |
|---|---|---|---|
| `daemon-swarm-proto` | `crates/contracts/` | serde, ciborium (wasm32-clean) | P |
| `det-core` | `crates/contracts/` | none (std only) | E |
| `daemon-train-sdk` | `crates/contracts/` | serde, ciborium | E |
| `daemon-swarm-net` | `crates/swarm/` | proto, tokio, reqwest | R |
| `daemon-swarm-run` | `crates/swarm/` | proto, net, tokio | R |
| `daemon-swarm-coordinator` | `crates/swarm/` | proto, axum, tokio | P |
| `daemon-swarm-observe` | `crates/swarm/` | proto, serde | P |
| `daemon-train` | `crates/coprocessor/` | proto, wasmtime, burn, blake3, xxhash-rust (+ bin) | E |
| `daemon-train-client` | `crates/coprocessor/` | daemon-common, daemon-provision, tokio | R |

## Dependency pins (root `[workspace.dependencies]`, resolved versions)

| Crate | Requirement | Resolved | Features |
|---|---|---|---|
| `wasmtime` | `46` | `46.0.1` | `default-features = false` + `runtime`, `cranelift`, `pooling-allocator` (fuel + epoch-interrupt are `Config` levers, no feature; no WASI) |
| `burn` | `0.21` | `0.21.0` | `default-features = false` + `std`, `ndarray`, `autodiff` (NO GPU backends) |
| `blake3` | `1` | `1.8.5` | default |
| `xxhash-rust` | `0.8` | `0.8.15` | `xxh3` |
| `ed25519-dalek` | `2` | `2.2.0` | default (already in-tree transitively; declared for the swarm lanes, wired in with envelope signing — lane P) |

`ciborium` (`0.2.2`) was already a workspace dep.

### deny.toml change

One documented advisory ignore added (licenses / bans / sources needed **no** changes):

- **`RUSTSEC-2025-0141`** (bincode unmaintained). `bincode 2.0.1` is an **unconditional** dep of
  `burn-core 0.21` (burn's record (de)serialization). It is an unmaintained-status advisory only
  (no CVE); the bincode team ceased development, so the advisory flags the crate itself — **no
  version pin or alternative resolves it** (the task's pin-over-ignore preference does not apply).
  burn is isolated to the `daemon-train` worker fault domain and never linked into the node process
  (§10.1). Matches the existing unmaintained-only ignores (paste / proc-macro-error2 / ttf-parser).
  Re-evaluate when burn moves off bincode.

## flake.nix change

The devShell toolchain now combines the pinned stable toolchain with
`fenix … targets.wasm32-unknown-unknown.stable.rust-std` (`rustToolchainDev` / `craneLibDev`),
scoped to the dev shell so package/build outputs keep the lean host-only toolchain. This is what
lets `xtask build-guests` cross-compile the guest modules in-shell.

## Lane file-ownership (disjoint; keep to your set)

| Lane | Owns (create / edit only within) |
|---|---|
| **P** (`swarm/p1`) | `crates/contracts/daemon-swarm-proto/`, `crates/swarm/daemon-swarm-coordinator/`, `crates/swarm/daemon-swarm-observe/`, `daemon-swarm.cddl` (new, repo root or the api crate per §10.4 authoring rules) |
| **R** (`swarm/r1`) | `crates/swarm/daemon-swarm-net/`, `crates/swarm/daemon-swarm-run/`, `crates/coprocessor/daemon-train-client/`, `tests/daemon-swarm-e2e/` (new), `bins/` |
| **E** (`swarm/e1`) | `crates/contracts/det-core/`, `crates/contracts/daemon-train-sdk/`, `crates/coprocessor/daemon-train/`, `guests/`, the `xtask build-guests` subcommand |

Cross-lane dependency edges are already wired via `[workspace.dependencies]` path entries (a lane
consuming another lane's crate uses `{ workspace = true }` and does **not** edit that crate).

## FROZEN files — single-writer rule (non-negotiable)

After the Wave-0 scaffold (this ledger commit), the following are **FROZEN**. Lane agents MUST NOT
modify them; a change here would collide across all three lanes and break the disjoint-merge
guarantee. Route any needed change through the integration owner as a separate, coordinated commit
on `integrations/swarm`.

- **`Cargo.toml`** (root) — workspace members glob, `exclude = ["guests"]`, `[workspace.dependencies]`, `[workspace.lints]`, profiles.
- **`deny.toml`** — advisory/license/ban/source policy.
- **`flake.nix`** — devShell toolchain + targets.

Adding a **new member crate** to a lane is fine (the `crates/*/*` glob picks it up with no root
edit). Adding a **new third-party dependency** requires a root `Cargo.toml` change → it is NOT a
lane action; request it from the integration owner (who also re-runs `cargo deny check`).

## Notes for lane agents (not obvious from the code)

- **Gates (from the worktree root, all currently green):** `cargo fmt --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, `cargo deny check`,
  `cargo test --workspace`, `cargo run -p xtask -- build-guests`, `typos docs/specs/`. Run
  everything via `nix develop --command …`.
- **reqwest is banned (clippy `disallowed_types`).** `daemon-swarm-net` declares `reqwest` for the
  egress plane, but a raw `reqwest::Client`/`ClientBuilder` fails the clippy gate workspace-wide
  (see `clippy.toml`). Route outbound HTTP through `daemon_egress::EgressClient`. No client is
  constructed in the scaffold.
- **fs / process / env bans** also live in `clippy.toml` (`daemon_core::ContainedRoot`,
  `daemon_provision`/`daemon-processes` for spawns, `EnvPolicy` for child env). `xtask` is
  `#[allow(clippy::disallowed_methods)]` crate-wide, which is why `build-guests` may call
  `Command::new("cargo")` directly.
- **Heavy trees (`wasmtime`, `burn`) build in the default workspace gate here** because
  `daemon-train` declares them directly (scaffold). In the shipped product they move to out-of-gate
  per-backend Nix lanes (§10.1); lane E should preserve that intent (feature-gate / lane-split the
  worker) rather than leaving burn/wasmtime on the default path forever.
- **No GPU backends** are in the graph — burn is `ndarray + autodiff` only. A stray `wgpu-*` set
  sits in `Cargo.lock` as an unreachable orphan (not compiled, not seen by `cargo deny`); do not
  "clean it up" by enabling a wgpu feature.
- **`guests/` is a SEPARATE workspace** (`exclude = ["guests"]`). It links `daemon-train-sdk` by
  path (`../../crates/contracts/daemon-train-sdk`) and builds only via `xtask build-guests` for
  `wasm32-unknown-unknown`. `guests/target/`, `guests/Cargo.lock`, and `*.wasm` are gitignored (lane
  E may choose to commit `guests/Cargo.lock` later for reproducible artifacts).
- **wasm32 rust-std is only in the dev shell.** A bare `cargo build --target
  wasm32-unknown-unknown` outside `nix develop` will fail — always use the dev shell.
- **`daemon-swarm.cddl` does not exist yet** — lane P creates it and (per §10.4) wires a swarm
  parity check; if you extend the `xtask cddl` gate for it, that xtask edit is shared tooling, not a
  frozen-file change, but coordinate it.
- Contracts crates that must stay dependency-lean (`daemon-swarm-proto` serde+ciborium,
  `det-core` std-only, `daemon-train-sdk` serde+ciborium) hand-roll their error types
  (`std::error::Error`) instead of using `thiserror`. Keep them lean — `daemon-swarm-proto` and
  `daemon-train-sdk` are on the `wasm32` path.
