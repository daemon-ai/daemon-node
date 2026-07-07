# AGENTS.md — daemon-node (Rust)

Nix-managed workspace. There are NO host tools — run cargo inside the devShell:
`nix develop --command <cmd>` (or the superproject `just` recipes).

## Required gate before finishing

- `nix develop --command cargo fmt`                                    # leave `cargo fmt --check` clean
- `nix develop --command cargo clippy --workspace --all-targets -- -D warnings`
- `nix develop --command cargo deny check`
- `nix develop --command cargo test --workspace`                       # or `cargo nextest run`
- `nix develop --command cargo test -p daemon-api --features arbitrary` # CDDL conformance (see below)

Never bypass the pre-commit hook (no `git commit --no-verify`).

## daemon-node is authoritative — clients stay thin

This daemon is the single authority for domain state, business logic, validation, persistence,
and orchestration. Every client (`daemon-app` GUI / TUI / WASM, `daemon-cli`) is a thin renderer
of node state that sends intents back — so design the API accordingly:

- New behavior lands HERE first, behind `ApiRequest` / `ApiResponse`, then clients consume it.
  Never plan a feature that needs a client to compute domain results locally.
- Responses carry decisions, not raw material for clients to re-derive: expose computed/derived
  state (statuses, aggregates, eligibility, ordering) instead of expecting clients to recompute
  it. If two clients would each re-implement the same rule, that rule belongs in the node.
- Requests are intents ("do X"), and the node validates them fully server-side; client-side
  checks are UX sugar, never the enforcement point.

## Wire contract (CDDL) — keep it in lockstep with the Rust types

[`crates/contracts/daemon-api/daemon-api.cddl`](crates/contracts/daemon-api/daemon-api.cddl) is the
**single, authoritative** wire contract: one hand-authored, zcbor-generatable file that both documents
every CBOR shape and generates the client C codec. The Rust serde types are the source of truth; the
CDDL must mirror them exactly and stay clean.

- **Any change to a Rust type exposed over the wire MUST be reflected in the CDDL.** "Exposed over the
  wire" = `ApiRequest` / `ApiResponse` and every type reachable from them across `daemon-api`,
  `daemon-common`, and `daemon-protocol` (adding/removing/renaming/retyping a field or enum variant,
  or adding a new variant). Update `daemon-api.cddl` in the same change, then `just update-codec` to
  regenerate the vendored codec.
- **No stale or loose rules.** Remove rules that are no longer reachable from `api-request` /
  `api-response` (unreachable rules are unvalidated and silently drift). Prefer concrete shapes over
  `any`; only use `any` for genuinely large/opaque payloads and say why in a comment.
- **Authoring rules (so zcbor generates cleanly):** quote every map key (`"key":`), give each union
  arm its own named rule, label same-type tuple elements (`[k: tstr, v: tstr]`), and suffix a rule
  with `-t` when its name would otherwise collide with a map member's coder (e.g. map `origin`.`scope`
  vs rule `origin-scope` → `origin-scope-t`). Rust `Vec<u8>`/`[u8; N]` serialize as a CBOR array of
  ints (`byte-array`), not `bstr` (no `serde_bytes` is used).
- **The gates prove it (drift = failing test):** `cargo test -p daemon-api --test conformance`
  (cddl-cat, real fixtures + negatives), `cargo test -p daemon-api --features arbitrary`
  (proptest: arbitrary values across every variant validate against the CDDL), `cargo run -p xtask --
  verify-codec` (zcbor C decoder vs ciborium), and `cargo run -p xtask -- cddl` (variant parity). If
  you change a wire type without updating the CDDL, these fail. `just conformance` runs them.

## Lint policy

- Lint levels are workspace-wide via `[workspace.lints]` in the root `Cargo.toml`; member crates
  opt in with `[lints]\nworkspace = true`. New crates MUST add that stanza.
- `todo!`, `unimplemented!`, and `dbg!` are denied — do not leave them in committed code.
- Prefer fixing a lint over silencing it. If an `#[allow(...)]` is truly warranted, scope it to
  the item and add a one-line reason; never blanket-allow at module or crate level.
- Don't add dependencies you don't use — `cargo machete` (`just audit-cleanup`) will flag them.

## Versioning

`VERSION` (repo root, clean SemVer) is the source of truth; `[workspace.package].version` mirrors it
because Cargo needs a literal (every member inherits via `version.workspace = true`; internal path
deps carry no `version`). `daemon-common`'s `build.rs` appends a git build-metadata suffix
(`+<n>.g<hash>[.dirty]` from `git describe`, or the Nix-injected `DAEMON_BUILD_ID`) to form
`daemon_common::VERSION` — what `daemon --version` and `daemon-cli --version` print. Do NOT bake a
version literal anywhere else.

Bump in the monorepo with `just set-version daemon-node X.Y.Z` (writes `VERSION` and syncs
`Cargo.toml`); standalone, edit `VERSION` and `[workspace.package].version` together. The
`just check-version` gate (part of `just lint`) fails if they drift.

## Features / engines

- The default workspace gate builds default features only. The `llama` / `mistralrs` / `hyperon`
  engine lanes need native libs and are deliberately separate flake outputs (e.g.
  `nix build .#daemon-infer-llama`) — do NOT switch the gate to `--all-features`.
- The devShell exports a prebuilt shared llama.cpp (`LLAMA_PREBUILT_DIR`, including `libmtmd`),
  so `cargo build -p daemon-infer --features llama,mtmd,dynamic-link` links in seconds without a
  cmake step. If llama.cpp starts compiling from source inside the devShell, the prebuilt env
  wiring is broken — stop and investigate rather than waiting it out.
- The engine flake outputs keep their heavy deps (llama.cpp / candle / hyperon) in per-lane
  cached `buildDepsOnly` layers: a workspace source change rebuilds only the leaf crates, so
  don't avoid these lanes on cost grounds.
- Miri (UB over the FFI/codec `unsafe` surface) and cargo-fuzz use the nightly shell:
  `nix develop .#nightly` (see `just miri` / `just fuzz`).
