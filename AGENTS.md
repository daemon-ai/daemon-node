# AGENTS.md — daemon-node (Rust)

Nix-managed workspace. There are NO host tools — run cargo inside the devShell:
`nix develop --command <cmd>` (or the superproject `just` recipes).

## Required gate before finishing

- `nix develop --command cargo fmt`                                    # leave `cargo fmt --check` clean
- `nix develop --command cargo clippy --workspace --all-targets -- -D warnings`
- `nix develop --command cargo deny check`
- `nix develop --command cargo test --workspace`                       # or `cargo nextest run`

Never bypass the pre-commit hook (no `git commit --no-verify`).

## Lint policy

- Lint levels are workspace-wide via `[workspace.lints]` in the root `Cargo.toml`; member crates
  opt in with `[lints]\nworkspace = true`. New crates MUST add that stanza.
- `todo!`, `unimplemented!`, and `dbg!` are denied — do not leave them in committed code.
- Prefer fixing a lint over silencing it. If an `#[allow(...)]` is truly warranted, scope it to
  the item and add a one-line reason; never blanket-allow at module or crate level.
- Don't add dependencies you don't use — `cargo machete` (`just audit-cleanup`) will flag them.

## Features / engines

- The default workspace gate builds default features only. The `llama` / `mistralrs` / `hyperon`
  engine lanes need native libs and are deliberately separate flake outputs (e.g.
  `nix build .#daemon-infer-llama`) — do NOT switch the gate to `--all-features`.
- Miri (UB over the FFI/codec `unsafe` surface) and cargo-fuzz use the nightly shell:
  `nix develop .#nightly` (see `just miri` / `just fuzz`).
