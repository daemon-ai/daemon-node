#!/usr/bin/env bash
# Guest-build reproducibility shim (daemon-vhc C2 lead-in): make the SDK-linking guest `.wasm`
# byte-identical across checkout paths (worktrees / machines / CI).
#
# WHY. cargo derives each *path* package's crate-disambiguator (`-C metadata`) from the package's
# absolute manifest directory (workspace-external path deps are hashed by absolute path, cargo
# `PackageId::stable_hash`). rustc folds that disambiguator into the crate's `StableCrateId`, which
# seeds symbol-hash / `DefPathHash` ordering — so the *same* source compiled under two checkout
# paths links its functions/types/elems in a different order and the `.wasm` bytes differ, even
# though `--remap-path-prefix` already normalises every embedded path *string*. Registry crates are
# unaffected (cargo hashes them by their immutable registry id, not a path), so only the local
# `crates/vhc/*` path packages drift. This shim pins their `-C metadata` to a value derived solely
# from the crate name, which is unique within a guest's link closure — removing the only remaining
# path input while leaving `-C extra-filename` (cargo's on-disk output naming) untouched.
#
# This is wired via `crates/vhc/guests/.cargo/config.toml` (`build.rustc-wrapper`), so EVERY cargo
# invocation in the guests workspace — `xtask build-guests`, the host test harnesses' `ensure_built`,
# and manual builds — goes through it identically, keeping the committed `guests/guests.blake3`
# reproducible with no per-call-site coordination. It is intentionally a no-op for every crate whose
# source is not under `crates/vhc/` (registry deps, the sysroot, build scripts, the `rustc -vV`
# probe): those already hash stably.
set -euo pipefail

rustc="$1"; shift
args=("$@")

is_local=0
crate_name=""
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    */crates/vhc/*) is_local=1 ;;
    --crate-name) crate_name="${args[i + 1]:-}" ;;
  esac
done

# getrandom backend selection (C1 compute lead-in): the Burn tree the compute guests link pulls
# `rand` -> `getrandom`, whose wasm32-unknown-unknown build REFUSES to compile without an explicit
# backend choice. The right backend for a wasmtime guest (no JS host!) is `custom` — the
# `__getrandom_v03_custom` definition lives in `daemon-vhc-sdk-compute` (deterministic fill;
# module RNG policy is `sys@2::rng_seed`, never ambient entropy). The cfg is consumed only by
# getrandom's own build, so it is appended here — the one seat EVERY guest-workspace rustc
# invocation passes through (xtask build-guests, the host test harnesses' ensure_built, manual
# builds), exactly the no-per-call-site-coordination argument the metadata pin above uses. Scoped
# to the getrandom crate on wasm32 so no other crate sees an unexpected cfg.
if [[ "$crate_name" == "getrandom" ]]; then
  for a in "${args[@]}"; do
    if [[ "$a" == "wasm32-unknown-unknown" ]]; then
      args+=("--cfg" 'getrandom_backend="custom"')
      break
    fi
  done
fi

if [[ "$is_local" == 1 && -n "$crate_name" ]]; then
  newargs=()
  for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[i]}" == "-C" && "${args[i + 1]:-}" == metadata=* ]]; then
      newargs+=("-C" "metadata=vhcpin-${crate_name}")
      i=$((i + 1))
      continue
    fi
    newargs+=("${args[i]}")
  done
  exec "$rustc" "${newargs[@]}"
fi

exec "$rustc" "${args[@]}"
