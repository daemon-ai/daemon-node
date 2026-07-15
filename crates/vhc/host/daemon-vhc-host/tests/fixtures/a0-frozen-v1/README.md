# A0 frozen v1 compatibility fixture (refactor §5 A0)

The standing pre-refactor regression: a content-addressed bundle capturing one deterministic
CPU run of the **pre-Phase-0-rename v1 five-phase driver** over the **immutable pre-refactor
`tiny-llama` module bytes**. The named tier-1 replay test
(`daemon-vhc-host/tests/a0_frozen_fixture.rs`, wired into `xtask swarm-ci-det`) reloads this
bundle on the current tree and must reproduce the recorded transcript **bit-for-bit under the v1
driver** — every later phase keeps it green until the Phase E v1 sunset, after which the same
fixture is kept with its expected result flipped to a clean `AbiUnsupportedMajor` refusal
(decisions D5; never deleted).

## Bundle contents

| File | What | blake3 |
|---|---|---|
| `tiny_llama.pre-refactor.wasm` | The immutable pre-refactor module bytes (143958 bytes), read from the pre-refactor capture worktree's built guests — **never recompiled** | `d9aa630f8ab4fa28cc1d1085dc4de88154f9f019e806430fbb496a2ad770634c` |
| `envelope.signed.cbor` | The exact schema-major-1 `SignedEnvelope` wire bytes (canonical CBOR; envelope carries `[run]`/`[experiment]`/`[artifacts]`/`[data]`/`[requirements]`/`[phases]`, pins the module + corpus by blake3, signed by the deterministic test key `SigningKey::from_bytes([7;32])`) | `f5d23e18961d064c0f3d63c7739f9cc913bead6f000aaad78d47d87a4e8fd990` (envelope hash `3820fab94404bb3381271f38330a72b3d733120b290f778dfcf19e4ad904d168`) |
| `expected.json` | Every pin (hashes, window, batch-derivation rule, run shape) + the expected transcript: per-round payload blake3 + post-ingest det-lane state digest | (self-describing) |
| `capture/` | The capture crate — the documented, re-runnable capture command (below) | — |

**Corpus input** (pinned by hash, bytes vendored once in this repo at
`crates/vhc/host/daemon-vhc-session/tests/fixtures/tinystories/` — byte-identical in the
pre-refactor tree, verified at capture): `shard-0000.bin`
(`96da080176dabf76a9321ec4df2332f1089b7c77dfbdea39d1a3894186393ae8`, 262144 u16-le tokens) under
`manifest.json` (`blake3` recorded in `expected.json`). The replay test re-verifies both hashes
before use, so the reference stays content-addressed without duplicating 2 MiB of shards.

**Input derivation (the pinned seed/window rule)**: for batch index `b = round * H + step`,
token `i` (of 2 sequences × 8 tokens) is `raw[(4096 + b*16 + i) % 262144] % 64` over the
u16-le-decoded shard-0 stream. Run shape: 4 rounds, `H = 3` (the module manifest's
`steps_per_round` for the pinned sparse_loco tiny config), self-ingest of one payload per round
staged as `PeerId([1;32])`, CPU backend (`BackendKind::Cpu`, `EngineConfig::default()`).

## Provenance note on the module bytes

Guest `.wasm` builds are byte-reproducible **within** one checkout but not across
worktrees/machines (cargo's path-keyed `-C metadata` reorders codegen; see
`wasm_backend_determinism.rs`). The committed `guests.blake3` of any tree is therefore an
advisory record of that checkout's build, and the bytes frozen here are the built artifact of
**the designated pre-refactor capture worktree** (`swarm-p3-integration` @ `6706fda`) whose
regenerated manifest recorded exactly this hash. The bundle is self-contained: the fixture's
authority is its own recorded blake3 + transcript, not any tree's `guests.blake3`.

## Capture command (regenerates the whole bundle from the pre-refactor tree)

```sh
# Requires the pre-refactor worktree at ../swarm-p3-integration relative to this checkout's
# parent (i.e. both worktrees side by side), checked out at 6706fda with its guests built
# (cargo run -p xtask -- build-guests there, once). Builds and runs in the PRE-REFACTOR
# devShell + target dir; makes no modification to that tree; writes the bundle here.
cd ../swarm-p3-integration && nix develop --command \
  cargo run -j5 --release --manifest-path \
  ../vhc-a0/crates/vhc/host/daemon-vhc-host/tests/fixtures/a0-frozen-v1/capture/Cargo.toml
```

(Adjust the two worktree names to your checkout layout; the capture crate resolves the
pre-refactor tree by relative path — see `capture/Cargo.toml`.)

Captured 2026-07-15; re-running the capture reproduces `expected.json` bit-for-bit.
