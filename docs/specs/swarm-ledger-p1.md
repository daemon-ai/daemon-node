# Swarm-training MVP — lane P1 ledger (consensus / proto)

Wave-1 working record for lane **P1** (`swarm/p1`). Scope is the consensus/proto contract in
`crates/contracts/daemon-swarm-proto` plus its new CDDL file. Read the program ledger
([`swarm-mvp-ledger.md`](swarm-mvp-ledger.md)) first for the branch map and the frozen-file rule.

## Base + branch

- **Repo:** `daemon-node` (Rust backend submodule; standalone checkout).
- **Base commit:** `d442cd8` (`docs(specs): swarm MVP program ledger`) — the Wave-0 scaffold tip on
  `integrations/swarm`.
- **Branch:** `swarm/p1`, forked from `d442cd8`. Integrates back into `integrations/swarm` at
  Merge 1.
- **Worktree:** `/home/j/experiments/daemon-worktree/swarm-proto` (isolated; never touches the main
  read-only checkout at `/home/j/experiments/daemon`).

## Scope this wave (lane P1, Wave 1)

Everything lands inside `crates/contracts/daemon-swarm-proto/` (a crate this lane owns outright,
including its `Cargo.toml` dev/normal dependencies) and the new
`crates/contracts/daemon-swarm-proto/daemon-swarm.cddl`. The other P-lane crates
(`daemon-swarm-coordinator`, `daemon-swarm-observe`) are out of scope for Wave 1 — the purified
`tick` (PROTO-1..3/14) and event sourcing land in a later wave on top of the types frozen here.

| Area | Spec / TDD grounding | Deliverable |
|---|---|---|
| Canonical CBOR encoder | RFC 8949 §4.2; ABI §6.1; TDD HOST-13 (proto half) | `canonical` module: deterministic bytes from any `serde::Serialize` value |
| Envelope + freeze/verify | spec §6.1, §16; TDD PROTO-11 | `envelope` module: schema types, validation, freeze→hash→sign, verify |
| Capability sets | spec §6.5, §16; TDD PROTO-12 | `capability` module: `name@version` typed sets, subset admission |
| Merkle set commitments | spec §6.4; TDD PROTO-5 (root/proof half) | `merkle` module: blake3 root over sorted `(peer, hash)` + membership proofs |
| Seven round messages + Join/Heartbeat + CDDL | spec §6.4, §7.3; TDD PROTO-13/19 | `messages` + `version` modules, `daemon-swarm.cddl`, conformance suite |
| Round state digest schedule | spec §5.6; TDD PROTO-18 | `digest` module: seed-keyed sampled-block xxh3-128 |

Out of scope (later P-lane waves, noted so integration knows where they land): purified `tick`
(PROTO-1..4/14..17), committee/assignment math (PROTO-8/10, `golden_psyche_parity`), heartbeat
drop counters (PROTO-7), replay oracle (PROTO-20), the `SwarmApi` wire mirror (WIRE-1..4), and the
observe projection (§3.9).

## Dependencies (all inside my own `Cargo.toml`; root `Cargo.toml` is FROZEN, untouched)

Normal deps, each `{ workspace = true }` from the pins in root `[workspace.dependencies]`:
`serde`, `ciborium`, `blake3`, `xxhash-rust` (xxh3), `ed25519-dalek`. No `tokio`, no Burn, no
wasmtime — the crate stays `wasm32-unknown-unknown`-clean (verified in the gates below).

Dev deps (ownership: a crate's own dev-dependencies are within its lane — root `Cargo.toml` is not
touched): `cddl-cat` (explicit `=0.7.1`, matching `daemon-api`'s pin, since it is **not** in root
`[workspace.dependencies]`) and `proptest` (`{ workspace = true }`).

Error handling: hand-rolled `SwarmProtoError` enum (no `thiserror`) per the Wave-0 lean-contract
note. No `todo!`/`unimplemented!`/`dbg!`.

## Seams exported (freeze at Merge 1)

These are the public surface other lanes build on; treat as stable after Merge 1.

- **Canonical codec** — `canonical::to_canonical_vec<T: Serialize>(&T) -> Result<Vec<u8>, SwarmProtoError>`,
  `canonical::from_canonical_slice<T: DeserializeOwned>(&[u8]) -> Result<T, SwarmProtoError>`.
- **Byte newtypes** — `PeerId([u8;32])`, `Hash([u8;32])` (blake3), `Root([u8;32])`,
  `Signature([u8;64])`, `Seed([u8;32])`; all CBOR `bstr`. `blake3_hash(&[u8]) -> Hash`.
- **Signing** — `SigningKey`/`VerifyingKey` re-exported helpers; `sign_canonical`/`verify_canonical`
  and `Signed<T>` wrapper (canonical-CBOR-of-body signed by node identity).
- **Envelope** — `Envelope` (+ `RunSection`/`ExperimentSection`/`Artifact`/`DataSection`/
  `Requirements`/`Phases`), `Envelope::validate`, `Envelope::freeze(&SigningKey) -> FrozenEnvelope`,
  `FrozenEnvelope::verify`, `FrozenEnvelope::{bytes, hash, config_bytes, signature, signer}`.
- **Capability** — `Capability { name, version }`, `CapabilitySet`, `CapabilitySet::admits(required)`.
- **Set commitment** — `SetCommitment { root, count }`, `commit_set(&[(PeerId, Hash)])`,
  `MembershipProof`, `SetCommitment::verify_membership(...)`.
- **Messages** — `RoundOpen`, `Commitment`, `Attestation`, `StorageReceipt`, `RoundRecord`,
  `Digest`, `Straggle`, `Join`, `Heartbeat`; the `SwarmMessage` externally-tagged enum; the signed
  wire frame `SignedMessage { version, payload, signer, sig }` with `sign`/`verify`; the authored
  `daemon-swarm.cddl`.
- **Version** — `SwarmProtoVersion(u16)` + `SWARM_PROTO_VERSION` const + `check_join(peer)` exact
  match.
- **Digest** — `StateLayout`, `DigestSchedule`, `derive_schedule`, `digest_state`.

## Planned commit slices (each green per the gates; TDD tight test+impl slices)

1. `mirror(P1): ledger` — this file.
2. `feat(swarm-proto): canonical RFC 8949 §4.2 CBOR encoder (green)` — `canonical` + RFC/adversarial tests.
3. `feat(swarm-proto): byte newtypes, blake3 hashing, ed25519 signing (green)` — `bytes`/`hash`/`sign`.
4. `feat(swarm-proto): run envelope schema + freeze/verify (green)` — `envelope` + PROTO-11 tests.
5. `feat(swarm-proto): capability sets + subset admission (green)` — `capability` + PROTO-12 tests.
6. `feat(swarm-proto): merkle set commitments + proofs (green)` — `merkle` + PROTO-5 root/proof tests.
7. `feat(swarm-proto): round message set + CDDL + proto version (green)` — `messages`/`version` +
   `daemon-swarm.cddl` + PROTO-13/19 conformance.
8. `feat(swarm-proto): round state digest schedule (green)` — `digest` + PROTO-18 tests.

Slices may be reordered/merged if the gates dictate, but every commit must pass
`cargo fmt --check`, `cargo clippy -p daemon-swarm-proto --all-targets -- -D warnings`,
`cargo test -p daemon-swarm-proto`, and `cargo build --target wasm32-unknown-unknown -p
daemon-swarm-proto`; the full-workspace gates run once before the lane is called done.

## Notes for Merge 1 integration

- **`SWARM_PROTO_VERSION` moves `0 → 1`** and its type changes from a bare `u32` const to a
  `SwarmProtoVersion(u16)` newtype (spec §7.3/§16 name it `u16`). The Wave-0 scaffold's
  `proto_version_is_stable` placeholder test is replaced by the real exact-match join check.
- **Canonicalization approach:** ciborium is used only to turn a serde value into a
  `ciborium::value::Value`; the final bytes are emitted by our own writer (definite lengths, shortest
  ints, RFC 8949 §4.2.2 shortest floats incl. f16, map keys sorted by encoded-key bytes). Documented
  in the `canonical` module docs; the RFC float/int vectors are the oracle.
- **CDDL strictness:** `cddl-cat` (the in-process validator `daemon-api` uses) does not enforce
  `bstr .size N`; fixed-length byte fields are declared as plain `bstr` in the CDDL and their lengths
  are enforced by the Rust newtypes. Same convention as `daemon-api.cddl`'s `content-hash`.
- **CDDL location:** `crates/contracts/daemon-swarm-proto/daemon-swarm.cddl` (beside the crate, the
  `daemon-api.cddl` pattern), not the repo root — keeps it inside the lane's owned tree.
- **No `xtask` edit:** conformance fixtures are generated in-process from the Rust types (real
  canonical bytes) and validated against the CDDL, so no shared `xtask cddl` change is needed this
  wave. A later wave can wire a `just`/`xtask` swarm-CDDL parity gate (shared tooling, coordinated).
- **Consensus-critical invariants downstream lanes must not re-derive:** the canonical encoder, the
  merkle set root, and the digest schedule are the bit-identity seams (spec §5.6/§6.4). Runtime/host
  lanes consume them via `workspace = true`; they must never fork a second encoding.
