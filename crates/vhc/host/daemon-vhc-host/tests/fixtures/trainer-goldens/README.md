# The v2-native trainer goldens (successor to the recorded v1 parity oracle)

This is the standing **drift oracle for the compute@2 trainer** (`tiny-llama-c3`): a recorded,
content-addressed bundle the per-tier reproduction tests
(`daemon-vhc-host/tests/trainer_goldens.rs`) check the trainer against, instead of the recorded v1
parity oracle (`../v1-parity-oracle/`). It is the v2-native successor that lets the v1 recording +
`v2_parity.rs` retire later (retirement plan §3) — nothing v1 is deleted by capturing it.

The recorded lane is the compute@2 trainer guest driven through a **single-peer barrier
whole-run**: it trains, commits its own outer update, and ingests that same committed set. Every
recorded value is the trainer's own v2-native output — no v1 inputs feed the trajectory. Only the
matched init and the config literals are inherited from the v1 oracle bundle, which is what anchors
the provenance chain below.

## Bundle contents

Every file is pinned by blake3 in `expected.json`; the reproduction tests re-verify each pin on
load (content-addressing).

| Path | What |
|---|---|
| `expected.json` | Every pin + the recorded per-round digests + the schedule + the exact model/profile config literals + `captured_from` provenance |
| `init.f32le.bin` | The matched init θ (flat f32-le; concatenated params in registration order), inherited from the v1 oracle so the goldens sit on the exact trajectory the C3c equality proof pins |
| `trained-round-{0,1}.f32le.bin` | Per-round trained θ (post-inner-steps, pre-ingest) — the guest's tag-2 publish; the native-lane (tolerance-class) comparison surface |
| `payload-round-{0,1}.bin` | Per-round committed payload bytes — **the trainer's OWN** sealed `SparseLoco` update (the guest publishes only the commitment hash, so the capture reconstructs the bytes natively and cross-checks them against that hash) |
| `capture/` | The documented, re-runnable capture crate (command below) |

The recorded per-round digests are the guest's tag-4 post-ingest det digests (the equality-class
oracle). At the capture commit they coincide bit-for-bit with the v1 oracle's c3 digests
(`abcf2612…`, `574cf418…`) — captured here from the autonomous v2-native self-ingest lane.

## Provenance chain (v1 oracle → C3c green → these goldens)

1. **v1 parity oracle** (`../v1-parity-oracle/`, recorded pre-sunset from the live v1 five-phase
   driver; decisions D5). Supplies this bundle's **matched init** (`init.f32le.bin`) and the exact
   **model/profile config literals**, and is the source of the recorded v1 det digests.
2. **`c3_parity` C3c green at the capture commit.** The test
   `c3_reauthored_tiny_llama_parity_vs_v1_oracle_cpu` (`daemon-vhc-host/tests/c3_parity.rs`) proves
   the compute@2 trainer, **fed the v1 oracle's recorded committed payloads**, reproduces the v1
   det digests **bit-for-bit** (the det lane is an equality class) and the trained θ **within the
   `OpClass::Optimizer` band** (the native lane is a tolerance class), on the pinned 2-round ×
   2-step, single-peer, 1-layer, seq-9 parity shape. Verified green at the capture commit:

   - commit `e93217b9` (`vhc-integration`; `tiny_llama_c3.wasm` pinned in `guests.blake3` at this
     commit)
   - run: `cargo test -p daemon-vhc-host --test c3_parity --features burn-ndarray \
     c3_reauthored_tiny_llama_parity_vs_v1_oracle_cpu` →
     `det-lane digests bit-exact across 2 rounds (v1 oracle reproduced)`; trained-θ max |Δ|
     6.557e-7 / 1.574e-7 (Optimizer band rtol 2e-4 / atol 2e-5).

3. **These goldens.** Captured at commit `e93217b9` from the compute@2 trainer's **own** single-peer
   barrier whole-run (self-ingest). The transitivity: step 2 proves the trainer lane is faithful to
   v1 (digest equality + Optimizer-band θ) at this exact commit and guest build; this bundle then
   records that same trustworthy lane's autonomous trajectory as the standing oracle going forward.

### Exact comparison surface (stated per the retirement-plan audit)

- C3c's digest equality holds over a **v1-payload-driven** trajectory (the c3 guest ingests the v1
  oracle's recorded committed payloads); the c3 guest's *own* committed payload bytes are not
  compared in that test. Payload-wire byte-identity is pinned separately at the profile-library
  level (`daemon-vhc-sdk-profiles` goldens: "the Section payload wire is byte-identical to the v1
  container encoding").
- This bundle therefore records the trainer's **own** committed payload bytes
  (`payload-round-*.bin`) — captured from the self-ingest lane, hash-verified against the guest's
  tag-3 commitment at capture. These are v2-native bytes (they differ from the v1 oracle's recorded
  update bytes), while the post-ingest digests coincide with v1's.

## What the reproduction tiers assert

`daemon-vhc-host/tests/trainer_goldens.rs`:

- **cpu** and **burn-ndarray** tiers drive the trainer against the recorded golden payloads and
  assert the tag-4 digests reproduce the golden **bit-for-bit** (equality class) and the tag-2 θ
  within the **Optimizer** band (tolerance class).
- **straggle → catch-up** leg (ported from `v2_parity.rs`'s
  `catch_up_after_straggle_reproduces_v1_digests_cpu` onto the compute@2 trainer): round 0's record
  arrives before its committed payload is fetchable (straggle), the payload lands, and `RoundOpen(1)`
  ingests round 0 (catch-up) and trains round 1 in one event slice — the run must end in exactly the
  clean run's final det state (the recorded final digest) with round-1 θ within the band.
- **wgpu** and **cuda** tiers (feature-gated, self-skipping without hardware) exercise the trainer's
  compute@2 kernels on the device via the op-journal replay seam (the `compute_replay.rs`
  mechanism) and reproduce the round-0 θ within the tolerance class.

### Backend independence of the det digest

The post-ingest digest is a pure function of `(init, ingested committed payloads)` via the
`daemon-vhc-det` fixed-order fp32 kernels (`SparseLoco`'s replicated det state is empty — its error
feedback is local, never digested). It is therefore **bit-identical across backends** by
construction, so the cpu and burn-ndarray tiers reproduce it identically. The only backend-sensitive
output is the trained θ (the native/tolerance lane), which the wgpu/cuda tiers check on the device.

### Note on the compute@2 execution backend at this commit

At the capture commit the compute@2 host execution backend is the ndarray `ComputeRunner`
regardless of `EngineConfig.backend` (the driver wires no GPU compute runner yet — that lands with
the GPU/CUDA workstream). Driving the guest under a GPU `BackendKind` would still execute on
ndarray, so the wgpu/cuda golden tiers do **not** drive the whole guest on the device; they replay
the trainer's recorded compute@2 op journal against a device `ComputeRunner<B>` — genuine on-device
execution of the same kernels — and check θ within tolerance. This is the lane that retires the
plan's "first GPU run may expose a compute@2 det-lane divergence" risk.

## Capture command (re-runnable on the current tree)

Unlike the v1 oracle's capture crate (which builds only on a pre-sunset tree), this one drives the
live compute@2 path and regenerates on the current tree:

```
# from the checkout root, build the guests first (the capture reads tiny_llama_c3.wasm):
nix develop --command bash -c 'cd crates/vhc/guests && \
  env -u CARGO_TARGET_DIR cargo build --release --target wasm32-unknown-unknown'
# then run the capture (it drives the guest, reconstructs + hash-verifies the payloads, and
# rewrites the whole bundle incl. expected.json):
cd crates/vhc/host/daemon-vhc-host/tests/fixtures/trainer-goldens/capture
CAPTURE_COMMIT=$(git rev-parse --short HEAD) nix develop <checkout> --command cargo run
```

The capture runs the whole recording **twice** and asserts byte-identity before it writes anything
— reproducibility is proven at capture time, not assumed. Verified byte-identical across two runs at
`e93217b9`.
