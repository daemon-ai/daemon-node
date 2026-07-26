// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The FROZEN fleet-ceremony trainer model configuration — the production validation tier.
//!
//! This is the single source of truth for the ceremony model geometry (the program spec's
//! ceremony section cites this module by path and restates the parameters for the reader).
//! It is a real multi-layer TinyLlama-class decoder — a ~0.79 B-parameter model — sized to the
//! amended validation fleet whose memory FLOOR is the M4 Mac's 32 GiB unified memory (the only
//! Metal trainer seat). It is explicitly NOT the 64-dim structural acceptance tier
//! ([`crate::live_genesis`]).
//!
//! # Sizing under the streaming det-fold substrate (amended fleet)
//!
//! The binding constraint is no longer the wasm32 4 GiB linear-memory ceiling that invalidated
//! this tier under the old resident design: canonical det-lane state is now **host-side**,
//! chunk-addressed, and streamed (the guest folds at O(chunks in flight)). Two memory pools
//! matter on the floor peer instead:
//!
//! - **On-device training working set** — the inner-loop master + gradient + both AdamW moments
//!   at 16 B/param ≈ **11.72 GiB** of fp32 device state (before activations). On the M4's 32 GiB
//!   unified memory the usable Metal working set is comparable to the old 24 GB-class discrete
//!   target, so this leaves real activation headroom on the floor peer. Growing the model toward
//!   1.0–1.2 B would erode exactly that headroom on exactly that box for zero gate value — the
//!   ceremony's ratified gate is a STRUCTURAL proof (digest agreement, churn, restore, replay),
//!   not a scale study (which the program defers).
//! - **Host-side retained det-state** — the state store's retained roots at the ratified cadence
//!   ≈ 5 families ≈ **14.65 GiB** (see [`CEREMONY_RETAINED_STATE_BYTES`]). On the M4's *unified*
//!   pool this cannot be RAM-resident alongside the device working set (11.72 + 14.65 GiB exceeds
//!   32 GiB before activations/OS), so the host state store is **disk-backed** — the arithmetic
//!   reason is pinned by [`tests::ceremony_geometry_is_frozen`].
//!
//! Largest single tensor: the tied token embedding, `vocab × d_model × 4 B` = 192 MiB — far
//! under every amended-fleet peer's per-buffer ceiling (Strix Halo wgpu/RADV ≈ 2047 MiB, the
//! Windows 5090 wgpu-DX12 lane, and the M4 Metal `maxBufferLength`).
//!
//! FROZEN: these values are ceremony inputs. Changing any of them re-derives the genesis, the
//! matched init (the seed-form `expected_root`), and every fleet-preflight sizing check — treat
//! any edit as a new ceremony candidate, never a tweak. The corpus pin (manifest hash +
//! tokenizer — the ratified TinyStories corpus under the TinyLlama SentencePiece tokenizer) is
//! frozen separately when the ceremony corpus is published; [`ceremony_model_value`]
//! deliberately builds only the `model` half of the trainer config so the corpus half cannot be
//! guessed at here.
//!
//! Nothing in the acceptance suite consumes this module (the acceptance tier stays 64-dim); it
//! is tracked so the ceremony genesis is authored from a reviewed, pinned artifact rather than
//! transcribed prose.

use std::collections::{BTreeMap, BTreeSet};

use ciborium::value::Value;

use crate::coordinator_config::{
    coordinator_role_config, CoordinatorAuthoring, PhaseDeadlines, RoundSchedule,
    EVENT_CLOCK_DEADLINE_S, WALL_CLOCK_TICK_PERIOD_MS,
};
use daemon_vhc_net::PublishedArtifact;
use daemon_vhc_proto::det_state::{
    derive_state_chunk_size, family_byte_len, family_fold, validate_checkpoint_cadence,
    validate_profile_chunk, validate_state_chunk_size, FamilyEntry,
};
use daemon_vhc_proto::envelope::Access;
use daemon_vhc_proto::genesis::{
    ChannelDecl, GenesisEnvelope, Identities, RoleEntry, RoleGrants, RunSection, SnapshotArtifact,
    StateContract, StateInit, TransportSelection, GENESIS_SCHEMA_MAJOR,
};
use daemon_vhc_proto::{blake3_hash, FrozenGenesis, Hash, PeerId, Seed, SigningKey};

/// Residual width.
pub const CEREMONY_D_MODEL: u32 = 1536;
/// Transformer blocks (real multi-layer depth — the acceptance tier runs 2).
pub const CEREMONY_N_LAYERS: u32 = 24;
/// Attention heads (`n_kv_heads == n_heads` — the guest runs full MHA).
pub const CEREMONY_N_HEADS: u32 = 24;
/// Per-head width (`n_heads · head_dim == d_model`).
pub const CEREMONY_HEAD_DIM: u32 = 64;
/// Vocabulary (tied input/output embedding): a power-of-two ceiling over the ceremony
/// tokenizer's id space (the in-guest `token % vocab` clamp is the identity for well-formed
/// corpora, the established discipline).
pub const CEREMONY_VOCAB: u32 = 32_768;
/// Sequence length.
pub const CEREMONY_SEQ_LEN: u32 = 2_048;
/// SwiGLU hidden = `ffn_mult · d_model` (= 4608).
pub const CEREMONY_FFN_MULT: u32 = 3;

/// AdamW learning rate.
pub const CEREMONY_LR: f64 = 3.0e-4;
/// AdamW β₁.
pub const CEREMONY_BETA1: f64 = 0.9;
/// AdamW β₂.
pub const CEREMONY_BETA2: f64 = 0.95;
/// AdamW ε.
pub const CEREMONY_ADAM_EPS: f64 = 1.0e-8;
/// AdamW decoupled weight decay.
pub const CEREMONY_WD: f64 = 0.1;
/// RoPE base.
pub const CEREMONY_ROPE_THETA: f64 = 10_000.0;
/// RMSNorm epsilon.
pub const CEREMONY_RMSNORM_EPS: f64 = 1.0e-5;

/// Inner optimizer steps per round — the trainer config's `steps_per_round`.
pub const CEREMONY_STEPS_PER_ROUND: u32 = 30;
/// Sequences per inner step — the trainer config's `micro_batch`.
pub const CEREMONY_MICRO_BATCH: u32 = 1;
/// Fetch-recovery budget before a stalled peer leaves for the epoch (`stall_rounds_max`).
pub const CEREMONY_STALL_ROUNDS_MAX: u32 = 4;
/// Record absences before a silent member is dropped from the roster (`k_absences`).
pub const CEREMONY_K_ABSENCES: u32 = 3;

/// The ceremony coordinator's real-timer period: the fleet run is a LIVE deployment, so its
/// clock measures wall time and the authored `*_s` timers ([`CeremonyRunTimers`]) mean seconds.
/// Without it the guest's `#[serde(default)] tick_period_ms = 0` leaves the coordinator on the
/// deterministic event-driven clock, where a "300 s" warmup is 300 delivered events and a quiet
/// run's deadline never arrives.
pub const CEREMONY_TICK_PERIOD_MS: u64 = WALL_CLOCK_TICK_PERIOD_MS;

/// The frozen total parameter count of the ceremony geometry (pinned by
/// [`tests::ceremony_geometry_is_frozen`]; the sum of [`ceremony_param_numels`]).
pub const CEREMONY_PARAM_COUNT: u64 = 786_507_264;

/// The on-device fp32 training working set per param: master + gradient + AdamW m + AdamW v.
pub const CEREMONY_DEVICE_BYTES_PER_PARAM: u64 = 16;

/// The **on-device training working set** in bytes ([`CEREMONY_PARAM_COUNT`] ×
/// [`CEREMONY_DEVICE_BYTES_PER_PARAM`]) ≈ 11.72 GiB — the inner-loop fp32 state each trainer
/// peer holds on its accelerator, before activations. Fits the M4 32 GiB unified floor's usable
/// Metal working set with activation headroom.
pub const CEREMONY_DEVICE_STATE_BYTES: u64 = CEREMONY_PARAM_COUNT * CEREMONY_DEVICE_BYTES_PER_PARAM;

/// The **host-side retained det-state** in bytes at the ratified retention/cadence defaults
/// (`state_retain_roots = 2`): ≈ 5 distinct families (2 master roots + ef + a sealed
/// checkpoint's adamw_m/adamw_v; master/ef dedup with the retained roots) ≈ 5 × 4 B/param. This
/// is the retained-bytes figure the disk-backing decision is measured against — pinned as a
/// standing assertion by the host state-store suite so a future retention/cadence change that
/// blows past the disk-backing assumption is caught by a test, not a fleet incident.
pub const CEREMONY_RETAINED_STATE_BYTES: u64 = 5 * 4 * CEREMONY_PARAM_COUNT;

/// The amended fleet's tightest per-buffer allocation ceiling (`i32::MAX`-clamped wgpu
/// `max_buffer_size` ≈ 2047 MiB, the Strix Halo RADV and Windows 5090 wgpu-DX12 lanes; the M4
/// Metal `maxBufferLength` is larger still). The largest ceremony tensor MUST fit under it.
pub const CEREMONY_PER_BUFFER_CEILING_BYTES: u64 = 2047 * (1 << 20);

/// The memory floor peer's nameplate unified-memory budget: the M4 Mac's 32 GiB (the only Metal
/// trainer seat under the amended fleet). Both the on-device working set and any RAM-resident
/// host state would draw from this one pool.
pub const CEREMONY_FLOOR_UNIFIED_BYTES: u64 = 32 * (1 << 30);

/// A conservative **usable** budget on the 32 GiB unified floor after the OS, the Metal
/// framework's working-set reservation, the worker/host process, and the wasm runtime take their
/// share (Apple's `recommendedMaxWorkingSetSize` on 32 GiB unified is ~22–24 GiB). The device
/// working set must fit under this WITH activation headroom; the device set plus a RAM-resident
/// retained store must NOT — that is the forcing arithmetic for disk backing.
pub const CEREMONY_FLOOR_USABLE_BYTES: u64 = 24 * (1 << 30);

/// The frozen per-parameter numels of the ceremony geometry, in the guest's registration order
/// (token embedding; per block: attn-norm, wq, wk, wv, wo, ffn-norm, w1, w3, w2; final norm) —
/// the same layout arithmetic the trainer guest derives from its `ModelCfg`.
#[must_use]
pub fn ceremony_param_numels() -> Vec<usize> {
    let d = CEREMONY_D_MODEL as usize;
    let qdim = (CEREMONY_N_HEADS * CEREMONY_HEAD_DIM) as usize;
    let hidden = (CEREMONY_FFN_MULT * CEREMONY_D_MODEL) as usize;
    let vocab = CEREMONY_VOCAB as usize;
    let mut out = vec![vocab * d];
    for _ in 0..CEREMONY_N_LAYERS {
        out.extend([
            d,
            d * qdim,
            d * qdim,
            d * qdim,
            qdim * d,
            d,
            d * hidden,
            d * hidden,
            hidden * d,
        ]);
    }
    out.push(d);
    out
}

/// The frozen ceremony trainer `model` config map (raw canonical-CBOR value against the trainer
/// guest's documented `ModelCfg` schema) — the reviewed artifact the ceremony genesis authoring
/// embeds verbatim. The corpus/`live` half is composed at genesis authoring, once the ceremony
/// corpus manifest is published and pinned.
#[must_use]
pub fn ceremony_model_value() -> Value {
    let text = |s: &str| Value::Text(s.into());
    let uint = |v: u32| Value::Integer(u64::from(v).into());
    Value::Map(vec![
        (text("d_model"), uint(CEREMONY_D_MODEL)),
        (text("n_layers"), uint(CEREMONY_N_LAYERS)),
        (text("n_heads"), uint(CEREMONY_N_HEADS)),
        (text("head_dim"), uint(CEREMONY_HEAD_DIM)),
        (text("vocab"), uint(CEREMONY_VOCAB)),
        (text("seq_len"), uint(CEREMONY_SEQ_LEN)),
        (text("ffn_mult"), uint(CEREMONY_FFN_MULT)),
        (text("rope_theta"), Value::Float(CEREMONY_ROPE_THETA)),
        (text("rmsnorm_eps"), Value::Float(CEREMONY_RMSNORM_EPS)),
        (text("lr"), Value::Float(CEREMONY_LR)),
        (text("beta1"), Value::Float(CEREMONY_BETA1)),
        (text("beta2"), Value::Float(CEREMONY_BETA2)),
        (text("adam_eps"), Value::Float(CEREMONY_ADAM_EPS)),
        (text("wd"), Value::Float(CEREMONY_WD)),
    ])
}

// ---- the ceremony state contract: seed-derived init + pinned expected_root (§6, [SF-5]) --------

/// The frozen ceremony **init seed** (the 32-byte expansion seed of the seed-derived genesis init,
/// D-SF2). Every peer expands the matched init deterministically from this seed + the versioned
/// distribution and cross-checks its sealed master fold against [`CEREMONY_EXPECTED_ROOT`] — a
/// mismatch is a typed init failure, never a silent divergence. A pinned ceremony input: changing
/// it re-derives the expected root and is a new ceremony candidate.
pub const CEREMONY_INIT_SEED: [u8; 32] = [0xCE; 32];

/// The versioned seed-init distribution the ceremony expands under ([`daemon_vhc_det::
/// SEED_INIT_DIST_V1`]) — the derivation identity dual-compiled in `daemon-vhc-det`.
pub const CEREMONY_INIT_DIST: u64 = daemon_vhc_det::SEED_INIT_DIST_V1;

/// The compression **profile chunk** for the ceremony geometry. It MUST divide every parameter's
/// numel; for this geometry the 1536-wide RMSNorm parameters make `chunk | 1536` binding, so the
/// profile chunk IS `d_model` (the SparseLoco default 4096 does not divide 1536 and would refuse
/// at the first norm parameter — design §3.2).
#[must_use]
pub fn ceremony_profile_chunk() -> u64 {
    u64::from(CEREMONY_D_MODEL)
}

/// The run-pinned `state_chunk_size` for the ceremony ([`derive_state_chunk_size`] over the
/// profile chunk): the largest integer multiple of the profile chunk's byte width (`1536 × 4`)
/// that is ≤ ~4 MiB — `682 × 6144 = 4,190,208` bytes ≈ 3.996 MiB.
#[must_use]
pub fn ceremony_state_chunk_size() -> u64 {
    derive_state_chunk_size(ceremony_profile_chunk())
}

/// The pinned **expected state root** ([SF-5]): the `master`-family fold of the seed-derived
/// matched init over the ceremony layout at [`ceremony_state_chunk_size`]. Committed here so it
/// is reviewable and so a change to the seed, the distribution, or the fold definition trips
/// [`tests::ceremony_expected_root_reproduces_the_pin`] (a stop-and-report digest-value movement)
/// rather than silently re-deriving. Reproduced from the seed by [`ceremony_expected_state_root`].
pub const CEREMONY_EXPECTED_ROOT: [u8; 32] = [
    0x56, 0xd8, 0x43, 0xda, 0x34, 0x24, 0xc3, 0xc2, 0x56, 0xdc, 0xfa, 0x23, 0xa6, 0xc9, 0x05, 0x09,
    0xb2, 0x1f, 0x20, 0x2b, 0xb2, 0xe0, 0x9d, 0x1d, 0x18, 0x8f, 0x41, 0xd1, 0xb5, 0x61, 0xb3, 0x99,
];

/// Compute the ceremony master-family fold from the seed-derived init, **streaming** the
/// expansion window-by-window (O(`state_chunk_size`) memory ≈ 4 MiB — never materializing the
/// ~2.93 GiB family), per-parameter chunked exactly as [`daemon_vhc_proto::det_state`] folds it.
/// This is what the genesis authoring pins as `expected_root` and what every peer reproduces to
/// cross-check its own expansion.
#[must_use]
pub fn ceremony_expected_state_root() -> Hash {
    let chunk_size = ceremony_state_chunk_size();
    let elems_per_chunk = (chunk_size / 4) as usize;
    let numels = ceremony_param_numels();
    let mut chunk_hashes: Vec<Hash> = Vec::new();
    let mut window: Vec<f32> = Vec::with_capacity(elems_per_chunk);
    for (i, &numel) in numels.iter().enumerate() {
        // Per-parameter chunking: a parameter never spans a chunk boundary; its last chunk is
        // short (mirrors `family_chunk_hashes`' per-parameter `.chunks(chunk_size)`).
        let mut off = 0usize;
        while off < numel {
            let take = elems_per_chunk.min(numel - off);
            daemon_vhc_det::seed_init_window(
                &CEREMONY_INIT_SEED,
                CEREMONY_INIT_DIST,
                i as u64,
                off,
                take,
                &mut window,
            )
            .expect("the ceremony distribution id is implemented");
            let mut bytes = Vec::with_capacity(take * 4);
            for &v in &window {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            chunk_hashes.push(blake3_hash(&bytes));
            off += take;
        }
    }
    let byte_len = family_byte_len(&numels.iter().map(|&n| n as u64).collect::<Vec<_>>());
    family_fold(chunk_size, byte_len, &chunk_hashes)
}

/// The frozen ceremony **state contract** ([SF-5], D-SF2): the derived `state_chunk_size` + a
/// seed-form init pin the guest expands, self-seals, and cross-checks against the expected root.
/// The single source of truth the ceremony genesis authoring embeds (envelope `state_contract` +
/// the trainer config's `state` field). Uses the pinned [`CEREMONY_EXPECTED_ROOT`].
#[must_use]
pub fn ceremony_state_contract() -> StateContract {
    StateContract {
        chunk_size: ceremony_state_chunk_size(),
        init: StateInit::Seed {
            seed: Seed(CEREMONY_INIT_SEED),
            dist: CEREMONY_INIT_DIST,
            expected_root: Hash(CEREMONY_EXPECTED_ROOT),
        },
    }
}

// ---- ceremony genesis authoring (§15 W-SF6; the trainer + coordinator envelope) ---------------

/// The FROZEN ceremony compression profile (SparseLoco), raw canonical CBOR against the trainer
/// guest's schema. `chunk` is the profile chunk ([`ceremony_profile_chunk`], = `d_model`); `topk`
/// selects 64 updates per compression chunk per peer (design §7.4). The profile choice does NOT
/// affect the seed-init `expected_root` (that folds the init, not the profile) but its `chunk`
/// IS the state-contract geometry driver via [`derive_state_chunk_size`].
#[must_use]
fn ceremony_profile_value() -> Value {
    let text = |s: &str| Value::Text(s.into());
    let uint = |v: u64| Value::Integer(v.into());
    Value::Map(vec![
        (text("h"), uint(3)),
        (text("ef_decay"), Value::Float(0.95)),
        (text("chunk"), uint(ceremony_profile_chunk())),
        (text("topk"), uint(64)),
        (text("bits"), uint(2)),
        (text("outer_alpha"), Value::Float(1.0)),
        (text("clip"), Value::Bool(true)),
    ])
}

/// The FROZEN ceremony trainer guest config (raw canonical CBOR): the frozen model
/// ([`ceremony_model_value`]) + the SparseLoco profile + the seed-form state contract
/// ([`ceremony_state_contract`]) + the fleet trainer roster + the `live` section naming the run +
/// the corpus manifest pin. This is the `da_init`/`da_build` config the trainer role receives.
#[must_use]
pub fn ceremony_trainer_config(run_label: &str, corpus_manifest: Hash, roster: &[PeerId]) -> Value {
    let text = |s: &str| Value::Text(s.into());
    let live = Value::Map(vec![
        (text("run_label"), text(run_label)),
        (
            text("manifest"),
            Value::serialized(&corpus_manifest).expect("manifest hash value"),
        ),
    ]);
    let Value::Map(mut fields) = ceremony_trainer_config_harness(roster) else {
        unreachable!("the harness form is a map")
    };
    fields.push((text("live"), live));
    Value::Map(fields)
}

/// The FROZEN ceremony trainer config in its **harness** form: [`ceremony_trainer_config`] without
/// the `live` section, i.e. the same frozen model + profile + seed-form state contract driven over
/// host staging instead of module-driven corpus/wire traffic (the guest's documented
/// absent-`live` contract).
///
/// This is what a gate that must exercise the ceremony GEOMETRY without the ceremony's data plane
/// feeds — the real-geometry init gate (`tests/ceremony_geometry.rs`). It shares the frozen halves
/// with the fleet form by construction, so a geometry or state-contract edit cannot move one
/// without the other.
#[must_use]
pub fn ceremony_trainer_config_harness(roster: &[PeerId]) -> Value {
    let text = |s: &str| Value::Text(s.into());
    let roster_val = Value::Array(roster.iter().map(|p| Value::Bytes(p.0.to_vec())).collect());
    let peer = roster.first().map_or_else(
        || Value::Bytes(vec![0u8; 32]),
        |p| Value::Bytes(p.0.to_vec()),
    );
    Value::Map(vec![
        (text("model"), ceremony_model_value()),
        (text("peer"), peer),
        (text("roster"), roster_val),
        (
            text("steps_per_round"),
            Value::Integer(u64::from(CEREMONY_STEPS_PER_ROUND).into()),
        ),
        (
            text("micro_batch"),
            Value::Integer(u64::from(CEREMONY_MICRO_BATCH).into()),
        ),
        (
            text("stall_rounds_max"),
            Value::Integer(u64::from(CEREMONY_STALL_ROUNDS_MAX).into()),
        ),
        (text("profile"), ceremony_profile_value()),
        (
            text("state"),
            Value::serialized(&ceremony_state_contract()).expect("state contract value"),
        ),
    ])
}

/// The frozen ceremony trainer config in its **round-walk gate** form: the harness form
/// ([`ceremony_trainer_config_harness`]) with exactly ONE documented deviation, so a gate can
/// drive the ROUND path (θ export → `make_update` → ingest → quiesce) at the real ceremony
/// geometry on a CPU lane in minutes.
///
/// - `steps_per_round = 0` — the round opens, commits, fences and exports WITHOUT the training
///   math. A single 30-step inner loop at seq 2048 over this 24-layer decoder is hours of ndarray
///   CPU; the round path's residency class is a property of the state walks (which windows are
///   read back, folded, emitted and applied), not of the gradients that produced θ. The trainer
///   goldens own the training math (bit-exact at a toy geometry); this owns the geometry.
///
/// The compression profile is the FROZEN one — `topk = 64`, the fleet's density. It used to be a
/// caller-chosen deviation because the committed payload crossed the guest boundary as one decoded
/// container (~210 MB per peer at this density, which no wasm32 linear memory holds); with the
/// container range-addressable and read a fold window at a time out of the host buffer it arrives
/// in ([SF-R3]), the payload plane streams like every other, so the gate drives the real density.
///
/// Everything else — the frozen model, the state contract, the pinned `expected_root`, the window
/// size, the roster shape — is shared with the fleet form by construction.
#[must_use]
pub fn ceremony_trainer_config_round_walk(roster: &[PeerId]) -> Value {
    let Value::Map(fields) = ceremony_trainer_config_harness(roster) else {
        unreachable!("the harness form is a map")
    };
    let patched = fields
        .into_iter()
        .map(|(k, v)| match &k {
            Value::Text(name) if name == "steps_per_round" => (k, Value::Integer(0u64.into())),
            _ => (k, v),
        })
        .collect();
    Value::Map(patched)
}

/// The frozen ceremony trainer config in its **training-step gate** form: the harness form with
/// real training math at the frozen profile, over a SHORTENED sequence.
///
/// # Why this form exists
///
/// [`ceremony_trainer_config_round_walk`] deliberately drops the inner loop (`steps_per_round = 0`)
/// so the state walks can be gated at the real geometry in minutes. That leaves one seam untested
/// at ceremony scale: the round path WITH the optimizer running — 786_507_264 real parameters
/// stepped through forward → backward → `inner_update` → the round-final fence → the θ export the
/// commitment is built from. Every lane that runs the training math (the trainer goldens) runs it
/// at a toy geometry, and every lane at this geometry skips it, so nothing in the battery observed
/// what a fleet peer does between "init finished" and "round 0 committed".
///
/// # The bound, and why it still covers the seam
///
/// `seq_len` is the ONE deviation, and it is the only knob that buys the difference: a step's
/// arithmetic is `O(parameters × tokens)`, so the frozen 2048-token sequence makes a single CPU
/// step ~9.7 TFLOP — hours per round, which is not a gate. `steps_per_round` and the sequence are
/// the caller's, and the recommended bound is a short sequence with more than one step.
///
/// What shortening the sequence does NOT change is exactly what this lane is for. The parameter
/// layout, the optimizer state, the gradient buffers, the per-parameter device traffic, the
/// compression profile (`topk = 64`) and the export/commit walks are all `O(parameters)` and run at
/// FULL ceremony size at any sequence length; only the activation tensors scale with the sequence.
/// So the residency, readback, fuel and device-op behaviour of a real training step at this
/// geometry is exercised, while the part that is merely expensive is not. `> 1` step is required
/// because the AdamW accumulation boundary (`inner_update`) only behaves like a round's inner loop
/// when there is more than one step to accumulate across.
///
/// Everything else — the model geometry, the state contract, the pinned `expected_root`, the window
/// size, the profile, the roster shape — is shared with the fleet form by construction.
///
/// # Panics
/// If `seq_len` is zero (a step with no tokens is not a training step).
#[must_use]
pub fn ceremony_trainer_config_training_step(
    roster: &[PeerId],
    steps_per_round: u64,
    seq_len: u32,
) -> Value {
    assert!(seq_len > 0, "a training step needs at least one token");
    let Value::Map(fields) = ceremony_trainer_config_harness(roster) else {
        unreachable!("the harness form is a map")
    };
    let patched = fields
        .into_iter()
        .map(|(k, v)| match &k {
            Value::Text(name) if name == "steps_per_round" => {
                (k, Value::Integer(steps_per_round.into()))
            }
            Value::Text(name) if name == "model" => {
                let Value::Map(model) = v else {
                    unreachable!("the frozen model is a map")
                };
                let model = model
                    .into_iter()
                    .map(|(mk, mv)| match &mk {
                        Value::Text(f) if f == "seq_len" => {
                            (mk, Value::Integer(u64::from(seq_len).into()))
                        }
                        _ => (mk, mv),
                    })
                    .collect();
                (k, Value::Map(model))
            }
            _ => (k, v),
        })
        .collect();
    Value::Map(patched)
}

/// The reduced parameter layout the live-corpus staging gate trains — the ONE deviation that lane
/// takes from the frozen fleet trainer, and the mirror image of
/// [`ceremony_trainer_config_training_step`]'s.
///
/// That lane shortens the SEQUENCE and keeps the parameters; this one keeps the SEQUENCE and
/// shortens the parameters. Between them the frozen `(seq_len, vocab)` and the frozen parameter
/// count are each driven through a real training step, and neither lane costs a datacenter.
pub const STAGING_GATE_D_MODEL: u32 = 64;
/// See [`STAGING_GATE_D_MODEL`].
pub const STAGING_GATE_N_LAYERS: u32 = 2;
/// See [`STAGING_GATE_D_MODEL`].
pub const STAGING_GATE_N_HEADS: u32 = 2;
/// See [`STAGING_GATE_D_MODEL`] (`n_heads · head_dim == d_model`, as the frozen layout requires).
pub const STAGING_GATE_HEAD_DIM: u32 = 32;
/// See [`STAGING_GATE_D_MODEL`].
pub const STAGING_GATE_FFN_MULT: u32 = 2;

/// The per-parameter numels of [`ceremony_trainer_config_live_staging`]'s reduced layout, in the
/// guest's registration order (the same arithmetic as [`ceremony_param_numels`]).
#[must_use]
pub fn staging_gate_param_numels() -> Vec<usize> {
    let d = STAGING_GATE_D_MODEL as usize;
    let qdim = (STAGING_GATE_N_HEADS * STAGING_GATE_HEAD_DIM) as usize;
    let hidden = (STAGING_GATE_FFN_MULT * STAGING_GATE_D_MODEL) as usize;
    let vocab = CEREMONY_VOCAB as usize;
    let mut out = vec![vocab * d];
    for _ in 0..STAGING_GATE_N_LAYERS {
        out.extend([
            d,
            d * qdim,
            d * qdim,
            d * qdim,
            qdim * d,
            d,
            d * hidden,
            d * hidden,
            hidden * d,
        ]);
    }
    out.push(d);
    out
}

/// The state contract for the reduced staging layout: the derived chunk size plus the seed-form
/// init pin, authored here so the guest's own `expected_root` cross-check runs for real.
#[must_use]
pub fn staging_gate_state_contract() -> StateContract {
    let seed = [0x5eu8; 32];
    let dist = daemon_vhc_det::SEED_INIT_DIST_V1;
    let chunk_size = derive_state_chunk_size(u64::from(STAGING_GATE_D_MODEL));
    let param_bytes: Vec<Vec<u8>> = staging_gate_param_numels()
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let vals =
                daemon_vhc_det::seed_init_param(&seed, dist, i as u64, n).expect("known dist");
            let mut b = Vec::with_capacity(n * 4);
            for v in vals {
                b.extend_from_slice(&v.to_le_bytes());
            }
            b
        })
        .collect();
    let views: Vec<&[u8]> = param_bytes.iter().map(Vec::as_slice).collect();
    let expected_root = FamilyEntry::author(&views, chunk_size)
        .expect("the staging layout authors a matched-init fold")
        .fold;
    StateContract {
        chunk_size,
        init: StateInit::Seed {
            seed: Seed(seed),
            dist,
            expected_root,
        },
    }
}

/// The frozen ceremony trainer config in its **live-corpus staging gate** form: the fleet's OWN
/// `live` section over a reduced parameter layout.
///
/// # Why this form exists
///
/// Every other trainer form drops the `live` section, so the guest takes its batches from host
/// staging and its whole module-driven data path — manifest fetch → chunk registration → planned
/// segments → `ArtifactRange` → `stage_fetched_batches` → the round's first `train_step` — is
/// driven by nothing. The fleet genesis is the only config that carries `live`, which made the
/// fleet the only place that path had ever run.
///
/// # What is FROZEN here (all of what the gap was about)
///
/// The `live` section itself, `seq_len`, `vocab`, `steps_per_round`, `micro_batch`,
/// `stall_rounds_max`, and the compression profile's shape — so the round plans the fleet's 30
/// single-sequence inner steps over the fleet's 2048-token sequences against a real chunk-addressed
/// manifest, and the forward pass sees the frozen `(seq_len, vocab)` activation geometry.
///
/// # The bound
///
/// `d_model`/`n_layers`/`n_heads`/`head_dim`/`ffn_mult` are reduced
/// ([`STAGING_GATE_D_MODEL`] and friends), because a 30-step round over 786_507_264 parameters at
/// 2048 tokens is ~290 TFLOP — not a gate on any CPU. What that does NOT touch is what this lane
/// is for: the staging path is parameter-independent, and the forward pass's own guest-resident
/// working set is a function of `(seq_len, vocab)` — the target rows and the `rows × vocab` logit
/// plane they index into — both of which are the frozen values here. `ceremony_training_step` covers the
/// complementary axis (the frozen parameter count at a shortened sequence), so between the two
/// lanes a real training step is driven at full size along each axis.
///
/// The profile `chunk` follows the same rule as the frozen one — it must divide every numel, so it
/// IS the reduced `d_model` — and the state contract is authored for this layout
/// ([`staging_gate_state_contract`]), so the guest's `expected_root` cross-check runs for real.
#[must_use]
pub fn ceremony_trainer_config_live_staging(
    run_label: &str,
    corpus_manifest: Hash,
    roster: &[PeerId],
) -> Value {
    let text = |s: &str| Value::Text(s.into());
    let uint = |v: u64| Value::Integer(v.into());
    let peer = roster.first().map_or_else(
        || Value::Bytes(vec![0u8; 32]),
        |p| Value::Bytes(p.0.to_vec()),
    );
    let model = Value::Map(vec![
        (text("d_model"), uint(u64::from(STAGING_GATE_D_MODEL))),
        (text("n_layers"), uint(u64::from(STAGING_GATE_N_LAYERS))),
        (text("n_heads"), uint(u64::from(STAGING_GATE_N_HEADS))),
        (text("head_dim"), uint(u64::from(STAGING_GATE_HEAD_DIM))),
        (text("vocab"), uint(u64::from(CEREMONY_VOCAB))),
        (text("seq_len"), uint(u64::from(CEREMONY_SEQ_LEN))),
        (text("ffn_mult"), uint(u64::from(STAGING_GATE_FFN_MULT))),
        (text("rope_theta"), Value::Float(CEREMONY_ROPE_THETA)),
        (text("rmsnorm_eps"), Value::Float(CEREMONY_RMSNORM_EPS)),
        (text("lr"), Value::Float(CEREMONY_LR)),
        (text("beta1"), Value::Float(CEREMONY_BETA1)),
        (text("beta2"), Value::Float(CEREMONY_BETA2)),
        (text("adam_eps"), Value::Float(CEREMONY_ADAM_EPS)),
        (text("wd"), Value::Float(CEREMONY_WD)),
    ]);
    let Value::Map(profile) = ceremony_profile_value() else {
        unreachable!("the frozen profile is a map")
    };
    let profile = Value::Map(
        profile
            .into_iter()
            .map(|(k, v)| match &k {
                Value::Text(f) if f == "chunk" => (k, uint(u64::from(STAGING_GATE_D_MODEL))),
                _ => (k, v),
            })
            .collect(),
    );
    Value::Map(vec![
        (text("model"), model),
        (text("peer"), peer),
        (
            text("roster"),
            Value::Array(roster.iter().map(|p| Value::Bytes(p.0.to_vec())).collect()),
        ),
        (
            text("steps_per_round"),
            uint(u64::from(CEREMONY_STEPS_PER_ROUND)),
        ),
        (text("micro_batch"), uint(u64::from(CEREMONY_MICRO_BATCH))),
        (
            text("stall_rounds_max"),
            uint(u64::from(CEREMONY_STALL_ROUNDS_MAX)),
        ),
        (text("profile"), profile),
        (
            text("state"),
            Value::serialized(&staging_gate_state_contract()).expect("state contract value"),
        ),
        (
            text("live"),
            Value::Map(vec![
                (text("run_label"), text(run_label)),
                (
                    text("manifest"),
                    Value::serialized(&corpus_manifest).expect("manifest hash value"),
                ),
            ]),
        ),
    ])
}

/// The real fleet RUN TIMERS + stop condition the ceremony operator calibrates at preflight and
/// binds into the coordinator config — closing the documented "the fleet operator tunes real
/// timers at preflight" seam inside this reviewed module instead of at a CLI.
///
/// These are WALL-CLOCK SECONDS: the ceremony coordinator config arms the real timer
/// ([`CEREMONY_TICK_PERIOD_MS`]), so a tuned value is the wall it says it is. The [`Default`]
/// values (warmup / round / witness / cooldown = 1_000_000 s, stop = 1_000_000 rounds) are the
/// effectively-infinite shape: every phase then exits through its event-driven fast path.
/// Only an operator that sets real values moves the run id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CeremonyRunTimers {
    /// The join/warmup wall the coordinator waits through before round 0 (`warmup_s`).
    pub warmup_s: u64,
    /// The per-round training-phase wall ceiling (`round_train_max_s`).
    pub round_max_s: u64,
    /// The witness/finalization-phase wall (`round_witness_s`).
    pub witness_s: u64,
    /// The end-of-run cooldown wall (`cooldown_s`).
    pub cooldown_s: u64,
    /// The run's stop condition, in completed rounds (`StopCondition::Rounds`).
    pub stop_rounds: u64,
}

impl Default for CeremonyRunTimers {
    fn default() -> Self {
        // The effectively-infinite shape: every phase exits through its fast path (type docs).
        Self {
            warmup_s: EVENT_CLOCK_DEADLINE_S,
            round_max_s: EVENT_CLOCK_DEADLINE_S,
            witness_s: EVENT_CLOCK_DEADLINE_S,
            cooldown_s: EVENT_CLOCK_DEADLINE_S,
            stop_rounds: 1_000_000,
        }
    }
}

/// The knobs the ceremony genesis authoring binds around the FROZEN model + state contract. The
/// corpus manifest pin, the fleet trust set, the roster, and the module hashes are CEREMONY-TIME
/// inputs (the published corpus + the fleet's certified peer identities, supplied by the preflight
/// operator); everything model-shaped is frozen in this module.
pub struct CeremonyGenesisSpec<'a> {
    /// The human/registry-facing run label.
    pub run_label: &'a str,
    /// The pinned coordinator module blake3 (`coordinator_quorum.wasm`).
    pub coordinator_module: Hash,
    /// The pinned trainer module blake3 (`tiny_llama.wasm`).
    pub trainer_module: Hash,
    /// The pinned corpus manifest hash (the published TinyStories corpus under the TinyLlama
    /// SentencePiece tokenizer) — committed as the run's data identity.
    pub corpus_manifest: Hash,
    /// The published corpus objects to map + grant, `(artifact name, published object)` — the
    /// manifest, the tokenizer and every shard fold (the trainer's `data@2` fetch grants). Each
    /// object carries its KIND, which is what fixes the key it is published at
    /// ([`PublishedArtifact`]).
    pub corpus_artifacts: &'a [(String, PublishedArtifact)],
    /// The sequence length the corpus was tokenized at (coordinator run config).
    pub seq_len: u64,
    /// Every participating peer's genesis-trusted base identity (the trust set).
    pub trusted_bases: &'a [PeerId],
    /// The trainer assignment roster (the fleet's trainer peer identities).
    pub roster: &'a [PeerId],
    /// The run's upgrade authority (unanimous module-upgrade signers; empty = an immutable run).
    pub upgrade_authority: Vec<PeerId>,
    /// Minimum healthy peers to leave `WaitingForMembers` (the amended fleet floor is 3 trainers).
    pub min_peers: u32,
    /// Roster ceiling.
    pub max_peers: u32,
    /// The remote checkpoint cadence in rounds (D-SF3); validated against `payload_retention`.
    pub remote_ckpt_cadence_rounds: u64,
    /// The payload retention floor in rounds (`0` = unbounded — no cadence constraint).
    pub payload_retention_rounds: u64,
    /// The real fleet run timers + stop condition, in wall-clock seconds.
    /// [`CeremonyRunTimers::default`] is the effectively-infinite shape; the preflight operator
    /// tunes it from the measured per-round wall on the slowest box.
    pub timers: CeremonyRunTimers,
}

/// Author + freeze the ceremony genesis (§15 W-SF6): the trainer role carrying the FROZEN model +
/// the seed-form state contract (pinned `expected_root`) + the corpus pin, the coordinator role,
/// the fleet trust set, and the upgrade authority. Signs with `author`. The corpus + init pins
/// commit into the genesis hash (the run's cryptographic id).
///
/// # Errors
/// A human-readable failure when a genesis-authoring rule is violated (profile chunk does not
/// divide the layout, an invalid state chunk size, a checkpoint cadence that could strand a
/// rejoiner past payload retention, a round schedule the trainers cannot slice, or wall-clock
/// timers authored onto a clock that does not measure wall time) or the envelope fails to
/// validate/freeze.
pub fn ceremony_genesis(
    spec: &CeremonyGenesisSpec<'_>,
    author: &SigningKey,
) -> Result<FrozenGenesis, String> {
    // Genesis-authoring geometry rules ([SF-5], §3.2, §7.4) — refuse up front, never at the
    // kernel's first use on the fleet.
    let numels: Vec<u64> = ceremony_param_numels().iter().map(|&n| n as u64).collect();
    validate_profile_chunk(ceremony_profile_chunk(), &numels).map_err(|e| e.to_string())?;
    validate_state_chunk_size(ceremony_state_chunk_size(), ceremony_profile_chunk())
        .map_err(|e| e.to_string())?;
    validate_checkpoint_cadence(
        spec.remote_ckpt_cadence_rounds,
        spec.payload_retention_rounds,
    )
    .map_err(|e| e.to_string())?;

    // Artifacts: the two modules + the published corpus objects (manifest, tokenizer, shards).
    //
    // Every `url` is derived from [`PublishedArtifact`] — the one place the run's published key
    // scheme is spelled — so the url a role fetches from is by construction the key the run's own
    // publisher (`xtask publish-module` / `publish-corpus`) writes. Spelling the scheme here as
    // well is how the corpus urls came to omit the publisher's suffix, which made every
    // genesis-pinned corpus fetch presign a key nothing had published.
    //
    // The publish layout also prefixes a run (`runs/<run>/modules/<blake3>.wasm`), but the run id is
    // the blake3 of THIS envelope — it cannot be known while authoring the envelope that defines it
    // (circular) — so the url carries the run-relative path and the presign surface prefixes the run
    // (`daemon_vhc_net::r2_object_key`, §11.3).
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "coordinator.wasm".to_string(),
        SnapshotArtifact {
            url: PublishedArtifact::Module(spec.coordinator_module).url(),
            blake3: spec.coordinator_module,
            size: None,
        },
    );
    artifacts.insert(
        "worker.wasm".to_string(),
        SnapshotArtifact {
            url: PublishedArtifact::Module(spec.trainer_module).url(),
            blake3: spec.trainer_module,
            size: None,
        },
    );
    let mut granted: BTreeSet<Hash> = BTreeSet::new();
    for (name, object) in spec.corpus_artifacts {
        let content_id = object.content_id();
        artifacts.insert(
            name.clone(),
            SnapshotArtifact {
                url: object.url(),
                blake3: content_id,
                size: None,
            },
        );
        granted.insert(content_id);
    }
    granted.insert(spec.corpus_manifest);

    let control_channel = |artifact_grants: BTreeSet<Hash>| RoleGrants {
        channels: vec![ChannelDecl {
            id: 0,
            name: "control".into(),
            class: 0,
            direction: 2,
            max_frame_bytes: 1 << 20,
            rate_per_min: 600,
            spool_frames: Some(256),
            replay_window: Some(1024),
            per_sender_quota: Some(64),
        }],
        artifacts: artifact_grants,
        ..RoleGrants::default()
    };

    let coordinator_base = spec
        .trusted_bases
        .first()
        .copied()
        .unwrap_or(PeerId([0; 32]));

    // The coordinator's opaque config (the guest's `da_init` shape): the round schedule derived
    // from the frozen trainer config, on the real clock the authored timers are denominated in.
    let coord_config = ceremony_coordinator_config(spec)?;

    let mut roles = BTreeMap::new();
    roles.insert(
        "coordinator".to_string(),
        RoleEntry {
            // No execution-requirement structure yet: the real one is obtained from the module's own
            // assessment export per the authoring flow, which is why nothing is hand-authored here.
            // `validate` refuses a runnable envelope carrying none, so this fails closed and loudly.
            execution: None,
            lane: "coordinator".into(),
            module: "coordinator.wasm".into(),
            abi: "vhc@2".into(),
            config: coord_config,
            grants: control_channel(BTreeSet::new()),
            device_min: daemon_vhc_proto::DeviceMinimums::default(),
        },
    );
    roles.insert(
        "trainer".to_string(),
        RoleEntry {
            // No execution-requirement structure yet: the real one is obtained from the module's own
            // assessment export per the authoring flow, which is why nothing is hand-authored here.
            // `validate` refuses a runnable envelope carrying none, so this fails closed and loudly.
            execution: None,
            lane: "trainer".into(),
            module: "worker.wasm".into(),
            abi: "vhc@2".into(),
            config: ceremony_trainer_config(spec.run_label, spec.corpus_manifest, spec.roster),
            grants: control_channel(granted),
            device_min: daemon_vhc_proto::DeviceMinimums::default(),
        },
    );

    let genesis = GenesisEnvelope {
        run: RunSection {
            schema: GENESIS_SCHEMA_MAJOR,
            run_label: spec.run_label.to_string(),
            min_peers: spec.min_peers,
            max_peers: spec.max_peers,
            access: Access::Org,
        },
        roles,
        artifacts,
        corpus_manifest: Some(spec.corpus_manifest),
        state_contract: Some(ceremony_state_contract()),
        authority: ceremony_authority(coordinator_base),
        transport: TransportSelection::default(),
        identities: Identities {
            coordinator: Some(coordinator_base),
            coordinator_set: spec.trusted_bases.to_vec(),
            upgrade_authority: spec.upgrade_authority.clone(),
        },
    };
    genesis
        .freeze(author)
        .map_err(|e| format!("freeze the ceremony genesis: {e}"))
}

/// The ceremony's round schedule for a `peers`-strong trainer roster, DERIVED from the frozen
/// trainer config: [`CEREMONY_STEPS_PER_ROUND`] × [`CEREMONY_MICRO_BATCH`] × `peers` sequences per
/// round, i.e. one micro-batch per inner step per peer.
///
/// The round window and the inner loop are one relationship, so the authoring computes the window
/// from the config it embeds rather than spelling a second number beside it: a window sized to
/// anything else (the roster size, say) leaves each peer a share the 30-step inner loop cannot
/// divide, and an unsliceable share plans no fetches at all — the peer trains nothing and never
/// commits.
#[must_use]
pub fn ceremony_round_schedule(peers: u32) -> RoundSchedule {
    RoundSchedule::derived(CEREMONY_STEPS_PER_ROUND, CEREMONY_MICRO_BATCH, peers)
}

/// The coordinator role's opaque `{state, tick_period_ms, verify_availability}` config (the
/// `da_init` shape), authored from the ceremony run parameters through the shared authoring seat
/// ([`crate::coordinator_config`]): the round schedule derived from the frozen trainer config, and
/// the real timer that makes the operator's calibrated seconds wall-clock seconds.
///
/// # Errors
/// The authoring refusals of [`coordinator_role_config`] (an unsliceable round schedule, or
/// wall-clock deadlines on a clock that does not measure wall time).
fn ceremony_coordinator_config(spec: &CeremonyGenesisSpec<'_>) -> Result<Value, String> {
    use daemon_vhc_proto::envelope::StopCondition;

    let peers = u32::try_from(spec.roster.len())
        .map_err(|_| "the ceremony trainer roster does not fit a peer count".to_string())?;
    coordinator_role_config(&CoordinatorAuthoring {
        run_label: spec.run_label,
        min_peers: spec.min_peers,
        max_peers: spec.max_peers,
        epoch_rounds: 0,
        stall_rounds_max: CEREMONY_STALL_ROUNDS_MAX,
        k_absences: CEREMONY_K_ABSENCES,
        seq_len: spec.seq_len,
        // The window the trainers can actually consume: derived from the trainer config this same
        // genesis embeds, over the roster it assigns against.
        schedule: ceremony_round_schedule(peers),
        // The real fleet timers, defaulting to the effectively-infinite values — byte-stable when
        // untuned; the preflight operator sets them from the slowest box's measured wall.
        deadlines: PhaseDeadlines {
            warmup_s: spec.timers.warmup_s,
            round_train_max_s: spec.timers.round_max_s,
            round_witness_s: spec.timers.witness_s,
            cooldown_s: spec.timers.cooldown_s,
        },
        tick_period_ms: CEREMONY_TICK_PERIOD_MS,
        stop: StopCondition::Rounds(spec.timers.stop_rounds),
        // The fleet coordinator does NOT availability-verify commitments against the content
        // plane (the guest default): an owner decision, not an authoring one.
        verify_availability: false,
    })
}

/// The envelope `authority` config (opaque to the host; D1's consensus SDK interprets it): the
/// single-key coordinator topology over the ceremony coordinator identity.
fn ceremony_authority(coordinator_base: PeerId) -> Value {
    use daemon_vhc_sdk_consensus::{AuthorityConfig, SingleKey, Topology, DEFAULT_RECORDS_CHANNEL};
    AuthorityConfig {
        topology: Topology::SingleKey(SingleKey::new(coordinator_base)),
        records_channel: DEFAULT_RECORDS_CHANNEL,
    }
    .encode()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frozen geometry's arithmetic, pinned to the AMENDED-fleet invariants (memory floor =
    /// the M4's 32 GiB unified memory; canonical state host-side and streamed, not in guest
    /// linear memory). A drift in any constant moves these sums and fails here.
    #[test]
    fn ceremony_geometry_is_frozen() {
        assert_eq!(
            CEREMONY_N_HEADS * CEREMONY_HEAD_DIM,
            CEREMONY_D_MODEL,
            "full-MHA width identity"
        );
        let numels = ceremony_param_numels();
        assert_eq!(
            numels.len(),
            (2 + 9 * CEREMONY_N_LAYERS) as usize,
            "embedding + 9 params/block + final norm"
        );
        let total = numels.iter().sum::<usize>() as u64;
        assert_eq!(total, CEREMONY_PARAM_COUNT, "the frozen parameter count");

        // On-device training working set (16 B/param) fits the M4 32 GiB unified floor with
        // activation headroom — the usable Metal working set on that box is comparable to the
        // old 24 GB-class discrete target, so the fp32 device state stays in the 11–16 GiB band.
        let device_state = total * CEREMONY_DEVICE_BYTES_PER_PARAM;
        assert_eq!(device_state, CEREMONY_DEVICE_STATE_BYTES);
        assert!(
            device_state > 11 * (1 << 30),
            "sized to the fleet: not shrunk below the ratified tier"
        );
        assert!(
            device_state < 16 * (1 << 30),
            "on-device fp32 state leaves activation headroom on the M4 unified floor"
        );

        // The device working set fits the usable budget WITH activation headroom…
        assert!(
            device_state < CEREMONY_FLOOR_USABLE_BYTES,
            "on-device set fits the floor peer's usable unified budget"
        );
        // …and the disk-backing decision is FORCED by arithmetic, not assumed: on the M4's single
        // unified pool the on-device set PLUS a RAM-resident retained det-state store overruns
        // that usable budget (the raw sum is 82% of the whole 32 GiB nameplate, leaving under
        // 6 GiB for activations/OS/host) — so the host state store MUST be disk-backed (only the
        // ~5-family retained figure lands on disk; RAM keeps index/lengths/refcounts).
        assert_eq!(
            CEREMONY_RETAINED_STATE_BYTES,
            5 * 4 * total,
            "≈5 retained families at the ratified cadence"
        );
        assert!(
            device_state + CEREMONY_RETAINED_STATE_BYTES > CEREMONY_FLOOR_USABLE_BYTES,
            "RAM-resident retained state would overrun the floor peer's usable budget beside the \
             device set — the arithmetic reason the state store is disk-backed"
        );

        // The largest single tensor (the tied embedding) stays under the tightest amended-fleet
        // per-buffer ceiling (wgpu/RADV + wgpu-DX12 ≈ 2047 MiB; Metal maxBufferLength is larger).
        let largest = numels.iter().max().copied().unwrap_or(0) as u64 * 4;
        assert_eq!(
            largest,
            u64::from(CEREMONY_VOCAB) * u64::from(CEREMONY_D_MODEL) * 4
        );
        assert!(
            largest < CEREMONY_PER_BUFFER_CEILING_BYTES,
            "largest tensor fits every fleet peer's per-buffer ceiling"
        );
    }

    /// The **standing retained-bytes assertion** (the disk-backing measurement, pinned): the
    /// host state store's retained roots at the ratified retention defaults, over the ceremony
    /// family geometry, MUST equal [`CEREMONY_RETAINED_STATE_BYTES`] (≈ 14.65 GiB) — and that
    /// figure, beside the on-device set, MUST overrun the floor peer's usable unified budget (the
    /// reason the store is disk-backed). This is tied to `STATE_RETAIN_ROOTS_DEFAULT` and the
    /// proto family-byte arithmetic, so a future retention/cadence change that would blow past the
    /// disk-backing assumption trips HERE — a test, not a fleet incident.
    #[test]
    #[allow(clippy::assertions_on_constants)] // deliberately pins constant sizing relationships
    fn ceremony_retained_state_matches_the_retention_model() {
        let numels: Vec<u64> = ceremony_param_numels().iter().map(|&n| n as u64).collect();
        // One consensus/replica family is the flat f32-le image of the layout — 4 B/param.
        let family_bytes = daemon_vhc_proto::det_state::family_byte_len(&numels);
        assert_eq!(
            family_bytes,
            4 * CEREMONY_PARAM_COUNT,
            "one family = 4 B/param"
        );

        // The retained set at defaults (design §8.2): `state_retain_roots` master roots (the
        // round base + the freshly sealed round) + 1 ef root + a sealed checkpoint's adamw_m and
        // adamw_v (its master/ef by-ref families dedup with the already-retained roots). ≈ 5
        // distinct families.
        let master_roots = daemon_vhc_proto::STATE_RETAIN_ROOTS_DEFAULT;
        let ef_roots = 1u64;
        let checkpoint_moment_families = 2u64; // adamw_m + adamw_v
        let retained_families = master_roots + ef_roots + checkpoint_moment_families;
        assert_eq!(retained_families, 5, "the ratified retained-family count");
        assert_eq!(
            family_bytes * retained_families,
            CEREMONY_RETAINED_STATE_BYTES,
            "retained roots at the ratified cadence == the pinned disk-backing figure"
        );

        // The disk-backing regime holds: retained roots + the on-device set overrun the floor
        // peer's usable unified budget, so they cannot both be RAM-resident.
        assert!(
            CEREMONY_DEVICE_STATE_BYTES + CEREMONY_RETAINED_STATE_BYTES
                > CEREMONY_FLOOR_USABLE_BYTES,
            "retained state must be disk-backed on the memory floor peer"
        );
    }

    /// The pinned expected root IS the fold of the seed-derived init over the ceremony layout —
    /// the [SF-5] admission cross-check every peer reproduces. This guards the seed, the versioned
    /// distribution (`daemon-vhc-det` dist v1), and the fold definition: any change trips here as
    /// a digest-value movement (stop-and-report), never a silent re-derivation.
    #[test]
    fn ceremony_expected_root_reproduces_the_pin() {
        let root = ceremony_expected_state_root();
        println!("CEREMONY_EXPECTED_ROOT = {}", root.to_hex());
        assert_ne!(
            root,
            Hash([0u8; 32]),
            "a real fold, not the zero placeholder"
        );
        assert_eq!(
            root,
            Hash(CEREMONY_EXPECTED_ROOT),
            "seed expansion reproduces the pinned expected_root"
        );
    }

    /// The ceremony state contract honours the genesis-authoring geometry rules ([SF-5], §3.2):
    /// the profile chunk divides every parameter numel, the state chunk size is a valid multiple
    /// of the profile-chunk byte width, and the pinned init root is the sealed seed expansion.
    #[test]
    fn ceremony_state_contract_honours_the_geometry_rules() {
        use daemon_vhc_proto::det_state::{validate_profile_chunk, validate_state_chunk_size};
        let numels: Vec<u64> = ceremony_param_numels().iter().map(|&n| n as u64).collect();
        // chunk | every numel (the 1536-wide norms make chunk = d_model binding).
        validate_profile_chunk(ceremony_profile_chunk(), &numels).expect("profile chunk divides");
        // The SparseLoco default 4096 is a refusal at this geometry.
        assert!(
            validate_profile_chunk(4096, &numels).is_err(),
            "default 4096 refuses"
        );
        // state_chunk_size is a non-zero multiple of chunk × 4.
        validate_state_chunk_size(ceremony_state_chunk_size(), ceremony_profile_chunk())
            .expect("state chunk size valid");
        assert_eq!(ceremony_state_chunk_size(), 682 * 6144);
        // The contract carries the seed form with the pinned, non-zero root.
        let contract = ceremony_state_contract();
        assert_eq!(contract.chunk_size, ceremony_state_chunk_size());
        match contract.init {
            StateInit::Seed {
                seed,
                dist,
                expected_root,
            } => {
                assert_eq!(seed.0, CEREMONY_INIT_SEED);
                assert_eq!(dist, CEREMONY_INIT_DIST);
                assert_eq!(expected_root, ceremony_expected_state_root());
            }
            StateInit::Manifest { .. } => panic!("the ceremony uses seed-derived init"),
        }
    }

    /// The ceremony genesis authors from the frozen model + a (ceremony-time) corpus pin + trust
    /// set: it validates, freezes, re-opens, and commits the corpus + seed-init pins into the run
    /// id — the executable-locally proof of the W-SF6 authoring path (the real fleet supplies the
    /// published corpus + certified peer identities at preflight).
    #[test]
    fn ceremony_genesis_authors_and_commits_the_pins() {
        let author = SigningKey::from_bytes(&[0x42; 32]);
        let base = |n: u8| daemon_vhc_proto::peer_id(&SigningKey::from_bytes(&[n; 32]));
        let trusted = [base(1), base(2), base(3)]; // Strix Halo + M4 + Windows 5090
        let manifest = Hash([0xAB; 32]);
        let corpus_artifacts = vec![
            (
                "corpus-manifest.cbor".to_string(),
                PublishedArtifact::CorpusManifest(manifest),
            ),
            (
                "shard-0.bin".to_string(),
                PublishedArtifact::CorpusShard(Hash([0x01; 32])),
            ),
        ];
        let spec = CeremonyGenesisSpec {
            run_label: "vhc-ceremony",
            coordinator_module: Hash([0xC0; 32]),
            trainer_module: Hash([0x7A; 32]),
            corpus_manifest: manifest,
            corpus_artifacts: &corpus_artifacts,
            seq_len: u64::from(CEREMONY_SEQ_LEN),
            trusted_bases: &trusted,
            roster: &trusted,
            upgrade_authority: vec![base(1)],
            min_peers: 3,
            max_peers: 3,
            remote_ckpt_cadence_rounds: 20,
            payload_retention_rounds: 64,
            timers: CeremonyRunTimers::default(),
        };
        let frozen = ceremony_genesis(&spec, &author).expect("author ceremony genesis");

        // Re-open the frozen wire (verifies signature + re-derives the hash) and validate.
        let reopened = FrozenGenesis::open(
            frozen.bytes().to_vec(),
            *frozen.signature(),
            *frozen.signer(),
        )
        .expect("reopen ceremony genesis");
        assert_eq!(reopened.run_id(), frozen.run_id());
        let env = reopened.decode().expect("decode");
        env.validate().expect("envelope validates");

        // The corpus + seed-init pins are committed (a different pin would be a different run id).
        assert_eq!(env.corpus_manifest, Some(manifest));
        let contract = env.state_contract.expect("state contract present");
        assert_eq!(contract, ceremony_state_contract());
        match contract.init {
            StateInit::Seed { expected_root, .. } => {
                assert_eq!(expected_root, Hash(CEREMONY_EXPECTED_ROOT))
            }
            StateInit::Manifest { .. } => panic!("seed-derived init"),
        }

        // Both canonical roles present; the trainer carries the FROZEN model verbatim.
        assert!(env.roles.contains_key("coordinator"));
        let trainer = env.roles.get("trainer").expect("trainer role");
        let Value::Map(cfg) = &trainer.config else {
            panic!("trainer config is a map");
        };
        let model = cfg
            .iter()
            .find_map(|(k, v)| matches!(k, Value::Text(t) if t == "model").then_some(v))
            .expect("model in trainer config");
        // Canonical CBOR reorders map keys on the freeze round-trip; compare the canonical
        // encodings (order-independent) to prove the frozen model is embedded verbatim.
        assert_eq!(
            daemon_vhc_proto::to_canonical_vec(model).unwrap(),
            daemon_vhc_proto::to_canonical_vec(&ceremony_model_value()).unwrap(),
            "frozen model embedded verbatim"
        );

        // Changing the cadence past the retention floor is refused at authoring.
        let mut bad = spec;
        bad.payload_retention_rounds = 30; // 20 + one churn slot (20) = 40 > 30
        assert!(
            ceremony_genesis(&bad, &author).is_err(),
            "cadence↔retention refused"
        );
    }

    /// The real fleet timers thread into the committed coordinator config: a tuned value moves the
    /// run id (the timers are inside the envelope that defines the run's identity).
    #[test]
    fn ceremony_timers_thread_into_the_genesis() {
        let author = SigningKey::from_bytes(&[0x42; 32]);
        let base = |n: u8| daemon_vhc_proto::peer_id(&SigningKey::from_bytes(&[n; 32]));
        let trusted = [base(1), base(2), base(3)];
        let manifest = Hash([0xAB; 32]);
        let corpus_artifacts = vec![(
            "corpus-manifest.cbor".to_string(),
            PublishedArtifact::CorpusManifest(manifest),
        )];
        let make = |timers: CeremonyRunTimers| CeremonyGenesisSpec {
            run_label: "vhc-ceremony",
            coordinator_module: Hash([0xC0; 32]),
            trainer_module: Hash([0x7A; 32]),
            corpus_manifest: manifest,
            corpus_artifacts: &corpus_artifacts,
            seq_len: u64::from(CEREMONY_SEQ_LEN),
            trusted_bases: &trusted,
            roster: &trusted,
            upgrade_authority: vec![base(1)],
            min_peers: 3,
            max_peers: 3,
            remote_ckpt_cadence_rounds: 8,
            payload_retention_rounds: 64,
            timers,
        };

        let default_run = ceremony_genesis(&make(CeremonyRunTimers::default()), &author)
            .expect("author with default timers");
        let tuned_run = ceremony_genesis(
            &make(CeremonyRunTimers {
                warmup_s: 120,
                round_max_s: 360,
                witness_s: 60,
                cooldown_s: 60,
                stop_rounds: 48,
            }),
            &author,
        )
        .expect("author with tuned timers");

        // The tuned timers live inside the envelope, so they move the run's cryptographic id.
        assert_ne!(
            default_run.run_id(),
            tuned_run.run_id(),
            "tuned fleet timers must change the committed run id"
        );
    }

    /// The committed coordinator config and the committed trainer config describe the SAME round:
    /// the window the coordinator opens is the roster's worth of the trainer's own inner loop, and
    /// the clock the deadlines are counted on measures wall time.
    ///
    /// These two halves are what a fleet round is made of. A window sized to anything but the
    /// trainer's schedule leaves each peer a share the inner loop cannot divide (no inner steps,
    /// no fetches, no commitment), and deadlines authored onto the event-driven clock count
    /// delivered events, so a quiet round cannot even time out.
    #[test]
    fn the_committed_round_schedule_matches_the_committed_trainer_config() {
        use crate::coordinator_config::WALL_CLOCK_TICK_PERIOD_MS;
        use daemon_vhc_sdk_consensus::coordinator::CoordinatorState;

        let author = SigningKey::from_bytes(&[0x42; 32]);
        let base = |n: u8| daemon_vhc_proto::peer_id(&SigningKey::from_bytes(&[n; 32]));
        let trusted = [base(1), base(2), base(3)];
        let manifest = Hash([0xAB; 32]);
        let corpus_artifacts = vec![(
            "corpus-manifest.cbor".to_string(),
            PublishedArtifact::CorpusManifest(manifest),
        )];
        let spec = CeremonyGenesisSpec {
            run_label: "vhc-ceremony",
            coordinator_module: Hash([0xC0; 32]),
            trainer_module: Hash([0x7A; 32]),
            corpus_manifest: manifest,
            corpus_artifacts: &corpus_artifacts,
            seq_len: u64::from(CEREMONY_SEQ_LEN),
            trusted_bases: &trusted,
            roster: &trusted,
            upgrade_authority: vec![base(1)],
            min_peers: 3,
            max_peers: 3,
            remote_ckpt_cadence_rounds: 8,
            payload_retention_rounds: 64,
            // The calibrated fleet timers: seconds, on a clock that measures seconds.
            timers: CeremonyRunTimers {
                warmup_s: 300,
                round_max_s: 600,
                witness_s: 300,
                cooldown_s: 60,
                stop_rounds: 48,
            },
        };
        let env = ceremony_genesis(&spec, &author)
            .expect("author ceremony genesis")
            .decode()
            .expect("decode");

        let field = |cfg: &Value, name: &str| -> Value {
            let Value::Map(entries) = cfg else {
                panic!("a role config is a map");
            };
            entries
                .iter()
                .find_map(|(k, v)| matches!(k, Value::Text(t) if t == name).then(|| v.clone()))
                .unwrap_or_else(|| panic!("`{name}` in the role config"))
        };
        let uint = |cfg: &Value, name: &str| -> u64 {
            u64::try_from(i128::from(
                field(cfg, name).as_integer().expect("an integer field"),
            ))
            .expect("a non-negative field")
        };

        // The trainer's half: the frozen inner loop over the fleet roster.
        let trainer = &env.roles.get("trainer").expect("trainer role").config;
        let steps = uint(trainer, "steps_per_round");
        let micro = uint(trainer, "micro_batch");
        let Value::Array(roster) = field(trainer, "roster") else {
            panic!("the trainer roster is an array");
        };
        assert_eq!(
            (steps, micro, roster.len()),
            (
                u64::from(CEREMONY_STEPS_PER_ROUND),
                u64::from(CEREMONY_MICRO_BATCH),
                trusted.len()
            )
        );

        // The coordinator's half: the window, and the clock the deadlines live on.
        let coordinator = &env
            .roles
            .get("coordinator")
            .expect("coordinator role")
            .config;
        let state: CoordinatorState = field(coordinator, "state")
            .deserialized()
            .expect("the coordinator state decodes");
        let expected = steps * micro * roster.len() as u64;
        assert_eq!(expected, 90, "30 inner steps × 1 sequence × 3 peers");
        assert_eq!(u64::from(state.config.global_batch.start), expected);
        assert_eq!(u64::from(state.config.global_batch.end), expected);
        assert_eq!(u64::from(state.config.steps_per_round), steps);
        assert_eq!(
            u64::from(state.config.global_batch.start) / roster.len() as u64 % steps,
            0,
            "a peer's share of the window is a whole number of inner steps"
        );
        assert_eq!(
            uint(coordinator, "tick_period_ms"),
            WALL_CLOCK_TICK_PERIOD_MS,
            "the authored seconds are counted on a clock that measures seconds"
        );
        assert_eq!(state.config.warmup_s, 300);
        assert_eq!(state.config.round_train_max_s, 600);
        assert_eq!(state.config.round_witness_s, 300);
        assert_eq!(state.config.cooldown_s, 60);
    }

    /// A roster the derivation cannot serve is an authoring refusal, not a run that parks: an
    /// empty trainer roster assigns nobody any sequences.
    #[test]
    fn ceremony_genesis_refuses_a_roster_it_cannot_schedule() {
        let author = SigningKey::from_bytes(&[0x42; 32]);
        let base = |n: u8| daemon_vhc_proto::peer_id(&SigningKey::from_bytes(&[n; 32]));
        let trusted = [base(1), base(2), base(3)];
        let manifest = Hash([0xAB; 32]);
        let corpus_artifacts = vec![(
            "corpus-manifest.cbor".to_string(),
            PublishedArtifact::CorpusManifest(manifest),
        )];
        let spec = CeremonyGenesisSpec {
            run_label: "vhc-ceremony",
            coordinator_module: Hash([0xC0; 32]),
            trainer_module: Hash([0x7A; 32]),
            corpus_manifest: manifest,
            corpus_artifacts: &corpus_artifacts,
            seq_len: u64::from(CEREMONY_SEQ_LEN),
            trusted_bases: &trusted,
            roster: &[],
            upgrade_authority: vec![base(1)],
            min_peers: 3,
            max_peers: 3,
            remote_ckpt_cadence_rounds: 8,
            payload_retention_rounds: 64,
            timers: CeremonyRunTimers::default(),
        };
        let err = ceremony_genesis(&spec, &author).expect_err("an empty roster must refuse");
        assert!(err.contains("assignment roster is empty"), "{err}");
    }

    /// The derivation is the trainer config's own arithmetic, at every fleet size.
    #[test]
    fn the_round_schedule_is_derived_from_the_frozen_trainer_config() {
        for peers in 1..=3u32 {
            let schedule = ceremony_round_schedule(peers);
            assert_eq!(
                schedule.global_batch,
                CEREMONY_STEPS_PER_ROUND * CEREMONY_MICRO_BATCH * peers
            );
            assert_eq!(schedule.sequences_per_peer(), CEREMONY_STEPS_PER_ROUND);
            schedule
                .validate()
                .expect("the derived schedule is sliceable");
        }
        // The single-peer smoke and the three-peer fleet: 30 and 90 sequences per round.
        assert_eq!(ceremony_round_schedule(1).global_batch, 30);
        assert_eq!(ceremony_round_schedule(3).global_batch, 90);
    }

    #[test]
    fn ceremony_model_value_round_trips_canonically() {
        let v = ceremony_model_value();
        let bytes = daemon_vhc_proto::to_canonical_vec(&v).expect("canonical encode");
        let back: Value = ciborium::de::from_reader(bytes.as_slice()).expect("decode");
        // Canonical encoding reorders map keys; the round trip preserves the ENTRIES.
        let pairs = |val: &Value| -> Vec<(String, Value)> {
            let Value::Map(entries) = val else {
                panic!("model config is a map");
            };
            let mut out: Vec<(String, Value)> = entries
                .iter()
                .map(|(k, v)| match k {
                    Value::Text(t) => (t.clone(), v.clone()),
                    other => panic!("non-text key {other:?}"),
                })
                .collect();
            out.sort_by(|a, b| a.0.cmp(&b.0));
            out
        };
        assert_eq!(pairs(&v), pairs(&back));
        assert_eq!(pairs(&v).len(), 14, "the frozen field set");
    }
}
