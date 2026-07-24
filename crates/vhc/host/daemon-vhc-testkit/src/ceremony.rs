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

use daemon_vhc_proto::det_state::{
    derive_state_chunk_size, family_byte_len, family_fold, validate_checkpoint_cadence,
    validate_profile_chunk, validate_state_chunk_size,
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
    let roster_val = Value::Array(roster.iter().map(|p| Value::Bytes(p.0.to_vec())).collect());
    let peer = roster.first().map_or_else(
        || Value::Bytes(vec![0u8; 32]),
        |p| Value::Bytes(p.0.to_vec()),
    );
    let live = Value::Map(vec![
        (text("run_label"), text(run_label)),
        (
            text("manifest"),
            Value::serialized(&corpus_manifest).expect("manifest hash value"),
        ),
    ]);
    Value::Map(vec![
        (text("model"), ceremony_model_value()),
        (text("peer"), peer),
        (text("roster"), roster_val),
        (text("steps_per_round"), Value::Integer(30u64.into())),
        (text("micro_batch"), Value::Integer(1u64.into())),
        (text("stall_rounds_max"), Value::Integer(4u64.into())),
        (text("profile"), ceremony_profile_value()),
        (
            text("state"),
            Value::serialized(&ceremony_state_contract()).expect("state contract value"),
        ),
        (text("live"), live),
    ])
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
    /// The published corpus objects to map + grant, `(artifact name, content hash)` — the
    /// manifest plus its shard folds (the trainer's `data@2` fetch grants).
    pub corpus_artifacts: &'a [(String, Hash)],
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
}

/// Author + freeze the ceremony genesis (§15 W-SF6): the trainer role carrying the FROZEN model +
/// the seed-form state contract (pinned `expected_root`) + the corpus pin, the coordinator role,
/// the fleet trust set, and the upgrade authority. Signs with `author`. The corpus + init pins
/// commit into the genesis hash (the run's cryptographic id).
///
/// # Errors
/// A human-readable failure when a genesis-authoring geometry rule is violated (profile chunk
/// does not divide the layout, an invalid state chunk size, or a checkpoint cadence that could
/// strand a rejoiner past payload retention) or the envelope fails to validate/freeze.
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

    // Artifacts: the two modules + the published corpus objects (manifest + shards).
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "coordinator.wasm".to_string(),
        SnapshotArtifact {
            url: format!("r2://mods/{}.wasm", spec.coordinator_module.to_hex()),
            blake3: spec.coordinator_module,
            size: None,
        },
    );
    artifacts.insert(
        "worker.wasm".to_string(),
        SnapshotArtifact {
            url: format!("r2://mods/{}.wasm", spec.trainer_module.to_hex()),
            blake3: spec.trainer_module,
            size: None,
        },
    );
    let mut granted: BTreeSet<Hash> = BTreeSet::new();
    for (name, hash) in spec.corpus_artifacts {
        artifacts.insert(
            name.clone(),
            SnapshotArtifact {
                url: format!("r2://corpus/{}", hash.to_hex()),
                blake3: *hash,
                size: None,
            },
        );
        granted.insert(*hash);
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

    // The coordinator's opaque config (the guest's `da_init` shape), event-driven synthetic clock
    // — the deterministic authoring shape; the fleet operator tunes real timers at preflight.
    let coord_config = ceremony_coordinator_config(spec, coordinator_base);

    let mut roles = BTreeMap::new();
    roles.insert(
        "coordinator".to_string(),
        RoleEntry {
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

/// The coordinator role's opaque `{state, …}` config (the `da_init` shape), authored from the
/// ceremony run parameters — the event-driven synthetic-clock shape (real fleet timers are a
/// preflight operator choice).
fn ceremony_coordinator_config(spec: &CeremonyGenesisSpec<'_>, coordinator_base: PeerId) -> Value {
    use daemon_vhc_proto::envelope::{GlobalBatch, StopCondition};
    use daemon_vhc_proto::{CapabilitySet, VHC_PROTO_VERSION};
    use daemon_vhc_sdk_consensus::coordinator::{
        CoordinatorState, RunConfig as CoordinatorRunConfig,
    };

    let _ = coordinator_base;
    let run_config = CoordinatorRunConfig {
        run_id: spec.run_label.to_string(),
        proto_version: VHC_PROTO_VERSION,
        envelope_hash: Hash([0u8; 32]),
        required_capabilities: CapabilitySet::new(),
        min_peers: spec.min_peers,
        max_peers: spec.max_peers,
        warmup_s: 1_000_000,
        round_train_max_s: 1_000_000,
        round_witness_s: 1_000_000,
        cooldown_s: 1_000_000,
        epoch_rounds: 0,
        stall_rounds_max: 4,
        global_batch: GlobalBatch {
            start: spec.max_peers.max(1),
            end: spec.max_peers.max(1),
            ramp_rounds: 1,
        },
        stop: StopCondition::Rounds(1_000_000),
        steps_per_round: 30,
        seq_len: spec.seq_len,
        witness_target: 0,
        overlap_bps: 0,
        k_absences: 3,
        verification_percent: 0,
        authorized: Vec::new(),
    };
    let state = CoordinatorState::new(run_config, Seed([0x33; 32]), 0);
    Value::Map(vec![(
        Value::Text("state".into()),
        Value::serialized(&state).expect("coordinator state to cbor value"),
    )])
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
            ("corpus-manifest.cbor".to_string(), manifest),
            ("shard-0.bin".to_string(), Hash([0x01; 32])),
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
