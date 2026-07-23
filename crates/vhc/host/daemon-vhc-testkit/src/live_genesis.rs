// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Genesis authoring for the MULTI-PROCESS acceptance topology (architecture §6): one
//! coordinator role (the seat-claimed `coordinator_quorum.wasm`) plus one trainer role
//! (`tiny_llama.wasm` in its module-driven LIVE mode) over a genesis-pinned, chunk-addressed
//! corpus manifest — the run three REAL node processes join through their product API.
//!
//! This is an AUTHORING seat only (the testkit is the harness-exempt place allowed to link the
//! consensus schema for fixture authorship): the acceptance suite consumes the frozen wire bytes
//! and never decodes a round message itself.
//!
//! The trainer tier is the ratified structural acceptance configuration: 64-dim residual width
//! with REAL multi-layer structure (two transformer blocks), small enough for the CPU/ndarray
//! lane. Assignment runs REPLICATED (a single-member roster shared by every trainer instance:
//! each trains the round's whole global window) — per-peer sharded assignment needs a module
//! identity surface the ABI does not expose yet, and digest agreement is equally meaningful over
//! replicated windows.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use ciborium::value::Value;

use daemon_vhc_proto::envelope::{Access, GlobalBatch, StopCondition};
use daemon_vhc_proto::genesis::{
    ChannelDecl, GenesisEnvelope, Identities, RoleEntry, RoleGrants, RunSection, SnapshotArtifact,
    TransportSelection, GENESIS_SCHEMA_MAJOR,
};
use daemon_vhc_proto::{
    blake3_hash, to_canonical_vec, CapabilitySet, CorpusManifest, Hash, PeerId, Seed,
    SignedEnvelope, SigningKey, VHC_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, RunConfig as CoordinatorRunConfig};
use daemon_vhc_sdk_consensus::{AuthorityConfig, SingleKey, Topology, DEFAULT_RECORDS_CHANNEL};

/// The structural acceptance trainer geometry: 64-dim, two REAL transformer blocks, 4 heads.
const D_MODEL: u32 = 64;
const N_LAYERS: u32 = 2;
const N_HEADS: u32 = 4;
const HEAD_DIM: u32 = 16;
const FFN_MULT: u32 = 2;

/// The replicated-assignment placeholder identity: every trainer instance shares this one-member
/// roster, so each trains the round's whole global window (module docs above).
const REPLICATED_PEER: [u8; 32] = [0x11; 32];

/// One authored acceptance genesis: the signed wire bytes plus the identities the suite pins.
pub struct LiveGenesis {
    /// The canonical-CBOR [`SignedEnvelope`] wire bytes (what the registry serves).
    pub wire: Vec<u8>,
    /// The run's cryptographic identity (the frozen genesis hash).
    pub genesis_hash: [u8; 32],
    /// The corpus manifest's content hash (the `corpus_manifest` pin).
    pub manifest_hash: [u8; 32],
    /// The corpus objects the payload plane must hold, `(content key, bytes)`: the manifest
    /// under its blake3, each shard under its chunk-fold identity.
    pub corpus_objects: Vec<([u8; 32], Vec<u8>)>,
}

/// The coordinator's liveness timing. The baseline tiers run EVENT-DRIVEN (no timer, effectively
/// infinite deadlines: every phase exits on the all-submitted fast path, so a wall-clock stall can
/// never mis-finalize a healthy round). The churn tiers arm the real timer so a SILENT member is
/// survivable: a round whose member vanished finalizes at `round_train_max_s` synthetic ticks with
/// an absence mark, `k_absences` drops the member, the `min_peers` floor breach forces the
/// cooldown, and the cooldown exit materializes staged rejoins into the next epoch's roster.
#[derive(Clone, Copy)]
pub struct LiveTiming {
    /// Real coordinator timer period in ms (`0` = event-driven synthetic clock, no timer).
    pub tick_period_ms: u64,
    /// Warmup timeout in synthetic ticks (early-exits when every member is ready).
    pub warmup_s: u64,
    /// Round training deadline in synthetic ticks (early-exits when all committed).
    pub round_train_max_s: u64,
    /// Witness/evidence deadline in synthetic ticks (early-exits when all evidenced).
    pub round_witness_s: u64,
    /// Cooldown dwell in synthetic ticks (the pending-join materialization point).
    pub cooldown_s: u64,
}

impl Default for LiveTiming {
    fn default() -> Self {
        Self {
            tick_period_ms: 0,
            warmup_s: 1_000_000,
            round_train_max_s: 1_000_000,
            round_witness_s: 1_000_000,
            cooldown_s: 1_000_000,
        }
    }
}

impl LiveTiming {
    /// The churn tier: a 500ms real tick (~2 synthetic ticks/s, plus one per delivered event).
    /// Deadlines are sized so a HEALTHY round (~1-2s wall, ~10-15 tick-equivalents) never grazes
    /// them, while a vanished member costs ~20s of wall time before its round finalizes with an
    /// absence mark; warmup exits by timeout when a ghost (dead incarnation still in the roster)
    /// can never signal readiness; cooldown resolves promptly so staged rejoins materialize.
    #[must_use]
    pub fn churn() -> Self {
        Self {
            tick_period_ms: 500,
            warmup_s: 30,
            round_train_max_s: 40,
            round_witness_s: 20,
            cooldown_s: 2,
        }
    }
}

/// The authored-genesis knobs the acceptance topology varies.
pub struct LiveGenesisSpec<'a> {
    /// The run label (registry key + coordinator admission id).
    pub run_label: &'a str,
    /// The coordinator module bytes + the artifact URL its assess resolves it from.
    pub coordinator_wasm: &'a [u8],
    /// A resolvable artifact URL for the coordinator module (e.g. `file://…`).
    pub coordinator_url: String,
    /// The trainer module bytes + the artifact URL its assess resolves it from.
    pub trainer_wasm: &'a [u8],
    /// A resolvable artifact URL for the trainer module.
    pub trainer_url: String,
    /// The vendored chunk-addressed corpus fixture directory (`corpus-manifest.cbor` + the
    /// fold-named shard files beside it).
    pub corpus_dir: &'a Path,
    /// The genesis-trusted base identities (every participating node's base key).
    pub trusted_bases: &'a [PeerId],
    /// Members the run waits for before warmup ends.
    pub min_peers: u32,
    /// Roster capacity.
    pub max_peers: u32,
    /// Epoch length in rounds (`0` = no epoch boundaries; churn/rejoin tiers need boundaries
    /// for pending members to materialize).
    pub epoch_rounds: u32,
    /// Sequences per round across the roster (replicated: each trainer trains all of them).
    pub global_batch: u32,
    /// Inner steps per round.
    pub steps_per_round: u32,
    /// Record absences before a silent member drops.
    pub k_absences: u32,
    /// The coordinator's liveness timing (see [`LiveTiming`]).
    pub timing: LiveTiming,
    /// The run's upgrade authority (whose unanimous signatures authorize a module-upgrade
    /// record — architecture §5.4). Empty = an immutable run (no upgrade admits).
    pub upgrade_authority: Vec<PeerId>,
}

/// Derive the grants-document hash a role's admission authors for `module` under this genesis
/// (the module's linked worlds ∪ the genesis role grant list — the same deterministic
/// derivation assess/join/switch use), i.e. the `grants_hash` anchor a module-upgrade record
/// must carry (architecture §5.4; the worker re-derives and compares fail-closed).
///
/// # Errors
/// A human-readable failure (undecodable wire, unknown role, engine/linking failure).
pub fn role_grants_hash(
    genesis_wire: &[u8],
    role: &str,
    module: &[u8],
) -> Result<[u8; 32], String> {
    let wire: SignedEnvelope = daemon_vhc_proto::from_canonical_slice(genesis_wire)
        .map_err(|e| format!("genesis wire: {e}"))?;
    let frozen = daemon_vhc_proto::FrozenGenesis::open(wire.bytes, wire.signature, wire.signer)
        .map_err(|e| format!("open genesis: {e}"))?;
    let env = frozen
        .decode()
        .map_err(|e| format!("decode genesis: {e}"))?;
    let entry = env
        .roles
        .get(role)
        .ok_or_else(|| format!("role `{role}` absent from the genesis role set"))?;
    let worker = daemon_vhc_host::Worker::new(daemon_vhc_host::EngineConfig::default())
        .map_err(|e| format!("engine: {e}"))?;
    let linked = daemon_vhc_host::linked_worlds(&worker, module)
        .map_err(|e| format!("linked worlds: {e}"))?;
    let grants = daemon_vhc_proto::GrantsDoc::author(&linked, &entry.grants).to_canonical_bytes();
    Ok(*blake3::hash(&grants).as_bytes())
}

/// Author the acceptance genesis: the coordinator role carrying its opaque
/// `{state: CoordinatorState}` config, the trainer role carrying the LIVE tiny-llama config
/// (module-driven corpus + wire announcements), the corpus manifest pinned as a mapped
/// artifact, and every participating node's base identity genesis-trusted.
///
/// # Panics
/// On unreadable fixture files or CBOR authoring failures — fixture authorship fails loud.
// Reads the vendored corpus fixture from disk: this is harness authoring tooling (never a shipped
// node path), so the workspace's raw-fs ban does not apply here.
#[allow(clippy::disallowed_methods)]
#[must_use]
pub fn live_genesis(spec: &LiveGenesisSpec<'_>) -> LiveGenesis {
    let manifest_bytes = std::fs::read(spec.corpus_dir.join("corpus-manifest.cbor"))
        .expect("read the vendored corpus manifest");
    let manifest = CorpusManifest::from_canonical_bytes(&manifest_bytes)
        .expect("the vendored corpus manifest parses");
    let manifest_hash = blake3_hash(&manifest_bytes);

    // The corpus objects the payload plane serves: the manifest by content hash, each shard by
    // its fold identity (the fixture files are fold-named by the authoring pipeline).
    let mut corpus_objects = vec![(manifest_hash.0, manifest_bytes.clone())];
    let mut granted: BTreeSet<Hash> = BTreeSet::new();
    granted.insert(manifest_hash);
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "corpus-manifest.cbor".to_string(),
        SnapshotArtifact {
            url: format!("r2://corpus/{}.cbor", manifest_hash.to_hex()),
            blake3: manifest_hash,
            size: Some(manifest_bytes.len() as u64),
        },
    );
    for (i, shard) in manifest.shards.iter().enumerate() {
        let bytes = std::fs::read(
            spec.corpus_dir
                .join(format!("{}.bin", shard.shard_hash.to_hex())),
        )
        .expect("read a vendored corpus shard");
        granted.insert(shard.shard_hash);
        artifacts.insert(
            format!("shard-{i}.bin"),
            SnapshotArtifact {
                url: format!("r2://corpus/{}.bin", shard.shard_hash.to_hex()),
                blake3: shard.shard_hash,
                size: Some(bytes.len() as u64),
            },
        );
        corpus_objects.push((shard.shard_hash.0, bytes));
    }

    let coordinator_hash = blake3_hash(spec.coordinator_wasm);
    let trainer_hash = blake3_hash(spec.trainer_wasm);
    artifacts.insert(
        "coordinator.wasm".to_string(),
        SnapshotArtifact {
            url: spec.coordinator_url.clone(),
            blake3: coordinator_hash,
            size: None,
        },
    );
    artifacts.insert(
        "worker.wasm".to_string(),
        SnapshotArtifact {
            url: spec.trainer_url.clone(),
            blake3: trainer_hash,
            size: None,
        },
    );

    // The coordinator's opaque config: the authored RunConfig + genesis CoordinatorState (the
    // guest's `da_init` shape; event-driven synthetic clock).
    let run_config = CoordinatorRunConfig {
        run_id: spec.run_label.to_string(),
        proto_version: VHC_PROTO_VERSION,
        envelope_hash: Hash([0u8; 32]),
        required_capabilities: CapabilitySet::new(),
        min_peers: spec.min_peers,
        max_peers: spec.max_peers,
        warmup_s: spec.timing.warmup_s,
        round_train_max_s: spec.timing.round_train_max_s,
        round_witness_s: spec.timing.round_witness_s,
        cooldown_s: spec.timing.cooldown_s,
        epoch_rounds: u64::from(spec.epoch_rounds),
        stall_rounds_max: 4,
        global_batch: GlobalBatch {
            start: spec.global_batch,
            end: spec.global_batch,
            ramp_rounds: 1,
        },
        stop: StopCondition::Rounds(1_000_000),
        steps_per_round: spec.steps_per_round,
        seq_len: u64::from(manifest.seq_len),
        witness_target: 0,
        overlap_bps: 0,
        k_absences: spec.k_absences,
        verification_percent: 0,
        authorized: Vec::new(),
    };
    let state = CoordinatorState::new(run_config, Seed([0x33; 32]), 0);
    let coord_config = Value::Map(vec![
        (
            Value::Text("state".into()),
            Value::serialized(&state).expect("state to cbor value"),
        ),
        // A churn tier arms the coordinator's real timer so deadlines actually pass; the
        // event-driven default (0) never arms one.
        (
            Value::Text("tick_period_ms".into()),
            Value::Integer(spec.timing.tick_period_ms.into()),
        ),
        // The live deployment shape: commitments are availability-verified against the run's
        // content-addressed plane (coordinator-as-storage-client, §6.4 I6). The deterministic
        // harness rings leave this off (the guest default).
        (Value::Text("verify_availability".into()), Value::Bool(true)),
    ]);

    // Grants: the control channel both modules declare; the trainer additionally holds the
    // corpus artifact grants (the manifest + every shard fold) its data@2 fetches admit under.
    // Numeric quotas stay unset (0 = inherit the lane ceiling, tighten-only): a real multi-layer
    // TinyLlama step's compute-queue depth rides the Trainer lane ceiling
    // (`ParticipationLane::trainer_launch_defaults`), which the production path sizes for training.
    let control_channel = |artifact_grants: BTreeSet<Hash>| RoleGrants {
        channels: vec![ChannelDecl {
            id: 0,
            name: "control".into(),
            class: 0,     // authoritative
            direction: 2, // bidirectional
            max_frame_bytes: 1 << 20,
            rate_per_min: 600,
            spool_frames: Some(256),
            replay_window: Some(1024),
            per_sender_quota: Some(64),
        }],
        artifacts: artifact_grants,
        ..RoleGrants::default()
    };

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
            config: trainer_live_config(spec.run_label, &manifest, manifest_hash),
            grants: control_channel(granted),
            device_min: daemon_vhc_proto::DeviceMinimums::default(),
        },
    );

    let coordinator_base = spec
        .trusted_bases
        .first()
        .copied()
        .unwrap_or(PeerId([0; 32]));
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
        corpus_manifest: Some(manifest_hash),
        state_contract: Some(live_state_contract(64)),
        authority: AuthorityConfig {
            topology: Topology::SingleKey(SingleKey::new(coordinator_base)),
            records_channel: DEFAULT_RECORDS_CHANNEL,
        }
        .encode(),
        transport: TransportSelection::default(),
        identities: Identities {
            coordinator: Some(coordinator_base),
            coordinator_set: spec.trusted_bases.to_vec(),
            upgrade_authority: spec.upgrade_authority.clone(),
        },
    };
    let author = SigningKey::from_bytes(&[0x42; 32]);
    let frozen = genesis
        .freeze(&author)
        .expect("freeze the acceptance genesis");
    let genesis_hash = frozen.run_id().0;
    let wire = SignedEnvelope {
        bytes: frozen.bytes().to_vec(),
        signature: *frozen.signature(),
        signer: *frozen.signer(),
    };
    LiveGenesis {
        wire: to_canonical_vec(&wire).expect("wire encode"),
        genesis_hash,
        manifest_hash: manifest_hash.0,
        corpus_objects,
    }
}

/// The trainer's canonical parameter element counts for the structural tier (`ModelCfg::
/// param_numels` — token embedding, per-block 9 params × [`N_LAYERS`], final norm).
fn param_numels(vocab: u32) -> Vec<usize> {
    let (d, qdim, hidden, vocab) = (
        D_MODEL as usize,
        (N_HEADS * HEAD_DIM) as usize,
        (FFN_MULT * D_MODEL) as usize,
        vocab as usize,
    );
    let mut out = vec![vocab * d];
    for _ in 0..N_LAYERS {
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

fn text(s: &str) -> Value {
    Value::Text(s.into())
}

fn uint(v: u64) -> Value {
    Value::Integer(v.into())
}

/// The LIVE trainer guest config (raw canonical CBOR against the guest's documented schema):
/// the structural model geometry sized to the corpus manifest's sequence length + tokenizer
/// vocabulary headroom, the replicated single-member roster, the SparseLoco profile, the
/// deterministic matched init, and the `live` section naming the run + the manifest pin.
fn trainer_live_config(run_label: &str, manifest: &CorpusManifest, manifest_hash: Hash) -> Value {
    // The model vocabulary: a fixed power-of-two ceiling over the fixture tokenizer's ids (the
    // in-guest `token % vocab` clamp is then the identity for well-formed fixtures).
    let vocab = 64u32;
    let model = Value::Map(vec![
        (text("d_model"), uint(u64::from(D_MODEL))),
        (text("n_layers"), uint(u64::from(N_LAYERS))),
        (text("n_heads"), uint(u64::from(N_HEADS))),
        (text("head_dim"), uint(u64::from(HEAD_DIM))),
        (text("vocab"), uint(u64::from(vocab))),
        (text("seq_len"), uint(u64::from(manifest.seq_len))),
        (text("ffn_mult"), uint(u64::from(FFN_MULT))),
        (text("rope_theta"), Value::Float(10_000.0)),
        (text("rmsnorm_eps"), Value::Float(1.0e-5)),
        (text("lr"), Value::Float(4.0e-4)),
        (text("beta1"), Value::Float(0.9)),
        (text("beta2"), Value::Float(0.95)),
        (text("adam_eps"), Value::Float(1.0e-8)),
        (text("wd"), Value::Float(0.1)),
    ]);
    let profile = Value::Map(vec![
        (text("h"), uint(3)),
        (text("ef_decay"), Value::Float(0.95)),
        (text("chunk"), uint(64)),
        (text("topk"), uint(8)),
        (text("bits"), uint(2)),
        (text("outer_alpha"), Value::Float(1.0)),
        (text("clip"), Value::Bool(false)),
    ]);
    let live = Value::Map(vec![
        (text("run_label"), text(run_label)),
        (
            text("manifest"),
            Value::serialized(&manifest_hash).expect("manifest hash value"),
        ),
    ]);
    Value::Map(vec![
        (text("model"), model),
        (text("peer"), Value::Bytes(REPLICATED_PEER.to_vec())),
        (
            text("roster"),
            Value::Array(vec![Value::Bytes(REPLICATED_PEER.to_vec())]),
        ),
        (text("steps_per_round"), uint(2)),
        (text("micro_batch"), uint(1)),
        (text("stall_rounds_max"), uint(4)),
        (text("profile"), profile),
        (
            text("state"),
            Value::serialized(&live_state_contract(vocab)).expect("state contract value"),
        ),
        (text("live"), live),
    ])
}

/// The live run's genesis seed-form state contract (§6.1a) for the trainer layout at `vocab`:
/// the derived state chunk size + a pinned `(seed, dist)` the guest expands, self-seals, and
/// cross-checks against `expected_root` (replaces the deleted inline init).
pub fn live_state_contract(vocab: u32) -> daemon_vhc_proto::genesis::StateContract {
    use daemon_vhc_proto::det_state::{derive_state_chunk_size, FamilyEntry};
    use daemon_vhc_proto::genesis::{StateContract, StateInit};
    let seed = [0x5eu8; 32];
    let dist = daemon_vhc_det::SEED_INIT_DIST_V1;
    let chunk_size = derive_state_chunk_size(64);
    let param_bytes: Vec<Vec<u8>> = param_numels(vocab)
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
        .expect("author")
        .fold;
    StateContract {
        chunk_size,
        init: StateInit::Seed {
            seed: daemon_vhc_proto::Seed(seed),
            dist,
            expected_root,
        },
    }
}
