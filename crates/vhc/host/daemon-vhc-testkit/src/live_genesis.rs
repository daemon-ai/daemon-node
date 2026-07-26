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

use crate::coordinator_config::{
    coordinator_role_config, CoordinatorAuthoring, PhaseDeadlines, RoundSchedule,
};
use daemon_vhc_net::PublishedArtifact;
use daemon_vhc_proto::envelope::{Access, StopCondition};
use daemon_vhc_proto::genesis::{
    ChannelDecl, GenesisEnvelope, Identities, RoleEntry, RoleGrants, RunSection, SnapshotArtifact,
    TransportSelection, GENESIS_SCHEMA_MAJOR,
};
use daemon_vhc_proto::{
    blake3_hash, to_canonical_vec, CorpusManifest, Hash, PeerId, SignedEnvelope, SigningKey,
};
use daemon_vhc_sdk_consensus::{AuthorityConfig, SingleKey, Topology, DEFAULT_RECORDS_CHANNEL};

/// The structural acceptance trainer geometry: 64-dim, two REAL transformer blocks, 4 heads.
const D_MODEL: u32 = 64;
const N_LAYERS: u32 = 2;
const N_HEADS: u32 = 4;
const HEAD_DIM: u32 = 16;
const FFN_MULT: u32 = 2;

/// Sequences per inner step — the trainer config's `micro_batch`, and the value the coordinator's
/// round window is authored around (one config, so the two cannot disagree).
const MICRO_BATCH: u32 = 1;
/// Fetch-recovery budget before a stalled peer leaves for the epoch.
const STALL_ROUNDS_MAX: u32 = 4;

/// The replicated-assignment placeholder identity: every trainer instance shares this one-member
/// roster, so each trains the round's whole global window (module docs above).
const REPLICATED_PEER: [u8; 32] = [0x11; 32];

/// The length of that roster: the round window is split one way.
const REPLICATED_ROSTER_PEERS: u32 = 1;

/// One authored acceptance genesis: the signed wire bytes plus the identities the suite pins.
pub struct LiveGenesis {
    /// The canonical-CBOR [`SignedEnvelope`] wire bytes (what the registry serves).
    pub wire: Vec<u8>,
    /// The run's cryptographic identity (the frozen genesis hash).
    pub genesis_hash: [u8; 32],
    /// The corpus manifest's content hash (the `corpus_manifest` pin).
    pub manifest_hash: [u8; 32],
    /// The corpus objects a run's content planes must hold — each carrying the **genesis-pinned
    /// url** it is published at, so a harness cannot stage them anywhere the envelope does not say.
    pub corpus_objects: Vec<CorpusObject>,
}

/// One published corpus object: the content id the module names, the url the envelope pins, and
/// the bytes.
///
/// Carrying the url (rather than the id alone) is what keeps a staging harness honest. The run-time
/// artifact plane resolves a pinned content id at the url the ENVELOPE commits —
/// [`PublishedArtifact`]'s `corpus/<manifest blake3>.cbor`, `corpus/<fold>.bin` (ABI §12.7 [CC-7],
/// what `xtask publish-corpus` writes) — which
/// is a different namespace from the committed-payload plane's `payload/<blake3>`. A harness that
/// staged corpus objects under the payload key instead proved only that the payload plane works: it
/// left the published layout ungated, and the first fleet trainer to reach the data plane died
/// fetching its genesis-pinned corpus manifest from a key nothing had ever published it at.
#[derive(Clone, Debug)]
pub struct CorpusObject {
    /// The content id the guest's `data.fetch` names: the manifest's blake3, or a shard's
    /// domain-separated chunk FOLD (which never equals `blake3(bytes)`).
    pub id: [u8; 32],
    /// The genesis artifact-map url this object is published at (`r2://corpus/<…>`).
    pub url: String,
    /// The object bytes.
    pub bytes: Vec<u8>,
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
    /// The per-role execution requirements this seat **places**, derived upstream from each module's
    /// own assessment output.
    ///
    /// This spec does carry module bytes, so deriving in-seat would be possible — it is an input
    /// anyway, for two reasons. It keeps all three authoring seats the same shape, so there is no
    /// seat where the reader has to ask whether the requirement was placed or invented; and it keeps
    /// an envelope builder from standing up a wasm engine as a side effect of authoring. Callers
    /// derive through [`daemon_vhc_host::run::author_execution`], which is the single seam.
    pub execution: daemon_vhc_proto::AuthoredExecution,
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

    // The corpus objects the run's content planes serve: the manifest by content hash, each shard
    // by its fold identity (the fixture files are fold-named by the authoring pipeline). Each
    // carries the url the envelope pins it at, derived from [`PublishedArtifact`] — the same key
    // scheme `xtask publish-corpus` writes at, so this genesis cannot spell a key of its own.
    let manifest_url = PublishedArtifact::CorpusManifest(manifest_hash).url();
    let mut corpus_objects = vec![CorpusObject {
        id: manifest_hash.0,
        url: manifest_url.clone(),
        bytes: manifest_bytes.clone(),
    }];
    let mut granted: BTreeSet<Hash> = BTreeSet::new();
    granted.insert(manifest_hash);
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "corpus-manifest.cbor".to_string(),
        SnapshotArtifact {
            url: manifest_url,
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
        let url = PublishedArtifact::CorpusShard(shard.shard_hash).url();
        artifacts.insert(
            format!("shard-{i}.bin"),
            SnapshotArtifact {
                url: url.clone(),
                blake3: shard.shard_hash,
                size: Some(bytes.len() as u64),
            },
        );
        corpus_objects.push(CorpusObject {
            id: shard.shard_hash.0,
            url,
            bytes,
        });
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

    // The coordinator's opaque `da_init` config, through the shared authoring seat (the one place
    // a genesis's round schedule and clock are decided — `crate::coordinator_config`). A churn
    // tier arms the coordinator's real timer so deadlines actually pass; the event-driven default
    // (0) never arms one, and the seat refuses wall-clock deadlines authored beside it.
    //
    // `verify_availability`: the live deployment shape — commitments are availability-verified
    // against the run's content-addressed plane (coordinator-as-storage-client, §6.4 I6). The
    // deterministic harness rings leave this off (the guest default).
    let coord_config = coordinator_role_config(&CoordinatorAuthoring {
        run_label: spec.run_label,
        min_peers: spec.min_peers,
        max_peers: spec.max_peers,
        epoch_rounds: u64::from(spec.epoch_rounds),
        stall_rounds_max: STALL_ROUNDS_MAX,
        k_absences: spec.k_absences,
        seq_len: u64::from(manifest.seq_len),
        // Assignment is REPLICATED (module docs): the trainer config's roster has one member, so
        // the whole round window is every trainer's own interval and the split is one-way.
        schedule: RoundSchedule::explicit(
            spec.global_batch,
            spec.steps_per_round,
            MICRO_BATCH,
            REPLICATED_ROSTER_PEERS,
        ),
        deadlines: PhaseDeadlines {
            warmup_s: spec.timing.warmup_s,
            round_train_max_s: spec.timing.round_train_max_s,
            round_witness_s: spec.timing.round_witness_s,
            cooldown_s: spec.timing.cooldown_s,
        },
        tick_period_ms: spec.timing.tick_period_ms,
        stop: StopCondition::Rounds(1_000_000),
        verify_availability: true,
    })
    .expect("the acceptance genesis authors a round its trainers can slice");

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
            // Placed, not composed: what the module's own assessment produced, derived upstream. A
            // role absent from the authored set stays `None`, which `validate` refuses for a runnable
            // envelope — defaulting here would be this seat inventing a resource requirement.
            execution: spec.execution.for_role("coordinator"),
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
            // Placed, not composed: what the module's own assessment produced, derived upstream. A
            // role absent from the authored set stays `None`, which `validate` refuses for a runnable
            // envelope — defaulting here would be this seat inventing a resource requirement.
            execution: spec.execution.for_role("trainer"),
            lane: "trainer".into(),
            module: "worker.wasm".into(),
            abi: "vhc@2".into(),
            config: trainer_live_config(
                spec.run_label,
                &manifest,
                manifest_hash,
                spec.steps_per_round,
            ),
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
fn trainer_live_config(
    run_label: &str,
    manifest: &CorpusManifest,
    manifest_hash: Hash,
    steps_per_round: u32,
) -> Value {
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
        // The inner loop the coordinator's round window was authored around: one config, so the
        // window a round opens is always a whole number of this trainer's steps.
        (text("steps_per_round"), uint(u64::from(steps_per_round))),
        (text("micro_batch"), uint(u64::from(MICRO_BATCH))),
        (text("stall_rounds_max"), uint(u64::from(STALL_ROUNDS_MAX))),
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

/// An [`AuthoredExecution`](daemon_vhc_proto::AuthoredExecution) over the canonical trivial plan,
/// covering the `coordinator` and `trainer` roles, for a **fixture** that authors an envelope without
/// a module to assess.
///
/// Named for what it is. Most negative and geometry fixtures pin module digests that resolve to no
/// bytes at all — they are testing an authoring rule, not a module — so there is nothing to assess and
/// requiring one would only force every such fixture to carry a wasm blob it never runs. What this
/// must never become is a shortcut on the real path: the plan inside reports
/// `is_module_derived() == false`, and a scan rule keeps this function out of non-test targets.
#[must_use]
pub fn fixture_authored_execution() -> daemon_vhc_proto::AuthoredExecution {
    fixture_authored_execution_for(&["coordinator", "trainer"])
}

/// The same, for a named role set.
#[must_use]
pub fn fixture_authored_execution_for(roles: &[&str]) -> daemon_vhc_proto::AuthoredExecution {
    let plan = daemon_vhc_proto::ModuleDerivedPlan::fixture(
        daemon_vhc_proto::LogicalResourcePlan::trivial(
            daemon_vhc_proto::WASM_GUEST_LINEAR_FLOOR_BYTES,
        ),
    );
    let grant = daemon_vhc_proto::ExecutionGrant {
        logical_resource_plan_hash: plan.plan().plan_hash().expect("the trivial plan hashes"),
        scope: daemon_vhc_proto::SelectionScope::UniformRun,
        values: std::collections::BTreeMap::new(),
    };
    let mut authored = daemon_vhc_proto::AuthoredExecution::new();
    for role in roles {
        authored = authored
            .derive(
                role,
                &plan,
                vec!["cpu".to_string()],
                daemon_vhc_proto::ProfileCertificationRequirements::default(),
                daemon_vhc_proto::HardwareIndependentMinima::default(),
                Some(&grant),
            )
            .expect(
                "the canonical trivial plan and its empty uniform grant derive by construction",
            );
    }
    authored
}

/// Derive the live cluster's per-role execution requirements from the two real modules it runs.
///
/// The counterpart of [`fixture_authored_execution`] for a caller that *has* modules: the acceptance
/// cluster runs real guests, so its envelope pins what those guests' own assessment said rather than
/// a fixture's stand-in. That is the whole point of the seam — a harness with modules has no excuse
/// to author a requirement nothing derived.
///
/// # Errors
/// A human-readable failure when a module cannot be assessed or its plan will not derive.
pub fn live_execution(
    coordinator_wasm: &[u8],
    trainer_wasm: &[u8],
) -> Result<daemon_vhc_proto::AuthoredExecution, String> {
    crate::ceremony::ceremony_execution(coordinator_wasm, trainer_wasm)
}
