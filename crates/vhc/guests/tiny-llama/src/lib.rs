// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `tiny-llama` — the reference worker training module: `models::TinyLlama`, formerly an
//! SDK-hosted stand-in, re-authored here as a real Burn model that runs inside the guest.
//!
//! The model ([`model::TinyLlamaModel`]) is ordinary Burn over `Autodiff<HostBackend>`: every tensor op
//! crosses `compute@2` as `CBOR(burn_ir::OperationIr)`; the autodiff tape walks guest-side with
//! zero intermediate readbacks. The comm profile is `daemon-vhc-sdk-profiles::SparseLoco` — its
//! det-lane ingest runs **in-guest with zero host support** (the `daemon-vhc-det` compatibility
//! path, architecture §3.2), over guest-held canonical masters/round bases. Choreography is
//! `BarrierRound` (sdk-rounds) with `Vec<u8>` payloads: committed payloads arrive as kind-0
//! staged bytes and are **blake3-verified in-guest** at `Committed::mint`.
//!
//! ## Config (canonical CBOR map)
//!
//! `{"model": ModelCfg, "peer": bstr32, "roster": [bstr32…], "steps_per_round": uint,
//! "micro_batch": uint, "stall_rounds_max": uint, "profile": SparseLocoCfg, "init": [f32…]}` —
//! `init` is the canonical flat state (concatenated params in registration order; matched init).
//!
//! ## Wire (module-defined, all on the `control` channel)
//!
//! - staged **in** (kind-0 bytes): `[0, round, step, sequences, seq_len, tokens_le]` a batch;
//!   `[1, round, peer32, payload]` a committed payload (unwrapped before the mint hash check).
//! - published **out**: `[2, round, theta_le]` the trained θ (canonical order, post-inner-steps —
//!   the C3b/C3c native-lane comparison surface); `[3, round, hash32]` the round commitment
//!   voice; `[4, round, digest16]` the post-ingest det digest (the v1 digest formula, computed
//!   in-guest).
//! - **payload plane**: the round's sealed committed container is `payload_put` before the tag-3
//!   voice (B1 discipline — the guest authors and externalizes its own payload; the embedder's
//!   async-runtime seat services the put and verifies the tag-3 hash covers exactly those bytes).
//!
//! ## Checkpoint / migration (ABI §10.2)
//!
//! On `Quiesce` the guest snapshots its state as a typed manifest of four flat f32-le sections:
//! `master` (the canonical det-lane masters — consensus-canonical, class 0; post-ingest this IS
//! the round base), `ef` (the profile's error-feedback residuals), and `adamw_m`/`adamw_v` (the
//! AdamW moments). The latter three are replica-local (class 1) and digest-invisible but
//! REQUIRED for continuity: the next round's `make_update` reads `ef` and the next round's
//! training reads the moments (v1 semantics — the outer step never resets them), so a restore
//! without them forks the committed-payload trajectory even though the ingest-side digest would
//! still match. The moments live device-side, so the quiesce is an async walk exactly like the
//! round export (fence → `export` → completions → stage + `snapshot_state` → `QuiesceReady`).
//! `da_migrate` restores all four and `run` rebuilds the model from them. A quiesce is a
//! between-rounds fence: trained-but-uncommitted state is deliberately not snapshotted (the
//! upgrade transaction drains at the round boundary).
//!
//! ## The export walk (why `make_update` is split)
//!
//! `BarrierRound` calls `make_update` synchronously inside the `RoundOpen` slice, but θ lives
//! device-side: extraction is fence → `export` → `Completion(BufferHandle)` → `read` through the
//! event loop (architecture §3.2/§3.4 — no synchronous readback exists). So the driver's
//! `make_update` returns empty (its `Commit` outbound is dropped), and the real profile
//! `make_update` + publishes run when the round's export completions finish — the same
//! deferred-voice shape as the retired bridge trainer's held commit.

mod model;

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;

use daemon_vhc_proto::capability::CapabilitySet;
use daemon_vhc_proto::corpus::{CorpusManifest, Endianness, TokenWidth};
use daemon_vhc_proto::det_state::{
    encode_checkpoint_doc, family_byte_len, CkptDocSection, DetStateChunkMap, DetStateManifest,
    FamilyRef, MASTER_FAMILY,
};
use daemon_vhc_proto::genesis::{StateContract, StateInit};
use daemon_vhc_proto::{blake3_hash, from_canonical_slice, to_canonical_vec, Hash, PeerId, Seed};
use daemon_vhc_proto::{IrohId, StateDigest};
use daemon_vhc_sdk::{
    GuestModule, MigrationDescriptor, MigrationSection, ModuleDecl, SectionDecl, SectionReader,
    StateManifest,
};
use daemon_vhc_sdk_compute::{export_tensor, fence, AutodiffHostBackend, HostBackend};
use daemon_vhc_sdk_consensus::digest::DigestCarry;
use daemon_vhc_sdk_consensus::fold_walk::Window;
use daemon_vhc_sdk_consensus::messages::{
    Commitment, Digest, Heartbeat, Join, RecordEntry, RoundOpen, ThroughputClass, VhcMessage,
};
use daemon_vhc_sdk_profiles::streaming::{
    f32s_to_le_bytes, le_bytes_to_f32s, SparseLocoIngestWalk, SparseLocoUpdateWalk,
    UpdateWindowInputs,
};
use daemon_vhc_sdk_profiles::{encode_payload, Section, SparseLocoCfg};
use daemon_vhc_sdk_rounds::{
    interval_for, slice_interval, BarrierRound, Committed, IngestOutcome, PayloadSource, RoundCfg,
    RoundExperiment, StepCtx as RoundStepCtx,
};
use serde::Deserialize;

/// The digest block size (the pinned det-lane granularity, matching `digest_state`).
const DIGEST_BLOCK: u32 = 64;
/// The in-flight window bound for the streamed fold walks (bounded read-ahead; the honest fuel
/// claim is per-window, §5.5). Small — the harness geometry is a handful of windows.
const WALK_IN_FLIGHT: u64 = 4;
/// The max in-flight restore window fetches (bounded read-ahead): the three restore families are
/// streamed with refill so the walk never exceeds the admitted `max_outstanding_ops`.
const RESTORE_IN_FLIGHT: usize = 4;
/// The error-feedback family tag (replica-local, digest-invisible — never in the state root).
const EF_FAMILY: &str = "ef";
/// The AdamW first-moment family tag (replica-local; materialized as a sealed family only at
/// checkpoint/drain from the device export).
const ADAMW_M_FAMILY: &str = "adamw_m";
/// The AdamW second-moment family tag (replica-local; as above).
const ADAMW_V_FAMILY: &str = "adamw_v";

use model::{ModelCfg, TinyLlamaModel};

const EV_FRAME: u64 = 0;
const EV_PAYLOAD_READY: u64 = 1;
const EV_TIMER: u64 = 2;
const EV_STOP: u64 = 4;
const EV_FENCE: u64 = 5;
const EV_COMPLETION: u64 = 6;
const EV_QUIESCE: u64 = 7;
/// `da_run` outcome after a §10.2 snapshot-accepted drain.
const OUTCOME_QUIESCE_READY: u32 = 2;
/// `da_migrate` Incompatible detail: a section is missing/unknown or its length mismatches the
/// model layout (§10.2 — module-defined detail codes are ≥ 16).
const MIGRATE_INCOMPATIBLE_SECTIONS: u32 = 16;

/// Per-step queue-depth-reset fences start here; round-final fences are `round + 1`.
const STEP_FENCE_BASE: u64 = 1 << 32;

#[derive(Deserialize)]
struct GuestCfg {
    model: ModelCfg,
    peer: PeerId,
    roster: Vec<PeerId>,
    steps_per_round: u32,
    micro_batch: u32,
    stall_rounds_max: u32,
    profile: SparseLocoCfg,
    /// The genesis **state contract** (§6.3): the run-pinned `state_chunk_size` + the init pin
    /// (seed-derived or content-addressed artifact). Replaces the deleted inline `init: Vec<f32>`
    /// — canonical state is chunk-addressed host-side now, so the config drops from gigabytes to
    /// bytes and the matched init is expanded (seed) or fetched (artifact) at run start.
    state: StateContract,
    /// The MODULE-DRIVEN live mode (absent = the harness contract above): the module announces
    /// itself on the control plane (Join/ready-Heartbeat), fetches its own training data from
    /// the genesis-pinned chunk-addressed corpus via `data@2`, fetches committed peer payloads
    /// from the content-addressed payload plane, and publishes the wire control messages
    /// (Commitment/Digest) the run's coordinator consumes — no host staging anywhere.
    #[serde(default)]
    live: Option<LiveCfg>,
}

/// The live-mode config half (module-driven data + wire announcements).
#[derive(Deserialize)]
struct LiveCfg {
    /// The run label the coordinator admits joins under (`RunConfig.run_id`).
    run_label: String,
    /// The genesis-pinned chunk-addressed corpus manifest's content hash (a granted artifact).
    manifest: Hash,
    /// The **local** checkpoint cadence in rounds (D-SF3): the round boundaries at which a
    /// checkpoint is considered. The canonical families are already sealed every round, so a local
    /// checkpoint is pointer bookkeeping — cheap, restorable by a co-located/LAN peer. `0` disables
    /// the cadence (the drain snapshot remains the only pointer source); absent = every round.
    #[serde(default = "default_ckpt_every")]
    ckpt_every: u64,
    /// The **remote** upload cadence in rounds (D-SF3, byte-budgeted): every `remote_ckpt_every`-th
    /// checkpoint boundary, the ONE deterministically-elected publisher for that slot uploads the
    /// by-reference document + its family chunks to the payload plane (others skip — R identical
    /// uploads per round are pure waste). `0` = upload at every local boundary (the default when
    /// unset). Genesis authoring refuses a cadence that could strand a rejoiner past
    /// `payload_retention_rounds` ([`daemon_vhc_proto::det_state::validate_checkpoint_cadence`]).
    #[serde(default)]
    remote_ckpt_every: u64,
}

/// The default live checkpoint cadence: every ingested round.
fn default_ckpt_every() -> u64 {
    1
}

/// One staged batch: `(sequences, seq_len, tokens)` in training order.
type BatchItem = (u32, u32, Vec<u32>);

/// The shared model/profile/det-lane state both the driver-called experiment and the event loop
/// mutate (wasm is single-threaded; `Rc<RefCell>` is the natural share).
struct Core {
    model: TinyLlamaModel<AutodiffHostBackend>,
    /// The profile config — the fold walks' geometry. No resident `SparseLoco` (the profile's
    /// error-feedback state is a host-side sealed family now, [`Self::ef_fold`]).
    profile_cfg: SparseLocoCfg,
    /// The parameter numels (registration order) — the walk geometry.
    numels: Vec<usize>,
    /// Per-parameter byte base offsets into the flat family image (prefix sums of `numel × 4`) —
    /// maps a fold [`Window`] `(param, param_off)` to an absolute `data@2::fetch` offset.
    family_base: Vec<u64>,
    /// The run-pinned `state_chunk_size` (the fold-walk window size).
    window_size: u64,
    /// The current round-base master family fold: the init fold before round 0, then each round's
    /// sealed master ([SF-R1] self-sealed) — the artifact the ingest AND update walks read their
    /// round-base windows from. THE canonical master lives here (host-side), not resident.
    master_fold: [u8; 32],
    /// The current master family's ordered chunk hashes (accumulated as each chunk is emitted) —
    /// what a by-reference checkpoint section ([SF-6]) lists for the sealed master, so the
    /// checkpoint costs zero extra reads of the already-sealed family.
    master_chunks: Vec<Hash>,
    /// The current error-feedback family fold: a zeroed family sealed at boot, then each
    /// `make_update`'s emitted ef family ([SF-R1] self-sealed). The update walk reads its ef
    /// windows from here; the replica-local ef lives here, not resident.
    ef_fold: [u8; 32],
    /// The current ef family's ordered chunk hashes (as [`Self::master_chunks`], for the ef
    /// by-reference section).
    ef_chunks: Vec<Hash>,
    /// The in-flight streamed ingest walk (one at a time; the barrier serializes on its seal).
    ingest_walk: Option<IngestWalkState>,
    /// The in-flight streamed `make_update` walk (kicked off when the round's θ export completes).
    update_walk: Option<UpdateWalkState>,
    /// Whether the next kicked-off ingest voices its digest at seal. A `RoundRecord`-triggered
    /// (or resume-triggered) ingest voices; a catch-up ingest kicked off inside `on_round_open`
    /// folds SILENTLY (the digest is implicit in the folded state) — the original guest dropped
    /// `on_round_open`'s outbounds and voiced only from the record handler.
    ingest_voices: bool,
    /// Host-staged batches, FIFO in training order.
    batches: VecDeque<BatchItem>,
    /// Accumulated micro-batch gradients of the current inner step (summed in arrival order,
    /// the v1 grad-buffer accumulation).
    pending_grads: Option<Vec<burn::tensor::Tensor<HostBackend, 1>>>,
    /// Fence-id mint for the per-step depth-reset fences.
    next_step_fence: u64,
}

/// The in-flight streamed ingest walk (ABI §12.14, design §3.4/§5.4): the resident
/// `SparseLoco::ingest` re-expressed as a completion-driven multi-slice state machine over
/// round-base windows fetched from [`Core::master_fold`], emitting master windows into a
/// `state_open` stream and threading the digest carry — kicked off by
/// [`RoundExperiment::begin_ingest`], driven by fetch completions, sealed in the final slice.
struct IngestWalkState {
    /// The round being folded.
    round: u64,
    /// The fold engine (owns the payloads, the schedule, and the digest carry).
    walk: SparseLocoIngestWalk,
    /// The open master-family write stream (`state_open`).
    stream: u64,
    /// The round-base family fold this walk reads its windows from.
    base_fold: [u8; 32],
    /// Outstanding round-base fetches: `data@2` op → window ordinal.
    ops: BTreeMap<u64, u64>,
    /// Whether this walk's digest is voiced at seal (record-triggered) or folds silently
    /// (a catch-up ingest kicked off inside `on_round_open`).
    voice: bool,
    /// The emitted master assembled per parameter — a TRANSIENT device-upload buffer freed with
    /// the walk (the post-ingest master uploads to the device once sealed; no resident master).
    master_assembly: Vec<Vec<f32>>,
    /// Scratch for the f32→le state-emit seam.
    byte_buf: Vec<u8>,
    /// The emitted master family's ordered chunk hashes (accumulated per `state_emit`) — moved
    /// into [`Core::master_chunks`] at seal, so a checkpoint references the sealed master by fold.
    chunk_hashes: Vec<Hash>,
}

/// The in-flight streamed `make_update` walk (design §5.4): the resident
/// `SparseLoco::make_update` re-expressed as a completion-driven walk over (θ, round-base, ef)
/// windows — θ resident-decoded from the round's device export, round-base + ef windows fetched
/// from the sealed folds — emitting the NEW ef family into a `state_open` stream and assembling
/// the payload sections; sealed, it puts the committed container and voices the tag-3 commitment.
struct UpdateWalkState {
    /// The round this update commits.
    round: u64,
    /// The fold engine (owns the schedule + the per-parameter section fragments).
    walk: SparseLocoUpdateWalk,
    /// The trained θ (device export), decoded per parameter — sliced per window.
    theta: Vec<Vec<f32>>,
    /// The new ef family write stream (`state_open("ef")`).
    ef_stream: u64,
    /// The round-base (master r−1) fold the θ⁽ᵗ⁾ windows read from.
    base_fold: [u8; 32],
    /// The prior ef family fold the ef windows read from.
    ef_fold: [u8; 32],
    /// Outstanding round-base window fetches: `data@2` op → window ordinal.
    rb_ops: BTreeMap<u64, u64>,
    /// Outstanding ef window fetches: `data@2` op → window ordinal.
    ef_ops: BTreeMap<u64, u64>,
    /// Arrived round-base windows awaiting their ef pair: ordinal → values.
    rb: BTreeMap<u64, Vec<f32>>,
    /// Arrived ef windows awaiting their round-base pair: ordinal → values.
    ef: BTreeMap<u64, Vec<f32>>,
    /// Scratch for the f32→le state-emit seam.
    byte_buf: Vec<u8>,
    /// The emitted NEW ef family's ordered chunk hashes — moved into [`Core::ef_chunks`] at seal.
    ef_chunks: Vec<Hash>,
}

/// The `RoundExperiment` adapter: v1 call points over the shared core.
struct C3Round {
    core: Rc<RefCell<Core>>,
}

impl RoundExperiment<Vec<u8>> for C3Round {
    fn train_step(&mut self, ctx: &RoundStepCtx) {
        let mut core = self.core.borrow_mut();
        let (sequences, seq_len, tokens) = core
            .batches
            .pop_front()
            .expect("a staged batch per train_step (the harness stages in training order)");
        let step_seqs = u32::try_from(ctx.micro.end - ctx.micro.start).unwrap_or(1);
        // v1 loss scaling: size / step_seqs (`api.rs::loss_scale`) — 1.0 in the parity harness.
        let loss_scale = f64::from(sequences) / f64::from(step_seqs.max(1));
        let grads =
            core.model
                .forward_backward(&tokens, sequences as usize, seq_len as usize, loss_scale);
        // Accumulate across the inner step's micro-batches (the v1 grad buffer).
        core.pending_grads = Some(match core.pending_grads.take() {
            None => grads,
            Some(acc) => acc.into_iter().zip(grads).map(|(a, g)| a.add(g)).collect(),
        });
    }

    fn inner_update(&mut self, inner_step: u32) {
        let mut core = self.core.borrow_mut();
        let grads = core
            .pending_grads
            .take()
            .expect("train_step accumulated gradients");
        core.model.adamw_apply(&grads, inner_step);
        // Reset the compute queue-depth window without blocking (§3.3: fencing reclaims depth;
        // the Event::Fence is ignored — only round-final fences are awaited).
        let id = STEP_FENCE_BASE + core.next_step_fence;
        core.next_step_fence += 1;
        fence(id);
    }

    fn make_update(&mut self, _round: u64) -> Vec<u8> {
        // θ lives device-side; the real profile make_update runs on the round's export
        // completions (module docs). The driver's Commit outbound is dropped in `emit`.
        Vec::new()
    }

    fn ingest(&mut self, _round: u64, _committed: &Committed<Vec<u8>>) -> [u8; 16] {
        // The streaming trainer never folds synchronously — it defers via `begin_ingest` and
        // voices the digest when the walk seals (the deferred-ingest seam, sdk-rounds).
        unreachable!("the streaming trainer defers ingest via begin_ingest")
    }

    /// Kick off the streamed det-lane ingest ([SF-4], design §5.4): decode the record-ordered
    /// committed payloads, open a `master` write stream, build the `SparseLocoIngestWalk` with a
    /// round-seeded digest carry, and issue the opening round-base window reads against
    /// [`Core::master_fold`] (the init fold before round 0, then the prior round's sealed master).
    /// The walk is stashed in `Core` and driven by fetch completions ([`drive_ingest_completion`]);
    /// the digest is voiced at seal via [`BarrierRound::finish_ingest`]. Returns `Deferred`.
    fn begin_ingest(&mut self, round: u64, committed: &Committed<Vec<u8>>) -> IngestOutcome {
        let payloads: Vec<Vec<Section>> = committed
            .items()
            .iter()
            .map(|it| {
                daemon_vhc_sdk_profiles::decode_payload(&it.bytes)
                    .expect("committed payload decodes (hash-verified at mint)")
            })
            .collect();
        let mut core = self.core.borrow_mut();
        let (numels, window_size, base_fold) =
            (core.numels.clone(), core.window_size, core.master_fold);
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&round.to_le_bytes());
        let carry = DigestCarry::new(&Seed(seed), DIGEST_BLOCK);
        let numels_u64: Vec<u64> = numels.iter().map(|&n| n as u64).collect();
        let byte_len = family_byte_len(&numels_u64);
        let mut walk = SparseLocoIngestWalk::new(
            &core.profile_cfg,
            &numels,
            window_size,
            WALK_IN_FLIGHT,
            &payloads,
            carry,
        )
        .expect("ingest walk geometry (profile chunk divides every numel; window aligned)");
        let opening = walk.start().expect("ingest walk start");
        let stream = daemon_vhc_sdk::state_open(MASTER_FAMILY, byte_len);
        let mut ops = BTreeMap::new();
        for w in &opening.issue {
            let off = core.family_base[w.param as usize] + w.param_off;
            let op = daemon_vhc_sdk::data_fetch(&base_fold, off, w.len);
            ops.insert(op, w.ordinal);
        }
        let voice = core.ingest_voices;
        let master_assembly = numels.iter().map(|&n| vec![0.0f32; n]).collect();
        core.ingest_walk = Some(IngestWalkState {
            round,
            walk,
            stream,
            base_fold,
            ops,
            voice,
            master_assembly,
            byte_buf: Vec::new(),
            chunk_hashes: Vec::new(),
        });
        IngestOutcome::Deferred
    }
}

/// The guest's payload source at the barrier: unwrapped committed payloads by `(round, peer)`;
/// `Committed::mint` re-verifies each against the record-listed blake3 in-guest.
struct PayloadMap {
    map: BTreeMap<(u64, PeerId), Vec<u8>>,
}

impl PayloadSource<Vec<u8>> for PayloadMap {
    fn payload(&mut self, round: u64, peer: &PeerId) -> Option<Vec<u8>> {
        self.map.get(&(round, *peer)).cloned()
    }
}

// -- the GuestModule under `main!` -------------------------------------------------------------------

/// The by-reference families a `da_migrate` restore carries into `run` ([SF-6]): the restoring
/// instance registers each fold ([SF-R2]) and streams it window-by-window in `da_run` — no bulk
/// bytes cross the migrate seam (whose only legal read is `read_back(kind=3)`, §6.6, used solely
/// for the inline round watermark).
struct RestoredRefs {
    master: FamilyRef,
    ef: FamilyRef,
    adamw_m: FamilyRef,
    adamw_v: FamilyRef,
    /// The snapshot's resync watermark (the last round its state folds), when the snapshot
    /// recorded one — the restored driver never re-ingests at or below it (§9 restore).
    round: Option<u64>,
}

struct TinyLlama {
    cfg_bytes: Vec<u8>,
    restored: Option<RestoredRefs>,
}

impl GuestModule for TinyLlama {
    fn decl() -> ModuleDecl {
        ModuleDecl {
            name: "tiny-llama",
            version: env!("CARGO_PKG_VERSION"),
            // The det-state write surface (state_open/emit/seal) + register_state_chunks force
            // the minor-3 declaration (ABI §1.3 step 5; the highest introducing minor imported).
            abi_minor: 3,
            channels: vec![0],
            // Guest-side det-lane state (masters + bases + ef) + decode scratch; device holds
            // the working weights + AdamW moments + activations.
            host_state_bytes: 16 << 20,
            host_scratch_bytes: 16 << 20,
            device_state_bytes: 32 << 20,
            device_scratch_bytes: 32 << 20,
        }
    }

    fn init(config: &[u8], _grants: &[u8]) -> Result<Self, u32> {
        // Imports are illegal during `da_init` (ABI §6.6): parse-only here; the model (whose
        // build crosses compute@2) is constructed at `run` start, where imports are legal.
        if from_canonical_slice::<GuestCfg>(config).is_err() {
            return Err(16);
        }
        Ok(Self {
            cfg_bytes: config.to_vec(),
            restored: None,
        })
    }

    fn run(&mut self) -> u32 {
        let cfg: GuestCfg =
            from_canonical_slice(&self.cfg_bytes).expect("config validated at init");
        run_module(cfg, self.restored.take())
    }

    /// The §10.2 consuming protocol (v2, [SF-6]): record the by-reference `master`/`ef`/`adamw_m`/
    /// `adamw_v` family descriptors (fold + geometry + chunk hashes) the host built from the
    /// checkpoint document, plus the inline `round` watermark. `da_migrate` does NO bulk reads
    /// (§6.6 legality unchanged) — `da_run` registers each fold ([SF-R2]) and STREAMS it.
    fn migrate(&mut self, descriptor: &MigrationDescriptor, reader: &mut dyn SectionReader) -> u32 {
        let mut master: Option<FamilyRef> = None;
        let mut ef: Option<FamilyRef> = None;
        let mut adamw_m: Option<FamilyRef> = None;
        let mut adamw_v: Option<FamilyRef> = None;
        let mut round: Option<u64> = None;
        for binding in &descriptor.sections {
            match binding {
                MigrationSection::Inline { name, staging_id } if name == "round" => {
                    // The resync watermark: 8 bytes LE; `u64::MAX` = the snapshot predates any
                    // ingest (no watermark). The one legal `read_back(kind=3)` in `da_migrate`.
                    let bytes = reader.read(*staging_id);
                    let Ok(raw) = <[u8; 8]>::try_from(bytes.as_slice()) else {
                        return MIGRATE_INCOMPATIBLE_SECTIONS;
                    };
                    let val = u64::from_le_bytes(raw);
                    round = (val != u64::MAX).then_some(val);
                }
                MigrationSection::ByRef { name, family } => match name.as_str() {
                    "master" => master = Some(family.clone()),
                    "ef" => ef = Some(family.clone()),
                    "adamw_m" => adamw_m = Some(family.clone()),
                    "adamw_v" => adamw_v = Some(family.clone()),
                    _ => return MIGRATE_INCOMPATIBLE_SECTIONS,
                },
                MigrationSection::Inline { .. } => return MIGRATE_INCOMPATIBLE_SECTIONS,
            }
        }
        let (Some(master), Some(ef), Some(adamw_m), Some(adamw_v)) = (master, ef, adamw_m, adamw_v)
        else {
            return MIGRATE_INCOMPATIBLE_SECTIONS;
        };
        self.restored = Some(RestoredRefs {
            master,
            ef,
            adamw_m,
            adamw_v,
            round,
        });
        0
    }
}

daemon_vhc_sdk::main!(TinyLlama);

/// Publish `[tag, round, bytes]` on the control channel.
fn publish_tagged(tag: u64, round: u64, bytes: &[u8]) {
    let v = ciborium::value::Value::Array(vec![
        ciborium::value::Value::from(tag),
        ciborium::value::Value::from(round),
        ciborium::value::Value::Bytes(bytes.to_vec()),
    ]);
    if let Ok(payload) = to_canonical_vec(&v) {
        let _ = daemon_vhc_sdk::publish(0, &payload);
    }
}

/// Publish a canonical-CBOR wire control message on the records channel (live mode): the
/// session signs the §12.1 transport envelope; the coordinator's tick consumes the payload.
fn publish_wire(msg: &VhcMessage) {
    if let Ok(payload) = to_canonical_vec(msg) {
        let _ = daemon_vhc_sdk::publish(0, &payload);
    }
}

/// One pending batch slot of an in-flight round open (live mode): the slot's sequence count and
/// its fetched shard segments, in corpus order.
struct PendingBatch {
    sequences: u32,
    segs: Vec<Option<Vec<u8>>>,
}

/// A round open awaiting its `data@2` corpus fetches (live mode).
struct PendingOpen {
    ro: RoundOpen,
    /// fetch op → (slot index, segment index).
    ops: BTreeMap<u64, (usize, usize)>,
    /// Batch slots in training order (step-major, then micro-window).
    slots: Vec<PendingBatch>,
}

/// A round record awaiting its committed-payload fetches (live mode).
struct PendingRecord {
    rr: daemon_vhc_sdk_consensus::messages::RoundRecord,
    entries: Vec<RecordEntry>,
    /// fetch op → the entry's peer (the payload lands under `(round, peer)`).
    ops: BTreeMap<u64, PeerId>,
}

/// The corpus geometry the live mode fetches against, derived from the verified manifest.
struct LiveCorpus {
    manifest: CorpusManifest,
    total_sequences: u64,
}

/// A periodic live checkpoint's export walk ([SF-6]). The master + ef families are already sealed,
/// so the checkpoint references them by fold (`master_fref`/`ef_fref` captured at the boundary —
/// ZERO extra bytes read); only the AdamW moments export off the device (async completions), and
/// are sealed into their own families at finish. The same async shape as the quiesce walk, without
/// ending the run.
struct CkptWalk {
    /// The ingested round this checkpoint's state folds (the restore watermark).
    round: u64,
    /// The already-sealed master + ef family references (captured at the boundary).
    master_fref: FamilyRef,
    ef_fref: FamilyRef,
    /// The exported AdamW moment tensors (all `m` then all `v`), per parameter, once landed.
    collected: Vec<Option<Vec<f32>>>,
    /// Outstanding moment export ops → collected index.
    ops: BTreeMap<u64, usize>,
}

/// The live-mode session state (module-driven data + wire announcements).
struct LiveState {
    cfg: LiveCfg,
    /// This peer id (for the per-slot publisher election).
    peer: PeerId,
    /// The run roster (for the per-slot publisher election).
    roster: Vec<PeerId>,
    corpus: Option<LiveCorpus>,
    manifest_op: Option<u64>,
    /// Whether a round has been observed (stops the periodic Join/Heartbeat re-announce).
    admitted: bool,
    pending_open: Option<PendingOpen>,
    pending_records: BTreeMap<u64, PendingRecord>,
    /// payload_put op → the commitment to publish once the store write is durable.
    pending_puts: BTreeMap<u64, Commitment>,
    /// The in-flight periodic checkpoint export walk (one at a time; a boundary that fires
    /// while one is in flight skips its cadence slot rather than queueing).
    pending_ckpt: Option<CkptWalk>,
    /// Outstanding checkpoint-document `payload_put` ops (their completions are drained; the
    /// pointer publication rides the host's put seam, not a wire message).
    ckpt_puts: std::collections::BTreeSet<u64>,
}

impl LiveState {
    fn new(cfg: LiveCfg, peer: PeerId, roster: Vec<PeerId>) -> Self {
        Self {
            cfg,
            peer,
            roster,
            corpus: None,
            manifest_op: None,
            admitted: false,
            pending_open: None,
            pending_records: BTreeMap::new(),
            pending_puts: BTreeMap::new(),
            pending_ckpt: None,
            ckpt_puts: std::collections::BTreeSet::new(),
        }
    }

    /// Start the periodic checkpoint export walk at an ingested-round boundary, when the cadence
    /// says so, no walk is already in flight, and this peer is the slot's designated publisher.
    /// The already-sealed master + ef families are captured by-reference here (zero extra reads);
    /// only the AdamW moments export.
    ///
    /// D-SF3 publication policy: the LOCAL cadence (`ckpt_every`) gates the checkpoint boundary
    /// (the folds already exist — a local checkpoint is bookkeeping); the REMOTE cadence
    /// (`remote_ckpt_every`) + a ONE-per-slot deterministic publisher election gate the upload, so
    /// a replicated group uploads once per slot, not once per peer. A slot whose publisher has died
    /// simply goes unpublished; the next slot's rotation covers it (the one-slot slack term).
    fn maybe_start_checkpoint(&mut self, core: &Rc<RefCell<Core>>, round: u64) {
        if self.cfg.ckpt_every == 0 || !round.is_multiple_of(self.cfg.ckpt_every) {
            return;
        }
        // Remote upload cadence + single-publisher gating. `remote_ckpt_every == 0` uploads at
        // every local boundary (the pre-cadence default). A non-boundary remote round is a
        // local-only checkpoint: the fold already exists, nothing to upload.
        let remote = if self.cfg.remote_ckpt_every == 0 {
            self.cfg.ckpt_every
        } else {
            self.cfg.remote_ckpt_every
        };
        if remote == 0 || !round.is_multiple_of(remote) {
            return;
        }
        let slot = round / remote;
        // The election seed is the corpus manifest hash — run-specific and identical across every
        // peer, so all peers derive the same publisher for the slot without exchanging a message.
        let seed = Seed(self.cfg.manifest.0);
        let publisher = daemon_vhc_sdk_consensus::assignment::elect_checkpoint_publisher_for_slot(
            &self.roster,
            &seed,
            slot,
        );
        if publisher != Some(self.peer) {
            return; // not this slot's publisher — skip (others upload; R identical uploads waste)
        }
        if self.pending_ckpt.is_some() {
            return;
        }
        let (tensors, master_fref, ef_fref) = {
            let c = core.borrow();
            let byte_len = family_byte_len(&numels_u64(&c.numels));
            (
                c.model.moment_tensors(),
                family_ref(c.master_fold, &c.master_chunks, byte_len, c.window_size),
                family_ref(c.ef_fold, &c.ef_chunks, byte_len, c.window_size),
            )
        };
        let n = tensors.len();
        let mut ops = BTreeMap::new();
        for (i, t) in tensors.into_iter().enumerate() {
            ops.insert(export_tensor(t), i);
        }
        self.pending_ckpt = Some(CkptWalk {
            round,
            master_fref,
            ef_fref,
            collected: vec![None; n],
            ops,
        });
    }

    /// Finish a completed checkpoint walk ([SF-6]): seal the exported AdamW moments into their own
    /// families (accumulating their chunk hashes), author the by-reference checkpoint DOCUMENT
    /// (master + ef + adamw_m + adamw_v by fold, plus the inline round watermark) via the shared
    /// codec, and put it on the payload plane. A live checkpoint moves ZERO family bytes locally —
    /// the families are already sealed; the referenced chunks are uploaded host-side on recognition
    /// of the document. Best-effort: a failed export abandons the walk (the next cadence retries).
    fn finish_checkpoint(&mut self, core: &Rc<RefCell<Core>>, walk: CkptWalk) {
        let mut moments = Vec::with_capacity(walk.collected.len());
        for c in walk.collected {
            match c {
                Some(v) => moments.push(v),
                None => return, // a failed moment export — abandon this cadence slot
            }
        }
        let n = moments.len() / 2;
        let (window_size, byte_len) = {
            let c = core.borrow();
            (c.window_size, family_byte_len(&numels_u64(&c.numels)))
        };
        let (m_fold, m_chunks) = seal_family(ADAMW_M_FAMILY, &moments[..n], window_size);
        let (v_fold, v_chunks) = seal_family(ADAMW_V_FAMILY, &moments[n..], window_size);
        let parts = vec![
            CkptPart::Family {
                name: MASTER_FAMILY,
                class: 0,
                fref: walk.master_fref,
            },
            CkptPart::Family {
                name: EF_FAMILY,
                class: 1,
                fref: walk.ef_fref,
            },
            CkptPart::Family {
                name: ADAMW_M_FAMILY,
                class: 1,
                fref: family_ref(m_fold, &m_chunks, byte_len, window_size),
            },
            CkptPart::Family {
                name: ADAMW_V_FAMILY,
                class: 1,
                fref: family_ref(v_fold, &v_chunks, byte_len, window_size),
            },
            CkptPart::Inline {
                name: "round",
                class: 1,
                bytes: walk.round.to_le_bytes().to_vec(),
            },
        ];
        let Some(manifest_bytes) = ckpt_manifest_bytes(&parts) else {
            return;
        };
        let Ok(bytes) = encode_checkpoint_doc(&manifest_bytes, &ckpt_sections(&parts)) else {
            return;
        };
        let buf = daemon_vhc_sdk::create_from(&bytes);
        let op = daemon_vhc_sdk::payload_put(buf);
        daemon_vhc_sdk::buffer_release(buf);
        self.ckpt_puts.insert(op);
    }

    /// Announce this peer to the coordinator: a Join plus the ready-Heartbeat that drives the
    /// warmup fast path. Re-published on a timer until the first round opens — control frames
    /// are fire-and-forget on the plane, and a frame published before the receivers ingested
    /// this session's certificate announcement is refused and lost (never replayed), so the
    /// module re-announces until it observes admission (a duplicate Join is a typed advisory
    /// reject coordinator-side, never an error).
    fn announce(&self) {
        publish_wire(&VhcMessage::Join(Join {
            run_id: self.cfg.run_label.clone(),
            iroh_id: IrohId([0u8; 32]),
            class: ThroughputClass::C1,
            capabilities: CapabilitySet::new(),
            envelope_hash: None,
        }));
        publish_wire(&VhcMessage::Heartbeat(Heartbeat {
            round: 0,
            ready: Some(true),
        }));
    }
}

/// Read one completion's `Ok(BufferHandle)` payload as raw bytes (`None` on a failed op).
fn completion_bytes(ev: &daemon_vhc_sdk::Event) -> Option<Vec<u8>> {
    let ciborium::value::Value::Array(result) = ev.items.get(2)? else {
        return None;
    };
    let ok = result
        .first()
        .and_then(|v| v.as_integer())
        .is_some_and(|n| i128::from(n) == 0);
    if !ok {
        return None;
    }
    let handle = result
        .get(1)
        .and_then(|v| v.as_integer())
        .map(|n| u64::try_from(i128::from(n)).unwrap_or(0))?;
    let bytes = daemon_vhc_sdk::read_buffer(handle);
    daemon_vhc_sdk::buffer_release(handle);
    Some(bytes)
}

/// The per-round export walk state: op → param index, collected values, the round in flight.
struct ExportState {
    round: u64,
    collected: Vec<Option<Vec<f32>>>,
    ops: BTreeMap<u64, usize>,
}

/// The quiesce snapshot's async collection ([SF-6]): only the AdamW moments export off the device
/// (op → moment index, all `m` then all `v`). The already-sealed master + ef families are captured
/// by-reference at drain start (`master_fref`/`ef_fref`); the moments are sealed into their own
/// families at finish and the four sections are declared by fold — the host reconstructs the
/// by-ref FamilyRefs from its state store, and the drain moves ZERO family bytes.
struct QuiesceWalk {
    collected: Vec<Option<Vec<f32>>>,
    ops: BTreeMap<u64, usize>,
    /// The already-sealed master + ef family references (captured at drain start).
    master_fref: FamilyRef,
    ef_fref: FamilyRef,
}

/// Decode one export completion's payload: `[status, handle]` → the tensor's f32 vec (None on a
/// nonzero status or an undecodable payload).
fn completion_tensor(ev: &daemon_vhc_sdk::Event) -> Option<Vec<f32>> {
    let ciborium::value::Value::Array(result) = ev.items.get(2)? else {
        return None;
    };
    let ok = result
        .first()
        .and_then(|v| v.as_integer())
        .is_some_and(|n| i128::from(n) == 0);
    if !ok {
        return None;
    }
    let handle = result
        .get(1)
        .and_then(|v| v.as_integer())
        .map(|n| u64::try_from(i128::from(n)).unwrap_or(0))?;
    let bytes = daemon_vhc_sdk::read_buffer(handle);
    daemon_vhc_sdk::buffer_release(handle);
    let data = daemon_vhc_sdk_compute::decode_tensor_data(&bytes);
    data.to_vec::<f32>().ok()
}

/// Plan one round's batch slots + `data@2` fetches from the assigned interval (live mode):
/// slots in training order (step-major, then micro-window), each slot's sequences coalesced
/// into per-shard contiguous byte segments.
fn plan_open_fetches(corpus: &LiveCorpus, ro: &RoundOpen, round_cfg: &RoundCfg) -> PendingOpen {
    let interval = interval_for(ro.batch, ro.seed, &round_cfg.roster, &round_cfg.peer);
    let steps = slice_interval(interval, round_cfg.steps_per_round, round_cfg.micro_batch);
    let mut slots = Vec::new();
    let mut ops = BTreeMap::new();
    for step in &steps {
        for mb in &step.micro {
            // Global sequence ids wrap modulo the corpus (the established window rule); the SDK
            // planner coalesces them into maximal per-shard contiguous ranges.
            let seqs: Vec<u64> = (mb.start..mb.end)
                .map(|s| s % corpus.total_sequences)
                .collect();
            let fetches = daemon_vhc_sdk::plan_window(&corpus.manifest, &seqs)
                .expect("the assigned window plans over the verified manifest");
            let slot_idx = slots.len();
            let mut segs = Vec::with_capacity(fetches.len());
            for (seg_idx, f) in fetches.iter().enumerate() {
                let op = daemon_vhc_sdk::data_fetch(&f.shard_hash, f.range_off, f.range_len);
                ops.insert(op, (slot_idx, seg_idx));
                segs.push(None);
            }
            slots.push(PendingBatch {
                sequences: u32::try_from(mb.end - mb.start).unwrap_or(0),
                segs,
            });
        }
    }
    PendingOpen {
        ro: ro.clone(),
        ops,
        slots,
    }
}

/// Decode a pending open's fetched segments into staged batches (training order), clamped into
/// the model's vocabulary (`token % vocab` — the deterministic tokenizer-to-model shim applied
/// identically by every peer).
fn stage_fetched_batches(
    core: &Rc<RefCell<Core>>,
    corpus: &LiveCorpus,
    open: &mut PendingOpen,
    vocab: u32,
) {
    let seq_len = corpus.manifest.seq_len;
    let width = corpus.manifest.token_width;
    let little = corpus.manifest.endianness == Endianness::Little;
    for slot in &mut open.slots {
        let mut raw = Vec::new();
        for seg in &mut slot.segs {
            raw.extend_from_slice(&seg.take().expect("all segments fetched"));
        }
        let tokens: Vec<u32> = match width {
            TokenWidth::U16 => raw
                .chunks_exact(2)
                .map(|c| {
                    let t = if little {
                        u16::from_le_bytes([c[0], c[1]])
                    } else {
                        u16::from_be_bytes([c[0], c[1]])
                    };
                    u32::from(t) % vocab
                })
                .collect(),
            TokenWidth::U32 => raw
                .chunks_exact(4)
                .map(|c| {
                    let t = if little {
                        u32::from_le_bytes([c[0], c[1], c[2], c[3]])
                    } else {
                        u32::from_be_bytes([c[0], c[1], c[2], c[3]])
                    };
                    t % vocab
                })
                .collect(),
        };
        core.borrow_mut()
            .batches
            .push_back((slot.sequences, seq_len, tokens));
    }
}

/// Voice the barrier outbounds: the tag-4 det digest (the journal-oracle voice) always; the
/// wire `Digest` control message additionally in live mode (the coordinator's liveness/desync
/// accounting input).
fn emit_round_outbounds(wire: bool, out: &[daemon_vhc_sdk_rounds::Outbound]) {
    for o in out {
        if let daemon_vhc_sdk_rounds::Outbound::RoundComplete { round, digest }
        | daemon_vhc_sdk_rounds::Outbound::CaughtUp { round, digest } = o
        {
            publish_tagged(4, *round, digest);
            if wire {
                publish_wire(&VhcMessage::Digest(Digest {
                    round: *round,
                    digest: StateDigest(*digest),
                }));
            }
        }
    }
}

/// Run the barrier on a fully-staged record and voice the outbounds.
fn dispatch_record(
    driver: &mut BarrierRound<C3Round>,
    payloads: &mut PayloadMap,
    wire: bool,
    pending: PendingRecord,
) {
    let out = driver.on_round_record(&pending.rr, pending.entries, payloads);
    emit_round_outbounds(wire, &out);
}

/// Open a harness-mode round, honoring the async fold serialization. If an ingest is mid-flight
/// (round r's sealed master is round r+1's round base), the round waits in `deferred_open` until
/// the fold seals. Otherwise the barrier runs `on_round_open` — which may itself kick off a
/// deferred CATCH-UP ingest (a straggled earlier round whose payload just became fetchable), in
/// which case training is deferred behind that fold too. Only a round that actually trained (a
/// `Commit` outbound) opens its θ export walk; a genuinely stalled round trains nothing.
fn try_open_round(
    core: &Rc<RefCell<Core>>,
    driver: &mut BarrierRound<C3Round>,
    payloads: &mut PayloadMap,
    numels: &[usize],
    ro: RoundOpen,
    export: &mut Option<ExportState>,
    deferred_open: &mut Option<RoundOpen>,
) {
    if driver.ingest_in_flight() {
        *deferred_open = Some(ro);
        return;
    }
    // A catch-up ingest kicked off inside `on_round_open` folds SILENTLY (its digest is implicit
    // in the folded state — the record handler is the only digest voice).
    core.borrow_mut().ingest_voices = false;
    let out = driver.on_round_open(&ro, payloads);
    // Record-triggered ingests (and finish_ingest resumes) voice again after this open.
    core.borrow_mut().ingest_voices = true;
    emit_round_outbounds(false, &out);
    if driver.ingest_in_flight() {
        // `on_round_open` kicked off a catch-up ingest; this round's training waits for its seal.
        *deferred_open = Some(ro);
    } else if out
        .iter()
        .any(|o| matches!(o, daemon_vhc_sdk_rounds::Outbound::Commit { .. }))
    {
        *export = Some(ExportState {
            round: ro.round,
            collected: vec![None; numels.len()],
            ops: BTreeMap::new(),
        });
        fence(ro.round + 1); // the round-final fence the walk waits for
    }
    // else: a genuinely stalled round (payload not yet fetchable) — no training, no export.
}

/// Per-parameter byte base offsets into the flat family image (prefix sums of `numel × 4`).
fn family_base_offsets(numels: &[usize]) -> Vec<u64> {
    let mut bases = Vec::with_capacity(numels.len());
    let mut acc = 0u64;
    for &n in numels {
        bases.push(acc);
        acc += (n as u64) * 4;
    }
    bases
}

/// Seal a resident family into a self-sealed fold ([SF-R1]) by streaming it through
/// `state_open`/`state_emit`/`state_seal` in per-parameter `window_size`-byte windows (the
/// fold-walk chunking, so the sealed fold IS the family fold the walk schedule assumes). Used to
/// give a resident init (seed-expanded or restored) a fetchable round-base fold.
fn seal_family(tag: &str, params: &[Vec<f32>], window_size: u64) -> ([u8; 32], Vec<Hash>) {
    let byte_len: u64 = params.iter().map(|p| (p.len() as u64) * 4).sum();
    let stream = daemon_vhc_sdk::state_open(tag, byte_len);
    let step = (window_size / 4).max(1) as usize; // elements per window
    let mut buf = Vec::new();
    // Accumulate each emitted chunk's blake3 in emit order — the fold-walk window IS the state
    // chunk, so this is exactly the family's ordered chunk-hash list ([SF-R1] self-sealed), what a
    // by-reference checkpoint section ([SF-6]) needs without re-reading the sealed family.
    let mut chunks = Vec::new();
    for p in params {
        let mut off = 0usize;
        while off < p.len() {
            let end = (off + step).min(p.len());
            f32s_to_le_bytes(&p[off..end], &mut buf);
            daemon_vhc_sdk::state_emit(stream, &buf);
            chunks.push(blake3_hash(&buf));
            off = end;
        }
    }
    (daemon_vhc_sdk::state_seal(stream), chunks)
}

/// One section the trainer's checkpoint document declares ([SF-6]): a by-reference already-sealed
/// family (zero bytes moved) or a small inline blob (the round watermark).
enum CkptPart {
    Family {
        name: &'static str,
        class: u64,
        fref: FamilyRef,
    },
    Inline {
        name: &'static str,
        class: u64,
        bytes: Vec<u8>,
    },
}

/// The parameter numels as `u64` (registration order) — the layout the family geometry helpers take.
fn numels_u64(numels: &[usize]) -> Vec<u64> {
    numels.iter().map(|&n| n as u64).collect()
}

/// Assemble a [`FamilyRef`] for a self-sealed family from its fold + accumulated chunk hashes.
fn family_ref(fold: [u8; 32], chunk_hashes: &[Hash], byte_len: u64, chunk_size: u64) -> FamilyRef {
    FamilyRef {
        fold: Hash(fold),
        byte_len,
        chunk_size,
        chunk_hashes: chunk_hashes.to_vec(),
    }
}

/// The §10.2 state-manifest bytes for a checkpoint document: a by-ref section declares its family
/// FOLD as `hash` and `byte_len` as `size` (the [SF-6] alternative); an inline section declares
/// the blake3 + length of its bytes. The host decodes this at the value level (dependency wall).
fn ckpt_manifest_bytes(parts: &[CkptPart]) -> Option<Vec<u8>> {
    let sections: Vec<SectionDecl> = parts
        .iter()
        .map(|p| match p {
            CkptPart::Family { name, class, fref } => SectionDecl {
                name: (*name).to_string(),
                schema: 1,
                hash: fref.fold,
                size: fref.byte_len,
                class: *class,
            },
            CkptPart::Inline { name, class, bytes } => SectionDecl {
                name: (*name).to_string(),
                schema: 1,
                hash: blake3_hash(bytes),
                size: bytes.len() as u64,
                class: *class,
            },
        })
        .collect();
    let manifest = StateManifest {
        schema: 1,
        module: Hash([0u8; 32]),
        sections,
    };
    to_canonical_vec(&manifest).ok()
}

/// The checkpoint-document section array in manifest order.
fn ckpt_sections(parts: &[CkptPart]) -> Vec<CkptDocSection> {
    parts
        .iter()
        .map(|p| match p {
            CkptPart::Family { name, fref, .. } => {
                CkptDocSection::ByRef((*name).to_string(), fref.clone())
            }
            CkptPart::Inline { name, bytes, .. } => {
                CkptDocSection::Inline((*name).to_string(), bytes.clone())
            }
        })
        .collect()
}

/// The async artifact-form init boot (§6.1b): fetch the pinned det-state manifest, register its
/// master family for length-aware ranged fetch ([SF-R2]), then stream the master windows back to
/// assemble the resident init and upload it to the device. The model boots from zeros; this
/// replaces those with the matched init before the first round trains.
struct BootState {
    /// The pinned init det-state manifest hash (a granted plain artifact).
    manifest: Hash,
    /// The outstanding manifest fetch, until it lands.
    manifest_op: Option<u64>,
    /// Outstanding master-window fetches: `data@2` op → window ordinal.
    window_ops: BTreeMap<u64, u64>,
    /// The master family's window schedule (set once the manifest lands).
    schedule: Vec<Window>,
    /// The assembling init (per parameter), filled as windows arrive.
    assembled: Vec<Vec<f32>>,
    /// Whether the init is fully assembled + uploaded.
    done: bool,
}

impl BootState {
    fn new(manifest: Hash, numels: &[usize]) -> Self {
        Self {
            manifest,
            manifest_op: None,
            window_ops: BTreeMap::new(),
            schedule: Vec::new(),
            assembled: numels.iter().map(|&n| vec![0.0f32; n]).collect(),
            done: false,
        }
    }
}

/// Drive one completion against the artifact-form init boot; returns `true` if it was consumed.
/// On the manifest completion: parse + register the master family ([SF-R2]) and issue every
/// master-window read. On a window completion: assemble it; when the family is complete, upload
/// the init to the device and mark the boot done.
fn drive_boot_completion(
    core: &Rc<RefCell<Core>>,
    boot: &mut BootState,
    ev: &daemon_vhc_sdk::Event,
    op: u64,
) -> bool {
    if boot.manifest_op == Some(op) {
        boot.manifest_op = None;
        let bytes = completion_bytes(ev).expect("the pinned init manifest fetches (fail loud)");
        let manifest =
            DetStateManifest::from_canonical_bytes(&bytes).expect("the init manifest parses");
        let (numels, window_size) = {
            let c = core.borrow();
            (c.numels.clone(), c.window_size)
        };
        let numels_u64: Vec<u64> = numels.iter().map(|&n| n as u64).collect();
        let master = manifest
            .families
            .get(MASTER_FAMILY)
            .expect("the init manifest carries a master family");
        let descriptor = DetStateChunkMap::derive(window_size, &numels_u64, &master.chunk_hashes)
            .expect("the fetched master geometry matches the layout")
            .to_canonical_bytes()
            .expect("descriptor cbor");
        let status = daemon_vhc_sdk::data_register_state_chunks(&descriptor);
        assert_eq!(
            status, 0,
            "the init master fold is a granted artifact (fail loud)"
        );
        {
            let mut c = core.borrow_mut();
            c.master_fold = master.fold.0;
            // The externally-registered master's chunk hashes (from the fetched manifest) — the
            // by-ref list for a checkpoint taken before the first ingest re-seals a native master.
            c.master_chunks = master.chunk_hashes.clone();
        }
        boot.schedule = daemon_vhc_sdk_consensus::fold_walk::windows(&numels_u64, window_size);
        let (base_fold, family_base) = {
            let c = core.borrow();
            (c.master_fold, c.family_base.clone())
        };
        for w in &boot.schedule {
            let off = family_base[w.param as usize] + w.param_off;
            let fop = daemon_vhc_sdk::data_fetch(&base_fold, off, w.len);
            boot.window_ops.insert(fop, w.ordinal);
        }
        return true;
    }
    let Some(ordinal) = boot.window_ops.remove(&op) else {
        return false;
    };
    let bytes = completion_bytes(ev).expect("an init master window fetches (fail loud)");
    let vals = le_bytes_to_f32s(&bytes).expect("init window is an f32-le image");
    let window = boot.schedule[usize::try_from(ordinal).expect("ordinal fits usize")];
    let off = (window.param_off / 4) as usize;
    boot.assembled[window.param as usize][off..off + vals.len()].copy_from_slice(&vals);
    if boot.window_ops.is_empty() {
        let mut c = core.borrow_mut();
        // Upload the assembled matched init to the device (transient — not retained resident), and
        // seal a zeroed ef family for round 0 (the update walk reads its ef windows from it).
        let flat = std::mem::take(&mut boot.assembled);
        c.model.set_params_from_flat(&flat);
        let window_size = c.window_size;
        let zeroed: Vec<Vec<f32>> = c.numels.iter().map(|&n| vec![0.0f32; n]).collect();
        let (ef_fold, ef_chunks) = seal_family(EF_FAMILY, &zeroed, window_size);
        c.ef_fold = ef_fold;
        c.ef_chunks = ef_chunks;
        boot.done = true;
    }
    true
}

/// The streaming checkpoint REHYDRATION walk ([SF-6], design §7.3): a restoring instance registers
/// the checkpoint's family folds ([SF-R2]) and fetches their windows on demand — master → device
/// weights, adamw_m/v → device moments — with no whole-state materialization beyond the transient
/// per-parameter upload buffers (freed once uploaded). The `ef` family is registered but not
/// streamed here: the `make_update` walk reads its windows directly from the adopted fold.
struct RestoreState {
    /// The shared per-parameter window schedule (same layout + window size for every family).
    schedule: Vec<Window>,
    /// The three streamed family folds, indexed by family code: 0 master, 1 adamw_m, 2 adamw_v.
    folds: [[u8; 32]; 3],
    /// Per-parameter byte base offsets (maps a window to an absolute fetch offset).
    family_base: Vec<u64>,
    /// Not-yet-issued `(family, window ordinal)` fetches, in order — drained with bounded refill
    /// so the walk never exceeds `max_outstanding_ops` (the families are large; issuing every
    /// window at once would breach the op grant).
    pending: VecDeque<(u8, u64)>,
    /// In-flight fetches: `data@2` op → `(family, window ordinal)`.
    inflight: BTreeMap<u64, (u8, u64)>,
    /// The assembling master / moments (per parameter), filled as windows arrive.
    master_assembled: Vec<Vec<f32>>,
    m_assembled: Vec<Vec<f32>>,
    v_assembled: Vec<Vec<f32>>,
    /// Windows not yet assembled across all three families.
    remaining: usize,
    /// Whether the rehydration is complete (weights + moments uploaded).
    done: bool,
}

impl RestoreState {
    /// Register the four checkpoint folds ([SF-R2]), adopt the master + ef folds as the round
    /// base, and kick off the bounded master/adamw_m/adamw_v window streaming. Called at `da_run`
    /// start (imports legal), never in `da_migrate`.
    fn begin(core: &Rc<RefCell<Core>>, refs: &RestoredRefs) -> Self {
        let (numels, window_size, family_base) = {
            let c = core.borrow();
            (c.numels.clone(), c.window_size, c.family_base.clone())
        };
        let numels_u64: Vec<u64> = numels.iter().map(|&n| n as u64).collect();
        // Register every externally-sourced fold so `data@2::fetch` resolves length-aware ([SF-R2]).
        for fref in [&refs.master, &refs.ef, &refs.adamw_m, &refs.adamw_v] {
            let desc = DetStateChunkMap::derive(fref.chunk_size, &numels_u64, &fref.chunk_hashes)
                .expect("restore family geometry matches the layout")
                .to_canonical_bytes()
                .expect("descriptor cbor");
            assert_eq!(
                daemon_vhc_sdk::data_register_state_chunks(&desc),
                0,
                "the restore family fold is a granted artifact (fail loud)"
            );
        }
        // Adopt the master + ef folds as the round base — the folds already exist in the plane, so
        // no re-seal: the ingest/update walks read their windows straight from these folds.
        {
            let mut c = core.borrow_mut();
            c.master_fold = refs.master.fold.0;
            c.master_chunks = refs.master.chunk_hashes.clone();
            c.ef_fold = refs.ef.fold.0;
            c.ef_chunks = refs.ef.chunk_hashes.clone();
        }
        let schedule = daemon_vhc_sdk_consensus::fold_walk::windows(&numels_u64, window_size);
        let mut pending: VecDeque<(u8, u64)> = VecDeque::new();
        for family in 0u8..3 {
            for w in &schedule {
                pending.push_back((family, w.ordinal));
            }
        }
        let assembled = || numels.iter().map(|&n| vec![0.0f32; n]).collect::<Vec<_>>();
        let mut st = Self {
            schedule,
            folds: [refs.master.fold.0, refs.adamw_m.fold.0, refs.adamw_v.fold.0],
            family_base,
            remaining: pending.len(),
            pending,
            inflight: BTreeMap::new(),
            master_assembled: assembled(),
            m_assembled: assembled(),
            v_assembled: assembled(),
            done: false,
        };
        st.issue_more();
        st
    }

    /// Issue pending window fetches up to the in-flight bound (bounded read-ahead).
    fn issue_more(&mut self) {
        while self.inflight.len() < RESTORE_IN_FLIGHT {
            let Some((family, ordinal)) = self.pending.pop_front() else {
                break;
            };
            let w = self.schedule[usize::try_from(ordinal).expect("ordinal fits usize")];
            let off = self.family_base[w.param as usize] + w.param_off;
            let op = daemon_vhc_sdk::data_fetch(&self.folds[family as usize], off, w.len);
            self.inflight.insert(op, (family, ordinal));
        }
    }
}

/// Drive one completion against the in-flight restore walk; returns `true` if it was consumed.
/// Assembles the arrived window into its family buffer, refills the in-flight window, and when
/// master + both moments are fully streamed uploads the weights (`set_params_from_flat`) +
/// moments (`set_moments_from_flat`) to the device and marks the rehydration done.
fn drive_restore_completion(
    core: &Rc<RefCell<Core>>,
    restore: &mut RestoreState,
    ev: &daemon_vhc_sdk::Event,
    op: u64,
) -> bool {
    let Some((family, ordinal)) = restore.inflight.remove(&op) else {
        return false;
    };
    let bytes = completion_bytes(ev).expect("a restore window fetches (fail loud)");
    let vals = le_bytes_to_f32s(&bytes).expect("restore window is an f32-le image");
    let window = restore.schedule[usize::try_from(ordinal).expect("ordinal fits usize")];
    let off = (window.param_off / 4) as usize;
    let target = match family {
        0 => &mut restore.master_assembled,
        1 => &mut restore.m_assembled,
        _ => &mut restore.v_assembled,
    };
    target[window.param as usize][off..off + vals.len()].copy_from_slice(&vals);
    restore.remaining -= 1;
    restore.issue_more();
    if restore.remaining == 0 {
        let mut c = core.borrow_mut();
        let master = std::mem::take(&mut restore.master_assembled);
        c.model.set_params_from_flat(&master);
        let m = std::mem::take(&mut restore.m_assembled);
        let v = std::mem::take(&mut restore.v_assembled);
        c.model.set_moments_from_flat(&m, &v);
        restore.done = true;
    }
    true
}

/// The outcome of routing a completion to the in-flight ingest walk.
enum IngestStep {
    /// The op is not an ingest-walk window read (route it elsewhere).
    NotMine,
    /// A window folded (or a read was issued); the walk continues.
    Progressed,
    /// The final slice sealed the fold: the new master fold is durable, the master is uploaded,
    /// and the round's digest is ready — the caller voices it via `finish_ingest`.
    Sealed {
        /// The ingested round.
        round: u64,
        /// The post-ingest det digest (the carry finalize).
        digest: [u8; 16],
        /// Whether this round's digest is voiced (record-triggered) or folds silently (catch-up
        /// inside `on_round_open`).
        voice: bool,
    },
}

/// Drive one completion against the in-flight streamed ingest walk. Folds the maximal contiguous
/// run now available (each fold `state_emit`s its master window and advances the carry, and
/// assembles the value into a TRANSIENT per-parameter buffer for the device upload — no resident
/// master), refills the read window, and at seal closes the fold, uploads the new master, and
/// advances the round-base fold. The caller finishes the barrier + digest voice.
fn drive_ingest_completion(
    core: &Rc<RefCell<Core>>,
    ev: &daemon_vhc_sdk::Event,
    op: u64,
) -> IngestStep {
    match core.borrow().ingest_walk.as_ref() {
        Some(w) if w.ops.contains_key(&op) => {}
        _ => return IngestStep::NotMine,
    }
    let bytes = completion_bytes(ev).expect("round-base window fetch completes (fail loud)");
    let vals = le_bytes_to_f32s(&bytes).expect("round-base window is an f32-le image");

    let sealed;
    {
        let mut core_mut = core.borrow_mut();
        let Core {
            ingest_walk,
            family_base,
            ..
        } = &mut *core_mut;
        let w = ingest_walk.as_mut().expect("ingest walk present");
        let ordinal = w.ops.remove(&op).expect("op is an outstanding window read");
        let slice = w
            .walk
            .on_window_ready(ordinal, &vals)
            .expect("the walk accepts the completed window");
        for (window, master_vals) in &slice.emitted {
            f32s_to_le_bytes(master_vals, &mut w.byte_buf);
            daemon_vhc_sdk::state_emit(w.stream, &w.byte_buf);
            w.chunk_hashes.push(blake3_hash(&w.byte_buf));
            let off = (window.param_off / 4) as usize;
            w.master_assembly[window.param as usize][off..off + master_vals.len()]
                .copy_from_slice(master_vals);
        }
        for window in &slice.issue {
            let off = family_base[window.param as usize] + window.param_off;
            let fop = daemon_vhc_sdk::data_fetch(&w.base_fold, off, window.len);
            w.ops.insert(fop, window.ordinal);
        }
        sealed = slice.sealed;
    }
    if !sealed {
        return IngestStep::Progressed;
    }
    // The sealing slice: close the fold (master r becomes the round base of r+1), finalize the
    // carry, and upload the freshly-assembled master to the device from the transient buffer
    // (freed with the walk — no resident master survives).
    let mut core_mut = core.borrow_mut();
    let mut state = core_mut.ingest_walk.take().expect("the sealed walk");
    let round = state.round;
    let voice = state.voice;
    let fold = daemon_vhc_sdk::state_seal(state.stream);
    let digest = *state
        .walk
        .seal()
        .expect("the walk sealed after every window folded")
        .finalize()
        .as_bytes();
    core_mut.master_fold = fold;
    core_mut.master_chunks = std::mem::take(&mut state.chunk_hashes);
    let flat = std::mem::take(&mut state.master_assembly);
    core_mut.model.set_params_from_flat(&flat);
    IngestStep::Sealed {
        round,
        digest,
        voice,
    }
}

/// Kick off the streamed `make_update` walk for `round`: `theta` is the round's device-exported
/// parameters (resident, decoded), the round-base (master r−1) + prior ef windows fetch from the
/// sealed folds, and the NEW ef family emits into a fresh `state_open` stream. Stashed in `Core`,
/// driven by [`drive_update_completion`].
fn start_update_walk(core: &Rc<RefCell<Core>>, round: u64, theta: Vec<Vec<f32>>) {
    let mut c = core.borrow_mut();
    let numels = c.numels.clone();
    let numels_u64: Vec<u64> = numels.iter().map(|&n| n as u64).collect();
    let byte_len = family_byte_len(&numels_u64);
    let (window_size, base_fold, ef_fold) = (c.window_size, c.master_fold, c.ef_fold);
    let mut walk = SparseLocoUpdateWalk::new(&c.profile_cfg, &numels, window_size, WALK_IN_FLIGHT)
        .expect("update walk geometry (profile chunk divides every numel; window aligned)");
    let opening = walk.start().expect("update walk start");
    let ef_stream = daemon_vhc_sdk::state_open(EF_FAMILY, byte_len);
    let mut st = UpdateWalkState {
        round,
        walk,
        theta,
        ef_stream,
        base_fold,
        ef_fold,
        rb_ops: BTreeMap::new(),
        ef_ops: BTreeMap::new(),
        rb: BTreeMap::new(),
        ef: BTreeMap::new(),
        byte_buf: Vec::new(),
        ef_chunks: Vec::new(),
    };
    for w in &opening.issue {
        let off = c.family_base[w.param as usize] + w.param_off;
        st.rb_ops.insert(
            daemon_vhc_sdk::data_fetch(&st.base_fold, off, w.len),
            w.ordinal,
        );
        st.ef_ops.insert(
            daemon_vhc_sdk::data_fetch(&st.ef_fold, off, w.len),
            w.ordinal,
        );
    }
    c.update_walk = Some(st);
}

/// The outcome of routing a completion to the in-flight `make_update` walk.
enum UpdateStep {
    /// Not an update-walk window read.
    NotMine,
    /// A window arrived / folded; the walk continues.
    Progressed,
    /// The final slice sealed the ef family + assembled the payload sections — the caller puts the
    /// committed container and voices the tag-3 commitment.
    Sealed {
        /// The committed round.
        round: u64,
        /// The sealed committed-container bytes.
        payload: Vec<u8>,
        /// Its blake3 (the commitment hash).
        hash: Hash,
    },
}

/// Drive one completion against the in-flight `make_update` walk. Each window needs its round-base
/// AND ef pair; when both arrive the fold runs (Δ → ef accumulate → top-k → pack), emits the new
/// ef window, and appends the payload section fragment; at seal it closes the ef fold and returns
/// the assembled payload for the tag-3 voice.
fn drive_update_completion(
    core: &Rc<RefCell<Core>>,
    ev: &daemon_vhc_sdk::Event,
    op: u64,
) -> UpdateStep {
    let is_rb = {
        let c = core.borrow();
        match c.update_walk.as_ref() {
            Some(st) if st.rb_ops.contains_key(&op) => true,
            Some(st) if st.ef_ops.contains_key(&op) => false,
            _ => return UpdateStep::NotMine,
        }
    };
    let bytes = completion_bytes(ev).expect("update window fetch completes (fail loud)");
    let vals = le_bytes_to_f32s(&bytes).expect("update window is an f32-le image");
    let mut c = core.borrow_mut();
    let sealed = {
        let Core {
            update_walk,
            family_base,
            ..
        } = &mut *c;
        let st = update_walk.as_mut().expect("update walk present");
        let ordinal = if is_rb {
            st.rb_ops.remove(&op)
        } else {
            st.ef_ops.remove(&op)
        }
        .expect("op is an outstanding update window read");
        if is_rb {
            st.rb.insert(ordinal, vals);
        } else {
            st.ef.insert(ordinal, vals);
        }
        if st.rb.contains_key(&ordinal) && st.ef.contains_key(&ordinal) {
            let round_base = st.rb.remove(&ordinal).expect("round-base present");
            let ef = st.ef.remove(&ordinal).expect("ef present");
            let window = st.walk.schedule()[usize::try_from(ordinal).expect("ordinal fits")];
            let (poff, elems) = (
                usize::try_from(window.param_off / 4).expect("offset fits"),
                usize::try_from(window.len / 4).expect("window fits"),
            );
            let theta = st.theta[window.param as usize][poff..poff + elems].to_vec();
            let step = st
                .walk
                .on_window_ready(
                    ordinal,
                    UpdateWindowInputs {
                        theta,
                        round_base,
                        ef,
                    },
                )
                .expect("the update walk accepts the completed window");
            for (_w, ef_new) in &step.emitted {
                f32s_to_le_bytes(ef_new, &mut st.byte_buf);
                daemon_vhc_sdk::state_emit(st.ef_stream, &st.byte_buf);
                st.ef_chunks.push(blake3_hash(&st.byte_buf));
            }
            for w in &step.issue {
                let off = family_base[w.param as usize] + w.param_off;
                st.rb_ops.insert(
                    daemon_vhc_sdk::data_fetch(&st.base_fold, off, w.len),
                    w.ordinal,
                );
                st.ef_ops.insert(
                    daemon_vhc_sdk::data_fetch(&st.ef_fold, off, w.len),
                    w.ordinal,
                );
            }
            step.sealed
        } else {
            false
        }
    };
    if !sealed {
        return UpdateStep::Progressed;
    }
    let mut st = c.update_walk.take().expect("the sealed update walk");
    c.ef_fold = daemon_vhc_sdk::state_seal(st.ef_stream);
    c.ef_chunks = std::mem::take(&mut st.ef_chunks);
    let sections = st
        .walk
        .seal()
        .expect("the update walk sealed after every window folded");
    let payload = encode_payload(&sections);
    let hash = blake3_hash(&payload);
    UpdateStep::Sealed {
        round: st.round,
        payload,
        hash,
    }
}

#[allow(clippy::too_many_lines)]
fn run_module(mut cfg: GuestCfg, restored: Option<RestoredRefs>) -> u32 {
    let numels = cfg.model.param_numels();
    let vocab = cfg.model.vocab;
    let mut live = cfg
        .live
        .take()
        .map(|c| LiveState::new(c, cfg.peer, cfg.roster.clone()));
    let restored_round = restored.as_ref().and_then(|r| r.round);
    let window_size = cfg.state.chunk_size;
    let family_base = family_base_offsets(&numels);
    // The init source (§6): a restore rebuilds from the snapshot; a fresh join expands the seed
    // (synchronous, sealed + `expected_root`-checked here) or fetches the content-addressed
    // artifact (async — the model boots from zeros and the fetched init uploads when the boot
    // walk completes, [`BootState`]). `booted == false` defers the first round until the init is
    // resident.
    let mut boot: Option<BootState> = None;
    // The init source (§6) / restore rebuild:
    //   - Restore ([SF-6]): the model boots from zeros; a [`RestoreState`] walk registers the
    //     checkpoint's family folds ([SF-R2]) and STREAMS master → device weights and adamw_m/v →
    //     device moments, adopting the master/ef folds as the round base. `booted` stays false
    //     until the walk lands (the first round defers behind it, like the artifact boot).
    //   - Seed: deterministic expansion, sealed + `expected_root`-cross-checked synchronously.
    //   - Artifact: fetched asynchronously ([`BootState`]); boots from zeros until it lands.
    let init: Vec<Vec<f32>> = match &restored {
        Some(_) => numels.iter().map(|&n| vec![0.0f32; n]).collect(),
        None => match cfg.state.init {
            StateInit::Seed { seed, dist, .. } => numels
                .iter()
                .enumerate()
                .map(|(i, &n)| {
                    daemon_vhc_det::seed_init_param(&seed.0, dist, i as u64, n)
                        .expect("the genesis seed-init distribution id is implemented")
                })
                .collect(),
            StateInit::Manifest { manifest } => {
                boot = Some(BootState::new(manifest, &numels));
                numels.iter().map(|&n| vec![0.0f32; n]).collect()
            }
        },
    };
    // Seed init is resident synchronously; a restore or an artifact fetch defers the first round.
    let mut booted = restored.is_none() && boot.is_none();
    let device = daemon_vhc_sdk_compute::device();
    let model = TinyLlamaModel::<AutodiffHostBackend>::from_flat(cfg.model.clone(), device, &init);
    // The canonical master + replica-local ef live host-side as sealed folds (no resident copy).
    // Seed-init seals both synchronously now — master from the expansion (cross-checked against
    // the pin) and a zeroed ef; restore adopts the checkpoint's folds; the artifact form
    // registers/seals at boot.
    let mut master_fold = [0u8; 32];
    let mut master_chunks: Vec<Hash> = Vec::new();
    let mut ef_fold = [0u8; 32];
    let mut ef_chunks: Vec<Hash> = Vec::new();
    if booted {
        let (mf, mc) = seal_family(MASTER_FAMILY, &init, window_size);
        master_fold = mf;
        master_chunks = mc;
        let zeroed: Vec<Vec<f32>> = numels.iter().map(|&n| vec![0.0f32; n]).collect();
        let (ff, fc) = seal_family(EF_FAMILY, &zeroed, window_size);
        ef_fold = ff;
        ef_chunks = fc;
        // Seed-init admission cross-check (§6.1a): the sealed expansion MUST reproduce the pin.
        if let StateInit::Seed { expected_root, .. } = cfg.state.init {
            assert_eq!(
                master_fold, expected_root.0,
                "seed-init sealed fold does not match the pinned expected_root (typed init failure)"
            );
        }
    }
    let core = Rc::new(RefCell::new(Core {
        model,
        profile_cfg: cfg.profile.clone(),
        numels: numels.clone(),
        family_base,
        window_size,
        master_fold,
        master_chunks,
        ef_fold,
        ef_chunks,
        ingest_walk: None,
        update_walk: None,
        ingest_voices: true,
        batches: VecDeque::new(),
        pending_grads: None,
        next_step_fence: 0,
    }));
    // Restore ([SF-6]): register the checkpoint's family folds ([SF-R2]), adopt the master/ef
    // folds as the round base, and kick off the streamed rehydration (master → weights,
    // adamw_m/v → moments; ef needs no materialization — the update walk reads its windows).
    let mut restore: Option<RestoreState> = restored
        .as_ref()
        .map(|refs| RestoreState::begin(&core, refs));
    // Fresh artifact-form join: kick off the init boot (fetch the pinned manifest).
    if let Some(b) = boot.as_mut() {
        b.manifest_op = Some(daemon_vhc_sdk::data_fetch(&b.manifest.0, 0, 0));
    }
    let round_cfg = RoundCfg {
        peer: cfg.peer,
        roster: cfg.roster.clone(),
        steps_per_round: cfg.steps_per_round,
        micro_batch: cfg.micro_batch,
        stall_rounds_max: cfg.stall_rounds_max,
    };
    let mut driver = BarrierRound::new(C3Round { core: core.clone() }, round_cfg.clone());
    // A restored instance never re-ingests at or below the snapshot's watermark (§9 restore).
    if restored_round.is_some() {
        driver.resume_from(restored_round);
    }
    let mut payloads = PayloadMap {
        map: BTreeMap::new(),
    };
    let mut export: Option<ExportState> = None;
    let mut quiesce: Option<QuiesceWalk> = None;
    // A harness RoundOpen that arrives before the async init boot completes waits here.
    let mut deferred_open: Option<RoundOpen> = None;

    // Live mode boots by fetching the genesis-pinned corpus manifest (everything else waits on
    // it: chunk registration precedes any shard range fetch, and the Join announcement waits so
    // the first opened round finds this peer trainable).
    if let Some(l) = live.as_mut() {
        l.manifest_op = Some(daemon_vhc_sdk::data_fetch(&l.cfg.manifest.0, 0, 0));
    }

    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    loop {
        let ev = daemon_vhc_sdk::next_event(&mut buf);
        match ev.tag {
            EV_STOP => {
                // Deliberate leak: any device tensor still held here would enqueue its
                // `OperationIr::Drop` through `submit_op` AFTER Stop — a `PhaseViolation`
                // (ABI §4.4). Instance teardown force-reclaims every handle (§7.3), so
                // forgetting the model (both `Rc` holders) is the conforming shutdown.
                std::mem::forget(driver);
                std::mem::forget(core);
                return 0;
            }
            EV_QUIESCE => {
                // The §10.2 producing protocol, phase 1: the AdamW moments live device-side, so
                // the snapshot is an async export walk (same shape as the round's θ export —
                // Fence/Completion events keep delivering during a drain, §4.4). The manifest is
                // authored and submitted when the last completion lands (EV_COMPLETION below).
                let (tensors, master_fref, ef_fref) = {
                    let c = core.borrow();
                    let byte_len = family_byte_len(&numels_u64(&c.numels));
                    (
                        c.model.moment_tensors(),
                        family_ref(c.master_fold, &c.master_chunks, byte_len, c.window_size),
                        family_ref(c.ef_fold, &c.ef_chunks, byte_len, c.window_size),
                    )
                };
                let mut ops = BTreeMap::new();
                let n = tensors.len();
                for (i, t) in tensors.into_iter().enumerate() {
                    ops.insert(export_tensor(t), i);
                }
                // The master + ef families are already sealed — captured by-reference (zero extra
                // reads); only the AdamW moments export here ([SF-6]).
                quiesce = Some(QuiesceWalk {
                    collected: vec![None; n],
                    ops,
                    master_fref,
                    ef_fref,
                });
            }
            EV_PAYLOAD_READY => {
                // All harness staging is kind-0 bytes; the wrapper tag routes it.
                let staging_id = ev.uint(1);
                let bytes = daemon_vhc_sdk::read_back_bytes(staging_id, 0);
                let Ok(ciborium::value::Value::Array(items)) =
                    ciborium::from_reader::<ciborium::value::Value, _>(bytes.as_slice())
                else {
                    continue; // not a recognized staged wrapper — module policy is to ignore
                };
                let uint = |i: usize| -> u64 {
                    items
                        .get(i)
                        .and_then(ciborium::value::Value::as_integer)
                        .map(|n| u64::try_from(i128::from(n)).unwrap_or(0))
                        .unwrap_or(0)
                };
                match uint(0) {
                    0 => {
                        // A batch: [0, round, step, sequences, seq_len, tokens_le].
                        let (sequences, seq_len) = (uint(3) as u32, uint(4) as u32);
                        let tokens: Vec<u32> = match items.get(5) {
                            Some(ciborium::value::Value::Bytes(b)) => b
                                .chunks_exact(4)
                                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect(),
                            _ => continue,
                        };
                        core.borrow_mut()
                            .batches
                            .push_back((sequences, seq_len, tokens));
                    }
                    1 => {
                        // A committed payload: [1, round, peer32, payload].
                        let round = uint(1);
                        let peer: [u8; 32] = match items.get(2) {
                            Some(ciborium::value::Value::Bytes(b)) => {
                                match b.as_slice().try_into() {
                                    Ok(p) => p,
                                    Err(_) => continue,
                                }
                            }
                            _ => continue,
                        };
                        let payload = match items.get(3) {
                            Some(ciborium::value::Value::Bytes(b)) => b.clone(),
                            _ => continue,
                        };
                        payloads.map.insert((round, PeerId(peer)), payload);
                    }
                    _ => {}
                }
            }
            EV_TIMER => {
                // Live mode: re-announce Join + ready-Heartbeat until the first round opens
                // (fire-and-forget frames published before the peers ingested this session's
                // certificate are refused and lost — the module re-announces, module policy).
                if let Some(l) = live.as_ref() {
                    if !l.admitted && l.corpus.is_some() {
                        l.announce();
                        daemon_vhc_sdk::set_timer(500);
                    }
                }
            }
            EV_FRAME => {
                let payload = ev.bytes(4);
                let Ok(msg) = from_canonical_slice::<VhcMessage>(&payload) else {
                    continue;
                };
                match msg {
                    VhcMessage::RoundOpen(ro) => {
                        let round = ro.round;
                        if let Some(l) = live.as_mut() {
                            // Module-driven data: plan + fetch this round's assigned corpus
                            // ranges; the driver's open runs when the last segment lands
                            // (EV_COMPLETION below). One open in flight at a time — the
                            // coordinator opens rounds strictly after records, so a second
                            // in-flight open means this peer is already stalled and sits the
                            // round out (the ladder catches it up at the next record).
                            l.admitted = true;
                            let Some(corpus) = l.corpus.as_ref() else {
                                continue; // manifest not ready — sit the round out
                            };
                            if l.pending_open.is_some() {
                                continue;
                            }
                            l.pending_open = Some(plan_open_fetches(corpus, &ro, &round_cfg));
                            continue;
                        }
                        // Harness mode: batches were host-staged; train + (dropped) commit,
                        // then the export walk for this round's θ. If the async init boot has not
                        // finished, this round waits until the matched init is resident.
                        let _ = round;
                        if !booted {
                            deferred_open = Some(ro);
                            continue;
                        }
                        try_open_round(
                            &core,
                            &mut driver,
                            &mut payloads,
                            &numels,
                            ro,
                            &mut export,
                            &mut deferred_open,
                        );
                    }
                    VhcMessage::RoundRecord(rr) => {
                        let entries: Vec<RecordEntry> = rr.inline.clone().unwrap_or_default();
                        if live.is_some() {
                            // Module-driven payload staging: fetch every record-listed
                            // committed payload from the content-addressed plane (self
                            // included — idempotent and uniform), then run the barrier.
                            if let Some(last) = driver.last_ingested() {
                                if rr.round <= last {
                                    continue; // at/below the watermark — never re-fetched
                                }
                            }
                            let mut ops = BTreeMap::new();
                            for e in &entries {
                                if !payloads.map.contains_key(&(rr.round, e.peer)) {
                                    let op = daemon_vhc_sdk::payload_get(&e.hash.0);
                                    ops.insert(op, e.peer);
                                }
                            }
                            let pending = PendingRecord { rr, entries, ops };
                            let pending_round = pending.rr.round;
                            if pending.ops.is_empty() {
                                dispatch_record(&mut driver, &mut payloads, true, pending);
                                // The ingested-round boundary: the periodic live checkpoint
                                // cadence fires here (post-ingest state, spec §9).
                                if let Some(l) = live.as_mut() {
                                    l.maybe_start_checkpoint(&core, pending_round);
                                }
                            } else if let Some(l) = live.as_mut() {
                                l.pending_records.insert(pending_round, pending);
                            }
                            continue;
                        }
                        let out = driver.on_round_record(&rr, entries, &mut payloads);
                        emit_round_outbounds(false, &out);
                    }
                    _ => {}
                }
            }
            EV_FENCE => {
                let id = ev.uint(1);
                let Some(st) = export.as_mut() else { continue };
                if id != st.round + 1 {
                    continue; // a per-step depth-reset fence — not awaited
                }
                // The device passed the round's training: export every param (device → sealed
                // buffer; the completions carry CBOR(TensorData)).
                let tensors = core.borrow().model.export_tensors();
                for (i, t) in tensors.into_iter().enumerate() {
                    let op = export_tensor(t);
                    st.ops.insert(op, i);
                }
            }
            EV_COMPLETION => {
                let op = ev.uint(1);
                // The async init boot ([SF-R2] artifact form): manifest fetch → register → master
                // window reads → upload. When it completes, a round that arrived early runs now.
                if let Some(b) = boot.as_mut() {
                    if drive_boot_completion(&core, b, &ev, op) {
                        if b.done {
                            booted = true;
                            boot = None;
                            if let Some(ro) = deferred_open.take() {
                                try_open_round(
                                    &core,
                                    &mut driver,
                                    &mut payloads,
                                    &numels,
                                    ro,
                                    &mut export,
                                    &mut deferred_open,
                                );
                            }
                        }
                        continue;
                    }
                }
                // The streamed checkpoint rehydration ([SF-6]): master/adamw_m/adamw_v window reads.
                // When the last lands (weights + moments uploaded), a round that arrived early runs.
                if let Some(rs) = restore.as_mut() {
                    if drive_restore_completion(&core, rs, &ev, op) {
                        if rs.done {
                            booted = true;
                            restore = None;
                            if let Some(ro) = deferred_open.take() {
                                try_open_round(
                                    &core,
                                    &mut driver,
                                    &mut payloads,
                                    &numels,
                                    ro,
                                    &mut export,
                                    &mut deferred_open,
                                );
                            }
                        }
                        continue;
                    }
                }
                // The in-flight streamed ingest walk's round-base window reads. At seal, voice the
                // round digest (finish_ingest) and — now that the fold is durable and the master
                // uploaded — retry any round whose training was waiting behind it (catch-up).
                match drive_ingest_completion(&core, &ev, op) {
                    IngestStep::NotMine => {}
                    IngestStep::Progressed => continue,
                    IngestStep::Sealed {
                        round,
                        digest,
                        voice,
                    } => {
                        let out = driver.finish_ingest(round, digest, &mut payloads);
                        // A catch-up ingest kicked off inside `on_round_open` folds silently; only
                        // a record-triggered ingest voices its digest (matches the resident guest,
                        // which dropped `on_round_open`'s outbounds).
                        if voice {
                            emit_round_outbounds(live.is_some(), &out);
                        }
                        if !driver.ingest_in_flight() {
                            if let Some(ro) = deferred_open.take() {
                                try_open_round(
                                    &core,
                                    &mut driver,
                                    &mut payloads,
                                    &numels,
                                    ro,
                                    &mut export,
                                    &mut deferred_open,
                                );
                            }
                        }
                        continue;
                    }
                }
                // The in-flight streamed make_update walk's (round-base, ef) window reads. At seal
                // the new ef family is durable; externalize the committed container and voice the
                // tag-3 commitment (the wire Commitment rides the put completion in live mode).
                match drive_update_completion(&core, &ev, op) {
                    UpdateStep::NotMine => {}
                    UpdateStep::Progressed => continue,
                    UpdateStep::Sealed {
                        round,
                        payload,
                        hash,
                    } => {
                        let buf = daemon_vhc_sdk::create_from(&payload);
                        let put_op = daemon_vhc_sdk::payload_put(buf);
                        daemon_vhc_sdk::buffer_release(buf);
                        publish_tagged(3, round, &hash.0);
                        if let Some(l) = live.as_mut() {
                            l.pending_puts.insert(
                                put_op,
                                Commitment {
                                    round,
                                    payload: hash,
                                    size: payload.len() as u64,
                                    locators: Vec::new(),
                                },
                            );
                        }
                        continue;
                    }
                }
                // Live-mode completion routing first: the manifest fetch, a pending open's
                // corpus segments, a pending record's committed payloads, a durable payload put.
                if let Some(l) = live.as_mut() {
                    if l.manifest_op == Some(op) {
                        l.manifest_op = None;
                        let bytes = completion_bytes(&ev)
                            .expect("the genesis-pinned corpus manifest fetches (fail loud)");
                        let manifest = CorpusManifest::from_canonical_bytes(&bytes)
                            .expect("the fetched corpus manifest parses");
                        // Register every shard's chunk map: after registration a shard's fold
                        // identity is range-fetchable with covering-chunk verification.
                        for i in 0..manifest.shards.len() {
                            let desc = daemon_vhc_sdk::chunk_descriptor(&manifest, i)
                                .expect("manifest shard yields a chunk descriptor");
                            let status = daemon_vhc_sdk::data_register_chunks(&desc);
                            assert_eq!(status, 0, "chunk registration is granted (fail loud)");
                        }
                        let total_sequences = manifest.total_sequences();
                        assert!(total_sequences > 0, "the pinned corpus is non-empty");
                        l.corpus = Some(LiveCorpus {
                            manifest,
                            total_sequences,
                        });
                        // Announce (and keep re-announcing until the first round opens).
                        l.announce();
                        daemon_vhc_sdk::set_timer(500);
                        continue;
                    }
                    if let Some(commitment) = l.pending_puts.remove(&op) {
                        // The sealed container is durable on the plane: NOW the wire commitment
                        // is publishable (the coordinator's availability check will find it).
                        publish_wire(&VhcMessage::Commitment(commitment));
                        continue;
                    }
                    if l.ckpt_puts.remove(&op) {
                        // The checkpoint document is durable; the pointer publication rides the
                        // host's put seam (spec §9), never a wire message.
                        continue;
                    }
                    // The periodic checkpoint walk's moment exports (spec §9): collect; a failed
                    // export abandons the cadence slot (training continues; the next fires).
                    let ckpt_hit = l.pending_ckpt.as_mut().and_then(|walk| {
                        walk.ops
                            .remove(&op)
                            .map(|idx| match completion_tensor(&ev) {
                                Some(t) => {
                                    walk.collected[idx] = Some(t);
                                    true
                                }
                                None => false,
                            })
                    });
                    match ckpt_hit {
                        Some(true) => {
                            let done = l.pending_ckpt.as_ref().is_some_and(|w| {
                                w.ops.is_empty() && w.collected.iter().all(Option::is_some)
                            });
                            if done {
                                let walk = l.pending_ckpt.take().expect("the in-flight checkpoint");
                                l.finish_checkpoint(&core, walk);
                            }
                            continue;
                        }
                        Some(false) => {
                            l.pending_ckpt = None;
                            continue;
                        }
                        None => {}
                    }
                    let open_hit = l.pending_open.as_mut().and_then(|open| {
                        open.ops.remove(&op).map(|(slot, seg)| {
                            open.slots[slot].segs[seg] = Some(
                                completion_bytes(&ev)
                                    .expect("a granted corpus range fetch completes (fail loud)"),
                            );
                            open.ops.is_empty()
                        })
                    });
                    if let Some(done) = open_hit {
                        if done {
                            let mut open = l.pending_open.take().expect("the in-flight open");
                            let corpus = l.corpus.as_ref().expect("corpus ready");
                            stage_fetched_batches(&core, corpus, &mut open, vocab);
                            let ro = open.ro;
                            let round = ro.round;
                            let _out = driver.on_round_open(&ro, &mut payloads);
                            export = Some(ExportState {
                                round,
                                collected: vec![None; numels.len()],
                                ops: BTreeMap::new(),
                            });
                            fence(round + 1);
                        }
                        continue;
                    }
                    let record_round = l
                        .pending_records
                        .iter_mut()
                        .find_map(|(round, p)| p.ops.remove(&op).map(|peer| (*round, peer)));
                    if let Some((round, peer)) = record_round {
                        let bytes = completion_bytes(&ev)
                            .expect("a record-listed committed payload fetches (fail loud)");
                        payloads.map.insert((round, peer), bytes);
                        let done = l
                            .pending_records
                            .get(&round)
                            .is_some_and(|p| p.ops.is_empty());
                        if done {
                            let pending = l
                                .pending_records
                                .remove(&round)
                                .expect("the completed record");
                            dispatch_record(&mut driver, &mut payloads, true, pending);
                            // The ingested-round boundary: the periodic live checkpoint
                            // cadence fires here (post-ingest state, spec §9).
                            l.maybe_start_checkpoint(&core, round);
                        }
                        continue;
                    }
                }
                // The quiesce walk's completions (§10.2 phase 2): collect the exported moments;
                // when the last lands, author + submit the typed manifest and QuiesceReady.
                if quiesce.as_ref().is_some_and(|w| w.ops.contains_key(&op)) {
                    {
                        let walk = quiesce.as_mut().expect("quiesce present");
                        let idx = walk.ops.remove(&op).expect("checked present");
                        walk.collected[idx] = Some(
                            completion_tensor(&ev)
                                .expect("moment export completes (quiesce fails loud)"),
                        );
                    }
                    let done = quiesce.as_ref().is_some_and(|w| {
                        w.ops.is_empty() && w.collected.iter().all(Option::is_some)
                    });
                    if done {
                        let walk = quiesce.take().expect("quiesce present");
                        let mut moments = walk.collected;
                        let moments: Vec<Vec<f32>> = moments
                            .iter_mut()
                            .map(|c| c.take().expect("collected"))
                            .collect();
                        let n = moments.len() / 2;
                        let (window_size, byte_len) = {
                            let c = core.borrow();
                            (c.window_size, family_byte_len(&numels_u64(&c.numels)))
                        };
                        // Seal the exported moments into their own families ([SF-6]); master + ef
                        // are already sealed and referenced by fold (`walk.master_fref`/`ef_fref`).
                        let (m_fold, m_chunks) =
                            seal_family(ADAMW_M_FAMILY, &moments[..n], window_size);
                        let (v_fold, v_chunks) =
                            seal_family(ADAMW_V_FAMILY, &moments[n..], window_size);
                        let mut parts = vec![
                            CkptPart::Family {
                                name: MASTER_FAMILY,
                                class: 0, // consensus-canonical (the digest-covered masters)
                                fref: walk.master_fref,
                            },
                            CkptPart::Family {
                                name: EF_FAMILY,
                                class: 1, // replica-local; continuity-required
                                fref: walk.ef_fref,
                            },
                            CkptPart::Family {
                                name: ADAMW_M_FAMILY,
                                class: 1,
                                fref: family_ref(m_fold, &m_chunks, byte_len, window_size),
                            },
                            CkptPart::Family {
                                name: ADAMW_V_FAMILY,
                                class: 1,
                                fref: family_ref(v_fold, &v_chunks, byte_len, window_size),
                            },
                        ];
                        // The resync watermark (§9): LIVE-MODE ONLY — the deterministic harness
                        // rings pin the four-section manifest shape, and only the live path
                        // restores through the watermark.
                        if live.is_some() {
                            parts.push(CkptPart::Inline {
                                name: "round",
                                class: 1,
                                bytes: driver
                                    .last_ingested()
                                    .unwrap_or(u64::MAX)
                                    .to_le_bytes()
                                    .to_vec(),
                            });
                        }
                        // Stage inline sections host-side; by-ref sections need no staging — the
                        // host reconstructs their FamilyRef from its own state store on
                        // `snapshot_state` (the families are sealed there).
                        for p in &parts {
                            if let CkptPart::Inline { bytes, .. } = p {
                                let _ = daemon_vhc_sdk::stage_state(bytes);
                            }
                        }
                        let manifest_bytes =
                            ckpt_manifest_bytes(&parts).expect("state-manifest cbor");
                        let status = daemon_vhc_sdk::snapshot_state(&manifest_bytes);
                        assert_eq!(status, 0, "snapshot_state rejected the trainer manifest");
                        // Same deliberate-leak shutdown discipline as Stop (§7.3).
                        std::mem::forget(driver);
                        std::mem::forget(core);
                        return OUTCOME_QUIESCE_READY;
                    }
                    continue;
                }
                let Some(st) = export.as_mut() else { continue };
                let Some(param_idx) = st.ops.remove(&op) else {
                    continue; // an import ack or another op — ignored
                };
                let Some(ciborium::value::Value::Array(result)) = ev.items.get(2) else {
                    continue;
                };
                let ok = result
                    .first()
                    .and_then(|v| v.as_integer())
                    .is_some_and(|n| i128::from(n) == 0);
                if !ok {
                    publish_tagged(9, st.round, b"export-failed");
                    continue;
                }
                let handle = result
                    .get(1)
                    .and_then(|v| v.as_integer())
                    .map(|n| u64::try_from(i128::from(n)).unwrap_or(0))
                    .unwrap_or(0);
                let bytes = daemon_vhc_sdk::read_buffer(handle);
                daemon_vhc_sdk::buffer_release(handle);
                let data = daemon_vhc_sdk_compute::decode_tensor_data(&bytes);
                st.collected[param_idx] = data.to_vec::<f32>().ok();

                if st.ops.is_empty() && st.collected.iter().all(Option::is_some) {
                    // All exports landed: publish trained θ, then the profile's real update.
                    let round = st.round;
                    let theta: Vec<Vec<f32>> = st
                        .collected
                        .iter_mut()
                        .map(|c| c.take().expect("collected"))
                        .collect();
                    export = None;

                    // The trained-θ voice is the harness-tier native comparison surface; the
                    // live plane skips it (a large frame per round with no live consumer).
                    if live.is_none() {
                        let mut theta_le = Vec::new();
                        for t in &theta {
                            for v in t {
                                theta_le.extend_from_slice(&v.to_le_bytes());
                            }
                        }
                        publish_tagged(2, round, &theta_le);
                    }

                    // The outer update streams: kick off the make_update walk over (θ, round-base,
                    // ef) windows — round-base + ef fetch from the sealed folds, the new ef family
                    // emits into a fresh stream. It voices the tag-3 commitment when it seals
                    // (drive_update_completion → UpdateStep::Sealed below).
                    start_update_walk(&core, round, theta);
                }
            }
            _ => {}
        }
    }
}
