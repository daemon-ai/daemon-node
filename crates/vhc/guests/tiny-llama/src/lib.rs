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
    SequenceSlice, StateManifest,
};
use daemon_vhc_sdk_compute::{export_tensor, fence, AutodiffHostBackend, HostBackend};
use daemon_vhc_sdk_consensus::digest::DigestCarry;
use daemon_vhc_sdk_consensus::fold_walk::{FoldWalk, Window};
use daemon_vhc_sdk_consensus::messages::{
    Commitment, Digest, Heartbeat, Join, RecordEntry, RoundOpen, ThroughputClass, VhcMessage,
};
use daemon_vhc_sdk_profiles::payload::HEADER_BYTES;
use daemon_vhc_sdk_profiles::streaming::{
    f32s_to_le_bytes, le_bytes_to_f32s, IngestFetch, IngestPart, SparseLocoIngestWalk,
    SparseLocoUpdateWalk, UpdateWindowInputs,
};
use daemon_vhc_sdk_profiles::SparseLocoCfg;
use daemon_vhc_sdk_rounds::{
    interval_for, slice_interval, BarrierRound, Committed, HostStaged, IngestOutcome,
    PayloadSource, RoundCfg, RoundExperiment, StepCtx as RoundStepCtx,
};
use serde::Deserialize;

/// The digest block size (the pinned det-lane granularity, matching `digest_state`).
const DIGEST_BLOCK: u32 = 64;
/// The module's own linear-memory floor, independent of geometry: the wasm image's static data plus
/// the allocator arena a Rust `cdylib` of this size touches before it folds anything (the Burn
/// router's op-encoding buffers, the event-frame scratch, the parsed config). Added to the
/// geometry-derived working set in [`TinyLlama::decl_for_config`]; measured on the real-geometry
/// gates, which print the run's peak linear memory beside the claim this figure feeds.
///
/// This module's own figure, well above the universal toolchain floor the SDK lifts every claim to
/// ([`daemon_vhc_sdk::module::WASM_LINEAR_MEMORY_FLOOR_BYTES`]): that one is what a bare `cdylib`
/// costs, this one is what THIS image costs.
const MODULE_BASELINE_BYTES: u64 = 12 << 20;
/// The in-flight window bound for the streamed fold walks (bounded read-ahead; the honest fuel
/// claim is per-window, §5.5). Small — the harness geometry is a handful of windows.
const WALK_IN_FLIGHT: u64 = 4;
/// The max in-flight restore window fetches (bounded read-ahead): the three restore families are
/// streamed with refill so the walk never exceeds the admitted `max_outstanding_ops`.
const RESTORE_IN_FLIGHT: usize = 4;
/// The guest linear-memory budget a streamed walk's IN-FLIGHT window set may occupy — the second
/// half of the bounded-guest-memory invariant (§3.2). [`WALK_IN_FLIGHT`] bounds the window COUNT;
/// at a real geometry a window is ~4 MiB and a walk holds several inputs per window (the update
/// walk holds θ + round-base + ef), so a fixed count is a byte budget that scales with the
/// geometry — exactly the class this module streams to avoid. [`walk_in_flight`] therefore takes
/// the min of the two: toy geometries keep the historical count (their windows are tiny), and the
/// fleet geometry falls back to as few as one window in flight.
const WALK_WINDOW_BUDGET_BYTES: u64 = 16 << 20;
/// The ceiling on the harness-tier trained-θ voice (tag 2), in bytes of the flat θ image: the
/// control-channel `max_frame_bytes` grant a run authors (the fleet ceremony authors exactly
/// 1 MiB — `daemon_vhc_testkit::ceremony`). The voice is a PARITY surface for the toy-geometry
/// comparison lanes (the live plane never consumes it and already skips it); at a real geometry
/// the θ image is gigabytes, which is neither a publishable frame nor a bounded guest buffer, so
/// the voice is skipped rather than assembled.
const THETA_VOICE_MAX_BYTES: u64 = 1 << 20;

/// The in-flight window bound for a walk holding `inputs_per_window` window-sized buffers per
/// window — `min(WALK_IN_FLIGHT, WALK_WINDOW_BUDGET_BYTES / (inputs × window_size))`, at least 1
/// (a walk that can never issue would never finish). See [`WALK_WINDOW_BUDGET_BYTES`].
fn walk_in_flight(window_size: u64, inputs_per_window: u64) -> u64 {
    let per_window = window_size.saturating_mul(inputs_per_window).max(1);
    (WALK_WINDOW_BUDGET_BYTES / per_window).clamp(1, WALK_IN_FLIGHT)
}
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

/// `da_run` outcomes for EXTERNAL INPUT this module refuses.
///
/// Each of the boundaries below reads bytes the module did not author — a fetched artifact, a
/// window named by a frame off the wire, a peer's committed payload. Malformed input there is an
/// ordinary, expected outcome of talking to other parties, so it MUST leave the run with a status
/// that names WHICH boundary refused and WHY. An abort would instead reach the host as an
/// undifferentiated guest trap, which is the failure class that has repeatedly cost a fleet
/// attempt: the trap says the guest died, never that a peer sent a container from a different
/// compression profile.
///
/// The codes are module-defined and disjoint from the `da_migrate` detail codes above.
const OUTCOME_CORPUS_MANIFEST_INVALID: u32 = 32;
const OUTCOME_CORPUS_RANGE_INVALID: u32 = 33;
const OUTCOME_ROUND_WINDOW_UNPLANNABLE: u32 = 34;
const OUTCOME_COMMITTED_PAYLOAD_INVALID: u32 = 35;

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
    /// A latched typed refusal of external input, set at the boundary that read it and returned as
    /// the run's outcome at the next event-loop turn. Boundaries reached from inside the barrier
    /// driver cannot return a run status themselves, so they latch it here instead of aborting.
    refusal: Option<u32>,
}

/// The in-flight streamed ingest walk (ABI §12.14, design §3.4/§5.4): the resident
/// `SparseLoco::ingest` re-expressed as a completion-driven multi-slice state machine over
/// round-base windows fetched from [`Core::master_fold`], emitting master windows into a
/// `state_open` stream and threading the digest carry — kicked off by
/// [`RoundExperiment::begin_ingest`], driven by fetch completions, sealed in the final slice.
struct IngestWalkState {
    /// The round being folded.
    round: u64,
    /// The fold engine (owns the schedule, the clip carries, and the digest carry).
    walk: SparseLocoIngestWalk,
    /// The open master-family write stream (`state_open`).
    stream: u64,
    /// The round-base family fold this walk reads its windows from.
    base_fold: [u8; 32],
    /// The record-ordered committed payloads, as the host BUFFERS `payload_get` delivered them
    /// ([SF-R3]): the host blake3-verified each against its record-listed hash before completing
    /// (architecture §3.4), and the fold reads section rows out of them with ranged `read_into` —
    /// no peer's payload ever enters linear memory whole.
    peer_buffers: Vec<u64>,
    /// Outstanding round-base fetches: `data@2` op → window ordinal.
    ops: BTreeMap<u64, u64>,
    /// Whether this walk's digest is voiced at seal (record-triggered) or folds silently
    /// (a catch-up ingest kicked off inside `on_round_open`).
    voice: bool,
    /// The barrier ingest size: the number of record-listed committed payloads this walk folds.
    /// At the barrier every committed payload is fetchable-and-verified before the fold begins, so
    /// this is BOTH the round's `committed` and this peer's `ingested` count — the honest
    /// guest-known bookkeeping reported alongside the digest over the reserved metric plane.
    committed: u32,
    /// Scratch for the f32→le state-emit seam.
    byte_buf: Vec<u8>,
    /// The emitted master family's ordered chunk hashes (accumulated per `state_emit`) — moved
    /// into [`Core::master_chunks`] at seal, so a checkpoint references the sealed master by fold.
    chunk_hashes: Vec<Hash>,
}

/// The in-flight streamed `make_update` walk (design §5.4): the resident
/// `SparseLoco::make_update` re-expressed as a completion-driven walk over (θ, round-base, ef)
/// windows — **θ exported off the device one window at a time**, round-base + ef windows fetched
/// from the sealed folds — emitting the NEW ef family into a `state_open` stream and assembling
/// the payload sections; sealed, it puts the committed container and voices the tag-3 commitment.
///
/// The walk owns the round's θ as DEVICE HANDLES (`theta_dev`), not values: a window's θ is a
/// device `slice` of the held handle, exported and read back at window granularity. Holding the
/// handles is what makes the lazy export honest — the post-ingest master apply re-materializes the
/// model's leaves, and these handles keep this round's θ alive (and unaliased) until the walk
/// seals, so every window exports the value the round actually trained.
struct UpdateWalkState {
    /// The round this update commits.
    round: u64,
    /// The fold engine (owns the schedule + the per-parameter section fragments).
    walk: SparseLocoUpdateWalk,
    /// The round's trained θ as device handles, canonical order (never values).
    theta_dev: Vec<burn::tensor::Tensor<HostBackend, 1>>,
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
    /// Outstanding θ window exports: `compute@2` op → window ordinal.
    theta_ops: BTreeMap<u64, u64>,
    /// Arrived round-base windows awaiting their siblings: ordinal → values.
    rb: BTreeMap<u64, Vec<f32>>,
    /// Arrived ef windows awaiting their siblings: ordinal → values.
    ef: BTreeMap<u64, Vec<f32>>,
    /// Arrived θ windows awaiting their siblings: ordinal → values.
    theta: BTreeMap<u64, Vec<f32>>,
    /// Scratch for the f32→le state-emit seam.
    byte_buf: Vec<u8>,
    /// The emitted NEW ef family's ordered chunk hashes — moved into [`Core::ef_chunks`] at seal.
    ef_chunks: Vec<Hash>,
    /// The open committed-payload buffer stream: the container is APPENDED window by window and
    /// sealed into the `BufferHandle` `payload_put` takes, so the producing side never holds it
    /// (the emit-side half of [SF-R3]).
    payload_stream: u64,
    /// The harness-tier trained-θ voice image (tag 2), filled window-by-window at the window's
    /// absolute family offset and published at seal. `None` when the voice is skipped: live mode
    /// (no consumer) or a θ image past [`THETA_VOICE_MAX_BYTES`].
    theta_voice: Option<Vec<u8>>,
}

/// The `RoundExperiment` adapter: v1 call points over the shared core.
struct C3Round {
    core: Rc<RefCell<Core>>,
}

impl RoundExperiment<HostStaged> for C3Round {
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

    fn ingest(&mut self, _round: u64, _committed: &Committed<HostStaged>) -> [u8; 16] {
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
    fn begin_ingest(&mut self, round: u64, committed: &Committed<HostStaged>) -> IngestOutcome {
        // The committed set is R host BUFFERS, not R decoded containers: the host verified each
        // against the record-listed hash before delivering its completion, and the fold reads the
        // rows it needs out of them ([SF-R3]).
        let peer_buffers: Vec<u64> = committed.items().iter().map(|it| it.bytes.0).collect();
        let mut core = self.core.borrow_mut();
        let (numels, window_size, base_fold) =
            (core.numels.clone(), core.window_size, core.master_fold);
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&round.to_le_bytes());
        let carry = DigestCarry::new(&Seed(seed), DIGEST_BLOCK);
        let numels_u64: Vec<u64> = numels.iter().map(|&n| n as u64).collect();
        let byte_len = family_byte_len(&numels_u64);
        // Two window-sized buffers per window in flight: the fetched round base and the master the
        // fold emits from it.
        let mut walk = SparseLocoIngestWalk::new(
            &core.profile_cfg,
            &numels,
            window_size,
            walk_in_flight(window_size, 2),
            peer_buffers.len(),
            carry,
        )
        .expect("ingest walk geometry (profile chunk divides every numel; window aligned)");
        // Cross-check every peer's container against the run's own geometry from its 40-byte
        // header alone — a producer that compressed under a different density or windowing is a
        // typed refusal here, never a mis-decode of its rows deeper in the fold.
        for &buf in &peer_buffers {
            let header = daemon_vhc_sdk::read_range(buf, 0, HEADER_BYTES as usize);
            if walk.layout().check_header(&header).is_err() {
                core.refusal = Some(OUTCOME_COMMITTED_PAYLOAD_INVALID);
                return IngestOutcome::Deferred;
            }
        }
        let opening = walk.start().expect("ingest walk start");
        let stream = daemon_vhc_sdk::state_open(MASTER_FAMILY, byte_len);
        let ops = BTreeMap::new();
        let voice = core.ingest_voices;
        // The barrier folds exactly the record-listed committed set (every item fetchable +
        // blake3-verified at `Committed::mint`), so the ingested count IS the committed count.
        let committed = u32::try_from(committed.items().len()).unwrap_or(u32::MAX);
        core.ingest_walk = Some(IngestWalkState {
            round,
            walk,
            stream,
            base_fold,
            peer_buffers,
            ops,
            voice,
            committed,
            byte_buf: Vec::new(),
            chunk_hashes: Vec::new(),
        });
        // Serve the opening reads: the payload rows resolve synchronously out of their buffers
        // (a prompt ranged `read_into`), the round-base windows are async `data@2` fetches.
        let Core {
            ingest_walk,
            family_base,
            model,
            ..
        } = &mut *core;
        let st = ingest_walk.as_mut().expect("the ingest walk just stashed");
        drive_ingest_fetches(st, family_base, model, opening.issue);
        IngestOutcome::Deferred
    }
}

/// The guest's payload source at the barrier: the committed payloads by `(round, peer)` as the host
/// BUFFERS `payload_get` delivered ([SF-R3]).
///
/// The repr is [`HostStaged`] — the mint's host-verified class: the host blake3-checks a fetched
/// payload against the requested content address before it delivers the completion (architecture
/// §3.4), so re-hashing it in-guest would mean reading the whole blob into linear memory to
/// re-derive a fact the host already established. Ordering and all-or-nothing semantics at the mint
/// are unchanged.
struct PayloadMap {
    map: BTreeMap<(u64, PeerId), HostStaged>,
}

impl PayloadSource<HostStaged> for PayloadMap {
    fn payload(&mut self, round: u64, peer: &PeerId) -> Option<HostStaged> {
        self.map.get(&(round, *peer)).copied()
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
            // The incremental buffer-staging trio (buffer_open/append/seal) is the highest
            // introducing minor this module imports, so it fixes the declaration at 4 (ABI §1.3
            // step 5); the det-state write surface + register_state_chunks sit below it at 3.
            abi_minor: 4,
            channels: vec![0],
            // The config-independent fallback (a harness that hands no decodable config). The
            // honest, geometry-derived figures are `decl_for_config` below — this module's
            // footprint is a function of the run it is admitted for, not a constant.
            host_state_bytes: 16 << 20,
            host_scratch_bytes: 16 << 20,
            device_state_bytes: 32 << 20,
            device_scratch_bytes: 32 << 20,
        }
    }

    /// The **config-dependent** declaration (ABI §9.1, streaming det fold §5.5): this module's
    /// footprint is a function of the geometry it is admitted for, and the host enforces the
    /// declaration as the sandbox's memory cap — so it is derived here from the same constants that
    /// bound the walks, not guessed.
    ///
    /// Linear memory (`host_state_bytes`, the hard tier the pooling allocator meters exactly) is
    /// **O(fold windows + payload section rows + bookkeeping)** at ANY geometry, because both planes
    /// stream: state families window by window ([`walk_in_flight`]) and committed payloads through
    /// ranged reads of the host buffers they arrive in ([SF-R3]).
    ///
    /// The three walks' working sets are **summed, not maxed**. They never overlap in time — the
    /// init expansion finishes before a round opens, and the barrier defers a round open behind an
    /// in-flight fold — but wasm linear memory never shrinks and the guest allocator does not return
    /// pages, so a phase's peak is not recovered for the next one: their block sizes differ enough
    /// (one window vs four vs two-plus-per-peer-rows) that freed blocks do not satisfy the next
    /// phase's request. The high-water is therefore cumulative across phases, which is exactly what
    /// "linear memory never shrinks, so a module's transient peak IS its long-lived floor" means.
    /// `ceremony_round`'s measured-peak assertion is what keeps this derivation honest at the real
    /// geometry; it is why the sum, not the max, is declared.
    ///
    /// Host-side staging (`host_scratch_bytes`, the peak tier) is the outgoing committed container,
    /// which the module appends into a host buffer and never holds itself.
    fn decl_for_config(config: &[u8]) -> ModuleDecl {
        let mut decl = Self::decl();
        let Ok(cfg) = from_canonical_slice::<GuestCfg>(config) else {
            return decl;
        };
        let numels = cfg.model.param_numels();
        if numels.is_empty() || cfg.profile.chunk == 0 {
            return decl;
        }
        let numels_u64: Vec<u64> = numels.iter().map(|&n| n as u64).collect();
        let window = cfg.state.chunk_size.max(1);
        // A window never spans a parameter, so the widest one is bounded by the largest parameter.
        let largest_param = numels_u64.iter().max().copied().unwrap_or(0) * 4;
        let w = window.min(largest_param.max(1));
        let peers = cfg.roster.len().max(1) as u64;
        let (chunk, k) = (u64::from(cfg.profile.chunk), u64::from(cfg.profile.topk));
        let rows = (w / (chunk * 4)).max(1);
        // One window's payload rows per peer: the decoded values + indices, plus the packed bytes
        // the ranged read lands (values row stride + the bit-packed index rows).
        let stride = 2 + (k * u64::from(cfg.profile.bits)).div_ceil(8);
        let idx_bits = u64::from(daemon_vhc_det::index_bits(chunk as usize).unwrap_or(32));
        let payload_window = rows * k * 8 + rows * stride + (rows * k * idx_bits).div_ceil(8);
        // The seed/artifact init expansion: the window it generates, its f32-le image, and the
        // zeroed `ef` window it seals beside it.
        let init_peak = 3 * w;
        // The update walk: θ, round base, ef and the emitted ef window, plus the folded window's
        // payload section pair on its way to `buffer_append`.
        let update_peak = walk_in_flight(window, 4) * (4 * w + payload_window);
        // The ingest walk: the round-base and master windows, plus one window's decoded section
        // rows per committed peer.
        let ingest_peak = walk_in_flight(window, 2) * (2 * w + peers * payload_window);
        // Bookkeeping that scales with the geometry: the four families' chunk-hash lists (master +
        // ef every round, both AdamW moments at a checkpoint/drain) and the per-parameter tables.
        let chunks = daemon_vhc_proto::det_state::family_chunk_count(&numels_u64, window);
        let bookkeeping = 4 * chunks * 32 + numels_u64.len() as u64 * 256;
        // The training step's guest-resident working set, in two named terms because they live on
        // two different scales:
        //
        //   * per MICRO-BATCH — the input and target id rows the forward pass builds (`i64` each),
        //     alive only for the step that consumes them;
        //   * per ROUND — a live round plans and stages the whole round's window before it trains
        //     the first step, so the fetched corpus bytes (one `token_width`-wide element per
        //     token, bounded by the covering-chunk grid the fetch plan splits on) and their decoded
        //     `u32` tokens are ALL resident across the round's inner loop. Declaring these at
        //     micro-batch scale would under-declare them by `steps_per_round`.
        //
        // Both terms are O(TOKENS), and keeping them there is a standing requirement, not an
        // observation: the causal mask and the target selection are O(tokens²) and O(tokens × vocab),
        // which at the fleet geometry are 16 MiB and 256 MiB — multiples of this whole claim. Both
        // are built and held ON DEVICE (`model.rs`). A forward pass that materializes either one in
        // linear memory does not overrun a budget, it exhausts the sandbox's memory cap and aborts
        // the guest.
        let tokens_mb = u64::from(cfg.model.seq_len) * u64::from(cfg.micro_batch);
        let tokens_round = tokens_mb * u64::from(cfg.steps_per_round.max(1));
        let step_peak = tokens_mb * (8 + 8) + tokens_round * (4 + 4);
        decl.host_state_bytes =
            MODULE_BASELINE_BYTES + init_peak + update_peak + ingest_peak + bookkeeping + step_peak;
        // The outgoing container: one section pair per window, staged host-side.
        let total_rows = numels_u64.iter().sum::<u64>() / chunk;
        decl.host_scratch_bytes =
            total_rows * stride + (total_rows * k * idx_bits).div_ceil(8) + HEADER_BYTES;
        decl
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
/// where each of its sequences sits inside the round's coalesced fetches, in corpus order.
struct PendingBatch {
    sequences: u32,
    slices: Vec<SequenceSlice>,
}

/// A round open awaiting its `data@2` corpus fetches (live mode).
struct PendingOpen {
    ro: RoundOpen,
    /// fetch op → index into [`PendingOpen::fetched`].
    ops: BTreeMap<u64, usize>,
    /// The byte length each fetch was planned for — what its completion must deliver exactly.
    planned: Vec<u64>,
    /// The round's coalesced fetches, one per covering chunk the window touches; `None` until the
    /// fetch completes.
    fetched: Vec<Option<Vec<u8>>>,
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
    /// The streamed AdamW moment seal (window-by-window off the device).
    moments: MomentSealWalk,
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
        let (tensors, numels, window_size, master_fref, ef_fref) = {
            let c = core.borrow();
            let byte_len = family_byte_len(&numels_u64(&c.numels));
            (
                c.model.moment_tensors(),
                c.numels.clone(),
                c.window_size,
                family_ref(c.master_fold, &c.master_chunks, byte_len, c.window_size),
                family_ref(c.ef_fold, &c.ef_chunks, byte_len, c.window_size),
            )
        };
        self.pending_ckpt = Some(CkptWalk {
            round,
            master_fref,
            ef_fref,
            moments: MomentSealWalk::begin(tensors, &numels, window_size),
        });
    }

    /// Finish a completed checkpoint walk ([SF-6]): seal the exported AdamW moments into their own
    /// families (accumulating their chunk hashes), author the by-reference checkpoint DOCUMENT
    /// (master + ef + adamw_m + adamw_v by fold, plus the inline round watermark) via the shared
    /// codec, and put it on the payload plane. A live checkpoint moves ZERO family bytes locally —
    /// the families are already sealed; the referenced chunks are uploaded host-side on recognition
    /// of the document. Best-effort: a failed export abandons the walk (the next cadence retries).
    fn finish_checkpoint(&mut self, core: &Rc<RefCell<Core>>, walk: CkptWalk) {
        let window_size = core.borrow().window_size;
        let (m_fref, v_fref) = walk.moments.families(window_size);
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
                fref: m_fref,
            },
            CkptPart::Family {
                name: ADAMW_V_FAMILY,
                class: 1,
                fref: v_fref,
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

/// One completion's `Ok(BufferHandle)` (`None` on a failed op) — the handle KEPT, not read: the
/// caller decides whether to pull the whole buffer into linear memory or range-read it ([SF-R3]).
fn completion_handle(ev: &daemon_vhc_sdk::Event) -> Option<u64> {
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
    result
        .get(1)
        .and_then(|v| v.as_integer())
        .map(|n| u64::try_from(i128::from(n)).unwrap_or(0))
}

/// Read one completion's `Ok(BufferHandle)` payload as raw bytes and release it (`None` on a failed
/// op) — for the SMALL completions whose whole object is the point (a manifest, a state window).
fn completion_bytes(ev: &daemon_vhc_sdk::Event) -> Option<Vec<u8>> {
    let handle = completion_handle(ev)?;
    let bytes = daemon_vhc_sdk::read_buffer(handle);
    daemon_vhc_sdk::buffer_release(handle);
    Some(bytes)
}

/// One completion's `Ok(Hash)` payload — the 32-byte content address a `payload_put` reports once
/// the object is durable. The producing guest learns its own commitment hash here rather than
/// hashing the container itself: it never held the container to hash.
fn completion_hash(ev: &daemon_vhc_sdk::Event) -> Option<[u8; 32]> {
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
    match result.get(1) {
        Some(ciborium::value::Value::Bytes(b)) => b.as_slice().try_into().ok(),
        _ => None,
    }
}

/// The quiesce snapshot's async moment seal ([SF-6]): only the AdamW moments export off the
/// device, and they do so window-by-window ([`MomentSealWalk`]) — never as resident families. The
/// already-sealed master + ef families are captured by-reference at drain start
/// (`master_fref`/`ef_fref`); the four sections are declared by fold, so the host reconstructs the
/// by-ref FamilyRefs from its state store and the drain moves ZERO family bytes.
struct QuiesceWalk {
    moments: MomentSealWalk,
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
/// slots in training order (step-major, then micro-window), over ONE fetch plan for the whole
/// round.
///
/// The window is planned whole rather than micro-batch by micro-batch. A micro-batch-at-a-time
/// plan issues one range per micro-batch, and when several of them fall inside the same covering
/// chunk — which is what the ceremony corpus's shard-sized chunks guarantee for a single peer's
/// round — the host transfers and verifies that whole chunk once per range. Planning the round
/// whole fetches each covering chunk exactly once; each fetch stays bounded by the chunk, so the
/// bytes the guest holds are bounded by the manifest's declared `chunk_size` and never by the
/// shard.
///
/// Returns `None` when the window a `RoundOpen` names does not plan over the pinned manifest: the
/// frame is external input, so an unplannable window is a typed refusal at the caller, never an
/// abort inside the planner.
fn plan_open_fetches(
    corpus: &LiveCorpus,
    ro: &RoundOpen,
    round_cfg: &RoundCfg,
) -> Option<PendingOpen> {
    let interval = interval_for(ro.batch, ro.seed, &round_cfg.roster, &round_cfg.peer);
    let steps = slice_interval(interval, round_cfg.steps_per_round, round_cfg.micro_batch);
    // The round's sequences in training order, plus where each slot's run starts in them. Global
    // sequence ids wrap modulo the corpus (the established window rule).
    let mut window: Vec<u64> = Vec::new();
    let mut spans: Vec<(usize, usize, u32)> = Vec::new();
    for step in &steps {
        for mb in &step.micro {
            let start = window.len();
            window.extend((mb.start..mb.end).map(|s| s % corpus.total_sequences));
            spans.push((
                start,
                window.len() - start,
                u32::try_from(mb.end - mb.start).unwrap_or(0),
            ));
        }
    }
    let plan = daemon_vhc_sdk::plan_covering_window(&corpus.manifest, &window).ok()?;
    let mut ops = BTreeMap::new();
    for (idx, f) in plan.fetches.iter().enumerate() {
        ops.insert(
            daemon_vhc_sdk::data_fetch(&f.shard_hash, f.range_off, f.range_len),
            idx,
        );
    }
    let slots = spans
        .into_iter()
        .map(|(start, count, sequences)| PendingBatch {
            sequences,
            slices: plan.sequences[start..start + count].to_vec(),
        })
        .collect();
    Some(PendingOpen {
        ro: ro.clone(),
        ops,
        planned: plan.fetches.iter().map(|f| f.range_len).collect(),
        fetched: (0..plan.fetches.len()).map(|_| None).collect(),
        slots,
    })
}

/// Decode a pending open's fetched segments into staged batches (training order), clamped into
/// the model's vocabulary (`token % vocab` — the deterministic tokenizer-to-model shim applied
/// identically by every peer).
fn stage_fetched_batches(
    core: &Rc<RefCell<Core>>,
    corpus: &LiveCorpus,
    open: &PendingOpen,
    vocab: u32,
) {
    let seq_len = corpus.manifest.seq_len;
    let width = corpus.manifest.token_width;
    let little = corpus.manifest.endianness == Endianness::Little;
    for slot in &open.slots {
        let mut raw = Vec::new();
        for seq in &slot.slices {
            for part in &seq.parts {
                let bytes = open.fetched[part.fetch]
                    .as_deref()
                    .expect("every planned fetch landed before staging");
                let off = part.offset as usize;
                raw.extend_from_slice(&bytes[off..off + part.len as usize]);
            }
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

/// Report the round outcome over the reserved metric plane for every digest-voicing outbound (the
/// opacity-safe live digest surface, ABI `round_metrics`). Called alongside [`emit_round_outbounds`]
/// at the ingest seal, where the barrier ingest size is known: `committed == ingested` (every
/// record-listed payload folded to reach the digest), and `stalled` distinguishes a straggled
/// round that caught up (`CaughtUp`) from an on-time ingest (`RoundComplete`). This is a strictly
/// ADDITIONAL host-visible report of the SAME digest the guest already voiced as tag-4; it touches
/// no det-lane math, no round logic, and no frame vocabulary.
fn report_round_metrics(committed: u32, out: &[daemon_vhc_sdk_rounds::Outbound]) {
    for o in out {
        let (round, digest, stalled) = match o {
            daemon_vhc_sdk_rounds::Outbound::RoundComplete { round, digest } => {
                (*round, *digest, false)
            }
            daemon_vhc_sdk_rounds::Outbound::CaughtUp { round, digest } => (*round, *digest, true),
            _ => continue,
        };
        daemon_vhc_sdk::report_round_outcome(round, committed, committed, stalled, digest);
    }
}

/// Run the barrier on a fully-staged record and voice the outbounds.
fn dispatch_record(
    driver: &mut BarrierRound<C3Round, HostStaged>,
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
    driver: &mut BarrierRound<C3Round, HostStaged>,
    payloads: &mut PayloadMap,
    ro: RoundOpen,
    export: &mut Option<u64>,
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
        *export = Some(ro.round);
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

/// The streamed seal of the two device-resident AdamW moment families (`adamw_m`, then
/// `adamw_v`) — the checkpoint/drain counterpart of [`stream_seed_init`].
///
/// A moment family lives device-side, so sealing it means reading it back, and reading a family
/// back WHOLE is exactly the residency class this module streams to avoid (~2.93 GiB per family at
/// the fleet geometry, with the tied embedding alone a single 192 MiB readback — past both the
/// linear-memory cap and the per-slice readback allowance). The walk exports ONE window at a time
/// (a device `slice` of the held handle → `export`), emits each window into the family's
/// `state_open` stream in ascending window order — the pinned fold-walk schedule, whose windows
/// ARE the family's chunks, so the sealed fold is bit-identical to a resident seal's — and
/// accumulates the [SF-R1] chunk hashes. Peak guest memory is the in-flight window set.
struct MomentSealWalk {
    /// The moment tensors as device handles: all of `m`, then all of `v` (canonical order).
    tensors: Vec<burn::tensor::Tensor<HostBackend, 1>>,
    /// The per-family window schedule (both halves share the layout).
    schedule: Vec<Window>,
    /// Which half is streaming: 0 = `adamw_m`, 1 = `adamw_v`.
    half: usize,
    /// The fold cursor over `schedule` for the half in flight (bounded read-ahead + ascending
    /// emit order for arbitrary completion arrival order).
    walk: FoldWalk,
    /// The in-flight window bound both halves run at.
    in_flight: u64,
    /// The half's open write stream.
    stream: u64,
    /// Outstanding window exports: `compute@2` op → window ordinal.
    ops: BTreeMap<u64, u64>,
    /// Arrived-but-not-yet-emitted windows (emit order is ascending by contract).
    pending: BTreeMap<u64, Vec<f32>>,
    /// The half's accumulated chunk hashes.
    chunks: Vec<Hash>,
    /// Scratch for the f32→le state-emit seam.
    byte_buf: Vec<u8>,
    /// The sealed halves in order: `(fold, chunk hashes)` for `adamw_m`, then `adamw_v`.
    sealed: Vec<([u8; 32], Vec<Hash>)>,
    /// The family byte length (shared by both halves).
    byte_len: u64,
}

/// The outcome of routing a completion to a [`MomentSealWalk`].
enum MomentStep {
    /// Not one of this walk's window exports.
    NotMine,
    /// A window emitted (or a half sealed); the walk continues.
    Progressed,
    /// A window export reported a device failure — the caller decides (a drain fails loud, a
    /// periodic checkpoint abandons its cadence slot).
    Failed,
    /// Both moment families are sealed ([`MomentSealWalk::families`] is ready).
    Sealed,
}

impl MomentSealWalk {
    /// Open `adamw_m` and issue its first window exports. `tensors` is `moment_tensors()` (all of
    /// `m`, then all of `v`).
    fn begin(
        tensors: Vec<burn::tensor::Tensor<HostBackend, 1>>,
        numels: &[usize],
        window_size: u64,
    ) -> Self {
        let numels_u64 = numels_u64(numels);
        let schedule = daemon_vhc_sdk_consensus::fold_walk::windows(&numels_u64, window_size);
        let byte_len = family_byte_len(&numels_u64);
        // Two window-sized buffers per window in flight: the exported readback and its le image.
        let in_flight = walk_in_flight(window_size, 2);
        let mut walk = FoldWalk::new(schedule.len() as u64, in_flight);
        let opening = walk.start();
        let mut me = Self {
            tensors,
            schedule,
            half: 0,
            walk,
            in_flight,
            stream: daemon_vhc_sdk::state_open(ADAMW_M_FAMILY, byte_len),
            ops: BTreeMap::new(),
            pending: BTreeMap::new(),
            chunks: Vec::new(),
            byte_buf: Vec::new(),
            sealed: Vec::new(),
            byte_len,
        };
        me.issue(&opening.issue);
        me
    }

    /// Export the listed windows of the half in flight (device `slice` → `export`).
    fn issue(&mut self, ordinals: &[u64]) {
        let base = self.half * (self.tensors.len() / 2);
        for &ordinal in ordinals {
            let w = self.schedule[usize::try_from(ordinal).expect("ordinal fits usize")];
            let off = usize::try_from(w.param_off / 4).expect("offset fits usize");
            let elems = usize::try_from(w.len / 4).expect("window fits usize");
            let window = self.tensors[base + w.param as usize]
                .clone()
                .slice([off..off + elems]);
            self.ops.insert(export_tensor(window), ordinal);
        }
    }

    /// Drive one completion against the walk.
    fn on_completion(&mut self, ev: &daemon_vhc_sdk::Event, op: u64) -> MomentStep {
        let Some(ordinal) = self.ops.remove(&op) else {
            return MomentStep::NotMine;
        };
        let Some(vals) = completion_tensor(ev) else {
            return MomentStep::Failed;
        };
        let actions = self
            .walk
            .on_completion(ordinal)
            .expect("the moment walk accepts its own outstanding window");
        self.pending.insert(ordinal, vals);
        for folded in &actions.fold {
            let vals = self
                .pending
                .remove(folded)
                .expect("a folded window is stashed");
            f32s_to_le_bytes(&vals, &mut self.byte_buf);
            daemon_vhc_sdk::state_emit(self.stream, &self.byte_buf);
            self.chunks.push(blake3_hash(&self.byte_buf));
        }
        self.issue(&actions.issue);
        if actions.seal {
            let fold = daemon_vhc_sdk::state_seal(self.stream);
            self.sealed.push((fold, std::mem::take(&mut self.chunks)));
            if self.half == 1 {
                return MomentStep::Sealed;
            }
            // On to `adamw_v`: same schedule, a fresh cursor and stream.
            self.half = 1;
            self.walk = FoldWalk::new(self.schedule.len() as u64, self.in_flight);
            self.stream = daemon_vhc_sdk::state_open(ADAMW_V_FAMILY, self.byte_len);
            let opening = self.walk.start();
            self.issue(&opening.issue);
        }
        MomentStep::Progressed
    }

    /// The sealed `(adamw_m, adamw_v)` family references, after [`MomentStep::Sealed`].
    fn families(&self, chunk_size: u64) -> (FamilyRef, FamilyRef) {
        let fref = |i: usize| {
            let (fold, chunks) = &self.sealed[i];
            family_ref(*fold, chunks, self.byte_len, chunk_size)
        };
        (fref(0), fref(1))
    }
}

/// Expand the seed-derived matched init (§6.1a) and land it in its two homes in ONE bounded pass:
/// every `window_size`-byte window is emitted into the master family's seal stream (accumulating
/// the chunk hash, so the sealed fold IS the fold the walk schedule assumes and the chunk list is
/// the [SF-R1] self-sealed one a by-reference checkpoint section needs) and written straight into
/// the device parameter. Returns the sealed master fold + its ordered chunk hashes.
///
/// The expansion is counter-based, so a window expands independently of the split
/// (`daemon_vhc_det::seed_init_window` exists for exactly this, and the genesis authoring folds the
/// pinned `expected_root` the same streaming way): peak guest memory is ONE window, not one
/// parameter and never one family — at the fleet-ceremony geometry a resident family is ~2.93 GiB.
fn stream_seed_init(
    model: &mut TinyLlamaModel<AutodiffHostBackend>,
    numels: &[usize],
    seed: &[u8; 32],
    dist: u64,
    window_size: u64,
) -> ([u8; 32], Vec<Hash>) {
    let stream = daemon_vhc_sdk::state_open(MASTER_FAMILY, family_byte_len(&numels_u64(numels)));
    let step = (window_size / 4).max(1) as usize; // elements per window
    let mut window: Vec<f32> = Vec::with_capacity(step);
    let mut bytes: Vec<u8> = Vec::with_capacity(step * 4);
    let mut chunks = Vec::new();
    for (i, &numel) in numels.iter().enumerate() {
        // Per-parameter chunking: a parameter never spans a chunk boundary; its last chunk is
        // short (the `det_state` family chunking every fold walk assumes).
        let mut off = 0usize;
        while off < numel {
            let take = step.min(numel - off);
            daemon_vhc_det::seed_init_window(seed, dist, i as u64, off, take, &mut window)
                .expect("the genesis seed-init distribution id is implemented");
            f32s_to_le_bytes(&window, &mut bytes);
            daemon_vhc_sdk::state_emit(stream, &bytes);
            chunks.push(blake3_hash(&bytes));
            model.write_param_window(i, off, &window);
            off += take;
        }
    }
    (daemon_vhc_sdk::state_seal(stream), chunks)
}

/// Seal an ALL-ZERO family into a self-sealed fold ([SF-R1]) — the fresh-join `ef` residuals —
/// from one reusable zero window, so the cost is the fold, not a resident family. Chunked exactly
/// like [`stream_seed_init`], so the fold is the one a zeroed family would have produced.
fn seal_zeroed_family(tag: &str, numels: &[usize], window_size: u64) -> ([u8; 32], Vec<Hash>) {
    let stream = daemon_vhc_sdk::state_open(tag, family_byte_len(&numels_u64(numels)));
    let step = (window_size / 4).max(1) as usize; // elements per window
    let zeros = vec![0u8; step * 4];
    let mut chunks = Vec::new();
    for &numel in numels {
        let mut off = 0usize;
        while off < numel {
            let take = step.min(numel - off);
            let chunk = &zeros[..take * 4];
            daemon_vhc_sdk::state_emit(stream, chunk);
            chunks.push(blake3_hash(chunk));
            off += take;
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
/// master family for length-aware ranged fetch ([SF-R2]), then stream the master windows back —
/// each window written STRAIGHT into its device parameter ([`TinyLlamaModel::write_param_window`],
/// the same landing the seed form uses), never assembled into a resident family. The model boots
/// from zeros; this replaces those with the matched init before the first round trains.
struct BootState {
    /// The pinned init det-state manifest hash (a granted plain artifact).
    manifest: Hash,
    /// The outstanding manifest fetch, until it lands.
    manifest_op: Option<u64>,
    /// Outstanding master-window fetches: `data@2` op → window ordinal.
    window_ops: BTreeMap<u64, u64>,
    /// The master family's window schedule (set once the manifest lands).
    schedule: Vec<Window>,
    /// Not-yet-issued window ordinals, ascending — drained with bounded refill so neither the
    /// `max_outstanding_ops` grant nor the in-flight window budget is breached.
    queued: VecDeque<u64>,
    /// The in-flight window bound (one fetched window buffer per window).
    in_flight: usize,
    /// Whether every master window has landed on the device.
    done: bool,
}

impl BootState {
    fn new(manifest: Hash) -> Self {
        Self {
            manifest,
            manifest_op: None,
            window_ops: BTreeMap::new(),
            schedule: Vec::new(),
            queued: VecDeque::new(),
            in_flight: RESTORE_IN_FLIGHT,
            done: false,
        }
    }

    /// Issue queued window fetches up to the in-flight bound (bounded read-ahead).
    fn issue_more(&mut self, base_fold: &[u8; 32], family_base: &[u64]) {
        while self.window_ops.len() < self.in_flight {
            let Some(ordinal) = self.queued.pop_front() else {
                break;
            };
            let w = self.schedule[usize::try_from(ordinal).expect("ordinal fits usize")];
            let off = family_base[w.param as usize] + w.param_off;
            let op = daemon_vhc_sdk::data_fetch(base_fold, off, w.len);
            self.window_ops.insert(op, ordinal);
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
        boot.queued = boot.schedule.iter().map(|w| w.ordinal).collect();
        // One fetched window buffer per window in flight.
        boot.in_flight = usize::try_from(walk_in_flight(window_size, 1)).unwrap_or(1);
        let (base_fold, family_base) = {
            let c = core.borrow();
            (c.master_fold, c.family_base.clone())
        };
        boot.issue_more(&base_fold, &family_base);
        return true;
    }
    let Some(ordinal) = boot.window_ops.remove(&op) else {
        return false;
    };
    let bytes = completion_bytes(ev).expect("an init master window fetches (fail loud)");
    let vals = le_bytes_to_f32s(&bytes).expect("init window is an f32-le image");
    let window = boot.schedule[usize::try_from(ordinal).expect("ordinal fits usize")];
    let off = usize::try_from(window.param_off / 4).expect("offset fits usize");
    let (base_fold, family_base) = {
        let mut c = core.borrow_mut();
        // Land the window straight on the device — no resident init image at any geometry.
        c.model
            .write_param_window(window.param as usize, off, &vals);
        (c.master_fold, c.family_base.clone())
    };
    boot.issue_more(&base_fold, &family_base);
    if boot.window_ops.is_empty() && boot.queued.is_empty() {
        // The matched init is resident ON DEVICE; seal a zeroed ef family for round 0 (the update
        // walk reads its ef windows from it).
        let mut c = core.borrow_mut();
        let (window_size, numels) = (c.window_size, c.numels.clone());
        let (ef_fold, ef_chunks) = seal_zeroed_family(EF_FAMILY, &numels, window_size);
        c.ef_fold = ef_fold;
        c.ef_chunks = ef_chunks;
        boot.done = true;
    }
    true
}

/// The streaming checkpoint REHYDRATION walk ([SF-6], design §7.3): a restoring instance registers
/// the checkpoint's family folds ([SF-R2]) and fetches their windows on demand — master → device
/// weights, adamw_m/v → device moments — each window landing STRAIGHT on the device
/// ([`TinyLlamaModel::write_param_window`] / `write_moment_window`), so peak guest memory is the
/// in-flight window set at any geometry. The `ef` family is registered but not streamed here: the
/// `make_update` walk reads its windows directly from the adopted fold.
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
    /// The in-flight window bound (one fetched window buffer per window).
    in_flight: usize,
    /// Windows not yet landed across all three families.
    remaining: usize,
    /// Whether the rehydration is complete (weights + moments on the device).
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
        let mut st = Self {
            schedule,
            folds: [refs.master.fold.0, refs.adamw_m.fold.0, refs.adamw_v.fold.0],
            family_base,
            remaining: pending.len(),
            pending,
            inflight: BTreeMap::new(),
            in_flight: usize::try_from(walk_in_flight(window_size, 1)).unwrap_or(1),
            done: false,
        };
        st.issue_more();
        st
    }

    /// Issue pending window fetches up to the in-flight bound (bounded read-ahead).
    fn issue_more(&mut self) {
        while self.inflight.len() < self.in_flight {
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
    let off = usize::try_from(window.param_off / 4).expect("offset fits usize");
    let param = window.param as usize;
    {
        // Land the window on the device as it arrives — the rehydration never holds a family.
        let model = &mut core.borrow_mut().model;
        match family {
            0 => model.write_param_window(param, off, &vals),
            1 => model.write_moment_window(false, param, off, &vals),
            _ => model.write_moment_window(true, param, off, &vals),
        }
    }
    restore.remaining -= 1;
    restore.issue_more();
    if restore.remaining == 0 {
        restore.done = true;
    }
    true
}

/// Serve a set of ingest-walk reads to fixpoint ([SF-R3]).
///
/// A committed-payload read is a RANGE of a host buffer the guest already holds, so it is a prompt
/// `read_into` served right here — no op, no completion, no whole-payload decode; peak residency is
/// one window's section rows per peer (~hundreds of KB). Only the round-base state windows are
/// asynchronous `data@2` fetches, which is also what paces the walk: each completion drives one
/// window's fold, so per-slice work stays bounded by construction.
///
/// Folded windows are `state_emit`ted and written straight into their device parameter as they are
/// produced, so the post-ingest master lands at window granularity and no resident master exists.
fn drive_ingest_fetches(
    st: &mut IngestWalkState,
    family_base: &[u64],
    model: &mut TinyLlamaModel<AutodiffHostBackend>,
    issue: Vec<IngestFetch>,
) -> bool {
    let mut queue: VecDeque<IngestFetch> = issue.into();
    let mut sealed = false;
    while let Some(fetch) = queue.pop_front() {
        let Some((offset, len)) = fetch.span else {
            // A round-base state window: the asynchronous half.
            let off = family_base[fetch.window.param as usize] + fetch.window.param_off;
            let op = daemon_vhc_sdk::data_fetch(&st.base_fold, off, fetch.window.len);
            st.ops.insert(op, fetch.window.ordinal);
            continue;
        };
        let peer = match fetch.part {
            IngestPart::Values(p) | IngestPart::Indices(p) => p as usize,
            IngestPart::RoundBase => unreachable!("the round base carries no payload span"),
        };
        let bytes = daemon_vhc_sdk::read_range(
            st.peer_buffers[peer],
            offset,
            usize::try_from(len).expect("a section span fits usize"),
        );
        let slice = st
            .walk
            .on_part_ready(fetch.part, fetch.window.ordinal, &bytes)
            .expect("the walk accepts the section rows it asked for");
        emit_ingest_windows(st, model, &slice.emitted);
        queue.extend(slice.issue);
        sealed |= slice.sealed;
    }
    sealed
}

/// Emit one slice's folded master windows: into the family's write stream (accumulating the [SF-R1]
/// chunk hash) and straight onto the device parameter.
fn emit_ingest_windows(
    st: &mut IngestWalkState,
    model: &mut TinyLlamaModel<AutodiffHostBackend>,
    emitted: &[(Window, Vec<f32>)],
) {
    for (window, master_vals) in emitted {
        f32s_to_le_bytes(master_vals, &mut st.byte_buf);
        daemon_vhc_sdk::state_emit(st.stream, &st.byte_buf);
        st.chunk_hashes.push(blake3_hash(&st.byte_buf));
        let off = usize::try_from(window.param_off / 4).expect("offset fits usize");
        model.write_param_window(window.param as usize, off, master_vals);
    }
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
        /// The barrier ingest size (committed == ingested) — reported with the digest over the
        /// reserved metric plane.
        committed: u32,
    },
}

/// Drive one completion against the in-flight streamed ingest walk. Folds the maximal contiguous
/// run now available — each fold `state_emit`s its master window, advances the carry, and writes
/// the same window STRAIGHT into its device parameter, so the post-ingest master lands on the
/// device at window granularity and no resident master ever exists — refills the read window, and
/// at seal closes the fold and advances the round base. The caller finishes the barrier + digest
/// voice.
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

    let sealed;
    {
        let mut core_mut = core.borrow_mut();
        let Core {
            ingest_walk,
            family_base,
            model,
            ..
        } = &mut *core_mut;
        let w = ingest_walk.as_mut().expect("ingest walk present");
        let ordinal = w.ops.remove(&op).expect("op is an outstanding window read");
        let slice = w
            .walk
            .on_part_ready(IngestPart::RoundBase, ordinal, &bytes)
            .expect("the walk accepts the completed window");
        emit_ingest_windows(w, model, &slice.emitted);
        sealed = slice.sealed | drive_ingest_fetches(w, family_base, model, slice.issue);
    }
    if !sealed {
        return IngestStep::Progressed;
    }
    // The sealing slice: close the fold (master r becomes the round base of r+1) and finalize the
    // carry. The device already carries the new master — every folded window was written as it
    // was emitted.
    let mut core_mut = core.borrow_mut();
    let mut state = core_mut.ingest_walk.take().expect("the sealed walk");
    // The peers' payload buffers have been fully read; give their quota back (§3.4 ownership).
    for buf in state.peer_buffers.drain(..) {
        daemon_vhc_sdk::buffer_release(buf);
    }
    let round = state.round;
    let voice = state.voice;
    let committed = state.committed;
    let fold = daemon_vhc_sdk::state_seal(state.stream);
    let digest = *state
        .walk
        .seal()
        .expect("the walk sealed after every window folded")
        .finalize()
        .as_bytes();
    core_mut.master_fold = fold;
    core_mut.master_chunks = std::mem::take(&mut state.chunk_hashes);
    IngestStep::Sealed {
        round,
        digest,
        voice,
        committed,
    }
}

/// Kick off the streamed `make_update` walk for `round` over its three window sources: the trained
/// θ (a device `slice` + `export` per window, off the handles captured at the round-final fence),
/// the round base (master r−1) and the prior ef family (both `data@2` window fetches from their
/// sealed folds). The NEW ef family emits into a fresh `state_open` stream. Stashed in `Core`,
/// driven by [`drive_update_completion`].
///
/// `voice` requests the harness-tier tag-2 trained-θ frame; it is honored only when the θ image is
/// a publishable frame ([`THETA_VOICE_MAX_BYTES`]).
fn start_update_walk(
    core: &Rc<RefCell<Core>>,
    round: u64,
    theta_dev: Vec<burn::tensor::Tensor<HostBackend, 1>>,
    voice: bool,
) {
    let mut c = core.borrow_mut();
    let numels = c.numels.clone();
    let numels_u64: Vec<u64> = numels.iter().map(|&n| n as u64).collect();
    let byte_len = family_byte_len(&numels_u64);
    let (window_size, base_fold, ef_fold) = (c.window_size, c.master_fold, c.ef_fold);
    // Four window-sized buffers per window in flight: θ, the round base, ef, and the new ef the
    // fold emits.
    let mut walk = SparseLocoUpdateWalk::new(
        &c.profile_cfg,
        &numels,
        window_size,
        walk_in_flight(window_size, 4),
    )
    .expect("update walk geometry (profile chunk divides every numel; window aligned)");
    let opening = walk.start().expect("update walk start");
    let ef_stream = daemon_vhc_sdk::state_open(EF_FAMILY, byte_len);
    // Open the committed container and lay its fixed header down first; every folded window
    // appends its own section pair after it, in fold order (the layout's own order).
    let payload_stream = daemon_vhc_sdk::buffer_open();
    daemon_vhc_sdk::buffer_append(payload_stream, &walk.payload_header());
    let theta_voice = (voice && byte_len <= THETA_VOICE_MAX_BYTES)
        .then(|| vec![0u8; usize::try_from(byte_len).expect("a voiced θ image fits usize")]);
    let mut st = UpdateWalkState {
        round,
        walk,
        theta_dev,
        ef_stream,
        base_fold,
        ef_fold,
        rb_ops: BTreeMap::new(),
        ef_ops: BTreeMap::new(),
        theta_ops: BTreeMap::new(),
        rb: BTreeMap::new(),
        ef: BTreeMap::new(),
        theta: BTreeMap::new(),
        byte_buf: Vec::new(),
        ef_chunks: Vec::new(),
        payload_stream,
        theta_voice,
    };
    issue_update_windows(&mut st, &c.family_base, &opening.issue);
    c.update_walk = Some(st);
}

/// Issue one window's three reads: the round-base + ef `data@2` fetches and the θ device export.
fn issue_update_windows(st: &mut UpdateWalkState, family_base: &[u64], issue: &[Window]) {
    for w in issue {
        let off = family_base[w.param as usize] + w.param_off;
        st.rb_ops.insert(
            daemon_vhc_sdk::data_fetch(&st.base_fold, off, w.len),
            w.ordinal,
        );
        st.ef_ops.insert(
            daemon_vhc_sdk::data_fetch(&st.ef_fold, off, w.len),
            w.ordinal,
        );
        let elem_off = usize::try_from(w.param_off / 4).expect("offset fits usize");
        let elems = usize::try_from(w.len / 4).expect("window fits usize");
        let theta_window = st.theta_dev[w.param as usize]
            .clone()
            .slice([elem_off..elem_off + elems]);
        st.theta_ops.insert(export_tensor(theta_window), w.ordinal);
    }
}

/// The outcome of routing a completion to the in-flight `make_update` walk.
enum UpdateStep {
    /// Not an update-walk window read.
    NotMine,
    /// A window arrived / folded; the walk continues.
    Progressed,
    /// The final slice sealed the ef family and the committed container — the caller has PUT the
    /// container and voices the tag-3 commitment when the put reports its content address.
    Sealed {
        /// The committed round.
        round: u64,
        /// The container's byte length (the record entry's `size`).
        size: u64,
        /// The in-flight `payload_put` whose completion carries the commitment hash.
        put_op: u64,
        /// The harness-tier trained-θ image to voice as tag 2 before the commitment, when the
        /// walk assembled one (see [`UpdateWalkState::theta_voice`]).
        theta_voice: Option<Vec<u8>>,
    },
}

/// Which of a window's three sources a completion carries.
enum UpdateSource {
    RoundBase,
    Ef,
    /// The θ window's device export (a `CBOR(TensorData)` readback, not an f32-le fetch).
    Theta,
}

/// Drive one completion against the in-flight `make_update` walk. Each window needs all three of
/// its sources (θ export, round base, ef); when the last one arrives the fold runs (Δ → ef
/// accumulate → top-k → pack), emits the new ef window, and appends the payload section fragment;
/// at seal it closes the ef fold and returns the assembled payload for the tag-3 voice.
fn drive_update_completion(
    core: &Rc<RefCell<Core>>,
    ev: &daemon_vhc_sdk::Event,
    op: u64,
) -> UpdateStep {
    let source = {
        let c = core.borrow();
        match c.update_walk.as_ref() {
            Some(st) if st.rb_ops.contains_key(&op) => UpdateSource::RoundBase,
            Some(st) if st.ef_ops.contains_key(&op) => UpdateSource::Ef,
            Some(st) if st.theta_ops.contains_key(&op) => UpdateSource::Theta,
            _ => return UpdateStep::NotMine,
        }
    };
    let vals = match source {
        UpdateSource::Theta => {
            completion_tensor(ev).expect("a θ window export completes (fail loud)")
        }
        _ => {
            let bytes = completion_bytes(ev).expect("update window fetch completes (fail loud)");
            le_bytes_to_f32s(&bytes).expect("update window is an f32-le image")
        }
    };
    let mut c = core.borrow_mut();
    let sealed = {
        let Core {
            update_walk,
            family_base,
            ..
        } = &mut *c;
        let st = update_walk.as_mut().expect("update walk present");
        let ordinal = match source {
            UpdateSource::RoundBase => st.rb_ops.remove(&op),
            UpdateSource::Ef => st.ef_ops.remove(&op),
            UpdateSource::Theta => st.theta_ops.remove(&op),
        }
        .expect("op is an outstanding update window read");
        match source {
            UpdateSource::RoundBase => st.rb.insert(ordinal, vals),
            UpdateSource::Ef => st.ef.insert(ordinal, vals),
            UpdateSource::Theta => st.theta.insert(ordinal, vals),
        };
        if st.rb.contains_key(&ordinal)
            && st.ef.contains_key(&ordinal)
            && st.theta.contains_key(&ordinal)
        {
            let round_base = st.rb.remove(&ordinal).expect("round-base present");
            let ef = st.ef.remove(&ordinal).expect("ef present");
            let theta = st.theta.remove(&ordinal).expect("θ present");
            let window = st.walk.schedule()[usize::try_from(ordinal).expect("ordinal fits")];
            // The harness-tier θ voice, assembled at the window's absolute family offset (the
            // walk folds ascending, but the voice is offset-addressed, so arrival order is moot).
            if let Some(image) = st.theta_voice.as_mut() {
                let at = usize::try_from(family_base[window.param as usize] + window.param_off)
                    .expect("voiced θ offset fits usize");
                for (i, v) in theta.iter().enumerate() {
                    image[at + i * 4..at + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
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
            // The window's committed-payload section pair leaves for the host buffer as it is
            // folded; the walk keeps only the running length.
            for (_w, bytes) in &step.payload {
                daemon_vhc_sdk::buffer_append(st.payload_stream, bytes.as_slice());
            }
            let issue = step.issue.clone();
            issue_update_windows(st, family_base, &issue);
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
    let theta_voice = st.theta_voice.take();
    let size = st
        .walk
        .seal()
        .expect("the update walk sealed after every window folded and tiled its layout");
    // Seal the appended container into the buffer `payload_put` takes. The commitment hash comes
    // back on the put's completion — the guest never held the container to hash it itself.
    let buf = daemon_vhc_sdk::buffer_seal(st.payload_stream);
    let put_op = daemon_vhc_sdk::payload_put(buf);
    daemon_vhc_sdk::buffer_release(buf);
    UpdateStep::Sealed {
        round: st.round,
        size,
        put_op,
        theta_voice,
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
    // The init source (§6) / restore rebuild. The model is allocated on-device as zeros in every
    // form and the real values are written in bounded windows — a guest-resident family is
    // O(parameters) (~2.93 GiB per copy at the fleet-ceremony geometry), which no wasm32 linear
    // memory holds, and holding one would break the streaming design's bounded-guest-memory
    // invariant (§3.2: the guest folds at O(chunks in flight)).
    //   - Restore ([SF-6]): a [`RestoreState`] walk registers the checkpoint's family folds
    //     ([SF-R2]) and STREAMS master → device weights and adamw_m/v → device moments, adopting
    //     the master/ef folds as the round base. `booted` stays false until the walk lands (the
    //     first round defers behind it, like the artifact boot).
    //   - Seed: deterministic expansion, streamed + sealed + `expected_root`-cross-checked
    //     synchronously ([`stream_seed_init`]).
    //   - Artifact: fetched asynchronously ([`BootState`]); trains from zeros until it lands.
    let mut boot: Option<BootState> = None;
    let device = daemon_vhc_sdk_compute::device();
    let mut model = TinyLlamaModel::<AutodiffHostBackend>::zeros(cfg.model.clone(), device);
    // The canonical master + replica-local ef live host-side as sealed folds (no resident copy).
    // Seed-init seals both synchronously — master from the streamed expansion (cross-checked
    // against the pin) and a zeroed ef; restore adopts the checkpoint's folds; the artifact form
    // registers/seals at boot.
    let mut master_fold = [0u8; 32];
    let mut master_chunks: Vec<Hash> = Vec::new();
    let mut ef_fold = [0u8; 32];
    let mut ef_chunks: Vec<Hash> = Vec::new();
    if restored.is_none() {
        match cfg.state.init {
            StateInit::Seed {
                seed,
                dist,
                expected_root,
            } => {
                let (mf, mc) = stream_seed_init(&mut model, &numels, &seed.0, dist, window_size);
                master_fold = mf;
                master_chunks = mc;
                let (ff, fc) = seal_zeroed_family(EF_FAMILY, &numels, window_size);
                ef_fold = ff;
                ef_chunks = fc;
                // Seed-init admission cross-check (§6.1a): the sealed expansion MUST reproduce
                // the pin.
                assert_eq!(
                    master_fold, expected_root.0,
                    "seed-init sealed fold does not match the pinned expected_root (typed init failure)"
                );
            }
            StateInit::Manifest { manifest } => boot = Some(BootState::new(manifest)),
        }
    }
    // Seed init is resident synchronously; a restore or an artifact fetch defers the first round.
    let mut booted = restored.is_none() && boot.is_none();
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
        refusal: None,
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
    let mut driver: BarrierRound<C3Round, HostStaged> =
        BarrierRound::new(C3Round { core: core.clone() }, round_cfg.clone());
    // A restored instance never re-ingests at or below the snapshot's watermark (§9 restore).
    if restored_round.is_some() {
        driver.resume_from(restored_round);
    }
    let mut payloads = PayloadMap {
        map: BTreeMap::new(),
    };
    // Record-listed committed payloads awaiting their `payload_get` completions (both modes).
    let mut pending_records: BTreeMap<u64, PendingRecord> = BTreeMap::new();
    // `payload_put` op → the (round, size) whose commitment its content address will voice.
    let mut pending_commit: BTreeMap<u64, (u64, u64)> = BTreeMap::new();
    // The round awaiting its round-final fence (its θ export walk starts there).
    let mut export: Option<u64> = None;
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
        // A boundary that read malformed external input latched its typed refusal rather than
        // aborting; the run ends on it here, naming which boundary refused.
        if let Some(code) = core.borrow().refusal {
            return code;
        }
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
                let (tensors, numels, window_size, master_fref, ef_fref) = {
                    let c = core.borrow();
                    let byte_len = family_byte_len(&numels_u64(&c.numels));
                    (
                        c.model.moment_tensors(),
                        c.numels.clone(),
                        c.window_size,
                        family_ref(c.master_fold, &c.master_chunks, byte_len, c.window_size),
                        family_ref(c.ef_fold, &c.ef_chunks, byte_len, c.window_size),
                    )
                };
                // The master + ef families are already sealed — captured by-reference (zero extra
                // reads); only the AdamW moments export here ([SF-6]), window by window.
                quiesce = Some(QuiesceWalk {
                    moments: MomentSealWalk::begin(tensors, &numels, window_size),
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
                    // Kind 1 (`[1, round, peer32, payload]`) is RETIRED: a committed payload never
                    // enters linear memory, in either mode. Both the harness and the live plane
                    // hand the guest the record's content address and it `payload_get`s the blob
                    // into a host buffer it then range-reads ([SF-R3]) — one code path, which is
                    // also what keeps the real-geometry gates honest about the toy tier (§1.3/§10.2).
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
                            let Some(open) = plan_open_fetches(corpus, &ro, &round_cfg) else {
                                return OUTCOME_ROUND_WINDOW_UNPLANNABLE;
                            };
                            l.pending_open = Some(open);
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
                            ro,
                            &mut export,
                            &mut deferred_open,
                        );
                    }
                    VhcMessage::RoundRecord(rr) => {
                        let entries: Vec<RecordEntry> = rr.inline.clone().unwrap_or_default();
                        // Module-driven payload custody, in BOTH modes: fetch every record-listed
                        // committed payload from the content-addressed plane (self included —
                        // idempotent and uniform) into a host buffer, then run the barrier. The
                        // host hash-verifies each fetch against the record's address before the
                        // completion lands, and the fold reads rows out of the buffer.
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
                            dispatch_record(&mut driver, &mut payloads, live.is_some(), pending);
                            // The live checkpoint fires at SEAL (in the ingest-walk `Sealed`
                            // handler), not here: `dispatch_record` only KICKS OFF the async
                            // deferred fold, so the canonical master has not yet advanced to
                            // this round — capturing now would strand a restorer a round back.
                        } else {
                            pending_records.insert(pending_round, pending);
                        }
                    }
                    _ => {}
                }
            }
            EV_FENCE => {
                let id = ev.uint(1);
                let Some(round) = export else { continue };
                if id != round + 1 {
                    continue; // a per-step depth-reset fence — not awaited
                }
                export = None;
                // The device passed the round's training. Capture θ as DEVICE HANDLES (no bytes
                // cross here) and hand them to the `make_update` walk, which exports them one
                // window at a time: a whole-parameter export is a whole-parameter readback (the
                // tied embedding alone is 192 MiB at the fleet geometry — past the linear-memory
                // cap AND the per-slice readback allowance), and a whole-family θ is the residency
                // class this module streams to avoid. The trained-θ voice is the harness-tier
                // native comparison surface; the live plane skips it (a large frame per round
                // with no live consumer).
                let theta_dev = core.borrow().model.export_tensors();
                start_update_walk(&core, round, theta_dev, live.is_none());
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
                        committed,
                    } => {
                        let out = driver.finish_ingest(round, digest, &mut payloads);
                        // A catch-up ingest kicked off inside `on_round_open` folds silently; only
                        // a record-triggered ingest voices its digest (matches the resident guest,
                        // which dropped `on_round_open`'s outbounds).
                        if voice {
                            emit_round_outbounds(live.is_some(), &out);
                            // Additionally report the round outcome over the reserved metric plane
                            // (the opacity-safe LIVE digest surface, ABI `round_metrics`): the host
                            // role session recognizes these metric NAMES and folds them into a
                            // `RoundOutcome` event — it never decodes this guest's `[4, round,
                            // digest]` frame. `committed == ingested` at the barrier (every
                            // record-listed payload folded); `stalled` is the caught-up class.
                            report_round_metrics(committed, &out);
                            // The ingested-round boundary (spec §9): the deferred fold has NOW
                            // sealed, so `core.master_fold` IS this round's canonical master and a
                            // live checkpoint's by-ref master agrees with its inline round
                            // watermark. Capturing at record dispatch (before the async seal) sealed
                            // the PRIOR round's master under this round's watermark, so a restorer
                            // resumed one outer step behind the survivor and diverged at its first
                            // post-restore round. Only record-triggered ingests reach a cadence
                            // boundary — the resident guest's synchronous checkpoint point.
                            if let Some(l) = live.as_mut() {
                                l.maybe_start_checkpoint(&core, round);
                            }
                        }
                        if !driver.ingest_in_flight() {
                            if let Some(ro) = deferred_open.take() {
                                try_open_round(
                                    &core,
                                    &mut driver,
                                    &mut payloads,
                                    ro,
                                    &mut export,
                                    &mut deferred_open,
                                );
                            }
                        }
                        continue;
                    }
                }
                // The in-flight streamed make_update walk's (θ export, round-base, ef) window
                // reads. At seal the new ef family is durable; voice the harness-tier trained θ,
                // externalize the committed container and voice the tag-3 commitment (the wire
                // Commitment rides the put completion in live mode).
                match drive_update_completion(&core, &ev, op) {
                    UpdateStep::NotMine => {}
                    UpdateStep::Progressed => continue,
                    UpdateStep::Sealed {
                        round,
                        size,
                        put_op,
                        theta_voice,
                    } => {
                        if let Some(image) = theta_voice {
                            publish_tagged(2, round, &image);
                        }
                        // The commitment voice waits for the put: its completion carries the
                        // container's content address (which is also when it is durable, so the
                        // coordinator's availability check and every peer's fetch can find it).
                        pending_commit.insert(put_op, (round, size));
                        continue;
                    }
                }
                // The committed container is durable: voice the commitment the put's content
                // address names.
                if let Some((round, size)) = pending_commit.remove(&op) {
                    let hash = Hash(
                        completion_hash(&ev).expect("the committed container PUTs (fail loud)"),
                    );
                    publish_tagged(3, round, &hash.0);
                    if live.is_some() {
                        publish_wire(&VhcMessage::Commitment(Commitment {
                            round,
                            payload: hash,
                            size,
                            locators: Vec::new(),
                        }));
                    }
                    continue;
                }
                // Live-mode completion routing first: the manifest fetch, a pending open's
                // corpus segments, a pending record's committed payloads, a durable payload put.
                if let Some(l) = live.as_mut() {
                    if l.manifest_op == Some(op) {
                        l.manifest_op = None;
                        // EXTERNAL INPUT: the corpus manifest is a fetched artifact. The host
                        // proved its bytes match the genesis pin; nothing proved they decode to a
                        // manifest this run can plan over, so every step of accepting it is a
                        // typed refusal rather than an abort.
                        let Some(bytes) = completion_bytes(&ev) else {
                            return OUTCOME_CORPUS_MANIFEST_INVALID;
                        };
                        let Ok(manifest) = CorpusManifest::from_canonical_bytes(&bytes) else {
                            return OUTCOME_CORPUS_MANIFEST_INVALID;
                        };
                        // Register every shard's chunk map: after registration a shard's fold
                        // identity is range-fetchable with covering-chunk verification.
                        for i in 0..manifest.shards.len() {
                            let Ok(desc) = daemon_vhc_sdk::chunk_descriptor(&manifest, i) else {
                                return OUTCOME_CORPUS_MANIFEST_INVALID;
                            };
                            if daemon_vhc_sdk::data_register_chunks(&desc) != 0 {
                                return OUTCOME_CORPUS_MANIFEST_INVALID;
                            }
                        }
                        let total_sequences = manifest.total_sequences();
                        if total_sequences == 0 {
                            return OUTCOME_CORPUS_MANIFEST_INVALID;
                        }
                        l.corpus = Some(LiveCorpus {
                            manifest,
                            total_sequences,
                        });
                        // Announce (and keep re-announcing until the first round opens).
                        l.announce();
                        daemon_vhc_sdk::set_timer(500);
                        continue;
                    }
                    if l.ckpt_puts.remove(&op) {
                        // The checkpoint document is durable; the pointer publication rides the
                        // host's put seam (spec §9), never a wire message.
                        continue;
                    }
                    // The periodic checkpoint walk's windowed moment exports (spec §9): emit into
                    // the moment families' streams; a failed export abandons the cadence slot
                    // (training continues; the next boundary fires a fresh walk).
                    let ckpt_step = l
                        .pending_ckpt
                        .as_mut()
                        .map(|walk| walk.moments.on_completion(&ev, op));
                    match ckpt_step {
                        Some(MomentStep::Progressed) => continue,
                        Some(MomentStep::Sealed) => {
                            let walk = l.pending_ckpt.take().expect("the in-flight checkpoint");
                            l.finish_checkpoint(&core, walk);
                            continue;
                        }
                        Some(MomentStep::Failed) => {
                            l.pending_ckpt = None;
                            continue;
                        }
                        Some(MomentStep::NotMine) | None => {}
                    }
                    // EXTERNAL INPUT: a corpus range's completion. The host verified the bytes
                    // against the shard's registered chunk hashes, which says nothing about the
                    // op having succeeded or the delivered span being the one that was planned —
                    // and a short span would otherwise slice out of bounds mid-stage.
                    let open_hit = l.pending_open.as_mut().and_then(|open| {
                        open.ops.remove(&op).map(|idx| match completion_bytes(&ev) {
                            Some(bytes) if bytes.len() as u64 == open.planned[idx] => {
                                open.fetched[idx] = Some(bytes);
                                Ok(open.ops.is_empty())
                            }
                            _ => Err(()),
                        })
                    });
                    if let Some(done) = open_hit {
                        let Ok(done) = done else {
                            return OUTCOME_CORPUS_RANGE_INVALID;
                        };
                        if done {
                            let open = l.pending_open.take().expect("the in-flight open");
                            let corpus = l.corpus.as_ref().expect("corpus ready");
                            stage_fetched_batches(&core, corpus, &open, vocab);
                            let ro = open.ro;
                            let round = ro.round;
                            let _out = driver.on_round_open(&ro, &mut payloads);
                            export = Some(round);
                            fence(round + 1);
                        }
                        continue;
                    }
                }
                // A record-listed committed payload landed as a host BUFFER (both modes): stash
                // the handle — never its bytes — and run the barrier once the record's whole set
                // is fetched.
                let record_round = pending_records
                    .iter_mut()
                    .find_map(|(round, p)| p.ops.remove(&op).map(|peer| (*round, peer)));
                if let Some((round, peer)) = record_round {
                    let handle = completion_handle(&ev)
                        .expect("a record-listed committed payload fetches (fail loud)");
                    payloads.map.insert((round, peer), HostStaged(handle));
                    let done = pending_records
                        .get(&round)
                        .is_some_and(|p| p.ops.is_empty());
                    if done {
                        let pending = pending_records
                            .remove(&round)
                            .expect("the completed record");
                        dispatch_record(&mut driver, &mut payloads, live.is_some(), pending);
                        // The live checkpoint fires at SEAL (in the ingest-walk `Sealed`
                        // handler), not here: `dispatch_record` only kicks off the async
                        // deferred fold, so the canonical master is not yet this round's.
                    }
                    continue;
                }
                // The quiesce walk's completions (§10.2 phase 2): emit the exported moment windows
                // into their family streams; when both seal, author + submit the typed manifest
                // and QuiesceReady.
                let quiesce_step = quiesce.as_mut().map(|w| w.moments.on_completion(&ev, op));
                match quiesce_step {
                    Some(MomentStep::Progressed) => continue,
                    Some(MomentStep::Failed) => {
                        panic!("a moment window export failed (the drain fails loud, §10.2)")
                    }
                    Some(MomentStep::Sealed) => {
                        let walk = quiesce.take().expect("quiesce present");
                        let window_size = core.borrow().window_size;
                        // The moments are sealed into their own families ([SF-6]); master + ef
                        // are already sealed and referenced by fold (`walk.master_fref`/`ef_fref`).
                        let (m_fref, v_fref) = walk.moments.families(window_size);
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
                                fref: m_fref,
                            },
                            CkptPart::Family {
                                name: ADAMW_V_FAMILY,
                                class: 1,
                                fref: v_fref,
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
                    Some(MomentStep::NotMine) | None => {}
                }
                // Anything else (an import ack, a released op) is event-loop noise.
            }
            _ => {}
        }
    }
}
