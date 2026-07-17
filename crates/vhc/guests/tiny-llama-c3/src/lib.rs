// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `tiny-llama-c3` — the **re-authored** reference worker module (Phase C, track C3; refactor §7
//! "models leave the SDK": `models::TinyLlama` becomes a real Burn model in the guest).
//!
//! The model ([`model::C3Llama`]) is ordinary Burn over `Autodiff<HostBackend>`: every tensor op
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
//! deferred-voice shape as tiny-llama-v2's held commit.

mod model;

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;

use daemon_vhc_proto::messages::{RecordEntry, SwarmMessage};
use daemon_vhc_proto::{
    blake3_hash, digest_state, from_canonical_slice, to_canonical_vec, Hash, PeerId, Seed,
};
use daemon_vhc_sdk_compute::{export_tensor, fence, AutodiffHostBackend, HostBackend};
use daemon_vhc_sdk_profiles::{encode_payload, IngestParam, ParamView, SparseLoco, SparseLocoCfg};
use daemon_vhc_sdk_rounds::{
    BarrierRound, Committed, PayloadSource, RoundCfg, RoundExperiment, StepCtx as RoundStepCtx,
};
use daemon_vhc_sdk_v2::{
    build_manifest, MigrationDescriptor, ModuleDecl, OwnedSection, SectionReader, V2Module,
};
use serde::Deserialize;

use model::{C3Llama, ModelCfg};

const EV_FRAME: u64 = 0;
const EV_PAYLOAD_READY: u64 = 1;
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
    /// The canonical flat init (concatenated params, registration order) — matched init.
    init: Vec<f32>,
}

/// One staged batch: `(sequences, seq_len, tokens)` in training order.
type BatchItem = (u32, u32, Vec<u32>);

/// The shared model/profile/det-lane state both the driver-called experiment and the event loop
/// mutate (wasm is single-threaded; `Rc<RefCell>` is the natural share).
struct Core {
    model: C3Llama<AutodiffHostBackend>,
    profile: SparseLoco,
    /// Canonical det-lane masters (guest-side, registration order).
    master: Vec<Vec<f32>>,
    /// The round bases θ⁽ᵗ⁾ (post-ingest master snapshots).
    round_base: Vec<Vec<f32>>,
    /// Host-staged batches, FIFO in training order.
    batches: VecDeque<BatchItem>,
    /// Accumulated micro-batch gradients of the current inner step (summed in arrival order,
    /// the v1 grad-buffer accumulation).
    pending_grads: Option<Vec<burn::tensor::Tensor<HostBackend, 1>>>,
    /// Fence-id mint for the per-step depth-reset fences.
    next_step_fence: u64,
}

impl Core {
    /// The post-ingest det digest — the exact v1 `WasmBackend::digest_of` formula over the
    /// canonical state (masters, then the profile's replicated det state), computed in-guest.
    fn digest_of(&self, round: u64) -> [u8; 16] {
        let mut state = Vec::new();
        for m in &self.master {
            for v in m {
                state.extend_from_slice(&v.to_le_bytes());
            }
        }
        for r in self.profile.replicated_state() {
            for v in r {
                state.extend_from_slice(&v.to_le_bytes());
            }
        }
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&round.to_le_bytes());
        let d = digest_state(&Seed(seed), 64, u32::MAX, &state);
        *d.as_bytes()
    }
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

    fn ingest(&mut self, round: u64, committed: &Committed<Vec<u8>>) -> [u8; 16] {
        let mut core = self.core.borrow_mut();
        let payloads: Vec<_> = committed
            .items()
            .iter()
            .map(|it| {
                daemon_vhc_sdk_profiles::decode_payload(&it.bytes)
                    .expect("committed payload decodes (hash-verified at mint)")
            })
            .collect();
        // Det-lane ingest over the guest-held canonical state (zero host support).
        {
            let Core {
                profile,
                master,
                round_base,
                ..
            } = &mut *core;
            let mut params: Vec<IngestParam<'_>> = master
                .iter_mut()
                .zip(round_base.iter())
                .map(|(m, b)| IngestParam {
                    master: m,
                    round_base: b,
                })
                .collect();
            profile
                .ingest(&mut params, &payloads)
                .expect("det ingest over verified committed payloads");
        }
        // The post-ingest master becomes the working weights and the next round base (the v1
        // ingest epilogue, guest-owned here).
        let flat = core.master.clone();
        core.model.set_params_from_flat(&flat);
        core.round_base = flat;
        core.digest_of(round)
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

// -- the V2Module under `main!` -------------------------------------------------------------------

/// Flat state a `da_migrate` restore carries into `run` (split by the model layout there).
struct Restored {
    master: Vec<f32>,
    ef: Vec<f32>,
    adamw_m: Vec<f32>,
    adamw_v: Vec<f32>,
}

struct TinyLlamaC3 {
    cfg_bytes: Vec<u8>,
    restored: Option<Restored>,
}

impl V2Module for TinyLlamaC3 {
    fn decl() -> ModuleDecl {
        ModuleDecl {
            name: "tiny-llama-c3",
            version: env!("CARGO_PKG_VERSION"),
            // compute@2 imports force the Phase-C minor (ABI §1.3 step 5).
            abi_minor: 2,
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

    /// The §10.2 consuming protocol: read the `master`/`ef`/`adamw_m`/`adamw_v` flat f32-le
    /// sections staged by the old instance's snapshot; `run` splits them by the model layout and
    /// rebuilds from them.
    fn migrate(&mut self, descriptor: &MigrationDescriptor, reader: &mut dyn SectionReader) -> u32 {
        let cfg: GuestCfg =
            from_canonical_slice(&self.cfg_bytes).expect("config validated at init");
        let total: usize = cfg.model.param_numels().iter().sum();
        let mut master: Option<Vec<f32>> = None;
        let mut ef: Option<Vec<f32>> = None;
        let mut adamw_m: Option<Vec<f32>> = None;
        let mut adamw_v: Option<Vec<f32>> = None;
        for binding in &descriptor.sections {
            let bytes = reader.read(binding.staging_id);
            let vals: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            if vals.len() != total {
                return MIGRATE_INCOMPATIBLE_SECTIONS;
            }
            match binding.name.as_str() {
                "master" => master = Some(vals),
                "ef" => ef = Some(vals),
                "adamw_m" => adamw_m = Some(vals),
                "adamw_v" => adamw_v = Some(vals),
                _ => return MIGRATE_INCOMPATIBLE_SECTIONS,
            }
        }
        let (Some(master), Some(ef), Some(adamw_m), Some(adamw_v)) = (master, ef, adamw_m, adamw_v)
        else {
            return MIGRATE_INCOMPATIBLE_SECTIONS;
        };
        self.restored = Some(Restored {
            master,
            ef,
            adamw_m,
            adamw_v,
        });
        0
    }
}

daemon_vhc_sdk_v2::main!(TinyLlamaC3);

/// Split the flat init into per-param canonical vectors.
fn split_flat(flat: &[f32], numels: &[usize]) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(numels.len());
    let mut off = 0;
    for &n in numels {
        out.push(flat[off..off + n].to_vec());
        off += n;
    }
    assert_eq!(off, flat.len(), "init length matches the canonical layout");
    out
}

/// Publish `[tag, round, bytes]` on the control channel.
fn publish_tagged(tag: u64, round: u64, bytes: &[u8]) {
    let v = ciborium::value::Value::Array(vec![
        ciborium::value::Value::from(tag),
        ciborium::value::Value::from(round),
        ciborium::value::Value::Bytes(bytes.to_vec()),
    ]);
    if let Ok(payload) = to_canonical_vec(&v) {
        let _ = daemon_vhc_sdk_v2::publish(0, &payload);
    }
}

/// The per-round export walk state: op → param index, collected values, the round in flight.
struct ExportState {
    round: u64,
    collected: Vec<Option<Vec<f32>>>,
    ops: BTreeMap<u64, usize>,
}

/// The quiesce snapshot's export walk: op → moment index (all `m`, then all `v`).
struct QuiesceWalk {
    collected: Vec<Option<Vec<f32>>>,
    ops: BTreeMap<u64, usize>,
}

/// Decode one export completion's payload: `[status, handle]` → the tensor's f32 vec (None on a
/// nonzero status or an undecodable payload).
fn completion_tensor(ev: &daemon_vhc_sdk_v2::Event) -> Option<Vec<f32>> {
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
    let bytes = daemon_vhc_sdk_v2::read_buffer(handle);
    daemon_vhc_sdk_v2::buffer_release(handle);
    let data = daemon_vhc_sdk_compute::decode_tensor_data(&bytes);
    data.to_vec::<f32>().ok()
}

/// Flatten per-param canonical vectors to f32-le bytes (the snapshot section encoding).
fn flat_le(params: &[Vec<f32>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(params.iter().map(Vec::len).sum::<usize>() * 4);
    for p in params {
        for v in p {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

#[allow(clippy::too_many_lines)]
fn run_module(cfg: GuestCfg, restored: Option<Restored>) -> u32 {
    let numels = cfg.model.param_numels();
    // A restored instance rebuilds from the snapshot's masters + error feedback + AdamW moments
    // (post-ingest state: master IS the round base); a fresh instance starts from the config's
    // matched init with zeroed local state.
    let (init, restored_local) = match restored {
        Some(r) => (
            split_flat(&r.master, &numels),
            Some((
                split_flat(&r.ef, &numels),
                split_flat(&r.adamw_m, &numels),
                split_flat(&r.adamw_v, &numels),
            )),
        ),
        None => (split_flat(&cfg.init, &numels), None),
    };
    let mut profile = SparseLoco::new(cfg.profile.clone(), &numels);
    let device = daemon_vhc_sdk_compute::device();
    let mut model = C3Llama::<AutodiffHostBackend>::from_flat(cfg.model.clone(), device, &init);
    if let Some((ef, m, v)) = restored_local {
        profile
            .restore_ef(ef)
            .expect("snapshot ef matches the model layout (validated at da_migrate)");
        model.set_moments_from_flat(&m, &v);
    }
    let core = Rc::new(RefCell::new(Core {
        model,
        profile,
        master: init.clone(),
        round_base: init,
        batches: VecDeque::new(),
        pending_grads: None,
        next_step_fence: 0,
    }));
    let mut driver = BarrierRound::new(
        C3Round { core: core.clone() },
        RoundCfg {
            peer: cfg.peer,
            roster: cfg.roster.clone(),
            steps_per_round: cfg.steps_per_round,
            micro_batch: cfg.micro_batch,
            stall_rounds_max: cfg.stall_rounds_max,
        },
    );
    let mut payloads = PayloadMap {
        map: BTreeMap::new(),
    };
    let mut export: Option<ExportState> = None;
    let mut quiesce: Option<QuiesceWalk> = None;

    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    loop {
        let ev = daemon_vhc_sdk_v2::next_event(&mut buf);
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
                let tensors = core.borrow().model.moment_tensors();
                let mut ops = BTreeMap::new();
                let n = tensors.len();
                for (i, t) in tensors.into_iter().enumerate() {
                    ops.insert(export_tensor(t), i);
                }
                quiesce = Some(QuiesceWalk {
                    collected: vec![None; n],
                    ops,
                });
            }
            EV_PAYLOAD_READY => {
                // All harness staging is kind-0 bytes; the wrapper tag routes it.
                let staging_id = ev.uint(1);
                let bytes = daemon_vhc_sdk_v2::read_back_bytes(staging_id, 0);
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
            EV_FRAME => {
                let payload = ev.bytes(4);
                let Ok(msg) = from_canonical_slice::<SwarmMessage>(&payload) else {
                    continue;
                };
                match msg {
                    SwarmMessage::RoundOpen(ro) => {
                        let round = ro.round;
                        // Train + (dropped) commit; then the export walk for this round's θ.
                        let _out = driver.on_round_open(&ro, &mut payloads);
                        export = Some(ExportState {
                            round,
                            collected: vec![None; numels.len()],
                            ops: BTreeMap::new(),
                        });
                        fence(round + 1); // the round-final fence the walk waits for
                    }
                    SwarmMessage::RoundRecord(rr) => {
                        let entries: Vec<RecordEntry> = rr.inline.clone().unwrap_or_default();
                        let out = driver.on_round_record(&rr, entries, &mut payloads);
                        for o in out {
                            if let daemon_vhc_sdk_rounds::Outbound::RoundComplete {
                                round,
                                digest,
                            }
                            | daemon_vhc_sdk_rounds::Outbound::CaughtUp { round, digest } = o
                            {
                                publish_tagged(4, round, &digest);
                            }
                        }
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
                // The quiesce walk's completions (§10.2 phase 2): collect the exported moments;
                // when the last lands, author + submit the typed manifest and QuiesceReady.
                if let Some(walk) = quiesce.as_mut() {
                    if let Some(idx) = walk.ops.remove(&op) {
                        walk.collected[idx] = Some(
                            completion_tensor(&ev)
                                .expect("moment export completes (quiesce fails loud)"),
                        );
                        if walk.ops.is_empty() && walk.collected.iter().all(Option::is_some) {
                            let moments: Vec<Vec<f32>> = walk
                                .collected
                                .iter_mut()
                                .map(|c| c.take().expect("collected"))
                                .collect();
                            let n = moments.len() / 2;
                            let (master_le, ef_le) = {
                                let c = core.borrow();
                                (flat_le(&c.master), flat_le(c.profile.ef_state()))
                            };
                            // `module` is zeroed — a module cannot hash its own bytes; the host
                            // verifies sections by content hash (the drill-pair convention).
                            let sections = [
                                OwnedSection {
                                    name: "master".to_string(),
                                    schema: 1,
                                    class: 0, // consensus-canonical (the digest-covered masters)
                                    bytes: master_le,
                                },
                                OwnedSection {
                                    name: "ef".to_string(),
                                    schema: 1,
                                    class: 1, // replica-local; continuity-required
                                    bytes: ef_le,
                                },
                                OwnedSection {
                                    name: "adamw_m".to_string(),
                                    schema: 1,
                                    class: 1,
                                    bytes: flat_le(&moments[..n]),
                                },
                                OwnedSection {
                                    name: "adamw_v".to_string(),
                                    schema: 1,
                                    class: 1,
                                    bytes: flat_le(&moments[n..]),
                                },
                            ];
                            for s in &sections {
                                let _staging_id = daemon_vhc_sdk_v2::stage_state(&s.bytes);
                            }
                            let manifest = build_manifest(Hash([0u8; 32]), 1, &sections);
                            let manifest_bytes =
                                to_canonical_vec(&manifest).expect("state-manifest cbor");
                            let status = daemon_vhc_sdk_v2::snapshot_state(&manifest_bytes);
                            assert_eq!(status, 0, "snapshot_state rejected the trainer manifest");
                            // Same deliberate-leak shutdown discipline as Stop (§7.3).
                            std::mem::forget(driver);
                            std::mem::forget(core);
                            return OUTCOME_QUIESCE_READY;
                        }
                        continue;
                    }
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
                let bytes = daemon_vhc_sdk_v2::read_buffer(handle);
                daemon_vhc_sdk_v2::buffer_release(handle);
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

                    let mut theta_le = Vec::new();
                    for t in &theta {
                        for v in t {
                            theta_le.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    publish_tagged(2, round, &theta_le);

                    let mut core_mut = core.borrow_mut();
                    let Core {
                        profile,
                        round_base,
                        ..
                    } = &mut *core_mut;
                    let views: Vec<ParamView<'_>> = theta
                        .iter()
                        .zip(round_base.iter())
                        .map(|(t, b)| ParamView {
                            theta: t,
                            round_base: b,
                        })
                        .collect();
                    let sections = profile.make_update(&views);
                    let payload = encode_payload(&sections);
                    // Externalize the sealed committed container (B1: the guest authors its own
                    // payload) — put BEFORE the tag-3 voice so the embedder that observes the
                    // commitment can pair it with the serviced bytes.
                    let buf = daemon_vhc_sdk_v2::create_from(&payload);
                    let _op = daemon_vhc_sdk_v2::payload_put(buf);
                    daemon_vhc_sdk_v2::buffer_release(buf);
                    publish_tagged(3, round, &blake3_hash(&payload).0);
                }
            }
            _ => {}
        }
    }
}
