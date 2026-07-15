// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! TinyLlama on `BarrierRound` under the major-2 event loop — the A2 parity guest.
//!
//! The math is the v1 guest's, verbatim: `daemon_vhc_sdk::models::TinyLlama` driving the frozen
//! `tabi@1` vocabulary (the §2.5 bridge). What moved is the CHOREOGRAPHY: the round logic runs
//! in-guest as `BarrierRound<TinyLlamaRound, HostStaged>` — the control inversion the det-digest
//! parity lane proves changed nothing about the math.
//!
//! Config (canonical CBOR map): `{"model": TinyLlamaCfg, "peer": bstr32, "roster": [bstr32…],
//! "steps_per_round": uint, "micro_batch": uint, "stall_rounds_max": uint}`.
//!
//! Event wiring: `PayloadReady{kind 1}` queues a staged batch (FIFO — the plumbing stages in
//! training order); `PayloadReady{kind 2}` queues a staged update (FIFO — staged in record-listed
//! order, the same order the record lists, so the per-peer lookup is positional);
//! `Frame(control)` decodes a `SwarmMessage` (`RoundOpen` → train + commit; `RoundRecord` →
//! barrier ingest); `Stop` → `Ok`.
//!
//! OUTBOUND SEALING GAP (Phase-A structural, recorded for the B1 brief): a bridge guest cannot
//! serialize its own update container to payload bytes (`update_bytes` is host-side; the
//! guest-visible path is Phase B's `payload_put`/`create_from`). `make_update` therefore builds
//! the container (the same math/registration side effects as v1) and returns EMPTY payload bytes;
//! the commitment's evidentiary hash at Phase A comes from the plumbing. The parity oracle covers
//! the gap transitively: the ingested state digest is a function of the same params the container
//! derived from, so identical digests ⇒ identical container inputs.

use std::cell::RefCell;
use std::collections::VecDeque;

use daemon_vhc_proto::messages::{Digest, Straggle, StraggleStatus, SwarmMessage};
use daemon_vhc_proto::{from_canonical_slice, to_canonical_vec, PeerId};
use daemon_vhc_sdk::models::{TinyLlama, TinyLlamaCfg};
use daemon_vhc_sdk::prelude::*;
use daemon_vhc_sdk_rounds::{
    BarrierRound, HostStaged, PayloadSource, RoundCfg, RoundExperiment, Staged,
    StepCtx as RoundStepCtx,
};
use serde::Deserialize;

// -- required exports beyond the experiment (ABI §2.1) --------------------------------------------

/// major 2, minor 0 (the `experiment!` macro is v1-shaped; hand-wire the v2 exports).
#[no_mangle]
pub extern "C" fn da_abi() -> u32 {
    2 << 16
}

/// The host requests guest buffers through the SDK allocator (ABI §2.4).
#[no_mangle]
pub extern "C" fn da_alloc(size: u32, align: u32) -> u32 {
    daemon_vhc_sdk::rt::da_alloc(size, align)
}

/// Paired release (ABI §2.4).
#[no_mangle]
pub extern "C" fn da_free(ptr: u32, size: u32, align: u32) {
    daemon_vhc_sdk::rt::da_free(ptr, size, align)
}

fn text(s: &str) -> ciborium::value::Value {
    ciborium::value::Value::Text(s.into())
}
fn uint_v(v: u64) -> ciborium::value::Value {
    ciborium::value::Value::Integer(v.into())
}

/// Manifest: the `control` channel + the bridge world (ABI §2.3).
#[no_mangle]
pub extern "C" fn da_manifest(_c: u32, _cl: u32) -> u64 {
    let m = ciborium::value::Value::Map(vec![
        (text("name"), text("tiny-llama-v2")),
        (text("version"), text(env!("CARGO_PKG_VERSION"))),
        (text("sdk"), text("daemon-vhc-sdk")),
        (text("abi"), uint_v(u64::from(2u32 << 16))),
        (
            text("channels"),
            ciborium::value::Value::Array(vec![uint_v(0)]),
        ),
    ]);
    let mut b = Vec::new();
    ciborium::into_writer(&m, &mut b).expect("cbor");
    daemon_vhc_sdk::rt::emit_cbor(&b)
}

/// A small honest claim (ABI §9.1) — the tiny parity model; SDK claim derivation is the macro
/// sitting.
#[no_mangle]
pub extern "C" fn da_claim(_c: u32, _cl: u32, _g: u32, _gl: u32) -> u64 {
    let tier = |d: u64, h: u64| {
        ciborium::value::Value::Map(vec![(text("device"), uint_v(d)), (text("host"), uint_v(h))])
    };
    let claim = ciborium::value::Value::Map(vec![
        (text("hard_accountable"), tier(0, 1 << 20)),
        (text("declared_peak"), tier(0, 64 << 20)),
        (text("workspace"), tier(0, 1 << 20)),
        (
            text("under_pressure"),
            ciborium::value::Value::Array(vec![uint_v(0), uint_v(1)]),
        ),
    ]);
    let mut b = Vec::new();
    ciborium::into_writer(&claim, &mut b).expect("cbor");
    daemon_vhc_sdk::rt::emit_cbor(&b)
}

// -- the round experiment over the SDK model -------------------------------------------------------

#[derive(Deserialize)]
struct GuestCfg {
    model: TinyLlamaCfg,
    /// This peer (proto `PeerId` — its serde impl is the wire form the harness authors).
    peer: PeerId,
    roster: Vec<PeerId>,
    steps_per_round: u32,
    micro_batch: u32,
    stall_rounds_max: u32,
}

/// The `RoundExperiment` adapter: the SDK model's math at exactly the v1 call points, batches
/// consumed from the host-staged FIFO.
struct TinyLlamaRound {
    model: TinyLlama,
    /// Host-staged batch tokens, FIFO in training order (`read_back` kind-1 handles).
    batches: RefCell<VecDeque<u64>>,
}

impl RoundExperiment<HostStaged> for TinyLlamaRound {
    fn train_step(&mut self, ctx: &RoundStepCtx) {
        let staging_id = self
            .batches
            .borrow_mut()
            .pop_front()
            .expect("a staged batch per train_step (plumbing stages in training order)");
        let handle = daemon_vhc_sdk_v2::read_back_uint(staging_id, 1);
        let batch = Batch::from_handle(handle);
        let step_seqs = (ctx.micro.end - ctx.micro.start) as u32;
        self.model.step(
            &batch,
            &StepCtx {
                inner_step: ctx.inner_step,
                mb_index: ctx.mb_index,
                mb_count: ctx.mb_count,
                step_seqs,
            },
        );
    }

    fn inner_update(&mut self, inner_step: u32) {
        self.model.inner_update(inner_step);
    }

    fn make_update(&mut self, round: u64) -> Vec<u8> {
        // Same math + container registration as v1; the sealed bytes stay host-side (the
        // OUTBOUND SEALING GAP in the module docs — Phase B's payload_put closes it).
        let _container = self.model.make_update(round);
        Vec::new()
    }

    fn ingest(&mut self, round: u64, staged: &Staged<HostStaged>) -> [u8; 16] {
        // Resolve each host-staged token to its upd_* index (record-listed order). The bridge
        // opens a fresh staged window per ingest slice (v1's per-entry `staged.clear()`), so the
        // indices are 0-based per round and the plain SDK `UpdatesView` reads them unchanged.
        let mut count = 0u32;
        for item in staged.items() {
            let idx = daemon_vhc_sdk_v2::read_back_uint(item.bytes.0, 2);
            debug_assert_eq!(u64::from(count), idx, "record-listed staging order");
            count += 1;
        }
        let view = UpdatesView::with_count(count);
        self.model.ingest(round, &view);
        // The agreement digest is host mechanism at this seam (the pump exports the canonical
        // state; digests publish via the plumbing) — the round core only needs a marker here.
        let mut d = [0u8; 16];
        d[..8].copy_from_slice(&round.to_le_bytes());
        d
    }
}

/// The guest's payload source: kind-2 staging tokens FIFO, positional per record entry (the
/// plumbing stages in record-listed order — single-lookup-per-entry, matching `Staged::mint`'s
/// iteration order).
struct FifoUpdates {
    queue: VecDeque<u64>,
}

impl PayloadSource<HostStaged> for FifoUpdates {
    fn payload(&mut self, _round: u64, _peer: &PeerId) -> Option<HostStaged> {
        self.queue.pop_front().map(HostStaged)
    }
}

// -- da_init / da_run --------------------------------------------------------------------------------

thread_local! {
    static STATE: RefCell<Option<(BarrierRound<TinyLlamaRound, HostStaged>, FifoUpdates)>> =
        const { RefCell::new(None) };
}

/// Build the model (bridge registration — legal exactly here, §2.5) + the round core.
///
/// # Safety
/// Called once by the host before `da_run`; `cfg_ptr` is a host-written span.
#[no_mangle]
pub unsafe extern "C" fn da_init(cfg_ptr: u32, cfg_len: u32, _g: u32, _gl: u32) -> u32 {
    let cfg_bytes = std::slice::from_raw_parts(cfg_ptr as *const u8, cfg_len as usize).to_vec();
    let Ok(cfg) = from_canonical_slice::<GuestCfg>(&cfg_bytes) else {
        return 16; // module-defined init failure: bad config
    };
    let mut model_cfg = Vec::new();
    if ciborium::into_writer(&cfg.model, &mut model_cfg).is_err() {
        return 17;
    }
    let model = TinyLlama::build(&Config::from_bytes(model_cfg));
    let round_cfg = RoundCfg {
        peer: cfg.peer,
        roster: cfg.roster,
        steps_per_round: cfg.steps_per_round,
        micro_batch: cfg.micro_batch,
        stall_rounds_max: cfg.stall_rounds_max,
    };
    let driver = BarrierRound::new(
        TinyLlamaRound {
            model,
            batches: RefCell::new(VecDeque::new()),
        },
        round_cfg,
    );
    STATE.with(|s| {
        *s.borrow_mut() = Some((
            driver,
            FifoUpdates {
                queue: VecDeque::new(),
            },
        ));
    });
    0
}

const EV_FRAME: u64 = 0;
const EV_PAYLOAD_READY: u64 = 1;
const EV_STOP: u64 = 4;
const STAGED_KIND_BATCH: u64 = 1;
const STAGED_KIND_UPDATE: u64 = 2;

/// Voice the round core's outbound actions on the control channel (§6.2): commitments, straggle
/// heartbeats, and round digests — module-authored payloads the host signs + sequences.
fn emit(out: Vec<daemon_vhc_sdk_rounds::Outbound>) {
    use daemon_vhc_sdk_rounds::Outbound;
    for o in out {
        let msg = match o {
            // Phase A: the sealed payload stays host-side (the outbound sealing gap — module
            // docs); the commitment frame still voices the round's completion.
            Outbound::Commit { commitment, .. } => SwarmMessage::Commitment(commitment),
            Outbound::RoundComplete { round, digest } | Outbound::CaughtUp { round, digest } => {
                SwarmMessage::Digest(Digest {
                    round,
                    digest: daemon_vhc_proto::StateDigest::new(digest),
                })
            }
            Outbound::Straggle { round, fetching } => SwarmMessage::Straggle(Straggle {
                round,
                status: if fetching {
                    StraggleStatus::Fetching
                } else {
                    StraggleStatus::Stalled
                },
            }),
            Outbound::Left { .. } => continue,
        };
        if let Ok(bytes) = to_canonical_vec(&msg) {
            let _ = daemon_vhc_sdk_v2::publish(0, &bytes);
        }
    }
}

/// The inverted loop: pull events, feed the round core (§3.1).
#[no_mangle]
pub extern "C" fn da_run() -> u32 {
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    loop {
        let ev = daemon_vhc_sdk_v2::next_event(&mut buf);
        match ev.tag {
            EV_STOP => return 0,
            EV_PAYLOAD_READY => {
                let staging_id = ev.uint(1);
                // meta.kind selects the queue (§4.2 payload-meta).
                let kind = match ev.items.get(3) {
                    Some(ciborium::value::Value::Map(m)) => m
                        .iter()
                        .find_map(|(k, v)| match k {
                            ciborium::value::Value::Text(t) if t == "kind" => v.as_integer(),
                            _ => None,
                        })
                        .map(|n| u64::try_from(i128::from(n)).unwrap_or(0))
                        .unwrap_or(0),
                    _ => 0,
                };
                STATE.with(|s| {
                    let mut st = s.borrow_mut();
                    let (driver, updates) = st.as_mut().expect("da_init ran");
                    match kind {
                        STAGED_KIND_BATCH => {
                            driver
                                .experiment()
                                .batches
                                .borrow_mut()
                                .push_back(staging_id);
                        }
                        STAGED_KIND_UPDATE => updates.queue.push_back(staging_id),
                        _ => {}
                    }
                });
            }
            EV_FRAME => {
                let payload = ev.bytes(4);
                let Ok(msg) = from_canonical_slice::<SwarmMessage>(&payload) else {
                    continue; // not a coordinator record — module policy is to ignore
                };
                let out = STATE.with(|s| {
                    let mut st = s.borrow_mut();
                    let (driver, updates) = st.as_mut().expect("da_init ran");
                    match msg {
                        SwarmMessage::RoundOpen(ro) => driver.on_round_open(&ro, updates),
                        SwarmMessage::RoundRecord(rr) => {
                            let entries = rr.inline.clone().unwrap_or_default();
                            driver.on_round_record(&rr, entries, updates)
                        }
                        _ => Vec::new(),
                    }
                });
                emit(out);
            }
            _ => {}
        }
    }
}
