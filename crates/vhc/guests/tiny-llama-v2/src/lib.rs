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

use daemon_vhc_sdk_v2::{ModuleDecl, V2Module};

use daemon_vhc_proto::messages::{Digest, Straggle, StraggleStatus, SwarmMessage};
use daemon_vhc_proto::{from_canonical_slice, to_canonical_vec, PeerId};
use daemon_vhc_sdk::models::{TinyLlama, TinyLlamaCfg};
use daemon_vhc_sdk::prelude::*;
use daemon_vhc_sdk_rounds::{
    BarrierRound, HostStaged, PayloadSource, RoundCfg, RoundExperiment, Staged,
    StepCtx as RoundStepCtx,
};
use serde::Deserialize;

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

// -- the V2Module under `main!` (ABI §2.1/§10.1) -----------------------------------------------------

/// TinyLlama-on-`BarrierRound` as a [`V2Module`]: `main!` derives the manifest + tiered claim
/// from this declaration and emits every required export (including `da_migrate`, which answers
/// `Incompatible` until a `MigrateState` impl lands — the honest Phase-A default).
struct TinyLlamaV2 {
    driver: BarrierRound<TinyLlamaRound, HostStaged>,
    updates: FifoUpdates,
}

impl V2Module for TinyLlamaV2 {
    fn decl() -> ModuleDecl {
        ModuleDecl {
            name: "tiny-llama-v2",
            version: env!("CARGO_PKG_VERSION"),
            channels: vec![0],
            // The tiny parity model: ~1 MiB accountable state (params + queues), ~1 MiB
            // transient decode/staging scratch; device residency is host mechanism (§2.5).
            host_state_bytes: 1 << 20,
            host_scratch_bytes: 1 << 20,
            device_state_bytes: 0,
            device_scratch_bytes: 0,
        }
    }

    /// Build the model (bridge registration — legal exactly here, §2.5) + the round core.
    fn init(config: &[u8], _grants: &[u8]) -> Result<Self, u32> {
        let Ok(cfg) = from_canonical_slice::<GuestCfg>(config) else {
            return Err(16); // module-defined init failure: bad config
        };
        let mut model_cfg = Vec::new();
        if ciborium::into_writer(&cfg.model, &mut model_cfg).is_err() {
            return Err(17);
        }
        let model = TinyLlama::build(&Config::from_bytes(model_cfg));
        let round_cfg = RoundCfg {
            peer: cfg.peer,
            roster: cfg.roster,
            steps_per_round: cfg.steps_per_round,
            micro_batch: cfg.micro_batch,
            stall_rounds_max: cfg.stall_rounds_max,
        };
        Ok(Self {
            driver: BarrierRound::new(
                TinyLlamaRound {
                    model,
                    batches: RefCell::new(VecDeque::new()),
                },
                round_cfg,
            ),
            updates: FifoUpdates {
                queue: VecDeque::new(),
            },
        })
    }

    fn run(&mut self) -> u32 {
        event_loop(&mut self.driver, &mut self.updates)
    }
}

daemon_vhc_sdk_v2::main!(TinyLlamaV2);

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
fn event_loop(
    driver: &mut BarrierRound<TinyLlamaRound, HostStaged>,
    updates: &mut FifoUpdates,
) -> u32 {
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
            }
            EV_FRAME => {
                let payload = ev.bytes(4);
                let Ok(msg) = from_canonical_slice::<SwarmMessage>(&payload) else {
                    continue; // not a coordinator record — module policy is to ignore
                };
                let out = match msg {
                    SwarmMessage::RoundOpen(ro) => driver.on_round_open(&ro, updates),
                    SwarmMessage::RoundRecord(rr) => {
                        let entries = rr.inline.clone().unwrap_or_default();
                        driver.on_round_record(&rr, entries, updates)
                    }
                    _ => Vec::new(),
                };
                emit(out);
            }
            _ => {}
        }
    }
}
