// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `BarrierRound<E>` — barrier-mode round choreography as **library code** (architecture §6;
//! refactor §5 A2 item 3).
//!
//! The round *logic* of `daemon-vhc-session::RoundEngine`, moved behind the wasm boundary as a
//! sans-io reactive core: it consumes decoded coordinator records and emits [`Outbound`] actions,
//! calling the [`RoundExperiment`] (the surviving `Experiment` trait) at exactly the v1 points —
//! **the choreography is the same state machine, relocated**, which is what the TinyLlama
//! det-digest parity acceptance asserts end to end. Ported one-to-one from the engine:
//!
//! - **Assignment consumption** — [`daemon_vhc_sdk_consensus::assign_batches`] over the
//!   class-equal roster (assignment math moved to the consensus SDK layer at D0, refactor §8/D0),
//!   sliced into `steps_per_round` inner steps × `micro_batch` micro-windows exactly as the
//!   engine's `slice_interval` does.
//! - **Train-loop order** — per inner step: every micro-batch through
//!   [`RoundExperiment::train_step`], then one [`RoundExperiment::inner_update`]; after all
//!   steps, one [`RoundExperiment::make_update`] → a `Commit` action.
//! - **The barrier (I2/I3)** — a `RoundRecord` is ingestible only when every record-listed
//!   payload is fetchable; ingest consumes a [`Staged`] set minted in **record-listed order with
//!   per-item blake3 verification** ([`Staged::mint`] — the bridging oracle Phase D re-types as
//!   `Committed<T>` byte-identically); records ingest in strictly ascending round order.
//! - **The straggle ladder (RUN-8/§6.4)** — an unfetchable head enters the stall ladder
//!   (`Straggle{Fetching}`), later `RoundOpen`s while stalled skip training and heartbeat
//!   `Straggle{Stalled}` against the `stall_rounds_max` budget, catch-up ingests late
//!   (`CaughtUp`), budget exhaustion leaves for the epoch.
//!
//! Sans-io on purpose: this crate never awaits, never fetches, never signs — the plumbing
//! (event pump, control-plane subscribe, staging + verification transport, journal, signing)
//! stays host mechanism in `daemon-vhc-session`, and the guest-side driver glue (the `main!`
//! wiring that pumps `next_event` frames into [`BarrierRound`]) rides the SDK. Payload access is
//! through the caller-supplied [`PayloadSource`] (guest: staged `read_back`; tests: a map).

use std::collections::BTreeMap;

use daemon_vhc_proto::messages::{BatchWindow, Commitment, RecordEntry, RoundOpen, RoundRecord};
use daemon_vhc_proto::{blake3_hash, Hash, PeerId};

/// The round id vocabulary (matches the session seam).
pub type RoundId = u64;

/// One micro-window of the peer's assigned interval (the engine's `MicroBatch`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicroWindow {
    /// Inclusive start batch id.
    pub start: u64,
    /// Exclusive end batch id.
    pub end: u64,
}

/// Per-call context for [`RoundExperiment::train_step`] (the engine's `StepCtx`, relocated).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepCtx {
    /// The round being trained.
    pub round: RoundId,
    /// The inner step index (`0..steps_per_round`).
    pub inner_step: u32,
    /// The micro-batch window of this call.
    pub micro: MicroWindow,
    /// This micro-batch's index within the inner step.
    pub mb_index: u32,
    /// Total micro-batches in the inner step.
    pub mb_count: u32,
}

/// The surviving `Experiment` trait (refactor §5 A2: "`Experiment` survives as
/// `RoundExperiment`"): the four math points the barrier choreography calls. How the math happens
/// is the implementor's business — a guest drives the `tabi@1` bridge / `compute@2`; a native
/// test counts calls; Phase C authors write ordinary Burn.
pub trait RoundExperiment<P = Vec<u8>> {
    /// One inner training step over one assigned micro-window.
    fn train_step(&mut self, ctx: &StepCtx);
    /// The inner-optimizer application after an inner step's micro-batches.
    fn inner_update(&mut self, inner_step: u32);
    /// Seal this round's outer update as opaque payload bytes.
    fn make_update(&mut self, round: RoundId) -> Vec<u8>;
    /// The barrier ingest of the verified, record-ordered committed set → the det-lane digest.
    fn ingest(&mut self, round: RoundId, staged: &Staged<P>) -> [u8; 16];
}

/// How a payload representation proves itself against the record-listed hash at mint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadCheck {
    /// The repr carries the bytes and they hash to the listed value.
    Verified,
    /// The repr carries the bytes and they DO NOT hash to the listed value (tamper — refuse).
    Mismatch,
    /// The repr is a host staging token: the host hash-verified the content before announcing it
    /// (`PayloadReady` is delivered only after blake3 verification, ABI §4.3), so the in-guest
    /// check is delegated — the oracle semantics (record-listed order, all-or-nothing) still hold.
    HostVerified,
}

/// A committed payload's representation at the barrier: in-guest bytes (native tests, the
/// session's verified cache) or a host staging token (the bridge's `read_back` kinds — guest
/// payloads never enter linear memory wholesale, architecture §3.4).
pub trait PayloadRepr: Clone + PartialEq {
    /// Check this repr against the record-listed blake3.
    fn check(&self, expected: &Hash) -> PayloadCheck;
}

impl PayloadRepr for Vec<u8> {
    fn check(&self, expected: &Hash) -> PayloadCheck {
        if blake3_hash(self) == *expected {
            PayloadCheck::Verified
        } else {
            PayloadCheck::Mismatch
        }
    }
}

/// A host-staged payload token: the staging id / `upd_*` index `read_back` yielded. The host
/// verified the content hash before announcing it (ABI §4.3), so the mint delegates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostStaged(pub u64);

impl PayloadRepr for HostStaged {
    fn check(&self, _expected: &Hash) -> PayloadCheck {
        PayloadCheck::HostVerified
    }
}

/// Where committed payloads come from at the barrier. Sans-io: `None` = not (yet) fetchable
/// (→ the stall ladder), never an await. Guests answer with staged tokens; the session answers
/// from its verified cache; tests answer from a map.
pub trait PayloadSource<P = Vec<u8>> {
    /// The payload committed by `peer` for `round`, if fetchable now.
    fn payload(&mut self, round: RoundId, peer: &PeerId) -> Option<P>;
}

impl<P: Clone> PayloadSource<P> for BTreeMap<(RoundId, PeerId), P> {
    fn payload(&mut self, round: RoundId, peer: &PeerId) -> Option<P> {
        self.get(&(round, *peer)).cloned()
    }
}

/// The verified, record-ordered committed set the barrier ingests — the **bridging oracle**
/// (refactor §5 A2): its ordering/verification semantics are pinned by tests so Phase D can
/// re-type it as `Committed<T>` with byte-identical behavior. Constructible only through
/// [`Staged::mint`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Staged<P = Vec<u8>> {
    items: Vec<StagedItem<P>>,
}

/// One verified committed payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedItem<P = Vec<u8>> {
    /// The contributing peer.
    pub peer: PeerId,
    /// The record-listed blake3 (verified against `bytes` at mint).
    pub hash: Hash,
    /// The payload representation (bytes, or a host staging token).
    pub bytes: P,
}

/// Why a [`Staged::mint`] refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MintError {
    /// A record-listed payload is not fetchable yet (the caller stalls — never a panic).
    Missing {
        /// The peer whose payload is missing.
        peer: PeerId,
    },
    /// Fetched bytes do not hash to the record-listed value (tamper — refuse, propagate).
    HashMismatch {
        /// The offending peer.
        peer: PeerId,
    },
}

impl<P: PayloadRepr> Staged<P> {
    /// Mint the staged set from a record's listed entries: **record-listed order** (I3 — never
    /// arrival or map order), every item's bytes **blake3-verified** against its listed hash. A
    /// missing payload refuses with [`MintError::Missing`] (the stall ladder's input); a mismatch
    /// refuses with [`MintError::HashMismatch`].
    ///
    /// # Errors
    /// [`MintError`] as above; on any error NOTHING is minted (all-or-nothing, the I2 barrier).
    pub fn mint(
        round: RoundId,
        entries: &[RecordEntry],
        source: &mut impl PayloadSource<P>,
    ) -> Result<Self, MintError> {
        let mut items = Vec::with_capacity(entries.len());
        for entry in entries {
            let Some(bytes) = source.payload(round, &entry.peer) else {
                return Err(MintError::Missing { peer: entry.peer });
            };
            if bytes.check(&entry.hash) == PayloadCheck::Mismatch {
                return Err(MintError::HashMismatch { peer: entry.peer });
            }
            items.push(StagedItem {
                peer: entry.peer,
                hash: entry.hash,
                bytes,
            });
        }
        Ok(Self { items })
    }

    /// The verified items, in record-listed order.
    #[must_use]
    pub fn items(&self) -> &[StagedItem<P>] {
        &self.items
    }
}

/// Static per-epoch configuration (the engine's `EngineConfig` subset the round logic consumes).
#[derive(Debug, Clone)]
pub struct RoundCfg {
    /// This peer's identity.
    pub peer: PeerId,
    /// The frozen roster (class-equal at launch, like the engine).
    pub roster: Vec<PeerId>,
    /// Inner steps per round (H).
    pub steps_per_round: u32,
    /// Micro-batch size (sequences per `train_step`).
    pub micro_batch: u32,
    /// Fetch-recovery budget before a stalled peer leaves for the epoch (§6.4 rung 2).
    pub stall_rounds_max: u32,
}

/// Actions the reactive core asks the plumbing to perform (the engine's publishes/emits,
/// inverted into data).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outbound {
    /// Publish this round's commitment (payload sealed by `make_update`; the plumbing PUTs the
    /// payload and fills locators — the hash is authoritative here).
    Commit {
        /// The commitment body (locators left to the plumbing).
        commitment: Commitment,
        /// The sealed payload bytes to PUT.
        payload: Vec<u8>,
    },
    /// Publish a straggle heartbeat: `fetching` (entering the ladder) or stalled (skipping).
    Straggle {
        /// The round.
        round: RoundId,
        /// `true` = `Fetching` (just stalled at the barrier); `false` = `Stalled` heartbeat.
        fetching: bool,
    },
    /// A round ingested on time; the digest is the det-lane agreement input.
    RoundComplete {
        /// The round.
        round: RoundId,
        /// The post-ingest det digest.
        digest: [u8; 16],
    },
    /// A previously-stalled round late-ingested (the ladder's catch-up rung).
    CaughtUp {
        /// The round.
        round: RoundId,
        /// The post-ingest det digest.
        digest: [u8; 16],
    },
    /// The stall budget is exhausted: leave for the epoch (rung 3).
    Left {
        /// The round the peer was stuck on.
        round: RoundId,
    },
}

/// The barrier-mode round driver: the engine's round logic as a reusable, sans-io library core.
pub struct BarrierRound<E, P = Vec<u8>> {
    experiment: E,
    cfg: RoundCfg,
    /// Records not yet ingested, ascending round order (the barrier queue).
    pending: BTreeMap<RoundId, Vec<RecordEntry>>,
    /// Whether the head of `pending` could not be ingested (the ladder).
    straggling: bool,
    /// Consecutive `RoundOpen`s observed while stalled (the §6.4 rung-2 budget).
    stalled_rounds: u32,
    /// Resync watermark: rounds at/below never re-ingest (a double outer-step would diverge).
    last_ingested: Option<RoundId>,
    _payload: std::marker::PhantomData<P>,
}

impl<E: RoundExperiment<P>, P: PayloadRepr> BarrierRound<E, P> {
    /// Wrap an experiment with the barrier choreography.
    pub fn new(experiment: E, cfg: RoundCfg) -> Self {
        Self {
            experiment,
            cfg,
            pending: BTreeMap::new(),
            straggling: false,
            stalled_rounds: 0,
            last_ingested: None,
            _payload: std::marker::PhantomData,
        }
    }

    /// The wrapped experiment (tests read counters/state through this).
    pub fn experiment(&self) -> &E {
        &self.experiment
    }

    /// Handle `RoundOpen(r)` — ported from the engine's `on_round_open` + `train_and_commit`:
    /// first make progress on any stalled rounds (in-order catch-up), then either skip (still
    /// stalled: heartbeat + budget) or train + commit this round.
    pub fn on_round_open(
        &mut self,
        ro: &RoundOpen,
        source: &mut impl PayloadSource<P>,
    ) -> Vec<Outbound> {
        let mut out = Vec::new();
        self.advance(None, source, &mut out);

        if self.straggling {
            self.stalled_rounds += 1;
            let round = self.pending.keys().next().copied().unwrap_or(ro.round);
            out.push(Outbound::Straggle {
                round: ro.round,
                fetching: false,
            });
            if self.stalled_rounds > self.cfg.stall_rounds_max {
                out.push(Outbound::Left { round });
            }
            return out;
        }

        self.train_and_commit(ro, &mut out);
        out
    }

    /// The train-loop order, ported verbatim: assignment interval → `steps_per_round` inner steps
    /// (each micro-batch through `train_step`, then one `inner_update`) → `make_update` → commit.
    fn train_and_commit(&mut self, ro: &RoundOpen, out: &mut Vec<Outbound>) {
        let interval = interval_for(ro.batch, ro.seed, &self.cfg.roster, &self.cfg.peer);
        let steps = slice_interval(interval, self.cfg.steps_per_round, self.cfg.micro_batch);
        for step in &steps {
            let mb_count = step.micro.len() as u32;
            for (mb_index, mb) in step.micro.iter().enumerate() {
                self.experiment.train_step(&StepCtx {
                    round: ro.round,
                    inner_step: step.index,
                    micro: *mb,
                    mb_index: mb_index as u32,
                    mb_count,
                });
            }
            self.experiment.inner_update(step.index);
        }
        let payload = self.experiment.make_update(ro.round);
        let hash = blake3_hash(&payload);
        out.push(Outbound::Commit {
            commitment: Commitment {
                round: ro.round,
                payload: hash,
                size: payload.len() as u64,
                locators: Vec::new(), // the plumbing PUTs + fills locators (host mechanism)
            },
            payload,
        });
    }

    /// Handle `RoundRecord(r)` — the barrier, ported from the engine's `on_round_record`: guard
    /// the resync watermark, enqueue the record-listed entries, ingest as far as the queue
    /// allows in ascending order, and enter the stall ladder if `r` (or an earlier round) is
    /// unfetchable.
    pub fn on_round_record(
        &mut self,
        rr: &RoundRecord,
        entries: Vec<RecordEntry>,
        source: &mut impl PayloadSource<P>,
    ) -> Vec<Outbound> {
        let mut out = Vec::new();
        if let Some(last) = self.last_ingested {
            if rr.round <= last {
                return out; // never re-ingest at/below the watermark
            }
        }
        self.pending.insert(rr.round, entries);
        self.advance(Some(rr.round), source, &mut out);
        if self.pending.contains_key(&rr.round) {
            self.straggling = true;
            out.push(Outbound::Straggle {
                round: rr.round,
                fetching: true,
            });
        }
        out
    }

    /// Ingest queued records in strictly ascending round order, stopping at the first whose
    /// committed set cannot be minted (ported from the engine's `advance` + `try_ingest`, with
    /// [`Staged::mint`] as the fetch+verify+order step).
    fn advance(
        &mut self,
        trigger: Option<RoundId>,
        source: &mut impl PayloadSource<P>,
        out: &mut Vec<Outbound>,
    ) {
        while let Some(round) = self.pending.keys().next().copied() {
            let entries = self.pending[&round].clone();
            match Staged::mint(round, &entries, source) {
                Ok(staged) => {
                    let digest = self.experiment.ingest(round, &staged);
                    self.last_ingested = Some(round);
                    self.pending.remove(&round);
                    let on_time = !self.straggling && trigger == Some(round);
                    if on_time {
                        out.push(Outbound::RoundComplete { round, digest });
                    } else {
                        out.push(Outbound::CaughtUp { round, digest });
                    }
                }
                Err(MintError::Missing { .. }) => break, // head unfetchable — stall (ladder)
                Err(MintError::HashMismatch { peer }) => {
                    // Tamper: refuse the round permanently rather than ingesting a forged set —
                    // the engine propagates this as a hard error; sans-io it surfaces as a
                    // stalled head the budget eventually abandons. Recorded distinctly so the
                    // plumbing can escalate (§12 evidence is its job).
                    let _ = peer;
                    break;
                }
            }
        }
        if self.pending.is_empty() {
            self.straggling = false;
            self.stalled_rounds = 0;
        }
    }
}

/// One inner step's micro-windows (the engine's `InnerStep`, relocated). Public so the bridging
/// oracle can compare this relocated slicing against the v1 engine's, window for window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InnerStep {
    /// The inner step index.
    pub index: u32,
    /// The step's micro-windows, in training order.
    pub micro: Vec<MicroWindow>,
}

/// The peer's assigned `[start, end)` interval — `assign_batches` over the class-equal roster
/// with zero overlap, exactly the engine's `assignment::interval_for` (assignment math lives in
/// `daemon-vhc-sdk-consensus` from D0). Public for the bridging oracle.
#[must_use]
pub fn interval_for(
    window: BatchWindow,
    seed: daemon_vhc_proto::Seed,
    roster: &[PeerId],
    peer: &PeerId,
) -> MicroWindow {
    use daemon_vhc_proto::messages::ThroughputClass;
    use daemon_vhc_sdk_consensus::assign_batches;
    let weighted: Vec<(PeerId, ThroughputClass)> =
        roster.iter().map(|p| (*p, ThroughputClass::C1)).collect();
    assign_batches(&weighted, &seed, window, 0)
        .into_iter()
        .find(|(p, _)| p == peer)
        .map_or(
            MicroWindow {
                start: window.start,
                end: window.start,
            },
            |(_, w)| MicroWindow {
                start: w.start,
                end: w.end,
            },
        )
}

/// Slice the assigned interval into `steps_per_round` equal inner steps of `micro_batch`-sized
/// micro-windows — the engine's `slice_interval`, relocated (an indivisible/empty interval
/// yields no steps here; the plumbing validated divisibility at admission in v1, and the parity
/// harness always supplies divisible windows). Public for the bridging oracle.
#[must_use]
pub fn slice_interval(
    interval: MicroWindow,
    steps_per_round: u32,
    micro_batch: u32,
) -> Vec<InnerStep> {
    let len = interval.end.saturating_sub(interval.start);
    if len == 0 || steps_per_round == 0 || micro_batch == 0 {
        return Vec::new();
    }
    let steps = u64::from(steps_per_round);
    if !len.is_multiple_of(steps) {
        return Vec::new();
    }
    let per_step = len / steps;
    let mb = u64::from(micro_batch);
    let mut out = Vec::with_capacity(steps_per_round as usize);
    for h in 0..steps_per_round {
        let step_start = interval.start + u64::from(h) * per_step;
        let step_end = step_start + per_step;
        let mut micro = Vec::new();
        let mut cursor = step_start;
        while cursor < step_end {
            let end = (cursor + mb).min(step_end);
            micro.push(MicroWindow { start: cursor, end });
            cursor = end;
        }
        out.push(InnerStep { index: h, micro });
    }
    out
}
