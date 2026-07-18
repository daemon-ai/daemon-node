// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The worker input-replay verifier — **skeleton** (refactor §5 A1; ABI companion §8.7).
//!
//! A1 delivers the journal substrate + the coordinator-oracle parity now (in parallel with A0). The
//! *worker* input-replay verifier — which re-drives a wasm guest through the **host runtime** from a
//! recorded journal and asserts every outbound decision matches — needs the dual-dispatch worker and
//! the `da_run`/`next_event` event loop, so its real wiring lands after A0 merges and **completes in
//! A2** (there is no event stream to replay end-to-end until then). This module is therefore the
//! *shape*: the typed outcomes (§8.7), a [`ReplayPlan`] derived from a journal, and a [`GuestUnderReplay`]
//! seam a sim/host driver implements — plus a sim-fed test harness proving the shape holds. It drives
//! **nothing real yet** and links neither the worker binary nor the host dispatch code (A0's
//! territory).
//!
//! Fixed here (the A1 contract the A2 wiring must honor — ABI §8.7):
//! * **Input replay is bit-exact on decisions**: re-feeding recorded event frames + recorded
//!   nondeterministic import results reproduces every publish/timer/read-back/branch; the recorded
//!   results are replayed, kernels are not re-executed.
//! * **Missing referenced content** → the typed [`ReplayOutcome::MissingPayload`]
//!   (`ReplayMissingPayload`), identifying the hash + ordinal; the run is reported **incomplete,
//!   never a pass**.
//! * **Recorded terminal faults** (tag 9 kinds 1–2) → the verifier re-drives up to the recorded
//!   ordinal then **injects the recorded terminal fact** ([`ReplayOutcome::TerminalFault`]); no
//!   wall-clock mechanism is re-armed.

use daemon_vhc_proto::Hash;

use super::record::{Body, Record, SidecarRef};

/// A single recorded step to re-feed the guest during replay (ABI §8.7): an event frame, a
/// nondeterministic import result, or the recorded terminal fact.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReplayStep {
    /// A delivered event frame (tag 1) at a logical time.
    Event {
        /// The record ordinal.
        ord: u64,
        /// Logical delivery time (§6.5).
        at: u64,
        /// The exact frame bytes to deliver.
        frame: Vec<u8>,
    },
    /// A recorded read-back result (tag 2), inline or referencing a sidecar (§8.5).
    ReadBack {
        /// The record ordinal (the sidecar nonce input).
        ord: u64,
        /// The read-back source.
        src: u64,
        /// The read-back kind.
        kind: u64,
        /// The read-back status.
        status: u64,
        /// The recorded value: inline bytes, or a sidecar reference to fetch.
        value: ReadBackValue,
    },
    /// A recorded clock reading (tag 3).
    Clock {
        /// The record ordinal.
        ord: u64,
        /// The recorded `now` value.
        now: u64,
    },
    /// A recorded device-profile delivery (tag 15, Phase B): the probe's measurement is a
    /// nondeterministic input — replay feeds the recorded bytes, never a fresh probe.
    DeviceProfile {
        /// The record ordinal.
        ord: u64,
        /// The recorded profile bytes.
        profile: Vec<u8>,
    },
    /// The recorded terminal fact (tag 9) to inject as the replay outcome.
    Terminal {
        /// The record ordinal.
        ord: u64,
        /// Terminal kind (0 outcome, 1 trap, 2 forced interruption).
        kind: u64,
    },
}

/// A recorded read-back value: inline, or a sidecar to fetch (which may be missing at replay — §8.7).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadBackValue {
    /// Inline plaintext (`<= READBACK_INLINE_MAX`).
    Inline(Vec<u8>),
    /// A content-addressed encrypted sidecar to fetch (§8.5).
    Sidecar(SidecarRef),
}

/// A recorded outbound decision the guest is expected to reproduce (the oracle). Extended in A2 with
/// timers / read-back requests; publish is the Phase-A minimal subset (ABI §8.7, refactor §5 A2).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExpectedDecision {
    /// A publish (tag 4): channel + resulting seq + payload hash.
    Publish {
        /// The record ordinal.
        ord: u64,
        /// The channel.
        channel: u64,
        /// The durable seq.
        seq: u64,
        /// blake3 of the guest payload.
        hash: Hash,
    },
}

/// The full replay plan derived from a journal: the steps to re-feed, in order, and the recorded
/// outbound decisions to assert against (ABI §8.7). Built by [`ReplayPlan::from_records`].
#[derive(Clone, Debug, Default)]
pub struct ReplayPlan {
    /// The ordered steps to re-feed the guest.
    pub steps: Vec<ReplayStep>,
    /// The recorded outbound decisions (the oracle the guest's actions must match).
    pub expected: Vec<ExpectedDecision>,
}

impl ReplayPlan {
    /// Derive a replay plan from a journal's records (in ordinal order).
    #[must_use]
    pub fn from_records(records: &[Record]) -> Self {
        let mut plan = ReplayPlan::default();
        for record in records {
            match &record.body {
                Body::Event(e) => plan.steps.push(ReplayStep::Event {
                    ord: record.ord,
                    at: e.at,
                    frame: e.frame.clone(),
                }),
                Body::ReadBack(r) => {
                    let value = match (&r.value, &r.sidecar) {
                        (Some(v), None) => ReadBackValue::Inline(v.clone()),
                        (None, Some(s)) => ReadBackValue::Sidecar(s.clone()),
                        // A well-formed journal has exactly one; treat a malformed record as empty
                        // inline (the A2 driver will surface it, this skeleton stays total).
                        _ => ReadBackValue::Inline(Vec::new()),
                    };
                    plan.steps.push(ReplayStep::ReadBack {
                        ord: record.ord,
                        src: r.src,
                        kind: r.kind,
                        status: r.status,
                        value,
                    });
                }
                Body::Clock(c) => plan.steps.push(ReplayStep::Clock {
                    ord: record.ord,
                    now: c.now,
                }),
                Body::DeviceProfile(d) => plan.steps.push(ReplayStep::DeviceProfile {
                    ord: record.ord,
                    profile: d.profile.clone(),
                }),
                Body::Publish(p) => plan.expected.push(ExpectedDecision::Publish {
                    ord: record.ord,
                    channel: p.channel,
                    seq: p.seq,
                    hash: p.hash,
                }),
                Body::Terminal(t) => plan.steps.push(ReplayStep::Terminal {
                    ord: record.ord,
                    kind: t.kind,
                }),
                // Other records (headers, snapshots, instantiation, etc.) are context, not steps.
                _ => {}
            }
        }
        plan
    }
}

/// The typed result of a worker replay (ABI §8.7). `Diverged`/`MissingPayload` are never a pass.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReplayOutcome {
    /// Every recorded decision reproduced bit-exactly.
    Pass {
        /// The number of decisions verified.
        decisions: u64,
    },
    /// The guest's outbound action diverged from the recorded one (the first divergence).
    Diverged(WorkerDivergence),
    /// A referenced content-addressed payload/sidecar could not be fetched (`ReplayMissingPayload`,
    /// §8.7): the hash + the ordinal that needed it. The run is **incomplete, never a pass**.
    MissingPayload {
        /// The content address that could not be fetched.
        hash: Hash,
        /// The journal ordinal that referenced it.
        ord: u64,
    },
    /// A recorded terminal fault (tag 9 kind 1–2) was injected at its recorded ordinal (§8.7).
    TerminalFault {
        /// The ordinal at which the recorded fault was injected.
        ord: u64,
        /// Terminal kind (1 trap, 2 forced interruption).
        kind: u64,
    },
}

/// The first divergence between a recorded decision and the guest's replayed one (ABI §8.7; the
/// existing [`ReplayDivergence`](crate::replay::ReplayDivergence) shape, generalized to the worker).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerDivergence {
    /// The journal ordinal at which the guest and the record disagree.
    pub ord: u64,
    /// The recorded (oracle) decision.
    pub recorded: ExpectedDecision,
    /// What the guest produced (`None` if it produced no decision where one was recorded).
    pub replayed: Option<ExpectedDecision>,
    /// A human-readable summary.
    pub detail: String,
}

/// The seam a real replay driver implements (A2): a wasm guest re-driven through the **host runtime**
/// (never the SDK — `host/daemon-vhc-observe`), fed recorded frames + import results, yielding its
/// outbound decisions. A1 ships only the trait + a sim mock (see the tests); A2 implements it over
/// the event loop.
pub trait GuestUnderReplay {
    /// Deliver a recorded event frame and return the guest's outbound decisions for that slice.
    fn deliver_event(&mut self, ord: u64, at: u64, frame: &[u8]) -> Vec<ExpectedDecision>;
    /// Supply a recorded nondeterministic import result the guest requested (read-back/clock).
    fn supply_import(&mut self, step: &ReplayStep);
}

/// Fetch a recorded sidecar during replay (the journal's [`fetch_sidecar`](super::store::Journal::fetch_sidecar)
/// in production; a map in tests). Returns `None` when the sidecar is missing → `ReplayMissingPayload`.
pub trait PayloadSource {
    /// Fetch the plaintext behind a sidecar reference, or `None` if it cannot be fetched (§8.7).
    fn fetch(&self, sref: &SidecarRef, ord: u64) -> Option<Vec<u8>>;
}

/// Drive a [`GuestUnderReplay`] through a [`ReplayPlan`], asserting decisions match and applying the
/// §8.7 semantics (missing-payload, terminal-injection). This is the **skeleton** driver: A2 replaces
/// the mock guest with the real host-runtime event loop; the outcome contract is fixed here.
pub fn run_replay<G: GuestUnderReplay, P: PayloadSource>(
    plan: &ReplayPlan,
    guest: &mut G,
    payloads: &P,
) -> ReplayOutcome {
    let mut expected = plan.expected.iter();
    let mut verified = 0u64;

    for step in &plan.steps {
        match step {
            ReplayStep::ReadBack {
                ord,
                value: ReadBackValue::Sidecar(sref),
                ..
            } => {
                // §8.7: a missing referenced payload is a typed incomplete outcome, never a pass.
                if payloads.fetch(sref, *ord).is_none() {
                    return ReplayOutcome::MissingPayload {
                        hash: sref.hash,
                        ord: *ord,
                    };
                }
                guest.supply_import(step);
            }
            ReplayStep::ReadBack { .. }
            | ReplayStep::Clock { .. }
            | ReplayStep::DeviceProfile { .. } => {
                guest.supply_import(step);
            }
            ReplayStep::Terminal { ord, kind } => {
                // §8.7: inject the recorded terminal fact as the outcome (kinds 1–2). Kind 0
                // (outcome) is a clean completion and does not short-circuit.
                if *kind != 0 {
                    return ReplayOutcome::TerminalFault {
                        ord: *ord,
                        kind: *kind,
                    };
                }
            }
            ReplayStep::Event { ord, at, frame } => {
                let decisions = guest.deliver_event(*ord, *at, frame);
                for produced in decisions {
                    match expected.next() {
                        Some(recorded) if *recorded == produced => verified += 1,
                        Some(recorded) => {
                            return ReplayOutcome::Diverged(WorkerDivergence {
                                ord: decision_ord(&produced),
                                recorded: recorded.clone(),
                                replayed: Some(produced),
                                detail: "replayed decision differs from the recorded one".into(),
                            });
                        }
                        None => {
                            return ReplayOutcome::Diverged(WorkerDivergence {
                                ord: decision_ord(&produced),
                                recorded: produced.clone(),
                                replayed: Some(produced),
                                detail: "guest produced a decision with none recorded".into(),
                            });
                        }
                    }
                }
            }
        }
    }

    // Any recorded decisions the guest failed to produce is a divergence (missing decisions).
    if let Some(recorded) = expected.next() {
        return ReplayOutcome::Diverged(WorkerDivergence {
            ord: decision_ord(recorded),
            recorded: recorded.clone(),
            replayed: None,
            detail: "recorded decision was not reproduced by the guest".into(),
        });
    }

    ReplayOutcome::Pass {
        decisions: verified,
    }
}

fn decision_ord(d: &ExpectedDecision) -> u64 {
    match d {
        ExpectedDecision::Publish { ord, .. } => *ord,
    }
}
