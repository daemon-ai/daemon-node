// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The deterministic discrete-event runtime (architecture §6): drives N native [`SimModule`]s over
//! the virtual worlds, with the Phase-A closed capability subset exposed through [`SimCtx`].
//!
//! A module reacts to events ([`SimModule::on_event`]) instead of pulling them from a blocking
//! `next_event`; this callback shape is the single-threaded, deterministic native form of the same
//! inverted loop (the wasm/host boundary needs the parked-thread pull; a native discrete-event sim
//! does not). Capability semantics match the host: `publish` stamps a durable channel-scoped seq
//! (§12.2), `set_timer` mints a never-reused per-instance id (§6.3), `now` is slice-constant and
//! monotone (§6.5). Because every draw is a deterministic function of the seed and the event order,
//! running the identical setup twice yields a byte-identical [`RunTranscript`] — the SDK-side
//! analogue of the host's §8.7 input replay.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use crate::clock::{Timers, VirtualClock};
use crate::net::{Trace, VirtualNet};

/// The Phase-A event set (ABI §4.2), virtualized. `Fence`/`Completion`/`PayloadReady`/`Budget` are
/// reserved for later phases (they arrive with the compute/buffer worlds), so the native subset is
/// the A2 closed one plus the terminal `Stop`/`Quiesce`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimEvent {
    /// An authoritative/gossip frame delivered on a channel (ABI §4.2 `Frame`).
    Frame {
        /// The channel the frame arrived on (ABI §6.2).
        channel: u32,
        /// The sender's durable channel-scoped sequence number (§12.2).
        seq: u64,
        /// The publishing peer index.
        sender: usize,
        /// The opaque payload bytes (module policy decodes them).
        payload: Vec<u8>,
    },
    /// A one-shot timer fired (ABI §4.2 `Timer`), carrying its logical fire time.
    Timer {
        /// The timer id returned by [`SimCtx::set_timer`].
        id: u64,
        /// The logical time the timer fired (ABI §6.3).
        fired_at: u64,
    },
    /// The run is ending (ABI §4.4 `Stop`); no imports are legal after it is handled.
    Stop {
        /// A `STOP_REASON_*` code (ABI §4.4).
        reason: u64,
    },
    /// A quiesce drain has begun (ABI §4.4 `Quiesce`) — reserved; delivered by upgrade drills.
    Quiesce {
        /// A `QUIESCE_REASON_*` code (ABI §4.4).
        reason: u64,
    },
}

/// A native module driven by the simulator — the SDK-side authoring shape (the wasm blob authors
/// the same algorithm against `daemon-vhc-sdk`'s raw event loop instead).
pub trait SimModule {
    /// Called once at logical time 0 before any event (the `da_run` prologue, ABI §3.1): arm the
    /// first timer, subscribe, publish an opening frame.
    fn init(&mut self, ctx: &mut SimCtx);

    /// Handle one delivered event (one slice of the inverted loop, ABI §3.1).
    fn on_event(&mut self, ctx: &mut SimCtx, ev: &SimEvent);
}

/// A published frame recorded in the [`RunTranscript`] (the decision transcript replay compares).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedFrame {
    /// The publishing peer index.
    pub peer: usize,
    /// The channel (ABI §6.2).
    pub channel: u32,
    /// The durable channel-scoped seq the peer stamped (§12.2).
    pub seq: u64,
    /// blake3 of the payload bytes (content address; bulk bytes are not copied into the transcript).
    pub payload_hash: [u8; 32],
    /// The payload length in bytes.
    pub payload_len: usize,
}

/// The deterministic record of a run: every publish (the module decisions), metrics, and the count
/// of delivered events. Two runs of the identical setup produce equal transcripts.
#[derive(Debug, Clone, Default)]
pub struct RunTranscript {
    /// Every published frame, in scheduling order.
    pub publishes: Vec<PublishedFrame>,
    /// Emitted metrics `(peer, name, value)` — egress, journaled here for inspection.
    pub metrics: Vec<(usize, String, f64)>,
    /// Total events delivered to modules (timers + frames; excludes suppressed/dropped).
    pub events_delivered: u64,
}

/// Bounds on a run so a wedged or non-terminating module cannot hang the gate.
#[derive(Debug, Clone, Copy)]
pub struct RunLimits {
    /// Stop after this many delivered events (a hard safety ceiling).
    pub max_events: u64,
    /// Stop once the logical clock reaches this time (`u64::MAX` = no horizon).
    pub horizon_ms: u64,
    /// Deliver a terminal `Stop{RUN_COMPLETE}` to every module when the run ends.
    pub deliver_stop: bool,
}

impl Default for RunLimits {
    fn default() -> Self {
        Self {
            max_events: 100_000,
            horizon_ms: u64::MAX,
            deliver_stop: true,
        }
    }
}

/// Per-peer durable runtime state that persists across event slices: the timer table (§6.3) and
/// the per-channel sequence counters (§12.2).
#[derive(Debug, Clone, Default)]
struct PeerRuntime {
    timers: Timers,
    seq_by_channel: BTreeMap<u32, u64>,
}

impl PeerRuntime {
    fn new() -> Self {
        Self {
            timers: Timers::new(),
            seq_by_channel: BTreeMap::new(),
        }
    }

    /// Allocate the next durable seq on `channel` (starts at 0, monotone, never reused — §12.2).
    fn next_seq(&mut self, channel: u32) -> u64 {
        let s = self.seq_by_channel.entry(channel).or_insert(0);
        let seq = *s;
        *s += 1;
        seq
    }
}

/// The capability handle a module is given for one event slice (or `init`). It exposes the Phase-A
/// closed subset natively and buffers the effects the simulator applies after the slice returns.
pub struct SimCtx<'a> {
    now: u64,
    peer: usize,
    rt: &'a mut PeerRuntime,
    published: Vec<(u32, u64, Vec<u8>)>,
    armed: Vec<(u64, u64)>,
    metrics: Vec<(String, f64)>,
    leave: Option<u32>,
}

impl<'a> SimCtx<'a> {
    fn new(now: u64, peer: usize, rt: &'a mut PeerRuntime) -> Self {
        Self {
            now,
            peer,
            rt,
            published: Vec::new(),
            armed: Vec::new(),
            metrics: Vec::new(),
            leave: None,
        }
    }

    /// The slice-constant logical time (ABI §6.5 `now`).
    #[must_use]
    pub fn now(&self) -> u64 {
        self.now
    }

    /// This module's peer index within the run.
    #[must_use]
    pub fn peer(&self) -> usize {
        self.peer
    }

    /// Publish opaque payload bytes on `channel` (ABI §6.2); returns the stamped durable
    /// channel-scoped seq (§12.2). Routing to subscribers (with trace latency/loss) happens after
    /// the slice returns, exactly as the host commits the frame before transmitting.
    pub fn publish(&mut self, channel: u32, payload: &[u8]) -> u64 {
        let seq = self.rt.next_seq(channel);
        self.published.push((channel, seq, payload.to_vec()));
        seq
    }

    /// Arm a one-shot timer `delay_ms` in the future (ABI §6.3); returns its never-reused id.
    pub fn set_timer(&mut self, delay_ms: u64) -> u64 {
        let id = self.rt.timers.arm();
        self.armed.push((id, delay_ms));
        id
    }

    /// Cancel a timer (ABI §6.3): `CANCEL_CANCELLED` if it had not fired (its event is suppressed),
    /// else `CANCEL_ALREADY_FIRED_OR_UNKNOWN`.
    pub fn cancel_timer(&mut self, id: u64) -> u32 {
        self.rt.timers.cancel(id)
    }

    /// Emit an advisory metric (ABI §6.5 `emit_metric`) — egress; recorded in the transcript.
    pub fn emit_metric(&mut self, name: &str, value: f64) {
        if value.is_finite() {
            self.metrics.push((name.to_string(), value));
        }
    }

    /// Request to leave the run with `outcome` (the native analogue of returning from `da_run`).
    pub fn leave(&mut self, outcome: u32) {
        self.leave = Some(outcome);
    }
}

/// The buffered effects of one slice, applied by the simulator after the module returns.
struct Effects {
    published: Vec<(u32, u64, Vec<u8>)>,
    armed: Vec<(u64, u64)>,
    metrics: Vec<(String, f64)>,
    leave: Option<u32>,
}

impl SimCtx<'_> {
    fn into_effects(self) -> Effects {
        Effects {
            published: self.published,
            armed: self.armed,
            metrics: self.metrics,
            leave: self.leave,
        }
    }
}

/// What kind of event a scheduled entry delivers.
#[derive(Debug, Clone)]
enum PendingKind {
    Timer {
        id: u64,
    },
    Frame {
        channel: u32,
        seq: u64,
        sender: usize,
        payload: Vec<u8>,
    },
}

/// A scheduled delivery, ordered by `(at, order)` — `order` a global monotone tiebreak so the queue
/// is a total order and thus fully deterministic.
struct Scheduled {
    at: u64,
    order: u64,
    target: usize,
    kind: PendingKind,
}

impl PartialEq for Scheduled {
    fn eq(&self, other: &Self) -> bool {
        self.at == other.at && self.order == other.order
    }
}
impl Eq for Scheduled {}
impl Ord for Scheduled {
    fn cmp(&self, other: &Self) -> Ordering {
        // Natural order: earlier `at` then lower `order` is "smaller". `BinaryHeap` is a max-heap,
        // so the driver wraps entries in `Reverse` to pop the smallest (earliest) first.
        self.at
            .cmp(&other.at)
            .then_with(|| self.order.cmp(&other.order))
    }
}
impl PartialOrd for Scheduled {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The deterministic discrete-event simulator over the virtual worlds.
pub struct Simulator {
    net: VirtualNet,
    trace: Trace,
    clock: VirtualClock,
    queue: BinaryHeap<std::cmp::Reverse<Scheduled>>,
    order: u64,
    runtimes: Vec<PeerRuntime>,
    transcript: RunTranscript,
}

impl Simulator {
    /// A fresh simulator over `net` under `trace`.
    #[must_use]
    pub fn new(net: VirtualNet, trace: Trace) -> Self {
        let peers = net.peers();
        Self {
            net,
            trace,
            clock: VirtualClock::new(),
            queue: BinaryHeap::new(),
            order: 0,
            runtimes: (0..peers).map(|_| PeerRuntime::new()).collect(),
            transcript: RunTranscript::default(),
        }
    }

    fn schedule(&mut self, at: u64, target: usize, kind: PendingKind) {
        let order = self.order;
        self.order += 1;
        self.queue.push(std::cmp::Reverse(Scheduled {
            at,
            order,
            target,
            kind,
        }));
    }

    /// Run `modules` (one per peer, indexed 0..peers) to `limits`, returning the modules (for state
    /// inspection) and the deterministic transcript. `modules.len()` must equal the net's peer count.
    ///
    /// # Panics
    /// If `modules.len()` does not match the net's peer count.
    pub fn run<M: SimModule>(
        mut self,
        mut modules: Vec<M>,
        limits: RunLimits,
    ) -> (Vec<M>, RunTranscript) {
        assert_eq!(
            modules.len(),
            self.net.peers(),
            "one module per peer (got {}, net has {})",
            modules.len(),
            self.net.peers()
        );

        // init every module at logical time 0 (ABI §3.1 prologue).
        for (peer, module) in modules.iter_mut().enumerate() {
            let mut rt = std::mem::take(&mut self.runtimes[peer]);
            let mut ctx = SimCtx::new(0, peer, &mut rt);
            module.init(&mut ctx);
            let effects = ctx.into_effects();
            self.runtimes[peer] = rt;
            self.apply(peer, effects);
        }

        let mut left = vec![false; modules.len()];
        while let Some(std::cmp::Reverse(entry)) = self.queue.pop() {
            if self.transcript.events_delivered >= limits.max_events {
                break;
            }
            if entry.at > limits.horizon_ms {
                break;
            }
            self.clock.advance_to(entry.at);
            let now = self.clock.now();
            let peer = entry.target;
            if left[peer] {
                continue;
            }
            let ev = match entry.kind {
                PendingKind::Timer { id } => {
                    // Suppress a fire whose timer a later `cancel` retired (§6.3), and a fire for a
                    // peer offline (churned out) at fire time.
                    if !self.runtimes[peer].timers.is_live(id) || !self.trace.online(peer, now) {
                        continue;
                    }
                    self.runtimes[peer].timers.fire(id);
                    SimEvent::Timer { id, fired_at: now }
                }
                PendingKind::Frame {
                    channel,
                    seq,
                    sender,
                    payload,
                } => {
                    // Deliver only if the recipient is online at arrival (else the frame is lost to
                    // churn — a dropped advisory observation, never a stall here).
                    if !self.trace.online(peer, now) {
                        continue;
                    }
                    SimEvent::Frame {
                        channel,
                        seq,
                        sender,
                        payload,
                    }
                }
            };
            self.transcript.events_delivered += 1;
            let effects = self.deliver(peer, &ev, &mut modules[peer]);
            if effects.leave.is_some() {
                left[peer] = true;
            }
            self.apply(peer, effects);
        }

        if limits.deliver_stop {
            let stop = SimEvent::Stop {
                reason: daemon_vhc_abi::STOP_REASON_RUN_COMPLETE,
            };
            for peer in 0..modules.len() {
                if left[peer] {
                    continue;
                }
                let effects = self.deliver(peer, &stop, &mut modules[peer]);
                // Post-Stop imports are illegal (ABI §4.4); the sim ignores any effects.
                let _ = effects;
            }
        }

        (modules, self.transcript)
    }

    /// Deliver one event to `module`, returning its buffered effects (kept separate from `apply` so
    /// the borrow of `self` and the borrow of `module` never overlap).
    fn deliver<M: SimModule>(&mut self, peer: usize, ev: &SimEvent, module: &mut M) -> Effects {
        let mut rt = std::mem::take(&mut self.runtimes[peer]);
        let mut ctx = SimCtx::new(self.clock.now(), peer, &mut rt);
        module.on_event(&mut ctx, ev);
        let effects = ctx.into_effects();
        self.runtimes[peer] = rt;
        effects
    }

    /// Apply a slice's buffered effects: route publishes (with trace latency/loss) and schedule
    /// armed timers. Effect processing order is deterministic (publishes then timers, in buffer
    /// order), and the global `order` counter makes the resulting queue a total order.
    fn apply(&mut self, peer: usize, effects: Effects) {
        let now = self.clock.now();
        for (channel, seq, payload) in effects.published {
            let payload_hash = *blake3::hash(&payload).as_bytes();
            self.transcript.publishes.push(PublishedFrame {
                peer,
                channel,
                seq,
                payload_hash,
                payload_len: payload.len(),
            });
            // The sender must be online to transmit (it is — it just ran); route to every other
            // peer under the trace's latency + loss model.
            for to in self.net.recipients(peer, channel) {
                if !self.trace.delivered(peer, to, channel, seq) {
                    continue;
                }
                let at = now + self.trace.latency_ms(peer, to, channel, seq);
                self.schedule(
                    at,
                    to,
                    PendingKind::Frame {
                        channel,
                        seq,
                        sender: peer,
                        payload: payload.clone(),
                    },
                );
            }
        }
        for (id, delay) in effects.armed {
            self.schedule(now + delay, peer, PendingKind::Timer { id });
        }
        for (name, value) in effects.metrics {
            self.transcript.metrics.push((peer, name, value));
        }
    }
}
