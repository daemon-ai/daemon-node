// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The driver-side journaling seam (ABI §8) — dependency-inverted over A1's substrate.
//!
//! The v2 event-loop driver is **born audited**: it emits every §8 observation (delivered events,
//! nondeterministic import results, publish outcomes, timer arms/cancels, clock readings, the
//! terminal fact) from its first commit. The concrete crash-safe segmented store lives in
//! `daemon-vhc-observe::journal` (A1), and the architecture's dependency direction is
//! `daemon-vhc-observe → daemon-vhc-host` (the replay verifiers drive modules *through* the host
//! runtime) — so the driver cannot link the store. This trait is the inversion: the driver writes
//! through a [`JournalSink`]; whoever links both crates (the session/worker wiring, and the tier-1
//! tests) adapts the real `Journal` onto it. [`MemorySink`] is the in-memory test double.
//!
//! Method-per-record-kind (rather than one `append(Body)`) keeps this crate free of the record
//! types (they live with the store) while making every §8.3 tag the driver can produce explicit.
//! Ordering/durability obligations (§8.4) are noted per method; the adapter maps them onto
//! `append` vs `append_committed`.

/// A journaling failure. Journal writes are load-bearing (§8.4: a publish MUST NOT return before
/// its barrier commits), so sink errors abort the affected import/run rather than being dropped.
#[derive(Debug, thiserror::Error)]
#[error("journal sink: {0}")]
pub struct SinkError(pub String);

/// The §8.3 records the Phase-A driver produces, as a write-only seam (see module docs).
///
/// All methods take `&mut self`; the driver serializes access behind its pump lock, so an adapter
/// needs no interior synchronization.
pub trait JournalSink: Send {
    /// tag 0 — the run header (admitted manifest/config/grants/claim/channels/device bytes,
    /// negotiated abi/worlds/bridge). Written once, first.
    #[allow(clippy::too_many_arguments)]
    fn run_header(
        &mut self,
        abi: u64,
        worlds: &[(String, u64)],
        bridge: bool,
        manifest: &[u8],
        config: &[u8],
        grants: &[u8],
        claim: &[u8],
        channels: &[u8],
        device: &[u8],
    ) -> Result<(), SinkError>;

    /// tag 13 — an instantiation (counter = the §7.1 generation seed; reason 0/1/2; `at` logical).
    /// Written at instantiation, before any guest code (§8.3/§10.3).
    fn instantiation(&mut self, counter: u64, reason: u64, at: u64) -> Result<(), SinkError>;

    /// tag 11 — the `da_init` call (config/grants blake3 pins + the guest status, §9.4 step 11).
    fn init(
        &mut self,
        config_hash: [u8; 32],
        grants_hash: [u8; 32],
        status: u64,
    ) -> Result<(), SinkError>;

    /// tag 1 — a delivered event: the exact frame bytes `next_event` returned, with the logical
    /// delivery time. Written before the guest observes the frame (§8.4 rule 4).
    fn event(&mut self, at: u64, frame: &[u8]) -> Result<(), SinkError>;

    /// tag 12 — the complete original signed wire frame behind an authoritative `Frame` event
    /// (§8.6), inline at Phase A.
    fn signed_frame(
        &mut self,
        channel: u64,
        seq: u64,
        sender: [u8; 32],
        frame: &[u8],
    ) -> Result<(), SinkError>;

    /// The durable channel-scoped sequence counter: the next seq this channel may publish
    /// (recovered from committed tag-4 records on open — §8.4 rule 2, §12.2).
    fn next_seq(&mut self, channel: u64) -> u64;

    /// tag 4 — a publish: the durable seq advance + the complete signed outbound frame. MUST be
    /// committed (barrier crossed) before this returns — `publish` returns to the guest only after
    /// (§6.2/§8.4 rule 2).
    fn publish(
        &mut self,
        channel: u64,
        seq: u64,
        payload: &[u8],
        frame: &[u8],
    ) -> Result<(), SinkError>;

    /// tag 3 — a `now()` clock reading (§6.5: clocks are not messages but must be captured).
    fn clock(&mut self, now: u64) -> Result<(), SinkError>;

    /// tag 5 — a timer arm (§6.3).
    fn timer_arm(&mut self, id: u64, delay: u64, armed_at: u64) -> Result<(), SinkError>;

    /// tag 6 — a timer-cancel outcome (§6.3 — nondeterministic: whether the cancel raced the fire).
    fn timer_cancel(&mut self, id: u64, status: u64) -> Result<(), SinkError>;

    /// tag 2 — a `read_back` `Ok` value (§6.4; `NeedCapacity` exchanges are recordless by design).
    fn read_back(
        &mut self,
        src: u64,
        kind: u64,
        status: u64,
        value: &[u8],
    ) -> Result<(), SinkError>;

    /// tag 15 — a `device_profile` delivery (§8.3; Phase B): the profile is a nondeterministic
    /// input (the probe's measurement), recorded verbatim per delivery so replay feeds the
    /// recorded bytes rather than re-probing.
    fn device_profile(&mut self, profile: &[u8]) -> Result<(), SinkError>;

    /// tag 7 — an advisory drop/coalesce (§4.7: every drop or coalesce MUST be journaled).
    /// `class`: 0 payload-ready, 1 timer, 2 gossip, 3 budget; `rule`: the fixed coalesce code.
    fn drop_coalesced(
        &mut self,
        class: u64,
        rule: u64,
        timer_id: Option<u64>,
        hash: Option<[u8; 32]>,
    ) -> Result<(), SinkError>;

    /// tag 14 — a completion arrival (ABI §7.5/§8.3, Phase B): the op plus its standalone
    /// canonical `completion-result` bytes. Written at ARRIVAL (enqueue), before the completion
    /// event is deliverable — completion results and their order are nondeterministic inputs
    /// (§8.1).
    fn completion(&mut self, op: u64, result: &[u8]) -> Result<(), SinkError>;

    /// tag 9 — the terminal fact (kind 0 = outcome, 1 = trap, 2 = forced interruption). MUST be
    /// committed before it is reported (§8.4 rule 2).
    fn terminal(
        &mut self,
        kind: u64,
        outcome: Option<u64>,
        trap: Option<(String, String, String, String)>,
    ) -> Result<(), SinkError>;
}

/// A shared sink: the embedder keeps a handle for inspection/commit while the driver writes.
/// (The driver already serializes writes behind its pump lock; this adds shared ownership only.)
impl<S: JournalSink> JournalSink for std::sync::Arc<std::sync::Mutex<S>> {
    fn run_header(
        &mut self,
        abi: u64,
        worlds: &[(String, u64)],
        bridge: bool,
        manifest: &[u8],
        config: &[u8],
        grants: &[u8],
        claim: &[u8],
        channels: &[u8],
        device: &[u8],
    ) -> Result<(), SinkError> {
        self.lock().expect("sink lock").run_header(
            abi, worlds, bridge, manifest, config, grants, claim, channels, device,
        )
    }
    fn instantiation(&mut self, counter: u64, reason: u64, at: u64) -> Result<(), SinkError> {
        self.lock()
            .expect("sink lock")
            .instantiation(counter, reason, at)
    }
    fn init(
        &mut self,
        config_hash: [u8; 32],
        grants_hash: [u8; 32],
        status: u64,
    ) -> Result<(), SinkError> {
        self.lock()
            .expect("sink lock")
            .init(config_hash, grants_hash, status)
    }
    fn event(&mut self, at: u64, frame: &[u8]) -> Result<(), SinkError> {
        self.lock().expect("sink lock").event(at, frame)
    }
    fn signed_frame(
        &mut self,
        channel: u64,
        seq: u64,
        sender: [u8; 32],
        frame: &[u8],
    ) -> Result<(), SinkError> {
        self.lock()
            .expect("sink lock")
            .signed_frame(channel, seq, sender, frame)
    }
    fn next_seq(&mut self, channel: u64) -> u64 {
        self.lock().expect("sink lock").next_seq(channel)
    }
    fn publish(
        &mut self,
        channel: u64,
        seq: u64,
        payload: &[u8],
        frame: &[u8],
    ) -> Result<(), SinkError> {
        self.lock()
            .expect("sink lock")
            .publish(channel, seq, payload, frame)
    }
    fn clock(&mut self, now: u64) -> Result<(), SinkError> {
        self.lock().expect("sink lock").clock(now)
    }
    fn timer_arm(&mut self, id: u64, delay: u64, armed_at: u64) -> Result<(), SinkError> {
        self.lock()
            .expect("sink lock")
            .timer_arm(id, delay, armed_at)
    }
    fn timer_cancel(&mut self, id: u64, status: u64) -> Result<(), SinkError> {
        self.lock().expect("sink lock").timer_cancel(id, status)
    }
    fn read_back(
        &mut self,
        src: u64,
        kind: u64,
        status: u64,
        value: &[u8],
    ) -> Result<(), SinkError> {
        self.lock()
            .expect("sink lock")
            .read_back(src, kind, status, value)
    }
    fn device_profile(&mut self, profile: &[u8]) -> Result<(), SinkError> {
        self.lock().expect("sink lock").device_profile(profile)
    }
    fn drop_coalesced(
        &mut self,
        class: u64,
        rule: u64,
        timer_id: Option<u64>,
        hash: Option<[u8; 32]>,
    ) -> Result<(), SinkError> {
        self.lock()
            .expect("sink lock")
            .drop_coalesced(class, rule, timer_id, hash)
    }
    fn completion(&mut self, op: u64, result: &[u8]) -> Result<(), SinkError> {
        self.lock().expect("sink lock").completion(op, result)
    }
    fn terminal(
        &mut self,
        kind: u64,
        outcome: Option<u64>,
        trap: Option<(String, String, String, String)>,
    ) -> Result<(), SinkError> {
        self.lock()
            .expect("sink lock")
            .terminal(kind, outcome, trap)
    }
}

/// One recorded entry of the in-memory test sink — a readable mirror of the §8.3 bodies.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum SinkEntry {
    RunHeader {
        abi: u64,
        bridge: bool,
    },
    Instantiation {
        counter: u64,
        reason: u64,
        at: u64,
    },
    Init {
        status: u64,
    },
    Event {
        at: u64,
        frame: Vec<u8>,
    },
    SignedFrame {
        channel: u64,
        seq: u64,
        sender: [u8; 32],
        frame: Vec<u8>,
    },
    Publish {
        channel: u64,
        seq: u64,
        payload_hash: [u8; 32],
        frame: Vec<u8>,
    },
    Clock {
        now: u64,
    },
    TimerArm {
        id: u64,
        delay: u64,
        armed_at: u64,
    },
    TimerCancel {
        id: u64,
        status: u64,
    },
    ReadBack {
        src: u64,
        kind: u64,
        status: u64,
        value: Vec<u8>,
    },
    DeviceProfile {
        profile: Vec<u8>,
    },
    Drop {
        class: u64,
        rule: u64,
        timer_id: Option<u64>,
        hash: Option<[u8; 32]>,
    },
    Completion {
        op: u64,
        result: Vec<u8>,
    },
    Terminal {
        kind: u64,
        outcome: Option<u64>,
    },
}

impl SinkEntry {
    /// The §8.3 tag this entry mirrors.
    #[must_use]
    pub fn tag(&self) -> u8 {
        match self {
            Self::RunHeader { .. } => 0,
            Self::Event { .. } => 1,
            Self::ReadBack { .. } => 2,
            Self::Clock { .. } => 3,
            Self::Publish { .. } => 4,
            Self::TimerArm { .. } => 5,
            Self::TimerCancel { .. } => 6,
            Self::Drop { .. } => 7,
            Self::Terminal { .. } => 9,
            Self::Init { .. } => 11,
            Self::SignedFrame { .. } => 12,
            Self::Instantiation { .. } => 13,
            Self::Completion { .. } => 14,
            Self::DeviceProfile { .. } => 15,
        }
    }
}

/// The in-memory [`JournalSink`] test double: append-order entries + per-channel seq counters.
#[derive(Debug, Default)]
pub struct MemorySink {
    /// Every entry, in append order.
    pub entries: Vec<SinkEntry>,
    seq_high: std::collections::BTreeMap<u64, Option<u64>>,
}

impl MemorySink {
    /// A fresh, empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl JournalSink for MemorySink {
    fn run_header(
        &mut self,
        abi: u64,
        _worlds: &[(String, u64)],
        bridge: bool,
        _manifest: &[u8],
        _config: &[u8],
        _grants: &[u8],
        _claim: &[u8],
        _channels: &[u8],
        _device: &[u8],
    ) -> Result<(), SinkError> {
        self.entries.push(SinkEntry::RunHeader { abi, bridge });
        Ok(())
    }

    fn instantiation(&mut self, counter: u64, reason: u64, at: u64) -> Result<(), SinkError> {
        self.entries.push(SinkEntry::Instantiation {
            counter,
            reason,
            at,
        });
        Ok(())
    }

    fn init(
        &mut self,
        _config_hash: [u8; 32],
        _grants_hash: [u8; 32],
        status: u64,
    ) -> Result<(), SinkError> {
        self.entries.push(SinkEntry::Init { status });
        Ok(())
    }

    fn event(&mut self, at: u64, frame: &[u8]) -> Result<(), SinkError> {
        self.entries.push(SinkEntry::Event {
            at,
            frame: frame.to_vec(),
        });
        Ok(())
    }

    fn signed_frame(
        &mut self,
        channel: u64,
        seq: u64,
        sender: [u8; 32],
        frame: &[u8],
    ) -> Result<(), SinkError> {
        self.entries.push(SinkEntry::SignedFrame {
            channel,
            seq,
            sender,
            frame: frame.to_vec(),
        });
        Ok(())
    }

    fn next_seq(&mut self, channel: u64) -> u64 {
        self.seq_high
            .get(&channel)
            .copied()
            .flatten()
            .map_or(0, |h| h + 1)
    }

    fn publish(
        &mut self,
        channel: u64,
        seq: u64,
        payload: &[u8],
        frame: &[u8],
    ) -> Result<(), SinkError> {
        self.entries.push(SinkEntry::Publish {
            channel,
            seq,
            payload_hash: *blake3::hash(payload).as_bytes(),
            frame: frame.to_vec(),
        });
        self.seq_high.insert(channel, Some(seq));
        Ok(())
    }

    fn clock(&mut self, now: u64) -> Result<(), SinkError> {
        self.entries.push(SinkEntry::Clock { now });
        Ok(())
    }

    fn timer_arm(&mut self, id: u64, delay: u64, armed_at: u64) -> Result<(), SinkError> {
        self.entries.push(SinkEntry::TimerArm {
            id,
            delay,
            armed_at,
        });
        Ok(())
    }

    fn timer_cancel(&mut self, id: u64, status: u64) -> Result<(), SinkError> {
        self.entries.push(SinkEntry::TimerCancel { id, status });
        Ok(())
    }

    fn read_back(
        &mut self,
        src: u64,
        kind: u64,
        status: u64,
        value: &[u8],
    ) -> Result<(), SinkError> {
        self.entries.push(SinkEntry::ReadBack {
            src,
            kind,
            status,
            value: value.to_vec(),
        });
        Ok(())
    }

    fn device_profile(&mut self, profile: &[u8]) -> Result<(), SinkError> {
        self.entries.push(SinkEntry::DeviceProfile {
            profile: profile.to_vec(),
        });
        Ok(())
    }

    fn drop_coalesced(
        &mut self,
        class: u64,
        rule: u64,
        timer_id: Option<u64>,
        hash: Option<[u8; 32]>,
    ) -> Result<(), SinkError> {
        self.entries.push(SinkEntry::Drop {
            class,
            rule,
            timer_id,
            hash,
        });
        Ok(())
    }

    fn completion(&mut self, op: u64, result: &[u8]) -> Result<(), SinkError> {
        self.entries.push(SinkEntry::Completion {
            op,
            result: result.to_vec(),
        });
        Ok(())
    }

    fn terminal(
        &mut self,
        kind: u64,
        outcome: Option<u64>,
        _trap: Option<(String, String, String, String)>,
    ) -> Result<(), SinkError> {
        self.entries.push(SinkEntry::Terminal { kind, outcome });
        Ok(())
    }
}
