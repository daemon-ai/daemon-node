// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The virtual network: channel pub/sub with **trace-driven** latency, churn, and session-length
//! models (architecture §6). Everything is a deterministic function of a seed plus the message
//! coordinates `(from, to, channel, seq)`, so a run replays bit-for-bit — the SDK-side analogue of
//! the WAN-trace replay the host testkit does over production blobs.
//!
//! The model is transport-independent (architecture §2): the guest never observes it, exactly as at
//! runtime. Routing/scheduling of deliveries lives in the [`crate::sim::Simulator`]; this module
//! owns only the *model* (how long a frame takes, whether it is lost, whether a peer is reachable).

/// A splitmix64 step — the deterministic per-message randomness source (no host entropy, so replay
/// is exact).
fn splitmix(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn mix4(seed: u64, a: u64, b: u64, c: u64, d: u64) -> u64 {
    let mut h = seed;
    for v in [a, b, c, d] {
        h = splitmix(h ^ v);
    }
    h
}

/// A per-peer offline window `[start_ms, end_ms)` on the logical clock — the churn / session-length
/// primitive. A peer offline at delivery time neither receives inbound frames nor has its outbound
/// frames delivered (its session has lapsed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OfflineWindow {
    /// The peer index this window applies to.
    pub peer: usize,
    /// Inclusive start (logical ms).
    pub start_ms: u64,
    /// Exclusive end (logical ms); `u64::MAX` means "never returns" (a permanent departure).
    pub end_ms: u64,
}

/// A deterministic, seeded WAN trace: latency, loss (churn), and session windows. Built fluently;
/// [`Trace::reliable`] is the zero-loss, fixed-latency default a convergence gate uses.
#[derive(Debug, Clone)]
pub struct Trace {
    seed: u64,
    base_latency_ms: u64,
    jitter_ms: u64,
    drop_permille: u64,
    offline: Vec<OfflineWindow>,
}

impl Trace {
    /// A reliable trace: fixed `base_latency_ms`, no jitter, no loss, no churn. The convergence-gate
    /// default (deterministic and lossless, so an averaging run provably converges).
    #[must_use]
    pub fn reliable(base_latency_ms: u64) -> Self {
        Self {
            seed: 0,
            base_latency_ms,
            jitter_ms: 0,
            drop_permille: 0,
            offline: Vec::new(),
        }
    }

    /// A seeded trace with jitter and per-message loss — the adversarial-suite shape (loss models
    /// gossip churn and transient partitions).
    #[must_use]
    pub fn seeded(seed: u64, base_latency_ms: u64, jitter_ms: u64, drop_permille: u64) -> Self {
        Self {
            seed,
            base_latency_ms,
            jitter_ms,
            drop_permille,
            offline: Vec::new(),
        }
    }

    /// Add a per-peer offline window (churn / bounded session length). Chainable.
    #[must_use]
    pub fn offline(mut self, peer: usize, start_ms: u64, end_ms: u64) -> Self {
        self.offline.push(OfflineWindow {
            peer,
            start_ms,
            end_ms,
        });
        self
    }

    /// The one-way delivery latency for a frame — deterministic in `(seed, from, to, channel, seq)`.
    #[must_use]
    pub fn latency_ms(&self, from: usize, to: usize, channel: u32, seq: u64) -> u64 {
        if self.jitter_ms == 0 {
            return self.base_latency_ms;
        }
        let r = mix4(self.seed, from as u64, to as u64, u64::from(channel), seq);
        self.base_latency_ms + (r % (self.jitter_ms + 1))
    }

    /// Whether a frame survives the link (loss model). Deterministic; a lossless trace never drops.
    #[must_use]
    pub fn delivered(&self, from: usize, to: usize, channel: u32, seq: u64) -> bool {
        if self.drop_permille == 0 {
            return true;
        }
        let r = mix4(
            self.seed ^ 0xD1CE,
            from as u64,
            to as u64,
            u64::from(channel),
            seq,
        );
        (r % 1000) >= self.drop_permille
    }

    /// Whether `peer` is online (reachable) at logical time `at` — false inside any offline window.
    #[must_use]
    pub fn online(&self, peer: usize, at: u64) -> bool {
        !self
            .offline
            .iter()
            .any(|w| w.peer == peer && at >= w.start_ms && at < w.end_ms)
    }
}

impl Default for Trace {
    /// A reliable 10 ms-latency trace.
    fn default() -> Self {
        Self::reliable(10)
    }
}

/// The virtual network topology: how many peers, and which channels gossip to every other peer.
/// Phase A has one channel (`control`, id 0, bidirectional, ABI §6.2), used here as the gossip
/// plane every averager peer publishes its running mean on.
#[derive(Debug, Clone)]
pub struct VirtualNet {
    peers: usize,
}

impl VirtualNet {
    /// A fully-connected gossip mesh of `peers` peers.
    #[must_use]
    pub fn mesh(peers: usize) -> Self {
        Self { peers }
    }

    /// The peer count.
    #[must_use]
    pub fn peers(&self) -> usize {
        self.peers
    }

    /// The recipients of a frame `from` publishes on `channel`: every other peer (gossip excludes
    /// the sender — a peer does not receive its own publish).
    #[must_use]
    pub fn recipients(&self, from: usize, _channel: u32) -> Vec<usize> {
        (0..self.peers).filter(|&p| p != from).collect()
    }
}
