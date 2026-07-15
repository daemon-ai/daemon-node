// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Acceptance toys authored natively against the runtime (architecture §9 expressiveness catalog).
//!
//! [`SparseAverager`] is the **SPARTA-shaped** row: continuous sparse averaging with **no rounds**
//! and **no coordinator** — a timer-driven gossip loop using only the A2 closed subset (`set_timer`,
//! `publish`, and the delivered `Frame`/`Timer` events). Each tick it det-averages its own value
//! with the latest value gossiped by every peer and republishes the result; over a fully-connected
//! reliable trace the mesh converges to the exact global mean, and the averaging rides the shared
//! `daemon-vhc-det` kernels, so the arithmetic is bit-identical to the host det lane.
//!
//! The same algorithm ships as a wasm blob (the pinned `toy_averager.wasm`, timers + publish),
//! driven under the host testkit — the two-layer proof of architecture §6. The gossip *ingest* half
//! extends onto B1's landed `net@` vocabulary when this rebases on B1.

use crate::sim::{SimCtx, SimEvent, SimModule};

/// The gossip channel the averager publishes/receives on (the Phase-A `control` channel, ABI §6.2).
pub const GOSSIP_CHANNEL: u32 = daemon_vhc_abi::DEFAULT_CHANNEL_CONTROL_ID;

/// Encode a det value vector as its raw little-endian f32 bytes (opaque payload, ABI §6.2).
fn encode(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Decode a det value vector from raw little-endian f32 bytes; empty on a malformed length.
fn decode(bytes: &[u8]) -> Vec<f32> {
    if !bytes.len().is_multiple_of(4) {
        return Vec::new();
    }
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// det-average a set of equal-length vectors via the shared fixed-order kernels (bit-identical to
/// the host). Empty input returns empty.
fn det_average(vecs: &[Vec<f32>]) -> Vec<f32> {
    if vecs.is_empty() {
        return Vec::new();
    }
    let refs: Vec<&[f32]> = vecs.iter().map(Vec::as_slice).collect();
    let sum = daemon_vhc_det::det_sum(&refs).expect("averaged vectors share a shape");
    daemon_vhc_det::det_scale(&sum, 1.0 / vecs.len() as f64)
}

/// A continuous (round-less, coordinator-less) SPARTA-shaped averager.
pub struct SparseAverager {
    value: Vec<f32>,
    latest: std::collections::BTreeMap<usize, Vec<f32>>,
    tick_ms: u64,
    ticks_remaining: u32,
}

impl SparseAverager {
    /// A peer starting at `initial`, mixing every `tick_ms` for `ticks` ticks.
    #[must_use]
    pub fn new(initial: Vec<f32>, tick_ms: u64, ticks: u32) -> Self {
        Self {
            value: initial,
            latest: std::collections::BTreeMap::new(),
            tick_ms,
            ticks_remaining: ticks,
        }
    }

    /// The peer's current value (its converging estimate of the global mean).
    #[must_use]
    pub fn value(&self) -> &[f32] {
        &self.value
    }
}

impl SimModule for SparseAverager {
    fn init(&mut self, ctx: &mut SimCtx) {
        // Announce the opening value so every peer starts with everyone's initial (ABI §3.1
        // prologue), then arm the first mixing tick.
        ctx.publish(GOSSIP_CHANNEL, &encode(&self.value));
        ctx.set_timer(self.tick_ms);
    }

    fn on_event(&mut self, ctx: &mut SimCtx, ev: &SimEvent) {
        match ev {
            SimEvent::Frame {
                sender, payload, ..
            } => {
                let peer_value = decode(payload);
                if !peer_value.is_empty() {
                    self.latest.insert(*sender, peer_value);
                }
            }
            SimEvent::Timer { .. } => {
                if self.ticks_remaining == 0 {
                    return;
                }
                self.ticks_remaining -= 1;
                // Continuous sparse averaging: mix own value with every peer's latest known value.
                let mut vecs = vec![self.value.clone()];
                vecs.extend(self.latest.values().cloned());
                self.value = det_average(&vecs);
                ctx.publish(GOSSIP_CHANNEL, &encode(&self.value));
                if let Some(&first) = self.value.first() {
                    ctx.emit_metric("mean0", f64::from(first));
                }
                if self.ticks_remaining > 0 {
                    ctx.set_timer(self.tick_ms);
                }
            }
            SimEvent::Stop { .. } | SimEvent::Quiesce { .. } => {}
        }
    }
}
