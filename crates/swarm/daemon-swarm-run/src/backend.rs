// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`TrainerBackend`] — the R↔E seam (spec §5.1/§10.2, ABI §2.3 lifecycle).
//!
//! The participant runtime (lane R) drives the round structure; the engine (lane E's `daemon-train`
//! worker) fills in the math. This trait is that boundary, deliberately **engine-agnostic**: every
//! signature is opaque bytes + plain structs — no `burn`, no `wasmtime`, no tensor types leak
//! across it, so the same round loop (Wave 2) hosts the [`StubBackend`] here and the real Burn/wasm
//! host later.
//!
//! The lifecycle mirrors the ABI guest exports (§5.1):
//! `build` (da_build) → `assess` (meta mode) → per round: `train_step` (da_step) ×
//! micro-batches, `inner_update` (da_inner_update) at accumulation boundaries, `make_update`
//! (da_make_update) at round end, then `ingest` (da_ingest_updates) over the staged committed set →
//! a post-ingest [`StateDigest`]; plus checkpoint save/load (§9).
//!
//! [`StubBackend`] is a deterministic fake (xxh3 of inputs) so Wave 2 can build + test the round
//! loop over a real seam before the engine exists.

use xxhash_rust::xxh3::{xxh3_128, xxh3_64};

use crate::seam::{ContentHash, PeerId, RoundId};

/// The engine seam the participant runtime drives (ABI §2.3 lifecycle).
pub trait TrainerBackend: Send {
    /// The backend's error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// `da_build`: register params + persistent state from the envelope's `[experiment.config]`
    /// bytes. Must be called before any training entry point.
    fn build(&mut self, config: &[u8]) -> Result<(), Self::Error>;

    /// Meta-mode footprint / eligibility on this peer (ABI §2.4, §6.5) — read-only, no allocation.
    fn assess(&self, meta: &AssessMeta) -> Result<Assessment, Self::Error>;

    /// `da_step`: one micro-batch — forward + backward (accumulate). `ctx` carries the accumulation
    /// position + this step's sequence total so loss scaling is exact.
    fn train_step(&mut self, batch: &BatchRef, ctx: StepCtx) -> Result<StepStats, Self::Error>;

    /// `da_inner_update`: apply the inner optimizer at an accumulation boundary.
    fn inner_update(&mut self, inner_step: u32) -> Result<(), Self::Error>;

    /// `da_make_update`: at round end, compress this peer's progress into opaque payload bytes (the
    /// object the payload plane moves + hashes; the swarm never parses it, §7.3).
    fn make_update(&mut self, round: RoundId) -> Result<Vec<u8>, Self::Error>;

    /// `da_ingest_updates`: decode + aggregate + outer step over the committed set, staged **in
    /// record order** (§6.4 I3), returning the post-ingest state digest (§5.6). Ordering is a
    /// consensus input: the caller must stage in `RoundRecord` order (sorted by node pubkey bytes).
    fn ingest(
        &mut self,
        round: RoundId,
        staged: &[StagedPayload],
    ) -> Result<StateDigest, Self::Error>;

    /// Serialize the checkpointable state (canonical params + `replicated` persistents) to bytes.
    fn checkpoint_save(&self) -> Result<Vec<u8>, Self::Error>;

    /// Restore state from [`TrainerBackend::checkpoint_save`] bytes (resync / rejoin, §9).
    fn checkpoint_load(&mut self, bytes: &[u8]) -> Result<(), Self::Error>;
}

/// Meta-mode inputs: this peer's effective resources after policy (§6.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessMeta {
    /// Effective VRAM available to training, in MiB.
    pub effective_vram_mb: u64,
    /// Effective host RAM available to training, in MiB.
    pub effective_ram_mb: u64,
}

/// Meta-mode output: eligibility + footprint estimates (§6.5, ABI §2.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assessment {
    /// Whether this peer can host the experiment under its effective resources.
    pub eligible: bool,
    /// Human-readable reasons (why-not, or informational headroom notes).
    pub reasons: Vec<String>,
    /// Estimated VRAM footprint, in MiB.
    pub vram_mb_estimate: u64,
    /// Estimated host-RAM footprint, in MiB.
    pub ram_mb_estimate: u64,
    /// Estimated per-round payload size, in bytes.
    pub payload_bytes_estimate: u64,
}

/// A materialized micro-batch handed to the engine (the host data pipeline owns tokenization; the
/// engine sees ready token ids).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchRef {
    /// The token ids of this micro-batch (row-major over its sequences).
    pub tokens: Vec<u32>,
    /// The sequence length (tokens per sequence).
    pub seq_len: u32,
}

/// The `da_step` context (ABI §4): accumulation position + this step's sequence total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepCtx {
    /// The inner step this micro-batch belongs to (`0..steps_per_round`).
    pub inner_step: u32,
    /// This micro-batch's index within the step's accumulation.
    pub mb_index: u32,
    /// The number of micro-batches accumulated this step.
    pub mb_count: u32,
    /// The total sequences in this step (for exact loss scaling).
    pub step_seqs: u32,
}

/// A per-step readout (loss; norms would join here).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepStats {
    /// The step's loss readout.
    pub loss: f32,
}

/// One committed payload staged for ingest, in record order (§6.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StagedPayload {
    /// The peer that produced this payload (node pubkey — the record ordering key).
    pub peer: PeerId,
    /// The payload's content hash (verified before staging).
    pub hash: ContentHash,
    /// The opaque payload bytes (`da_make_update` output from that peer).
    pub bytes: Vec<u8>,
}

/// A post-ingest state digest (xxh3-128 over sampled state, §5.6) — the cross-peer agreement probe.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StateDigest(pub [u8; 16]);

impl StateDigest {
    /// The raw 16-byte digest.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Lowercase hex rendering (32 chars).
    #[must_use]
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(32);
        for b in self.0 {
            s.push(char::from_digit((b >> 4) as u32, 16).expect("nibble"));
            s.push(char::from_digit((b & 0xf) as u32, 16).expect("nibble"));
        }
        s
    }
}

impl core::fmt::Debug for StateDigest {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "StateDigest({})", self.to_hex())
    }
}

/// A deterministic stub [`TrainerBackend`] whose "updates" are xxh3 of its inputs.
///
/// It holds a tiny in-memory state dict (`Vec<u64>`) seeded from the config bytes and folds batches
/// / staged payloads into it with fixed, order-sensitive mixing — enough for the Wave-2 round loop
/// to exercise the full lifecycle and for tests to assert determinism + record-order sensitivity,
/// with no engine, GPU, or wasm.
pub struct StubBackend {
    /// The canonical state dict; `None` until `build` is called.
    params: Option<Vec<u64>>,
    /// The current step's accumulator (reset at each `inner_update`).
    accum: u64,
}

impl StubBackend {
    /// The number of state-dict entries the stub maintains.
    const STATE_LEN: usize = 16;

    /// A fresh, unbuilt stub.
    #[must_use]
    pub fn new() -> Self {
        Self {
            params: None,
            accum: 0,
        }
    }

    fn params_mut(&mut self) -> Result<&mut Vec<u64>, StubError> {
        self.params.as_mut().ok_or(StubError::NotBuilt)
    }

    fn params(&self) -> Result<&Vec<u64>, StubError> {
        self.params.as_ref().ok_or(StubError::NotBuilt)
    }

    /// The little-endian bytes of the state dict (the digest / payload preimage).
    fn state_bytes(params: &[u64]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(params.len() * 8);
        for p in params {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        buf
    }
}

impl Default for StubBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainerBackend for StubBackend {
    type Error = StubError;

    fn build(&mut self, config: &[u8]) -> Result<(), StubError> {
        let seed = xxh3_64(config);
        let params = (0..Self::STATE_LEN)
            .map(|i| xxh3_64(&[config, &(i as u64).to_le_bytes()].concat()) ^ seed)
            .collect();
        self.params = Some(params);
        self.accum = 0;
        Ok(())
    }

    fn assess(&self, meta: &AssessMeta) -> Result<Assessment, StubError> {
        // A trivial footprint model: fixed per-state-entry cost. Deterministic + engine-free.
        let vram = (Self::STATE_LEN as u64) * 8;
        let ram = vram * 2;
        let payload = (Self::STATE_LEN as u64) * 8 + 8;
        let eligible = meta.effective_vram_mb >= vram && meta.effective_ram_mb >= ram;
        let reasons = if eligible {
            vec!["stub backend fits".into()]
        } else {
            vec![format!(
                "insufficient resources: need vram>={vram}MiB ram>={ram}MiB, have vram={} ram={}",
                meta.effective_vram_mb, meta.effective_ram_mb
            )]
        };
        Ok(Assessment {
            eligible,
            reasons,
            vram_mb_estimate: vram,
            ram_mb_estimate: ram,
            payload_bytes_estimate: payload,
        })
    }

    fn train_step(&mut self, batch: &BatchRef, ctx: StepCtx) -> Result<StepStats, StubError> {
        self.params()?; // ensure built
        let mut token_bytes = Vec::with_capacity(batch.tokens.len() * 4);
        for t in &batch.tokens {
            token_bytes.extend_from_slice(&t.to_le_bytes());
        }
        let step_mix = xxh3_64(&token_bytes)
            ^ (u64::from(ctx.inner_step) << 32)
            ^ u64::from(ctx.mb_index)
            ^ (u64::from(ctx.mb_count) << 16)
            ^ (u64::from(ctx.step_seqs) << 48);
        self.accum = self.accum.wrapping_add(step_mix).rotate_left(7);
        // A stable pseudo-loss readout in a sane range (0, 12].
        let loss = 1.0 + (self.accum % 1000) as f32 / 100.0;
        Ok(StepStats { loss })
    }

    fn inner_update(&mut self, inner_step: u32) -> Result<(), StubError> {
        let accum = self.accum;
        let params = self.params_mut()?;
        let idx = inner_step as usize % params.len();
        params[idx] = params[idx].wrapping_add(accum);
        params[0] ^= accum.rotate_left((inner_step % 63) + 1);
        self.accum = 0;
        Ok(())
    }

    fn make_update(&mut self, round: RoundId) -> Result<Vec<u8>, StubError> {
        let params = self.params()?;
        let mut preimage = round.to_le_bytes().to_vec();
        preimage.extend_from_slice(&Self::state_bytes(params));
        // The payload frame: round (8) ++ xxh3-128 of (round ++ state) (16) — deterministic bytes.
        let digest = xxh3_128(&preimage);
        let mut out = round.to_le_bytes().to_vec();
        out.extend_from_slice(&digest.to_le_bytes());
        Ok(out)
    }

    fn ingest(
        &mut self,
        round: RoundId,
        staged: &[StagedPayload],
    ) -> Result<StateDigest, StubError> {
        let params = self.params_mut()?;
        // Fold staged payloads in the caller's (record) order; the position index makes ingest
        // order-sensitive, so a reordering bug diverges the digest loudly (§6.4 I3).
        for (position, payload) in staged.iter().enumerate() {
            let mix = xxh3_64(&payload.bytes)
                .wrapping_mul(2 * position as u64 + 1)
                .wrapping_add(round);
            let idx = position % params.len();
            params[idx] = params[idx].wrapping_add(mix);
        }
        let digest = xxh3_128(&Self::state_bytes(params));
        Ok(StateDigest(digest.to_le_bytes()))
    }

    fn checkpoint_save(&self) -> Result<Vec<u8>, StubError> {
        let params = self.params()?;
        let mut buf = Vec::new();
        ciborium::into_writer(&(params, self.accum), &mut buf)
            .map_err(|e| StubError::Codec(e.to_string()))?;
        Ok(buf)
    }

    fn checkpoint_load(&mut self, bytes: &[u8]) -> Result<(), StubError> {
        let (params, accum): (Vec<u64>, u64) =
            ciborium::from_reader(bytes).map_err(|e| StubError::Codec(e.to_string()))?;
        self.params = Some(params);
        self.accum = accum;
        Ok(())
    }
}

/// Errors surfaced by [`StubBackend`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StubError {
    /// A training entry point was called before `build`.
    #[error("stub backend used before build()")]
    NotBuilt,
    /// A checkpoint (de)serialization step failed.
    #[error("stub checkpoint codec error: {0}")]
    Codec(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn built(config: &[u8]) -> StubBackend {
        let mut b = StubBackend::new();
        b.build(config).unwrap();
        b
    }

    fn payload(peer: u8, bytes: &[u8]) -> StagedPayload {
        StagedPayload {
            peer: PeerId([peer; 32]),
            hash: ContentHash::of(bytes),
            bytes: bytes.to_vec(),
        }
    }

    #[test]
    fn make_update_is_deterministic_and_config_sensitive() {
        let mut a = built(b"experiment-config");
        let mut b = built(b"experiment-config");
        assert_eq!(a.make_update(0).unwrap(), b.make_update(0).unwrap());

        let mut c = built(b"different-config");
        assert_ne!(a.make_update(0).unwrap(), c.make_update(0).unwrap());
        // The round number changes the payload frame.
        assert_ne!(a.make_update(0).unwrap(), a.make_update(1).unwrap());
    }

    #[test]
    fn training_changes_the_update() {
        let mut a = built(b"cfg");
        let before = a.make_update(0).unwrap();
        let batch = BatchRef {
            tokens: vec![1, 2, 3, 4],
            seq_len: 4,
        };
        a.train_step(
            &batch,
            StepCtx {
                inner_step: 0,
                mb_index: 0,
                mb_count: 1,
                step_seqs: 4,
            },
        )
        .unwrap();
        a.inner_update(0).unwrap();
        assert_ne!(before, a.make_update(0).unwrap());
    }

    #[test]
    fn ingest_is_deterministic_and_order_sensitive() {
        let p1 = payload(0x01, b"update-alpha");
        let p2 = payload(0x02, b"update-beta");

        // Same order, same config -> identical digest.
        let d_forward_a = built(b"cfg").ingest(5, &[p1.clone(), p2.clone()]).unwrap();
        let d_forward_b = built(b"cfg").ingest(5, &[p1.clone(), p2.clone()]).unwrap();
        assert_eq!(d_forward_a, d_forward_b);

        // Reordering the staged set changes the digest (record order is a consensus input).
        let d_reversed = built(b"cfg").ingest(5, &[p2, p1]).unwrap();
        assert_ne!(d_forward_a, d_reversed);
    }

    #[test]
    fn checkpoint_round_trips_state() {
        let mut a = built(b"cfg");
        let batch = BatchRef {
            tokens: vec![9, 8, 7],
            seq_len: 3,
        };
        a.train_step(
            &batch,
            StepCtx {
                inner_step: 1,
                mb_index: 0,
                mb_count: 1,
                step_seqs: 3,
            },
        )
        .unwrap();
        a.inner_update(1).unwrap();
        let saved = a.checkpoint_save().unwrap();
        let expect = a.make_update(2).unwrap();

        let mut restored = StubBackend::new();
        restored.checkpoint_load(&saved).unwrap();
        assert_eq!(restored.make_update(2).unwrap(), expect);
    }

    #[test]
    fn entry_points_require_build() {
        let mut fresh = StubBackend::new();
        let err = fresh
            .train_step(
                &BatchRef {
                    tokens: vec![],
                    seq_len: 1,
                },
                StepCtx {
                    inner_step: 0,
                    mb_index: 0,
                    mb_count: 1,
                    step_seqs: 0,
                },
            )
            .unwrap_err();
        assert!(matches!(err, StubError::NotBuilt));
        assert!(matches!(fresh.make_update(0), Err(StubError::NotBuilt)));
    }

    #[test]
    fn assess_reflects_resources() {
        let stub = StubBackend::new();
        let ok = stub
            .assess(&AssessMeta {
                effective_vram_mb: 10_000,
                effective_ram_mb: 20_000,
            })
            .unwrap();
        assert!(ok.eligible);

        let tight = stub
            .assess(&AssessMeta {
                effective_vram_mb: 0,
                effective_ram_mb: 0,
            })
            .unwrap();
        assert!(!tight.eligible);
        assert!(!tight.reasons.is_empty());
    }
}
