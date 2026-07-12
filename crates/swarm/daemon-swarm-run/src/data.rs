// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The pre-tokenized shard data pipeline (spec §8) + `BatchId` mapping (§6.3).
//!
//! Corpora are pre-tokenized offline into fixed-width shards (u16/u32 token streams); a
//! [`Manifest`] (`manifest.json`) lists the shards, their sizes, token counts, and a blake3 per
//! shard. Peers map [`BatchId`] intervals to `(shard, offset)` **purely locally** ([`Manifest::locate`])
//! and slice their assigned interval into `steps_per_round` inner steps × micro-batches
//! ([`slice_interval`]) — no per-batch RPC.
//!
//! [`SyntheticCorpus`] generates a deterministic seeded corpus (u16 tokens) for tests, so the round
//! loop (Wave 2) and the worker (Wave 3) have a data source with no external download.

use serde::{Deserialize, Serialize};

use daemon_swarm_proto::{blake3_hash, Hash};

use crate::seam::BatchId;

/// Whether `s` is a well-formed blake3 content-hash hex string: exactly `Hash::LEN * 2` (64) hex
/// digits. The manifest stores per-shard hashes as hex text (JSON), so this validates the string
/// form of proto's canonical [`Hash`] without materializing one.
fn is_blake3_hex(s: &str) -> bool {
    s.len() == Hash::LEN * 2 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// The token element width of a shard's fixed-width stream (spec §8: u16/u32).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TokenWidth {
    /// 16-bit tokens (vocab ≤ 65 536) — the MVP width.
    U16,
    /// 32-bit tokens (large-vocab corpora).
    U32,
}

impl TokenWidth {
    /// The width in bytes.
    #[must_use]
    pub fn bytes(self) -> u64 {
        match self {
            TokenWidth::U16 => 2,
            TokenWidth::U32 => 4,
        }
    }
}

/// One shard entry of the manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardDesc {
    /// The shard's file name (relative to the manifest root).
    pub name: String,
    /// The shard's size in bytes.
    pub bytes: u64,
    /// The number of tokens in the shard.
    pub tokens: u64,
    /// The shard's blake3 content hash (lowercase hex).
    pub blake3: String,
}

/// The pre-tokenized corpus manifest (`manifest.json`, §8).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// The token element width shared by every shard.
    pub token_width: TokenWidth,
    /// The sequence length (tokens per training sequence).
    pub seq_len: u32,
    /// The shards, in data-window order.
    pub shards: Vec<ShardDesc>,
}

/// The location of one sequence within the corpus: which shard, and the token offset into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchLocation {
    /// The index of the shard holding the sequence.
    pub shard: usize,
    /// The token offset of the sequence's first token within that shard.
    pub token_offset: u64,
}

impl Manifest {
    /// Parse + validate a `manifest.json` document.
    pub fn from_json(json: &str) -> Result<Self, DataError> {
        let manifest: Manifest =
            serde_json::from_str(json).map_err(|e| DataError::Parse(e.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Serialize to a `manifest.json` document.
    pub fn to_json(&self) -> Result<String, DataError> {
        serde_json::to_string_pretty(self).map_err(|e| DataError::Parse(e.to_string()))
    }

    /// Validate structural + numeric consistency (RUN-3 `invalid_manifest_rejected`).
    pub fn validate(&self) -> Result<(), DataError> {
        if self.shards.is_empty() {
            return Err(DataError::EmptyManifest);
        }
        if self.seq_len == 0 {
            return Err(DataError::ZeroSeqLen);
        }
        let width = self.token_width.bytes();
        let seq_len = u64::from(self.seq_len);
        for (i, shard) in self.shards.iter().enumerate() {
            if shard.tokens == 0 {
                return Err(DataError::ZeroShardTokens(i));
            }
            if shard.bytes != shard.tokens * width {
                return Err(DataError::ShardSizeMismatch {
                    shard: i,
                    expected: shard.tokens * width,
                    declared: shard.bytes,
                });
            }
            // Each shard holds whole sequences, so a BatchId never straddles a shard boundary.
            if !shard.tokens.is_multiple_of(seq_len) {
                return Err(DataError::ShardNotSeqAligned {
                    shard: i,
                    tokens: shard.tokens,
                    seq_len: self.seq_len,
                });
            }
            if !is_blake3_hex(&shard.blake3) {
                return Err(DataError::BadShardHash(i));
            }
        }
        Ok(())
    }

    /// The total number of tokens across all shards.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.shards.iter().map(|s| s.tokens).sum()
    }

    /// The total number of whole sequences across all shards (the `BatchId` range upper bound).
    #[must_use]
    pub fn total_sequences(&self) -> u64 {
        self.total_tokens() / u64::from(self.seq_len)
    }

    /// Map a [`BatchId`] (a sequence index over the data window) to its `(shard, token_offset)`.
    pub fn locate(&self, batch: BatchId) -> Result<BatchLocation, DataError> {
        let seq_len = u64::from(self.seq_len);
        let mut cursor = 0u64;
        for (shard, desc) in self.shards.iter().enumerate() {
            let seqs = desc.tokens / seq_len;
            if batch < cursor + seqs {
                let seq_in_shard = batch - cursor;
                return Ok(BatchLocation {
                    shard,
                    token_offset: seq_in_shard * seq_len,
                });
            }
            cursor += seqs;
        }
        Err(DataError::BatchOutOfRange {
            batch,
            total: self.total_sequences(),
        })
    }
}

/// A half-open interval of [`BatchId`]s assigned to a peer for one round (§6.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchInterval {
    /// Inclusive start.
    pub start: BatchId,
    /// Exclusive end.
    pub end: BatchId,
}

impl BatchInterval {
    /// Construct `[start, end)`.
    #[must_use]
    pub fn new(start: BatchId, end: BatchId) -> Self {
        Self { start, end }
    }

    /// The number of sequences in the interval.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Whether the interval is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// One micro-batch: a half-open `[start, end)` slice of a peer's interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MicroBatch {
    /// Inclusive start.
    pub start: BatchId,
    /// Exclusive end.
    pub end: BatchId,
}

/// One inner step: the micro-batches trained + accumulated before an `inner_update` (§5.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InnerStep {
    /// The step index within the round (`0..steps_per_round`).
    pub index: u32,
    /// The micro-batches making up this step's global batch share.
    pub micro_batches: Vec<MicroBatch>,
}

/// Slice a peer's assigned `interval` into `steps_per_round` inner steps, each chunked into
/// micro-batches of `micro_batch` sequences (the last per step may be smaller) — RUN-3
/// `interval_slices_into_h_steps`.
///
/// The cadence (`steps_per_round`) is uniform across peers (§6.3), so the interval must divide
/// evenly into the steps; a non-divisible interval is [`DataError::IntervalNotDivisible`].
pub fn slice_interval(
    interval: BatchInterval,
    steps_per_round: u32,
    micro_batch: u32,
) -> Result<Vec<InnerStep>, DataError> {
    if interval.is_empty() {
        return Err(DataError::EmptyInterval);
    }
    if steps_per_round == 0 {
        return Err(DataError::ZeroSteps);
    }
    if micro_batch == 0 {
        return Err(DataError::ZeroMicroBatch);
    }
    let len = interval.len();
    let steps = u64::from(steps_per_round);
    if !len.is_multiple_of(steps) {
        return Err(DataError::IntervalNotDivisible {
            len,
            steps: steps_per_round,
        });
    }
    let per_step = len / steps;
    let mb = u64::from(micro_batch);
    let mut out = Vec::with_capacity(steps_per_round as usize);
    for h in 0..steps_per_round {
        let step_start = interval.start + u64::from(h) * per_step;
        let step_end = step_start + per_step;
        let mut micro_batches = Vec::new();
        let mut cursor = step_start;
        while cursor < step_end {
            let end = (cursor + mb).min(step_end);
            micro_batches.push(MicroBatch { start: cursor, end });
            cursor = end;
        }
        out.push(InnerStep {
            index: h,
            micro_batches,
        });
    }
    Ok(out)
}

/// One generated shard: its file name + in-memory token bytes.
pub type ShardBlob = (String, Vec<u8>);

/// A deterministic synthetic corpus generator (seeded u16 tokens) for tests.
pub struct SyntheticCorpus;

impl SyntheticCorpus {
    /// The synthetic vocabulary size (tokens are `value % VOCAB`).
    pub const VOCAB: u64 = 32_000;

    /// Deterministically generate the little-endian u16 token bytes of one shard: `tokens` tokens
    /// derived from `seed` (via splitmix64 over the token index).
    #[must_use]
    pub fn shard_bytes(seed: u64, tokens: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(tokens as usize * 2);
        for i in 0..tokens {
            let token = (splitmix64(seed ^ i) % Self::VOCAB) as u16;
            out.extend_from_slice(&token.to_le_bytes());
        }
        out
    }

    /// Generate a full corpus: `num_shards` shards of `tokens_per_shard` tokens each, returning the
    /// validated [`Manifest`] plus the in-memory shard bytes `(name, bytes)`. `tokens_per_shard`
    /// must be a multiple of `seq_len` (each shard holds whole sequences).
    pub fn generate(
        seed: u64,
        num_shards: u32,
        tokens_per_shard: u64,
        seq_len: u32,
    ) -> Result<(Manifest, Vec<ShardBlob>), DataError> {
        if seq_len == 0 {
            return Err(DataError::ZeroSeqLen);
        }
        if tokens_per_shard == 0 {
            return Err(DataError::ZeroShardTokens(0));
        }
        let mut shards = Vec::new();
        let mut blobs = Vec::new();
        for s in 0..num_shards {
            let name = format!("shard-{s:04}.bin");
            let bytes = Self::shard_bytes(seed ^ u64::from(s), tokens_per_shard);
            shards.push(ShardDesc {
                name: name.clone(),
                bytes: bytes.len() as u64,
                tokens: tokens_per_shard,
                blake3: blake3_hash(&bytes).to_hex(),
            });
            blobs.push((name, bytes));
        }
        let manifest = Manifest {
            token_width: TokenWidth::U16,
            seq_len,
            shards,
        };
        manifest.validate()?;
        Ok((manifest, blobs))
    }
}

/// A small, fast, deterministic mixing function (splitmix64) for the synthetic corpus.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Errors surfaced by the data pipeline.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DataError {
    /// The manifest JSON could not be parsed/serialized.
    #[error("manifest json error: {0}")]
    Parse(String),
    /// The manifest declared no shards.
    #[error("manifest has no shards")]
    EmptyManifest,
    /// `seq_len` was zero.
    #[error("seq_len must be non-zero")]
    ZeroSeqLen,
    /// A shard declared zero tokens.
    #[error("shard {0} has zero tokens")]
    ZeroShardTokens(usize),
    /// A shard's byte size did not equal `tokens * token_width`.
    #[error(
        "shard {shard} size mismatch: expected {expected} bytes, manifest declares {declared}"
    )]
    ShardSizeMismatch {
        /// The shard index.
        shard: usize,
        /// The byte size implied by `tokens * token_width`.
        expected: u64,
        /// The byte size the manifest declares.
        declared: u64,
    },
    /// A shard's token count was not a multiple of `seq_len` (a `BatchId` would straddle shards).
    #[error("shard {shard} tokens {tokens} not a multiple of seq_len {seq_len}")]
    ShardNotSeqAligned {
        /// The shard index.
        shard: usize,
        /// The shard's token count.
        tokens: u64,
        /// The manifest's sequence length.
        seq_len: u32,
    },
    /// A shard's blake3 field was not a valid 64-char hex digest.
    #[error("shard {0} has a malformed blake3 hash")]
    BadShardHash(usize),
    /// A `BatchId` fell outside the corpus's sequence range.
    #[error("batch {batch} out of range (total sequences {total})")]
    BatchOutOfRange {
        /// The requested batch id.
        batch: BatchId,
        /// The total number of sequences available.
        total: u64,
    },
    /// A peer's interval did not divide evenly into `steps_per_round`.
    #[error("interval of {len} sequences does not divide into {steps} steps")]
    IntervalNotDivisible {
        /// The interval length.
        len: u64,
        /// The requested step count.
        steps: u32,
    },
    /// `slice_interval` was given an empty interval.
    #[error("cannot slice an empty interval")]
    EmptyInterval,
    /// `steps_per_round` was zero.
    #[error("steps_per_round must be non-zero")]
    ZeroSteps,
    /// `micro_batch` was zero.
    #[error("micro_batch must be non-zero")]
    ZeroMicroBatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shard(name: &str, tokens: u64, width: TokenWidth) -> ShardDesc {
        ShardDesc {
            name: name.into(),
            bytes: tokens * width.bytes(),
            tokens,
            blake3: blake3_hash(name.as_bytes()).to_hex(),
        }
    }

    fn manifest(seq_len: u32, shards: Vec<ShardDesc>) -> Manifest {
        Manifest {
            token_width: TokenWidth::U16,
            seq_len,
            shards,
        }
    }

    #[test]
    fn manifest_batchid_maps() {
        // shard0: 8 tokens = 2 seqs; shard1: 12 tokens = 3 seqs; seq_len 4 -> 5 sequences total.
        let m = manifest(
            4,
            vec![
                shard("a", 8, TokenWidth::U16),
                shard("b", 12, TokenWidth::U16),
            ],
        );
        m.validate().unwrap();
        assert_eq!(m.total_sequences(), 5);
        assert_eq!(
            m.locate(0).unwrap(),
            BatchLocation {
                shard: 0,
                token_offset: 0
            }
        );
        assert_eq!(
            m.locate(1).unwrap(),
            BatchLocation {
                shard: 0,
                token_offset: 4
            }
        );
        assert_eq!(
            m.locate(2).unwrap(),
            BatchLocation {
                shard: 1,
                token_offset: 0
            }
        );
        assert_eq!(
            m.locate(3).unwrap(),
            BatchLocation {
                shard: 1,
                token_offset: 4
            }
        );
        assert_eq!(
            m.locate(4).unwrap(),
            BatchLocation {
                shard: 1,
                token_offset: 8
            }
        );
        assert!(matches!(
            m.locate(5),
            Err(DataError::BatchOutOfRange { batch: 5, total: 5 })
        ));
    }

    #[test]
    fn invalid_manifest_rejected() {
        assert_eq!(
            manifest(4, vec![]).validate(),
            Err(DataError::EmptyManifest)
        );

        let mut bad_size = manifest(4, vec![shard("a", 8, TokenWidth::U16)]);
        bad_size.shards[0].bytes = 99;
        assert!(matches!(
            bad_size.validate(),
            Err(DataError::ShardSizeMismatch { shard: 0, .. })
        ));

        // 10 tokens is not a multiple of seq_len 4.
        let unaligned = manifest(4, vec![shard("a", 10, TokenWidth::U16)]);
        assert!(matches!(
            unaligned.validate(),
            Err(DataError::ShardNotSeqAligned { shard: 0, .. })
        ));

        let mut bad_hash = manifest(4, vec![shard("a", 8, TokenWidth::U16)]);
        bad_hash.shards[0].blake3 = "not-hex".into();
        assert_eq!(bad_hash.validate(), Err(DataError::BadShardHash(0)));
    }

    #[test]
    fn interval_slices_into_h_steps() {
        // 24 sequences, 3 steps, micro-batch 4 -> each step is 8 seqs = 2 micro-batches of 4.
        let steps = slice_interval(BatchInterval::new(0, 24), 3, 4).unwrap();
        assert_eq!(steps.len(), 3);
        for (h, step) in steps.iter().enumerate() {
            assert_eq!(step.index, h as u32);
            assert_eq!(step.micro_batches.len(), 2);
            let base = h as u64 * 8;
            assert_eq!(
                step.micro_batches[0],
                MicroBatch {
                    start: base,
                    end: base + 4
                }
            );
            assert_eq!(
                step.micro_batches[1],
                MicroBatch {
                    start: base + 4,
                    end: base + 8
                }
            );
        }

        // A ragged micro-batch: 8 seqs, 2 steps (4 each), mb=3 -> [0,3),[3,4) then [4,7),[7,8).
        let ragged = slice_interval(BatchInterval::new(0, 8), 2, 3).unwrap();
        assert_eq!(ragged[0].micro_batches.last().unwrap().end, 4);
        assert_eq!(ragged[1].micro_batches[0], MicroBatch { start: 4, end: 7 });
    }

    #[test]
    fn slice_interval_rejects_bad_inputs() {
        assert_eq!(
            slice_interval(BatchInterval::new(5, 5), 2, 1),
            Err(DataError::EmptyInterval)
        );
        assert!(matches!(
            slice_interval(BatchInterval::new(0, 10), 3, 4),
            Err(DataError::IntervalNotDivisible { len: 10, steps: 3 })
        ));
        assert_eq!(
            slice_interval(BatchInterval::new(0, 8), 0, 4),
            Err(DataError::ZeroSteps)
        );
        assert_eq!(
            slice_interval(BatchInterval::new(0, 8), 2, 0),
            Err(DataError::ZeroMicroBatch)
        );
    }

    #[test]
    fn synthetic_corpus_is_deterministic() {
        let a = SyntheticCorpus::shard_bytes(0xDAE0_7E57, 16);
        let b = SyntheticCorpus::shard_bytes(0xDAE0_7E57, 16);
        assert_eq!(a, b, "same seed -> same bytes");
        assert_ne!(
            a,
            SyntheticCorpus::shard_bytes(0xDAE0_7E58, 16),
            "different seed -> different bytes"
        );
        assert_eq!(a.len(), 16 * 2, "u16 width");
    }

    #[test]
    fn synthetic_corpus_generates_valid_manifest() {
        let (manifest, blobs) = SyntheticCorpus::generate(0xDAE0_7E57, 3, 32, 8).unwrap();
        manifest.validate().unwrap();
        assert_eq!(manifest.shards.len(), 3);
        assert_eq!(manifest.total_sequences(), 3 * (32 / 8));
        // Every manifest hash matches the generated shard bytes (the fetch-time integrity check).
        for (desc, (name, bytes)) in manifest.shards.iter().zip(blobs.iter()) {
            assert_eq!(&desc.name, name);
            assert_eq!(desc.blake3, blake3_hash(bytes).to_hex());
        }
        // Mapping works end-to-end over the generated corpus.
        assert_eq!(
            manifest.locate(4).unwrap(),
            BatchLocation {
                shard: 1,
                token_offset: 0
            }
        );
    }
}
