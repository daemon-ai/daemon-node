// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The pre-tokenized shard data pipeline (spec §8) + `BatchId` mapping (§6.3).
//!
//! **HARNESS-ERA** (`#[cfg(any(test, feature = "harness"))]`): this legacy JSON-manifest corpus
//! pipeline feeds the retained `RoundEngine` and the harness suites only. The production corpus
//! contract is chunk-addressed — shards are named by their committed chunk-map fold identity,
//! fetched through `data@2`, and chunk-verified by the run pump — and no production path reads
//! `manifest.json`.
//!
//! Corpora are pre-tokenized offline into fixed-width shards (u16/u32 token streams); a
//! [`Manifest`] (`manifest.json`) lists the shards, their sizes, token counts, and a blake3 per
//! shard. Peers map [`BatchId`] intervals to `(shard, offset)` **purely locally** ([`Manifest::locate`])
//! and slice their assigned interval into `steps_per_round` inner steps × micro-batches
//! ([`slice_interval`]) — no per-batch RPC.
//!
//! [`SyntheticCorpus`] generates a deterministic seeded corpus (u16 tokens) for tests, so the round
//! loop and the worker have a data source with no external download.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use daemon_vhc_proto::{blake3_hash, Hash};

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
///
/// The provenance fields (`tokenizer`/`tokenizer_revision`/`dataset`/`dataset_revision`) are an
/// **additive** extension (M1): `#[serde(default)]` + `skip_serializing_if` keeps every
/// earlier manifest (which carries only `token_width`/`seq_len`/`shards`) valid, and a manifest
/// written without provenance is byte-identical to the old shape. They record how the shards were
/// produced (`xtask tokenize-corpus`) so a run is reproducible and auditable (spec §8/§9).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// The token element width shared by every shard.
    pub token_width: TokenWidth,
    /// The sequence length (tokens per training sequence).
    pub seq_len: u32,
    /// The shards, in data-window order.
    pub shards: Vec<ShardDesc>,
    /// The tokenizer identity the corpus was tokenized with (e.g. `"gpt2"`), if recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
    /// The pinned tokenizer revision (HF commit SHA / tag), if recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_revision: Option<String>,
    /// The source dataset identity (e.g. `"roneneldan/TinyStories"`), if recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    /// The pinned dataset revision (HF commit SHA / tag), if recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_revision: Option<String>,
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

    /// The set of shard indices covering the sequence window `[start_seq, start_seq + seq_count)`,
    /// wrapping modulo [`Manifest::total_sequences`] (a peer stages **only** these shards — the §8
    /// RAM-bounded windowing, the fleet artifact-distribution path). `seq_count == 0` (or `>= total`) means the whole corpus.
    ///
    /// `start_seq` is taken modulo the total, matching [`Corpus::sequence`]'s wrap. A window that
    /// runs off the end wraps to the front (two contiguous ranges), so a run whose data cursor wraps
    /// still stages the shards it will read.
    #[must_use]
    pub fn shards_covering(&self, start_seq: u64, seq_count: u64) -> BTreeSet<usize> {
        let total = self.total_sequences();
        if total == 0 {
            return BTreeSet::new();
        }
        if seq_count == 0 || seq_count >= total {
            return (0..self.shards.len()).collect();
        }
        let seq_len = u64::from(self.seq_len);
        // Per-shard sequence bounds `[start, end)` in the global sequence index space.
        let mut bounds = Vec::with_capacity(self.shards.len());
        let mut cum = 0u64;
        for s in &self.shards {
            let seqs = s.tokens / seq_len;
            bounds.push((cum, cum + seqs));
            cum += seqs;
        }
        let start = start_seq % total;
        // The window is one contiguous range, or two when it wraps past `total`.
        let ranges: [(u64, u64); 2] = if start + seq_count <= total {
            [(start, start + seq_count), (0, 0)]
        } else {
            [(start, total), (0, (start + seq_count) - total)]
        };
        let mut out = BTreeSet::new();
        for (rs, re) in ranges {
            if rs >= re {
                continue;
            }
            for (i, (bs, be)) in bounds.iter().enumerate() {
                if *bs < re && rs < *be {
                    out.insert(i);
                }
            }
        }
        out
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
            tokenizer: None,
            tokenizer_revision: None,
            dataset: None,
            dataset_revision: None,
        };
        manifest.validate()?;
        Ok((manifest, blobs))
    }
}

/// An in-memory pre-tokenized corpus: a validated [`Manifest`] plus the shard bytes it describes.
///
/// The round engine reads a peer's assigned micro-batches out of this without any external
/// download. [`Corpus::sequence`] maps a [`BatchId`] to its `seq_len` token ids (wrapping the id
/// into range, so a monotonically-advancing round cursor never runs off the end of a fixed test
/// corpus).
#[derive(Clone, Debug)]
pub struct Corpus {
    manifest: Manifest,
    /// One slot per manifest shard. `Some` = resident bytes; `None` = not staged (a **windowed**
    /// corpus stages only the shards its assignment touches — the fleet artifact-distribution path). A fully-resident corpus
    /// (`from_parts`/`synthetic`) has every slot `Some`.
    shards: Vec<Option<Vec<u8>>>,
}

impl Corpus {
    /// Build a corpus from a validated manifest + the shard bytes it describes — the **real** data
    /// path (the runtime fetches shards per the manifest and constructs this; `SyntheticCorpus` is
    /// the CI stand-in). The shards must match the manifest 1:1 (count + per-shard byte length), and
    /// each shard's content is blake3-verified against its [`ShardDesc`] (the §8 integrity check), so
    /// a corrupt or reordered shard is rejected here rather than surfacing as NaN downstream.
    pub fn from_parts(manifest: Manifest, shards: Vec<Vec<u8>>) -> Result<Self, DataError> {
        manifest.validate()?;
        if shards.len() != manifest.shards.len() {
            return Err(DataError::ShardCountMismatch {
                manifest: manifest.shards.len(),
                provided: shards.len(),
            });
        }
        for (i, (desc, bytes)) in manifest.shards.iter().zip(shards.iter()).enumerate() {
            verify_shard(i, desc, bytes)?;
        }
        Ok(Self {
            manifest,
            shards: shards.into_iter().map(Some).collect(),
        })
    }

    /// Build a **windowed** corpus: a validated manifest plus only the shards a peer has staged
    /// (`resident`: shard-index → bytes) — the fetch-only-assigned-shards distribution path (spec §8).
    /// Each resident shard is blake3-verified against its [`ShardDesc`]; a non-resident shard is left
    /// `None` and addressing it via [`Corpus::sequence`] is a typed [`DataError::ShardNotResident`]
    /// (never a silent NaN). Use [`Manifest::shards_covering`] to compute which indices to stage.
    pub fn windowed(
        manifest: Manifest,
        resident: BTreeMap<usize, Vec<u8>>,
    ) -> Result<Self, DataError> {
        manifest.validate()?;
        let mut shards: Vec<Option<Vec<u8>>> = vec![None; manifest.shards.len()];
        for (idx, bytes) in resident {
            let desc = manifest
                .shards
                .get(idx)
                .ok_or(DataError::ShardIndexOutOfRange {
                    shard: idx,
                    total: manifest.shards.len(),
                })?;
            verify_shard(idx, desc, &bytes)?;
            shards[idx] = Some(bytes);
        }
        Ok(Self { manifest, shards })
    }

    /// Build a deterministic synthetic corpus (`num_shards` × `tokens_per_shard` u16 tokens).
    pub fn synthetic(
        seed: u64,
        num_shards: u32,
        tokens_per_shard: u64,
        seq_len: u32,
    ) -> Result<Self, DataError> {
        let (manifest, blobs) =
            SyntheticCorpus::generate(seed, num_shards, tokens_per_shard, seq_len)?;
        let shards = blobs.into_iter().map(|(_, bytes)| Some(bytes)).collect();
        Ok(Self { manifest, shards })
    }

    /// The number of shards resident in memory (`== manifest.shards.len()` for a full corpus; fewer
    /// for a windowed one).
    #[must_use]
    pub fn resident_shards(&self) -> usize {
        self.shards.iter().filter(|s| s.is_some()).count()
    }

    /// The corpus manifest.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The total number of whole sequences in the corpus.
    #[must_use]
    pub fn total_sequences(&self) -> u64 {
        self.manifest.total_sequences()
    }

    /// The token ids of the sequence at `batch` (wrapped into range), as `u32`s.
    pub fn sequence(&self, batch: BatchId) -> Result<Vec<u32>, DataError> {
        let total = self.total_sequences();
        if total == 0 {
            return Err(DataError::EmptyManifest);
        }
        let loc = self.manifest.locate(batch % total)?;
        let seq_len = self.manifest.seq_len as usize;
        let width = self.manifest.token_width.bytes() as usize;
        let shard = self.shards[loc.shard]
            .as_deref()
            .ok_or(DataError::ShardNotResident { shard: loc.shard })?;
        let start = loc.token_offset as usize * width;
        let mut tokens = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let off = start + i * width;
            let token = match self.manifest.token_width {
                TokenWidth::U16 => u32::from(u16::from_le_bytes([shard[off], shard[off + 1]])),
                TokenWidth::U32 => {
                    u32::from_le_bytes([shard[off], shard[off + 1], shard[off + 2], shard[off + 3]])
                }
            };
            tokens.push(token);
        }
        Ok(tokens)
    }
}

/// Verify one shard's bytes against its [`ShardDesc`] (byte length + blake3 content hash) — the §8
/// fetch-time integrity check shared by [`Corpus::from_parts`] and [`Corpus::windowed`].
fn verify_shard(index: usize, desc: &ShardDesc, bytes: &[u8]) -> Result<(), DataError> {
    if bytes.len() as u64 != desc.bytes {
        return Err(DataError::ShardSizeMismatch {
            shard: index,
            expected: desc.bytes,
            declared: bytes.len() as u64,
        });
    }
    if blake3_hash(bytes).to_hex().as_str() != desc.blake3 {
        return Err(DataError::ShardHashMismatch { shard: index });
    }
    Ok(())
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
    /// The number of provided shards did not match the manifest ([`Corpus::from_parts`]).
    #[error("manifest declares {manifest} shards but {provided} were provided")]
    ShardCountMismatch {
        /// The shard count the manifest declares.
        manifest: usize,
        /// The number of shard blobs provided.
        provided: usize,
    },
    /// A provided shard's content blake3 did not match its manifest entry ([`Corpus::from_parts`]).
    #[error("shard {shard} content blake3 does not match the manifest")]
    ShardHashMismatch {
        /// The shard index.
        shard: usize,
    },
    /// A [`Corpus::windowed`] resident entry named a shard index outside the manifest.
    #[error("resident shard index {shard} out of range (manifest has {total} shards)")]
    ShardIndexOutOfRange {
        /// The offending shard index.
        shard: usize,
        /// The number of shards in the manifest.
        total: usize,
    },
    /// A [`BatchId`] addressed a shard that was not staged in a windowed corpus ([`Corpus::windowed`]).
    /// The peer must stage every shard its assignment touches (see [`Manifest::shards_covering`]).
    #[error("shard {shard} is not resident (windowed corpus: stage the assigned shards first)")]
    ShardNotResident {
        /// The non-resident shard index.
        shard: usize,
    },
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
            tokenizer: None,
            tokenizer_revision: None,
            dataset: None,
            dataset_revision: None,
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
    fn corpus_sequence_reads_tokens_and_wraps() {
        let corpus = Corpus::synthetic(0xDAE0_7E57, 2, 32, 8).unwrap();
        assert_eq!(corpus.total_sequences(), 2 * (32 / 8));
        let s0 = corpus.sequence(0).unwrap();
        assert_eq!(s0.len(), 8, "one sequence = seq_len tokens");
        // Wrapping: sequence(total) == sequence(0).
        assert_eq!(corpus.sequence(corpus.total_sequences()).unwrap(), s0);
        // Distinct sequences differ (deterministic synthetic tokens).
        assert_ne!(corpus.sequence(1).unwrap(), s0);
    }

    #[test]
    fn manifest_provenance_is_additive_and_back_compatible() {
        // A earlier manifest (no provenance keys at all) still parses (RUN-3 back-compat).
        let old = r#"{
            "token_width": "u16",
            "seq_len": 4,
            "shards": [{"name":"a","bytes":16,"tokens":8,
                "blake3":"0000000000000000000000000000000000000000000000000000000000000000"}]
        }"#;
        let m = Manifest::from_json(old).unwrap();
        assert_eq!(m.tokenizer, None);
        assert_eq!(m.dataset, None);
        // A provenance-less manifest serializes WITHOUT the new keys (byte-identical old shape).
        let json = m.to_json().unwrap();
        assert!(!json.contains("tokenizer"), "no tokenizer key when unset");
        assert!(!json.contains("dataset"), "no dataset key when unset");

        // A provenance-carrying manifest round-trips and preserves every field.
        let mut prov = m.clone();
        prov.tokenizer = Some("gpt2".into());
        prov.tokenizer_revision = Some("607a30d783dfa663caf39e06633721c8d4cfcd7e".into());
        prov.dataset = Some("roneneldan/TinyStories".into());
        prov.dataset_revision = Some("main".into());
        let round = Manifest::from_json(&prov.to_json().unwrap()).unwrap();
        assert_eq!(round, prov);
        // A new-shape manifest read by any consumer still validates structurally.
        round.validate().unwrap();
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

    #[test]
    fn shards_covering_selects_only_the_window() {
        // 4 shards × 4 seqs each (16 tokens, seq_len 4) → 16 sequences total.
        let m = manifest(
            4,
            vec![
                shard("a", 16, TokenWidth::U16),
                shard("b", 16, TokenWidth::U16),
                shard("c", 16, TokenWidth::U16),
                shard("d", 16, TokenWidth::U16),
            ],
        );
        m.validate().unwrap();
        assert_eq!(m.total_sequences(), 16);
        // Window [0,8) covers shards 0,1 only — NOT the whole corpus.
        assert_eq!(
            m.shards_covering(0, 8),
            BTreeSet::from([0, 1]),
            "a peer stages only the shards its window touches"
        );
        // A window that spans a boundary picks up both neighbours.
        assert_eq!(m.shards_covering(3, 3), BTreeSet::from([0, 1]));
        // Wrap-around: [14,18) → seqs 14,15 (shard 3) + 0,1 (shard 0).
        assert_eq!(m.shards_covering(14, 4), BTreeSet::from([0, 3]));
        // seq_count 0 or >= total means the whole corpus.
        assert_eq!(m.shards_covering(0, 0), BTreeSet::from([0, 1, 2, 3]));
        assert_eq!(m.shards_covering(0, 999), BTreeSet::from([0, 1, 2, 3]));
    }

    #[test]
    fn windowed_corpus_serves_resident_and_rejects_absent() {
        // A real (full) synthetic corpus gives us the manifest + true shard bytes.
        let (manifest, blobs) = SyntheticCorpus::generate(0xD1CE, 4, 32, 8).unwrap();
        // Stage only shards 0 and 2 (a windowed subset).
        let mut resident = BTreeMap::new();
        resident.insert(0usize, blobs[0].1.clone());
        resident.insert(2usize, blobs[2].1.clone());
        let windowed = Corpus::windowed(manifest.clone(), resident).unwrap();
        assert_eq!(windowed.resident_shards(), 2);

        // total_sequences is the manifest's (all shards) → 4 × 4 = 16.
        assert_eq!(windowed.total_sequences(), 16);
        // Shard 0 covers batches 0..4; a resident batch reads cleanly + matches the full corpus.
        let full =
            Corpus::from_parts(manifest, blobs.into_iter().map(|(_, b)| b).collect()).unwrap();
        assert_eq!(windowed.sequence(0).unwrap(), full.sequence(0).unwrap());
        // Shard 1 (batches 4..8) is NOT resident → typed ShardNotResident, never a silent read.
        assert!(matches!(
            windowed.sequence(4),
            Err(DataError::ShardNotResident { shard: 1 })
        ));
    }

    #[test]
    fn windowed_corpus_rejects_bad_index_and_tampered_shard() {
        let (manifest, blobs) = SyntheticCorpus::generate(0xD1CE, 2, 32, 8).unwrap();
        // Out-of-range index.
        let mut bad_idx = BTreeMap::new();
        bad_idx.insert(9usize, blobs[0].1.clone());
        assert!(matches!(
            Corpus::windowed(manifest.clone(), bad_idx),
            Err(DataError::ShardIndexOutOfRange { shard: 9, .. })
        ));
        // Tampered bytes for a valid index.
        let mut tampered = BTreeMap::new();
        tampered.insert(0usize, vec![0xFFu8; blobs[0].1.len()]);
        assert!(matches!(
            Corpus::windowed(manifest, tampered),
            Err(DataError::ShardHashMismatch { shard: 0 })
        ));
    }
}
