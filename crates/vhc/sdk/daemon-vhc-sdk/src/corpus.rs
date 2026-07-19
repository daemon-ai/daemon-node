// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Corpus windowing as **module policy** — the SDK home of the data-window math (Phase B,
//! refactor §6 "corpus windowing, by layer").
//!
//! Architecture §3.2 (`data@`): "windowing, batching, and data scheduling are worker-module
//! policy, not host features"; §8's authority table places "data windowing/scheduling" with the
//! worker module. This module is that policy's SDK home: the pure windowing math of the host's
//! v1 pipeline (`daemon-vhc-session::data`) — manifest → `BatchId` location → shard-window
//! coverage → token extraction — ported behind the wasm boundary, byte-for-byte in semantics
//! (pinned by the cross-layer equivalence oracle in `tests/daemon-vhc-e2e`).
//!
//! **The layering, stated honestly for the bridge era:**
//!
//! - **Host mechanism** — fetch-by-hash: the manifest object and the shards it names are
//!   content-addressed artifacts the host fetches and blake3-verifies (the worker's existing
//!   `build_corpus` fetch path; the `data@2` fetch imports expose the same mechanism to guests).
//!   The host never decides *which* data trains — it moves verified bytes.
//! - **Module policy** — this code: which sequences form the round's batches
//!   (`daemon-vhc-sdk-rounds`' assignment slicing), which shards the active window needs
//!   ([`Manifest::shards_covering`] — the prefetch *request* is policy even when the fetch is
//!   mechanism), and how token windows are cut from shard bytes ([`CorpusWindow::sequence`]).
//! - Under the transitional `tabi@1` bridge, batch *content* still reaches the guest through
//!   host staging (`read_back` kind 1) because tensors are host-side; the plumbing mirrors the
//!   module's window arithmetic (same proto assignment math, dual-compiled). When `data@2`
//!   fetch + `compute@2` land, a module fetches shards itself and this code is the whole path.
//!
//! The v1 host pipeline (`daemon-vhc-session::data`) stays untouched for the retained v1 driver
//! until the Phase-E sunset; `batch_tokens@1` stays v1-compat byte-identical.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// A sequence index over the corpus's data window (the proto/engine `BatchId` vocabulary).
pub type BatchId = u64;

/// The token element width of a shard's fixed-width stream (u16/u32).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TokenWidth {
    /// 16-bit tokens (vocab ≤ 65 536).
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

/// The pre-tokenized corpus manifest — the same `manifest.json` schema the v1 host pipeline
/// reads (provenance fields additive + optional, exactly as there).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// The token element width shared by every shard.
    pub token_width: TokenWidth,
    /// The sequence length (tokens per training sequence).
    pub seq_len: u32,
    /// The shards, in data-window order.
    pub shards: Vec<ShardDesc>,
    /// The tokenizer identity, if recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
    /// The pinned tokenizer revision, if recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_revision: Option<String>,
    /// The source dataset identity, if recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    /// The pinned dataset revision, if recorded.
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

/// Whether `s` is a well-formed blake3 hex digest (64 hex chars).
fn is_blake3_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

impl Manifest {
    /// Parse + validate a `manifest.json` document (the artifact a module `data@`-fetches by
    /// the envelope's pinned hash).
    ///
    /// # Errors
    /// [`CorpusError`] on parse or structural failure.
    pub fn from_json(json: &str) -> Result<Self, CorpusError> {
        let manifest: Manifest =
            serde_json::from_str(json).map_err(|e| CorpusError::Parse(e.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validate structural + numeric consistency (the v1 pipeline's rules, unchanged).
    ///
    /// # Errors
    /// [`CorpusError`] naming the offending shard/field.
    pub fn validate(&self) -> Result<(), CorpusError> {
        if self.shards.is_empty() {
            return Err(CorpusError::EmptyManifest);
        }
        if self.seq_len == 0 {
            return Err(CorpusError::ZeroSeqLen);
        }
        let width = self.token_width.bytes();
        let seq_len = u64::from(self.seq_len);
        for (i, shard) in self.shards.iter().enumerate() {
            if shard.tokens == 0 {
                return Err(CorpusError::ZeroShardTokens(i));
            }
            if shard.bytes != shard.tokens * width {
                return Err(CorpusError::ShardSizeMismatch {
                    shard: i,
                    expected: shard.tokens * width,
                    declared: shard.bytes,
                });
            }
            // Each shard holds whole sequences, so a BatchId never straddles a shard boundary.
            if shard.tokens % seq_len != 0 {
                return Err(CorpusError::ShardNotSeqAligned {
                    shard: i,
                    tokens: shard.tokens,
                    seq_len: self.seq_len,
                });
            }
            if !is_blake3_hex(&shard.blake3) {
                return Err(CorpusError::BadShardHash(i));
            }
        }
        Ok(())
    }

    /// The total number of tokens across all shards.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.shards.iter().map(|s| s.tokens).sum()
    }

    /// The total number of whole sequences across all shards.
    #[must_use]
    pub fn total_sequences(&self) -> u64 {
        self.total_tokens() / u64::from(self.seq_len)
    }

    /// Map a [`BatchId`] to its `(shard, token_offset)` — identical arithmetic to the v1
    /// pipeline's `locate` (the equivalence oracle pins it).
    ///
    /// # Errors
    /// [`CorpusError::BatchOutOfRange`] past the last sequence.
    pub fn locate(&self, batch: BatchId) -> Result<BatchLocation, CorpusError> {
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
        Err(CorpusError::BatchOutOfRange {
            batch,
            total: self.total_sequences(),
        })
    }

    /// The shard indices covering `[start_seq, start_seq + seq_count)` (wrapping) — the
    /// module's **prefetch request**: policy decides which shards its window needs; the host
    /// mechanism fetches + verifies them. `seq_count == 0` (or `>= total`) means every shard.
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
        let mut bounds = Vec::with_capacity(self.shards.len());
        let mut cum = 0u64;
        for s in &self.shards {
            let seqs = s.tokens / seq_len;
            bounds.push((cum, cum + seqs));
            cum += seqs;
        }
        let start = start_seq % total;
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

/// A windowed corpus: the validated manifest plus the shard bytes the module's window actually
/// staged (fetched-by-hash through host mechanism, resident in guest/module memory). The
/// module-policy analogue of the v1 host pipeline's windowed `Corpus`.
#[derive(Clone, Debug)]
pub struct CorpusWindow {
    manifest: Manifest,
    /// One slot per manifest shard; `None` = not staged by this window.
    shards: Vec<Option<Vec<u8>>>,
}

impl CorpusWindow {
    /// Assemble a window from the manifest plus the resident shards (shard index → bytes),
    /// verifying every resident shard's byte length and blake3 against its manifest entry — the
    /// fetch-time integrity rule, applied at the policy layer too (never trust a staging layer
    /// you didn't have to).
    ///
    /// # Errors
    /// [`CorpusError`] on a bad index, length mismatch, or content-hash mismatch.
    pub fn assemble(
        manifest: Manifest,
        resident: BTreeMap<usize, Vec<u8>>,
    ) -> Result<Self, CorpusError> {
        manifest.validate()?;
        let mut shards: Vec<Option<Vec<u8>>> = vec![None; manifest.shards.len()];
        for (idx, bytes) in resident {
            let desc = manifest
                .shards
                .get(idx)
                .ok_or(CorpusError::ShardIndexOutOfRange {
                    shard: idx,
                    total: manifest.shards.len(),
                })?;
            if bytes.len() as u64 != desc.bytes {
                return Err(CorpusError::ShardSizeMismatch {
                    shard: idx,
                    expected: desc.bytes,
                    declared: bytes.len() as u64,
                });
            }
            let got = blake3::hash(&bytes).to_hex().to_string();
            if got != desc.blake3 {
                return Err(CorpusError::ShardHashMismatch { shard: idx });
            }
            shards[idx] = Some(bytes);
        }
        Ok(Self { manifest, shards })
    }

    /// The corpus manifest.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The number of resident shards.
    #[must_use]
    pub fn resident_shards(&self) -> usize {
        self.shards.iter().filter(|s| s.is_some()).count()
    }

    /// The total number of whole sequences in the corpus (all shards, resident or not).
    #[must_use]
    pub fn total_sequences(&self) -> u64 {
        self.manifest.total_sequences()
    }

    /// The token ids of the sequence at `batch` (wrapped into range), as `u32`s — identical
    /// extraction arithmetic to the v1 pipeline's `sequence` (oracle-pinned). A non-resident
    /// shard is a typed error, never a silent read.
    ///
    /// # Errors
    /// [`CorpusError::ShardNotResident`] when the window did not stage the shard.
    pub fn sequence(&self, batch: BatchId) -> Result<Vec<u32>, CorpusError> {
        let total = self.total_sequences();
        if total == 0 {
            return Err(CorpusError::EmptyManifest);
        }
        let loc = self.manifest.locate(batch % total)?;
        let seq_len = self.manifest.seq_len as usize;
        let width = self.manifest.token_width.bytes() as usize;
        let shard = self.shards[loc.shard]
            .as_deref()
            .ok_or(CorpusError::ShardNotResident { shard: loc.shard })?;
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

/// Errors surfaced by the corpus-window policy layer. Hand-rolled `Display` (this crate keeps
/// its dependency set minimal, like the contracts crates it sits beside).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CorpusError {
    /// The manifest JSON could not be parsed.
    Parse(String),
    /// The manifest declared no shards.
    EmptyManifest,
    /// `seq_len` was zero.
    ZeroSeqLen,
    /// A shard declared zero tokens.
    ZeroShardTokens(usize),
    /// A shard's byte size did not equal `tokens * token_width` (or the staged bytes' length).
    ShardSizeMismatch {
        /// The shard index.
        shard: usize,
        /// The expected byte size.
        expected: u64,
        /// The declared/observed byte size.
        declared: u64,
    },
    /// A shard's token count was not a multiple of `seq_len`.
    ShardNotSeqAligned {
        /// The shard index.
        shard: usize,
        /// The shard's token count.
        tokens: u64,
        /// The manifest's sequence length.
        seq_len: u32,
    },
    /// A shard's blake3 field was not a valid 64-char hex digest.
    BadShardHash(usize),
    /// A staged shard's content blake3 did not match its manifest entry.
    ShardHashMismatch {
        /// The shard index.
        shard: usize,
    },
    /// A resident entry named a shard index outside the manifest.
    ShardIndexOutOfRange {
        /// The offending shard index.
        shard: usize,
        /// The number of shards in the manifest.
        total: usize,
    },
    /// A [`BatchId`] addressed a shard this window did not stage.
    ShardNotResident {
        /// The non-resident shard index.
        shard: usize,
    },
    /// A `BatchId` fell outside the corpus's sequence range.
    BatchOutOfRange {
        /// The requested batch id.
        batch: BatchId,
        /// The total number of sequences available.
        total: u64,
    },
    /// An assignment derivation named a peer slot outside the roster.
    AssignmentOutOfRange {
        /// The requested peer index.
        peer: u32,
        /// The roster size.
        roster: u32,
    },
}

impl std::fmt::Display for CorpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "manifest json error: {e}"),
            Self::EmptyManifest => write!(f, "manifest has no shards"),
            Self::ZeroSeqLen => write!(f, "seq_len must be non-zero"),
            Self::ZeroShardTokens(i) => write!(f, "shard {i} has zero tokens"),
            Self::ShardSizeMismatch {
                shard,
                expected,
                declared,
            } => write!(
                f,
                "shard {shard} size mismatch: expected {expected} bytes, got {declared}"
            ),
            Self::ShardNotSeqAligned {
                shard,
                tokens,
                seq_len,
            } => write!(
                f,
                "shard {shard} tokens {tokens} not a multiple of seq_len {seq_len}"
            ),
            Self::BadShardHash(i) => write!(f, "shard {i} has a malformed blake3 hash"),
            Self::ShardHashMismatch { shard } => {
                write!(
                    f,
                    "shard {shard} content blake3 does not match the manifest"
                )
            }
            Self::ShardIndexOutOfRange { shard, total } => {
                write!(f, "shard index {shard} out of range ({total} shards)")
            }
            Self::ShardNotResident { shard } => {
                write!(f, "shard {shard} is not resident in this window")
            }
            Self::BatchOutOfRange { batch, total } => {
                write!(f, "batch {batch} out of range (total sequences {total})")
            }
            Self::AssignmentOutOfRange { peer, roster } => {
                write!(f, "peer index {peer} out of range (roster size {roster})")
            }
        }
    }
}

impl std::error::Error for CorpusError {}

// ---- the chunk-addressed corpus contract (module policy over the proto mechanism) ---------------
//
// The production data path: a canonical-CBOR [`CorpusManifest`] (genesis-pinned by hash) whose
// shards are chunk-addressed fold identities. Everything below is MODULE policy — which
// sequences this peer trains (deterministic assignment with a per-epoch reshuffle), which byte
// ranges those sequences need (windowing), and the `data@2` calls that fetch them — while the
// host stays mechanism-only (registered chunk maps, covering-span service, chunk verification).
// The JSON `Manifest` above is the legacy harness-era schema and retires with its consumers.

pub use daemon_vhc_proto::corpus::{
    chunk_count, covering_span, ChunkMap, CorpusManifest, Endianness, SequenceBoundary, ShardEntry,
    TokenizerId, CORPUS_DEFAULT_CHUNK_SIZE, CORPUS_MANIFEST_FORMAT,
};
use daemon_vhc_proto::domains::ASSIGN_SALT;
use daemon_vhc_proto::{blake3_hash, to_canonical_vec};

/// The deterministic inputs a module's data assignment derives from: run identity, epoch, and
/// this peer's slot in the epoch's ordered trainer roster. Every peer computes every peer's
/// assignment from the same inputs — no host involvement, no coordination round-trip.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssignmentParams {
    /// The run's cryptographic identity (the genesis hash).
    pub genesis_hash: [u8; 32],
    /// The epoch — the reshuffle key: a new epoch re-derives a fresh layout (ratified
    /// per-epoch reshuffle).
    pub epoch: u64,
    /// The number of peers sharing the trainer role this epoch.
    pub roster_size: u32,
    /// This peer's slot in the epoch's ordered roster (`< roster_size`).
    pub peer_index: u32,
}

/// One peer's epoch assignment: sequence-id ranges over `[0, total_sequences)`, disjoint across
/// `peer_index` and jointly covering the whole corpus.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Assignment {
    /// The assigned sequence ids, as at most two contiguous ranges (the epoch rotation may
    /// wrap one stripe around the corpus end).
    pub sequences: Vec<core::ops::Range<u64>>,
}

impl Assignment {
    /// The number of sequences assigned.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.sequences.iter().map(|r| r.end - r.start).sum()
    }

    /// Whether the assignment is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate the assigned sequence ids in training order.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.sequences.iter().flat_map(Clone::clone)
    }
}

/// The epoch's assignment seed: `blake3(ASSIGN_SALT ‖ genesis_hash ‖ u64le(epoch))` — the same
/// domain-separated shuffle family as the round assignment math, keyed by run + epoch so every
/// epoch re-lays the corpus (ratified per-epoch reshuffle).
fn assignment_seed(genesis_hash: &[u8; 32], epoch: u64) -> [u8; 32] {
    let mut buf = Vec::with_capacity(ASSIGN_SALT.len() + 32 + 8);
    buf.extend_from_slice(ASSIGN_SALT);
    buf.extend_from_slice(genesis_hash);
    buf.extend_from_slice(&epoch.to_le_bytes());
    blake3_hash(&buf).0
}

/// A hash-derived uniform index in `[0, bound)` (bound > 0): `blake3(seed ‖ u64le(label))`
/// first 8 bytes LE, reduced. Self-contained (no RNG state) so the derivation is auditable
/// from the seed alone.
fn seeded_index(seed: &[u8; 32], label: u64, bound: u64) -> u64 {
    let mut buf = Vec::with_capacity(40);
    buf.extend_from_slice(seed);
    buf.extend_from_slice(&label.to_le_bytes());
    let h = blake3_hash(&buf);
    let mut le = [0u8; 8];
    le.copy_from_slice(&h.0[..8]);
    u64::from_le_bytes(le) % bound
}

/// Derive this peer's deterministic epoch assignment (module policy — ratified construction):
///
/// 1. the epoch seed rotates the sequence space by `offset = seed_index % total`;
/// 2. the roster's stripe order is a seed-driven (hash-based Fisher–Yates) permutation of
///    `[0, roster_size)` — which stripe a peer holds reshuffles every epoch;
/// 3. stripes partition the rotated space contiguously with largest-remainder sizing, so the
///    full roster covers every sequence exactly once and stripes are disjoint by construction.
///
/// Identical inputs give identical output on every peer; distinct `peer_index` give disjoint
/// coverage; the union over the roster is the whole corpus.
///
/// # Errors
/// [`CorpusError::AssignmentOutOfRange`] when `peer_index >= roster_size` or the roster is
/// empty; [`CorpusError::EmptyManifest`] for a corpus with no sequences.
pub fn derive_assignment(
    manifest: &CorpusManifest,
    params: &AssignmentParams,
) -> Result<Assignment, CorpusError> {
    if params.roster_size == 0 || params.peer_index >= params.roster_size {
        return Err(CorpusError::AssignmentOutOfRange {
            peer: params.peer_index,
            roster: params.roster_size,
        });
    }
    let total = manifest.total_sequences();
    if total == 0 {
        return Err(CorpusError::EmptyManifest);
    }
    let seed = assignment_seed(&params.genesis_hash, params.epoch);
    let offset = seeded_index(&seed, u64::MAX, total);

    // The stripe permutation (hash-based Fisher–Yates over the roster slots).
    let n = params.roster_size as usize;
    let mut perm: Vec<u32> = (0..params.roster_size).collect();
    for j in (1..n).rev() {
        let k = seeded_index(&seed, j as u64, (j + 1) as u64) as usize;
        perm.swap(j, k);
    }
    let stripe = perm[params.peer_index as usize] as u64;

    // Largest-remainder equal sizing: the first (total % n) stripes carry one extra sequence.
    let n64 = params.roster_size as u64;
    let base = total / n64;
    let extra = total % n64;
    let start_rot = stripe * base + stripe.min(extra);
    let size = base + u64::from(stripe < extra);
    if size == 0 {
        return Ok(Assignment::default());
    }

    // Un-rotate into real sequence ids: the stripe [start_rot, start_rot+size) maps to
    // (i + offset) % total — one range, or two when it wraps the corpus end.
    let start = (start_rot + offset) % total;
    let end = start + size;
    let mut sequences = Vec::with_capacity(2);
    if end <= total {
        sequences.push(start..end);
    } else {
        sequences.push(start..total);
        sequences.push(0..(end - total));
    }
    Ok(Assignment { sequences })
}

/// The byte range of one sequence: `(shard index, byte offset into the shard, byte length)` —
/// pure windowing math over the manifest geometry (whole sequences per shard, so a sequence is
/// always one contiguous in-shard range).
///
/// # Errors
/// [`CorpusError::BatchOutOfRange`] past the last sequence.
pub fn sequence_byte_range(
    manifest: &CorpusManifest,
    seq_id: u64,
) -> Result<(usize, u64, u64), CorpusError> {
    let seq_len = u64::from(manifest.seq_len);
    let width = manifest.token_width.bytes();
    let mut cursor = 0u64;
    for (i, shard) in manifest.shards.iter().enumerate() {
        let seqs = shard.token_count / seq_len;
        if seq_id < cursor + seqs {
            let off = (seq_id - cursor) * seq_len * width;
            return Ok((i, off, seq_len * width));
        }
        cursor += seqs;
    }
    Err(CorpusError::BatchOutOfRange {
        batch: seq_id,
        total: manifest.total_sequences(),
    })
}

/// One `data@2::fetch` call the module's window plan asks for: a byte range of one shard,
/// named by the shard's fold identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeFetch {
    /// The shard's artifact identity (the chunk fold — grant, registration, and fetch key).
    pub shard_hash: [u8; 32],
    /// Range start (bytes into the shard).
    pub range_off: u64,
    /// Range length in bytes.
    pub range_len: u64,
}

/// Plan the fetches for a set of sequences: each sequence's byte range, coalesced into maximal
/// contiguous per-shard ranges in input order (module policy decides the order; adjacent
/// sequences of one shard collapse into one ranged fetch).
///
/// # Errors
/// [`CorpusError::BatchOutOfRange`] when any id is past the last sequence.
pub fn plan_window(
    manifest: &CorpusManifest,
    seqs: &[u64],
) -> Result<Vec<RangeFetch>, CorpusError> {
    let mut out: Vec<(usize, u64, u64)> = Vec::new(); // (shard, off, len)
    for &seq in seqs {
        let (shard, off, len) = sequence_byte_range(manifest, seq)?;
        match out.last_mut() {
            Some((s, o, l)) if *s == shard && *o + *l == off => *l += len,
            _ => out.push((shard, off, len)),
        }
    }
    Ok(out
        .into_iter()
        .map(|(shard, range_off, range_len)| RangeFetch {
            shard_hash: manifest.shards[shard].shard_hash.0,
            range_off,
            range_len,
        })
        .collect())
}

/// Build the canonical-CBOR `register_chunks` descriptor for shard `shard` —
/// `[chunk_size, token_count, byte_len, [c_0, …]]` — exactly what the host's fold check
/// re-derives the granted identity from.
///
/// # Errors
/// [`CorpusError::ShardIndexOutOfRange`] for an unknown shard.
pub fn chunk_descriptor(manifest: &CorpusManifest, shard: usize) -> Result<Vec<u8>, CorpusError> {
    let entry = manifest
        .shards
        .get(shard)
        .ok_or(CorpusError::ShardIndexOutOfRange {
            shard,
            total: manifest.shards.len(),
        })?;
    let hashes: Vec<ciborium::value::Value> = entry
        .chunk_hashes
        .iter()
        .map(|h| ciborium::value::Value::Bytes(h.0.to_vec()))
        .collect();
    let doc = ciborium::value::Value::Array(vec![
        ciborium::value::Value::from(manifest.chunk_size),
        ciborium::value::Value::from(entry.token_count),
        ciborium::value::Value::from(entry.byte_len),
        ciborium::value::Value::Array(hashes),
    ]);
    to_canonical_vec(&doc).map_err(|e| CorpusError::Parse(format!("chunk descriptor: {e}")))
}

/// Register shard `shard`'s chunk map with the host (`data@2::register_chunks`) — call once per
/// shard BEFORE ranging it; a chunk-addressed identity has no whole-object hash to verify
/// against, so an unregistered shard fetch can never verify.
///
/// # Errors
/// [`CorpusError::ShardIndexOutOfRange`] for an unknown shard. A fold outside the admitted
/// grants traps host-side (`GrantViolation`) — that is an admission fault, not a return.
#[cfg(target_arch = "wasm32")]
pub fn register_shard_chunks(manifest: &CorpusManifest, shard: usize) -> Result<(), CorpusError> {
    let desc = chunk_descriptor(manifest, shard)?;
    let status = crate::abi::data_register_chunks(&desc);
    if status == 0 {
        Ok(())
    } else {
        Err(CorpusError::Parse(format!(
            "register_chunks returned status {status}"
        )))
    }
}

/// Decode one sequence's token ids from its fetched byte range, honoring the manifest's pinned
/// token width AND endianness (never the reader's platform).
///
/// # Errors
/// [`CorpusError::ShardSizeMismatch`]-shaped refusal when `bytes` is not exactly
/// `seq_len × width` (a short read is a typed error, never a silent truncation).
pub fn decode_sequence_tokens(
    manifest: &CorpusManifest,
    bytes: &[u8],
) -> Result<Vec<u32>, CorpusError> {
    let width = manifest.token_width.bytes() as usize;
    let expected = manifest.seq_len as usize * width;
    if bytes.len() != expected {
        return Err(CorpusError::ShardSizeMismatch {
            shard: 0,
            expected: expected as u64,
            declared: bytes.len() as u64,
        });
    }
    let mut tokens = Vec::with_capacity(manifest.seq_len as usize);
    for chunk in bytes.chunks_exact(width) {
        let token = match (manifest.token_width, manifest.endianness) {
            (daemon_vhc_proto::TokenWidth::U16, Endianness::Little) => {
                u32::from(u16::from_le_bytes([chunk[0], chunk[1]]))
            }
            (daemon_vhc_proto::TokenWidth::U16, Endianness::Big) => {
                u32::from(u16::from_be_bytes([chunk[0], chunk[1]]))
            }
            (daemon_vhc_proto::TokenWidth::U32, Endianness::Little) => {
                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
            }
            (daemon_vhc_proto::TokenWidth::U32, Endianness::Big) => {
                u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
            }
        };
        tokens.push(token);
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shard(name: &str, tokens: u64) -> ShardDesc {
        // A structurally-valid placeholder hash (content checks are `assemble`'s job).
        ShardDesc {
            name: name.into(),
            bytes: tokens * 2,
            tokens,
            blake3: "00".repeat(32),
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

    /// Deterministic little-endian u16 shard bytes with tokens `< vocab`.
    fn shard_bytes(seed: u64, tokens: u64, vocab: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(tokens as usize * 2);
        let mut s = seed;
        for _ in 0..tokens {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.extend_from_slice(&(((s >> 33) % vocab) as u16).to_le_bytes());
        }
        out
    }

    fn real_manifest(seq_len: u32, per_shard: u64, n: u64) -> (Manifest, Vec<Vec<u8>>) {
        let mut shards = Vec::new();
        let mut blobs = Vec::new();
        for i in 0..n {
            let bytes = shard_bytes(i ^ 0xD1CE, per_shard, 64);
            shards.push(ShardDesc {
                name: format!("shard-{i:04}.bin"),
                bytes: bytes.len() as u64,
                tokens: per_shard,
                blake3: blake3::hash(&bytes).to_hex().to_string(),
            });
            blobs.push(bytes);
        }
        (manifest(seq_len, shards), blobs)
    }

    #[test]
    fn locate_maps_across_shard_boundaries() {
        let m = manifest(4, vec![shard("a", 8), shard("b", 12)]);
        m.validate().unwrap();
        assert_eq!(m.total_sequences(), 5);
        assert_eq!(
            m.locate(2).unwrap(),
            BatchLocation {
                shard: 1,
                token_offset: 0
            }
        );
        assert!(matches!(
            m.locate(5),
            Err(CorpusError::BatchOutOfRange { batch: 5, total: 5 })
        ));
    }

    #[test]
    fn shards_covering_wraps_and_selects_only_the_window() {
        let m = manifest(
            4,
            vec![
                shard("a", 16),
                shard("b", 16),
                shard("c", 16),
                shard("d", 16),
            ],
        );
        assert_eq!(m.shards_covering(0, 8), BTreeSet::from([0, 1]));
        assert_eq!(m.shards_covering(14, 4), BTreeSet::from([0, 3]));
        assert_eq!(m.shards_covering(0, 0), BTreeSet::from([0, 1, 2, 3]));
    }

    #[test]
    fn assemble_verifies_and_sequence_reads_resident_only() {
        let (m, blobs) = real_manifest(9, 18, 2);
        let mut resident = BTreeMap::new();
        resident.insert(0usize, blobs[0].clone());
        let w = CorpusWindow::assemble(m, resident).unwrap();
        assert_eq!(w.resident_shards(), 1);
        assert_eq!(w.total_sequences(), 4);
        let s0 = w.sequence(0).unwrap();
        assert_eq!(s0.len(), 9);
        assert!(s0.iter().all(|t| *t < 64), "small-vocab tokens");
        // Wrapping addresses shard 0 again at batch 4 (= 0 mod 4).
        assert_eq!(w.sequence(4).unwrap(), s0);
        // Shard 1 is not resident: typed, never silent.
        assert!(matches!(
            w.sequence(2),
            Err(CorpusError::ShardNotResident { shard: 1 })
        ));
    }

    #[test]
    fn assemble_rejects_tampered_and_out_of_range() {
        let (m, blobs) = real_manifest(9, 18, 2);
        let mut tampered = BTreeMap::new();
        tampered.insert(0usize, vec![0xFF; blobs[0].len()]);
        assert!(matches!(
            CorpusWindow::assemble(m.clone(), tampered),
            Err(CorpusError::ShardHashMismatch { shard: 0 })
        ));
        let mut bad_idx = BTreeMap::new();
        bad_idx.insert(9usize, blobs[0].clone());
        assert!(matches!(
            CorpusWindow::assemble(m, bad_idx),
            Err(CorpusError::ShardIndexOutOfRange { shard: 9, .. })
        ));
    }

    // ---- the chunk-addressed contract's policy math ---------------------------------------------

    /// A valid chunked manifest: 3 shards × 32/64/32 tokens (u16), seq_len 8, chunk_size 32.
    fn chunked_manifest() -> CorpusManifest {
        let blobs = [
            shard_bytes(11, 32, 64),
            shard_bytes(22, 64, 64),
            shard_bytes(33, 32, 64),
        ];
        let shards: Vec<ShardEntry> = blobs
            .iter()
            .map(|b| CorpusManifest::author_shard(b, b.len() as u64 / 2, 32).unwrap())
            .collect();
        let total = shards.iter().map(|s| s.token_count).sum();
        CorpusManifest {
            format_version: CORPUS_MANIFEST_FORMAT,
            token_width: daemon_vhc_proto::TokenWidth::U16,
            endianness: Endianness::Little,
            seq_len: 8,
            sequence_boundary: SequenceBoundary::WholeSequencesPerShard,
            eos_id: None,
            pad_id: None,
            chunk_size: 32,
            tokenizer: TokenizerId {
                hash: daemon_vhc_proto::blake3_hash(b"tok"),
                name: "tok".into(),
                revision: "r1".into(),
            },
            total_tokens: total,
            shards,
        }
    }

    #[test]
    fn assignment_is_deterministic_disjoint_and_covering() {
        let m = chunked_manifest();
        assert_eq!(m.total_sequences(), 16);
        for epoch in [0u64, 1, 7] {
            let mut seen = [0u32; 16];
            let mut first: Option<Assignment> = None;
            for peer in 0..3u32 {
                let p = AssignmentParams {
                    genesis_hash: [0xAB; 32],
                    epoch,
                    roster_size: 3,
                    peer_index: peer,
                };
                let a = derive_assignment(&m, &p).unwrap();
                let b = derive_assignment(&m, &p).unwrap();
                assert_eq!(a, b, "identical inputs, identical assignment");
                for s in a.iter() {
                    seen[s as usize] += 1;
                }
                if peer == 0 {
                    first = Some(a);
                }
            }
            assert!(
                seen.iter().all(|&c| c == 1),
                "epoch {epoch}: the roster covers every sequence exactly once (disjoint)"
            );
            let _ = first;
        }
        // Out-of-roster is typed.
        assert!(matches!(
            derive_assignment(
                &m,
                &AssignmentParams {
                    genesis_hash: [0xAB; 32],
                    epoch: 0,
                    roster_size: 2,
                    peer_index: 2,
                }
            ),
            Err(CorpusError::AssignmentOutOfRange { peer: 2, roster: 2 })
        ));
    }

    #[test]
    fn epoch_reshuffle_moves_the_layout() {
        let m = chunked_manifest();
        let assignment_at = |epoch: u64| {
            derive_assignment(
                &m,
                &AssignmentParams {
                    genesis_hash: [0xCD; 32],
                    epoch,
                    roster_size: 4,
                    peer_index: 1,
                },
            )
            .unwrap()
            .sequences
        };
        let layouts: Vec<_> = (0..8).map(assignment_at).collect();
        assert!(
            layouts.windows(2).any(|w| w[0] != w[1]),
            "eight consecutive epochs never all share one layout"
        );
    }

    #[test]
    fn window_plan_coalesces_adjacent_sequences_per_shard() {
        let m = chunked_manifest();
        // Sequences 0..4 live in shard 0 (32 tokens / 8 = 4 seqs), 4..12 in shard 1.
        let (s0, off0, len0) = sequence_byte_range(&m, 0).unwrap();
        assert_eq!((s0, off0, len0), (0, 0, 16));
        let (s1, off1, _) = sequence_byte_range(&m, 4).unwrap();
        assert_eq!((s1, off1), (1, 0));
        assert!(matches!(
            sequence_byte_range(&m, 16),
            Err(CorpusError::BatchOutOfRange { batch: 16, .. })
        ));

        // 0,1 coalesce; 5 opens shard 1; 2 re-opens shard 0 (order is policy, preserved).
        let plan = plan_window(&m, &[0, 1, 5, 2]).unwrap();
        assert_eq!(plan.len(), 3);
        assert_eq!(plan[0].shard_hash, m.shards[0].shard_hash.0);
        assert_eq!((plan[0].range_off, plan[0].range_len), (0, 32));
        assert_eq!(plan[1].shard_hash, m.shards[1].shard_hash.0);
        assert_eq!((plan[1].range_off, plan[1].range_len), (16, 16));
        assert_eq!((plan[2].range_off, plan[2].range_len), (32, 16));
    }

    #[test]
    fn chunk_descriptor_round_trips_the_fold() {
        let m = chunked_manifest();
        let desc = chunk_descriptor(&m, 1).unwrap();
        // The descriptor decodes back to the registered geometry whose fold IS the shard id
        // (the host-side check this feeds).
        let v: ciborium::value::Value = ciborium::de::from_reader(desc.as_slice()).unwrap();
        let ciborium::value::Value::Array(parts) = v else {
            panic!("descriptor shape");
        };
        assert_eq!(parts.len(), 4);
        assert!(matches!(
            chunk_descriptor(&m, 9),
            Err(CorpusError::ShardIndexOutOfRange { shard: 9, total: 3 })
        ));
    }

    #[test]
    fn sequence_tokens_decode_honors_width_and_endianness() {
        let mut m = chunked_manifest();
        m.seq_len = 2;
        let le = decode_sequence_tokens(&m, &[0x01, 0x02, 0x03, 0x04]).unwrap();
        assert_eq!(le, vec![0x0201, 0x0403]);
        m.endianness = Endianness::Big;
        let be = decode_sequence_tokens(&m, &[0x01, 0x02, 0x03, 0x04]).unwrap();
        assert_eq!(be, vec![0x0102, 0x0304]);
        // A short read is typed, never a silent truncation.
        assert!(decode_sequence_tokens(&m, &[0x01, 0x02]).is_err());
    }

    #[test]
    fn manifest_json_round_trips_the_v1_schema() {
        // The exact v1 `manifest.json` shape (provenance-less) parses and validates.
        let json = r#"{
            "token_width": "u16",
            "seq_len": 4,
            "shards": [{"name":"a","bytes":16,"tokens":8,
                "blake3":"0000000000000000000000000000000000000000000000000000000000000000"}]
        }"#;
        let m = Manifest::from_json(json).unwrap();
        assert_eq!(m.total_sequences(), 2);
        assert_eq!(m.tokenizer, None);
    }
}
