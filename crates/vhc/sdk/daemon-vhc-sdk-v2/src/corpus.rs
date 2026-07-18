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
        }
    }
}

impl std::error::Error for CorpusError {}

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
