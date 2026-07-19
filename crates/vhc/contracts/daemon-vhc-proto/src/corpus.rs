// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **chunk-addressed corpus manifest** — the data contract that makes `data@2` byte-range
//! fetches verifiable without whole-shard downloads (architecture §3.2: the host fetches
//! verified byte ranges; the module owns assignment/windowing).
//!
//! # The chain of custody
//!
//! ```text
//! chunk bytes  →  c_i = blake3(chunk_i)                              (content)
//! c_0 … c_{n-1} → shard_hash = fold(chunk_size, token_count,
//!                                   byte_len, chunk hashes)          (order + geometry)
//! shard hashes → manifest → manifest_hash = blake3(canonical CBOR)   (the corpus identity)
//! manifest_hash → genesis `[run] corpus_manifest` pin                (the run's data root)
//! ```
//!
//! - A **chunk** is identified by its plain blake3 — so chunks ride every content-addressed
//!   seam unchanged (the `ContentStore` plane, the §8.7 replay payload table).
//! - A **shard**'s artifact identity is the domain-separated [`shard_fold`] over its ordered
//!   chunk hashes and geometry — *not* `blake3(shard bytes)`. That is deliberate: a fetched
//!   range is verified from the manifest's covering chunk hashes alone, and a whole-shard
//!   fetch of a chunk-addressed shard can never pass the pump's plain-hash verification by
//!   accident (the fold is only derivable through the chunk list a module registers).
//! - The **manifest** is canonical CBOR ([`crate::to_canonical_vec`]) so its hash is
//!   reproducible across authors; the genesis pins that hash, closing the chain.
//!
//! # Sequence and chunk alignment (ratified rules)
//!
//! - `chunk_size` is a whole multiple of the token width: **no token ever spans a chunk
//!   boundary**, so any token-aligned range's covering chunks are self-contained.
//! - Every shard holds **whole sequences** (`token_count % seq_len == 0`): a sequence never
//!   spans a shard boundary, so module-side windowing needs no cross-shard stitching.
//!
//! The host consumes only the *mechanism* half of this module (the fold + covering-span math,
//! via the chunk map a module registers at run time); it never decodes the manifest itself —
//! which sequences train, in which order, stays module policy (the SDK's corpus layer).

use serde::{Deserialize, Serialize};

use crate::bytes::Hash;
use crate::canonical::{from_canonical_slice, to_canonical_vec};
use crate::domains::CORPUS_SHARD_DOMAIN;
use crate::error::VhcProtoError;
use crate::hash::blake3_hash;

/// The corpus-manifest format major this build understands. A layout or derivation change of
/// any kind is a new major (the domain-registry identity rule); the old major keeps naming the
/// old scheme forever.
pub const CORPUS_MANIFEST_FORMAT: u32 = 1;

/// The ratified default chunk size (4 MiB): small enough that a covering-chunk fetch of one
/// training window stays cheap, large enough that per-shard chunk-hash lists stay tiny
/// (a 256 MiB shard carries 64 chunk hashes = 2 KiB).
pub const CORPUS_DEFAULT_CHUNK_SIZE: u64 = 4 << 20;

/// The token element width of a shard's fixed-width token stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TokenWidth {
    /// 16-bit tokens (vocab ≤ 65 536) — the ratified width for the TinyLlama-class corpora.
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

/// The byte order of every token in every shard — pinned in the manifest so a corpus authored
/// on any host reads identically on every peer (never inferred from the reader's platform).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Endianness {
    /// Little-endian token bytes (the authoring default).
    Little,
    /// Big-endian token bytes.
    Big,
}

/// The sequence-boundary rule the corpus was authored under (pinned, not inferred).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceBoundary {
    /// Every shard holds whole sequences (`token_count % seq_len == 0`); a sequence never spans
    /// a shard boundary. The only ratified rule at format 1.
    WholeSequencesPerShard,
}

/// The tokenizer identity: the tokenizer artifact itself is content-addressed (its `blake3` is
/// a run artifact like any other), with name + revision as human provenance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerId {
    /// blake3 of the tokenizer artifact bytes (e.g. `tokenizer.json`) — the fetchable identity.
    pub hash: Hash,
    /// The tokenizer's human name (e.g. an upstream repo id).
    pub name: String,
    /// The pinned upstream revision the artifact was resolved from.
    pub revision: String,
}

/// One shard entry: identity (the chunk fold), geometry, and the ordered per-chunk hashes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardEntry {
    /// The shard's artifact identity — [`shard_fold`] over the geometry + `chunk_hashes`. This
    /// is the hash a module grants, registers, and `data@2`-fetches by, and the payload-store
    /// key (`corpus/<hex>.bin`).
    pub shard_hash: Hash,
    /// The shard's total byte length.
    pub byte_len: u64,
    /// The number of tokens in the shard.
    pub token_count: u64,
    /// blake3 of each `chunk_size`-sized chunk, in order (the last chunk may be shorter).
    pub chunk_hashes: Vec<Hash>,
}

/// The chunk-addressed corpus manifest (see the module docs for the custody chain). Canonical
/// CBOR is the one wire form; [`CorpusManifest::manifest_hash`] is the identity the genesis pins.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusManifest {
    /// Format major — MUST be [`CORPUS_MANIFEST_FORMAT`].
    pub format_version: u32,
    /// The token element width shared by every shard.
    pub token_width: TokenWidth,
    /// The pinned token byte order.
    pub endianness: Endianness,
    /// Tokens per training sequence.
    pub seq_len: u32,
    /// The sequence-boundary rule (format 1: whole sequences per shard).
    pub sequence_boundary: SequenceBoundary,
    /// The end-of-sequence token id, where the tokenizer defines one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eos_id: Option<u32>,
    /// The padding token id, where the authoring pipeline pads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_id: Option<u32>,
    /// The chunk size in bytes, pinned corpus-wide (a whole multiple of the token width; the
    /// last chunk of each shard may be shorter).
    pub chunk_size: u64,
    /// The tokenizer identity (itself a content-addressed run artifact).
    pub tokenizer: TokenizerId,
    /// The total token count across all shards (redundant with the shard list; validated).
    pub total_tokens: u64,
    /// The shards, in data-window order.
    pub shards: Vec<ShardEntry>,
}

/// The registered chunk map for one shard — exactly what the host needs to serve verified byte
/// ranges: geometry + ordered chunk hashes. A module derives it from the manifest
/// ([`CorpusManifest::chunk_map`]) and registers it with the host, which re-derives the fold
/// and admits the map only when the fold IS a granted artifact hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkMap {
    /// The chunk size in bytes.
    pub chunk_size: u64,
    /// The shard's token count (folded into the identity — geometry is committed, not advisory).
    pub token_count: u64,
    /// The shard's byte length.
    pub byte_len: u64,
    /// blake3 of each chunk, in order.
    pub chunk_hashes: Vec<Hash>,
}

impl ChunkMap {
    /// Structural validity: non-degenerate geometry and a chunk list exactly covering
    /// `byte_len` at `chunk_size`.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        self.chunk_size > 0
            && self.byte_len > 0
            && self.chunk_hashes.len() as u64 == chunk_count(self.byte_len, self.chunk_size)
    }

    /// The shard identity this map folds to ([`shard_fold`]).
    #[must_use]
    pub fn fold(&self) -> Hash {
        shard_fold(
            self.chunk_size,
            self.token_count,
            self.byte_len,
            &self.chunk_hashes,
        )
    }

    /// The byte length of chunk `i` (the last chunk may be short).
    #[must_use]
    pub fn chunk_len(&self, i: u64) -> u64 {
        let start = i * self.chunk_size;
        self.byte_len.saturating_sub(start).min(self.chunk_size)
    }
}

/// The number of chunks covering `byte_len` bytes at `chunk_size` (0 for degenerate inputs).
#[must_use]
pub fn chunk_count(byte_len: u64, chunk_size: u64) -> u64 {
    if chunk_size == 0 {
        return 0;
    }
    byte_len.div_ceil(chunk_size)
}

/// blake3 of each `chunk_size`-sized window of `bytes`, in order (the chunk identities).
#[must_use]
pub fn chunk_hashes(bytes: &[u8], chunk_size: u64) -> Vec<Hash> {
    if chunk_size == 0 {
        return Vec::new();
    }
    bytes
        .chunks(usize::try_from(chunk_size).unwrap_or(usize::MAX))
        .map(blake3_hash)
        .collect()
}

/// The **shard fold** — the shard's artifact identity:
///
/// ```text
/// blake3(CORPUS_SHARD_DOMAIN ++ u64le(chunk_size) ++ u64le(token_count)
///                            ++ u64le(byte_len)   ++ c_0 ++ … ++ c_{n-1})
/// ```
///
/// Geometry is folded in so the same bytes under a different chunk size (or a re-declared token
/// count) are a *different* artifact; the domain prefix separates the fold from every other
/// blake3 derivation in the subsystem.
#[must_use]
pub fn shard_fold(chunk_size: u64, token_count: u64, byte_len: u64, chunks: &[Hash]) -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CORPUS_SHARD_DOMAIN);
    hasher.update(&chunk_size.to_le_bytes());
    hasher.update(&token_count.to_le_bytes());
    hasher.update(&byte_len.to_le_bytes());
    for c in chunks {
        hasher.update(&c.0);
    }
    Hash(*hasher.finalize().as_bytes())
}

/// The chunk-aligned **covering span** of the byte range `[off, end)` within a `byte_len`-byte
/// shard chunked at `chunk_size`: the smallest chunk-aligned `[span_off, span_off + span_len)`
/// containing the range (clamped to `byte_len`). An empty range covers nothing (`span_len` 0).
///
/// The caller has already bounds-checked `off <= end <= byte_len`.
#[must_use]
pub fn covering_span(byte_len: u64, chunk_size: u64, off: u64, end: u64) -> (u64, u64) {
    if off >= end || chunk_size == 0 {
        return (off, 0);
    }
    let span_off = (off / chunk_size) * chunk_size;
    let span_end = (end.div_ceil(chunk_size) * chunk_size).min(byte_len);
    (span_off, span_end - span_off)
}

impl CorpusManifest {
    /// Structural + numeric validation (every rule the format pins):
    ///
    /// - known `format_version`; non-zero `seq_len` and `chunk_size`;
    /// - `chunk_size % token_width == 0` (no token ever spans a chunk boundary);
    /// - per shard: non-zero tokens, `byte_len == token_count × width`, whole sequences
    ///   (`token_count % seq_len == 0`), a chunk list exactly covering `byte_len`, and a
    ///   `shard_hash` that IS the fold of the entry's own geometry + chunk hashes;
    /// - `total_tokens` equals the shard sum.
    pub fn validate(&self) -> Result<(), VhcProtoError> {
        if self.format_version != CORPUS_MANIFEST_FORMAT {
            return Err(VhcProtoError::Validation(format!(
                "unknown corpus-manifest format {} (this build understands \
                 {CORPUS_MANIFEST_FORMAT})",
                self.format_version
            )));
        }
        if self.shards.is_empty() {
            return Err(VhcProtoError::Validation(
                "corpus manifest has no shards".into(),
            ));
        }
        if self.seq_len == 0 {
            return Err(VhcProtoError::Validation("seq_len must be > 0".into()));
        }
        let width = self.token_width.bytes();
        if self.chunk_size == 0 || self.chunk_size % width != 0 {
            return Err(VhcProtoError::Validation(format!(
                "chunk_size {} must be a non-zero multiple of the token width {width} (no \
                 token may span a chunk boundary)",
                self.chunk_size
            )));
        }
        let mut total = 0u64;
        for (i, shard) in self.shards.iter().enumerate() {
            if shard.token_count == 0 {
                return Err(VhcProtoError::Validation(format!(
                    "shard {i} has zero tokens"
                )));
            }
            if shard.byte_len != shard.token_count * width {
                return Err(VhcProtoError::Validation(format!(
                    "shard {i} byte_len {} != token_count {} × width {width}",
                    shard.byte_len, shard.token_count
                )));
            }
            if shard.token_count % u64::from(self.seq_len) != 0 {
                return Err(VhcProtoError::Validation(format!(
                    "shard {i} token_count {} is not a multiple of seq_len {} (sequences must \
                     not span shard boundaries)",
                    shard.token_count, self.seq_len
                )));
            }
            let expected_chunks = chunk_count(shard.byte_len, self.chunk_size);
            if shard.chunk_hashes.len() as u64 != expected_chunks {
                return Err(VhcProtoError::Validation(format!(
                    "shard {i} declares {} chunk hashes, geometry needs {expected_chunks}",
                    shard.chunk_hashes.len()
                )));
            }
            let fold = shard_fold(
                self.chunk_size,
                shard.token_count,
                shard.byte_len,
                &shard.chunk_hashes,
            );
            if fold != shard.shard_hash {
                return Err(VhcProtoError::Validation(format!(
                    "shard {i} shard_hash {} is not the fold of its own chunk list ({})",
                    shard.shard_hash.to_hex(),
                    fold.to_hex()
                )));
            }
            total += shard.token_count;
        }
        if total != self.total_tokens {
            return Err(VhcProtoError::Validation(format!(
                "total_tokens {} != shard sum {total}",
                self.total_tokens
            )));
        }
        Ok(())
    }

    /// Author one shard entry from its raw bytes (chunk, hash, fold) — the `tokenize-corpus`
    /// authoring primitive and the test-fixture builder.
    ///
    /// # Errors
    /// [`VhcProtoError::Validation`] on degenerate geometry (`chunk_size == 0`, empty bytes, or
    /// a byte length that is not `token_count × width` — caught later by [`Self::validate`]).
    pub fn author_shard(
        bytes: &[u8],
        token_count: u64,
        chunk_size: u64,
    ) -> Result<ShardEntry, VhcProtoError> {
        if chunk_size == 0 || bytes.is_empty() {
            return Err(VhcProtoError::Validation(
                "author_shard needs non-empty bytes and a non-zero chunk size".into(),
            ));
        }
        let chunks = chunk_hashes(bytes, chunk_size);
        let byte_len = bytes.len() as u64;
        Ok(ShardEntry {
            shard_hash: shard_fold(chunk_size, token_count, byte_len, &chunks),
            byte_len,
            token_count,
            chunk_hashes: chunks,
        })
    }

    /// Serialize to the one wire form (canonical CBOR), after validation.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, VhcProtoError> {
        self.validate()?;
        to_canonical_vec(self)
    }

    /// The manifest identity: blake3 of the canonical bytes (what the genesis pins).
    pub fn manifest_hash(&self) -> Result<Hash, VhcProtoError> {
        Ok(blake3_hash(&self.to_canonical_bytes()?))
    }

    /// Decode + validate a manifest from its canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, VhcProtoError> {
        let manifest: Self = from_canonical_slice(bytes)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// The registered-chunk-map view of shard `i` (`None` out of range).
    #[must_use]
    pub fn chunk_map(&self, shard: usize) -> Option<ChunkMap> {
        let entry = self.shards.get(shard)?;
        Some(ChunkMap {
            chunk_size: self.chunk_size,
            token_count: entry.token_count,
            byte_len: entry.byte_len,
            chunk_hashes: entry.chunk_hashes.clone(),
        })
    }

    /// The total number of whole sequences across all shards.
    #[must_use]
    pub fn total_sequences(&self) -> u64 {
        self.total_tokens / u64::from(self.seq_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Deterministic little-endian u16 token bytes.
    fn shard_bytes(seed: u64, tokens: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(tokens as usize * 2);
        let mut s = seed;
        for _ in 0..tokens {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.extend_from_slice(&(((s >> 33) % 64) as u16).to_le_bytes());
        }
        out
    }

    fn tokenizer() -> TokenizerId {
        TokenizerId {
            hash: blake3_hash(b"tokenizer-fixture"),
            name: "fixture-tokenizer".into(),
            revision: "deadbeef".into(),
        }
    }

    /// A valid two-shard manifest: seq_len 8, chunk_size 32 (16 u16 tokens per chunk).
    fn manifest() -> (CorpusManifest, Vec<Vec<u8>>) {
        let blobs = vec![shard_bytes(1, 64), shard_bytes(2, 32)];
        let shards: Vec<ShardEntry> = blobs
            .iter()
            .map(|b| CorpusManifest::author_shard(b, b.len() as u64 / 2, 32).unwrap())
            .collect();
        let total = shards.iter().map(|s| s.token_count).sum();
        (
            CorpusManifest {
                format_version: CORPUS_MANIFEST_FORMAT,
                token_width: TokenWidth::U16,
                endianness: Endianness::Little,
                seq_len: 8,
                sequence_boundary: SequenceBoundary::WholeSequencesPerShard,
                eos_id: Some(2),
                pad_id: None,
                chunk_size: 32,
                tokenizer: tokenizer(),
                total_tokens: total,
                shards,
            },
            blobs,
        )
    }

    #[test]
    fn round_trips_canonically_and_hash_is_reproducible() {
        let (m, _) = manifest();
        let bytes_a = m.to_canonical_bytes().unwrap();
        let bytes_b = m.clone().to_canonical_bytes().unwrap();
        assert_eq!(bytes_a, bytes_b, "canonical encoding is deterministic");
        let back = CorpusManifest::from_canonical_bytes(&bytes_a).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.manifest_hash().unwrap(), m.manifest_hash().unwrap());
        assert_eq!(m.total_sequences(), 12);
    }

    #[test]
    fn fold_binds_geometry_and_order() {
        let (m, blobs) = manifest();
        let entry = &m.shards[0];
        let map = m.chunk_map(0).unwrap();
        assert!(map.is_well_formed());
        assert_eq!(map.fold(), entry.shard_hash);
        // A different chunk size over the same bytes is a different identity.
        let other = CorpusManifest::author_shard(&blobs[0], entry.token_count, 64).unwrap();
        assert_ne!(other.shard_hash, entry.shard_hash);
        // Reordered chunk hashes are a different identity.
        let mut swapped = entry.chunk_hashes.clone();
        swapped.swap(0, 1);
        assert_ne!(
            shard_fold(32, entry.token_count, entry.byte_len, &swapped),
            entry.shard_hash
        );
        // A re-declared token count is a different identity.
        assert_ne!(
            shard_fold(
                32,
                entry.token_count + 8,
                entry.byte_len,
                &entry.chunk_hashes
            ),
            entry.shard_hash
        );
        // And the fold is never the plain content hash of the shard bytes.
        assert_ne!(entry.shard_hash, blake3_hash(&blobs[0]));
    }

    #[test]
    fn validation_rejects_each_broken_rule() {
        let (good, _) = manifest();

        let mut m = good.clone();
        m.format_version = 2;
        assert!(m.validate().is_err(), "unknown format");

        let mut m = good.clone();
        m.chunk_size = 33; // not a multiple of the u16 width
        assert!(m.validate().is_err(), "token spanning a chunk boundary");

        let mut m = good.clone();
        m.shards[0].token_count += 4; // breaks byte_len == tokens × width
        assert!(m.validate().is_err(), "geometry mismatch");

        let mut m = good.clone();
        m.seq_len = 7; // 32 % 7 != 0
        assert!(m.validate().is_err(), "sequence spanning a shard boundary");

        let mut m = good.clone();
        m.shards[0].chunk_hashes.pop();
        assert!(m.validate().is_err(), "chunk list does not cover byte_len");

        let mut m = good.clone();
        m.shards[0].shard_hash = blake3_hash(b"not-the-fold");
        assert!(m.validate().is_err(), "shard_hash is not the fold");

        let mut m = good.clone();
        m.total_tokens += 1;
        assert!(m.validate().is_err(), "total_tokens mismatch");

        let mut m = good;
        m.shards.clear();
        assert!(m.validate().is_err(), "empty manifest");
    }

    #[test]
    fn chunk_hashes_match_manual_slicing() {
        let bytes = shard_bytes(3, 40); // 80 bytes; chunk 32 → 32 + 32 + 16
        let hashes = chunk_hashes(&bytes, 32);
        assert_eq!(hashes.len(), 3);
        assert_eq!(hashes[0], blake3_hash(&bytes[..32]));
        assert_eq!(hashes[1], blake3_hash(&bytes[32..64]));
        assert_eq!(hashes[2], blake3_hash(&bytes[64..]));
        let map = ChunkMap {
            chunk_size: 32,
            token_count: 40,
            byte_len: 80,
            chunk_hashes: hashes,
        };
        assert_eq!(map.chunk_len(0), 32);
        assert_eq!(map.chunk_len(2), 16, "last chunk is short");
    }

    proptest! {
        /// The covering span always contains the range, is chunk-aligned at its start, ends
        /// chunk-aligned or at byte_len, and is minimal (shrinking either edge by one chunk
        /// loses coverage).
        #[test]
        fn covering_span_covers_and_is_minimal(
            byte_len in 1u64..100_000,
            chunk_size in 1u64..5_000,
            a in 0u64..100_000,
            b in 0u64..100_000,
        ) {
            let off = a.min(byte_len);
            let end = (a + b % 1_000).min(byte_len);
            let (span_off, span_len) = covering_span(byte_len, chunk_size, off, end);
            if off >= end {
                prop_assert_eq!(span_len, 0);
            } else {
                let span_end = span_off + span_len;
                // Covers the range…
                prop_assert!(span_off <= off && end <= span_end);
                // …within the shard…
                prop_assert!(span_end <= byte_len);
                // …chunk-aligned at the start, aligned-or-terminal at the end…
                prop_assert_eq!(span_off % chunk_size, 0);
                prop_assert!(span_end % chunk_size == 0 || span_end == byte_len);
                // …and minimal: one chunk narrower on either side loses coverage.
                prop_assert!(span_off + chunk_size > off);
                prop_assert!(span_end < end + chunk_size);
            }
        }

        /// author_shard's fold reproduces from the manual pipeline for arbitrary geometry.
        #[test]
        fn author_shard_fold_reproduces(tokens in 1u64..512, chunk_tokens in 1u64..64) {
            let bytes = shard_bytes(tokens, tokens);
            let chunk_size = chunk_tokens * 2;
            let entry = CorpusManifest::author_shard(&bytes, tokens, chunk_size).unwrap();
            let manual = shard_fold(
                chunk_size,
                tokens,
                bytes.len() as u64,
                &chunk_hashes(&bytes, chunk_size),
            );
            prop_assert_eq!(entry.shard_hash, manual);
            prop_assert_eq!(
                entry.chunk_hashes.len() as u64,
                chunk_count(bytes.len() as u64, chunk_size)
            );
        }
    }
}
