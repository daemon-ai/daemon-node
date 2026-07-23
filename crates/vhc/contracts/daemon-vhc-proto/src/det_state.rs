// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **chunk-addressed det-state contract** — the corpus custody chain (see [`crate::corpus`])
//! instantiated for canonical training state, under its own derivation domain
//! ([`crate::domains::DET_STATE_DOMAIN`]).
//!
//! # The chain of custody
//!
//! ```text
//! chunk bytes   → c_i = blake3(chunk_i)                              (content)
//! c_0 … c_{n-1} → family_fold = blake3(domain ‖ u64le(chunk_size)
//!                                      ‖ u64le(byte_len) ‖ c_0 ‖ …)  (order + geometry)
//! family folds  → det-state manifest → blake3(canonical CBOR)        (the round state root)
//! ```
//!
//! - A **chunk** is identified by its plain blake3, riding every content-addressed seam
//!   unchanged.
//! - A **family** (`master`, `replicated:<name>` — the flat f32-le concatenation of its
//!   per-parameter vectors in registration order) has the domain-separated [`family_fold`] as
//!   its artifact identity — *not* `blake3(family bytes)` — so a byte range verifies from the
//!   manifest's covering chunk hashes alone, without materializing the family. The fold
//!   deliberately omits the corpus fold's `token_count`; det-state geometry is
//!   `(chunk_size, byte_len)`.
//! - The **manifest** ([`DetStateManifest`]) is the per-round consensus-canonical statement;
//!   its canonical-CBOR blake3 ([`DetStateManifest::state_root`]) is the **round state root**.
//!   Every peer derives the identical manifest from its own sealed chunks — an agreement
//!   primitive, not a message.
//!
//! # Chunking rules (normative)
//!
//! - Chunking is **per parameter**: a parameter never spans a chunk boundary (each parameter is
//!   independently chunked at `chunk_size`; its last chunk may be short) — the state-plane
//!   mirror of the corpus rule "no token ever spans a chunk boundary".
//! - The **profile-chunk constraint** ([`validate_profile_chunk`]): the compression profile's
//!   `chunk` MUST divide every parameter's numel (the det kernels hard-refuse a non-multiple
//!   layout only at first use; genesis authoring validates it up front).
//! - [`validate_state_chunk_size`]: the state `chunk_size` MUST be a non-zero integer multiple
//!   of the profile chunk's byte width (`chunk × 4`), so no compression chunk ever spans a
//!   state chunk. [`derive_state_chunk_size`] picks the authoring default (the largest such
//!   multiple ≤ 4 MiB — the corpus cost point).
//! - [`validate_checkpoint_cadence`]: a rejoiner replays forward from the freshest reachable
//!   remote checkpoint only across *retained* payloads, so the remote publication cadence plus
//!   one cadence slot of publisher-churn slack must fit inside the payload-retention floor.
//!
//! # Checkpoint-document v2 section forms
//!
//! [`FamilyRef`] and [`CkptDocSection`] are the wire forms a chunked checkpoint document's
//! sections take: inline bytes for small sections (e.g. the 8-byte round watermark), a
//! by-reference family descriptor for large ones (the already-sealed family artifacts — zero
//! additional bytes moved). Minted here as contract types; the producing/consuming seams adopt
//! them in their own changes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bytes::Hash;
use crate::canonical::{from_canonical_slice, to_canonical_vec};
use crate::domains::DET_STATE_DOMAIN;
use crate::error::VhcProtoError;
use crate::hash::blake3_hash;

/// The det-state manifest format major this build understands. A layout or derivation change of
/// any kind is a new major (the domain-registry identity rule).
pub const DET_STATE_MANIFEST_FORMAT: u32 = 1;

/// The byte width of one state element (families are f32-le by contract).
pub const STATE_ELEM_BYTES: u64 = 4;

/// The authoring cost point for [`derive_state_chunk_size`]: ~4 MiB, the corpus default's
/// per-chunk cost point (fewer per-op costs vs tighter guest memory bound).
pub const STATE_CHUNK_SIZE_TARGET: u64 = 4 << 20;

/// The consensus-canonical family every det-state manifest MUST carry.
pub const MASTER_FAMILY: &str = "master";

/// The name prefix of profile-declared replicated consensus-canonical families
/// (e.g. `replicated:outer-momentum`).
pub const REPLICATED_FAMILY_PREFIX: &str = "replicated:";

/// Grant name: raw-byte token bucket + per-emit ceiling on state-chunk writes.
pub const STATE_WRITE_BUDGET_GRANT: &str = "state-write-budget";
/// Grant name: live retained bytes across sealed families.
pub const STATE_STORE_BYTES_GRANT: &str = "state-store-bytes";
/// Grant name: concurrent open state streams.
pub const STATE_STREAMS_MAX_GRANT: &str = "state-streams-max";
/// Default number of sealed roots retained per family: the current round base and the freshly
/// sealed round.
pub const STATE_RETAIN_ROOTS_DEFAULT: u64 = 2;

/// The **family fold** — a state family's artifact identity:
///
/// ```text
/// blake3(DET_STATE_DOMAIN ++ u64le(chunk_size) ++ u64le(byte_len) ++ c_0 ++ … ++ c_{n-1})
/// ```
///
/// Geometry is folded in so the same bytes under a different chunk size are a *different*
/// artifact; the domain prefix separates the fold from every other blake3 derivation in the
/// subsystem (including the corpus shard fold over the identical chunk list).
#[must_use]
pub fn family_fold(chunk_size: u64, byte_len: u64, chunks: &[Hash]) -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DET_STATE_DOMAIN);
    hasher.update(&chunk_size.to_le_bytes());
    hasher.update(&byte_len.to_le_bytes());
    for c in chunks {
        hasher.update(&c.0);
    }
    Hash(*hasher.finalize().as_bytes())
}

/// The number of chunks covering one parameter of `numel` elements at `chunk_size` (0 for
/// degenerate inputs). Parameters are independently chunked; the last chunk may be short.
#[must_use]
pub fn param_chunk_count(numel: u64, chunk_size: u64) -> u64 {
    if chunk_size == 0 {
        return 0;
    }
    (numel * STATE_ELEM_BYTES).div_ceil(chunk_size)
}

/// The total chunk count of a family over the parameter layout (per-parameter chunking).
#[must_use]
pub fn family_chunk_count(numels: &[u64], chunk_size: u64) -> u64 {
    numels
        .iter()
        .map(|&n| param_chunk_count(n, chunk_size))
        .sum()
}

/// The byte length of a param-shaped family (the flat f32-le concatenation).
#[must_use]
pub fn family_byte_len(numels: &[u64]) -> u64 {
    numels.iter().map(|&n| n * STATE_ELEM_BYTES).sum()
}

/// blake3 of each per-parameter `chunk_size`-sized window of `params`, in registration order —
/// the family's ordered chunk identities under the per-parameter chunking rule.
#[must_use]
pub fn family_chunk_hashes(params: &[&[u8]], chunk_size: u64) -> Vec<Hash> {
    if chunk_size == 0 {
        return Vec::new();
    }
    let step = usize::try_from(chunk_size).unwrap_or(usize::MAX);
    params
        .iter()
        .flat_map(|p| p.chunks(step))
        .map(blake3_hash)
        .collect()
}

/// The per-chunk byte lengths of a param-shaped family under the per-parameter chunking rule, in
/// order — the length-aware twin of [`family_chunk_hashes`]. Interior parameters end in a short
/// tail (a parameter never spans a chunk boundary), so the sequence is NOT a uniform grid; these
/// are the lengths a [`DetStateChunkMap`] carries so an externally-registered fold resolves ranges
/// against actual offsets ([SF-R2]).
#[must_use]
pub fn family_chunk_lens(numels: &[u64], chunk_size: u64) -> Vec<u32> {
    let mut lens = Vec::new();
    if chunk_size == 0 {
        return lens;
    }
    for &n in numels {
        let mut remaining = n * STATE_ELEM_BYTES;
        while remaining > 0 {
            let take = remaining.min(chunk_size);
            #[allow(clippy::cast_possible_truncation)]
            lens.push(take as u32);
            remaining -= take;
        }
    }
    lens
}

/// A **det-state registration descriptor** ([SF-R2], ruling of 2026-07-23): what a guest passes to
/// register an *externally-sourced* family fold (artifact-form init, restore roots) so a
/// subsequent `data@2::fetch` resolves ranges **length-aware**. Per-parameter chunking gives a
/// family interior short tails, so — unlike the uniform-grid corpus [`crate::corpus::ChunkMap`] —
/// the covering geometry needs explicit per-chunk lengths.
///
/// The lengths are **guest-derived from the layout** (numels + chunk_size, [`family_chunk_lens`]),
/// NOT a new field on [`FamilyRef`] or [`DetStateManifest`]: the family fold already pins the
/// ordered chunk hashes; the lengths are only framing hints for where to split served bytes. A
/// lying descriptor cannot corrupt — a wrong split re-hashes to something other than the granted
/// `c_i` and fails [`DetStateChunkMap::verify_covering_span`] chunk-by-chunk. So root minimality
/// and digest continuity are preserved with no new trust surface (the deciding argument for the
/// ruling): the fold is the integrity anchor, the lengths are self-correcting framing. This is the
/// symmetric twin of the self-sealed state store's `(hash, len)` model — self-sealed folds carry
/// their emitted lengths, externally-registered folds carry derived lengths, both resolve by
/// walking actual offsets.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetStateChunkMap {
    /// The state chunk size the family was chunked at.
    pub chunk_size: u64,
    /// The family's total byte length.
    pub byte_len: u64,
    /// `(chunk blake3, chunk len)` in fold order — lengths derived from the parameter layout.
    pub chunks: Vec<(Hash, u32)>,
}

impl DetStateChunkMap {
    /// Derive a descriptor from a family's ordered chunk hashes + the parameter layout: pair each
    /// hash with its per-parameter-chunking length ([`family_chunk_lens`]). The `chunk_hashes`
    /// come from the fetched [`FamilyRef`]/[`DetStateManifest`]; the guest already knows `numels`.
    ///
    /// # Errors
    /// [`VhcProtoError::Validation`] when the derived length count does not match the supplied
    /// chunk-hash count (the fetched geometry disagrees with the layout — a typed refusal, never a
    /// silent mismatch).
    pub fn derive(
        chunk_size: u64,
        numels: &[u64],
        chunk_hashes: &[Hash],
    ) -> Result<Self, VhcProtoError> {
        let lens = family_chunk_lens(numels, chunk_size);
        if lens.len() != chunk_hashes.len() {
            return Err(VhcProtoError::Validation(format!(
                "det-state descriptor: layout yields {} chunks but the fetched family lists {}",
                lens.len(),
                chunk_hashes.len()
            )));
        }
        Ok(Self {
            chunk_size,
            byte_len: family_byte_len(numels),
            chunks: chunk_hashes.iter().copied().zip(lens).collect(),
        })
    }

    /// The family fold this descriptor covers (its artifact identity — the value that must be a
    /// granted artifact for the registered fold to be fetchable, exactly the corpus posture).
    #[must_use]
    pub fn fold(&self) -> Hash {
        let hashes: Vec<Hash> = self.chunks.iter().map(|(h, _)| *h).collect();
        family_fold(self.chunk_size, self.byte_len, &hashes)
    }

    /// Structural validity: non-degenerate geometry, every chunk `0 < len ≤ chunk_size`, and the
    /// lengths summing to exactly `byte_len` (the framing is well-formed). The fold is NOT
    /// self-checkable here (it is derived, not carried) — the host admits the map only when
    /// [`Self::fold`] is a granted artifact, and verification re-hashes each served chunk.
    pub fn validate(&self) -> Result<(), VhcProtoError> {
        if self.chunk_size == 0 || self.byte_len == 0 || self.chunks.is_empty() {
            return Err(VhcProtoError::Validation(
                "det-state chunk map needs non-zero geometry and a chunk list".into(),
            ));
        }
        let mut total = 0u64;
        for (i, (_, len)) in self.chunks.iter().enumerate() {
            let len = u64::from(*len);
            if len == 0 || len > self.chunk_size {
                return Err(VhcProtoError::Validation(format!(
                    "det-state chunk {i} length {len} is not in 1..={}",
                    self.chunk_size
                )));
            }
            total += len;
        }
        if total != self.byte_len {
            return Err(VhcProtoError::Validation(format!(
                "det-state chunk lengths sum to {total}, not the declared byte_len {}",
                self.byte_len
            )));
        }
        Ok(())
    }

    /// The minimal chunk-aligned covering span of the byte range `[off, end)` under the actual
    /// per-chunk offsets (the det-state analogue of [`crate::corpus::covering_span`], length-aware
    /// rather than uniform-grid): returns `(span_off, span_len)` where `span_off` is the start of
    /// the first chunk touching `off` and the span ends at the end of the last chunk touching
    /// `end`. `(off, 0)` for an empty range.
    #[must_use]
    pub fn covering_span(&self, off: u64, end: u64) -> (u64, u64) {
        if off >= end {
            return (off, 0);
        }
        let mut cursor = 0u64;
        let mut span_off = 0u64;
        let mut span_end = 0u64;
        let mut started = false;
        for (_, len) in &self.chunks {
            let chunk_start = cursor;
            let chunk_end = cursor + u64::from(*len);
            cursor = chunk_end;
            if chunk_end <= off {
                continue;
            }
            if chunk_start >= end {
                break;
            }
            if !started {
                span_off = chunk_start;
                started = true;
            }
            span_end = chunk_end;
        }
        if !started {
            return (off, 0);
        }
        (span_off, span_end - span_off)
    }

    /// Verify a covering span served from an untrusted store against the descriptor's fold-pinned
    /// chunk hashes, walking the actual per-chunk offsets: `span_off` must fall on a chunk
    /// boundary, and every chunk the span covers must blake3 to the descriptor's hash at that
    /// position. Returns the verified bytes, or the first mismatch described. A lying length or a
    /// tampered store fails HERE — the fold-committed hashes are the anchor.
    ///
    /// # Errors
    /// A `String` describing a boundary/length/hash violation.
    pub fn verify_covering_span(&self, span_off: u64, bytes: &[u8]) -> Result<(), VhcProtoError> {
        // Locate the starting chunk index (span_off must be a chunk boundary).
        let mut cursor = 0u64;
        let mut index = 0usize;
        while index < self.chunks.len() && cursor < span_off {
            cursor += u64::from(self.chunks[index].1);
            index += 1;
        }
        if cursor != span_off {
            return Err(VhcProtoError::Validation(format!(
                "det-state span_off {span_off} is not a chunk boundary"
            )));
        }
        let mut pos = 0usize;
        while pos < bytes.len() {
            let Some((hash, len)) = self.chunks.get(index) else {
                return Err(VhcProtoError::Validation(format!(
                    "det-state span reaches past the chunk list (chunk {index})"
                )));
            };
            let end = pos + *len as usize;
            if end > bytes.len() {
                return Err(VhcProtoError::Validation(format!(
                    "det-state span truncates chunk {index} ({} of {len} bytes)",
                    bytes.len() - pos
                )));
            }
            if blake3_hash(&bytes[pos..end]) != *hash {
                return Err(VhcProtoError::Validation(format!(
                    "det-state chunk {index} does not hash to the fold-registered chunk hash"
                )));
            }
            pos = end;
            index += 1;
        }
        Ok(())
    }

    /// The ordered `(chunk hash, chunk len)` pairs that a chunk-aligned covering span
    /// `[span_off, span_off + span_len)` is composed of — the list a chunk-keyed resolver fetches
    /// and concatenates to reassemble the span, symmetric with the replay-side chunk-keyed
    /// materialization ([SF-R2]). `span_off` MUST be a chunk boundary and the span MUST end on a
    /// chunk boundary (both hold for a span produced by [`Self::covering_span`]); anything else is
    /// a typed refusal so the resolver never fetches a mis-aligned or over-reaching span.
    ///
    /// # Errors
    /// A `String` describing a boundary/length violation.
    pub fn covering_chunks(
        &self,
        span_off: u64,
        span_len: u64,
    ) -> Result<Vec<(Hash, u32)>, VhcProtoError> {
        // Locate the starting chunk (span_off must be a chunk boundary).
        let mut cursor = 0u64;
        let mut index = 0usize;
        while index < self.chunks.len() && cursor < span_off {
            cursor += u64::from(self.chunks[index].1);
            index += 1;
        }
        if cursor != span_off {
            return Err(VhcProtoError::Validation(format!(
                "det-state span_off {span_off} is not a chunk boundary"
            )));
        }
        let mut out = Vec::new();
        let mut remaining = span_len;
        while remaining > 0 {
            let Some(&(hash, len)) = self.chunks.get(index) else {
                return Err(VhcProtoError::Validation(format!(
                    "det-state covering span [{span_off}, +{span_len}) reaches past the chunk list \
                     (chunk {index})"
                )));
            };
            out.push((hash, len));
            remaining = remaining.checked_sub(u64::from(len)).ok_or_else(|| {
                VhcProtoError::Validation(format!(
                    "det-state covering span [{span_off}, +{span_len}) does not end on a chunk \
                     boundary (chunk {index} overshoots)"
                ))
            })?;
            index += 1;
        }
        Ok(out)
    }

    /// Encode to the one wire form (canonical CBOR) — the descriptor a guest passes to the host's
    /// det-state chunk registration import, after [`Self::validate`].
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, VhcProtoError> {
        self.validate()?;
        to_canonical_vec(self)
    }

    /// Decode + validate a descriptor from its canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, VhcProtoError> {
        let map: Self = from_canonical_slice(bytes)?;
        map.validate()?;
        Ok(map)
    }
}

/// The layout binding: the parameter count and the blake3 of the canonical-CBOR numels list —
/// enough for a peer to refuse a manifest authored over a different registration-order layout
/// without carrying the (large) list itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutBinding {
    /// Parameter count (registration order).
    pub params: u64,
    /// blake3 of the canonical-CBOR encoding of the numels list.
    pub numels: Hash,
}

impl LayoutBinding {
    /// Derive the binding from the parameter numels (registration order).
    ///
    /// # Errors
    /// Propagates a codec error (structurally unreachable for a `u64` list).
    pub fn of_numels(numels: &[u64]) -> Result<Self, VhcProtoError> {
        Ok(Self {
            params: numels.len() as u64,
            numels: blake3_hash(&to_canonical_vec(&numels)?),
        })
    }
}

/// One family entry: identity (the fold), geometry, and the ordered per-chunk hashes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FamilyEntry {
    /// The family's artifact identity — [`family_fold`] over the manifest's `chunk_size` +
    /// this entry's geometry + `chunk_hashes`.
    pub fold: Hash,
    /// The family's total byte length.
    pub byte_len: u64,
    /// blake3 of each chunk under the per-parameter chunking rule, in order.
    pub chunk_hashes: Vec<Hash>,
}

impl FamilyEntry {
    /// Author one family entry from its per-parameter byte images (registration order) — chunk,
    /// hash, fold.
    ///
    /// # Errors
    /// [`VhcProtoError::Validation`] on degenerate geometry (`chunk_size == 0`, no parameters,
    /// or an empty parameter).
    pub fn author(params: &[&[u8]], chunk_size: u64) -> Result<Self, VhcProtoError> {
        if chunk_size == 0 || params.is_empty() || params.iter().any(|p| p.is_empty()) {
            return Err(VhcProtoError::Validation(
                "a family needs non-empty parameters and a non-zero chunk size".into(),
            ));
        }
        let chunk_hashes = family_chunk_hashes(params, chunk_size);
        let byte_len = params.iter().map(|p| p.len() as u64).sum();
        Ok(Self {
            fold: family_fold(chunk_size, byte_len, &chunk_hashes),
            byte_len,
            chunk_hashes,
        })
    }
}

/// The per-round **det-state manifest**: the consensus-canonical statement of the round's sealed
/// state families. Canonical CBOR is the one wire form; [`DetStateManifest::state_root`] is the
/// round state root. It covers only consensus-canonical families ([`MASTER_FAMILY`] +
/// `replicated:<name>`…); replica-local families (error feedback, optimizer moments) have
/// per-family folds that appear only inside checkpoint documents, never in the shared root.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetStateManifest {
    /// Format major — MUST be [`DET_STATE_MANIFEST_FORMAT`].
    pub format: u32,
    /// The genesis hash — binds the root to the run.
    pub run_id: Hash,
    /// The ingested round this state folds.
    pub round: u64,
    /// The parameter-layout binding.
    pub layout: LayoutBinding,
    /// The state chunk size in bytes, pinned run-wide by the genesis state contract.
    pub chunk_size: u64,
    /// The consensus-canonical families, keyed by name.
    pub families: BTreeMap<String, FamilyEntry>,
}

impl DetStateManifest {
    /// Structural + numeric validation (every rule the format pins from the manifest alone):
    ///
    /// - known `format`; non-zero `chunk_size`; at least one parameter in the layout;
    /// - a [`MASTER_FAMILY`] entry present; every other family named `replicated:<name>`;
    /// - per family: non-zero `byte_len`, a chunk list whose count is *plausible* for
    ///   per-parameter chunking (between the contiguous floor and floor + one short tail per
    ///   parameter), and a `fold` that IS the fold of the entry's own geometry + chunk hashes.
    ///
    /// Exact per-parameter chunk counts need the numels list — [`Self::validate_with_numels`].
    pub fn validate(&self) -> Result<(), VhcProtoError> {
        if self.format != DET_STATE_MANIFEST_FORMAT {
            return Err(VhcProtoError::Validation(format!(
                "unknown det-state manifest format {} (this build understands \
                 {DET_STATE_MANIFEST_FORMAT})",
                self.format
            )));
        }
        if self.chunk_size == 0 {
            return Err(VhcProtoError::Validation(
                "det-state chunk_size must be > 0".into(),
            ));
        }
        if self.layout.params == 0 {
            return Err(VhcProtoError::Validation(
                "det-state layout binds zero parameters".into(),
            ));
        }
        if !self.families.contains_key(MASTER_FAMILY) {
            return Err(VhcProtoError::Validation(format!(
                "det-state manifest lacks the `{MASTER_FAMILY}` family"
            )));
        }
        for (name, family) in &self.families {
            if name != MASTER_FAMILY && !name.starts_with(REPLICATED_FAMILY_PREFIX) {
                return Err(VhcProtoError::Validation(format!(
                    "family `{name}` is neither `{MASTER_FAMILY}` nor \
                     `{REPLICATED_FAMILY_PREFIX}<name>` (only consensus-canonical families \
                     enter the state root)"
                )));
            }
            if family.byte_len == 0 {
                return Err(VhcProtoError::Validation(format!(
                    "family `{name}` has zero bytes"
                )));
            }
            // Per-parameter chunking: the count is the contiguous floor plus at most one short
            // tail chunk per parameter.
            let floor = family.byte_len.div_ceil(self.chunk_size);
            let ceiling = family
                .byte_len
                .div_ceil(self.chunk_size)
                .saturating_add(self.layout.params);
            let declared = family.chunk_hashes.len() as u64;
            if declared < floor || declared > ceiling {
                return Err(VhcProtoError::Validation(format!(
                    "family `{name}` declares {declared} chunks; per-parameter chunking of {} \
                     bytes at {} needs between {floor} and {ceiling}",
                    family.byte_len, self.chunk_size
                )));
            }
            let fold = family_fold(self.chunk_size, family.byte_len, &family.chunk_hashes);
            if fold != family.fold {
                return Err(VhcProtoError::Validation(format!(
                    "family `{name}` fold {} is not the fold of its own chunk list ({})",
                    family.fold.to_hex(),
                    fold.to_hex()
                )));
            }
        }
        Ok(())
    }

    /// Layout-aware validation: [`Self::validate`] plus, against the actual numels list, the
    /// layout binding ([`LayoutBinding::of_numels`]) and each family's exact byte length and
    /// per-parameter chunk count (every family is param-shaped by contract).
    pub fn validate_with_numels(&self, numels: &[u64]) -> Result<(), VhcProtoError> {
        self.validate()?;
        let binding = LayoutBinding::of_numels(numels)?;
        if binding != self.layout {
            return Err(VhcProtoError::Validation(
                "det-state layout binding does not match the parameter numels".into(),
            ));
        }
        let byte_len = family_byte_len(numels);
        let chunks = family_chunk_count(numels, self.chunk_size);
        for (name, family) in &self.families {
            if family.byte_len != byte_len {
                return Err(VhcProtoError::Validation(format!(
                    "family `{name}` byte_len {} != layout byte length {byte_len}",
                    family.byte_len
                )));
            }
            if family.chunk_hashes.len() as u64 != chunks {
                return Err(VhcProtoError::Validation(format!(
                    "family `{name}` declares {} chunks; per-parameter chunking needs {chunks}",
                    family.chunk_hashes.len()
                )));
            }
        }
        Ok(())
    }

    /// Serialize to the one wire form (canonical CBOR), after validation.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, VhcProtoError> {
        self.validate()?;
        to_canonical_vec(self)
    }

    /// The **round state root**: blake3 of the canonical bytes — the derivable agreement object
    /// every peer computes identically from its own sealed chunks.
    pub fn state_root(&self) -> Result<Hash, VhcProtoError> {
        Ok(blake3_hash(&self.to_canonical_bytes()?))
    }

    /// Decode + validate a manifest from its canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, VhcProtoError> {
        let manifest: Self = from_canonical_slice(bytes)?;
        manifest.validate()?;
        Ok(manifest)
    }
}

/// A by-reference family descriptor inside a chunked checkpoint document: everything a consumer
/// needs to register + range-fetch the already-sealed family artifact (zero section bytes moved
/// for a family the host store already holds).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FamilyRef {
    /// The family's artifact identity ([`family_fold`]).
    pub fold: Hash,
    /// The family's total byte length.
    pub byte_len: u64,
    /// The chunk size the family was sealed at.
    pub chunk_size: u64,
    /// blake3 of each chunk, in order.
    pub chunk_hashes: Vec<Hash>,
}

impl FamilyRef {
    /// Structural validity: non-degenerate geometry and a `fold` that IS the fold of the
    /// descriptor's own geometry + chunk hashes.
    pub fn validate(&self) -> Result<(), VhcProtoError> {
        if self.chunk_size == 0 || self.byte_len == 0 || self.chunk_hashes.is_empty() {
            return Err(VhcProtoError::Validation(
                "family ref needs non-zero geometry and a chunk list".into(),
            ));
        }
        let fold = family_fold(self.chunk_size, self.byte_len, &self.chunk_hashes);
        if fold != self.fold {
            return Err(VhcProtoError::Validation(format!(
                "family ref fold {} is not the fold of its own chunk list ({})",
                self.fold.to_hex(),
                fold.to_hex()
            )));
        }
        Ok(())
    }
}

/// A checkpoint-document v2 section: inline bytes for small sections (the 8-byte round
/// watermark), by-reference for large families. On the wire each form is a 2-element array
/// (`[name, bytes]` / `[name, family-ref]`), distinguished by the second element's shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CkptDocSection {
    /// `[name, family-ref]` — a by-reference family section.
    ByRef(String, FamilyRef),
    /// `[name, bytes]` — an inline section.
    Inline(String, Vec<u8>),
}

impl CkptDocSection {
    /// The section name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::ByRef(name, _) | Self::Inline(name, _) => name,
        }
    }
}

/// The profile-chunk constraint (genesis-authoring rule): the compression profile's `chunk`
/// MUST divide every parameter's numel — the det kernels refuse a non-multiple layout only at
/// first use, so authoring validates it up front. For the ceremony geometry every numel is a
/// multiple of `d_model` and the norm parameters ARE `d_model`, so the profile chunk must
/// divide `d_model` itself.
pub fn validate_profile_chunk(profile_chunk: u64, numels: &[u64]) -> Result<(), VhcProtoError> {
    if profile_chunk == 0 {
        return Err(VhcProtoError::Validation(
            "profile chunk must be > 0".into(),
        ));
    }
    if numels.is_empty() {
        return Err(VhcProtoError::Validation(
            "profile-chunk validation needs a non-empty parameter layout".into(),
        ));
    }
    for (i, &numel) in numels.iter().enumerate() {
        if !numel.is_multiple_of(profile_chunk) {
            return Err(VhcProtoError::Validation(format!(
                "profile chunk {profile_chunk} does not divide parameter {i}'s numel {numel} \
                 (the det kernels refuse a non-multiple layout)"
            )));
        }
    }
    Ok(())
}

/// The state-chunk-size rule (genesis-validated): `state_chunk_size` MUST be a non-zero integer
/// multiple of the profile chunk's byte width (`chunk × 4`), so no compression chunk ever spans
/// a state chunk and every fold window keeps profile-chunk-aligned interior boundaries.
pub fn validate_state_chunk_size(
    state_chunk_size: u64,
    profile_chunk: u64,
) -> Result<(), VhcProtoError> {
    if profile_chunk == 0 {
        return Err(VhcProtoError::Validation(
            "profile chunk must be > 0".into(),
        ));
    }
    let width = profile_chunk * STATE_ELEM_BYTES;
    if state_chunk_size == 0 || !state_chunk_size.is_multiple_of(width) {
        return Err(VhcProtoError::Validation(format!(
            "state chunk_size {state_chunk_size} must be a non-zero integer multiple of the \
             profile chunk byte width {width} (chunk {profile_chunk} × {STATE_ELEM_BYTES})"
        )));
    }
    Ok(())
}

/// The authoring derivation for `state_chunk_size`: the **largest** integer multiple of the
/// profile chunk's byte width (`chunk × 4`) that is ≤ [`STATE_CHUNK_SIZE_TARGET`] (~4 MiB, the
/// corpus cost point), falling back to one chunk width when a single compression chunk already
/// exceeds the target. Derived, not defaulted — the result is pinned in the genesis state
/// contract and satisfies [`validate_state_chunk_size`] by construction.
#[must_use]
pub fn derive_state_chunk_size(profile_chunk: u64) -> u64 {
    let width = profile_chunk.max(1) * STATE_ELEM_BYTES;
    let multiples = (STATE_CHUNK_SIZE_TARGET / width).max(1);
    multiples * width
}

/// The cadence↔retention bound (genesis-validation rule): a rejoiner replays forward from the
/// freshest reachable remote checkpoint only across *retained* payloads, so the remote
/// publication cadence plus **one cadence slot of publisher-churn slack** (a slot whose
/// deterministic publisher died goes unpublished; the next slot's rotation covers it) must fit
/// inside the payload-retention floor:
///
/// ```text
/// remote_cadence_rounds + remote_cadence_rounds ≤ payload_retention_rounds
/// ```
///
/// `payload_retention_rounds == 0` means unbounded retention (no constraint);
/// `remote_cadence_rounds == 0` means remote publication is disabled (nothing to bound).
pub fn validate_checkpoint_cadence(
    remote_cadence_rounds: u64,
    payload_retention_rounds: u64,
) -> Result<(), VhcProtoError> {
    if payload_retention_rounds == 0 || remote_cadence_rounds == 0 {
        return Ok(());
    }
    let need = remote_cadence_rounds.saturating_mul(2);
    if need > payload_retention_rounds {
        return Err(VhcProtoError::Validation(format!(
            "remote checkpoint cadence {remote_cadence_rounds} + one publisher-churn slot \
             exceeds payload_retention_rounds {payload_retention_rounds} (a rejoiner could \
             strand past retention); retention must be ≥ {need} or the cadence tightened"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::shard_fold;
    use proptest::prelude::*;

    /// Deterministic per-parameter f32-le byte images (seeded splitmix64).
    fn param_bytes(seed: u64, numel: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(numel as usize * 4);
        let mut s = seed;
        for _ in 0..numel {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            #[allow(clippy::cast_precision_loss)]
            out.extend_from_slice(&(((s >> 40) as f32) / 16_777_216.0).to_le_bytes());
        }
        out
    }

    fn manifest_of(numels: &[u64], chunk_size: u64) -> DetStateManifest {
        let params: Vec<Vec<u8>> = numels
            .iter()
            .enumerate()
            .map(|(i, &n)| param_bytes(i as u64 + 1, n))
            .collect();
        let views: Vec<&[u8]> = params.iter().map(Vec::as_slice).collect();
        let master = FamilyEntry::author(&views, chunk_size).unwrap();
        DetStateManifest {
            format: DET_STATE_MANIFEST_FORMAT,
            run_id: Hash([7; 32]),
            round: 3,
            layout: LayoutBinding::of_numels(numels).unwrap(),
            chunk_size,
            families: [(MASTER_FAMILY.to_string(), master)].into_iter().collect(),
        }
    }

    #[test]
    fn fold_binds_geometry_order_and_domain() {
        let numels = [16u64, 4, 8];
        let params: Vec<Vec<u8>> = numels.iter().map(|&n| param_bytes(n, n)).collect();
        let views: Vec<&[u8]> = params.iter().map(Vec::as_slice).collect();
        let entry = FamilyEntry::author(&views, 32).unwrap();

        // A different chunk size over the same bytes is a different identity.
        let other = FamilyEntry::author(&views, 16).unwrap();
        assert_ne!(other.fold, entry.fold);

        // Reordered chunk hashes are a different identity.
        let mut swapped = entry.chunk_hashes.clone();
        swapped.swap(0, 1);
        assert_ne!(family_fold(32, entry.byte_len, &swapped), entry.fold);

        // Never the plain content hash of the family bytes.
        let flat: Vec<u8> = params.concat();
        assert_ne!(entry.fold, blake3_hash(&flat));

        // Domain separation: the corpus shard fold over the IDENTICAL chunk list + geometry is
        // a different identity (token_count 0 chosen to make the corpus preimage minimal).
        assert_ne!(
            entry.fold,
            shard_fold(32, 0, entry.byte_len, &entry.chunk_hashes)
        );
    }

    #[test]
    fn chunking_is_per_parameter_with_short_tails() {
        // numels 16/4/8 f32 → 64/16/32 bytes; chunk 24 → per-param 3+1+2 chunks, tails short.
        let numels = [16u64, 4, 8];
        assert_eq!(family_chunk_count(&numels, 24), 6);
        assert_eq!(family_byte_len(&numels), 112);
        let params: Vec<Vec<u8>> = numels.iter().map(|&n| param_bytes(n, n)).collect();
        let views: Vec<&[u8]> = params.iter().map(Vec::as_slice).collect();
        let hashes = family_chunk_hashes(&views, 24);
        assert_eq!(hashes.len(), 6);
        // Parameter boundaries hold: chunk 3 is the WHOLE second parameter (16 bytes < 24), and
        // differs from any contiguous-flat slicing that would straddle the boundary.
        assert_eq!(hashes[3], blake3_hash(&params[1]));
        let flat: Vec<u8> = params.concat();
        assert_ne!(hashes[3], blake3_hash(&flat[72..96]));
    }

    #[test]
    fn manifest_round_trips_and_state_root_is_reproducible() {
        let m = manifest_of(&[16, 4, 8], 24);
        let bytes_a = m.to_canonical_bytes().unwrap();
        let bytes_b = m.clone().to_canonical_bytes().unwrap();
        assert_eq!(bytes_a, bytes_b, "canonical encoding is deterministic");
        let back = DetStateManifest::from_canonical_bytes(&bytes_a).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.state_root().unwrap(), m.state_root().unwrap());
        m.validate_with_numels(&[16, 4, 8]).unwrap();
    }

    #[test]
    fn manifest_validation_rejects_each_broken_rule() {
        let good = manifest_of(&[16, 4, 8], 24);

        let mut m = good.clone();
        m.format = 2;
        assert!(m.validate().is_err(), "unknown format");

        let mut m = good.clone();
        m.chunk_size = 0;
        assert!(m.validate().is_err(), "zero chunk size");

        let mut m = good.clone();
        m.families.clear();
        assert!(m.validate().is_err(), "master family required");

        let mut m = good.clone();
        let entry = m.families[MASTER_FAMILY].clone();
        m.families.insert("ef".into(), entry);
        assert!(m.validate().is_err(), "replica-local family name refused");

        let mut m = good.clone();
        m.families.get_mut(MASTER_FAMILY).unwrap().fold = blake3_hash(b"not-the-fold");
        assert!(m.validate().is_err(), "fold is not the chunk-list fold");

        let mut m = good.clone();
        m.families
            .get_mut(MASTER_FAMILY)
            .unwrap()
            .chunk_hashes
            .truncate(2);
        assert!(m.validate().is_err(), "chunk list below the coverage floor");

        // A replicated family under the proper prefix is accepted.
        let mut m = good.clone();
        let entry = m.families[MASTER_FAMILY].clone();
        m.families.insert("replicated:outer-momentum".into(), entry);
        m.validate().unwrap();

        // Layout-aware: wrong numels refuse on the binding.
        assert!(good.validate_with_numels(&[16, 4, 9]).is_err());
        // …and a right-binding but mis-chunked family refuses on the exact count: chunk 32
        // gives 2+1+1 = 4 chunks, not the declared 6.
        let mut m = good;
        m.chunk_size = 32;
        let params: Vec<Vec<u8>> = [16u64, 4, 8]
            .iter()
            .enumerate()
            .map(|(i, &n)| param_bytes(i as u64 + 1, n))
            .collect();
        let views: Vec<&[u8]> = params.iter().map(Vec::as_slice).collect();
        let hashes = family_chunk_hashes(&views, 24); // wrong chunking for chunk_size 32
        let fam = m.families.get_mut(MASTER_FAMILY).unwrap();
        fam.chunk_hashes = hashes;
        fam.fold = family_fold(32, fam.byte_len, &fam.chunk_hashes);
        assert!(m.validate_with_numels(&[16, 4, 8]).is_err());
    }

    #[test]
    fn family_ref_and_doc_sections_round_trip() {
        let params: Vec<Vec<u8>> = [16u64, 4].iter().map(|&n| param_bytes(n, n)).collect();
        let views: Vec<&[u8]> = params.iter().map(Vec::as_slice).collect();
        let entry = FamilyEntry::author(&views, 24).unwrap();
        let fref = FamilyRef {
            fold: entry.fold,
            byte_len: entry.byte_len,
            chunk_size: 24,
            chunk_hashes: entry.chunk_hashes,
        };
        fref.validate().unwrap();
        let mut broken = fref.clone();
        broken.byte_len += 1;
        assert!(broken.validate().is_err(), "geometry is fold-committed");

        let sections = vec![
            CkptDocSection::Inline("round".into(), 7u64.to_le_bytes().to_vec()),
            CkptDocSection::ByRef("master".into(), fref),
        ];
        let wire = to_canonical_vec(&sections).unwrap();
        let back: Vec<CkptDocSection> = from_canonical_slice(&wire).unwrap();
        assert_eq!(back, sections);
        assert_eq!(back[0].name(), "round");
        assert_eq!(back[1].name(), "master");
    }

    #[test]
    fn det_state_chunk_map_derives_lengths_and_matches_the_fold() {
        // numels 16/4/8 f32 → 64/16/32 bytes; chunk 24 → per-param 3+1+2 = 6 chunks.
        let numels = [16u64, 4, 8];
        let params: Vec<Vec<u8>> = numels
            .iter()
            .enumerate()
            .map(|(i, &n)| param_bytes(i as u64 + 1, n))
            .collect();
        let views: Vec<&[u8]> = params.iter().map(Vec::as_slice).collect();
        let entry = FamilyEntry::author(&views, 24).unwrap();

        let lens = family_chunk_lens(&numels, 24);
        assert_eq!(
            lens,
            vec![24, 24, 16, 16, 24, 8],
            "per-parameter short tails"
        );

        let map = DetStateChunkMap::derive(24, &numels, &entry.chunk_hashes).unwrap();
        map.validate().unwrap();
        // The descriptor covers the SAME fold identity as the manifest family entry.
        assert_eq!(map.fold(), entry.fold);
        assert_eq!(map.byte_len, 112);
        assert_eq!(map.chunks.len(), 6);

        // Round-trips through canonical CBOR.
        let wire = map.to_canonical_bytes().unwrap();
        assert_eq!(DetStateChunkMap::from_canonical_bytes(&wire).unwrap(), map);

        // A layout that disagrees with the fetched chunk-hash count is a typed refusal.
        assert!(DetStateChunkMap::derive(24, &[16, 4], &entry.chunk_hashes).is_err());
    }

    #[test]
    fn det_state_verify_walks_actual_offsets_and_catches_lies() {
        let numels = [16u64, 4, 8];
        let params: Vec<Vec<u8>> = numels
            .iter()
            .enumerate()
            .map(|(i, &n)| param_bytes(i as u64 + 1, n))
            .collect();
        let flat: Vec<u8> = params.concat(); // the family byte image
        let views: Vec<&[u8]> = params.iter().map(Vec::as_slice).collect();
        let entry = FamilyEntry::author(&views, 24).unwrap();
        let map = DetStateChunkMap::derive(24, &numels, &entry.chunk_hashes).unwrap();

        // A whole-family span at a boundary verifies.
        map.verify_covering_span(0, &flat).unwrap();

        // The covering span of an interior range is chunk-aligned to actual offsets; serving it
        // verifies, and the served span is exactly the covering chunks.
        let (span_off, span_len) = map.covering_span(30, 70);
        assert_eq!(
            span_off, 24,
            "starts at the boundary of the chunk containing 30"
        );
        let served = flat[span_off as usize..(span_off + span_len) as usize].to_vec();
        map.verify_covering_span(span_off, &served).unwrap();

        // A tampered byte fails re-verification (the fold-pinned hashes are the anchor).
        let mut bad = flat.clone();
        bad[0] ^= 0xff;
        assert!(map.verify_covering_span(0, &bad).is_err());

        // A span_off that is not a chunk boundary is refused.
        assert!(map.verify_covering_span(10, &flat[10..]).is_err());

        // A lying length (claim a full-width tail) makes the split re-hash wrong → refused.
        let mut lying = map.clone();
        let last = lying.chunks.len() - 1;
        lying.chunks[last].1 = 24; // was 8
        lying.byte_len += 16;
        assert!(lying.verify_covering_span(0, &flat).is_err());
    }

    #[test]
    fn det_state_covering_chunks_reassembles_span_and_rejects_misalignment() {
        let numels = [16u64, 4, 8];
        let params: Vec<Vec<u8>> = numels
            .iter()
            .enumerate()
            .map(|(i, &n)| param_bytes(i as u64 + 1, n))
            .collect();
        let flat: Vec<u8> = params.concat();
        let views: Vec<&[u8]> = params.iter().map(Vec::as_slice).collect();
        let entry = FamilyEntry::author(&views, 24).unwrap();
        let map = DetStateChunkMap::derive(24, &numels, &entry.chunk_hashes).unwrap();

        // The covering span of an interior range decomposes into exactly its constituent chunks,
        // in order, and the concatenated chunk bytes ARE the covering span the resolver serves.
        let (span_off, span_len) = map.covering_span(30, 70);
        let chunks = map.covering_chunks(span_off, span_len).unwrap();
        let listed_len: u64 = chunks.iter().map(|(_, l)| u64::from(*l)).sum();
        assert_eq!(listed_len, span_len, "listed chunk lengths sum to the span");
        // Each listed hash is the blake3 of the corresponding slice of the family image.
        let mut pos = span_off as usize;
        for (hash, len) in &chunks {
            let end = pos + *len as usize;
            assert_eq!(blake3_hash(&flat[pos..end]), *hash);
            pos = end;
        }
        // A whole-family span lists every chunk.
        let all = map.covering_chunks(0, map.byte_len).unwrap();
        assert_eq!(all.len(), map.chunks.len());
        // A non-boundary offset and an over-reaching span are typed refusals.
        assert!(map.covering_chunks(10, 24).is_err(), "off not a boundary");
        assert!(
            map.covering_chunks(0, map.byte_len + 1).is_err(),
            "span past the chunk list"
        );
    }

    #[test]
    fn det_state_chunk_map_validate_rejects_broken_framing() {
        let numels = [8u64];
        let params: Vec<Vec<u8>> = numels.iter().map(|&n| param_bytes(1, n)).collect();
        let views: Vec<&[u8]> = params.iter().map(Vec::as_slice).collect();
        let entry = FamilyEntry::author(&views, 16).unwrap();
        let good = DetStateChunkMap::derive(16, &numels, &entry.chunk_hashes).unwrap();

        let mut m = good.clone();
        m.chunks[0].1 = 0;
        assert!(m.validate().is_err(), "zero-length chunk");

        let mut m = good.clone();
        m.chunks[0].1 = 99; // > chunk_size
        assert!(m.validate().is_err(), "chunk longer than chunk_size");

        let mut m = good;
        m.byte_len += 1; // lengths no longer sum to byte_len
        assert!(m.validate().is_err(), "length sum mismatch");
    }

    #[test]
    fn state_chunk_size_rule_and_derivation() {
        // chunk 64 → width 256 B → the largest multiple ≤ 4 MiB is exactly 4 MiB.
        assert_eq!(derive_state_chunk_size(64), 4 << 20);
        validate_state_chunk_size(4 << 20, 64).unwrap();
        // chunk 1536 → width 6144 B → 682 multiples (the ceremony-geometry cost point).
        assert_eq!(derive_state_chunk_size(1536), 682 * 6144);
        validate_state_chunk_size(682 * 6144, 1536).unwrap();
        // A giant profile chunk falls back to one chunk width.
        assert_eq!(derive_state_chunk_size(2 << 20), (2 << 20) * 4);
        // Refusals: zero, and any non-multiple of chunk × 4.
        assert!(validate_state_chunk_size(0, 64).is_err());
        assert!(validate_state_chunk_size((4 << 20) + 1, 64).is_err());
        assert!(
            validate_state_chunk_size(4 << 20, 1536).is_err(),
            "4 MiB is not a 6144 multiple"
        );
        assert!(validate_state_chunk_size(64, 0).is_err());
    }

    #[test]
    fn cadence_retention_bound() {
        // cadence 20 + one churn slot (20) needs retention ≥ 40.
        validate_checkpoint_cadence(20, 40).unwrap();
        assert!(validate_checkpoint_cadence(20, 39).is_err());
        // Unbounded retention / disabled remote publication: no constraint.
        validate_checkpoint_cadence(20, 0).unwrap();
        validate_checkpoint_cadence(0, 8).unwrap();
    }

    proptest! {
        /// The derived state chunk size always satisfies its own validation rule and never
        /// exceeds the target unless one chunk width already does.
        #[test]
        fn derived_chunk_size_is_always_valid(profile_chunk in 1u64..2_000_000) {
            let derived = derive_state_chunk_size(profile_chunk);
            prop_assert!(validate_state_chunk_size(derived, profile_chunk).is_ok());
            let width = profile_chunk * STATE_ELEM_BYTES;
            if width <= STATE_CHUNK_SIZE_TARGET {
                prop_assert!(derived <= STATE_CHUNK_SIZE_TARGET);
                prop_assert!(derived + width > STATE_CHUNK_SIZE_TARGET, "largest multiple");
            } else {
                prop_assert_eq!(derived, width);
            }
        }

        /// Authored families reproduce from the manual pipeline for arbitrary geometry, and the
        /// chunk count matches the per-parameter arithmetic.
        #[test]
        fn author_family_fold_reproduces(
            numels in proptest::collection::vec(1u64..64, 1..6),
            chunk_size in 1u64..128,
        ) {
            let params: Vec<Vec<u8>> = numels
                .iter()
                .enumerate()
                .map(|(i, &n)| param_bytes(i as u64, n))
                .collect();
            let views: Vec<&[u8]> = params.iter().map(Vec::as_slice).collect();
            let entry = FamilyEntry::author(&views, chunk_size).unwrap();
            let manual = family_fold(
                chunk_size,
                family_byte_len(&numels),
                &family_chunk_hashes(&views, chunk_size),
            );
            prop_assert_eq!(entry.fold, manual);
            prop_assert_eq!(
                entry.chunk_hashes.len() as u64,
                family_chunk_count(&numels, chunk_size)
            );
        }
    }
}
