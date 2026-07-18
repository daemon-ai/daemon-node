// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Typed checkpoint manifests (architecture §5.3; refactor §9 Phase E; ABI §10.2).
//!
//! A checkpoint is **a typed manifest, not a blob**: it names, as separate content-addressed
//! sections each carrying a schema version, the run's recoverable state — *consensus state* (det
//! lane), *module/role state* (opaque module serialization), *worker-local recoverable state*, the
//! *data cursor*, and the *journal position* (architecture §5.3). The manifest makes explicit which
//! sections are **consensus-canonical** (peers must agree on them byte-for-byte, so a digest
//! attestation over them is meaningful — [`crate::attestation`]) and which are **role/replica-local**
//! (peers legitimately differ in native params, optimizer state, caches).
//!
//! ## One discipline with migration (ABI §10.2)
//!
//! The section-declaration shape here is deliberately a **superset** of the ABI §10.2
//! `state-section-decl` the migration path speaks (`daemon-vhc-sdk::migrate`): a checkpoint's
//! sections carry the same `name`/`schema`/`hash`/`size`/`class` fields, plus a semantic
//! [`SectionKind`] tag naming which of the five architecture-§5.3 sections it is. So the upgrade
//! transaction's "snapshot state + journal cursor" (architecture §5.4; E2's upgrade transaction)
//! *is* a checkpoint manifest — checkpointing and migration are one discipline. E2 threads its
//! snapshot through this type; E3 (cold join) consumes it plus [`crate::attestation`].
//!
//! ## Content addressing (architecture §5.3, §8)
//!
//! The manifest itself is content-addressed: [`CheckpointManifest::content_hash`] is the blake3 of
//! its canonical-CBOR encoding, and that hash is the identity a coordinator record names, an
//! attestation signs over, and a late joiner fetches by. Each section's `hash` is the blake3 of the
//! (host-staged, hash-verified) section bytes, so the whole checkpoint is verifiable end-to-end
//! against a single committed hash.

use daemon_vhc_proto::{blake3_hash, from_canonical_slice, to_canonical_vec, Hash, StateDigest};
use serde::{Deserialize, Serialize};

/// The schema version of the checkpoint-manifest envelope defined by this module (distinct from a
/// module's own per-section `schema` versions). Bumped only on a breaking manifest-shape change;
/// additive fields ride `#[serde(default)]` (the established discipline).
pub const CHECKPOINT_MANIFEST_SCHEMA: u64 = 1;

/// Which of the five architecture-§5.3 checkpoint sections a declaration names. The numeric values
/// are permanent wire tags (never renumbered); additive kinds append.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(into = "u64", try_from = "u64")]
#[non_exhaustive]
pub enum SectionKind {
    /// Consensus state on the det lane — the bytes every peer must agree on. A digest attestation
    /// (architecture §5.3, [`crate::attestation::AttestationTier::Digest`]) is a claim about this
    /// section's digest.
    Consensus,
    /// Module / role state — the module's opaque serialization (params, optimizer state). Role- or
    /// replica-local: peers legitimately differ here (architecture §5.3).
    Module,
    /// Worker-local recoverable state (caches, scratch) — never consensus-canonical.
    WorkerLocal,
    /// The data cursor — how far this replica has consumed its corpus window.
    DataCursor,
    /// The journal position — the execution-identity journal ordinal the checkpoint captures, from
    /// which record-replay catch-up resumes (architecture §5.3; §8.1 journal cursor).
    JournalPosition,
}

impl SectionKind {
    /// The permanent numeric wire tag.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::Consensus => 0,
            Self::Module => 1,
            Self::WorkerLocal => 2,
            Self::DataCursor => 3,
            Self::JournalPosition => 4,
        }
    }

    /// The default canonicity [`SectionClass`] for this kind: only [`SectionKind::Consensus`] is
    /// consensus-canonical; the rest are role/replica-local (architecture §5.3). A section MAY
    /// override this (e.g. a canonical shared base carried in the module section), so the class is
    /// stored explicitly on each [`CheckpointSection`].
    #[must_use]
    pub const fn default_class(self) -> SectionClass {
        match self {
            Self::Consensus => SectionClass::ConsensusCanonical,
            _ => SectionClass::RoleLocal,
        }
    }
}

impl From<SectionKind> for u64 {
    fn from(k: SectionKind) -> u64 {
        k.tag()
    }
}

impl TryFrom<u64> for SectionKind {
    type Error = CheckpointError;
    fn try_from(v: u64) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Consensus),
            1 => Ok(Self::Module),
            2 => Ok(Self::WorkerLocal),
            3 => Ok(Self::DataCursor),
            4 => Ok(Self::JournalPosition),
            other => Err(CheckpointError::UnknownSectionKind(other)),
        }
    }
}

/// The canonicity class of a section (ABI §10.2 `class`): whether peers must agree on it byte-for-byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "u64", try_from = "u64")]
pub enum SectionClass {
    /// Consensus-canonical: every peer's copy is byte-identical (ABI §10.2 class 0).
    ConsensusCanonical,
    /// Role- or replica-local: peers legitimately differ (ABI §10.2 class 1).
    RoleLocal,
}

impl From<SectionClass> for u64 {
    fn from(c: SectionClass) -> u64 {
        match c {
            SectionClass::ConsensusCanonical => 0,
            SectionClass::RoleLocal => 1,
        }
    }
}

impl TryFrom<u64> for SectionClass {
    type Error = CheckpointError;
    fn try_from(v: u64) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::ConsensusCanonical),
            1 => Ok(Self::RoleLocal),
            other => Err(CheckpointError::UnknownSectionClass(other)),
        }
    }
}

/// One declared checkpoint section (architecture §5.3; a superset of ABI §10.2 `state-section-decl`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointSection {
    /// Section name, unique within one manifest (e.g. `"consensus"`, `"module"`, `"data-cursor"`).
    pub name: String,
    /// Which of the five §5.3 sections this is.
    pub kind: SectionKind,
    /// Per-section schema version (module-defined for module/role state; 1 for host-owned sections).
    pub schema: u64,
    /// blake3 of the section bytes (the content address of the section on the payload plane).
    pub hash: Hash,
    /// Section byte length.
    pub size: u64,
    /// Canonicity class (ABI §10.2 `class`).
    pub class: SectionClass,
}

impl CheckpointSection {
    /// Build a section declaration from its bytes, hashing them and defaulting the class from the
    /// kind. Use [`CheckpointSection::with_class`] to override the class.
    #[must_use]
    pub fn from_bytes(
        name: impl Into<String>,
        kind: SectionKind,
        schema: u64,
        bytes: &[u8],
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            schema,
            hash: blake3_hash(bytes),
            size: bytes.len() as u64,
            class: kind.default_class(),
        }
    }

    /// Override the canonicity class (e.g. a consensus-canonical base carried in a module section).
    #[must_use]
    pub fn with_class(mut self, class: SectionClass) -> Self {
        self.class = class;
        self
    }
}

/// A typed checkpoint manifest (architecture §5.3): the sectioned, content-addressed, schema-versioned
/// description of a run's recoverable state at one `(epoch, round)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointManifest {
    /// This module's manifest-envelope schema version ([`CHECKPOINT_MANIFEST_SCHEMA`]).
    pub schema: u64,
    /// The run identity (genesis hash, architecture §5.1) this checkpoint belongs to.
    pub run_id: Hash,
    /// The epoch this checkpoint captures (transition-chain head, architecture §5.1).
    pub epoch: u64,
    /// The round (post-ingest state) this checkpoint captures.
    pub round: u64,
    /// The producing module hash (execution identity, architecture §5.1).
    pub module: Hash,
    /// The consensus-state digest this checkpoint reproduces (the det-lane agreement probe, §5.6) —
    /// the value a [`crate::attestation::AttestationTier::Digest`] attestation is signed against.
    pub digest: StateDigest,
    /// The declared sections, in declaration order.
    pub sections: Vec<CheckpointSection>,
}

impl CheckpointManifest {
    /// Start assembling a manifest at `(run_id, epoch, round, module)` reproducing `digest`.
    #[must_use]
    pub fn builder(
        run_id: Hash,
        epoch: u64,
        round: u64,
        module: Hash,
        digest: StateDigest,
    ) -> CheckpointManifestBuilder {
        CheckpointManifestBuilder {
            manifest: Self {
                schema: CHECKPOINT_MANIFEST_SCHEMA,
                run_id,
                epoch,
                round,
                module,
                digest,
                sections: Vec::new(),
            },
        }
    }

    /// The content address of this manifest: blake3 of its canonical-CBOR encoding. This is the
    /// identity a coordinator record names, an attestation signs over, and a late joiner fetches by
    /// (architecture §5.3, §8).
    ///
    /// # Errors
    /// Propagates a codec error (structurally unreachable for this type).
    pub fn content_hash(&self) -> Result<Hash, CheckpointError> {
        Ok(blake3_hash(&self.to_wire()?))
    }

    /// Canonical-CBOR wire bytes.
    ///
    /// # Errors
    /// Propagates a codec error (structurally unreachable for this type).
    pub fn to_wire(&self) -> Result<Vec<u8>, CheckpointError> {
        to_canonical_vec(self).map_err(|e| CheckpointError::Codec(e.to_string()))
    }

    /// Decode a manifest from its wire bytes.
    ///
    /// # Errors
    /// [`CheckpointError::Codec`] on malformed/non-canonical bytes.
    pub fn from_wire(bytes: &[u8]) -> Result<Self, CheckpointError> {
        from_canonical_slice(bytes).map_err(|e| CheckpointError::Codec(e.to_string()))
    }

    /// The first section of the given kind, if present.
    #[must_use]
    pub fn section(&self, kind: SectionKind) -> Option<&CheckpointSection> {
        self.sections.iter().find(|s| s.kind == kind)
    }

    /// The consensus-canonical section (architecture §5.3), if the manifest carries one.
    #[must_use]
    pub fn consensus_section(&self) -> Option<&CheckpointSection> {
        self.section(SectionKind::Consensus)
    }

    /// Validate structural invariants: a supported schema, unique section names, and no two sections
    /// of the same kind. (Section *bytes* are verified against `hash` at load time, not here.)
    ///
    /// # Errors
    /// [`CheckpointError::UnsupportedSchema`], [`CheckpointError::DuplicateSectionName`], or
    /// [`CheckpointError::DuplicateSectionKind`].
    pub fn validate(&self) -> Result<(), CheckpointError> {
        if self.schema != CHECKPOINT_MANIFEST_SCHEMA {
            return Err(CheckpointError::UnsupportedSchema(self.schema));
        }
        for (i, s) in self.sections.iter().enumerate() {
            if self.sections[..i].iter().any(|p| p.name == s.name) {
                return Err(CheckpointError::DuplicateSectionName(s.name.clone()));
            }
            if self.sections[..i].iter().any(|p| p.kind == s.kind) {
                return Err(CheckpointError::DuplicateSectionKind(s.kind));
            }
        }
        Ok(())
    }
}

/// Assembles a [`CheckpointManifest`], hashing section bytes as they are added.
#[derive(Debug, Clone)]
pub struct CheckpointManifestBuilder {
    manifest: CheckpointManifest,
}

impl CheckpointManifestBuilder {
    /// Declare a section from its bytes (class defaulted from the kind, architecture §5.3).
    #[must_use]
    pub fn section(
        mut self,
        name: impl Into<String>,
        kind: SectionKind,
        schema: u64,
        bytes: &[u8],
    ) -> Self {
        self.manifest
            .sections
            .push(CheckpointSection::from_bytes(name, kind, schema, bytes));
        self
    }

    /// Declare a pre-built section (for a caller that overrode the class or hashed bytes elsewhere).
    #[must_use]
    pub fn push(mut self, section: CheckpointSection) -> Self {
        self.manifest.sections.push(section);
        self
    }

    /// Finish, validating structural invariants.
    ///
    /// # Errors
    /// As [`CheckpointManifest::validate`].
    pub fn build(self) -> Result<CheckpointManifest, CheckpointError> {
        self.manifest.validate()?;
        Ok(self.manifest)
    }
}

/// Errors surfaced by the checkpoint-manifest layer. (`Display`/`Error` are hand-written, not
/// `thiserror`-derived, to keep this crate dependency-light + wasm32-clean like the rest of it.)
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CheckpointError {
    /// A manifest declared a schema version this build does not support.
    UnsupportedSchema(u64),
    /// Two sections shared a name.
    DuplicateSectionName(String),
    /// Two sections shared a kind.
    DuplicateSectionKind(SectionKind),
    /// A section-kind wire tag was not recognized.
    UnknownSectionKind(u64),
    /// A section-class wire tag was not recognized.
    UnknownSectionClass(u64),
    /// A manifest (de)serialization step failed.
    Codec(String),
}

impl core::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(
                f,
                "unsupported checkpoint-manifest schema {v} (this build speaks {CHECKPOINT_MANIFEST_SCHEMA})"
            ),
            Self::DuplicateSectionName(n) => write!(f, "duplicate checkpoint section name {n:?}"),
            Self::DuplicateSectionKind(k) => write!(f, "duplicate checkpoint section kind {k:?}"),
            Self::UnknownSectionKind(t) => write!(f, "unknown checkpoint section kind tag {t}"),
            Self::UnknownSectionClass(t) => write!(f, "unknown checkpoint section class tag {t}"),
            Self::Codec(e) => write!(f, "checkpoint manifest codec error: {e}"),
        }
    }
}

impl std::error::Error for CheckpointError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(seed: u8) -> Hash {
        Hash([seed; 32])
    }

    fn sample() -> CheckpointManifest {
        CheckpointManifest::builder(h(1), 3, 7, h(2), StateDigest([0xAB; 16]))
            .section("consensus", SectionKind::Consensus, 1, b"det-state-digest")
            .section("module", SectionKind::Module, 4, b"opaque-module-bytes")
            .section("data-cursor", SectionKind::DataCursor, 1, b"\x07\x00")
            .section(
                "journal-position",
                SectionKind::JournalPosition,
                1,
                b"\x2a\x00",
            )
            .build()
            .unwrap()
    }

    #[test]
    fn manifest_wire_round_trips_and_is_content_addressed() {
        let m = sample();
        let wire = m.to_wire().unwrap();
        let back = CheckpointManifest::from_wire(&wire).unwrap();
        assert_eq!(back, m);
        // Content hash is stable across independent encodings (canonical CBOR).
        assert_eq!(m.content_hash().unwrap(), back.content_hash().unwrap());
        // Any field change changes the content address.
        let mut m2 = m.clone();
        m2.round += 1;
        assert_ne!(m.content_hash().unwrap(), m2.content_hash().unwrap());
    }

    #[test]
    fn section_kinds_and_classes_are_declared_per_architecture_5_3() {
        let m = sample();
        assert_eq!(
            m.consensus_section().unwrap().class,
            SectionClass::ConsensusCanonical
        );
        assert_eq!(
            m.section(SectionKind::Module).unwrap().class,
            SectionClass::RoleLocal
        );
        assert_eq!(
            m.section(SectionKind::JournalPosition).unwrap().class,
            SectionClass::RoleLocal
        );
    }

    #[test]
    fn section_bytes_are_content_addressed() {
        let s =
            CheckpointSection::from_bytes("module", SectionKind::Module, 4, b"opaque-module-bytes");
        assert_eq!(s.hash, blake3_hash(b"opaque-module-bytes"));
        assert_eq!(s.size, b"opaque-module-bytes".len() as u64);
    }

    #[test]
    fn validate_rejects_duplicate_names_and_kinds() {
        let dup_name = CheckpointManifest::builder(h(1), 0, 0, h(2), StateDigest([0; 16]))
            .section("x", SectionKind::Consensus, 1, b"a")
            .section("x", SectionKind::Module, 1, b"b")
            .build();
        assert!(matches!(
            dup_name,
            Err(CheckpointError::DuplicateSectionName(_))
        ));

        let dup_kind = CheckpointManifest::builder(h(1), 0, 0, h(2), StateDigest([0; 16]))
            .section("a", SectionKind::Module, 1, b"a")
            .section("b", SectionKind::Module, 1, b"b")
            .build();
        assert!(matches!(
            dup_kind,
            Err(CheckpointError::DuplicateSectionKind(_))
        ));
    }

    #[test]
    fn unknown_schema_is_rejected() {
        let mut m = sample();
        m.schema = 999;
        assert!(matches!(
            m.validate(),
            Err(CheckpointError::UnsupportedSchema(999))
        ));
    }

    #[test]
    fn section_kind_tags_are_stable() {
        assert_eq!(SectionKind::Consensus.tag(), 0);
        assert_eq!(SectionKind::Module.tag(), 1);
        assert_eq!(SectionKind::WorkerLocal.tag(), 2);
        assert_eq!(SectionKind::DataCursor.tag(), 3);
        assert_eq!(SectionKind::JournalPosition.tag(), 4);
    }
}
