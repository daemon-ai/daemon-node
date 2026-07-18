// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Migration scaffolding (ABI §10.2): the typed **state-manifest** protocol — state never
//! crosses an upgrade as one opaque byte-slice; it is named, versioned, hashed sections, the
//! same shape as checkpoint manifests, so checkpointing and migration are one discipline.
//!
//! Phase-A scope (refactor §5 A2): the wire types, the producing/consuming traits, and the sim
//! round-trip — a tested surface for Phase E's upgrade transaction to call. Full host-side
//! materialization (`stage_state`/`snapshot_state` staging, the §10.3 transaction) is Phase E.

use daemon_vhc_proto::{blake3_hash, from_canonical_slice, to_canonical_vec, Hash};
use serde::{Deserialize, Serialize};

/// One declared state section (ABI §10.2 `state-section-decl`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionDecl {
    /// Section name, e.g. `"consensus"`, `"optimizer"`, `"data-cursor"`.
    pub name: String,
    /// Per-section schema version.
    pub schema: u64,
    /// blake3 of the section bytes.
    pub hash: Hash,
    /// Section byte length.
    pub size: u64,
    /// 0 = consensus-canonical, 1 = role/replica-local (architecture §5.3).
    pub class: u64,
}

/// The state manifest (ABI §10.2 `state-manifest`) — journaled verbatim on snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateManifest {
    /// Module-defined state schema version.
    pub schema: u64,
    /// The producing module hash.
    pub module: Hash,
    /// The declared sections, in staging order.
    pub sections: Vec<SectionDecl>,
}

/// One restore binding (ABI §10.2 `migration-section`): the staging ID is carried **in the
/// descriptor itself** — the consuming module is not in `da_run` and sees no `PayloadReady`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationSection {
    /// = the corresponding `SectionDecl::name`.
    pub name: String,
    /// The restore staging ID (`read_back(id, kind = 3)`, legal during `da_migrate`, §6.6).
    pub staging_id: u64,
}

/// The host-produced migration descriptor (ABI §10.2) — never section bytes, never old memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationDescriptor {
    /// The old module's accepted manifest, verbatim.
    pub manifest: StateManifest,
    /// Restore bindings, in `manifest.sections` order.
    pub sections: Vec<MigrationSection>,
}

impl MigrationDescriptor {
    /// Canonical CBOR wire bytes (what `da_migrate` receives).
    ///
    /// # Errors
    /// Propagates the codec error (structurally unreachable for these types).
    pub fn to_wire(&self) -> Result<Vec<u8>, daemon_vhc_proto::VhcProtoError> {
        to_canonical_vec(self)
    }

    /// Decode the descriptor `da_migrate` received.
    ///
    /// # Errors
    /// Malformed/non-canonical descriptor bytes.
    pub fn from_wire(bytes: &[u8]) -> Result<Self, daemon_vhc_proto::VhcProtoError> {
        from_canonical_slice(bytes)
    }
}

/// A section produced by the old module's snapshot (bytes still guest-side; `stage_state`
/// seals them host-side at Phase E — in sim the harness holds them directly).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedSection {
    /// Section name (unique within one snapshot).
    pub name: String,
    /// Per-section schema version.
    pub schema: u64,
    /// 0 = consensus-canonical, 1 = role/replica-local.
    pub class: u64,
    /// The section bytes.
    pub bytes: Vec<u8>,
}

/// Reads staged section bytes by staging ID: `read_back(kind = 3)` on the wasm guest; an
/// in-memory map in sim (the round-trip tests) and in the Phase-E host harness.
pub trait SectionReader {
    /// The bytes behind `staging_id` (hash-verified host-side before staging, §10.2).
    fn read(&mut self, staging_id: u64) -> Vec<u8>;
}

/// The producing/consuming pair a migratable module implements; `main!`'s `da_migrate` and the
/// Phase-E quiesce path drive it. Kept separate from [`crate::module::V2Module`] so a
/// non-migratable module implements nothing extra.
pub trait MigrateState: Sized {
    /// Snapshot the module state as named sections (the `Quiesce{Upgrade}` drain, §10.2).
    fn snapshot(&self) -> Vec<OwnedSection>;

    /// Reconstruct state from a descriptor + staged sections. Returns `0` (`Ready`) or
    /// `1`/`≥16` (`Incompatible` detail, §10.2).
    fn restore(&mut self, descriptor: &MigrationDescriptor, reader: &mut dyn SectionReader) -> u32;
}

/// Build the state manifest for a snapshot (hashes + sizes computed here, in-guest at Phase E).
#[must_use]
pub fn build_manifest(module: Hash, schema: u64, sections: &[OwnedSection]) -> StateManifest {
    StateManifest {
        schema,
        module,
        sections: sections
            .iter()
            .map(|s| SectionDecl {
                name: s.name.clone(),
                schema: s.schema,
                hash: blake3_hash(&s.bytes),
                size: s.bytes.len() as u64,
                class: s.class,
            })
            .collect(),
    }
}

/// An in-memory [`SectionReader`] for sim round-trips: staging IDs are indices the harness
/// assigned. Verifies the manifest hash discipline on read, exactly as the host does before
/// staging (§10.2) — a corrupted section fails loud in the test, not silently in the restore.
pub struct SimSections {
    sections: Vec<(u64, Vec<u8>)>,
}

impl SimSections {
    /// Stage `sections` under synthetic guest-style staging IDs (`(1 << 63) | counter`, §10.2).
    #[must_use]
    pub fn stage(sections: &[OwnedSection]) -> (Self, Vec<MigrationSection>) {
        let mut staged = Vec::new();
        let mut bindings = Vec::new();
        for (i, s) in sections.iter().enumerate() {
            let id = (1u64 << 63) | (i as u64 + 1);
            staged.push((id, s.bytes.clone()));
            bindings.push(MigrationSection {
                name: s.name.clone(),
                staging_id: id,
            });
        }
        (Self { sections: staged }, bindings)
    }
}

impl SectionReader for SimSections {
    fn read(&mut self, staging_id: u64) -> Vec<u8> {
        self.sections
            .iter()
            .find(|(id, _)| *id == staging_id)
            .map(|(_, b)| b.clone())
            .expect("staged section for staging_id")
    }
}

/// The sim round-trip (the Phase-A acceptance for this scaffolding): snapshot `old` → manifest →
/// descriptor → restore into `new`. Returns `da_migrate`'s status code (`0` = `Ready`).
pub fn roundtrip<S: MigrateState>(old: &S, new: &mut S, module: Hash, schema: u64) -> u32 {
    let sections = old.snapshot();
    let manifest = build_manifest(module, schema, &sections);
    let (mut reader, bindings) = SimSections::stage(&sections);
    let descriptor = MigrationDescriptor {
        manifest,
        sections: bindings,
    };
    // The wire round-trip is part of the oracle: what `da_migrate` decodes is what E encodes.
    let wire = descriptor.to_wire().expect("descriptor wire");
    let decoded = MigrationDescriptor::from_wire(&wire).expect("descriptor decode");
    assert_eq!(decoded, descriptor, "descriptor codec round-trip");
    new.restore(&decoded, &mut reader)
}
