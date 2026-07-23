// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Migration scaffolding (ABI §10.2): the typed **state-manifest** protocol — state never
//! crosses an upgrade as one opaque byte-slice; it is named, versioned, hashed sections, the
//! same shape as checkpoint manifests, so checkpointing and migration are one discipline.
//!
//! Phase-A scope (refactor §5 A2): the wire types, the producing/consuming traits, and the sim
//! round-trip — a tested surface for Phase E's upgrade transaction to call. Full host-side
//! materialization (`stage_state`/`snapshot_state` staging, the §10.3 transaction) is Phase E.

use daemon_vhc_proto::det_state::FamilyRef;
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

/// One restore binding (ABI §10.2 `migration-section`) — carried **in the descriptor itself**
/// (the consuming module is not in `da_run` and sees no `PayloadReady`). Two forms, distinguished
/// structurally (untagged) by the second field, mirroring the checkpoint-document section forms
/// ([SF-6]):
///
/// - **Inline** (`{name, staging_id}`): small state (the round watermark) staged host-side and
///   read via `read_back(id, kind = 3)`, legal during `da_migrate` (§6.6).
/// - **By-reference** (`{name, family}`): an already-sealed family the restoring instance
///   REGISTERS ([SF-R2]) and streams window-by-window in `da_run` — never bulk-read in
///   `da_migrate` (whose only legal read is `read_back(kind = 3)`, §6.6). This is how the
///   `VhcProtoVersion`-2 descriptor carries master/ef/adamw families without moving their bytes
///   through the migrate seam.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MigrationSection {
    /// `{name, family}` — a by-reference family the new instance registers + streams in `da_run`.
    ByRef {
        /// = the corresponding `SectionDecl::name`.
        name: String,
        /// The already-sealed family artifact (fold + geometry + ordered chunk hashes).
        family: FamilyRef,
    },
    /// `{name, staging_id}` — an inline section staged host-side (`read_back(kind = 3)`).
    Inline {
        /// = the corresponding `SectionDecl::name`.
        name: String,
        /// The restore staging ID (`read_back(id, kind = 3)`, legal during `da_migrate`, §6.6).
        staging_id: u64,
    },
}

impl MigrationSection {
    /// The section name (either form).
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::ByRef { name, .. } | Self::Inline { name, .. } => name,
        }
    }
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
/// Phase-E quiesce path drive it. Kept separate from [`crate::module::GuestModule`] so a
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
            bindings.push(MigrationSection::Inline {
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

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_proto::det_state::FamilyRef;

    #[test]
    fn migration_section_by_ref_round_trips_through_wire() {
        let fref = FamilyRef {
            fold: Hash([7u8; 32]),
            byte_len: 96,
            chunk_size: 24,
            chunk_hashes: vec![
                Hash([1u8; 32]),
                Hash([2u8; 32]),
                Hash([3u8; 32]),
                Hash([4u8; 32]),
            ],
        };
        let descriptor = MigrationDescriptor {
            manifest: StateManifest {
                schema: 1,
                module: Hash([0u8; 32]),
                sections: vec![
                    SectionDecl {
                        name: "master".into(),
                        schema: 1,
                        hash: fref.fold,
                        size: fref.byte_len,
                        class: 0,
                    },
                    SectionDecl {
                        name: "round".into(),
                        schema: 1,
                        hash: blake3_hash(&7u64.to_le_bytes()),
                        size: 8,
                        class: 1,
                    },
                ],
            },
            sections: vec![
                MigrationSection::ByRef {
                    name: "master".into(),
                    family: fref.clone(),
                },
                MigrationSection::Inline {
                    name: "round".into(),
                    staging_id: (1u64 << 63) | 1,
                },
            ],
        };
        let wire = descriptor.to_wire().expect("descriptor to_wire");
        let back = MigrationDescriptor::from_wire(&wire).expect("descriptor from_wire");
        assert_eq!(
            back, descriptor,
            "by-ref + inline sections survive the wire round-trip"
        );
        match &back.sections[0] {
            MigrationSection::ByRef { name, family } => {
                assert_eq!(name, "master");
                assert_eq!(family, &fref);
            }
            other => panic!("first section should be by-ref, got {other:?}"),
        }
    }
}
