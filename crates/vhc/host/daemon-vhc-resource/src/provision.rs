// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **provisioned-profile file** — how certified Backend Execution Profiles reach a worker
//! (`docs/specs/vhc-architecture-spec.md` §9.6, `[PC-12]`).
//!
//! ## What this module is, and is deliberately not
//!
//! A worker composes a Physical Estimate only from an *authenticated* profile, and authentication
//! needs three owner-side inputs the worker cannot invent: the profiles themselves with their
//! trust envelopes, the machine owner's acceptance policy, and the lane's sanity bounds. Those are
//! **data provisioned by the node operator**, so this module is exactly a file format plus its
//! read/write: canonical CBOR at a directory the node hands the worker by path reference
//! ([`PROFILE_DIR_ENV`]), mirroring how the identity store travels
//! (`daemon_vhc_session::keystore::IDENTITY_DIR_ENV`).
//!
//! What it is **not** is an authoring path. Nothing here constructs a profile, an envelope or a
//! policy; production code may read and carry the operator's values, and the minting of
//! development artifacts stays in dev tooling (the `test-support` fixtures for tests, an `xtask`
//! command for real boxes) — a production binary that could mint a profile would be a production
//! binary that could vouch for itself, which is the substitution the trust gate exists to refuse.
//!
//! ## Absence is a state, not an error
//!
//! A box with no provisioned file is today's truthful configuration: the worker passes no resource
//! authority and a certification-minor module refuses `EstimateNotComposable`, typed. So
//! [`load_from_env`] returns `Option` — absent env, absent directory and absent file are all
//! `None` — while a file that *exists* but cannot be decoded is an error, because a corrupt
//! provisioning is an operator's problem to hear about, never something to silently run without.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use daemon_vhc_proto::{from_canonical_slice, to_canonical_vec};
use serde::{Deserialize, Serialize};

use crate::planner::LaneEstimateBounds;
use crate::profile::BackendExecutionProfile;
use crate::store::{ProfileStore, StoreRefusal};
use crate::trust::{ProfileAcceptancePolicy, ProfileTrustEnvelope};

/// The environment variable naming the directory that holds the provisioned-profile file. A path
/// REFERENCE — profile bytes never ride the command wire, exactly like the identity store.
pub const PROFILE_DIR_ENV: &str = "DAEMON_VHC_PROFILE_DIR";

/// The file inside the profile directory. One file, not one-per-profile: the owner policy and the
/// lane bounds are statements about the whole box, and scattering them across per-profile files
/// would invite two files to disagree about them.
pub const PROFILES_FILE_NAME: &str = "profiles.cbor";

/// Schema identity for [`ProvisionedProfiles`]' canonical encoding.
pub const PROVISIONED_PROFILES_SCHEMA: u32 = 1;

/// One provisioned profile: the profile and the trust envelope that vouches for it. Always the
/// pair — a profile without its envelope is unauthenticatable by construction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvisionedEntry {
    /// The Backend Execution Profile.
    pub profile: BackendExecutionProfile,
    /// The trust envelope binding it.
    pub envelope: ProfileTrustEnvelope,
}

/// Everything the operator provisions for profile authentication on one box: the profiles with
/// their envelopes, the owner's acceptance policy, and the lane's estimate bounds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvisionedProfiles {
    /// Encoding identity, [`PROVISIONED_PROFILES_SCHEMA`].
    pub schema: u32,
    /// The machine owner's acceptance policy — whose vouching this box accepts.
    pub owner_policy: ProfileAcceptancePolicy,
    /// The lane's sanity bounds per backend class. Owner-side lane configuration: the lane refuses
    /// composition for a class with no bounds (`LaneStatesNoBoundsForClass`), so a provisioning
    /// that stocks a class's profile without its bounds would be self-defeating.
    pub lane_bounds: LaneEstimateBounds,
    /// The provisioned profiles.
    pub entries: Vec<ProvisionedEntry>,
}

impl ProvisionedProfiles {
    /// Stock a [`ProfileStore`] with every provisioned entry.
    ///
    /// # Errors
    /// [`StoreRefusal`] when an entry does not hold together (digest mismatch, collision) — a
    /// refusal about the provisioned data, surfaced verbatim.
    pub fn stock(&self, store: &mut ProfileStore) -> Result<(), StoreRefusal> {
        for entry in &self.entries {
            store.insert(entry.profile.clone(), entry.envelope.clone())?;
        }
        Ok(())
    }
}

/// Why reading or writing a provisioned file refused.
#[derive(Debug, thiserror::Error)]
pub enum ProvisionError {
    /// The file exists but is not a valid provisioned-profile file.
    #[error("provisioned profiles at {path}: {detail}")]
    Malformed {
        /// The offending file.
        path: PathBuf,
        /// What refused.
        detail: String,
    },
    /// The file or directory could not be read or written.
    #[error("provisioned profiles I/O at {path}: {source}")]
    Io {
        /// The path involved.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },
    /// The file carries a schema this build does not read.
    #[error("provisioned profiles at {path}: schema {actual} (this build reads {expected})")]
    Schema {
        /// The offending file.
        path: PathBuf,
        /// What the file said.
        actual: u32,
        /// What this build reads.
        expected: u32,
    },
}

/// Write the provisioned set as canonical CBOR into `dir` (created if absent).
///
/// # Errors
/// [`ProvisionError`] on encoding or I/O failure.
// Raw fs, declared: the destination is the node-owned profile directory (operator configuration,
// the node derives and creates it under its own data dir) — never an attacker-influenced path,
// so `ContainedRoot` containment adds nothing here. Same posture as the identity keystore.
#[allow(clippy::disallowed_methods)]
pub fn write(dir: &Path, set: &ProvisionedProfiles) -> Result<PathBuf, ProvisionError> {
    std::fs::create_dir_all(dir).map_err(|source| ProvisionError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    let path = dir.join(PROFILES_FILE_NAME);
    let bytes = to_canonical_vec(set).map_err(|e| ProvisionError::Malformed {
        path: path.clone(),
        detail: e.to_string(),
    })?;
    std::fs::write(&path, bytes).map_err(|source| ProvisionError::Io {
        path: path.clone(),
        source,
    })?;
    Ok(path)
}

/// Read the provisioned set from `dir`, if the file is present.
///
/// `Ok(None)` when the directory or the file does not exist — the un-provisioned box, a state and
/// not an error. Everything else that goes wrong is a typed refusal: a file that exists speaks for
/// the operator, and mis-reading it silently would run the box on a configuration nobody stated.
///
/// # Errors
/// [`ProvisionError`] when the file exists but cannot be read or decoded, or carries a schema this
/// build does not read.
// Raw fs, declared: the source is the node-owned profile directory the spawning node named by
// path reference (operator configuration, like the identity keystore) — not attacker-influenced.
#[allow(clippy::disallowed_methods)]
pub fn load(dir: &Path) -> Result<Option<ProvisionedProfiles>, ProvisionError> {
    let path = dir.join(PROFILES_FILE_NAME);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(ProvisionError::Io { path, source }),
    };
    let set: ProvisionedProfiles =
        from_canonical_slice(&bytes).map_err(|e| ProvisionError::Malformed {
            path: path.clone(),
            detail: e.to_string(),
        })?;
    if set.schema != PROVISIONED_PROFILES_SCHEMA {
        return Err(ProvisionError::Schema {
            path,
            actual: set.schema,
            expected: PROVISIONED_PROFILES_SCHEMA,
        });
    }
    Ok(Some(set))
}

/// Read the provisioned set from the directory [`PROFILE_DIR_ENV`] names, if both the variable and
/// the file are present.
///
/// # Errors
/// As [`load`]: a named-but-unreadable file refuses rather than degrading to `None`.
pub fn load_from_env() -> Result<Option<ProvisionedProfiles>, ProvisionError> {
    match std::env::var_os(PROFILE_DIR_ENV) {
        Some(dir) => load(Path::new(&dir)),
        None => Ok(None),
    }
}

/// The set of backend-class slugs the provisioned entries cover — what this box can compose for.
#[must_use]
pub fn provisioned_classes(set: &ProvisionedProfiles) -> BTreeSet<String> {
    set.entries
        .iter()
        .map(|e| e.profile.backend_class.slug().to_string())
        .collect()
}

#[cfg(test)]
// Raw fs in fixtures: test tooling writing into its own tempdir, never a shipped path.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::revision::BackendClass;

    fn provisioned_fixture() -> ProvisionedProfiles {
        let running = crate::revision::fixtures::revision(BackendClass::Cpu);
        let profile = crate::trust::fixtures::profile_for(&running);
        let envelope = crate::trust::fixtures::envelope_for(&profile, &running);
        let store = ProfileStore::new();
        let owner_policy = crate::trust::fixtures::policy_for(&store);
        ProvisionedProfiles {
            schema: PROVISIONED_PROFILES_SCHEMA,
            owner_policy,
            lane_bounds: LaneEstimateBounds {
                by_backend_class: [("cpu".to_string(), [0u64, 1 << 40])].into_iter().collect(),
            },
            entries: vec![ProvisionedEntry { profile, envelope }],
        }
    }

    #[test]
    fn round_trips_through_the_file_and_stocks_a_store() {
        let dir = tempfile::tempdir().expect("tempdir");
        let set = provisioned_fixture();
        write(dir.path(), &set).expect("write");
        let loaded = load(dir.path()).expect("load").expect("present");
        assert_eq!(loaded, set);
        let mut store = ProfileStore::new();
        loaded.stock(&mut store).expect("stock");
        assert_eq!(
            provisioned_classes(&loaded).into_iter().collect::<Vec<_>>(),
            vec!["cpu".to_string()]
        );
    }

    #[test]
    fn absent_directory_and_absent_file_are_none_not_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(load(dir.path()).expect("empty dir is a state").is_none());
        assert!(load(&dir.path().join("never-created"))
            .expect("absent dir is a state")
            .is_none());
    }

    #[test]
    fn a_present_but_corrupt_file_refuses_rather_than_degrading_to_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(PROFILES_FILE_NAME), b"not cbor").expect("write");
        let err = load(dir.path()).expect_err("corrupt file must refuse");
        assert!(matches!(err, ProvisionError::Malformed { .. }), "{err}");
    }

    #[test]
    fn a_foreign_schema_refuses_typed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut set = provisioned_fixture();
        set.schema = PROVISIONED_PROFILES_SCHEMA + 1;
        // Bypass `write`'s value as-is: the writer writes what it is given; the READER owns the
        // schema gate, because the reader is the one about to act on the contents.
        write(dir.path(), &set).expect("write");
        let err = load(dir.path()).expect_err("foreign schema must refuse");
        assert!(matches!(err, ProvisionError::Schema { actual, .. } if actual == set.schema));
    }

    #[test]
    fn env_reference_is_a_path_reference() {
        // Drift-pin: the identity-store convention is a path reference in a stable variable name;
        // renaming it breaks every spawn site silently, so the name is pinned here.
        assert_eq!(PROFILE_DIR_ENV, "DAEMON_VHC_PROFILE_DIR");
        assert_eq!(PROFILES_FILE_NAME, "profiles.cbor");
    }
}
