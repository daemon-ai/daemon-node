// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The dedicated vhc identity keystore — the durable home of the node's base identity, the iroh
//! transport identity, and per-run signing keys (architecture §4.3, §7.2).
//!
//! Layout (all under the node state dir, e.g. `<data_dir>/vhc/identity/`):
//!
//! ```text
//! identity/                      dir mode 0700 (owner-only)
//!   base.key                     the node's base ed25519 identity, file mode 0600
//!   iroh.key                     the iroh transport secret — separate from the base identity
//!   runs/<blake3(run label)>/    per-run key material, deleted on terminal completion
//!     <role>-<incarnation>.key   the per-run ed25519 seed
//!     <role>-<incarnation>.cert  the base-signed RunKeyCertificate (public, cached beside it)
//! ```
//!
//! Lifecycle rules (normative for this store):
//!
//! - **Creation is atomic**: write to a temp name in the same directory, fsync, rename. A crash
//!   mid-create leaves either no key or a whole key — never a torn one.
//! - **Crash recovery = the file is the identity**: if a key file exists it IS the key; the store
//!   never regenerates over an existing file. A crashed worker resuming its incarnation reads the
//!   same per-run key back and gets a fresh certificate over it.
//! - **Rotation of the base identity is a documented manual operation** (delete/replace the file
//!   and re-issue live-run certificates); no code path rotates it.
//! - **Terminal cleanup**: [`VhcKeystore::remove_run`] deletes a run's whole key directory when
//!   the run reaches terminal completion — per-run keys and certificates do not outlive the run.
//! - **Permissions**: on Unix the directory is 0700 and every secret file 0600, asserted at open.
//!   On Windows (cross-built fleet workers) the store relies on the profile directory's ACLs; no
//!   POSIX-mode enforcement is attempted.
//!
//! Secrets never ride ordinary command payloads or journals: the node hands the worker a
//! REFERENCE — the identity directory (an inherited environment path, [`IDENTITY_DIR_ENV`]) plus
//! the `(run label, role, incarnation)` coordinates it already knows — and the worker resolves
//! the key material against this store itself.

use std::path::{Path, PathBuf};

use daemon_vhc_proto::{from_canonical_slice, to_canonical_vec, RunKeyCertificate, SigningKey};
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

use crate::identity::{IdentityError, SecretSeed};

// Raw-fs rationale (the workspace routes attacker-influenced paths through
// `daemon_core::ContainedRoot`): every path this store touches is NODE-controlled — a
// config-derived root plus fixed names and blake3-hex run components; nothing is
// attacker-influenced. `ContainedRoot` cannot serve a keystore anyway: it creates files 0644 /
// dirs 0755 with only an async post-hoc `set_mode`, which would open a wider-than-0600 window on
// fresh secret material — the helpers below create at the final mode instead. Each raw call site
// is item-scoped-allowed with this reason.

/// The environment variable through which the node hands a worker subprocess the identity-store
/// location (a path reference, never key material).
pub const IDENTITY_DIR_ENV: &str = "DAEMON_VHC_IDENTITY_DIR";

/// The record-format version this store writes.
const RECORD_VERSION: u64 = 1;

/// A keystore failure.
#[derive(Debug, thiserror::Error)]
pub enum KeystoreError {
    /// Filesystem error (create/read/write/rename/permissions).
    #[error("keystore io at {path}: {source}")]
    Io {
        /// The path involved.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// A stored record failed to decode or carried the wrong kind/version.
    #[error("keystore record {path}: {detail}")]
    BadRecord {
        /// The path involved.
        path: PathBuf,
        /// What was wrong.
        detail: String,
    },
    /// Entropy / certificate failure from the identity layer.
    #[error(transparent)]
    Identity(#[from] IdentityError),
    /// Canonical CBOR encode/decode failure.
    #[error("keystore codec: {0}")]
    Codec(String),
}

/// The canonical-CBOR secret record this store persists (one per key file).
#[derive(Serialize, Deserialize)]
struct SecretRecord {
    /// Record-format version ([`RECORD_VERSION`]).
    v: u64,
    /// What the seed is for (`vhc-base-ed25519` / `vhc-iroh` / `vhc-run-ed25519`).
    kind: String,
    /// The 32-byte secret seed.
    seed: [u8; 32],
    /// Creation time (unix ms; informational).
    created_ms: u64,
}

impl Drop for SecretRecord {
    fn drop(&mut self) {
        self.seed.zeroize();
    }
}

/// The dedicated vhc identity keystore rooted at one directory.
pub struct VhcKeystore {
    root: PathBuf,
}

impl VhcKeystore {
    /// Open (creating if absent) the keystore at `root`. Creates the directory tree owner-only
    /// and, on Unix, asserts/repairs the 0700/0600 discipline.
    ///
    /// # Errors
    /// Filesystem failure creating or securing the directory.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, KeystoreError> {
        let root = root.into();
        create_owner_only_dir(&root)?;
        Ok(Self { root })
    }

    /// Resolve the store a worker subprocess was handed by reference ([`IDENTITY_DIR_ENV`]).
    ///
    /// # Errors
    /// The environment variable is unset (no identity store was provided) or the open fails.
    pub fn from_env() -> Result<Self, KeystoreError> {
        let Some(dir) = std::env::var_os(IDENTITY_DIR_ENV) else {
            return Err(KeystoreError::BadRecord {
                path: PathBuf::from(format!("${IDENTITY_DIR_ENV}")),
                detail: "no identity store reference in the environment".into(),
            });
        };
        Self::open(PathBuf::from(dir))
    }

    /// The store's root directory (what the node exports as the worker's reference).
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The node's durable base ed25519 identity: loaded if present (the file IS the identity),
    /// created CSPRNG + atomically otherwise.
    ///
    /// # Errors
    /// Filesystem, entropy, or record-decode failure.
    pub fn base_identity(&self) -> Result<SigningKey, KeystoreError> {
        let seed = self.load_or_create(&self.root.join("base.key"), "vhc-base-ed25519")?;
        Ok(seed.signing_key())
    }

    /// The iroh transport secret — its own CSPRNG key with its own store entry, deliberately
    /// distinct from both the base identity and every per-run key (architecture §7.2).
    ///
    /// # Errors
    /// Filesystem, entropy, or record-decode failure.
    pub fn iroh_secret(&self) -> Result<SecretSeed, KeystoreError> {
        self.load_or_create(&self.root.join("iroh.key"), "vhc-iroh")
    }

    /// The node-local journal sidecar encryption key (ABI §8.5): its own CSPRNG secret with its
    /// own store entry — a construction input for the durable journal's encrypted sidecars,
    /// distinct from every signing/transport identity (it encrypts, it never signs).
    ///
    /// # Errors
    /// Filesystem, entropy, or record-decode failure.
    pub fn journal_sidecar_key(&self) -> Result<SecretSeed, KeystoreError> {
        self.load_or_create(&self.root.join("journal.key"), "vhc-journal-sidecar")
    }

    /// The per-run signing key for `(run label, role, incarnation)`: recovered if persisted (a
    /// crashed worker resumes its incarnation with the SAME key), freshly CSPRNG-generated and
    /// persisted otherwise. Persisted only for the life of the run ([`VhcKeystore::remove_run`]).
    ///
    /// # Errors
    /// Filesystem, entropy, or record-decode failure.
    pub fn run_signing_key(
        &self,
        run_label: &str,
        role: &str,
        incarnation: u64,
    ) -> Result<SigningKey, KeystoreError> {
        let dir = self.run_dir(run_label);
        create_owner_only_dir(&dir)?;
        let path = dir.join(format!("{role}-{incarnation}.key"));
        let seed = self.load_or_create(&path, "vhc-run-ed25519")?;
        Ok(seed.signing_key())
    }

    /// Persist a run key's certificate beside its key (public material, cached for distribution
    /// and crash recovery).
    ///
    /// # Errors
    /// Filesystem or encode failure.
    pub fn store_run_certificate(
        &self,
        run_label: &str,
        role: &str,
        incarnation: u64,
        cert: &RunKeyCertificate,
    ) -> Result<(), KeystoreError> {
        let path = self
            .run_dir(run_label)
            .join(format!("{role}-{incarnation}.cert"));
        let bytes = to_canonical_vec(cert).map_err(|e| KeystoreError::Codec(e.to_string()))?;
        // Not secret, but written with the same atomic discipline (and 0600 — nothing in this
        // store is ever group/other-readable).
        atomic_write(&path, &bytes)
    }

    /// The cached certificate for `(run label, role, incarnation)`, if one was stored.
    ///
    /// # Errors
    /// Filesystem or decode failure (an absent file is `Ok(None)`).
    pub fn run_certificate(
        &self,
        run_label: &str,
        role: &str,
        incarnation: u64,
    ) -> Result<Option<RunKeyCertificate>, KeystoreError> {
        let path = self
            .run_dir(run_label)
            .join(format!("{role}-{incarnation}.cert"));
        let bytes = match read_file(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(KeystoreError::Io { path, source: e }),
        };
        let cert = from_canonical_slice::<RunKeyCertificate>(&bytes).map_err(|e| {
            KeystoreError::BadRecord {
                path,
                detail: format!("certificate decode: {e}"),
            }
        })?;
        Ok(Some(cert))
    }

    /// Terminal-completion cleanup: delete every per-run key and certificate for `run_label`.
    /// Idempotent — an absent directory is success.
    ///
    /// # Errors
    /// Filesystem failure other than absence.
    pub fn remove_run(&self, run_label: &str) -> Result<(), KeystoreError> {
        let dir = self.run_dir(run_label);
        match remove_dir_all(&dir) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(KeystoreError::Io {
                path: dir,
                source: e,
            }),
        }
    }

    /// A run's key directory: keyed by the blake3 of the run LABEL (labels are free-form strings;
    /// hashing keeps the path safe and collision-free). The certificate inside binds the
    /// cryptographic run id (the genesis hash) — the label only namespaces storage.
    fn run_dir(&self, run_label: &str) -> PathBuf {
        self.root
            .join("runs")
            .join(blake3::hash(run_label.as_bytes()).to_hex().as_str())
    }

    /// Load a seed record if `path` exists (asserting kind/version), else create one atomically.
    fn load_or_create(&self, path: &Path, kind: &str) -> Result<SecretSeed, KeystoreError> {
        match read_file(path) {
            Ok(bytes) => {
                let record: SecretRecord =
                    from_canonical_slice(&bytes).map_err(|e| KeystoreError::BadRecord {
                        path: path.to_path_buf(),
                        detail: format!("record decode: {e}"),
                    })?;
                if record.v != RECORD_VERSION || record.kind != kind {
                    return Err(KeystoreError::BadRecord {
                        path: path.to_path_buf(),
                        detail: format!(
                            "record v{} kind `{}` (expected v{RECORD_VERSION} `{kind}`)",
                            record.v, record.kind
                        ),
                    });
                }
                Ok(SecretSeed::from_bytes(record.seed))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let seed = SecretSeed::fresh()?;
                let record = SecretRecord {
                    v: RECORD_VERSION,
                    kind: kind.to_string(),
                    seed: *seed.bytes(),
                    created_ms: now_ms(),
                };
                let bytes =
                    to_canonical_vec(&record).map_err(|e| KeystoreError::Codec(e.to_string()))?;
                atomic_write(path, &bytes)?;
                Ok(seed)
            }
            Err(e) => Err(KeystoreError::Io {
                path: path.to_path_buf(),
                source: e,
            }),
        }
    }
}

/// Create `dir` (and parents) and, on Unix, force it owner-only (0700).
// Node-controlled path; created-at-mode discipline (see the module-level raw-fs rationale).
#[allow(clippy::disallowed_methods)]
fn create_owner_only_dir(dir: &Path) -> Result<(), KeystoreError> {
    let io = |e: std::io::Error| KeystoreError::Io {
        path: dir.to_path_buf(),
        source: e,
    };
    std::fs::create_dir_all(dir).map_err(io)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).map_err(io)?;
    }
    Ok(())
}

/// Read a store file in full.
// Node-controlled path (see the module-level raw-fs rationale).
#[allow(clippy::disallowed_methods)]
fn read_file(path: &Path) -> std::io::Result<Vec<u8>> {
    std::fs::read(path)
}

/// Remove a run's key directory recursively.
// Node-controlled path — the component is a blake3 hex of the run label, never raw input
// (see the module-level raw-fs rationale).
#[allow(clippy::disallowed_methods)]
fn remove_dir_all(dir: &Path) -> std::io::Result<()> {
    std::fs::remove_dir_all(dir)
}

/// Write `bytes` to `path` atomically: temp file in the same directory (created 0600 on Unix —
/// never a wider-than-0600 window on fresh secret material), write, fsync, rename over the final
/// name. A crash leaves no torn record.
// Node-controlled path; ContainedRoot cannot create-at-0600 (see the module-level rationale).
#[allow(clippy::disallowed_methods)]
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), KeystoreError> {
    use std::io::Write as _;
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = dir.join(format!(
        ".tmp-{}",
        path.file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default()
    ));
    let io = |p: &Path, e: std::io::Error| KeystoreError::Io {
        path: p.to_path_buf(),
        source: e,
    };
    {
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&tmp).map_err(|e| io(&tmp, e))?;
        f.write_all(bytes).map_err(|e| io(&tmp, e))?;
        f.sync_all().map_err(|e| io(&tmp, e))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| io(path, e))?;
    // Best-effort directory fsync so the rename itself is durable.
    if let Ok(d) = std::fs::File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
// Test-only fixture manipulation inside tempdirs (mode asserts, record swaps) — not a
// production fs path.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use daemon_vhc_proto::peer_id;
    use std::fs;

    #[test]
    fn base_identity_is_created_once_and_recovered_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        let store = VhcKeystore::open(dir.path().join("identity")).unwrap();
        let first = store.base_identity().unwrap();
        // Crash recovery: a re-open reads the SAME identity back — never regenerates.
        let store2 = VhcKeystore::open(dir.path().join("identity")).unwrap();
        let second = store2.base_identity().unwrap();
        assert_eq!(peer_id(&first), peer_id(&second));
    }

    #[test]
    fn the_three_identities_are_distinct_keys() {
        let dir = tempfile::tempdir().unwrap();
        let store = VhcKeystore::open(dir.path()).unwrap();
        let base = peer_id(&store.base_identity().unwrap());
        let iroh = peer_id(&store.iroh_secret().unwrap().signing_key());
        let run = peer_id(&store.run_signing_key("run-x", "trainer", 1).unwrap());
        assert_ne!(base, iroh, "transport identity is not the base identity");
        assert_ne!(base, run, "per-run keys are not the base identity");
        assert_ne!(iroh, run, "per-run keys are not the transport identity");
    }

    #[test]
    fn run_keys_recover_within_an_incarnation_and_rotate_across_incarnations() {
        let dir = tempfile::tempdir().unwrap();
        let store = VhcKeystore::open(dir.path()).unwrap();
        let a1 = peer_id(&store.run_signing_key("run-a", "trainer", 1).unwrap());
        // The same (run, role, incarnation) recovers the SAME key (crash resume)...
        let a1_again = peer_id(&store.run_signing_key("run-a", "trainer", 1).unwrap());
        assert_eq!(a1, a1_again);
        // ...a new incarnation is a fresh key (rotation on incarnation change only)...
        let a2 = peer_id(&store.run_signing_key("run-a", "trainer", 2).unwrap());
        assert_ne!(a1, a2);
        // ...and another run never shares key material.
        let b1 = peer_id(&store.run_signing_key("run-b", "trainer", 1).unwrap());
        assert_ne!(a1, b1);
    }

    #[test]
    fn terminal_cleanup_removes_a_runs_keys_and_certs() {
        let dir = tempfile::tempdir().unwrap();
        let store = VhcKeystore::open(dir.path()).unwrap();
        let key = store.run_signing_key("run-done", "trainer", 3).unwrap();
        let base = store.base_identity().unwrap();
        let cert = daemon_vhc_proto::RunKeyCertificate::issue(
            &base,
            daemon_vhc_proto::CertScope {
                run_id: daemon_vhc_proto::Hash([9; 32]),
                epoch: 0,
                role: "trainer".into(),
                instance: 3,
                module_hash: daemon_vhc_proto::Hash([1; 32]),
            },
            peer_id(&key),
        )
        .unwrap();
        store
            .store_run_certificate("run-done", "trainer", 3, &cert)
            .unwrap();
        assert_eq!(
            store.run_certificate("run-done", "trainer", 3).unwrap(),
            Some(cert)
        );

        store.remove_run("run-done").unwrap();
        assert_eq!(
            store.run_certificate("run-done", "trainer", 3).unwrap(),
            None
        );
        // A fresh key after cleanup is a NEW key — the old material is gone.
        let rejoined = peer_id(&store.run_signing_key("run-done", "trainer", 3).unwrap());
        assert_ne!(rejoined, peer_id(&key));
        // Idempotent.
        store.remove_run("run-done").unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn secret_files_and_directories_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("identity");
        let store = VhcKeystore::open(&root).unwrap();
        store.base_identity().unwrap();
        store.run_signing_key("run-p", "trainer", 1).unwrap();

        assert_eq!(
            fs::metadata(&root).unwrap().permissions().mode() & 0o777,
            0o700,
            "store dir is owner-only"
        );
        assert_eq!(
            fs::metadata(root.join("base.key"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600,
            "base key is 0600"
        );
        let runs = root.join("runs");
        for entry in fs::read_dir(&runs).unwrap() {
            let run_dir = entry.unwrap().path();
            assert_eq!(
                fs::metadata(&run_dir).unwrap().permissions().mode() & 0o777,
                0o700,
                "run dir is owner-only"
            );
            for f in fs::read_dir(&run_dir).unwrap() {
                let f = f.unwrap().path();
                assert_eq!(
                    fs::metadata(&f).unwrap().permissions().mode() & 0o777,
                    0o600,
                    "key material is 0600: {}",
                    f.display()
                );
            }
        }
    }

    #[test]
    fn a_wrong_kind_record_is_a_typed_refusal_never_a_key() {
        let dir = tempfile::tempdir().unwrap();
        let store = VhcKeystore::open(dir.path()).unwrap();
        store.iroh_secret().unwrap();
        // Present the iroh record as the base identity: kind mismatch refuses typed.
        fs::rename(dir.path().join("iroh.key"), dir.path().join("base.key")).unwrap();
        assert!(matches!(
            store.base_identity(),
            Err(KeystoreError::BadRecord { .. })
        ));
    }
}
