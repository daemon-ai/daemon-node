// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **on-disk spill** for the host state store (design §8.1): a synchronous,
//! content-addressed chunk directory the [`crate::run::state_store::StateStore`] writes canonical
//! det-lane chunks through when a run pins a state directory, so the retained families live on
//! disk instead of resident RAM.
//!
//! # Why this exists (the memory floor)
//!
//! The streamed det-fold substrate moves canonical state host-side; at the ceremony tier the
//! retained roots at the ratified cadence are ≈ 14.65 GiB (≈ 5 families). On the fleet's memory
//! floor peer — the M4 Mac's 32 GiB *unified* memory — that cannot be RAM-resident beside the
//! on-device training working set (≈ 11.72 GiB) without overrunning the usable budget before
//! activations/OS. So the state store spills chunk BYTES to disk and keeps only the index
//! (lengths, refcounts, seal order, token bucket) in RAM. This closes the divergence between the
//! design (§8.1: "an `FsContentStore`-class store rooted in the run's state directory") and the
//! first landed store, which was entirely RAM-resident.
//!
//! # Sanctioned raw-fs home
//!
//! Like the journal store (`daemon-vhc-journal`), this is a host-owned durable store on the
//! synchronous guest-driver thread, so it uses `std::fs` directly under a scoped
//! `disallowed_methods` allow rather than the async [`ContainedRoot`] seam (the pump thread is
//! not async). No spawn / env mutation here; paths are derived from content hashes (hex), never
//! from attacker-influenced input.
//!
//! # Custody is preserved, fail-loud
//!
//! The store is the write-side verifier by construction (the host hashed the bytes at
//! `state_emit` before handing them here), and every read re-hashes the served bytes against the
//! requested content address: a disk object that is missing or does not re-hash is the SAME loud,
//! typed fault as a resident miss — never a silent substitution (design §8: "the pump is the sole
//! verifier"; the custody cross-check does not weaken because the bytes moved to disk).

// Sanctioned raw-fs home (module docs): host-owned durable state-chunk spill on the synchronous
// driver thread. No spawn / env mutation here.
#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use daemon_vhc_proto::{blake3_hash, Hash};

/// A read failure from the spill — mapped by the state store onto the same loud outcomes a
/// resident miss / custody violation produces.
#[derive(Debug)]
pub enum SpillReadError {
    /// The object is not on disk (an evicted or never-written chunk).
    Missing,
    /// The object is present but does not re-hash to the requested content address — a custody
    /// violation (tamper or corruption), surfaced loudly, never silently served.
    Custody,
    /// A lower-level IO failure reading the object.
    Io(String),
}

impl std::fmt::Display for SpillReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing => write!(f, "state chunk object is not on disk"),
            Self::Custody => write!(
                f,
                "custody violation: spilled state chunk does not re-hash to its content address"
            ),
            Self::Io(e) => write!(f, "state chunk spill IO error: {e}"),
        }
    }
}

/// A synchronous content-addressed chunk spill rooted at a per-instance directory
/// (`<run state dir>/state/<role>-<instance>/<blake3-hex>`). Objects are named by their chunk
/// blake3, so writes are idempotent (same bytes ⇒ same address ⇒ same file) and nothing
/// attacker-influenced ever shapes a path.
pub struct SpillStore {
    root: PathBuf,
    /// A per-store counter for unique temp names (atomic tmp→rename), so a torn write is never
    /// observed at the content address even if the same object is re-written.
    tmp_seq: AtomicU64,
    /// The ambient disk custodian + charge scope (Phase 6): every spilled chunk reserves its
    /// bytes before it lands, and an evicted chunk discharges them. `None` = uncustodied.
    custody: Option<(std::sync::Arc<daemon_vhc_custody::DiskCustodian>, String)>,
}

impl SpillStore {
    /// Open (creating if missing) a spill rooted at `root`.
    ///
    /// # Errors
    /// [`std::io::Error`] if the root cannot be created.
    pub fn open(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        let custody = daemon_vhc_custody::ambient_for(&root);
        Ok(Self {
            root,
            tmp_seq: AtomicU64::new(0),
            custody,
        })
    }

    fn object_path(&self, hash: &[u8; 32]) -> PathBuf {
        self.root.join(Hash(*hash).to_hex())
    }

    /// Write one chunk content-addressed (atomic tmp→rename). Idempotent: an object already
    /// present at the address is left untouched (same bytes by construction).
    ///
    /// # Errors
    /// [`std::io::Error`] on a write/rename failure.
    pub fn write(&self, hash: &[u8; 32], bytes: &[u8]) -> std::io::Result<()> {
        let path = self.object_path(hash);
        if path.exists() {
            return Ok(());
        }
        // Reserve before the bytes land (Phase 6): a spill that does not fit refuses typed
        // (`HostStorageExhausted` at the state-store seam), never a raw mid-write ENOSPC.
        let reservation = match &self.custody {
            None => None,
            Some((custodian, scope)) => Some(
                custodian
                    .reserve(
                        scope,
                        bytes.len() as u64,
                        daemon_vhc_custody::WriteClass::Normal,
                    )
                    .map_err(|refusal| refusal.to_io())?,
            ),
        };
        let seq = self.tmp_seq.fetch_add(1, Ordering::Relaxed);
        let tmp = self
            .root
            .join(format!("{}.tmp-{seq}", Hash(*hash).to_hex()));
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, &path)?;
        if let Some(r) = reservation {
            r.commit();
        }
        Ok(())
    }

    /// Read one chunk and re-hash it against `hash` (the custody cross-check). Returns the
    /// verified bytes; a missing object is [`SpillReadError::Missing`] and a mismatch is
    /// [`SpillReadError::Custody`] — the store never serves unverified bytes.
    ///
    /// # Errors
    /// [`SpillReadError`] on miss / custody violation / IO failure.
    pub fn read_verify(&self, hash: &[u8; 32]) -> Result<Vec<u8>, SpillReadError> {
        let path = self.object_path(hash);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(SpillReadError::Missing)
            }
            Err(e) => return Err(SpillReadError::Io(e.to_string())),
        };
        if blake3_hash(&bytes) != Hash(*hash) {
            return Err(SpillReadError::Custody);
        }
        Ok(bytes)
    }

    /// Remove one chunk object (retention eviction / torn-fold GC). A missing object is not an
    /// error — eviction is idempotent. A custodied removal discharges the reclaimed bytes.
    pub fn remove(&self, hash: &[u8; 32]) {
        let path = self.object_path(hash);
        let reclaimed = std::fs::symlink_metadata(&path).map(|m| m.len()).ok();
        if std::fs::remove_file(&path).is_ok() {
            if let (Some(bytes), Some((custodian, scope))) = (reclaimed, &self.custody) {
                custodian.discharge(scope, bytes);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_root(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "daemon-vhc-state-spill-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        p
    }

    #[test]
    fn write_read_round_trips_by_content_address() {
        let root = tmp_root("roundtrip");
        let store = SpillStore::open(&root).unwrap();
        let bytes = b"a-canonical-state-chunk".to_vec();
        let hash = blake3_hash(&bytes).0;
        store.write(&hash, &bytes).unwrap();
        assert_eq!(store.read_verify(&hash).unwrap(), bytes);
        // Idempotent re-write.
        store.write(&hash, &bytes).unwrap();
        assert_eq!(store.read_verify(&hash).unwrap(), bytes);
        store.remove(&hash);
        assert!(matches!(
            store.read_verify(&hash).unwrap_err(),
            SpillReadError::Missing
        ));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Phase 6: a spill rooted under the exported custody root reserves every chunk against the
    /// per-run quota (Normal class) and refuses TYPED at exhaustion; an eviction discharges the
    /// bytes, so capacity actually returns. Other tests' spills live outside the custody root
    /// and stay uncustodied.
    #[test]
    fn custodied_spill_refuses_at_quota_and_discharges_on_eviction() {
        let custody_root = tmp_root("custody-root");
        std::fs::create_dir_all(&custody_root).unwrap();
        std::env::set_var(daemon_vhc_custody::CUSTODY_ROOT_ENV, &custody_root);
        std::env::set_var(daemon_vhc_custody::DISK_RUN_QUOTA_MB_ENV, "1");
        std::env::set_var(daemon_vhc_custody::DISK_RESERVE_MB_ENV, "0");
        std::env::set_var(daemon_vhc_custody::DISK_EMERGENCY_MB_ENV, "0");

        let store = SpillStore::open(custody_root.join("run-scope/trainer-1/state")).unwrap();
        let chunk = vec![0x5Au8; 700 * 1024];
        let h1 = blake3_hash(&chunk).0;
        store.write(&h1, &chunk).unwrap();
        // A second 700 KiB chunk overflows the 1 MiB scope quota: typed refusal.
        let chunk2 = vec![0xA5u8; 700 * 1024];
        let h2 = blake3_hash(&chunk2).0;
        let err = store.write(&h2, &chunk2).unwrap_err();
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::QuotaExceeded,
            "typed, not raw ENOSPC"
        );
        // Eviction discharges the ledger; the refused chunk now fits.
        store.remove(&h1);
        store.write(&h2, &chunk2).unwrap();

        std::env::remove_var(daemon_vhc_custody::CUSTODY_ROOT_ENV);
        std::env::remove_var(daemon_vhc_custody::DISK_RUN_QUOTA_MB_ENV);
        std::env::remove_var(daemon_vhc_custody::DISK_RESERVE_MB_ENV);
        std::env::remove_var(daemon_vhc_custody::DISK_EMERGENCY_MB_ENV);
        let _ = std::fs::remove_dir_all(&custody_root);
    }

    #[test]
    fn tampered_object_fails_custody_loud() {
        let root = tmp_root("custody");
        let store = SpillStore::open(&root).unwrap();
        let bytes = b"honest-chunk".to_vec();
        let hash = blake3_hash(&bytes).0;
        store.write(&hash, &bytes).unwrap();
        // Overwrite the object at its address with different bytes (tamper/corruption). The
        // re-hash on read catches it — the SAME loud fault a resident custody violation raises.
        std::fs::write(root.join(Hash(hash).to_hex()), b"tampered").unwrap();
        assert!(matches!(
            store.read_verify(&hash).unwrap_err(),
            SpillReadError::Custody
        ));
        let _ = std::fs::remove_dir_all(&root);
    }
}
