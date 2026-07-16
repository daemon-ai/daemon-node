// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The crash-safe segmented journal substrate (ABI companion §8).
//!
//! This is the on-disk substrate that makes policy-determinism operational (architecture §3.6,
//! ABI §8): a **crash-safe, segmented, append-only** journal recording everything a guest could have
//! branched on, so re-feeding it reproduces every decision bit-for-bit. It generalizes the in-memory
//! [`MessageLog`](crate::MessageLog) / [`RunCapture`](crate::RunCapture) capture onto durable storage
//! (refactor §5 A1). The wire/record format is the ABI companion's ([`record`]); the framing,
//! durability, and chaining are §8.2/§8.4; large values go to encrypted content-addressed sidecars
//! ([`sidecar`], §8.5).
//!
//! Module map (how §8 maps to code):
//! * [`record`] — the §8.3 tagged-union record types + canonical CBOR codec, validated against
//!   [`daemon_vhc_abi::JOURNAL_CDDL`].
//! * [`segment`] — the §8.2 segment layout: header framing, the BLAKE3 hash chain across segments,
//!   per-record CRC32C, the self-excluding `seal` record, and crash-recovery scanning.
//! * [`sidecar`] — the §8.5 XChaCha20-Poly1305 encrypted, content-addressed sidecar store with a
//!   pluggable [`sidecar::KeyProvider`] (the node-local key is a construction input, never invented
//!   here).
//! * [`Journal`] — ties segments + sidecars together with the §8.4 commit barriers, segment rotation,
//!   and crash recovery (reconciling the durable seq counter + spool against the journal).
//! * [`oracle`] — the coordinator-oracle migration: the existing replay oracle over journal-backed
//!   capture (its public behavior pinned by the existing tests).
//! * [`verifier`] — the worker input-replay verifier **skeleton** (types + a sim-fed harness shape);
//!   its real host-runtime wiring lands after A0 merges and completes in A2 (refactor §5 A1).

// Sanctioned raw-fs home (Phase 4 hardening / clippy.toml): the journal root is a **host-owned,
// node-chosen** directory (the journal lives in the worker subprocess, decisions D1; the path is
// never attacker/guest-influenced), and the §8.4 durability model needs low-level `fdatasync`
// commit barriers + `set_len` tail truncation that `ContainedRoot`'s safe-path-resolution API does
// not model. Per clippy.toml, "outside module X" is expressed only via a declared `#[allow]` anchor
// at the sanctioned home — this is that anchor. No process spawn / env mutation happens here, so the
// shared-lint coupling residual is inert.
#![allow(clippy::disallowed_methods)]

pub mod archive;
pub mod consensus;
pub mod oracle;
pub mod record;
pub mod segment;
pub mod sidecar;
pub mod store;
pub mod verifier;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub use archive::{
    detect_fork, ArchiveError, ChainHead, ForkEvidence, RecordArchive, ReplicationPolicy,
    RetentionPolicy, SignedHead,
};
pub use consensus::{
    extract_consensus_capture, recover_chain_from_archive, replay_consensus_from_archive,
    ConsensusCapture, ConsensusReplayError, ConsensusReplayReport, RecoveredChain,
};
pub use record::{Body, ExecIdentity, Record};
pub use segment::{scan_bytes, scan_file, ScanResult, SegmentHeader, SegmentWriter};
pub use sidecar::{KeyProvider, SidecarError, SidecarStore, StaticKey};
pub use store::{Journal, RotatePolicy};

use daemon_vhc_abi::JOURNAL_FORMAT_VERSION;

/// Errors surfaced by the journal substrate.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum JournalError {
    /// An I/O error (open/append/fdatasync/rotate).
    #[error("journal io error: {0}")]
    Io(#[from] io::Error),
    /// A canonical-CBOR (de)serialization step failed.
    #[error("journal codec error: {0}")]
    Codec(String),
    /// A record carried a tag not in the §8.3 grammar (0..=17; 18..=63 reserved).
    #[error("unknown journal record tag: {0}")]
    UnknownTag(u8),
    /// A segment file's magic / format / header CRC did not validate.
    #[error("corrupt segment header: {0}")]
    BadHeader(String),
    /// The BLAKE3 segment chain did not verify (a segment's `prev` or a seal hash mismatched).
    #[error("segment chain broken: {0}")]
    ChainBroken(String),
    /// A sidecar failed AEAD verification or its plaintext content-address check (§8.5).
    #[error("sidecar error: {0}")]
    Sidecar(String),
}

/// Where a journal's segment files (and sidecars) live on disk.
///
/// Segments are named `segment-<NNNNNNNN>.dvhcjrn` (zero-padded ordinal, sorted lexicographically ==
/// numerically); sidecars live under a `sidecars/` subdirectory named by plaintext blake3 (§8.5).
#[derive(Clone, Debug)]
pub struct JournalPaths {
    root: PathBuf,
}

impl JournalPaths {
    /// A journal rooted at `root` (created if absent).
    ///
    /// # Errors
    /// [`JournalError::Io`] if the directory tree cannot be created.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, JournalError> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        fs::create_dir_all(root.join("sidecars"))?;
        Ok(Self { root })
    }

    /// The journal root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The path of segment ordinal `n`.
    #[must_use]
    pub fn segment(&self, n: u64) -> PathBuf {
        self.root.join(format!("segment-{n:08}.dvhcjrn"))
    }

    /// The sidecars subdirectory.
    #[must_use]
    pub fn sidecars(&self) -> PathBuf {
        self.root.join("sidecars")
    }

    /// The segment ordinals present on disk, ascending.
    ///
    /// # Errors
    /// [`JournalError::Io`] if the directory cannot be read.
    pub fn existing_segments(&self) -> Result<Vec<u64>, JournalError> {
        let mut ords = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix("segment-") {
                if let Some(num) = rest.strip_suffix(".dvhcjrn") {
                    if let Ok(n) = num.parse::<u64>() {
                        ords.push(n);
                    }
                }
            }
        }
        ords.sort_unstable();
        Ok(ords)
    }
}

/// fsync a directory so a freshly-created segment's directory entry is durable (§8.4: "on segment
/// creation, also the directory entry").
pub(crate) fn fsync_dir(dir: &Path) -> io::Result<()> {
    // Opening a directory read-only and calling `sync_all` fsyncs the directory entry on Unix.
    let f = fs::File::open(dir)?;
    f.sync_all()
}

/// The journal format version this build writes (ABI §8.2 / §8.3 tag-0 `format`).
#[must_use]
pub fn format_version() -> u32 {
    JOURNAL_FORMAT_VERSION
}
