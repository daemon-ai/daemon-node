// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The crash-safe segmented journal substrate (ABI companion §8).
//!
//! This is the on-disk substrate that makes policy-determinism operational (architecture §3.6,
//! ABI §8): a **crash-safe, segmented, append-only** journal recording everything a guest could have
//! branched on, so re-feeding it reproduces every decision bit-for-bit. The wire/record format is
//! the ABI companion's ([`record`]); the framing, durability, and chaining are §8.2/§8.4; large
//! values go to encrypted content-addressed sidecars ([`sidecar`], §8.5).
//!
//! Extracted from `daemon-vhc-observe` so production crates (the role session's durable sink, the
//! worker binary) can link the store without the observe crate's oracle tooling: observe decodes
//! SDK round schemas for its replay oracles, which no production host graph may link (ABI §12.5
//! [OWN-3]). This crate is schema-free by construction — records, segments, sidecars, and the
//! commit-barrier store move bytes and hashes only. `daemon-vhc-observe` re-exports everything
//! here under its original `journal::` paths, so the archive/oracle/verifier tooling and their
//! suites are unchanged consumers.
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

// Sanctioned raw-fs home (Phase 4 hardening / clippy.toml): the journal root is a **host-owned,
// node-chosen** directory (the journal lives in the worker subprocess, decisions D1; the path is
// never attacker/guest-influenced), and the §8.4 durability model needs low-level `fdatasync`
// commit barriers + `set_len` tail truncation that `ContainedRoot`'s safe-path-resolution API does
// not model. Per clippy.toml, "outside module X" is expressed only via a declared `#[allow]` anchor
// at the sanctioned home — this is that anchor. No process spawn / env mutation happens here, so the
// shared-lint coupling residual is inert.
#![allow(clippy::disallowed_methods)]
#![forbid(unsafe_code)]

pub mod record;
pub mod segment;
pub mod sidecar;
pub mod store;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

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
///
/// A directory is not an ordinary file, and the two platforms disagree about it, so this barrier is
/// per-platform rather than one idiom that happens to compile everywhere:
///
/// * **Unix** — the POSIX idiom: open the directory and `sync_all` it. A failure here is a real
///   durability failure and propagates.
/// * **Windows** — `File::open` on a directory (`GENERIC_READ`, no flags) is REFUSED with
///   `ERROR_ACCESS_DENIED` (os error 5); a directory handle requires `FILE_FLAG_BACKUP_SEMANTICS`,
///   and `FlushFileBuffers` on that handle additionally requires WRITE access (a read-only or
///   zero-access directory handle also returns error 5 — measured on the fleet's Windows box). So
///   the Windows arm asks for exactly that handle, and treats a refusal as SUCCESS rather than a
///   journal failure: Windows guarantees nothing about flushing a directory handle, while NTFS
///   commits a file's creation with the file's own metadata — which every caller here has just
///   flushed (`File::sync_all` is `FlushFileBuffers` on the segment). The barrier is taken wherever
///   the platform offers it, and can never turn a healthy journal into a failed run.
///
/// Running the Unix idiom on Windows is what made every fresh journal there terminal: the first
/// segment's header wrote, this call was denied, and each worker generation died `FailedRetryable`
/// on a freshly created directory (`journal io error: Access is denied. (os error 5)`) — a run that
/// never reached a single round.
#[cfg(unix)]
pub(crate) fn fsync_dir(dir: &Path) -> io::Result<()> {
    let f = fs::File::open(dir)?;
    f.sync_all()
}

#[cfg(windows)]
pub(crate) fn fsync_dir(dir: &Path) -> io::Result<()> {
    use std::os::windows::fs::OpenOptionsExt as _;

    /// `FILE_FLAG_BACKUP_SEMANTICS` — the flag that makes `CreateFileW` hand back a handle to a
    /// directory at all.
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;

    if let Ok(handle) = fs::OpenOptions::new()
        .write(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(dir)
    {
        let _ = handle.sync_all();
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn fsync_dir(_dir: &Path) -> io::Result<()> {
    Ok(())
}

/// The journal format version this build writes (ABI §8.2 / §8.3 tag-0 `format`).
#[must_use]
pub fn format_version() -> u32 {
    JOURNAL_FORMAT_VERSION
}

/// The §8.4 directory barrier's PLATFORM discipline. The substrate's behavioral suites live with
/// `daemon-vhc-observe` (which re-exports this crate); these stay here because they assert
/// `pub(crate)` platform semantics — and because this crate cross-compiles to Windows on its own,
/// so the Windows arm can be run on a real Windows box as a `--target x86_64-pc-windows-gnu` test
/// binary instead of being taken on faith.
#[cfg(test)]
mod tests {
    use super::*;

    /// A unique scratch directory under the platform temp dir (no dev-dependencies in this crate).
    fn scratch(tag: &str) -> PathBuf {
        let unique = format!(
            "daemon-vhc-journal-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let dir = std::env::temp_dir().join(unique);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// The segment-creation path end to end: create → append → commit → re-open, each of which
    /// takes the directory barrier. Portable by design — on Windows this is the regression test for
    /// the barrier that refused a freshly created journal directory with `Access is denied`
    /// (os error 5) and terminated every worker generation before its first round.
    #[test]
    fn segment_create_and_reopen_take_the_directory_barrier() {
        let dir = scratch("segment-barrier");
        let paths = JournalPaths::open(&dir).expect("journal paths");
        let id = record::ExecIdentity {
            run_id: daemon_vhc_proto::Hash([1u8; 32]),
            epoch: 0,
            role: "trainer".into(),
            instance: 1,
            module: daemon_vhc_proto::Hash([2u8; 32]),
        };
        let header = segment::SegmentHeader {
            id,
            segment: 0,
            prev_blake3: segment::GENESIS_PREV,
        };
        let path = paths.segment(0);
        let mut writer = segment::SegmentWriter::create(&path, &header)
            .expect("create segment 0 (§8.4 barrier)");
        writer.commit().expect("commit barrier");
        let bytes = fs::read(&path).expect("read segment");
        drop(writer);
        segment::SegmentWriter::reopen(&path, &bytes, 0).expect("re-open segment 0 (§8.4 barrier)");
        fsync_dir(paths.root()).expect("directory barrier on the journal root");
        fsync_dir(&paths.sidecars()).expect("directory barrier on the sidecar dir");
        let _ = fs::remove_dir_all(&dir);
    }

    /// Windows only: pin the platform semantics the barrier is split for. The Unix idiom
    /// (`File::open` on a directory) is denied — os error 5, the `ERROR_ACCESS_DENIED` the fleet box
    /// reported — while the barrier itself succeeds on the same directory. If a future edit puts the
    /// Unix idiom back on this platform, this fails instead of the fleet.
    #[cfg(windows)]
    #[test]
    fn windows_directory_handles_need_backup_semantics() {
        let dir = scratch("windows-dir-handle");
        let denied = fs::File::open(&dir).expect_err("a plain directory open must be refused");
        assert_eq!(
            denied.raw_os_error(),
            Some(5),
            "expected ERROR_ACCESS_DENIED from a directory open without FILE_FLAG_BACKUP_SEMANTICS, \
             got {denied:?}"
        );
        fsync_dir(&dir).expect("the §8.4 barrier must succeed where the Unix idiom cannot");
        let _ = fs::remove_dir_all(&dir);
    }
}
