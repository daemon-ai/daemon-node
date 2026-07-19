// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Sealed record-archive publication (architecture §7.4; the acceptance digest/replay oracle's
//! product artifact) — the coordinator's durable journal segments published to the payload plane.
//!
//! A coordinator's journal (ABI §8) is a chain of segments; a SEALED segment is immutable and
//! content-addressed by its complete-file blake3 (the §8.2 chain link). Publishing the archive
//! uploads each sealed segment's bytes to the content-addressed payload plane and returns the
//! ordered addresses + the head (the last sealed segment) — the anchor a late joiner or the
//! offline replay oracle fetches to re-derive the run.
//!
//! Schema-free by construction: this reads the segment SUBSTRATE (`daemon-vhc-journal`) and moves
//! opaque bytes to a [`ContentStore`] — it never decodes a round message (the oracle that
//! interprets the archive is harness tooling, not this production path).

use daemon_vhc_journal::{scan_file, JournalPaths};
use daemon_vhc_net::{ContentHash, ContentStore};

/// The published archive: the sealed segments' content addresses in chain order, and the head
/// (the last sealed segment) — the pointer an archive reader starts from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PublishedArchive {
    /// The sealed segments' content addresses, ascending by ordinal (chain order).
    pub segments: Vec<ContentHash>,
    /// The head address (the last sealed segment), or `None` when nothing is sealed yet.
    pub head: Option<ContentHash>,
}

/// An archive-publication failure.
#[derive(Debug, thiserror::Error)]
pub enum ArchiveError {
    /// A journal substrate error (scan / read).
    #[error("journal: {0}")]
    Journal(String),
    /// A content-store put failure.
    #[error("content store: {0}")]
    Store(String),
    /// A published segment's store address did not match its complete-file blake3 (the store is
    /// broken — the addresses must agree by construction).
    #[error("archive address mismatch: segment complete-file {expected} != store {actual}")]
    AddressMismatch {
        /// The segment's complete-file blake3 (hex).
        expected: String,
        /// The address the store returned (hex).
        actual: String,
    },
}

/// Publish every SEALED segment of the journal at `journal_dir` to `store`, returning their
/// content addresses in chain order + the head. The unsealed (active) tail is NOT published: only
/// immutable segments are archive material. Idempotent — re-publishing puts identical bytes at
/// identical addresses.
///
/// # Errors
/// A journal read/scan failure, a store put failure, or an address disagreement.
pub async fn publish_journal_archive(
    journal_dir: &std::path::Path,
    store: &dyn ContentStore,
) -> Result<PublishedArchive, ArchiveError> {
    let paths =
        JournalPaths::open(journal_dir).map_err(|e| ArchiveError::Journal(e.to_string()))?;
    let ordinals = paths
        .existing_segments()
        .map_err(|e| ArchiveError::Journal(e.to_string()))?;

    let mut segments = Vec::new();
    for ord in ordinals {
        let path = paths.segment(ord);
        let scan = scan_file(&path).map_err(|e| ArchiveError::Journal(e.to_string()))?;
        // Only immutable (sealed) segments are archive material; a torn/active tail is skipped.
        if !scan.sealed {
            continue;
        }
        let bytes = read_segment(&path).map_err(|e| ArchiveError::Journal(e.to_string()))?;
        let address = store
            .put_content(&bytes)
            .await
            .map_err(|e| ArchiveError::Store(e.to_string()))?;
        let expected = ContentHash(scan.complete_file_blake3);
        if address != expected {
            return Err(ArchiveError::AddressMismatch {
                expected: expected.to_hex(),
                actual: address.to_hex(),
            });
        }
        segments.push(address);
    }
    let head = segments.last().copied();
    Ok(PublishedArchive { segments, head })
}

// The journal root is a host-owned, node-chosen directory (never attacker-influenced); the
// segment file read mirrors the journal substrate's own sanctioned raw-fs discipline.
#[allow(clippy::disallowed_methods)]
fn read_segment(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    std::fs::read(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_host::run::{JournalSink, RunIdentity};
    use daemon_vhc_net::MemoryContentStore;

    use crate::journal_home::{journal_dir, DurableSink};

    fn identity() -> RunIdentity {
        RunIdentity {
            run_id: [0x11; 32],
            epoch: 0,
            role: "coordinator".into(),
            instance: 1,
            module: [0x22; 32],
        }
    }

    #[tokio::test]
    async fn sealed_segments_publish_content_addressed_and_refetch() {
        let dir = tempfile::tempdir().unwrap();
        let jdir = journal_dir(dir.path(), "archive-run", "coordinator", 1);
        // Write enough records to seal at least one segment (RotatePolicy default rolls at a
        // record threshold), then leave the journal.
        {
            let mut sink = DurableSink::open(&jdir, &identity(), [0x5C; 32]).unwrap();
            sink.run_header(
                2 << 16,
                &[("vhc".into(), 2)],
                false,
                b"m",
                b"c",
                b"g",
                b"cl",
                b"ch",
                b"d",
            )
            .unwrap();
            for i in 0..2048u64 {
                sink.event(i, b"opaque-record").unwrap();
            }
            sink.terminal(0, Some(0), None).unwrap();
        }

        let store = MemoryContentStore::new();
        let archive = publish_journal_archive(&jdir, &store).await.unwrap();
        assert!(
            !archive.segments.is_empty(),
            "at least one segment sealed + published"
        );
        assert_eq!(archive.head, archive.segments.last().copied());
        // Every published address re-fetches (content-addressed, verified).
        for addr in &archive.segments {
            let bytes = store.get_content(addr).await.expect("archived segment");
            assert_eq!(
                daemon_vhc_proto::blake3_hash(&bytes),
                *addr,
                "the archived segment is addressed by its own bytes"
            );
        }
        // Idempotent re-publish yields the same addresses.
        let again = publish_journal_archive(&jdir, &store).await.unwrap();
        assert_eq!(again.segments, archive.segments);
    }
}
