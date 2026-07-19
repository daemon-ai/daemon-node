// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The [`Journal`] — the crash-safe segmented append-only store (ABI companion §8.2/§8.4/§8.5).
//!
//! [`Journal`] ties the pieces together: it appends records to the current segment ([`segment`]),
//! rotates + seals segments at a host-configured threshold (§8.2), enforces the §8.4 commit barriers
//! (fdatasync), routes oversize read-back values to encrypted content-addressed sidecars ([`sidecar`],
//! §8.5), and on [`Journal::open`] performs crash recovery: it verifies the BLAKE3 segment chain,
//! truncates a torn tail to the last durable frame boundary (no silent corruption past a CRC/chain
//! break), reconstructs the next record ordinal, and reconciles the durable channel-scoped sequence
//! counters against the committed tag-4 publish records (§8.4 rule 2, §12.2).

// Sanctioned raw-fs home (see journal/mod.rs): host-owned journal root + low-level durability. No
// spawn / env mutation here.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;

use daemon_vhc_abi::READBACK_INLINE_MAX;
use daemon_vhc_proto::blake3_hash;

use super::record::{Body, ExecIdentity, PublishRec, ReadBackRec, Record, SidecarRef};
use super::segment::{scan_file, ScanResult, SegmentHeader, SegmentWriter, GENESIS_PREV};
use super::sidecar::{KeyProvider, SidecarStore};
use super::{JournalError, JournalPaths};

/// Host-configurable segment rotation policy (§8.2: segments roll at a size/record threshold).
#[derive(Clone, Copy, Debug)]
pub struct RotatePolicy {
    /// Roll (seal + start a new segment) once a segment reaches this many records.
    pub max_records: u64,
}

impl Default for RotatePolicy {
    fn default() -> Self {
        // A conservative default; hosts tune it. Kept small enough that tests exercise rotation.
        Self { max_records: 1024 }
    }
}

/// A crash-safe, segmented, append-only journal for one run-instance (§8).
pub struct Journal<K: KeyProvider> {
    paths: JournalPaths,
    id: ExecIdentity,
    writer: SegmentWriter,
    current_segment: u64,
    next_ord: u64,
    rotate: RotatePolicy,
    sidecars: SidecarStore<K>,
    /// Highest committed publish `seq` per channel — the durable counter recovered per §8.4 rule 2.
    seq_high: BTreeMap<u64, u64>,
    /// The current instantiation counter (nonce input for sidecars, §8.5). Advanced by the host on
    /// each (re-)instantiation; A1's substrate defaults it to 0 and lets the caller set it.
    instantiation_counter: u64,
}

impl<K: KeyProvider + Clone> Journal<K> {
    /// Create a fresh journal at `root` (first segment, ordinal 0) for execution identity `id`.
    ///
    /// # Errors
    /// [`JournalError`] on any filesystem / encode failure.
    pub fn create(
        root: impl AsRef<std::path::Path>,
        id: ExecIdentity,
        key: K,
        rotate: RotatePolicy,
    ) -> Result<Self, JournalError> {
        let paths = JournalPaths::open(root)?;
        if !paths.existing_segments()?.is_empty() {
            return Err(JournalError::BadHeader(
                "create called on a non-empty journal directory; use open".into(),
            ));
        }
        let header = SegmentHeader {
            id: id.clone(),
            segment: 0,
            prev_blake3: GENESIS_PREV,
        };
        let writer = SegmentWriter::create(paths.segment(0), &header)?;
        let sidecars = SidecarStore::open(paths.sidecars(), id.clone(), key)?;
        Ok(Self {
            paths,
            id,
            writer,
            current_segment: 0,
            next_ord: 0,
            rotate,
            sidecars,
            seq_high: BTreeMap::new(),
            instantiation_counter: 0,
        })
    }

    /// Open an existing journal, performing crash recovery (§8.2/§8.4): verify the segment chain,
    /// truncate a torn tail, recover the next ordinal + the durable seq counters, and re-open the last
    /// (unsealed) segment for appends — or start a fresh next segment if the last one was sealed.
    ///
    /// # Errors
    /// [`JournalError`] on a broken chain, unreadable header, or filesystem failure.
    pub fn open(
        root: impl AsRef<std::path::Path>,
        id: ExecIdentity,
        key: K,
        rotate: RotatePolicy,
    ) -> Result<Self, JournalError> {
        let paths = JournalPaths::open(root)?;
        let ords = paths.existing_segments()?;
        if ords.is_empty() {
            return Self::create(paths.root(), id, key, rotate);
        }

        let recovery = Self::recover(&paths, &ords)?;
        let sidecars = SidecarStore::open(paths.sidecars(), id.clone(), key)?;

        let (writer, current_segment) = if recovery.last_sealed {
            // Last segment is immutable; start the next one, chaining off its complete-file hash.
            let next = recovery.last_ordinal + 1;
            let header = SegmentHeader {
                id: id.clone(),
                segment: next,
                prev_blake3: recovery.last_file_blake3,
            };
            let writer = SegmentWriter::create(paths.segment(next), &header)?;
            (writer, next)
        } else {
            // Re-open the last (unsealed) segment for continued appends; truncate its torn tail.
            let intact = &recovery.last_intact_bytes;
            let writer = SegmentWriter::reopen(
                paths.segment(recovery.last_ordinal),
                intact,
                recovery.last_records,
            )?;
            (writer, recovery.last_ordinal)
        };

        Ok(Self {
            paths,
            id,
            writer,
            current_segment,
            next_ord: recovery.next_ord,
            rotate,
            sidecars,
            seq_high: recovery.seq_high,
            instantiation_counter: 0,
        })
    }

    /// Walk every segment in order, verifying the chain + reconciling recovery state (§8.2/§8.4).
    fn recover(paths: &JournalPaths, ords: &[u64]) -> Result<Recovery, JournalError> {
        let mut expected_prev = GENESIS_PREV;
        let mut next_ord = 0u64;
        let mut seq_high: BTreeMap<u64, u64> = BTreeMap::new();
        let mut last_ordinal = 0u64;
        let mut last_sealed = false;
        let mut last_file_blake3 = GENESIS_PREV;
        let mut last_records = 0u64;
        let mut last_intact_bytes = Vec::new();

        for (idx, &ord) in ords.iter().enumerate() {
            let scan: ScanResult = scan_file(paths.segment(ord))?;
            if scan.header.segment != ord {
                return Err(JournalError::ChainBroken(format!(
                    "segment file {ord} carries header ordinal {}",
                    scan.header.segment
                )));
            }
            if scan.header.prev_blake3 != expected_prev {
                return Err(JournalError::ChainBroken(format!(
                    "segment {ord} prev_blake3 does not match the previous segment's hash"
                )));
            }
            let is_last = idx + 1 == ords.len();
            // A non-last segment MUST be sealed (a mid-chain unsealed segment means the chain forked).
            if !is_last && !scan.sealed {
                return Err(JournalError::ChainBroken(format!(
                    "non-terminal segment {ord} is unsealed"
                )));
            }
            for record in &scan.records {
                next_ord = next_ord.max(record.ord + 1);
                // Re-key at every run-header (§8.1): a run-header opens a new execution-identity
                // span (a live upgrade's seam), and the §12.2 signed stream is scoped to that
                // identity — the durable per-channel counters recover from the CURRENT span's
                // publishes only, never a retired incarnation's.
                if matches!(record.body, Body::RunHeader(_)) {
                    seq_high.clear();
                }
                if let Body::Publish(p) = &record.body {
                    let e = seq_high.entry(p.channel).or_insert(0);
                    *e = (*e).max(p.seq);
                }
            }
            expected_prev = scan.complete_file_blake3;
            if is_last {
                last_ordinal = ord;
                last_sealed = scan.sealed;
                last_file_blake3 = scan.complete_file_blake3;
                last_records = scan.records.len() as u64;
                // Re-read the intact prefix bytes so a reopen can seed the hasher + truncate the tail.
                let mut bytes = std::fs::read(paths.segment(ord))?;
                bytes.truncate(scan.durable_len as usize);
                last_intact_bytes = bytes;
            }
        }

        Ok(Recovery {
            next_ord,
            seq_high,
            last_ordinal,
            last_sealed,
            last_file_blake3,
            last_records,
            last_intact_bytes,
        })
    }

    /// The execution identity keying this journal.
    #[must_use]
    pub fn id(&self) -> &ExecIdentity {
        &self.id
    }

    /// The next record ordinal that [`Journal::append`] will assign.
    #[must_use]
    pub fn next_ord(&self) -> u64 {
        self.next_ord
    }

    /// The current (open) segment ordinal.
    #[must_use]
    pub fn current_segment(&self) -> u64 {
        self.current_segment
    }

    /// Set the current instantiation counter (§7.1 / §8.5 nonce input). The host advances this on
    /// each (re-)instantiation and journals it as a tag-13 record; A1's substrate just tracks it.
    pub fn set_instantiation_counter(&mut self, counter: u64) {
        self.instantiation_counter = counter;
    }

    /// The durable, channel-scoped sequence counter: the next seq that channel may publish, strictly
    /// above the highest committed tag-4 `seq` (§8.4 rule 2, §12.2; never reused across crashes).
    #[must_use]
    pub fn next_seq(&self, channel: u64) -> u64 {
        self.seq_high.get(&channel).map_or(0, |h| h + 1)
    }

    /// Append a record (write, not necessarily commit — §8.4 rule 3/4). Rotates the segment first if
    /// the rotation threshold is reached. Returns the assigned ordinal.
    ///
    /// # Errors
    /// [`JournalError`] on rotate/encode/write failure.
    pub fn append(&mut self, body: Body) -> Result<u64, JournalError> {
        self.maybe_rotate()?;
        let ord = self.next_ord;
        self.writer.append(&Record::new(ord, body))?;
        self.next_ord += 1;
        Ok(ord)
    }

    /// Append a record and immediately cross a commit barrier (§8.4 rule 2: publish / terminal /
    /// snapshot must be committed before they are observable). Returns the assigned ordinal.
    ///
    /// # Errors
    /// [`JournalError`] on rotate/encode/write/sync failure.
    pub fn append_committed(&mut self, body: Body) -> Result<u64, JournalError> {
        let ord = self.append(body)?;
        self.commit()?;
        Ok(ord)
    }

    /// The §8.4 commit barrier: fdatasync the current segment so every record written before it is
    /// durable.
    ///
    /// # Errors
    /// [`JournalError::Io`] on sync failure.
    pub fn commit(&mut self) -> Result<(), JournalError> {
        self.writer.commit()
    }

    /// Journal a `publish` (§8.4 rule 2): advance the durable channel seq counter, write the tag-4
    /// record, and cross a commit barrier before returning — the atomic batch a publish requires.
    /// Returns `(ord, seq)`.
    ///
    /// # Errors
    /// [`JournalError`] on encode/write/sync failure.
    pub fn publish(
        &mut self,
        channel: u64,
        payload: &[u8],
        frame: Vec<u8>,
    ) -> Result<(u64, u64), JournalError> {
        let seq = self.next_seq(channel);
        let hash = blake3_hash(payload);
        let ord = self.append_committed(Body::Publish(PublishRec {
            channel,
            seq,
            hash,
            frame,
        }))?;
        self.seq_high.insert(channel, seq);
        Ok((ord, seq))
    }

    /// Journal a `read_back` value (§8.5): inline iff plaintext `<= READBACK_INLINE_MAX`, else an
    /// encrypted content-addressed sidecar keyed by the record ordinal. Returns the assigned ordinal.
    ///
    /// # Errors
    /// [`JournalError`] on encode/write/encrypt failure.
    pub fn read_back(
        &mut self,
        src: u64,
        kind: u64,
        status: u64,
        value: &[u8],
    ) -> Result<u64, JournalError> {
        self.maybe_rotate()?;
        let ord = self.next_ord;
        let seg = self.current_segment;
        let body = if value.len() <= READBACK_INLINE_MAX {
            Body::ReadBack(ReadBackRec {
                src,
                kind,
                status,
                value: Some(value.to_vec()),
                sidecar: None,
            })
        } else {
            let sref: SidecarRef =
                self.sidecars
                    .put(ord, self.instantiation_counter, seg, value)?;
            Body::ReadBack(ReadBackRec {
                src,
                kind,
                status,
                value: None,
                sidecar: Some(sref),
            })
        };
        self.writer.append(&Record::new(ord, body))?;
        self.next_ord += 1;
        Ok(ord)
    }

    /// Fetch a sidecar plaintext by reference (§8.5), for replay/audit. `ord` is the referencing
    /// record's ordinal (the nonce input).
    ///
    /// # Errors
    /// [`super::sidecar::SidecarError`] on missing/verify/io failure.
    pub fn fetch_sidecar(
        &self,
        sref: &SidecarRef,
        ord: u64,
    ) -> Result<Vec<u8>, super::sidecar::SidecarError> {
        self.sidecars.get(sref, ord, self.instantiation_counter)
    }

    /// Seal the current segment cleanly + start the next one (§8.2). The seal is a §8.4 commit barrier.
    ///
    /// # Errors
    /// [`JournalError`] on seal/create failure.
    pub fn roll(&mut self) -> Result<(), JournalError> {
        let file_hash = self.writer.seal()?;
        let next = self.current_segment + 1;
        let header = SegmentHeader {
            id: self.id.clone(),
            segment: next,
            prev_blake3: file_hash,
        };
        self.writer = SegmentWriter::create(self.paths.segment(next), &header)?;
        self.current_segment = next;
        Ok(())
    }

    /// The live-upgrade seam (§8.1/§10.3): CONTINUE this journal — one file series — under a new
    /// execution identity. Seals the current segment and opens the next one with the incoming
    /// incarnation's identity in its header (the seam forces a segment roll, so every segment
    /// header still matches the identity of the records it contains); the per-journal record
    /// ordinal stays globally monotone across the seam; the per-channel publish counters reset
    /// (the new `(run, epoch, role, instance, channel)` stream opens at seq 0 — the never-reused
    /// stream scope of §12.2, disjoint from the retired incarnation's by construction).
    ///
    /// The caller writes the incoming incarnation's own tag-0 run-header as the new span's first
    /// record (the driver does this at instantiation).
    ///
    /// # Errors
    /// [`JournalError`] on seal/create failure.
    pub fn roll_to_identity(&mut self, id: ExecIdentity) -> Result<(), JournalError> {
        self.id = id.clone();
        self.sidecars.set_identity(id);
        self.roll()?;
        self.seq_high.clear();
        Ok(())
    }

    /// Open a journal at the live-upgrade seam (§8.1): crash-recover the retired incarnation's
    /// records (they remain as the prefix), then [`Journal::roll_to_identity`] so appends land in
    /// a fresh segment carrying the incoming identity. Refuses on an empty directory — a seam
    /// continues an existing log; a fresh incarnation uses [`Journal::open`].
    ///
    /// # Errors
    /// [`JournalError`] on a broken chain, an empty directory, or filesystem failure.
    pub fn open_continuation(
        root: impl AsRef<std::path::Path>,
        id: ExecIdentity,
        key: K,
        rotate: RotatePolicy,
    ) -> Result<Self, JournalError> {
        let paths = JournalPaths::open(&root)?;
        if paths.existing_segments()?.is_empty() {
            return Err(JournalError::BadHeader(
                "continuation open on an empty journal directory (a seam continues an existing \
                 log; use open for a fresh incarnation)"
                    .into(),
            ));
        }
        let mut journal = Self::open(root, id.clone(), key, rotate)?;
        journal.roll_to_identity(id)?;
        Ok(journal)
    }

    fn maybe_rotate(&mut self) -> Result<(), JournalError> {
        if self.writer.records() >= self.rotate.max_records {
            self.roll()?;
        }
        Ok(())
    }

    /// The journal's on-disk paths.
    #[must_use]
    pub fn paths(&self) -> &JournalPaths {
        &self.paths
    }

    /// Read back every record across every segment in order (for replay / audit). Verifies the chain
    /// as it goes (§8.2). A torn tail on the final segment is silently discarded (already truncated).
    ///
    /// # Errors
    /// [`JournalError`] on a broken chain or read failure.
    pub fn read_all_records(&self) -> Result<Vec<Record>, JournalError> {
        let ords = self.paths.existing_segments()?;
        let mut out = Vec::new();
        let mut expected_prev = GENESIS_PREV;
        for &ord in &ords {
            let scan = scan_file(self.paths.segment(ord))?;
            if scan.header.prev_blake3 != expected_prev {
                return Err(JournalError::ChainBroken(format!(
                    "segment {ord} prev_blake3 mismatch during read_all"
                )));
            }
            for record in scan.records {
                // The seal record is framing metadata, not a journal observation.
                if !matches!(record.body, Body::Seal(_)) {
                    out.push(record);
                }
            }
            expected_prev = scan.complete_file_blake3;
        }
        Ok(out)
    }
}

/// Crash-recovery state reconstructed from the segment chain (§8.2/§8.4).
struct Recovery {
    next_ord: u64,
    seq_high: BTreeMap<u64, u64>,
    last_ordinal: u64,
    last_sealed: bool,
    last_file_blake3: [u8; 32],
    last_records: u64,
    last_intact_bytes: Vec<u8>,
}
