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
use std::sync::Arc;

use daemon_vhc_abi::READBACK_INLINE_MAX;
use daemon_vhc_custody::{DiskCustodian, Reservation, WriteClass};
use daemon_vhc_proto::blake3_hash;

use super::record::{Body, ExecIdentity, PublishRec, ReadBackRec, Record, SidecarRef};
use super::segment::{scan_file, ScanResult, SegmentHeader, SegmentWriter, GENESIS_PREV};
use super::sidecar::{KeyProvider, SidecarStore};
use super::{JournalError, JournalPaths};

/// Host-configurable segment rotation policy (§8.2: segments roll at a size/record threshold,
/// and optionally at an age bound — the archive **recovery-point cadence**).
#[derive(Clone, Debug)]
pub struct RotatePolicy {
    /// Roll (seal + start a new segment) once a segment reaches this many records.
    pub max_records: u64,
    /// Roll a non-empty segment once it has been open this long (checked on append — a quiet
    /// journal has nothing new to protect, so it never rolls on time alone). `None` disables
    /// the age bound. This is what keeps the remote reconstruction point from going stale
    /// behind a large record count: the sealed-segment publisher only ever sees sealed
    /// segments, so the open segment's age bounds the recovery-point lag.
    pub max_open: Option<std::time::Duration>,
    /// An EXTERNAL recovery-point request cell (Gate B' round-aware seal pacing): whenever the
    /// cell's value exceeds the count of requests this journal has already honored, the next
    /// append seals the (non-empty) open segment first, exactly like the age bound. The journal
    /// stays vocabulary-free — the owner encodes its own pacing (the session bumps the cell as
    /// the committed-round watermark drifts past the archive tip); this substrate sees only "a
    /// recovery point was requested". `None` disables the seam.
    pub roll_request: Option<std::sync::Arc<std::sync::atomic::AtomicU64>>,
}

impl Default for RotatePolicy {
    fn default() -> Self {
        // A conservative default; hosts tune it. Kept small enough that tests exercise rotation.
        Self {
            max_records: 1024,
            max_open: None,
            roll_request: None,
        }
    }
}

/// One sealed segment, as reported to the seal hook the instant [`Journal::roll`] seals it —
/// everything an incremental archive publisher needs to publish exactly this segment and its
/// attested head, with no directory scan.
#[derive(Clone, Debug)]
pub struct SealedSegment {
    /// The execution identity in the sealed segment's header (the identity span that wrote it —
    /// NOT necessarily the journal's current identity at a live-upgrade seam).
    pub id: ExecIdentity,
    /// The sealed segment's 0-based ordinal in the series.
    pub segment: u64,
    /// The sealed segment file's path.
    pub path: std::path::PathBuf,
    /// The complete-file BLAKE3 — the segment's content address (§8.2 chain link).
    pub segment_blake3: [u8; 32],
    /// The previous segment's complete-file BLAKE3 ([`GENESIS_PREV`] at segment 0).
    pub prev_blake3: [u8; 32],
    /// The number of records the segment carries (excluding the seal frame).
    pub records: u64,
}

/// The seal hook: invoked synchronously inside [`Journal::roll`] immediately after the seal
/// barrier lands. Implementations must be cheap and non-blocking (hand off to a channel).
pub type SealHook = Box<dyn FnMut(&SealedSegment) + Send>;

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
    /// The identity written into the CURRENT segment's header (lags [`Journal::id`] between a
    /// `roll_to_identity` identity change and the roll itself — the sealed segment reports the
    /// identity that actually wrote it).
    current_header_id: ExecIdentity,
    /// The current segment's `prev_blake3` chain link (what the seal hook reports).
    current_prev: [u8; 32],
    /// When the current segment was opened (the [`RotatePolicy::max_open`] age clock).
    opened_at: std::time::Instant,
    /// How many external recovery-point requests ([`RotatePolicy::roll_request`]) this journal
    /// has already honored — the next append rolls first whenever the cell exceeds this.
    roll_honored: u64,
    /// The identity in SEGMENT 0's header — the series' **founding identity** (the archive chain
    /// scope; constant across live-upgrade seams).
    founding_id: ExecIdentity,
    /// The incremental-publication seam: invoked on every seal (see [`SealHook`]).
    on_seal: Option<SealHook>,
    /// The ambient disk custodian + charge scope (Phase 6), attached at open when the journal
    /// root lives under [`daemon_vhc_custody::CUSTODY_ROOT_ENV`]. `None` = uncustodied (tests,
    /// non-VHC embedders) — every reservation is a no-op.
    custody: Option<(Arc<DiskCustodian>, String)>,
}

/// The current value of the external recovery-point request cell (0 when the seam is unwired).
fn requests_outstanding(rotate: &RotatePolicy) -> u64 {
    rotate
        .roll_request
        .as_ref()
        .map_or(0, |cell| cell.load(std::sync::atomic::Ordering::Relaxed))
}

/// The reserved margin for segment bookkeeping writes whose exact size is not worth
/// pre-encoding (a segment header at create, the tag-17 seal frame at roll) — both are
/// [`WriteClass::Critical`]: sealing the active journal must always succeed during pressure
/// handling, and a roll that cannot open its successor wedges the recovery stream.
const SEGMENT_OVERHEAD_BYTES: u64 = 4096;

/// The sidecar envelope margin above the plaintext (encryption tag + header + the
/// content-address filename's directory entry) reserved per §8.5 sidecar write.
const SIDECAR_OVERHEAD_BYTES: u64 = 128;

/// Reserve `bytes` against the ambient custodian (no-op when uncustodied). A refusal surfaces
/// as [`JournalError::Io`] with the typed exhaustion kind ([`daemon_vhc_custody::CustodyRefusal::to_io`]),
/// so the durable-sink seam classifies it `HostStorageExhausted` — never `BadModule`.
fn reserve_with(
    custody: &Option<(Arc<DiskCustodian>, String)>,
    bytes: u64,
    class: WriteClass,
) -> Result<Option<Reservation>, JournalError> {
    match custody {
        None => Ok(None),
        Some((custodian, scope)) => custodian
            .reserve(scope, bytes, class)
            .map(Some)
            .map_err(|refusal| JournalError::Io(refusal.to_io())),
    }
}

/// The write class of one record body: the recovery-critical records — the terminal pair
/// (tag 11) and the checkpoint/upgrade snapshot anchor (tag 10) — may draw into the emergency
/// margin and are quota-exempt (refusing them trades a bounded overrun for a forked run);
/// everything else is a [`WriteClass::Normal`] durable write.
fn class_of(body: &Body) -> WriteClass {
    match body {
        Body::Terminal(_) | Body::Snapshot(_) => WriteClass::Critical,
        _ => WriteClass::Normal,
    }
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
        let custody = daemon_vhc_custody::ambient_for(paths.root());
        let header = SegmentHeader {
            id: id.clone(),
            segment: 0,
            prev_blake3: GENESIS_PREV,
        };
        let reservation = reserve_with(&custody, SEGMENT_OVERHEAD_BYTES, WriteClass::Critical)?;
        let writer = SegmentWriter::create(paths.segment(0), &header)?;
        if let Some(r) = reservation {
            r.commit();
        }
        let sidecars = SidecarStore::open(paths.sidecars(), id.clone(), key)?;
        // Requests raised BEFORE this journal existed are already honored by construction:
        // a fresh (empty) chain has no un-sealed history to protect.
        let roll_honored = requests_outstanding(&rotate);
        Ok(Self {
            paths,
            id: id.clone(),
            writer,
            current_segment: 0,
            next_ord: 0,
            rotate,
            sidecars,
            seq_high: BTreeMap::new(),
            instantiation_counter: 0,
            current_header_id: id.clone(),
            current_prev: GENESIS_PREV,
            opened_at: std::time::Instant::now(),
            roll_honored,
            founding_id: id,
            on_seal: None,
            custody,
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
            // A pruned chain that lost ALL its retained segments is damage, not a fresh start:
            // re-creating segment 0 here would fork the anchored (archived) history.
            if crate::ChainAnchor::load(&paths)?.is_some() {
                return Err(JournalError::ChainBroken(
                    "an anchored (pruned) chain has no retained segments; refusing to re-create \
                     from genesis"
                        .into(),
                ));
            }
            return Self::create(paths.root(), id, key, rotate);
        }

        let recovery = Self::recover(&paths, &ords)?;
        let sidecars = SidecarStore::open(paths.sidecars(), id.clone(), key)?;
        let custody = daemon_vhc_custody::ambient_for(paths.root());

        let (writer, current_segment, current_header_id, current_prev) = if recovery.last_sealed {
            // Last segment is immutable; start the next one, chaining off its complete-file hash.
            let next = recovery.last_ordinal + 1;
            let header = SegmentHeader {
                id: id.clone(),
                segment: next,
                prev_blake3: recovery.last_file_blake3,
            };
            let reservation = reserve_with(&custody, SEGMENT_OVERHEAD_BYTES, WriteClass::Critical)?;
            let writer = SegmentWriter::create(paths.segment(next), &header)?;
            if let Some(r) = reservation {
                r.commit();
            }
            (writer, next, id.clone(), recovery.last_file_blake3)
        } else {
            // Re-open the last (unsealed) segment for continued appends; truncate its torn tail.
            let intact = &recovery.last_intact_bytes;
            let writer = SegmentWriter::reopen(
                paths.segment(recovery.last_ordinal),
                intact,
                recovery.last_records,
            )?;
            (
                writer,
                recovery.last_ordinal,
                recovery.last_header_id,
                recovery.last_prev_blake3,
            )
        };

        // Requests raised against a PREVIOUS journal instance don't carry over: the pacing owner
        // re-derives its lag from live state and re-raises if the gap still stands.
        let roll_honored = requests_outstanding(&rotate);
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
            current_header_id,
            current_prev,
            opened_at: std::time::Instant::now(),
            roll_honored,
            founding_id: recovery.founding_id,
            on_seal: None,
            custody,
        })
    }

    /// Walk every segment in order, verifying the chain + reconciling recovery state (§8.2/§8.4).
    ///
    /// A pruned chain (archive-then-prune reclaimed the archived prefix) anchors at its
    /// [`ChainAnchor`](crate::ChainAnchor) instead of genesis: leftover files below the anchor are
    /// skipped (prune debris — proven archived before the anchor advanced past them), the first
    /// retained segment's `prev_blake3` verifies against the anchor's recorded predecessor hash,
    /// and a MISSING anchored first segment refuses (that is damage, not pruning).
    fn recover(paths: &JournalPaths, ords: &[u64]) -> Result<Recovery, JournalError> {
        let anchor = crate::ChainAnchor::load(paths)?;
        let (ords, mut expected_prev) = match &anchor {
            Some(a) => {
                let retained: Vec<u64> =
                    ords.iter().copied().filter(|o| *o >= a.first_ord).collect();
                match retained.first() {
                    Some(first) if *first == a.first_ord => (retained, a.prev_blake3),
                    Some(first) => {
                        return Err(JournalError::ChainBroken(format!(
                            "the chain anchor names segment {} as the first retained segment, \
                             but the earliest present is {first}",
                            a.first_ord
                        )));
                    }
                    None => {
                        return Err(JournalError::ChainBroken(format!(
                            "the chain anchor names segment {} as the first retained segment, \
                             but no segment at or above it exists",
                            a.first_ord
                        )));
                    }
                }
            }
            None => (ords.to_vec(), GENESIS_PREV),
        };
        let ords = &ords[..];
        let mut next_ord = 0u64;
        let mut seq_high: BTreeMap<u64, u64> = BTreeMap::new();
        let mut last_ordinal = 0u64;
        let mut last_sealed = false;
        let mut last_file_blake3 = GENESIS_PREV;
        let mut last_records = 0u64;
        let mut last_intact_bytes = Vec::new();
        let mut founding_id = None;
        let mut last_header_id = None;
        let mut last_prev_blake3 = GENESIS_PREV;

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
            if idx == 0 {
                founding_id = Some(scan.header.id.clone());
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
            if is_last {
                last_ordinal = ord;
                last_sealed = scan.sealed;
                last_file_blake3 = scan.complete_file_blake3;
                last_records = scan.records.len() as u64;
                last_header_id = Some(scan.header.id.clone());
                last_prev_blake3 = expected_prev;
                // Re-read the intact prefix bytes so a reopen can seed the hasher + truncate the tail.
                let mut bytes = std::fs::read(paths.segment(ord))?;
                bytes.truncate(scan.durable_len as usize);
                last_intact_bytes = bytes;
            }
            expected_prev = scan.complete_file_blake3;
        }

        // `ords` is non-empty on every recover() path, so both identities were captured.
        let founding_id = founding_id
            .ok_or_else(|| JournalError::ChainBroken("recover on an empty segment list".into()))?;
        let last_header_id = last_header_id.unwrap_or_else(|| founding_id.clone());

        Ok(Recovery {
            next_ord,
            seq_high,
            last_ordinal,
            last_sealed,
            last_file_blake3,
            last_records,
            last_intact_bytes,
            founding_id,
            last_header_id,
            last_prev_blake3,
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
    /// Every append reserves its EXACT framed byte count with the ambient disk custodian before
    /// the bytes touch disk (Phase 6): a write either fits or refuses typed — never a raw
    /// `ENOSPC` discovered mid-write. Recovery-critical records ([`class_of`]) may draw into the
    /// emergency margin.
    ///
    /// # Errors
    /// [`JournalError`] on rotate/encode/write failure, or a typed capacity refusal.
    pub fn append(&mut self, body: Body) -> Result<u64, JournalError> {
        self.maybe_rotate()?;
        let ord = self.next_ord;
        let class = class_of(&body);
        let framed = SegmentWriter::encode(&Record::new(ord, body))?;
        let reservation = reserve_with(&self.custody, framed.len() as u64, class)?;
        self.writer.append_framed(&framed)?;
        if let Some(r) = reservation {
            r.commit();
        }
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
            // The sidecar is the BULK write of the pair — reserve the plaintext plus the
            // encryption envelope margin before it lands.
            let sidecar_reservation = reserve_with(
                &self.custody,
                value.len() as u64 + SIDECAR_OVERHEAD_BYTES,
                WriteClass::Normal,
            )?;
            let sref: SidecarRef =
                self.sidecars
                    .put(ord, self.instantiation_counter, seg, value)?;
            if let Some(r) = sidecar_reservation {
                r.commit();
            }
            Body::ReadBack(ReadBackRec {
                src,
                kind,
                status,
                value: None,
                sidecar: Some(sref),
            })
        };
        let framed = SegmentWriter::encode(&Record::new(ord, body))?;
        let reservation = reserve_with(&self.custody, framed.len() as u64, WriteClass::Normal)?;
        self.writer.append_framed(&framed)?;
        if let Some(r) = reservation {
            r.commit();
        }
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

    /// Seal the current segment cleanly + start the next one (§8.2). The seal is a §8.4 commit
    /// barrier. Fires the seal hook (when armed) the moment the seal lands — the incremental
    /// archive-publication seam.
    ///
    /// # Errors
    /// [`JournalError`] on seal/create failure.
    pub fn roll(&mut self) -> Result<(), JournalError> {
        // One Critical margin covers the pair of bookkeeping writes a roll performs (the seal
        // frame + the successor's header): sealing must always succeed during pressure handling.
        let reservation =
            reserve_with(&self.custody, SEGMENT_OVERHEAD_BYTES, WriteClass::Critical)?;
        let records = self.writer.records();
        let file_hash = self.writer.seal()?;
        if let Some(hook) = self.on_seal.as_mut() {
            hook(&SealedSegment {
                id: self.current_header_id.clone(),
                segment: self.current_segment,
                path: self.paths.segment(self.current_segment),
                segment_blake3: file_hash,
                prev_blake3: self.current_prev,
                records,
            });
        }
        let next = self.current_segment + 1;
        let header = SegmentHeader {
            id: self.id.clone(),
            segment: next,
            prev_blake3: file_hash,
        };
        self.writer = SegmentWriter::create(self.paths.segment(next), &header)?;
        if let Some(r) = reservation {
            r.commit();
        }
        self.current_segment = next;
        self.current_header_id = self.id.clone();
        self.current_prev = file_hash;
        self.opened_at = std::time::Instant::now();
        Ok(())
    }

    /// Arm the seal hook (see [`SealHook`]). At most one; re-arming replaces.
    pub fn set_seal_hook(&mut self, hook: SealHook) {
        self.on_seal = Some(hook);
    }

    /// The identity in segment 0's header — the series' founding identity (the archive chain
    /// scope: constant across live-upgrade seams, unlike [`Journal::id`]).
    #[must_use]
    pub fn founding_id(&self) -> &ExecIdentity {
        &self.founding_id
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
    /// An EMPTY current segment (e.g. the successor a terminal tail-seal opened) is re-headered
    /// in place under the incoming identity instead of sealed: a content-free segment sealed
    /// under the retired identity would demand an attested head from a key the successor does
    /// not hold, for records that do not exist. Same ordinal, same chain link — the series stays
    /// intact.
    ///
    /// # Errors
    /// [`JournalError`] on seal/create failure.
    pub fn roll_to_identity(&mut self, id: ExecIdentity) -> Result<(), JournalError> {
        self.id = id.clone();
        self.sidecars.set_identity(id.clone());
        if self.writer.records() == 0 {
            let header = SegmentHeader {
                id: id.clone(),
                segment: self.current_segment,
                prev_blake3: self.current_prev,
            };
            // Replace the content-free file: only a header (no records) exists at this ordinal,
            // and the recreate writes the same ordinal + chain link under the new identity.
            // Net-zero capacity (a header replaces a header) — reserve the margin anyway so the
            // recreate is governed like every other segment-file creation.
            let reservation =
                reserve_with(&self.custody, SEGMENT_OVERHEAD_BYTES, WriteClass::Critical)?;
            std::fs::remove_file(self.paths.segment(self.current_segment))?;
            self.writer = SegmentWriter::create(self.paths.segment(self.current_segment), &header)?;
            if let Some(r) = reservation {
                r.commit();
            }
            self.current_header_id = id;
            self.opened_at = std::time::Instant::now();
        } else {
            self.roll()?;
        }
        self.seq_high.clear();
        Ok(())
    }

    /// Open a journal at the live-upgrade seam (§8.1): crash-recover the retired incarnation's
    /// records (they remain as the prefix), then [`Journal::roll_to_identity`] so appends land in
    /// a fresh segment carrying the incoming identity. Refuses on an empty directory — a seam
    /// continues an existing log; a fresh incarnation uses [`Journal::open`].
    ///
    /// `on_seal` is armed BEFORE the seam roll: the retiring span's final segment seals right
    /// here, and its seal must reach the archive publisher like every other (the head chain is
    /// dense — a swallowed seam seal would wedge publication at the next ordinal).
    ///
    /// # Errors
    /// [`JournalError`] on a broken chain, an empty directory, or filesystem failure.
    pub fn open_continuation(
        root: impl AsRef<std::path::Path>,
        id: ExecIdentity,
        key: K,
        rotate: RotatePolicy,
        on_seal: Option<SealHook>,
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
        if let Some(hook) = on_seal {
            journal.set_seal_hook(hook);
        }
        journal.roll_to_identity(id)?;
        Ok(journal)
    }

    fn maybe_rotate(&mut self) -> Result<(), JournalError> {
        let records = self.writer.records();
        let over_count = records >= self.rotate.max_records;
        // The age bound only ever seals a NON-empty segment (an empty roll would churn the chain
        // with content-free segments), and only on append — a quiet journal never rolls on time
        // alone (it has nothing new to protect).
        let over_age = records > 0
            && self
                .rotate
                .max_open
                .is_some_and(|max| self.opened_at.elapsed() >= max);
        // An external recovery-point request ([`RotatePolicy::roll_request`]) seals on the same
        // non-empty/append-time discipline as the age bound; an empty segment marks the request
        // honored without churning the chain (there is nothing new to protect — the previous
        // seal IS the requested recovery point).
        let requested = requests_outstanding(&self.rotate);
        let over_request = requested > self.roll_honored;
        if over_request && records == 0 {
            self.roll_honored = requested;
        }
        if over_count || over_age || (over_request && records > 0) {
            self.roll_honored = requested.max(self.roll_honored);
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

/// Seal every ABANDONED unsealed segment with records in the journal at `dir` — the crash cut
/// of a chain whose writer died (§8.2). The intact prefix (scan-verified, torn suffix
/// truncated) is sealed IN PLACE, making the records archive material: a coordinator
/// reconstruction that consumes a crashed predecessor's tail must leave those records sealed,
/// or the successor chain's boot capture folds history the archive can never re-derive (the
/// c15k defect-16 shape: a 99-record unsealed tail consumed at recovery, skipped by every
/// later archive-only replay).
///
/// A record-free unsealed segment (a header-only active file) is left alone — it carries
/// nothing to protect and recovery reopens it for appends. Idempotent: a second pass finds
/// everything sealed and returns empty.
///
/// The caller must hold the series' single-writer discipline (the crashed writer is dead;
/// no live sink has the directory open).
///
/// # Errors
/// [`JournalError`] on an unlistable home or a seal that cannot be written (the caller must
/// then treat the tail as NOT consumable — replaying records that cannot become durable
/// recreates the divergence this exists to prevent).
pub fn seal_abandoned_tail(dir: &std::path::Path) -> Result<Vec<u64>, JournalError> {
    let paths = JournalPaths::open(dir)?;
    let mut sealed = Vec::new();
    for ord in paths.existing_segments()? {
        let path = paths.segment(ord);
        let scan = scan_file(&path)?;
        if scan.sealed || scan.records.is_empty() {
            continue;
        }
        let mut bytes = std::fs::read(&path).map_err(JournalError::Io)?;
        bytes.truncate(scan.durable_len as usize);
        let mut writer = SegmentWriter::reopen(&path, &bytes, scan.records.len() as u64)?;
        writer.seal()?;
        sealed.push(ord);
    }
    Ok(sealed)
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
    founding_id: ExecIdentity,
    last_header_id: ExecIdentity,
    last_prev_blake3: [u8; 32],
}
