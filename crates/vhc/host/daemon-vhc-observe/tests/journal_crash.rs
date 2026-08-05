// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Crash safety of the segmented journal (ABI companion §8.2/§8.4; conformance §13
//! "crash-recovery test"): torn-write / kill-mid-append at arbitrary byte offsets, reopen, and
//! verify — the chain is intact up to the last durable barrier, recovery lands on a clean point, no
//! silent corruption survives past a CRC/chain break, and the durable seq counter is never reused.

// Crash tests deliberately truncate/corrupt raw segment files under a throwaway temp dir; the
// Phase-4 fs guardrail targets production paths, so a test-scoped allow is sanctioned.
#![allow(clippy::disallowed_methods)]

use std::sync::atomic::{AtomicU64, Ordering};

use daemon_vhc_observe::journal::record::{Body, ClockRec, EventRec, ExecIdentity, Record};
use daemon_vhc_observe::journal::segment::{
    scan_bytes, SegmentHeader, SegmentWriter, GENESIS_PREV,
};
use daemon_vhc_observe::journal::sidecar::StaticKey;
use daemon_vhc_observe::journal::store::{Journal, RotatePolicy};
use daemon_vhc_proto::Hash;

fn tempdir() -> std::path::PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let mut base = std::env::temp_dir();
    base.push(format!(
        "dvhc-journal-crash-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn ident() -> ExecIdentity {
    ExecIdentity {
        run_id: Hash([1; 32]),
        epoch: 1,
        role: "trainer".into(),
        instance: 7,
        module: Hash([2; 32]),
    }
}

fn event(ord: u64, at: u64) -> Record {
    Record::new(
        ord,
        Body::Event(EventRec {
            at,
            frame: format!("frame-{ord}").into_bytes(),
        }),
    )
}

/// A segment written with N committed records; truncating at EVERY byte offset must never yield a
/// partial/garbage record — recovery always lands on a clean frame boundary (§8.2).
#[test]
fn truncation_at_every_offset_recovers_to_a_clean_boundary() {
    let dir = tempdir();
    let path = dir.join("segment-00000000.dvhcjrn");
    let header = SegmentHeader {
        id: ident(),
        segment: 0,
        prev_blake3: GENESIS_PREV,
    };
    let mut w = SegmentWriter::create(&path, &header).unwrap();
    let records: Vec<Record> = (0..8).map(|i| event(i, i * 10)).collect();
    for r in &records {
        w.append(r).unwrap();
    }
    w.commit().unwrap();
    let full = std::fs::read(&path).unwrap();

    // The full header length (magic + version + prev + len + body + CRC): a prefix shorter than this
    // cannot even yield a clean recovery point.
    let (_, header_len) = SegmentHeader::decode(&full).unwrap();

    // Compute the byte offset of every clean frame boundary (header end + cumulative frame sizes).
    let clean = scan_bytes(&full).unwrap();
    let mut boundaries = vec![];
    {
        // Reconstruct boundaries by re-scanning prefixes: a prefix scans without truncation exactly
        // when it ends on a boundary.
        for len in 0..=full.len() {
            if let Ok(s) = scan_bytes(&full[..len]) {
                if !s.truncated {
                    boundaries.push(len);
                }
            }
        }
    }
    assert_eq!(
        clean.records.len(),
        8,
        "the full segment holds all 8 records"
    );

    for len in 0..full.len() {
        let prefix = &full[..len];
        match scan_bytes(prefix) {
            Ok(scan) => {
                // Every recovered record is one of the originals, in order, with no partials.
                assert_eq!(scan.records.as_slice(), &records[..scan.records.len()]);
                // durable_len is a real frame boundary <= len.
                assert!(scan.durable_len as usize <= len);
                assert!(boundaries.contains(&(scan.durable_len as usize)));
                // truncated <=> we cut inside/after the last recoverable frame.
                if (scan.durable_len as usize) < len {
                    assert!(scan.truncated);
                }
            }
            Err(_) => {
                // Only prefixes too short to even hold the complete header are unrecoverable.
                assert!(len < header_len, "len {len} < header_len {header_len}");
            }
        }
    }
}

/// A single-byte corruption inside a record body is caught by CRC32C: that frame and everything after
/// it is discarded; nothing past the CRC break is silently accepted (§8.2).
#[test]
fn crc_break_discards_the_tail_without_silent_corruption() {
    let dir = tempdir();
    let path = dir.join("segment-00000000.dvhcjrn");
    let header = SegmentHeader {
        id: ident(),
        segment: 0,
        prev_blake3: GENESIS_PREV,
    };
    let mut w = SegmentWriter::create(&path, &header).unwrap();
    for i in 0..5 {
        w.append(&event(i, i)).unwrap();
    }
    w.commit().unwrap();
    let mut bytes = std::fs::read(&path).unwrap();

    // Corrupt a byte well into the file (inside the 3rd/4th record body region).
    let mid = bytes.len() * 3 / 4;
    bytes[mid] ^= 0xff;

    let scan = scan_bytes(&bytes).unwrap();
    assert!(scan.truncated, "a CRC break truncates the tail");
    assert!(
        scan.records.len() < 5,
        "records after the break are discarded ({} of 5 kept)",
        scan.records.len()
    );
    // Whatever survived is a correct prefix — no garbage record decoded past the break.
    for (i, r) in scan.records.iter().enumerate() {
        assert_eq!(r.ord, i as u64);
    }
}

/// Kill-mid-append then reopen the whole `Journal`: the durable seq counter resumes strictly above
/// the last committed publish (never reused, §8.4 rule 2 / §12.2), the next ordinal is correct, and
/// appends continue cleanly on the truncated journal.
#[test]
fn reopen_after_torn_append_never_reuses_seq_and_continues() {
    let dir = tempdir();
    let root = dir.join("journal");

    // Write a journal: 3 committed publishes on channel 0 (seq 0,1,2) then some uncommitted events.
    {
        let mut j = Journal::create(
            &root,
            ident(),
            StaticKey::new([9u8; 32]),
            RotatePolicy::default(),
        )
        .unwrap();
        for i in 0..3u64 {
            let (_, seq) = j
                .publish(
                    0,
                    format!("payload-{i}").as_bytes(),
                    format!("frame-{i}").into_bytes(),
                )
                .unwrap();
            assert_eq!(seq, i, "seq is monotone from 0");
        }
        // Written-but-uncommitted observations (safe to lose together with the tail — §8.4 rule 3).
        j.append(Body::Clock(ClockRec { now: 100 })).unwrap();
        j.append(Body::Event(EventRec {
            at: 101,
            frame: b"tail".to_vec(),
        }))
        .unwrap();
        // Simulate a crash *before* the trailing commit by dropping the writer without commit()
        // and then tearing the file below.
    }

    // Simulate a torn write: truncate the segment file by a few bytes (mid-frame tail).
    let seg = root.join("segment-00000000.dvhcjrn");
    let full = std::fs::read(&seg).unwrap();
    let torn_len = full.len() - 3;
    std::fs::write(&seg, &full[..torn_len]).unwrap();

    // Reopen: recovery truncates the torn tail and reconciles the seq counter.
    let mut j = Journal::open(
        &root,
        ident(),
        StaticKey::new([9u8; 32]),
        RotatePolicy::default(),
    )
    .unwrap();
    // The 3 committed publishes survive → next seq is strictly above the highest committed (3).
    assert_eq!(
        j.next_seq(0),
        3,
        "seq counter resumes above the last committed publish"
    );

    // The reopened journal can continue appending; a new publish gets seq 3 (never reuses 0..=2).
    let (_, seq) = j.publish(0, b"resumed", b"resumed-frame".to_vec()).unwrap();
    assert_eq!(seq, 3);

    // Everything reads back with an intact chain and no partial records.
    let records = j.read_all_records().unwrap();
    let publishes: Vec<u64> = records
        .iter()
        .filter_map(|r| match &r.body {
            Body::Publish(p) => Some(p.seq),
            _ => None,
        })
        .collect();
    assert_eq!(
        publishes,
        vec![0, 1, 2, 3],
        "seqs are contiguous + never reused"
    );
}

/// A torn tail spanning a segment boundary: recovery re-opens the last (unsealed) segment, discards
/// its torn tail, and the multi-segment chain stays verifiable.
#[test]
fn multi_segment_chain_recovers_after_tear_on_last_segment() {
    let dir = tempdir();
    let root = dir.join("journal");
    // Force rotation every 2 records so we get several segments.
    let rotate = RotatePolicy {
        max_records: 2,
        ..RotatePolicy::default()
    };

    {
        let mut j = Journal::create(&root, ident(), StaticKey::new([3u8; 32]), rotate).unwrap();
        for i in 0..7u64 {
            j.append(Body::Event(EventRec {
                at: i,
                frame: format!("e{i}").into_bytes(),
            }))
            .unwrap();
        }
        j.commit().unwrap();
    }

    // Tear the last segment file.
    let segs = daemon_vhc_observe::journal::JournalPaths::open(&root)
        .unwrap()
        .existing_segments()
        .unwrap();
    assert!(segs.len() >= 3, "rotation produced multiple segments");
    let last = *segs.last().unwrap();
    let last_path = root.join(format!("segment-{last:08}.dvhcjrn"));
    let full = std::fs::read(&last_path).unwrap();
    if full.len() > 2 {
        std::fs::write(&last_path, &full[..full.len() - 2]).unwrap();
    }

    // Reopen: the chain across all segments verifies; the torn tail on the last is discarded.
    let j = Journal::open(&root, ident(), StaticKey::new([3u8; 32]), rotate).unwrap();
    let records = j.read_all_records().unwrap();
    // Events recovered are a contiguous prefix by ordinal (no gaps, no partials).
    for (i, r) in records.iter().enumerate() {
        assert_eq!(r.ord, i as u64);
    }
    assert!(!records.is_empty());
}

/// The seal hook (the incremental archive-publication seam) reports, for EVERY roll, exactly the
/// claim an archive head publishes: the sealed segment's identity, ordinal, path, content hash,
/// chain link, and record count — and the reported chain links thread (each `prev_blake3` is the
/// previous report's `segment_blake3`, from the genesis link).
#[test]
fn the_seal_hook_reports_the_publishable_chain_claim_on_every_roll() {
    use std::sync::{Arc, Mutex};

    use daemon_vhc_observe::journal::store::SealedSegment;

    let dir = tempdir();
    let root = dir.join("journal");
    let rotate = RotatePolicy {
        max_records: 2,
        ..RotatePolicy::default()
    };
    let seals: Arc<Mutex<Vec<SealedSegment>>> = Arc::new(Mutex::new(Vec::new()));

    let mut j = Journal::create(&root, ident(), StaticKey::new([3u8; 32]), rotate).unwrap();
    let sink = seals.clone();
    j.set_seal_hook(Box::new(move |s| sink.lock().unwrap().push(s.clone())));
    for i in 0..5u64 {
        j.append(Body::Event(EventRec {
            at: i,
            frame: format!("e{i}").into_bytes(),
        }))
        .unwrap();
    }
    j.commit().unwrap();

    let seals = seals.lock().unwrap();
    assert_eq!(seals.len(), 2, "5 records at max 2/segment roll twice");
    let mut prev = GENESIS_PREV;
    for (i, s) in seals.iter().enumerate() {
        assert_eq!(s.segment, i as u64);
        assert_eq!(s.id, ident(), "the sealed segment's own header identity");
        assert_eq!(s.records, 2);
        assert_eq!(s.prev_blake3, prev, "the chain link threads");
        // The reported content hash IS the sealed file's blake3 (the content address a
        // publisher uploads under).
        let bytes = std::fs::read(&s.path).unwrap();
        assert_eq!(
            Hash(s.segment_blake3),
            daemon_vhc_proto::blake3_hash(&bytes),
            "segment_blake3 is the complete sealed file's hash"
        );
        prev = s.segment_blake3;
    }
}

/// The [`RotatePolicy::max_open`] age bound (the archive recovery-point cadence): a non-empty
/// segment older than the bound rolls on the next append; an EMPTY segment never rolls on time
/// alone (no content-free chain churn).
#[test]
fn the_age_bound_rolls_a_stale_nonempty_segment_but_never_an_empty_one() {
    use std::sync::atomic::AtomicU64 as Counter;
    use std::sync::Arc;

    let dir = tempdir();
    let root = dir.join("journal");
    let rotate = RotatePolicy {
        max_records: 1_000,
        max_open: Some(std::time::Duration::from_millis(30)),
    };
    let rolls = Arc::new(Counter::new(0));

    let mut j = Journal::create(&root, ident(), StaticKey::new([3u8; 32]), rotate).unwrap();
    let n = rolls.clone();
    j.set_seal_hook(Box::new(move |_| {
        n.fetch_add(1, Ordering::Relaxed);
    }));

    // An empty segment aged past the bound: the next append must land in it, not roll it.
    std::thread::sleep(std::time::Duration::from_millis(40));
    j.append(Body::Clock(daemon_vhc_observe::journal::record::ClockRec {
        now: 1,
    }))
    .unwrap();
    assert_eq!(
        rolls.load(Ordering::Relaxed),
        0,
        "an empty segment never rolls on age"
    );

    // The segment is now non-empty; once aged, the NEXT append seals it first.
    std::thread::sleep(std::time::Duration::from_millis(40));
    j.append(Body::Clock(daemon_vhc_observe::journal::record::ClockRec {
        now: 2,
    }))
    .unwrap();
    assert_eq!(
        rolls.load(Ordering::Relaxed),
        1,
        "a stale non-empty segment seals before the next append"
    );
    assert_eq!(j.current_segment(), 1);
}

/// The founding identity is segment 0's header identity, surviving both a reopen and a
/// live-upgrade seam ([`Journal::roll_to_identity`]) — the archive chain scope is the SERIES',
/// not the current span's.
#[test]
fn the_founding_identity_survives_reopen_and_the_upgrade_seam() {
    let dir = tempdir();
    let root = dir.join("journal");

    {
        let mut j = Journal::create(
            &root,
            ident(),
            StaticKey::new([3u8; 32]),
            RotatePolicy::default(),
        )
        .unwrap();
        j.append(Body::Clock(daemon_vhc_observe::journal::record::ClockRec {
            now: 1,
        }))
        .unwrap();
        j.commit().unwrap();
    }

    // The seam: continue the series under a successor identity.
    let successor = ExecIdentity {
        instance: 8,
        epoch: 2,
        ..ident()
    };
    {
        let mut j = Journal::open_continuation(
            &root,
            successor.clone(),
            StaticKey::new([3u8; 32]),
            RotatePolicy::default(),
            None,
        )
        .unwrap();
        assert_eq!(*j.id(), successor, "the live identity is the successor's");
        assert_eq!(
            *j.founding_id(),
            ident(),
            "the founding identity stays the series founder's across the seam"
        );
        j.append(Body::Clock(daemon_vhc_observe::journal::record::ClockRec {
            now: 2,
        }))
        .unwrap();
        j.commit().unwrap();
    }

    // And across a plain reopen after the seam.
    let j = Journal::open(
        &root,
        successor.clone(),
        StaticKey::new([3u8; 32]),
        RotatePolicy::default(),
    )
    .unwrap();
    assert_eq!(*j.founding_id(), ident());
}
