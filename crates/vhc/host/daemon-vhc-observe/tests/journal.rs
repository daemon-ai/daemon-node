// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Journal substrate behavior (ABI companion §8.2/§8.3/§8.5, §13):
//! record round-trips + grammar conformance for every tag, segment framing + BLAKE3 chain + CRC32C,
//! clean-roll seal, encrypted content-addressed sidecars (round-trip + missing + tamper), and the
//! worker input-replay verifier skeleton shape (§8.7).

// Tests read/write raw fixture files under a throwaway temp dir (never an attacker path); the
// Phase-4 fs guardrail targets production paths, so a test-scoped allow is sanctioned.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;

use daemon_vhc_abi::{JOURNAL_CDDL, JOURNAL_RECORD_TAGS, READBACK_INLINE_MAX};
use daemon_vhc_proto::{blake3_hash, Hash};

use daemon_vhc_observe::journal::record::{
    Body, ClockRec, CompletionRec, ConditionRec, DeviceProfileRec, DropId, DropRec, EventRec,
    EvidenceRef, ExecIdentity, ExecutionGrantRec, InitRec, InstantiationRec, PublishRec,
    ReadBackRec, Record, RunHeader, SealRec, SidecarRef, SignedFrameRec, SnapshotRec, TerminalRec,
    ThrottleRec, TimerArmRec, TimerCancelRec, TrapInfo,
};
use daemon_vhc_observe::journal::segment::{
    scan_bytes, SegmentHeader, SegmentWriter, GENESIS_PREV,
};
use daemon_vhc_observe::journal::sidecar::{SidecarError, SidecarStore, StaticKey};
use daemon_vhc_observe::journal::verifier::{
    run_replay, ExpectedDecision, GuestUnderReplay, PayloadSource, ReplayOutcome, ReplayPlan,
    ReplayStep,
};

fn h(n: u8) -> Hash {
    Hash([n; 32])
}

fn ident() -> ExecIdentity {
    ExecIdentity {
        run_id: h(1),
        epoch: 4,
        role: "trainer".into(),
        instance: 42,
        module: h(2),
    }
}

/// One representative record per §8.3 tag (0..=17), in tag order.
fn all_bodies() -> Vec<Body> {
    let mut worlds = BTreeMap::new();
    worlds.insert("vhc".to_string(), 0u64);
    worlds.insert("net".to_string(), 0u64);
    vec![
        Body::RunHeader(Box::new(RunHeader {
            run_id: h(1),
            epoch: 4,
            role: "trainer".into(),
            instance: 42,
            module: h(2),
            abi: 2 << 16,
            worlds,
            bridge: true,
            manifest: b"manifest".to_vec(),
            config: b"config".to_vec(),
            grants: b"grants".to_vec(),
            claim: Some(b"claim".to_vec()),
            channels: b"channels".to_vec(),
            device: b"device".to_vec(),
            resource_plan: None,
            resource_plan_hash: None,
            physical_estimate: None,
            physical_estimate_hash: None,
            aggregate_estimate: None,
            aggregate_estimate_hash: None,
            execution_grant: None,
            execution_grant_hash: None,
            format: 1,
        })),
        Body::Event(EventRec {
            at: 12,
            frame: b"delivered-frame".to_vec(),
        }),
        Body::ReadBack(ReadBackRec {
            src: 0,
            kind: 1,
            status: 0,
            value: Some(b"inline".to_vec()),
            sidecar: None,
        }),
        Body::Clock(ClockRec { now: 123_456 }),
        Body::Publish(PublishRec {
            channel: 0,
            seq: 1,
            hash: h(9),
            frame: b"signed-wire-frame".to_vec(),
        }),
        Body::TimerArm(TimerArmRec {
            id: 1,
            delay: 1000,
            armed_at: 50,
        }),
        Body::TimerCancel(TimerCancelRec { id: 1, status: 0 }),
        Body::Drop(DropRec {
            class: 0,
            rule: 0,
            dropped: DropId {
                hash: Some(h(3)),
                seq: Some(7),
                ..Default::default()
            },
        }),
        Body::Throttle(ThrottleRec {
            paused: false,
            duty_pct: 80,
            vram_cap_bytes: 8_000_000_000,
        }),
        Body::Terminal(TerminalRec {
            kind: 1,
            outcome: None,
            trap: Some(TrapInfo {
                code: "BudgetEpoch".into(),
                import: "next_event".into(),
                context: "slice".into(),
                detail: "spun".into(),
            }),
        }),
        Body::Snapshot(SnapshotRec {
            manifest: b"state-manifest".to_vec(),
        }),
        Body::Init(InitRec {
            config_hash: h(4),
            grants_hash: h(5),
            status: 0,
        }),
        Body::SignedFrame(SignedFrameRec {
            channel: 0,
            seq: 1,
            sender: h(6),
            frame: None,
            evidence: Some(EvidenceRef {
                hash: h(7),
                locator: "archive://x".into(),
            }),
        }),
        Body::Instantiation(InstantiationRec {
            counter: 0,
            reason: 0,
            at: 7,
        }),
        Body::Completion(CompletionRec {
            op: 0,
            result: b"result".to_vec(),
        }),
        Body::DeviceProfile(DeviceProfileRec {
            profile: b"profile".to_vec(),
        }),
        Body::Condition(ConditionRec {
            code: "SpoolExhausted".into(),
            detail: "channel 0".into(),
        }),
        Body::Seal(SealRec {
            segment_blake3: h(8),
            records: 18,
        }),
        // tag 18 — the grant-application result a certification run records after the export returns.
        // It was added to the grammar without a sample here, so this set stopped covering the record
        // set it claims to cover and the assertion below caught it as soon as anyone ran the suite.
        Body::ExecutionGrant(ExecutionGrantRec {
            execution_grant_hash: h(18),
            status: 0,
        }),
    ]
}

#[test]
fn every_record_tag_round_trips_and_conforms_to_the_grammar() {
    let bodies = all_bodies();
    // Cover the full §8.3 record set.
    let tags: Vec<u8> = bodies.iter().map(Body::tag).collect();
    for t in JOURNAL_RECORD_TAGS {
        assert!(tags.contains(t), "tag {t} missing from the round-trip set");
    }
    assert_eq!(tags, (0u8..=18).collect::<Vec<_>>(), "tags in 0..=18 order");

    for (i, body) in bodies.into_iter().enumerate() {
        let record = Record::new(i as u64, body);
        let bytes = record.to_canonical().expect("encode");

        // Round-trips through the Rust codec.
        let back = Record::from_canonical(&bytes).expect("decode");
        assert_eq!(back, record, "record tag {} round-trip", record.tag());

        // Canonical: a second encode is byte-identical.
        assert_eq!(back.to_canonical().unwrap(), bytes);

        // Conforms to the authoritative ABI grammar (§8.3 / §13).
        cddl_cat::validate_cbor_bytes("journal-record", JOURNAL_CDDL, &bytes)
            .unwrap_or_else(|e| panic!("tag {} failed grammar validation: {e:?}", record.tag()));
    }
}

#[test]
fn read_back_sidecar_branch_conforms() {
    // The other arm of the read-back group choice (sidecar instead of inline value).
    let record = Record::new(
        2,
        Body::ReadBack(ReadBackRec {
            src: 0,
            kind: 1,
            status: 0,
            value: None,
            sidecar: Some(SidecarRef {
                hash: h(1),
                size: 999_999,
                seg: 3,
            }),
        }),
    );
    let bytes = record.to_canonical().unwrap();
    assert_eq!(Record::from_canonical(&bytes).unwrap(), record);
    cddl_cat::validate_cbor_bytes("journal-record", JOURNAL_CDDL, &bytes).unwrap();
}

#[test]
fn unknown_tag_is_rejected() {
    // A hand-built [64, 0, {}] array is a well-formed CBOR array but an unassigned tag.
    let v = ciborium::value::Value::Array(vec![
        ciborium::value::Value::Integer(64.into()),
        ciborium::value::Value::Integer(0.into()),
        ciborium::value::Value::Map(vec![]),
    ]);
    let bytes = daemon_vhc_proto::to_canonical_vec(&v).unwrap();
    assert!(Record::from_canonical(&bytes).is_err());
}

// ----- segment framing + chain + seal (§8.2) -----

fn write_segment(records: &[Record], prev: [u8; 32]) -> (Vec<u8>, [u8; 32]) {
    let dir = tempdir();
    let path = dir.join("segment-00000000.dvhcjrn");
    let header = SegmentHeader {
        id: ident(),
        segment: 0,
        prev_blake3: prev,
    };
    let mut w = SegmentWriter::create(&path, &header).unwrap();
    for r in records {
        w.append(r).unwrap();
    }
    w.commit().unwrap();
    let file_hash = w.file_blake3();
    let bytes = std::fs::read(&path).unwrap();
    (bytes, file_hash)
}

#[test]
fn segment_scans_records_and_reports_clean_end() {
    let records: Vec<Record> = all_bodies()
        .into_iter()
        .take(6)
        .enumerate()
        .map(|(i, b)| Record::new(i as u64, b))
        .collect();
    let (bytes, file_hash) = write_segment(&records, GENESIS_PREV);

    let scan = scan_bytes(&bytes).unwrap();
    assert_eq!(scan.header.segment, 0);
    assert_eq!(scan.header.prev_blake3, GENESIS_PREV);
    assert!(!scan.sealed);
    assert!(!scan.truncated);
    assert_eq!(scan.records, records);
    assert_eq!(scan.durable_len, bytes.len() as u64);
    assert_eq!(scan.complete_file_blake3, file_hash);
}

#[test]
fn seal_excludes_itself_and_chains() {
    let dir = tempdir();
    let path = dir.join("segment-00000000.dvhcjrn");
    let header = SegmentHeader {
        id: ident(),
        segment: 0,
        prev_blake3: GENESIS_PREV,
    };
    let mut w = SegmentWriter::create(&path, &header).unwrap();
    for (i, b) in all_bodies().into_iter().take(3).enumerate() {
        w.append(&Record::new(i as u64, b)).unwrap();
    }
    let file_hash = w.seal().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let scan = scan_bytes(&bytes).unwrap();
    assert!(scan.sealed, "a cleanly-rolled segment ends with a seal");
    assert!(!scan.truncated);
    // The last record is the seal; its hash covers the segment EXCLUDING its own frame.
    let last = scan.records.last().unwrap();
    match &last.body {
        Body::Seal(SealRec { records, .. }) => assert_eq!(*records, 3),
        other => panic!("expected a seal record, got {other:?}"),
    }
    // The complete-file hash (post-seal) is the chain link the next segment carries.
    assert_eq!(scan.complete_file_blake3, file_hash);

    // A tampered seal hash is caught as a broken chain.
    // (Flip a byte inside the seal's segment_blake3 field — detected by scan.)
    let mut tampered = bytes.clone();
    // The seal frame is the last frame; corrupting the CRC-protected body would trip the CRC first,
    // so instead corrupt a body byte and confirm the CRC catches it (no silent acceptance).
    let n = tampered.len();
    tampered[n - 6] ^= 0xff;
    let scan2 = scan_bytes(&tampered).unwrap();
    assert!(
        scan2.truncated,
        "a corrupted seal frame is discarded, not silently accepted"
    );
}

// ----- sidecars (§8.5) -----

#[test]
fn sidecar_round_trips_and_is_content_addressed() {
    let dir = tempdir();
    let store = SidecarStore::open(&dir, ident(), StaticKey::new([7u8; 32])).unwrap();

    let plaintext = vec![0xABu8; READBACK_INLINE_MAX * 3];
    let sref = store.put(5, 1, 0, &plaintext).unwrap();
    assert_eq!(sref.hash, blake3_hash(&plaintext));
    assert_eq!(sref.size, plaintext.len() as u64);
    assert!(store.contains(&sref));

    // Decrypt + content-address verify.
    let got = store.get(&sref, 5, 1).unwrap();
    assert_eq!(got, plaintext);

    // Content-addressed: re-putting identical bytes yields the same ref and no error.
    let sref2 = store.put(5, 1, 0, &plaintext).unwrap();
    assert_eq!(sref2.hash, sref.hash);
}

#[test]
fn sidecar_missing_is_typed_not_silent() {
    let dir = tempdir();
    let store = SidecarStore::open(&dir, ident(), StaticKey::new([7u8; 32])).unwrap();
    let sref = SidecarRef {
        hash: h(200),
        size: 10,
        seg: 0,
    };
    match store.get(&sref, 1, 0) {
        Err(SidecarError::Missing { hash }) => assert_eq!(hash, h(200)),
        other => panic!("expected a typed Missing, got {other:?}"),
    }
}

#[test]
fn sidecar_wrong_nonce_or_tamper_fails_aead() {
    let dir = tempdir();
    let store = SidecarStore::open(&dir, ident(), StaticKey::new([7u8; 32])).unwrap();
    let plaintext = vec![0x11u8; READBACK_INLINE_MAX + 1];
    let sref = store.put(5, 1, 0, &plaintext).unwrap();

    // Wrong instantiation counter (nonce input) → AEAD auth fails (belt-and-braces, §8.5).
    assert!(matches!(
        store.get(&sref, 5, 999),
        Err(SidecarError::Verify(_))
    ));
    // Wrong referencing ordinal (nonce + AAD header) → fails.
    assert!(matches!(
        store.get(&sref, 6, 1),
        Err(SidecarError::Verify(_))
    ));

    // On-disk ciphertext tamper → AEAD auth fails.
    let path = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().map(|x| x == "dvhcsc").unwrap_or(false))
        .unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let n = bytes.len();
    bytes[n - 1] ^= 0xff;
    std::fs::write(&path, &bytes).unwrap();
    assert!(matches!(
        store.get(&sref, 5, 1),
        Err(SidecarError::Verify(_))
    ));
}

#[test]
fn sidecar_cross_journal_splice_is_rejected() {
    let dir = tempdir();
    // Journal A writes a sidecar.
    let store_a = SidecarStore::open(&dir, ident(), StaticKey::new([7u8; 32])).unwrap();
    let plaintext = vec![0x22u8; READBACK_INLINE_MAX + 1];
    let sref = store_a.put(5, 1, 0, &plaintext).unwrap();

    // Journal B (different execution identity, different key) tries to read A's spliced sidecar.
    let mut other = ident();
    other.instance = 99; // a different incarnation → different owner header + key
    let store_b = SidecarStore::open(&dir, other, StaticKey::new([8u8; 32])).unwrap();
    assert!(matches!(
        store_b.get(&sref, 5, 1),
        Err(SidecarError::Verify(_))
    ));
}

// ----- worker input-replay verifier skeleton (§8.7) -----

/// A sim guest that replays exactly what it was told (echoes recorded publishes) — the trivial
/// bit-exact guest. A2 replaces this with the real host-runtime event loop.
struct EchoGuest {
    to_emit: Vec<ExpectedDecision>,
}
impl GuestUnderReplay for EchoGuest {
    fn deliver_event(&mut self, _ord: u64, _at: u64, _frame: &[u8]) -> Vec<ExpectedDecision> {
        std::mem::take(&mut self.to_emit)
    }
    fn supply_import(&mut self, _step: &ReplayStep) {}
}

struct NoPayloads;
impl PayloadSource for NoPayloads {
    fn fetch(&self, _sref: &SidecarRef, _ord: u64) -> Option<Vec<u8>> {
        None
    }
}
struct AllPayloads;
impl PayloadSource for AllPayloads {
    fn fetch(&self, _sref: &SidecarRef, _ord: u64) -> Option<Vec<u8>> {
        Some(vec![0u8; 1])
    }
}

#[test]
fn verifier_skeleton_pass_diverge_missing_terminal() {
    // A plan: one event that should produce one publish.
    let expected = ExpectedDecision::Publish {
        ord: 1,
        channel: 0,
        seq: 0,
        hash: h(9),
    };
    let plan = ReplayPlan {
        steps: vec![ReplayStep::Event {
            ord: 0,
            at: 0,
            frame: b"f".to_vec(),
        }],
        expected: vec![expected.clone()],
        composition: Default::default(),
    };

    // Pass: the guest reproduces the recorded publish.
    let mut good = EchoGuest {
        to_emit: vec![expected.clone()],
    };
    assert_eq!(
        run_replay(&plan, &mut good, &AllPayloads),
        ReplayOutcome::Pass { decisions: 1 }
    );

    // Diverged: the guest emits a different publish.
    let mut bad = EchoGuest {
        to_emit: vec![ExpectedDecision::Publish {
            ord: 1,
            channel: 0,
            seq: 0,
            hash: h(10),
        }],
    };
    assert!(matches!(
        run_replay(&plan, &mut bad, &AllPayloads),
        ReplayOutcome::Diverged(_)
    ));

    // MissingPayload: a sidecar read-back step whose payload can't be fetched (§8.7, never a pass).
    let miss_plan = ReplayPlan {
        steps: vec![ReplayStep::ReadBack {
            ord: 3,
            src: 0,
            kind: 1,
            status: 0,
            value: daemon_vhc_observe::journal::verifier::ReadBackValue::Sidecar(SidecarRef {
                hash: h(50),
                size: 10,
                seg: 0,
            }),
        }],
        expected: vec![],
        composition: Default::default(),
    };
    let mut g = EchoGuest { to_emit: vec![] };
    assert_eq!(
        run_replay(&miss_plan, &mut g, &NoPayloads),
        ReplayOutcome::MissingPayload {
            hash: h(50),
            ord: 3
        }
    );

    // TerminalFault: a recorded trap (kind 1) is injected at its ordinal (§8.7).
    let term_plan = ReplayPlan {
        steps: vec![ReplayStep::Terminal { ord: 9, kind: 1 }],
        expected: vec![],
        composition: Default::default(),
    };
    let mut g2 = EchoGuest { to_emit: vec![] };
    assert_eq!(
        run_replay(&term_plan, &mut g2, &AllPayloads),
        ReplayOutcome::TerminalFault { ord: 9, kind: 1 }
    );
}

#[test]
fn verifier_plan_from_journal_records() {
    let records = vec![
        Record::new(
            0,
            Body::Event(EventRec {
                at: 1,
                frame: b"e".to_vec(),
            }),
        ),
        Record::new(
            1,
            Body::Publish(PublishRec {
                channel: 0,
                seq: 0,
                hash: h(9),
                frame: b"p".to_vec(),
            }),
        ),
        Record::new(2, Body::Clock(ClockRec { now: 5 })),
    ];
    let plan = ReplayPlan::from_records(&records);
    assert_eq!(plan.steps.len(), 2); // event + clock
    assert_eq!(plan.expected.len(), 1); // publish
}

// ----- head↔segment identity binding (the shared archive-reader verifier) -----

mod binding {
    use super::*;
    use daemon_vhc_observe::journal::binding::{verify_head_binding, HeadClaim};
    use daemon_vhc_observe::journal::ScanResult;

    /// A real sealed segment (through the writer): 3 records, ordinal 4, a non-genesis prev.
    fn sealed_scan() -> ScanResult {
        let dir = super::tempdir();
        let path = dir.join("segment-00000004.dvhcjrn");
        let header = SegmentHeader {
            id: ident(),
            segment: 4,
            prev_blake3: [0xAB; 32],
        };
        let mut w = SegmentWriter::create(&path, &header).unwrap();
        for ord in 0..3u64 {
            w.append(&Record::new(ord, Body::Clock(ClockRec { now: 1000 + ord })))
                .unwrap();
        }
        w.seal().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        scan_bytes(&bytes).unwrap()
    }

    fn claim() -> HeadClaim<'static> {
        HeadClaim {
            run_id: h(1),
            epoch: 4,
            role: "trainer",
            module: h(2),
            chain_instance: Some(42),
            segment: 4,
            prev_hash: Hash([0xAB; 32]),
            records: 3,
        }
    }

    #[test]
    fn a_matching_head_binds() {
        verify_head_binding(&sealed_scan(), &claim()).expect("binds");
    }

    /// The adoption shape (defect 16): a claim form without the chain scope (the harness
    /// `ChainHead`, whose `instance` is the ATTESTING span) skips the instance comparison but
    /// still binds everything else.
    #[test]
    fn a_scopeless_claim_skips_the_chain_scope_check() {
        let mut c = claim();
        c.chain_instance = None;
        verify_head_binding(&sealed_scan(), &c).expect("binds without the chain scope");
    }

    /// Every claimed field is load-bearing: mutate each one and the binding must refuse,
    /// naming the disagreement.
    #[test]
    fn every_identity_field_is_load_bearing() {
        let scan = sealed_scan();
        let cases: Vec<(&str, HeadClaim<'static>)> = vec![
            ("run_id", {
                let mut c = claim();
                c.run_id = h(0xEE);
                c
            }),
            ("epoch", {
                let mut c = claim();
                c.epoch = 9;
                c
            }),
            ("role", {
                let mut c = claim();
                c.role = "coordinator";
                c
            }),
            ("module", {
                let mut c = claim();
                c.module = h(0xEE);
                c
            }),
            ("chain scope", {
                let mut c = claim();
                c.chain_instance = Some(43);
                c
            }),
            ("ordinal", {
                let mut c = claim();
                c.segment = 5;
                c
            }),
            ("prev link", {
                let mut c = claim();
                c.prev_hash = h(0xCD);
                c
            }),
            ("record count", {
                let mut c = claim();
                c.records = 4;
                c
            }),
        ];
        for (field, c) in cases {
            let err = verify_head_binding(&scan, &c)
                .expect_err(&format!("a mutated {field} must refuse"));
            assert!(
                err.0.contains("!=") || err.0.contains("attests"),
                "{field}: the refusal names the disagreement, got {err}"
            );
        }
    }

    /// An unsealed segment (no clean roll) never binds — archive material is sealed only.
    #[test]
    fn an_unsealed_segment_refuses() {
        let dir = super::tempdir();
        let path = dir.join("segment-00000004.dvhcjrn");
        let header = SegmentHeader {
            id: ident(),
            segment: 4,
            prev_blake3: [0xAB; 32],
        };
        let mut w = SegmentWriter::create(&path, &header).unwrap();
        w.append(&Record::new(0, Body::Clock(ClockRec { now: 1 })))
            .unwrap();
        w.commit().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let scan = scan_bytes(&bytes).unwrap();
        let err = verify_head_binding(&scan, &claim()).expect_err("unsealed refuses");
        assert!(err.0.contains("sealed"), "got {err}");
    }

    /// A seal whose declared count disagrees with the records the scan actually recovered
    /// refuses even when the head repeats the same wrong number — the segment is internally
    /// inconsistent, and the head's count would otherwise vacuously "match".
    #[test]
    fn a_lying_seal_count_refuses() {
        let dir = super::tempdir();
        let path = dir.join("segment-00000004.dvhcjrn");
        let header = SegmentHeader {
            id: ident(),
            segment: 4,
            prev_blake3: [0xAB; 32],
        };
        // Hand-build the seal with a wrong count but a CORRECT self-excluding hash, so the
        // scan accepts it and only the binding catches the lie.
        let mut w = SegmentWriter::create(&path, &header).unwrap();
        w.append(&Record::new(0, Body::Clock(ClockRec { now: 1 })))
            .unwrap();
        let pre_seal = w.file_blake3();
        let seal = Record::new(
            1,
            Body::Seal(SealRec {
                segment_blake3: Hash(pre_seal),
                records: 2, // lies: only 1 record precedes the seal
            }),
        );
        w.append_framed(&SegmentWriter::encode(&seal).unwrap())
            .unwrap();
        w.commit().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let scan = scan_bytes(&bytes).unwrap();
        let mut c = claim();
        c.records = 2; // the head repeats the lie
        let err = verify_head_binding(&scan, &c).expect_err("internal inconsistency refuses");
        assert!(err.0.contains("recovered"), "got {err}");
    }
}

// ----- a tiny tempdir helper (no external dev-dep) -----

fn tempdir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let mut base = std::env::temp_dir();
    let unique = format!(
        "dvhc-journal-test-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    );
    base.push(unique);
    std::fs::create_dir_all(&base).unwrap();
    base
}
