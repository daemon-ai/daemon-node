// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Driver unit tests: pump queue policies (§4.7), chunk-addressed completion verification,
//! timer determinism (§6.3), stop-cut semantics (§4.4), the §10.2 manifest/descriptor wire, and
//! the §12.1 signed-frame envelope.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use ciborium::value::Value;
use daemon_vhc_abi::{COMP_ERR_HASH_MISMATCH, EV_TAG_STOP, FRAME_ENVELOPE_DOMAIN_V2};
use daemon_vhc_proto::sign::verify_bytes;
use daemon_vhc_proto::{peer_id, to_canonical_vec, SigningKey};
use wasmtime::StoreLimitsBuilder;

use crate::run::buffer::BufferTable;
use crate::run::completion::{CompError, CompletionResult};
use crate::run::driver::BufferStreams;
use crate::run::journal::{JournalSink, MemorySink, SinkEntry};
use crate::run::ops::{OpRequest, OpTable};
use crate::run::streams::StreamTable;
use crate::trap::Trap;

use super::chunks::{decode_chunk_descriptor, verify_covering_span};
use super::config::{DeliverVerdict, OpOutcome, RunIdentity};
use super::host::{build_signed_frame, Host, SliceState};
use super::migration::{build_migration_descriptor, decode_manifest_sections};
use super::pump::{fire_due_timers, ArmedTimer, PumpHandle, PumpShared, PumpState};

fn test_state(sink: Box<dyn JournalSink>) -> PumpState {
    PumpState {
        queue: VecDeque::new(),
        timers: Vec::new(),
        next_timer_id: 1,
        staged: std::collections::BTreeMap::new(),
        next_host_staging_id: 1,
        next_guest_staging_id: 1,
        sink,
        timer_depth: 2,
        payload_depth: 4,
        gossip_depth: 2,
        spool_frames: 4,
        per_sender_quota: 2,
        auth_spooled: 0,
        auth_per_sender: std::collections::HashMap::new(),
        spool_exhausted_reported: false,
        gossip_arrivals: std::collections::HashMap::new(),
        metrics: Vec::new(),
        logs: Vec::new(),
        guest_panic: None,
        published: Vec::new(),
        buffers: BufferTable::new(0, 0, 0),
        ops: OpTable::new(0, 0),
        chunk_maps: std::collections::HashMap::new(),
        state_chunk_maps: std::collections::HashMap::new(),
        data_read_budget: 0,
        data_read_used: 0,
        streams: StreamTable::new(0),
        state: crate::run::state_store::StateStore::new(
            crate::run::state_store::StateStoreConfig::default(),
        ),
        guest_memory_high_water: 0,
        buffer_streams: BufferStreams::default(),
        op_requests: Vec::new(),
        stop_enqueued: false,
        stop_cut: None,
        draining: false,
        drain_deadline_at: None,
        accepted_snapshot: None,
        egress_hook: None,
        migrate_validated: false,
    }
}

fn test_pump(sink: Box<dyn JournalSink>) -> PumpHandle {
    PumpHandle {
        shared: Arc::new(PumpShared {
            state: Mutex::new(test_state(sink)),
            wake: Condvar::new(),
            t0: Instant::now(),
            hold: AtomicBool::new(false),
        }),
    }
}

fn signed_stub() -> Vec<u8> {
    b"signed-frame-stub".to_vec()
}

/// A chunked fixture: 80 bytes at chunk_size 32 (two full chunks + one short).
fn chunk_fixture() -> (daemon_vhc_proto::ChunkMap, Vec<u8>) {
    let bytes: Vec<u8> = (0u8..80).collect();
    let map = daemon_vhc_proto::ChunkMap {
        chunk_size: 32,
        token_count: 40,
        byte_len: 80,
        chunk_hashes: daemon_vhc_proto::chunk_hashes(&bytes, 32),
    };
    (map, bytes)
}

#[test]
fn covering_span_verification_accepts_true_chunks_and_refuses_lies() {
    let (map, bytes) = chunk_fixture();
    // The full span verifies; a mid-span range's covering chunks verify.
    assert_eq!(verify_covering_span(&map, 0, bytes.clone()).unwrap(), bytes);
    assert!(verify_covering_span(&map, 32, bytes[32..].to_vec()).is_ok());
    // One flipped byte in any covering chunk is a described refusal.
    let mut tampered = bytes.clone();
    tampered[40] ^= 0xFF;
    let err = verify_covering_span(&map, 0, tampered).unwrap_err();
    assert!(err.contains("chunk 1"), "{err}");
    // A truncated span is refused, never partially accepted.
    assert!(verify_covering_span(&map, 0, bytes[..40].to_vec())
        .unwrap_err()
        .contains("truncates chunk 1"));
    // A span past the chunk list is refused.
    let mut overlong = bytes.clone();
    overlong.extend_from_slice(&[0u8; 32]);
    assert!(verify_covering_span(&map, 0, overlong)
        .unwrap_err()
        .contains("past the chunk list"));
}

#[test]
fn chunk_descriptor_decode_round_trips_and_rejects_malformed() {
    let (map, _) = chunk_fixture();
    let hashes: Vec<ciborium::value::Value> = map
        .chunk_hashes
        .iter()
        .map(|h| ciborium::value::Value::Bytes(h.0.to_vec()))
        .collect();
    let doc = ciborium::value::Value::Array(vec![
        ciborium::value::Value::from(map.chunk_size),
        ciborium::value::Value::from(map.token_count),
        ciborium::value::Value::from(map.byte_len),
        ciborium::value::Value::Array(hashes),
    ]);
    let desc = daemon_vhc_proto::to_canonical_vec(&doc).unwrap();
    let decoded = decode_chunk_descriptor(&desc).unwrap();
    assert_eq!(decoded, map);
    assert_eq!(decoded.fold(), map.fold());

    assert!(decode_chunk_descriptor(b"junk").is_err(), "not CBOR");
    // Degenerate geometry (chunk list shorter than the byte length needs) is refused.
    let bad = ciborium::value::Value::Array(vec![
        ciborium::value::Value::from(32u64),
        ciborium::value::Value::from(40u64),
        ciborium::value::Value::from(80u64),
        ciborium::value::Value::Array(vec![ciborium::value::Value::Bytes(vec![0u8; 32])]),
    ]);
    let bad_desc = daemon_vhc_proto::to_canonical_vec(&bad).unwrap();
    assert!(decode_chunk_descriptor(&bad_desc)
        .unwrap_err()
        .contains("degenerate"));
}

#[test]
fn chunked_completion_verifies_covering_chunks_then_slices_the_range() {
    let (map, bytes) = chunk_fixture();
    let fold = map.fold();
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let pump = test_pump(Box::new(sink.clone()));
    {
        let mut st = pump.shared.state.lock().unwrap();
        st.chunk_maps.insert(fold.0, map.clone());
    }
    // The guest asked for [40, 60); chunk 1 ([32, 64)) covers it entirely.
    let (span_off, span_len) =
        daemon_vhc_proto::covering_span(map.byte_len, map.chunk_size, 40, 60);
    assert_eq!((span_off, span_len), (32, 32));
    let request = OpRequest::ArtifactRange {
        hash: fold.0,
        range_off: 40,
        range_len: 20,
        span_off,
        span_len,
    };
    let op = {
        let mut st = pump.shared.state.lock().unwrap();
        let op = st.ops.begin(request.clone()).unwrap();
        st.op_requests.push((op, request));
        op
    };
    // A span answer with true chunks completes Ok(handle) carrying exactly the range.
    let handle = pump
        .complete_op(
            op,
            OpOutcome::RangeDone {
                bytes: bytes[32..64].to_vec(),
            },
        )
        .unwrap()
        .expect("range completion mints a buffer");
    let st = pump.shared.state.lock().unwrap();
    let buf = st.buffers.resolve(handle).unwrap();
    assert_eq!(buf.as_slice(), &bytes[40..60]);
}

#[test]
fn chunked_completion_refuses_tampered_spans_typed() {
    let (map, bytes) = chunk_fixture();
    let fold = map.fold();
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let pump = test_pump(Box::new(sink.clone()));
    {
        let mut st = pump.shared.state.lock().unwrap();
        st.chunk_maps.insert(fold.0, map.clone());
    }
    let request = OpRequest::ArtifactRange {
        hash: fold.0,
        range_off: 0,
        range_len: 16,
        span_off: 0,
        span_len: 32,
    };
    let op = {
        let mut st = pump.shared.state.lock().unwrap();
        st.ops.begin(request).unwrap()
    };
    let mut lied = bytes[..32].to_vec();
    lied[3] ^= 0x01;
    let minted = pump
        .complete_op(op, OpOutcome::RangeDone { bytes: lied })
        .unwrap();
    assert!(minted.is_none(), "no buffer for a refused span");
    // The journaled completion is the typed HashMismatch — the guest never saw the bytes.
    let entries = sink.lock().unwrap().entries.clone();
    let completion = entries
        .iter()
        .find_map(|e| match e {
            SinkEntry::Completion { result, .. } => Some(result.clone()),
            _ => None,
        })
        .expect("completion journaled");
    let decoded = CompletionResult::decode(&completion).unwrap();
    assert!(matches!(
        decoded,
        CompletionResult::Err(CompError { code, .. })
            if code == COMP_ERR_HASH_MISMATCH
    ));
}

/// A whole-object answer for a chunked request (the in-process content-store seat) is
/// span-extracted + chunk-verified by the pump — same trust path, different transfer shape.
#[test]
fn chunked_completion_accepts_whole_object_answers() {
    let (map, bytes) = chunk_fixture();
    let fold = map.fold();
    let pump = test_pump(Box::new(MemorySink::new()));
    {
        let mut st = pump.shared.state.lock().unwrap();
        st.chunk_maps.insert(fold.0, map.clone());
    }
    let request = OpRequest::ArtifactRange {
        hash: fold.0,
        range_off: 70,
        range_len: 0, // to the end
        span_off: 64,
        span_len: 16,
    };
    let op = {
        let mut st = pump.shared.state.lock().unwrap();
        st.ops.begin(request).unwrap()
    };
    let handle = pump
        .complete_op(
            op,
            OpOutcome::FetchDone {
                artifact: bytes.clone(),
            },
        )
        .unwrap()
        .expect("whole-object answer verifies");
    let st = pump.shared.state.lock().unwrap();
    assert_eq!(st.buffers.resolve(handle).unwrap().as_slice(), &bytes[70..]);
}

#[test]
fn authoritative_spool_backpressures_and_journals_the_typed_stall() {
    // test_state: spool_frames = 4, per_sender_quota = 2 (§4.7: bounded, never drops).
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let pump = test_pump(Box::new(sink.clone()));
    let s1 = [1u8; 32];
    let s2 = [2u8; 32];
    let s3 = [3u8; 32];
    // Per-sender quota: sender 1's third undelivered frame back-pressures HIM only.
    assert_eq!(
        pump.deliver_frame(0, 0, s1, b"a".to_vec(), signed_stub())
            .unwrap(),
        DeliverVerdict::Accepted
    );
    assert_eq!(
        pump.deliver_frame(0, 1, s1, b"b".to_vec(), signed_stub())
            .unwrap(),
        DeliverVerdict::Accepted
    );
    assert_eq!(
        pump.deliver_frame(0, 2, s1, b"c".to_vec(), signed_stub())
            .unwrap(),
        DeliverVerdict::SenderQuota,
        "per-sender quota bounds the DoS vector (§4.7)"
    );
    // Other senders proceed until the SPOOL bound (4).
    assert_eq!(
        pump.deliver_frame(0, 0, s2, b"d".to_vec(), signed_stub())
            .unwrap(),
        DeliverVerdict::Accepted
    );
    assert_eq!(
        pump.deliver_frame(0, 1, s2, b"e".to_vec(), signed_stub())
            .unwrap(),
        DeliverVerdict::Accepted
    );
    assert_eq!(
        pump.deliver_frame(0, 0, s3, b"f".to_vec(), signed_stub())
            .unwrap(),
        DeliverVerdict::SpoolFull,
        "genuine spool exhaustion back-pressures (never a drop)"
    );
    // The typed stall was journaled ONCE for the episode (§6.7 tag 16), even on a re-hit.
    assert_eq!(
        pump.deliver_frame(0, 0, s3, b"f".to_vec(), signed_stub())
            .unwrap(),
        DeliverVerdict::SpoolFull
    );
    let entries = &sink.lock().unwrap().entries;
    let stalls: Vec<_> = entries
        .iter()
        .filter(|e| matches!(e, SinkEntry::Condition { code, .. } if code == "SpoolExhausted"))
        .collect();
    assert_eq!(stalls.len(), 1, "one condition per exhaustion episode");
    // Nothing was dropped: the reliable class holds every accepted frame.
    assert!(!entries.iter().any(|e| matches!(e, SinkEntry::Drop { .. })));
}

#[test]
fn the_egress_hook_fires_on_registration_and_on_guest_egress() {
    // The embedder egress wake: registering fires once (nothing already landed is silently
    // unannounced), and every subsequent guest-egress landing fires it again. The hook is a
    // pure signal — the embedder still drains through `published`/`take_op_requests`.
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let pump = test_pump(Box::new(sink));
    let fires = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = fires.clone();
    pump.set_egress_hook(Arc::new(move || {
        counter.fetch_add(1, Ordering::SeqCst);
    }));
    assert_eq!(
        fires.load(Ordering::SeqCst),
        1,
        "fires once at registration"
    );
    // Simulate guest egress landing under the pump lock (the import-body path).
    {
        let mut st = pump.shared.state.lock().unwrap();
        st.published.push((0, 0, b"frame".to_vec()));
        st.note_egress();
        st.metrics.push(("loss".into(), 1.0));
        st.note_egress();
    }
    assert_eq!(
        fires.load(Ordering::SeqCst),
        3,
        "each landing wakes the embedder"
    );
}

#[test]
fn gossip_class_drops_oldest_at_depth_and_journals_identity() {
    // test_state: gossip_depth = 2 (§4.7 drop-oldest, journaled tag 7 class 2).
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let pump = test_pump(Box::new(sink.clone()));
    let g = [9u8; 32];
    pump.deliver_gossip(5, g, b"g0".to_vec()).unwrap();
    pump.deliver_gossip(5, g, b"g1".to_vec()).unwrap();
    pump.deliver_gossip(5, g, b"g2".to_vec()).unwrap();
    let entries = &sink.lock().unwrap().entries;
    let drops: Vec<&SinkEntry> = entries
        .iter()
        .filter(|e| matches!(e, SinkEntry::Drop { class: 2, .. }))
        .collect();
    assert_eq!(drops.len(), 1, "third arrival drops the OLDEST");
    let SinkEntry::Drop { rule, dropped, .. } = drops[0] else {
        unreachable!()
    };
    assert_eq!(*rule, daemon_vhc_abi::COALESCE_DROP_OLDEST);
    assert_eq!(
        (dropped.channel, dropped.sender, dropped.seq),
        (Some(5), Some(g), Some(0)),
        "the drop names the oldest arrival's full identity"
    );
}

#[test]
fn payload_ready_dedups_by_hash_and_bounds_depth() {
    // test_state: payload_depth = 4 (§4.7 class 0: dedup-by-hash + bounded queue).
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let pump = test_pump(Box::new(sink.clone()));
    // Identical bytes coalesce: one announcement, one journaled dedup.
    let id1 = pump.stage_payload(b"same".to_vec(), None).unwrap();
    let id2 = pump.stage_payload(b"same".to_vec(), None).unwrap();
    assert_eq!(id1, id2, "dedup returns the already-staged id");
    // Distinct hashes beyond the depth drop the OLDEST announcement (and unstage it).
    for i in 0u8..4 {
        pump.stage_payload(vec![i], None).unwrap();
    }
    let entries = &sink.lock().unwrap().entries;
    let dedups = entries
        .iter()
        .filter(
            |e| matches!(e, SinkEntry::Drop { class: 0, rule, .. } if *rule == daemon_vhc_abi::COALESCE_DEDUP_HASH),
        )
        .count();
    assert!(
        dedups >= 2,
        "the dedup + the depth drop are journaled: {entries:?}"
    );
}

#[test]
fn due_timers_fire_in_deterministic_fire_at_then_id_order() {
    let mut st = test_state(Box::new(MemorySink::new()));
    st.timer_depth = 16;
    st.timers = vec![
        ArmedTimer { id: 3, fire_at: 10 },
        ArmedTimer { id: 1, fire_at: 10 },
        ArmedTimer { id: 2, fire_at: 5 },
        ArmedTimer { id: 4, fire_at: 99 }, // not due
    ];
    fire_due_timers(&mut st, 20).unwrap();
    let fired: Vec<u64> = st.queue.iter().filter_map(|q| q.timer_id).collect();
    assert_eq!(fired, vec![2, 1, 3], "(fire_at, id) ascending");
    assert_eq!(st.timers.len(), 1, "undue timer stays armed");
}

#[test]
fn timer_queue_depth_drops_oldest_and_journals_it() {
    let mut st = test_state(Box::new(MemorySink::new()));
    st.timers = vec![
        ArmedTimer { id: 1, fire_at: 1 },
        ArmedTimer { id: 2, fire_at: 2 },
        ArmedTimer { id: 3, fire_at: 3 },
    ];
    // Depth 2: firing all three drops the oldest queued Timer (id 1), journaled (§4.7).
    fire_due_timers(&mut st, 10).unwrap();
    let queued: Vec<u64> = st.queue.iter().filter_map(|q| q.timer_id).collect();
    assert_eq!(queued, vec![2, 3]);
}

#[test]
fn stop_cut_already_passed_enqueues_stop_immediately_and_fences_timers() {
    let pump = test_pump(Box::new(MemorySink::new()));
    {
        let mut st = pump.shared.state.lock().unwrap();
        st.published.push((0, 0, b"frame".to_vec()));
        // A due timer armed at registration time: it must never fire past the cut.
        st.timers.push(ArmedTimer { id: 7, fire_at: 0 });
    }
    pump.stop_at_publishes(1, 0).unwrap();
    let mut st = pump.shared.state.lock().unwrap();
    assert!(st.stop_enqueued, "cut already passed: stop registers now");
    assert_eq!(st.queue.len(), 1);
    assert_eq!(st.queue[0].tag, EV_TAG_STOP);
    // The delivery loop's gate: with stop enqueued, due timers never fire (§4.4).
    fire_due_timers_gated(&mut st, 100).unwrap();
    assert_eq!(st.queue.len(), 1, "no Timer enters the stream behind Stop");
}

/// The `next_event` loop's exact firing condition, extracted for the gate assertion above.
fn fire_due_timers_gated(st: &mut PumpState, now: u64) -> Result<(), Trap> {
    if !st.draining && !st.stop_enqueued {
        fire_due_timers(st, now)?;
    }
    Ok(())
}

#[test]
fn stop_cut_pending_yields_to_explicit_stop_and_stays_idempotent() {
    let pump = test_pump(Box::new(MemorySink::new()));
    pump.stop_at_publishes(5, 0).unwrap();
    {
        let st = pump.shared.state.lock().unwrap();
        assert!(!st.stop_enqueued, "cut not reached: no stop yet");
        assert_eq!(st.stop_cut, Some((5, 0)));
    }
    pump.stop(1).unwrap();
    pump.stop(1).unwrap(); // idempotent
    pump.stop_at_publishes(0, 2).unwrap(); // registration after stop is a no-op
    let st = pump.shared.state.lock().unwrap();
    let stops = st.queue.iter().filter(|q| q.tag == EV_TAG_STOP).count();
    assert_eq!(stops, 1, "exactly one terminal Stop");
    assert_eq!(st.stop_cut, None, "an explicit stop clears the cut");
}

#[test]
fn manifest_sections_decode_and_descriptor_round_trips() {
    use ciborium::value::Value;
    // A minimal §10.2 state-manifest: schema/module + one section decl.
    let section_bytes = b"counter-state".to_vec();
    let manifest = Value::Map(vec![
        (Value::Text("schema".into()), Value::Integer(1.into())),
        (Value::Text("module".into()), Value::Bytes(vec![7u8; 32])),
        (
            Value::Text("sections".into()),
            Value::Array(vec![Value::Map(vec![
                (Value::Text("name".into()), Value::Text("counter".into())),
                (Value::Text("schema".into()), Value::Integer(1.into())),
                (
                    Value::Text("hash".into()),
                    Value::Bytes(blake3::hash(&section_bytes).as_bytes().to_vec()),
                ),
                (
                    Value::Text("size".into()),
                    Value::Integer((section_bytes.len() as u64).into()),
                ),
                (Value::Text("class".into()), Value::Integer(0.into())),
            ])]),
        ),
    ]);
    let mut manifest_bytes = Vec::new();
    ciborium::into_writer(&manifest, &mut manifest_bytes).unwrap();

    let decls = decode_manifest_sections(&manifest_bytes).expect("decodes");
    assert_eq!(decls.len(), 1);
    assert_eq!(decls[0].name, "counter");
    assert_eq!(decls[0].size, section_bytes.len() as u64);
    assert_eq!(&decls[0].hash, blake3::hash(&section_bytes).as_bytes());

    // The descriptor embeds the manifest value verbatim + the restore bindings (§10.2).
    let desc = build_migration_descriptor(
        &manifest_bytes,
        &[super::migration::RestoreBinding::Inline {
            name: "counter".into(),
            staging_id: 42,
        }],
    )
    .expect("builds");
    let v: Value = ciborium::de::from_reader(desc.as_slice()).unwrap();
    let Value::Map(entries) = v else {
        panic!("descriptor is a map")
    };
    let get = |name: &str| {
        entries
            .iter()
            .find_map(|(k, val)| match k {
                Value::Text(t) if t == name => Some(val.clone()),
                _ => None,
            })
            .expect("descriptor key")
    };
    assert_eq!(get("manifest"), manifest, "manifest embedded verbatim");
    let Value::Array(sections) = get("sections") else {
        panic!("sections is an array")
    };
    assert_eq!(sections.len(), 1);

    // Malformed manifests are refused, not misread.
    assert!(decode_manifest_sections(b"not-cbor").is_err());
    assert!(
        decode_manifest_sections(&[0xa0]).is_err(),
        "empty map: no sections"
    );
}

#[test]
fn signed_frame_carries_the_full_scope_tuple_and_verifies() {
    // §12.1: [envelope, payload, sig]; the signature over the canonical envelope; every scope
    // field host-built. Verify with the plain proto primitives a third party would use.
    let signing = SigningKey::from_bytes(&[9u8; 32]);
    let sender = peer_id(&signing).0;
    let host = Host {
        shared: Arc::new(PumpShared {
            state: Mutex::new(test_state(Box::new(MemorySink::new()))),
            wake: Condvar::new(),
            t0: Instant::now(),
            hold: AtomicBool::new(false),
        }),
        limits: StoreLimitsBuilder::new().build(),
        trap: None,
        slice: SliceState {
            in_init: false,
            in_migrate: false,
            stopped: false,
            draining: false,
            now: 0,
            op_calls: 0,
            readback_bytes: 0,
            pending_next: None,
            pending_readback: None,
            pending_readback_value: None,
            pending_device: None,
        },
        fuel_per_slice: 0,
        op_budget: 0,
        epoch_ticks: 1,
        max_readback_bytes: 0,
        max_frame_bytes: 0,
        hard_accountable_host_bytes: 0,
        accountable_staged_bytes: 0,
        migration_max_sections: 0,
        migration_max_section_bytes: 0,
        migration_restore: false,
        compute: None,
        compute_queue_depth: 0,
        compute_ops_since_fence: 0,
        compute_fault_after_ops: None,
        compute_ops_total: 0,
        signing,
        rng_seed: [0u8; 32],
        device_bytes: Vec::new(),
        granted_artifacts: std::collections::BTreeSet::new(),
        identity: RunIdentity {
            run_id: [1u8; 32],
            epoch: 4,
            role: "trainer".into(),
            instance: 7,
            module: [2u8; 32],
        },
        sender,
    };
    let payload = b"opaque-payload";
    let frame = build_signed_frame(&host, 0, 42, payload).unwrap();

    let v: Value = ciborium::de::from_reader(frame.as_slice()).unwrap();
    let Value::Array(parts) = v else {
        panic!("frame is [envelope, payload, sig]")
    };
    assert_eq!(parts.len(), 3);
    let Value::Map(env) = &parts[0] else {
        panic!("envelope is a map")
    };
    let get = |k: &str| {
        env.iter()
            .find(|(key, _)| matches!(key, Value::Text(t) if t == k))
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| panic!("envelope field {k}"))
    };
    assert_eq!(get("domain"), Value::from(FRAME_ENVELOPE_DOMAIN_V2));
    assert_eq!(get("epoch"), Value::from(4u64));
    assert_eq!(get("instance"), Value::from(7u64));
    assert_eq!(get("channel"), Value::from(0u64));
    assert_eq!(get("seq"), Value::from(42u64));
    assert_eq!(
        get("payload_hash"),
        Value::Bytes(blake3::hash(payload).as_bytes().to_vec())
    );
    // The payload is carried verbatim; the signature verifies over the canonical envelope.
    assert_eq!(parts[1], Value::Bytes(payload.to_vec()));
    let Value::Bytes(sig) = &parts[2] else {
        panic!("sig bytes")
    };
    let env_bytes = to_canonical_vec(&parts[0]).unwrap();
    let sig64: [u8; 64] = sig.as_slice().try_into().unwrap();
    verify_bytes(
        &daemon_vhc_proto::PeerId(sender),
        &daemon_vhc_proto::Signature(sig64),
        &env_bytes,
    )
    .expect("§12.1 signature verifies");
}
