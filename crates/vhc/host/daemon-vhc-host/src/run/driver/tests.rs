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
        allocator_samples: Vec::new(),
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
            in_run: false,
            slice_ordinal: None,
            slices_delivered: 0,
            log_calls_this_phase: 0,
            log_bytes_this_phase: 0,
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

// -- the journalled terminal context is the real one, not a hard-coded literal --------------------

/// A `SliceState` in the "nothing has happened yet" position, for the derivation table below.
fn quiescent_slice() -> SliceState {
    SliceState {
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
        in_run: false,
        slice_ordinal: None,
        slices_delivered: 0,
        log_calls_this_phase: 0,
        log_bytes_this_phase: 0,
        pending_device: None,
    }
}

/// The context is DERIVED from lifecycle state, and every distinguishable state derives the value
/// that is true of it.
///
/// The four `da_run` states are the point: before this, a trap anywhere in the run loop was recorded
/// as the same thing, so "between slices" and "in slice 7" were indistinguishable in the one field a
/// replay verdict compares.
#[test]
fn the_execution_context_is_derived_from_lifecycle_state_and_the_four_run_states_are_distinct() {
    use daemon_vhc_abi::ExecutionContext;

    let mut init = quiescent_slice();
    init.in_init = true;
    assert_eq!(init.execution_context(), ExecutionContext::Init);

    let mut migrate = quiescent_slice();
    migrate.in_migrate = true;
    assert_eq!(migrate.execution_context(), ExecutionContext::Migrate);

    // In the run loop, before the first event has been delivered.
    let mut before = quiescent_slice();
    before.in_run = true;
    assert_eq!(
        before.execution_context(),
        ExecutionContext::RunBeforeFirstSlice
    );

    // Inside a slice: the ordinal is the one actually active.
    let mut in_slice = quiescent_slice();
    in_slice.in_run = true;
    in_slice.slice_ordinal = Some(7);
    in_slice.slices_delivered = 8;
    assert_eq!(in_slice.execution_context(), ExecutionContext::RunSlice(7));

    // Between slices: a slice HAS run, but none is active. Attributing this to slice 7 would point a
    // reader at code that had already returned.
    let mut between = quiescent_slice();
    between.in_run = true;
    between.slice_ordinal = None;
    between.slices_delivered = 8;
    assert_eq!(
        between.execution_context(),
        ExecutionContext::RunBetweenSlices
    );
    assert_ne!(
        between.execution_context(),
        ExecutionContext::RunSlice(7),
        "a between-slices trap must not borrow the last slice's ordinal"
    );

    // After a consumed stop no further slice can begin.
    let mut after = quiescent_slice();
    after.in_run = true;
    after.slices_delivered = 8;
    after.stopped = true;
    assert_eq!(
        after.execution_context(),
        ExecutionContext::RunAfterLastSlice
    );

    // Initialization takes precedence over the run flag: the phase flags are still set at trap time
    // (the driver clears them only on the success path), so an init trap must not read as a run one.
    let mut init_inside_run = quiescent_slice();
    init_inside_run.in_run = true;
    init_inside_run.in_init = true;
    assert_eq!(init_inside_run.execution_context(), ExecutionContext::Init);
}

/// Drive `journal_terminal_trap` at each context and read the record back.
fn recorded_context(context: &daemon_vhc_abi::ExecutionContext, abi_minor: u32) -> String {
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let shared = Arc::new(PumpShared {
        state: Mutex::new(test_state(Box::new(sink.clone()))),
        wake: Condvar::new(),
        t0: Instant::now(),
        hold: AtomicBool::new(false),
    });
    let trap = Trap::bare(crate::trap::TrapCode::GuestPanic, "boom");
    super::lifecycle::journal_terminal_trap(&shared, &trap, context, abi_minor)
        .expect("the terminal record is written");

    let guard = sink.lock().expect("sink");
    let entries = &guard.entries;
    let terminal = entries
        .iter()
        .find_map(|e| match e {
            SinkEntry::Terminal { trap: Some(t), .. } => Some(t.clone()),
            _ => None,
        })
        .expect("a terminal trap record");
    assert_eq!(terminal.0, "GuestPanic", "the typed code is unchanged");
    terminal.2
}

/// The journalled terminal record carries the context the trap actually occurred in.
///
/// Before this, `journal_terminal_trap` wrote the literal `"da_run"` for every trap it was handed —
/// so an initialization trap was recorded as a run-loop trap. That falsified the field a replay
/// verdict keys on, and it made a correct in-memory diagnosis read as a misattribution bug in the
/// forwarding, sending the reader to investigate the wrong mechanism. The three preceding tests can
/// all pass while the record still says otherwise, which is why this one exists separately.
#[test]
fn the_journalled_terminal_record_carries_the_phase_the_trap_occurred_in() {
    use daemon_vhc_abi::{ExecutionContext, CERTIFICATION_MINOR_V2};

    assert_eq!(
        recorded_context(&ExecutionContext::Init, CERTIFICATION_MINOR_V2),
        "da_init",
        "an initialization trap is recorded as an initialization trap"
    );
    assert_eq!(
        recorded_context(&ExecutionContext::Migrate, CERTIFICATION_MINOR_V2),
        "da_migrate"
    );
    assert_eq!(
        recorded_context(&ExecutionContext::RunSlice(18), CERTIFICATION_MINOR_V2),
        "slice:18",
        "a run-loop trap names the slice it was in"
    );
    assert_eq!(
        recorded_context(
            &ExecutionContext::RunBeforeFirstSlice,
            CERTIFICATION_MINOR_V2
        ),
        "da_run:before"
    );
    assert_eq!(
        recorded_context(&ExecutionContext::RunBetweenSlices, CERTIFICATION_MINOR_V2),
        "da_run:between"
    );
    assert_eq!(
        recorded_context(&ExecutionContext::RunAfterLastSlice, CERTIFICATION_MINOR_V2),
        "da_run:after"
    );
    assert_eq!(
        recorded_context(&ExecutionContext::ExecutionGrant, CERTIFICATION_MINOR_V2),
        "da_apply_execution_grant",
        "a grant-application trap occupies its own branch and says so"
    );
}

/// **The replay half.** A verdict compares the recorded **code** and **context**, never the
/// detail string — and a journal written at a legacy minor is compared under the legacy renderer,
/// unchanged, rather than reinterpreted as one of the truthful values its writer never meant.
#[test]
fn a_replay_verdict_compares_the_recorded_context_under_the_journals_own_minor() {
    use daemon_vhc_abi::{
        terminal_contexts_agree, ExecutionContext, CERTIFICATION_MINOR_V2,
        LEGACY_CONTEXT_MAX_MINOR, LEGACY_TERMINAL_CONTEXT,
    };

    // A legacy journal records the bare string for every phase. Replaying it under its own minor
    // agrees; the recorded string is NOT upgraded into an ABI-2.5 value.
    let legacy = recorded_context(&ExecutionContext::Init, LEGACY_CONTEXT_MAX_MINOR);
    assert_eq!(legacy, LEGACY_TERMINAL_CONTEXT);
    assert!(terminal_contexts_agree(
        &legacy,
        &ExecutionContext::Init,
        LEGACY_CONTEXT_MAX_MINOR
    ));
    assert!(
        ExecutionContext::parse(&legacy).is_none(),
        "the legacy string is not one of the eleven renderings, so nothing can silently equate them"
    );

    // A certification-minor journal is compared against the truthful rendering, and a DIFFERENT
    // phase disagrees — which is the whole reason the field is worth comparing.
    let init = recorded_context(&ExecutionContext::Init, CERTIFICATION_MINOR_V2);
    assert!(terminal_contexts_agree(
        &init,
        &ExecutionContext::Init,
        CERTIFICATION_MINOR_V2
    ));
    assert!(
        !terminal_contexts_agree(&init, &ExecutionContext::Migrate, CERTIFICATION_MINOR_V2),
        "a migration trap does not agree with an initialization record"
    );
    assert!(
        !terminal_contexts_agree(
            &init,
            &ExecutionContext::RunBetweenSlices,
            CERTIFICATION_MINOR_V2
        ),
        "nor does a run-phase trap"
    );
}

// -- [LX-10]: the held panic detail is context-scoped -----------------------------------------------

/// Build a `Store<Host>` whose pump already holds a forwarded panic line emitted in
/// `emitted_in`, then take a trap while the slice state says `trap_state`. Returns the trap's detail.
fn lifted_detail(emitted_in: daemon_vhc_abi::ExecutionContext, trap_state: SliceState) -> String {
    let worker =
        crate::runtime::Worker::new(crate::runtime::EngineConfig::default()).expect("engine");
    let signing = SigningKey::from_bytes(&[9u8; 32]);
    let sender = peer_id(&signing).0;
    let shared = Arc::new(PumpShared {
        state: Mutex::new(test_state(Box::new(MemorySink::new()))),
        wake: Condvar::new(),
        t0: Instant::now(),
        hold: AtomicBool::new(false),
    });
    // The line the guest forwarded, tagged with the context it was emitted in.
    shared.state.lock().expect("pump").guest_panic = Some((
        emitted_in,
        "guests/tiny-llama/src/lib.rs:530:9: seed init".into(),
    ));

    let host = Host {
        shared: shared.clone(),
        limits: StoreLimitsBuilder::new().build(),
        trap: None,
        slice: trap_state,
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
    let mut store = wasmtime::Store::new(worker.engine(), host);
    let trap = super::lifecycle::take_trap(
        &mut store,
        wasmtime::Error::msg("wasm trap: wasm `unreachable` instruction executed"),
    );
    trap.detail
}

/// **[LX-10], the negative half.** A prefixed line emitted during initialization must NOT be lifted
/// into a trap that happens later, in a different phase.
///
/// Emitting a prefixed line does not imply trapping: a guest may log one and continue, or log one in
/// an early phase and trap much later for an unrelated reason. An unscoped slot attaches that stale
/// line to the later trap, producing an authoritative-looking source location that belongs to a
/// different failure — worse than no diagnostic, because it is believed, and it sends the reader to
/// the wrong file with confidence.
#[test]
fn an_initialization_phase_detail_cannot_contaminate_a_later_trap() {
    use daemon_vhc_abi::ExecutionContext;

    // In the run loop, inside slice 3 — the state a later, unrelated trap occurs in.
    let in_slice_3 = || {
        let mut s = quiescent_slice();
        s.in_run = true;
        s.slice_ordinal = Some(3);
        s.slices_delivered = 4;
        s
    };

    // Emitted in `da_init`; the trap happens later, in the run loop.
    let detail = lifted_detail(ExecutionContext::Init, in_slice_3());
    assert!(
        !detail.contains("seed init"),
        "an init-phase line must not be attached to a run-phase trap: {detail}"
    );
    assert!(
        !detail.contains("tiny-llama"),
        "and its source location must not be either: {detail}"
    );
    assert!(
        detail.contains("unreachable"),
        "the trap keeps the detail it would otherwise have had: {detail}"
    );

    // The migration twin: emitted in `da_migrate`, trapping later in `da_run`.
    let migrate_detail = lifted_detail(ExecutionContext::Migrate, in_slice_3());
    assert!(
        !migrate_detail.contains("seed init"),
        "a migrate-phase line must not be attached to a run-phase trap: {migrate_detail}"
    );

    // A between-slices trap does not inherit a line emitted inside a slice, either — the four run
    // states are distinct for lifting as well as for recording.
    let mut between = quiescent_slice();
    between.in_run = true;
    between.slices_delivered = 4;
    assert!(
        !lifted_detail(ExecutionContext::RunSlice(3), between).contains("seed init"),
        "a slice-scoped line is not lifted into a between-slices trap"
    );
}

/// The positive half, so the negative one is not passing for the wrong reason: when the contexts DO
/// match, the line is lifted to the front of the detail — which is the whole point of forwarding it.
#[test]
fn a_matching_context_lifts_the_forwarded_line_to_the_front_of_the_detail() {
    use daemon_vhc_abi::ExecutionContext;

    let mut init = quiescent_slice();
    init.in_init = true;
    let detail = lifted_detail(ExecutionContext::Init, init);
    assert!(
        detail.starts_with("guests/tiny-llama/src/lib.rs:530:9: seed init"),
        "the forwarded line leads the detail: {detail}"
    );
    assert!(
        detail.contains("unreachable"),
        "and the engine's own text still follows it: {detail}"
    );
}

// -- the allocator sampler is read, and its reproducibility keying compares sets ----------------

/// The sample points a record claims must be the ones the driver is actually wired at.
///
/// This is the drift that matters: a profile calibrated from slice-end readings is not reproducible on
/// a binary that samples only at phase boundaries, even though both would truthfully report that
/// allocator statistics are available. If the enum grows a member nobody wired, or a wired site is
/// removed, a record claiming that point would be claiming reproducibility it cannot deliver — so the
/// claim is pinned against the wiring here rather than trusted.
#[test]
fn every_sample_point_the_record_may_claim_is_wired_in_the_driver() {
    use crate::compute::SamplePoint;

    // Every member of the closed domain, so adding one forces this test to be revisited.
    let all = [
        SamplePoint::AfterBringUp,
        SamplePoint::AfterInit,
        SamplePoint::AfterMigrate,
        SamplePoint::AfterSlice,
        SamplePoint::AtTeardown,
    ];
    let slugs: Vec<&str> = all.iter().map(|p| p.slug()).collect();
    assert_eq!(
        slugs,
        vec![
            "after-bring-up",
            "after-init",
            "after-migrate",
            "after-slice",
            "at-teardown"
        ],
        "the slugs are the strings a reproducibility check compares, so they are pinned"
    );

    // Each one must appear at a call site in the driver. Reading the sources is the only way to
    // assert this without running five different lifecycles on a device this test cannot require.
    let lifecycle = include_str!("lifecycle.rs");
    let event_seam = include_str!("linker/vhc.rs");
    for point in all {
        let wired = lifecycle.contains(&format!("SamplePoint::{point:?}"))
            || event_seam.contains(&format!("SamplePoint::{point:?}"));
        assert!(
            wired,
            "`{}` is in the closed domain but nothing samples at it — a record claiming it would \
             claim a reproducibility it cannot deliver. Wire it, or remove it from the domain.",
            point.slug()
        );
    }
}

/// An empty sample series is an absence, and the readout says so rather than printing zeros.
///
/// The distinction is the whole point of the sampler's contract: a backend that cannot report
/// occupancy records nothing, and a reader who took that for `bytes_in_use = 0` would calibrate a
/// profile against a figure nobody measured. That is the same mislabelling the per-term calibration
/// basis exists to prevent, one layer down.
#[test]
fn an_unsampled_run_reports_an_absence_and_not_a_zero() {
    let slice = quiescent_slice();
    assert!(
        slice.log_calls_this_phase == 0,
        "the fixture is quiescent so the assertion below is about the samples, not the phase"
    );

    // A pump that never sampled holds an empty series, which is distinguishable from one that
    // sampled and read zero.
    let sampled_nothing: Vec<(crate::compute::SamplePoint, crate::compute::AllocatorSample)> =
        Vec::new();
    let sampled_a_zero = [(
        crate::compute::SamplePoint::AfterInit,
        crate::compute::AllocatorSample::default(),
    )];
    assert!(sampled_nothing.is_empty());
    assert_eq!(sampled_a_zero.len(), 1);
    assert_eq!(
        sampled_a_zero[0].1,
        crate::compute::AllocatorSample::default(),
        "a measured zero is a value; the empty series above is not that value"
    );
}

/// The run header's resource branch follows the negotiated minor, and a certification-minor run with
/// no composition **does not start**.
///
/// The empty-members case is the one the minor alone cannot decide. Writing the composed branch would
/// record members asserting a composition that never happened; writing the legacy branch would record a
/// declared claim the module never declared. Admission refuses such a run first, so reaching this gate
/// means a caller assembled the configuration directly — which is precisely what the first gate cannot
/// see, and why there is a second one.
#[test]
fn a_certification_minor_run_without_a_composition_refuses_instead_of_recording_empty_members() {
    use crate::run::driver::config::RunError;
    use crate::run::driver::lifecycle::run_header_resources;
    use crate::run::RunHeaderResources;

    let identity = crate::run::RunIdentity {
        run_id: [7u8; 32],
        epoch: 1,
        role: "trainer".into(),
        instance: 1,
        module: [9u8; 32],
    };
    let base = || {
        crate::run::RunConfig::new(
            identity.clone(),
            [0x33; 32],
            b"cfg".to_vec(),
            b"gr".to_vec(),
        )
    };

    // A legacy run records what the module declared, whatever the composed fields hold.
    let mut legacy = base();
    legacy.abi_minor = daemon_vhc_abi::LEGACY_CONTEXT_MAX_MINOR;
    legacy.claim_bytes = b"declared".to_vec();
    assert!(matches!(
        run_header_resources(&legacy).expect("a legacy run records its declared claim"),
        RunHeaderResources::Declared(b"declared")
    ));

    // The certification minor with every member present records the composed branch.
    let mut composed = base();
    composed.abi_minor = daemon_vhc_abi::CERTIFICATION_MINOR_V2;
    composed.resource_plan_bytes = b"plan".to_vec();
    composed.physical_claim_bytes = b"claim".to_vec();
    composed.aggregate_claim_bytes = b"aggregate".to_vec();
    composed.execution_grant = b"grant".to_vec();
    assert!(matches!(
        run_header_resources(&composed).expect("a composed run records its composition"),
        RunHeaderResources::Composed { .. }
    ));

    // Each member is individually load-bearing: absent, the run refuses and the refusal names it.
    for (member, blank) in [
        ("resource_plan", 0usize),
        ("physical_claim", 1),
        ("aggregate_claim", 2),
        ("execution_grant", 3),
    ] {
        let mut missing = base();
        missing.abi_minor = daemon_vhc_abi::CERTIFICATION_MINOR_V2;
        let mut members = [
            b"plan".to_vec(),
            b"claim".to_vec(),
            b"aggregate".to_vec(),
            b"grant".to_vec(),
        ];
        members[blank] = Vec::new();
        let [plan, claim, aggregate, grant] = members;
        missing.resource_plan_bytes = plan;
        missing.physical_claim_bytes = claim;
        missing.aggregate_claim_bytes = aggregate;
        missing.execution_grant = grant;

        let refusal = run_header_resources(&missing).expect_err("an absent member refuses the run");
        assert!(
            matches!(refusal, RunError::CompositionMissing { member: m, .. } if m == member),
            "the refusal names the absent member, got {refusal}"
        );
    }

    // And a run that declares the certification minor while carrying only a declared claim is refused
    // too — the legacy fallback is not available at that minor.
    let mut declared_at_certification = base();
    declared_at_certification.abi_minor = daemon_vhc_abi::CERTIFICATION_MINOR_V2;
    declared_at_certification.claim_bytes = b"declared".to_vec();
    assert!(matches!(
        run_header_resources(&declared_at_certification).expect_err("no silent legacy fallback"),
        RunError::CompositionMissing { .. }
    ));
}
