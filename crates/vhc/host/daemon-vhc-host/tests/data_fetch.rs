// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
//! The B2 `data@2` fetch conformance (architecture §3.2 the data world; refactor §6):
//!
//! - **The corpus-window fetch, end to end** (`test-data-v2`, abi 2.1): the GUEST's corpus policy
//!   (`sdk-v2::corpus::Manifest::locate`) decides which byte range of which shard it needs; the
//!   host fetches the whole artifact (the embedder servicer stands in for the resolver + content
//!   cache), verifies it against the **committed hash**, slices the range, and completes
//!   `Ok(BufferHandle)` (tag 6). The published window must equal the range sliced host-side AND
//!   the session corpus pipeline's `sequence()` tokens — tying the fetch path to the corpus
//!   oracle.
//! - **Grants**: a fetch outside the admitted artifact set traps `GrantViolation` typed, before
//!   any op is issued (which artifacts a module may touch is a grant).
//! - **Pinning**: a servicer answering WRONG bytes completes `Err(HashMismatch)` — hosts
//!   fetch-and-verify against the committed hash, never trust the wire. An out-of-bounds range
//!   completes `Err(StoreRefused)` (the sub-resource rule: ranges are slices OF verified
//!   content, so bounds are judged after verification).
//! - **Credentials never enter the sandbox**: the `OpRequest::ArtifactFetch` the embedder
//!   receives carries ONLY `(hash, range)` — asserted in the servicer; the import signature has
//!   no URL/locator/credential input at all.
//! - **Journal + replay**: the fetch completion is journaled (tag 14) and the run re-drives
//!   bit-for-bit with the artifact materialized from the content-addressed payload table
//!   (extended for artifacts per §8.7).
//!
//! Dev/test harness: shells `cargo build` for the guests (the event_loop.rs pattern), so the
//! fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ciborium::value::Value;
use daemon_vhc_host::run::{
    replay, start_run, MemorySink, OpOutcome, OpRequest, PumpHandle, ReplayEnd, ReplayScript,
    RunConfig, RunEnd, RunIdentity, SinkEntry,
};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_session::data::{Corpus, SyntheticCorpus};

fn guest_wasm() -> Vec<u8> {
    daemon_vhc_guest_build::guest_wasm("test_data_v2")
}

// -- the corpus fixture (the artifact map's committed content) --------------------------------------

/// Two shards × 36 tokens, seq_len 9 (u16): the corpus whose shards are the committed artifacts.
fn fixture() -> (daemon_vhc_session::data::Manifest, Vec<Vec<u8>>) {
    let (manifest, blobs) = SyntheticCorpus::generate(0xB2DA, 2, 36, 9).expect("synthetic");
    (manifest, blobs.into_iter().map(|(_, b)| b).collect())
}

fn blake3_of(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

/// Guest config: `{"mode", "manifest", "shard", "seq"}` (canonical CBOR).
fn guest_config(mode: u64, manifest_json: &str, shard: [u8; 32], seq: u64) -> Vec<u8> {
    let v = Value::Map(vec![
        (Value::from("mode"), Value::from(mode)),
        (Value::from("manifest"), Value::from(manifest_json)),
        (Value::from("shard"), Value::Bytes(shard.to_vec())),
        (Value::from("seq"), Value::from(seq)),
    ]);
    let mut b = Vec::new();
    ciborium::into_writer(&v, &mut b).expect("config cbor");
    b
}

/// Guest config for the chunked modes: `{"mode", "shard" (the fold), "desc", "off", "len"}`.
fn chunked_config(mode: u64, shard: [u8; 32], desc: &[u8], off: u64, len: u64) -> Vec<u8> {
    let v = Value::Map(vec![
        (Value::from("mode"), Value::from(mode)),
        (Value::from("shard"), Value::Bytes(shard.to_vec())),
        (Value::from("desc"), Value::Bytes(desc.to_vec())),
        (Value::from("off"), Value::from(off)),
        (Value::from("len"), Value::from(len)),
    ]);
    let mut b = Vec::new();
    ciborium::into_writer(&v, &mut b).expect("config cbor");
    b
}

/// A chunk-addressed shard fixture: 80 deterministic bytes at chunk_size 32 (two full chunks +
/// one short), its fold identity, and the guest-facing register_chunks descriptor.
fn chunked_fixture() -> (daemon_vhc_proto::ChunkMap, Vec<u8>, [u8; 32], Vec<u8>) {
    let bytes: Vec<u8> = (0u8..80).map(|b| b.wrapping_mul(37)).collect();
    let map = daemon_vhc_proto::ChunkMap {
        chunk_size: 32,
        token_count: 40,
        byte_len: 80,
        chunk_hashes: daemon_vhc_proto::chunk_hashes(&bytes, 32),
    };
    let fold = map.fold().0;
    let hashes: Vec<Value> = map
        .chunk_hashes
        .iter()
        .map(|h| Value::Bytes(h.0.to_vec()))
        .collect();
    let doc = Value::Array(vec![
        Value::from(map.chunk_size),
        Value::from(map.token_count),
        Value::from(map.byte_len),
        Value::Array(hashes),
    ]);
    let desc = daemon_vhc_proto::to_canonical_vec(&doc).expect("descriptor cbor");
    (map, bytes, fold, desc)
}

/// The embedder servicer — the resolver/content-cache seat. `tamper`: hashes to answer with
/// corrupted bytes (the pinning negative). Asserts the request carries ONLY content
/// coordinates (hash + range/span): no URL, locator, or credential can reach — or leave — the
/// sandbox through this surface. Chunk-addressed requests (`ArtifactRange`) are served with
/// ONLY the covering span (recorded into `spans_served` so tests can assert the whole shard
/// never crossed).
fn service_fetches(
    pump: &PumpHandle,
    artifacts: &std::collections::HashMap<[u8; 32], Vec<u8>>,
    tamper: &[[u8; 32]],
) {
    service_fetches_spans(pump, artifacts, tamper, &mut Vec::new());
}

/// [`service_fetches`], recording every served covering span as `(span_off, span_len)`.
fn service_fetches_spans(
    pump: &PumpHandle,
    artifacts: &std::collections::HashMap<[u8; 32], Vec<u8>>,
    tamper: &[[u8; 32]],
    spans_served: &mut Vec<(u64, u64)>,
) {
    for (op, request) in pump.take_op_requests() {
        match request {
            OpRequest::ArtifactFetch {
                hash,
                range_off: _,
                range_len: _,
            } => {
                let artifact = artifacts
                    .get(&hash)
                    .unwrap_or_else(|| panic!("servicer asked for an unknown artifact"))
                    .clone();
                let artifact = if tamper.contains(&hash) {
                    vec![0xFF; artifact.len()] // the wire lied — the pump must catch it
                } else {
                    artifact
                };
                pump.complete_op(op, OpOutcome::FetchDone { artifact })
                    .expect("fetch completion");
            }
            OpRequest::ArtifactRange {
                hash,
                span_off,
                span_len,
                ..
            } => {
                let artifact = artifacts
                    .get(&hash)
                    .unwrap_or_else(|| panic!("servicer asked for an unknown shard"));
                let lo = span_off as usize;
                let hi = lo + span_len as usize;
                let mut bytes = artifact[lo..hi].to_vec();
                if tamper.contains(&hash) {
                    bytes[0] ^= 0xFF; // one lying byte in one covering chunk
                }
                spans_served.push((span_off, span_len));
                pump.complete_op(op, OpOutcome::RangeDone { bytes })
                    .expect("range completion");
            }
            other => panic!("the data guest issues only artifact requests, got {other:?}"),
        }
    }
}

/// Run the guest to `publishes` published frames, servicing fetches; returns (published, entries, end).
#[allow(clippy::type_complexity)]
fn drive(
    config: Vec<u8>,
    granted: Vec<[u8; 32]>,
    artifacts: std::collections::HashMap<[u8; 32], Vec<u8>>,
    tamper: &[[u8; 32]],
    publishes: usize,
    instance: u64,
) -> (Vec<(u64, u64, Vec<u8>)>, Vec<SinkEntry>, RunEnd) {
    let wasm = guest_wasm();
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let sel = select_driver(&worker, &wasm, Some(blake3::hash(&wasm).as_bytes()))
        .expect("minor-2 data module admitted");
    assert_eq!((sel.major, sel.minor), (2, 2));

    let identity = RunIdentity {
        run_id: [0xDA; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let mut run_cfg = RunConfig::new(identity, [0x77; 32], config, Vec::new());
    run_cfg.granted_artifacts = granted.into_iter().collect();
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, &wasm, run_cfg, Box::new(sink.clone())).expect("start");
    let pump = run.pump.clone();

    let deadline = Instant::now() + Duration::from_secs(30);
    while pump.published().len() < publishes {
        service_fetches(&pump, &artifacts, tamper);
        // A trapped run will never publish; bail out to wait() below.
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let published = pump.published();
    if published.len() >= publishes {
        pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
            .expect("stop");
    }
    let end = run.wait().expect("guest thread clean");
    let entries = sink.lock().expect("sink").entries.clone();
    (published, entries, end)
}

/// [`drive`] with a data-read budget, recording the served covering spans.
#[allow(clippy::type_complexity)]
fn drive_chunked(
    config: Vec<u8>,
    granted: Vec<[u8; 32]>,
    artifacts: std::collections::HashMap<[u8; 32], Vec<u8>>,
    tamper: &[[u8; 32]],
    publishes: usize,
    instance: u64,
    budget: u64,
) -> (
    Vec<(u64, u64, Vec<u8>)>,
    Vec<SinkEntry>,
    RunEnd,
    Vec<(u64, u64)>,
) {
    let wasm = guest_wasm();
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let identity = RunIdentity {
        run_id: [0xDB; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let mut run_cfg = RunConfig::new(identity, [0x77; 32], config, Vec::new());
    run_cfg.granted_artifacts = granted.into_iter().collect();
    run_cfg.data_read_budget_bytes = budget;
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, &wasm, run_cfg, Box::new(sink.clone())).expect("start");
    let pump = run.pump.clone();

    let mut spans = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    while pump.published().len() < publishes {
        service_fetches_spans(&pump, &artifacts, tamper, &mut spans);
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let published = pump.published();
    if published.len() >= publishes {
        pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
            .expect("stop");
    }
    let end = run.wait().expect("guest thread clean");
    let entries = sink.lock().expect("sink").entries.clone();
    (published, entries, end, spans)
}

/// The payload bytes of published frame `i` (the §12.1 `[envelope, payload, sig]` wire form).
fn published_payload(published: &[(u64, u64, Vec<u8>)], i: usize) -> Vec<u8> {
    let v: Value = ciborium::de::from_reader(published[i].2.as_slice()).expect("frame cbor");
    let Value::Array(parts) = v else {
        panic!("frame shape")
    };
    let Value::Bytes(payload) = &parts[1] else {
        panic!("payload shape")
    };
    payload.clone()
}

/// Mode 0: policy picks the window, mechanism fetches + verifies + slices, the module feeds the
/// batch — and the whole run replays bit-for-bit from the journal + the artifact table.
#[test]
fn corpus_window_fetch_end_to_end_and_replays() {
    let (manifest, blobs) = fixture();
    let json = manifest.to_json().expect("manifest json");
    // Sequence 1 lives in shard 0 at token_offset 9 (36 tokens/shard ÷ seq_len 9 = 4 seqs/shard).
    let shard0 = blake3_of(&blobs[0]);
    let seq: u64 = 1;
    let expected_window = blobs[0][9 * 2..18 * 2].to_vec();
    // The corpus-pipeline oracle: the same window as session tokens, re-encoded LE u16.
    let corpus = Corpus::from_parts(manifest.clone(), blobs.clone()).expect("corpus");
    let expected_tokens: Vec<u8> = corpus
        .sequence(seq)
        .expect("sequence")
        .into_iter()
        .flat_map(|t| u16::try_from(t).expect("u16 corpus").to_le_bytes())
        .collect();
    assert_eq!(expected_window, expected_tokens, "fixture self-consistency");

    let artifacts: std::collections::HashMap<[u8; 32], Vec<u8>> =
        [(shard0, blobs[0].clone())].into_iter().collect();
    let config = guest_config(0, &json, shard0, seq);
    let (published, entries, end) =
        drive(config.clone(), vec![shard0], artifacts.clone(), &[], 1, 1);
    assert!(matches!(end, RunEnd::Outcome(0)), "end: {end:?}");
    assert_eq!(
        published_payload(&published, 0),
        expected_window,
        "the module fed exactly the window its policy located"
    );

    // The fetch completion is journaled (tag 14), Ok-variant.
    let completions: Vec<&SinkEntry> = entries
        .iter()
        .filter(|e| matches!(e, SinkEntry::Completion { .. }))
        .collect();
    assert_eq!(completions.len(), 1, "one fetch, one tag-14 record");

    // Replay: the artifact materializes from the content-addressed table (extended for
    // artifacts); every decision reproduces bit-for-bit.
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let mut script = ReplayScript::from_entries(&entries);
    script.payloads.insert(shard0, blobs[0].clone());
    let replayed = replay(&worker, &guest_wasm(), &config, &[], script).expect("harness");
    assert_eq!(replayed.end, ReplayEnd::Outcome(0));
    let recorded: Vec<(u64, u64, [u8; 32])> = entries
        .iter()
        .filter_map(|e| match e {
            SinkEntry::Publish {
                channel,
                seq,
                payload_hash,
                ..
            } => Some((*channel, *seq, *payload_hash)),
            _ => None,
        })
        .collect();
    let redriven: Vec<(u64, u64, [u8; 32])> = replayed
        .decisions
        .iter()
        .map(|d| (d.channel, d.seq, d.payload_hash))
        .collect();
    assert_eq!(recorded, redriven, "decisions reproduce bit-for-bit");

    // And a replay WITHOUT the artifact table entry is the typed missing-payload divergence.
    let script_missing = ReplayScript::from_entries(&entries);
    let replayed = replay(&worker, &guest_wasm(), &config, &[], script_missing).expect("h");
    match replayed.end {
        ReplayEnd::Diverged(msg) => assert!(msg.contains("ReplayMissingPayload"), "{msg}"),
        other => panic!("expected ReplayMissingPayload, got {other:?}"),
    }
}

/// Mode 1: which artifacts a module may touch is a GRANT — an ungranted hash is a typed
/// `GrantViolation` trap before any op is issued (and no tag-14 record exists).
#[test]
fn ungranted_artifact_fetch_traps_grant_violation() {
    let (manifest, blobs) = fixture();
    let json = manifest.to_json().expect("json");
    let shard0 = blake3_of(&blobs[0]);
    // The guest (mode 1) fetches [0xAB; 32], which is NOT granted (only shard0 is).
    let config = guest_config(1, &json, shard0, 0);
    let (_, entries, end) = drive(
        config,
        vec![shard0],
        std::collections::HashMap::new(),
        &[],
        1,
        2,
    );
    match end {
        RunEnd::Trapped(trap) => {
            assert_eq!(trap.code, daemon_vhc_host::TrapCode::GrantViolation);
            assert!(trap.detail.contains("artifact"), "{}", trap.detail);
        }
        other => panic!("expected GrantViolation, got {other:?}"),
    }
    assert!(
        !entries
            .iter()
            .any(|e| matches!(e, SinkEntry::Completion { .. })),
        "no op was issued, no completion journaled"
    );
}

/// Mode 2 + 3: the pinning semantics as completions — an out-of-bounds range is
/// `Err(StoreRefused)` (3); tampered wire bytes are caught by the pump's whole-artifact
/// verification as `Err(HashMismatch)` (4). The guest publishes the code byte it observed.
#[test]
fn range_bounds_and_tamper_complete_typed_errors() {
    let (manifest, blobs) = fixture();
    let json = manifest.to_json().expect("json");
    let shard0 = blake3_of(&blobs[0]);
    let artifacts: std::collections::HashMap<[u8; 32], Vec<u8>> =
        [(shard0, blobs[0].clone())].into_iter().collect();

    // Mode 2: absurd range_off → StoreRefused (3).
    let (published, _, end) = drive(
        guest_config(2, &json, shard0, 0),
        vec![shard0],
        artifacts.clone(),
        &[],
        1,
        3,
    );
    assert!(matches!(end, RunEnd::Outcome(0)));
    assert_eq!(
        published_payload(&published, 0),
        vec![u8::try_from(daemon_vhc_abi::COMP_ERR_STORE_REFUSED).unwrap()],
        "out-of-bounds range completes StoreRefused"
    );

    // Mode 3: the servicer tampers → the pump completes HashMismatch (4); the guest NEVER sees
    // the corrupted bytes (fetch-and-verify against the committed hash).
    let (published, _, end) = drive(
        guest_config(3, &json, shard0, 0),
        vec![shard0],
        artifacts,
        &[shard0],
        1,
        4,
    );
    assert!(matches!(end, RunEnd::Outcome(0)));
    assert_eq!(
        published_payload(&published, 0),
        vec![u8::try_from(daemon_vhc_abi::COMP_ERR_HASH_MISMATCH).unwrap()],
        "tampered artifact completes HashMismatch"
    );
}

// ---- the chunk-addressed corpus contract (register_chunks + covering-span range fetch) ----------

/// Mode 4: register → range-fetch → the embedder moves ONLY the covering span, the pump
/// verifies the covering chunks and slices the exact range — and the run replays bit-for-bit
/// from CHUNK-keyed payload-table entries (no whole-shard object exists anywhere).
#[test]
fn chunked_range_fetch_end_to_end_and_replays() {
    let (map, bytes, fold, desc) = chunked_fixture();
    // The guest asks for [40, 60): chunk 1 ([32, 64)) covers it entirely.
    let config = chunked_config(4, fold, &desc, 40, 20);
    let artifacts: std::collections::HashMap<[u8; 32], Vec<u8>> =
        [(fold, bytes.clone())].into_iter().collect();
    let (published, entries, end, spans) =
        drive_chunked(config.clone(), vec![fold], artifacts, &[], 1, 1, 0);
    assert!(matches!(end, RunEnd::Outcome(0)), "end: {end:?}");
    assert_eq!(
        published_payload(&published, 0),
        bytes[40..60].to_vec(),
        "the module fed exactly the window its policy located"
    );
    assert_eq!(
        spans,
        vec![(32, 32)],
        "the embedder served ONLY the covering chunk, never the whole shard"
    );

    // Replay: the artifact table is CHUNK-addressed — each chunk under its own blake3.
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let mut script = ReplayScript::from_entries(&entries);
    for (i, chunk) in bytes.chunks(32).enumerate() {
        script
            .payloads
            .insert(map.chunk_hashes[i].0, chunk.to_vec());
    }
    let replayed = replay(&worker, &guest_wasm(), &config, &[], script).expect("harness");
    assert_eq!(replayed.end, ReplayEnd::Outcome(0));
    let recorded: Vec<(u64, u64, [u8; 32])> = entries
        .iter()
        .filter_map(|e| match e {
            SinkEntry::Publish {
                channel,
                seq,
                payload_hash,
                ..
            } => Some((*channel, *seq, *payload_hash)),
            _ => None,
        })
        .collect();
    let redriven: Vec<(u64, u64, [u8; 32])> = replayed
        .decisions
        .iter()
        .map(|d| (d.channel, d.seq, d.payload_hash))
        .collect();
    assert_eq!(recorded, redriven, "decisions reproduce bit-for-bit");

    // A replay whose payload table lacks the covering chunk is the typed missing-payload
    // divergence — never a silent pass.
    let script_missing = ReplayScript::from_entries(&entries);
    let replayed = replay(&worker, &guest_wasm(), &config, &[], script_missing).expect("h");
    match replayed.end {
        ReplayEnd::Diverged(msg) => assert!(msg.contains("ReplayMissingPayload"), "{msg}"),
        other => panic!("expected ReplayMissingPayload, got {other:?}"),
    }
}

/// Mode 5: one lying byte in a covering chunk → the pump completes `Err(HashMismatch)`; the
/// guest never sees the corrupted window.
#[test]
fn chunked_tampered_span_completes_hash_mismatch() {
    let (_map, bytes, fold, desc) = chunked_fixture();
    let artifacts: std::collections::HashMap<[u8; 32], Vec<u8>> =
        [(fold, bytes)].into_iter().collect();
    let (published, _, end, _) = drive_chunked(
        chunked_config(5, fold, &desc, 0, 16),
        vec![fold],
        artifacts,
        &[fold],
        1,
        2,
        0,
    );
    assert!(matches!(end, RunEnd::Outcome(0)));
    assert_eq!(
        published_payload(&published, 0),
        vec![u8::try_from(daemon_vhc_abi::COMP_ERR_HASH_MISMATCH).unwrap()],
        "a lying covering chunk completes HashMismatch"
    );
}

/// Mode 6: the cumulative data-read budget covers exactly one window — the first fetch feeds
/// the batch, the identical second fetch completes `Err(GrantExhausted)` (typed, at the call).
#[test]
fn chunked_read_budget_exhaustion_is_typed() {
    let (_map, bytes, fold, desc) = chunked_fixture();
    let artifacts: std::collections::HashMap<[u8; 32], Vec<u8>> =
        [(fold, bytes.clone())].into_iter().collect();
    let (published, _, end, _) = drive_chunked(
        chunked_config(6, fold, &desc, 8, 24),
        vec![fold],
        artifacts,
        &[],
        2,
        3,
        24, // exactly one 24-byte window
    );
    assert!(matches!(end, RunEnd::Outcome(0)));
    assert_eq!(
        published_payload(&published, 0),
        bytes[8..32].to_vec(),
        "the budgeted first window feeds"
    );
    assert_eq!(
        published_payload(&published, 1),
        vec![u8::try_from(daemon_vhc_abi::COMP_ERR_GRANT_EXHAUSTED).unwrap()],
        "the second identical fetch breaches the cumulative budget"
    );
}

/// Mode 7: bounds are knowable at the call on a REGISTERED shard — an absurd offset completes
/// `Err(StoreRefused)` immediately (no embedder round-trip happens at all).
#[test]
fn chunked_out_of_bounds_range_refuses_at_the_call() {
    let (_map, bytes, fold, desc) = chunked_fixture();
    let artifacts: std::collections::HashMap<[u8; 32], Vec<u8>> =
        [(fold, bytes)].into_iter().collect();
    let (published, _, end, spans) = drive_chunked(
        chunked_config(7, fold, &desc, 0, 0),
        vec![fold],
        artifacts,
        &[],
        1,
        4,
        0,
    );
    assert!(matches!(end, RunEnd::Outcome(0)));
    assert_eq!(
        published_payload(&published, 0),
        vec![u8::try_from(daemon_vhc_abi::COMP_ERR_STORE_REFUSED).unwrap()],
        "registered bounds refuse at the call"
    );
    assert!(spans.is_empty(), "no store round-trip for a refused range");
}

/// Mode 8: registering a chunk map whose fold is NOT a granted artifact traps `GrantViolation`
/// at the call — a module cannot smuggle chunk identities for content it was not granted.
#[test]
fn ungranted_chunk_registration_traps_grant_violation() {
    let (_map, _bytes, fold, desc) = chunked_fixture();
    // Grant something else entirely; the descriptor's fold is not in the set.
    let other = [0x55u8; 32];
    let (_, entries, end, _) = drive_chunked(
        chunked_config(8, fold, &desc, 0, 0),
        vec![other],
        std::collections::HashMap::new(),
        &[],
        1,
        5,
        0,
    );
    match end {
        RunEnd::Trapped(trap) => {
            assert_eq!(trap.code, daemon_vhc_host::TrapCode::GrantViolation);
            assert!(trap.detail.contains("fold"), "{}", trap.detail);
        }
        other => panic!("expected GrantViolation, got {other:?}"),
    }
    assert!(
        !entries
            .iter()
            .any(|e| matches!(e, SinkEntry::Completion { .. })),
        "no op was issued, no completion journaled"
    );
}
