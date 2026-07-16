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
//! Dev/test harness: shells `cargo build` for the guests (the v2_event_loop.rs pattern), so the
//! fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use ciborium::value::Value;
use daemon_vhc_host::v2::{
    replay_v2, start_run, MemorySink, OpOutcome, OpRequest, PumpHandle, ReplayEnd, ReplayScript,
    RunEnd, RunIdentity, SinkEntry, V2RunConfig,
};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_session::data::{Corpus, SyntheticCorpus};

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_remap_rustflags() -> String {
    let root = guests_root();
    let checkout = root.ancestors().nth(3).unwrap_or(&root).to_path_buf();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cargo"));
    format!(
        "--remap-path-prefix={}=/daemon-node --remap-path-prefix={}=/cargo",
        checkout.display(),
        cargo_home.display(),
    )
}

static BUILD: Once = Once::new();

fn guest_wasm() -> Vec<u8> {
    BUILD.call_once(|| {
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join("target/wasm32-unknown-unknown/release/test_data_v2.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
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

/// The embedder servicer — the resolver/content-cache seat. `tamper`: hashes to answer with
/// corrupted bytes (the pinning negative). Asserts the request carries ONLY (hash, range): no
/// URL, locator, or credential can reach — or leave — the sandbox through this surface.
fn service_fetches(
    pump: &PumpHandle,
    artifacts: &std::collections::HashMap<[u8; 32], Vec<u8>>,
    tamper: &[[u8; 32]],
) {
    for (op, request) in pump.take_op_requests() {
        let OpRequest::ArtifactFetch {
            hash,
            range_off: _,
            range_len: _,
        } = request
        else {
            panic!("the data guest issues only ArtifactFetch requests, got {request:?}");
        };
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
        .expect("minor-1 data module admitted");
    assert_eq!((sel.major, sel.minor), (2, 1));

    let identity = RunIdentity {
        run_id: [0xDA; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let mut run_cfg = V2RunConfig::new(identity, [0x77; 32], config, Vec::new());
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
    let replayed = replay_v2(&worker, &guest_wasm(), &config, &[], script).expect("harness");
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
    let replayed = replay_v2(&worker, &guest_wasm(), &config, &[], script_missing).expect("h");
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
