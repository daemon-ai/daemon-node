// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **The pipeline acceptance toy, graduated to tensor buffers** (Phase C, refactor §7 "the
//! pipeline toy graduates to tensor buffers here"; architecture §9's "SWARM pipeline stage" row):
//! two `pipeline-stage` module instances exchange **exported device tensors over
//! credit-controlled streams** — `compute.export` → `BufferHandle` → `stream_write` on the
//! producer; `stream_read` → `compute.import` → an on-device transform (`t * 2`) → re-export on
//! the consumer — over an in-memory transport (the same `take_op_requests`/`complete_op`/
//! `grant_credit` seam the session will drive).
//!
//! Flow-control pin (unchanged from Phase B): the producer issues ALL its writes at once against
//! a one-chunk credit window, so the pump provably HOLDS the surplus (their transport requests
//! appear only after the consumer's reads replenish credit) — asserted over the transport's
//! event log, which is deterministic because the hold/release ordering is pump-enforced.
//!
//! Dev/test harness: shells `cargo build` for the guests, so fs/process bans are allowed
//! file-wide.
#![allow(clippy::disallowed_methods)]

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use daemon_vhc_host::run::{
    replay, start_run, MemorySink, OpOutcome, OpRequest, PumpHandle, ReplayEnd, ReplayScript,
    RunConfig, RunEnd, RunIdentity, SinkEntry,
};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};

const N_CHUNKS: u8 = 3;
/// f32 elements per chunk tensor (the Phase-B byte recipe, widened to floats).
const CHUNK_LEN: u8 = 16;

const PEER_A: [u8; 32] = [0xAA; 32];
const PEER_B: [u8; 32] = [0xBB; 32];

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

fn pipeline_wasm() -> Vec<u8> {
    BUILD.call_once(|| {
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join("target/wasm32-unknown-unknown/release/pipeline_stage.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Chunk `i`'s f32 elements exactly as the guest derives them.
fn chunk_floats(i: u8) -> Vec<f32> {
    (0..CHUNK_LEN)
        .map(|j| f32::from(i.wrapping_mul(31).wrapping_add(j)))
        .collect()
}

/// The `CBOR(TensorData)` bytes an export of `values` produces (the host runner and the test
/// serialize with the same ciborium writer over the same pinned burn `TensorData`).
fn tensor_bytes(values: Vec<f32>) -> Vec<u8> {
    let data = burn::tensor::TensorData::new(values, [CHUNK_LEN as usize]);
    let mut out = Vec::new();
    ciborium::into_writer(&data, &mut out).expect("TensorData encodes");
    out
}

/// One exported chunk tensor's wire size = one chunk of writable credit: the producer's 3-write
/// burst must hold 2 writes.
fn credit() -> u64 {
    tensor_bytes(chunk_floats(0)).len() as u64
}

fn config(role: u8, peer: [u8; 32]) -> Vec<u8> {
    let mut c = vec![role, N_CHUNKS, CHUNK_LEN];
    c.extend_from_slice(&peer);
    c
}

/// The in-memory two-pump transport: routes opens↔accepts by peer identity, moves written chunks
/// to the destination's read queue, and replenishes the WRITER's credit as the reader consumes —
/// the async-runtime seat of architecture §3.3, minus any real network.
struct MemNet {
    /// (pump idx, local stream) → (peer pump idx, peer stream).
    pairs: HashMap<(usize, u64), (usize, u64)>,
    /// Bytes in flight toward (pump idx, stream), FIFO.
    inflight: HashMap<(usize, u64), VecDeque<Vec<u8>>>,
    /// Reads awaiting bytes on (pump idx, stream), FIFO.
    pending_reads: HashMap<(usize, u64), VecDeque<u64>>,
    /// Accepts standing per pump idx.
    pending_accepts: Vec<VecDeque<u64>>,
    /// Opens awaiting a matching accept: (from idx, op, target idx).
    pending_opens: Vec<(usize, u64, usize)>,
    /// The payload store (stage B's final put).
    store: HashMap<[u8; 32], Vec<u8>>,
    /// The deterministic transport event log (the flow-control pin's substrate).
    log: Vec<String>,
}

impl MemNet {
    fn new(n: usize) -> Self {
        Self {
            pairs: HashMap::new(),
            inflight: HashMap::new(),
            pending_reads: HashMap::new(),
            pending_accepts: vec![VecDeque::new(); n],
            pending_opens: Vec::new(),
            store: HashMap::new(),
            log: Vec::new(),
        }
    }

    /// One service pass over both pumps.
    fn service(&mut self, pumps: &[(PumpHandle, [u8; 32])]) {
        for (idx, (pump, _)) in pumps.iter().enumerate() {
            for (op, request) in pump.take_op_requests() {
                match request {
                    OpRequest::StreamAccept => {
                        self.pending_accepts[idx].push_back(op);
                    }
                    OpRequest::StreamOpen { peer } => {
                        let target = pumps
                            .iter()
                            .position(|(_, id)| *id == peer)
                            .expect("known peer");
                        self.pending_opens.push((idx, op, target));
                    }
                    OpRequest::StreamWrite { stream, bytes } => {
                        self.log.push(format!("write-req:{idx}"));
                        let (t, remote) = self.pairs[&(idx, stream)];
                        self.inflight
                            .entry((t, remote))
                            .or_default()
                            .push_back(bytes.to_vec());
                        pump.complete_op(op, OpOutcome::WriteDone).expect("write");
                    }
                    OpRequest::StreamRead { stream } => {
                        self.pending_reads
                            .entry((idx, stream))
                            .or_default()
                            .push_back(op);
                    }
                    OpRequest::PayloadPut { bytes } => {
                        self.store
                            .insert(*blake3::hash(&bytes).as_bytes(), bytes.to_vec());
                        pump.complete_op(op, OpOutcome::PutDone).expect("put");
                    }
                    other => panic!("unexpected op request: {other:?}"),
                }
            }
        }
        // Pair opens with standing accepts.
        let opens = std::mem::take(&mut self.pending_opens);
        for (from, open_op, target) in opens {
            if let Some(accept_op) = self.pending_accepts[target].pop_front() {
                let sa = pumps[from]
                    .0
                    .complete_op(open_op, OpOutcome::OpenDone { credit: credit() })
                    .expect("open")
                    .expect("minted stream");
                let sb = pumps[target]
                    .0
                    .complete_op(accept_op, OpOutcome::AcceptDone { credit: credit() })
                    .expect("accept")
                    .expect("minted stream");
                self.pairs.insert((from, sa), (target, sb));
                self.pairs.insert((target, sb), (from, sa));
            } else {
                self.pending_opens.push((from, open_op, target));
            }
        }
        // Match reads with in-flight bytes; the reader's consumption replenishes the WRITER's
        // credit (§3.3) — which releases held writes pump-side.
        let keys: Vec<(usize, u64)> = self.pending_reads.keys().copied().collect();
        for key in keys {
            loop {
                let has_both = self.pending_reads.get(&key).is_some_and(|q| !q.is_empty())
                    && self.inflight.get(&key).is_some_and(|q| !q.is_empty());
                if !has_both {
                    break;
                }
                let op = self
                    .pending_reads
                    .get_mut(&key)
                    .unwrap()
                    .pop_front()
                    .unwrap();
                let bytes = self.inflight.get_mut(&key).unwrap().pop_front().unwrap();
                let len = bytes.len() as u64;
                pumps[key.0]
                    .0
                    .complete_op(op, OpOutcome::ReadDone { bytes })
                    .expect("read");
                let (src_idx, src_stream) = self.pairs[&key];
                self.log.push(format!("grant:{src_idx}"));
                pumps[src_idx].0.grant_credit(src_stream, len);
            }
        }
    }
}

fn frame_payload(frame: &[u8]) -> Vec<u8> {
    let v: ciborium::value::Value = ciborium::de::from_reader(frame).expect("frame cbor");
    let ciborium::value::Value::Array(parts) = v else {
        panic!("frame shape");
    };
    let ciborium::value::Value::Bytes(payload) = &parts[1] else {
        panic!("payload");
    };
    payload.clone()
}

#[test]
fn two_stage_pipeline_exchanges_exported_tensors_under_credit_flow_control() {
    let wasm = pipeline_wasm();
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let sel = select_driver(&worker, &wasm, Some(blake3::hash(&wasm).as_bytes()))
        .expect("pipeline guest admitted");
    assert_eq!(
        (sel.major, sel.minor),
        (2, daemon_vhc_abi::COMPUTE_MINOR_V2),
        "the graduated toy imports compute@2, so selection lands at the Phase-C minor"
    );

    let mk = |role: u8, instance: u64, peer: [u8; 32], seed: u8| {
        let identity = RunIdentity {
            run_id: [0xE1; 32],
            epoch: 0,
            role: if role == 0 { "stage-a" } else { "stage-b" }.to_string(),
            instance,
            module: *blake3::hash(&wasm).as_bytes(),
        };
        let cfg = RunConfig::new(identity, [seed; 32], config(role, peer), Vec::new());
        let sink = Arc::new(Mutex::new(MemorySink::new()));
        let run = start_run(&worker, &wasm, cfg, Box::new(sink.clone())).expect("start");
        (run, sink)
    };
    // A (producer) talks to B; B (consumer) accepts.
    let (run_a, sink_a) = mk(0, 1, PEER_B, 0xA1);
    let (run_b, sink_b) = mk(1, 2, PEER_A, 0xB1);

    let pumps = [(run_a.pump.clone(), PEER_A), (run_b.pump.clone(), PEER_B)];
    let mut net = MemNet::new(2);

    // Drive until A published "sent" and B published the 32-byte commitment hash.
    let deadline = Instant::now() + Duration::from_secs(30);
    while run_a.pump.published().is_empty() || run_b.pump.published().is_empty() {
        net.service(&pumps);
        assert!(
            Instant::now() < deadline,
            "pipeline stalled: log {:?}",
            net.log
        );
        std::thread::sleep(Duration::from_millis(2));
    }

    // The acceptance: B committed EXACTLY the produced tensors, DOUBLED ON-DEVICE — the tensor
    // path end to end (produce → export → stream → import → transform → re-export → commit).
    let expected: Vec<u8> = (0..N_CHUNKS)
        .flat_map(|i| tensor_bytes(chunk_floats(i).into_iter().map(|v| v * 2.0).collect()))
        .collect();
    let expected_hash = *blake3::hash(&expected).as_bytes();
    assert_eq!(frame_payload(&run_a.pump.published()[0].2), b"sent");
    assert_eq!(
        frame_payload(&run_b.pump.published()[0].2),
        expected_hash.to_vec(),
        "the consumer's commitment hash covers exactly the doubled chunk tensors"
    );
    assert_eq!(
        net.store.get(&expected_hash),
        Some(&expected),
        "the transformed tensor content round-tripped the stream intact"
    );

    // The flow-control pin: with a one-chunk window, the producer's 2nd/3rd write REQUESTS only
    // reach the transport after a credit grant (the pump held them) — deterministic, because the
    // hold/release ordering is pump-enforced.
    let first_grant = net
        .log
        .iter()
        .position(|e| e == "grant:0")
        .expect("a grant happened");
    let writes_before_grant = net.log[..first_grant]
        .iter()
        .filter(|e| *e == "write-req:0")
        .count();
    assert_eq!(
        writes_before_grant, 1,
        "exactly ONE write fit the credit window before the first replenishment: {:?}",
        net.log
    );
    assert_eq!(
        net.log.iter().filter(|e| *e == "write-req:0").count(),
        usize::from(N_CHUNKS),
        "every held write was eventually released by credit"
    );

    // Clean stop + bit-exact replay of BOTH journals (streams, holds, and credit timing are
    // host/transport mechanism — the journaled delivered sequence is the whole guest input).
    for (run, sink, cfg_role, cfg_peer) in
        [(run_a, sink_a, 0u8, PEER_B), (run_b, sink_b, 1u8, PEER_A)]
    {
        run.pump
            .stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
            .expect("stop");
        assert!(matches!(run.wait().expect("thread"), RunEnd::Outcome(0)));
        let entries: Vec<SinkEntry> = sink.lock().expect("sink").entries.clone();
        let script = ReplayScript::from_entries(&entries);
        let replayed = replay(&worker, &wasm, &config(cfg_role, cfg_peer), &[], script)
            .expect("replay harness");
        assert_eq!(
            replayed.end,
            ReplayEnd::Outcome(0),
            "role {cfg_role} replays"
        );
        let recorded: Vec<[u8; 32]> = entries
            .iter()
            .filter_map(|e| match e {
                SinkEntry::Publish { payload_hash, .. } => Some(*payload_hash),
                _ => None,
            })
            .collect();
        let redriven: Vec<[u8; 32]> = replayed.decisions.iter().map(|d| d.payload_hash).collect();
        assert_eq!(
            recorded, redriven,
            "role {cfg_role} decisions replay bit-exact"
        );
    }
}
