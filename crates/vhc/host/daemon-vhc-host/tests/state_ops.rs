// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The det-state op conformance + replay suite (ABI §12.14 [SF-4]/[SF-R1]/[SF-7]; §13
// conformance row "state-op conformance (framing traps, torn folds)"):
//
// - the write vocabulary end-to-end: `state_open`/`state_emit`/`state_seal` against the real
//   event-loop driver, the sealed fold reproducing `daemon_vhc_proto::family_fold` bit-exactly;
// - [SF-R1]: a self-sealed root is fetchable with NO `register_chunks` and NO grant entry,
//   serviced host-locally through the ordinary Completion protocol;
// - framing + grant traps, each typed ([SF-4] coarse framing; [SF-7] budget vocabulary);
// - the degenerate single-window geometry (the 64-dim acceptance shape) on the same code path;
// - torn folds: only `state_seal` mints durable artifacts — an opened-but-unsealed stream's
//   chunks are force-reclaimed at teardown (clean return AND trap), and a restarted instance
//   observes nothing ([SF-4] crash rule);
// - tier-1 replay: `state_emit` re-executes over reproduced guest memory into the replay-side
//   state chunk store (no journal record — the journal stays O(records)); the journaled
//   `state_seal` fold is the O(1) divergence cross-check; a fetch of a self-sealed root
//   materializes from the replay-side store — never `ReplayMissingPayload`.
//
// The fixture is hand-authored WAT (no wasm32 toolchain, no guest workspace, no `guests.blake3`
// churn in this lane): behavioral event-loop guests are legible as text. Real-guest state-op
// conformance through the SDK lands with the trainer-guest wave, which owns guest re-pins.

use std::sync::{Arc, Mutex};

use daemon_vhc_host::run::{
    replay, start_run, MemorySink, ReplayEnd, ReplayScript, RunConfig, RunEnd, RunIdentity,
    SinkEntry, StateStoreStats,
};
use daemon_vhc_host::{select_driver, EngineConfig, TrapCode, Worker};

/// The mode-selected conformance guest (see file header). Layout: the admitted config's one
/// mode byte is stashed at 32; the 12-byte family image lives at 1024 (`AAAAAAAABBBB`); family
/// tags at 2048 (`master`) / 2060 (`ef`); sealed folds land at 3072+; the event buffer at 4096;
/// `read_into` output at 5120. The `read_into` handle is the deterministic first host-partition
/// buffer (ABI §7.1: a pure function of the journaled completion order —
/// `pack_handle(BUFFER, seed 1, HOST_INDEX_BASE)`).
const FIXTURE_WAT: &str = r#"
(module
  (import "vhc@2" "next_event" (func $next_event (param i32 i32) (result i64)))
  (import "vhc@2" "state_open" (func $state_open (param i32 i32 i64) (result i64)))
  (import "vhc@2" "state_emit" (func $state_emit (param i64 i32 i32) (result i64)))
  (import "vhc@2" "state_seal" (func $state_seal (param i64 i32) (result i32)))
  (import "vhc@2" "read_into" (func $read_into (param i64 i64 i32 i32) (result i64)))
  (import "data@2" "fetch" (func $fetch (param i32 i64 i64) (result i64)))
  (import "net@2" "publish" (func $publish (param i32 i32 i32) (result i64)))
  (memory (export "memory") 1)
  (data (i32.const 1024) "AAAAAAAABBBB")
  (data (i32.const 2048) "master")
  (data (i32.const 2060) "ef")
  (func (export "da_abi") (result i32) (i32.const 0x20003)) ;; major 2, minor 3
  (func (export "da_alloc") (param i32 i32) (result i32) (i32.const 64))
  (func (export "da_free") (param i32 i32))
  (func (export "da_manifest") (result i32) (i32.const 0))
  (func (export "da_claim") (result i32) (i32.const 0))
  (func (export "da_init") (param i32 i32 i32 i32) (result i32)
    (i32.store8 (i32.const 32) (i32.load8_u (local.get 0)))
    (i32.const 0))
  ;; One 12-byte round: [8 shared bytes][4-byte tail tagged by $r], sealed to fold @ $out.
  (func $seal_round (param $r i32) (param $out i32)
    (local $s i64)
    (local.set $s (call $state_open (i32.const 2048) (i32.const 6) (i64.const 12)))
    (drop (call $state_emit (local.get $s) (i32.const 1024) (i32.const 8)))
    (i32.store8 (i32.const 1032) (local.get $r))
    (drop (call $state_emit (local.get $s) (i32.const 1032) (i32.const 4)))
    (drop (call $state_seal (local.get $s) (i32.const 3072)))
    ;; copy the fold to $out (folds accumulate for mode 9's evicted-root fetch)
    (i64.store (local.get $out) (i64.load (i32.const 3072)))
    (i64.store (i32.add (local.get $out) (i32.const 8)) (i64.load (i32.const 3080)))
    (i64.store (i32.add (local.get $out) (i32.const 16)) (i64.load (i32.const 3088)))
    (i64.store (i32.add (local.get $out) (i32.const 24)) (i64.load (i32.const 3096))))
  (func (export "da_run") (result i32)
    (local $mode i32) (local $s i64) (local $n i64)
    (local.set $mode (i32.load8_u (i32.const 32)))
    ;; mode 0 — happy path: open/emit/seal, publish the fold, fetch a boundary-crossing range
    ;; of the self-sealed root [SF-R1], publish the fetched bytes.
    (if (i32.eq (local.get $mode) (i32.const 0)) (then
      (local.set $s (call $state_open (i32.const 2048) (i32.const 6) (i64.const 12)))
      (drop (call $state_emit (local.get $s) (i32.const 1024) (i32.const 8)))
      (drop (call $state_emit (local.get $s) (i32.const 1032) (i32.const 4)))
      (drop (call $state_seal (local.get $s) (i32.const 3072)))
      (drop (call $publish (i32.const 0) (i32.const 3072) (i32.const 32)))
      (drop (call $fetch (i32.const 3072) (i64.const 2) (i64.const 7)))
      (drop (call $next_event (i32.const 4096) (i32.const 512)))
      (local.set $n (call $read_into
        (i64.const 0x0800000180000000) (i64.const 0) (i32.const 5120) (i32.const 16)))
      (drop (call $publish (i32.const 0) (i32.const 5120) (i32.wrap_i64 (local.get $n))))
      (return (i32.const 1))))
    ;; mode 1 — misframed emit: a 9-byte chunk under chunk_size 8 traps typed.
    (if (i32.eq (local.get $mode) (i32.const 1)) (then
      (local.set $s (call $state_open (i32.const 2048) (i32.const 6) (i64.const 12)))
      (drop (call $state_emit (local.get $s) (i32.const 1024) (i32.const 9)))
      (return (i32.const 1))))
    ;; mode 2 — incomplete seal: 8 of the declared 12 bytes.
    (if (i32.eq (local.get $mode) (i32.const 2)) (then
      (local.set $s (call $state_open (i32.const 2048) (i32.const 6) (i64.const 12)))
      (drop (call $state_emit (local.get $s) (i32.const 1024) (i32.const 8)))
      (drop (call $state_seal (local.get $s) (i32.const 3072)))
      (return (i32.const 1))))
    ;; mode 3 — a second concurrent stream under state-streams-max = 1.
    (if (i32.eq (local.get $mode) (i32.const 3)) (then
      (drop (call $state_open (i32.const 2048) (i32.const 6) (i64.const 12)))
      (drop (call $state_open (i32.const 2060) (i32.const 2) (i64.const 12)))
      (return (i32.const 1))))
    ;; mode 4 — an 8-byte emit under a 4-byte per-emit write-budget ceiling.
    (if (i32.eq (local.get $mode) (i32.const 4)) (then
      (local.set $s (call $state_open (i32.const 2048) (i32.const 6) (i64.const 12)))
      (drop (call $state_emit (local.get $s) (i32.const 1024) (i32.const 8)))
      (return (i32.const 1))))
    ;; mode 5 — emit to a never-issued stream id.
    (if (i32.eq (local.get $mode) (i32.const 5)) (then
      (drop (call $state_emit (i64.const 0x8000000000000063) (i32.const 1024) (i32.const 4)))
      (return (i32.const 1))))
    ;; mode 6 — state_open with no state contract provisioned (chunk_size 0).
    (if (i32.eq (local.get $mode) (i32.const 6)) (then
      (drop (call $state_open (i32.const 2048) (i32.const 6) (i64.const 12)))
      (return (i32.const 1))))
    ;; mode 7 — degenerate single-window geometry (chunk_size ≥ byte_len): the same code path.
    (if (i32.eq (local.get $mode) (i32.const 7)) (then
      (local.set $s (call $state_open (i32.const 2048) (i32.const 6) (i64.const 12)))
      (drop (call $state_emit (local.get $s) (i32.const 1024) (i32.const 12)))
      (drop (call $state_seal (local.get $s) (i32.const 3072)))
      (drop (call $publish (i32.const 0) (i32.const 3072) (i32.const 32)))
      (return (i32.const 1))))
    ;; mode 8 — a torn fold: open + emit, never seal, return cleanly.
    (if (i32.eq (local.get $mode) (i32.const 8)) (then
      (local.set $s (call $state_open (i32.const 2048) (i32.const 6) (i64.const 12)))
      (drop (call $state_emit (local.get $s) (i32.const 1024) (i32.const 8)))
      (return (i32.const 1))))
    ;; mode 9 — retention: three sealed rounds under retain_roots 2, then fetch round 0's
    ;; (evicted) fold — it left the fetchable set, so the grant check refuses it.
    (if (i32.eq (local.get $mode) (i32.const 9)) (then
      (call $seal_round (i32.const 0) (i32.const 3200))
      (call $seal_round (i32.const 1) (i32.const 3232))
      (call $seal_round (i32.const 2) (i32.const 3264))
      (drop (call $fetch (i32.const 3200) (i64.const 0) (i64.const 0)))
      (return (i32.const 1))))
    (i32.const 15)))
"#;

fn fixture_wasm() -> Vec<u8> {
    wat::parse_str(FIXTURE_WAT).expect("fixture wat")
}

fn identity(instance: u64, wasm: &[u8]) -> RunIdentity {
    RunIdentity {
        run_id: [0x5F; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance,
        module: *blake3::hash(wasm).as_bytes(),
    }
}

/// Run the fixture in `mode` with `tune` applied to the run config; the guest needs no external
/// servicing (self-sealed fetches complete host-locally), so this drives straight to the end.
#[allow(clippy::type_complexity)] // the drive-harness tuple, the data_fetch.rs convention
fn run_mode(
    mode: u8,
    instance: u64,
    tune: impl FnOnce(&mut RunConfig),
) -> (
    Vec<(u64, u64, Vec<u8>)>,
    Vec<SinkEntry>,
    RunEnd,
    StateStoreStats,
) {
    let wasm = fixture_wasm();
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let mut cfg = RunConfig::new(
        identity(instance, &wasm),
        [0x66; 32],
        vec![mode],
        Vec::new(),
    );
    cfg.state_chunk_size = 8;
    tune(&mut cfg);
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, &wasm, cfg, Box::new(sink.clone())).expect("start");
    let pump = run.pump.clone();
    let end = run.wait().expect("guest thread clean");
    let entries = sink.lock().expect("sink").entries.clone();
    (pump.published(), entries, end, pump.state_store_stats())
}

/// The §12.1 frame's payload bytes.
fn payload_of(frame: &[u8]) -> Vec<u8> {
    let v: ciborium::value::Value = ciborium::de::from_reader(frame).expect("frame cbor");
    let ciborium::value::Value::Array(parts) = v else {
        panic!("frame shape")
    };
    let ciborium::value::Value::Bytes(payload) = &parts[1] else {
        panic!("payload shape")
    };
    payload.clone()
}

/// The fold the fixture's 12-byte family seals at chunk_size 8: [8 bytes][4-byte tail].
fn expected_fold() -> [u8; 32] {
    let hashes = vec![
        daemon_vhc_proto::Hash(*blake3::hash(b"AAAAAAAA").as_bytes()),
        daemon_vhc_proto::Hash(*blake3::hash(b"BBBB").as_bytes()),
    ];
    daemon_vhc_proto::family_fold(8, 12, &hashes).0
}

fn expect_trap(end: &RunEnd, code: TrapCode, needle: &str) {
    match end {
        RunEnd::Trapped(t) => {
            assert_eq!(t.code, code, "trap: {t}");
            assert!(t.detail.contains(needle), "detail: {}", t.detail);
        }
        other => panic!("expected a {code:?} trap, got {other:?}"),
    }
}

// -- selection: the minor-3 surface admits end-to-end ---------------------------------------------

#[test]
fn state_importing_module_selects_at_minor_3() {
    let wasm = fixture_wasm();
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let sel = select_driver(&worker, &wasm, Some(blake3::hash(&wasm).as_bytes()))
        .expect("a state-importing module admits on this host");
    assert_eq!((sel.major, sel.minor), (2, 3), "the det-state minor");
}

// -- the write vocabulary + [SF-R1] ---------------------------------------------------------------

#[test]
fn open_emit_seal_publishes_the_proto_fold_and_self_sealed_fetch_serves_ranges() {
    let (published, entries, end, stats) = run_mode(0, 1, |_| {});
    assert!(matches!(end, RunEnd::Outcome(1)), "end: {end:?}");
    assert_eq!(published.len(), 2, "the fold + the fetched range");
    // The sealed fold IS the proto-side family fold (chunk_size 8, byte_len 12, two chunks).
    assert_eq!(payload_of(&published[0].2), expected_fold().to_vec());
    // [SF-R1]: the range [2, 9) of the self-sealed root — crossing the chunk boundary — was
    // served host-locally with NO grant entry, NO register_chunks, NO embedder round-trip.
    assert_eq!(payload_of(&published[1].2), b"AAAAAAB".to_vec());
    // The artifact is retained (12 bytes, 2 chunks), nothing left open.
    assert_eq!(
        stats,
        StateStoreStats {
            sealed_folds: 1,
            open_streams: 0,
            retained_bytes: 12,
            chunk_objects: 2
        }
    );
    // Journal shape ([SF-4]): NO record per emit (the journal stays O(records)); exactly one
    // kind-6 seal record carrying the 32-byte fold; the fetch's tag-14 completion.
    let seal_records: Vec<_> = entries
        .iter()
        .filter_map(|e| match e {
            SinkEntry::ReadBack { kind, value, .. }
                if *kind == u64::from(daemon_vhc_abi::READBACK_KIND_STATE_SEAL) =>
            {
                Some(value.clone())
            }
            _ => None,
        })
        .collect();
    assert_eq!(seal_records.len(), 1, "one seal, one nr record");
    assert_eq!(seal_records[0], expected_fold().to_vec());
    assert_eq!(
        entries
            .iter()
            .filter(|e| matches!(e, SinkEntry::ReadBack { .. }))
            .count(),
        1,
        "emits are recordless (dc class)"
    );
    assert_eq!(
        entries
            .iter()
            .filter(|e| matches!(e, SinkEntry::Completion { .. }))
            .count(),
        1,
        "the self-sealed fetch completes through the ordinary protocol"
    );
}

#[test]
fn degenerate_single_window_geometry_runs_the_same_path() {
    // chunk_size (16) ≥ byte_len (12): one chunk, one window — the 64-dim acceptance shape.
    let (published, _, end, stats) = run_mode(7, 1, |cfg| cfg.state_chunk_size = 16);
    assert!(matches!(end, RunEnd::Outcome(1)), "end: {end:?}");
    let hashes = vec![daemon_vhc_proto::Hash(
        *blake3::hash(b"AAAAAAAABBBB").as_bytes(),
    )];
    assert_eq!(
        payload_of(&published[0].2),
        daemon_vhc_proto::family_fold(16, 12, &hashes).0.to_vec()
    );
    assert_eq!(stats.sealed_folds, 1);
    assert_eq!(stats.chunk_objects, 1, "the whole family is one chunk");
}

// -- framing + grant traps ([SF-4]/[SF-7]) --------------------------------------------------------

#[test]
fn misframed_emit_traps_typed() {
    let (_, _, end, stats) = run_mode(1, 1, |_| {});
    expect_trap(&end, TrapCode::StateMisframedEmit, "chunk of 9 bytes");
    // The trap force-reclaimed the open stream: nothing durable, nothing staged (crash rule).
    assert_eq!(stats.open_streams, 0);
    assert_eq!(stats.chunk_objects, 0);
}

#[test]
fn incomplete_seal_traps_typed() {
    let (_, _, end, _) = run_mode(2, 1, |_| {});
    expect_trap(&end, TrapCode::StateIncompleteSeal, "8 of the declared 12");
}

#[test]
fn streams_max_grant_refuses_the_second_stream() {
    let (_, _, end, _) = run_mode(3, 1, |cfg| cfg.state_streams_max = 1);
    expect_trap(&end, TrapCode::GrantViolation, "state-streams-max");
}

#[test]
fn per_emit_write_budget_refuses_typed() {
    let (_, _, end, _) = run_mode(4, 1, |cfg| cfg.state_emit_max_bytes = 4);
    expect_trap(&end, TrapCode::GrantViolation, "state-write-budget");
}

#[test]
fn unknown_stream_is_an_invalid_handle() {
    let (_, _, end, _) = run_mode(5, 1, |_| {});
    expect_trap(&end, TrapCode::InvalidHandle, "names no open state stream");
}

#[test]
fn unprovisioned_state_plane_refuses_open() {
    let (_, _, end, _) = run_mode(6, 1, |cfg| cfg.state_chunk_size = 0);
    expect_trap(&end, TrapCode::GrantViolation, "not provisioned");
}

// -- retention ([SF-7]) ---------------------------------------------------------------------------

#[test]
fn retention_evicts_the_oldest_root_and_its_fetch_refuses() {
    // Three sealed rounds under retain_roots 2: round 0's root leaves the fetchable set; the
    // guest's fetch of it falls through [SF-R1] to the grant check and refuses typed.
    let (_, _, end, stats) = run_mode(9, 1, |cfg| cfg.state_retain_roots = 2);
    expect_trap(
        &end,
        TrapCode::GrantViolation,
        "not in the admitted artifact set",
    );
    assert_eq!(stats.sealed_folds, 2, "rounds 1 and 2 retained");
    // Content-addressed dedup: the 8-byte chunk shared by every round is stored once —
    // 8 (shared) + 4 + 4 tail bytes, not 2 × 12.
    assert_eq!(stats.retained_bytes, 16);
    assert_eq!(stats.chunk_objects, 3);
}

// -- torn folds ([SF-4] crash rule) ---------------------------------------------------------------

#[test]
fn torn_folds_are_never_durable_and_gc_on_teardown_and_restart() {
    // A clean return with an unsealed stream: teardown force-reclaims the staged chunks.
    let (published, entries, end, stats) = run_mode(8, 1, |_| {});
    assert!(matches!(end, RunEnd::Outcome(1)));
    assert!(published.is_empty());
    assert_eq!(
        stats,
        StateStoreStats {
            sealed_folds: 0,
            open_streams: 0,
            retained_bytes: 0,
            chunk_objects: 0
        },
        "nothing durable, nothing staged"
    );
    assert!(
        !entries.iter().any(|e| matches!(
            e,
            SinkEntry::ReadBack { kind, .. }
                if *kind == u64::from(daemon_vhc_abi::READBACK_KIND_STATE_SEAL)
        )),
        "no seal record: the journal carries no durable-root evidence for a torn fold"
    );
    // The restarted instance (the crash-recovery shape: a fresh incarnation of the same
    // module) observes an EMPTY store and works normally — the torn fold left no residue.
    let (published, _, end, stats) = run_mode(0, 2, |_| {});
    assert!(matches!(end, RunEnd::Outcome(1)), "restart: {end:?}");
    assert_eq!(payload_of(&published[0].2), expected_fold().to_vec());
    assert_eq!(stats.sealed_folds, 1);
}

// -- tier-1 replay ([SF-4] journal/replay classes) ------------------------------------------------

/// Record mode 0 and return `(entries, config)` for replay assertions.
fn recorded_happy_path() -> (Vec<SinkEntry>, Vec<u8>) {
    let (_, entries, end, _) = run_mode(0, 3, |_| {});
    assert!(matches!(end, RunEnd::Outcome(1)));
    (entries, vec![0u8])
}

#[test]
fn recorded_fold_replays_bit_exact_with_no_payload_archive() {
    let (entries, config) = recorded_happy_path();
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let mut script = ReplayScript::from_entries(&entries);
    script.state_chunk_size = 8;
    assert_eq!(script.state_seals.len(), 1, "the kind-6 record routed");
    assert!(
        script.payloads.is_empty(),
        "no payload table: self-sealed roots are NOT archived (O(records) journal)"
    );
    let replayed = replay(&worker, &fixture_wasm(), &config, &[], script).expect("harness");
    assert_eq!(
        replayed.end,
        ReplayEnd::Outcome(1),
        "no ReplayMissingPayload"
    );
    // Every publish reproduces bit-for-bit: the fold AND the fetched range (which replay
    // materialized from the replay-side state chunk store, not the payload table).
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
    assert_eq!(recorded.len(), 2);
}

#[test]
fn a_divergent_replay_trips_the_seal_cross_check() {
    let (entries, config) = recorded_happy_path();
    // Tamper the recorded seal fold (the nr record): replay re-derives the true fold over the
    // re-executed emits and MUST refuse the recording, O(1) at the seal.
    let tampered: Vec<SinkEntry> = entries
        .iter()
        .cloned()
        .map(|e| match e {
            SinkEntry::ReadBack {
                src,
                kind,
                status,
                mut value,
            } if kind == u64::from(daemon_vhc_abi::READBACK_KIND_STATE_SEAL) => {
                value[0] ^= 0xFF;
                SinkEntry::ReadBack {
                    src,
                    kind,
                    status,
                    value,
                }
            }
            other => other,
        })
        .collect();
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let mut script = ReplayScript::from_entries(&tampered);
    script.state_chunk_size = 8;
    let replayed = replay(&worker, &fixture_wasm(), &config, &[], script).expect("harness");
    match replayed.end {
        ReplayEnd::Diverged(msg) => {
            assert!(msg.contains("state_seal fold mismatch"), "{msg}");
        }
        other => panic!("expected the seal cross-check divergence, got {other:?}"),
    }
}

#[test]
fn a_torn_fold_recording_replays_cleanly() {
    // The torn-fold journal (open + emit, no seal, clean return) replays without divergence:
    // the re-executed emit lands in the replay-side store and is simply never sealed.
    let (_, entries, end, _) = run_mode(8, 4, |_| {});
    assert!(matches!(end, RunEnd::Outcome(1)));
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let mut script = ReplayScript::from_entries(&entries);
    script.state_chunk_size = 8;
    assert!(
        script.state_seals.is_empty(),
        "no seal record for a torn fold"
    );
    let replayed = replay(&worker, &fixture_wasm(), &[0x08], &[], script).expect("harness");
    assert_eq!(replayed.end, ReplayEnd::Outcome(1));
    assert!(replayed.decisions.is_empty());
}

// -- grants sourcing ([SF-7] via the generic WorldGrant bounds) -----------------------------------

#[test]
fn state_grant_bounds_source_from_the_vhc_world_grant() {
    use daemon_vhc_host::run::apply_state_grant_bounds;
    use daemon_vhc_proto::{GrantBound, RoleGrants, WorldGrant};

    let mut role = RoleGrants::default();
    let mut bounds = std::collections::BTreeMap::new();
    bounds.insert(
        daemon_vhc_proto::STATE_WRITE_BUDGET_GRANT.to_string(),
        GrantBound {
            max_bytes: Some(1 << 22),
            max_per_slice: None,
            rate_per_min: Some(1 << 30),
            max_outstanding: None,
            values: Vec::new(),
        },
    );
    bounds.insert(
        daemon_vhc_proto::STATE_STORE_BYTES_GRANT.to_string(),
        GrantBound {
            max_bytes: Some(16 << 30),
            max_per_slice: None,
            rate_per_min: None,
            max_outstanding: None,
            values: Vec::new(),
        },
    );
    bounds.insert(
        daemon_vhc_proto::STATE_STREAMS_MAX_GRANT.to_string(),
        GrantBound {
            max_bytes: None,
            max_per_slice: None,
            rate_per_min: None,
            max_outstanding: Some(4),
            values: Vec::new(),
        },
    );
    bounds.insert(
        daemon_vhc_host::run::STATE_RETAIN_ROOTS_GRANT.to_string(),
        GrantBound {
            max_bytes: None,
            max_per_slice: None,
            rate_per_min: None,
            max_outstanding: Some(3),
            values: Vec::new(),
        },
    );
    role.worlds
        .insert("vhc@2".to_string(), WorldGrant { minor: 3, bounds });

    let wasm = fixture_wasm();
    let mut cfg = RunConfig::new(identity(9, &wasm), [0; 32], Vec::new(), Vec::new());
    apply_state_grant_bounds(&role, &mut cfg);
    assert_eq!(cfg.state_emit_max_bytes, 1 << 22);
    assert_eq!(cfg.state_write_rate_per_min, 1 << 30);
    assert_eq!(cfg.state_store_bytes, 16 << 30);
    assert_eq!(cfg.state_streams_max, 4);
    assert_eq!(cfg.state_retain_roots, 3);

    // An absent grant leaves the defaults: unbounded budgets, the design-default retention.
    let mut cfg = RunConfig::new(identity(9, &wasm), [0; 32], Vec::new(), Vec::new());
    apply_state_grant_bounds(&RoleGrants::default(), &mut cfg);
    assert_eq!(cfg.state_emit_max_bytes, 0);
    assert_eq!(
        cfg.state_retain_roots,
        daemon_vhc_proto::STATE_RETAIN_ROOTS_DEFAULT
    );
}
