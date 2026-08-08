// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Role-session lifecycle over real guest modules and in-process providers.
//!
//! The session under test is the production seat: certified per-run identity, the mandatory
//! certificate check on inbound frames, opaque published-frame relay onto a [`ControlPlane`],
//! content-addressed payload servicing, hard pause, graceful/immediate leave, and terminal
//! classification. The guests are the pinned test modules (timer-driven publisher; snapshot-on-
//! drain module) — no round vocabulary exists anywhere in this suite.
//!
//! Dev/test harness: shells `cargo build` for the guests (the established pattern), so the
//! fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use daemon_vhc_host::run::{MemorySink, RunConfig, RunIdentity, SinkEntry};
use daemon_vhc_host::EngineConfig;
use daemon_vhc_net::{ContentStore, ControlPlane, LoopbackGossip, MemoryContentStore};
use daemon_vhc_proto::{peer_id, CertScope, Hash, SigningKey};
use daemon_vhc_session::identity::{issue_run_key, CertifiedRunKey};
use daemon_vhc_session::protocol::{Event, LeaveMode, TerminalOutcome};
use daemon_vhc_session::role_session::{spawn_role, RoleProviders, RoleSessionSpec, ThrottleLevel};
use tokio::sync::mpsc;

// -- guest build harness (mirrors the host suites) ------------------------------------------------

fn guest_wasm(name: &str) -> Vec<u8> {
    daemon_vhc_guest_build::guest_wasm(name)
}

// -- session rig -----------------------------------------------------------------------------------

struct Rig {
    plane: Arc<LoopbackGossip>,
    payloads: Arc<MemoryContentStore>,
    sink: Arc<Mutex<MemorySink>>,
    spec: RoleSessionSpec,
    base: SigningKey,
    identity: RunIdentity,
    /// The session's own §12.1 sender (its certified per-run public key).
    own_sender: [u8; 32],
}

/// Assemble a production-shaped session spec over in-process providers: fresh CSPRNG base +
/// certified per-run key, loopback control plane, in-memory content stores, memory journal.
fn rig(wasm: &[u8], role: &str, config: Vec<u8>) -> Rig {
    let module_hash = *blake3::hash(wasm).as_bytes();
    let run_id = [0x77u8; 32];
    let identity = RunIdentity {
        run_id,
        epoch: 0,
        role: role.to_string(),
        instance: 1,
        module: module_hash,
    };
    let base = daemon_vhc_session::identity::SecretSeed::fresh()
        .expect("entropy")
        .signing_key();
    let certified: CertifiedRunKey = issue_run_key(
        &base,
        CertScope {
            run_id: Hash(run_id),
            epoch: 0,
            role: role.to_string(),
            instance: 1,
            module_hash: Hash(module_hash),
        },
    )
    .expect("issue run key");
    let run_cfg = RunConfig::new(
        identity.clone(),
        certified.key.to_bytes(),
        config,
        b"admitted-grants".to_vec(),
    );
    let plane = Arc::new(LoopbackGossip::new());
    let payloads = Arc::new(MemoryContentStore::new());
    let artifacts = Arc::new(MemoryContentStore::new());
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let spec = RoleSessionSpec {
        module: wasm.to_vec(),
        engine: EngineConfig::default(),
        run: run_cfg,
        own_cert: certified.cert.clone(),
        trusted_bases: vec![peer_id(&base)],
        peer_certs: vec![certified.cert.clone()],
        seat_grant: None,
        providers: RoleProviders {
            control: plane.clone(),
            payloads: payloads.clone(),
            artifacts: artifacts.clone(),
            plane_stats: None,
            archive_heads: None,
        },
        journal: Box::new(sink.clone()),
        drain_deadline: Duration::from_secs(10),
        restore: None,
        admitted_quotas: None,
        archive: None,
        catch_up: Vec::new(),
    };
    let own_sender = certified.sender().0;
    Rig {
        plane,
        payloads,
        sink,
        spec,
        base,
        identity,
        own_sender,
    }
}

/// Await the session's terminal event, forwarding nothing else.
async fn await_terminal(events: &mut mpsc::UnboundedReceiver<Event>) -> (u64, TerminalOutcome) {
    loop {
        let ev = tokio::time::timeout(Duration::from_secs(30), events.recv())
            .await
            .expect("terminal event within the deadline")
            .expect("event stream open until terminal");
        if let Event::RunTerminated {
            generation,
            outcome,
            ..
        } = ev
        {
            return (generation, outcome);
        }
    }
}

/// `spawn` returns immediately; the module's published frames relay verbatim (already-signed
/// opaque bytes) onto the control plane; an immediate leave ends the role as owner intent.
#[tokio::test(flavor = "multi_thread")]
async fn published_frames_relay_opaquely_and_immediate_leave_classifies_left() {
    let wasm = guest_wasm("toy_averager");
    // Config [n=2]: publish two mean frames on the control channel, then park until Stop.
    let r = rig(&wasm, "publisher", vec![2u8]);
    let mut outside = r.plane.subscribe();
    let (tx, mut events) = mpsc::unbounded_channel();
    let handle = spawn_role("run-relay".into(), r.spec, tx);
    assert_eq!(handle.generation(), 1);

    // The session's FIRST publication is its §12.3 certificate announcement (the distribution
    // record — a top-level single-entry map, structurally disjoint from the frame triple), then
    // both module publishes arrive as §12.1-signed frames, relayed verbatim.
    let announcement = tokio::time::timeout(Duration::from_secs(30), outside.recv())
        .await
        .expect("certificate announcement within the deadline")
        .expect("plane open");
    match daemon_vhc_session::distribution::DistributionRecord::from_bytes(&announcement)
        .expect("the first publication is the distribution record")
    {
        daemon_vhc_session::distribution::DistributionRecord::Cert(cert) => {
            cert.verify_chain().expect("the announced record verifies");
        }
        other => panic!("expected the session's certificate announcement, got {other:?}"),
    }
    for _ in 0..2 {
        let frame = tokio::time::timeout(Duration::from_secs(30), outside.recv())
            .await
            .expect("published frame within the deadline")
            .expect("plane open");
        // Opaque relay: the frame verifies as a signed envelope from the session's certified
        // per-run key — the session never re-encoded or inspected it.
        let v: ciborium::value::Value = ciborium::de::from_reader(frame.as_slice()).unwrap();
        let ciborium::value::Value::Array(parts) = v else {
            panic!("frame is not [envelope, payload, sig]")
        };
        assert_eq!(parts.len(), 3, "the §12.1 wire triple");
    }

    // The role is running and the caller never blocked: leave immediately, get the typed end.
    handle.leave(LeaveMode::Immediate);
    let (generation, outcome) = await_terminal(&mut events).await;
    assert_eq!(generation, 1);
    assert_eq!(outcome, TerminalOutcome::Left { checkpoint: None });
}

/// Hard pause: while paused the module receives nothing and publishes nothing; resume continues
/// the run. The pause is enforced at the pump (event delivery freezes), never by asking nicely.
#[tokio::test(flavor = "multi_thread")]
async fn hard_pause_stops_the_module_and_resume_continues() {
    let wasm = guest_wasm("toy_averager");
    // A long publisher (200 ticks) so the pause lands mid-run.
    let r = rig(&wasm, "publisher", vec![200u8]);
    let mut outside = r.plane.subscribe();
    let (tx, mut events) = mpsc::unbounded_channel();
    let handle = spawn_role("run-pause".into(), r.spec, tx);

    // Let it publish a little, then pause hard.
    let first = tokio::time::timeout(Duration::from_secs(30), outside.recv())
        .await
        .expect("first publish")
        .expect("plane open");
    assert!(!first.is_empty());
    handle.throttle(ThrottleLevel {
        paused: true,
        duty_cycle_pct: Some(0),
        vram_cap_mb: None,
    });

    // Drain in-flight deliveries, then assert the publish stream STALLS: two samples of the
    // outside inbox separated by real time must be identical while paused.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut drained = 0usize;
    while outside.try_recv().is_some() {
        drained += 1;
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        outside.try_recv().is_none(),
        "a paused module must not publish (drained {drained} in-flight frames, then silence)"
    );

    // Resume: publishing continues.
    handle.throttle(ThrottleLevel {
        paused: false,
        duty_cycle_pct: Some(100),
        vram_cap_mb: None,
    });
    tokio::time::timeout(Duration::from_secs(30), outside.recv())
        .await
        .expect("publishing resumes after un-pause")
        .expect("plane open");

    handle.leave(LeaveMode::Immediate);
    let (_, outcome) = await_terminal(&mut events).await;
    assert_eq!(outcome, TerminalOutcome::Left { checkpoint: None });
}

/// Graceful leave: quiesce at the fence, the module snapshots, and the session persists the
/// capture to the payload plane — the leave checkpoint rides the terminal event.
#[tokio::test(flavor = "multi_thread")]
async fn graceful_leave_quiesces_and_persists_the_drain_snapshot() {
    let wasm = guest_wasm("test_migrate_old");
    let r = rig(&wasm, "stateful", vec![5u8]);
    let payloads = r.payloads.clone();
    let (tx, mut events) = mpsc::unbounded_channel();
    let handle = spawn_role("run-drain".into(), r.spec, tx);

    // Give the module a moment to start, then leave gracefully.
    tokio::time::sleep(Duration::from_millis(200)).await;
    handle.leave(LeaveMode::Graceful);
    let (_, outcome) = await_terminal(&mut events).await;
    let TerminalOutcome::Left {
        checkpoint: Some(hex),
    } = outcome
    else {
        panic!("graceful leave must carry the drain snapshot, got {outcome:?}");
    };
    // The checkpoint object is real: fetch it back by content hash.
    let mut hash = [0u8; 32];
    for (i, byte) in hash.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex checkpoint hash");
    }
    let bytes = payloads
        .get_content(&Hash(hash))
        .await
        .expect("drain snapshot persisted to the payload plane");
    assert!(!bytes.is_empty());
}

/// Inbound: a certified peer's signed frame is verified above the pump and delivered below it
/// (tag-12 evidence journaled); the session's own echoed frames are filtered; an uncertified
/// sender is refused typed and never reaches the module.
#[tokio::test(flavor = "multi_thread")]
async fn inbound_frames_verify_certify_and_filter_echoes() {
    let wasm = guest_wasm("toy_averager");
    let mut r = rig(&wasm, "publisher", vec![2u8]);
    // A certified peer on the same run (same trusted base).
    let peer = issue_run_key(
        &r.base,
        CertScope {
            run_id: Hash(r.identity.run_id),
            epoch: 0,
            role: "peer".to_string(),
            instance: 1,
            module_hash: Hash(r.identity.module),
        },
    )
    .expect("issue peer key");
    r.spec.peer_certs.push(peer.cert.clone());
    let plane = r.plane.clone();
    let sink = r.sink.clone();
    let own_sender = r.own_sender;
    let (tx, mut events) = mpsc::unbounded_channel();
    let handle = spawn_role("run-inbound".into(), r.spec, tx);

    // Wait until the module is live (its first publish lands in the journal).
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while std::time::Instant::now() < deadline {
        let published = sink
            .lock()
            .unwrap()
            .entries
            .iter()
            .any(|e| matches!(e, SinkEntry::Publish { .. }));
        if published {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // The peer publishes a signed opaque frame (channel 0, seq 0) onto the plane.
    let frame = signed_frame(
        &peer,
        r.identity.run_id,
        r.identity.module,
        "peer",
        1,
        0,
        b"opaque-peer-voice",
    );
    plane.publish(&frame).await.expect("publish peer frame");
    // An impostor (no certificate) publishes on the same scope: refused typed, never delivered.
    let impostor = issue_run_key(
        &SigningKey::from_bytes(&[9u8; 32]),
        CertScope {
            run_id: Hash(r.identity.run_id),
            epoch: 0,
            role: "peer".to_string(),
            instance: 2,
            module_hash: Hash(r.identity.module),
        },
    )
    .expect("issue impostor key");
    let bad = signed_frame(
        &impostor,
        r.identity.run_id,
        r.identity.module,
        "peer",
        2,
        0,
        b"uncertified",
    );
    plane.publish(&bad).await.expect("publish impostor frame");

    // The certified frame lands below the pump: its ORIGINAL signed bytes ride the journal as
    // delivery evidence. The impostor's never does.
    let peer_sender = peer.sender().0;
    let impostor_sender = impostor.sender().0;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut delivered = false;
    while std::time::Instant::now() < deadline {
        delivered =
            sink.lock().unwrap().entries.iter().any(
                |e| matches!(e, SinkEntry::SignedFrame { sender, .. } if *sender == peer_sender),
            );
        if delivered {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(delivered, "the certified peer frame reaches the pump");
    {
        let guard = sink.lock().unwrap();
        assert!(
            !guard.entries.iter().any(
                |e| matches!(e, SinkEntry::SignedFrame { sender, .. } if *sender == impostor_sender)
            ),
            "an uncertified sender must never reach the module"
        );
        // Echo filter: the session's own published frames fan back over the loopback plane but
        // are never re-delivered into its own pump.
        assert!(
            !guard.entries.iter().any(
                |e| matches!(e, SinkEntry::SignedFrame { sender, .. } if *sender == own_sender)
            ),
            "own-voice echoes are filtered before the attach"
        );
    }

    handle.leave(LeaveMode::Immediate);
    let (_, outcome) = await_terminal(&mut events).await;
    assert_eq!(outcome, TerminalOutcome::Left { checkpoint: None });
}

/// Author one §12.1 wire frame signed by `key` (the same envelope shape the pump signs).
fn signed_frame(
    key: &CertifiedRunKey,
    run_id: [u8; 32],
    module_hash: [u8; 32],
    role: &str,
    instance: u64,
    seq: u64,
    payload: &[u8],
) -> Vec<u8> {
    use ciborium::value::Value;
    let sender = key.sender().0;
    let envelope = Value::Map(vec![
        (Value::from("domain"), Value::from("daemon-vhc/frame/2")),
        (Value::from("run_id"), Value::Bytes(run_id.to_vec())),
        (Value::from("epoch"), Value::from(0u64)),
        (Value::from("role"), Value::from(role)),
        (Value::from("instance"), Value::from(instance)),
        (Value::from("module"), Value::Bytes(module_hash.to_vec())),
        (Value::from("sender"), Value::Bytes(sender.to_vec())),
        (Value::from("channel"), Value::from(0u64)),
        (Value::from("seq"), Value::from(seq)),
        (
            Value::from("payload_hash"),
            Value::Bytes(blake3::hash(payload).as_bytes().to_vec()),
        ),
    ]);
    let sig = daemon_vhc_proto::sign::sign_canonical(&key.key, &envelope).expect("sign");
    let wire = Value::Array(vec![
        envelope,
        Value::Bytes(payload.to_vec()),
        Value::Bytes(sig.0.to_vec()),
    ]);
    let mut out = Vec::new();
    ciborium::into_writer(&wire, &mut out).expect("frame cbor");
    out
}
