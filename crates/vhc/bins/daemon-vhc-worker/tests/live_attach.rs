// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![cfg(feature = "vhc-net")]

//! The LIVE transport attach end-to-end over the loopback relay: two REAL worker processes —
//! separate identity stores, separate run-state roots — join one run through node-authored
//! `SessionCredentials` (WS control plane + the filesystem content store), and the §12.3
//! certificate distribution closes the trust loop with no out-of-band exchange:
//!
//! 1. each session announces its own certificate on the plane at attach;
//! 2. the peer ingests it (chain-verified, base genesis-trusted) and then DELIVERS the
//!    announcer's §12.1 frames — pinned by the tag-12 evidence records in the receiving peer's
//!    durable journal, sender-matched against the announcer's persisted certificate;
//! 3. leave/terminal classification works over the live plane exactly as over the smoke seat.
//!
//! Dev/test harness: shells cargo for the guest build; the env/spawn bans target the shipped
//! node, so they are allowed file-wide here.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command as StdCommand;
use std::sync::Once;
use std::time::Duration;

use ciborium::value::Value;
use daemon_common::SessionId;
use daemon_provision::{Placement, PlacementSpec, ProcessProvisioner, Provisioner};
use daemon_vhc_net::ws_relay::MockWsCoordinator;
use daemon_vhc_proto::genesis::{
    ChannelDecl, Identities, RoleEntry, RoleGrants, RunSection, SnapshotArtifact,
    TransportSelection, GENESIS_SCHEMA_MAJOR,
};
use daemon_vhc_proto::{
    blake3_hash, peer_id, to_canonical_vec, GenesisEnvelope, PeerId, SignedEnvelope, SigningKey,
};
use daemon_vhc_session::journal_home::{self, RUN_DIR_ENV};
use daemon_vhc_session::keystore::{VhcKeystore, IDENTITY_DIR_ENV};
use daemon_vhc_session::protocol::{
    self, Command, Event, JoinPolicy, LeaveMode, PolicyMode, SessionCredentials, TerminalOutcome,
    WsAuthSpec,
};

const RUN_LABEL: &str = "live-attach-smoke";
const WORKER_ROLE: &str = "publisher";

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

fn module_path(name: &str) -> PathBuf {
    BUILD.call_once(|| {
        let status = StdCommand::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests");
        assert!(status.success(), "building guest modules failed");
    });
    guests_root().join(format!("target/wasm32-unknown-unknown/release/{name}.wasm"))
}

/// Author + freeze the genesis: the publisher guest as the worker role, the pinned consensus
/// coordinator blob declared (never launched here), and the `[identities]` section naming BOTH
/// workers' base identities as trusted certificate issuers — the trust roots the §12.3
/// distribution records chain to (never ambient config).
fn genesis_wire(trusted_bases: &[PeerId]) -> (Vec<u8>, [u8; 32]) {
    let coord_wasm = std::fs::read(module_path("coordinator_quorum")).expect("coordinator blob");
    let publisher_wasm = std::fs::read(module_path("toy_averager")).expect("publisher blob");

    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "coordinator.wasm".to_string(),
        SnapshotArtifact {
            url: format!("file://{}", module_path("coordinator_quorum").display()),
            blake3: blake3_hash(&coord_wasm),
            size: None,
        },
    );
    artifacts.insert(
        "publisher.wasm".to_string(),
        SnapshotArtifact {
            url: format!("file://{}", module_path("toy_averager").display()),
            blake3: blake3_hash(&publisher_wasm),
            size: None,
        },
    );

    let control_channel = || RoleGrants {
        channels: vec![ChannelDecl {
            id: 0,
            name: "control".into(),
            class: 0,
            direction: 2,
            max_frame_bytes: 1 << 20,
            rate_per_min: 600,
            spool_frames: Some(256),
            replay_window: Some(1024),
            per_sender_quota: Some(64),
        }],
        ..RoleGrants::default()
    };
    let mut roles = BTreeMap::new();
    roles.insert(
        "coordinator".to_string(),
        RoleEntry {
            lane: "coordinator".into(),
            module: "coordinator.wasm".into(),
            abi: "vhc@2".into(),
            config: Value::Map(Vec::new()),
            grants: control_channel(),
            device_min: daemon_vhc_proto::DeviceMinimums::default(),
        },
    );
    roles.insert(
        WORKER_ROLE.to_string(),
        RoleEntry {
            lane: "trainer".into(),
            module: "publisher.wasm".into(),
            abi: "vhc@2".into(),
            // The publisher guest reads its config as raw bytes: a small CBOR unsigned is the
            // tick count — two publishes, then park until stopped.
            config: Value::from(2u8),
            grants: control_channel(),
            device_min: daemon_vhc_proto::DeviceMinimums {
                gpu: Some(1), // optional
                ram_bytes: Some(1 << 20),
                ..Default::default()
            },
        },
    );

    let env = GenesisEnvelope {
        run: RunSection {
            schema: GENESIS_SCHEMA_MAJOR,
            run_label: RUN_LABEL.to_string(),
            min_peers: 1,
            max_peers: 4,
            access: daemon_vhc_proto::envelope::Access::Org,
        },
        roles,
        artifacts,
        corpus_manifest: None,
        authority: Value::Null,
        transport: TransportSelection::default(),
        identities: Identities {
            coordinator: trusted_bases.first().copied(),
            coordinator_set: trusted_bases.to_vec(),
            upgrade_authority: Vec::new(),
        },
    };
    let author = SigningKey::from_bytes(&[0x42; 32]);
    let frozen = env.freeze(&author).expect("freeze genesis");
    let genesis_hash = frozen.run_id().0;
    let wire = SignedEnvelope {
        bytes: frozen.bytes().to_vec(),
        signature: *frozen.signature(),
        signer: *frozen.signer(),
    };
    (to_canonical_vec(&wire).expect("wire"), genesis_hash)
}

fn policy() -> JoinPolicy {
    JoinPolicy {
        mode: PolicyMode::Always,
        vram_cap_mb: 0,
        duty_cycle_pct: 100,
        schedule: None,
    }
}

struct Cut {
    writer: daemon_provision::CutWriter,
    reader: daemon_provision::CutReader,
    _child: daemon_provision::ChildGuard,
}

impl Cut {
    async fn send(&self, cmd: &Command) {
        let bytes = protocol::encode(cmd).expect("encode command");
        self.writer.send(&bytes).await.expect("send command");
    }

    async fn next(&mut self, deadline: Duration) -> Event {
        let bytes = tokio::time::timeout(deadline, self.reader.recv())
            .await
            .expect("worker event within the deadline")
            .expect("worker cut open");
        protocol::decode::<Event>(&bytes).expect("decodable event")
    }

    async fn until<T>(
        &mut self,
        deadline: Duration,
        mut pick: impl FnMut(&Event) -> Option<T>,
    ) -> T {
        let cut_deadline = tokio::time::Instant::now() + deadline;
        loop {
            let remaining = cut_deadline
                .checked_duration_since(tokio::time::Instant::now())
                .expect("event deadline exhausted");
            let ev = self.next(remaining).await;
            if let Event::Error { detail, .. } = &ev {
                panic!("worker error: {detail}");
            }
            if let Some(out) = pick(&ev) {
                return out;
            }
        }
    }
}

/// One worker seat: its own identity store, its own run-state root, its own subprocess.
struct Seat {
    identity: tempfile::TempDir,
    run_dir: tempfile::TempDir,
    cut: Cut,
}

async fn spawn_seat(tag: &str) -> Seat {
    let identity = tempfile::tempdir().expect("identity tempdir");
    VhcKeystore::open(identity.path()).expect("init keystore");
    let run_dir = tempfile::tempdir().expect("run-state tempdir");

    let session = SessionId::new(format!("live-attach-{tag}"));
    let spec = PlacementSpec {
        program: env!("CARGO_BIN_EXE_daemon-vhc-worker").into(),
        args: Vec::new(),
        env: vec![
            ("DAEMON_VHC_LANE_GPU_OPTIONAL".to_string(), "1".to_string()),
            (
                IDENTITY_DIR_ENV.to_string(),
                identity.path().display().to_string(),
            ),
            (
                RUN_DIR_ENV.to_string(),
                run_dir.path().display().to_string(),
            ),
            // NO in-process plane: the join must ride the live credentials or refuse.
        ],
    };
    let Placement { channel, child } = ProcessProvisioner::new()
        .place(&session, spec)
        .await
        .expect("spawn worker binary");
    let (writer, reader) = channel.split();
    let mut cut = Cut {
        writer,
        reader,
        _child: child,
    };
    let ready = cut.next(Duration::from_secs(30)).await;
    assert!(matches!(ready, Event::Ready { .. }), "got {ready:?}");
    Seat {
        identity,
        run_dir,
        cut,
    }
}

/// Assess + live-join one seat against the relay; returns once the role reports `running`.
async fn join_live(seat: &mut Seat, wire: &[u8], creds: &SessionCredentials, step: Duration) {
    seat.cut
        .send(&Command::AssessRun {
            envelope: wire.to_vec(),
            role: None,
        })
        .await;
    let elig = seat
        .cut
        .until(step, |ev| match ev {
            Event::Assessed(e) => Some(e.clone()),
            _ => None,
        })
        .await;
    assert!(elig.eligible, "publisher admits: {:?}", elig.reasons);
    // Play the node: mint incarnation 1's identity in this seat's keystore and stamp it into the
    // delivered tuple (the worker resolves the key + certificate read-only).
    let mut tuple = elig.admitted_tuple.clone().expect("assessed tuple");
    tuple.incarnation = 1;
    {
        let keystore = VhcKeystore::open(seat.identity.path()).expect("open keystore");
        daemon_vhc_session::provisioning::provision_run_identity(
            &keystore,
            &daemon_vhc_session::provisioning::ProvisionScope {
                run_label: RUN_LABEL,
                genesis_hash: tuple.genesis_hash,
                epoch: 0,
                role: &tuple.role,
                incarnation: 1,
                module_hash: tuple.module_hash,
            },
        )
        .expect("provision run identity");
    }
    seat.cut
        .send(&Command::JoinRun {
            run_id: RUN_LABEL.into(),
            coordinator: String::new(), // the credentials carry the ws_base
            credentials: creds.to_bytes().expect("encode credentials"),
            policy: policy(),
            admitted_tuple: Some(tuple),
        })
        .await;
    seat.cut
        .until(step, |ev| match ev {
            Event::RunPhase { run_id, phase, .. } if run_id == RUN_LABEL && phase == "running" => {
                Some(())
            }
            _ => None,
        })
        .await;
}

/// Leave a seat and return the classified terminal outcome.
async fn leave(seat: &mut Seat, step: Duration) -> TerminalOutcome {
    seat.cut
        .send(&Command::Leave {
            run_id: RUN_LABEL.into(),
            mode: LeaveMode::Immediate,
        })
        .await;
    seat.cut
        .until(step, |ev| match ev {
            Event::RunTerminated {
                run_id, outcome, ..
            } if run_id == RUN_LABEL => Some(outcome.clone()),
            _ => None,
        })
        .await
}

/// The §12.1 senders of the tag-12 signed-frame evidence records in a seat's durable journal.
fn journaled_frame_senders(seat: &Seat) -> Vec<[u8; 32]> {
    let jdir = journal_home::journal_dir(seat.run_dir.path(), RUN_LABEL, WORKER_ROLE, 1);
    let mut senders = Vec::new();
    for entry in std::fs::read_dir(&jdir).expect("journal dir") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("dvhcjrn") {
            continue;
        }
        let scan = daemon_vhc_journal::scan_file(&path).expect("scan segment");
        for record in scan.records {
            if let daemon_vhc_journal::record::Body::SignedFrame(sf) = record.body {
                senders.push(sf.sender.0);
            }
        }
    }
    senders
}

/// The headline live-attach proof (module docs above).
#[tokio::test]
async fn two_workers_exchange_certified_frames_over_the_live_ws_plane() {
    let relay = MockWsCoordinator::start().await;
    let step = Duration::from_secs(120);

    // Two seats with SEPARATE identity stores; genesis names both base identities as trusted
    // certificate issuers (the §12.3 trust roots).
    let mut a = spawn_seat("a").await;
    let mut b = spawn_seat("b").await;
    let base_a = peer_id(
        &VhcKeystore::open(a.identity.path())
            .expect("open keystore a")
            .base_identity()
            .expect("base a"),
    );
    let base_b = peer_id(
        &VhcKeystore::open(b.identity.path())
            .expect("open keystore b")
            .base_identity()
            .expect("base b"),
    );
    let (wire, genesis_hash) = genesis_wire(&[base_a, base_b]);

    let creds = SessionCredentials {
        genesis_hash,
        ws_base: Some(relay.base_url()),
        ws_auth: WsAuthSpec::None,
        iroh: None,
        presign_base: None, // the filesystem content store under the run-state root
        peer_certs: Vec::new(), // trust arrives ON the plane, as distribution records
        secret_ref: None,   // unauthenticated local relay lane
        expires_at_ms: 0,
        restore: None,
    };

    // B first (subscribed before A attaches), then A: A's certificate announcement + frames
    // relay to B in publish order.
    join_live(&mut b, &wire, &creds, step).await;
    relay.wait_peers(1).await;
    join_live(&mut a, &wire, &creds, step).await;
    relay.wait_peers(2).await;

    // Each seat announced its certificate then published two guest ticks: 2 announcements +
    // 4 frames must reach the relay. Poll (bounded) so the drain below never races the guests.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    while relay.received() < 6 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "relay saw only {} of 6 expected publications",
            relay.received()
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // A short grace for relay→B delivery + B's pump to journal the delivered frames (the
    // terminal commit below flushes them durably before we scan).
    tokio::time::sleep(Duration::from_secs(2)).await;

    let a_cert = VhcKeystore::open(a.identity.path())
        .expect("open keystore a")
        .run_certificate(RUN_LABEL, WORKER_ROLE, 1)
        .expect("read cert a")
        .expect("cert a persisted");
    let a_sender = a_cert.body.run_key.0;

    // Leave A (its publishes are already relayed), then B; both classify.
    assert_eq!(
        leave(&mut a, step).await,
        TerminalOutcome::Left { checkpoint: None }
    );
    assert_eq!(
        leave(&mut b, step).await,
        TerminalOutcome::Left { checkpoint: None }
    );

    // B's journal carries tag-12 evidence records from A's certified sender: the inbound frames
    // VERIFIED (chain to a genesis-trusted base via the on-plane distribution record) and
    // DELIVERED below the pump. The announcement itself is not journaled (it is not a frame).
    let senders = journaled_frame_senders(&b);
    assert!(
        senders.contains(&a_sender),
        "B journaled delivered frames from A's certified sender \
         (distribution → verification → delivery); senders seen: {}",
        senders.len()
    );

    // And the relay actually carried the full traffic (announcements + frames, both peers).
    assert!(
        relay.received() >= 6,
        "relay saw {} frames",
        relay.received()
    );
}
