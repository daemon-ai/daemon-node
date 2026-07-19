// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![cfg(feature = "vhc-net")]

//! The coordinator-seat two-process smoke: a seat-claiming COORDINATOR-role node process and a
//! TRAINER node process over the promoted relay fixture, with real keystores, durable journals,
//! and the filesystem content store.
//!
//! The test process plays the node for both seats:
//! 1. it claims the coordinator seat in the normative CAS fold (`FakeSeatRegistry`) with the
//!    coordinator's node-provisioned identity, then launches the coordinator role through the
//!    UNCHANGED role-session path (directed role — zero role branching in the session);
//! 2. it peer-authorizes the stored lease (signature + genesis-trusted base + supersession floor)
//!    and hands the trainer the coordinator's endpoint + certificate as bootstrap trust;
//! 3. the trainer joins and its frames verify against the seat holder's certificate.
//!
//! Fencing-is-safe-not-seamless is pinned: a live contender's claim loses TYPED against the CAS,
//! and the coordinator role runs to a clean leave. (The seat is role-agnostic by design, so the
//! smoke points the coordinator role at the same publisher guest — the proof is the seat lifecycle
//! + directed launch through the one session path, not a specific coordinator algorithm.)

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
use daemon_vhc_net::FakeSeatRegistry;
use daemon_vhc_node::seat::{author_claim, authorize_incumbent, derive_bid, CoordinatorSeat};
use daemon_vhc_proto::genesis::{
    ChannelDecl, Identities, RoleEntry, RoleGrants, RunSection, SnapshotArtifact,
    TransportSelection, GENESIS_SCHEMA_MAJOR,
};
use daemon_vhc_proto::{
    blake3_hash, peer_id, to_canonical_vec, ControlEndpoint, GenesisEnvelope, PeerId,
    RevocationLedger, SeatDecision, SeatState, SignedEnvelope, SigningKey, DEFAULT_SEAT_SKEW_MS,
};
use daemon_vhc_session::journal_home::RUN_DIR_ENV;
use daemon_vhc_session::keystore::{VhcKeystore, IDENTITY_DIR_ENV};
use daemon_vhc_session::protocol::{
    self, AdmittedTuple, Command, Event, JoinPolicy, PolicyMode, SessionCredentials,
    TerminalOutcome, WsAuthSpec,
};
use daemon_vhc_session::provisioning::{provision_run_identity, ProvisionScope};

const RUN_LABEL: &str = "seat-smoke";
const COORD_ROLE: &str = "coordinator";
const TRAINER_ROLE: &str = "trainer";

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

fn publisher_wasm() -> Vec<u8> {
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
    let path = guests_root().join("target/wasm32-unknown-unknown/release/toy_averager.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// A genesis naming a coordinator role AND a trainer role — BOTH at the publisher guest (the seat
/// is role-agnostic). `[identities]` names both node base identities as trusted issuers.
fn genesis_wire(trusted_bases: &[PeerId]) -> (Vec<u8>, [u8; 32]) {
    let wasm = publisher_wasm();
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "role.wasm".to_string(),
        SnapshotArtifact {
            url: format!(
                "file://{}",
                guests_root()
                    .join("target/wasm32-unknown-unknown/release/toy_averager.wasm")
                    .display()
            ),
            blake3: blake3_hash(&wasm),
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
    let role_entry = |lane: &str| RoleEntry {
        lane: lane.into(),
        module: "role.wasm".into(),
        abi: "vhc@2".into(),
        config: Value::from(2u8), // two publishes, then park until stopped
        grants: control_channel(),
        device_min: daemon_vhc_proto::DeviceMinimums {
            gpu: Some(1),
            ram_bytes: Some(1 << 20),
            ..Default::default()
        },
    };
    let mut roles = BTreeMap::new();
    roles.insert(COORD_ROLE.to_string(), role_entry("coordinator"));
    roles.insert(TRAINER_ROLE.to_string(), role_entry("trainer"));

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
        authority: Value::Null,
        transport: TransportSelection::default(),
        corpus_manifest: None,
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

struct Seat {
    identity: tempfile::TempDir,
    // Held so the run-state dir (journals + fs payload) outlives the seat; not read directly.
    _run_dir: tempfile::TempDir,
    cut: Cut,
}

async fn spawn_seat(tag: &str) -> Seat {
    let identity = tempfile::tempdir().expect("identity tempdir");
    VhcKeystore::open(identity.path()).expect("init keystore");
    let run_dir = tempfile::tempdir().expect("run-state tempdir");
    let session = SessionId::new(format!("seat-smoke-{tag}"));
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
        _run_dir: run_dir,
        cut,
    }
}

/// Assess a directed role and return the assessed tuple.
async fn assess_role(seat: &mut Seat, wire: &[u8], role: &str, step: Duration) -> AdmittedTuple {
    seat.cut
        .send(&Command::AssessRun {
            envelope: wire.to_vec(),
            role: Some(role.to_string()),
        })
        .await;
    let elig = seat
        .cut
        .until(step, |ev| match ev {
            Event::Assessed(e) => Some(e.clone()),
            _ => None,
        })
        .await;
    assert!(elig.eligible, "{role} admits: {:?}", elig.reasons);
    elig.admitted_tuple.expect("assessed tuple")
}

/// Provision identity for `(role, incarnation)` in the seat's keystore (the node's job).
fn provision(seat: &Seat, tuple: &AdmittedTuple, role: &str, incarnation: u64) {
    let keystore = VhcKeystore::open(seat.identity.path()).expect("open keystore");
    provision_run_identity(
        &keystore,
        &ProvisionScope {
            run_label: RUN_LABEL,
            genesis_hash: tuple.genesis_hash,
            epoch: 0,
            role,
            incarnation,
            module_hash: tuple.module_hash,
        },
    )
    .expect("provision run identity");
}

async fn join_role(
    seat: &mut Seat,
    mut tuple: AdmittedTuple,
    role: &str,
    incarnation: u64,
    creds: &SessionCredentials,
    step: Duration,
) {
    tuple.incarnation = incarnation;
    seat.cut
        .send(&Command::JoinRun {
            run_id: RUN_LABEL.into(),
            coordinator: String::new(),
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
    let _ = role;
}

async fn leave(seat: &mut Seat, step: Duration) -> TerminalOutcome {
    seat.cut
        .send(&Command::Leave {
            run_id: RUN_LABEL.into(),
            mode: daemon_vhc_session::protocol::LeaveMode::Immediate,
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

#[tokio::test]
async fn coordinator_seat_claim_launch_and_trainer_lease_resolve() {
    let relay = MockWsCoordinator::start().await;
    let registry = FakeSeatRegistry::new();
    let step = Duration::from_secs(120);
    let now = 1_000_000u64;

    let mut coord = spawn_seat("coord").await;
    let mut trainer = spawn_seat("trainer").await;
    let coord_base = peer_id(
        &VhcKeystore::open(coord.identity.path())
            .unwrap()
            .base_identity()
            .unwrap(),
    );
    let trainer_base = peer_id(
        &VhcKeystore::open(trainer.identity.path())
            .unwrap()
            .base_identity()
            .unwrap(),
    );
    let (wire, genesis_hash) = genesis_wire(&[coord_base, trainer_base]);
    let ws = relay.base_url();

    // -- the coordinator node claims the seat, then launches the coordinator role -------------
    let coord_tuple = assess_role(&mut coord, &wire, COORD_ROLE, step).await;
    let bid = derive_bid(
        &registry.read(RUN_LABEL, COORD_ROLE),
        now,
        DEFAULT_SEAT_SKEW_MS,
    )
    .expect("virgin slot bids");
    // Provision the coordinator identity at the bid incarnation, then author + CAS the lease.
    provision(&coord, &coord_tuple, COORD_ROLE, bid);
    let seat_scope = CoordinatorSeat {
        run_label: RUN_LABEL,
        genesis_hash,
        role: COORD_ROLE,
        epoch: 0,
        module_hash: coord_tuple.module_hash,
        endpoint: ControlEndpoint {
            ws: Some(ws.clone()),
            iroh_ticket: None,
        },
    };
    let lease = author_claim(
        &VhcKeystore::open(coord.identity.path()).unwrap(),
        &seat_scope,
        bid,
        now,
    )
    .expect("author claim");
    assert_eq!(
        registry.claim(RUN_LABEL, &lease, now).decision,
        SeatDecision::Accepted,
        "the coordinator wins the virgin seat"
    );
    let coord_creds = SessionCredentials {
        genesis_hash,
        ws_base: Some(ws.clone()),
        ws_auth: WsAuthSpec::None,
        iroh: None,
        presign_base: None,
        peer_certs: Vec::new(),
        secret_ref: None,
        expires_at_ms: 0,
        restore: None,
    };
    join_role(&mut coord, coord_tuple, COORD_ROLE, bid, &coord_creds, step).await;

    // -- a live contender loses TYPED against the CAS (fencing is safety) ----------------------
    let contender_dir = tempfile::tempdir().unwrap();
    let contender = VhcKeystore::open(contender_dir.path()).unwrap();
    provision_run_identity(
        &contender,
        &ProvisionScope {
            run_label: RUN_LABEL,
            genesis_hash,
            epoch: 0,
            role: COORD_ROLE,
            incarnation: bid + 1,
            module_hash: coord_tuple_module(&registry),
        },
    )
    .unwrap();
    let contender_lease = author_claim(&contender, &seat_scope, bid + 1, now + 1_000).unwrap();
    assert_eq!(
        registry
            .claim(RUN_LABEL, &contender_lease, now + 1_000)
            .decision,
        SeatDecision::RejectedHeld,
        "a contender must not displace the live coordinator lease"
    );

    // -- the TRAINER node resolves the seat: authorize the incumbent, then join ----------------
    let stored = match registry.read(RUN_LABEL, COORD_ROLE) {
        SeatState::Leased(l) => *l,
        other => panic!("expected a lease: {other:?}"),
    };
    let authorized = authorize_incumbent(
        &stored,
        &[coord_base, trainer_base],
        &RevocationLedger::new(),
        now + 2_000,
        DEFAULT_SEAT_SKEW_MS,
    )
    .expect("the trainer authorizes the coordinator's lease");
    let coord_ws = authorized.endpoint.ws.clone().expect("ws endpoint");

    let trainer_tuple = assess_role(&mut trainer, &wire, TRAINER_ROLE, step).await;
    provision(&trainer, &trainer_tuple, TRAINER_ROLE, 1);
    let trainer_creds = SessionCredentials {
        genesis_hash,
        ws_base: Some(coord_ws), // the seat-published endpoint
        ws_auth: WsAuthSpec::None,
        iroh: None,
        presign_base: None,
        // Bootstrap trust: the seat holder's certificate (so the trainer verifies its frames).
        peer_certs: vec![authorized.certificate.clone()],
        secret_ref: None,
        expires_at_ms: 0,
        restore: None,
    };
    join_role(
        &mut trainer,
        trainer_tuple,
        TRAINER_ROLE,
        1,
        &trainer_creds,
        step,
    )
    .await;

    // Both roles reached `running` over the relay: the seat mechanism launched the coordinator
    // through the same session path and the trainer joined against the verified lease endpoint.
    relay.wait_peers(2).await;

    // Clean leave for both — the coordinator role runs to a classified terminal.
    assert_eq!(
        leave(&mut coord, step).await,
        TerminalOutcome::Left { checkpoint: None }
    );
    assert_eq!(
        leave(&mut trainer, step).await,
        TerminalOutcome::Left { checkpoint: None }
    );
}

/// The coordinator module hash, read once from the seeded lease's scope (a helper to keep the
/// contender's provisioning scope identical to the incumbent's).
fn coord_tuple_module(registry: &FakeSeatRegistry) -> [u8; 32] {
    match registry.read(RUN_LABEL, COORD_ROLE) {
        SeatState::Leased(l) => l.body.module_hash.0,
        _ => panic!("seat should be leased"),
    }
}
