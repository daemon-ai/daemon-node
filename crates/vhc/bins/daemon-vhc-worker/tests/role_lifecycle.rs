// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The responsive worker command loop over the REAL binary: `JoinRun` spawns a role session and
//! returns to the loop immediately, commands keep answering while roles run, two roles run
//! concurrently, `Leave` is real (a classified terminal event), and `Shutdown` drains every
//! session. Driven directly over the provision cut so command/event interleaving is observable
//! (the supervisor's request/reply seam hides it).
//!
//! The joined module is the timer-driven publisher guest: it publishes a few frames, then parks
//! in its event loop until told to stop — a live role instance with no external dependencies.
//! The worker binds the in-process plane seat (`DAEMON_VHC_INPROC_PLANE=1`), so the whole
//! lifecycle runs through the production role-session path with no network.
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
use daemon_vhc_proto::genesis::{
    ChannelDecl, Identities, RoleEntry, RoleGrants, RunSection, SnapshotArtifact,
    TransportSelection, GENESIS_SCHEMA_MAJOR,
};
use daemon_vhc_proto::{
    blake3_hash, to_canonical_vec, GenesisEnvelope, SignedEnvelope, SigningKey,
};
use daemon_vhc_session::keystore::{VhcKeystore, IDENTITY_DIR_ENV};
use daemon_vhc_session::protocol::AdmittedTuple;
use daemon_vhc_session::protocol::{
    self, Command, Event, JoinPolicy, LeaveMode, PolicyMode, TerminalOutcome,
};
use daemon_vhc_session::provisioning::{provision_run_identity, ProvisionScope};

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

/// Author + freeze a genesis for `run_label`: the publisher guest as the worker role (its config
/// is the guest's raw byte contract — a small CBOR integer whose encoding is the tick count),
/// the pinned consensus coordinator blob as the coordinator role (declared, never launched
/// here — the seat-claim mechanism owns coordinator launch).
fn genesis_wire(run_label: &str) -> Vec<u8> {
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
        "publisher".to_string(),
        RoleEntry {
            lane: "trainer".into(),
            module: "publisher.wasm".into(),
            abi: "vhc@2".into(),
            // The publisher guest reads its config as raw bytes: byte 0 is the tick count. A
            // small CBOR unsigned encodes as that single byte, so `2` = two publishes, then
            // park until stopped.
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
            run_label: run_label.to_string(),
            min_peers: 1,
            max_peers: 4,
            access: daemon_vhc_proto::envelope::Access::Org,
        },
        roles,
        artifacts,
        corpus_manifest: None,
        authority: Value::Null,
        transport: TransportSelection::default(),
        identities: Identities::default(),
    };
    let author = SigningKey::from_bytes(&[0x42; 32]);
    let frozen = env.freeze(&author).expect("freeze genesis");
    let wire = SignedEnvelope {
        bytes: frozen.bytes().to_vec(),
        signature: *frozen.signature(),
        signer: *frozen.signer(),
    };
    to_canonical_vec(&wire).expect("wire")
}

/// Play the node's authorship role for a direct-drive test: stamp the minted incarnation into
/// the assessed tuple and PROVISION the per-run identity (mint key + issue certificate under the
/// base identity) in the keystore the worker resolves read-only. Returns the delivery tuple.
fn provision_and_stamp(
    identity_dir: &std::path::Path,
    run_label: &str,
    mut tuple: AdmittedTuple,
    incarnation: u64,
) -> AdmittedTuple {
    tuple.incarnation = incarnation;
    let keystore = VhcKeystore::open(identity_dir).expect("open keystore");
    provision_run_identity(
        &keystore,
        &ProvisionScope {
            run_label,
            genesis_hash: tuple.genesis_hash,
            epoch: 0,
            role: &tuple.role,
            incarnation,
            module_hash: tuple.module_hash,
        },
    )
    .expect("provision run identity");
    tuple
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

    /// The next decodable event within `deadline`; worker errors panic the test.
    async fn next(&mut self, deadline: Duration) -> Event {
        let bytes = tokio::time::timeout(deadline, self.reader.recv())
            .await
            .expect("worker event within the deadline")
            .expect("worker cut open");
        protocol::decode::<Event>(&bytes).expect("decodable event")
    }

    /// Read events until `pick` accepts one, failing loud on worker errors.
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

async fn spawn_worker(identity_dir: &std::path::Path, run_dir: &std::path::Path) -> Cut {
    let session = SessionId::new("role-lifecycle-worker");
    let spec = PlacementSpec {
        program: env!("CARGO_BIN_EXE_daemon-vhc-worker").into(),
        args: Vec::new(),
        env: vec![
            ("DAEMON_VHC_LANE_GPU_OPTIONAL".to_string(), "1".to_string()),
            (
                IDENTITY_DIR_ENV.to_string(),
                identity_dir.display().to_string(),
            ),
            // The node-delivered run-state root: every role instance journals durably under it
            // (`<root>/<blake3(label)>/<role>-<incarnation>/journal`).
            (
                daemon_vhc_session::journal_home::RUN_DIR_ENV.to_string(),
                run_dir.display().to_string(),
            ),
            // The in-process plane seat: the production role-session path with no network.
            ("DAEMON_VHC_INPROC_PLANE".to_string(), "1".to_string()),
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
    // The spawn handshake.
    let ready = cut.next(Duration::from_secs(30)).await;
    assert!(matches!(ready, Event::Ready { .. }), "got {ready:?}");
    cut
}

/// The whole responsive lifecycle over one worker process: join returns immediately; the loop
/// answers while a role runs; a second role runs concurrently; leave is a classified terminal;
/// shutdown drains the rest.
#[tokio::test]
async fn command_loop_stays_responsive_across_join_leave_shutdown() {
    let identity = tempfile::tempdir().expect("identity tempdir");
    VhcKeystore::open(identity.path()).expect("init keystore");
    let run_dir = tempfile::tempdir().expect("run-state tempdir");
    let mut cut = spawn_worker(identity.path(), run_dir.path()).await;
    let step = Duration::from_secs(120);

    // Role A: assess + join. The join reply is immediate (phase `joining`), and the role then
    // reports `running` — the command loop never blocks on the run.
    cut.send(&Command::AssessRun {
        envelope: genesis_wire("role-a"),
        role: None,
        switch_target: None,
    })
    .await;
    let elig = cut
        .until(step, |ev| match ev {
            Event::Assessed(e) => Some(e.clone()),
            _ => None,
        })
        .await;
    assert!(elig.eligible, "publisher admits: {:?}", elig.reasons);
    let tuple_a = provision_and_stamp(
        identity.path(),
        "role-a",
        elig.admitted_tuple.clone().expect("assessed tuple"),
        1,
    );
    cut.send(&Command::JoinRun {
        run_id: "role-a".into(),
        coordinator: String::new(),
        credentials: Vec::new(),
        policy: policy(),
        admitted_tuple: Some(Box::new(tuple_a)),
    })
    .await;
    cut.until(step, |ev| match ev {
        Event::RunPhase {
            run_id,
            phase,
            generation,
            ..
        } if run_id == "role-a" && phase == "running" => Some(*generation),
        _ => None,
    })
    .await;

    // Loop responsiveness while the role runs: a liveness probe answers at once.
    cut.send(&Command::Ping).await;
    cut.until(Duration::from_secs(10), |ev| {
        matches!(ev, Event::Pong).then_some(())
    })
    .await;

    // Role B joins CONCURRENTLY (a second assessment re-resolves the cached run; the map holds
    // both live handles).
    cut.send(&Command::AssessRun {
        envelope: genesis_wire("role-b"),
        role: None,
        switch_target: None,
    })
    .await;
    let elig_b = cut
        .until(step, |ev| match ev {
            Event::Assessed(e) => Some(e.clone()),
            _ => None,
        })
        .await;
    assert!(elig_b.eligible, "second role admits: {:?}", elig_b.reasons);
    let tuple_b = provision_and_stamp(
        identity.path(),
        "role-b",
        elig_b.admitted_tuple.clone().expect("assessed tuple"),
        1,
    );
    cut.send(&Command::JoinRun {
        run_id: "role-b".into(),
        coordinator: String::new(),
        credentials: Vec::new(),
        policy: policy(),
        admitted_tuple: Some(Box::new(tuple_b)),
    })
    .await;
    cut.until(step, |ev| match ev {
        Event::RunPhase { run_id, phase, .. } if run_id == "role-b" && phase == "running" => {
            Some(())
        }
        _ => None,
    })
    .await;

    // The governor lever lands mid-run without disturbing the loop (hard pause on, then off).
    cut.send(&Command::Throttle {
        vram_cap_mb: None,
        duty_cycle_pct: Some(0),
        paused: true,
    })
    .await;
    cut.send(&Command::Throttle {
        vram_cap_mb: None,
        duty_cycle_pct: Some(100),
        paused: false,
    })
    .await;
    cut.send(&Command::Ping).await;
    cut.until(Duration::from_secs(10), |ev| {
        matches!(ev, Event::Pong).then_some(())
    })
    .await;

    // Leave role A: a real leave — the classified terminal event names the run and its
    // generation, and role B keeps running (its terminal has not been emitted).
    cut.send(&Command::Leave {
        run_id: "role-a".into(),
        mode: LeaveMode::Immediate,
    })
    .await;
    let (gen_a, outcome_a) = cut
        .until(step, |ev| match ev {
            Event::RunTerminated {
                run_id,
                generation,
                outcome,
            } if run_id == "role-a" => Some((*generation, outcome.clone())),
            _ => None,
        })
        .await;
    assert_eq!(gen_a, 1);
    assert_eq!(outcome_a, TerminalOutcome::Left { checkpoint: None });
    // The role journaled DURABLY into its node-delivered per-incarnation home: the terminated
    // instance's first segment exists on disk and is non-empty (ABI §8 — the in-memory sink is
    // gone from the referenced path; journals outlive the run as the oracle's product input).
    let journal_a =
        daemon_vhc_session::journal_home::journal_dir(run_dir.path(), "role-a", "publisher", 1);
    let segment0 = journal_a.join("segment-00000000.dvhcjrn");
    let seg_meta = std::fs::metadata(&segment0)
        .unwrap_or_else(|e| panic!("durable segment at {}: {e}", segment0.display()));
    assert!(seg_meta.len() > 0, "segment 0 carries records");
    cut.send(&Command::Ping).await;
    cut.until(Duration::from_secs(10), |ev| {
        matches!(ev, Event::Pong).then_some(())
    })
    .await;

    // Shutdown drains role B: its classified terminal arrives before the process exits.
    cut.send(&Command::Shutdown).await;
    let outcome_b = cut
        .until(step, |ev| match ev {
            Event::RunTerminated {
                run_id, outcome, ..
            } if run_id == "role-b" => Some(outcome.clone()),
            _ => None,
        })
        .await;
    assert_eq!(outcome_b, TerminalOutcome::Left { checkpoint: None });
}

/// A production join REFUSES typed when no per-run identity was provisioned for the delivered
/// incarnation — the worker never mints (base-key custody stays with the node). The negative
/// incarnation test is real now: the delivered tuple names an incarnation the keystore has no
/// key for.
#[tokio::test]
async fn join_refuses_when_no_identity_was_provisioned() {
    let identity = tempfile::tempdir().expect("identity tempdir");
    VhcKeystore::open(identity.path()).expect("init keystore");
    let run_dir = tempfile::tempdir().expect("run-state tempdir");
    let mut cut = spawn_worker(identity.path(), run_dir.path()).await;
    let step = Duration::from_secs(120);

    cut.send(&Command::AssessRun {
        envelope: genesis_wire("role-unprovisioned"),
        role: None,
        switch_target: None,
    })
    .await;
    let elig = cut
        .until(step, |ev| match ev {
            Event::Assessed(e) => Some(e.clone()),
            _ => None,
        })
        .await;
    assert!(elig.eligible);

    // Deliver a tuple stamped with an incarnation the keystore has NO provisioned key for —
    // NOTHING was provisioned. The join must refuse typed, never mint, never run.
    let mut tuple = elig.admitted_tuple.clone().expect("assessed tuple");
    tuple.incarnation = 99;
    cut.send(&Command::JoinRun {
        run_id: "role-unprovisioned".into(),
        coordinator: String::new(),
        credentials: Vec::new(),
        policy: policy(),
        admitted_tuple: Some(Box::new(tuple)),
    })
    .await;
    // Read raw (the `until` helper treats a worker Error as a failure; here the Error IS the
    // expected outcome).
    let detail = loop {
        match cut.next(step).await {
            Event::Error { detail, .. } => break detail,
            Event::RunPhase { run_id, phase, .. }
                if run_id == "role-unprovisioned" && phase == "running" =>
            {
                panic!("an unprovisioned join must never run")
            }
            _ => {}
        }
    };
    assert!(
        detail.contains("no per-run identity was provisioned"),
        "typed no-identity refusal, got: {detail}"
    );

    cut.send(&Command::Shutdown).await;
}
