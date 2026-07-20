// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The live module switch over the REAL worker binary (ABI §10.3): the certificate re-issuance
//! handshake end to end — a switch whose post-switch identity was never provisioned refuses
//! typed with the old module untouched; after the node-side provisioning (the new incarnation's
//! key and re-issued certificate in the keystore) the same switch activates, the generation
//! advances, the command loop keeps answering, and the DURABLE journal continues as one file
//! series across the seam (sealed retired span, new identity header, publish sequences
//! restarting at 0).
//!
//! The joined module is the migrate drill pair's FROM side (it snapshots its counter at the
//! drain fence); the target is the TO side (it restores the counter through `da_migrate`). The
//! target bytes reach the worker through the explicit hash-verified module-source override — the
//! upgrade-time peer of the join's module override.
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
use daemon_vhc_session::journal_home::RUN_DIR_ENV;
use daemon_vhc_session::keystore::{VhcKeystore, IDENTITY_DIR_ENV};
use daemon_vhc_session::protocol::{
    self, AdmittedTuple, Command, Event, JoinPolicy, LeaveMode, PolicyMode, TerminalOutcome,
};
use daemon_vhc_session::provisioning::{provision_run_identity, ProvisionScope};

const ROLE: &str = "counter";

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

/// Author + freeze a genesis whose worker role runs the migrate drill's FROM module.
fn genesis_wire(run_label: &str) -> Vec<u8> {
    let coord_wasm = std::fs::read(module_path("coordinator_quorum")).expect("coordinator blob");
    let from_wasm = std::fs::read(module_path("test_migrate_old")).expect("drill FROM blob");

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
        "counter.wasm".to_string(),
        SnapshotArtifact {
            url: format!("file://{}", module_path("test_migrate_old").display()),
            blake3: blake3_hash(&from_wasm),
            size: None,
        },
    );

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
        ROLE.to_string(),
        RoleEntry {
            lane: "trainer".into(),
            module: "counter.wasm".into(),
            abi: "vhc@2".into(),
            // The drill module reads its config as raw bytes: byte 0 seeds the counter. A small
            // CBOR unsigned encodes as that single byte.
            config: Value::from(0u8),
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

fn control_channel() -> RoleGrants {
    RoleGrants {
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
    }
}

fn policy() -> JoinPolicy {
    JoinPolicy {
        mode: PolicyMode::Always,
        vram_cap_mb: 0,
        duty_cycle_pct: 100,
        schedule: None,
    }
}

/// The admitted tuple a node would author for the switch target at `epoch`/`incarnation`: the
/// artifact-addressed fields rederived exactly as the session will (grants from the genesis role
/// grants; the claim re-evaluated over the admitted funnel).
fn switch_tuple(
    genesis_hash: [u8; 32],
    new_wasm: &[u8],
    incarnation: u64,
) -> (AdmittedTuple, [u8; 32], [u8; 32]) {
    let new_module = *blake3::hash(new_wasm).as_bytes();
    let worker =
        daemon_vhc_host::Worker::new(daemon_vhc_host::EngineConfig::default()).expect("engine");
    let linked = daemon_vhc_host::linked_worlds(&worker, new_wasm).expect("linked worlds");
    let grants =
        daemon_vhc_proto::GrantsDoc::author(&linked, &control_channel()).to_canonical_bytes();
    let grants_hash = *blake3::hash(&grants).as_bytes();
    let envelope_grants = daemon_vhc_host::run::EnvelopeRoleGrants {
        grants: control_channel(),
        run_artifacts: std::collections::BTreeSet::new(),
    };
    // The admitted role config carries UNCHANGED across the switch: the drill role's config is
    // the single CBOR byte `0` (the counter seed) — the claim + config hash are computed over
    // exactly what the migrated instance initializes with.
    let config = to_canonical_vec(&Value::from(0u8)).expect("role config bytes");
    let admission = daemon_vhc_host::run::admit(
        &worker,
        new_wasm,
        Some(&new_module),
        &config,
        &grants,
        &daemon_vhc_host::run::ParticipationLane {
            gpu: 1,
            vram_bytes: 0,
            ram_bytes: 0,
            disk_bytes: 0,
            ..daemon_vhc_host::run::ParticipationLane::trainer_launch_defaults()
        },
        &daemon_vhc_host::run::DeviceProfile {
            gpu: true,
            vram_bytes: 8 << 30,
            ram_bytes: 16 << 30,
            disk_bytes: 100 << 30,
        },
        &daemon_vhc_host::run::OwnerPolicy {
            participation_enabled: true,
            vram_cap_bytes: 0,
            host_cap_bytes: 0,
        },
        None,
        Some(&envelope_grants),
    )
    .expect("target admits");
    let tuple = AdmittedTuple {
        module_hash: new_module,
        config_hash: *blake3::hash(&config).as_bytes(),
        grants_hash,
        claim_hash: *blake3::hash(&admission.claim_bytes).as_bytes(),
        genesis_hash,
        role: ROLE.to_string(),
        incarnation,
        device_profile_rev: 0,
        owner_policy_rev: 0,
        backend: "cpu".to_string(),
        gpu_index: 0,
    };
    (tuple, new_module, grants_hash)
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

async fn spawn_worker(identity_dir: &std::path::Path, run_dir: &std::path::Path) -> Cut {
    let session = SessionId::new("module-switch-worker");
    let spec = PlacementSpec {
        program: env!("CARGO_BIN_EXE_daemon-vhc-worker").into(),
        args: Vec::new(),
        env: vec![
            ("DAEMON_VHC_LANE_GPU_OPTIONAL".to_string(), "1".to_string()),
            (
                IDENTITY_DIR_ENV.to_string(),
                identity_dir.display().to_string(),
            ),
            (RUN_DIR_ENV.to_string(), run_dir.display().to_string()),
            ("DAEMON_VHC_INPROC_PLANE".to_string(), "1".to_string()),
            // The switch target's module-source override (hash-verified at pre-flight): the
            // upgrade-time peer of the join's module override.
            (
                "DAEMON_VHC_SWITCH_MODULE".to_string(),
                module_path("test_migrate_new").display().to_string(),
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
    cut
}

/// The re-issuance handshake over the real binary: an unprovisioned switch refuses typed with
/// the old module untouched; the provisioned switch activates under the new incarnation; the
/// durable journal continues as one chained file series across the seam.
#[tokio::test]
async fn switch_reissues_identity_and_continues_the_durable_journal() {
    let identity = tempfile::tempdir().expect("identity tempdir");
    VhcKeystore::open(identity.path()).expect("init keystore");
    let run_dir = tempfile::tempdir().expect("run-state tempdir");
    let mut cut = spawn_worker(identity.path(), run_dir.path()).await;
    let step = Duration::from_secs(120);
    let run_label = "switch-run";

    // Assess + node-side provisioning + join, exactly the lifecycle suite's authorship shape.
    cut.send(&Command::AssessRun {
        envelope: genesis_wire(run_label),
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
    assert!(
        elig.eligible,
        "drill FROM module admits: {:?}",
        elig.reasons
    );
    let mut tuple = elig.admitted_tuple.clone().expect("assessed tuple");
    tuple.incarnation = 1;
    let keystore = VhcKeystore::open(identity.path()).expect("open keystore");
    provision_run_identity(
        &keystore,
        &ProvisionScope {
            run_label,
            genesis_hash: tuple.genesis_hash,
            epoch: 0,
            role: ROLE,
            incarnation: 1,
            module_hash: tuple.module_hash,
        },
    )
    .expect("provision the join identity");
    cut.send(&Command::JoinRun {
        run_id: run_label.into(),
        coordinator: String::new(),
        credentials: Vec::new(),
        policy: policy(),
        admitted_tuple: Some(Box::new(tuple.clone())),
    })
    .await;
    cut.until(step, |ev| match ev {
        Event::RunPhase {
            phase, generation, ..
        } if phase == "running" => Some(*generation),
        _ => None,
    })
    .await;

    // The switch target + the node-authored post-switch tuple (incarnation 2, epoch 1).
    let new_wasm = std::fs::read(module_path("test_migrate_new")).expect("drill TO blob");
    let (switch_tuple, new_module, grants_hash) = switch_tuple(tuple.genesis_hash, &new_wasm, 2);

    // UNPROVISIONED: the re-issuance handshake fails closed — typed refusal, old module intact.
    cut.send(&Command::SwitchModule {
        run_id: run_label.into(),
        epoch: 1,
        role: ROLE.into(),
        new_module,
        grants_hash,
        deadline_ms: 10_000,
        admitted_tuple: Some(Box::new(switch_tuple.clone())),
    })
    .await;
    let reason = cut
        .until(step, |ev| match ev {
            Event::SwitchRefused { reason, .. } => Some(reason.clone()),
            Event::ModuleSwitched { .. } => {
                panic!("an unprovisioned switch must never activate")
            }
            _ => None,
        })
        .await;
    assert!(
        reason.contains("no per-run identity was provisioned"),
        "typed refusal names the missing provisioning: {reason}"
    );
    // The loop answers and the old role keeps running.
    cut.send(&Command::Ping).await;
    cut.until(Duration::from_secs(10), |ev| {
        matches!(ev, Event::Pong).then_some(())
    })
    .await;

    // PROVISIONED: the node mints the new incarnation's key and re-issues the certificate bound
    // to (run, epoch 1, role, incarnation 2, new module), then the same switch activates.
    provision_run_identity(
        &keystore,
        &ProvisionScope {
            run_label,
            genesis_hash: tuple.genesis_hash,
            epoch: 1,
            role: ROLE,
            incarnation: 2,
            module_hash: new_module,
        },
    )
    .expect("provision the post-switch identity");
    cut.send(&Command::SwitchModule {
        run_id: run_label.into(),
        epoch: 1,
        role: ROLE.into(),
        new_module,
        grants_hash,
        deadline_ms: 10_000,
        admitted_tuple: Some(Box::new(switch_tuple)),
    })
    .await;
    let (epoch, module, generation) = cut
        .until(step, |ev| match ev {
            Event::ModuleSwitched {
                epoch,
                module,
                generation,
                ..
            } => Some((*epoch, *module, *generation)),
            Event::SwitchRefused { reason, .. } => panic!("provisioned switch refused: {reason}"),
            Event::RunTerminated { outcome, .. } => panic!("switch left the run: {outcome:?}"),
            _ => None,
        })
        .await;
    assert_eq!(epoch, 1);
    assert_eq!(module, new_module);
    assert_eq!(generation, 2, "the switch minted the new incarnation");

    // The loop still answers; the migrated role leaves cleanly under its NEW generation.
    cut.send(&Command::Ping).await;
    cut.until(Duration::from_secs(10), |ev| {
        matches!(ev, Event::Pong).then_some(())
    })
    .await;
    cut.send(&Command::Leave {
        run_id: run_label.into(),
        mode: LeaveMode::Immediate,
    })
    .await;
    let (gen_t, outcome) = cut
        .until(step, |ev| match ev {
            Event::RunTerminated {
                generation,
                outcome,
                ..
            } => Some((*generation, outcome.clone())),
            _ => None,
        })
        .await;
    assert_eq!(gen_t, 2);
    assert_eq!(outcome, TerminalOutcome::Left { checkpoint: None });
    cut.send(&Command::Shutdown).await;

    // The DURABLE seam: one chained file series in the JOIN incarnation's home — the retired
    // span sealed under identity (epoch 0, instance 1), the continuation under (epoch 1,
    // instance 2), with the announce publish opening the new stream at seq 0.
    let jdir = daemon_vhc_session::journal_home::journal_dir(run_dir.path(), run_label, ROLE, 1);
    let mut ords: Vec<u64> = std::fs::read_dir(&jdir)
        .expect("journal home exists")
        .filter_map(|e| {
            let name = e.ok()?.file_name();
            let name = name.to_string_lossy();
            name.strip_prefix("segment-")?
                .strip_suffix(".dvhcjrn")?
                .parse()
                .ok()
        })
        .collect();
    ords.sort_unstable();
    assert!(
        ords.len() >= 2,
        "the seam rolled a segment (found {ords:?})"
    );
    let first = daemon_vhc_journal::scan_file(jdir.join(format!("segment-{:08}.dvhcjrn", ords[0])))
        .expect("scan retired span");
    assert!(
        first.sealed,
        "the retired span's segment sealed at the seam"
    );
    assert_eq!(first.header.id.instance, 1);
    assert_eq!(first.header.id.epoch, 0);
    let last = daemon_vhc_journal::scan_file(
        jdir.join(format!("segment-{:08}.dvhcjrn", ords[ords.len() - 1])),
    )
    .expect("scan continuation span");
    assert_eq!(last.header.id.instance, 2);
    assert_eq!(last.header.id.epoch, 1);
    let post_seam_publish_seqs: Vec<u64> = last
        .records
        .iter()
        .filter_map(|r| match &r.body {
            daemon_vhc_journal::Body::Publish(p) => Some(p.seq),
            _ => None,
        })
        .collect();
    assert_eq!(
        post_seam_publish_seqs.first(),
        Some(&0),
        "the new incarnation's publish stream opened at seq 0"
    );
}
