// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **Roster floor self-heal regressions** (the join-transaction recovery invariant): a roster
//! publish refused STALE restarts the join transaction from **verified own-base evidence** —
//! and ONLY from it.
//!
//! - A stored record that verifies to this node's OWN base identity for this instance's exact
//!   scope is a fresher execution of our own ladder (a predecessor a restart lost track of):
//!   the counter is repaired strictly above it and authorship restarts once, minting a fresh
//!   incarnation that supersedes the stored floor. The join then completes.
//! - A stored record by a FOREIGN base occupying our slot is a collision (or a lying registry):
//!   the join FAILS CLOSED typed — no counter mutation, no floor adopted, no join. Registry
//!   metadata is retry signal, never trusted state (ABI §12.13 [ROSTER-1]).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ciborium::value::Value;
use daemon_api::{VhcApi, VhcPolicy, VhcPolicyMode};
use daemon_vhc_node::credentials::RunInstanceIdentity;
use daemon_vhc_node::roster::author_roster_record;
use daemon_vhc_node::service::{VhcError, WorkerControl};
use daemon_vhc_node::{DiscoveredRun, RunDiscovery, VhcService, VhcServiceParts, VhcStore};
use daemon_vhc_proto::genesis::{
    ChannelDecl, Identities, RoleEntry, RoleGrants, RunSection, SnapshotArtifact,
    TransportSelection, GENESIS_SCHEMA_MAJOR,
};
use daemon_vhc_proto::{
    blake3_hash, peer_id, to_canonical_vec, GenesisEnvelope, RosterRecord, SignedEnvelope,
    SigningKey,
};
use daemon_vhc_session::config::VhcConfig;
use daemon_vhc_session::keystore::VhcKeystore;
use daemon_vhc_session::protocol::{AdmittedTuple, Eligibility, Hardware, JoinPolicy, LeaveMode};

const RUN: &str = "floor-run";
const ROLE: &str = "trainer";
const COORD: &str = "https://coord.example/api/v1/vhc";

/// A minimal signed genesis for the run: coordinator + trainer roles over dummy pinned
/// artifacts, the node's base in the trust set. Returns the canonical `SignedEnvelope` wire
/// bytes and the genesis hash (the run's cryptographic id the admitted tuple must carry).
fn genesis_wire(trusted_bases: &[daemon_vhc_proto::PeerId]) -> (Vec<u8>, [u8; 32]) {
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "role.wasm".to_string(),
        SnapshotArtifact {
            url: "file:///dev/null".into(),
            blake3: blake3_hash(b"module-bytes"),
            size: None,
        },
    );
    let role_entry = || RoleEntry {
        // A fixture envelope: this exercises the roster floor, nothing resource-shaped — the
        // shared trivial construction every compute-free module emits.
        execution: Some(
            daemon_vhc_proto::RoleExecutionRequirements::fixture_over_trivial_plan(vec![
                "cpu".to_string()
            ]),
        ),
        lane: "trainer".into(),
        module: "role.wasm".into(),
        abi: "vhc@2".into(),
        config: Value::from(1u8),
        grants: RoleGrants {
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
        },
        device_min: daemon_vhc_proto::DeviceMinimums::default(),
    };
    let mut roles = BTreeMap::new();
    roles.insert("coordinator".to_string(), role_entry());
    roles.insert(ROLE.to_string(), role_entry());
    let env = GenesisEnvelope {
        run: RunSection {
            schema: GENESIS_SCHEMA_MAJOR,
            run_label: RUN.to_string(),
            min_peers: 1,
            max_peers: 4,
            access: daemon_vhc_proto::envelope::Access::Org,
        },
        roles,
        artifacts,
        corpus_manifest: None,
        state_contract: None,
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
    (
        to_canonical_vec(&wire).expect("encode signed envelope"),
        genesis_hash,
    )
}

/// A worker seam whose assess admits the trainer at the REAL genesis hash (so node-side
/// authorship — the roster publish under test — actually runs).
struct AdmittingWorker {
    genesis_hash: [u8; 32],
    joins: Mutex<Vec<String>>,
}

#[async_trait]
impl WorkerControl for AdmittingWorker {
    async fn probe(&self) -> Result<Hardware, VhcError> {
        Ok(Hardware::default())
    }
    async fn assess(&self, _e: Vec<u8>, _r: Option<String>) -> Result<Eligibility, VhcError> {
        Ok(Eligibility {
            eligible: true,
            reasons: vec!["admitted".into()],
            headroom: vec![
                (
                    daemon_vhc_abi::RESERVATION_DEVICE_BYTES_KEY.into(),
                    256 << 20,
                ),
                (daemon_vhc_abi::RESERVATION_HOST_BYTES_KEY.into(), 512 << 20),
            ],
            refusal_code: None,
            admitted_tuple: Some(AdmittedTuple {
                module_hash: blake3_hash(b"module-bytes").0,
                genesis_hash: self.genesis_hash,
                role: ROLE.to_string(),
                incarnation: 0,
                ..AdmittedTuple::default()
            }),
        })
    }
    async fn join(
        &self,
        run_id: String,
        _coordinator: String,
        _credentials: Vec<u8>,
        _policy: JoinPolicy,
        _tuple: Option<AdmittedTuple>,
    ) -> Result<(), VhcError> {
        self.joins.lock().unwrap().push(run_id);
        Ok(())
    }
    async fn leave(&self, _run_id: String, _mode: LeaveMode) -> Result<(), VhcError> {
        Ok(())
    }
    async fn throttle(
        &self,
        _vram: Option<u32>,
        _duty: Option<u8>,
        _paused: bool,
    ) -> Result<(), VhcError> {
        Ok(())
    }
}

/// A discovery seam whose FIRST roster publish is refused STALE with a configured stored
/// record; later publishes are accepted and recorded (the successful re-authored publish).
struct StaleRosterDiscovery {
    envelope: Vec<u8>,
    stale_stored: Mutex<Option<RosterRecord>>,
    accepted: Mutex<Vec<RosterRecord>>,
}

#[async_trait]
impl RunDiscovery for StaleRosterDiscovery {
    async fn list_runs(&self) -> Result<Vec<DiscoveredRun>, VhcError> {
        Ok(Vec::new())
    }
    async fn get_run(&self, run_id: &str) -> Result<Option<DiscoveredRun>, VhcError> {
        Ok(Some(DiscoveredRun {
            run_id: run_id.to_string(),
            coordinator: COORD.to_string(),
            envelope_hash: "deadbeef".into(),
            proto_version: 3,
        }))
    }
    async fn fetch_envelope(&self, _run_id: &str) -> Result<Vec<u8>, VhcError> {
        Ok(self.envelope.clone())
    }
    async fn publish_roster(&self, _run_id: &str, record: &RosterRecord) -> Result<(), VhcError> {
        if let Some(stored) = self.stale_stored.lock().unwrap().take() {
            return Err(VhcError::RosterStale {
                stored_incarnation: stored.body.incarnation,
                stored: Some(Box::new(stored)),
            });
        }
        self.accepted.lock().unwrap().push(record.clone());
        Ok(())
    }
    async fn fetch_roster(&self, _run_id: &str) -> Result<Vec<RosterRecord>, VhcError> {
        Ok(Vec::new())
    }
}

fn iroh_config() -> VhcConfig {
    VhcConfig {
        enabled: true,
        coordinator_allowlist: vec![COORD.into()],
        iroh: daemon_vhc_session::config::IrohConfig {
            enabled: true,
            relays: String::new(),
            bind_port: 0,
            advertise_ips: vec!["127.0.0.1".into()],
        },
        ..VhcConfig::default()
    }
}

fn policy() -> VhcPolicy {
    VhcPolicy {
        mode: VhcPolicyMode::Idle,
        vram_cap_mb: 8_000,
        duty_cycle_pct: 90,
        schedule: None,
    }
}

/// Author the "stored" roster record a registry would return on the stale refusal: a record
/// under `keystore`'s identity ladder at `incarnation`, scoped to the run.
fn stored_record(keystore: &VhcKeystore, genesis_hash: [u8; 32], incarnation: u64) -> RosterRecord {
    author_roster_record(
        keystore,
        &RunInstanceIdentity {
            run_label: RUN,
            genesis_hash,
            epoch: 0,
            role: ROLE,
            incarnation,
            module_hash: blake3_hash(b"module-bytes").0,
        },
        vec!["127.0.0.1:4444".into()],
        None,
        1_000,
    )
    .expect("author stored record")
}

/// Verified OWN-BASE evidence repairs the floor: the stale refusal carries a record from THIS
/// node's own ladder at incarnation 5; the transaction restarts once, mints strictly above it,
/// republishes, and the join completes.
#[tokio::test]
async fn own_base_stale_roster_repairs_the_floor_and_the_join_restarts_once() {
    let identity = tempfile::tempdir().unwrap();
    let keystore = VhcKeystore::open(identity.path()).unwrap();
    let own_base = peer_id(&keystore.base_identity().unwrap());
    let (envelope, genesis_hash) = genesis_wire(&[own_base]);

    let stored = stored_record(&keystore, genesis_hash, 5);
    let discovery = Arc::new(StaleRosterDiscovery {
        envelope,
        stale_stored: Mutex::new(Some(stored)),
        accepted: Mutex::new(Vec::new()),
    });
    let worker = Arc::new(AdmittingWorker {
        genesis_hash,
        joins: Mutex::new(Vec::new()),
    });
    let svc = VhcService::new(VhcServiceParts {
        config: iroh_config(),
        store: VhcStore::open_in_memory().unwrap(),
        worker: worker.clone(),
        feed: None,
        discovery: Some(discovery.clone()),
        budget: None,
        worker_factory: None,
        identity_dir: Some(identity.path().to_path_buf()),
        seat_directory: None,
    });

    svc.vhc_join(RUN.into(), policy(), "op".into())
        .await
        .expect("the join self-heals over the repaired floor");

    // The restarted authorship republished ABOVE the stored floor (5): mint-above guarantees
    // strictly greater, and the record is this node's own fresh execution identity.
    let accepted = discovery.accepted.lock().unwrap();
    assert_eq!(accepted.len(), 1, "exactly one accepted publish");
    assert!(
        accepted[0].body.incarnation > 5,
        "the re-authored record supersedes the stored floor (got {})",
        accepted[0].body.incarnation
    );
    assert_eq!(worker.joins.lock().unwrap().as_slice(), [RUN.to_string()]);
}

/// A FOREIGN base's record in our slot is a collision, not a floor: the join fails closed
/// typed, nothing is republished, and no counter repair happens.
#[tokio::test]
async fn foreign_base_stale_roster_fails_closed_and_adopts_no_floor() {
    let identity = tempfile::tempdir().unwrap();
    let keystore = VhcKeystore::open(identity.path()).unwrap();
    let own_base = peer_id(&keystore.base_identity().unwrap());
    let (envelope, genesis_hash) = genesis_wire(&[own_base]);

    // The stored record comes from a DIFFERENT node installation (its own keystore + base).
    let foreign_dir = tempfile::tempdir().unwrap();
    let foreign = VhcKeystore::open(foreign_dir.path()).unwrap();
    let stored = stored_record(&foreign, genesis_hash, 9);

    let discovery = Arc::new(StaleRosterDiscovery {
        envelope,
        stale_stored: Mutex::new(Some(stored)),
        accepted: Mutex::new(Vec::new()),
    });
    let worker = Arc::new(AdmittingWorker {
        genesis_hash,
        joins: Mutex::new(Vec::new()),
    });
    let svc = VhcService::new(VhcServiceParts {
        config: iroh_config(),
        store: VhcStore::open_in_memory().unwrap(),
        worker: worker.clone(),
        feed: None,
        discovery: Some(discovery.clone()),
        budget: None,
        worker_factory: None,
        identity_dir: Some(identity.path().to_path_buf()),
        seat_directory: None,
    });

    let err = svc
        .vhc_join(RUN.into(), policy(), "op".into())
        .await
        .expect_err("a foreign-base record in our slot fails the join closed");
    assert!(
        err.to_string().contains("failing closed"),
        "typed fail-closed refusal, got: {err}"
    );
    assert!(
        discovery.accepted.lock().unwrap().is_empty(),
        "nothing was republished over a collision"
    );
    assert!(
        worker.joins.lock().unwrap().is_empty(),
        "no join was issued"
    );
}
