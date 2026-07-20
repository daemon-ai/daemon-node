// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The node-side **record consumption seam** (architecture §5.4; ABI §10.3):
//! `VhcApi::vhc_switch_module` takes a committed canonical-CBOR `UpgradeRecord`, validates it
//! FAIL-CLOSED against the run's rebuilt transition chain (genesis + the node's persisted
//! record mirror), provisions the post-switch identity, and drives `switch_module` through the
//! worker-control seam. The worker here is a trait-level fake recording every call — the LIVE
//! transaction below it is proven by the worker/session suites and the acceptance gate; THIS
//! suite proves the node's validation, provisioning, persistence, and refusal surfaces.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use daemon_api::{VhcApi, VhcSwitchOutcome};
use daemon_vhc_node::service::{VhcError, WorkerControl};
use daemon_vhc_node::{DiscoveredRun, RunDiscovery, VhcService, VhcServiceParts, VhcStore};
use daemon_vhc_proto::envelope::{Access, DeviceMinimums};
use daemon_vhc_proto::genesis::{
    GenesisEnvelope, Identities, RoleEntry, RoleGrants, RunSection, SnapshotArtifact,
    TransportSelection, GENESIS_SCHEMA_MAJOR,
};
use daemon_vhc_proto::{
    blake3_hash, peer_id, to_canonical_vec, Hash, SignedEnvelope, SigningKey, UpgradeRecord,
};
use daemon_vhc_session::config::VhcConfig;
use daemon_vhc_session::protocol::{
    AdmittedTuple, Eligibility, Hardware, JoinPolicy, LeaveMode, SwitchTarget,
};
use daemon_vhc_supervisor::SwitchOutcome;

const RUN: &str = "run-upgrade";
const COORD: &str = "https://coord.example/api/v1/vhc";

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn hash(n: u8) -> Hash {
    Hash([n; 32])
}

/// A two-role genesis whose trainer role is pinned to `old_module`, with `key(1)` as the
/// single-key upgrade authority. Returns the canonical `SignedEnvelope` wire bytes + run id.
fn genesis_wire(old_module: Hash) -> (Vec<u8>, Hash, GenesisEnvelope) {
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "trainer-mod".to_string(),
        SnapshotArtifact {
            url: "r2://mods/trainer.wasm".into(),
            blake3: old_module,
            size: None,
        },
    );
    artifacts.insert(
        "coord-mod".to_string(),
        SnapshotArtifact {
            url: "r2://mods/coord.wasm".into(),
            blake3: hash(2),
            size: None,
        },
    );
    let mut roles = BTreeMap::new();
    for (name, module, lane) in [
        ("trainer", "trainer-mod", "trainer"),
        ("coordinator", "coord-mod", "coordinator"),
    ] {
        roles.insert(
            name.to_string(),
            RoleEntry {
                lane: lane.into(),
                module: module.into(),
                abi: "vhc@2".into(),
                config: ciborium::value::Value::Map(vec![]),
                grants: RoleGrants::default(),
                device_min: DeviceMinimums::default(),
            },
        );
    }
    let genesis = GenesisEnvelope {
        run: RunSection {
            schema: GENESIS_SCHEMA_MAJOR,
            run_label: RUN.into(),
            min_peers: 1,
            max_peers: 8,
            access: Access::Org,
        },
        roles,
        artifacts,
        corpus_manifest: None,
        authority: ciborium::value::Value::Map(vec![]),
        transport: TransportSelection::default(),
        identities: Identities {
            upgrade_authority: vec![peer_id(&key(1))],
            ..Default::default()
        },
    };
    let frozen = genesis.freeze(&key(200)).expect("freeze genesis");
    let run_id = *frozen.run_id();
    let wire = SignedEnvelope {
        bytes: frozen.bytes().to_vec(),
        signature: *frozen.signature(),
        signer: *frozen.signer(),
    };
    (
        to_canonical_vec(&wire).expect("wire encode"),
        run_id,
        genesis,
    )
}

/// The discovery fake: serves the REAL frozen genesis wire (consumption re-verifies it).
struct FakeDiscovery {
    envelope: Vec<u8>,
}

#[async_trait]
impl RunDiscovery for FakeDiscovery {
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
}

/// One recorded `switch_module` drive: what the node handed the worker.
#[derive(Clone)]
struct SwitchCall {
    epoch: u64,
    role: String,
    new_module: [u8; 32],
    grants_hash: [u8; 32],
    deadline_ms: u64,
    tuple: Option<AdmittedTuple>,
}

/// A recording worker fake: assess/join succeed; `assess_switch` answers an eligible verdict
/// with a post-switch tuple; `switch_module` records its args and answers the scripted outcome.
struct FakeWorker {
    genesis_hash: [u8; 32],
    switch_calls: Mutex<Vec<SwitchCall>>,
    switch_outcome: Mutex<SwitchOutcome>,
}

impl FakeWorker {
    fn new(genesis_hash: [u8; 32]) -> Arc<Self> {
        Arc::new(Self {
            genesis_hash,
            switch_calls: Mutex::new(Vec::new()),
            switch_outcome: Mutex::new(SwitchOutcome::Activated {
                epoch: 0,
                module: [0; 32],
                retries: 0,
            }),
        })
    }
}

#[async_trait]
impl WorkerControl for FakeWorker {
    async fn probe(&self) -> Result<Hardware, VhcError> {
        Ok(Hardware::default())
    }
    async fn assess(
        &self,
        _envelope: Vec<u8>,
        _role: Option<String>,
    ) -> Result<Eligibility, VhcError> {
        Ok(Eligibility {
            eligible: true,
            ..Eligibility::default()
        })
    }
    async fn assess_switch(
        &self,
        _envelope: Vec<u8>,
        role: Option<String>,
        target: SwitchTarget,
    ) -> Result<Eligibility, VhcError> {
        Ok(Eligibility {
            eligible: true,
            reasons: vec!["switch target admitted".into()],
            headroom: Vec::new(),
            refusal_code: None,
            admitted_tuple: Some(AdmittedTuple {
                module_hash: target.new_module,
                config_hash: *blake3::hash(&[]).as_bytes(),
                grants_hash: target.grants_hash,
                claim_hash: [0xCC; 32],
                genesis_hash: self.genesis_hash,
                role: role.unwrap_or_else(|| "trainer".into()),
                incarnation: 0,
                device_profile_rev: 0,
                owner_policy_rev: 0,
                backend: "cpu".into(),
                gpu_index: 0,
            }),
        })
    }
    async fn join(
        &self,
        _run_id: String,
        _coordinator: String,
        _credentials: Vec<u8>,
        _policy: JoinPolicy,
        _admitted_tuple: Option<daemon_vhc_session::protocol::AdmittedTuple>,
    ) -> Result<(), VhcError> {
        Ok(())
    }
    async fn leave(&self, _run_id: String, _mode: LeaveMode) -> Result<(), VhcError> {
        Ok(())
    }
    async fn throttle(
        &self,
        _vram_cap_mb: Option<u32>,
        _duty_cycle_pct: Option<u8>,
        _paused: bool,
    ) -> Result<(), VhcError> {
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    async fn switch_module(
        &self,
        _run_id: String,
        epoch: u64,
        role: String,
        new_module: [u8; 32],
        grants_hash: [u8; 32],
        deadline_ms: u64,
        admitted_tuple: Option<daemon_vhc_session::protocol::AdmittedTuple>,
    ) -> Result<SwitchOutcome, VhcError> {
        self.switch_calls.lock().unwrap().push(SwitchCall {
            epoch,
            role,
            new_module,
            grants_hash,
            deadline_ms,
            tuple: admitted_tuple,
        });
        let mut outcome = self.switch_outcome.lock().unwrap().clone();
        if let SwitchOutcome::Activated {
            epoch: e, module, ..
        } = &mut outcome
        {
            *e = epoch;
            *module = new_module;
        }
        Ok(outcome)
    }
}

/// A joined service over the real genesis wire, with node-side identity authorship enabled.
async fn joined_service(
    envelope: Vec<u8>,
    worker: Arc<FakeWorker>,
    identity_dir: &std::path::Path,
) -> Arc<VhcService> {
    let config = VhcConfig {
        enabled: true,
        coordinator_allowlist: vec![COORD.into()],
        ..VhcConfig::default()
    };
    let svc = Arc::new(VhcService::new(VhcServiceParts {
        config,
        store: VhcStore::open_in_memory().unwrap(),
        worker,
        feed: None,
        discovery: Some(Arc::new(FakeDiscovery { envelope })),
        budget: None,
        worker_factory: None,
        identity_dir: Some(identity_dir.to_path_buf()),
        seat_directory: None,
    }));
    svc.vhc_join(
        RUN.into(),
        daemon_api::VhcPolicy::default(),
        "op-join".into(),
    )
    .await
    .expect("join");
    svc
}

fn author_record(run_id: Hash, old_module: Hash, new_module: Hash, signer: &SigningKey) -> Vec<u8> {
    let record = UpgradeRecord::author(
        run_id,
        1,
        run_id,
        "trainer",
        old_module,
        new_module,
        7,
        hash(50),
        blake3_hash(&[]),
        &[signer],
    )
    .expect("author record");
    to_canonical_vec(&record).expect("record wire")
}

#[tokio::test]
async fn authorized_record_drives_the_switch_and_persists_the_advance() {
    let old_module = hash(11);
    let new_module = hash(42);
    let (envelope, run_id, _genesis) = genesis_wire(old_module);
    let identity = tempfile::tempdir().unwrap();
    let worker = FakeWorker::new(run_id.0);
    let svc = joined_service(envelope, worker.clone(), identity.path()).await;
    let joined_instance = svc
        .store()
        .get_run(RUN)
        .unwrap()
        .expect("joined row")
        .instance;

    let record = author_record(run_id, old_module, new_module, &key(1));
    let outcome = svc
        .vhc_switch_module(RUN.into(), record, "op-switch".into())
        .await
        .expect("switch op");
    let VhcSwitchOutcome::Activated {
        epoch,
        module_hash,
        retries,
    } = outcome
    else {
        panic!("expected activation, got {outcome:?}");
    };
    assert_eq!(epoch, 1);
    assert_eq!(module_hash, new_module.to_hex());
    assert_eq!(retries, 0);

    // The worker was driven with the record-derived, node-provisioned facts.
    let calls = worker.switch_calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    let call = &calls[0];
    assert_eq!(call.epoch, 1);
    assert_eq!(call.role, "trainer");
    assert_eq!(call.new_module, new_module.0);
    assert_eq!(call.grants_hash, hash(50).0);
    assert!(call.deadline_ms > 0, "the drain deadline is clamped, not 0");
    let tuple = call.tuple.as_ref().expect("post-switch tuple delivered");
    assert!(
        tuple.incarnation > joined_instance,
        "the minted post-switch incarnation {} supersedes {joined_instance}",
        tuple.incarnation
    );
    assert_eq!(tuple.module_hash, new_module.0);

    // The post-switch identity was PROVISIONED before the command: key + certificate bound to
    // (run, epoch 1, trainer, new incarnation, new module) resolve read-only from the keystore.
    let keystore = daemon_vhc_session::keystore::VhcKeystore::open(identity.path()).unwrap();
    let cert = keystore
        .run_certificate(RUN, "trainer", tuple.incarnation)
        .expect("keystore read")
        .expect("post-switch certificate provisioned");
    assert_eq!(cert.body.scope.epoch, 1);
    assert_eq!(cert.body.scope.module_hash, new_module);
    assert_eq!(cert.body.scope.instance, tuple.incarnation);

    // The store persisted the advance: execution identity at epoch 1 / the new incarnation, the
    // module observability hash, the backfilled RunId, and the record mirror (the next switch's
    // chain-rebuild input).
    let row = svc.store().get_run(RUN).unwrap().expect("row");
    assert_eq!(row.epoch, 1);
    assert_eq!(row.instance, tuple.incarnation);
    assert_eq!(row.module_hash, Some(new_module.0));
    assert_eq!(row.run_id_hash, Some(run_id.0));
    assert_eq!(svc.store().upgrade_records(RUN).unwrap().len(), 1);

    // A REPLAYED consumption of the same record refuses typed (strictly-monotone epochs; the
    // mirror already carries epoch 1) — nothing re-drives the worker.
    let replay = author_record(run_id, old_module, new_module, &key(1));
    let outcome = svc
        .vhc_switch_module(RUN.into(), replay, "op-replay".into())
        .await
        .expect("replay op");
    assert!(
        matches!(outcome, VhcSwitchOutcome::Refused { ref reason } if reason.contains("epoch")),
        "replay must refuse on epoch monotonicity: {outcome:?}"
    );
    assert_eq!(worker.switch_calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn unauthorized_and_misdirected_records_refuse_without_touching_the_worker() {
    let old_module = hash(11);
    let new_module = hash(42);
    let (envelope, run_id, _genesis) = genesis_wire(old_module);
    let identity = tempfile::tempdir().unwrap();
    let worker = FakeWorker::new(run_id.0);
    let svc = joined_service(envelope, worker.clone(), identity.path()).await;

    // Signed by a NON-authority key: refused (fail closed).
    let rogue = author_record(run_id, old_module, new_module, &key(9));
    let outcome = svc
        .vhc_switch_module(RUN.into(), rogue, "op-rogue".into())
        .await
        .expect("rogue op");
    assert!(
        matches!(outcome, VhcSwitchOutcome::Refused { ref reason } if reason.contains("not authorized")),
        "non-authority record must refuse: {outcome:?}"
    );

    // A record for a DIFFERENT role than the held instance: refused before validation work.
    let wrong_role = UpgradeRecord::author(
        run_id,
        1,
        run_id,
        "coordinator",
        hash(2),
        new_module,
        7,
        hash(50),
        blake3_hash(&[]),
        &[&key(1)],
    )
    .unwrap();
    let outcome = svc
        .vhc_switch_module(
            RUN.into(),
            to_canonical_vec(&wrong_role).unwrap(),
            "op-role".into(),
        )
        .await
        .expect("wrong-role op");
    assert!(
        matches!(outcome, VhcSwitchOutcome::Refused { ref reason } if reason.contains("role")),
        "wrong-role record must refuse: {outcome:?}"
    );

    // A stale old_module (not the role's current pin): the chain refuses it.
    let stale = author_record(run_id, hash(77), new_module, &key(1));
    let outcome = svc
        .vhc_switch_module(RUN.into(), stale, "op-stale".into())
        .await
        .expect("stale op");
    assert!(
        matches!(outcome, VhcSwitchOutcome::Refused { .. }),
        "stale-old-module record must refuse: {outcome:?}"
    );

    // Garbage bytes: refused typed, never a panic.
    let outcome = svc
        .vhc_switch_module(RUN.into(), vec![0xFF, 0x00, 0x13], "op-junk".into())
        .await
        .expect("junk op");
    assert!(
        matches!(outcome, VhcSwitchOutcome::Refused { ref reason } if reason.contains("undecodable")),
        "junk bytes must refuse typed: {outcome:?}"
    );

    // For a run with NO live instance: refused (nothing to switch).
    let record = author_record(run_id, old_module, new_module, &key(1));
    let outcome = svc
        .vhc_switch_module("run-unknown".into(), record, "op-unknown".into())
        .await
        .expect("unknown-run op");
    assert!(
        matches!(outcome, VhcSwitchOutcome::Refused { ref reason } if reason.contains("no live role-instance")),
        "unknown run must refuse: {outcome:?}"
    );

    // The worker's switch surface was never touched by any refusal.
    assert!(worker.switch_calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_post_fence_left_outcome_travels_in_band() {
    let old_module = hash(11);
    let new_module = hash(42);
    let (envelope, run_id, _genesis) = genesis_wire(old_module);
    let identity = tempfile::tempdir().unwrap();
    let worker = FakeWorker::new(run_id.0);
    *worker.switch_outcome.lock().unwrap() = SwitchOutcome::Left {
        reason: "migration exhausted its retry budget: Incompatible".into(),
    };
    let svc = joined_service(envelope, worker.clone(), identity.path()).await;

    let record = author_record(run_id, old_module, new_module, &key(1));
    let outcome = svc
        .vhc_switch_module(RUN.into(), record, "op-left".into())
        .await
        .expect("left op");
    assert!(
        matches!(outcome, VhcSwitchOutcome::Left { ref reason } if reason.contains("exhausted")),
        "the post-fence exit travels in-band: {outcome:?}"
    );
    // A left switch persists NO advance: the row's identity stays at the joined epoch and no
    // record mirror is written.
    let row = svc.store().get_run(RUN).unwrap().expect("row");
    assert_eq!(row.epoch, 0);
    assert!(svc.store().upgrade_records(RUN).unwrap().is_empty());
}
