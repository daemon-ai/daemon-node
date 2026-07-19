// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The resident coordinator seat keeper over the normative CAS fold: the service's keeper pass
//! claims the seat for a joined coordinator-role run, heartbeats it, releases it on owner pause
//! (the floor persists), and stands by against a live incumbent; the keeper drops a fenced lease
//! when the seat moves (fencing-is-safe-not-seamless).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use daemon_api::{VhcApi, VhcPolicy, VhcPolicyMode};
use daemon_vhc_net::{FakeSeatRegistry, SeatClaimOutcome};
use daemon_vhc_node::service::{VhcError, WorkerControl};
use daemon_vhc_node::{
    SeatCandidate, SeatDirectory, SeatKeeper, SeatNote, VhcService, VhcServiceParts, VhcStore,
};
use daemon_vhc_proto::{
    ControlEndpoint, SeatDecision, SeatLease, SeatRelease, SeatState, DEFAULT_SEAT_TTL_MS,
};
use daemon_vhc_session::config::VhcConfig;
use daemon_vhc_session::keystore::VhcKeystore;
use daemon_vhc_session::protocol::{self, AdmittedTuple, Eligibility, Hardware, JoinPolicy};

const RUN: &str = "seat-run";
const ROLE: &str = "coordinator";
const GENESIS: [u8; 32] = [0x9E; 32];
const MODULE: [u8; 32] = [0xC0; 32];

/// The normative CAS fold behind the [`SeatDirectory`] seam, with a settable clock so tests
/// drive expiry deterministically (`None` = wall clock, the service-integration mode).
struct FakeDirectory {
    registry: FakeSeatRegistry,
    now_ms: Mutex<Option<u64>>,
}

impl FakeDirectory {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            registry: FakeSeatRegistry::new(),
            now_ms: Mutex::new(None),
        })
    }
    fn set_now(&self, now: u64) {
        *self.now_ms.lock().unwrap() = Some(now);
    }
    fn now(&self) -> u64 {
        self.now_ms.lock().unwrap().unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
        })
    }
}

#[async_trait]
impl SeatDirectory for FakeDirectory {
    async fn read_seat(&self, run: &str, role: &str) -> Result<SeatState, VhcError> {
        Ok(self.registry.read(run, role))
    }
    async fn claim_seat(&self, run: &str, lease: &SeatLease) -> Result<SeatClaimOutcome, VhcError> {
        let resp = self.registry.claim(run, lease, self.now());
        Ok(match resp.decision {
            SeatDecision::Accepted => SeatClaimOutcome::Won(lease.clone()),
            decision => SeatClaimOutcome::Lost {
                decision,
                state: resp.state,
            },
        })
    }
    async fn renew_seat(&self, run: &str, lease: &SeatLease) -> Result<SeatClaimOutcome, VhcError> {
        let resp = self.registry.renew(run, lease, self.now());
        Ok(match resp.decision {
            SeatDecision::Accepted => SeatClaimOutcome::Won(lease.clone()),
            decision => SeatClaimOutcome::Lost {
                decision,
                state: resp.state,
            },
        })
    }
    async fn release_seat(
        &self,
        run: &str,
        role: &str,
        release: &SeatRelease,
    ) -> Result<(), VhcError> {
        let resp = self.registry.release(run, role, release, self.now());
        if resp.decision == SeatDecision::Accepted {
            Ok(())
        } else {
            Err(VhcError::Discovery(format!(
                "seat release refused: {:?}",
                resp.decision
            )))
        }
    }
}

/// A minimal worker seam (the seat keeper never touches the worker).
struct IdleWorker;

#[async_trait]
impl WorkerControl for IdleWorker {
    async fn probe(&self) -> Result<Hardware, VhcError> {
        Ok(Hardware::default())
    }
    async fn assess(&self, _e: Vec<u8>, _r: Option<String>) -> Result<Eligibility, VhcError> {
        Ok(Eligibility::default())
    }
    async fn join(
        &self,
        _run_id: String,
        _coordinator: String,
        _credentials: Vec<u8>,
        _policy: JoinPolicy,
        _admitted_tuple: Option<AdmittedTuple>,
    ) -> Result<(), VhcError> {
        Ok(())
    }
    async fn leave(
        &self,
        _run_id: String,
        _mode: daemon_vhc_session::protocol::LeaveMode,
    ) -> Result<(), VhcError> {
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

fn tuple() -> AdmittedTuple {
    AdmittedTuple {
        module_hash: MODULE,
        genesis_hash: GENESIS,
        role: ROLE.to_string(),
        incarnation: 1,
        ..AdmittedTuple::default()
    }
}

fn candidate() -> SeatCandidate {
    SeatCandidate {
        run_label: RUN.to_string(),
        genesis_hash: GENESIS,
        role: ROLE.to_string(),
        epoch: 0,
        module_hash: MODULE,
        endpoint: ControlEndpoint {
            ws: Some("wss://coord.example/runs/seat-run/ws".into()),
            iroh_ticket: None,
        },
    }
}

/// The service-resident pass: a joined coordinator-role run is covered — claim on the first
/// pass, heartbeat renew on the next, fenced release on owner pause (the floor persists), and
/// stand-by (no claim) while another claimant holds the slot after the release.
#[tokio::test]
async fn keeper_claims_renews_and_releases_on_pause() {
    let identity = tempfile::tempdir().unwrap();
    VhcKeystore::open(identity.path()).unwrap();
    let directory = FakeDirectory::new();
    let svc = Arc::new(VhcService::new(VhcServiceParts {
        config: VhcConfig {
            enabled: true,
            seat_claim: true,
            ..VhcConfig::default()
        },
        store: VhcStore::open_in_memory().unwrap(),
        worker: Arc::new(IdleWorker),
        feed: None,
        discovery: None,
        budget: None,
        worker_factory: None,
        identity_dir: Some(identity.path().to_path_buf()),
        seat_directory: Some(directory.clone()),
    }));
    svc.bind_self();

    // A durable joined intent whose admitted role is the seat role.
    let policy = VhcPolicy {
        mode: VhcPolicyMode::Always,
        vram_cap_mb: 0,
        duty_cycle_pct: 100,
        schedule: None,
    };
    svc.store()
        .put_join_intent(
            RUN,
            "wss://coord.example",
            &policy,
            None,
            &Default::default(),
        )
        .unwrap();
    svc.store().set_execution_identity(RUN, 0, ROLE, 1).unwrap();
    svc.store()
        .set_admitted_tuple(RUN, &protocol::encode(&tuple()).unwrap())
        .unwrap();

    // First pass: the virgin slot is claimed at bid 0.
    let notes = svc.seat_tick().await.unwrap();
    assert!(
        matches!(
            notes.as_slice(),
            [SeatNote::Claimed { run_label, incarnation: 0 }] if run_label == RUN
        ),
        "got {notes:?}"
    );
    assert!(matches!(
        directory.registry.read(RUN, ROLE),
        SeatState::Leased(_)
    ));

    // Second pass: the held lease heartbeats (a renew, never a takeover).
    let notes = svc.seat_tick().await.unwrap();
    assert!(
        matches!(notes.as_slice(), [SeatNote::Renewed { run_label }] if run_label == RUN),
        "got {notes:?}"
    );

    // Owner pause: the seat releases FENCED — the slot unclaims, the token floor persists.
    svc.vhc_pause(RUN.to_string(), "op-p".into()).await.unwrap();
    match directory.registry.read(RUN, ROLE) {
        SeatState::Unclaimed { last_fencing_token } => {
            assert_eq!(
                last_fencing_token,
                Some(0),
                "the floor survives the release"
            );
        }
        other => panic!("expected the tombstoned slot, got {other:?}"),
    }
    // A paused run leaves the keeper's coverage entirely (no claim while paused).
    assert!(svc.seat_tick().await.unwrap().is_empty());

    // A successor claims at floor + 1; after resume-of-intent the keeper STANDS BY against the
    // live incumbent (fencing is safety, not a fight). The intent axis is enough to re-cover the
    // run: flip it back to joined directly (a full resume needs a live worker seam).
    let other_identity = tempfile::tempdir().unwrap();
    let other_keystore = VhcKeystore::open(other_identity.path()).unwrap();
    let successor = daemon_vhc_node::seat::author_claim(
        &other_keystore,
        &daemon_vhc_node::seat::CoordinatorSeat {
            run_label: RUN,
            genesis_hash: GENESIS,
            role: ROLE,
            epoch: 0,
            module_hash: MODULE,
            endpoint: candidate().endpoint,
        },
        1,
        directory.now(),
    )
    .unwrap();
    assert_eq!(
        directory
            .registry
            .claim(RUN, &successor, directory.now())
            .decision,
        SeatDecision::Accepted
    );
    svc.store()
        .set_desired_state(RUN, daemon_vhc_node::DesiredState::Joined)
        .unwrap();
    let notes = svc.seat_tick().await.unwrap();
    assert!(
        notes.is_empty(),
        "stand by against a live incumbent: {notes:?}"
    );
    match directory.registry.read(RUN, ROLE) {
        SeatState::Leased(lease) => assert_eq!(lease.body.incarnation, 1, "the successor holds"),
        other => panic!("expected the successor's lease, got {other:?}"),
    }
}

/// Fencing at the keeper: a held lease whose seat moved (expiry + takeover) is DROPPED on the
/// refused renew — the fenced claimant never fights, and a later pass bids fresh at floor + 1.
#[tokio::test]
async fn keeper_drops_a_fenced_lease_on_refused_renew() {
    let identity = tempfile::tempdir().unwrap();
    VhcKeystore::open(identity.path()).unwrap();
    let directory = FakeDirectory::new();
    let keeper = SeatKeeper::new(directory.clone(), identity.path().to_path_buf());
    let t0 = 1_000_000u64;
    directory.set_now(t0);

    // Claim the virgin slot at bid 0.
    let notes = keeper.tick(&[candidate()], t0).await;
    assert!(matches!(
        notes.as_slice(),
        [SeatNote::Claimed { incarnation: 0, .. }]
    ));
    assert_eq!(keeper.held_incarnation(RUN), Some(0));

    // The lease expires unrenewed; a standby takes the seat at floor + 1.
    let t1 = t0 + DEFAULT_SEAT_TTL_MS * 2;
    directory.set_now(t1);
    let other_identity = tempfile::tempdir().unwrap();
    let other_keystore = VhcKeystore::open(other_identity.path()).unwrap();
    let takeover = daemon_vhc_node::seat::author_claim(
        &other_keystore,
        &daemon_vhc_node::seat::CoordinatorSeat {
            run_label: RUN,
            genesis_hash: GENESIS,
            role: ROLE,
            epoch: 0,
            module_hash: MODULE,
            endpoint: candidate().endpoint,
        },
        1,
        t1,
    )
    .unwrap();
    assert_eq!(
        directory.registry.claim(RUN, &takeover, t1).decision,
        SeatDecision::Accepted,
        "an expired lease is taken over at floor + 1"
    );

    // The old holder's renew is REFUSED: the keeper drops the fenced lease (never a fight).
    let notes = keeper.tick(&[candidate()], t1).await;
    assert!(
        matches!(notes.as_slice(), [SeatNote::Fenced { run_label, .. }] if run_label == RUN),
        "got {notes:?}"
    );
    assert_eq!(keeper.held_incarnation(RUN), None);

    // While the new incumbent is live, the keeper stands by (no bid derives).
    let notes = keeper.tick(&[candidate()], t1 + 1_000).await;
    assert!(notes.is_empty(), "stand by: {notes:?}");
}
