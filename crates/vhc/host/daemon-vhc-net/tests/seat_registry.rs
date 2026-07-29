// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Coordinator seat-lease suite: [`RegistryClient`]'s seat surface against a mock registry that
//! serves the frozen HTTP contract (`GET`/`PUT` `…/seat/:role`, `POST …/seat/:role/heartbeat`,
//! `DELETE …/seat/:role`; canonical CBOR both ways; 200 accepted / 409 refused) backed by the
//! [`FakeSeatRegistry`] fixture — i.e. by the normative `SeatSlot::fold`.
//!
//! The registry-posture split is asserted explicitly: the registry ACCEPTS a structurally-valid
//! lease whose authority is garbage (untrusted base, forged signature) — it is untrusted storage
//! and never judges authority — while a PEER verifying the same stored object refuses it typed.
//! Fencing (the token CAS + the certificate supersession floor) is what protects the run, never
//! the registry's opinion.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use daemon_egress::{EgressClient, EgressConfig};
use daemon_vhc_net::{FakeSeatRegistry, RegistryClient, RunId, SeatClaimOutcome};
use daemon_vhc_proto::cert::{CertScope, RunKeyCertificate};
use daemon_vhc_proto::domains::{SEAT_LEASE_DOMAIN, SEAT_RELEASE_DOMAIN};
use daemon_vhc_proto::{
    peer_id, to_canonical_vec, ControlEndpoint, Hash, PeerId, RevocationLedger, SeatDecision,
    SeatLease, SeatLeaseBody, SeatLeaseError, SeatMutationResponse, SeatRelease, SeatReleaseBody,
    SeatState, SigningKey, DEFAULT_SEAT_HEARTBEAT_MS, DEFAULT_SEAT_SKEW_MS,
};
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const RUN_LABEL: &str = "seat-run";
const ROLE: &str = "coordinator";
const TTL_MS: u64 = 30_000;

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn run_hash() -> Hash {
    Hash([0x11; 32])
}

fn module_hash() -> Hash {
    Hash([0xCC; 32])
}

fn body(claimant: PeerId, incarnation: u64, issued_at_ms: u64) -> SeatLeaseBody {
    SeatLeaseBody {
        domain: SEAT_LEASE_DOMAIN.to_string(),
        run_id: run_hash(),
        role: ROLE.to_string(),
        epoch: 0,
        incarnation,
        fencing_token: incarnation,
        claimant,
        module_hash: module_hash(),
        endpoint: ControlEndpoint {
            ws: Some(format!("wss://registry.example/runs/{RUN_LABEL}/ws")),
            iroh_ticket: None,
        },
        issued_at_ms,
        expires_at_ms: issued_at_ms + TTL_MS,
        heartbeat_interval_ms: DEFAULT_SEAT_HEARTBEAT_MS,
    }
}

/// A claim by run key `seed`, certified by `base`, at `incarnation`, issued at `issued_at_ms`.
fn lease(base: &SigningKey, seed: u8, incarnation: u64, issued_at_ms: u64) -> SeatLease {
    let run_key = key(seed);
    let claimant = peer_id(&run_key);
    let b = body(claimant, incarnation, issued_at_ms);
    let cert = RunKeyCertificate::issue(base, b.cert_scope(), claimant).expect("issue cert");
    SeatLease::claim(&run_key, cert, b).expect("author lease")
}

// -- the mock server: the frozen HTTP surface over the FakeSeatRegistry fixture --------------------

/// Shared registry + registry clock. The clock is test-advanced (no wall time), mirroring how the
/// remote registry judges expiry with ITS clock, not the claimant's.
#[derive(Clone)]
struct SeatBackend {
    registry: Arc<FakeSeatRegistry>,
    now_ms: Arc<AtomicU64>,
}

impl SeatBackend {
    fn new() -> Self {
        Self {
            registry: Arc::new(FakeSeatRegistry::with_skew(DEFAULT_SEAT_SKEW_MS)),
            now_ms: Arc::new(AtomicU64::new(1_000)),
        }
    }

    fn set_now(&self, now_ms: u64) {
        self.now_ms.store(now_ms, Ordering::SeqCst);
    }
}

/// Parse `(run, role)` out of `/api/v1/vhc/runs/:id/seat/:role[/heartbeat]`.
fn seat_path(req: &Request) -> (String, String) {
    let segments: Vec<&str> = req.url.path().trim_matches('/').split('/').collect();
    let runs = segments.iter().position(|s| *s == "runs").expect("runs");
    (
        segments[runs + 1].to_string(),
        segments[runs + 3].to_string(),
    )
}

fn cbor_response(status: u16, resp: &SeatMutationResponse) -> ResponseTemplate {
    ResponseTemplate::new(status).set_body_bytes(to_canonical_vec(resp).expect("encode response"))
}

fn mutation_status(resp: &SeatMutationResponse) -> u16 {
    match resp.decision {
        SeatDecision::Accepted => 200,
        _ => 409,
    }
}

struct ReadSeat(SeatBackend);
impl Respond for ReadSeat {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let (run, role) = seat_path(req);
        let state = self.0.registry.read(&run, &role);
        ResponseTemplate::new(200).set_body_bytes(to_canonical_vec(&state).expect("encode state"))
    }
}

struct ClaimSeat(SeatBackend);
impl Respond for ClaimSeat {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let (run, _) = seat_path(req);
        let lease: SeatLease = match daemon_vhc_proto::from_canonical_slice(&req.body) {
            Ok(l) => l,
            Err(e) => return ResponseTemplate::new(400).set_body_string(e.to_string()),
        };
        let now = self.0.now_ms.load(Ordering::SeqCst);
        let resp = self.0.registry.claim(&run, &lease, now);
        cbor_response(mutation_status(&resp), &resp)
    }
}

struct RenewSeat(SeatBackend);
impl Respond for RenewSeat {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let (run, _) = seat_path(req);
        let lease: SeatLease = match daemon_vhc_proto::from_canonical_slice(&req.body) {
            Ok(l) => l,
            Err(e) => return ResponseTemplate::new(400).set_body_string(e.to_string()),
        };
        let now = self.0.now_ms.load(Ordering::SeqCst);
        let resp = self.0.registry.renew(&run, &lease, now);
        cbor_response(mutation_status(&resp), &resp)
    }
}

struct ReleaseSeat(SeatBackend);
impl Respond for ReleaseSeat {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let (run, role) = seat_path(req);
        let release: SeatRelease = match daemon_vhc_proto::from_canonical_slice(&req.body) {
            Ok(r) => r,
            Err(e) => return ResponseTemplate::new(400).set_body_string(e.to_string()),
        };
        let now = self.0.now_ms.load(Ordering::SeqCst);
        let resp = self.0.registry.release(&run, &role, &release, now);
        cbor_response(mutation_status(&resp), &resp)
    }
}

async fn seat_server(backend: &SeatBackend) -> (MockServer, RegistryClient) {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path_regex(r"^/api/v1/vhc/runs/[^/]+/seat/[^/]+$"))
        .respond_with(ReadSeat(backend.clone()))
        .mount(&server)
        .await;
    Mock::given(method("PUT"))
        .and(path_regex(r"^/api/v1/vhc/runs/[^/]+/seat/[^/]+$"))
        .respond_with(ClaimSeat(backend.clone()))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path_regex(r"^/api/v1/vhc/runs/[^/]+/seat/[^/]+/heartbeat$"))
        .respond_with(RenewSeat(backend.clone()))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path_regex(r"^/api/v1/vhc/runs/[^/]+/seat/[^/]+$"))
        .respond_with(ReleaseSeat(backend.clone()))
        .mount(&server)
        .await;
    let egress = EgressClient::new(EgressConfig::default()).expect("egress");
    let base = format!("{}/api/v1/vhc", server.uri());
    let client = RegistryClient::new(egress, base).with_bearer("vhc-token");
    (server, client)
}

// -- the suites -------------------------------------------------------------------------------------

#[tokio::test]
async fn full_seat_lifecycle_over_the_frozen_http_surface() {
    let backend = SeatBackend::new();
    let (_server, client) = seat_server(&backend).await;
    let run = RunId::new(RUN_LABEL);
    let base = key(1);
    let trusted = [peer_id(&base)];

    // Unclaimed, no floor.
    assert_eq!(
        client.read_seat(&run, ROLE).await.expect("read"),
        Some(SeatState::Unclaimed {
            last_fencing_token: None
        })
    );

    // Claim wins; the stored lease reads back byte-equal and authorizes peer-side.
    let l0 = lease(&base, 2, 0, 1_000);
    match client.claim_seat(&run, &l0).await.expect("claim") {
        SeatClaimOutcome::Won(l) => assert_eq!(l, l0),
        lost => panic!("first claim must win: {lost:?}"),
    }
    match client.read_seat(&run, ROLE).await.expect("read") {
        Some(SeatState::Leased(l)) => {
            assert_eq!(*l, l0);
            l.authorize(&trusted, 2_000, DEFAULT_SEAT_SKEW_MS)
                .expect("the stored lease authorizes against the genesis-trusted base");
        }
        other => panic!("expected the stored lease: {other:?}"),
    }

    // Renew (heartbeat): a re-signed body under the same identity/token extends expiry.
    backend.set_now(11_000);
    let run_key = key(2);
    let mut renewed_body = l0.body.clone();
    renewed_body.issued_at_ms = 11_000;
    renewed_body.expires_at_ms = 11_000 + TTL_MS;
    let renewed = SeatLease::claim(&run_key, l0.certificate.clone(), renewed_body).expect("renew");
    assert!(matches!(
        client.renew_seat(&run, &renewed).await.expect("renew"),
        SeatClaimOutcome::Won(_)
    ));

    // Release: the seat unclaims but the fencing floor persists (tokens never reset).
    let release = SeatRelease::sign(
        &run_key,
        SeatReleaseBody {
            domain: SEAT_RELEASE_DOMAIN.to_string(),
            run_id: run_hash(),
            role: ROLE.to_string(),
            incarnation: 0,
            fencing_token: 0,
            claimant: peer_id(&run_key),
        },
    )
    .expect("sign release");
    client
        .release_seat(&run, ROLE, &release)
        .await
        .expect("release accepted");
    assert_eq!(
        client.read_seat(&run, ROLE).await.expect("read"),
        Some(SeatState::Unclaimed {
            last_fencing_token: Some(0)
        })
    );

    // A stale release (already released) is a typed refusal, not a silent success.
    assert!(client.release_seat(&run, ROLE, &release).await.is_err());
}

#[tokio::test]
async fn concurrent_claims_produce_one_winner_and_typed_losers_who_re_read() {
    let backend = SeatBackend::new();
    let (_server, client) = seat_server(&backend).await;
    let run = RunId::new(RUN_LABEL);
    let base = key(1);

    // Four claimants race the virgin slot at the same first token.
    let leases: Vec<SeatLease> = (0..4).map(|i| lease(&base, 10 + i, 0, 1_000)).collect();
    let (a, b, c, d) = tokio::join!(
        client.claim_seat(&run, &leases[0]),
        client.claim_seat(&run, &leases[1]),
        client.claim_seat(&run, &leases[2]),
        client.claim_seat(&run, &leases[3]),
    );

    let mut winners = Vec::new();
    let mut losers = Vec::new();
    for outcome in [a, b, c, d] {
        match outcome.expect("claim transport") {
            SeatClaimOutcome::Won(l) => winners.push(l),
            SeatClaimOutcome::Lost { decision, state } => losers.push((decision, state)),
        }
    }
    assert_eq!(winners.len(), 1, "exactly one claimant wins the CAS");
    assert_eq!(losers.len(), 3, "every other claimant loses typed");
    let winner = &winners[0];
    for (decision, state) in &losers {
        // A live lease by another claimant refuses as HELD; the carried state is the re-read —
        // the loser learns the incumbent without a second round-trip.
        assert_eq!(*decision, SeatDecision::RejectedHeld);
        assert_eq!(*state, SeatState::Leased(Box::new(winner.clone())));
    }

    // A loser bidding token+1 while the incumbent is live still loses (fencing never overrides
    // liveness — takeover needs expiry first).
    let eager = lease(&base, 20, 1, 2_000);
    assert!(matches!(
        client.claim_seat(&run, &eager).await.expect("claim"),
        SeatClaimOutcome::Lost {
            decision: SeatDecision::RejectedHeld,
            ..
        }
    ));
}

#[tokio::test]
async fn a_fenced_stale_claimant_is_refused_by_registry_and_peers() {
    let backend = SeatBackend::new();
    let (_server, client) = seat_server(&backend).await;
    let run = RunId::new(RUN_LABEL);
    let base = key(1);
    let trusted = [peer_id(&base)];

    // Incarnation 0 claims, then goes silent past expiry + skew.
    let l0 = lease(&base, 2, 0, 1_000);
    assert!(matches!(
        client.claim_seat(&run, &l0).await.expect("claim"),
        SeatClaimOutcome::Won(_)
    ));
    backend.set_now(1_000 + TTL_MS + DEFAULT_SEAT_SKEW_MS + 1);

    // The standby takes over at exactly floor + 1 (a fresh incarnation).
    let l1 = lease(&base, 3, 1, 40_000);
    assert!(matches!(
        client.claim_seat(&run, &l1).await.expect("takeover"),
        SeatClaimOutcome::Won(_)
    ));

    // REGISTRY fence: the stale claimant's renew at the old token is a typed conflict carrying
    // the current state (it learns it was superseded).
    let stale = match client.renew_seat(&run, &l0).await.expect("stale renew") {
        SeatClaimOutcome::Lost { decision, state } => (decision, state),
        won => panic!("a superseded renew must lose: {won:?}"),
    };
    assert_eq!(
        stale.0,
        SeatDecision::RejectedFencingConflict {
            expected: 1,
            got: 0
        }
    );
    assert_eq!(stale.1, SeatState::Leased(Box::new(l1.clone())));

    // PEER fence (architecture §6.3.1): observing the takeover's certificate advances the
    // supersession floor, so the old incarnation's scope is judged dead — with or without the
    // registry's answer, and with no explicit revocation record delivered.
    let mut ledger = RevocationLedger::new();
    ledger.observe_certificates(std::slice::from_ref(&l1.certificate));
    let stale_scope = l0.body.cert_scope();
    assert!(
        ledger
            .judge(
                &stale_scope,
                &l0.body.claimant,
                &l0.certificate.base_identity
            )
            .is_err(),
        "the superseded incarnation is below the supersession floor"
    );
    let live_scope = l1.body.cert_scope();
    assert!(ledger
        .judge(
            &live_scope,
            &l1.body.claimant,
            &l1.certificate.base_identity
        )
        .is_ok());
    // The new lease itself authorizes; the fenced one still carries a chain-valid cert — the
    // floor, not the chain, is what kills it (supersession is the safety floor).
    assert!(l1.authorize(&trusted, 40_000, DEFAULT_SEAT_SKEW_MS).is_ok());
    assert!(
        l0.authorize(&trusted, 40_000, DEFAULT_SEAT_SKEW_MS)
            .is_err(),
        "and the stale lease is expired by wall clock anyway"
    );
}

#[tokio::test]
async fn the_registry_stores_what_peers_refuse() {
    // The untrusted-registry posture split: the registry is storage — it ACCEPTS a structurally-valid
    // lease whose authority is garbage; PEERS refuse the same object typed. No panic, no silent
    // drop on either side.
    let backend = SeatBackend::new();
    let (_server, client) = seat_server(&backend).await;
    let run = RunId::new(RUN_LABEL);
    let honest_base = key(1);
    let trusted = [peer_id(&honest_base)];

    // An attacker with its own base identity authors a self-consistent lease.
    let attacker_base = key(66);
    let forged = lease(&attacker_base, 67, 0, 1_000);
    match client.claim_seat(&run, &forged).await.expect("claim") {
        SeatClaimOutcome::Won(stored) => {
            // Structurally accepted and stored — the registry never judged authority…
            assert_eq!(stored, forged);
        }
        lost => panic!("the registry must accept the structurally-valid forgery: {lost:?}"),
    }
    // …and every peer refuses it: the certificate chains to a base the genesis never named.
    let stored = match client.read_seat(&run, ROLE).await.expect("read") {
        Some(SeatState::Leased(l)) => l,
        other => panic!("stored: {other:?}"),
    };
    assert_eq!(
        stored.authorize(&trusted, 2_000, DEFAULT_SEAT_SKEW_MS),
        Err(SeatLeaseError::UntrustedBase)
    );

    // A tampered self-signature: still stored by a conforming registry (no signature checks
    // registry-side), still refused typed by peers.
    let mut tampered = lease(&honest_base, 2, 1, 2_000);
    tampered.sig.0[0] ^= 0xff;
    backend.set_now(1_000 + TTL_MS + DEFAULT_SEAT_SKEW_MS + 1); // let the forgery expire first
    match client.claim_seat(&run, &tampered).await.expect("claim") {
        SeatClaimOutcome::Won(_) => {}
        lost => panic!("the registry must store the tampered lease too: {lost:?}"),
    }
    assert_eq!(
        tampered.verify_signature(),
        Err(SeatLeaseError::BadSignature)
    );
    assert_eq!(
        tampered.authorize(&trusted, 2_000, DEFAULT_SEAT_SKEW_MS),
        Err(SeatLeaseError::BadSignature)
    );

    // And a certificate whose scope does not match the lease (wrong module) is a typed cert
    // refusal peer-side — while remaining registry-acceptable.
    let run_key = key(5);
    let claimant = peer_id(&run_key);
    let b = body(claimant, 2, 3_000);
    let wrong_scope = CertScope {
        module_hash: Hash([0xDD; 32]),
        ..b.cert_scope()
    };
    let cert = RunKeyCertificate::issue(&honest_base, wrong_scope, claimant).expect("cert");
    let mismatched = SeatLease::claim(&run_key, cert, b).expect("lease");
    backend.set_now(2_000 + TTL_MS + DEFAULT_SEAT_SKEW_MS + 1);
    assert!(matches!(
        client.claim_seat(&run, &mismatched).await.expect("claim"),
        SeatClaimOutcome::Won(_)
    ));
    assert!(matches!(
        mismatched.authorize(&trusted, 3_500, DEFAULT_SEAT_SKEW_MS),
        Err(SeatLeaseError::Cert(_))
    ));
}
