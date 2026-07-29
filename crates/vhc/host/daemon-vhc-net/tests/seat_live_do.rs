// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! LIVE seat-lease lifecycle against the deployed dev registry (the cloud seat slots behind
//! the frozen node↔cloud contract, ABI spec §12.4): claim → read-back + peer-side authorize →
//! heartbeat/renew → a live contender loses typed with the incumbent as its re-read → release →
//! tombstone floor. The registry stores and CASes; every acceptance judgment asserted here runs
//! CLIENT-side (`SeatLease::authorize` against the test-local trusted base) — the live half of
//! the untrusted-storage posture the offline suites pin.
//!
//! SKIPS cleanly (the `ws_live_do` convention) unless `VHC_LIVE_REGISTRY_URL` (or
//! `VHC_LIVE_WS_URL`) is set, so it never runs in the offline workspace gate. Drive it after
//! seeding a run (`apps/vhc scripts/seed_run.mjs <run>` in the cloud repo):
//!   VHC_LIVE_REGISTRY_URL=https://daemon-vhc-dev.<acct>.workers.dev/api/v1/vhc \
//!     VHC_LIVE_SEAT_RUN=run-seat-live \
//!     cargo test -p daemon-vhc-net --test seat_live_do -- --nocapture
//!
//! Idempotent against a durable slot: the bid token is derived from the CURRENT slot state
//! (floor + 1 after a previous run's release/expiry), so re-runs never assume a virgin slot.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use daemon_egress::{EgressClient, EgressConfig};
use daemon_vhc_net::{RegistryClient, RunId, SeatClaimOutcome};
use daemon_vhc_proto::cert::RunKeyCertificate;
use daemon_vhc_proto::domains::{SEAT_LEASE_DOMAIN, SEAT_RELEASE_DOMAIN};
use daemon_vhc_proto::{
    blake3_hash, peer_id, ControlEndpoint, Hash, SeatDecision, SeatLease, SeatLeaseBody,
    SeatRelease, SeatReleaseBody, SeatState, SigningKey, DEFAULT_SEAT_HEARTBEAT_MS,
    DEFAULT_SEAT_SKEW_MS,
};

const ROLE: &str = "coordinator";
const TTL_MS: u64 = 30_000;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_millis() as u64
}

/// A test-local signing key from a label + wall clock (live smoke only — production keys are
/// CSPRNG-minted by the identity subsystem; this test's trust root is its own base key, so
/// nothing here can authenticate against a real run's genesis).
fn test_key(label: &str) -> SigningKey {
    let seed = blake3_hash(format!("{label}/{}/{}", std::process::id(), now_ms()).as_bytes());
    SigningKey::from_bytes(seed.as_bytes())
}

fn live_client(base_url: &str) -> RegistryClient {
    let egress = EgressClient::new(EgressConfig::default()).expect("egress client");
    RegistryClient::new(egress, base_url).with_internal(
        std::env::var("VHC_LIVE_ORG").unwrap_or_else(|_| "org_live".into()),
        std::env::var("VHC_LIVE_ACTOR").unwrap_or_else(|_| "key:live".into()),
    )
}

fn body(
    run_id: Hash,
    incarnation: u64,
    claimant: daemon_vhc_proto::PeerId,
    issued_at_ms: u64,
    ws_url: &str,
) -> SeatLeaseBody {
    SeatLeaseBody {
        domain: SEAT_LEASE_DOMAIN.to_string(),
        run_id,
        role: ROLE.to_string(),
        epoch: 0,
        incarnation,
        leadership_term: incarnation,
        claimant,
        module_hash: Hash([0xCC; 32]),
        endpoint: ControlEndpoint {
            ws: Some(ws_url.to_string()),
            iroh_ticket: None,
        },
        issued_at_ms,
        expires_at_ms: issued_at_ms + TTL_MS,
        heartbeat_interval_ms: DEFAULT_SEAT_HEARTBEAT_MS,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn live_seat_lifecycle_against_the_deployed_registry() {
    let Ok(base_url) =
        std::env::var("VHC_LIVE_REGISTRY_URL").or_else(|_| std::env::var("VHC_LIVE_WS_URL"))
    else {
        eprintln!(
            "SKIP live seat lifecycle: set VHC_LIVE_REGISTRY_URL \
             (e.g. https://daemon-vhc-dev.<acct>.workers.dev/api/v1/vhc)"
        );
        return;
    };
    let run_label = std::env::var("VHC_LIVE_SEAT_RUN").unwrap_or_else(|_| "run-seat-live".into());
    let run = RunId::new(run_label.clone());
    let client = live_client(&base_url);

    // The lease body's run identity: the registry structurally stores whatever 32-byte id the
    // slot was keyed with first; peers bind it to the genesis hash. For the live smoke the id
    // is derived from the run label so re-runs address the same slot consistently.
    let run_id = blake3_hash(run_label.as_bytes());
    let ws_url = format!("{base_url}/runs/{run_label}/ws");

    // -- derive an idempotent bid from the CURRENT slot state --------------------------------
    let state = client
        .read_seat(&run, ROLE)
        .await
        .expect("read seat")
        .expect("the seeded run exists (seed it with the cloud seed script)");
    let bid = match &state {
        SeatState::Unclaimed {
            last_leadership_term,
        } => last_leadership_term.map_or(0, |f| f + 1),
        SeatState::Leased(held) => {
            // A previous run's lease may still be live for up to TTL + skew: wait it out.
            let deadline = held.body.expires_at_ms.saturating_add(DEFAULT_SEAT_SKEW_MS) + 2_000;
            let wait = deadline.saturating_sub(now_ms());
            assert!(
                wait < 120_000,
                "held lease expires unreasonably far in the future ({wait} ms)"
            );
            if wait > 0 {
                eprintln!("waiting {wait} ms for the previous holder's lease to expire");
                tokio::time::sleep(Duration::from_millis(wait)).await;
            }
            held.body.leadership_term + 1
        }
    };

    // -- identities: a fresh per-run key certified by a test-local base ----------------------
    let base_key = test_key("seat-live/base");
    let trusted = [peer_id(&base_key)];
    let run_key = test_key("seat-live/run-key");
    let claimant = peer_id(&run_key);

    // -- claim at the bid (this smoke reuses the bid as the execution incarnation; the
    //    identity-scope semantics live in the proto fold vectors) ----------------------------
    let b0 = body(run_id, bid, claimant, now_ms(), &ws_url);
    let cert = RunKeyCertificate::issue(&base_key, b0.cert_scope(), claimant).expect("cert");
    let lease = SeatLease::claim(&run_key, cert.clone(), b0).expect("author lease");
    match client.claim_seat(&run, &lease).await.expect("claim") {
        SeatClaimOutcome::Won(l) => assert_eq!(l, lease),
        SeatClaimOutcome::Lost { decision, state } => {
            panic!("claim at the derived bid must win: {decision:?} / {state:?}")
        }
    }

    // -- read-back: the stored object round-trips and authorizes PEER-side --------------------
    match client
        .read_seat(&run, ROLE)
        .await
        .expect("read")
        .expect("run")
    {
        SeatState::Leased(stored) => {
            assert_eq!(*stored, lease, "the registry echoes the stored lease");
            stored
                .authorize(&trusted, now_ms(), DEFAULT_SEAT_SKEW_MS)
                .expect("the stored lease authorizes against the trusted base");
        }
        other => panic!("expected the stored lease: {other:?}"),
    }

    // -- heartbeat/renew: a re-signed body under the same identity + term ---------------------
    let mut renewed_body = lease.body.clone();
    renewed_body.issued_at_ms = now_ms();
    renewed_body.expires_at_ms = renewed_body.issued_at_ms + TTL_MS;
    let renewed = SeatLease::claim(&run_key, cert, renewed_body).expect("renew lease");
    match client.renew_seat(&run, &renewed).await.expect("renew") {
        SeatClaimOutcome::Won(_) => {}
        lost => panic!("the holder's renew must win: {lost:?}"),
    }

    // -- a live contender loses TYPED, with the incumbent carried as its re-read --------------
    let contender_key = test_key("seat-live/contender");
    let contender_id = peer_id(&contender_key);
    let cb = body(run_id, bid + 1, contender_id, now_ms(), &ws_url);
    let contender_cert =
        RunKeyCertificate::issue(&base_key, cb.cert_scope(), contender_id).expect("cert");
    let contender = SeatLease::claim(&contender_key, contender_cert, cb).expect("lease");
    match client.claim_seat(&run, &contender).await.expect("contend") {
        SeatClaimOutcome::Lost { decision, state } => {
            assert_eq!(decision, SeatDecision::RejectedHeld);
            match state {
                SeatState::Leased(incumbent) => assert_eq!(
                    *incumbent, renewed,
                    "the refusal carries the incumbent as the loser's re-read"
                ),
                other => panic!("refusal state: {other:?}"),
            }
        }
        SeatClaimOutcome::Won(_) => panic!("a contender must not displace a live lease"),
    }

    // -- release: the seat unclaims; the tombstone floor persists -----------------------------
    let release = SeatRelease::sign(
        &run_key,
        SeatReleaseBody {
            domain: SEAT_RELEASE_DOMAIN.to_string(),
            run_id,
            role: ROLE.to_string(),
            incarnation: bid,
            leadership_term: bid,
            claimant,
        },
    )
    .expect("sign release");
    client
        .release_seat(&run, ROLE, &release)
        .await
        .expect("release accepted");
    match client
        .read_seat(&run, ROLE)
        .await
        .expect("read")
        .expect("run")
    {
        SeatState::Unclaimed {
            last_leadership_term,
        } => {
            assert_eq!(
                last_leadership_term,
                Some(bid),
                "the floor survives the release (terms never reset)"
            );
        }
        other => panic!("expected the tombstoned slot: {other:?}"),
    }
    eprintln!("live seat lifecycle green at bid term {bid} against {base_url}");
}
