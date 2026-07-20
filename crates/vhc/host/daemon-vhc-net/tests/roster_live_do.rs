// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! LIVE iroh-roster lifecycle against the deployed dev registry (the cloud roster slots behind
//! the frozen node↔cloud contract): publish → snapshot read-back + peer-side authorize →
//! re-address republish supersedes → a stale republish loses typed with the stored record as its
//! re-read → a rejoin (higher incarnation) supersedes. The registry stores and upserts; every
//! acceptance judgment asserted here runs CLIENT-side (`RosterRecord::authorize` against the
//! test-local trusted base) — the live half of the untrusted-storage posture the offline suites
//! pin.
//!
//! SKIPS cleanly (the `ws_live_do` convention) unless `VHC_LIVE_REGISTRY_URL` (or
//! `VHC_LIVE_WS_URL`) is set, so it never runs in the offline workspace gate. Drive it after
//! seeding a run (`apps/vhc scripts/seed_run.mjs <run>` in the cloud repo):
//!   VHC_LIVE_REGISTRY_URL=https://daemon-vhc-dev.<acct>.workers.dev/api/v1/vhc \
//!     VHC_LIVE_ROSTER_RUN=run-roster-live \
//!     cargo test -p daemon-vhc-net --test roster_live_do -- --nocapture
//!
//! Idempotent against durable slots: freshness stamps derive from the wall clock, so re-runs
//! always publish at a fresher `(incarnation, issued_at_ms)` than any previous run left behind.

use std::time::{SystemTime, UNIX_EPOCH};

use daemon_egress::{EgressClient, EgressConfig};
use daemon_vhc_net::{RegistryClient, RosterPublishOutcome, RunId};
use daemon_vhc_proto::bytes::IrohId;
use daemon_vhc_proto::cert::RunKeyCertificate;
use daemon_vhc_proto::domains::ROSTER_RECORD_DOMAIN;
use daemon_vhc_proto::{
    blake3_hash, peer_id, Hash, PeerId, RosterDecision, RosterRecord, RosterRecordBody,
    RosterRecordError, SigningKey,
};

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
    sender: PeerId,
    endpoint_id: IrohId,
    incarnation: u64,
    issued_at_ms: u64,
) -> RosterRecordBody {
    RosterRecordBody {
        domain: ROSTER_RECORD_DOMAIN.to_string(),
        run_id,
        role: "trainer".to_string(),
        epoch: 0,
        incarnation,
        sender,
        module_hash: Hash([0xCC; 32]),
        endpoint_id,
        direct_addrs: vec!["127.0.0.1:4550".to_string()],
        relay_url: None,
        issued_at_ms,
    }
}

fn record(
    base: &SigningKey,
    run_key: &SigningKey,
    run_id: Hash,
    endpoint_id: IrohId,
    incarnation: u64,
    issued_at_ms: u64,
) -> RosterRecord {
    let sender = peer_id(run_key);
    let b = body(run_id, sender, endpoint_id, incarnation, issued_at_ms);
    let cert = RunKeyCertificate::issue(base, b.cert_scope(), sender).expect("issue cert");
    RosterRecord::publish(run_key, cert, b).expect("author record")
}

#[tokio::test]
async fn live_roster_lifecycle_against_the_dev_registry() {
    let Some(base_url) = std::env::var("VHC_LIVE_REGISTRY_URL")
        .or_else(|_| std::env::var("VHC_LIVE_WS_URL"))
        .ok()
    else {
        eprintln!("skipping: set VHC_LIVE_REGISTRY_URL to run the live roster lifecycle");
        return;
    };
    let run_label =
        std::env::var("VHC_LIVE_ROSTER_RUN").unwrap_or_else(|_| "run-roster-live".into());
    let run = RunId::new(&run_label);
    let client = live_client(&base_url);

    // Fresh identities per run of the test: freshness starts above anything a prior run stored.
    let base = test_key("roster-live/base");
    let run_key = test_key("roster-live/run-key");
    let endpoint_id = IrohId(*blake3_hash(format!("ep/{}", now_ms()).as_bytes()).as_bytes());
    let run_id = Hash(*blake3_hash(run_label.as_bytes()).as_bytes());
    let incarnation = 1;
    let t0 = now_ms();

    // 1. Publish, then read back + authorize CLIENT-side (the registry never vouches).
    let first = record(&base, &run_key, run_id, endpoint_id, incarnation, t0);
    assert_eq!(
        client
            .publish_roster(&run, &first)
            .await
            .expect("live publish"),
        RosterPublishOutcome::Accepted
    );
    let fetched = client.fetch_roster(&run).await.expect("live fetch");
    let mine = fetched
        .iter()
        .find(|r| r.body.endpoint_id == endpoint_id)
        .expect("published record is served back");
    assert_eq!(mine.body, first.body);
    mine.authorize(&[peer_id(&base)])
        .expect("the read-back record authorizes against the test-local trusted base");
    // ...and under a DIFFERENT trust set the same stored object is refused (posture check).
    assert_eq!(
        mine.authorize(&[peer_id(&test_key("roster-live/stranger"))]),
        Err(RosterRecordError::UntrustedBase)
    );

    // 2. Re-address republish (same incarnation, later issue) supersedes.
    let readdressed = record(&base, &run_key, run_id, endpoint_id, incarnation, t0 + 1);
    assert_eq!(
        client
            .publish_roster(&run, &readdressed)
            .await
            .expect("live republish"),
        RosterPublishOutcome::Accepted
    );

    // 3. A stale republish loses typed, carrying the stored record as the re-read.
    match client
        .publish_roster(&run, &first)
        .await
        .expect("live stale publish")
    {
        RosterPublishOutcome::Refused { decision, stored } => {
            assert_eq!(
                decision,
                RosterDecision::RejectedStale {
                    stored_incarnation: incarnation,
                    stored_issued_at_ms: t0 + 1
                }
            );
            assert_eq!(stored.expect("stored record").body, readdressed.body);
        }
        other => panic!("expected the stale refusal, got {other:?}"),
    }

    // 4. A rejoin (higher incarnation, fresh per-run key — the rotation rule) supersedes.
    let rejoin_key = test_key("roster-live/run-key-2");
    let rejoined = record(&base, &rejoin_key, run_id, endpoint_id, incarnation + 1, t0);
    assert_eq!(
        client
            .publish_roster(&run, &rejoined)
            .await
            .expect("live rejoin publish"),
        RosterPublishOutcome::Accepted
    );
    let fetched = client.fetch_roster(&run).await.expect("live re-fetch");
    let mine = fetched
        .iter()
        .find(|r| r.body.endpoint_id == endpoint_id)
        .expect("rejoined record is served back");
    assert_eq!(mine.body.incarnation, incarnation + 1);
    println!(
        "live roster lifecycle green against {base_url} (run `{run_label}`, {} entries served)",
        fetched.len()
    );
}
