// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Iroh roster suite: [`RegistryClient`]'s roster surface against a mock registry that serves
//! the frozen HTTP contract (`GET`/`PUT {base}/runs/:id/roster`; canonical CBOR both ways; 200
//! accepted / 409 refused / 400 shape error) backed by the [`FakeRosterRegistry`] fixture —
//! i.e. by the normative `RosterSlot::fold`.
//!
//! The registry-posture split is asserted explicitly: the registry ACCEPTS a structurally-valid
//! record whose authority is garbage (untrusted base, forged signature) — it is untrusted
//! storage and never judges authority — while a PEER verifying the same stored object refuses it
//! typed. Freshness precedence (`(incarnation, issued_at_ms)`, grouped by `(role, base)`) is
//! what protects readers from stale addresses, never the registry's opinion.

use std::sync::Arc;

use daemon_egress::{EgressClient, EgressConfig};
use daemon_vhc_net::{FakeRosterRegistry, RegistryClient, RosterPublishOutcome, RunId};
use daemon_vhc_proto::bytes::IrohId;
use daemon_vhc_proto::cert::RunKeyCertificate;
use daemon_vhc_proto::domains::ROSTER_RECORD_DOMAIN;
use daemon_vhc_proto::{
    freshest_per_node, peer_id, to_canonical_vec, Hash, PeerId, RosterDecision, RosterRecord,
    RosterRecordBody, RosterRecordError, SigningKey,
};
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const RUN_LABEL: &str = "roster-run";

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn body(sender: PeerId, endpoint: u8, incarnation: u64, issued_at_ms: u64) -> RosterRecordBody {
    RosterRecordBody {
        domain: ROSTER_RECORD_DOMAIN.to_string(),
        run_id: Hash([0x11; 32]),
        role: "trainer".to_string(),
        epoch: 0,
        incarnation,
        sender,
        module_hash: Hash([0xCC; 32]),
        endpoint_id: IrohId([endpoint; 32]),
        direct_addrs: vec![format!("127.0.0.1:{}", 4000 + u16::from(endpoint))],
        relay_url: None,
        issued_at_ms,
    }
}

/// A record by run key `seed`, certified by `base`, publishing endpoint `endpoint`.
fn record(
    base: &SigningKey,
    seed: u8,
    endpoint: u8,
    incarnation: u64,
    issued_at_ms: u64,
) -> RosterRecord {
    let run_key = key(seed);
    let sender = peer_id(&run_key);
    let b = body(sender, endpoint, incarnation, issued_at_ms);
    let cert = RunKeyCertificate::issue(base, b.cert_scope(), sender).expect("issue cert");
    RosterRecord::publish(&run_key, cert, b).expect("author record")
}

// -- the mock server: the frozen HTTP surface over the FakeRosterRegistry fixture ------------------

fn run_of(req: &Request) -> String {
    let segments: Vec<&str> = req.url.path().trim_matches('/').split('/').collect();
    let runs = segments.iter().position(|s| *s == "runs").expect("runs");
    segments[runs + 1].to_string()
}

struct PublishRoster(Arc<FakeRosterRegistry>);
impl Respond for PublishRoster {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let record: RosterRecord = match daemon_vhc_proto::from_canonical_slice(&req.body) {
            Ok(r) => r,
            Err(e) => return ResponseTemplate::new(400).set_body_string(e.to_string()),
        };
        let resp = self.0.publish(&run_of(req), &record);
        let status = match resp.decision {
            RosterDecision::Accepted => 200,
            _ => 409,
        };
        ResponseTemplate::new(status)
            .set_body_bytes(to_canonical_vec(&resp).expect("encode response"))
    }
}

struct ReadRoster(Arc<FakeRosterRegistry>);
impl Respond for ReadRoster {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let snapshot = self.0.snapshot(&run_of(req));
        ResponseTemplate::new(200)
            .set_body_bytes(to_canonical_vec(&snapshot).expect("encode snapshot"))
    }
}

async fn roster_server(registry: Arc<FakeRosterRegistry>) -> (MockServer, RegistryClient) {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path_regex(r"^/api/v1/vhc/runs/[^/]+/roster$"))
        .respond_with(ReadRoster(registry.clone()))
        .mount(&server)
        .await;
    Mock::given(method("PUT"))
        .and(path_regex(r"^/api/v1/vhc/runs/[^/]+/roster$"))
        .respond_with(PublishRoster(registry))
        .mount(&server)
        .await;
    let egress = EgressClient::new(EgressConfig::default()).expect("egress");
    let base = format!("{}/api/v1/vhc", server.uri());
    let client = RegistryClient::new(egress, base).with_bearer("vhc-token");
    (server, client)
}

// -- the suites -------------------------------------------------------------------------------------

/// The publish → fetch → verify lifecycle: two nodes publish, a reader fetches the snapshot,
/// authorizes every entry against the genesis-trusted bases, and reduces by freshness precedence.
#[tokio::test]
async fn publish_fetch_verify_lifecycle() {
    let registry = Arc::new(FakeRosterRegistry::new());
    let (_server, client) = roster_server(registry).await;
    let run = RunId::new(RUN_LABEL);

    let base_a = key(1);
    let base_b = key(9);
    let a = record(&base_a, 2, 0x55, 1, 1_000);
    let b = record(&base_b, 4, 0x66, 1, 2_000);

    assert_eq!(
        client.publish_roster(&run, &a).await.expect("publish a"),
        RosterPublishOutcome::Accepted
    );
    assert_eq!(
        client.publish_roster(&run, &b).await.expect("publish b"),
        RosterPublishOutcome::Accepted
    );

    let fetched = client.fetch_roster(&run).await.expect("fetch");
    assert_eq!(fetched.len(), 2);

    // Peer-side verification: both entries authorize against the run's trusted bases...
    let trusted = [peer_id(&base_a), peer_id(&base_b)];
    let verified: Vec<RosterRecord> = fetched
        .into_iter()
        .filter(|r| r.authorize(&trusted).is_ok())
        .collect();
    assert_eq!(verified.len(), 2);
    // ...and freshness reduction keeps one record per (role, base) node.
    let reduced = freshest_per_node(verified);
    assert_eq!(reduced.len(), 2);
}

/// A re-addressed republish (same incarnation, later issue) supersedes; a stale republish comes
/// back as a typed 409 refusal carrying the stored record (the publisher's re-read).
#[tokio::test]
async fn republish_supersedes_and_stale_is_refused_with_the_stored_record() {
    let registry = Arc::new(FakeRosterRegistry::new());
    let (_server, client) = roster_server(registry).await;
    let run = RunId::new(RUN_LABEL);
    let base = key(1);

    let first = record(&base, 2, 0x55, 1, 1_000);
    assert_eq!(
        client.publish_roster(&run, &first).await.expect("publish"),
        RosterPublishOutcome::Accepted
    );

    let readdressed = record(&base, 2, 0x55, 1, 2_000);
    assert_eq!(
        client
            .publish_roster(&run, &readdressed)
            .await
            .expect("republish"),
        RosterPublishOutcome::Accepted
    );

    let stale = record(&base, 2, 0x55, 1, 1_500);
    match client.publish_roster(&run, &stale).await.expect("stale") {
        RosterPublishOutcome::Refused { decision, stored } => {
            assert_eq!(
                decision,
                RosterDecision::RejectedStale {
                    stored_incarnation: 1,
                    stored_issued_at_ms: 2_000
                }
            );
            assert_eq!(
                stored.expect("stored record travels with the refusal").body,
                readdressed.body
            );
        }
        other => panic!("expected a stale refusal, got {other:?}"),
    }

    // A rejoin (higher incarnation) supersedes regardless of wall clock.
    let rejoined = record(&base, 2, 0x55, 2, 500);
    assert_eq!(
        client
            .publish_roster(&run, &rejoined)
            .await
            .expect("rejoin"),
        RosterPublishOutcome::Accepted
    );
    let fetched = client.fetch_roster(&run).await.expect("fetch");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].body.incarnation, 2);
}

/// The registry-posture split: the registry ACCEPTS a structurally-valid record whose authority
/// is garbage (a certificate chained to a base no genesis names) — untrusted storage never
/// judges authority — while a PEER verifying the same stored object refuses it typed.
#[tokio::test]
async fn registry_stores_what_peers_refuse() {
    let registry = Arc::new(FakeRosterRegistry::new());
    let (_server, client) = roster_server(registry).await;
    let run = RunId::new(RUN_LABEL);

    let rogue_base = key(66);
    let rogue = record(&rogue_base, 7, 0x77, 1, 1_000);
    assert_eq!(
        client.publish_roster(&run, &rogue).await.expect("publish"),
        RosterPublishOutcome::Accepted,
        "the registry stores; it never judges"
    );

    let fetched = client.fetch_roster(&run).await.expect("fetch");
    assert_eq!(fetched.len(), 1);
    // The peer's judgment: the base is not genesis-trusted → typed refusal, address never dialed.
    let trusted = [peer_id(&key(1))];
    assert_eq!(
        fetched[0].authorize(&trusted),
        Err(RosterRecordError::UntrustedBase)
    );
}

/// An unknown run's roster reads as empty (the 404 → empty mapping), and a non-CBOR publish body
/// is a shape error, not a stored object.
#[tokio::test]
async fn empty_roster_and_shape_errors() {
    let registry = Arc::new(FakeRosterRegistry::new());
    let (server, client) = roster_server(registry).await;

    let fetched = client
        .fetch_roster(&RunId::new("never-published"))
        .await
        .expect("fetch empty");
    assert!(fetched.is_empty());

    // A garbage body → 400 from the shape gate → a typed client error (never a decode panic).
    let egress = EgressClient::new(EgressConfig::default()).expect("egress");
    let raw = daemon_egress::EgressRequest::put(
        format!("{}/api/v1/vhc/runs/{RUN_LABEL}/roster", server.uri()),
        b"not-cbor".to_vec(),
    );
    let resp = egress
        .execute(raw, daemon_egress::Redirects::None)
        .await
        .expect("raw put");
    assert_eq!(resp.status().as_u16(), 400);
}
