// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Run discovery + envelope fetch (A1): [`RegistryClient`] against a mock `apps/swarm` registry —
//! `GET /runs` (list), `GET /runs/:id` (detail + 404), and the presign→object→blake3-verify envelope
//! fetch. A registry that serves the wrong envelope bytes is rejected before `AssessRun` ever sees
//! them (the §12 tamper path).

use daemon_egress::{EgressClient, EgressConfig};
use daemon_swarm_net::RunId;
use daemon_swarm_net::{RegistryClient, RunDescriptor, SwarmNetError};
use daemon_swarm_proto::blake3_hash;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const ENVELOPE_BYTES: &[u8] = b"frozen-envelope-canonical-cbor-bytes";

fn descriptor(run_id: &str, envelope_hash: &str) -> serde_json::Value {
    json!({
        "run_id": run_id,
        "schema": 1,
        "proto_version": 3,
        "envelope_hash": envelope_hash,
        "author_pubkey": "00",
        "artifacts": [{ "path": "envelope.cbor", "blake3": envelope_hash, "size": ENVELOPE_BYTES.len() }],
        "update_max_bytes": 1_048_576,
        "min_peers": 1,
        "max_peers": 8,
        "rounds": 10,
        "created_at": 42,
        "envelope_key": format!("runs/{run_id}/envelope.cbor")
    })
}

async fn registry_with(envelope_hash: &str) -> (MockServer, RegistryClient) {
    let server = MockServer::start().await;
    let base = format!("{}/api/v1/swarm", server.uri());
    let d = descriptor("run-1", envelope_hash);

    Mock::given(method("GET"))
        .and(path("/api/v1/swarm/runs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [d.clone()] })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/v1/swarm/runs/run-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": d })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/v1/swarm/runs/missing"))
        .respond_with(
            ResponseTemplate::new(404)
                .set_body_json(json!({ "error": { "message": "run not found" } })),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/v1/swarm/runs/run-1/presign"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "url": format!("{}/obj/envelope", server.uri()),
            "expires_at": now_plus_900(),
            "headers": {}
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/obj/envelope"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(ENVELOPE_BYTES.to_vec()))
        .mount(&server)
        .await;

    let egress = EgressClient::new(EgressConfig::default()).expect("egress");
    let client = RegistryClient::new(egress, base).with_bearer("swarm-token");
    (server, client)
}

fn now_plus_900() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() + 900)
        .unwrap_or(900)
}

#[tokio::test]
async fn lists_and_fetches_run_descriptors() {
    let hash = blake3_hash(ENVELOPE_BYTES).to_hex();
    let (_server, client) = registry_with(&hash).await;

    let runs = client.list_runs().await.expect("list");
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].run_id, "run-1");

    let one: Option<RunDescriptor> = client.get_run("run-1").await.expect("get");
    assert_eq!(one.expect("some").proto_version, 3);

    assert!(
        client
            .get_run("missing")
            .await
            .expect("get missing")
            .is_none(),
        "a 404 is Ok(None), not an error"
    );
}

#[tokio::test]
async fn fetches_and_verifies_envelope() {
    let hash = blake3_hash(ENVELOPE_BYTES).to_hex();
    let (_server, client) = registry_with(&hash).await;
    let d = client.get_run("run-1").await.expect("get").expect("some");

    let bytes = client
        .fetch_envelope(&RunId::new("run-1"), &d)
        .await
        .expect("fetch envelope");
    assert_eq!(
        bytes, ENVELOPE_BYTES,
        "the frozen envelope bytes round-trip"
    );
}

#[tokio::test]
async fn rejects_envelope_with_wrong_hash() {
    // The descriptor claims a hash that does NOT match the served bytes — the tamper path (§12).
    let wrong = blake3_hash(b"a different envelope").to_hex();
    let (_server, client) = registry_with(&wrong).await;
    let d = client.get_run("run-1").await.expect("get").expect("some");

    let err = client
        .fetch_envelope(&RunId::new("run-1"), &d)
        .await
        .expect_err("must reject a hash mismatch");
    assert!(
        matches!(err, SwarmNetError::HashMismatch { .. }),
        "expected HashMismatch, got {err:?}"
    );
}
