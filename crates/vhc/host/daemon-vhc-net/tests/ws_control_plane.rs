// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![cfg(feature = "ws")]
// Integration test crate: reading/writing the committed golden fixture uses std::fs directly.
#![allow(clippy::disallowed_methods)]

//! WS coordinator client suite (A1): the canonical-CBOR framing golden, dissemination relay over the
//! mock `RunCoordinatorDO`, coordinator broadcast delivery, reconnect + resubscribe, and auth-header
//! plumbing. The framing golden pins the byte-exact `SignedMessage` frame the DO consumes
//! (`apps/vhc` `codec.ts` decodes the same bytes) so a canonicalization regression fails loud.

mod common;

use std::time::Duration;

use common::ws_harness::{fast_reconnect, no_reconnect, MockWsCoordinator};
use common::{recv_timeout, signed_heartbeat_bytes, signing_key, DELIVER, GRACE};
use daemon_vhc_net::ws_client::HEARTBEAT_PREFIX;
use daemon_vhc_net::{ControlPlane, WsAuth};
use daemon_vhc_proto::{from_canonical_slice, to_canonical_vec, Hash, VHC_PROTO_VERSION};
use daemon_vhc_sdk_consensus::messages::{Commitment, Locator};
use daemon_vhc_sdk_consensus::{SignedMessage, VhcMessage};

const FRAME_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ws-frame-commitment.cbor"
);

/// A deterministic signed `Commitment` frame — the golden the DO framing must reproduce byte-for-byte.
fn golden_commitment_frame() -> Vec<u8> {
    let key = signing_key(7);
    let payload = VhcMessage::Commitment(Commitment {
        round: 3,
        payload: Hash::new([0xab; 32]),
        size: 4096,
        locators: vec![Locator::StoreKey(
            "runs/run-golden/rounds/3/aabb.upd".to_string(),
        )],
    });
    let signed = SignedMessage::sign(&key, VHC_PROTO_VERSION, payload).expect("sign");
    to_canonical_vec(&signed).expect("encode")
}

/// Regeneration helper (vendored/generated golden, per the A1 brief): rewrite the committed frame
/// fixture. Run explicitly after an intentional proto change:
/// `cargo test -p daemon-vhc-net --features ws regenerate_ws_frame_fixture -- --ignored`.
#[test]
#[ignore = "regeneration helper; writes the committed golden fixture"]
fn regenerate_ws_frame_fixture() {
    std::fs::write(FRAME_FIXTURE, golden_commitment_frame()).expect("write fixture");
}

/// The framing golden: the canonical-CBOR `SignedMessage` frame is byte-stable, matches the committed
/// fixture (the bytes the cloud DO consumes), round-trips, and its signature verifies.
#[test]
fn ws_frame_golden_is_byte_stable_and_verifies() {
    let bytes = golden_commitment_frame();
    let fixture = std::fs::read(FRAME_FIXTURE).expect(
        "committed golden fixture present (regenerate with the ignored `regenerate_ws_frame_fixture`)",
    );
    assert_eq!(
        bytes, fixture,
        "canonical-CBOR SignedMessage frame drifted from the committed golden (cross-repo DO contract)"
    );
    let decoded: SignedMessage = from_canonical_slice(&bytes).expect("decode");
    decoded.verify().expect("golden frame signature verifies");
    assert_eq!(
        to_canonical_vec(&decoded).expect("re-encode"),
        bytes,
        "canonical re-encode must be byte-identical"
    );
}

/// The DO disseminates an inbound frame to the OTHER peers (never echoes the sender): peer A's
/// publish reaches peer B over the relay, and A's own subscriber gets it once (self-delivery).
#[tokio::test(flavor = "multi_thread")]
async fn ws_publish_relays_to_other_peers() {
    let coord = MockWsCoordinator::start().await;
    let a = coord.client("run-1", WsAuth::None, no_reconnect()).await;
    let b = coord.client("run-1", WsAuth::None, no_reconnect()).await;
    coord.wait_peers(2).await;
    let mut sub_a = a.subscribe();
    let mut sub_b = b.subscribe();

    let frame = signed_heartbeat_bytes(&signing_key(1), 1);
    a.publish(&frame).await.expect("publish");

    assert_eq!(
        recv_timeout(&mut sub_a, DELIVER).await.as_deref(),
        Some(frame.as_slice()),
        "publisher self-delivers its own frame once"
    );
    assert_eq!(
        recv_timeout(&mut sub_b, DELIVER).await.as_deref(),
        Some(frame.as_slice()),
        "the other peer receives the relayed frame"
    );
    assert!(recv_timeout(&mut sub_a, GRACE).await.is_none());
    assert!(recv_timeout(&mut sub_b, GRACE).await.is_none());
}

/// A coordinator emission (RoundOpen / StorageReceipt / RoundRecord) reaches every connected peer.
#[tokio::test(flavor = "multi_thread")]
async fn ws_coordinator_broadcast_reaches_all_peers() {
    let coord = MockWsCoordinator::start().await;
    let a = coord.client("run-1", WsAuth::None, no_reconnect()).await;
    let b = coord.client("run-1", WsAuth::None, no_reconnect()).await;
    coord.wait_peers(2).await;
    let mut sub_a = a.subscribe();
    let mut sub_b = b.subscribe();

    let frame = signed_heartbeat_bytes(&signing_key(2), 5);
    coord.broadcast(frame.clone());

    assert_eq!(
        recv_timeout(&mut sub_a, DELIVER).await.as_deref(),
        Some(frame.as_slice())
    );
    assert_eq!(
        recv_timeout(&mut sub_b, DELIVER).await.as_deref(),
        Some(frame.as_slice())
    );
}

/// A severed socket reconnects with backoff and re-sends the registered resubscribe frame (the
/// peer's Join), and post-reconnect coordinator emissions are delivered.
#[tokio::test(flavor = "multi_thread")]
async fn ws_reconnects_and_resubscribes() {
    let coord = MockWsCoordinator::start().await;
    let plane = coord.client("run-1", WsAuth::None, fast_reconnect()).await;
    coord.wait_peers(1).await;
    assert_eq!(plane.connect_count(), 1);

    // Register the peer's Join as a resubscribe frame (sent now + re-sent on every reconnect).
    let join = signed_heartbeat_bytes(&signing_key(9), 0);
    plane.add_resubscribe_frame(join);
    wait_until(Duration::from_secs(2), || coord.received() >= 1).await;

    let mut sub = plane.subscribe();

    // Force a disconnect; the plane must reconnect (connect_count grows) and re-send the Join.
    coord.sever();
    wait_until(Duration::from_secs(5), || plane.connect_count() >= 2).await;
    coord.wait_peers(1).await;
    wait_until(Duration::from_secs(2), || coord.received() >= 2).await;
    assert!(
        plane.is_connected(),
        "the plane is up again after reconnect"
    );

    // A post-reconnect coordinator emission still reaches the subscriber.
    let frame = signed_heartbeat_bytes(&signing_key(3), 7);
    coord.broadcast(frame.clone());
    assert_eq!(
        recv_timeout(&mut sub, DELIVER).await.as_deref(),
        Some(frame.as_slice())
    );
}

/// A server heartbeat frame (the DO's `DVHC-HB1` liveness signal): prefix + seq + unix-ms.
fn heartbeat_frame(seq: u64) -> Vec<u8> {
    let mut f = Vec::with_capacity(HEARTBEAT_PREFIX.len() + 16);
    f.extend_from_slice(HEARTBEAT_PREFIX);
    f.extend_from_slice(&seq.to_be_bytes());
    f.extend_from_slice(&0u64.to_be_bytes());
    f
}

/// A server heartbeat is consumed by the plane (liveness bookkeeping + counters) and never
/// reaches a subscriber; a regular frame alone never arms the deafness deadline.
#[tokio::test(flavor = "multi_thread")]
async fn ws_heartbeat_is_consumed_never_fanned_out() {
    let coord = MockWsCoordinator::start().await;
    let mut rc = fast_reconnect();
    rc.binary_silence_deadline = Duration::from_millis(400);
    let plane = coord.client("run-hb", WsAuth::None, rc).await;
    coord.wait_peers(1).await;
    let mut sub = plane.subscribe();

    // A regular frame delivers to the subscriber but must NOT arm the deafness deadline: well
    // past the 400 ms deadline the plane is still on its first connection (quiet phases on a
    // pre-heartbeat server never trigger reconnect cycling).
    let frame = signed_heartbeat_bytes(&signing_key(4), 1);
    coord.broadcast(frame.clone());
    assert_eq!(
        recv_timeout(&mut sub, DELIVER).await.as_deref(),
        Some(frame.as_slice())
    );
    tokio::time::sleep(Duration::from_millis(900)).await;
    assert_eq!(
        plane.connect_count(),
        1,
        "a regular frame must not arm the deafness deadline"
    );

    // A heartbeat is counted and consumed — the subscriber never sees it.
    coord.broadcast(heartbeat_frame(1));
    wait_until(Duration::from_secs(2), || {
        plane.stats().heartbeats_received >= 1
    })
    .await;
    let stats = plane.stats();
    assert!(stats.binary_received >= 2, "both binaries counted");
    assert_eq!(stats.frames_delivered, 1, "only the real frame fanned out");
    assert!(stats.last_binary_unix_ms > 0, "the inbound watermark moved");
    assert!(recv_timeout(&mut sub, GRACE).await.is_none());
}

/// The deaf-fan-out regression: once a server heartbeat armed the expected-progress deadline,
/// Binary silence past it forces a reconnect + resubscribe even though the socket itself stays
/// healthy (the relay keeps answering Pings throughout — Pong is never delivery evidence).
#[tokio::test(flavor = "multi_thread")]
async fn ws_binary_silence_after_heartbeat_forces_deaf_reconnect() {
    let coord = MockWsCoordinator::start().await;
    let mut rc = fast_reconnect();
    rc.binary_silence_deadline = Duration::from_millis(400);
    let plane = coord.client("run-deaf", WsAuth::None, rc).await;
    coord.wait_peers(1).await;

    // Arm the deadline with one heartbeat, then go Binary-silent (the deaf fan-out).
    coord.broadcast(heartbeat_frame(1));
    wait_until(Duration::from_secs(2), || {
        plane.stats().heartbeats_received >= 1
    })
    .await;

    // The plane must declare deafness and cycle the connection.
    wait_until(Duration::from_secs(5), || plane.connect_count() >= 2).await;
    assert!(
        plane.stats().deaf_reconnects >= 1,
        "counted as a deaf cycle"
    );

    // The fresh connection is unarmed (no heartbeat arrived on it), so the plane does not keep
    // cycling — and a post-reconnect broadcast still reaches the subscriber.
    let stable = plane.connect_count();
    tokio::time::sleep(Duration::from_millis(900)).await;
    assert_eq!(
        plane.connect_count(),
        stable,
        "no reconnect cycling while the new connection is unarmed"
    );
    let mut sub = plane.subscribe();
    let frame = signed_heartbeat_bytes(&signing_key(5), 2);
    coord.broadcast(frame.clone());
    assert_eq!(
        recv_timeout(&mut sub, DELIVER).await.as_deref(),
        Some(frame.as_slice())
    );
}

/// The Bearer credential is stamped on the upgrade request (the gateway path).
#[tokio::test(flavor = "multi_thread")]
async fn ws_stamps_bearer_auth_header() {
    let coord = MockWsCoordinator::start().await;
    let _plane = coord
        .client(
            "run-1",
            WsAuth::Bearer("secret-token".into()),
            no_reconnect(),
        )
        .await;
    coord.wait_peers(1).await;
    let headers = coord.last_headers();
    assert!(
        headers
            .iter()
            .any(|(k, v)| k == "authorization" && v == "Bearer secret-token"),
        "expected the Bearer auth header, got {headers:?}"
    );
}

/// The internal identity headers are stamped on the upgrade request (the direct-to-worker dev path).
#[tokio::test(flavor = "multi_thread")]
async fn ws_stamps_internal_identity_headers() {
    let coord = MockWsCoordinator::start().await;
    let _plane = coord
        .client(
            "run-1",
            WsAuth::Internal {
                org_id: "org-7".into(),
                actor: "key:k9".into(),
            },
            no_reconnect(),
        )
        .await;
    coord.wait_peers(1).await;
    let headers = coord.last_headers();
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-daemon-org-id" && v == "org-7"));
    assert!(headers
        .iter()
        .any(|(k, v)| k == "x-daemon-actor" && v == "key:k9"));
}

/// Poll `cond` until true or the deadline elapses (panics on timeout).
async fn wait_until<F: Fn() -> bool>(within: Duration, cond: F) {
    let deadline = std::time::Instant::now() + within;
    while !cond() {
        if std::time::Instant::now() > deadline {
            panic!("condition not met within {within:?}");
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}
