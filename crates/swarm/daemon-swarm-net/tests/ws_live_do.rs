// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![cfg(feature = "ws")]
#![allow(clippy::disallowed_methods)]

//! Merge-1 cross-lane LIVE check (A1 node WS client ↔ C1 wasm-tick `RunCoordinatorDO`).
//!
//! This is the first real node↔cloud contact: A1's `WsControlPlane` dials the C1 coordinator DO
//! running under `wrangler dev` (not the in-process mock). It asserts the committed canonical-CBOR
//! `SignedMessage` framing golden is accepted + relayed **byte-for-byte** by the real DO, that the
//! Loopback/Iroh delivery contract holds against the DO (publisher self-delivers, the DO relays to
//! the *other* peer, no echo/no duplicate), and that a registered resubscribe frame reaches the peer.
//!
//! It SKIPS cleanly (like C1's `r2-smoke`) unless `SWARM_LIVE_WS_URL` is set, so it never runs in the
//! offline workspace gate. Drive it after seeding a run against wrangler-dev (port 8795):
//!   SWARM_LIVE_WS_URL=http://127.0.0.1:8795/api/v1/swarm SWARM_LIVE_RUN_ID=run-live \
//!     cargo test -p daemon-swarm-net --features ws --test ws_live_do -- --nocapture

mod common;

use std::time::{Duration, Instant};

use common::{recv_timeout, signed_heartbeat_bytes, signing_key, DELIVER, GRACE};
use daemon_swarm_net::{ControlPlane, ReconnectConfig, WsAuth, WsConfig, WsControlPlane};
use daemon_swarm_proto::messages::{Commitment, Heartbeat, Join, Locator, ThroughputClass};
use daemon_swarm_proto::{
    from_canonical_slice, to_canonical_vec, CapabilitySet, Hash, IrohId, SignedMessage,
    SwarmMessage, SWARM_PROTO_VERSION,
};

/// The committed A1 framing golden (matches `tests/fixtures/ws-frame-commitment.cbor`): a signed
/// `Commitment` frame. `size: 4096` stays under the run's `update_max_bytes`, so the DO's §7.3
/// receive-side cap does not drop it before relay.
fn golden_commitment_frame() -> Vec<u8> {
    let key = signing_key(7);
    let payload = SwarmMessage::Commitment(Commitment {
        round: 3,
        payload: Hash::new([0xab; 32]),
        size: 4096,
        locators: vec![Locator::StoreKey(
            "runs/run-golden/rounds/3/aabb.upd".to_string(),
        )],
    });
    let signed = SignedMessage::sign(&key, SWARM_PROTO_VERSION, payload).expect("sign");
    to_canonical_vec(&signed).expect("encode")
}

fn live_auth() -> WsAuth {
    WsAuth::Internal {
        org_id: std::env::var("SWARM_LIVE_ORG").unwrap_or_else(|_| "org_live".into()),
        actor: std::env::var("SWARM_LIVE_ACTOR").unwrap_or_else(|_| "key:live".into()),
    }
}

async fn connect_live(base_url: &str, run_id: &str) -> WsControlPlane {
    WsControlPlane::connect(WsConfig {
        base_url: base_url.to_string(),
        run_id: run_id.to_string(),
        auth: live_auth(),
        reconnect: ReconnectConfig::default(),
    })
    .await
    .expect("connect WsControlPlane to the live coordinator DO")
}

/// The headline live check: golden framing byte-for-byte over the real DO, self-delivery + relay, and
/// resubscribe-frame delivery.
#[tokio::test(flavor = "multi_thread")]
async fn live_ws_framing_relay_and_resubscribe_against_wrangler_dev() {
    let Ok(base_url) = std::env::var("SWARM_LIVE_WS_URL") else {
        eprintln!(
            "SKIP live_ws_do: set SWARM_LIVE_WS_URL (e.g. http://127.0.0.1:8795/api/v1/swarm)"
        );
        return;
    };
    let run_id = std::env::var("SWARM_LIVE_RUN_ID").unwrap_or_else(|_| "run-live".into());

    let a = connect_live(&base_url, &run_id).await;
    let b = connect_live(&base_url, &run_id).await;
    assert_eq!(a.connect_count(), 1, "peer A connected once");
    assert_eq!(b.connect_count(), 1, "peer B connected once");
    assert!(
        a.is_connected() && b.is_connected(),
        "both peers connected to the DO"
    );

    let mut sub_a = a.subscribe();
    let mut sub_b = b.subscribe();

    // Framing byte-check: A publishes the committed golden; the real DO relays the exact bytes to B
    // (the sender is excluded) and A self-delivers its own publish once.
    let golden = golden_commitment_frame();
    a.publish(&golden).await.expect("publish golden frame");

    assert_eq!(
        recv_timeout(&mut sub_a, DELIVER).await.as_deref(),
        Some(golden.as_slice()),
        "publisher self-delivers the golden frame once"
    );
    assert_eq!(
        recv_timeout(&mut sub_b, DELIVER).await.as_deref(),
        Some(golden.as_slice()),
        "the DO relays the golden frame to the other peer BYTE-FOR-BYTE"
    );
    assert!(
        recv_timeout(&mut sub_a, GRACE).await.is_none(),
        "no duplicate to the publisher"
    );
    assert!(
        recv_timeout(&mut sub_b, GRACE).await.is_none(),
        "no duplicate to the other peer"
    );

    // Resubscribe delivery: register B's Join (re-sent on every (re)connect); A receives it via relay.
    let join = signed_heartbeat_bytes(&signing_key(11), 0);
    b.add_resubscribe_frame(join.clone());
    assert_eq!(
        recv_timeout(&mut sub_a, DELIVER).await.as_deref(),
        Some(join.as_slice()),
        "the registered resubscribe frame is delivered over the live DO"
    );

    a.shutdown().await;
    b.shutdown().await;
}

/// RoundEngine-adjacent progression smoke: a run-bound `Join` + a readiness `Heartbeat` published
/// over the WS plane drive the real wasm-tick DO through admission → warmup → a signed `RoundOpen`
/// coordinator emission that reaches the joining peer. Proves the DO's decision core (the compiled
/// `daemon_swarm_coordinator::tick`) advances the run in response to A1-client frames — the
/// coordinator half of the RoundEngine-over-WsControlPlane loop (the full stub-backend round loop is
/// the Merge-2 node↔cloud↔worker item, gated by A3 worker attach).
#[tokio::test(flavor = "multi_thread")]
async fn live_ws_join_ready_progresses_to_round_open() {
    let Ok(base_url) = std::env::var("SWARM_LIVE_WS_URL") else {
        eprintln!("SKIP live_ws_do progression: set SWARM_LIVE_WS_URL");
        return;
    };
    // A dedicated single-peer run (min_peers=1) seeded by scripts/seed_run.mjs.
    let run_id = std::env::var("SWARM_LIVE_PROG_RUN").unwrap_or_else(|_| "run-prog".into());

    let peer = connect_live(&base_url, &run_id).await;
    let mut sub = peer.subscribe();
    let key = signing_key(21);

    let join = SignedMessage::sign(
        &key,
        SWARM_PROTO_VERSION,
        SwarmMessage::Join(Join {
            run_id: run_id.clone(),
            iroh_id: IrohId([0x21; 32]),
            class: ThroughputClass::C2,
            capabilities: CapabilitySet::new(),
            envelope_hash: None,
        }),
    )
    .expect("sign join");
    peer.publish(&to_canonical_vec(&join).expect("encode join"))
        .await
        .expect("publish join");

    // Readiness heartbeat: lets the coordinator exit Warmup early (all admitted members ready, §6.2).
    let ready = SignedMessage::sign(
        &key,
        SWARM_PROTO_VERSION,
        SwarmMessage::Heartbeat(Heartbeat {
            round: 0,
            ready: Some(true),
        }),
    )
    .expect("sign heartbeat");
    peer.publish(&to_canonical_vec(&ready).expect("encode heartbeat"))
        .await
        .expect("publish ready heartbeat");

    // The DO broadcasts coordinator emissions to ALL peers (incl. the sender): await a RoundOpen.
    let mut saw_round_open = false;
    let deadline = Instant::now() + Duration::from_secs(45);
    while Instant::now() < deadline {
        match recv_timeout(&mut sub, Duration::from_secs(5)).await {
            Some(frame) => {
                if let Ok(sm) = from_canonical_slice::<SignedMessage>(&frame) {
                    if matches!(sm.payload, SwarmMessage::RoundOpen(_)) {
                        saw_round_open = true;
                        break;
                    }
                }
            }
            None => {
                // keep waiting up to the deadline (warmup/join-window timers are alarm-driven)
            }
        }
    }
    assert!(
        saw_round_open,
        "the wasm-tick DO advanced to a signed RoundOpen after a run-bound Join + ready heartbeat"
    );
    peer.shutdown().await;
}
