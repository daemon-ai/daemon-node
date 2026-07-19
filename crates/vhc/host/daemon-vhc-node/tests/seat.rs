// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The coordinator seat manager end-to-end over the normative CAS fold (the `FakeSeatRegistry`,
//! the same fold a conforming remote registry runs): claim → peer-authorize → heartbeat → a live
//! contender loses TYPED → release preserves the tombstone floor → a superseded lease is refused
//! by the supersession floor (fencing-is-safe-not-seamless). The node authors + signs; every
//! authority judgment is peer-side (`authorize_incumbent`), the registry storing structurally.

use daemon_vhc_net::FakeSeatRegistry;
use daemon_vhc_node::seat::{
    author_claim, author_release, author_renew, authorize_incumbent, derive_bid, CoordinatorSeat,
};
use daemon_vhc_proto::{
    peer_id, ControlEndpoint, RevocationLedger, SeatDecision, SeatState, DEFAULT_SEAT_SKEW_MS,
};
use daemon_vhc_session::keystore::VhcKeystore;

const RUN: &str = "seat-run";
const ROLE: &str = "coordinator";
const GENESIS: [u8; 32] = [0x9E; 32];
const MODULE: [u8; 32] = [0xC0; 32];

fn seat<'a>(endpoint: ControlEndpoint) -> CoordinatorSeat<'a> {
    CoordinatorSeat {
        run_label: RUN,
        genesis_hash: GENESIS,
        role: ROLE,
        epoch: 0,
        module_hash: MODULE,
        endpoint,
    }
}

fn ws_endpoint(url: &str) -> ControlEndpoint {
    ControlEndpoint {
        ws: Some(url.to_string()),
        iroh_ticket: None,
    }
}

#[test]
fn seat_lifecycle_claim_heartbeat_fence_release_and_supersession() {
    let dir = tempfile::tempdir().expect("tempdir");
    // The node's identity store: its base identity is the genesis-trusted issuer here.
    let keystore = VhcKeystore::open(dir.path()).expect("keystore");
    let base = peer_id(&keystore.base_identity().expect("base"));
    let trusted = [base];
    let registry = FakeSeatRegistry::new();
    let now = 1_000_000u64;

    // -- claim: unclaimed slot → bid 0 (floor none), author + CAS-accept ----------------------
    let state = registry.read(RUN, ROLE);
    assert_eq!(
        derive_bid(&state, now, DEFAULT_SEAT_SKEW_MS),
        Some(0),
        "a virgin slot's first bid is 0"
    );
    let ep = ws_endpoint("wss://coord.example/runs/seat-run/ws");
    let lease = author_claim(&keystore, &seat(ep.clone()), 0, now).expect("author claim");
    let resp = registry.claim(RUN, &lease, now);
    assert_eq!(
        resp.decision,
        SeatDecision::Accepted,
        "first claim wins CAS"
    );

    // -- peer-side authorize: the stored lease verifies + is not below the floor ---------------
    let mut floor = RevocationLedger::new();
    let stored = match registry.read(RUN, ROLE) {
        SeatState::Leased(l) => *l,
        other => panic!("expected a lease, got {other:?}"),
    };
    let authorized = authorize_incumbent(&stored, &trusted, &floor, now, DEFAULT_SEAT_SKEW_MS)
        .expect("the stored lease authorizes to the genesis-trusted base");
    assert_eq!(authorized.endpoint, ep);
    assert_eq!(authorized.incarnation, 0);
    // The trainer now treats incarnation 0 as the coordinator slot's floor.
    floor.observe_certificates(std::slice::from_ref(&stored.certificate));

    // -- heartbeat: a re-signed body under the same token renews (never a takeover) ------------
    let renew = author_renew(&keystore, &seat(ep.clone()), &stored, now + 5_000).expect("renew");
    assert_eq!(
        registry.renew(RUN, &renew, now + 5_000).decision,
        SeatDecision::Accepted,
        "the holder's heartbeat renews"
    );

    // -- a live contender loses TYPED (the incumbent is not expired) ---------------------------
    // A fresh identity store stands in for a different node.
    let other_dir = tempfile::tempdir().expect("tempdir");
    let other = VhcKeystore::open(other_dir.path()).expect("keystore");
    // While the incumbent is live, `derive_bid` says stand by (no bid).
    let live_state = registry.read(RUN, ROLE);
    assert_eq!(
        derive_bid(&live_state, now + 6_000, DEFAULT_SEAT_SKEW_MS),
        None,
        "a live incumbent means stand by"
    );
    // Even if a contender force-bids `floor + 1`, the CAS refuses it while the incumbent holds.
    let contender = author_claim(&other, &seat(ep.clone()), 1, now + 6_000).expect("author");
    let lost = registry.claim(RUN, &contender, now + 6_000);
    assert_eq!(
        lost.decision,
        SeatDecision::RejectedHeld,
        "a contender must not displace a live lease"
    );

    // -- release: the seat unclaims; the tombstone floor persists ------------------------------
    let release = author_release(&keystore, &seat(ep.clone()), 0).expect("release");
    assert_eq!(
        registry.release(RUN, ROLE, &release, now + 7_000).decision,
        SeatDecision::Accepted,
        "the holder releases"
    );
    match registry.read(RUN, ROLE) {
        SeatState::Unclaimed { last_fencing_token } => {
            assert_eq!(last_fencing_token, Some(0), "the floor survives release");
        }
        other => panic!("expected the tombstoned slot, got {other:?}"),
    }

    // -- takeover after release: the next bid is floor + 1 = 1 ---------------------------------
    let post = registry.read(RUN, ROLE);
    assert_eq!(
        derive_bid(&post, now + 8_000, DEFAULT_SEAT_SKEW_MS),
        Some(1)
    );
    let successor = author_claim(&other, &seat(ep.clone()), 1, now + 8_000).expect("author");
    assert_eq!(
        registry.claim(RUN, &successor, now + 8_000).decision,
        SeatDecision::Accepted,
        "a takeover at floor + 1 wins after release"
    );

    // -- fencing is safety: the OLD claimant's incarnation-0 lease is dead once incarnation 1 --
    // exists, regardless of what any registry stores (supersession floor).
    let mut floor2 = RevocationLedger::new();
    floor2.observe_certificates(std::slice::from_ref(&successor.certificate)); // floor = 1
    let stale = authorize_incumbent(
        &stored,
        &trusted,
        &floor2,
        now + 8_000,
        DEFAULT_SEAT_SKEW_MS,
    );
    assert!(
        stale.is_err(),
        "the superseded incarnation-0 lease is refused by the supersession floor"
    );

    // -- an untrusted base never authorizes (authority is peer-side, never the registry) -------
    let stranger_dir = tempfile::tempdir().expect("tempdir");
    let stranger = VhcKeystore::open(stranger_dir.path()).expect("keystore");
    let stranger_lease =
        author_claim(&stranger, &seat(ep), 5, now + 9_000).expect("author stranger lease");
    let empty_floor = RevocationLedger::new();
    let refused = authorize_incumbent(
        &stranger_lease,
        &trusted, // does NOT include the stranger's base
        &empty_floor,
        now + 9_000,
        DEFAULT_SEAT_SKEW_MS,
    );
    assert!(
        refused.is_err(),
        "a lease from an untrusted base is refused peer-side"
    );
}
