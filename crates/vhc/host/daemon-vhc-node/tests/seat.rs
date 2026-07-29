// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The coordinator seat manager end-to-end over the normative CAS fold (the `FakeSeatRegistry`,
//! the same fold a conforming remote registry runs): claim → peer-authorize → heartbeat → a live
//! contender loses TYPED → release preserves the tombstone floor → a superseded lease is refused
//! by the VERIFIED leadership-term floor (fencing-is-safe-not-seamless; [SEAT-1] v2). The node
//! authors + signs; every authority judgment is peer-side (`authorize_incumbent`), the registry
//! storing structurally. The two identities stay separate throughout: execution incarnations are
//! claimant-local ordinals, leadership terms order the one cross-base seat.

use daemon_vhc_net::FakeSeatRegistry;
use daemon_vhc_node::seat::{
    author_claim, author_release, author_renew, authorize_incumbent, derive_bid, CoordinatorSeat,
};
use daemon_vhc_proto::{
    peer_id, ControlEndpoint, RevocationLedger, SeatDecision, SeatLeaseError, SeatState,
    SeatTermLedger, DEFAULT_SEAT_SKEW_MS,
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

    // -- claim: unclaimed slot → term bid 0 (no floor anywhere), author + CAS-accept ----------
    let state = registry.read(RUN, ROLE);
    assert_eq!(
        derive_bid(&state, None, now, DEFAULT_SEAT_SKEW_MS),
        Some(0),
        "a virgin slot with no persisted floor bids term 0"
    );
    let ep = ws_endpoint("wss://coord.example/runs/seat-run/ws");
    // Execution incarnation 4: a claimant-local ordinal (this node's counter is at 4 from
    // unrelated churn) — deliberately NOT the term, to pin the separation.
    let lease = author_claim(&keystore, &seat(ep.clone()), 4, 0, now).expect("author claim");
    let resp = registry.claim(RUN, &lease, now);
    assert_eq!(
        resp.decision,
        SeatDecision::Accepted,
        "first claim wins CAS"
    );

    // -- peer-side authorize: signature/chain/expiry + per-base execution floor + term floor --
    let revocations = RevocationLedger::new();
    let mut terms = SeatTermLedger::new();
    let stored = match registry.read(RUN, ROLE) {
        SeatState::Leased(l) => *l,
        other => panic!("expected a lease, got {other:?}"),
    };
    let authorized = authorize_incumbent(
        &stored,
        &trusted,
        &revocations,
        &terms,
        now,
        DEFAULT_SEAT_SKEW_MS,
    )
    .expect("the stored lease authorizes to the genesis-trusted base");
    assert_eq!(authorized.endpoint, ep);
    assert_eq!(authorized.incarnation, 4);
    assert_eq!(authorized.leadership_term, 0);
    // Only NOW — after the full acceptance — may the grant feed the term floor.
    terms.observe_verified_grant(&stored);

    // -- heartbeat: a re-signed body under the same identity + term renews (never a takeover) --
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
        derive_bid(&live_state, None, now + 6_000, DEFAULT_SEAT_SKEW_MS),
        None,
        "a live incumbent means stand by"
    );
    // Even if a contender force-bids a higher term, the CAS refuses it while the incumbent holds.
    let contender = author_claim(&other, &seat(ep.clone()), 1, 1, now + 6_000).expect("author");
    let lost = registry.claim(RUN, &contender, now + 6_000);
    assert_eq!(
        lost.decision,
        SeatDecision::RejectedHeld,
        "a contender must not displace a live lease"
    );

    // -- release: the seat unclaims; the tombstone floor persists ------------------------------
    let release = author_release(&keystore, &seat(ep.clone()), 4, 0).expect("release");
    assert_eq!(
        registry.release(RUN, ROLE, &release, now + 7_000).decision,
        SeatDecision::Accepted,
        "the holder releases"
    );
    match registry.read(RUN, ROLE) {
        SeatState::Unclaimed {
            last_leadership_term,
        } => {
            assert_eq!(last_leadership_term, Some(0), "the floor survives release");
        }
        other => panic!("expected the tombstoned slot, got {other:?}"),
    }

    // -- takeover after release: the next bid is above the tombstone floor ---------------------
    let post = registry.read(RUN, ROLE);
    assert_eq!(
        derive_bid(&post, None, now + 8_000, DEFAULT_SEAT_SKEW_MS),
        Some(1)
    );
    // The successor's own execution incarnation is 1 (ITS ladder — nothing to do with the old
    // holder's 4: incarnations are per-base, never a cross-base order).
    let successor = author_claim(&other, &seat(ep.clone()), 1, 1, now + 8_000).expect("author");
    assert_eq!(
        registry.claim(RUN, &successor, now + 8_000).decision,
        SeatDecision::Accepted,
        "a takeover above the floor wins after release"
    );

    // -- fencing is safety: the OLD claimant's term-0 lease is dead once a VERIFIED term-1 grant
    // is observed, regardless of what any registry stores — and it is the TERM ledger that kills
    // it, not a certificate floor (a generic coordinator-role certificate that never won the
    // seat must not fence anyone; that is the retired `role_floor` defect).
    let successor_grant = authorize_incumbent(
        &successor,
        &[peer_id(&other.base_identity().expect("base"))],
        &revocations,
        &terms,
        now + 8_000,
        DEFAULT_SEAT_SKEW_MS,
    )
    .expect("the successor's grant authorizes to its own trusted base");
    assert_eq!(successor_grant.leadership_term, 1);
    terms.observe_verified_grant(&successor);
    let stale = authorize_incumbent(
        &stored,
        &trusted,
        &revocations,
        &terms,
        now + 8_000,
        DEFAULT_SEAT_SKEW_MS,
    );
    assert!(
        matches!(
            stale,
            Err(SeatLeaseError::TermSuperseded { got: 0, floor: 1 })
        ),
        "the superseded term-0 lease is refused by the verified term floor: {stale:?}"
    );

    // -- an untrusted base never authorizes (authority is peer-side, never the registry) -------
    let stranger_dir = tempfile::tempdir().expect("tempdir");
    let stranger = VhcKeystore::open(stranger_dir.path()).expect("keystore");
    let stranger_lease =
        author_claim(&stranger, &seat(ep), 5, 9, now + 9_000).expect("author stranger lease");
    let refused = authorize_incumbent(
        &stranger_lease,
        &trusted, // does NOT include the stranger's base
        &revocations,
        &SeatTermLedger::new(),
        now + 9_000,
        DEFAULT_SEAT_SKEW_MS,
    );
    assert!(
        refused.is_err(),
        "a lease from an untrusted base is refused peer-side"
    );
}
