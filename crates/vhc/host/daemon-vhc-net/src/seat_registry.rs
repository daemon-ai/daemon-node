// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`FakeSeatRegistry`] — the local seat-registry fixture: an in-memory, thread-safe seat-slot
//! store applying the **normative CAS fold** (`daemon_vhc_proto::SeatSlot::fold`) under a single
//! writer, exactly as a conforming remote registry does behind the frozen HTTP surface
//! (`PUT/POST/GET/DELETE {base}/runs/:id/seat/:role[…]`).
//!
//! This is a **test/acceptance fixture, never a production registry**: it exists so the client
//! suites here and the multi-process acceptance suite can exercise claim/renew/read/release,
//! CAS races, fencing, expiry, and the tombstone floor without a network. Serving it over HTTP is
//! the harness's concern (the suites mount it behind an in-process mock server); the semantics
//! live entirely in the shared fold, so local and cloud behavior can only diverge if one of them
//! stops applying the fold — which the shared test vectors
//! (`daemon-vhc-proto/tests/fixtures/seat-cas-vectors.json`) catch.
//!
//! Faithful to the registry posture (untrusted storage): **structural validation only** — the
//! fold checks domains, bounded ordinals, windows, endpoint presence, slot consistency, expiry
//! by the registry clock + skew grace, and the sparse leadership-term CAS ([SEAT-1] v2). It
//! never verifies a signature and never judges authority; peers do that.

use std::collections::BTreeMap;
use std::sync::Mutex;

use daemon_vhc_proto::{
    SeatLease, SeatMutationResponse, SeatRelease, SeatRequest, SeatSlot, SeatState,
    DEFAULT_SEAT_SKEW_MS,
};

/// One seat slot per `(run label, role)` key — the registry's storage granularity (the cloud
/// stores per-run, per-role; the run label keys the run like the descriptor store does).
type SlotKey = (String, String);

/// The in-memory seat registry fixture. Interior mutability so one instance can back concurrent
/// claimants (the CAS-race suites) and an HTTP responder simultaneously.
#[derive(Debug, Default)]
pub struct FakeSeatRegistry {
    slots: Mutex<BTreeMap<SlotKey, SeatSlot>>,
    skew_ms: u64,
}

impl FakeSeatRegistry {
    /// A fixture with the default skew grace ([`DEFAULT_SEAT_SKEW_MS`]).
    #[must_use]
    pub fn new() -> Self {
        Self::with_skew(DEFAULT_SEAT_SKEW_MS)
    }

    /// A fixture with an explicit skew grace (milliseconds).
    #[must_use]
    pub fn with_skew(skew_ms: u64) -> Self {
        Self {
            slots: Mutex::new(BTreeMap::new()),
            skew_ms,
        }
    }

    /// Claim / take over a seat (the `PUT {base}/runs/:id/seat/:role` semantics): the fold's
    /// compare-and-set with increment against the slot keyed by `(run, lease.body.role)`.
    pub fn claim(&self, run: &str, lease: &SeatLease, now_ms: u64) -> SeatMutationResponse {
        self.apply(
            run,
            &lease.body.role.clone(),
            SeatRequest::Claim(lease.clone()),
            now_ms,
        )
    }

    /// Renew (heartbeat) a held lease (the `POST …/seat/:role/heartbeat` semantics).
    pub fn renew(&self, run: &str, lease: &SeatLease, now_ms: u64) -> SeatMutationResponse {
        self.apply(
            run,
            &lease.body.role.clone(),
            SeatRequest::Renew(lease.clone()),
            now_ms,
        )
    }

    /// Release a held seat (the `DELETE …/seat/:role` semantics); the tombstone floor persists.
    pub fn release(
        &self,
        run: &str,
        role: &str,
        release: &SeatRelease,
        now_ms: u64,
    ) -> SeatMutationResponse {
        self.apply(run, role, SeatRequest::Release(release.clone()), now_ms)
    }

    /// Read a seat slot (the `GET …/seat/:role` semantics). A never-touched slot reads as
    /// unclaimed with no floor — run existence is the descriptor store's concern, not the seat
    /// slot's.
    pub fn read(&self, run: &str, role: &str) -> SeatState {
        self.slots
            .lock()
            .expect("seat slots mutex")
            .get(&(run.to_string(), role.to_string()))
            .map_or(
                SeatState::Unclaimed {
                    last_leadership_term: None,
                },
                SeatSlot::state,
            )
    }

    /// Apply one mutation under the single-writer lock — the serialization every conforming
    /// registry provides around the pure fold.
    fn apply(
        &self,
        run: &str,
        role: &str,
        request: SeatRequest,
        now_ms: u64,
    ) -> SeatMutationResponse {
        let mut slots = self.slots.lock().expect("seat slots mutex");
        let slot = slots
            .entry((run.to_string(), role.to_string()))
            .or_default();
        let (next, decision) = slot.fold(&request, now_ms, self.skew_ms);
        *slot = next;
        SeatMutationResponse {
            decision,
            state: slot.state(),
        }
    }
}
