// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`FakeRosterRegistry`] — the local roster-registry fixture: an in-memory, thread-safe roster
//! store applying the **normative monotonic-upsert fold** (`daemon_vhc_proto::RosterSlot::fold`)
//! under a single writer, exactly as a conforming remote registry does behind the frozen HTTP
//! surface (`GET`/`PUT {base}/runs/:id/roster`).
//!
//! This is a **test/acceptance fixture, never a production registry**: it exists so the client
//! suites here and the multi-process acceptance suite can exercise publish/snapshot, the
//! freshness-key precedence, and the entry cap without a network. Serving it over HTTP is the
//! harness's concern; the semantics live entirely in the shared fold, so local and cloud behavior
//! can only diverge if one of them stops applying the fold — which the shared test vectors
//! (`daemon-vhc-proto/tests/fixtures/roster-vectors.json`) catch.
//!
//! Faithful to the registry posture (untrusted storage): **structural validation only** — the
//! fold checks the domain tag, dialability, size caps, slot consistency, and the freshness-key
//! monotonic upsert. It never verifies a signature and never judges authority; peers do that.

use std::collections::BTreeMap;
use std::sync::Mutex;

use daemon_vhc_proto::bytes::IrohId;
use daemon_vhc_proto::{
    RosterDecision, RosterMutationResponse, RosterRecord, RosterSlot, RosterSnapshot,
    MAX_ROSTER_ENTRIES,
};

/// One roster slot per `(run label, endpoint id, role)` key — the registry's storage
/// granularity.
///
/// The ROLE is load-bearing: a node's single iroh endpoint is shared by its co-located role
/// instances (a coordinator and its co-trainer), and each role's record carries the freshness of
/// its OWN incarnation ladder. An endpoint-only slot compared those ladders against each other:
/// whichever sibling had churned higher owned the slot, and the other's publish was refused
/// `RejectedStale` forever — observed live on the two-box WAN rung as a node that could never
/// rejoin its own run after a restart (the coordinator's low ladder against the trainer's high
/// stored record). Readers already group records by `(role, base identity)`
/// ([`RosterRecord::group_key`]); one slot per role is that same projection, and the normative
/// fold is untouched — freshness now only ever compares within one ladder.
type SlotKey = (String, IrohId, String);

/// The in-memory roster registry fixture. Interior mutability so one instance can back
/// concurrent publishers and an HTTP responder simultaneously.
#[derive(Debug, Default)]
pub struct FakeRosterRegistry {
    slots: Mutex<BTreeMap<SlotKey, RosterSlot>>,
}

impl FakeRosterRegistry {
    /// An empty fixture.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish (upsert) a roster record for `run` (the `PUT {base}/runs/:id/roster` semantics):
    /// the fold's monotonic freshness upsert against the slot keyed by the record's
    /// `(endpoint_id, role)`, plus the per-run entry cap for a NEW entry key.
    pub fn publish(&self, run: &str, record: &RosterRecord) -> RosterMutationResponse {
        let mut slots = self.slots.lock().expect("roster slots mutex");
        let key = (
            run.to_string(),
            record.body.endpoint_id,
            record.body.role.clone(),
        );
        if !slots.contains_key(&key) {
            let run_entries = slots.keys().filter(|(r, _, _)| r == run).count();
            if run_entries >= MAX_ROSTER_ENTRIES {
                return RosterMutationResponse {
                    decision: RosterDecision::RejectedStructural {
                        reason: format!("run roster is full ({MAX_ROSTER_ENTRIES} entries)"),
                    },
                    record: None,
                };
            }
        }
        let slot = slots.entry(key).or_default();
        let (next, decision) = slot.fold(record);
        *slot = next;
        RosterMutationResponse {
            decision,
            record: slot.record.clone(),
        }
    }

    /// Read a run's roster snapshot (the `GET {base}/runs/:id/roster` semantics). A never-touched
    /// run reads as an empty snapshot — run existence is the descriptor store's concern.
    pub fn snapshot(&self, run: &str) -> RosterSnapshot {
        let slots = self.slots.lock().expect("roster slots mutex");
        RosterSnapshot {
            entries: slots
                .iter()
                .filter(|((r, _, _), _)| r == run)
                .filter_map(|(_, slot)| slot.record.clone())
                .collect(),
        }
    }
}
