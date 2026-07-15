// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Admission: capability negotiation + join gating (spec §6.5; TDD PROTO-12/13).
//!
//! A pure predicate over the frozen [`Join`] message + the run config. The coordinator controls
//! *who may* join (auth); the peer alone decides *whether it can* (self-assessment, §6.5) — this is
//! the coordinator's half. Checks, in order: run-id, proto-version exact match, (optional) envelope
//! hash, capability **subset** (`required ⊆ advertised`, via the frozen [`CapabilitySet::admits`]),
//! roster capacity, duplicate.
//!
//! The `Join` carries an optional `envelope_hash` (Wave-3 additive carrier), threaded here as
//! `asserted_hash`: `tick` forwards `join.envelope_hash.as_ref()`, so a peer that assessed a
//! different envelope is rejected with `EnvelopeHashMismatch`; a legacy join that omits the hash
//! passes `None` and skips the check (back-compat).

use daemon_swarm_proto::messages::Join;
use daemon_swarm_proto::{CapabilitySet, Hash, PeerId, SwarmProtoVersion};

use crate::config::RunConfig;
use crate::io::AdmissionReject;
use crate::state::{Member, Phase};

/// A candidate join to evaluate: the signing `peer`, its signed-frame `version`, the [`Join`]
/// payload, and an optional asserted envelope hash (see the module note).
#[derive(Clone, Copy, Debug)]
pub struct JoinCandidate<'a> {
    /// The join's signer (node identity).
    pub peer: PeerId,
    /// The signed frame's proto version.
    pub version: SwarmProtoVersion,
    /// The join payload.
    pub join: &'a Join,
    /// An asserted envelope hash to check against the run (`None` skips the check).
    pub asserted_hash: Option<&'a Hash>,
}

/// Decide whether `cand.peer` may join, per §6.5.
///
/// `phase` gates joins out of halted phases; `roster`/`pending` bound capacity + duplicates (a
/// previously-`Dropped` member may rejoin).
pub fn admit(
    config: &RunConfig,
    phase: Phase,
    roster: &[Member],
    pending: &[Member],
    cand: &JoinCandidate<'_>,
) -> Result<(), AdmissionReject> {
    if cand.join.run_id != config.run_id {
        return Err(AdmissionReject::RunIdMismatch);
    }
    if cand.version != config.proto_version {
        return Err(AdmissionReject::VersionMismatch {
            expected: config.proto_version,
            got: cand.version,
        });
    }
    if let Some(h) = cand.asserted_hash {
        if *h != config.envelope_hash {
            return Err(AdmissionReject::EnvelopeHashMismatch);
        }
    }
    // Capability subset: advertised (join) must admit the run's required set.
    check_capabilities(&cand.join.capabilities, &config.required_capabilities)?;
    if phase.is_halted() {
        return Err(AdmissionReject::NotAccepting(phase));
    }
    // A currently-active member (healthy roster entry or a pending join) is a duplicate; a
    // previously-Dropped member is allowed to rejoin.
    let active_dup = roster.iter().any(|m| m.peer == cand.peer && m.is_healthy())
        || pending.iter().any(|m| m.peer == cand.peer);
    if active_dup {
        return Err(AdmissionReject::DuplicatePeer);
    }
    // Capacity counts healthy members + staged pending joins.
    let occupancy = roster.iter().filter(|m| m.is_healthy()).count() + pending.len();
    if occupancy >= config.max_peers as usize {
        return Err(AdmissionReject::RosterFull);
    }
    Ok(())
}

fn check_capabilities(
    advertised: &CapabilitySet,
    required: &CapabilitySet,
) -> Result<(), AdmissionReject> {
    let missing = advertised.missing(required);
    if missing.is_empty() {
        Ok(())
    } else {
        Err(AdmissionReject::MissingCapabilities(missing))
    }
}
