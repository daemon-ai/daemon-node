// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`RunDiscovery`] — the run-discovery + envelope-fetch seam the join flow drives (spec §6.1/§6.5;
//! A1).
//!
//! [`VhcService::vhc_join`](crate::VhcService) used to derive eligibility from a hardware
//! probe against a hardcoded allowlist coordinator (a placeholder). A1 replaces that with real
//! discovery: resolve the run from the coordinator registry, fetch + blake3-verify the frozen
//! envelope, and hand it to the worker's existing `AssessRun` for a real §6.5 verdict **before**
//! `JoinRun`. This trait is the seam (a [`EgressRunDiscovery`] over
//! [`daemon_vhc_net::RegistryClient`] in production, a fake in tests) so the service is testable
//! without a live coordinator.

use async_trait::async_trait;
use daemon_vhc_net::{RegistryClient, RunId};

use crate::service::VhcError;

/// A published-checkpoint pointer resolved from the registry (spec §9): the round a
/// checkpoint covers and its content address, the late-join restore input. Pointers are keyed
/// per `(role, kind)` — a role restores only from its own role's state, and a periodic LIVE
/// pointer is a distinct slot from a graceful-leave DRAIN snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointPointer {
    /// The envelope role whose state the checkpoint captures.
    pub role: String,
    /// The pointer kind (`"live"` / `"drain"`).
    pub kind: String,
    /// The round the checkpoint captures.
    pub round: u64,
    /// blake3 of the checkpoint document (hex).
    pub hash: String,
    /// The checkpoint byte length (advisory).
    pub size: u64,
}

/// A discovered run: the coordination facts the node needs to assess + join (never experiment
/// config or module bytes — the seam rule).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscoveredRun {
    /// The run id (coordinator-assigned).
    pub run_id: String,
    /// The coordinator endpoint the run is served from (the WS `base_url` + presign base).
    pub coordinator: String,
    /// blake3 of the frozen envelope (hex) — the assert the peer joins under (§6.5).
    pub envelope_hash: String,
    /// The vhc proto version the run is pinned to (§16).
    pub proto_version: u32,
}

/// The discovery seam: list/resolve runs from the coordinator registry and fetch a run's frozen
/// envelope bytes (blake3-verified). Implemented by [`EgressRunDiscovery`] (real) + test fakes.
#[async_trait]
pub trait RunDiscovery: Send + Sync {
    /// Discover all runs the coordinator advertises (registry `GET /runs`).
    async fn list_runs(&self) -> Result<Vec<DiscoveredRun>, VhcError>;
    /// Resolve one run (`GET /runs/:id`); `None` if the coordinator does not know it.
    async fn get_run(&self, run_id: &str) -> Result<Option<DiscoveredRun>, VhcError>;
    /// Fetch the run's frozen envelope bytes (presigned GET + blake3-verify). Errors if the run is
    /// unknown or the bytes do not match the descriptor's hash.
    async fn fetch_envelope(&self, run_id: &str) -> Result<Vec<u8>, VhcError>;

    /// Publish this run's latest checkpoint pointer to the registry (spec §9), keyed by
    /// the `(role, kind)` slot it fills. The checkpoint DOCUMENT already lives on the payload
    /// plane (the session put it there); this records the round → content-address pointer a
    /// late joiner reads. Best-effort: the default is a no-op so fakes/offline nodes need not
    /// implement it.
    async fn publish_checkpoint(
        &self,
        run_id: &str,
        role: &str,
        kind: &str,
        round: u64,
        hash: &str,
        size: u64,
    ) -> Result<(), VhcError> {
        let _ = (run_id, role, kind, round, hash, size);
        Ok(())
    }

    /// Resolve `role`'s best restore pointer for this run (spec §9; the late-join restore
    /// input): the seat's OWN pointer first (correct replica-local semantics), falling back to
    /// a sibling seat's pointer in the same FAMILY only when the seat has published nothing —
    /// [`best_restore_pointer`]. Within each scope a drain snapshot never shadows a live
    /// restore source, and a pointer outside the family (a different role kind) is never
    /// consulted. `None` when the family has no pointer (a fresh start). Default `None` (no
    /// registry state).
    async fn fetch_checkpoint(
        &self,
        run_id: &str,
        role: &str,
    ) -> Result<Option<CheckpointPointer>, VhcError> {
        let _ = (run_id, role);
        Ok(None)
    }

    /// Publish this node's signed iroh roster record (`PUT {base}/runs/:id/roster`). The
    /// registry applies its structural monotonic upsert; a stale refusal is surfaced as a
    /// [`VhcError::Discovery`] (the join proceeds — a fresher record of OURS already stands).
    /// Default no-op so fakes/offline nodes need not implement it (the iroh plane is opt-in).
    async fn publish_roster(
        &self,
        run_id: &str,
        record: &daemon_vhc_proto::RosterRecord,
    ) -> Result<(), VhcError> {
        let _ = (run_id, record);
        Ok(())
    }

    /// Fetch the run's stored roster records (`GET {base}/runs/:id/roster`) — UNVERIFIED
    /// registry state: the caller judges every entry (`crate::roster::verified_iroh_roster`).
    /// Default empty (no registry state).
    async fn fetch_roster(
        &self,
        run_id: &str,
    ) -> Result<Vec<daemon_vhc_proto::RosterRecord>, VhcError> {
        let _ = run_id;
        Ok(Vec::new())
    }

    /// Fetch the run's stored archive-head records (`GET {base}/runs/:id/archive/heads`) —
    /// UNVERIFIED registry state: the caller judges every record
    /// (`daemon_vhc_proto::verify_chains` against the genesis-trusted bases) before trusting a
    /// lineage or a round claim. Default empty (no registry state), so fakes/offline nodes need
    /// not implement it.
    async fn fetch_archive_heads(
        &self,
        run_id: &str,
    ) -> Result<Vec<daemon_vhc_proto::ArchiveHeadRecord>, VhcError> {
        let _ = run_id;
        Ok(Vec::new())
    }

    /// The registry descriptor's AUTHORED total-round count (`rounds`, the `stop_rounds` figure
    /// registered at seed), `None` when the run is driven by another stop condition or the
    /// registry does not know it. REL-9(b)'s completion stand-down compares this authored
    /// figure against the VERIFIED archive-head round claim — descriptor metadata alone never
    /// proves progress. Default `None` (no registry state), so fakes/offline nodes need not
    /// implement it.
    async fn run_rounds(&self, run_id: &str) -> Result<Option<u64>, VhcError> {
        let _ = run_id;
        Ok(None)
    }
}

/// The role FAMILY a checkpoint pointer may FALL BACK across: authored per-seat roles
/// (`trainer-0`, `trainer-1`, …) share one family (`trainer`). A role without a seat suffix is
/// its own family (`coordinator`, legacy `trainer`).
///
/// Family scope is a FALLBACK, not an equivalence (Gate D', re-adjudicating the c15g defect-8
/// fix): the deterministic-state contract makes only the CLASS-0 consensus-canonical sections
/// identical across seats at a round boundary (the digests cover exactly those; c15g proved
/// live agreement) — a checkpoint DOCUMENT additionally carries class-1 replica-local sections
/// (tiny-llama's `ef` error-feedback residuals and AdamW moments are per-seat data-slice
/// trajectories), so restoring a sibling's doc ADOPTS foreign replica-local state.
/// Consensus-safe, but a recorded posture ([`crate::VhcService`] emits a
/// `sibling_restore_adopted` warning), never an inferred identity.
fn role_family(role: &str) -> &str {
    match role.rsplit_once('-') {
        Some((family, seat)) if !seat.is_empty() && seat.bytes().all(|b| b.is_ascii_digit()) => {
            family
        }
        _ => role,
    }
}

/// The restore preference (spec §9, Gate D' order): the seat's OWN pointers first — freshest
/// own `live`, else own `drain` — falling back to a SIBLING seat in the same family
/// ([`role_family`], same live-over-drain rule) only when the seat has published nothing.
///
/// Own-seat-first restores the correct replica-local semantics (class-1 optimizer/error-feedback
/// state is the seat's own trajectory); an own pointer's extra staleness is bridged by archive
/// catch-up (Gate B'), not by adopting a sibling's fresher doc. The sibling fallback keeps the
/// c15g defect-8 lesson: a crashed seat whose OWN slot was never published (alternating
/// publisher election) must not wedge `CheckpointStale` while a family sibling's pointer (the
/// identical class-0 deterministic state) stands published.
pub fn best_restore_pointer(
    pointers: &[CheckpointPointer],
    role: &str,
) -> Option<CheckpointPointer> {
    let family = role_family(role);
    let best = |scope: &dyn Fn(&&CheckpointPointer) -> bool| {
        let of_kind = |kind: &str| {
            pointers
                .iter()
                .filter(scope)
                .filter(|p| p.kind == kind)
                .max_by_key(|p| p.round)
                .cloned()
        };
        of_kind(daemon_vhc_net::CHECKPOINT_KIND_LIVE)
            .or_else(|| of_kind(daemon_vhc_net::CHECKPOINT_KIND_DRAIN))
    };
    best(&|p: &&CheckpointPointer| p.role == role)
        .or_else(|| best(&|p: &&CheckpointPointer| role_family(&p.role) == family))
}

/// The production [`RunDiscovery`]: a [`RegistryClient`] against a vhc coordinator base.
pub struct EgressRunDiscovery {
    registry: RegistryClient,
    coordinator: String,
}

impl EgressRunDiscovery {
    /// Wrap a configured [`RegistryClient`]; its base URL is the coordinator endpoint the discovered
    /// runs are served from (the WS + presign base).
    pub fn new(registry: RegistryClient) -> Self {
        let coordinator = registry.base_url().to_string();
        Self {
            registry,
            coordinator,
        }
    }
}

#[async_trait]
impl RunDiscovery for EgressRunDiscovery {
    async fn list_runs(&self) -> Result<Vec<DiscoveredRun>, VhcError> {
        let runs = self
            .registry
            .list_runs()
            .await
            .map_err(|e| VhcError::Discovery(e.to_string()))?;
        Ok(runs
            .into_iter()
            .map(|d| DiscoveredRun {
                run_id: d.run_id,
                coordinator: self.coordinator.clone(),
                envelope_hash: d.envelope_hash,
                proto_version: d.proto_version,
            })
            .collect())
    }

    async fn get_run(&self, run_id: &str) -> Result<Option<DiscoveredRun>, VhcError> {
        let run = self
            .registry
            .get_run(run_id)
            .await
            .map_err(|e| VhcError::Discovery(e.to_string()))?;
        Ok(run.map(|d| DiscoveredRun {
            run_id: d.run_id,
            coordinator: self.coordinator.clone(),
            envelope_hash: d.envelope_hash,
            proto_version: d.proto_version,
        }))
    }

    async fn fetch_envelope(&self, run_id: &str) -> Result<Vec<u8>, VhcError> {
        let descriptor = self
            .registry
            .get_run(run_id)
            .await
            .map_err(|e| VhcError::Discovery(e.to_string()))?
            .ok_or_else(|| VhcError::Discovery(format!("run {run_id} not found in registry")))?;
        self.registry
            .fetch_envelope(&RunId::new(run_id), &descriptor)
            .await
            .map_err(|e| VhcError::Discovery(e.to_string()))
    }

    async fn publish_checkpoint(
        &self,
        run_id: &str,
        role: &str,
        kind: &str,
        round: u64,
        hash: &str,
        size: u64,
    ) -> Result<(), VhcError> {
        self.registry
            .publish_checkpoint(run_id, role, kind, round, hash, size)
            .await
            .map_err(|e| VhcError::Discovery(e.to_string()))
    }

    async fn fetch_checkpoint(
        &self,
        run_id: &str,
        role: &str,
    ) -> Result<Option<CheckpointPointer>, VhcError> {
        let state = self
            .registry
            .fetch_state(run_id)
            .await
            .map_err(|e| VhcError::Discovery(e.to_string()))?;
        let pointers: Vec<CheckpointPointer> = state
            .map(|s| s.checkpoints)
            .unwrap_or_default()
            .into_iter()
            .map(|c| CheckpointPointer {
                role: c.role,
                kind: c.kind,
                round: c.round,
                hash: c.hash,
                size: c.size,
            })
            .collect();
        Ok(best_restore_pointer(&pointers, role))
    }

    async fn publish_roster(
        &self,
        run_id: &str,
        record: &daemon_vhc_proto::RosterRecord,
    ) -> Result<(), VhcError> {
        match self
            .registry
            .publish_roster(&RunId::new(run_id), record)
            .await
            .map_err(|e| VhcError::Discovery(e.to_string()))?
        {
            daemon_vhc_net::RosterPublishOutcome::Accepted => Ok(()),
            // A stale refusal surfaces TYPED with the slot's stored record: the join
            // transaction judges it (a verified own-base record is own-floor evidence; anything
            // else fails closed) — never an internal publish retry, and never a string the
            // caller has to parse.
            daemon_vhc_net::RosterPublishOutcome::Refused {
                decision:
                    daemon_vhc_proto::RosterDecision::RejectedStale {
                        stored_incarnation, ..
                    },
                stored,
            } => Err(VhcError::RosterStale {
                stored_incarnation,
                stored,
            }),
            daemon_vhc_net::RosterPublishOutcome::Refused { decision, .. } => Err(
                VhcError::Discovery(format!("roster publish refused: {decision:?}")),
            ),
        }
    }

    async fn fetch_roster(
        &self,
        run_id: &str,
    ) -> Result<Vec<daemon_vhc_proto::RosterRecord>, VhcError> {
        self.registry
            .fetch_roster(&RunId::new(run_id))
            .await
            .map_err(|e| VhcError::Discovery(e.to_string()))
    }

    async fn fetch_archive_heads(
        &self,
        run_id: &str,
    ) -> Result<Vec<daemon_vhc_proto::ArchiveHeadRecord>, VhcError> {
        use daemon_vhc_net::ArchiveHeadStore as _;
        self.registry
            .archive_head_store(&RunId::new(run_id))
            .fetch_heads()
            .await
            .map_err(|e| VhcError::Discovery(e.to_string()))
    }

    async fn run_rounds(&self, run_id: &str) -> Result<Option<u64>, VhcError> {
        let run = self
            .registry
            .get_run(run_id)
            .await
            .map_err(|e| VhcError::Discovery(e.to_string()))?;
        Ok(run.and_then(|d| d.rounds))
    }
}
