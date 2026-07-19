// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The resident coordinator **seat keeper** — the always-on claim/heartbeat/release loop over the
//! seat manager (architecture §6.3; ABI §12.4; D-P9).
//!
//! [`crate::seat`] owns the claimant mechanics (bid derivation, lease/renew/release authorship,
//! peer-side authorization); this module makes them RESIDENT: when the owner enables coordinator
//! duty (`[vhc] seat_claim`), the node service drives one keeper pass per heartbeat interval over
//! every joined run whose admitted role is the configured seat role —
//!
//! - **unheld slot**: read the slot, derive a bid (`None` while a live incumbent holds — stand
//!   by), author + CAS the claim; a lost race stands by (fencing-is-safe-not-seamless);
//! - **held lease**: re-sign with a fresh expiry under the SAME token and renew; a refused renew
//!   means the seat moved — the claimant is FENCED and drops the lease (never a takeover fight);
//! - **release**: an owner pause/leave (and node shutdown) releases the held lease signed, so the
//!   registry tombstones the token and the floor persists.
//!
//! The registry stays untrusted storage throughout: every outcome here is a structural CAS fold;
//! authority judgments live peer-side ([`crate::seat::authorize_incumbent`]).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use daemon_vhc_net::{RegistryClient, RunId, SeatClaimOutcome, VhcNetError};
use daemon_vhc_proto::{ControlEndpoint, SeatLease, SeatRelease, SeatState, DEFAULT_SEAT_SKEW_MS};
use daemon_vhc_session::keystore::VhcKeystore;

use crate::seat::{author_claim, author_release, author_renew, derive_bid, CoordinatorSeat};
use crate::service::VhcError;

/// The seat-slot directory seam — the registry's four seat operations, abstracted so the keeper
/// runs identically over the production [`RegistryClient`] and a test fold.
#[async_trait]
pub trait SeatDirectory: Send + Sync {
    /// Read a run's seat slot for `role` (an unknown slot reads unclaimed with no floor).
    async fn read_seat(&self, run: &str, role: &str) -> Result<SeatState, VhcError>;
    /// CAS a claim; a refusal comes back typed with the slot's current state.
    async fn claim_seat(&self, run: &str, lease: &SeatLease) -> Result<SeatClaimOutcome, VhcError>;
    /// Renew (heartbeat) a held lease under the same token.
    async fn renew_seat(&self, run: &str, lease: &SeatLease) -> Result<SeatClaimOutcome, VhcError>;
    /// Release a held seat (the registry tombstones the token; the floor persists).
    async fn release_seat(
        &self,
        run: &str,
        role: &str,
        release: &SeatRelease,
    ) -> Result<(), VhcError>;
}

fn net(e: VhcNetError) -> VhcError {
    VhcError::Discovery(e.to_string())
}

#[async_trait]
impl SeatDirectory for RegistryClient {
    async fn read_seat(&self, run: &str, role: &str) -> Result<SeatState, VhcError> {
        Ok(RegistryClient::read_seat(self, &RunId::new(run), role)
            .await
            .map_err(net)?
            .unwrap_or(SeatState::Unclaimed {
                last_fencing_token: None,
            }))
    }
    async fn claim_seat(&self, run: &str, lease: &SeatLease) -> Result<SeatClaimOutcome, VhcError> {
        RegistryClient::claim_seat(self, &RunId::new(run), lease)
            .await
            .map_err(net)
    }
    async fn renew_seat(&self, run: &str, lease: &SeatLease) -> Result<SeatClaimOutcome, VhcError> {
        RegistryClient::renew_seat(self, &RunId::new(run), lease)
            .await
            .map_err(net)
    }
    async fn release_seat(
        &self,
        run: &str,
        role: &str,
        release: &SeatRelease,
    ) -> Result<(), VhcError> {
        RegistryClient::release_seat(self, &RunId::new(run), role, release)
            .await
            .map_err(net)
    }
}

/// One run the keeper covers: the seat scope resolved from the durable join intent (the admitted
/// tuple's identity + the run row's coordinator endpoint).
#[derive(Clone, Debug)]
pub struct SeatCandidate {
    /// The run label (keystore namespace + registry run key).
    pub run_label: String,
    /// The run's cryptographic identity (genesis hash).
    pub genesis_hash: [u8; 32],
    /// The claimed role label (the configured seat role, matching the admitted tuple).
    pub role: String,
    /// The run epoch the lease is scoped to.
    pub epoch: u64,
    /// The pinned coordinator module hash (from the admitted tuple).
    pub module_hash: [u8; 32],
    /// The control-plane endpoint peers dial while this node holds the seat.
    pub endpoint: ControlEndpoint,
}

impl SeatCandidate {
    fn scope(&self) -> CoordinatorSeat<'_> {
        CoordinatorSeat {
            run_label: &self.run_label,
            genesis_hash: self.genesis_hash,
            role: &self.role,
            epoch: self.epoch,
            module_hash: self.module_hash,
            endpoint: self.endpoint.clone(),
        }
    }
}

/// What one keeper pass observed for a run (surfaced by the service as events).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SeatNote {
    /// The claim CAS was won: this node now holds the seat at the carried incarnation.
    Claimed {
        /// The run whose seat was claimed.
        run_label: String,
        /// The lease incarnation (== fencing token, [SEAT-1]).
        incarnation: u64,
    },
    /// The heartbeat renewed the held lease.
    Renewed {
        /// The run whose lease renewed.
        run_label: String,
    },
    /// A renew was refused — the seat moved and this claimant is fenced out; the lease dropped.
    Fenced {
        /// The run whose seat moved.
        run_label: String,
        /// The registry's typed verdict.
        detail: String,
    },
    /// A keeper step failed (authorship / directory transport); retried next pass.
    Error {
        /// The run the step was for.
        run_label: String,
        /// The failure detail.
        detail: String,
    },
}

/// The resident seat keeper: held leases + the directory + the identity store the claims are
/// authored against.
pub struct SeatKeeper {
    directory: Arc<dyn SeatDirectory>,
    identity_dir: PathBuf,
    held: Mutex<BTreeMap<String, SeatLease>>,
}

impl SeatKeeper {
    /// A keeper authoring against the identity store at `identity_dir`, over `directory`.
    pub fn new(directory: Arc<dyn SeatDirectory>, identity_dir: PathBuf) -> Self {
        Self {
            directory,
            identity_dir,
            held: Mutex::new(BTreeMap::new()),
        }
    }

    /// The held lease's incarnation for `run_label` (observability / tests).
    pub fn held_incarnation(&self, run_label: &str) -> Option<u64> {
        self.held
            .lock()
            .expect("seat keeper lock")
            .get(run_label)
            .map(|l| l.body.incarnation)
    }

    /// One keeper pass over `candidates`: renew every held lease, claim every unheld slot whose
    /// bid derives (a live incumbent means stand by). Every outcome is a note; one run's failure
    /// never blocks the rest.
    pub async fn tick(&self, candidates: &[SeatCandidate], now_ms: u64) -> Vec<SeatNote> {
        let mut notes = Vec::new();
        for candidate in candidates {
            let held = {
                let held = self.held.lock().expect("seat keeper lock");
                held.get(&candidate.run_label).cloned()
            };
            let step = match held {
                Some(lease) => self.renew(candidate, &lease, now_ms).await,
                None => self.claim(candidate, now_ms).await,
            };
            match step {
                Ok(Some(note)) => notes.push(note),
                Ok(None) => {}
                Err(e) => notes.push(SeatNote::Error {
                    run_label: candidate.run_label.clone(),
                    detail: e.to_string(),
                }),
            }
        }
        notes
    }

    /// Claim an unheld slot when the bid derives (unclaimed → floor + 1; expired incumbent →
    /// takeover at held + 1; live incumbent → stand by).
    async fn claim(
        &self,
        candidate: &SeatCandidate,
        now_ms: u64,
    ) -> Result<Option<SeatNote>, VhcError> {
        let state = self
            .directory
            .read_seat(&candidate.run_label, &candidate.role)
            .await?;
        let Some(bid) = derive_bid(&state, now_ms, DEFAULT_SEAT_SKEW_MS) else {
            return Ok(None); // a live incumbent holds — stand by
        };
        let keystore = self.keystore()?;
        let lease = author_claim(&keystore, &candidate.scope(), bid, now_ms)
            .map_err(|e| VhcError::Internal(format!("author seat claim: {e}")))?;
        match self
            .directory
            .claim_seat(&candidate.run_label, &lease)
            .await?
        {
            SeatClaimOutcome::Won(won) => {
                let incarnation = won.body.incarnation;
                self.held
                    .lock()
                    .expect("seat keeper lock")
                    .insert(candidate.run_label.clone(), won);
                Ok(Some(SeatNote::Claimed {
                    run_label: candidate.run_label.clone(),
                    incarnation,
                }))
            }
            // Lost the race: another claimant CASed first — stand by (never a takeover fight).
            SeatClaimOutcome::Lost { .. } => Ok(None),
        }
    }

    /// Heartbeat a held lease: re-sign with a fresh expiry under the SAME token. A refusal means
    /// the seat moved — this claimant is fenced and the lease drops.
    async fn renew(
        &self,
        candidate: &SeatCandidate,
        lease: &SeatLease,
        now_ms: u64,
    ) -> Result<Option<SeatNote>, VhcError> {
        let keystore = self.keystore()?;
        let renewed = author_renew(&keystore, &candidate.scope(), lease, now_ms)
            .map_err(|e| VhcError::Internal(format!("author seat renew: {e}")))?;
        match self
            .directory
            .renew_seat(&candidate.run_label, &renewed)
            .await?
        {
            SeatClaimOutcome::Won(won) => {
                self.held
                    .lock()
                    .expect("seat keeper lock")
                    .insert(candidate.run_label.clone(), won);
                Ok(Some(SeatNote::Renewed {
                    run_label: candidate.run_label.clone(),
                }))
            }
            SeatClaimOutcome::Lost { decision, .. } => {
                self.held
                    .lock()
                    .expect("seat keeper lock")
                    .remove(&candidate.run_label);
                Ok(Some(SeatNote::Fenced {
                    run_label: candidate.run_label.clone(),
                    detail: format!("{decision:?}"),
                }))
            }
        }
    }

    /// Release the held lease for `run_label` (owner pause/leave, node shutdown): a signed
    /// release the registry tombstones — the floor persists, a successor bids floor + 1. No-op
    /// when nothing is held.
    pub async fn release_run(&self, run_label: &str) -> Result<(), VhcError> {
        let Some(lease) = self
            .held
            .lock()
            .expect("seat keeper lock")
            .remove(run_label)
        else {
            return Ok(());
        };
        let scope = CoordinatorSeat {
            run_label,
            genesis_hash: lease.body.run_id.0,
            role: &lease.body.role,
            epoch: lease.body.epoch,
            module_hash: lease.body.module_hash.0,
            endpoint: lease.body.endpoint.clone(),
        };
        let keystore = self.keystore()?;
        let release = author_release(&keystore, &scope, lease.body.incarnation)
            .map_err(|e| VhcError::Internal(format!("author seat release: {e}")))?;
        self.directory
            .release_seat(run_label, &lease.body.role, &release)
            .await
    }

    /// Release every held lease (node shutdown — the fenced release, so successors take over at
    /// floor + 1 without waiting out the TTL). Best-effort per run.
    pub async fn release_all(&self) -> Vec<SeatNote> {
        let runs: Vec<String> = {
            let held = self.held.lock().expect("seat keeper lock");
            held.keys().cloned().collect()
        };
        let mut notes = Vec::new();
        for run in runs {
            if let Err(e) = self.release_run(&run).await {
                notes.push(SeatNote::Error {
                    run_label: run,
                    detail: format!("seat release failed: {e}"),
                });
            }
        }
        notes
    }

    fn keystore(&self) -> Result<VhcKeystore, VhcError> {
        VhcKeystore::open(&self.identity_dir)
            .map_err(|e| VhcError::Internal(format!("open identity keystore: {e}")))
    }
}
