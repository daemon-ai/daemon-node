// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **Fit Verdict** — the empirical admission authority the composed estimate is not
//! (`docs/specs/vhc-architecture-spec.md`, resource model).
//!
//! ## Why a verdict exists
//!
//! The composed [`PhysicalEstimate`](crate::planner::PhysicalEstimate) is a *conservative
//! estimate*: it exists to refuse cheaply (an estimate that already exceeds supply needs no
//! probe) and to size the enforced budget. It is not, and cannot be made, a proof that a module
//! fits on a device — allocators, drivers and framework pools do not honor arithmetic. The thing
//! that can answer "does it fit?" is the device itself: run the actual module on the actual
//! backend at the actual granted geometry under the enforced budget, in the sandbox that
//! contains it, and record what happened. That recorded answer is the Fit Verdict, and it — not
//! the estimate — is what final admission at a given geometry stands on.
//!
//! ## Memoization identity
//!
//! A verdict is memoized by [`FitProbeKey`]: the module's content hash, the backend
//! implementation revision it ran against, the plan it emitted, the grant that bound its
//! geometry, and the budget it ran under. Change any member and the verdict simply does not
//! apply — there is no "close enough" lookup, because a near-miss key is exactly a stale or
//! swapped artifact. Fleet feasibility for a frozen roster is then a set membership question:
//! every node holds a green verdict for its key. No search, no roster arithmetic.
//!
//! ## What a red verdict is
//!
//! A contained, typed outcome — the budget's trap, recorded. It is evidence about a
//! configuration, not an outage and not an escalation: the grant's declared space selects a
//! smaller geometry and probes again. Only an empty declared space at minimum geometry leaves
//! the resource model's jurisdiction, and what leaves is a *product* question (is this device in
//! or out), never a byte question.

use std::collections::BTreeMap;

use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Hash};
use serde::{Deserialize, Serialize};

/// Schema identity for [`FitVerdict`]'s canonical encoding.
pub const FIT_VERDICT_SCHEMA: u32 = 1;

/// The identity a fit probe ran as, and the identity its verdict is memoized by.
///
/// Every member is a digest of an artifact that already has a canonical encoding elsewhere in
/// the admission path; the key adds no vocabulary of its own. The budget rides in the key —
/// rather than in the outcome alone — because "fits under budget B" is a different statement
/// from "fits under budget B′", and a memo hit must never substitute one for the other.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FitProbeKey {
    /// blake3 of the module bytes that ran.
    pub module_hash: Hash,
    /// Digest of the backend implementation revision record the probe ran on. A driver,
    /// framework or kernel-set change moves this digest and invalidates every verdict under it.
    pub backend_revision_digest: Hash,
    /// The module's Logical Resource Plan hash — the shape that was probed.
    pub plan_hash: Hash,
    /// The Execution Grant's canonical-bytes digest — the geometry the probe bound.
    pub grant_hash: Hash,
    /// The enforced budget, in bytes, the probe ran under (the figure the estimate sized).
    pub budget_bytes: u64,
}

impl FitProbeKey {
    /// The key's canonical CBOR bytes.
    ///
    /// # Errors
    /// [`VerdictError::Encoding`] when canonical encoding fails.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, VerdictError> {
        to_canonical_vec(self).map_err(|e| VerdictError::Encoding(e.to_string()))
    }

    /// blake3 of the canonical bytes — the memo address of this probe identity.
    ///
    /// # Errors
    /// [`VerdictError::Encoding`] when canonical encoding fails.
    pub fn digest(&self) -> Result<Hash, VerdictError> {
        Ok(blake3_hash(&self.to_canonical_bytes()?))
    }
}

/// What the device said.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitOutcome {
    /// The probe ran the granted geometry to completion inside the enforced budget.
    Fits {
        /// The measured peak, as the governor accounted it. Evidence, not a new authority: the
        /// budget stays what admission enforces.
        measured_peak_bytes: u64,
    },
    /// The probe was contained by the budget or trapped, TYPED. The slug is the stable refusal
    /// vocabulary (the governor's trap or the engine's typed abort), so a red verdict can be
    /// read without re-running anything.
    Contained {
        /// The stable machine-readable slug of the trap that contained the probe.
        trap_slug: String,
    },
}

/// A content-addressed, memoized fit-probe result: the key it ran as and what the device said.
///
/// The verdict is the **admission authority at its key**. A green verdict admits that exact
/// (module, backend revision, plan, geometry, budget) tuple without re-probing; a red verdict
/// refuses it without re-probing; an absent verdict means the probe has not run — never that
/// the estimate gets to answer instead.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FitVerdict {
    /// Encoding identity, [`FIT_VERDICT_SCHEMA`].
    pub schema: u32,
    /// The probe identity this verdict memoizes.
    pub key: FitProbeKey,
    /// What the device said.
    pub outcome: FitOutcome,
}

impl FitVerdict {
    /// The verdict's canonical CBOR bytes.
    ///
    /// # Errors
    /// [`VerdictError::Encoding`] when canonical encoding fails.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, VerdictError> {
        to_canonical_vec(self).map_err(|e| VerdictError::Encoding(e.to_string()))
    }

    /// blake3 of the canonical bytes — the verdict's content address.
    ///
    /// # Errors
    /// [`VerdictError::Encoding`] when canonical encoding fails.
    pub fn digest(&self) -> Result<Hash, VerdictError> {
        Ok(blake3_hash(&self.to_canonical_bytes()?))
    }

    /// Whether this verdict admits its key.
    #[must_use]
    pub fn is_green(&self) -> bool {
        matches!(self.outcome, FitOutcome::Fits { .. })
    }
}

/// A memo of verdicts keyed by probe identity.
///
/// Deliberately a value store with no eviction and no "nearest" lookup: a verdict is either held
/// for exactly the key asked about, or the probe runs. The store refuses to overwrite a key with
/// a *different* verdict — two verdicts for one key means the key under-identifies the probe
/// (a nondeterministic probe or an unrecorded revision), and that is a defect to surface, not a
/// race to last-write-wins.
#[derive(Debug, Default)]
pub struct FitVerdictStore {
    verdicts: BTreeMap<Hash, FitVerdict>,
}

impl FitVerdictStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The verdict held for `key`, if the probe has run.
    ///
    /// # Errors
    /// [`VerdictError::Encoding`] when the key cannot be addressed.
    pub fn lookup(&self, key: &FitProbeKey) -> Result<Option<&FitVerdict>, VerdictError> {
        Ok(self.verdicts.get(&key.digest()?))
    }

    /// Record a verdict. Idempotent for an identical verdict; a *conflicting* verdict for the
    /// same key refuses typed.
    ///
    /// # Errors
    /// [`VerdictError::Encoding`] when addressing fails; [`VerdictError::Conflicting`] when the
    /// key already holds a different outcome.
    pub fn record(&mut self, verdict: FitVerdict) -> Result<(), VerdictError> {
        let address = verdict.key.digest()?;
        if let Some(held) = self.verdicts.get(&address) {
            if *held != verdict {
                return Err(VerdictError::Conflicting {
                    key_digest: address,
                });
            }
            return Ok(());
        }
        self.verdicts.insert(address, verdict);
        Ok(())
    }
}

/// Why a verdict operation refused.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum VerdictError {
    /// Canonical encoding failed.
    #[error("verdict encoding: {0}")]
    Encoding(String),
    /// One probe key produced two different outcomes — the key under-identifies the probe.
    #[error(
        "conflicting verdicts for probe key {key_digest:?}: one key produced two outcomes, so \
         the probe identity is under-specified (a nondeterministic probe or an unrecorded \
         backend revision)"
    )]
    Conflicting {
        /// The memo address both verdicts claimed.
        key_digest: Hash,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> FitProbeKey {
        FitProbeKey {
            module_hash: Hash([1; 32]),
            backend_revision_digest: Hash([2; 32]),
            plan_hash: Hash([3; 32]),
            grant_hash: Hash([4; 32]),
            budget_bytes: 64 << 20,
        }
    }

    fn green() -> FitVerdict {
        FitVerdict {
            schema: FIT_VERDICT_SCHEMA,
            key: key(),
            outcome: FitOutcome::Fits {
                measured_peak_bytes: 48 << 20,
            },
        }
    }

    /// The content address is a function of the canonical bytes and nothing else.
    #[test]
    fn the_verdict_is_content_addressed_deterministically() {
        assert_eq!(green().digest().unwrap(), green().digest().unwrap());
        let mut other = green();
        other.outcome = FitOutcome::Contained {
            trap_slug: "MemoryBudgetExceeded".into(),
        };
        assert_ne!(green().digest().unwrap(), other.digest().unwrap());
    }

    /// Any changed key member is a different memo address: no near-miss lookup exists.
    #[test]
    fn every_key_member_is_identity_bearing() {
        let base = key().digest().unwrap();
        let mutations: Vec<FitProbeKey> = vec![
            FitProbeKey {
                module_hash: Hash([9; 32]),
                ..key()
            },
            FitProbeKey {
                backend_revision_digest: Hash([9; 32]),
                ..key()
            },
            FitProbeKey {
                plan_hash: Hash([9; 32]),
                ..key()
            },
            FitProbeKey {
                grant_hash: Hash([9; 32]),
                ..key()
            },
            FitProbeKey {
                budget_bytes: (64 << 20) + 1,
                ..key()
            },
        ];
        for mutated in mutations {
            assert_ne!(
                mutated.digest().unwrap(),
                base,
                "a changed member must move the memo address: {mutated:?}"
            );
        }
    }

    /// A memo hit returns the held verdict; an absent key returns nothing rather than an answer.
    #[test]
    fn the_store_memoizes_and_absence_is_not_an_answer() {
        let mut store = FitVerdictStore::new();
        assert_eq!(store.lookup(&key()).unwrap(), None);
        store.record(green()).unwrap();
        assert_eq!(store.lookup(&key()).unwrap(), Some(&green()));
        // Idempotent re-record of the identical verdict.
        store.record(green()).unwrap();
    }

    /// One key, two outcomes: typed refusal, because the key under-identifies the probe.
    #[test]
    fn a_conflicting_verdict_for_one_key_refuses_typed() {
        let mut store = FitVerdictStore::new();
        store.record(green()).unwrap();
        let red = FitVerdict {
            schema: FIT_VERDICT_SCHEMA,
            key: key(),
            outcome: FitOutcome::Contained {
                trap_slug: "MemoryBudgetExceeded".into(),
            },
        };
        let err = store.record(red).unwrap_err();
        assert!(matches!(err, VerdictError::Conflicting { .. }));
    }

    /// Green admits, contained refuses — and a contained verdict names its trap.
    #[test]
    fn green_admits_and_contained_names_its_trap() {
        assert!(green().is_green());
        let red = FitVerdict {
            schema: FIT_VERDICT_SCHEMA,
            key: key(),
            outcome: FitOutcome::Contained {
                trap_slug: "MemoryBudgetExceeded".into(),
            },
        };
        assert!(!red.is_green());
        match red.outcome {
            FitOutcome::Contained { trap_slug } => assert_eq!(trap_slug, "MemoryBudgetExceeded"),
            FitOutcome::Fits { .. } => unreachable!(),
        }
    }
}
