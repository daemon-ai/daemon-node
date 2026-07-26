//! The profile loader/store seam: where Backend Execution Profiles come from, and the one gate
//! between having a profile and being allowed to compose against it.
//!
//! # No authority is baked in
//!
//! The store holds no opinion about who may vouch for a profile. Every authority question arrives as
//! **injected data** — the owner's and the run's acceptance policies, supplied per selection — and the
//! store's own code names no signer, no key, no default. This is deliberate and is not merely tidy
//! layering: who holds signing authority for this program's profiles is an open question at the time
//! of writing, and a store that had assumed an answer would have to be rewritten around whatever is
//! decided rather than simply handed a different policy.
//!
//! A store with no policy therefore accepts nothing, which is the correct floor. An empty authority
//! set on one side defers to the other; empty on both accepts nothing at all, because nothing has
//! vouched.
//!
//! # Why authentication happens at selection and not at insertion
//!
//! It is tempting to authenticate on the way in, so that everything held is known-good. That would be
//! wrong. Authentication is a question about a profile *and* a particular running binary *and* a
//! particular run's policy: the same stored profile is authentic for one run and inauthentic for the
//! next, on the same machine, because the run's policy or the running revision changed. A profile
//! authenticated once and then trusted would be a stale verdict travelling as a current one.
//!
//! So the store holds **candidates**, and [`ProfileStore::select`] is the only way to obtain an
//! [`AuthenticatedProfile`] — which is the only thing composition accepts. A caller cannot skip the
//! gate, because there is no other constructor.

use std::collections::BTreeMap;

use crate::profile::{BackendExecutionProfile, ProfileError};
use crate::revision::{BackendClass, BackendImplementationRevision};
use crate::trust::{
    authenticate, AuthenticationRefusal, ProfileAcceptancePolicy, ProfileTrustEnvelope,
};
use daemon_vhc_proto::bytes::Hash;

/// Why a profile could not be taken into the store.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum StoreRefusal {
    /// The profile does not validate on its own terms.
    #[error("the profile is not well-formed: {0}")]
    Invalid(String),
    /// The envelope does not bind the profile bytes presented.
    #[error(
        "the trust envelope binds profile {envelope:?} but the bytes presented digest to {actual:?}; a \
         store that accepted this would be holding a vouching for one profile filed under another"
    )]
    BindingMismatch {
        /// What the envelope claims to bind.
        envelope: Hash,
        /// What the bytes actually digest to.
        actual: Hash,
    },
    /// A different profile is already filed under this digest.
    #[error(
        "profile {0:?} is already present with different bytes — a digest collision here would mean \
         one of the two is not what its digest says"
    )]
    DigestCollision(Hash),
}

/// Why no profile could be selected for a lane.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum SelectionRefusal {
    /// Nothing in the store is even a candidate for this lane and revision.
    #[error(
        "no profile in the store prices the {class} backend at implementation revision \
         `{revision}`; {held} profile(s) are held, none of them for this lane"
    )]
    NoCandidate {
        /// The class asked for.
        class: &'static str,
        /// The running implementation revision.
        revision: String,
        /// How many profiles the store holds in total, so an operator can tell an empty store from
        /// a mis-stocked one.
        held: usize,
    },
    /// Candidates existed and every one was refused, with the reason for each.
    #[error(
        "every candidate profile for the {class} backend was refused ({} considered); this is a \
         refusal to run, not a fallback",
        refusals.len()
    )]
    AllCandidatesRefused {
        /// The class asked for.
        class: &'static str,
        /// Each candidate's digest and why it was refused.
        refusals: Vec<(Hash, AuthenticationRefusal)>,
    },
    /// More than one profile authenticated, and choosing between them is not the store's to make.
    #[error(
        "{} profiles authenticate for the {class} backend at this revision, and which one is \
         composed against changes the physical claim — so this is refused rather than resolved by \
         iteration order, which would make a run's own evidence unable to explain its numbers",
        digests.len()
    )]
    Ambiguous {
        /// The class asked for.
        class: &'static str,
        /// The digests that all authenticated.
        digests: Vec<Hash>,
    },
}

/// A profile held in the store, with the envelope that vouches for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredProfile {
    /// The profile.
    pub profile: BackendExecutionProfile,
    /// Its trust envelope.
    pub envelope: ProfileTrustEnvelope,
}

/// Everything authentication needs that is not the profile itself.
///
/// Assembled per selection by the caller, which is what keeps authority out of the store: the two
/// policies arrive as data, and the store reads them rather than holding any of its own.
pub struct AuthenticationContext<'a> {
    /// The machine owner's policy — the operator refusing use of their own machine.
    pub owner: &'a ProfileAcceptancePolicy,
    /// The run's policy — the run refusing to be executed on terms it does not accept.
    pub run: &'a ProfileAcceptancePolicy,
    /// The revision record of the binary and lane actually about to execute.
    pub running: &'a BackendImplementationRevision,
    /// The planner version about to compose.
    pub planner_version: u32,
    /// Now, for validity windows.
    pub now_ms: u64,
}

/// A profile that has authenticated, for this binary, under both policies, at this moment.
///
/// The **only** way to obtain one is [`ProfileStore::select`]. Composition takes this type rather
/// than a bare profile, so an unauthenticated profile cannot reach a physical claim: there is no
/// constructor to reach for, so the gate cannot be skipped by a caller who did not know it was there.
///
/// It borrows from the store, so it also cannot outlive the profile it names.
#[derive(Clone, Copy, Debug)]
pub struct AuthenticatedProfile<'a> {
    profile: &'a BackendExecutionProfile,
    digest: Hash,
}

impl<'a> AuthenticatedProfile<'a> {
    /// The profile.
    #[must_use]
    pub fn profile(&self) -> &'a BackendExecutionProfile {
        self.profile
    }

    /// Its digest — what the admitted tuple and the composition evidence record carry.
    #[must_use]
    pub fn digest(&self) -> Hash {
        self.digest
    }
}

/// The profiles this node holds, and the gate to using one.
#[derive(Clone, Debug, Default)]
pub struct ProfileStore {
    by_digest: BTreeMap<Hash, StoredProfile>,
}

impl ProfileStore {
    /// An empty store. It accepts nothing until it is stocked, and even then nothing until a policy
    /// names an authority.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a profile and its envelope in.
    ///
    /// Insertion checks only what is true regardless of any run: that the profile is well-formed, and
    /// that the envelope binds these exact bytes. It deliberately does **not** authenticate — see the
    /// module note on why a verdict cached at insertion would be a stale verdict served as a current
    /// one.
    ///
    /// # Errors
    /// [`StoreRefusal`] when the profile does not validate, when the envelope binds other bytes, or
    /// when a different profile is already filed under the same digest.
    pub fn insert(
        &mut self,
        profile: BackendExecutionProfile,
        envelope: ProfileTrustEnvelope,
    ) -> Result<Hash, StoreRefusal> {
        profile
            .validate()
            .map_err(|e: ProfileError| StoreRefusal::Invalid(e.to_string()))?;
        let digest = profile
            .profile_digest()
            .map_err(|e| StoreRefusal::Invalid(e.to_string()))?;
        if envelope.profile_digest != digest {
            return Err(StoreRefusal::BindingMismatch {
                envelope: envelope.profile_digest,
                actual: digest,
            });
        }
        let entry = StoredProfile { profile, envelope };
        if let Some(existing) = self.by_digest.get(&digest) {
            if *existing != entry {
                return Err(StoreRefusal::DigestCollision(digest));
            }
        }
        self.by_digest.insert(digest, entry);
        Ok(digest)
    }

    /// How many profiles are held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_digest.len()
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_digest.is_empty()
    }

    /// One profile by digest, unauthenticated — for reporting what is held, never for composing.
    #[must_use]
    pub fn get(&self, digest: Hash) -> Option<&StoredProfile> {
        self.by_digest.get(&digest)
    }

    /// Every held profile's digest and the lane it prices, for an operator reading what a node has.
    pub fn inventory(&self) -> impl Iterator<Item = (Hash, BackendClass, &str)> {
        self.by_digest.iter().map(|(digest, held)| {
            (
                *digest,
                held.profile.backend_class,
                held.profile.implementation_revision.as_str(),
            )
        })
    }

    /// Select the one profile this lane may compose against, authenticating it under both policies.
    ///
    /// The gate. Candidates are the profiles priced for the running lane's class and implementation
    /// revision; each is authenticated in full, and exactly one must survive.
    ///
    /// **Ambiguity is refused, not resolved.** If two profiles authenticate, which one is composed
    /// against changes the physical claim, and therefore changes what the node reserved and what it
    /// admitted. Picking the first by iteration order would make a run's own evidence unable to
    /// explain its own numbers — the record would name a profile whose selection had no reason.
    ///
    /// **Every candidate refused is a refusal to run**, and each refusal is reported. An operator
    /// needs to know that four profiles were considered and why each was rejected; a bare "no profile
    /// available" turns a policy problem, an expiry and a revision mismatch into the same message.
    ///
    /// # Errors
    /// [`SelectionRefusal`] when nothing is a candidate, when every candidate is refused, or when
    /// more than one authenticates.
    pub fn select<'a>(
        &'a self,
        context: &AuthenticationContext<'_>,
    ) -> Result<AuthenticatedProfile<'a>, SelectionRefusal> {
        let class = context.running.backend_class;
        let revision = &context.running.backend_implementation.revision;
        let candidates: Vec<(&Hash, &StoredProfile)> = self
            .by_digest
            .iter()
            .filter(|(_, held)| {
                held.profile.backend_class == class
                    && held.profile.implementation_revision == *revision
            })
            .collect();

        if candidates.is_empty() {
            return Err(SelectionRefusal::NoCandidate {
                class: class.slug(),
                revision: revision.clone(),
                held: self.by_digest.len(),
            });
        }

        let mut authenticated: Vec<(Hash, &BackendExecutionProfile)> = Vec::new();
        let mut refusals: Vec<(Hash, AuthenticationRefusal)> = Vec::new();
        for (digest, held) in candidates {
            match authenticate(
                &held.profile,
                &held.envelope,
                context.owner,
                context.run,
                context.running,
                context.planner_version,
                context.now_ms,
            ) {
                Ok(()) => authenticated.push((*digest, &held.profile)),
                Err(refusal) => refusals.push((*digest, refusal)),
            }
        }

        match authenticated.len() {
            0 => Err(SelectionRefusal::AllCandidatesRefused {
                class: class.slug(),
                refusals,
            }),
            1 => {
                let (digest, profile) = authenticated.remove(0);
                Ok(AuthenticatedProfile { profile, digest })
            }
            _ => Err(SelectionRefusal::Ambiguous {
                class: class.slug(),
                digests: authenticated.into_iter().map(|(d, _)| d).collect(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::revision::fixtures;
    use crate::trust::fixtures as trust_fixtures;

    fn stocked() -> (ProfileStore, BackendImplementationRevision) {
        let running = fixtures::revision(BackendClass::Vulkan);
        let profile = trust_fixtures::profile_for(&running);
        let envelope = trust_fixtures::envelope_for(&profile, &running);
        let mut store = ProfileStore::new();
        store.insert(profile, envelope).expect("stocks");
        (store, running)
    }

    fn context<'a>(
        owner: &'a ProfileAcceptancePolicy,
        run: &'a ProfileAcceptancePolicy,
        running: &'a BackendImplementationRevision,
    ) -> AuthenticationContext<'a> {
        AuthenticationContext {
            owner,
            run,
            running,
            planner_version: 1,
            now_ms: trust_fixtures::NOW,
        }
    }

    /// A stocked store selects its profile under policies that accept it.
    #[test]
    fn a_stocked_store_selects_under_accepting_policies() {
        let (store, running) = stocked();
        let policy = trust_fixtures::policy_for(&store);
        let selected = store
            .select(&context(&policy, &policy, &running))
            .expect("selects");
        assert_eq!(selected.profile().backend_class, BackendClass::Vulkan);
        assert_eq!(
            selected.digest(),
            selected.profile().profile_digest().expect("digest")
        );
    }

    /// **No authority is baked in.** A store stocked with a perfectly good profile selects nothing
    /// when neither side names an authority — because nothing has vouched, and a store cannot vouch
    /// on its own behalf.
    ///
    /// This is the property that lets the seam be built while who-may-sign is still undecided: the
    /// answer arrives as data, and until it does the honest verdict is that nothing is admissible.
    #[test]
    fn a_store_with_no_named_authority_selects_nothing() {
        let (store, running) = stocked();
        let mut nameless = trust_fixtures::policy_for(&store);
        nameless.accepted_authorities.clear();

        let refusal = store
            .select(&context(&nameless, &nameless, &running))
            .expect_err("nothing has vouched, so nothing is admissible");
        let SelectionRefusal::AllCandidatesRefused { refusals, .. } = refusal else {
            panic!("expected a per-candidate refusal, got {refusal}");
        };
        assert_eq!(refusals.len(), 1, "the one candidate was considered");
        assert!(
            matches!(refusals[0].1, AuthenticationRefusal::NoAuthorityNamed),
            "refused for want of an authority, not for some incidental reason: {:?}",
            refusals[0].1
        );
    }

    /// A lane the store holds nothing for is told so, with what it does hold — an operator needs to
    /// tell an empty store from one stocked for the wrong lane.
    #[test]
    fn a_lane_with_no_candidate_is_refused_with_the_count_held() {
        let (store, _) = stocked();
        let other_lane = fixtures::revision(BackendClass::Cuda);
        let policy = trust_fixtures::policy_for(&store);

        let refusal = store
            .select(&context(&policy, &policy, &other_lane))
            .expect_err("nothing prices this lane");
        assert!(
            matches!(
                refusal,
                SelectionRefusal::NoCandidate {
                    class: "cuda",
                    held: 1,
                    ..
                }
            ),
            "{refusal}"
        );
    }

    /// **Ambiguity is refused rather than resolved by iteration order.**
    ///
    /// Two profiles for the same lane and revision that both authenticate are not interchangeable:
    /// their cost terms differ, so which one is composed against changes the physical claim and
    /// therefore what the node reserved. Choosing silently would leave a run's evidence naming a
    /// profile whose selection had no reason behind it.
    #[test]
    fn two_authenticating_profiles_are_refused_rather_than_chosen_between() {
        let (mut store, running) = stocked();
        // A second profile for the same lane and revision, differing in a cost figure — so it is a
        // genuinely different price for the same hardware, not a duplicate.
        let mut second = trust_fixtures::profile_for(&running);
        second.allocation_ceilings.reported_bytes += 4096;
        let envelope = trust_fixtures::envelope_for(&second, &running);
        store.insert(second, envelope).expect("stocks the second");
        assert_eq!(store.len(), 2);

        let policy = trust_fixtures::policy_for(&store);
        let refusal = store
            .select(&context(&policy, &policy, &running))
            .expect_err("two authenticating profiles must not be silently disambiguated");
        let SelectionRefusal::Ambiguous { digests, .. } = refusal else {
            panic!("expected an ambiguity refusal, got {refusal}");
        };
        assert_eq!(digests.len(), 2);
    }

    /// An envelope that vouches for other bytes cannot be filed, because the store would then hold a
    /// vouching for one profile under another's name.
    #[test]
    fn an_envelope_binding_other_bytes_cannot_be_stored() {
        let running = fixtures::revision(BackendClass::Vulkan);
        let profile = trust_fixtures::profile_for(&running);
        let mut envelope = trust_fixtures::envelope_for(&profile, &running);
        envelope.profile_digest = Hash([0xEE; 32]);

        let refusal = ProfileStore::new()
            .insert(profile, envelope)
            .expect_err("a mis-bound envelope must not be stored");
        assert!(matches!(refusal, StoreRefusal::BindingMismatch { .. }));
    }

    /// Storing the same profile twice is idempotent — re-stocking a node is not an error.
    #[test]
    fn storing_the_same_profile_twice_is_idempotent() {
        let (mut store, running) = stocked();
        let profile = trust_fixtures::profile_for(&running);
        let envelope = trust_fixtures::envelope_for(&profile, &running);
        store.insert(profile, envelope).expect("re-stocks");
        assert_eq!(store.len(), 1);
    }
}
