// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **profile trust envelope** and admission-time authentication
//! (`docs/specs/vhc-architecture-spec.md` §9.7 `[PC-12]`).
//!
//! Content addressing proves **identity**, not **authority**. A digest says which bytes these are;
//! it does not say those bytes may be trusted to price a machine. A profile that understates
//! costs — through malice or through an honest error nobody caught — produces claims that admit
//! machines which then fail, and the failure arrives wearing a certified claim's provenance. So a
//! profile is authenticated before it is composed with, and a profile that is merely *present and
//! well-formed* has established nothing.
//!
//! ## The effective policy is the intersection
//!
//! A profile is acceptable only if **both** owner policy and run policy accept its authority and
//! bindings. Neither may broaden the other: the owner may always refuse use of its machine, and the
//! run may always refuse evidence outside its certification policy. A refusal identifies whether
//! owner policy, run policy, or both rejected it — because "the profile was refused" sends nobody
//! to the right conversation.
//!
//! ## Revision ranges name which numbering they constrain
//!
//! A driver can carry several revision numberings that do not order against each other, and one
//! backend supplies no driver revision at all. A range is therefore an explicit permitted **set**
//! against a **named** signal, not an interval: inventing a total order over version strings would
//! silently accept revisions nobody certified. Where the framework supplies no driver revision, the
//! OS build is the documented fallback signal, and a profile that constrains nothing comparable is
//! refused rather than treated as matching everything.

use std::collections::BTreeSet;

use daemon_vhc_proto::{Hash, PeerId};
use serde::{Deserialize, Serialize};

use crate::profile::{BackendExecutionProfile, ProfileError};
use crate::revision::{
    BackendImplementationRevision, OsFamily, RevisionSignal, SealedBinaryIdentity,
};

/// Which revision numbering a permitted set constrains.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RevisionNumbering {
    /// The driver's own version text.
    DriverVersion,
    /// The vendor's release numbering, which on at least one platform differs from the
    /// display-driver numbering for the same driver.
    VendorRelease,
    /// The OS build — the only implementation-revision signal that exists on a backend whose
    /// framework supplies no driver revision.
    OsBuild,
}

/// The permitted revisions for one numbering: an explicit set, against a named signal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevisionRange {
    /// Which numbering this constrains.
    pub numbering: RevisionNumbering,
    /// The exact permitted values. Explicit rather than an interval, because these strings have no
    /// reliable total order.
    pub permitted: BTreeSet<String>,
    /// For [`RevisionNumbering::OsBuild`], the OS family the permitted builds belong to. A build
    /// number means nothing without the family it numbers.
    pub os_family: Option<OsFamily>,
}

/// When a profile is valid, and how revocation is checked.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidityPolicy {
    /// Not valid before this instant, in milliseconds.
    pub not_before_ms: u64,
    /// Not valid after this instant, in milliseconds. A profile with no expiry is a profile nobody
    /// ever has to re-examine.
    pub not_after_ms: u64,
    /// The revocation list this profile is checked against.
    pub revocation_list_digest: Hash,
}

/// Who stands behind a profile.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileAuthority {
    /// The signing identity.
    pub signer: PeerId,
    /// The release authority's name, for a human reading a refusal.
    pub release_authority: String,
}

/// The trust envelope a profile carries, or is cryptographically bound to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileTrustEnvelope {
    /// The profile schema version this envelope binds.
    pub profile_schema: u32,
    /// The planner versions this profile is priced for. A profile priced for one composition
    /// algorithm is not valid under another.
    pub compatible_planner_versions: BTreeSet<u32>,
    /// The sealed backend binary this profile describes.
    pub sealed_binary: SealedBinaryIdentity,
    /// The digest of the conformance evidence that certified the profile.
    pub certification_evidence_digest: Hash,
    /// Who stands behind it.
    pub authority: ProfileAuthority,
    /// When it is valid, and how revocation is checked.
    pub validity: ValidityPolicy,
    /// The exact backend implementation revision it is valid for.
    pub implementation_revision: String,
    /// The permitted driver/API ranges, one per constrained numbering.
    pub permitted_revision_ranges: Vec<RevisionRange>,
    /// The digest of the profile bytes this envelope binds.
    pub profile_digest: Hash,
}

/// One side's acceptance policy. The effective policy is the intersection of the owner's and the
/// run's.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProfileAcceptancePolicy {
    /// The authorities this side accepts. Empty means this side names no authority of its own and
    /// defers to the other — it never means "any authority", which is why the intersection requires
    /// at least one side to name one.
    pub accepted_authorities: BTreeSet<PeerId>,
    /// The lowest profile schema this side accepts.
    pub min_profile_schema: u32,
    /// The planner versions this side accepts. Empty defers to the other side.
    pub accepted_planner_versions: BTreeSet<u32>,
    /// Whether this side requires a conformance-evidence digest.
    pub require_conformance_evidence: bool,
    /// Whether this side requires the per-allocation ceiling to have been **measured** rather than
    /// merely reported.
    pub require_measured_allocation_ceiling: bool,
    /// Whether this side requires every cost term to rest on *something* — no term with an absent
    /// calibration basis.
    pub require_full_calibration: bool,
    /// Profile digests this side has revoked.
    pub revoked_profiles: BTreeSet<Hash>,
}

/// Which policy refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicySide {
    /// The owner's policy — the machine's operator refusing use of their machine.
    Owner,
    /// The run's policy — the run refusing evidence outside its certification policy.
    Run,
    /// Both refused independently.
    Both,
}

impl std::fmt::Display for PolicySide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Owner => "owner policy",
            Self::Run => "run policy",
            Self::Both => "both owner and run policy",
        })
    }
}

/// Why a profile was not authenticated. Every variant is a typed refusal — never a fallback to an
/// unauthenticated profile, an older one, or an estimate.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AuthenticationRefusal {
    /// The envelope does not bind the profile bytes presented.
    #[error(
        "the trust envelope binds a different profile than the bytes presented; content addressing \
         is the one thing that must line up before anything else is considered"
    )]
    BindingMismatch,
    /// The envelope's own members disagree with the profile's.
    #[error("the trust envelope disagrees with the profile it binds: {0}")]
    EnvelopeInconsistent(String),
    /// No policy accepts the signer.
    #[error("{side} does not accept profile authority `{authority}`")]
    AuthorityRejected {
        /// Which policy refused.
        side: PolicySide,
        /// The release authority named in the envelope.
        authority: String,
    },
    /// Neither side named an accepted authority, so nothing has actually vouched for the profile.
    #[error(
        "neither owner nor run policy names an accepted profile authority; an empty intersection \
         accepts nothing, and defaulting to acceptance is how an unvouched profile prices a machine"
    )]
    NoAuthorityNamed,
    /// The schema is below what a policy accepts.
    #[error("{side} requires profile schema >= {required}, but the profile is schema {actual}")]
    SchemaTooOld {
        /// Which policy refused.
        side: PolicySide,
        /// The minimum it requires.
        required: u32,
        /// What the profile carries.
        actual: u32,
    },
    /// The planner about to run is not one the profile names.
    #[error(
        "the planner about to compose is version {planner}, which the profile is not priced for \
         (it names {named:?})"
    )]
    PlannerIncompatible {
        /// The planner that would run.
        planner: u32,
        /// The versions the profile names.
        named: BTreeSet<u32>,
    },
    /// A policy does not accept the planner version.
    #[error("{side} does not accept planner version {planner}")]
    PlannerRejected {
        /// Which policy refused.
        side: PolicySide,
        /// The planner version.
        planner: u32,
    },
    /// The profile is outside its validity window.
    #[error("the profile is not valid at {now_ms} ms (valid {not_before_ms}..={not_after_ms})")]
    OutsideValidity {
        /// The instant checked.
        now_ms: u64,
        /// Window start.
        not_before_ms: u64,
        /// Window end.
        not_after_ms: u64,
    },
    /// The profile has been revoked.
    #[error("{side} has revoked this profile")]
    Revoked {
        /// Which policy revoked it.
        side: PolicySide,
    },
    /// A policy requires conformance evidence the envelope does not carry.
    #[error("{side} requires a conformance-evidence digest, which this profile does not carry")]
    ConformanceEvidenceMissing {
        /// Which policy refused.
        side: PolicySide,
    },
    /// A policy requires a measured per-allocation ceiling.
    #[error("{side} requires a measured per-allocation ceiling: {detail}")]
    CeilingNotMeasured {
        /// Which policy refused.
        side: PolicySide,
        /// Why the ceiling is not certified.
        detail: String,
    },
    /// The profile carries figures measured **from allocator statistics** that this binary cannot
    /// reproduce. Directly-probed figures are unaffected.
    #[error(
        "the profile carries {measured} figure(s) measured from allocator statistics, but this \
         binary cannot report them; that evidence is unreproducible here, so the profile's own \
         certification is unverifiable on this binary"
    )]
    CalibrationUnreproducible {
        /// How many terms claim measurement.
        measured: usize,
    },
    /// A policy requires every term to rest on something.
    #[error("{side} requires full calibration, but {absent} term(s) have no calibration basis")]
    CalibrationIncomplete {
        /// Which policy refused.
        side: PolicySide,
        /// How many terms are uncalibrated.
        absent: usize,
    },
    /// The running implementation revision is not one the profile covers.
    #[error(
        "the running backend implementation revision `{running}` is not covered by this profile \
         (valid for `{profile}`); role execution is prevented until revalidation succeeds"
    )]
    ImplementationRevisionUnmatched {
        /// What the binary reports.
        running: String,
        /// What the profile names.
        profile: String,
    },
    /// The running driver/API revision is outside every permitted range.
    #[error(
        "the running {numbering:?} revision `{running}` is outside the profile's permitted set"
    )]
    DriverRevisionUnmatched {
        /// Which numbering was compared.
        numbering: RevisionNumbering,
        /// The running value.
        running: String,
    },
    /// The platform supplied nothing comparable, so no range can be evaluated.
    #[error(
        "the platform supplied no comparable implementation-revision signal — no driver revision \
         and no OS build — so the profile's permitted ranges cannot be evaluated; this is a typed \
         refusal rather than an assumed match, because a range evaluated against nothing admits \
         everything"
    )]
    NoComparableRevisionSignal,
    /// The profile prices a different backend class than the lane it is being authenticated against.
    #[error(
        "the profile prices the {profile} backend class but the running lane is {running} — a \
         profile for one backend cannot be authenticated against another, and admitting one would \
         be the silent-device-fallback failure arriving through the authentication path"
    )]
    BackendClassMismatch {
        /// The class the profile prices.
        profile: &'static str,
        /// The class the running lane serves.
        running: &'static str,
    },
    /// The sealed binary the envelope describes is not the one running.
    #[error("the trust envelope describes a different sealed binary than the one running")]
    SealedBinaryMismatch,
    /// The profile itself does not validate.
    #[error(transparent)]
    Profile(#[from] ProfileError),
}

/// The effective acceptance decision: both sides must accept, and neither may broaden the other.
///
/// Every check below is evaluated against the **intersection**. Where a check is one-sided (a
/// validity window, a binding) it is evaluated once and attributed to neither policy, because
/// neither policy chose it.
pub fn authenticate(
    profile: &BackendExecutionProfile,
    envelope: &ProfileTrustEnvelope,
    owner: &ProfileAcceptancePolicy,
    run: &ProfileAcceptancePolicy,
    running: &BackendImplementationRevision,
    planner_version: u32,
    now_ms: u64,
) -> Result<(), AuthenticationRefusal> {
    profile.validate()?;

    // 0. The lane, before anything else. A profile prices one backend class and is meaningless
    //    against another: its cost terms describe that backend's allocator, its pooling, its
    //    staging path.
    //
    //    This is deliberately a direct class comparison rather than a consequence of the identity
    //    and driver checks below, because relying on those is one mistake away from the bug that
    //    prompted it. This box's CPU-lane record carried the Vulkan adapter's identity and its RADV
    //    driver strings — filled from a shared graphics probe regardless of lane — so a Vulkan
    //    profile naming that driver range matched a CPU lane while the record plainly said `Cpu`
    //    beside it. The record is fixed at its source; this refuses the class of mistake, not the
    //    instance, so a future record that mis-attributes an adapter cannot cash it in here.
    if profile.backend_class != running.backend_class {
        return Err(AuthenticationRefusal::BackendClassMismatch {
            profile: profile.backend_class.slug(),
            running: running.backend_class.slug(),
        });
    }

    // 1. Identity first: the envelope must bind these bytes.
    if envelope.profile_digest != profile.profile_digest()? {
        return Err(AuthenticationRefusal::BindingMismatch);
    }
    if envelope.profile_schema != profile.schema {
        return Err(AuthenticationRefusal::EnvelopeInconsistent(format!(
            "envelope binds schema {} but the profile is schema {}",
            envelope.profile_schema, profile.schema
        )));
    }
    if envelope.certification_evidence_digest != profile.conformance_evidence_digest {
        return Err(AuthenticationRefusal::EnvelopeInconsistent(
            "envelope and profile name different conformance evidence".into(),
        ));
    }
    if envelope.implementation_revision != profile.implementation_revision {
        return Err(AuthenticationRefusal::EnvelopeInconsistent(
            "envelope and profile name different backend implementation revisions".into(),
        ));
    }
    if envelope.sealed_binary != running.sealed_binary {
        return Err(AuthenticationRefusal::SealedBinaryMismatch);
    }

    // 2. Authority, as an intersection. An empty set on one side defers; both empty accepts
    //    nothing, because nothing has vouched.
    let signer = &envelope.authority.signer;
    let owner_names = !owner.accepted_authorities.is_empty();
    let run_names = !run.accepted_authorities.is_empty();
    if !owner_names && !run_names {
        return Err(AuthenticationRefusal::NoAuthorityNamed);
    }
    let owner_ok = !owner_names || owner.accepted_authorities.contains(signer);
    let run_ok = !run_names || run.accepted_authorities.contains(signer);
    if !owner_ok || !run_ok {
        return Err(AuthenticationRefusal::AuthorityRejected {
            side: side_of(owner_ok, run_ok),
            authority: envelope.authority.release_authority.clone(),
        });
    }

    // 3. Schema floor, per side.
    let owner_schema_ok = profile.schema >= owner.min_profile_schema;
    let run_schema_ok = profile.schema >= run.min_profile_schema;
    if !owner_schema_ok || !run_schema_ok {
        return Err(AuthenticationRefusal::SchemaTooOld {
            side: side_of(owner_schema_ok, run_schema_ok),
            required: owner.min_profile_schema.max(run.min_profile_schema),
            actual: profile.schema,
        });
    }

    // 4. The planner about to run must be one the profile is priced for, and one both sides accept.
    if !envelope
        .compatible_planner_versions
        .contains(&planner_version)
    {
        return Err(AuthenticationRefusal::PlannerIncompatible {
            planner: planner_version,
            named: envelope.compatible_planner_versions.clone(),
        });
    }
    let owner_planner_ok = owner.accepted_planner_versions.is_empty()
        || owner.accepted_planner_versions.contains(&planner_version);
    let run_planner_ok = run.accepted_planner_versions.is_empty()
        || run.accepted_planner_versions.contains(&planner_version);
    if !owner_planner_ok || !run_planner_ok {
        return Err(AuthenticationRefusal::PlannerRejected {
            side: side_of(owner_planner_ok, run_planner_ok),
            planner: planner_version,
        });
    }

    // 5. Validity and revocation.
    if now_ms < envelope.validity.not_before_ms || now_ms > envelope.validity.not_after_ms {
        return Err(AuthenticationRefusal::OutsideValidity {
            now_ms,
            not_before_ms: envelope.validity.not_before_ms,
            not_after_ms: envelope.validity.not_after_ms,
        });
    }
    let owner_revoked = owner.revoked_profiles.contains(&envelope.profile_digest);
    let run_revoked = run.revoked_profiles.contains(&envelope.profile_digest);
    if owner_revoked || run_revoked {
        return Err(AuthenticationRefusal::Revoked {
            side: side_of(!owner_revoked, !run_revoked),
        });
    }

    // 6. Evidence requirements, per side.
    if envelope.certification_evidence_digest == Hash([0u8; 32]) {
        let owner_needs = owner.require_conformance_evidence;
        let run_needs = run.require_conformance_evidence;
        if owner_needs || run_needs {
            return Err(AuthenticationRefusal::ConformanceEvidenceMissing {
                side: side_of(!owner_needs, !run_needs),
            });
        }
    }
    if owner.require_measured_allocation_ceiling || run.require_measured_allocation_ceiling {
        if let Err(err) = profile.allocation_ceilings.effective_bytes() {
            return Err(AuthenticationRefusal::CeilingNotMeasured {
                side: side_of(
                    !owner.require_measured_allocation_ceiling,
                    !run.require_measured_allocation_ceiling,
                ),
                detail: err.to_string(),
            });
        }
    }

    // 6b. Calibration. A profile whose pooling or workspace terms were calibrated FROM allocator
    //     statistics must not be accepted by a binary that cannot produce them: the evidence behind
    //     those figures cannot be reproduced there, so the profile's certification is unverifiable
    //     on the very binary about to compose with it. This is the teeth on the
    //     `statistics_available` member, which would otherwise be a declaration nothing reads.
    //     The check keys on the *source* of the measurement, not on "was it measured": a directly
    //     probed figure is reproducible on any binary, and refusing it would make recording a
    //     truthful measurement the expensive choice while downgrading it to a weaker basis cleared
    //     the refusal — an incentive to mislabel, which is precisely what the per-term basis exists
    //     to prevent.
    if profile.requires_allocator_statistics() && !running.allocator.statistics_available {
        return Err(AuthenticationRefusal::CalibrationUnreproducible {
            measured: profile.statistics_dependent_terms().count()
                + usize::from(
                    profile
                        .headroom
                        .hidden_overhead_basis
                        .requires_allocator_statistics(),
                ),
        });
    }
    let calibration = profile.calibration_summary();
    if !calibration.is_complete() {
        let owner_needs = owner.require_full_calibration;
        let run_needs = run.require_full_calibration;
        if owner_needs || run_needs {
            return Err(AuthenticationRefusal::CalibrationIncomplete {
                side: side_of(!owner_needs, !run_needs),
                absent: calibration.absent,
            });
        }
    }

    // 7. Revision binding. Absolute: an unmatched revision prevents role execution until
    //    revalidation succeeds.
    if running.backend_implementation.revision != envelope.implementation_revision {
        return Err(AuthenticationRefusal::ImplementationRevisionUnmatched {
            running: running.backend_implementation.revision.clone(),
            profile: envelope.implementation_revision.clone(),
        });
    }
    check_revision_ranges(envelope, running)?;

    Ok(())
}

fn side_of(owner_ok: bool, run_ok: bool) -> PolicySide {
    match (owner_ok, run_ok) {
        (false, false) => PolicySide::Both,
        (false, true) => PolicySide::Owner,
        _ => PolicySide::Run,
    }
}

/// Compare the running driver/API signal against the profile's permitted ranges.
fn check_revision_ranges(
    envelope: &ProfileTrustEnvelope,
    running: &BackendImplementationRevision,
) -> Result<(), AuthenticationRefusal> {
    let signal = running.driver_api.revision_signal();
    let (numbering, value, family) = match &signal {
        RevisionSignal::DriverVersion(v) => (RevisionNumbering::DriverVersion, v, None),
        RevisionSignal::VendorRelease(v) => (RevisionNumbering::VendorRelease, v, None),
        RevisionSignal::OsBuild {
            family,
            build,
            version: _,
        } => (RevisionNumbering::OsBuild, build, Some(*family)),
        RevisionSignal::None => return Err(AuthenticationRefusal::NoComparableRevisionSignal),
    };

    let applicable: Vec<&RevisionRange> = envelope
        .permitted_revision_ranges
        .iter()
        .filter(|r| r.numbering == numbering && (r.os_family.is_none() || r.os_family == family))
        .collect();

    // A profile that constrains nothing about the signal the platform actually supplies has not
    // constrained the platform at all.
    if applicable.is_empty() {
        return Err(AuthenticationRefusal::DriverRevisionUnmatched {
            numbering,
            running: value.clone(),
        });
    }
    if applicable.iter().any(|r| r.permitted.contains(value)) {
        Ok(())
    } else {
        Err(AuthenticationRefusal::DriverRevisionUnmatched {
            numbering,
            running: value.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::fixtures::profile;
    use crate::revision::{BackendClass, DriverRevision, Maybe, ProducedBy, Unavailable};

    const NOW: u64 = 1_000;

    fn peer(n: u8) -> PeerId {
        PeerId([n; 32])
    }

    fn revision() -> BackendImplementationRevision {
        let mut rev = crate::revision::fixtures::revision(BackendClass::Vulkan);
        rev.sealed_binary = SealedBinaryIdentity {
            blake3: [1u8; 32],
            size_bytes: 44_323_328,
        };
        rev.produced_by = ProducedBy::WorkerProbePath;
        rev
    }

    fn envelope(p: &BackendExecutionProfile) -> ProfileTrustEnvelope {
        ProfileTrustEnvelope {
            profile_schema: p.schema,
            compatible_planner_versions: [1].into_iter().collect(),
            sealed_binary: SealedBinaryIdentity {
                blake3: [1u8; 32],
                size_bytes: 44_323_328,
            },
            certification_evidence_digest: p.conformance_evidence_digest,
            authority: ProfileAuthority {
                signer: peer(7),
                release_authority: "the release authority".into(),
            },
            validity: ValidityPolicy {
                not_before_ms: 0,
                not_after_ms: 10_000,
                revocation_list_digest: Hash([2u8; 32]),
            },
            implementation_revision: p.implementation_revision.clone(),
            permitted_revision_ranges: vec![RevisionRange {
                numbering: RevisionNumbering::DriverVersion,
                permitted: ["25.1.0".to_string()].into_iter().collect(),
                os_family: None,
            }],
            profile_digest: p.profile_digest().unwrap(),
        }
    }

    fn policy() -> ProfileAcceptancePolicy {
        ProfileAcceptancePolicy {
            accepted_authorities: [peer(7)].into_iter().collect(),
            min_profile_schema: 1,
            accepted_planner_versions: [1].into_iter().collect(),
            require_conformance_evidence: true,
            require_measured_allocation_ceiling: true,
            require_full_calibration: true,
            revoked_profiles: BTreeSet::new(),
        }
    }

    fn authenticate_with(
        owner: &ProfileAcceptancePolicy,
        run: &ProfileAcceptancePolicy,
    ) -> Result<(), AuthenticationRefusal> {
        let p = profile(BackendClass::Vulkan);
        let e = envelope(&p);
        authenticate(&p, &e, owner, run, &revision(), 1, NOW)
    }

    /// **MEAS-F8.** A device profile cannot authenticate against a CPU lane.
    ///
    /// The reachable version of this was not hypothetical. This box's CPU-lane record carried the
    /// *Vulkan* adapter's vendor and device ids and its RADV driver strings — filled from a shared
    /// graphics probe regardless of which lane the record described — while correctly reporting a CPU
    /// backend class and a software rasterizer beside them. A Vulkan profile naming a `Mesa 25.2.6`
    /// driver range therefore matched the CPU-lane record's driver signal exactly. That is the
    /// silent-device-fallback failure arriving *through* the authentication path rather than around
    /// it: a device role admitted against a CPU lane by a check that believed it had verified the
    /// driver.
    ///
    /// Two things had to change and this asserts the outcome of both. The record now carries only the
    /// identity of the adapter that serves its own lane, so a CPU-lane record has no device driver to
    /// compare against. And the class is compared directly, so the refusal does not *depend* on the
    /// record being right — a future record that mis-attributes an adapter still cannot cash it in.
    #[test]
    fn a_device_profile_cannot_authenticate_against_a_cpu_lane() {
        let device_profile = profile(BackendClass::Vulkan);
        let envelope = envelope(&device_profile);

        // A CPU-lane record that, in the shape of the defect, still carried a device driver string.
        let mut cpu_lane = revision();
        cpu_lane.backend_class = BackendClass::Cpu;
        cpu_lane.driver_api.driver.version_text = Maybe::Available("Mesa 25.2.6".into());

        let refusal = authenticate(
            &device_profile,
            &envelope,
            &policy(),
            &policy(),
            &cpu_lane,
            1,
            NOW,
        )
        .expect_err("a Vulkan profile must not authenticate against a CPU lane");

        assert!(
            matches!(
                refusal,
                AuthenticationRefusal::BackendClassMismatch {
                    profile: "vulkan",
                    running: "cpu"
                }
            ),
            "the refusal names both classes rather than failing later for an incidental reason: \
             {refusal}"
        );
        // And it is refused on the class, BEFORE the driver range is ever consulted — so it does not
        // rely on the driver string being absent, which is the property that survives a bad record.
        assert!(
            !format!("{refusal}").contains("Mesa"),
            "the driver was never compared: {refusal}"
        );
    }

    /// The same lane pairing in the other direction, so the check is not just refusing everything.
    #[test]
    fn a_profile_authenticates_against_its_own_lane() {
        let cpu_profile = profile(BackendClass::Cpu);
        let e = envelope(&cpu_profile);
        let mut cpu_lane = revision();
        cpu_lane.backend_class = BackendClass::Cpu;
        // The CPU lane's real revision signal is the OS build, not a vendor driver — there is no
        // vendor driver on this lane, which is exactly why its absence is typed as inapplicable.
        cpu_lane.driver_api.driver.version_text =
            Maybe::Unavailable(crate::revision::Unavailable::NotApplicableToLane);
        cpu_lane.driver_api.driver.vendor_release =
            Maybe::Unavailable(crate::revision::Unavailable::NotApplicableToLane);
        cpu_lane.driver_api.os.build = Maybe::Available("6.19.7".into());
        cpu_lane.driver_api.os.version = Maybe::Available("25.11".into());

        // Whatever else this fixture's ranges say, the class must not be what refuses it.
        assert!(
            !matches!(
                authenticate(&cpu_profile, &e, &policy(), &policy(), &cpu_lane, 1, NOW),
                Err(AuthenticationRefusal::BackendClassMismatch { .. })
            ),
            "a CPU profile must not be refused on the class by a CPU lane"
        );
    }
    #[test]
    fn a_bound_signed_current_profile_authenticates() {
        authenticate_with(&policy(), &policy()).expect("authenticates");
    }

    /// Content addressing is checked first and on its own: the envelope must bind these bytes
    /// before any policy question is worth asking.
    #[test]
    fn an_envelope_binding_other_bytes_is_refused() {
        let p = profile(BackendClass::Vulkan);
        let mut e = envelope(&p);
        e.profile_digest = Hash([0xEE; 32]);
        assert_eq!(
            authenticate(&p, &e, &policy(), &policy(), &revision(), 1, NOW),
            Err(AuthenticationRefusal::BindingMismatch)
        );
    }

    /// The effective policy is the intersection, and the refusal names which side rejected.
    #[test]
    fn acceptance_is_the_intersection_and_the_refusal_names_the_rejecting_policy() {
        let mut owner = policy();
        owner.accepted_authorities = [peer(9)].into_iter().collect();
        let err = authenticate_with(&owner, &policy()).unwrap_err();
        assert!(matches!(
            err,
            AuthenticationRefusal::AuthorityRejected {
                side: PolicySide::Owner,
                ..
            }
        ));
        assert!(err.to_string().contains("owner policy"));

        let mut run = policy();
        run.accepted_authorities = [peer(9)].into_iter().collect();
        let err = authenticate_with(&policy(), &run).unwrap_err();
        assert!(matches!(
            err,
            AuthenticationRefusal::AuthorityRejected {
                side: PolicySide::Run,
                ..
            }
        ));

        // Both refusing independently is reported as both.
        let mut both = policy();
        both.accepted_authorities = [peer(9)].into_iter().collect();
        let err = authenticate_with(&both, &both).unwrap_err();
        assert!(err.to_string().contains("both owner and run policy"));
    }

    /// Neither side naming an authority accepts nothing. Defaulting to acceptance is how an
    /// unvouched profile ends up pricing a machine.
    #[test]
    fn an_empty_authority_intersection_accepts_nothing() {
        let mut owner = policy();
        owner.accepted_authorities.clear();
        let mut run = policy();
        run.accepted_authorities.clear();
        assert_eq!(
            authenticate_with(&owner, &run),
            Err(AuthenticationRefusal::NoAuthorityNamed)
        );
    }

    /// One side may defer to the other, and deferring is not broadening.
    #[test]
    fn one_side_may_defer_to_the_other() {
        let mut owner = policy();
        owner.accepted_authorities.clear();
        owner.accepted_planner_versions.clear();
        authenticate_with(&owner, &policy()).expect("the run's authority governs");
    }

    #[test]
    fn a_planner_the_profile_is_not_priced_for_cannot_compose_with_it() {
        let p = profile(BackendClass::Vulkan);
        let e = envelope(&p);
        let err = authenticate(&p, &e, &policy(), &policy(), &revision(), 2, NOW).unwrap_err();
        assert!(matches!(
            err,
            AuthenticationRefusal::PlannerIncompatible { planner: 2, .. }
        ));
    }

    #[test]
    fn an_expired_or_not_yet_valid_profile_is_refused() {
        let p = profile(BackendClass::Vulkan);
        let e = envelope(&p);
        assert!(matches!(
            authenticate(&p, &e, &policy(), &policy(), &revision(), 1, 20_000).unwrap_err(),
            AuthenticationRefusal::OutsideValidity { .. }
        ));
        let mut early = e.clone();
        early.validity.not_before_ms = 5_000;
        assert!(matches!(
            authenticate(&p, &early, &policy(), &policy(), &revision(), 1, NOW).unwrap_err(),
            AuthenticationRefusal::OutsideValidity { .. }
        ));
    }

    #[test]
    fn either_side_may_revoke_and_the_refusal_names_it() {
        let p = profile(BackendClass::Vulkan);
        let e = envelope(&p);
        let mut owner = policy();
        owner.revoked_profiles.insert(e.profile_digest);
        assert_eq!(
            authenticate(&p, &e, &owner, &policy(), &revision(), 1, NOW),
            Err(AuthenticationRefusal::Revoked {
                side: PolicySide::Owner
            })
        );
    }

    /// An unmatched implementation revision prevents execution. Recording an obligation alone never
    /// permits it.
    #[test]
    fn an_unmatched_implementation_revision_prevents_execution() {
        let p = profile(BackendClass::Vulkan);
        let e = envelope(&p);
        let mut running = revision();
        running.backend_implementation.revision = "0.11.0".into();
        assert!(matches!(
            authenticate(&p, &e, &policy(), &policy(), &running, 1, NOW).unwrap_err(),
            AuthenticationRefusal::ImplementationRevisionUnmatched { .. }
        ));
    }

    #[test]
    fn a_driver_revision_outside_the_permitted_set_is_refused() {
        let p = profile(BackendClass::Vulkan);
        let e = envelope(&p);
        let mut running = revision();
        running.driver_api.driver.version_text = Maybe::Available("26.0.0".into());
        assert!(matches!(
            authenticate(&p, &e, &policy(), &policy(), &running, 1, NOW).unwrap_err(),
            AuthenticationRefusal::DriverRevisionUnmatched {
                numbering: RevisionNumbering::DriverVersion,
                ..
            }
        ));
    }

    /// A backend that supplies no driver revision is constrained on its OS build instead — the
    /// documented fallback — and the family is part of the comparison.
    #[test]
    fn a_backend_with_no_driver_revision_is_constrained_on_its_os_build() {
        let p = profile(BackendClass::Metal);
        let mut e = envelope(&p);
        e.permitted_revision_ranges = vec![RevisionRange {
            numbering: RevisionNumbering::OsBuild,
            permitted: ["6.19.7".to_string()].into_iter().collect(),
            os_family: Some(OsFamily::Linux),
        }];
        // The record is a Metal lane, matching the profile: a profile prices one backend class and
        // authenticating it against another is refused on the class before any range is consulted.
        let mut running = revision();
        running.backend_class = BackendClass::Metal;
        running.driver_api.driver = DriverRevision::default();
        authenticate(&p, &e, &policy(), &policy(), &running, 1, NOW)
            .expect("the OS build is the fallback signal");

        // A permitted set for the wrong family does not apply.
        e.permitted_revision_ranges[0].os_family = Some(OsFamily::Macos);
        assert!(matches!(
            authenticate(&p, &e, &policy(), &policy(), &running, 1, NOW).unwrap_err(),
            AuthenticationRefusal::DriverRevisionUnmatched { .. }
        ));
    }

    /// With nothing comparable supplied, no range can be evaluated — and a range evaluated against
    /// nothing would admit everything.
    #[test]
    fn no_comparable_signal_is_a_typed_refusal() {
        let p = profile(BackendClass::Metal);
        let e = envelope(&p);
        let mut running = revision();
        running.backend_class = BackendClass::Metal;
        running.driver_api.driver = DriverRevision::default();
        running.driver_api.os.build = Maybe::Unavailable(Unavailable::NotExposedByPlatform);
        assert_eq!(
            authenticate(&p, &e, &policy(), &policy(), &running, 1, NOW),
            Err(AuthenticationRefusal::NoComparableRevisionSignal)
        );
    }

    /// A policy may require the per-allocation ceiling to have been measured, which the reported
    /// figure alone cannot satisfy.
    #[test]
    fn a_policy_may_require_a_measured_allocation_ceiling() {
        let mut p = profile(BackendClass::Dx12);
        p.allocation_ceilings.measured_bytes = Maybe::default();
        let e = envelope(&p);
        let mut running = revision();
        running.backend_class = BackendClass::Dx12;
        let err = authenticate(&p, &e, &policy(), &policy(), &running, 1, NOW).unwrap_err();
        assert!(matches!(
            err,
            AuthenticationRefusal::CeilingNotMeasured { .. }
        ));

        // A policy that does not require it composes on the reported figure at its own risk — and it
        // has to say so on BOTH axes, because an unmeasured ceiling now shows up in the calibration
        // census as uncalibrated. That coupling is deliberate: the two flags are two ways of asking
        // the same question, and a policy that waived one while still demanding full calibration was
        // asking for something it had already given up.
        let mut relaxed = policy();
        relaxed.require_measured_allocation_ceiling = false;
        relaxed.require_full_calibration = false;
        authenticate(&p, &e, &relaxed, &relaxed, &running, 1, NOW)
            .expect("a policy may accept an unmeasured ceiling explicitly");
    }

    /// The teeth on `statistics_available`: a profile whose figures were *measured* from allocator
    /// statistics must not be accepted by a binary that cannot produce them, because the evidence
    /// behind those figures is unreproducible there.
    #[test]
    fn a_statistics_derived_figure_is_refused_by_a_binary_without_statistics() {
        let mut p = profile(BackendClass::Vulkan);
        p.standing_terms[0].calibration_basis =
            crate::profile::CalibrationBasis::MeasuredFromAllocatorStatistics;
        let e = envelope(&p);

        let mut running = revision();
        running.allocator.statistics_available = false;
        let err = authenticate(&p, &e, &policy(), &policy(), &running, 1, NOW).unwrap_err();
        assert!(matches!(
            err,
            AuthenticationRefusal::CalibrationUnreproducible { measured: 1 }
        ));

        // The same profile on a binary that can report statistics authenticates.
        authenticate(&p, &e, &policy(), &policy(), &revision(), 1, NOW)
            .expect("statistics available, so the evidence is reproducible");

        // A profile needing no statistics is unaffected by the binary's capability.
        let plain = profile(BackendClass::Vulkan);
        assert!(!plain.requires_allocator_statistics());
        authenticate(
            &plain,
            &envelope(&plain),
            &policy(),
            &policy(),
            &running,
            1,
            NOW,
        )
        .expect("a profile needing no allocator statistics authenticates anywhere");
    }

    /// A **directly probed** measurement is reproducible anywhere, so it must not be refused. The
    /// alternative punished honest recording: a truthful profile became unusable, while downgrading
    /// the figure to a weaker basis cleared the refusal — an incentive to mislabel, which is the
    /// exact failure the per-term basis exists to prevent.
    #[test]
    fn a_directly_probed_measurement_is_not_refused_for_want_of_statistics() {
        let mut p = profile(BackendClass::Vulkan);
        p.standing_terms[0].calibration_basis =
            crate::profile::CalibrationBasis::MeasuredByDirectProbe;
        assert!(!p.requires_allocator_statistics());

        let mut running = revision();
        running.allocator.statistics_available = false;
        authenticate(&p, &envelope(&p), &policy(), &policy(), &running, 1, NOW)
            .expect("an honestly recorded direct probe authenticates on any binary");
    }

    /// The hidden-overhead reserve is not a `CostTerm`, so a check that only walked the terms would
    /// let a statistics-derived hidden figure through on a binary that cannot reproduce it.
    #[test]
    fn a_statistics_derived_hidden_overhead_figure_is_covered_by_the_same_teeth() {
        let mut p = profile(BackendClass::Vulkan);
        p.headroom.hidden_overhead_basis =
            crate::profile::CalibrationBasis::MeasuredFromAllocatorStatistics;
        assert!(p.requires_allocator_statistics());
        let e = envelope(&p);
        let mut running = revision();
        running.allocator.statistics_available = false;
        assert!(matches!(
            authenticate(&p, &e, &policy(), &policy(), &running, 1, NOW).unwrap_err(),
            AuthenticationRefusal::CalibrationUnreproducible { .. }
        ));
    }

    /// A policy may require every term to rest on something, and the refusal names which side asked.
    #[test]
    fn a_policy_may_require_full_calibration() {
        let mut p = profile(BackendClass::Vulkan);
        p.standing_terms[0].calibration_basis = crate::profile::CalibrationBasis::Absent;
        let e = envelope(&p);

        let err = authenticate(&p, &e, &policy(), &policy(), &revision(), 1, NOW).unwrap_err();
        assert!(matches!(
            err,
            AuthenticationRefusal::CalibrationIncomplete { absent: 1, .. }
        ));

        let mut relaxed = policy();
        relaxed.require_full_calibration = false;
        authenticate(&p, &e, &relaxed, &relaxed, &revision(), 1, NOW)
            .expect("a policy may accept an uncalibrated term explicitly");
    }

    #[test]
    fn the_sealed_binary_must_be_the_one_running() {
        let p = profile(BackendClass::Vulkan);
        let mut e = envelope(&p);
        e.sealed_binary.blake3 = [0xAB; 32];
        assert_eq!(
            authenticate(&p, &e, &policy(), &policy(), &revision(), 1, NOW),
            Err(AuthenticationRefusal::SealedBinaryMismatch)
        );
    }

    #[test]
    fn an_envelope_disagreeing_with_its_profile_is_refused() {
        let p = profile(BackendClass::Vulkan);
        let mut e = envelope(&p);
        e.implementation_revision = "9.9.9".into();
        assert!(matches!(
            authenticate(&p, &e, &policy(), &policy(), &revision(), 1, NOW).unwrap_err(),
            AuthenticationRefusal::EnvelopeInconsistent(_)
        ));
    }
}
