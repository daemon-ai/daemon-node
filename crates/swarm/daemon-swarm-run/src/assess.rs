// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Staged run assessment (spec §6.5, ABI §6.2; TDD RUN-10).
//!
//! A peer assesses a run in two staged steps so it never fetches a module it cannot host:
//!
//! 1. **Envelope pre-screen** — a pure function of the *envelope* (declared required capabilities +
//!    tolerated round modes) against this peer's advertised capabilities and the coordinator's round
//!    mode. This runs **before any module fetch** ([`prescreen`]): a capability or round-mode
//!    mismatch is rejected without moving a byte of module code.
//! 2. **Manifest verification** — after the module is fetched, its `da_manifest` cadence block is
//!    re-derived and verified equal to the envelope's copy ([`verify_manifest`]); a mismatch means
//!    the envelope and the module disagree and the run is rejected (§6.1 hash-chain intent — the
//!    cadence a peer paces to must be the one the envelope froze).
//!
//! Both are pure + engine-free, so they are unit-testable without a worker (the eligibility *verdict*
//! over the real worker protocol is `daemon-train-client`'s `assess`, RUN-10's supervisor half).

use daemon_swarm_proto::capability::{Capability, CapabilitySet};

/// The pre-screen verdict (before module fetch).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Prescreen {
    /// The envelope pre-screen passed; the module may be fetched + assessed in meta mode.
    Eligible,
    /// The pre-screen rejected the run without fetching the module.
    Rejected(PrescreenReject),
}

/// Why the envelope pre-screen rejected a run (before any module fetch).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrescreenReject {
    /// The peer's advertised capabilities are missing ops the envelope requires (`required ⊄
    /// advertised`, §6.5).
    MissingCapabilities(Vec<Capability>),
    /// The coordinator's round mode is not one the envelope's module tolerates (§6.2 round modes).
    RoundModeIncompatible {
        /// The coordinator's configured round mode.
        coordinator: String,
        /// The round modes the envelope's module tolerates.
        module_supports: Vec<String>,
    },
}

/// Envelope pre-screen (RUN-10, step 1) — pure over the envelope + peer advertisement, **before any
/// module fetch**. Checks required-capability subset admission, then round-mode compatibility.
#[must_use]
pub fn prescreen(
    required: &CapabilitySet,
    advertised: &CapabilitySet,
    envelope_round_modes: &[String],
    coordinator_round_mode: &str,
) -> Prescreen {
    let missing = advertised.missing(required);
    if !missing.is_empty() {
        return Prescreen::Rejected(PrescreenReject::MissingCapabilities(missing));
    }
    if !envelope_round_modes
        .iter()
        .any(|m| m == coordinator_round_mode)
    {
        return Prescreen::Rejected(PrescreenReject::RoundModeIncompatible {
            coordinator: coordinator_round_mode.to_string(),
            module_supports: envelope_round_modes.to_vec(),
        });
    }
    Prescreen::Eligible
}

/// The manifest-vs-envelope cadence consistency check (RUN-10, step 2), run after the module is
/// fetched: the module's re-derived `da_manifest` cadence must equal the envelope's copy.
///
/// # Errors
///
/// [`ManifestMismatch`] when the re-derived cadence differs from the envelope's frozen value.
pub fn verify_manifest(
    envelope_cadence: u32,
    manifest_cadence: u32,
) -> Result<(), ManifestMismatch> {
    if envelope_cadence == manifest_cadence {
        Ok(())
    } else {
        Err(ManifestMismatch {
            envelope: envelope_cadence,
            manifest: manifest_cadence,
        })
    }
}

/// The module's re-derived cadence disagrees with the envelope's frozen copy (RUN-10, §6.2/§6.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ManifestMismatch {
    /// The cadence the envelope froze.
    pub envelope: u32,
    /// The cadence the module's `da_manifest` re-derived.
    pub manifest: u32,
}

impl core::fmt::Display for ManifestMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "manifest cadence {} != envelope cadence {}",
            self.manifest, self.envelope
        )
    }
}

impl std::error::Error for ManifestMismatch {}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(tokens: &[&str]) -> CapabilitySet {
        CapabilitySet::from_tokens(tokens.iter().copied()).unwrap()
    }

    #[test]
    fn prescreen_rejects_before_fetch() {
        // A missing required capability is rejected from the envelope + advertisement alone — no
        // module bytes are involved (the function takes none).
        let required = caps(&["dct2@1", "topk_chunk@1"]);
        let advertised = caps(&["dct2@1"]); // missing topk_chunk@1
        let modes = vec!["barrier".to_string()];
        match prescreen(&required, &advertised, &modes, "barrier") {
            Prescreen::Rejected(PrescreenReject::MissingCapabilities(m)) => {
                assert_eq!(m, vec![Capability::parse("topk_chunk@1").unwrap()]);
            }
            other => panic!("expected MissingCapabilities, got {other:?}"),
        }

        // A round-mode incompatibility is likewise pre-fetch.
        let ok_caps = caps(&["dct2@1", "topk_chunk@1"]);
        let barrier_only = vec!["barrier".to_string()];
        match prescreen(&required, &ok_caps, &barrier_only, "pipelined") {
            Prescreen::Rejected(PrescreenReject::RoundModeIncompatible { coordinator, .. }) => {
                assert_eq!(coordinator, "pipelined");
            }
            other => panic!("expected RoundModeIncompatible, got {other:?}"),
        }

        // Subset caps + a supported mode passes the pre-screen (proceed to fetch).
        assert_eq!(
            prescreen(&required, &ok_caps, &barrier_only, "barrier"),
            Prescreen::Eligible
        );
    }

    #[test]
    fn manifest_envelope_cadence_mismatch_rejected() {
        // The re-derived manifest cadence must equal the envelope's frozen copy.
        assert!(verify_manifest(30, 30).is_ok());
        let err = verify_manifest(30, 32).unwrap_err();
        assert_eq!(
            err,
            ManifestMismatch {
                envelope: 30,
                manifest: 32
            }
        );
        assert!(err.to_string().contains("!= envelope cadence 30"));
    }
}
