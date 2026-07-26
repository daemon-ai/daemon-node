// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **typed execution context** — one value, one derivation, four consumers
//! (`docs/specs/vhc-module-abi-spec.md` §7.6; architecture §9.6 `[LX-14]`).
//!
//! A single failure writes its execution context four times: it tags a forwarded panic line when
//! that line is emitted, it is compared when the line is lifted into a trap's detail, it
//! classifies the trap, and it is recorded as the terminal record's context where a replay
//! verdict keys on it. Those must be **one value from one derivation**. A host that scopes the
//! lift correctly but records a fixed context has not implemented the rule: the journal then
//! shows a run-phase trap carrying an initialization-phase message and location, which reads as a
//! misattribution bug in the forwarding — the diagnosis inverted, and the reader sent to
//! investigate the wrong mechanism.
//!
//! ## Why the domain has eleven members and not three
//!
//! The informal reduction to "`da_init` / `da_migrate` / a slice ordinal" is wrong in both
//! directions. It drops the three non-slice run states — a trap between slices, before the first
//! event, or after the last is not attributable to a slice and must not invent one — and it drops
//! the assessment and export contexts the record grammar already names. An active export takes
//! precedence over the instance containing it: while the claim, resource-plan or manifest export
//! is running, its own value is authoritative and [`ExecutionContext::Assessment`] applies only
//! outside those calls.
//!
//! ## Why rendering is selected by the negotiated ABI minor
//!
//! Existing journals record the bare string `da_run` for every terminal trap, whatever phase the
//! trap occurred in. Those bytes are evidence, and evidence is not rewritten. A journal written
//! under an ABI at or below [`LEGACY_CONTEXT_MAX_MINOR`] therefore renders — live and on replay —
//! through [`render_legacy_terminal`], and a replay compares the recorded string unchanged rather
//! than normalizing it into one of the eleven values it never meant.

use std::fmt;

/// The highest major-2 minor whose terminal records use the legacy renderer. A journal written at
/// or below this minor carries the bare [`LEGACY_TERMINAL_CONTEXT`] for every terminal trap.
pub const LEGACY_CONTEXT_MAX_MINOR: u32 = 4;

/// The single context string every terminal trap carried at ABI ≤ 2.4, whatever phase it occurred
/// in. Preserved verbatim so historical journals stay comparable as the evidence they are.
pub const LEGACY_TERMINAL_CONTEXT: &str = "da_run";

/// The closed domain of execution contexts (`[LX-14]`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExecutionContext {
    /// Inside `da_init`.
    Init,
    /// Inside `da_migrate`.
    Migrate,
    /// Inside `da_run`, before the first event has been delivered.
    RunBeforeFirstSlice,
    /// Inside `da_run`, while the event slice of this ordinal is active.
    RunSlice(u64),
    /// Inside `da_run`, after one slice has ended and before the next begins.
    RunBetweenSlices,
    /// Inside `da_run`, after the final slice has ended — including after a stop was consumed.
    RunAfterLastSlice,
    /// In assessment-instance setup or teardown, with no assessment export active.
    Assessment,
    /// Inside the legacy claim export.
    Claim,
    /// Inside `da_resource_plan`.
    ResourcePlan,
    /// Inside the manifest export.
    Manifest,
    /// Inside `da_apply_execution_grant`.
    ExecutionGrant,
}

/// The canonical text prefix of a slice context. The ordinal that follows is the canonical
/// unsigned base-10 rendering of a `u64`: digits `0`–`9`, no sign, no leading zero, no
/// whitespace, no locale, no alternate prefix.
pub const SLICE_CONTEXT_PREFIX: &str = "slice:";

impl ExecutionContext {
    /// The canonical ABI-2.5 text form. Total, one-way, and stable: two hosts, or a host and its
    /// replay, produce identical text for identical state.
    #[must_use]
    pub fn render(&self) -> String {
        match self {
            Self::Init => "da_init".to_string(),
            Self::Migrate => "da_migrate".to_string(),
            Self::RunBeforeFirstSlice => "da_run:before".to_string(),
            Self::RunBetweenSlices => "da_run:between".to_string(),
            Self::RunAfterLastSlice => "da_run:after".to_string(),
            Self::RunSlice(ordinal) => format!("{SLICE_CONTEXT_PREFIX}{ordinal}"),
            Self::Assessment => "assessment".to_string(),
            Self::Claim => "da_claim".to_string(),
            Self::ResourcePlan => "da_resource_plan".to_string(),
            Self::Manifest => "da_manifest".to_string(),
            Self::ExecutionGrant => "da_apply_execution_grant".to_string(),
        }
    }

    /// The context a terminal trap record carries, selected by the negotiated major-2 minor. At or
    /// below [`LEGACY_CONTEXT_MAX_MINOR`] every context renders as [`LEGACY_TERMINAL_CONTEXT`],
    /// which is exactly what those journals have always contained; from the certification minor
    /// the truthful eleven-value rendering applies.
    #[must_use]
    pub fn render_for_minor(&self, abi_minor: u32) -> String {
        if abi_minor <= LEGACY_CONTEXT_MAX_MINOR {
            render_legacy_terminal()
        } else {
            self.render()
        }
    }

    /// Parse a canonical ABI-2.5 rendering back into the typed value. Deliberately strict: a sign,
    /// leading zero, whitespace or alternate prefix on a slice ordinal is not a slice context.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "da_init" => Self::Init,
            "da_migrate" => Self::Migrate,
            "da_run:before" => Self::RunBeforeFirstSlice,
            "da_run:between" => Self::RunBetweenSlices,
            "da_run:after" => Self::RunAfterLastSlice,
            "assessment" => Self::Assessment,
            "da_claim" => Self::Claim,
            "da_resource_plan" => Self::ResourcePlan,
            "da_manifest" => Self::Manifest,
            "da_apply_execution_grant" => Self::ExecutionGrant,
            other => {
                let digits = other.strip_prefix(SLICE_CONTEXT_PREFIX)?;
                if digits.is_empty()
                    || !digits.bytes().all(|b| b.is_ascii_digit())
                    || (digits.len() > 1 && digits.starts_with('0'))
                {
                    return None;
                }
                Self::RunSlice(digits.parse().ok()?)
            }
        })
    }

    /// Whether this context is one of the four distinct `da_run` states.
    #[must_use]
    pub fn is_run_phase(&self) -> bool {
        matches!(
            self,
            Self::RunBeforeFirstSlice
                | Self::RunSlice(_)
                | Self::RunBetweenSlices
                | Self::RunAfterLastSlice
        )
    }
}

impl fmt::Display for ExecutionContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

/// The legacy terminal-trap context: the bare string every ABI ≤ 2.4 journal records.
#[must_use]
pub fn render_legacy_terminal() -> String {
    LEGACY_TERMINAL_CONTEXT.to_string()
}

/// Whether a recorded terminal context may be compared against a freshly rendered one. Replay
/// compares the recorded **code** and **context** and never the human-readable detail; a legacy
/// recording is compared under the legacy renderer, unchanged, because reinterpreting it as one of
/// the eleven values would assert something the writer never claimed.
#[must_use]
pub fn terminal_contexts_agree(recorded: &str, live: &ExecutionContext, abi_minor: u32) -> bool {
    recorded == live.render_for_minor(abi_minor)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every member renders exactly, and the boundary vectors the specification names are pinned.
    #[test]
    fn the_canonical_mapping_is_total_and_stable() {
        let vectors: &[(ExecutionContext, &str)] = &[
            (ExecutionContext::Init, "da_init"),
            (ExecutionContext::Migrate, "da_migrate"),
            (ExecutionContext::RunBeforeFirstSlice, "da_run:before"),
            (ExecutionContext::RunBetweenSlices, "da_run:between"),
            (ExecutionContext::RunAfterLastSlice, "da_run:after"),
            (ExecutionContext::RunSlice(0), "slice:0"),
            (ExecutionContext::RunSlice(1), "slice:1"),
            (
                ExecutionContext::RunSlice(u64::MAX),
                "slice:18446744073709551615",
            ),
            (ExecutionContext::Assessment, "assessment"),
            (ExecutionContext::Claim, "da_claim"),
            (ExecutionContext::ResourcePlan, "da_resource_plan"),
            (ExecutionContext::Manifest, "da_manifest"),
            (ExecutionContext::ExecutionGrant, "da_apply_execution_grant"),
        ];
        for (context, expected) in vectors {
            assert_eq!(&context.render(), expected);
            assert_eq!(
                ExecutionContext::parse(expected).as_ref(),
                Some(context),
                "`{expected}` round-trips"
            );
            // Two independent renderings of identical state are byte-identical.
            assert_eq!(context.render(), context.clone().render());
        }
    }

    #[test]
    fn slice_ordinals_reject_signs_whitespace_and_leading_zeroes() {
        for bogus in [
            "slice:+1",
            "slice:-1",
            "slice: 1",
            "slice:1 ",
            "slice:01",
            "slice:",
            "slice:0x1",
            "Slice:1",
            "da_run",
        ] {
            assert!(
                ExecutionContext::parse(bogus).is_none(),
                "`{bogus}` must not parse as a context"
            );
        }
        assert_eq!(
            ExecutionContext::parse("slice:0"),
            Some(ExecutionContext::RunSlice(0)),
            "zero is rendered and parsed as a single digit"
        );
    }

    /// A journal written at a legacy minor keeps its bare `da_run`; the certification minor is the
    /// first that produces the truthful rendering.
    #[test]
    fn rendering_is_selected_by_the_negotiated_minor() {
        let init = ExecutionContext::Init;
        for legacy in 0..=LEGACY_CONTEXT_MAX_MINOR {
            assert_eq!(init.render_for_minor(legacy), LEGACY_TERMINAL_CONTEXT);
        }
        assert_eq!(
            init.render_for_minor(LEGACY_CONTEXT_MAX_MINOR + 1),
            "da_init"
        );
        assert!(terminal_contexts_agree(
            "da_run",
            &ExecutionContext::Init,
            LEGACY_CONTEXT_MAX_MINOR
        ));
        assert!(!terminal_contexts_agree(
            "da_run",
            &ExecutionContext::Init,
            LEGACY_CONTEXT_MAX_MINOR + 1
        ));
    }

    /// A legacy recording is never upgraded into an ABI-2.5 value: the string `da_run` is not one
    /// of the eleven renderings, so no normalization can silently equate them.
    #[test]
    fn a_legacy_recording_is_not_normalized_into_a_certification_value() {
        assert!(ExecutionContext::parse(LEGACY_TERMINAL_CONTEXT).is_none());
        for context in [
            ExecutionContext::RunBeforeFirstSlice,
            ExecutionContext::RunBetweenSlices,
            ExecutionContext::RunAfterLastSlice,
            ExecutionContext::Init,
            ExecutionContext::Migrate,
        ] {
            assert_ne!(context.render(), LEGACY_TERMINAL_CONTEXT);
        }
    }

    #[test]
    fn the_four_run_states_are_distinct() {
        assert!(ExecutionContext::RunBeforeFirstSlice.is_run_phase());
        assert!(ExecutionContext::RunSlice(7).is_run_phase());
        assert!(ExecutionContext::RunBetweenSlices.is_run_phase());
        assert!(ExecutionContext::RunAfterLastSlice.is_run_phase());
        assert!(!ExecutionContext::Init.is_run_phase());
        assert!(!ExecutionContext::Assessment.is_run_phase());
        assert!(!ExecutionContext::ResourcePlan.is_run_phase());
        assert_ne!(
            ExecutionContext::RunBeforeFirstSlice,
            ExecutionContext::RunBetweenSlices
        );
    }
}
