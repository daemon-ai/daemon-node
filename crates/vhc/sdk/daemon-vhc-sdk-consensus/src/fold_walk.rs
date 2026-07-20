// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **deterministic slice-decomposition schedule** for the streamed det-lane fold walks —
//! the pinned contract the multi-slice fold engine and the streaming trainer guest build
//! against.
//!
//! # Why a schedule is a contract
//!
//! Resident ingest/make_update run as ONE synchronous call over all parameters inside one event
//! slice; the ABI resets fuel/op budgets per slice, so a multi-GiB single-slice fold traps long
//! before it hits memory. The streamed walks are therefore **completion-driven multi-slice
//! state machines**: each window read completes as an event, each slice folds a bounded number
//! of windows and issues the next reads, and the digest carry / family seal land in the walk's
//! final slice. For the result to stay in the det lane's equality class, the decomposition must
//! be pinned:
//!
//! - **Window enumeration** ([`windows`]) is per parameter, in registration order, ascending
//!   within each parameter (a parameter never spans a window; the last window of a parameter
//!   may be short) — exactly the per-parameter chunking of the det-state contract, so window
//!   ordinal ≡ family chunk ordinal.
//! - **Fold order is ascending, always** ([`FoldWalk`]): reads may complete out of order, but a
//!   completed window folds only when every earlier window has folded. Per-window fold math is
//!   window-local (the resident per-parameter operation sequence, window-sliced: record-ordered
//!   scatter-adds into a zeroed accumulator, rebase copy, single axpy — then emit, then the
//!   digest-carry advance), and windows partition each parameter ascending, so every f32 op
//!   executes with identical operands in identical order as the resident walk — the
//!   windowed ≡ resident bit-identity the parity suites prove.
//! - **Reads run ahead bounded**: at most `in_flight` windows are outstanding, issued in
//!   ascending order; per-slice work is bounded by construction (windows folded per slice ×
//!   window bytes), so the honest fuel claim is per-window, not per-round.
//! - The walk **seals in the final slice**: the seal action fires exactly once, in the slice
//!   that folds the last window.
//!
//! The degenerate geometry is the same code path: a window size ≥ every parameter's byte length
//! makes each parameter one window, and a single-parameter tier collapses to a one-window walk
//! (issue → fold → seal in one slice) — no resident-mode special case exists.
//!
//! Ingest and make_update share this schedule; they differ only in the per-window math and in
//! which family streams the fold emits. The shared vectors
//! (`tests/fixtures/fold-walk-vectors.json`) pin the enumeration and the invariance of the
//! fold/seal order under arbitrary completion permutations.

use daemon_vhc_proto::det_state::{family_chunk_count, STATE_ELEM_BYTES};

/// One fold window: a `(parameter, byte range)` pair under the per-parameter chunking rule.
/// `ordinal` is the walk position AND the family chunk ordinal (the identities coincide by
/// construction).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    /// Walk position / family chunk ordinal.
    pub ordinal: u64,
    /// Parameter index (registration order).
    pub param: u32,
    /// Byte offset within the parameter's f32-le image.
    pub param_off: u64,
    /// Window byte length (`window_size`, except a parameter's short tail).
    pub len: u64,
}

/// Enumerate the fold windows of a parameter layout at `window_size` bytes: parameters in
/// registration order, windows ascending within each parameter, tails short. Empty for
/// degenerate inputs (`window_size == 0` or an empty layout).
#[must_use]
pub fn windows(numels: &[u64], window_size: u64) -> Vec<Window> {
    if window_size == 0 {
        return Vec::new();
    }
    let mut out =
        Vec::with_capacity(usize::try_from(family_chunk_count(numels, window_size)).unwrap_or(0));
    let mut ordinal = 0u64;
    for (param, &numel) in numels.iter().enumerate() {
        let byte_len = numel * STATE_ELEM_BYTES;
        let mut off = 0u64;
        while off < byte_len {
            let len = (byte_len - off).min(window_size);
            out.push(Window {
                ordinal,
                param: u32::try_from(param).expect("registration order fits u32"),
                param_off: off,
                len,
            });
            ordinal += 1;
            off += len;
        }
    }
    out
}

/// The actions one event slice performs, in order: fold the listed windows (each fold implies
/// the full per-window sequence — window math, emit, digest-carry advance), then issue the
/// listed window reads, then seal if this was the walk's final slice.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SliceActions {
    /// Window ordinals to fold in this slice, ascending, contiguous with everything already
    /// folded.
    pub fold: Vec<u64>,
    /// Window ordinals whose reads this slice issues, ascending.
    pub issue: Vec<u64>,
    /// Whether this slice seals the walk (fires exactly once, with the last fold).
    pub seal: bool,
}

/// The completion-driven fold-walk state machine over `n` windows with at most `in_flight`
/// outstanding reads. Drive it with [`FoldWalk::start`] once, then [`FoldWalk::on_completion`]
/// per completed read; every returned [`SliceActions`] is deterministic in walk state alone, and
/// the concatenated fold order is `0..n` for EVERY arrival permutation (the pinned invariant).
#[derive(Clone, Debug)]
pub struct FoldWalk {
    total: u64,
    in_flight: u64,
    next_issue: u64,
    next_fold: u64,
    /// Completed-but-not-yet-foldable windows (out-of-order arrivals), kept sorted.
    pending: Vec<u64>,
    sealed: bool,
}

/// A completion for a window the walk never issued (or already folded) — a protocol violation
/// by the driver, surfaced typed so a guest can trap rather than diverge silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnexpectedCompletion(pub u64);

impl core::fmt::Display for UnexpectedCompletion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "completion for window {} that is not outstanding",
            self.0
        )
    }
}

impl std::error::Error for UnexpectedCompletion {}

impl FoldWalk {
    /// A walk over `total` windows with at most `in_flight` outstanding reads (`in_flight == 0`
    /// is clamped to 1 — a walk that can never issue would never finish).
    #[must_use]
    pub fn new(total: u64, in_flight: u64) -> Self {
        Self {
            total,
            in_flight: in_flight.max(1),
            next_issue: 0,
            next_fold: 0,
            pending: Vec::new(),
            sealed: false,
        }
    }

    /// The opening slice: issue the first `in_flight` reads (an empty walk seals immediately).
    pub fn start(&mut self) -> SliceActions {
        let mut actions = SliceActions::default();
        if self.total == 0 && !self.sealed {
            self.sealed = true;
            actions.seal = true;
            return actions;
        }
        self.issue_up_to_limit(&mut actions);
        actions
    }

    /// One window read completed: fold the maximal contiguous run now available (ascending from
    /// the walk's fold cursor), refill the read window, and seal when the last window folds.
    ///
    /// # Errors
    /// [`UnexpectedCompletion`] for a window that is not outstanding.
    pub fn on_completion(&mut self, ordinal: u64) -> Result<SliceActions, UnexpectedCompletion> {
        if ordinal < self.next_fold || ordinal >= self.next_issue || self.pending.contains(&ordinal)
        {
            return Err(UnexpectedCompletion(ordinal));
        }
        let idx = self.pending.partition_point(|&p| p < ordinal);
        self.pending.insert(idx, ordinal);

        let mut actions = SliceActions::default();
        // Fold the maximal contiguous completed run starting at the fold cursor.
        while self.pending.first() == Some(&self.next_fold) {
            self.pending.remove(0);
            actions.fold.push(self.next_fold);
            self.next_fold += 1;
        }
        self.issue_up_to_limit(&mut actions);
        if self.next_fold == self.total && !self.sealed {
            self.sealed = true;
            actions.seal = true;
        }
        Ok(actions)
    }

    /// Whether every window has folded and the seal has fired.
    #[must_use]
    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    fn issue_up_to_limit(&mut self, actions: &mut SliceActions) {
        let outstanding = self.next_issue - self.next_fold - self.pending.len() as u64;
        let mut room = self.in_flight.saturating_sub(outstanding);
        while room > 0 && self.next_issue < self.total {
            actions.issue.push(self.next_issue);
            self.next_issue += 1;
            room -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive a walk over `total` windows with the given arrival permutation, returning the
    /// concatenated fold order and the slice count where the seal fired.
    fn drive(total: u64, in_flight: u64, arrivals: &[u64]) -> (Vec<u64>, Option<usize>) {
        let mut walk = FoldWalk::new(total, in_flight);
        let mut issued: Vec<u64> = Vec::new();
        let mut folds = Vec::new();
        let mut seal_slice = None;
        let opening = walk.start();
        issued.extend(&opening.issue);
        if opening.seal {
            seal_slice = Some(0);
        }
        for (slice, &arrival) in arrivals.iter().enumerate() {
            assert!(issued.contains(&arrival), "arrivals only for issued reads");
            let actions = walk.on_completion(arrival).unwrap();
            folds.extend(&actions.fold);
            issued.extend(&actions.issue);
            if actions.seal {
                assert_eq!(seal_slice, None, "the seal fires exactly once");
                seal_slice = Some(slice + 1);
            }
        }
        (folds, seal_slice)
    }

    #[test]
    fn window_enumeration_is_per_parameter_ascending_with_short_tails() {
        // numels 16/4/8 f32 → 64/16/32 bytes at window 24 → 3+1+2 windows.
        let w = windows(&[16, 4, 8], 24);
        assert_eq!(w.len(), 6);
        assert_eq!(
            w.iter()
                .map(|w| (w.param, w.param_off, w.len))
                .collect::<Vec<_>>(),
            vec![
                (0, 0, 24),
                (0, 24, 24),
                (0, 48, 16), // parameter 0's short tail — never spans into parameter 1
                (1, 0, 16),  // parameter 1 is one short window
                (2, 0, 24),
                (2, 24, 8),
            ]
        );
        assert!(w.iter().enumerate().all(|(i, w)| w.ordinal == i as u64));
    }

    #[test]
    fn degenerate_geometry_is_the_same_code_path() {
        // A window size covering every parameter: one window per parameter.
        let w = windows(&[16, 4, 8], 1 << 20);
        assert_eq!(w.len(), 3);
        assert!(w.iter().all(|w| w.param_off == 0));
        // A single-parameter tier: a one-window walk that folds + seals in one slice.
        let (folds, seal) = drive(1, 4, &[0]);
        assert_eq!(folds, vec![0]);
        assert_eq!(seal, Some(1));
    }

    #[test]
    fn fold_order_is_ascending_for_out_of_order_arrivals() {
        // in_flight 3 over 5 windows, arrivals maximally reversed within the read window.
        let (folds, seal) = drive(5, 3, &[2, 1, 0, 4, 3]);
        assert_eq!(folds, vec![0, 1, 2, 3, 4]);
        assert_eq!(seal, Some(5));
        // In-order arrivals give the identical fold order.
        let (folds, seal) = drive(5, 3, &[0, 1, 2, 3, 4]);
        assert_eq!(folds, vec![0, 1, 2, 3, 4]);
        assert_eq!(seal, Some(5));
    }

    #[test]
    fn issues_never_exceed_in_flight_and_walk_seals_exactly_once() {
        let mut walk = FoldWalk::new(6, 2);
        let opening = walk.start();
        assert_eq!(opening.issue, vec![0, 1]);
        let a = walk.on_completion(1).unwrap(); // out of order: nothing foldable yet
        assert!(a.fold.is_empty());
        assert_eq!(a.issue, vec![2], "one completion frees one issue slot");
        let b = walk.on_completion(0).unwrap(); // unblocks the contiguous run 0..=1
        assert_eq!(b.fold, vec![0, 1]);
        // Window 2 is still outstanding, so exactly one slot frees.
        assert_eq!(b.issue, vec![3]);
        assert!(!b.seal);
        for ordinal in [2, 3, 4] {
            assert!(!walk.on_completion(ordinal).unwrap().seal);
        }
        let last = walk.on_completion(5).unwrap();
        assert_eq!(last.fold, vec![5]);
        assert!(last.seal);
        assert!(walk.is_sealed());
    }

    #[test]
    fn unexpected_completions_are_typed_refusals() {
        let mut walk = FoldWalk::new(3, 2);
        walk.start();
        assert_eq!(
            walk.on_completion(2),
            Err(UnexpectedCompletion(2)),
            "never issued"
        );
        walk.on_completion(0).unwrap();
        assert_eq!(
            walk.on_completion(0),
            Err(UnexpectedCompletion(0)),
            "already folded"
        );
        walk.on_completion(1).unwrap();
        assert_eq!(
            walk.on_completion(1),
            Err(UnexpectedCompletion(1)),
            "duplicate"
        );
    }

    #[test]
    fn empty_walk_seals_in_the_opening_slice() {
        let mut walk = FoldWalk::new(0, 4);
        let opening = walk.start();
        assert!(opening.seal && opening.fold.is_empty() && opening.issue.is_empty());
        assert!(walk.is_sealed());
    }
}
