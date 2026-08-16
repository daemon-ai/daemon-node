// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Incremental stop-sequence matching with holdback (engine-independent).
//!
//! A stop sequence can arrive split across token pieces, so a streaming generation must hold back
//! any output tail that could still be the *prefix* of a stop before surfacing it (the llama-server
//! / q1-2026 `StopDetector` pattern). [`StopMatcher::push`] returns the text that is provably not
//! part of a stop; when a full stop matches, generation cuts and the stop text is swallowed.
//!
//! Pure and engine-free, so it is unit-tested in the default stub build like [`crate::tooling`].

/// Incremental stop-sequence matcher. Feed decoded pieces with [`push`](Self::push); at end of
/// generation, [`flush`](Self::flush) releases any held-back tail that never completed a stop.
#[derive(Debug, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    /// The undecided tail: a suffix of the output so far that is a proper prefix of some stop.
    held: String,
}

impl StopMatcher {
    /// A matcher for `stops` (empty sequences are dropped; an empty set never holds anything back).
    pub fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            held: String::new(),
        }
    }

    /// Whether any stop sequences are configured.
    pub fn is_active(&self) -> bool {
        !self.stops.is_empty()
    }

    /// Feed `piece`; returns `(releasable, hit)`. `releasable` is output text that is provably not
    /// part of any stop; `hit` means a full stop matched — generation must cut, the stop text is
    /// swallowed, and anything the model produced before the stop has been released.
    pub fn push(&mut self, piece: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (piece.to_string(), false);
        }
        self.held.push_str(piece);

        // A completed stop anywhere in the pending tail: release what precedes it, swallow the rest.
        if let Some(at) = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()))
            .min()
        {
            let released = self.held[..at].to_string();
            self.held.clear();
            return (released, true);
        }

        // Hold back the longest tail that is still a prefix of some stop; release the remainder.
        let keep = self.longest_stop_prefix_suffix();
        let release_to = self.held.len() - keep;
        let released = self.held[..release_to].to_string();
        self.held.drain(..release_to);
        (released, false)
    }

    /// End of generation without a stop hit: release the held-back tail verbatim.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// The length (bytes) of the longest suffix of `held` that is a proper prefix of any stop.
    fn longest_stop_prefix_suffix(&self) -> usize {
        let mut longest = 0usize;
        for stop in &self.stops {
            // Try suffixes from the longest candidate down; the first prefix match wins for this stop.
            let max = stop.len().saturating_sub(1).min(self.held.len());
            for take in (longest + 1..=max).rev() {
                if !self.held.is_char_boundary(self.held.len() - take) {
                    continue;
                }
                let suffix = &self.held[self.held.len() - take..];
                if stop.starts_with(suffix) {
                    longest = longest.max(take);
                    break;
                }
            }
        }
        longest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn passthrough_without_stops() {
        let mut m = StopMatcher::new(&[]);
        assert!(!m.is_active());
        assert_eq!(m.push("hello"), ("hello".to_string(), false));
        assert_eq!(m.flush(), "");
    }

    #[test]
    fn whole_stop_in_one_piece_cuts_and_swallows() {
        let mut m = StopMatcher::new(&stops(&["<|im_end|>"]));
        let (out, hit) = m.push("answer<|im_end|>trailing");
        assert_eq!(out, "answer");
        assert!(hit);
    }

    #[test]
    fn stop_split_across_pieces_is_held_back_then_cut() {
        let mut m = StopMatcher::new(&stops(&["<|im_end|>"]));
        let (out, hit) = m.push("answer<|im_");
        assert_eq!(out, "answer");
        assert!(!hit);
        let (out, hit) = m.push("end|>");
        assert_eq!(out, "");
        assert!(hit);
    }

    #[test]
    fn false_prefix_is_released_on_divergence() {
        let mut m = StopMatcher::new(&stops(&["<|im_end|>"]));
        let (out, hit) = m.push("a<|im_");
        assert_eq!(out, "a");
        assert!(!hit);
        let (out, hit) = m.push("portant|>");
        assert_eq!(out, "<|im_portant|>");
        assert!(!hit);
        assert_eq!(m.flush(), "");
    }

    #[test]
    fn flush_releases_pending_tail_at_eog() {
        let mut m = StopMatcher::new(&stops(&["\nuser:"]));
        let (out, hit) = m.push("done\nuse");
        assert_eq!(out, "done");
        assert!(!hit);
        assert_eq!(m.flush(), "\nuse");
    }

    #[test]
    fn earliest_of_multiple_stops_wins() {
        let mut m = StopMatcher::new(&stops(&["STOP", "HALT"]));
        let (out, hit) = m.push("xHALTySTOP");
        assert_eq!(out, "x");
        assert!(hit);
    }

    #[test]
    fn multibyte_boundaries_are_respected() {
        let mut m = StopMatcher::new(&stops(&["終わり"]));
        let (out, hit) = m.push("答え終");
        assert_eq!(out, "答え");
        assert!(!hit);
        let (out, hit) = m.push("わり");
        assert_eq!(out, "");
        assert!(hit);
    }
}
