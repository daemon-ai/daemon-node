// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The watch-pattern rate-limit state machines (hermes `process_registry.py`
//! `_check_watch_patterns` / `_global_watch_admit`), ported as pure, clock-driven logic so the
//! limits are testable without threads or sleeps.
//!
//! Per-session hard rule: at most ONE emitted match per cooldown window; any match arriving inside
//! the window is dropped and counts ONE strike per window. After `strike_limit` consecutive strike
//! windows, watch is permanently disabled for the session and promoted to notify-on-complete. A
//! clean window (no drops) resets the strike run. On top sits a cross-session global circuit
//! breaker so concurrent siblings cannot collectively flood: over `max_per_window` admitted events
//! in one window trips a cooldown that drops (and counts) everything until it releases.

/// Per-session watch scan outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScanOutcome {
    /// The chunk matched, and the match is emitted (subject to the global breaker).
    Emit {
        /// The first pattern that matched.
        pattern: String,
        /// The matched lines (first 20, capped at 2000 chars).
        output: String,
        /// Matches dropped by the rate limit since the previous emission.
        suppressed: u32,
    },
    /// The match arrived inside the cooldown window and was dropped (counted).
    Dropped,
    /// This drop crossed the strike limit: watch is now disabled and the session is promoted to
    /// notify-on-complete; emit exactly one summary saying so.
    Disabled {
        /// Matches dropped by the rate limit in total (for the summary).
        suppressed: u32,
    },
}

/// Per-session watch rate-limit state (hermes' `_watch_*` fields on `ProcessSession`).
#[derive(Debug, Default)]
pub struct WatchState {
    /// Total matches emitted.
    pub hits: u32,
    /// Matches dropped by the rate limit since the last emission.
    pub suppressed: u32,
    /// Permanently disabled after the strike limit (promoted to notify-on-complete).
    pub disabled: bool,
    cooldown_until_ms: u64,
    strike_candidate: bool,
    consecutive_strikes: u32,
}

impl WatchState {
    /// Scan `chunk` against `patterns` at `now_ms`, applying the per-session rate limit. Returns
    /// `None` when nothing matched (or watch is disabled).
    pub fn scan(
        &mut self,
        now_ms: u64,
        min_interval_ms: u64,
        strike_limit: u32,
        patterns: &[String],
        chunk: &str,
    ) -> Option<ScanOutcome> {
        if patterns.is_empty() || self.disabled {
            return None;
        }
        let mut matched_lines: Vec<&str> = Vec::new();
        let mut matched_pattern: Option<&String> = None;
        for line in chunk.lines() {
            for pat in patterns {
                if line.contains(pat.as_str()) {
                    matched_lines.push(line.trim_end());
                    if matched_pattern.is_none() {
                        matched_pattern = Some(pat);
                    }
                    break; // one match per line is enough
                }
            }
        }
        if matched_lines.is_empty() {
            return None;
        }

        // Case 1: still inside the cooldown from the last emission — drop, count one strike per
        // window, and disable + promote at the strike limit.
        if self.cooldown_until_ms != 0 && now_ms < self.cooldown_until_ms {
            self.suppressed = self
                .suppressed
                .saturating_add(u32::try_from(matched_lines.len()).unwrap_or(u32::MAX));
            if !self.strike_candidate {
                self.strike_candidate = true;
                self.consecutive_strikes += 1;
                if self.consecutive_strikes >= strike_limit {
                    self.disabled = true;
                    return Some(ScanOutcome::Disabled {
                        suppressed: self.suppressed,
                    });
                }
            }
            return Some(ScanOutcome::Dropped);
        }

        // Case 2: cooldown expired. A prior window with no drops resets the strike run.
        if self.cooldown_until_ms != 0 && !self.strike_candidate {
            self.consecutive_strikes = 0;
        }
        self.strike_candidate = false;
        self.cooldown_until_ms = now_ms + min_interval_ms;
        self.hits += 1;
        let suppressed = self.suppressed;
        self.suppressed = 0;

        // Trim the matched output like hermes: first 20 lines, capped at 2000 chars.
        let mut output = matched_lines
            .iter()
            .take(20)
            .copied()
            .collect::<Vec<_>>()
            .join("\n");
        if output.len() > 2000 {
            let mut end = 2000;
            while end > 0 && !output.is_char_boundary(end) {
                end -= 1;
            }
            output.truncate(end);
            output.push_str("\n...(truncated)");
        }
        Some(ScanOutcome::Emit {
            pattern: matched_pattern
                .expect("matched lines imply a pattern")
                .clone(),
            output,
            suppressed,
        })
    }
}

/// The global admit decision for one emitted watch match.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlobalOutcome {
    /// A just-expired cooldown released with this many events suppressed during the trip (emit one
    /// "notifications resumed" summary).
    pub released: Option<u32>,
    /// This event tripped the breaker (emit one "overflow, suppressing" summary).
    pub tripped: bool,
    /// Whether this event is admitted through to the notifier.
    pub admitted: bool,
}

/// The cross-session circuit breaker (hermes' `_global_watch_*` registry fields).
#[derive(Debug, Default)]
pub struct GlobalBreaker {
    window_start_ms: u64,
    window_hits: u32,
    tripped_until_ms: u64,
    suppressed_during_trip: u32,
}

impl GlobalBreaker {
    /// Admit or drop one watch-match event at `now_ms`.
    pub fn admit(
        &mut self,
        now_ms: u64,
        max_per_window: u32,
        window_ms: u64,
        cooldown_ms: u64,
    ) -> GlobalOutcome {
        let mut outcome = GlobalOutcome::default();

        // Handle cooldown expiry first so the release summary precedes this event's decision.
        if self.tripped_until_ms != 0 && now_ms >= self.tripped_until_ms {
            let suppressed = self.suppressed_during_trip;
            self.tripped_until_ms = 0;
            self.suppressed_during_trip = 0;
            self.window_start_ms = now_ms;
            self.window_hits = 0;
            if suppressed > 0 {
                outcome.released = Some(suppressed);
            }
        }

        // Still in cooldown — drop and count.
        if self.tripped_until_ms != 0 && now_ms < self.tripped_until_ms {
            self.suppressed_during_trip += 1;
            return outcome; // admitted = false
        }

        // Slide the window.
        if now_ms.saturating_sub(self.window_start_ms) >= window_ms {
            self.window_start_ms = now_ms;
            self.window_hits = 0;
        }
        if self.window_hits >= max_per_window {
            self.tripped_until_ms = now_ms + cooldown_ms;
            self.suppressed_during_trip += 1;
            outcome.tripped = true;
            return outcome; // admitted = false
        }
        self.window_hits += 1;
        outcome.admitted = true;
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INTERVAL_MS: u64 = 15_000;
    const STRIKES: u32 = 3;

    fn pats(p: &[&str]) -> Vec<String> {
        p.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn emits_once_per_window_and_counts_suppressed() {
        let mut w = WatchState::default();
        let patterns = pats(&["READY"]);
        // First match emits.
        let out = w.scan(0, INTERVAL_MS, STRIKES, &patterns, "READY on :8080\n");
        assert!(matches!(out, Some(ScanOutcome::Emit { suppressed: 0, .. })));
        // Inside the window: dropped (one strike for the window, both drops counted).
        assert_eq!(
            w.scan(1_000, INTERVAL_MS, STRIKES, &patterns, "READY again"),
            Some(ScanOutcome::Dropped)
        );
        assert_eq!(
            w.scan(2_000, INTERVAL_MS, STRIKES, &patterns, "READY again"),
            Some(ScanOutcome::Dropped)
        );
        // Window expired: the next match emits and reports the two suppressed drops.
        match w.scan(INTERVAL_MS + 1, INTERVAL_MS, STRIKES, &patterns, "READY x") {
            Some(ScanOutcome::Emit { suppressed, .. }) => assert_eq!(suppressed, 2),
            other => panic!("expected emit, got {other:?}"),
        }
    }

    #[test]
    fn three_consecutive_strike_windows_disable_watch() {
        let mut w = WatchState::default();
        let patterns = pats(&["ERR"]);
        let mut now = 0u64;
        // Window 1 emit, then a drop (strike 1); window 2 emit+drop (strike 2); window 3 the drop
        // crosses the limit → Disabled.
        assert!(matches!(
            w.scan(now, INTERVAL_MS, STRIKES, &patterns, "ERR a"),
            Some(ScanOutcome::Emit { .. })
        ));
        assert_eq!(
            w.scan(now + 1, INTERVAL_MS, STRIKES, &patterns, "ERR b"),
            Some(ScanOutcome::Dropped)
        );
        now += INTERVAL_MS + 1;
        assert!(matches!(
            w.scan(now, INTERVAL_MS, STRIKES, &patterns, "ERR c"),
            Some(ScanOutcome::Emit { .. })
        ));
        assert_eq!(
            w.scan(now + 1, INTERVAL_MS, STRIKES, &patterns, "ERR d"),
            Some(ScanOutcome::Dropped)
        );
        now += INTERVAL_MS + 1;
        assert!(matches!(
            w.scan(now, INTERVAL_MS, STRIKES, &patterns, "ERR e"),
            Some(ScanOutcome::Emit { .. })
        ));
        match w.scan(now + 1, INTERVAL_MS, STRIKES, &patterns, "ERR f") {
            Some(ScanOutcome::Disabled { .. }) => {}
            other => panic!("expected Disabled at the third strike window, got {other:?}"),
        }
        assert!(w.disabled);
        // Disabled: further matches are invisible.
        assert_eq!(
            w.scan(now + 2, INTERVAL_MS, STRIKES, &patterns, "ERR g"),
            None
        );
    }

    #[test]
    fn a_clean_window_resets_the_strike_run() {
        let mut w = WatchState::default();
        let patterns = pats(&["X"]);
        // Emit + drop (strike 1).
        w.scan(0, INTERVAL_MS, STRIKES, &patterns, "X").unwrap();
        w.scan(1, INTERVAL_MS, STRIKES, &patterns, "X").unwrap();
        assert_eq!(w.consecutive_strikes, 1);
        // Next window emits with NO drop → clean.
        let t2 = INTERVAL_MS + 1;
        w.scan(t2, INTERVAL_MS, STRIKES, &patterns, "X").unwrap();
        // The window after that: the clean prior window reset the run.
        let t3 = t2 + INTERVAL_MS + 1;
        w.scan(t3, INTERVAL_MS, STRIKES, &patterns, "X").unwrap();
        assert_eq!(w.consecutive_strikes, 0, "clean window resets strikes");
    }

    #[test]
    fn match_output_is_trimmed_to_twenty_lines() {
        let mut w = WatchState::default();
        let patterns = pats(&["hit"]);
        let chunk = (0..30)
            .map(|i| format!("hit {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        match w.scan(0, INTERVAL_MS, STRIKES, &patterns, &chunk) {
            Some(ScanOutcome::Emit { output, .. }) => {
                assert_eq!(output.lines().count(), 20);
            }
            other => panic!("expected emit, got {other:?}"),
        }
    }

    const G_MAX: u32 = 15;
    const G_WINDOW_MS: u64 = 10_000;
    const G_COOLDOWN_MS: u64 = 30_000;

    #[test]
    fn global_breaker_trips_after_the_cap_and_releases_with_a_summary() {
        let mut g = GlobalBreaker::default();
        // 15 admits inside one window.
        for i in 0..G_MAX {
            let out = g.admit(u64::from(i), G_MAX, G_WINDOW_MS, G_COOLDOWN_MS);
            assert!(out.admitted, "event {i} admitted");
        }
        // The 16th trips.
        let tripped = g.admit(100, G_MAX, G_WINDOW_MS, G_COOLDOWN_MS);
        assert!(tripped.tripped && !tripped.admitted);
        // During the cooldown everything is dropped and counted.
        for t in [200u64, 300, 400] {
            let out = g.admit(t, G_MAX, G_WINDOW_MS, G_COOLDOWN_MS);
            assert!(!out.admitted && !out.tripped);
        }
        // After the cooldown, the first event releases (reporting 4 suppressed) and is admitted.
        let released = g.admit(100 + G_COOLDOWN_MS, G_MAX, G_WINDOW_MS, G_COOLDOWN_MS);
        assert_eq!(released.released, Some(4));
        assert!(released.admitted);
    }

    #[test]
    fn global_window_slides_and_resets_the_count() {
        let mut g = GlobalBreaker::default();
        for i in 0..G_MAX {
            assert!(
                g.admit(u64::from(i), G_MAX, G_WINDOW_MS, G_COOLDOWN_MS)
                    .admitted
            );
        }
        // A fresh window admits again instead of tripping.
        let out = g.admit(G_WINDOW_MS + 1, G_MAX, G_WINDOW_MS, G_COOLDOWN_MS);
        assert!(out.admitted && !out.tripped);
    }
}
