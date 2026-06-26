//! `daemon-schedule` — schedule parsing and next-fire computation for the I15 cron backing.
//!
//! A small, dependency-light crate that turns a human schedule expression into a [`Schedule`] and
//! answers the one question the resident cron scheduler asks each tick: *given an instant, when is
//! the next fire?* It supports three schedule kinds:
//!
//! - **cron** — full 5/6-field cron expressions (via `croner`, including `L`/`W`/`#` extensions),
//!   evaluated in an optional IANA timezone (via `chrono-tz`).
//! - **interval** — `@every <dur>`, ISO-8601 durations (`PT1H30M`), bare compound durations
//!   (`90s`, `2h`, `1h30m`), and the `@hourly`/`@daily`/… named shortcuts.
//! - **once** — a single ISO-8601 timestamp / date (fires exactly once).
//!
//! The crate is protocol-free: the lifecycle policy (`repeat`/`jitter`/overlap/catch-up) lives on
//! the wire `CronSpec`; the scheduler threads the relevant pieces (`repeat`, `jitter_secs`,
//! `timezone`) onto the [`Schedule`] via the builder methods and drives the rest itself.

#![forbid(unsafe_code)]

use chrono::{DateTime, TimeZone, Utc};
use chrono_tz::Tz;
use croner::Cron;
use std::str::FromStr;

/// The grace floor (seconds): a missed fire within this window still runs once. Mirrors Hermes'
/// `MIN_GRACE`, and is the fixed grace for one-shot jobs.
pub const MIN_GRACE_SECS: u64 = 120;
/// The grace ceiling (seconds): clamps the half-period grace so a long-period job does not accept an
/// arbitrarily stale catch-up. Mirrors Hermes' `MAX_GRACE` (2h).
pub const MAX_GRACE_SECS: u64 = 7200;

/// A parse failure for a schedule expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    /// The cron expression could not be parsed by `croner`.
    Cron(String),
    /// A duration/interval expression could not be parsed.
    Duration(String),
    /// A one-shot timestamp could not be parsed.
    Timestamp(String),
    /// The timezone string is not a valid IANA name.
    Timezone(String),
    /// The expression was empty.
    Empty,
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScheduleError::Cron(e) => write!(f, "invalid cron expression: {e}"),
            ScheduleError::Duration(e) => write!(f, "invalid interval/duration: {e}"),
            ScheduleError::Timestamp(e) => write!(f, "invalid one-shot timestamp: {e}"),
            ScheduleError::Timezone(e) => write!(f, "invalid timezone: {e}"),
            ScheduleError::Empty => write!(f, "empty schedule expression"),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// The parsed schedule shape (the discriminant of how next-fire is computed).
#[derive(Clone, Debug)]
enum Kind {
    /// A cron expression (the `croner` matcher).
    Cron(Box<Cron>),
    /// A fixed interval in seconds.
    Interval(u64),
    /// A single absolute fire instant (unix seconds).
    Once(i64),
}

/// A parsed schedule plus the lifecycle knobs the scheduler reads. Construct with [`Schedule::parse`]
/// then layer policy via [`Schedule::with_timezone`], [`Schedule::with_repeat`], and
/// [`Schedule::with_jitter`].
#[derive(Clone, Debug)]
pub struct Schedule {
    kind: Kind,
    tz: Option<Tz>,
    repeat: Option<u32>,
    jitter_secs: Option<u32>,
}

impl Schedule {
    /// Parse a schedule expression into a [`Schedule`] (UTC, no repeat/jitter until layered on).
    ///
    /// Resolution order: `@`-prefixed shortcuts, ISO-8601 durations (`P…`), one-shot timestamps
    /// (`YYYY-MM-DD…`), bare compound durations (`90s`, `1h30m`), then a cron expression.
    pub fn parse(expr: &str) -> Result<Self, ScheduleError> {
        let trimmed = expr.trim();
        if trimmed.is_empty() {
            return Err(ScheduleError::Empty);
        }
        let kind = parse_kind(trimmed)?;
        Ok(Self {
            kind,
            tz: None,
            repeat: None,
            jitter_secs: None,
        })
    }

    /// Set the IANA timezone used to evaluate cron expressions (interval/once are tz-agnostic). A
    /// `None` keeps the default (UTC). An unrecognized name is a [`ScheduleError::Timezone`].
    pub fn with_timezone(mut self, tz: Option<&str>) -> Result<Self, ScheduleError> {
        self.tz = match tz {
            Some(name) if !name.trim().is_empty() => {
                Some(Tz::from_str(name.trim()).map_err(|_| ScheduleError::Timezone(name.into()))?)
            }
            _ => None,
        };
        Ok(self)
    }

    /// Set the maximum number of fires before the job is exhausted (`None` = forever).
    pub fn with_repeat(mut self, repeat: Option<u32>) -> Self {
        self.repeat = repeat;
        self
    }

    /// Set the jitter window in seconds (`None`/`0` = fire exactly on time).
    pub fn with_jitter(mut self, jitter_secs: Option<u32>) -> Self {
        self.jitter_secs = jitter_secs;
        self
    }

    /// The configured repeat cap, if any.
    pub fn repeat(&self) -> Option<u32> {
        self.repeat
    }

    /// Whether this is a one-shot schedule (fires at most once, no recurrence).
    pub fn is_one_shot(&self) -> bool {
        matches!(self.kind, Kind::Once(_))
    }

    /// The next fire strictly after `after_unix` (unix seconds), or `None` when the schedule has no
    /// further occurrence (a one-shot already past `after`, or a cron search that hit its limit).
    pub fn next_after(&self, after_unix: u64) -> Option<u64> {
        match &self.kind {
            Kind::Interval(secs) => Some(after_unix.saturating_add(*secs)),
            Kind::Once(ts) => {
                if *ts as u64 > after_unix {
                    Some(*ts as u64)
                } else {
                    None
                }
            }
            Kind::Cron(cron) => cron_next(cron, after_unix, self.tz),
        }
    }

    /// The approximate period (seconds) between consecutive fires, computed from `from_unix`.
    /// `None` for one-shot schedules.
    pub fn period_hint(&self, from_unix: u64) -> Option<u64> {
        match &self.kind {
            Kind::Interval(secs) => Some(*secs),
            Kind::Once(_) => None,
            Kind::Cron(cron) => {
                let first = cron_next(cron, from_unix, self.tz)?;
                let second = cron_next(cron, first, self.tz)?;
                Some(second.saturating_sub(first))
            }
        }
    }

    /// The grace window (seconds) for a missed fire observed at `from_unix`: `clamp(period/2,
    /// MIN_GRACE, MAX_GRACE)` for recurring schedules, `MIN_GRACE` for one-shot/unknown. A fire
    /// overdue within the grace window is run once (catch-up); beyond it the schedule fast-forwards.
    pub fn grace_secs(&self, from_unix: u64) -> u64 {
        match self.period_hint(from_unix) {
            Some(period) => (period / 2).clamp(MIN_GRACE_SECS, MAX_GRACE_SECS),
            None => MIN_GRACE_SECS,
        }
    }

    /// A deterministic jitter offset in `0..=jitter_secs` derived from `seed` (e.g. the base fire
    /// time mixed with a per-job salt), so identically-scheduled jobs spread rather than thundering.
    pub fn jitter_offset(&self, seed: u64) -> u64 {
        match self.jitter_secs {
            Some(j) if j > 0 => splitmix64(seed) % (j as u64 + 1),
            _ => 0,
        }
    }
}

/// Detect and parse the schedule kind from a (trimmed, non-empty) expression.
fn parse_kind(expr: &str) -> Result<Kind, ScheduleError> {
    // 1. `@`-prefixed: `@every <dur>` interval, or a named shortcut mapped to cron.
    if let Some(rest) = expr.strip_prefix('@') {
        let lower = rest.to_ascii_lowercase();
        if let Some(dur) = lower.strip_prefix("every") {
            let secs = parse_duration_secs(dur.trim())
                .ok_or_else(|| ScheduleError::Duration(expr.into()))?;
            return Ok(Kind::Interval(secs));
        }
        let cron_expr = match lower.as_str() {
            "yearly" | "annually" => "0 0 1 1 *",
            "monthly" => "0 0 1 * *",
            "weekly" => "0 0 * * 0",
            "daily" | "midnight" => "0 0 * * *",
            "hourly" => "0 * * * *",
            other => return Err(ScheduleError::Cron(format!("unknown shortcut @{other}"))),
        };
        return parse_cron(cron_expr);
    }
    // 2. ISO-8601 duration (`P…`) -> interval.
    if expr.starts_with('P') || expr.starts_with('p') {
        if let Some(secs) = parse_iso8601_duration(expr) {
            return Ok(Kind::Interval(secs));
        }
        return Err(ScheduleError::Duration(expr.into()));
    }
    // 3. A one-shot timestamp/date (`YYYY-MM-DD…`).
    if looks_like_timestamp(expr) {
        return parse_once(expr).map(Kind::Once);
    }
    // 4. A bare compound duration (`90s`, `2h`, `1h30m`) -> interval.
    if let Some(secs) = parse_duration_secs(expr) {
        return Ok(Kind::Interval(secs));
    }
    // 5. Otherwise a cron expression.
    parse_cron(expr)
}

/// Parse a cron expression through croner.
fn parse_cron(expr: &str) -> Result<Kind, ScheduleError> {
    Cron::from_str(expr)
        .map(|c| Kind::Cron(Box::new(c)))
        .map_err(|e| ScheduleError::Cron(format!("{e}")))
}

/// Compute the next cron occurrence strictly after `after_unix` (in the schedule's timezone, or UTC).
fn cron_next(cron: &Cron, after_unix: u64, tz: Option<Tz>) -> Option<u64> {
    let after = DateTime::<Utc>::from_timestamp(after_unix as i64, 0)?;
    match tz {
        Some(tz) => {
            let local = after.with_timezone(&tz);
            let next = cron.find_next_occurrence(&local, false).ok()?;
            Some(next.timestamp().max(0) as u64)
        }
        None => {
            let next = cron.find_next_occurrence(&after, false).ok()?;
            Some(next.timestamp().max(0) as u64)
        }
    }
}

/// Heuristic: does the expression look like an absolute timestamp/date (`YYYY-MM-DD`…)?
fn looks_like_timestamp(expr: &str) -> bool {
    let bytes = expr.as_bytes();
    bytes.len() >= 10
        && bytes[0..4].iter().all(u8::is_ascii_digit)
        && bytes[4] == b'-'
        && bytes[5..7].iter().all(u8::is_ascii_digit)
        && bytes[7] == b'-'
        && bytes[8..10].iter().all(u8::is_ascii_digit)
}

/// Parse a one-shot timestamp: RFC-3339 (`2026-07-01T09:00:00Z`), a naive date-time
/// (`2026-07-01T09:00:00`, interpreted as UTC), or a bare date (`2026-07-01`, midnight UTC).
fn parse_once(expr: &str) -> Result<i64, ScheduleError> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(expr) {
        return Ok(dt.timestamp());
    }
    if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(expr, "%Y-%m-%dT%H:%M:%S") {
        return Ok(Utc.from_utc_datetime(&naive).timestamp());
    }
    if let Ok(date) = chrono::NaiveDate::parse_from_str(expr, "%Y-%m-%d") {
        if let Some(naive) = date.and_hms_opt(0, 0, 0) {
            return Ok(Utc.from_utc_datetime(&naive).timestamp());
        }
    }
    Err(ScheduleError::Timestamp(expr.into()))
}

/// Parse a bare compound duration (`90s`, `2h`, `1h30m`, `1d12h`, `2w`) into seconds. Units: `s`
/// seconds, `m` minutes, `h` hours, `d` days, `w` weeks. Returns `None` on any unrecognized shape.
fn parse_duration_secs(expr: &str) -> Option<u64> {
    let expr = expr.trim();
    if expr.is_empty() {
        return None;
    }
    let mut total: u64 = 0;
    let mut num: Option<u64> = None;
    let mut saw_unit = false;
    for ch in expr.chars() {
        if ch.is_ascii_digit() {
            num = Some(
                num.unwrap_or(0)
                    .checked_mul(10)?
                    .checked_add((ch as u8 - b'0') as u64)?,
            );
        } else {
            let n = num.take()?;
            let unit = match ch.to_ascii_lowercase() {
                's' => 1,
                'm' => 60,
                'h' => 3600,
                'd' => 86_400,
                'w' => 604_800,
                _ => return None,
            };
            total = total.checked_add(n.checked_mul(unit)?)?;
            saw_unit = true;
        }
    }
    // A trailing number with no unit (e.g. "30") is rejected: a schedule must be explicit.
    if num.is_some() || !saw_unit {
        return None;
    }
    Some(total)
}

/// Parse an ISO-8601 duration (`P[n]W` or `P[n]DT[n]H[n]M[n]S`) into seconds. Months/years are not
/// supported (no anchor date for a calendar period); such inputs return `None`.
fn parse_iso8601_duration(expr: &str) -> Option<u64> {
    let s = expr.strip_prefix(['P', 'p'])?;
    if s.is_empty() {
        return None;
    }
    // Weeks form: `PnW`.
    if let Some(weeks) = s.strip_suffix(['W', 'w']) {
        let n: u64 = weeks.parse().ok()?;
        return n.checked_mul(604_800);
    }
    let mut total: u64 = 0;
    let mut in_time = false;
    let mut num: Option<u64> = None;
    for ch in s.chars() {
        match ch {
            'T' | 't' => {
                if num.is_some() {
                    return None;
                }
                in_time = true;
            }
            c if c.is_ascii_digit() => {
                num = Some(
                    num.unwrap_or(0)
                        .checked_mul(10)?
                        .checked_add((c as u8 - b'0') as u64)?,
                );
            }
            'D' | 'd' => {
                total = total.checked_add(num.take()?.checked_mul(86_400)?)?;
            }
            'H' | 'h' if in_time => {
                total = total.checked_add(num.take()?.checked_mul(3600)?)?;
            }
            'M' | 'm' if in_time => {
                total = total.checked_add(num.take()?.checked_mul(60)?)?;
            }
            'S' | 's' if in_time => {
                total = total.checked_add(num.take()?)?;
            }
            // 'M' before 'T' would be months (unsupported), 'Y' is years (unsupported).
            _ => return None,
        }
    }
    if num.is_some() {
        return None;
    }
    Some(total)
}

/// A tiny SplitMix64 step — a fast, well-distributed integer mixer for deterministic jitter (no RNG
/// dependency, no per-process state).
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_forms_parse_and_advance() {
        for expr in ["@every 30m", "PT30M", "30m", "@every 1h30m"] {
            let s = Schedule::parse(expr).unwrap();
            assert!(!s.is_one_shot());
            let next = s.next_after(1000).unwrap();
            assert!(next > 1000, "{expr} should advance past now");
        }
        assert_eq!(
            Schedule::parse("@every 30m").unwrap().next_after(0),
            Some(1800)
        );
        assert_eq!(
            Schedule::parse("PT1H30M").unwrap().next_after(0),
            Some(5400)
        );
        assert_eq!(Schedule::parse("90s").unwrap().next_after(10), Some(100));
        assert_eq!(
            Schedule::parse("2w").unwrap().next_after(0),
            Some(1_209_600)
        );
        assert_eq!(Schedule::parse("P1W").unwrap().next_after(0), Some(604_800));
    }

    #[test]
    fn cron_next_occurrence_is_future_and_aligned() {
        // Daily at 09:00 UTC.
        let s = Schedule::parse("0 9 * * *").unwrap();
        // 2021-01-01T00:00:00Z = 1609459200; next 09:00 is +9h.
        let base = 1_609_459_200;
        assert_eq!(s.next_after(base), Some(base + 9 * 3600));
        // Period of a daily job is ~86400s; grace = clamp(43200, 120, 7200) = 7200.
        assert_eq!(s.grace_secs(base), 7200);
    }

    #[test]
    fn cron_timezone_shifts_the_fire() {
        let base = 1_609_459_200; // 2021-01-01T00:00:00Z
        let utc = Schedule::parse("0 9 * * *").unwrap();
        let berlin = Schedule::parse("0 9 * * *")
            .unwrap()
            .with_timezone(Some("Europe/Berlin"))
            .unwrap();
        // Berlin is UTC+1 in January, so 09:00 Berlin = 08:00 UTC, one hour earlier than 09:00 UTC.
        assert_eq!(utc.next_after(base), Some(base + 9 * 3600));
        assert_eq!(berlin.next_after(base), Some(base + 8 * 3600));
    }

    #[test]
    fn once_fires_then_exhausts() {
        // 2026-07-01T09:00:00Z
        let s = Schedule::parse("2026-07-01T09:00:00Z").unwrap();
        assert!(s.is_one_shot());
        let ts = s.next_after(0).unwrap();
        // After the fire instant there is no further occurrence.
        assert_eq!(s.next_after(ts), None);
        assert_eq!(s.next_after(ts + 1), None);
        // A bare date parses to midnight UTC.
        let d = Schedule::parse("2026-07-01").unwrap();
        assert!(d.is_one_shot());
        assert!(d.next_after(0).is_some());
    }

    #[test]
    fn invalid_inputs_error() {
        assert!(matches!(Schedule::parse("   "), Err(ScheduleError::Empty)));
        assert!(matches!(
            Schedule::parse("not a schedule at all !!"),
            Err(ScheduleError::Cron(_))
        ));
        assert!(matches!(
            Schedule::parse("0 9 * * *")
                .unwrap()
                .with_timezone(Some("Mars/Phobos")),
            Err(ScheduleError::Timezone(_))
        ));
        // Month/year ISO durations are unsupported (no anchor date).
        assert!(matches!(
            Schedule::parse("P1Y"),
            Err(ScheduleError::Duration(_))
        ));
        assert!(matches!(
            Schedule::parse("P1M"),
            Err(ScheduleError::Duration(_))
        ));
    }

    #[test]
    fn jitter_is_bounded_and_deterministic() {
        let s = Schedule::parse("0 * * * *").unwrap().with_jitter(Some(60));
        for seed in [0u64, 1, 42, 999_999] {
            let o = s.jitter_offset(seed);
            assert!(o <= 60);
            assert_eq!(o, s.jitter_offset(seed), "deterministic per seed");
        }
        // No jitter configured -> always 0.
        assert_eq!(Schedule::parse("0 * * * *").unwrap().jitter_offset(7), 0);
    }

    #[test]
    fn repeat_threads_through() {
        let s = Schedule::parse("@every 1h").unwrap().with_repeat(Some(3));
        assert_eq!(s.repeat(), Some(3));
    }
}
