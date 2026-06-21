//! Natural-language temporal parsing — port of `temporal_parser.py`.
//!
//! [`parse_nl_date`] runs a first-match priority chain (ISO -> slash EU/US -> named month -> relative
//! today/yesterday/tomorrow -> `last|this|next <weekday>` -> bare weekday -> week/month/year ->
//! `N units ago`/`in N units` -> vague), and [`extract_temporal`] wraps it with named-time-of-day
//! tagging (`temporal_parser.py` L106-L390). Deterministic; no LLM.

use chrono::{Datelike, Duration, NaiveDate, Utc, Weekday};
use regex::Regex;
use std::sync::OnceLock;

/// The result of temporal extraction (`temporal_parser.py` `extract_temporal` L385-L389).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Temporal {
    /// Resolved ISO date (`YYYY-MM-DD`), if any.
    pub event_date: Option<String>,
    /// Precision: `day | week | month | year | relative | unknown`.
    pub event_date_precision: String,
    /// Extracted temporal tags.
    pub temporal_tags: Vec<String>,
    /// The first/primary tag, if any.
    pub primary_signal: Option<String>,
}

/// Named times of day (`temporal_parser.py` `NAMED_TIMES` L48-L57) — tag-only signals.
const NAMED_TIMES: &[&str] = &[
    "morning", "afternoon", "evening", "night", "midnight", "noon", "dawn", "dusk",
];

/// Day name -> weekday (`temporal_parser.py` `DAY_MAP` L34).
fn day_map(name: &str) -> Option<Weekday> {
    Some(match name {
        "monday" | "mon" => Weekday::Mon,
        "tuesday" | "tue" => Weekday::Tue,
        "wednesday" | "wed" => Weekday::Wed,
        "thursday" | "thu" => Weekday::Thu,
        "friday" | "fri" => Weekday::Fri,
        "saturday" | "sat" => Weekday::Sat,
        "sunday" | "sun" => Weekday::Sun,
        _ => return None,
    })
}

/// Month name -> month number (`temporal_parser.py` `MONTH_MAP` L40).
fn month_map(name: &str) -> Option<u32> {
    Some(match name {
        "january" | "jan" => 1,
        "february" | "feb" => 2,
        "march" | "mar" => 3,
        "april" | "apr" => 4,
        "may" => 5,
        "june" | "jun" => 6,
        "july" | "jul" => 7,
        "august" | "aug" => 8,
        "september" | "sep" => 9,
        "october" | "oct" => 10,
        "november" | "nov" => 11,
        "december" | "dec" => 12,
        _ => return None,
    })
}

struct Patterns {
    iso: Regex,
    slash: Regex,
    named_month: Regex,
    rel_weekday: Regex,
    bare_weekday: Regex,
    week_month_year: Regex,
    ago: Regex,
    in_future: Regex,
}

fn patterns() -> &'static Patterns {
    static P: OnceLock<Patterns> = OnceLock::new();
    P.get_or_init(|| Patterns {
        iso: Regex::new(r"\b(\d{4})-(\d{2})-(\d{2})\b").unwrap(),
        slash: Regex::new(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b").unwrap(),
        named_month: Regex::new(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b",
        )
        .unwrap(),
        rel_weekday: Regex::new(
            r"\b(last|this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b",
        )
        .unwrap(),
        bare_weekday: Regex::new(
            r"\b(?:on\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        )
        .unwrap(),
        week_month_year: Regex::new(r"\b(this|last|next)\s+(week|month|year)\b").unwrap(),
        ago: Regex::new(
            r"\b(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+(ago|before|earlier|back)\b",
        )
        .unwrap(),
        in_future: Regex::new(
            r"\bin\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
        )
        .unwrap(),
    })
}

/// Lowercased full weekday name (`%A` in Python).
fn weekday_name(d: NaiveDate) -> String {
    match d.weekday() {
        Weekday::Mon => "monday",
        Weekday::Tue => "tuesday",
        Weekday::Wed => "wednesday",
        Weekday::Thu => "thursday",
        Weekday::Fri => "friday",
        Weekday::Sat => "saturday",
        Weekday::Sun => "sunday",
    }
    .to_string()
}

fn iso(d: NaiveDate) -> String {
    d.format("%Y-%m-%d").to_string()
}

/// `week-{isoweek}-{year}` (Python `f"week-{isocalendar()[1]}-{d.year}"`).
fn week_tag(d: NaiveDate) -> String {
    format!("week-{}-{}", d.iso_week().week(), d.year())
}

/// Day-grain tags for an absolute date: `[iso, week-W-Y, weekday]`.
fn day_tags(d: NaiveDate) -> Vec<String> {
    vec![iso(d), week_tag(d), weekday_name(d)]
}

/// Resolve a `this|last|next <weekday>` reference to a concrete date (`temporal_parser.py`
/// `_resolve_relative_day` L60-L103).
fn resolve_relative_day(reference: NaiveDate, day: Weekday, qualifier: &str) -> NaiveDate {
    let current = reference.weekday().num_days_from_monday() as i64;
    let target = day.num_days_from_monday() as i64;
    match qualifier {
        "this" => {
            let diff = (current - target).rem_euclid(7);
            if diff == 0 {
                reference
            } else {
                reference - Duration::days(diff)
            }
        }
        "last" => {
            let diff = (current - target + 7).rem_euclid(7) + 7;
            reference - Duration::days(diff)
        }
        "next" => {
            let mut diff = (target - current).rem_euclid(7);
            if diff == 0 {
                diff = 7;
            }
            reference + Duration::days(diff)
        }
        _ => reference,
    }
}

fn first_of_month(year: i32, month: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(year, month, 1).expect("first-of-month is always valid")
}

/// Parse a natural-language date expression against `reference`, returning
/// `(date, precision, tags)` or `None` (`temporal_parser.py` `parse_nl_date` L106-L354).
pub fn parse_nl_date(text: &str, reference: NaiveDate) -> Option<(NaiveDate, String, Vec<String>)> {
    let p = patterns();
    let lower = text.to_lowercase();

    // ---- Absolute: ISO ----
    if let Some(m) = p.iso.captures(text) {
        if let Some(d) = NaiveDate::from_ymd_opt(
            m[1].parse().ok()?,
            m[2].parse().ok()?,
            m[3].parse().ok()?,
        ) {
            return Some((d, "day".into(), day_tags(d)));
        }
    }

    // ---- Absolute: slash EU/US ----
    if let Some(m) = p.slash.captures(text) {
        let a: u32 = m[1].parse().ok()?;
        let b: u32 = m[2].parse().ok()?;
        let mut y: i32 = m[3].parse().ok()?;
        if y < 100 {
            y += 2000;
        }
        // First number > 12 => day/month/year (EU), else month/day/year (US).
        let built = if a > 12 {
            NaiveDate::from_ymd_opt(y, b, a)
        } else {
            NaiveDate::from_ymd_opt(y, a, b)
        };
        if let Some(d) = built {
            return Some((d, "day".into(), day_tags(d)));
        }
    }

    // ---- Absolute: named month + day (+ optional year) ----
    if let Some(m) = p.named_month.captures(&lower) {
        let month = month_map(&m[1]).unwrap_or(1);
        let day: u32 = m[2].parse().ok()?;
        let year: i32 = m
            .get(3)
            .and_then(|y| y.as_str().parse().ok())
            .unwrap_or_else(|| reference.year());
        if let Some(d) = NaiveDate::from_ymd_opt(year, month, day) {
            return Some((d, "day".into(), day_tags(d)));
        }
    }

    // ---- Relative day words ----
    if word(&lower, "today") {
        let d = reference;
        return Some((d, "day".into(), vec![iso(d), weekday_name(d)]));
    }
    if word(&lower, "yesterday") {
        let d = reference - Duration::days(1);
        return Some((
            d,
            "day".into(),
            vec![iso(d), weekday_name(d), "yesterday".into()],
        ));
    }
    if word(&lower, "tomorrow") {
        let d = reference + Duration::days(1);
        return Some((
            d,
            "day".into(),
            vec![iso(d), weekday_name(d), "tomorrow".into()],
        ));
    }

    // ---- last|this|next <weekday> ----
    if let Some(m) = p.rel_weekday.captures(&lower) {
        let qualifier = &m[1];
        let day_name = &m[2];
        if let Some(day) = day_map(day_name) {
            let d = resolve_relative_day(reference, day, qualifier);
            return Some((
                d,
                "day".into(),
                vec![iso(d), week_tag(d), day_name.to_string(), qualifier.to_string()],
            ));
        }
    }

    // ---- bare weekday ("on Monday" / "monday") ----
    if let Some(m) = p.bare_weekday.captures(&lower) {
        let day_name = &m[1];
        if let Some(day) = day_map(day_name) {
            let d = resolve_relative_day(reference, day, "this");
            return Some((
                d,
                "day".into(),
                vec![iso(d), week_tag(d), day_name.to_string()],
            ));
        }
    }

    // ---- this|last|next week|month|year ----
    if let Some(m) = p.week_month_year.captures(&lower) {
        let qualifier = &m[1];
        let unit = &m[2];
        return Some(week_month_year(reference, qualifier, unit));
    }

    // ---- N units ago / before / earlier / back ----
    if let Some(m) = p.ago.captures(&lower) {
        let num: i64 = m[1].parse().ok()?;
        let unit = &m[2];
        let d = (reference.and_hms_opt(0, 0, 0)? - unit_delta(num, unit)?).date();
        let precision = if unit == "day" || unit == "hour" { "day" } else { "week" };
        return Some((d, precision.into(), vec![iso(d), format!("{num}-{unit}s-ago")]));
    }

    // ---- in N units (future) ----
    if let Some(m) = p.in_future.captures(&lower) {
        let num: i64 = m[1].parse().ok()?;
        let unit = &m[2];
        let d = (reference.and_hms_opt(0, 0, 0)? + unit_delta(num, unit)?).date();
        let precision = if unit == "day" || unit == "hour" { "day" } else { "week" };
        return Some((d, precision.into(), vec![iso(d), format!("in-{num}-{unit}s")]));
    }

    // ---- Vague ----
    if word(&lower, "recently") || word(&lower, "lately") || lower.contains("not long ago") {
        return Some((reference, "relative".into(), vec!["recently".into()]));
    }
    if lower.contains("a while ago") || lower.contains("some time ago") || lower.contains("long ago")
    {
        return Some((reference, "relative".into(), vec!["vague".into()]));
    }

    None
}

/// Build the `this|last|next week|month|year` result (`temporal_parser.py` L233-L281).
fn week_month_year(reference: NaiveDate, qualifier: &str, unit: &str) -> (NaiveDate, String, Vec<String>) {
    match (qualifier, unit) {
        ("this", "week") => (reference, "week".into(), vec![week_tag(reference), "this-week".into()]),
        ("this", "month") => (
            reference,
            "month".into(),
            vec![format!("{}-{:02}", reference.year(), reference.month()), "this-month".into()],
        ),
        ("this", "year") => (
            reference,
            "year".into(),
            vec![reference.year().to_string(), "this-year".into()],
        ),
        ("last", "week") => {
            let d = reference - Duration::weeks(1);
            (d, "week".into(), vec![week_tag(d), "last-week".into()])
        }
        ("last", "month") => {
            let d = if reference.month() == 1 {
                first_of_month(reference.year() - 1, 12)
            } else {
                first_of_month(reference.year(), reference.month() - 1)
            };
            (d, "month".into(), vec![format!("{}-{:02}", d.year(), d.month()), "last-month".into()])
        }
        ("last", "year") => {
            let d = first_of_month(reference.year() - 1, 1);
            (d, "year".into(), vec![d.year().to_string(), "last-year".into()])
        }
        ("next", "week") => {
            let d = reference + Duration::weeks(1);
            (d, "week".into(), vec![week_tag(d), "next-week".into()])
        }
        ("next", "month") => {
            let d = if reference.month() == 12 {
                first_of_month(reference.year() + 1, 1)
            } else {
                first_of_month(reference.year(), reference.month() + 1)
            };
            (d, "month".into(), vec![format!("{}-{:02}", d.year(), d.month()), "next-month".into()])
        }
        ("next", "year") => {
            let d = first_of_month(reference.year() + 1, 1);
            (d, "year".into(), vec![d.year().to_string(), "next-year".into()])
        }
        _ => (reference, "unknown".into(), Vec::new()),
    }
}

/// The `timedelta` for `num` of `unit` (`temporal_parser.py` L293-L307): months ~ 30d, years ~ 365d.
fn unit_delta(num: i64, unit: &str) -> Option<Duration> {
    Some(match unit {
        "second" => Duration::seconds(num),
        "minute" => Duration::minutes(num),
        "hour" => Duration::hours(num),
        "day" => Duration::days(num),
        "week" => Duration::weeks(num),
        "month" => Duration::days(num * 30),
        "year" => Duration::days(num * 365),
        _ => return None,
    })
}

/// Whole-word containment (`\b<word>\b`), avoiding substring false-positives.
fn word(haystack: &str, w: &str) -> bool {
    haystack.split(|c: char| !c.is_alphanumeric()).any(|t| t == w)
}

/// Extract temporal info using the current UTC date as the reference.
pub fn extract_temporal(text: &str) -> Temporal {
    extract_temporal_with_ref(text, Utc::now().date_naive())
}

/// Extract temporal info against an explicit reference date (testable;
/// `temporal_parser.py` `extract_temporal` L357-L390).
pub fn extract_temporal_with_ref(text: &str, reference: NaiveDate) -> Temporal {
    let lower = text.to_lowercase();
    let mut time_tags: Vec<String> = Vec::new();
    for name in NAMED_TIMES {
        if word(&lower, name) {
            time_tags.push((*name).to_string());
            break;
        }
    }

    match parse_nl_date(text, reference) {
        None => {
            let primary = time_tags.first().cloned();
            Temporal {
                event_date: None,
                event_date_precision: "unknown".into(),
                temporal_tags: time_tags,
                primary_signal: primary,
            }
        }
        Some((d, precision, mut tags)) => {
            tags.extend(time_tags);
            let primary = tags.first().cloned();
            Temporal {
                event_date: Some(iso(d)),
                event_date_precision: precision,
                temporal_tags: tags,
                primary_signal: primary,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refd() -> NaiveDate {
        // Monday, 2026-05-18 (ISO week 21).
        NaiveDate::from_ymd_opt(2026, 5, 18).unwrap()
    }

    #[test]
    fn iso_absolute() {
        let t = extract_temporal_with_ref("ship on 2026-05-20", refd());
        assert_eq!(t.event_date.as_deref(), Some("2026-05-20"));
        assert_eq!(t.event_date_precision, "day");
        assert!(t.temporal_tags.contains(&"wednesday".to_string()));
        assert!(t.temporal_tags.iter().any(|x| x.starts_with("week-21-2026")));
    }

    #[test]
    fn invalid_iso_falls_through() {
        // 2026-02-29 is invalid; with no other signal -> unknown.
        let t = extract_temporal_with_ref("the date 2026-02-29 is bogus", refd());
        assert_eq!(t.event_date, None);
        assert_eq!(t.event_date_precision, "unknown");
    }

    #[test]
    fn slash_eu_vs_us() {
        // 20/05/2026 -> EU day/month (first > 12).
        let eu = parse_nl_date("on 20/05/2026", refd()).unwrap();
        assert_eq!(iso(eu.0), "2026-05-20");
        // 05/06/2026 -> US month/day.
        let us = parse_nl_date("on 05/06/2026", refd()).unwrap();
        assert_eq!(iso(us.0), "2026-05-06");
    }

    #[test]
    fn named_month_with_and_without_year() {
        let with = parse_nl_date("May 20, 2026 launch", refd()).unwrap();
        assert_eq!(iso(with.0), "2026-05-20");
        let without = parse_nl_date("due August 3rd", refd()).unwrap();
        assert_eq!(iso(without.0), "2026-08-03");
    }

    #[test]
    fn relative_today_yesterday_tomorrow() {
        assert_eq!(iso(parse_nl_date("today", refd()).unwrap().0), "2026-05-18");
        let y = parse_nl_date("yesterday", refd()).unwrap();
        assert_eq!(iso(y.0), "2026-05-17");
        assert!(y.2.contains(&"yesterday".to_string()));
        assert_eq!(iso(parse_nl_date("tomorrow", refd()).unwrap().0), "2026-05-19");
    }

    #[test]
    fn relative_weekdays() {
        // Reference is Monday 2026-05-18.
        // last Monday -> previous week's Monday (7 days back).
        assert_eq!(iso(parse_nl_date("last monday", refd()).unwrap().0), "2026-05-11");
        // this Friday -> most recent Friday on/before ref => 2026-05-15.
        assert_eq!(iso(parse_nl_date("this friday", refd()).unwrap().0), "2026-05-15");
        // next Friday -> upcoming Friday => 2026-05-22.
        assert_eq!(iso(parse_nl_date("next friday", refd()).unwrap().0), "2026-05-22");
        // next Monday when ref is Monday -> +7.
        assert_eq!(iso(parse_nl_date("next monday", refd()).unwrap().0), "2026-05-25");
    }

    #[test]
    fn week_month_year_units() {
        let lw = parse_nl_date("last week", refd()).unwrap();
        assert_eq!(lw.1, "week");
        assert!(lw.2.contains(&"last-week".to_string()));
        let nm = parse_nl_date("next month", refd()).unwrap();
        assert_eq!(iso(nm.0), "2026-06-01");
        assert_eq!(nm.1, "month");
        let ly = parse_nl_date("last year", refd()).unwrap();
        assert_eq!(iso(ly.0), "2025-01-01");
        assert_eq!(ly.1, "year");
    }

    #[test]
    fn intervals_ago_and_future() {
        let ago = parse_nl_date("3 days ago", refd()).unwrap();
        assert_eq!(iso(ago.0), "2026-05-15");
        assert!(ago.2.contains(&"3-days-ago".to_string()));
        let future = parse_nl_date("in 2 weeks", refd()).unwrap();
        assert_eq!(iso(future.0), "2026-06-01");
        assert!(future.2.contains(&"in-2-weeks".to_string()));
    }

    #[test]
    fn vague_and_named_time() {
        let v = extract_temporal_with_ref("we talked recently", refd());
        assert_eq!(v.event_date_precision, "relative");
        assert!(v.temporal_tags.contains(&"recently".to_string()));
        // Named time of day attaches even with a date.
        let t = extract_temporal_with_ref("met last monday morning", refd());
        assert!(t.temporal_tags.contains(&"morning".to_string()));
        // Named time alone (no date) -> unknown precision + the time tag.
        let only = extract_temporal_with_ref("see you in the evening", refd());
        assert_eq!(only.event_date_precision, "unknown");
        assert_eq!(only.primary_signal.as_deref(), Some("evening"));
    }

    #[test]
    fn no_signal_is_none() {
        let t = extract_temporal_with_ref("just a normal sentence", refd());
        assert_eq!(t.event_date, None);
        assert_eq!(t.event_date_precision, "unknown");
        assert!(t.temporal_tags.is_empty());
    }
}
