// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Argument coercion + content slicing for the `lcm_*` tools (`LCM:tools.py:100-299`).
//!
//! The parsers replicate Python's coercion rules over JSON args: `int(x)` truncates floats and
//! parses numeric strings, `float(x)` additionally accepts booleans, and `str(x or "")` maps every
//! falsy value to the empty string. Divergences are per-helper documented.

use crate::tokens::Tokenizer;
use serde_json::Value;

/// `str(value or "")` (Python truthiness): `null`/`false`/`0`/`""`/`[]`/`{}` become `""`, `true`
/// renders `True`, numbers render as themselves, arrays/objects fall back to their JSON rendering
/// (Python would render `repr`, which no caller feeds on purpose).
pub(super) fn py_str_or_empty(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::Bool(false)) => String::new(),
        Some(Value::Bool(true)) => "True".to_string(),
        Some(Value::Number(n)) => {
            if n.as_f64() == Some(0.0) {
                String::new()
            } else {
                n.to_string()
            }
        }
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) if a.is_empty() => String::new(),
        Some(Value::Object(o)) if o.is_empty() => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Python truthiness over a JSON value (`if row.get("tool_calls"):` and friends).
pub(super) fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// How Python renders a raw arg into an error message (`f"Node {node_id} …"`): strings bare,
/// booleans capitalized, everything else via its JSON rendering.
pub(super) fn py_display(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

/// `int(value)` (Python): booleans are ints, floats truncate toward zero, strings parse after
/// trimming (no float strings). `None` when the coercion would raise.
pub(super) fn coerce_int(value: &Value) -> Option<i64> {
    match value {
        Value::Bool(b) => Some(i64::from(*b)),
        Value::Number(n) => n.as_i64().or_else(|| {
            n.as_f64()
                .filter(|f| f.is_finite())
                .map(|f| f.trunc() as i64)
        }),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

/// `_parse_int_value` (`LCM:tools.py:123`): coerce, falling back to `default`. A missing key uses
/// the default without coercion (Python's `args.get(key, default)`).
pub(super) fn parse_int_value(value: Option<&Value>, default: i64) -> i64 {
    match value {
        None => default,
        Some(v) => coerce_int(v).unwrap_or(default),
    }
}

/// `_parse_non_negative_int` (`LCM:tools.py:130`).
pub(super) fn parse_non_negative_int(value: Option<&Value>, default: i64) -> i64 {
    parse_int_value(value, default).max(0)
}

/// `_parse_positive_int` (`LCM:tools.py:134`).
pub(super) fn parse_positive_int(value: Option<&Value>, default: i64) -> i64 {
    parse_int_value(value, default).max(1)
}

/// `_parse_strict_int` (`LCM:tools.py:184`): like [`coerce_int`] but rejecting booleans; `Err` is
/// the Python error string.
pub(super) fn parse_strict_int(value: &Value, name: &str) -> Result<i64, String> {
    if value.is_boolean() {
        return Err(format!("{name} must be an integer"));
    }
    coerce_int(value).ok_or_else(|| format!("{name} must be an integer"))
}

/// `_parse_optional_float` (`LCM:tools.py:138`): `float(value)` — accepts numbers, booleans, and
/// numeric strings (including `inf`/`nan` spellings, which Rust's `f64::from_str` shares).
pub(super) fn parse_optional_float(
    value: Option<&Value>,
    name: &str,
) -> Result<Option<f64>, String> {
    let Some(value) = value else { return Ok(None) };
    if value.is_null() {
        return Ok(None);
    }
    let parsed = match value {
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    };
    parsed
        .map(Some)
        .ok_or_else(|| format!("{name} must be a number"))
}

/// `_parse_optional_timestamp` (`LCM:tools.py:147`): a Unix-seconds number, a numeric string, or a
/// timezone-aware ISO 8601 string (a trailing `Z` normalizes to `+00:00`). Naive ISO datetimes are
/// rejected with the Python error text.
pub(super) fn parse_optional_timestamp(
    value: Option<&Value>,
    name: &str,
) -> Result<Option<f64>, String> {
    let Some(value) = value else { return Ok(None) };
    if value.is_null() {
        return Ok(None);
    }
    let type_err = || format!("{name} must be a Unix timestamp or timezone-aware ISO 8601 string");
    match value {
        Value::Bool(_) => Err(type_err()),
        Value::Number(n) => n.as_f64().map(Some).ok_or_else(type_err),
        _ => {
            let text = py_display(value);
            let text = text.trim();
            if text.is_empty() {
                return Err(format!("{name} must not be empty"));
            }
            if let Ok(v) = text.parse::<f64>() {
                return Ok(Some(v));
            }
            match parse_iso8601(text) {
                IsoParse::Aware(ts) => Ok(Some(ts)),
                IsoParse::Naive => Err(format!(
                    "{name} ISO timestamp must include a timezone offset or Z"
                )),
                IsoParse::Invalid => Err(type_err()),
            }
        }
    }
}

/// `_parse_grep_role` (`LCM:tools.py:174`).
pub(super) fn parse_grep_role(value: Option<&Value>) -> Result<Option<String>, String> {
    let Some(value) = value else { return Ok(None) };
    if value.is_null() {
        return Ok(None);
    }
    let role = py_str_or_empty(Some(value)).trim().to_string();
    const VALID: [&str; 5] = ["system", "user", "assistant", "tool", "unknown"];
    if VALID.contains(&role.as_str()) {
        Ok(Some(role))
    } else {
        Err("role must be one of: system, user, assistant, tool, unknown".to_string())
    }
}

/// `_parse_load_session_roles` (`LCM:tools.py:627`): an array of non-empty strings (each item
/// coerced via `str(item or "")`), deduplicated preserving order.
pub(super) fn parse_load_session_roles(value: Option<&Value>) -> Result<Vec<String>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let Some(items) = value.as_array() else {
        return Err("roles must be an array of strings".to_string());
    };
    let mut roles: Vec<String> = Vec::new();
    for item in items {
        let role = py_str_or_empty(Some(item)).trim().to_string();
        if role.is_empty() {
            return Err("roles must contain only non-empty strings".to_string());
        }
        if !roles.contains(&role) {
            roles.push(role);
        }
    }
    Ok(roles)
}

/// The outcome of parsing an ISO 8601 datetime string.
pub(super) enum IsoParse {
    /// Timezone-aware: the Unix timestamp in seconds.
    Aware(f64),
    /// A valid datetime without any UTC offset (Python rejects these for grep time bounds).
    Naive,
    /// Not a parseable ISO 8601 datetime.
    Invalid,
}

/// A hand-rolled subset of `datetime.fromisoformat` (the crate deliberately avoids a chrono
/// dependency): `YYYY-MM-DD[( |T)HH:MM[:SS[.ffffff]]][±HH[:]MM | ±HH | Z-normalized offset]`.
/// Python 3.11 accepts more exotic spellings (ordinal dates, `HH` alone); the tools only promise
/// the documented "ISO 8601 with timezone" contract.
pub(super) fn parse_iso8601(text: &str) -> IsoParse {
    let normalized = if let Some(stripped) = text.strip_suffix(['Z', 'z']) {
        format!("{stripped}+00:00")
    } else {
        text.to_string()
    };
    // Split off a UTC offset: a '+' anywhere after the date part, or a '-' after position 10
    // (earlier '-' are date separators).
    let bytes = normalized.as_bytes();
    let mut offset_pos: Option<usize> = None;
    for (i, b) in bytes.iter().enumerate().skip(10) {
        if *b == b'+' || *b == b'-' {
            offset_pos = Some(i);
            break;
        }
    }
    let (datetime_part, offset_secs) = match offset_pos {
        Some(pos) => {
            let Some(off) = parse_utc_offset(&normalized[pos..]) else {
                return IsoParse::Invalid;
            };
            (&normalized[..pos], Some(off))
        }
        None => (normalized.as_str(), None),
    };
    let Some(epoch_naive) = parse_naive_datetime(datetime_part) else {
        return IsoParse::Invalid;
    };
    match offset_secs {
        Some(off) => IsoParse::Aware(epoch_naive - off as f64),
        None => IsoParse::Naive,
    }
}

/// Parse `±HH:MM`, `±HHMM`, or `±HH` into signed seconds.
fn parse_utc_offset(s: &str) -> Option<i64> {
    let (sign, rest) = match s.as_bytes().first()? {
        b'+' => (1i64, &s[1..]),
        b'-' => (-1i64, &s[1..]),
        _ => return None,
    };
    let (hh, mm) = match rest.len() {
        2 => (rest.parse::<i64>().ok()?, 0),
        4 => (
            rest[..2].parse::<i64>().ok()?,
            rest[2..].parse::<i64>().ok()?,
        ),
        5 if rest.as_bytes()[2] == b':' => (
            rest[..2].parse::<i64>().ok()?,
            rest[3..].parse::<i64>().ok()?,
        ),
        _ => return None,
    };
    if !(0..=23).contains(&hh) || !(0..=59).contains(&mm) {
        return None;
    }
    Some(sign * (hh * 3600 + mm * 60))
}

/// Parse `YYYY-MM-DD[( |T)HH:MM[:SS[.frac]]]` into naive Unix seconds (as if UTC).
fn parse_naive_datetime(s: &str) -> Option<f64> {
    if s.len() < 10 {
        return None;
    }
    let (date, time) = s.split_at(10);
    let b = date.as_bytes();
    if b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let year: i64 = date[..4].parse().ok()?;
    let month: i64 = date[5..7].parse().ok()?;
    let day: i64 = date[8..10].parse().ok()?;
    if !(1..=12).contains(&month) || day < 1 || day > days_in_month(year, month) {
        return None;
    }
    let mut seconds = days_from_civil(year, month, day) as f64 * 86_400.0;
    if time.is_empty() {
        return Some(seconds);
    }
    let time = time.strip_prefix(['T', 't', ' '])?;
    let hh: i64 = time.get(..2)?.parse().ok()?;
    if time.as_bytes().get(2) != Some(&b':') {
        return None;
    }
    let mi: i64 = time.get(3..5)?.parse().ok()?;
    if !(0..=23).contains(&hh) || !(0..=59).contains(&mi) {
        return None;
    }
    seconds += (hh * 3600 + mi * 60) as f64;
    let rest = &time[5..];
    if rest.is_empty() {
        return Some(seconds);
    }
    let rest = rest.strip_prefix(':')?;
    let ss: i64 = rest.get(..2)?.parse().ok()?;
    if !(0..=59).contains(&ss) {
        return None;
    }
    seconds += ss as f64;
    let frac = &rest[2..];
    if frac.is_empty() {
        return Some(seconds);
    }
    let digits = frac.strip_prefix('.')?;
    if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let fraction: f64 = format!("0.{digits}").parse().ok()?;
    Some(seconds + fraction)
}

/// Days in `month` of `year` (proleptic Gregorian).
fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Howard Hinnant's `days_from_civil`: days since 1970-01-01 for a proleptic-Gregorian date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `_truncate_text_to_token_budget` (`LCM:tools.py:100`): the largest character prefix whose token
/// count fits `max_tokens`. Returns `(prefix, was_truncated)`; a non-positive budget returns
/// `("", text_was_nonempty)`.
pub(super) fn truncate_text_to_token_budget(
    tok: &Tokenizer,
    text: &str,
    max_tokens: i64,
) -> (String, bool) {
    if max_tokens <= 0 || text.is_empty() {
        return (String::new(), !text.is_empty());
    }
    if tok.count_text(text) as i64 <= max_tokens {
        return (text.to_string(), false);
    }
    let chars: Vec<char> = text.chars().collect();
    let (mut low, mut high) = (0i64, chars.len() as i64);
    let mut best = String::new();
    while low <= high {
        let mid = (low + high) / 2;
        let candidate: String = chars[..mid as usize].iter().collect();
        if tok.count_text(&candidate) as i64 <= max_tokens {
            best = candidate;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    (best, true)
}

/// One paged content slice (`_slice_content_for_response` / `_full_content_slice`,
/// `LCM:tools.py:201-235`). All offsets/counts are in characters (Python `len`).
pub(super) struct ContentSlice {
    pub content: String,
    pub content_chars: usize,
    pub content_offset: usize,
    pub content_returned_chars: usize,
    pub content_truncated: bool,
    pub next_content_offset: usize,
    pub has_more: bool,
}

/// `_slice_content_for_response` (`LCM:tools.py:201`): token-budgeted slice from a character
/// offset, guaranteeing at least one character of progress when anything remains.
pub(super) fn slice_content_for_response(
    tok: &Tokenizer,
    content: &str,
    max_tokens: i64,
    content_offset: usize,
) -> ContentSlice {
    let chars: Vec<char> = content.chars().collect();
    let content_offset = content_offset.min(chars.len());
    let tail: String = chars[content_offset..].iter().collect();
    let (mut sliced, _) = truncate_text_to_token_budget(tok, &tail, max_tokens);
    if sliced.is_empty() && content_offset < chars.len() {
        // A tiny token budget can fail to fit even the next character. Return one character anyway
        // so callers make deterministic, lossless cursor progress instead of receiving
        // has_more=true with the same content_offset forever.
        sliced = chars[content_offset].to_string();
    }
    let returned = sliced.chars().count();
    let next_content_offset = content_offset + returned;
    let has_more = next_content_offset < chars.len();
    ContentSlice {
        content: sliced,
        content_chars: chars.len(),
        content_offset,
        content_returned_chars: returned,
        content_truncated: has_more,
        next_content_offset: if has_more { next_content_offset } else { 0 },
        has_more,
    }
}

/// `_full_content_slice` (`LCM:tools.py:223`): everything from the offset, never marked truncated
/// (compact placeholder rows bypass the token budget).
pub(super) fn full_content_slice(content: &str, content_offset: usize) -> ContentSlice {
    let chars: Vec<char> = content.chars().collect();
    let content_offset = content_offset.min(chars.len());
    let sliced: String = chars[content_offset..].iter().collect();
    let returned = chars.len() - content_offset;
    ContentSlice {
        content: sliced,
        content_chars: chars.len(),
        content_offset,
        content_returned_chars: returned,
        content_truncated: false,
        next_content_offset: 0,
        has_more: false,
    }
}

/// `_slice_loaded_content` (`LCM:tools.py:644`): the char-capped `lcm_load_session` slice.
pub(super) fn slice_loaded_content(content: &str, max_content_chars: usize) -> ContentSlice {
    let chars: Vec<char> = content.chars().collect();
    let take = max_content_chars.min(chars.len());
    let sliced: String = chars[..take].iter().collect();
    let has_more = take < chars.len();
    ContentSlice {
        content: sliced,
        content_chars: chars.len(),
        content_offset: 0,
        content_returned_chars: take,
        content_truncated: has_more,
        next_content_offset: if has_more { take } else { 0 },
        has_more,
    }
}

/// `%.3g` (the Python timeout formatting in the `lcm_expand_query` degraded reason): three
/// significant digits, trailing zeros trimmed.
pub(super) fn format_sig3(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let formatted = format!("{value:.2e}");
    let (mantissa, exp) = formatted
        .split_once('e')
        .expect("{:e} always carries an exponent");
    let exp: i32 = exp.parse().expect("float exponent parses");
    // %g switches to scientific notation outside [1e-5, 1e+3) (relative to precision 3).
    if !(-5..3).contains(&exp) {
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        let sign = if exp < 0 { "-" } else { "+" };
        return format!("{mantissa}e{sign}{:02}", exp.abs());
    }
    let decimals = (2 - exp).max(0) as usize;
    let fixed = format!("{value:.decimals$}");
    if fixed.contains('.') {
        fixed
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    } else {
        fixed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn int_coercion_matches_python() {
        assert_eq!(coerce_int(&json!(3.9)), Some(3));
        assert_eq!(coerce_int(&json!("42")), Some(42));
        assert_eq!(coerce_int(&json!(" 7 ")), Some(7));
        assert_eq!(coerce_int(&json!(true)), Some(1));
        assert_eq!(coerce_int(&json!("3.5")), None);
        assert_eq!(coerce_int(&json!(null)), None);
        assert_eq!(parse_int_value(None, 10), 10);
        assert_eq!(parse_int_value(Some(&json!("nope")), 10), 10);
        assert!(parse_strict_int(&json!(true), "limit").is_err());
        assert_eq!(parse_strict_int(&json!("12"), "limit"), Ok(12));
    }

    #[test]
    fn optional_timestamp_accepts_aware_iso_and_rejects_naive() {
        // Epoch reference: 2021-03-01T00:00:00+00:00 == 1614556800.
        let ts = parse_optional_timestamp(Some(&json!("2021-03-01T00:00:00Z")), "time_from")
            .unwrap()
            .unwrap();
        assert_eq!(ts, 1_614_556_800.0);
        let offset =
            parse_optional_timestamp(Some(&json!("2021-03-01T01:30:00+01:30")), "time_from")
                .unwrap()
                .unwrap();
        assert_eq!(offset, 1_614_556_800.0);
        let naive = parse_optional_timestamp(Some(&json!("2021-03-01T00:00:00")), "time_from");
        assert_eq!(
            naive.unwrap_err(),
            "time_from ISO timestamp must include a timezone offset or Z"
        );
        let junk = parse_optional_timestamp(Some(&json!("not a time")), "time_to");
        assert_eq!(
            junk.unwrap_err(),
            "time_to must be a Unix timestamp or timezone-aware ISO 8601 string"
        );
        // Numeric strings and numbers pass straight through.
        assert_eq!(
            parse_optional_timestamp(Some(&json!("123.5")), "t").unwrap(),
            Some(123.5)
        );
        assert_eq!(
            parse_optional_timestamp(Some(&json!(9.0)), "t").unwrap(),
            Some(9.0)
        );
        // Booleans are rejected (Python's isinstance(value, bool) guard).
        assert!(parse_optional_timestamp(Some(&json!(true)), "t").is_err());
        // Fractional seconds survive.
        let frac = parse_optional_timestamp(Some(&json!("2021-03-01T00:00:00.250Z")), "t")
            .unwrap()
            .unwrap();
        assert_eq!(frac, 1_614_556_800.25);
    }

    #[test]
    fn leap_day_and_month_bounds_validate() {
        assert!(matches!(
            parse_iso8601("2024-02-29T00:00:00+00:00"),
            IsoParse::Aware(_)
        ));
        assert!(matches!(
            parse_iso8601("2023-02-29T00:00:00+00:00"),
            IsoParse::Invalid
        ));
        assert!(matches!(
            parse_iso8601("2023-13-01T00:00:00+00:00"),
            IsoParse::Invalid
        ));
    }

    #[test]
    fn slice_content_guarantees_progress_on_tiny_budget() {
        let tok = Tokenizer::heuristic();
        let s = slice_content_for_response(&tok, "abcdefgh", 0, 0);
        assert_eq!(s.content, "a", "one char of forced progress");
        assert!(s.has_more);
        assert_eq!(s.next_content_offset, 1);
        let done = slice_content_for_response(&tok, "abc", 1_000, 0);
        assert!(!done.has_more);
        assert_eq!(done.next_content_offset, 0, "Python reports 0, not null");
    }

    #[test]
    fn sig3_formatting_matches_python_percent_g() {
        assert_eq!(format_sig3(120.0), "120");
        assert_eq!(format_sig3(1.5), "1.5");
        assert_eq!(format_sig3(0.0005), "0.0005");
        assert_eq!(format_sig3(12345.0), "1.23e+04");
        assert_eq!(format_sig3(60.0), "60");
    }
}
