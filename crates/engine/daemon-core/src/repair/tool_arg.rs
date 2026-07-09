// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Tool-argument JSON repair (§9).
//!
//! Models emit tool-call arguments as JSON text that is frequently *almost* valid: wrapped in a
//! markdown code fence, carrying trailing commas, single-quoted, or truncated mid-object when the
//! response was cut off. [`repair_tool_args`] runs a small multi-pass repair and, on success,
//! re-serializes through `serde_json` so the tool always receives canonical JSON (stable key order,
//! no stray whitespace).
//!
//! The repair pipeline is a faithful port of hermes' `_repair_tool_call_arguments`
//! (`agent/message_sanitization.py:185`): empty/whitespace and the Python literal `None`
//! collapse to `{}`; a lenient control-char-in-string parse recovers llama.cpp-style payloads;
//! trailing commas are stripped, unclosed structures closed, and excess closers trimmed (bounded);
//! and unrepairable input falls back to `{}` (rather than being passed through) so a malformed
//! payload can never crash the upstream API with an "invalid tool call arguments" 400.

/// The outcome of repairing a tool-argument payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgRepair {
    /// The repaired (and canonicalized) argument string, or the original if repair failed.
    pub args: String,
    /// Whether any repair pass changed the input.
    pub repaired: bool,
    /// Whether the input looked truncated (an unbalanced/cut-off payload).
    pub truncated: bool,
}

/// Repair and canonicalize a tool-argument JSON payload (§9), a faithful port of hermes'
/// `_repair_tool_call_arguments`. The passes, in order:
/// 1. empty / whitespace-only → `{}`;
/// 2. the Python literal `None` → `{}`;
/// 3. parse as-is (already valid JSON);
/// 4. strip a surrounding markdown code fence and parse (a Rust-only convenience pass);
/// 5. lenient parse of raw control chars inside strings (emulates Python's `json.loads(strict=False)`);
/// 6. strip trailing commas, close unclosed structures, trim excess closers (bounded 50), and parse;
/// 7. escape control chars inside strings then parse;
/// 8. unrepairable → `{}` fallback.
///
/// The first pass that parses wins and its value is re-serialized canonically. Unlike Python
/// (which preserves object insertion order), the Rust port sorts object keys for determinism.
pub fn repair_tool_args(raw: &str) -> ArgRepair {
    let truncated = looks_truncated(raw);
    let raw_stripped = raw.trim();

    // Pass 1: empty / whitespace-only → empty object.
    if raw_stripped.is_empty() {
        return ArgRepair {
            args: "{}".to_string(),
            repaired: true,
            truncated: false,
        };
    }

    // Pass 2: the Python literal `None` → empty object.
    if raw_stripped == "None" {
        return ArgRepair {
            args: "{}".to_string(),
            repaired: true,
            truncated: false,
        };
    }

    // Pass 3: already valid JSON.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(raw_stripped) {
        return ArgRepair {
            args: canonicalize(&value),
            repaired: raw_stripped != raw || !is_canonical(raw, &value),
            truncated: false,
        };
    }

    // Pass 4 (Rust-only): strip a markdown code fence (```json ... ```), then parse. Hermes has no
    // fence pass, but models wrapping arguments in a fence is common enough to keep. Everything after
    // this operates on the de-fenced body.
    let defenced = strip_code_fence(raw);
    let body = if defenced != raw_stripped {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(defenced.trim()) {
            return ArgRepair {
                args: canonicalize(&value),
                repaired: true,
                truncated: false,
            };
        }
        defenced.trim().to_string()
    } else {
        raw_stripped.to_string()
    };

    // Pass 5: lenient parse of raw control chars inside strings. serde_json is strict, so we escape
    // the control chars first (equivalent to Python's `json.loads(strict=False)` for success cases),
    // then re-serialize to the canonical wire form.
    let escaped_body = escape_control_chars_in_json_strings(&body);
    if escaped_body != body {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&escaped_body) {
            return ArgRepair {
                args: canonicalize(&value),
                repaired: true,
                truncated: false,
            };
        }
    }

    // Pass 6: common structural repairs — strip trailing commas, close unclosed structures by
    // delimiter count, then trim excess closing delimiters (bounded to 50 iterations).
    let mut fixed = strip_trailing_commas(&body);
    let open_curly = count_char(&fixed, '{') as i64 - count_char(&fixed, '}') as i64;
    let open_bracket = count_char(&fixed, '[') as i64 - count_char(&fixed, ']') as i64;
    if open_curly > 0 {
        fixed.push_str(&"}".repeat(open_curly as usize));
    }
    if open_bracket > 0 {
        fixed.push_str(&"]".repeat(open_bracket as usize));
    }
    for _ in 0..50 {
        if serde_json::from_str::<serde_json::Value>(&fixed).is_ok() {
            break;
        }
        // Trim one excess closing brace/bracket. Both Python branches pop a single char, so they
        // collapse to one condition here (avoids clippy::if_same_then_else).
        let excess_curly =
            fixed.ends_with('}') && count_char(&fixed, '}') > count_char(&fixed, '{');
        let excess_square =
            fixed.ends_with(']') && count_char(&fixed, ']') > count_char(&fixed, '[');
        if excess_curly || excess_square {
            fixed.pop();
        } else {
            break;
        }
    }
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&fixed) {
        return ArgRepair {
            args: canonicalize(&value),
            repaired: true,
            truncated,
        };
    }

    // Pass 7: escape unescaped control chars inside strings of the structurally-repaired payload,
    // then retry. Catches cases where a control char coexists with another malformation (e.g. a
    // trailing comma) that pass 5 alone could not clear.
    let escaped_fixed = escape_control_chars_in_json_strings(&fixed);
    if escaped_fixed != fixed {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&escaped_fixed) {
            return ArgRepair {
                args: canonicalize(&value),
                repaired: true,
                truncated,
            };
        }
    }

    // Pass 8: unrepairable — fall back to an empty object so the upstream API request cannot be
    // rejected as "invalid tool call arguments" and kill the session.
    ArgRepair {
        args: "{}".to_string(),
        repaired: true,
        truncated,
    }
}

/// Count occurrences of a single ASCII character in `s` (matches Python's `str.count`, including
/// occurrences inside JSON string literals — the delimiter-balance heuristic is deliberately naive).
fn count_char(s: &str, needle: char) -> usize {
    s.chars().filter(|c| *c == needle).count()
}

/// Strip a trailing comma (plus any whitespace up to the closer) that immediately precedes a `}` or
/// `]`. A faithful port of Python's `re.sub(r',\s*([}\]])', r'\1', fixed)`: a single non-overlapping
/// pass over the whole string, not string-literal aware.
fn strip_trailing_commas(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == ',' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && (chars[j] == '}' || chars[j] == ']') {
                // Drop the comma and the intervening whitespace; the closer is emitted next.
                i = j;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Escape unescaped control chars (`0x00`–`0x1F`) that appear inside JSON string values with their
/// `\uXXXX` form. A faithful port of hermes' `_escape_invalid_chars_in_json_strings`: walks the raw
/// text tracking string state, passing already-escaped pairs through untouched.
fn escape_control_chars_in_json_strings(raw: &str) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(n);
    let mut in_string = false;
    let mut i = 0;
    while i < n {
        let ch = chars[i];
        if in_string {
            if ch == '\\' && i + 1 < n {
                out.push(ch);
                out.push(chars[i + 1]);
                i += 2;
                continue;
            }
            if ch == '"' {
                in_string = false;
                out.push(ch);
            } else if (ch as u32) < 0x20 {
                out.push_str(&format!("\\u{:04x}", ch as u32));
            } else {
                out.push(ch);
            }
        } else {
            if ch == '"' {
                in_string = true;
            }
            out.push(ch);
        }
        i += 1;
    }
    out
}

/// Whether a JSON payload looks truncated: brackets/quotes left unbalanced (ignoring escapes and
/// string contents). Cheap heuristic used to flag a likely cut-off response.
pub fn looks_truncated(raw: &str) -> bool {
    let mut depth_curly = 0i32;
    let mut depth_square = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for c in raw.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => depth_curly += 1,
            '}' => depth_curly -= 1,
            '[' => depth_square += 1,
            ']' => depth_square -= 1,
            _ => {}
        }
    }
    in_string || depth_curly != 0 || depth_square != 0
}

/// Re-serialize a parsed value canonically (compact, stable object key order via `serde_json`'s
/// `BTreeMap`-like ordering for `Value::Object`, which preserves insertion order; we sort for
/// determinism).
fn canonicalize(value: &serde_json::Value) -> String {
    // `serde_json::to_string` on a `Value` already drops insignificant whitespace; for determinism
    // we additionally sort object keys.
    let sorted = sort_keys(value);
    serde_json::to_string(&sorted).unwrap_or_else(|_| "{}".to_string())
}

/// Whether `raw` is already the canonical serialization of `value` (so we can report `repaired`
/// accurately for the happy path).
fn is_canonical(raw: &str, value: &serde_json::Value) -> bool {
    raw == canonicalize(value)
}

/// Recursively sort object keys for a deterministic canonical form.
fn sort_keys(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let mut out = serde_json::Map::new();
            for (k, v) in entries {
                out.insert(k.clone(), sort_keys(v));
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(sort_keys).collect())
        }
        other => other.clone(),
    }
}

/// Strip a surrounding markdown code fence (```json ... ``` or ``` ... ```), if present.
fn strip_code_fence(raw: &str) -> String {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        // Drop an optional language tag on the opening fence line.
        let rest = rest.split_once('\n').map(|x| x.1).unwrap_or("");
        let rest = rest.strip_suffix("```").unwrap_or(rest);
        return rest.trim().to_string();
    }
    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_json_is_canonicalized() {
        let r = repair_tool_args(r#"{"b": 1, "a": 2}"#);
        assert_eq!(r.args, r#"{"a":2,"b":1}"#);
        assert!(!r.truncated);
    }

    #[test]
    fn strips_markdown_fence() {
        let r = repair_tool_args("```json\n{\"path\": \"a.txt\"}\n```");
        assert_eq!(r.args, r#"{"path":"a.txt"}"#);
        assert!(r.repaired);
    }

    #[test]
    fn repairs_trailing_comma() {
        let r = repair_tool_args(r#"{"a": 1, "b": 2,}"#);
        assert_eq!(r.args, r#"{"a":1,"b":2}"#);
        assert!(r.repaired);
    }

    #[test]
    fn truncated_mid_string_falls_back_to_empty_object() {
        // Hermes semantics: a value truncated mid-string cannot be recovered by delimiter
        // balancing (closing the string would fabricate content), so it collapses to `{}`.
        let r = repair_tool_args(r#"{"cmd": "ls -la"#);
        assert!(r.truncated);
        assert_eq!(r.args, "{}");
    }

    #[test]
    fn unrepairable_falls_back_to_empty_object() {
        // Hermes returns `{}` for unrepairable input so the upstream API cannot reject the
        // request; the Rust port matches that (no passthrough of garbage).
        let r = repair_tool_args("not json at all !!!");
        assert_eq!(r.args, "{}");
        assert!(r.repaired);
    }

    #[test]
    fn looks_truncated_detects_imbalance() {
        assert!(looks_truncated(r#"{"a": [1, 2"#));
        assert!(!looks_truncated(r#"{"a": [1, 2]}"#));
        assert!(looks_truncated(r#"{"a": "open"#));
    }
}

/// Parity tests ported from hermes' `_repair_tool_call_arguments`
/// (`agent/message_sanitization.py:185`) and its test matrix
/// (`tests/run_agent/test_repair_tool_call_arguments.py`).
///
/// Every stage of the Python repair pipeline is now implemented, so these all pass; each
/// asserts the behavior of the corresponding Python test case.
#[cfg(test)]
mod parity {
    use super::*;

    /// Helper: does the repaired output parse as valid JSON? (Python asserts
    /// `json.loads(result)` succeeds for the "valid JSON at minimum" cases.)
    fn parses(raw: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(&repair_tool_args(raw).args).is_ok()
    }

    // ── Stage 1: empty / whitespace-only → {} ──────────────────────────────

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_empty_string_returns_empty_object (tests/run_agent/test_repair_tool_call_arguments.py:13)
    #[test]
    fn empty_string_returns_empty_object() {
        assert_eq!(repair_tool_args("").args, "{}");
    }

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_whitespace_only_returns_empty_object (tests/run_agent/test_repair_tool_call_arguments.py:16)
    #[test]
    fn whitespace_only_returns_empty_object() {
        assert_eq!(repair_tool_args("   \n\t  ").args, "{}");
    }

    // ── Stage 2: Python `None` literal → {} ────────────────────────────────

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_python_none_literal (tests/run_agent/test_repair_tool_call_arguments.py:25)
    #[test]
    fn python_none_literal_returns_empty_object() {
        assert_eq!(repair_tool_args("None").args, "{}");
    }

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_python_none_with_whitespace (tests/run_agent/test_repair_tool_call_arguments.py:28)
    #[test]
    fn python_none_with_whitespace_returns_empty_object() {
        assert_eq!(repair_tool_args("  None  ").args, "{}");
    }

    // ── Stage 3: trailing-comma repair (already handled — port for coverage) ─

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_trailing_comma_in_array (tests/run_agent/test_repair_tool_call_arguments.py:37)
    #[test]
    fn trailing_comma_in_array() {
        let r = repair_tool_args(r#"{"a": [1, 2,]}"#);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&r.args).unwrap(),
            serde_json::json!({"a": [1, 2]})
        );
    }

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_multiple_trailing_commas (tests/run_agent/test_repair_tool_call_arguments.py:42)
    #[test]
    fn multiple_trailing_commas() {
        let r = repair_tool_args(r#"{"a": 1, "b": 2,}"#);
        let v: serde_json::Value = serde_json::from_str(&r.args).unwrap();
        assert_eq!(v["a"], 1);
        assert_eq!(v["b"], 2);
    }

    // ── Stage 4: unclosed brackets (already handled — port for coverage) ────

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_unclosed_bracket_and_brace (tests/run_agent/test_repair_tool_call_arguments.py:55)
    #[test]
    fn unclosed_bracket_and_brace_yields_valid_json() {
        assert!(parses(r#"{"a": [1, 2"#));
    }

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_trailing_comma_plus_unclosed_brace (tests/run_agent/test_repair_tool_call_arguments.py:95)
    #[test]
    fn trailing_comma_plus_unclosed_brace_yields_valid_json() {
        assert!(parses(r#"{"a": 1, "b": 2,"#));
    }

    // ── Stage 5: excess closing delimiters (NOT handled → gap) ─────────────

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_extra_closing_brace (tests/run_agent/test_repair_tool_call_arguments.py:64)
    #[test]
    fn extra_closing_brace_is_trimmed() {
        let r = repair_tool_args(r#"{"key": "value"}}"#);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&r.args).unwrap(),
            serde_json::json!({"key": "value"})
        );
    }

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_extra_closing_bracket (tests/run_agent/test_repair_tool_call_arguments.py:69)
    #[test]
    fn extra_closing_bracket_yields_valid_json() {
        assert!(parses(r#"{"a": [1]]}"#));
    }

    // ── Stage 6: unrepairable → {} fallback (NOT handled → gap) ────────────

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_unrepairable_garbage_returns_empty_object (tests/run_agent/test_repair_tool_call_arguments.py:76)
    #[test]
    fn unrepairable_garbage_returns_empty_object() {
        assert_eq!(repair_tool_args("totally not json").args, "{}");
    }

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_unrepairable_partial_returns_empty_object (tests/run_agent/test_repair_tool_call_arguments.py:79)
    #[test]
    fn unrepairable_partial_returns_empty_object() {
        // A value truncated mid-string is unrepairable in hermes (brace-count
        // alone cannot recover it) → {}.
        assert_eq!(repair_tool_args(r#"{"truncated": "val"#).args, "{}");
    }

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_real_world_glm_truncation (tests/run_agent/test_repair_tool_call_arguments.py:101)
    #[test]
    fn glm_truncation_yields_valid_json() {
        // Truncated after a key's colon (`"background":`) → hermes falls back to {}.
        assert!(parses(
            r#"{"command": "ls -la /tmp", "timeout": 30, "background":"#
        ));
    }

    // ── Stage 0: lenient parse of literal control chars in strings (gap) ───

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_literal_newline_inside_string_value (tests/run_agent/test_repair_tool_call_arguments.py:113)
    #[test]
    fn literal_newline_inside_string_value() {
        let r = repair_tool_args("{\"summary\": \"line one\nline two\"}");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&r.args).unwrap(),
            serde_json::json!({"summary": "line one\nline two"})
        );
    }

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_literal_tab_inside_string_value (tests/run_agent/test_repair_tool_call_arguments.py:119)
    #[test]
    fn literal_tab_inside_string_value() {
        let r = repair_tool_args("{\"summary\": \"col1\tcol2\"}");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&r.args).unwrap(),
            serde_json::json!({"summary": "col1\tcol2"})
        );
    }

    // ── Stage 4: control-char escape fallback with trailing comma (gap) ────

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_control_chars_with_trailing_comma (tests/run_agent/test_repair_tool_call_arguments.py:135)
    #[test]
    fn control_chars_with_trailing_comma() {
        let r = repair_tool_args("{\"msg\": \"line\none\",}");
        let v: serde_json::Value = serde_json::from_str(&r.args).unwrap();
        assert!(v["msg"].as_str().unwrap().contains("line"));
    }
}
