// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Tool-argument JSON repair (§9).
//!
//! Models emit tool-call arguments as JSON text that is frequently *almost* valid: wrapped in a
//! markdown code fence, carrying trailing commas, single-quoted, or truncated mid-object when the
//! response was cut off. [`repair_tool_args`] runs a small multi-pass repair and, on success,
//! re-serializes through `serde_json` so the tool always receives canonical JSON (stable key order,
//! no stray whitespace). Unparseable-after-repair input is passed through untouched so a tool that
//! takes a non-JSON string still works.

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

/// Repair and canonicalize a tool-argument JSON payload (§9). The passes, in order:
/// 1. parse as-is;
/// 2. strip a surrounding markdown code fence and parse;
/// 3. strip trailing commas + balance brackets/quotes (truncation repair) and parse.
///
/// The first pass that parses wins and its value is re-serialized canonically; if none parse, the
/// original string is returned with `repaired = false`.
pub fn repair_tool_args(raw: &str) -> ArgRepair {
    let truncated = looks_truncated(raw);

    // Pass 1: already valid JSON.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(raw.trim()) {
        return ArgRepair {
            args: canonicalize(&value),
            repaired: raw.trim() != raw || !is_canonical(raw, &value),
            truncated: false,
        };
    }

    // Pass 2: strip a markdown code fence (```json ... ```), then parse.
    let defenced = strip_code_fence(raw);
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(defenced.trim()) {
        return ArgRepair {
            args: canonicalize(&value),
            repaired: true,
            truncated: false,
        };
    }

    // Pass 3: drop trailing commas and balance unclosed strings/brackets (truncation repair).
    let balanced = balance_and_clean(defenced.trim());
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&balanced) {
        return ArgRepair {
            args: canonicalize(&value),
            repaired: true,
            truncated,
        };
    }

    // Give up: pass the original through unchanged.
    ArgRepair {
        args: raw.to_string(),
        repaired: false,
        truncated,
    }
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

/// Remove trailing commas before `}`/`]` and close any unbalanced strings/brackets at the end (the
/// truncation-repair pass). Operates outside string literals only.
fn balance_and_clean(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 8);
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    let mut pending_comma: Option<usize> = None;

    for c in raw.chars() {
        if in_string {
            out.push(c);
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
            '"' => {
                in_string = true;
                pending_comma = None;
                out.push(c);
            }
            ',' => {
                pending_comma = Some(out.len());
                out.push(c);
            }
            '}' | ']' => {
                // Drop a trailing comma immediately before this closer.
                if let Some(pos) = pending_comma.take() {
                    if out[pos..].trim().is_empty() || out[pos..].chars().all(|x| x == ',') {
                        // remove from the comma position to end (only whitespace/comma)
                        out.truncate(pos);
                    }
                }
                stack.pop();
                out.push(c);
            }
            '{' => {
                stack.push('}');
                pending_comma = None;
                out.push(c);
            }
            '[' => {
                stack.push(']');
                pending_comma = None;
                out.push(c);
            }
            _ => {
                if !c.is_whitespace() {
                    pending_comma = None;
                }
                out.push(c);
            }
        }
    }

    // Close an unterminated string, then any open brackets, in reverse order.
    if in_string {
        out.push('"');
    }
    // Drop a dangling trailing comma at the very end.
    let trimmed_end = out.trim_end();
    if trimmed_end.ends_with(',') {
        let new_len = trimmed_end.len() - 1;
        out.truncate(new_len);
    }
    while let Some(closer) = stack.pop() {
        out.push(closer);
    }
    out
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
    fn repairs_truncated_object() {
        let r = repair_tool_args(r#"{"cmd": "ls -la"#);
        assert!(r.truncated);
        // Closed string + closed object => parseable canonical form.
        assert_eq!(r.args, r#"{"cmd":"ls -la"}"#);
    }

    #[test]
    fn unparseable_passes_through() {
        let r = repair_tool_args("not json at all !!!");
        assert_eq!(r.args, "not json at all !!!");
        assert!(!r.repaired);
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
/// `parity_gap_*` tests assert the DESIRED behavior per the Python source and are
/// expected to FAIL against the current Rust port — each documents a missing repair
/// stage. Plain-named tests port already-correct behavior and MUST PASS.
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
    fn parity_gap_empty_string_returns_empty_object() {
        assert_eq!(repair_tool_args("").args, "{}");
    }

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_whitespace_only_returns_empty_object (tests/run_agent/test_repair_tool_call_arguments.py:16)
    #[test]
    fn parity_gap_whitespace_only_returns_empty_object() {
        assert_eq!(repair_tool_args("   \n\t  ").args, "{}");
    }

    // ── Stage 2: Python `None` literal → {} ────────────────────────────────

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_python_none_literal (tests/run_agent/test_repair_tool_call_arguments.py:25)
    #[test]
    fn parity_gap_python_none_literal_returns_empty_object() {
        assert_eq!(repair_tool_args("None").args, "{}");
    }

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_python_none_with_whitespace (tests/run_agent/test_repair_tool_call_arguments.py:28)
    #[test]
    fn parity_gap_python_none_with_whitespace_returns_empty_object() {
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
    fn parity_gap_extra_closing_brace_is_trimmed() {
        let r = repair_tool_args(r#"{"key": "value"}}"#);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&r.args).unwrap(),
            serde_json::json!({"key": "value"})
        );
    }

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_extra_closing_bracket (tests/run_agent/test_repair_tool_call_arguments.py:69)
    #[test]
    fn parity_gap_extra_closing_bracket_yields_valid_json() {
        assert!(parses(r#"{"a": [1]]}"#));
    }

    // ── Stage 6: unrepairable → {} fallback (NOT handled → gap) ────────────

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_unrepairable_garbage_returns_empty_object (tests/run_agent/test_repair_tool_call_arguments.py:76)
    #[test]
    fn parity_gap_unrepairable_garbage_returns_empty_object() {
        assert_eq!(repair_tool_args("totally not json").args, "{}");
    }

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_unrepairable_partial_returns_empty_object (tests/run_agent/test_repair_tool_call_arguments.py:79)
    #[test]
    fn parity_gap_unrepairable_partial_returns_empty_object() {
        // A value truncated mid-string is unrepairable in hermes (brace-count
        // alone cannot recover it) → {}.
        assert_eq!(repair_tool_args(r#"{"truncated": "val"#).args, "{}");
    }

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_real_world_glm_truncation (tests/run_agent/test_repair_tool_call_arguments.py:101)
    #[test]
    fn parity_gap_glm_truncation_yields_valid_json() {
        // Truncated after a key's colon (`"background":`) → hermes falls back to {}.
        assert!(parses(
            r#"{"command": "ls -la /tmp", "timeout": 30, "background":"#
        ));
    }

    // ── Stage 0: lenient parse of literal control chars in strings (gap) ───

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_literal_newline_inside_string_value (tests/run_agent/test_repair_tool_call_arguments.py:113)
    #[test]
    fn parity_gap_literal_newline_inside_string_value() {
        let r = repair_tool_args("{\"summary\": \"line one\nline two\"}");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&r.args).unwrap(),
            serde_json::json!({"summary": "line one\nline two"})
        );
    }

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_literal_tab_inside_string_value (tests/run_agent/test_repair_tool_call_arguments.py:119)
    #[test]
    fn parity_gap_literal_tab_inside_string_value() {
        let r = repair_tool_args("{\"summary\": \"col1\tcol2\"}");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&r.args).unwrap(),
            serde_json::json!({"summary": "col1\tcol2"})
        );
    }

    // ── Stage 4: control-char escape fallback with trailing comma (gap) ────

    // parity: test_repair_tool_call_arguments.py::TestRepairToolCallArguments::test_control_chars_with_trailing_comma (tests/run_agent/test_repair_tool_call_arguments.py:135)
    #[test]
    fn parity_gap_control_chars_with_trailing_comma() {
        let r = repair_tool_args("{\"msg\": \"line\none\",}");
        let v: serde_json::Value = serde_json::from_str(&r.args).unwrap();
        assert!(v["msg"].as_str().unwrap().contains("line"));
    }
}
