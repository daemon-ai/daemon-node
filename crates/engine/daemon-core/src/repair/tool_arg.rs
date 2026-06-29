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
