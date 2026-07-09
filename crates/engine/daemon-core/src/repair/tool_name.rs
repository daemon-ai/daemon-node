// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Tool-name repair (§9).
//!
//! Models occasionally emit a tool name that is *close* to a registered one — wrong case, an extra
//! namespace prefix (`functions.read_file`), a hyphen where there should be an underscore, or stray
//! surrounding punctuation. [`repair_tool_name`] normalizes the raw name, tries an exact match, then
//! a fuzzy match against the registry (Levenshtein similarity via `strsim`). When nothing is close
//! enough it returns a [`NameRepairError`] listing the valid names, which the provider surfaces as a
//! protocol-valid tool error the model can correct on the next round.

use strsim::normalized_levenshtein;

/// The similarity floor for accepting a fuzzy tool-name match (0..1, 1 = identical). Below this the
/// name is rejected rather than risk dispatching to the wrong tool.
const FUZZY_THRESHOLD: f64 = 0.82;

/// A tool name that could not be resolved to a registered tool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameRepairError {
    /// The (normalized) name the model emitted.
    pub invalid: String,
    /// The valid tool names, for the corrective error message.
    pub valid: Vec<String>,
}

impl std::fmt::Display for NameRepairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown tool `{}`; valid tools: {}",
            self.invalid,
            self.valid.join(", ")
        )
    }
}

impl std::error::Error for NameRepairError {}

/// Resolve `raw` to a registered tool name (§9), a faithful port of hermes'
/// `repair_tool_call` (`agent/agent_runtime_helpers.py:1925`) layered on top of the Rust port's
/// existing namespace/quote normalization:
/// 1. trim VolcEngine XML-attribute pollution at the first `"`/`'`/`<`/`>` (issue #33007);
/// 2. exact match on the (namespace/quote/case/separator) normalized name;
/// 3. build a candidate set adding CamelCase→snake_case and class-like `_tool`/`-tool`/`tool`
///    suffix stripping (applied up to twice for double-tacked suffixes like `TodoTool_tool`), and
///    exact-match any candidate;
/// 4. else the closest fuzzy match above [`FUZZY_THRESHOLD`], else [`NameRepairError`].
///
/// Hermes returns `None` on failure; the Rust port returns `Err` with the valid names so the
/// provider can surface a protocol-valid corrective error. The fuzzy stage uses `strsim`
/// (normalized Levenshtein) where hermes uses `difflib.get_close_matches`; the two agree on the
/// accept/reject cases in the parity matrix.
pub fn repair_tool_name(raw: &str, valid: &[String]) -> Result<String, NameRepairError> {
    let cleaned = trim_xml_pollution(raw);
    let normalized = normalize(&cleaned);

    // Exact match on the normalized name (case-insensitively against normalized valid names).
    for name in valid {
        if normalize(name) == normalized {
            return Ok(name.clone());
        }
    }

    // Build the candidate set for class-like emissions, then exact-match any candidate against the
    // (normalized) valid names.
    let candidates = build_candidates(&cleaned);
    for name in valid {
        let nv = normalize(name);
        for cand in &candidates {
            if !cand.is_empty() && normalize(cand) == nv {
                return Ok(name.clone());
            }
        }
    }

    // Fuzzy match: pick the most similar valid name above the threshold.
    let best = valid
        .iter()
        .map(|name| (name, normalized_levenshtein(&normalized, &normalize(name))))
        .filter(|(_, score)| *score >= FUZZY_THRESHOLD)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    match best {
        Some((name, _)) => Ok(name.clone()),
        None => Err(NameRepairError {
            invalid: normalized,
            valid: valid.to_vec(),
        }),
    }
}

/// Trim VolcEngine-style XML-attribute pollution: some endpoints leak raw XML attribute fragments
/// into the tool name (e.g. `terminal" parameter="command" string="true`). Truncate at the first
/// `"`/`'`/`<`/`>` that is *not* at position 0 — a leading quote is left for [`normalize`] to strip,
/// and whitespace is never split on so legitimate `write file` still flows through. Faithful port of
/// the `_xml_sep` loop in hermes' `repair_tool_call`.
fn trim_xml_pollution(raw: &str) -> String {
    let mut s = raw.to_string();
    for sep in ['"', '\'', '<', '>'] {
        if let Some(idx) = s.find(sep) {
            if idx > 0 {
                s.truncate(idx);
            }
        }
    }
    s
}

/// Normalize a tool name for matching: trim, strip surrounding quotes/backticks, drop a leading
/// `functions.`/`tool.`/`tools.` namespace, lowercase, and unify `-`/space to `_`.
fn normalize(raw: &str) -> String {
    let mut s = raw
        .trim()
        .trim_matches(|c| c == '"' || c == '\'' || c == '`');
    for prefix in ["functions.", "function.", "tools.", "tool.", "namespace."] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest;
        }
    }
    s.to_ascii_lowercase()
        .chars()
        .map(|c| if c == '-' || c == ' ' { '_' } else { c })
        .collect()
}

/// Hermes' `_norm`: lowercase and unify `-`/space to `_` (no namespace/quote handling).
fn norm_simple(s: &str) -> String {
    s.to_ascii_lowercase()
        .chars()
        .map(|c| if c == '-' || c == ' ' { '_' } else { c })
        .collect()
}

/// Hermes' `_camel_snake`: insert `_` before each uppercase letter that is not the first char, then
/// lowercase (`TodoTool` → `todo_tool`, `WriteFileTool` → `write_file_tool`).
fn camel_snake(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && c.is_ascii_uppercase() {
            out.push('_');
        }
        out.push(c);
    }
    out.to_ascii_lowercase()
}

/// Hermes' `_strip_tool_suffix`: strip a trailing `_tool`/`-tool`/`tool` class-like suffix (checked
/// in that order), then trim any trailing `_`/`-`. Returns `None` when no suffix matched.
fn strip_tool_suffix(s: &str) -> Option<String> {
    let lc = s.to_ascii_lowercase();
    for suffix in ["_tool", "-tool", "tool"] {
        if lc.ends_with(suffix) {
            let cut = s.len() - suffix.len();
            let trimmed = s[..cut].trim_end_matches(['_', '-']);
            return Some(trimmed.to_string());
        }
    }
    None
}

/// Build the candidate set hermes assembles in `repair_tool_call`: the raw (xml-trimmed) name, its
/// lowercase, its `_norm` form, and its CamelCase→snake form, then class-like suffix stripping
/// applied up to twice (each stripped form re-expanded through `_norm` and `_camel_snake`).
fn build_candidates(base: &str) -> Vec<String> {
    let mut cands: Vec<String> = Vec::new();
    let push = |c: String, v: &mut Vec<String>| {
        if !v.contains(&c) {
            v.push(c);
        }
    };
    push(base.to_string(), &mut cands);
    push(base.to_ascii_lowercase(), &mut cands);
    push(norm_simple(base), &mut cands);
    push(camel_snake(base), &mut cands);

    for _ in 0..2 {
        let mut extra: Vec<String> = Vec::new();
        for c in &cands {
            if let Some(stripped) = strip_tool_suffix(c) {
                if !stripped.is_empty() {
                    extra.push(stripped.clone());
                    extra.push(norm_simple(&stripped));
                    extra.push(camel_snake(&stripped));
                }
            }
        }
        for e in extra {
            push(e, &mut cands);
        }
    }
    cands
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> Vec<String> {
        vec![
            "read_file".to_string(),
            "write_file".to_string(),
            "run_shell".to_string(),
        ]
    }

    #[test]
    fn exact_match_passes_through() {
        assert_eq!(
            repair_tool_name("read_file", &valid()).unwrap(),
            "read_file"
        );
    }

    #[test]
    fn normalizes_case_prefix_and_separators() {
        assert_eq!(
            repair_tool_name("functions.Read-File", &valid()).unwrap(),
            "read_file"
        );
        assert_eq!(
            repair_tool_name("`write file`", &valid()).unwrap(),
            "write_file"
        );
    }

    #[test]
    fn fuzzy_matches_a_typo() {
        // One transposed/dropped char is within threshold.
        assert_eq!(repair_tool_name("read_fil", &valid()).unwrap(), "read_file");
    }

    #[test]
    fn rejects_a_far_name() {
        let err = repair_tool_name("delete_everything", &valid()).unwrap_err();
        assert_eq!(err.invalid, "delete_everything");
        assert!(err.valid.contains(&"read_file".to_string()));
        assert!(err.to_string().contains("valid tools"));
    }
}

/// Parity tests ported from hermes' `repair_tool_call`
/// (`agent/agent_runtime_helpers.py:1925`; tests `tests/run_agent/test_repair_tool_call_name.py`).
/// On top of the case/separator/fuzzy handling the Rust port already had, it now
/// (a) splits CamelCase to snake_case, (b) strips `_tool`/`-tool`/`Tool` class-like
/// suffixes (up to twice), and (c) trims VolcEngine XML-attribute pollution at the
/// first `"`/`'`/`<`/`>`, so these all pass.
///
/// hermes returns `None` for an unresolved name; the Rust port returns `Err`, so we
/// treat `Err` as the parity equivalent of `None`.
#[cfg(test)]
mod parity {
    use super::*;

    fn valid() -> Vec<String> {
        [
            "todo",
            "patch",
            "browser_click",
            "browser_navigate",
            "web_search",
            "read_file",
            "write_file",
            "terminal",
            "execute_code",
            "session_search",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn repair(name: &str) -> Option<String> {
        repair_tool_name(name, &valid()).ok()
    }

    // ── Existing behavior that already works (must PASS) ───────────────────

    // parity: test_repair_tool_call_name.py::TestExistingBehaviorStillWorks::test_uppercase_simple (tests/run_agent/test_repair_tool_call_name.py:53)
    #[test]
    fn uppercase_simple() {
        assert_eq!(repair("TERMINAL").as_deref(), Some("terminal"));
    }

    // parity: test_repair_tool_call_name.py::TestExistingBehaviorStillWorks::test_fuzzy_near_miss (tests/run_agent/test_repair_tool_call_name.py:62)
    #[test]
    fn fuzzy_near_miss() {
        assert_eq!(repair("terminall").as_deref(), Some("terminal"));
    }

    // parity: test_repair_tool_call_name.py::TestExistingBehaviorStillWorks::test_unknown_returns_none (tests/run_agent/test_repair_tool_call_name.py:66)
    #[test]
    fn unknown_returns_none() {
        assert_eq!(repair("xyz_no_such_tool"), None);
    }

    // parity: test_repair_tool_call_name.py::TestEdgeCases::test_empty_string (tests/run_agent/test_repair_tool_call_name.py:104)
    #[test]
    fn empty_string_returns_none() {
        assert_eq!(repair(""), None);
    }

    // parity: test_repair_tool_call_name.py::TestEdgeCases::test_only_tool_suffix (tests/run_agent/test_repair_tool_call_name.py:107)
    #[test]
    fn only_tool_suffix_returns_none() {
        assert_eq!(repair("_tool"), None);
    }

    // parity: test_repair_tool_call_name.py::TestEdgeCases::test_very_long_name_does_not_match_by_accident (tests/run_agent/test_repair_tool_call_name.py:117)
    #[test]
    fn very_long_name_returns_none() {
        assert_eq!(repair("ThisIsNotRemotelyARealToolName_tool"), None);
    }

    // parity: test_repair_tool_call_name.py::TestVolcEngineXmlPollution::test_tool_name_with_trailing_quote_only (tests/run_agent/test_repair_tool_call_name.py:155)
    #[test]
    fn trailing_quote_only_is_trimmed() {
        assert_eq!(repair("terminal\"").as_deref(), Some("terminal"));
    }

    // parity: test_repair_tool_call_name.py::TestVolcEngineXmlPollution::test_leading_quote_falls_through_to_fuzzy_match (tests/run_agent/test_repair_tool_call_name.py:183)
    #[test]
    fn leading_and_trailing_quotes_resolve() {
        assert_eq!(repair("\"terminal\"").as_deref(), Some("terminal"));
    }

    // parity: test_repair_tool_call_name.py::TestVolcEngineXmlPollution::test_pollution_with_unknown_tool_root_still_fails (tests/run_agent/test_repair_tool_call_name.py:177)
    #[test]
    fn polluted_unknown_root_returns_none() {
        assert_eq!(repair("no_such_tool\" parameter=\"x\" string=\"true"), None);
    }

    // ── CamelCase → snake_case + class-like suffix stripping (gaps) ────────

    // parity: test_repair_tool_call_name.py::TestClassLikeEmissions::test_camel_case_with_underscore_tool_suffix (tests/run_agent/test_repair_tool_call_name.py:76)
    #[test]
    fn camel_case_with_underscore_tool_suffix() {
        assert_eq!(
            repair("BrowserClick_tool").as_deref(),
            Some("browser_click")
        );
    }

    // parity: test_repair_tool_call_name.py::TestClassLikeEmissions::test_camel_case_with_Tool_class_suffix (tests/run_agent/test_repair_tool_call_name.py:79)
    #[test]
    fn camel_case_with_tool_class_suffix() {
        assert_eq!(repair("PatchTool").as_deref(), Some("patch"));
    }

    // parity: test_repair_tool_call_name.py::TestClassLikeEmissions::test_double_tacked_class_and_snake_suffix (tests/run_agent/test_repair_tool_call_name.py:82)
    #[test]
    fn double_tacked_class_and_snake_suffix() {
        assert_eq!(repair("TodoTool_tool").as_deref(), Some("todo"));
    }

    // parity: test_repair_tool_call_name.py::TestClassLikeEmissions::test_simple_name_with_tool_suffix (tests/run_agent/test_repair_tool_call_name.py:87)
    #[test]
    fn simple_name_with_tool_suffix() {
        assert_eq!(repair("Patch_tool").as_deref(), Some("patch"));
    }

    // parity: test_repair_tool_call_name.py::TestClassLikeEmissions::test_simple_name_with_dash_tool_suffix (tests/run_agent/test_repair_tool_call_name.py:90)
    #[test]
    fn simple_name_with_dash_tool_suffix() {
        assert_eq!(repair("patch-tool").as_deref(), Some("patch"));
    }

    // parity: test_repair_tool_call_name.py::TestClassLikeEmissions::test_camel_case_preserves_multi_word_match (tests/run_agent/test_repair_tool_call_name.py:93)
    #[test]
    fn camel_case_preserves_multi_word_match() {
        assert_eq!(repair("WriteFileTool").as_deref(), Some("write_file"));
    }

    // parity: test_repair_tool_call_name.py::TestClassLikeEmissions::test_mixed_separators_and_suffix (tests/run_agent/test_repair_tool_call_name.py:97)
    #[test]
    fn mixed_separators_and_suffix() {
        assert_eq!(repair("write-file_Tool").as_deref(), Some("write_file"));
    }

    // ── VolcEngine XML-attribute pollution trimming (gaps) ────────────────

    // parity: test_repair_tool_call_name.py::TestVolcEngineXmlPollution::test_terminal_with_xml_attribute_pollution (tests/run_agent/test_repair_tool_call_name.py:136)
    #[test]
    fn terminal_with_xml_attribute_pollution() {
        assert_eq!(
            repair("terminal\" parameter=\"command\" string=\"true").as_deref(),
            Some("terminal")
        );
    }

    // parity: test_repair_tool_call_name.py::TestVolcEngineXmlPollution::test_execute_code_with_xml_attribute_pollution (tests/run_agent/test_repair_tool_call_name.py:141)
    #[test]
    fn execute_code_with_xml_attribute_pollution() {
        assert_eq!(
            repair("execute_code\" parameter=\"code\" string=\"true").as_deref(),
            Some("execute_code")
        );
    }

    // parity: test_repair_tool_call_name.py::TestVolcEngineXmlPollution::test_camel_case_tool_with_xml_pollution (tests/run_agent/test_repair_tool_call_name.py:149)
    #[test]
    fn camel_case_tool_with_xml_pollution() {
        assert_eq!(
            repair("BrowserClick_tool\" parameter=\"selector\" string=\"true").as_deref(),
            Some("browser_click")
        );
    }

    // parity: test_repair_tool_call_name.py::TestVolcEngineXmlPollution::test_tool_name_with_angle_bracket_pollution (tests/run_agent/test_repair_tool_call_name.py:159)
    #[test]
    fn tool_name_with_angle_bracket_pollution() {
        assert_eq!(
            repair("terminal<parameter=command").as_deref(),
            Some("terminal")
        );
    }

    // parity: test_repair_tool_call_name.py::TestVolcEngineXmlPollution::test_tool_name_with_single_quote_pollution (tests/run_agent/test_repair_tool_call_name.py:163)
    #[test]
    fn tool_name_with_single_quote_pollution() {
        assert_eq!(
            repair("terminal' parameter='command' string='true").as_deref(),
            Some("terminal")
        );
    }
}
