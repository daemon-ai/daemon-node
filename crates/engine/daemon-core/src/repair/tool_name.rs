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

/// Resolve `raw` to a registered tool name (§9): exact match on the normalized form, else the
/// closest fuzzy match above [`FUZZY_THRESHOLD`], else [`NameRepairError`].
pub fn repair_tool_name(raw: &str, valid: &[String]) -> Result<String, NameRepairError> {
    let normalized = normalize(raw);

    // Exact match on the normalized name (case-insensitively against normalized valid names).
    for name in valid {
        if normalize(name) == normalized {
            return Ok(name.clone());
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

/// Parity tests ported from hermes' `AIAgent._repair_tool_call`
/// (`tests/run_agent/test_repair_tool_call_name.py`). The Python routine, on top of
/// the case/separator/fuzzy handling the Rust port already has, also
/// (a) splits CamelCase to snake_case, (b) strips `_tool`/`-tool`/`Tool` class-like
/// suffixes (up to twice), and (c) trims VolcEngine XML-attribute pollution at the
/// first `"`/`'`/`<`/`>`. Those three are the gaps.
///
/// `parity_gap_*` tests assert the desired Python behavior and are expected to FAIL.
/// Plain-named tests port behavior the Rust port already handles and MUST PASS.
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
    fn parity_gap_camel_case_with_underscore_tool_suffix() {
        assert_eq!(
            repair("BrowserClick_tool").as_deref(),
            Some("browser_click")
        );
    }

    // parity: test_repair_tool_call_name.py::TestClassLikeEmissions::test_camel_case_with_Tool_class_suffix (tests/run_agent/test_repair_tool_call_name.py:79)
    #[test]
    fn parity_gap_camel_case_with_tool_class_suffix() {
        assert_eq!(repair("PatchTool").as_deref(), Some("patch"));
    }

    // parity: test_repair_tool_call_name.py::TestClassLikeEmissions::test_double_tacked_class_and_snake_suffix (tests/run_agent/test_repair_tool_call_name.py:82)
    #[test]
    fn parity_gap_double_tacked_class_and_snake_suffix() {
        assert_eq!(repair("TodoTool_tool").as_deref(), Some("todo"));
    }

    // parity: test_repair_tool_call_name.py::TestClassLikeEmissions::test_simple_name_with_tool_suffix (tests/run_agent/test_repair_tool_call_name.py:87)
    #[test]
    fn parity_gap_simple_name_with_tool_suffix() {
        assert_eq!(repair("Patch_tool").as_deref(), Some("patch"));
    }

    // parity: test_repair_tool_call_name.py::TestClassLikeEmissions::test_simple_name_with_dash_tool_suffix (tests/run_agent/test_repair_tool_call_name.py:90)
    #[test]
    fn parity_gap_simple_name_with_dash_tool_suffix() {
        assert_eq!(repair("patch-tool").as_deref(), Some("patch"));
    }

    // parity: test_repair_tool_call_name.py::TestClassLikeEmissions::test_camel_case_preserves_multi_word_match (tests/run_agent/test_repair_tool_call_name.py:93)
    #[test]
    fn parity_gap_camel_case_preserves_multi_word_match() {
        assert_eq!(repair("WriteFileTool").as_deref(), Some("write_file"));
    }

    // parity: test_repair_tool_call_name.py::TestClassLikeEmissions::test_mixed_separators_and_suffix (tests/run_agent/test_repair_tool_call_name.py:97)
    #[test]
    fn parity_gap_mixed_separators_and_suffix() {
        assert_eq!(repair("write-file_Tool").as_deref(), Some("write_file"));
    }

    // ── VolcEngine XML-attribute pollution trimming (gaps) ────────────────

    // parity: test_repair_tool_call_name.py::TestVolcEngineXmlPollution::test_terminal_with_xml_attribute_pollution (tests/run_agent/test_repair_tool_call_name.py:136)
    #[test]
    fn parity_gap_terminal_with_xml_attribute_pollution() {
        assert_eq!(
            repair("terminal\" parameter=\"command\" string=\"true").as_deref(),
            Some("terminal")
        );
    }

    // parity: test_repair_tool_call_name.py::TestVolcEngineXmlPollution::test_execute_code_with_xml_attribute_pollution (tests/run_agent/test_repair_tool_call_name.py:141)
    #[test]
    fn parity_gap_execute_code_with_xml_attribute_pollution() {
        assert_eq!(
            repair("execute_code\" parameter=\"code\" string=\"true").as_deref(),
            Some("execute_code")
        );
    }

    // parity: test_repair_tool_call_name.py::TestVolcEngineXmlPollution::test_camel_case_tool_with_xml_pollution (tests/run_agent/test_repair_tool_call_name.py:149)
    #[test]
    fn parity_gap_camel_case_tool_with_xml_pollution() {
        assert_eq!(
            repair("BrowserClick_tool\" parameter=\"selector\" string=\"true").as_deref(),
            Some("browser_click")
        );
    }

    // parity: test_repair_tool_call_name.py::TestVolcEngineXmlPollution::test_tool_name_with_angle_bracket_pollution (tests/run_agent/test_repair_tool_call_name.py:159)
    #[test]
    fn parity_gap_tool_name_with_angle_bracket_pollution() {
        assert_eq!(
            repair("terminal<parameter=command").as_deref(),
            Some("terminal")
        );
    }

    // parity: test_repair_tool_call_name.py::TestVolcEngineXmlPollution::test_tool_name_with_single_quote_pollution (tests/run_agent/test_repair_tool_call_name.py:163)
    #[test]
    fn parity_gap_tool_name_with_single_quote_pollution() {
        assert_eq!(
            repair("terminal' parameter='command' string='true").as_deref(),
            Some("terminal")
        );
    }
}
