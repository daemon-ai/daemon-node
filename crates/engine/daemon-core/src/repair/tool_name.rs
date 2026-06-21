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
