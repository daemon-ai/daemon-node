// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Canonical quantization FAMILIES (wire v48 quant filter) — the ONE normalization table.
//!
//! Raw quant tags (GGUF filename labels like `Q4_K_M`, repo-name method suffixes like `GPTQ`)
//! fold into coarse canonical families so the client's filter chips stay a short, stable
//! vocabulary. Two ordered groups share one wire field (deliberately conflated — one chip row
//! serves the user's intent; the canonical order keeps the groups visible):
//!
//! - Precision families: `Q1‥Q8` (GGML block quants; `Q4_0/Q4_1/Q4_K_* → Q4`), `IQ`
//!   (importance-aware i-quants — `IQ1_*` is NOT `Q1`), `TQ` (ternary), `INT2/INT4/INT8`
//!   (generic integer widths — `INT4 ≠ Q4`), `FP4`, `MXFP4` (microscaled block format, distinct
//!   from both `Q4` and `FP4`; GPT-OSS ships it), `FP8` (`E4M3/E5M2/FP8_* → FP8`), `F16`
//!   (`FP16 → F16`), `BF16`, `F32`. `F64` is deliberately excluded (not a useful bucket).
//! - Method formats (mainly repo-name suffixes on the repository-strategy search): `GPTQ AWQ
//!   EXL2 HQQ AQLM NF4 BNB4`. `NF4` is distinct from `INT4`/`FP4`/`Q4` (bitsandbytes/QLoRA).
//!
//! `GGUF` is a container, never a family — every llama.cpp hit would carry it (pure noise).

use daemon_common::ModelFile;

/// The canonical family order (precision first, then method formats). The UI renders only the
/// families PRESENT in served results, in this order — the exhaustive vocabulary never becomes
/// a chip wall.
pub const FAMILY_ORDER: &[&str] = &[
    // Precision families.
    "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q8", "IQ", "TQ", "INT2", "INT4", "INT8", "FP4", "MXFP4",
    "FP8", "F16", "BF16", "F32", //
    // Method formats.
    "GPTQ", "AWQ", "EXL2", "HQQ", "AQLM", "NF4", "BNB4",
];

/// The canonical family for one raw quant tag (a GGUF filename label such as `Q4_K_M`, or a
/// repo-name token such as `GPTQ`). `None` when the tag names no known family (e.g. `GGUF`).
pub fn family_of_label(label: &str) -> Option<&'static str> {
    let up = label.trim().to_ascii_uppercase();
    if up.is_empty() {
        return None;
    }
    // Exact method formats and fixed precision tokens first (most specific).
    match up.as_str() {
        "GPTQ" => return Some("GPTQ"),
        "AWQ" => return Some("AWQ"),
        "EXL2" => return Some("EXL2"),
        "HQQ" => return Some("HQQ"),
        "AQLM" => return Some("AQLM"),
        "NF4" => return Some("NF4"),
        "BNB4" => return Some("BNB4"),
        "BF16" => return Some("BF16"),
        "F32" | "FP32" => return Some("F32"),
        "F16" | "FP16" => return Some("F16"),
        "INT2" => return Some("INT2"),
        "INT4" => return Some("INT4"),
        "INT8" => return Some("INT8"),
        "FP4" => return Some("FP4"),
        _ => {}
    }
    // Prefixed families: order matters (MXFP4 before FP4-by-prefix, IQ before Q).
    if up.starts_with("MXFP4") {
        return Some("MXFP4");
    }
    if up.starts_with("FP8") || up.starts_with("E4M3") || up.starts_with("E5M2") {
        return Some("FP8");
    }
    if up.starts_with("IQ") {
        return Some("IQ");
    }
    if up.starts_with("TQ") {
        return Some("TQ");
    }
    // GGML block quants: `Q<digit>[_…]` → `Q<digit>` (known digits only; Q7 does not exist).
    if let Some(rest) = up.strip_prefix('Q') {
        let digit = rest.chars().next()?;
        // The remainder must not extend the digit (a `Q40` token is not a quant label).
        if rest.chars().nth(1).is_none_or(|c| !c.is_ascii_digit()) {
            return match digit {
                '1' => Some("Q1"),
                '2' => Some("Q2"),
                '3' => Some("Q3"),
                '4' => Some("Q4"),
                '5' => Some("Q5"),
                '6' => Some("Q6"),
                '8' => Some("Q8"),
                _ => None,
            };
        }
    }
    None
}

/// The distinct canonical families across a repo's file listing (from the parsed per-file quant
/// labels), in canonical order — the artifact-strategy (llama.cpp) enrichment.
pub fn families_from_files(files: &[ModelFile]) -> Vec<String> {
    let present: Vec<&'static str> = files
        .iter()
        .filter_map(|f| f.quant.as_deref().and_then(family_of_label))
        .collect();
    in_canonical_order(&present)
}

/// The canonical families named by a repo id (tokenized on non-alphanumerics) — the
/// repository-strategy (mistral.rs) enrichment, where method formats ride the repo NAME
/// (e.g. `TheBloke/Llama-2-7B-GPTQ`, `unsloth/llama-3-8b-bnb-4bit`).
pub fn families_from_repo_name(repo: &str) -> Vec<String> {
    let up = repo.to_ascii_uppercase();
    let tokens: Vec<&str> = up
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    let mut present: Vec<&'static str> = Vec::new();
    for (i, tok) in tokens.iter().enumerate() {
        if let Some(fam) = family_of_label(tok) {
            present.push(fam);
        }
        // The bitsandbytes naming convention splits across tokens: `…-bnb-4bit`.
        if *tok == "BNB" && tokens.get(i + 1).is_some_and(|n| *n == "4BIT") {
            present.push("BNB4");
        }
    }
    in_canonical_order(&present)
}

/// Distinct families in canonical order.
fn in_canonical_order(present: &[&'static str]) -> Vec<String> {
    FAMILY_ORDER
        .iter()
        .filter(|fam| present.contains(*fam))
        .map(|fam| (*fam).to_string())
        .collect()
}

/// Whether a hit's families intersect the selected filter set (case-insensitive; the client
/// sends canonical family tokens).
pub fn matches_filter(families: &[String], selected: &[String]) -> bool {
    families
        .iter()
        .any(|f| selected.iter().any(|s| s.eq_ignore_ascii_case(f)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str) -> ModelFile {
        ModelFile {
            path: path.to_string(),
            size_bytes: 1,
            quant: crate::gguf::quant_label(path),
            is_split: false,
            is_first_shard: false,
            is_mmproj: false,
        }
    }

    /// GGML block labels fold to their `Q<digit>` family; i-quants, ternary, and the float
    /// formats keep their own families; `INT4`, `NF4`, `FP4`, and `MXFP4` never fold into `Q4`.
    #[test]
    fn labels_fold_into_canonical_families() {
        assert_eq!(family_of_label("Q4_K_M"), Some("Q4"));
        assert_eq!(family_of_label("q4_0"), Some("Q4"));
        assert_eq!(family_of_label("Q8_0"), Some("Q8"));
        assert_eq!(family_of_label("Q6_K_L"), Some("Q6"));
        assert_eq!(family_of_label("IQ2_XS"), Some("IQ"));
        assert_eq!(family_of_label("IQ1_S"), Some("IQ")); // i-quants are NOT Q1
        assert_eq!(family_of_label("TQ1_0"), Some("TQ"));
        assert_eq!(family_of_label("MXFP4"), Some("MXFP4"));
        assert_eq!(family_of_label("FP4"), Some("FP4"));
        assert_eq!(family_of_label("E4M3"), Some("FP8"));
        assert_eq!(family_of_label("FP8_E4M3FN"), Some("FP8"));
        assert_eq!(family_of_label("FP16"), Some("F16"));
        assert_eq!(family_of_label("BF16"), Some("BF16"));
        assert_eq!(family_of_label("F32"), Some("F32"));
        assert_eq!(family_of_label("INT4"), Some("INT4"));
        assert_eq!(family_of_label("NF4"), Some("NF4"));
        assert_eq!(family_of_label("GPTQ"), Some("GPTQ"));
        // Containers and noise are never families.
        assert_eq!(family_of_label("GGUF"), None);
        assert_eq!(family_of_label("F64"), None);
        assert_eq!(family_of_label("Q40"), None);
        assert_eq!(family_of_label(""), None);
    }

    /// File listings produce distinct families in canonical order.
    #[test]
    fn files_produce_ordered_distinct_families() {
        let files = vec![
            file("model-F16.gguf"),
            file("model-Q4_K_M.gguf"),
            file("model-Q4_K_S.gguf"),
            file("model-IQ2_XS.gguf"),
            file("model-Q8_0.gguf"),
            file("README.md"),
        ];
        assert_eq!(families_from_files(&files), vec!["Q4", "Q8", "IQ", "F16"]);
    }

    /// Repo names carry method formats (and sometimes precision tokens); `GGUF` is ignored.
    #[test]
    fn repo_names_yield_method_families() {
        assert_eq!(
            families_from_repo_name("TheBloke/Llama-2-7B-GPTQ"),
            vec!["GPTQ"]
        );
        assert_eq!(
            families_from_repo_name("unsloth/llama-3-8b-bnb-4bit"),
            vec!["BNB4"]
        );
        assert_eq!(
            families_from_repo_name("bartowski/SmolLM2-135M-Instruct-GGUF"),
            Vec::<String>::new()
        );
        assert_eq!(
            families_from_repo_name("org/model-AWQ-INT4"),
            vec!["INT4", "AWQ"]
        );
    }

    /// The filter matches on any intersection, case-insensitively.
    #[test]
    fn filter_matches_any_selected_family() {
        let fams = vec!["Q4".to_string(), "F16".to_string()];
        assert!(matches_filter(&fams, &["q4".to_string()]));
        assert!(matches_filter(
            &fams,
            &["Q8".to_string(), "F16".to_string()]
        ));
        assert!(!matches_filter(&fams, &["Q8".to_string()]));
        assert!(!matches_filter(&[], &["Q4".to_string()]));
    }
}
