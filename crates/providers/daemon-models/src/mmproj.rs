// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Filename heuristics for pairing a language-model GGUF with its `mmproj` companion — the CLIP
//! vision-projector file llama.cpp loads *alongside* text weights (never as a chat model itself).
//!
//! Ported from the reference `MmprojMatcher` (daemon-q1-2026): analyze a filename stem into a
//! quant tag + normalized name tokens + an "is this an mmproj/projector?" hint, then score
//! model↔projector pairs by *mmproj-token recall* (`|mmproj ∩ model| / |mmproj|`). Two pairing
//! policies use it:
//!
//! - **Download-side** ([`best_companion`], threshold [`HF_PAIR_THRESHOLD`]): when a *text* GGUF
//!   is planned, scan the repo listing for the best-scoring projector — recall plus a `+0.10`
//!   same-quant bonus and a `+0.02` "mmproj" (vs "projector") naming bonus — and fetch it in the
//!   same job. A quant mismatch only loses the bonus (projector repos often ship fewer quants
//!   than the text weights).
//! - **Local-scan side** ([`best_registry_companion`], threshold [`LOCAL_PAIR_THRESHOLD`]): link
//!   pre-existing cataloged projectors to text records conservatively — a *hard*
//!   quant-compatibility gate (file-type labels when both known, else filename quant tags), then
//!   recall ≥ 0.8 with a deterministic path tie-break.

use std::collections::HashSet;

use daemon_common::{InstalledModel, ModelSource};

use crate::gguf;

/// The download-side pairing threshold (repo-listing scan; bonuses included).
pub const HF_PAIR_THRESHOLD: f64 = 0.75;

/// The local pairing threshold (cataloged records; pure recall after the quant gate).
pub const LOCAL_PAIR_THRESHOLD: f64 = 0.8;

/// The analyzed parts of a GGUF filename stem.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StemInfo {
    /// The normalized quant tag (e.g. `Q8_0`, `Q4_K_M`, `F16`), when recognizable.
    pub quant_tag: Option<String>,
    /// Normalized name tokens with the mmproj/projector markers and the quant tag removed.
    pub tokens: Vec<String>,
    /// Whether the stem carries an mmproj/projector marker (`mm-proj` normalizes to `mmproj`).
    pub is_mmproj_hint: bool,
}

/// The filename stem (final path component, `.gguf` stripped) of a repo-relative path.
fn stem_of(path: &str) -> &str {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.strip_suffix(".gguf")
        .or_else(|| name.strip_suffix(".GGUF"))
        .unwrap_or(name)
}

/// Analyze a filename stem into quant tag + normalized tokens + the mmproj hint.
pub fn analyze_stem(stem: &str) -> StemInfo {
    let norm = stem.to_ascii_lowercase().replace("mm-proj", "mmproj");
    let is_mmproj_hint = norm.contains("mmproj") || norm.contains("projector");
    let quant_tag = gguf::quant_label(stem);

    // Remove the quant substring before tokenizing so its fragments (`q4`/`k`/`m`) never pollute
    // the recall score.
    let mut s = norm;
    if let Some(q) = &quant_tag {
        let ql = q.to_ascii_lowercase();
        if let Some(pos) = s.find(&ql) {
            s.replace_range(pos..pos + ql.len(), "-");
        }
    }
    let tokens = s
        .split(['-', '.', '_'])
        .map(str::trim)
        .filter(|t| !t.is_empty() && *t != "mmproj" && *t != "projector")
        .map(str::to_string)
        .collect();

    StemInfo {
        quant_tag,
        tokens,
        is_mmproj_hint,
    }
}

/// Analyze a repo-relative path (or bare filename) by its stem.
pub fn analyze_path(path: &str) -> StemInfo {
    analyze_stem(stem_of(path))
}

/// Whether a filename/path carries the mmproj/projector marker (the "never a chat model" hint).
pub fn is_mmproj_path(path: &str) -> bool {
    analyze_path(path).is_mmproj_hint
}

/// Whether a cataloged model is a vision-projector artifact — the mmproj/projector filename hint
/// (on the HF file name or the on-disk file name) or the authoritative `arch == "clip"` GGUF
/// metadata. Projector records stay in the catalog (inventory / uninstall) but are never chat
/// models: excluded from offers and rejected by activate/resolve.
pub fn is_projector_record(record: &InstalledModel) -> bool {
    if record
        .arch
        .as_deref()
        .is_some_and(|a| a.eq_ignore_ascii_case("clip"))
    {
        return true;
    }
    let hf_file_hint = match &record.model.source {
        ModelSource::Hf { file: Some(f), .. } => is_mmproj_path(f),
        _ => false,
    };
    let local_hint = record
        .local_path
        .file_name()
        .is_some_and(|n| is_mmproj_path(&n.to_string_lossy()));
    hf_file_hint || local_hint
}

/// Mmproj-token recall: `|mmproj ∩ model| / |mmproj|` over normalized tokens.
pub fn token_recall_score(model_tokens: &[String], mmproj_tokens: &[String]) -> f64 {
    if mmproj_tokens.is_empty() {
        return 0.0;
    }
    let model: HashSet<&str> = model_tokens.iter().map(String::as_str).collect();
    let common = mmproj_tokens
        .iter()
        .filter(|t| model.contains(t.as_str()))
        .count();
    common as f64 / mmproj_tokens.len() as f64
}

/// The download-side pair score for a (text model, projector candidate) stem pair: token recall,
/// `+0.10` when the quant tags match, `+0.02` when the candidate says "mmproj" (naming preference
/// over "projector" on ties).
pub fn hf_pair_score(model: &StemInfo, candidate_path: &str, candidate: &StemInfo) -> f64 {
    let mut score = token_recall_score(&model.tokens, &candidate.tokens);
    if let (Some(mq), Some(cq)) = (&model.quant_tag, &candidate.quant_tag) {
        if mq == cq {
            score += 0.10;
        }
    }
    let lower = stem_of(candidate_path)
        .to_ascii_lowercase()
        .replace("mm-proj", "mmproj");
    if lower.contains("mmproj") {
        score += 0.02;
    }
    score
}

/// The best projector companion for `model_file` in a repo listing of `(path, size)` entries.
/// Returns `(path, size)` when the best candidate clears [`HF_PAIR_THRESHOLD`]; `None` when the
/// requested file is itself an mmproj (the reference rule: never pair a projector download).
pub fn best_companion<'a>(
    model_file: &str,
    listing: impl IntoIterator<Item = (&'a str, u64)>,
) -> Option<(String, u64)> {
    let model = analyze_path(model_file);
    if model.is_mmproj_hint {
        return None;
    }
    let mut best: Option<(f64, String, u64)> = None;
    for (path, size) in listing {
        if !gguf::is_gguf(path) || path == model_file {
            continue;
        }
        let cand = analyze_path(path);
        if !cand.is_mmproj_hint {
            continue;
        }
        let score = hf_pair_score(&model, path, &cand);
        let better = match &best {
            Some((s, p, _)) => score > *s || (score == *s && path < p.as_str()),
            None => true,
        };
        if better {
            best = Some((score, path.to_string(), size));
        }
    }
    best.and_then(|(score, path, size)| (score >= HF_PAIR_THRESHOLD).then_some((path, size)))
}

/// The conservative quant-compatibility gate for local pairing: authoritative file-type labels
/// when both sides carry one, else filename quant tags when both are known, else compatible.
pub fn quant_compatible(
    model_file_type: Option<&str>,
    model_quant: Option<&str>,
    proj_file_type: Option<&str>,
    proj_quant: Option<&str>,
) -> bool {
    if let (Some(a), Some(b)) = (model_file_type, proj_file_type) {
        return a.eq_ignore_ascii_case(b);
    }
    if let (Some(a), Some(b)) = (model_quant, proj_quant) {
        return a.eq_ignore_ascii_case(b);
    }
    true
}

/// One candidate the local pairing scan considers (a projector-classified catalog record).
#[derive(Clone, Debug)]
pub struct LocalCandidate {
    /// The on-disk path of the projector artifact.
    pub path: String,
    /// The record's authoritative GGUF file-type label, when known.
    pub file_type: Option<String>,
    /// The record's filename quant tag, when known.
    pub quant: Option<String>,
}

/// The best local projector companion for a text record: hard quant gate, recall ≥
/// [`LOCAL_PAIR_THRESHOLD`], deterministic lexicographic path tie-break.
pub fn best_registry_companion(
    model_stem: &str,
    model_file_type: Option<&str>,
    candidates: &[LocalCandidate],
) -> Option<String> {
    let model = analyze_stem(model_stem);
    if model.is_mmproj_hint {
        return None;
    }
    let mut best: Option<(f64, String)> = None;
    for cand in candidates {
        let info = analyze_path(&cand.path);
        if !quant_compatible(
            model_file_type,
            model.quant_tag.as_deref(),
            cand.file_type.as_deref(),
            cand.quant.as_deref(),
        ) {
            continue;
        }
        let score = token_recall_score(&model.tokens, &info.tokens);
        let better = match &best {
            Some((s, p)) => score > *s || (score == *s && cand.path < *p),
            None => true,
        };
        if better {
            best = Some((score, cand.path.clone()));
        }
    }
    best.and_then(|(score, path)| (score >= LOCAL_PAIR_THRESHOLD).then_some(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recall(model_stem: &str, proj_stem: &str) -> f64 {
        let m = analyze_stem(model_stem);
        let p = analyze_stem(proj_stem);
        token_recall_score(&m.tokens, &p.tokens)
    }

    /// The reference README vectors: hint + quant tag recognized, pair recall ≥ 0.8.
    #[test]
    fn reference_vectors_pair() {
        for (model, proj) in [
            (
                "mradermacher-olmOCR-2-7B-1025.Q8_0",
                "olmOCR-2-7B-1025.mmproj-Q8_0",
            ),
            ("prithivMLmods-chandra_Q8_0", "chandra-mmproj-q8_0"),
            ("sabafallah-deepseek-ocr_q8_0", "mmproj-deepseek-ocr-q8_0"),
        ] {
            let p = analyze_stem(proj);
            assert!(p.is_mmproj_hint, "{proj} should hint mmproj");
            assert_eq!(p.quant_tag.as_deref(), Some("Q8_0"), "{proj} quant");
            let m = analyze_stem(model);
            assert_eq!(m.quant_tag.as_deref(), Some("Q8_0"), "{model} quant");
            assert!(recall(model, proj) >= 0.8, "{model} vs {proj}");
        }
    }

    /// The SmolVLM shape that triggered the fatal: the projector is hinted, tokens match the text
    /// weights fully, and the *text* stem carries no mmproj hint.
    #[test]
    fn smolvlm_vectors() {
        let proj = analyze_stem("mmproj-SmolVLM-256M-Instruct-Q8_0");
        assert!(proj.is_mmproj_hint);
        assert_eq!(proj.quant_tag.as_deref(), Some("Q8_0"));
        assert_eq!(proj.tokens, vec!["smolvlm", "256m", "instruct"]);

        let text = analyze_stem("SmolVLM-256M-Instruct-f16");
        assert!(!text.is_mmproj_hint);
        assert_eq!(text.quant_tag.as_deref(), Some("F16"));
        assert!(
            (recall(
                "SmolVLM-256M-Instruct-f16",
                "mmproj-SmolVLM-256M-Instruct-Q8_0"
            ) - 1.0)
                .abs()
                < 1e-9
        );
    }

    /// `mm-proj` / `projector` spellings normalize to the hint.
    #[test]
    fn hint_spellings_normalize() {
        assert!(analyze_stem("foo-mm-proj-q8_0").is_mmproj_hint);
        assert!(analyze_stem("foo.projector.f16").is_mmproj_hint);
        assert!(!analyze_stem("foo-project-q8_0").is_mmproj_hint);
        assert!(is_mmproj_path("gguf/mmproj-model-f16.gguf"));
        assert!(!is_mmproj_path("gguf/model-f16.gguf"));
    }

    /// A quant mismatch gates the conservative local pairing (the reference hard rule).
    #[test]
    fn quant_mismatch_gates_local_pairing() {
        let a = analyze_stem("foo.Q8_0");
        let b = analyze_stem("foo.mmproj-Q4_K_M");
        assert!(!quant_compatible(
            None,
            a.quant_tag.as_deref(),
            None,
            b.quant_tag.as_deref()
        ));
        // File-type labels win over filename tags when both sides carry one.
        assert!(quant_compatible(
            Some("F16"),
            a.quant_tag.as_deref(),
            Some("f16"),
            b.quant_tag.as_deref()
        ));
        // Unknown quant on either side stays permissive.
        assert!(quant_compatible(None, None, None, Some("Q8_0")));
    }

    /// Download-side companion pick: recall + quant bonus + naming bonus, threshold 0.75; the
    /// same-quant projector wins over a mismatched one.
    #[test]
    fn best_companion_prefers_matching_quant() {
        let listing = [
            ("SmolVLM-256M-Instruct-Q8_0.gguf", 200u64),
            ("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf", 100u64),
            ("mmproj-SmolVLM-256M-Instruct-f16.gguf", 190u64),
            ("README.md", 1u64),
        ];
        let best = best_companion(
            "SmolVLM-256M-Instruct-Q8_0.gguf",
            listing.iter().map(|(p, s)| (*p, *s)),
        );
        assert_eq!(
            best,
            Some(("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf".to_string(), 100))
        );
    }

    /// A quant mismatch only loses the download-side bonus: the full-recall projector still pairs.
    #[test]
    fn best_companion_pairs_across_quants() {
        let listing = [("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf", 100u64)];
        let best = best_companion(
            "SmolVLM-256M-Instruct-f16.gguf",
            listing.iter().map(|(p, s)| (*p, *s)),
        );
        assert_eq!(
            best,
            Some(("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf".to_string(), 100))
        );
    }

    /// An unrelated projector stays unpaired (below threshold), and an mmproj request never pairs.
    #[test]
    fn best_companion_thresholds_and_mmproj_requests() {
        let listing = [("mmproj-totally-different-model-Q8_0.gguf", 100u64)];
        assert_eq!(
            best_companion(
                "SmolVLM-256M-Instruct-Q8_0.gguf",
                listing.iter().map(|(p, s)| (*p, *s)),
            ),
            None,
            "unrelated projector must not pair"
        );
        let listing = [("mmproj-SmolVLM-256M-Instruct-f16.gguf", 100u64)];
        assert_eq!(
            best_companion(
                "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
                listing.iter().map(|(p, s)| (*p, *s)),
            ),
            None,
            "an mmproj request never looks up a companion"
        );
    }

    /// Local registry pairing: quant gate + 0.8 recall threshold + deterministic tie-break.
    #[test]
    fn best_registry_companion_gates_and_tie_breaks() {
        let cands = vec![
            LocalCandidate {
                path: "/hub/b/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf".into(),
                file_type: None,
                quant: Some("Q8_0".into()),
            },
            LocalCandidate {
                path: "/hub/a/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf".into(),
                file_type: None,
                quant: Some("Q8_0".into()),
            },
        ];
        // Same score both: lexicographically smaller path wins (deterministic).
        assert_eq!(
            best_registry_companion("SmolVLM-256M-Instruct-Q8_0", None, &cands).as_deref(),
            Some("/hub/a/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")
        );
        // The hard quant gate: an F16 text model does not auto-link a Q8_0 projector locally.
        assert_eq!(
            best_registry_companion("SmolVLM-256M-Instruct-f16", None, &cands),
            None
        );
        // Below-threshold recall stays unpaired even when quant-compatible.
        let unrelated = vec![LocalCandidate {
            path: "/hub/mmproj-other-model-Q8_0.gguf".into(),
            file_type: None,
            quant: Some("Q8_0".into()),
        }];
        assert_eq!(
            best_registry_companion("SmolVLM-256M-Instruct-Q8_0", None, &unrelated),
            None
        );
    }
}
