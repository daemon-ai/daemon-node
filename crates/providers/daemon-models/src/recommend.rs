//! Hardware-aware quant recommender — the "tune"-like helper that picks a quantization a user can
//! actually run, regardless of engine.
//!
//! - **llama**: from a repo's GGUF file listing, choose the highest-quality quant whose on-disk size
//!   fits the memory budget (VRAM when present, else RAM, minus runtime/KV headroom). The result
//!   names the GGUF file to download.
//! - **mistral.rs**: ISQ is applied in-engine to the full-precision repo, so there is no file to
//!   pick; instead choose an ISQ level whose estimated resident size fits the budget, from the
//!   model's parameter count.
//!
//! The quality ordering is a static table aligned with `llama-cpp-4`'s `LlamaFtype` ranking, so we
//! never link the engine just to rank quants.

use daemon_common::{ModelEngine, ModelFile, QuantCandidate, QuantRecommendation};

use crate::gguf;

/// Fraction of the raw memory budget usable for model weights; the remainder is headroom for the KV
/// cache, activations, and runtime overhead.
const WEIGHT_BUDGET_FRACTION: f64 = 0.8;

/// GGUF quant labels ordered best-quality (and largest) first. Mirrors `LlamaFtype`'s quality
/// ranking. Index = rank: a lower index is higher quality.
const LLAMA_QUALITY: &[&str] = &[
    "F32", "BF16", "F16", "FP16", //
    "Q8_K", "Q8_1", "Q8_0", //
    "Q6_K", //
    "Q5_K_M", "Q5_K", "Q5_K_S", "Q5_1", "Q5_0", //
    "Q4_K_M", "Q4_K", "Q4_K_S", "Q4_1", "Q4_0", //
    "IQ4_NL", "IQ4_XS", //
    "Q3_K_L", "Q3_K_M", "Q3_K", "Q3_K_S", //
    "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS", //
    "Q2_K_S", "Q2_K", //
    "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", //
    "IQ1_M", "IQ1_S",
];

/// ISQ levels for mistral.rs, best-quality first, with an approximate bits-per-weight used to
/// estimate resident size from the parameter count.
const MISTRALRS_ISQ: &[(&str, f64)] = &[
    ("Q8_0", 8.5),
    ("Q6K", 6.6),
    ("Q5K", 5.5),
    ("Q4K", 4.5),
    ("Q3K", 3.4),
    ("Q2K", 2.6),
];

/// The quality rank of a quant label (lower is better); unknown labels rank worst.
fn quality_rank(label: &str) -> usize {
    let upper = label.to_ascii_uppercase();
    LLAMA_QUALITY
        .iter()
        .position(|q| *q == upper)
        .unwrap_or(LLAMA_QUALITY.len())
}

/// The portion of `budget_bytes` to fit weights into.
fn weight_budget(budget_bytes: u64) -> u64 {
    (budget_bytes as f64 * WEIGHT_BUDGET_FRACTION) as u64
}

/// One aggregated GGUF candidate: a quant, the file to name when downloading (first shard / single
/// file), and the total on-disk size (summed across shards).
struct LlamaCandidate {
    quant: String,
    file: String,
    size_bytes: u64,
}

/// Aggregate a repo's GGUF files into per-quant candidates: split shards of the same quant are
/// summed and represented by their first shard (the file a client names to pull the whole set).
fn aggregate_llama(files: &[ModelFile]) -> Vec<LlamaCandidate> {
    use std::collections::BTreeMap;
    // quant -> (named file, total size). The named file prefers a first shard / single file.
    let mut by_quant: BTreeMap<String, (Option<String>, u64)> = BTreeMap::new();
    for f in files {
        if !gguf::is_gguf(&f.path) {
            continue;
        }
        let Some(quant) = f.quant.clone().or_else(|| gguf::quant_label(&f.path)) else {
            continue;
        };
        let entry = by_quant.entry(quant).or_insert((None, 0));
        entry.1 = entry.1.saturating_add(f.size_bytes);
        let nameable = !f.is_split || f.is_first_shard;
        if nameable && entry.0.is_none() {
            entry.0 = Some(f.path.clone());
        }
    }
    by_quant
        .into_iter()
        .filter_map(|(quant, (file, size))| {
            file.map(|file| LlamaCandidate {
                quant,
                file,
                size_bytes: size,
            })
        })
        .collect()
}

/// Recommend a GGUF quant for llama from a repo's file listing and a memory budget.
pub fn recommend_llama(
    repo: &str,
    files: &[ModelFile],
    budget_bytes: u64,
) -> QuantRecommendation {
    let usable = weight_budget(budget_bytes);
    let mut aggregated = aggregate_llama(files);
    // Best quality first.
    aggregated.sort_by(|a, b| {
        quality_rank(&a.quant)
            .cmp(&quality_rank(&b.quant))
            .then_with(|| b.size_bytes.cmp(&a.size_bytes))
    });

    let candidates: Vec<QuantCandidate> = aggregated
        .iter()
        .map(|c| QuantCandidate {
            quant: c.quant.clone(),
            file: Some(c.file.clone()),
            size_bytes: Some(c.size_bytes),
            fits: c.size_bytes <= usable,
        })
        .collect();

    if aggregated.is_empty() {
        return QuantRecommendation {
            engine: ModelEngine::Llama,
            repo: repo.to_string(),
            file: None,
            quant: String::new(),
            size_bytes: None,
            budget_bytes,
            fits: false,
            reason: "no recognizable GGUF quant found in the repo".to_string(),
            candidates,
        };
    }

    // The best-quality candidate that fits the weight budget; else the smallest available.
    let chosen = aggregated
        .iter()
        .find(|c| c.size_bytes <= usable)
        .unwrap_or_else(|| {
            aggregated
                .iter()
                .min_by_key(|c| c.size_bytes)
                .expect("non-empty")
        });
    let fits = chosen.size_bytes <= usable;
    let reason = if fits {
        format!(
            "highest-quality GGUF fitting ~{} of budget ({} usable for weights)",
            human_bytes(budget_bytes),
            human_bytes(usable)
        )
    } else {
        format!(
            "smallest available GGUF; none fit the ~{} budget — expect offload/swap",
            human_bytes(budget_bytes)
        )
    };

    QuantRecommendation {
        engine: ModelEngine::Llama,
        repo: repo.to_string(),
        file: Some(chosen.file.clone()),
        quant: chosen.quant.clone(),
        size_bytes: Some(chosen.size_bytes),
        budget_bytes,
        fits,
        reason,
        candidates,
    }
}

/// Recommend an ISQ level for mistral.rs from the model's parameter count and a memory budget.
pub fn recommend_mistralrs(
    repo: &str,
    num_parameters: Option<u64>,
    budget_bytes: u64,
) -> QuantRecommendation {
    let usable = weight_budget(budget_bytes);

    let Some(params) = num_parameters.filter(|p| *p > 0) else {
        // Without a parameter count we cannot size the budget; default to a broadly safe level.
        let candidates = MISTRALRS_ISQ
            .iter()
            .map(|(q, _)| QuantCandidate {
                quant: (*q).to_string(),
                file: None,
                size_bytes: None,
                fits: true,
            })
            .collect();
        return QuantRecommendation {
            engine: ModelEngine::MistralRs,
            repo: repo.to_string(),
            file: None,
            quant: "Q4K".to_string(),
            size_bytes: None,
            budget_bytes,
            fits: true,
            reason: "parameter count unknown; defaulting to Q4K ISQ".to_string(),
            candidates,
        };
    };

    let est = |bits: f64| -> u64 { ((params as f64) * bits / 8.0) as u64 };
    let candidates: Vec<QuantCandidate> = MISTRALRS_ISQ
        .iter()
        .map(|(q, bits)| {
            let size = est(*bits);
            QuantCandidate {
                quant: (*q).to_string(),
                file: None,
                size_bytes: Some(size),
                fits: size <= usable,
            }
        })
        .collect();

    let chosen = MISTRALRS_ISQ
        .iter()
        .find(|(_, bits)| est(*bits) <= usable)
        .or_else(|| MISTRALRS_ISQ.last())
        .expect("non-empty isq table");
    let chosen_size = est(chosen.1);
    let fits = chosen_size <= usable;
    let reason = if fits {
        format!(
            "highest-quality ISQ fitting ~{} of budget for a {} model",
            human_bytes(budget_bytes),
            human_params(params)
        )
    } else {
        format!(
            "lowest ISQ level; a {} model does not fit ~{} — expect offload/swap",
            human_params(params),
            human_bytes(budget_bytes)
        )
    };

    QuantRecommendation {
        engine: ModelEngine::MistralRs,
        repo: repo.to_string(),
        file: None,
        quant: chosen.0.to_string(),
        size_bytes: Some(chosen_size),
        budget_bytes,
        fits,
        reason,
        candidates,
    }
}

/// Pick the highest-precision GGUF in a repo to use as a quantization *source* (F32 > F16/BF16 >
/// Q8_0 > …). Quantizing from a higher-precision source yields better results; this is the inverse
/// of the budget-fit pick. Returns `None` if the repo has no recognizable GGUF.
pub fn highest_precision_gguf(files: &[ModelFile]) -> Option<&ModelFile> {
    files
        .iter()
        .filter(|f| gguf::is_gguf(&f.path))
        .filter(|f| !f.is_split || f.is_first_shard)
        .filter(|f| f.quant.is_some() || gguf::quant_label(&f.path).is_some())
        .min_by(|a, b| {
            let qa = a.quant.clone().or_else(|| gguf::quant_label(&a.path)).unwrap_or_default();
            let qb = b.quant.clone().or_else(|| gguf::quant_label(&b.path)).unwrap_or_default();
            quality_rank(&qa)
                .cmp(&quality_rank(&qb))
                .then_with(|| b.size_bytes.cmp(&a.size_bytes))
        })
}

/// A compact human-readable byte size (GiB/MiB).
fn human_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.1} GiB", b / GIB)
    } else {
        format!("{:.0} MiB", b / MIB)
    }
}

/// A compact human-readable parameter count (e.g. `7B`, `1.5B`, `350M`).
fn human_params(params: u64) -> String {
    const B: f64 = 1e9;
    const M: f64 = 1e6;
    let p = params as f64;
    if p >= B {
        format!("{:.1}B", p / B)
    } else {
        format!("{:.0}M", p / M)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gguf_file(path: &str, size_bytes: u64) -> ModelFile {
        ModelFile {
            quant: gguf::quant_label(path),
            is_first_shard: gguf::is_first_shard(path),
            is_split: gguf::shard_spec(path).is_some(),
            path: path.to_string(),
            size_bytes,
        }
    }

    #[test]
    fn quality_rank_orders_known_quants() {
        assert!(quality_rank("Q8_0") < quality_rank("Q4_K_M"));
        assert!(quality_rank("Q4_K_M") < quality_rank("Q2_K"));
        assert!(quality_rank("F16") < quality_rank("Q8_0"));
        // Unknown labels rank worst.
        assert!(quality_rank("WAT") >= LLAMA_QUALITY.len());
    }

    #[test]
    fn llama_picks_highest_quality_that_fits() {
        let gib = 1024 * 1024 * 1024;
        let files = vec![
            gguf_file("m-Q8_0.gguf", 8 * gib),
            gguf_file("m-Q4_K_M.gguf", 4 * gib),
            gguf_file("m-Q2_K.gguf", 2 * gib),
        ];
        // 8 GiB budget -> 6.4 GiB usable -> Q8_0 (8) doesn't fit, Q4_K_M (4) fits and is best.
        let rec = recommend_llama("org/m", &files, 8 * gib);
        assert_eq!(rec.quant, "Q4_K_M");
        assert_eq!(rec.file.as_deref(), Some("m-Q4_K_M.gguf"));
        assert!(rec.fits);
        assert_eq!(rec.candidates.len(), 3);
    }

    #[test]
    fn llama_falls_back_to_smallest_when_nothing_fits() {
        let gib = 1024 * 1024 * 1024;
        let files = vec![
            gguf_file("m-Q8_0.gguf", 80 * gib),
            gguf_file("m-Q4_K_M.gguf", 40 * gib),
        ];
        let rec = recommend_llama("org/m", &files, 4 * gib);
        assert_eq!(rec.quant, "Q4_K_M");
        assert!(!rec.fits);
    }

    #[test]
    fn llama_sums_split_shards() {
        let gib = 1024 * 1024 * 1024;
        let files = vec![
            gguf_file("m-Q4_K_M-00001-of-00002.gguf", 3 * gib),
            gguf_file("m-Q4_K_M-00002-of-00002.gguf", 3 * gib),
        ];
        let rec = recommend_llama("org/m", &files, 100 * gib);
        // Named by the first shard, sized as the sum.
        assert_eq!(rec.file.as_deref(), Some("m-Q4_K_M-00001-of-00002.gguf"));
        assert_eq!(rec.size_bytes, Some(6 * gib));
    }

    #[test]
    fn llama_empty_when_no_gguf_quant() {
        let rec = recommend_llama("org/m", &[], 8 * 1024 * 1024 * 1024);
        assert!(!rec.fits);
        assert!(rec.file.is_none());
    }

    #[test]
    fn mistralrs_picks_isq_by_params_and_budget() {
        let gib = 1024 * 1024 * 1024;
        // 7B params, 24 GiB budget -> 19.2 GiB usable -> Q8_0 (~7.4 GiB) fits and is best.
        let rec = recommend_mistralrs("org/m", Some(7_000_000_000), 24 * gib);
        assert_eq!(rec.quant, "Q8_0");
        assert!(rec.fits);
    }

    #[test]
    fn mistralrs_downshifts_on_small_budget() {
        let gib = 1024 * 1024 * 1024;
        // 70B params, 24 GiB budget -> needs a low ISQ; ensure it does not pick Q8_0.
        let rec = recommend_mistralrs("org/m", Some(70_000_000_000), 24 * gib);
        assert_ne!(rec.quant, "Q8_0");
    }

    #[test]
    fn mistralrs_defaults_without_params() {
        let rec = recommend_mistralrs("org/m", None, 24 * 1024 * 1024 * 1024);
        assert_eq!(rec.quant, "Q4K");
        assert!(rec.fits);
    }
}
