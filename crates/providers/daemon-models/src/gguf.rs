//! GGUF filename heuristics + a magic-byte preflight.
//!
//! Ported (Rust-idiomatic, regex-free) from the old `HuggingFaceService` filename parsing: detect
//! the quantization label, recognize multi-part (split) shards by the llama.cpp
//! `-NNNNN-of-NNNNN.gguf` naming, and verify a downloaded file actually begins with the `GGUF`
//! magic before we catalog it.

use std::path::Path;

/// The `GGUF` magic at the head of every GGUF v2/v3 file.
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Known GGUF quantization labels, longest-first so a scan matches the most specific token (e.g.
/// `Q4_K_M` before `Q4_K` before `Q4`).
const QUANT_LABELS: &[&str] = &[
    "Q2_K_S", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K", "Q4_K_S", "Q4_K_M", "Q4_K", "Q4_0",
    "Q4_1", "Q5_K_S", "Q5_K_M", "Q5_K", "Q5_0", "Q5_1", "Q6_K", "Q8_0", "Q8_1", "Q8_K",
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    "IQ4_XS", "IQ4_NL", "BF16", "F16", "F32", "FP16",
];

/// Whether a repo file is a GGUF artifact (by extension).
pub fn is_gguf(path: &str) -> bool {
    path.to_ascii_lowercase().ends_with(".gguf")
}

/// The quantization label embedded in a filename, if recognizable (case-insensitive scan over the
/// known label set, most-specific first).
pub fn quant_label(filename: &str) -> Option<String> {
    let upper = filename.to_ascii_uppercase();
    QUANT_LABELS
        .iter()
        .find(|label| upper.contains(*label))
        .map(|label| (*label).to_string())
}

/// A parsed multi-part shard descriptor: `(index, total)` from `…-NNNNN-of-NNNNN.gguf`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardSpec {
    /// The 1-based shard index.
    pub index: u32,
    /// The total shard count.
    pub total: u32,
}

/// Recognize the llama.cpp split-shard naming `…-NNNNN-of-NNNNN.gguf` and extract `(index, total)`.
pub fn shard_spec(filename: &str) -> Option<ShardSpec> {
    let stem = filename.strip_suffix(".gguf").or_else(|| {
        let lower = filename.to_ascii_lowercase();
        lower.ends_with(".gguf").then(|| &filename[..filename.len() - 5])
    })?;
    // Find the "-of-" separator and read the 5-digit groups around it.
    let of = stem.rfind("-of-")?;
    let (before, after) = (&stem[..of], &stem[of + 4..]);
    let index_str = before.rsplit('-').next()?;
    let total_str = after;
    if index_str.len() != 5 || total_str.len() != 5 {
        return None;
    }
    let index: u32 = index_str.parse().ok()?;
    let total: u32 = total_str.parse().ok()?;
    (total > 0 && index >= 1 && index <= total).then_some(ShardSpec { index, total })
}

/// Whether a filename is the *first* shard of a split GGUF set (the file a client names to pull the
/// whole set).
pub fn is_first_shard(filename: &str) -> bool {
    matches!(shard_spec(filename), Some(s) if s.index == 1)
}

/// Given the first shard's filename, the full set of shard filenames (`00001-of-N … N-of-N`).
pub fn shard_set(first_shard: &str) -> Option<Vec<String>> {
    let spec = shard_spec(first_shard)?;
    let of = first_shard.rfind("-of-")?;
    let total_start = of + 4;
    // The index group is the 5 digits immediately before "-of-".
    let index_start = of - 5;
    let prefix = &first_shard[..index_start];
    let suffix = &first_shard[total_start + 5..]; // typically ".gguf"
    let total_str = &first_shard[total_start..total_start + 5];
    Some(
        (1..=spec.total)
            .map(|i| format!("{prefix}{i:05}-of-{total_str}{suffix}"))
            .collect(),
    )
}

/// Verify a file on disk begins with the GGUF magic. Cheap integrity gate before cataloging a
/// freshly downloaded GGUF (catches truncated / HTML-error-page downloads).
pub fn verify_gguf_magic(path: &Path) -> std::io::Result<bool> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut head = [0u8; 4];
    match f.read_exact(&mut head) {
        Ok(()) => Ok(&head == GGUF_MAGIC),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_quant_labels() {
        assert_eq!(quant_label("Model-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(quant_label("foo.q8_0.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(quant_label("model-f16.gguf").as_deref(), Some("F16"));
        assert_eq!(quant_label("tokenizer.json"), None);
    }

    #[test]
    fn detects_split_shards() {
        let first = "Meta-Llama-3-70B-Q4_K_M-00001-of-00009.gguf";
        let spec = shard_spec(first).unwrap();
        assert_eq!(spec, ShardSpec { index: 1, total: 9 });
        assert!(is_first_shard(first));
        assert!(!is_first_shard(
            "Meta-Llama-3-70B-Q4_K_M-00002-of-00009.gguf"
        ));
        let set = shard_set(first).unwrap();
        assert_eq!(set.len(), 9);
        assert_eq!(set[0], first);
        assert_eq!(set[8], "Meta-Llama-3-70B-Q4_K_M-00009-of-00009.gguf");
    }

    #[test]
    fn single_file_is_not_a_shard() {
        assert_eq!(shard_spec("Model-Q4_K_M.gguf"), None);
        assert!(!is_first_shard("Model-Q4_K_M.gguf"));
    }
}
