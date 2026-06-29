// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Configuration presets (`daemon-context-lcm-port-spec.md` §12.6).
//!
//! Read-only **suggestion** metadata — the port never mutates live config (`presets.py:14-16`). Two
//! shipped presets tuned for long-context codex engines; [`suggest_preset_for_engine`] picks one by
//! the model's context length. The suggestion is surfaced as a `preset_suggestion` field in
//! `lcm_status`; applying a preset is an operator decision, not an engine side effect.

/// An inert tuning preset (suggestion only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Preset {
    /// The preset name.
    pub name: &'static str,
    /// The suggested compaction threshold (fraction of the context window).
    pub context_threshold: f64,
    /// The suggested verbatim fresh-tail turn count.
    pub fresh_tail_count: usize,
    /// The suggested leaf-chunk size in tokens.
    pub leaf_chunk_tokens: usize,
    /// The inclusive lower bound of the context window (tokens) this preset targets.
    pub min_context_length: usize,
    /// A short human description.
    pub description: &'static str,
}

/// `codex_gpt_long_context` (`presets.py:127-130`): ≥200k windows.
pub const CODEX_GPT_LONG_CONTEXT: Preset = Preset {
    name: "codex_gpt_long_context",
    context_threshold: 0.75,
    fresh_tail_count: 24,
    leaf_chunk_tokens: 8_000,
    min_context_length: 200_000,
    description: "Long-context GPT/codex engines (>=200k window): compact late, keep a deep tail.",
};

/// `codex_spark_context` (`presets.py:127-130`): 110k-200k windows.
pub const CODEX_SPARK_CONTEXT: Preset = Preset {
    name: "codex_spark_context",
    context_threshold: 0.75,
    fresh_tail_count: 16,
    leaf_chunk_tokens: 8_000,
    min_context_length: 110_000,
    description: "Mid-context codex engines (110k-200k window): compact late, leaner tail.",
};

/// Every shipped preset (highest context floor first, so `suggest_preset_for_engine` picks the most
/// specific match).
pub const ALL: &[Preset] = &[CODEX_GPT_LONG_CONTEXT, CODEX_SPARK_CONTEXT];

/// Suggest a preset for an engine with `context_length` tokens (`suggest_preset_for_engine`,
/// `presets.py:316-330`), or `None` when no preset targets that window. Never mutates config.
pub fn suggest_preset_for_engine(context_length: usize) -> Option<&'static Preset> {
    ALL.iter().find(|p| context_length >= p.min_context_length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_most_specific_preset() {
        assert_eq!(
            suggest_preset_for_engine(256_000).unwrap().name,
            "codex_gpt_long_context"
        );
        assert_eq!(
            suggest_preset_for_engine(128_000).unwrap().name,
            "codex_spark_context"
        );
    }

    #[test]
    fn no_preset_for_small_windows() {
        assert!(suggest_preset_for_engine(32_000).is_none());
    }
}
