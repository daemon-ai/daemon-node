// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! LLM conflict detection — port of `llm_conflict_detector.py`.
//!
//! Opt-in tier-2 validator (`MNEMOSYNE_LLM_CONFLICT_DETECTION`) layered atop the deterministic
//! `(subject, predicate)` veracity-contradiction path that records `conflicts` rows during
//! consolidation. In the Rust port the LLM call routes through the daemon-core `Provider` (the same
//! seam used by [`crate::extract::Extractor`]); Python's exponential-backoff retry ladder
//! (`_call_conflict_llm_with_retry` L86-L128: 2 retries, 1s/2s delays) is ported, while the bespoke
//! HTTP client and sampling params (temperature 0.0) stay host concerns — the daemon-core `Request`
//! carries no sampling knobs. Cost accounting ports via [`crate::cost_log`]
//! ([`validate_conflict_pair_logged`] estimates tokens at ~4 chars each and appends a
//! fire-and-forget `cost_entries` row, `llm_conflict_detector.py` L184-L209). The prompt + JSON
//! contract are ported verbatim from `validate_conflict_pair` L135-L214.

use crate::extract::Extractor;
use serde::Deserialize;

/// The verdict returned by the LLM (`validate_conflict_pair` JSON schema L157-L163).
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ConflictVerdict {
    /// Whether the newer memory contradicts/supersedes the older one.
    #[serde(default)]
    pub is_conflict: bool,
    /// Verdict confidence `[0, 1]`.
    #[serde(default)]
    pub confidence: f64,
    /// The correct fact, summarized (when the model supplies it).
    #[serde(default)]
    pub correct_fact: Option<String>,
}

/// Build the conflict-validation prompt (`validate_conflict_pair` L147-L164), verbatim.
fn build_prompt(older_content: &str, newer_content: &str) -> String {
    format!(
        "You are an advanced agentic memory consolidation engine. Your task is to analyze two \
         memories and determine if they represent a factual contradiction or a conflict (where the \
         newer memory corrects, updates, or overrides the older one).\n\n\
         Older Memory: \"{older_content}\"\n\
         Newer Memory: \"{newer_content}\"\n\n\
         Analyze them carefully:\n\
         - If they are about different subjects or unrelated, there is NO conflict.\n\
         - If they represent chronological updates, corrections of errors, or changed preferences \
         (e.g. \"I love apples\" corrected by \"Actually I prefer oranges now\", or \"event is May \
         29\" vs \"event is June 5\"), this IS a conflict where the newer memory overrides the older \
         one.\n\
         - If they are near-duplicates or additions that complement each other without factual \
         contradiction, there is NO conflict.\n\n\
         You must respond ONLY with a valid JSON object matching this schema:\n\
         {{\n  \"is_conflict\": true or false,\n  \"confidence\": 0.0 to 1.0,\n  \
         \"correct_fact\": \"The correct fact summarized\",\n  \"reason\": \"Brief explanation\"\n}}\n"
    )
}

/// Isolate the JSON object from a model completion, tolerating Markdown fences / trailing prose
/// (`validate_conflict_pair` L171-L177).
fn strip_json(raw: &str) -> &str {
    let mut s = raw.trim();
    if let Some(rest) = s.strip_prefix("```json") {
        s = rest;
    } else if let Some(rest) = s.strip_prefix("```") {
        s = rest;
    }
    if let Some(idx) = s.rfind("```") {
        s = &s[..idx];
    }
    let s = s.trim();
    match (s.find('{'), s.rfind('}')) {
        (Some(a), Some(b)) if b >= a => &s[a..=b],
        _ => s,
    }
}

/// Parse a completion into a [`ConflictVerdict`] (`None` if it isn't the expected JSON object).
pub fn parse_verdict(raw: &str) -> Option<ConflictVerdict> {
    serde_json::from_str::<ConflictVerdict>(strip_json(raw)).ok()
}

/// Max retries after the first failed attempt (`_call_conflict_llm_with_retry` L86).
const MAX_RETRIES: u32 = 2;

/// Initial backoff delay in seconds, doubled per attempt (`_call_conflict_llm_with_retry` L87-L88).
const INITIAL_DELAY_SECS: f64 = 1.0;

/// Run the completion with exponential backoff (1s, 2s — `_call_conflict_llm_with_retry`
/// L90-L128). `None` when every attempt times out / errors.
async fn call_with_retry(extractor: &Extractor, prompt: &str) -> Option<String> {
    for attempt in 0..=MAX_RETRIES {
        if let Some(raw) = extractor.summarize(prompt.to_string()).await {
            return Some(raw);
        }
        if attempt < MAX_RETRIES {
            let delay = INITIAL_DELAY_SECS * 2f64.powi(attempt as i32);
            tokio::time::sleep(std::time::Duration::from_secs_f64(delay)).await;
        }
    }
    None
}

/// Ask the injected LLM whether `newer_content` contradicts/supersedes `older_content`
/// (`validate_conflict_pair` L135-L214), retrying failed calls with exponential backoff. `None`
/// in regex-only mode, when every attempt times out / errors, or when the completion isn't
/// parseable JSON (the Python `(False, 0.0, None)` failure shape; a parse failure is a model
/// answer, not retried).
pub async fn validate_conflict_pair(
    extractor: &Extractor,
    older_content: &str,
    newer_content: &str,
) -> Option<ConflictVerdict> {
    if !extractor.available() {
        return None;
    }
    let raw = call_with_retry(extractor, &build_prompt(older_content, newer_content)).await?;
    parse_verdict(&raw)
}

/// [`validate_conflict_pair`] plus the cost-log write (`llm_conflict_detector.py` L184-L209):
/// tokens are estimated at ~4 chars each for prompt and response, priced at the default tier,
/// and appended to the bank-adjacent `cost_log.db` as a fire-and-forget row (`memory_count = 2`
/// — the validated pair). Skipped for in-memory banks (no data dir to co-locate with).
pub async fn validate_conflict_pair_logged(
    extractor: &Extractor,
    engine: &crate::engine::Engine,
    older_content: &str,
    newer_content: &str,
) -> Option<ConflictVerdict> {
    if !extractor.available() {
        return None;
    }
    let prompt = build_prompt(older_content, newer_content);
    let input_t = crate::cost_log::estimate_tokens(&prompt);
    let raw = call_with_retry(extractor, &prompt).await?;
    let output_t = crate::cost_log::estimate_tokens(&raw);
    if engine.is_persistent() {
        let est = crate::cost_log::calculate_cost(input_t, output_t);
        if let Err(e) = crate::cost_log::log_cost(
            &engine.config().data_dir,
            engine.session_id(),
            2,
            input_t + output_t,
            est,
            "host-provider",
        ) {
            tracing::debug!(error = %e, "cost log write failed (non-fatal)");
        }
    }
    parse_verdict(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::Extractor;
    use daemon_core::MockProvider;
    use std::sync::Arc;

    #[test]
    fn parses_fenced_verdict() {
        let raw = "```json\n{\"is_conflict\": true, \"confidence\": 0.9, \
                   \"correct_fact\": \"prefers oranges\", \"reason\": \"updated\"}\n```";
        let v = parse_verdict(raw).expect("verdict");
        assert!(v.is_conflict);
        assert!((v.confidence - 0.9).abs() < 1e-9);
        assert_eq!(v.correct_fact.as_deref(), Some("prefers oranges"));
    }

    #[tokio::test]
    async fn regex_only_returns_none() {
        let e = Extractor::new();
        assert!(validate_conflict_pair(&e, "a", "b").await.is_none());
    }

    #[tokio::test]
    async fn injected_provider_validates() {
        let json = r#"{"is_conflict": true, "confidence": 0.8, "correct_fact": "B"}"#;
        let e = Extractor::with_provider(Arc::new(MockProvider::completing(json)));
        let v = validate_conflict_pair(&e, "old", "new")
            .await
            .expect("verdict");
        assert!(v.is_conflict);
    }
}
