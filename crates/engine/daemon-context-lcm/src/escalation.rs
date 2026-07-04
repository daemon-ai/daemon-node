// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Escalation & auxiliary-provider summarization (`daemon-context-lcm-port-spec.md` §7).
//!
//! A summary must shrink its source or it escalates: **L1** a detailed LLM summary, **L2** aggressive
//! LLM bullets at half budget, **L3** a deterministic head/tail truncation that always converges.
//! Each LLM level tries the aux fallback chain in order; a route's result only counts when it
//! *shrinks* the source (`accepts_result`, `LCM:escalation.py:210-215`) — a non-shrinking or empty
//! reply is recorded as a route failure and the next route is tried. L2 runs even when the whole L1
//! chain failed (`LCM:escalation.py:325-342`). A per-route [`SummaryCircuitBreaker`] (2 failures /
//! 300s) skips a failing aux provider. The aux model is a `daemon-core` [`Provider`]; per-call model
//! routing collapses to "the provider is the model" (§7.4), so escalation just builds a one-message
//! [`Request`] and calls [`Provider::chat`] under a timeout.

use crate::tokens::Tokenizer;
use daemon_core::provider::Failure;
use daemon_core::{Provider, Request, RequestMsg};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The default failure threshold when no config is supplied (`LCM:config.py:302`).
const DEFAULT_BREAKER_FAILURE_THRESHOLD: u32 = 2;
/// The default cooldown when no config is supplied (`LCM:config.py:304`).
const DEFAULT_BREAKER_COOLDOWN_SECONDS: u64 = 300;
/// The single-line focus-topic bound inside a prompt brief (`_normalized_focus_topic`,
/// `LCM:escalation.py:136`).
const FOCUS_TOPIC_MAX_CHARS: usize = 160;

/// The escalation level a summary was produced at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// Detailed LLM summary.
    L1,
    /// Aggressive LLM bullets (half budget).
    L2,
    /// Deterministic head/tail truncation (no LLM).
    L3,
}

/// A single-route circuit breaker for one aux summarization provider (§7.3). Thresholds are
/// config-driven (`summary_circuit_breaker_*`); a chain carries one breaker per route.
#[derive(Debug)]
pub struct SummaryCircuitBreaker {
    failures: u32,
    opened_at: Option<Instant>,
    failure_threshold: u32,
    cooldown: Duration,
}

impl Default for SummaryCircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

impl SummaryCircuitBreaker {
    /// A fresh, closed breaker with the spec-default thresholds (2 failures / 300s).
    pub fn new() -> Self {
        Self::with_config(
            DEFAULT_BREAKER_FAILURE_THRESHOLD,
            DEFAULT_BREAKER_COOLDOWN_SECONDS,
        )
    }

    /// A fresh, closed breaker with explicit thresholds (`failure_threshold` consecutive failures /
    /// `cooldown_seconds` open window).
    pub fn with_config(failure_threshold: u32, cooldown_seconds: u64) -> Self {
        Self {
            failures: 0,
            opened_at: None,
            failure_threshold: failure_threshold.max(1),
            cooldown: Duration::from_secs(cooldown_seconds),
        }
    }

    /// Whether the route is currently open (skip the LLM, fall to L3). The breaker self-heals after
    /// the cooldown elapses (half-open: the next call is allowed).
    pub fn is_open(&self) -> bool {
        matches!(self.opened_at, Some(t) if t.elapsed() < self.cooldown)
    }

    /// Record a successful call (closes the breaker).
    pub fn record_success(&mut self) {
        self.failures = 0;
        self.opened_at = None;
    }

    /// Record a failed call (opens the breaker at the threshold).
    pub fn record_failure(&mut self) {
        self.failures = self.failures.saturating_add(1);
        if self.failures >= self.failure_threshold {
            self.opened_at = Some(Instant::now());
        }
    }
}

/// The inputs to one escalation (a leaf chunk or a condensation group).
pub struct SummaryRequest<'a> {
    /// The text to summarize.
    pub text: &'a str,
    /// The token count of `text` (the acceptance bar: a summary must be strictly smaller).
    pub source_tokens: usize,
    /// The L1 token budget (leaf: 20% src cap 12k; condense: 40% src).
    pub token_budget: usize,
    /// The DAG depth being produced (drives the L1 guidance).
    pub depth: i64,
    /// An optional focus topic (rendered as a §7 focus brief; `""` for none).
    pub focus_topic: &'a str,
    /// Optional custom instructions (rendered as an "Additional instructions" block; `""` for none).
    pub custom_instructions: &'a str,
}

/// The result of one escalation: the summary text, the level it converged at, and whether the LLM
/// levels fell through on a *retry-worthy* failure (timeout / context-length class — the
/// `_is_retry_worthy_leaf_summary_error` markers, `LCM:engine.py:760-776`). The leaf-rescue ladder
/// (`LCM:engine.py:801-849`) uses that signal to retry with a smaller chunk instead of accepting a
/// degraded L3 truncation of an oversized one.
pub struct Escalated {
    /// The summary text.
    pub text: String,
    /// The level the escalation converged at.
    pub level: Level,
    /// `true` when the result is L3 *and* at least one aux route failed with a retry-worthy
    /// (timeout/context-overflow) failure — the chunk itself may be too big for the aux model.
    pub retry_worthy_failure: bool,
}

/// Summarize with 3-level escalation (§7).
///
/// `aux_chain` is the ordered fallback chain of aux providers (`summary_model` then
/// `summary_fallback_models`); `breakers` is the parallel per-route breaker slice. Each LLM level
/// tries the chain in order (skipping open routes); a route's reply counts only when it is
/// non-empty and strictly smaller than the source (otherwise the route records a failure and the
/// next is tried). L2 runs at `l2_budget_ratio` of the L1 budget even when the whole L1 chain
/// failed; L3 (`l3_truncate_tokens`) is local and always converges. `timeout` bounds each call.
pub async fn summarize_with_escalation(
    aux_chain: &[Arc<dyn Provider>],
    tok: &Tokenizer,
    breakers: &mut [SummaryCircuitBreaker],
    l2_budget_ratio: f64,
    l3_truncate_tokens: usize,
    timeout: Duration,
    req: SummaryRequest<'_>,
) -> Escalated {
    let accepts = |text: &str| tok.count_text(text) < req.source_tokens;

    // L1 — detailed summary over the fallback chain.
    let (l1, rw1) = call_summary_llm_chain(
        aux_chain,
        breakers,
        build_l1_prompt(&req),
        timeout,
        &accepts,
    )
    .await;
    if let Some(text) = l1 {
        return Escalated {
            text,
            level: Level::L1,
            retry_worthy_failure: false,
        };
    }

    // L2 — aggressive bullets at the reduced budget, even when the whole L1 chain failed
    // (`LCM:escalation.py:325-342`).
    let l2_budget = (((req.token_budget as f64) * l2_budget_ratio) as usize).max(1);
    let l2_prompt = build_l2_prompt(&req, l2_budget);
    let (l2, rw2) = call_summary_llm_chain(aux_chain, breakers, l2_prompt, timeout, &accepts).await;
    if let Some(text) = l2 {
        return Escalated {
            text,
            level: Level::L2,
            retry_worthy_failure: false,
        };
    }

    // L3 — deterministic truncation; always converges.
    Escalated {
        text: deterministic_truncate(tok, req.text, l3_truncate_tokens),
        level: Level::L3,
        retry_worthy_failure: rw1 || rw2,
    }
}

/// Try the aux fallback chain for one prompt (`_invoke_summary_llm_chain`, §7.3): skip open routes,
/// call each in order, and record per-route success/failure — where an empty or non-`accepts`
/// (non-shrinking) reply *is* a route failure (`accepts_result`, `LCM:escalation.py:210-215`).
/// Returns the first accepted text (or `None` when every route is open/failed) plus whether any
/// route failed with a retry-worthy (timeout/context-length) failure.
async fn call_summary_llm_chain(
    chain: &[Arc<dyn Provider>],
    breakers: &mut [SummaryCircuitBreaker],
    prompt: String,
    timeout: Duration,
    accepts: &(dyn Fn(&str) -> bool + Sync),
) -> (Option<String>, bool) {
    let mut skipped = 0usize;
    let mut retry_worthy = false;
    for (i, provider) in chain.iter().enumerate() {
        if breakers.get(i).is_some_and(|b| b.is_open()) {
            skipped += 1;
            tracing::warn!(route = i, "lcm: summary route skipped by open circuit");
            continue;
        }
        match call_summary_llm(provider.as_ref(), prompt.clone(), timeout).await {
            Ok(text) => {
                let text = strip_reasoning_blocks(&text);
                if !text.is_empty() && accepts(&text) {
                    if let Some(b) = breakers.get_mut(i) {
                        b.record_success();
                    }
                    return (Some(text), retry_worthy);
                }
                // A healthy route that produced an unusable (empty / non-shrinking) reply still
                // counts as a route failure so the chain advances and the breaker learns.
                if let Some(b) = breakers.get_mut(i) {
                    b.record_failure();
                }
            }
            Err(failure_retry_worthy) => {
                retry_worthy |= failure_retry_worthy;
                if let Some(b) = breakers.get_mut(i) {
                    b.record_failure();
                }
            }
        }
    }
    if skipped == chain.len() {
        tracing::warn!("lcm: summary fallback chain exhausted: all routes are temporarily open");
    }
    (None, retry_worthy)
}

/// Build the request to the aux provider and await its text under `timeout`. On failure returns
/// `Err(retry_worthy)` — whether the failure is timeout/context-length class (the
/// `_is_retry_worthy_leaf_summary_error` markers) and thus a candidate for the leaf-rescue ladder.
async fn call_summary_llm(
    aux: &dyn Provider,
    prompt: String,
    timeout: Duration,
) -> Result<String, bool> {
    let request = Request {
        system: String::new(),
        messages: vec![RequestMsg {
            role: "user".into(),
            content: prompt,
            ..Default::default()
        }],
        tools: Vec::new(),
        auth: None,
        constraint: None,
        cache_system: false,
    };
    match tokio::time::timeout(timeout, aux.chat(request)).await {
        Ok(Ok(out)) => Ok(out.text),
        Ok(Err(failure)) => Err(is_retry_worthy_failure(&failure)),
        Err(_elapsed) => Err(true),
    }
}

/// `_is_retry_worthy_leaf_summary_error` (`LCM:engine.py:760-776`): a timeout or a context-length /
/// payload-size class failure, where a smaller chunk plausibly succeeds.
fn is_retry_worthy_failure(failure: &Failure) -> bool {
    match failure {
        Failure::ContextOverflow(_) | Failure::PayloadTooLarge(_) => true,
        other => {
            const MARKERS: [&str; 10] = [
                "context length",
                "maximum context",
                "max context",
                "too many tokens",
                "token limit",
                "prompt is too long",
                "input too long",
                "request too large",
                "timed out",
                "timeout",
            ];
            let message = other.to_string().to_lowercase();
            MARKERS.iter().any(|m| message.contains(m))
        }
    }
}

/// Depth-aware L1 guidance, verbatim (`LCM:escalation.py:224-228`).
fn depth_guidance(depth: i64) -> &'static str {
    match depth {
        0 => "Preserve decisions, rationale, constraints, active tasks, file paths, commands, and specific values.",
        1 => "Distill into arc-level outcomes: what evolved, what was decided, current state. Drop per-turn detail.",
        _ => "Capture durable narrative: decisions in effect, completed milestones, timeline. Drop process detail.",
    }
}

/// Single-line, bounded focus topic for prompt injection (`_normalized_focus_topic`,
/// `LCM:escalation.py:136-141`).
fn normalized_focus_topic(focus_topic: &str) -> String {
    let normalized = focus_topic.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_with_ellipsis(&normalized, FOCUS_TOPIC_MAX_CHARS)
}

/// Truncate to `max_chars` characters, replacing the tail with `…` (the Python
/// `text[: max_chars - 1].rstrip() + "…"` shape). Shared with the auto-focus derivation.
pub(crate) fn truncate_with_ellipsis(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let head: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{}…", head.trim_end())
}

/// The verbatim L1 focus brief (`_build_l1_focus_brief`, `LCM:escalation.py:144-154`).
fn build_l1_focus_brief(focus_topic: &str) -> String {
    let topic = normalized_focus_topic(focus_topic);
    if topic.is_empty() {
        return String::new();
    }
    format!(
        "Focus brief:\n\
         Primary focus: {topic}\n\
         Preserve concrete decisions, constraints, files, commands, identifiers, and current state for this focus.\n\
         Spend roughly 60-70% of the summary budget on the focus when relevant.\n\
         Do not discard unrelated blockers or active tasks just because they are off-focus.\n"
    )
}

/// The verbatim L2 focus brief (`_build_l2_focus_brief`, `LCM:escalation.py:157-166`).
fn build_l2_focus_brief(focus_topic: &str) -> String {
    let topic = normalized_focus_topic(focus_topic);
    if topic.is_empty() {
        return String::new();
    }
    format!(
        "Focus brief:\n\
         Primary focus: {topic}\n\
         Prefer bullets that preserve decisions, blockers, files, commands, identifiers, and current state for this focus.\n\
         Keep other active tasks only when they are current blockers or handoff state.\n"
    )
}

/// The verbatim custom-instructions block (`LCM:escalation.py:233-235`).
fn custom_block(custom_instructions: &str) -> String {
    if custom_instructions.is_empty() {
        String::new()
    } else {
        format!("\nAdditional instructions:\n{custom_instructions}\n")
    }
}

/// The verbatim L1 prompt (`_build_l1_prompt`, `LCM:escalation.py:221-246`).
fn build_l1_prompt(req: &SummaryRequest<'_>) -> String {
    format!(
        "Summarize this conversation segment for future turns.\n\
         {guidance}\n\
         Remove repetition and conversational filler.\n\
         End with: \"Expand for details about: <what was compressed>\"\n\
         {focus}{custom}\n\n\
         Target ~{budget} tokens.\n\n\
         CONTENT:\n{content}",
        guidance = depth_guidance(req.depth),
        focus = build_l1_focus_brief(req.focus_topic),
        custom = custom_block(req.custom_instructions),
        budget = req.token_budget,
        content = req.text,
    )
}

/// The verbatim L2 prompt (`_build_l2_prompt`, `LCM:escalation.py:249-264`); `budget` is the
/// already-halved L2 budget.
fn build_l2_prompt(req: &SummaryRequest<'_>, budget: usize) -> String {
    format!(
        "Compress this into bullet points. Maximum {budget} tokens.\n\
         Keep only: decisions made, files changed, errors hit, current state.\n\
         Drop all reasoning, alternatives considered, and process detail.\n\
         {focus}{custom}\n\n\
         CONTENT:\n{content}",
        budget = budget,
        focus = build_l2_focus_brief(req.focus_topic),
        custom = custom_block(req.custom_instructions),
        content = req.text,
    )
}

/// Deterministic head/tail truncation (`_deterministic_truncate`, `LCM:escalation.py:267-285`):
/// return the text as-is when it already fits `max_tokens`; else `char_budget = max_tokens * 4`,
/// keep 40% head + 40% tail with a marker between. Operates on char boundaries so multi-byte text
/// is never split.
fn deterministic_truncate(tok: &Tokenizer, text: &str, max_tokens: usize) -> String {
    const MARKER: &str =
        "\n\n[...deterministic truncation — details available via lcm_expand...]\n\n";
    if tok.count_text(text) <= max_tokens {
        return text.to_string();
    }
    let char_budget = max_tokens.saturating_mul(4);
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= char_budget {
        return text.to_string();
    }
    let head = char_budget * 4 / 10;
    let tail = char_budget * 4 / 10;
    let head_s: String = chars.iter().take(head).collect();
    let tail_s: String = chars.iter().skip(chars.len() - tail).collect();
    format!("{head_s}{MARKER}{tail_s}")
}

/// Strip reasoning blocks from summary text before persisting (`_strip_reasoning_blocks`,
/// `LCM:escalation.py:90-95`). Removes `<think>`, `<thinking>`, `<reasoning>`, `<thought>`, and
/// `<REASONING_SCRATCHPAD>` paired blocks (case-insensitive), then trims. Shared with
/// [`crate::extraction`] (the extraction aux call strips reasoning the same way).
pub(crate) fn strip_reasoning_blocks(text: &str) -> String {
    const TAGS: [&str; 5] = [
        "think",
        "thinking",
        "reasoning",
        "thought",
        "reasoning_scratchpad",
    ];
    let mut out = text.to_string();
    for tag in TAGS {
        out = strip_tag(&out, tag);
    }
    out.trim().to_string()
}

/// Remove every `<tag>...</tag>` span (case-insensitive) from `text`.
fn strip_tag(text: &str, tag: &str) -> String {
    let lower = text.to_ascii_lowercase();
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    while let Some(rel) = lower[cursor..].find(&open) {
        let start = cursor + rel;
        out.push_str(&text[cursor..start]);
        match lower[start..].find(&close) {
            Some(end_rel) => cursor = start + end_rel + close.len(),
            None => {
                // Unterminated block: drop the rest.
                cursor = text.len();
                break;
            }
        }
    }
    out.push_str(&text[cursor..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use daemon_core::provider::{Capabilities, Failure, ModelOutput, ToolCallFormat};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// An aux provider that returns a fixed text (or a failure) for every call, counting calls.
    struct FixedAux {
        reply: Option<String>,
        failure: Option<fn() -> Failure>,
        calls: AtomicU64,
    }

    impl FixedAux {
        fn replying(text: &str) -> Self {
            Self {
                reply: Some(text.to_string()),
                failure: None,
                calls: AtomicU64::new(0),
            }
        }

        fn failing() -> Self {
            Self {
                reply: None,
                failure: None,
                calls: AtomicU64::new(0),
            }
        }

        fn failing_with(failure: fn() -> Failure) -> Self {
            Self {
                reply: None,
                failure: Some(failure),
                calls: AtomicU64::new(0),
            }
        }
    }

    #[async_trait]
    impl Provider for FixedAux {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_native_tools: false,
                supports_streaming: false,
                tool_call_format: ToolCallFormat::Native,
                max_context: Some(8192),
            }
        }
        async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            match &self.reply {
                Some(text) => Ok(ModelOutput {
                    text: text.clone(),
                    ..Default::default()
                }),
                None => Err(self
                    .failure
                    .map(|f| f())
                    .unwrap_or_else(|| Failure::Provider("aux down".into()))),
            }
        }
    }

    fn req<'a>(text: &'a str, src: usize) -> SummaryRequest<'a> {
        SummaryRequest {
            text,
            source_tokens: src,
            token_budget: 100,
            depth: 0,
            focus_topic: "",
            custom_instructions: "",
        }
    }

    fn chain(reply: Option<&str>) -> Vec<Arc<dyn Provider>> {
        vec![Arc::new(match reply {
            Some(s) => FixedAux::replying(s),
            None => FixedAux::failing(),
        })]
    }

    #[tokio::test]
    async fn l1_accepts_a_shorter_summary() {
        let aux = chain(Some("short"));
        let tok = Tokenizer::heuristic();
        let mut breakers = vec![SummaryCircuitBreaker::new()];
        let out = summarize_with_escalation(
            &aux,
            &tok,
            &mut breakers,
            0.5,
            512,
            Duration::from_secs(5),
            req("a very long source that is bigger than the summary", 1000),
        )
        .await;
        assert_eq!(out.level, Level::L1);
        assert_eq!(out.text, "short");
        assert!(!out.retry_worthy_failure);
    }

    #[tokio::test]
    async fn l3_converges_when_summary_never_shrinks() {
        // The aux echoes a long reply that never beats source_tokens=1, forcing L1+L2 to fail the
        // shrink test and fall to deterministic truncation.
        let long = "x".repeat(10_000);
        let aux = chain(Some(&long));
        let tok = Tokenizer::heuristic();
        let mut breakers = vec![SummaryCircuitBreaker::new()];
        let out = summarize_with_escalation(
            &aux,
            &tok,
            &mut breakers,
            0.5,
            64,
            Duration::from_secs(5),
            req(&long, 1),
        )
        .await;
        assert_eq!(out.level, Level::L3);
        assert!(out.text.len() < long.len(), "L3 truncates");
        assert!(out.text.contains("deterministic truncation"));
        assert!(
            !out.retry_worthy_failure,
            "non-shrinking replies are not retry-worthy"
        );
    }

    #[tokio::test]
    async fn l2_runs_even_when_the_l1_chain_fails() {
        // A provider that fails the first (L1) call and answers the second (L2) call.
        struct L2OnlyAux {
            calls: AtomicU64,
        }
        #[async_trait]
        impl Provider for L2OnlyAux {
            fn capabilities(&self) -> Capabilities {
                Capabilities {
                    supports_native_tools: false,
                    supports_streaming: false,
                    tool_call_format: ToolCallFormat::Native,
                    max_context: Some(8192),
                }
            }
            async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                if req.messages[0]
                    .content
                    .starts_with("Compress this into bullet points")
                {
                    Ok(ModelOutput {
                        text: "- terse bullets".into(),
                        ..Default::default()
                    })
                } else {
                    Err(Failure::Provider("L1 down".into()))
                }
            }
        }
        let aux: Vec<Arc<dyn Provider>> = vec![Arc::new(L2OnlyAux {
            calls: AtomicU64::new(0),
        })];
        let tok = Tokenizer::heuristic();
        let mut breakers = vec![SummaryCircuitBreaker::new()];
        let out = summarize_with_escalation(
            &aux,
            &tok,
            &mut breakers,
            0.5,
            512,
            Duration::from_secs(5),
            req(
                "a long enough source text for the shrink test to pass",
                1000,
            ),
        )
        .await;
        assert_eq!(out.level, Level::L2, "L2 ran despite the L1 chain failing");
        assert_eq!(out.text, "- terse bullets");
    }

    #[tokio::test]
    async fn non_shrinking_reply_fails_over_to_the_next_route() {
        // Route 0 replies verbosely (never shrinks); route 1 replies tersely. The accepts_result
        // gate must advance the chain within one level.
        let aux: Vec<Arc<dyn Provider>> = vec![
            Arc::new(FixedAux::replying(&"verbose ".repeat(400))),
            Arc::new(FixedAux::replying("terse")),
        ];
        let tok = Tokenizer::heuristic();
        let mut breakers = vec![SummaryCircuitBreaker::new(), SummaryCircuitBreaker::new()];
        let out = summarize_with_escalation(
            &aux,
            &tok,
            &mut breakers,
            0.5,
            512,
            Duration::from_secs(5),
            req("a source text of moderate length for the test", 100),
        )
        .await;
        assert_eq!(out.level, Level::L1);
        assert_eq!(out.text, "terse");
        // The verbose route recorded a failure (breaker not yet open at threshold 2).
        breakers[0].record_failure();
        assert!(breakers[0].is_open(), "first failure was recorded");
    }

    #[tokio::test]
    async fn open_breaker_falls_to_l3_without_calling_the_provider() {
        let aux = chain(None); // would error if called
        let tok = Tokenizer::heuristic();
        let mut breakers = vec![SummaryCircuitBreaker::new()];
        // Two failures open the (single) route.
        breakers[0].record_failure();
        breakers[0].record_failure();
        assert!(breakers[0].is_open());
        let src = "some source text to truncate ".repeat(50);
        let out = summarize_with_escalation(
            &aux,
            &tok,
            &mut breakers,
            0.5,
            16,
            Duration::from_secs(5),
            req(&src, 1),
        )
        .await;
        assert_eq!(out.level, Level::L3);
        assert!(!out.text.is_empty());
        assert!(!out.retry_worthy_failure, "open routes were never called");
    }

    #[tokio::test]
    async fn fallback_route_is_used_when_the_primary_route_is_open() {
        // Primary errors; fallback returns a usable summary. Open the primary first, then the chain
        // should skip it and succeed on the fallback.
        let mut breakers = vec![SummaryCircuitBreaker::new(), SummaryCircuitBreaker::new()];
        breakers[0].record_failure();
        breakers[0].record_failure();
        assert!(breakers[0].is_open());
        let aux: Vec<Arc<dyn Provider>> = vec![
            Arc::new(FixedAux::failing()),
            Arc::new(FixedAux::replying("fallback summary")),
        ];
        let tok = Tokenizer::heuristic();
        let out = summarize_with_escalation(
            &aux,
            &tok,
            &mut breakers,
            0.5,
            512,
            Duration::from_secs(5),
            req(
                "a long source that exceeds the fallback summary length",
                1000,
            ),
        )
        .await;
        assert_eq!(out.level, Level::L1);
        assert_eq!(out.text, "fallback summary");
    }

    #[tokio::test]
    async fn context_overflow_failure_is_retry_worthy() {
        let aux: Vec<Arc<dyn Provider>> = vec![Arc::new(FixedAux::failing_with(|| {
            Failure::ContextOverflow("prompt exceeds the window".into())
        }))];
        let tok = Tokenizer::heuristic();
        let mut breakers = vec![SummaryCircuitBreaker::new()];
        let src = "source ".repeat(100);
        let out = summarize_with_escalation(
            &aux,
            &tok,
            &mut breakers,
            0.5,
            64,
            Duration::from_secs(5),
            req(&src, 1000),
        )
        .await;
        assert_eq!(out.level, Level::L3);
        assert!(out.retry_worthy_failure, "overflow flags the rescue ladder");
    }

    #[tokio::test]
    async fn plain_provider_failure_is_not_retry_worthy() {
        let aux = chain(None); // Failure::Provider("aux down")
        let tok = Tokenizer::heuristic();
        let mut breakers = vec![SummaryCircuitBreaker::new()];
        let src = "source ".repeat(100);
        let out = summarize_with_escalation(
            &aux,
            &tok,
            &mut breakers,
            0.5,
            64,
            Duration::from_secs(5),
            req(&src, 1000),
        )
        .await;
        assert_eq!(out.level, Level::L3);
        assert!(!out.retry_worthy_failure);
    }

    #[test]
    fn prompts_carry_focus_brief_and_custom_instructions_verbatim() {
        let r = SummaryRequest {
            text: "BODY",
            source_tokens: 100,
            token_budget: 50,
            depth: 0,
            focus_topic: "fix   the\nparser bug",
            custom_instructions: "Answer in English.",
        };
        let l1 = build_l1_prompt(&r);
        assert!(l1.contains(
            "Preserve decisions, rationale, constraints, active tasks, file paths, commands, and specific values."
        ));
        assert!(l1.contains("Focus brief:\nPrimary focus: fix the parser bug\n"));
        assert!(
            l1.contains("Spend roughly 60-70% of the summary budget on the focus when relevant.")
        );
        assert!(l1.contains("\nAdditional instructions:\nAnswer in English.\n"));
        assert!(l1.contains("Target ~50 tokens."));
        assert!(l1.ends_with("CONTENT:\nBODY"));

        let l2 = build_l2_prompt(&r, 25);
        assert!(l2.starts_with("Compress this into bullet points. Maximum 25 tokens."));
        assert!(l2.contains("Prefer bullets that preserve decisions, blockers, files, commands, identifiers, and current state for this focus."));
        assert!(l2.contains("\nAdditional instructions:\nAnswer in English.\n"));
        assert!(l2.ends_with("CONTENT:\nBODY"));
    }

    #[test]
    fn depth_guidance_is_verbatim() {
        assert!(depth_guidance(1).starts_with("Distill into arc-level outcomes"));
        assert!(depth_guidance(2).starts_with("Capture durable narrative"));
        assert!(depth_guidance(7).starts_with("Capture durable narrative"));
    }

    #[test]
    fn focus_topic_is_normalized_and_bounded() {
        let long = "word ".repeat(100);
        let brief = build_l1_focus_brief(&long);
        let topic_line = brief.lines().nth(1).unwrap();
        assert!(topic_line.starts_with("Primary focus: "));
        assert!(topic_line.chars().count() <= "Primary focus: ".len() + FOCUS_TOPIC_MAX_CHARS);
        assert!(topic_line.ends_with('…'));
        assert!(build_l1_focus_brief("").is_empty());
        assert!(build_l2_focus_brief("  \n ").is_empty());
    }

    #[test]
    fn deterministic_truncate_returns_short_text_verbatim() {
        let tok = Tokenizer::heuristic();
        assert_eq!(
            deterministic_truncate(&tok, "short text", 512),
            "short text"
        );
        let long = "y".repeat(10_000);
        let out = deterministic_truncate(&tok, &long, 64);
        assert!(out.len() < long.len());
        assert!(out.contains("deterministic truncation"));
    }

    #[test]
    fn strips_reasoning_blocks() {
        let s = "Keep this <think>drop me</think> and <REASONING_SCRATCHPAD>this too</REASONING_SCRATCHPAD> end.";
        assert_eq!(strip_reasoning_blocks(s), "Keep this  and  end.");
    }
}
