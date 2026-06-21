//! Escalation & auxiliary-provider summarization (`daemon-context-lcm-port-spec.md` §7).
//!
//! A summary must shrink its source or it escalates: **L1** a detailed LLM summary, **L2** aggressive
//! LLM bullets at half budget, **L3** a deterministic head/tail truncation that always converges. A
//! per-route [`SummaryCircuitBreaker`] (2 failures / 300s) skips a failing aux provider and falls
//! straight to L3. The aux model is a `daemon-core` [`Provider`]; per-call model routing collapses to
//! "the provider is the model" (§7.4), so escalation just builds a one-message [`Request`] and calls
//! [`Provider::chat`] under a timeout.

use crate::tokens::Tokenizer;
use daemon_core::{Provider, Request, RequestMsg};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The default failure threshold when no config is supplied (`LCM:config.py:302`).
const DEFAULT_BREAKER_FAILURE_THRESHOLD: u32 = 2;
/// The default cooldown when no config is supplied (`LCM:config.py:304`).
const DEFAULT_BREAKER_COOLDOWN_SECONDS: u64 = 300;

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
    /// An optional focus topic (appended to the prompt; `""` for none).
    pub focus_topic: &'a str,
    /// Optional custom instructions (appended; `""` for none).
    pub custom_instructions: &'a str,
}

/// Summarize with 3-level escalation, returning the summary text and the level it converged at.
///
/// `aux_chain` is the ordered fallback chain of aux providers (`summary_model` then
/// `summary_fallback_models`); `breakers` is the parallel per-route breaker slice. Each LLM call
/// tries the chain in order (skipping open routes); if **all** routes are open/failing the level
/// falls through. `l2_budget_ratio` (0.50) and `l3_truncate_tokens` (512) are config; `timeout`
/// bounds each call. L3 is local and always converges.
pub async fn summarize_with_escalation(
    aux_chain: &[Arc<dyn Provider>],
    tok: &Tokenizer,
    breakers: &mut [SummaryCircuitBreaker],
    l2_budget_ratio: f64,
    l3_truncate_tokens: usize,
    timeout: Duration,
    req: SummaryRequest<'_>,
) -> (String, Level) {
    // L1 — detailed summary over the fallback chain.
    if let Some(text) = call_summary_llm_chain(aux_chain, breakers, build_l1_prompt(&req), timeout).await
    {
        let text = strip_reasoning_blocks(&text);
        if !text.is_empty() && tok.count_text(&text) < req.source_tokens {
            return (text, Level::L1);
        }
        // L2 — aggressive bullets at half budget (a route is healthy, just too verbose).
        let l2_budget = ((req.token_budget as f64) * l2_budget_ratio) as usize;
        let l2_prompt = build_l2_prompt(&req, l2_budget.max(1));
        if let Some(text2) = call_summary_llm_chain(aux_chain, breakers, l2_prompt, timeout).await {
            let text2 = strip_reasoning_blocks(&text2);
            if !text2.is_empty() && tok.count_text(&text2) < req.source_tokens {
                return (text2, Level::L2);
            }
        }
    }
    // L3 — deterministic truncation; always converges.
    (
        deterministic_truncate(req.text, l3_truncate_tokens),
        Level::L3,
    )
}

/// Try the aux fallback chain for one prompt (`_invoke_summary_llm_chain`, §7.3): skip open routes,
/// call each in order, record success/failure per route. Returns the first successful text, or
/// `None` when every route is open or failed.
async fn call_summary_llm_chain(
    chain: &[Arc<dyn Provider>],
    breakers: &mut [SummaryCircuitBreaker],
    prompt: String,
    timeout: Duration,
) -> Option<String> {
    for (i, provider) in chain.iter().enumerate() {
        if breakers.get(i).is_some_and(|b| b.is_open()) {
            continue;
        }
        match call_summary_llm(provider.as_ref(), prompt.clone(), timeout).await {
            Ok(text) => {
                if let Some(b) = breakers.get_mut(i) {
                    b.record_success();
                }
                return Some(text);
            }
            Err(()) => {
                if let Some(b) = breakers.get_mut(i) {
                    b.record_failure();
                }
            }
        }
    }
    None
}

/// Build the request to the aux provider and await its text under `timeout`. `Err(())` on any
/// transport failure or timeout (the caller maps that to a breaker failure + L3).
async fn call_summary_llm(aux: &dyn Provider, prompt: String, timeout: Duration) -> Result<String, ()> {
    let request = Request {
        system: String::new(),
        messages: vec![RequestMsg {
            role: "user".into(),
            content: prompt,
            ..Default::default()
        }],
        tools: Vec::new(),
        auth: None,
    };
    match tokio::time::timeout(timeout, aux.chat(request)).await {
        Ok(Ok(out)) => Ok(out.text),
        Ok(Err(_failure)) => Err(()),
        Err(_elapsed) => Err(()),
    }
}

/// Depth-aware L1 guidance (`LCM:escalation.py:221-229`).
fn depth_guidance(depth: i64) -> &'static str {
    match depth {
        0 => "Preserve decisions, rationale, constraints, tasks, paths, commands, values.",
        1 => "Arc-level outcomes; drop per-turn detail.",
        _ => "Durable narrative; drop process detail.",
    }
}

/// The verbatim L1 prompt (`_build_l1_prompt`, `LCM:escalation.py:237-246`).
fn build_l1_prompt(req: &SummaryRequest<'_>) -> String {
    let focus = if req.focus_topic.is_empty() {
        String::new()
    } else {
        format!("\nFocus on: {}", req.focus_topic)
    };
    let custom = if req.custom_instructions.is_empty() {
        String::new()
    } else {
        format!("\n{}", req.custom_instructions)
    };
    format!(
        "Summarize this conversation segment for future turns.\n\
         {guidance}\n\
         Remove repetition and conversational filler.\n\
         End with: \"Expand for details about: <what was compressed>\"\
         {focus}{custom}\n\n\
         Target ~{budget} tokens.\n\n\
         CONTENT:\n{content}",
        guidance = depth_guidance(req.depth),
        focus = focus,
        custom = custom,
        budget = req.token_budget,
        content = req.text,
    )
}

/// The verbatim L2 prompt (`_build_l2_prompt`, `LCM:escalation.py:258-264`); `budget` is the
/// already-halved L2 budget.
fn build_l2_prompt(req: &SummaryRequest<'_>, budget: usize) -> String {
    let focus = if req.focus_topic.is_empty() {
        String::new()
    } else {
        format!("\nFocus on: {}", req.focus_topic)
    };
    let custom = if req.custom_instructions.is_empty() {
        String::new()
    } else {
        format!("\n{}", req.custom_instructions)
    };
    format!(
        "Compress this into bullet points. Maximum {budget} tokens.\n\
         Keep only: decisions made, files changed, errors hit, current state.\n\
         Drop all reasoning, alternatives considered, and process detail.\
         {focus}{custom}\n\n\
         CONTENT:\n{content}",
        budget = budget,
        focus = focus,
        custom = custom,
        content = req.text,
    )
}

/// Deterministic head/tail truncation (`_deterministic_truncate`, `LCM:escalation.py:267-285`):
/// `char_budget = max_tokens * 4`; keep 40% head + 40% tail with a marker between. Operates on char
/// boundaries so multi-byte text is never split.
fn deterministic_truncate(text: &str, max_tokens: usize) -> String {
    const MARKER: &str = "\n\n[...deterministic truncation — details available via lcm_expand...]\n\n";
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
    const TAGS: [&str; 5] = ["think", "thinking", "reasoning", "thought", "reasoning_scratchpad"];
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

    /// An aux provider that returns a fixed text (or a failure) for every call.
    struct FixedAux {
        reply: Option<String>,
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
            match &self.reply {
                Some(text) => Ok(ModelOutput {
                    text: text.clone(),
                    ..Default::default()
                }),
                None => Err(Failure::Provider("aux down".into())),
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
        vec![Arc::new(FixedAux {
            reply: reply.map(|s| s.to_string()),
        })]
    }

    #[tokio::test]
    async fn l1_accepts_a_shorter_summary() {
        let aux = chain(Some("short"));
        let tok = Tokenizer::heuristic();
        let mut breakers = vec![SummaryCircuitBreaker::new()];
        let (text, level) = summarize_with_escalation(
            &aux,
            &tok,
            &mut breakers,
            0.5,
            512,
            Duration::from_secs(5),
            req("a very long source that is bigger than the summary", 1000),
        )
        .await;
        assert_eq!(level, Level::L1);
        assert_eq!(text, "short");
    }

    #[tokio::test]
    async fn l3_converges_when_summary_never_shrinks() {
        // The aux echoes a long reply that never beats source_tokens=1, forcing L1+L2 to fail the
        // shrink test and fall to deterministic truncation.
        let long = "x".repeat(10_000);
        let aux = chain(Some(&long));
        let tok = Tokenizer::heuristic();
        let mut breakers = vec![SummaryCircuitBreaker::new()];
        let (text, level) = summarize_with_escalation(
            &aux,
            &tok,
            &mut breakers,
            0.5,
            64,
            Duration::from_secs(5),
            req(&long, 1),
        )
        .await;
        assert_eq!(level, Level::L3);
        assert!(text.len() < long.len(), "L3 truncates");
        assert!(text.contains("deterministic truncation"));
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
        let (text, level) = summarize_with_escalation(
            &aux,
            &tok,
            &mut breakers,
            0.5,
            16,
            Duration::from_secs(5),
            req(&src, 1),
        )
        .await;
        assert_eq!(level, Level::L3);
        assert!(!text.is_empty());
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
            Arc::new(FixedAux { reply: None }),
            Arc::new(FixedAux { reply: Some("fallback summary".into()) }),
        ];
        let tok = Tokenizer::heuristic();
        let (text, level) = summarize_with_escalation(
            &aux,
            &tok,
            &mut breakers,
            0.5,
            512,
            Duration::from_secs(5),
            req("a long source that exceeds the fallback summary length", 1000),
        )
        .await;
        assert_eq!(level, Level::L1);
        assert_eq!(text, "fallback summary");
    }

    #[test]
    fn strips_reasoning_blocks() {
        let s = "Keep this <think>drop me</think> and <REASONING_SCRATCHPAD>this too</REASONING_SCRATCHPAD> end.";
        assert_eq!(strip_reasoning_blocks(s), "Keep this  and  end.");
    }
}
