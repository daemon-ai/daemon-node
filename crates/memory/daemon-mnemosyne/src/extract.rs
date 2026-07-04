// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! LLM extraction — port of `extraction.py`, over the `daemon-core` [`Provider`] seam.
//!
//! Mnemosyne does not own an LLM runtime. The host injects a [`Provider`] (the same chat backend the
//! engine uses, or a dedicated aux profile) and this thin wrapper runs a single structured-extraction
//! completion, mirroring the LCM `call_summary_llm` one-shot. With no provider the engine falls back
//! to the deterministic regex extraction in [`crate::knowledge`] (`extraction.py` host->local->remote
//! ladder collapses to: injected provider, else regex baseline).
//!
//! Extraction is async (it calls a model). The synchronous BEAM [`Engine`](crate::engine::Engine)
//! never calls a model inline: the async [`MnemosyneProvider`](crate::provider::MnemosyneProvider)
//! hooks extract here and pass the parsed result into the engine's sync ingest entrypoint.
//!
//! Two extraction schemas port from Python:
//! - **Message-level** ([`Extractor::extract`] → [`Extracted`]): the MEMORIA entity/triple/fact
//!   object used by the per-turn ingest path (`extraction.py`).
//! - **Conversation-level** ([`Extractor::extract_conversation_facts`] → [`ExtractedFact`]): the
//!   C13 fact-array schema (`extraction/prompts.py` + `extraction/client.py`) — SPO triples with
//!   per-message provenance (`source` index), timestamps, and confidence, used for batch
//!   conversation distillation.
//!
//! Every attempt is counted in the process-global [`diagnostics`] registry (C13.b), so silent
//! extraction failures are visible via [`diagnostics::get_extraction_stats`].

use daemon_core::{Provider, Request, RequestMsg};
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration;

pub mod diagnostics;

/// Default per-call extraction timeout (mirrors the LCM aux-LLM budget).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);

/// The conversation-level fact-extraction system prompt (`extraction/prompts.py`
/// `EXTRACTION_SYSTEM_PROMPT`, verbatim).
pub const EXTRACTION_SYSTEM_PROMPT: &str = r#"You extract structured facts from conversation messages. For each message or group of related messages, identify:

1. ENTITIES: People, projects, tools, versions, dates, numbers mentioned
2. RELATIONSHIPS: How entities relate to each other (uses, created, set, changed, prefers)
3. TEMPORAL ANCHORS: When something happened, deadlines, durations
4. CONTRADICTIONS: When a fact was later changed or updated

Return ONLY a JSON array of fact objects. Each fact must have:
- subject: the entity the fact is about (string)
- predicate: the relationship or action (string)
- object: the value or related entity (string)
- timestamp: ISO timestamp when this was stated (string, from message context)
- source: which message index this came from (integer, 0-based)
- confidence: 0.0-1.0 how certain you are (float)

RULES:
- One fact per relationship. "I use React 18.2 and Node.js 18" = 2 facts.
- Use lowercase for predicates: "uses", "set", "changed", "created", "prefers"
- Include versions and numbers as objects when available
- If a message states something changed, extract BOTH old and new facts
- If unclear, use confidence < 0.8

Format: [{"subject": "...", "predicate": "...", "object": "...", "timestamp": "...", "source": 0, "confidence": 0.95}]
"#;

/// The user-prompt template (`extraction/prompts.py` `EXTRACTION_USER_TEMPLATE`, verbatim modulo
/// the `{conversation_text}` interpolation).
pub fn extraction_user_prompt(conversation_text: &str) -> String {
    format!(
        "Extract all structured facts from the following conversation messages. Return ONLY the \
         JSON array, no other text.\n\nCONVERSATION:\n{conversation_text}\n\nFACTS:"
    )
}

/// One fact object from the conversation-level extraction schema (`extraction/prompts.py` fact
/// shape; `client.py` `extract_facts` returns a list of these).
#[derive(Clone, Debug, Default, Deserialize, PartialEq, serde::Serialize)]
pub struct ExtractedFact {
    /// The entity the fact is about.
    #[serde(default)]
    pub subject: String,
    /// The relationship or action (lowercase by prompt contract).
    #[serde(default)]
    pub predicate: String,
    /// The value or related entity.
    #[serde(default)]
    pub object: String,
    /// ISO timestamp when this was stated (from message context).
    #[serde(default)]
    pub timestamp: String,
    /// 0-based index of the source message.
    #[serde(default)]
    pub source: i64,
    /// Extraction confidence `[0, 1]`.
    #[serde(default = "default_confidence")]
    pub confidence: f64,
}

/// Parse a completion into the fact array (`client.py` `extract_facts` L197-L208: first `[` to
/// last `]`, must decode to a JSON list). `None` when no parseable array exists.
pub fn parse_fact_array(raw: &str) -> Option<Vec<ExtractedFact>> {
    let start = raw.find('[')?;
    let end = raw.rfind(']')?;
    if end <= start {
        return None;
    }
    serde_json::from_str::<Vec<ExtractedFact>>(&raw[start..=end]).ok()
}

/// A subject-predicate-object triple extracted by the LLM.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ExtractedTriple {
    /// Subject.
    pub subject: String,
    /// Predicate.
    pub predicate: String,
    /// Object.
    pub object: String,
    /// Optional confidence `[0, 1]` (defaults to 0.7 when the model omits it).
    #[serde(default = "default_confidence")]
    pub confidence: f64,
}

fn default_confidence() -> f64 {
    0.7
}

/// The structured result of one extraction pass (`extraction.py` MEMORIA JSON, adapted to the bits
/// the knowledge layer consumes: entities, SPO triples, and free-text statements).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Extracted {
    /// Salient entity mentions.
    #[serde(default)]
    pub entities: Vec<String>,
    /// Knowledge-graph triples.
    #[serde(default, alias = "kg", alias = "triples")]
    pub triples: Vec<ExtractedTriple>,
    /// High-signal free-text statements (persisted as `fact` annotations).
    #[serde(default, alias = "statements")]
    pub facts: Vec<String>,
}

impl Extracted {
    /// Whether the pass produced anything worth ingesting.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty() && self.triples.is_empty() && self.facts.is_empty()
    }
}

/// The extraction prompt (adapted from `extraction.py` `EXTRACTION_PROMPT_TEMPLATE` L42-L71): a
/// single completion that returns the strict JSON object [`Extracted`] deserializes.
fn build_prompt(text: &str) -> String {
    format!(
        "You are a structured-memory extractor. From the message below, extract ONLY high-signal, \
         long-term relevant items. Detect the language but preserve original casing and language.\n\
         Return STRICT JSON only (no prose, no code fences) in exactly this shape:\n\
         {{\"entities\": [\"Name\"], \"triples\": [{{\"subject\": \"S\", \"predicate\": \"p\", \
         \"object\": \"O\", \"confidence\": 0.8}}], \"facts\": [\"persistent statement\"]}}\n\
         Rules: only persistent, non-transient content (ignore weather, one-off chat, system text); \
         use semantic understanding, not keywords; if nothing qualifies, return empty arrays.\n\n\
         Message: {text}\n\nJSON:"
    )
}

/// Strip optional Markdown code fences and isolate the first JSON object (`extraction.py`
/// `_parse_facts` fence handling L89-L96).
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
    // Isolate the outermost object so trailing prose can't break the parse.
    match (s.find('{'), s.rfind('}')) {
        (Some(a), Some(b)) if b >= a => &s[a..=b],
        _ => s,
    }
}

/// Parse a model completion into [`Extracted`] (`None` if it isn't valid JSON of the right shape).
pub fn parse_extraction(raw: &str) -> Option<Extracted> {
    let json = strip_json(raw);
    serde_json::from_str::<Extracted>(json).ok()
}

/// An extraction backend handle: an optional injected [`Provider`].
#[derive(Clone, Default)]
pub struct Extractor {
    provider: Option<Arc<dyn Provider>>,
    timeout: Option<Duration>,
}

impl Extractor {
    /// A no-LLM extractor (regex baseline only).
    pub fn new() -> Self {
        Self::default()
    }

    /// An extractor backed by an injected provider.
    pub fn with_provider(provider: Arc<dyn Provider>) -> Self {
        Self {
            provider: Some(provider),
            timeout: None,
        }
    }

    /// Whether an LLM is available (false in regex-only mode).
    pub fn available(&self) -> bool {
        self.provider.is_some()
    }

    /// Run a one-shot structured extraction over `text`. `None` in regex-only mode, on timeout, on
    /// backend error, or when the completion isn't parseable JSON.
    ///
    /// Every attempt feeds the process-global [`diagnostics`] counters under the `host` tier
    /// (`core/extraction.py` wires the same `record_attempt`/`record_success`/`record_no_output`/
    /// `record_failure`/`record_call` set around its tier ladder; the Rust node has exactly one
    /// tier — the injected provider).
    pub async fn extract(&self, text: &str) -> Option<Extracted> {
        let provider = self.provider.as_ref()?;
        let diag = diagnostics::get_diagnostics();
        diag.record_attempt("host");
        let request = Request {
            system: String::new(),
            messages: vec![RequestMsg {
                role: "user".into(),
                content: build_prompt(text),
                ..Default::default()
            }],
            tools: Vec::new(),
            auth: None,
            constraint: None,
            cache_system: false,
        };
        let timeout = self.timeout.unwrap_or(DEFAULT_TIMEOUT);
        let out = match tokio::time::timeout(timeout, provider.chat(request)).await {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => {
                diag.record_failure("host", Some(&e.to_string()), Some("backend_error"));
                diag.record_call(false, false);
                return None;
            }
            Err(_) => {
                diag.record_failure("host", None, Some("timeout"));
                diag.record_call(false, false);
                return None;
            }
        };
        let Some(parsed) = parse_extraction(&out.text) else {
            diag.record_failure("host", None, Some("json_parse_failed"));
            diag.record_call(false, false);
            return None;
        };
        if parsed.is_empty() {
            diag.record_no_output("host");
            diag.record_call(false, true);
            None
        } else {
            diag.record_success("host");
            diag.record_call(true, false);
            Some(parsed)
        }
    }

    /// Conversation-level fact extraction (`client.py` `extract_facts` L156-L214 +
    /// `prompts.py`): format the messages as `[i] [role]: content` lines, send the C13 fact-array
    /// prompt, and parse the completion into [`ExtractedFact`]s. `None` in regex-only mode or on
    /// timeout/error; `Some(vec![])` when the model answers with an empty array. Feeds the same
    /// diagnostics counters as [`Self::extract`].
    pub async fn extract_conversation_facts(
        &self,
        messages: &[(String, String)],
    ) -> Option<Vec<ExtractedFact>> {
        let provider = self.provider.as_ref()?;
        let diag = diagnostics::get_diagnostics();
        diag.record_attempt("host");
        let conversation = messages
            .iter()
            .enumerate()
            .map(|(i, (role, content))| format!("[{i}] [{role}]: {content}"))
            .collect::<Vec<_>>()
            .join("\n");
        let request = Request {
            system: EXTRACTION_SYSTEM_PROMPT.to_string(),
            messages: vec![RequestMsg {
                role: "user".into(),
                content: extraction_user_prompt(&conversation),
                ..Default::default()
            }],
            tools: Vec::new(),
            auth: None,
            constraint: None,
            cache_system: false,
        };
        let timeout = self.timeout.unwrap_or(DEFAULT_TIMEOUT);
        let out = match tokio::time::timeout(timeout, provider.chat(request)).await {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => {
                diag.record_failure("host", Some(&e.to_string()), Some("backend_error"));
                diag.record_call(false, false);
                return None;
            }
            Err(_) => {
                diag.record_failure("host", None, Some("timeout"));
                diag.record_call(false, false);
                return None;
            }
        };
        let Some(facts) = parse_fact_array(&out.text) else {
            diag.record_failure("host", None, Some("json_parse_failed"));
            diag.record_call(false, false);
            return None;
        };
        if facts.is_empty() {
            diag.record_no_output("host");
            diag.record_call(false, true);
        } else {
            diag.record_success("host");
            diag.record_call(true, false);
        }
        Some(facts)
    }

    /// Run a one-shot summarization completion over `prompt` (used by sleep/degradation). Returns the
    /// raw model text, or `None` in regex-only mode / on timeout / on error.
    pub async fn summarize(&self, prompt: String) -> Option<String> {
        let provider = self.provider.as_ref()?;
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
        let timeout = self.timeout.unwrap_or(DEFAULT_TIMEOUT);
        match tokio::time::timeout(timeout, provider.chat(request)).await {
            Ok(Ok(out)) if !out.text.trim().is_empty() => Some(out.text),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_core::MockProvider;

    #[test]
    fn parses_fenced_json() {
        let raw = "```json\n{\"entities\":[\"Maya\"],\"triples\":[{\"subject\":\"Maya\",\
                   \"predicate\":\"works_at\",\"object\":\"Acme\"}],\"facts\":[\"my name is Maya\"]}\n```";
        let got = parse_extraction(raw).expect("parse");
        assert_eq!(got.entities, vec!["Maya".to_string()]);
        assert_eq!(got.triples.len(), 1);
        assert_eq!(got.triples[0].predicate, "works_at");
        assert!((got.triples[0].confidence - 0.7).abs() < 1e-9);
        assert_eq!(got.facts.len(), 1);
    }

    #[test]
    fn accepts_kg_alias_and_trailing_prose() {
        let raw = "Here you go: {\"kg\":[{\"subject\":\"a\",\"predicate\":\"b\",\"object\":\"c\",\
                   \"confidence\":0.9}]} -- done";
        let got = parse_extraction(raw).expect("parse");
        assert_eq!(got.triples.len(), 1);
        assert!((got.triples[0].confidence - 0.9).abs() < 1e-9);
    }

    #[tokio::test]
    async fn regex_only_extractor_returns_none() {
        let e = Extractor::new();
        assert!(!e.available());
        assert!(e.extract("Maya works at Acme").await.is_none());
    }

    #[tokio::test]
    async fn injected_provider_extracts() {
        let json = r#"{"entities":["Maya"],"triples":[],"facts":[]}"#;
        let e = Extractor::with_provider(Arc::new(MockProvider::completing(json)));
        let got = e.extract("Maya joined").await.expect("extraction");
        assert_eq!(got.entities, vec!["Maya".to_string()]);
    }

    #[test]
    fn fact_array_parses_with_surrounding_prose() {
        let raw = r#"Sure! FACTS: [{"subject":"user","predicate":"uses","object":"React 18.2",
            "timestamp":"2026-01-01T00:00:00Z","source":0,"confidence":0.95}] hope that helps"#;
        let facts = parse_fact_array(raw).expect("array");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].subject, "user");
        assert_eq!(facts[0].predicate, "uses");
        assert_eq!(facts[0].object, "React 18.2");
        assert_eq!(facts[0].source, 0);
        assert!((facts[0].confidence - 0.95).abs() < 1e-9);
    }

    #[test]
    fn fact_array_rejects_non_array_json() {
        assert!(parse_fact_array("{\"subject\": \"x\"}").is_none());
        assert!(parse_fact_array("no json here").is_none());
        assert_eq!(parse_fact_array("[]").expect("empty array"), vec![]);
    }

    #[tokio::test]
    async fn conversation_facts_round_trip() {
        let json = r#"[{"subject":"user","predicate":"prefers","object":"tabs",
            "timestamp":"2026-01-01T00:00:00Z","source":1,"confidence":0.9}]"#;
        let e = Extractor::with_provider(Arc::new(MockProvider::completing(json)));
        let msgs = vec![
            ("assistant".to_string(), "tabs or spaces?".to_string()),
            ("user".to_string(), "tabs, always".to_string()),
        ];
        let facts = e
            .extract_conversation_facts(&msgs)
            .await
            .expect("extraction");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].predicate, "prefers");
        assert_eq!(facts[0].source, 1);
    }
}
