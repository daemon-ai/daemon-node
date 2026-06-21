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

use daemon_core::{Provider, Request, RequestMsg};
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration;

/// Default per-call extraction timeout (mirrors the LCM aux-LLM budget).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);

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
    pub async fn extract(&self, text: &str) -> Option<Extracted> {
        let provider = self.provider.as_ref()?;
        let request = Request {
            system: String::new(),
            messages: vec![RequestMsg {
                role: "user".into(),
                content: build_prompt(text),
                ..Default::default()
            }],
            tools: Vec::new(),
            auth: None,
        };
        let timeout = self.timeout.unwrap_or(DEFAULT_TIMEOUT);
        let out = match tokio::time::timeout(timeout, provider.chat(request)).await {
            Ok(Ok(out)) => out,
            _ => return None,
        };
        let parsed = parse_extraction(&out.text)?;
        if parsed.is_empty() {
            None
        } else {
            Some(parsed)
        }
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
}
