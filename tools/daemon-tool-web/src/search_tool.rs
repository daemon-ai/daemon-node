//! The `web_search` tool: resolve a query through a [`WebSearchBackend`] and return ranked hits as
//! untrusted external data.

use std::sync::Arc;

use async_trait::async_trait;
use daemon_core::{Tool, ToolCall, ToolConcurrency, ToolOutcome, TurnCx};
use daemon_protocol::ToolDetail;
use serde::Deserialize;

use crate::backend::{SearchOpts, SearchResults, SearchTopic, WebError, WebSearchBackend};

/// The JSON-Schema advertised for `web_search`.
const WEB_SEARCH_SCHEMA: &str = r#"{
  "type": "object",
  "required": ["query"],
  "properties": {
    "query": {"type": "string", "description": "The search query."},
    "max_results": {"type": "integer", "description": "Maximum number of results (default 5)."},
    "topic": {"type": "string", "enum": ["general", "news"], "description": "Search topic hint (default general)."}
  }
}"#;

/// The default number of results when a call omits `max_results`.
const DEFAULT_MAX_RESULTS: u32 = 5;

/// The `web_search` tool.
pub struct WebSearchTool {
    backend: Arc<dyn WebSearchBackend>,
    default_max_results: u32,
}

impl WebSearchTool {
    /// A search tool over `backend`.
    pub fn new(backend: Arc<dyn WebSearchBackend>) -> Self {
        Self {
            backend,
            default_max_results: DEFAULT_MAX_RESULTS,
        }
    }

    /// Override the default result count.
    pub fn with_default_max_results(mut self, n: u32) -> Self {
        self.default_max_results = n;
        self
    }
}

#[derive(Debug, Deserialize)]
struct Args {
    query: String,
    #[serde(default)]
    max_results: Option<u32>,
    #[serde(default)]
    topic: Option<String>,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn schema(&self) -> &str {
        WEB_SEARCH_SCHEMA
    }

    fn concurrency(&self) -> ToolConcurrency {
        // Read-only network fetch with no shared-state mutation: safe to batch concurrently.
        ToolConcurrency::Parallel
    }

    async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> ToolOutcome {
        let args: Args = match serde_json::from_str(&call.args) {
            Ok(a) => a,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("web_search: invalid arguments: {e}"),
                )
            }
        };
        if args.query.trim().is_empty() {
            return ToolOutcome::text(call.call_id.clone(), false, "web_search: empty query");
        }
        let topic = match args.topic.as_deref() {
            Some("news") => SearchTopic::News,
            _ => SearchTopic::General,
        };
        let opts = SearchOpts {
            max_results: args.max_results.unwrap_or(self.default_max_results).clamp(1, 20),
            topic,
        };
        match self.backend.search(&args.query, &opts).await {
            Ok(results) => {
                let content = render(&results);
                let detail = ToolDetail {
                    kind: "web_search".to_string(),
                    body: serde_json::to_vec(&results).unwrap_or_default(),
                };
                ToolOutcome::untrusted_text(call.call_id.clone(), true, content).with_detail(detail)
            }
            Err(WebError::MissingKey(key)) => ToolOutcome::text(
                call.call_id.clone(),
                false,
                format!(
                    "web_search: backend '{}' is not configured. Set an API key for the '{key}' \
                     credential profile (CredentialSet) and retry.",
                    self.backend.name()
                ),
            ),
            Err(e) => ToolOutcome::text(
                call.call_id.clone(),
                false,
                format!("web_search ({}): {e}", self.backend.name()),
            ),
        }
    }
}

/// Render search results as a concise text block for the model.
fn render(results: &SearchResults) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "web_search \"{}\" via {} — {} result(s)\n",
        results.query,
        results.provider,
        results.hits.len()
    ));
    if let Some(answer) = &results.answer {
        out.push_str(&format!("\nanswer: {answer}\n"));
    }
    for (i, hit) in results.hits.iter().enumerate() {
        out.push_str(&format!("\n{}. {}\n   {}\n", i + 1, hit.title, hit.url));
        if !hit.snippet.trim().is_empty() {
            out.push_str(&format!("   {}\n", hit.snippet.trim()));
        }
    }
    out
}
