//! The `web_extract` tool: fetch + clean a page through an ordered chain of [`WebFetchBackend`]s
//! (a keyed hosted scraper first, the local readability fallback last) and return the content as
//! untrusted external data. The target URL is checked against the egress policy first.

use std::sync::Arc;

use async_trait::async_trait;
use daemon_core::{check_url, Tool, ToolCall, ToolConcurrency, ToolOutcome, TurnCx};
use daemon_protocol::ToolDetail;
use serde::Deserialize;

use crate::backend::{FetchFormat, FetchOpts, FetchedDoc, WebError, WebFetchBackend};

/// The JSON-Schema advertised for `web_extract`.
const WEB_EXTRACT_SCHEMA: &str = r#"{
  "type": "object",
  "required": ["url"],
  "properties": {
    "url": {"type": "string", "description": "The http(s) URL to fetch and extract."},
    "format": {"type": "string", "enum": ["markdown", "text"], "description": "Output format (default markdown)."}
  }
}"#;

/// The `web_extract` tool. Holds an ordered backend chain; unavailable backends (e.g. a keyed
/// scraper with no key) are skipped, and a backend reporting [`WebError::MissingKey`] falls through
/// to the next.
pub struct WebExtractTool {
    backends: Vec<Arc<dyn WebFetchBackend>>,
}

impl WebExtractTool {
    /// An extract tool over an ordered backend chain (tried front to back).
    pub fn new(backends: Vec<Arc<dyn WebFetchBackend>>) -> Self {
        Self { backends }
    }
}

#[derive(Debug, Deserialize)]
struct Args {
    url: String,
    #[serde(default)]
    format: Option<String>,
}

#[async_trait]
impl Tool for WebExtractTool {
    fn name(&self) -> &str {
        "web_extract"
    }

    fn schema(&self) -> &str {
        WEB_EXTRACT_SCHEMA
    }

    fn concurrency(&self) -> ToolConcurrency {
        // Read-only fetch + clean with no shared-state mutation: safe to batch concurrently.
        ToolConcurrency::Parallel
    }

    async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> ToolOutcome {
        let args: Args = match serde_json::from_str(&call.args) {
            Ok(a) => a,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("web_extract: invalid arguments: {e}"),
                )
            }
        };
        // Egress safety: http(s)-only, no loopback/private/link-local hosts (SSRF guard).
        if let Err(reject) = check_url(&args.url) {
            return ToolOutcome::text(
                call.call_id.clone(),
                false,
                format!("web_extract: {reject}"),
            );
        }
        let format = match args.format.as_deref() {
            Some("text") => FetchFormat::Text,
            _ => FetchFormat::Markdown,
        };
        let opts = FetchOpts { format };

        let mut last_err: Option<String> = None;
        let mut tried_any = false;
        for backend in &self.backends {
            if !backend.available() {
                continue;
            }
            tried_any = true;
            match backend.fetch(&args.url, &opts).await {
                Ok(doc) => {
                    let content = render(&doc);
                    let detail = ToolDetail {
                        kind: "web_extract".to_string(),
                        body: serde_json::to_vec(&doc).unwrap_or_default(),
                    };
                    return ToolOutcome::untrusted_text(call.call_id.clone(), true, content)
                        .with_detail(detail);
                }
                // A missing key just means "skip me"; continue to the fallback.
                Err(WebError::MissingKey(_)) => {
                    tried_any = false;
                    continue;
                }
                Err(e) => {
                    last_err = Some(format!("{} ({}): {e}", "web_extract", backend.name()));
                    continue;
                }
            }
        }
        let msg = match (tried_any, last_err) {
            (_, Some(err)) => err,
            (false, None) => {
                "web_extract: no fetch backend configured (set a scraper API key or enable the local fallback)"
                    .to_string()
            }
            (true, None) => "web_extract: no content extracted".to_string(),
        };
        ToolOutcome::text(call.call_id.clone(), false, msg)
    }
}

/// Render a fetched document as a text block for the model.
fn render(doc: &FetchedDoc) -> String {
    let mut out = String::new();
    out.push_str(&format!("web_extract: {} via {}\n", doc.url, doc.provider));
    if let Some(title) = &doc.title {
        out.push_str(&format!("\n# {title}\n"));
    }
    out.push('\n');
    out.push_str(&doc.content);
    out
}
