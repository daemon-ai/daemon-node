// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `browser` tool: a tagged-`op` surface over the supervised [`BrowserSupervisor`]
//! (metta-tool style). Navigation is egress-checked and optionally approval-gated; extracted page
//! content is returned as untrusted external data.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use daemon_common::ReqId;
use daemon_core::{check_url, Tool, ToolCall, ToolOutcome, TurnCx};
use daemon_protocol::{HostRequest, HostRequestKind, HostResponseBody, ToolDetail};
use serde::Deserialize;

use crate::supervisor::{BrowserSupervisor, ExtractFormat};

/// The JSON-Schema advertised for the `browser` tool (covers every op via the `op` discriminator).
const BROWSER_SCHEMA: &str = r#"{
  "type": "object",
  "required": ["op"],
  "properties": {
    "op": {
      "type": "string",
      "enum": ["navigate","extract","click","type","press_key","wait_for","screenshot","eval","reload","back","close"],
      "description": "The browser operation to perform."
    },
    "url": {"type": "string", "description": "Target URL (navigate)."},
    "selector": {"type": "string", "description": "CSS selector (click/type/press_key/wait_for)."},
    "text": {"type": "string", "description": "Text to type (type)."},
    "key": {"type": "string", "description": "Key to press, e.g. Enter (press_key)."},
    "format": {"type": "string", "enum": ["markdown","text","html"], "description": "Extraction format (extract, default markdown)."},
    "full_page": {"type": "boolean", "description": "Capture the full scrollable page (screenshot)."},
    "timeout_ms": {"type": "integer", "description": "Wait timeout in milliseconds (wait_for, default 10000)."},
    "js": {"type": "string", "description": "JavaScript expression to evaluate (eval)."}
  }
}"#;

/// Whether navigation requires an interactive host approval first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavApproval {
    /// Navigate freely (subject only to the egress safety check).
    Auto,
    /// Prompt the host (`HostRequest::Approval`) before each navigation.
    Ask,
}

/// The `browser` tool over a shared [`BrowserSupervisor`].
pub struct BrowserTool {
    browser: Arc<BrowserSupervisor>,
    nav_approval: NavApproval,
}

impl BrowserTool {
    /// A browser tool over `browser` that navigates freely (egress-checked only).
    pub fn new(browser: Arc<BrowserSupervisor>) -> Self {
        Self {
            browser,
            nav_approval: NavApproval::Auto,
        }
    }

    /// Require interactive host approval before each navigation.
    pub fn with_navigation_approval(mut self) -> Self {
        self.nav_approval = NavApproval::Ask;
        self
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Op {
    Navigate {
        url: String,
    },
    Extract {
        #[serde(default)]
        format: Option<String>,
    },
    Click {
        selector: String,
    },
    Type {
        selector: String,
        text: String,
    },
    PressKey {
        selector: String,
        key: String,
    },
    WaitFor {
        selector: String,
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    Screenshot {
        #[serde(default)]
        full_page: bool,
    },
    Eval {
        js: String,
    },
    Reload,
    Back,
    Close,
}

#[async_trait]
impl Tool for BrowserTool {
    fn name(&self) -> &str {
        "browser"
    }

    fn schema(&self) -> &str {
        BROWSER_SCHEMA
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let op: Op = match serde_json::from_str(&call.args) {
            Ok(op) => op,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("browser: invalid arguments: {e}"),
                )
            }
        };
        let id = call.call_id.clone();
        match op {
            Op::Navigate { url } => self.navigate(&id, &url, cx).await,
            Op::Extract { format } => {
                let fmt = match format.as_deref() {
                    Some("text") => ExtractFormat::Text,
                    Some("html") => ExtractFormat::Html,
                    _ => ExtractFormat::Markdown,
                };
                match self.browser.extract(fmt).await {
                    Ok((title, content)) => {
                        let body = serde_json::json!({ "title": title, "content": content });
                        let mut out = String::new();
                        if let Some(t) = &title {
                            out.push_str(&format!("# {t}\n\n"));
                        }
                        out.push_str(&content);
                        ToolOutcome::untrusted_text(id, true, out).with_detail(ToolDetail {
                            kind: "browser_extract".into(),
                            body: serde_json::to_vec(&body).unwrap_or_default(),
                        })
                    }
                    Err(e) => ToolOutcome::text(id, false, format!("browser extract: {e}")),
                }
            }
            Op::Click { selector } => self.simple(id, self.browser.click(&selector).await, "click"),
            Op::Type { selector, text } => {
                self.simple(id, self.browser.type_text(&selector, &text).await, "type")
            }
            Op::PressKey { selector, key } => self.simple(
                id,
                self.browser.press_key(&selector, &key).await,
                "press_key",
            ),
            Op::WaitFor {
                selector,
                timeout_ms,
            } => {
                let timeout = Duration::from_millis(timeout_ms.unwrap_or(10_000));
                self.simple(
                    id,
                    self.browser.wait_for(&selector, timeout).await,
                    "wait_for",
                )
            }
            Op::Screenshot { full_page } => match self.browser.screenshot(full_page).await {
                Ok(path) => {
                    let p = path.display().to_string();
                    ToolOutcome::text(id, true, format!("screenshot saved: {p}")).with_detail(
                        ToolDetail {
                            kind: "browser_screenshot".into(),
                            body: serde_json::to_vec(&serde_json::json!({ "path": p }))
                                .unwrap_or_default(),
                        },
                    )
                }
                Err(e) => ToolOutcome::text(id, false, format!("browser screenshot: {e}")),
            },
            Op::Eval { js } => match self.browser.eval(&js).await {
                // Page-evaluated output is untrusted (it reflects page-controlled data).
                Ok(value) => ToolOutcome::untrusted_text(id, true, value),
                Err(e) => ToolOutcome::text(id, false, format!("browser eval: {e}")),
            },
            Op::Reload => self.simple(id, self.browser.reload().await, "reload"),
            Op::Back => self.simple(id, self.browser.back().await, "back"),
            Op::Close => self.simple(id, self.browser.close().await, "close"),
        }
    }
}

impl BrowserTool {
    async fn navigate(&self, id: &str, url: &str, cx: &TurnCx<'_>) -> ToolOutcome {
        // Egress safety: http(s)-only, no loopback/private/link-local hosts.
        if let Err(reject) = check_url(url) {
            return ToolOutcome::text(id.to_string(), false, format!("browser navigate: {reject}"));
        }
        if self.nav_approval == NavApproval::Ask {
            let resp = cx
                .host
                .request(HostRequest {
                    request_id: ReqId(0),
                    kind: HostRequestKind::Approval {
                        prompt: format!("approve browser navigation to {url}"),
                    },
                })
                .await;
            let approved = matches!(resp.body, HostResponseBody::Approved(true));
            if !approved {
                return ToolOutcome::text(
                    id.to_string(),
                    false,
                    format!("browser navigate: denied by approval policy ({url})"),
                );
            }
        }
        match self.browser.navigate(url).await {
            Ok(current) => {
                ToolOutcome::text(id.to_string(), true, format!("navigated to {current}"))
            }
            Err(e) => ToolOutcome::text(id.to_string(), false, format!("browser navigate: {e}")),
        }
    }

    /// Map a unit-returning op result into a terse outcome.
    fn simple(
        &self,
        id: String,
        result: Result<(), crate::error::BrowserError>,
        op: &str,
    ) -> ToolOutcome {
        match result {
            Ok(()) => ToolOutcome::text(id, true, format!("browser {op}: ok")),
            Err(e) => ToolOutcome::text(id, false, format!("browser {op}: {e}")),
        }
    }
}
