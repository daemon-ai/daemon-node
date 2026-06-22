//! Browser-tool coverage (only meaningful with the `cdp` feature).
//!
//! The default test runs without a real Chromium: it builds the tool, exercises the schema/egress
//! reject, and confirms that an op needing the browser fails gracefully (crash-loop breaker) rather
//! than hanging when no browser can be launched. The full navigate/extract op exercise is `#[ignore]`
//! since it needs a system Chromium; run it with `--ignored` on a machine that has one.

#![cfg(feature = "cdp")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use daemon_common::{Budget, SessionId};
use daemon_core::events::EventSink;
use daemon_core::exec::LocalEnvironment;
use daemon_core::{Tool, ToolCall, ToolOutcome, TurnCx};
use daemon_protocol::{HostRequest, HostRequestHandler, HostResponse, HostResponseBody};
use daemon_tool_browser::{BrowserSettings, BrowserSupervisor, BrowserTool};

struct NoopHost;
#[async_trait]
impl HostRequestHandler for NoopHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
        HostResponse {
            request_id: req.request_id,
            body: HostResponseBody::Approved(true),
        }
    }
}

async fn run_tool(tool: &dyn Tool, args: &str) -> ToolOutcome {
    let events = EventSink::discarding();
    let exec = LocalEnvironment::sandbox("browser-tool-test");
    let host = NoopHost;
    let cx = TurnCx {
        cancel: tokio_util::sync::CancellationToken::new(),
        events: &events,
        host: &host,
        session_id: SessionId::new("s"),
        profile: None,
        budget: Budget::unlimited(),
        exec: &exec,
        tool_result_budget: 0,
        approval_policy: daemon_core::ApprovalPolicy::AutoAllow,
        pre_approved: false,
        checkpoints: None,
    };
    let call = ToolCall {
        call_id: "c1".into(),
        name: tool.name().to_string(),
        args: args.into(),
    };
    tool.run(&call, &cx).await
}

/// A supervisor pointed at a non-existent browser binary so a launch fails fast instead of finding a
/// real system Chromium during the no-browser smoke test.
fn unlaunchable_tool() -> BrowserTool {
    let settings = BrowserSettings {
        chrome_path: Some(PathBuf::from("/nonexistent/definitely-not-chrome")),
        headless: true,
        screenshot_dir: std::env::temp_dir().join("daemon_browser_test_shots"),
        launch_timeout: Duration::from_secs(2),
        auto_dismiss_dialogs: true,
    };
    BrowserTool::new(Arc::new(BrowserSupervisor::new(settings)))
}

#[tokio::test]
async fn schema_and_name_are_stable() {
    let tool = unlaunchable_tool();
    assert_eq!(tool.name(), "browser");
    assert!(tool.schema().contains("navigate"));
    assert!(tool.schema().contains("screenshot"));
}

#[tokio::test]
async fn invalid_args_reported() {
    let tool = unlaunchable_tool();
    let outcome = run_tool(&tool, "not json").await;
    assert!(!outcome.result.ok);
    assert!(outcome.result.content.contains("invalid arguments"));
}

#[tokio::test]
async fn navigate_rejects_unsafe_url() {
    let tool = unlaunchable_tool();
    let outcome = run_tool(&tool, r#"{"op":"navigate","url":"http://localhost:9000/x"}"#).await;
    assert!(!outcome.result.ok, "loopback host must be rejected pre-launch");
    assert!(!outcome.result.content.contains("navigated"));
}

#[tokio::test]
async fn navigate_without_browser_fails_gracefully() {
    // No real browser at the configured path: navigation surfaces a launch error, it does not hang.
    let tool = unlaunchable_tool();
    let outcome = run_tool(&tool, r#"{"op":"navigate","url":"https://example.com"}"#).await;
    assert!(!outcome.result.ok);
    assert!(outcome.result.content.contains("browser navigate"));
}

#[tokio::test]
#[ignore = "requires a system Chromium/Chrome; run with --ignored"]
async fn full_navigate_and_extract() {
    let settings = BrowserSettings {
        screenshot_dir: std::env::temp_dir().join("daemon_browser_it_shots"),
        ..BrowserSettings::default()
    };
    let tool = BrowserTool::new(Arc::new(BrowserSupervisor::new(settings)));
    let nav = run_tool(&tool, r#"{"op":"navigate","url":"https://example.com"}"#).await;
    assert!(nav.result.ok, "navigate: {}", nav.result.content);
    let extract = run_tool(&tool, r#"{"op":"extract","format":"text"}"#).await;
    assert!(extract.result.ok, "extract: {}", extract.result.content);
    assert!(extract.untrusted, "page content is untrusted");
    assert!(extract.result.content.to_lowercase().contains("example"));
}
