// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `vision_analyze` coverage, all deterministic (no live network): the SSRF pre-flight and the
//! per-redirect-hop guard (wiremock), size caps, workspace containment, the aux happy path against
//! in-process fake providers (request shape, trust classification), provider-failure mapping, the
//! empty-content retry, timeout, and cancellation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use daemon_common::{Budget, SessionId};
use daemon_core::{
    Capabilities, EventSink, Failure, LocalEnvironment, MockProvider, ModelOutput, Provider,
    Request, Tool, ToolCall, ToolCallFormat, ToolConcurrency, TurnCx,
};
use daemon_protocol::{HostRequest, HostRequestHandler, HostResponse, HostResponseBody};
use daemon_tool_vision::{VisionAnalyzeTool, VisionToolConfig};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Minimal PNG-magic bytes: the sniff is prefix-based, so no decodable body is needed.
fn png_fixture() -> Vec<u8> {
    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    bytes.extend_from_slice(&[0u8; 32]);
    bytes
}

fn png_data_url() -> String {
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png_fixture())
    )
}

fn mock_tool(reply: &str) -> VisionAnalyzeTool {
    VisionAnalyzeTool::new(
        Arc::new(MockProvider::completing(reply)),
        VisionToolConfig::default(),
    )
}

struct NoopHost;
#[async_trait]
impl HostRequestHandler for NoopHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
        HostResponse {
            request_id: req.request_id,
            body: HostResponseBody::Approved {
                approved: true,
                allow_permanent: false,
                reason: None,
            },
        }
    }
}

async fn run_with(
    tool: &VisionAnalyzeTool,
    args: &str,
    exec: &LocalEnvironment,
    cancel: CancellationToken,
) -> daemon_core::ToolOutcome {
    let events = EventSink::discarding();
    let host = NoopHost;
    let cx = TurnCx {
        cancel,
        events: &events,
        host: &host,
        session_id: SessionId::new("s"),
        profile: None,
        budget: Budget::unlimited(),
        exec,
        tool_result_budget: 0,
        approval_policy: daemon_core::ApprovalPolicy::AutoAllow,
        pre_approved: false,
        checkpoints: None,
        tool_timeout: None,
        session_allow: &[],
    };
    let call = ToolCall {
        call_id: "c1".into(),
        name: "vision_analyze".into(),
        args: args.into(),
    };
    tool.run(&call, &cx).await
}

async fn run_tool(tool: &VisionAnalyzeTool, args: &str) -> daemon_core::ToolOutcome {
    let exec = LocalEnvironment::sandbox("vision-tool-test");
    run_with(tool, args, &exec, CancellationToken::new()).await
}

fn result_json(outcome: &daemon_core::ToolOutcome) -> serde_json::Value {
    serde_json::from_str(&outcome.result.content).expect("result content is JSON")
}

/// A provider that records the last request it served and replies with fixed text.
struct CapturingProvider {
    last: Mutex<Option<Request>>,
    reply: String,
}

impl CapturingProvider {
    fn new(reply: &str) -> Self {
        Self {
            last: Mutex::new(None),
            reply: reply.to_string(),
        }
    }
}

fn test_capabilities() -> Capabilities {
    Capabilities {
        supports_native_tools: false,
        supports_streaming: false,
        tool_call_format: ToolCallFormat::Native,
        max_context: None,
    }
}

#[async_trait]
impl Provider for CapturingProvider {
    fn capabilities(&self) -> Capabilities {
        test_capabilities()
    }
    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
        *self.last.lock().expect("capture lock") = Some(req);
        Ok(ModelOutput {
            text: self.reply.clone(),
            ..Default::default()
        })
    }
}

/// A provider that fails every call with a fixed provider error message.
struct FailingProvider(String);
#[async_trait]
impl Provider for FailingProvider {
    fn capabilities(&self) -> Capabilities {
        test_capabilities()
    }
    async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
        Err(Failure::Provider(self.0.clone()))
    }
}

/// A provider that answers empty content first, then real text (drives the single retry).
struct EmptyThenTextProvider {
    calls: AtomicU64,
}
#[async_trait]
impl Provider for EmptyThenTextProvider {
    fn capabilities(&self) -> Capabilities {
        test_capabilities()
    }
    async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ModelOutput {
            text: if n == 0 {
                String::new()
            } else {
                "retried answer".to_string()
            },
            ..Default::default()
        })
    }
}

/// A provider whose chat never completes (drives timeout / cancellation deterministically).
struct PendingProvider;
#[async_trait]
impl Provider for PendingProvider {
    fn capabilities(&self) -> Capabilities {
        test_capabilities()
    }
    async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
        std::future::pending().await
    }
}

// --- SSRF: the pre-flight on the initial URL ---------------------------------------------------

#[tokio::test]
async fn rejects_private_loopback_and_metadata_urls() {
    let tool = mock_tool("never reached");
    for url in [
        "http://localhost:9000/secret.png",
        "http://127.0.0.1/x.png",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/internal.png",
        "http://[::1]/x.png",
    ] {
        let args = format!(r#"{{"image_url":"{url}","question":"q"}}"#);
        let outcome = run_tool(&tool, &args).await;
        assert!(!outcome.result.ok, "expected {url} to be rejected");
        let v = result_json(&outcome);
        assert_eq!(v["success"], false);
        assert!(
            v["analysis"]
                .as_str()
                .unwrap()
                .contains("egress safety policy"),
            "analysis should name the egress policy for {url}"
        );
    }
}

#[tokio::test]
async fn file_url_resolves_as_local_path_and_stays_contained() {
    let tool = mock_tool("never reached");
    // `file://` strips to a local path; an absolute path outside the workspace is rejected by
    // containment (never read).
    let outcome = run_tool(
        &tool,
        r#"{"image_url":"file:///etc/passwd","question":"q"}"#,
    )
    .await;
    assert!(!outcome.result.ok);
    let v = result_json(&outcome);
    assert!(v["analysis"]
        .as_str()
        .unwrap()
        .contains("could not be read from the workspace"));
    // A local path is inside the agent's trust boundary — not fenced as untrusted.
    assert!(!outcome.untrusted);
}

// --- SSRF: the per-redirect-hop guard (wiremock) ------------------------------------------------

#[tokio::test]
async fn redirect_to_private_address_is_blocked() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/start"))
        .respond_with(
            ResponseTemplate::new(302)
                .insert_header("Location", "http://169.254.169.254/latest/meta-data/"),
        )
        .mount(&server)
        .await;

    let tool = mock_tool("never reached");
    // The helper is called directly (bypassing the initial-URL pre-flight, which would reject the
    // loopback mock) — exactly the seam split `web_extract` tests use.
    let err = tool
        .fetch_image(&format!("{}/start", server.uri()))
        .await
        .expect_err("redirect into metadata space must be blocked");
    assert!(
        err.to_string().contains("blocked url"),
        "expected an SSRF rejection, got: {err}"
    );
}

#[tokio::test]
async fn redirect_without_location_and_error_status_are_unreachable() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/nolocation"))
        .respond_with(ResponseTemplate::new(302))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/missing"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let tool = mock_tool("never reached");
    let err = tool
        .fetch_image(&format!("{}/nolocation", server.uri()))
        .await
        .expect_err("redirect without Location");
    assert!(err.to_string().contains("without a Location header"));

    let err = tool
        .fetch_image(&format!("{}/missing", server.uri()))
        .await
        .expect_err("404 must fail");
    assert!(err.to_string().contains("404"));
}

#[tokio::test]
async fn fetch_returns_body_bytes_on_success() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/img.png"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(png_fixture()))
        .mount(&server)
        .await;

    let tool = mock_tool("unused");
    let bytes = tool
        .fetch_image(&format!("{}/img.png", server.uri()))
        .await
        .expect("fetch ok");
    assert_eq!(bytes, png_fixture());
}

// --- Size caps ----------------------------------------------------------------------------------

#[tokio::test]
async fn download_cap_rejects_oversized_bodies() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/big.png"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![0u8; 200]))
        .mount(&server)
        .await;

    let tool = VisionAnalyzeTool::new(
        Arc::new(MockProvider::completing("unused")),
        VisionToolConfig {
            max_download_bytes: 64,
            ..Default::default()
        },
    );
    let err = tool
        .fetch_image(&format!("{}/big.png", server.uri()))
        .await
        .expect_err("200-byte body over a 64-byte cap");
    assert!(err.to_string().contains("too large"), "got: {err}");
}

#[tokio::test]
async fn base64_cap_rejects_with_actionable_error() {
    let tool = VisionAnalyzeTool::new(
        Arc::new(MockProvider::completing("never reached")),
        VisionToolConfig {
            max_base64_bytes: 16,
            ..Default::default()
        },
    );
    let args = format!(r#"{{"image_url":"{}","question":"q"}}"#, png_data_url());
    let outcome = run_tool(&tool, &args).await;
    assert!(!outcome.result.ok);
    let v = result_json(&outcome);
    assert!(v["analysis"]
        .as_str()
        .unwrap()
        .contains("Downscale or compress"));
}

// --- Workspace containment ----------------------------------------------------------------------

#[tokio::test]
async fn local_path_escape_is_rejected() {
    let tool = mock_tool("never reached");
    let outcome = run_tool(
        &tool,
        r#"{"image_url":"../vision-escape.png","question":"q"}"#,
    )
    .await;
    assert!(!outcome.result.ok);
    let v = result_json(&outcome);
    assert!(v["analysis"]
        .as_str()
        .unwrap()
        .contains("could not be read from the workspace"));
}

#[tokio::test]
async fn local_image_is_analyzed_and_trusted() {
    let exec = LocalEnvironment::sandbox("vision-local-ok");
    daemon_core::ExecutionEnvironment::write(
        &exec,
        std::path::Path::new("shots/img.png"),
        &png_fixture(),
    )
    .await
    .expect("write fixture");

    let tool = mock_tool("A cat on a mat.");
    let outcome = run_with(
        &tool,
        r#"{"image_url":"shots/img.png","question":"what is it?"}"#,
        &exec,
        CancellationToken::new(),
    )
    .await;
    assert!(outcome.result.ok, "content: {}", outcome.result.content);
    let v = result_json(&outcome);
    assert_eq!(v["success"], true);
    assert_eq!(v["analysis"], "A cat on a mat.");
    // A workspace-local image stays inside the trust boundary.
    assert!(!outcome.untrusted);
    let detail = outcome.detail.expect("detail envelope");
    assert_eq!(detail.kind, "vision_analyze");
}

// --- The aux happy path (request shape + trust) --------------------------------------------------

#[tokio::test]
async fn data_url_flows_to_the_aux_provider_with_hermes_request_shape() {
    let provider = Arc::new(CapturingProvider::new("Described."));
    let tool = VisionAnalyzeTool::new(provider.clone(), VisionToolConfig::default());
    let args = format!(
        r#"{{"image_url":"{}","question":"what breed is the dog?"}}"#,
        png_data_url()
    );
    let outcome = run_tool(&tool, &args).await;
    assert!(outcome.result.ok);
    let v = result_json(&outcome);
    assert_eq!(v["success"], true);
    assert_eq!(v["analysis"], "Described.");
    // Externally-transported image (data: URL) => the analysis is fenced as untrusted.
    assert!(outcome.untrusted);

    let req = provider
        .last
        .lock()
        .expect("capture lock")
        .clone()
        .expect("provider saw the request");
    assert_eq!(req.messages.len(), 1);
    let msg = &req.messages[0];
    assert_eq!(msg.role, "user");
    assert!(msg
        .content
        .starts_with("Fully describe and explain everything about this image"));
    assert!(msg.content.contains("what breed is the dog?"));
    assert_eq!(msg.images.len(), 1);
    assert_eq!(msg.images[0].mime, "image/png");
    assert_eq!(
        msg.images[0].data_base64,
        base64::engine::general_purpose::STANDARD.encode(png_fixture())
    );
    assert_eq!(req.params.temperature, Some(0.1));
    assert_eq!(req.params.max_tokens, Some(2000));
    assert_eq!(req.task.as_deref(), Some("vision"));
    assert_eq!(req.auth, None, "no credential configured => env fallback");
}

#[tokio::test]
async fn tool_surface_is_readonly_parallel_and_self_timed() {
    let tool = mock_tool("unused");
    let call = ToolCall {
        call_id: "c".into(),
        name: "vision_analyze".into(),
        args: "{}".into(),
    };
    assert_eq!(tool.concurrency(), ToolConcurrency::Parallel);
    assert_eq!(tool.concurrency_for(&call), ToolConcurrency::Parallel);
    assert!(!tool.mutates());
    assert!(!tool.mutates_for(&call));
    // Self-limiting: the engine's per-tool timeout stage is opted out.
    assert_eq!(
        tool.call_timeout(&call, Some(Duration::from_secs(30))),
        None
    );
    assert!(serde_json::from_str::<serde_json::Value>(tool.schema()).is_ok());
}

// --- Failure mapping, retry, timeout, cancellation ----------------------------------------------

#[tokio::test]
async fn provider_without_vision_capability_maps_to_friendly_error() {
    let tool = VisionAnalyzeTool::new(
        Arc::new(FailingProvider(
            "model gpt-x does not support image input".to_string(),
        )),
        VisionToolConfig::default(),
    );
    let args = format!(r#"{{"image_url":"{}","question":"q"}}"#, png_data_url());
    let outcome = run_tool(&tool, &args).await;
    assert!(!outcome.result.ok);
    let v = result_json(&outcome);
    assert_eq!(v["success"], false);
    assert!(v["analysis"]
        .as_str()
        .unwrap()
        .contains("does not support image input"));
    assert!(v["error"]
        .as_str()
        .unwrap()
        .starts_with("Error analyzing image:"));
}

#[tokio::test]
async fn empty_content_is_retried_once() {
    let provider = Arc::new(EmptyThenTextProvider {
        calls: AtomicU64::new(0),
    });
    let tool = VisionAnalyzeTool::new(provider.clone(), VisionToolConfig::default());
    let args = format!(r#"{{"image_url":"{}","question":"q"}}"#, png_data_url());
    let outcome = run_tool(&tool, &args).await;
    assert!(outcome.result.ok);
    let v = result_json(&outcome);
    assert_eq!(v["analysis"], "retried answer");
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn aux_call_times_out_against_the_configured_deadline() {
    let tool = VisionAnalyzeTool::new(
        Arc::new(PendingProvider),
        VisionToolConfig {
            call_timeout: Duration::from_millis(50),
            ..Default::default()
        },
    );
    let args = format!(r#"{{"image_url":"{}","question":"q"}}"#, png_data_url());
    let outcome = run_tool(&tool, &args).await;
    assert!(!outcome.result.ok);
    let v = result_json(&outcome);
    assert!(v["error"].as_str().unwrap().contains("timed out"));
}

#[tokio::test]
async fn cancellation_interrupts_the_call() {
    let tool = VisionAnalyzeTool::new(Arc::new(PendingProvider), VisionToolConfig::default());
    let exec = LocalEnvironment::sandbox("vision-cancel");
    let cancel = CancellationToken::new();
    cancel.cancel();
    let args = format!(r#"{{"image_url":"{}","question":"q"}}"#, png_data_url());
    let outcome = run_with(&tool, &args, &exec, cancel).await;
    assert!(!outcome.result.ok);
    let v = result_json(&outcome);
    assert_eq!(v["analysis"], "Interrupted");
}

// --- Input validation ----------------------------------------------------------------------------

#[tokio::test]
async fn non_image_payloads_are_rejected() {
    let tool = mock_tool("never reached");
    let args = r#"{"image_url":"data:text/plain;base64,aGVsbG8=","question":"q"}"#;
    let outcome = run_tool(&tool, args).await;
    assert!(!outcome.result.ok);
    let v = result_json(&outcome);
    assert!(v["analysis"]
        .as_str()
        .unwrap()
        .contains("Only real image files"));
}

#[tokio::test]
async fn invalid_arguments_and_empty_url_fail_clearly() {
    let tool = mock_tool("never reached");
    let outcome = run_tool(&tool, "not json").await;
    assert!(!outcome.result.ok);
    assert!(outcome.result.content.contains("invalid arguments"));

    let outcome = run_tool(&tool, r#"{"image_url":"  ","question":"q"}"#).await;
    assert!(!outcome.result.ok);
    let v = result_json(&outcome);
    assert!(v["analysis"]
        .as_str()
        .unwrap()
        .contains("Invalid image source"));
}
