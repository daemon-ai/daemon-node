// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-tool-vision` — the `vision_analyze` chat tool (a [`daemon_core::Tool`]).
//!
//! The hermes `vision_analyze` aux-LLM path ("path B", `tools/vision_tools.py`): resolve an image
//! from an http(s) URL / workspace-local path / `data:` URL, validate it (SSRF egress policy with
//! per-redirect-hop re-checks, workspace containment, magic-byte MIME sniff, hard size caps), then
//! ask a **vision-capable auxiliary provider** to describe it and answer the model's question. The
//! result is the hermes-shaped JSON `{"success": bool, "analysis": string}`.
//!
//! Design decisions (coordinator-approved):
//! - No auto-resize: an image over the base64 cap is rejected with an actionable error rather than
//!   pulling an image-codec dependency. The caps mirror hermes (50 MiB fetch, 20 MiB base64).
//! - The sniffed MIME always wins over any declared type (`data:` mediatype, file extension).
//! - Trust is source-dependent: analyses of http(s)/`data:` images return as **untrusted** tool
//!   output (prompt injection rides inside images exactly like web pages); workspace-local images
//!   return trusted.
//! - The aux provider is injected (an [`Arc<dyn Provider>`]), mirroring how the host wires
//!   `lcm_aux`; credentials thread through `Request::auth` when configured, else the provider's
//!   environment fallback applies.

#![forbid(unsafe_code)]
// Phase 4: test code may use raw fs/reqwest/Command; the --lib pass still guards production.
#![cfg_attr(test, allow(clippy::disallowed_methods, clippy::disallowed_types))]

mod error;
mod input;
mod mime;

pub use error::VisionError;
pub use input::{classify_source, parse_data_url, ImageSource};
pub use mime::sniff_image_mime;

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use daemon_core::{
    check_url, ModelOutput, Provider, Request, RequestImage, RequestMsg, RequestParams, Tool,
    ToolCall, ToolConcurrency, ToolOutcome, TurnCx,
};
use daemon_protocol::ToolDetail;
use serde::Deserialize;

use crate::mime::SNIFF_HEADER_LEN;

/// The JSON-Schema advertised for `vision_analyze` (hermes `VISION_ANALYZE_SCHEMA` parity).
const VISION_ANALYZE_SCHEMA: &str = r#"{
  "type": "object",
  "required": ["image_url", "question"],
  "properties": {
    "image_url": {"type": "string", "description": "Image source: an http(s) URL, a workspace file path, or a data: URL. Supported formats: PNG, JPEG, GIF, BMP, WebP."},
    "question": {"type": "string", "description": "Your specific question or request about the image; the vision model fully describes the image and then answers it."}
  }
}"#;

/// Sampling temperature for the aux vision call (hermes `auxiliary.vision.temperature` default).
const VISION_TEMPERATURE: f64 = 0.1;
/// Output-token cap for the aux vision call (hermes `max_tokens=2000`).
const VISION_MAX_TOKENS: u32 = 2000;
/// How many redirects the download follows before giving up (each hop is re-validated).
const MAX_REDIRECT_HOPS: usize = 5;
/// Per-request download deadline (hermes `auxiliary.vision.download_timeout` default).
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(30);
/// The download user agent (a plain product identity, like `daemon-tool-web`'s).
const USER_AGENT: &str = "daemon-vision/0.1 (+https://github.com/daemon-ai/daemon-node)";
/// The full-description wrapper prepended to the model's question (hermes
/// `_handle_vision_analyze` parity).
const FULL_DESCRIPTION_PREFIX: &str =
    "Fully describe and explain everything about this image, then answer the following question:\n\n";
/// The analysis fallback when the aux model returns empty content twice (hermes parity).
const EMPTY_ANALYSIS_FALLBACK: &str =
    "There was a problem with the request and the image could not be analyzed.";

/// Construction-time tuning for [`VisionAnalyzeTool`] (the host maps its `[vision]` config here).
#[derive(Clone, Debug)]
pub struct VisionToolConfig {
    /// The bearer credential threaded into [`Request::auth`] for each aux call. `None` (the
    /// default) lets the provider fall back to its environment credential. Treat as a secret.
    pub auth: Option<String>,
    /// The aux vision-call deadline (hermes `auxiliary.vision.timeout` default: 120s).
    pub call_timeout: Duration,
    /// The hard cap on downloaded / inline image bytes (hermes `_VISION_MAX_DOWNLOAD_BYTES`).
    pub max_download_bytes: u64,
    /// The hard cap on the base64 payload handed to the provider (hermes `_MAX_BASE64_BYTES`).
    pub max_base64_bytes: usize,
}

impl Default for VisionToolConfig {
    fn default() -> Self {
        Self {
            auth: None,
            call_timeout: Duration::from_secs(120),
            max_download_bytes: 50 * 1024 * 1024,
            max_base64_bytes: 20 * 1024 * 1024,
        }
    }
}

/// The `vision_analyze` tool: resolve + validate an image, then describe it through the injected
/// vision-capable aux provider.
pub struct VisionAnalyzeTool {
    aux: Arc<dyn Provider>,
    // Straggler (scoped): vision predates and mirrors daemon-egress -- it is the *proven*
    // self-contained SSRF-safe pattern (Policy::none() + per-hop check_url via `next_hop`,
    // MAX_REDIRECT_HOPS). Kept as-is; dedupe into daemon-egress is a follow-up.
    #[allow(clippy::disallowed_types)]
    http: reqwest::Client,
    auth: Option<String>,
    call_timeout: Duration,
    max_download_bytes: u64,
    max_base64_bytes: usize,
}

impl VisionAnalyzeTool {
    /// A vision tool over the given aux provider. The shared HTTP client follows **no** redirects
    /// on its own — the fetch loop follows them manually so every hop passes the egress check.
    #[allow(clippy::disallowed_types)] // scoped straggler: self-contained SSRF-safe client (see struct)
    pub fn new(aux: Arc<dyn Provider>, cfg: VisionToolConfig) -> Self {
        let http = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .redirect(reqwest::redirect::Policy::none())
            .timeout(DOWNLOAD_TIMEOUT)
            .build()
            // Client construction fails only when the TLS backend cannot initialize — a
            // boot-environment defect. Failing loudly beats silently swapping in a default client
            // whose redirect-following would bypass the per-hop SSRF re-validation.
            .expect("vision: building the no-redirect HTTP client");
        Self {
            aux,
            http,
            auth: cfg.auth,
            call_timeout: cfg.call_timeout,
            max_download_bytes: cfg.max_download_bytes,
            max_base64_bytes: cfg.max_base64_bytes,
        }
    }

    /// Download an image, following up to [`MAX_REDIRECT_HOPS`] redirects with an egress re-check
    /// on **every** hop (a public URL 302-ing to `http://169.254.169.254/` is rejected — hermes
    /// `_ssrf_redirect_guard` parity), a `Content-Length` pre-check, and a streamed byte cap.
    ///
    /// The *initial* URL is deliberately not re-checked here: the tool's `run` pre-flight owns
    /// that (mirroring `web_extract`'s tool-level guard over an unchecked backend), which also
    /// keeps this helper exercisable against a loopback mock server in tests.
    pub async fn fetch_image(&self, url: &str) -> Result<Vec<u8>, VisionError> {
        let mut current = url.to_string();
        for _ in 0..=MAX_REDIRECT_HOPS {
            let resp = self
                .http
                .get(&current)
                .header(reqwest::header::ACCEPT, "image/*,*/*;q=0.8")
                .send()
                .await
                .map_err(|e| VisionError::Unreachable(e.to_string()))?;
            let status = resp.status();
            if status.is_redirection() {
                let location = resp
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .ok_or_else(|| {
                        VisionError::Unreachable(format!(
                            "redirect ({status}) without a Location header"
                        ))
                    })?;
                current = next_hop(&current, location)?;
                continue;
            }
            if !status.is_success() {
                return Err(VisionError::Unreachable(format!(
                    "fetch returned status {status}"
                )));
            }
            // Reject oversized images early via Content-Length, then enforce the same cap on the
            // actual streamed bytes (the header can lie or be absent).
            if let Some(len) = resp.content_length() {
                if len > self.max_download_bytes {
                    return Err(VisionError::TooLarge(format!(
                        "image is {} (Content-Length), over the {} download cap",
                        fmt_mb(len),
                        fmt_mb(self.max_download_bytes)
                    )));
                }
            }
            let mut resp = resp;
            let mut body: Vec<u8> = Vec::new();
            while let Some(chunk) = resp
                .chunk()
                .await
                .map_err(|e| VisionError::Unreachable(e.to_string()))?
            {
                if (body.len() + chunk.len()) as u64 > self.max_download_bytes {
                    return Err(VisionError::TooLarge(format!(
                        "image exceeds the {} download cap",
                        fmt_mb(self.max_download_bytes)
                    )));
                }
                body.extend_from_slice(&chunk);
            }
            return Ok(body);
        }
        Err(VisionError::Unreachable(format!(
            "too many redirects (limit {MAX_REDIRECT_HOPS})"
        )))
    }

    /// Resolve the image bytes for `source`, validate them, and run the aux vision call.
    async fn analyze(
        &self,
        source: &ImageSource,
        question: &str,
        cx: &TurnCx<'_>,
    ) -> Result<String, VisionError> {
        let bytes = match source {
            ImageSource::Http(url) => {
                // The pre-flight egress check on the initial URL; redirect hops are re-checked
                // inside the fetch loop.
                let checked =
                    check_url(url).map_err(|reject| VisionError::Ssrf(reject.to_string()))?;
                self.fetch_image(&checked.url).await?
            }
            ImageSource::DataUrl(raw) => {
                let bytes = parse_data_url(raw)?;
                if bytes.len() as u64 > self.max_download_bytes {
                    return Err(VisionError::TooLarge(format!(
                        "data: URL payload is {}, over the {} cap",
                        fmt_mb(bytes.len() as u64),
                        fmt_mb(self.max_download_bytes)
                    )));
                }
                bytes
            }
            // Workspace containment is enforced by the execution environment (`contain()`).
            ImageSource::Local(path) => cx
                .exec
                .read(path)
                .await
                .map_err(|e| VisionError::Workspace(e.to_string()))?,
        };

        // The magic-byte sniff is authoritative — a declared MIME (data: mediatype, extension)
        // never overrides what the bytes actually are.
        let mime =
            sniff_image_mime(&bytes[..bytes.len().min(SNIFF_HEADER_LEN)]).ok_or_else(|| {
                VisionError::NotImage(
                    "only real image files are supported (PNG, JPEG, GIF, BMP, WebP)".to_string(),
                )
            })?;

        // Exact padded-base64 length, computed before encoding so an oversized payload is
        // rejected without a giant transient allocation. No auto-resize by design: the caller
        // gets an actionable error instead (coordinator ruling).
        let encoded_len = bytes.len().div_ceil(3) * 4;
        if encoded_len > self.max_base64_bytes {
            return Err(VisionError::TooLarge(format!(
                "base64 payload would be {}, over the {} vision cap; downscale or compress the \
                 image and retry",
                fmt_mb(encoded_len as u64),
                fmt_mb(self.max_base64_bytes as u64)
            )));
        }
        let data_base64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        drop(bytes);

        let prompt = format!("{FULL_DESCRIPTION_PREFIX}{question}");
        let request = self.build_request(prompt, mime, data_base64);
        let out = self.chat_with_deadline(request.clone()).await?;
        if let Some(analysis) = extract_analysis(&out) {
            return Ok(analysis);
        }
        // Hermes parity: one bounded retry when the model answered with empty content
        // (reasoning-only responses happen on some vision stacks).
        let out = self.chat_with_deadline(request).await?;
        Ok(extract_analysis(&out).unwrap_or_else(|| EMPTY_ANALYSIS_FALLBACK.to_string()))
    }

    /// The multimodal aux request: one user message carrying the prompt text + the image part,
    /// hermes' sampling params (`temperature=0.1`, `max_tokens=2000`), and the `"vision"` task
    /// label. The configured bearer (when any) threads through `auth`.
    fn build_request(&self, prompt: String, mime: &str, data_base64: String) -> Request {
        Request {
            messages: vec![RequestMsg {
                role: "user".to_string(),
                content: prompt,
                images: vec![RequestImage {
                    mime: mime.to_string(),
                    data_base64,
                }],
                ..Default::default()
            }],
            auth: self.auth.clone(),
            ..Default::default()
        }
        .with_params(RequestParams {
            temperature: Some(VISION_TEMPERATURE),
            max_tokens: Some(VISION_MAX_TOKENS),
            ..Default::default()
        })
        .with_task("vision")
    }

    /// Run one aux chat bounded by the configured vision deadline.
    async fn chat_with_deadline(&self, request: Request) -> Result<ModelOutput, VisionError> {
        match tokio::time::timeout(self.call_timeout, self.aux.chat(request)).await {
            Ok(Ok(out)) => Ok(out),
            Ok(Err(failure)) => Err(VisionError::Provider(failure.to_string())),
            Err(_) => Err(VisionError::Timeout(self.call_timeout)),
        }
    }
}

/// The tool arguments. `question` tolerates absence (hermes' handler defaults it to empty).
#[derive(Debug, Deserialize)]
struct Args {
    image_url: String,
    #[serde(default)]
    question: String,
}

#[async_trait]
impl Tool for VisionAnalyzeTool {
    fn name(&self) -> &str {
        "vision_analyze"
    }

    fn schema(&self) -> &str {
        VISION_ANALYZE_SCHEMA
    }

    fn concurrency(&self) -> ToolConcurrency {
        // Read-only fetch + aux describe with no shared-state mutation: safe to batch concurrently.
        ToolConcurrency::Parallel
    }

    fn call_timeout(&self, _call: &ToolCall, _default: Option<Duration>) -> Option<Duration> {
        // Self-limiting: the aux call is bounded by `[vision]`'s own deadline and each download
        // request by DOWNLOAD_TIMEOUT, so the engine's per-tool timeout stage is opted out.
        None
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let args: Args = match serde_json::from_str(&call.args) {
            Ok(a) => a,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("vision_analyze: invalid arguments: {e}"),
                )
            }
        };
        if args.image_url.trim().is_empty() {
            return render_outcome(
                &call.call_id,
                Err(VisionError::BadInput("image_url is required".to_string())),
                false,
            );
        }
        let source = classify_source(&args.image_url);
        // Coordinator ruling: analyses of externally-transported images (http/data:) are untrusted
        // — prompt injection rides inside images exactly like web pages — while workspace-local
        // images stay inside the agent's own trust boundary.
        let untrusted = matches!(source, ImageSource::Http(_) | ImageSource::DataUrl(_));
        let outcome = tokio::select! {
            _ = cx.cancel.cancelled() => Err(VisionError::Cancelled),
            res = self.analyze(&source, &args.question, cx) => res,
        };
        render_outcome(&call.call_id, outcome, untrusted)
    }
}

/// Render an analysis outcome as the hermes-shaped result JSON (`{"success", "analysis"[, "error"]}`)
/// with a `vision_analyze` detail envelope, honoring the source-dependent trust classification.
fn render_outcome(
    call_id: &str,
    outcome: Result<String, VisionError>,
    untrusted: bool,
) -> ToolOutcome {
    let (ok, body) = match outcome {
        Ok(analysis) => (
            true,
            serde_json::json!({ "success": true, "analysis": analysis }),
        ),
        Err(e) => (
            false,
            serde_json::json!({
                "success": false,
                "error": format!("Error analyzing image: {e}"),
                "analysis": friendly_analysis(&e),
            }),
        ),
    };
    let content = serde_json::to_string_pretty(&body).unwrap_or_else(|_| body.to_string());
    let detail = ToolDetail {
        kind: "vision_analyze".to_string(),
        body: serde_json::to_vec(&body).unwrap_or_default(),
    };
    let outcome = if untrusted {
        ToolOutcome::untrusted_text(call_id.to_string(), ok, content)
    } else {
        ToolOutcome::text(call_id.to_string(), ok, content)
    };
    outcome.with_detail(detail)
}

/// Extract the analysis text from a model output: the content channel, falling back to the
/// reasoning channel when content is empty (hermes `extract_content_or_reasoning` parity).
fn extract_analysis(out: &ModelOutput) -> Option<String> {
    let text = out.text.trim();
    if !text.is_empty() {
        return Some(text.to_string());
    }
    let reasoning = out.reasoning.as_deref().unwrap_or("").trim();
    if !reasoning.is_empty() {
        return Some(reasoning.to_string());
    }
    None
}

/// A human/model-actionable `analysis` string for a failure — the hermes error-hint classification
/// (`vision_tools.py` error branch): billing, missing vision capability, and rejected-image
/// conditions each get a distinct explanation instead of a bare API error.
fn friendly_analysis(err: &VisionError) -> String {
    match err {
        VisionError::Provider(_) | VisionError::Timeout(_) => {
            let hint = err.to_string().to_ascii_lowercase();
            if [
                "402",
                "insufficient",
                "payment required",
                "credits",
                "billing",
            ]
            .iter()
            .any(|h| hint.contains(h))
            {
                format!(
                    "Insufficient credits or payment required. Please top up your API provider \
                     account and try again. Error: {err}"
                )
            } else if [
                "does not support",
                "not support image",
                "content policy",
                "content_policy",
                "multimodal",
                "unrecognized request argument",
                "image input",
            ]
            .iter()
            .any(|h| hint.contains(h))
            {
                format!(
                    "The configured vision model does not support image input or the request was \
                     not accepted by the server. Error: {err}"
                )
            } else if ["invalid_request", "image_url", "payload too large", "413"]
                .iter()
                .any(|h| hint.contains(h))
            {
                format!(
                    "The vision API rejected the image. This can happen when the image is in an \
                     unsupported format, corrupted, or too large. Try a smaller JPEG/PNG and \
                     retry. Error: {err}"
                )
            } else {
                format!(
                    "There was a problem with the request and the image could not be analyzed. \
                     Error: {err}"
                )
            }
        }
        VisionError::TooLarge(_) => format!(
            "The image is too large for vision analysis. Downscale or compress the image and \
             retry. Error: {err}"
        ),
        VisionError::NotImage(_) => format!(
            "Only real image files (PNG, JPEG, GIF, BMP, WebP) are supported for vision \
             analysis. Error: {err}"
        ),
        VisionError::Ssrf(_) => format!(
            "The image URL was blocked by the egress safety policy (private, loopback, and \
             link-local hosts are not allowed). Error: {err}"
        ),
        VisionError::Unreachable(_) => {
            format!("The image could not be downloaded. Error: {err}")
        }
        VisionError::BadInput(_) => format!(
            "Invalid image source. Provide an HTTP/HTTPS URL, a workspace file path, or a data: \
             URL. Error: {err}"
        ),
        VisionError::Workspace(_) => {
            format!("The local image path could not be read from the workspace. Error: {err}")
        }
        VisionError::Cancelled => "Interrupted".to_string(),
    }
}

/// Resolve one redirect hop: join `location` (absolute or relative) against the current URL and
/// re-validate the target against the egress policy — the redirect-based SSRF guard (hermes
/// `_ssrf_redirect_guard` parity). Every followed hop passes through here.
fn next_hop(current: &str, location: &str) -> Result<String, VisionError> {
    let base = reqwest::Url::parse(current)
        .map_err(|e| VisionError::Unreachable(format!("invalid url: {e}")))?;
    let next = base
        .join(location)
        .map_err(|e| VisionError::Unreachable(format!("invalid redirect location: {e}")))?;
    let next = next.to_string();
    check_url(&next).map_err(|reject| VisionError::Ssrf(reject.to_string()))?;
    Ok(next)
}

/// Format a byte count as a human-readable MB figure for error messages.
fn fmt_mb(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_analysis_prefers_text_then_reasoning() {
        let out = ModelOutput {
            text: "the answer".to_string(),
            reasoning: Some("thinking".to_string()),
            ..Default::default()
        };
        assert_eq!(extract_analysis(&out).as_deref(), Some("the answer"));

        let out = ModelOutput {
            text: "  ".to_string(),
            reasoning: Some("reasoning only".to_string()),
            ..Default::default()
        };
        assert_eq!(extract_analysis(&out).as_deref(), Some("reasoning only"));

        let out = ModelOutput::default();
        assert_eq!(extract_analysis(&out), None);
    }

    #[test]
    fn next_hop_joins_and_revalidates_redirect_targets() {
        // Absolute public target: allowed.
        assert_eq!(
            next_hop("https://example.com/a", "https://cdn.example.net/img.png").unwrap(),
            "https://cdn.example.net/img.png"
        );
        // Relative target joins against the current URL.
        assert_eq!(
            next_hop("https://example.com/dir/a", "b.png").unwrap(),
            "https://example.com/dir/b.png"
        );
        // Redirects into private / loopback / metadata space are rejected.
        for target in [
            "http://169.254.169.254/latest/meta-data/",
            "http://localhost:8080/x",
            "http://10.0.0.5/x",
            "http://[::1]/x",
        ] {
            assert!(
                matches!(
                    next_hop("https://example.com/a", target),
                    Err(VisionError::Ssrf(_))
                ),
                "expected {target} to be blocked"
            );
        }
        // A scheme downgrade to a non-http scheme is rejected by the same policy.
        assert!(matches!(
            next_hop("https://example.com/a", "file:///etc/passwd"),
            Err(VisionError::Ssrf(_))
        ));
    }

    #[test]
    fn friendly_analysis_classifies_provider_hints() {
        let billing = friendly_analysis(&VisionError::Provider("billing: 402 no credits".into()));
        assert!(billing.contains("Insufficient credits"));

        let no_vision = friendly_analysis(&VisionError::Provider(
            "model gpt-x does not support image input".into(),
        ));
        assert!(no_vision.contains("does not support image input"));

        let rejected = friendly_analysis(&VisionError::Provider(
            "payload too large: request body limit".into(),
        ));
        assert!(rejected.contains("rejected the image"));

        let generic = friendly_analysis(&VisionError::Provider("boom".into()));
        assert!(generic.contains("could not be analyzed"));
    }

    #[test]
    fn build_request_carries_image_params_task_and_auth() {
        struct NeverProvider;
        #[async_trait]
        impl Provider for NeverProvider {
            fn capabilities(&self) -> daemon_core::Capabilities {
                daemon_core::Capabilities {
                    supports_native_tools: false,
                    supports_streaming: false,
                    tool_call_format: daemon_core::ToolCallFormat::Native,
                    max_context: None,
                }
            }
            async fn chat(&self, _req: Request) -> Result<ModelOutput, daemon_core::Failure> {
                Ok(ModelOutput::default())
            }
        }
        let tool = VisionAnalyzeTool::new(
            Arc::new(NeverProvider),
            VisionToolConfig {
                auth: Some("sekrit".to_string()),
                ..Default::default()
            },
        );
        let req = tool.build_request("prompt".into(), "image/png", "QUJD".into());
        assert_eq!(req.messages.len(), 1);
        let msg = &req.messages[0];
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "prompt");
        assert_eq!(msg.images.len(), 1);
        assert_eq!(msg.images[0].mime, "image/png");
        assert_eq!(msg.images[0].data_base64, "QUJD");
        assert_eq!(req.params.temperature, Some(VISION_TEMPERATURE));
        assert_eq!(req.params.max_tokens, Some(VISION_MAX_TOKENS));
        assert_eq!(req.task.as_deref(), Some("vision"));
        assert_eq!(req.auth.as_deref(), Some("sekrit"));
    }
}
