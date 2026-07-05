// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The supervised browser session — a single lazily-launched Chromium driven over CDP, modeled on
//! the `daemon-metta` `MettaCoprocessor`: the process is spawned on first use, guarded behind an
//! async mutex (CDP commands are serialized per session), torn down + respawned after a transport
//! fault, and protected by a crash-loop breaker so a missing/broken Chromium fails fast.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chromiumoxide::cdp::browser_protocol::fetch::{
    ContinueRequestParams, EnableParams as FetchEnableParams, EventRequestPaused, FailRequestParams,
};
use chromiumoxide::cdp::browser_protocol::network::ErrorReason;
use chromiumoxide::cdp::browser_protocol::page::{
    CaptureScreenshotFormat, EventJavascriptDialogOpening, HandleJavaScriptDialogParams,
};
use chromiumoxide::page::ScreenshotParams;
use chromiumoxide::{Browser, BrowserConfig, Page};
use daemon_core::check_url;
use dom_smoothie::{Config as ReadCfg, Readability, TextMode};
use futures::StreamExt;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use crate::error::BrowserError;

/// The output format for [`BrowserSupervisor::extract`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtractFormat {
    /// Readability-cleaned Markdown.
    Markdown,
    /// Visible page text (`document.body.innerText`).
    Text,
    /// The full rendered HTML.
    Html,
}

/// Launch + behaviour settings for the supervised browser.
#[derive(Clone, Debug)]
pub struct BrowserSettings {
    /// An explicit Chromium/Chrome executable path; `None` lets chromiumoxide auto-detect.
    pub chrome_path: Option<PathBuf>,
    /// Run headless (the default). `false` shows a window (local debugging only).
    pub headless: bool,
    /// Where PNG screenshots are written.
    pub screenshot_dir: PathBuf,
    /// How long to wait for the browser to come up.
    pub launch_timeout: Duration,
    /// Auto-dismiss JS dialogs (`alert`/`confirm`/`beforeunload`) so they cannot wedge the session.
    pub auto_dismiss_dialogs: bool,
}

impl Default for BrowserSettings {
    fn default() -> Self {
        Self {
            chrome_path: None,
            headless: true,
            screenshot_dir: std::env::temp_dir().join("daemon_browser_screenshots"),
            launch_timeout: Duration::from_secs(20),
            auto_dismiss_dialogs: true,
        }
    }
}

/// How many consecutive failed launches trip the crash-loop breaker.
const MAX_LAUNCH_FAILURES: u32 = 3;

/// A live browser + its single working page, plus the background tasks that drive the CDP socket,
/// the per-request egress guard, and dialog handling. Dropping it closes the browser.
struct Session {
    browser: Browser,
    page: Page,
    _handler: JoinHandle<()>,
    _dialog: Option<JoinHandle<()>>,
    /// Re-validates every request the page makes (incl. server redirects) against the egress
    /// policy; dropped with the session.
    _egress: JoinHandle<()>,
}

/// The supervised browser. One active page; CDP ops are serialized behind the async mutex.
pub struct BrowserSupervisor {
    settings: BrowserSettings,
    session: Mutex<Option<Session>>,
    launch_failures: AtomicU32,
    screenshot_seq: AtomicU32,
}

impl BrowserSupervisor {
    /// A supervisor with the given settings; the browser is launched lazily on the first op.
    pub fn new(settings: BrowserSettings) -> Self {
        Self {
            settings,
            session: Mutex::new(None),
            launch_failures: AtomicU32::new(0),
            screenshot_seq: AtomicU32::new(0),
        }
    }

    /// Navigate the active page to `url`, returning the final URL.
    pub async fn navigate(&self, url: &str) -> Result<String, BrowserError> {
        self.run(|page| {
            let url = url.to_string();
            Box::pin(async move {
                page.goto(url)
                    .await
                    .map_err(|e| BrowserError::Cdp(e.to_string()))?;
                page.wait_for_navigation()
                    .await
                    .map_err(|e| BrowserError::Cdp(e.to_string()))?;
                let current = page
                    .url()
                    .await
                    .map_err(|e| BrowserError::Cdp(e.to_string()))?
                    .unwrap_or_default();
                // Defense-in-depth: re-validate the final resolved URL. The per-request Fetch
                // guard already blocks a redirect hop into private space mid-chain (navigation then
                // fails above), but this catches any http(s) landing URL that slipped through.
                if !current.is_empty() && should_block_egress(&current) {
                    return Err(BrowserError::Ssrf(format!(
                        "navigation resolved to a blocked host: {current}"
                    )));
                }
                Ok(current)
            })
        })
        .await
    }

    /// Extract the active page's content in the requested format.
    pub async fn extract(
        &self,
        format: ExtractFormat,
    ) -> Result<(Option<String>, String), BrowserError> {
        self.run(move |page| {
            Box::pin(async move {
                match format {
                    ExtractFormat::Html => {
                        let html = page
                            .content()
                            .await
                            .map_err(|e| BrowserError::Cdp(e.to_string()))?;
                        Ok((None, html))
                    }
                    ExtractFormat::Text => {
                        let text =
                            eval_string(page, "document.body ? document.body.innerText : ''")
                                .await?;
                        Ok((None, text))
                    }
                    ExtractFormat::Markdown => {
                        let url = page.url().await.ok().flatten().unwrap_or_default();
                        let html = page
                            .content()
                            .await
                            .map_err(|e| BrowserError::Cdp(e.to_string()))?;
                        Ok(readability_markdown(&url, &html))
                    }
                }
            })
        })
        .await
    }

    /// Click the first element matching `selector`.
    pub async fn click(&self, selector: &str) -> Result<(), BrowserError> {
        self.run(|page| {
            let selector = selector.to_string();
            Box::pin(async move {
                let el = page
                    .find_element(selector)
                    .await
                    .map_err(|e| BrowserError::Cdp(e.to_string()))?;
                el.click()
                    .await
                    .map_err(|e| BrowserError::Cdp(e.to_string()))?;
                Ok(())
            })
        })
        .await
    }

    /// Type `text` into the first element matching `selector` (after focusing it).
    pub async fn type_text(&self, selector: &str, text: &str) -> Result<(), BrowserError> {
        self.run(|page| {
            let selector = selector.to_string();
            let text = text.to_string();
            Box::pin(async move {
                let el = page
                    .find_element(selector)
                    .await
                    .map_err(|e| BrowserError::Cdp(e.to_string()))?;
                el.click()
                    .await
                    .map_err(|e| BrowserError::Cdp(e.to_string()))?;
                el.type_str(text)
                    .await
                    .map_err(|e| BrowserError::Cdp(e.to_string()))?;
                Ok(())
            })
        })
        .await
    }

    /// Press a key (e.g. `Enter`) on the first element matching `selector`.
    pub async fn press_key(&self, selector: &str, key: &str) -> Result<(), BrowserError> {
        self.run(|page| {
            let selector = selector.to_string();
            let key = key.to_string();
            Box::pin(async move {
                let el = page
                    .find_element(selector)
                    .await
                    .map_err(|e| BrowserError::Cdp(e.to_string()))?;
                el.press_key(key)
                    .await
                    .map_err(|e| BrowserError::Cdp(e.to_string()))?;
                Ok(())
            })
        })
        .await
    }

    /// Wait until `selector` appears (polling) or `timeout` elapses.
    pub async fn wait_for(&self, selector: &str, timeout: Duration) -> Result<(), BrowserError> {
        self.run(|page| {
            let selector = selector.to_string();
            Box::pin(async move {
                let deadline = std::time::Instant::now() + timeout;
                loop {
                    if page.find_element(selector.clone()).await.is_ok() {
                        return Ok(());
                    }
                    if std::time::Instant::now() >= deadline {
                        return Err(BrowserError::Cdp(format!(
                            "timed out waiting for selector '{selector}'"
                        )));
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            })
        })
        .await
    }

    /// Evaluate a JavaScript expression and return its JSON-stringified result.
    pub async fn eval(&self, js: &str) -> Result<String, BrowserError> {
        self.run(|page| {
            let js = js.to_string();
            Box::pin(async move { eval_string(page, &js).await })
        })
        .await
    }

    /// Screenshot the active page; writes a PNG under the screenshot dir and returns its path.
    pub async fn screenshot(&self, full_page: bool) -> Result<PathBuf, BrowserError> {
        let dir = self.settings.screenshot_dir.clone();
        let seq = self.screenshot_seq.fetch_add(1, Ordering::Relaxed);
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = dir.join(format!("shot_{ts}_{seq}.png"));
        let path2 = path.clone();
        let bytes = self
            .run(move |page| {
                Box::pin(async move {
                    let params = ScreenshotParams::builder()
                        .format(CaptureScreenshotFormat::Png)
                        .full_page(full_page)
                        .build();
                    page.screenshot(params)
                        .await
                        .map_err(|e| BrowserError::Cdp(e.to_string()))
                })
            })
            .await?;
        // Scoped: the screenshot dir is the fixed, daemon-controlled artifact dir (default
        // `$TMPDIR/daemon_browser_screenshots`) and the filename is daemon-generated (`shot_<ts>_<seq>.png`)
        // -- no workspace/session/agent-supplied path component, so ContainedRoot is not required here.
        #[allow(clippy::disallowed_methods)]
        {
            std::fs::create_dir_all(&dir).map_err(|e| BrowserError::Io(e.to_string()))?;
            std::fs::write(&path2, &bytes).map_err(|e| BrowserError::Io(e.to_string()))?;
        }
        Ok(path)
    }

    /// Reload the active page.
    pub async fn reload(&self) -> Result<(), BrowserError> {
        self.eval("location.reload()").await.map(|_| ())
    }

    /// Navigate back in history.
    pub async fn back(&self) -> Result<(), BrowserError> {
        self.eval("history.back()").await.map(|_| ())
    }

    /// Close the browser (the next op relaunches it).
    pub async fn close(&self) -> Result<(), BrowserError> {
        let mut guard = self.session.lock().await;
        if let Some(mut session) = guard.take() {
            let _ = session.browser.close().await;
            let _ = session.browser.wait().await;
        }
        Ok(())
    }

    /// Run `op` against the active page, launching the session if needed and tearing it down on a
    /// transport fault so the next call respawns.
    async fn run<F, R>(&self, op: F) -> Result<R, BrowserError>
    where
        F: for<'p> FnOnce(
            &'p Page,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<R, BrowserError>> + Send + 'p>,
        >,
    {
        let mut guard = self.session.lock().await;
        if guard.is_none() {
            *guard = Some(self.launch().await?);
        }
        let session = guard.as_ref().expect("session present after launch");
        let result = op(&session.page).await;
        if result.is_err() {
            // Tear down on fault; the next op relaunches a clean session.
            if let Some(mut s) = guard.take() {
                let _ = s.browser.close().await;
            }
        }
        result
    }

    /// Launch a fresh browser + page, wiring the CDP-driving task and optional dialog auto-dismiss.
    async fn launch(&self) -> Result<Session, BrowserError> {
        if self.launch_failures.load(Ordering::Relaxed) >= MAX_LAUNCH_FAILURES {
            return Err(BrowserError::CrashLoop(MAX_LAUNCH_FAILURES));
        }
        match self.try_launch().await {
            Ok(session) => {
                self.launch_failures.store(0, Ordering::Relaxed);
                Ok(session)
            }
            Err(e) => {
                self.launch_failures.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    async fn try_launch(&self) -> Result<Session, BrowserError> {
        let mut builder = BrowserConfig::builder().launch_timeout(self.settings.launch_timeout);
        if !self.settings.headless {
            builder = builder.with_head();
        }
        if let Some(path) = &self.settings.chrome_path {
            builder = builder.chrome_executable(path);
        }
        let config = builder.build().map_err(BrowserError::Launch)?;
        let (browser, mut handler) = Browser::launch(config)
            .await
            .map_err(|e| BrowserError::Launch(e.to_string()))?;
        let handler_task = tokio::spawn(async move {
            while let Some(ev) = handler.next().await {
                if ev.is_err() {
                    break;
                }
            }
        });
        let page = browser
            .new_page("about:blank")
            .await
            .map_err(|e| BrowserError::Launch(e.to_string()))?;
        // Install the per-request egress guard BEFORE any navigation. If it cannot be wired we fail
        // the launch rather than run an unguarded browser (fail-closed) — an unguarded page would
        // follow a redirect into loopback/link-local/private space with no re-check.
        let egress_task = spawn_egress_guard(&page).await?;
        let dialog_task = if self.settings.auto_dismiss_dialogs {
            spawn_dialog_dismisser(&page).await
        } else {
            None
        };
        Ok(Session {
            browser,
            page,
            _handler: handler_task,
            _dialog: dialog_task,
            _egress: egress_task,
        })
    }
}

/// Whether a request URL must be blocked by the egress guard: an `http(s)` request to a
/// private/loopback/link-local/otherwise-disallowed host (per [`check_url`]). Non-`http(s)` schemes
/// (`data:`, `blob:`, `about:`, …) are not host egress and are always allowed — blocking them would
/// break legitimate rendering without closing any SSRF vector.
fn should_block_egress(url: &str) -> bool {
    let lower = url.trim().to_ascii_lowercase();
    let is_http = lower.starts_with("http://") || lower.starts_with("https://");
    is_http && check_url(url).is_err()
}

/// Enable the CDP Fetch domain and spawn a task that re-validates **every** request the page makes
/// — including server-driven redirect hops, which Chromium follows in its own network stack — against
/// the egress policy, failing (`BlockedByClient`) any that resolves to a disallowed host. This is
/// the browser-native equivalent of the HTTP tools' manual, per-hop `check_url` redirect loop: the
/// reqwest egress client cannot see Chromium's redirects, so the guard lives at the interception
/// point. Fails closed — a setup error aborts the launch.
async fn spawn_egress_guard(page: &Page) -> Result<JoinHandle<()>, BrowserError> {
    // Subscribe BEFORE enabling so no paused request is missed (a missed pause hangs the page).
    let mut paused = page
        .event_listener::<EventRequestPaused>()
        .await
        .map_err(|e| BrowserError::Cdp(e.to_string()))?;
    // No patterns => every request pauses at the Request stage until we continue/fail it.
    page.execute(FetchEnableParams::default())
        .await
        .map_err(|e| BrowserError::Cdp(e.to_string()))?;
    let page = page.clone();
    Ok(tokio::spawn(async move {
        while let Some(ev) = paused.next().await {
            let request_id = ev.request_id.clone();
            let result = if should_block_egress(&ev.request.url) {
                page.execute(FailRequestParams::new(
                    request_id,
                    ErrorReason::BlockedByClient,
                ))
                .await
                .map(|_| ())
            } else {
                page.execute(ContinueRequestParams::new(request_id))
                    .await
                    .map(|_| ())
            };
            // A continue/fail can race a page teardown; that is benign (the session is gone).
            let _ = result;
        }
    }))
}

/// Evaluate `js` and return its result as a string (objects are JSON-encoded).
async fn eval_string(page: &Page, js: &str) -> Result<String, BrowserError> {
    let result = page
        .evaluate(js)
        .await
        .map_err(|e| BrowserError::Cdp(e.to_string()))?;
    match result.into_value::<serde_json::Value>() {
        Ok(serde_json::Value::String(s)) => Ok(s),
        Ok(other) => Ok(other.to_string()),
        Err(_) => Ok(String::new()),
    }
}

/// Convert page HTML to readability Markdown; returns `(title, markdown)`.
fn readability_markdown(url: &str, html: &str) -> (Option<String>, String) {
    let cfg = ReadCfg {
        text_mode: TextMode::Markdown,
        ..Default::default()
    };
    let doc_url = if url.is_empty() { None } else { Some(url) };
    match Readability::new(html, doc_url, Some(cfg)).and_then(|mut r| r.parse()) {
        Ok(article) => {
            let title = {
                let t = article.title.to_string();
                (!t.trim().is_empty()).then_some(t)
            };
            (title, article.text_content.trim().to_string())
        }
        // Fall back to raw HTML if extraction fails.
        Err(_) => (None, html.to_string()),
    }
}

/// Spawn a task that auto-dismisses JS dialogs so a modal cannot wedge the session.
async fn spawn_dialog_dismisser(page: &Page) -> Option<JoinHandle<()>> {
    let mut events = page
        .event_listener::<EventJavascriptDialogOpening>()
        .await
        .ok()?;
    let page = page.clone();
    Some(tokio::spawn(async move {
        while events.next().await.is_some() {
            if let Ok(params) = HandleJavaScriptDialogParams::builder()
                .accept(false)
                .build()
            {
                let _ = page.execute(params).await;
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Fetch-interceptor predicate: block only `http(s)` requests to disallowed hosts; let
    /// non-host schemes through. This is the decision the per-request guard makes on every hop,
    /// including a server redirect — so a `302` into link-local/loopback/private space is failed
    /// mid-navigation. Exercised without a live Chromium (pure function).
    #[test]
    fn egress_guard_blocks_only_disallowed_http_hosts() {
        // Public http(s): allowed.
        assert!(!should_block_egress("https://example.com/a"));
        assert!(!should_block_egress("http://1.1.1.1/"));

        // Private / loopback / link-local / metadata http(s): blocked (this is the redirect-SSRF
        // fix — a hop resolving here is failed).
        for u in [
            "http://169.254.169.254/latest/meta-data/",
            "http://127.0.0.1/x",
            "http://localhost/x",
            "http://10.0.0.5/x",
            "http://192.168.1.1/x",
            "http://[::1]/x",
            "http://[fc00::1]/x",
        ] {
            assert!(should_block_egress(u), "expected {u} to be blocked");
        }

        // Non-http(s) schemes are not host egress and must pass (or rendering breaks).
        assert!(!should_block_egress("data:image/png;base64,iVBORw0KGgo="));
        assert!(!should_block_egress("about:blank"));
        assert!(!should_block_egress("blob:https://example.com/uuid"));
    }
}
