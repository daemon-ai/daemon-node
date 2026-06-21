//! The dependency-light local fetch backend: `reqwest` GET + `dom_smoothie` (Mozilla-readability)
//! extraction to Markdown/text. No API key, so it is always [`available`](WebFetchBackend::available)
//! and serves as the `web_extract` fallback when no hosted scraper key is configured. It does not
//! render JavaScript.

use async_trait::async_trait;
use dom_smoothie::{Config, Readability, TextMode};

use crate::backend::{FetchFormat, FetchOpts, FetchedDoc, WebError, WebFetchBackend};

/// A default user agent so servers do not reject the fetch as a bare client.
const DEFAULT_UA: &str = "daemon-web-extract/0.1 (+https://github.com/example/daemon)";

/// The local reqwest+readability [`WebFetchBackend`].
pub struct LocalFetch {
    http: reqwest::Client,
}

impl LocalFetch {
    /// A local backend with the default user agent.
    pub fn new() -> Self {
        Self::with_user_agent(DEFAULT_UA)
    }

    /// A local backend with a custom user agent.
    pub fn with_user_agent(ua: &str) -> Self {
        let http = reqwest::Client::builder()
            .user_agent(ua)
            .build()
            .unwrap_or_default();
        Self { http }
    }
}

impl Default for LocalFetch {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WebFetchBackend for LocalFetch {
    fn name(&self) -> &str {
        "local"
    }

    async fn fetch(&self, url: &str, opts: &FetchOpts) -> Result<FetchedDoc, WebError> {
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|e| WebError::Http(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(WebError::Http(format!("fetch returned status {status}")));
        }
        let html = resp
            .text()
            .await
            .map_err(|e| WebError::Decode(e.to_string()))?;
        extract(url, &html, opts)
    }
}

/// Extract the main content of `html` (synchronous; isolated so tests can exercise it without a
/// network round-trip).
pub fn extract(url: &str, html: &str, opts: &FetchOpts) -> Result<FetchedDoc, WebError> {
    let text_mode = match opts.format {
        FetchFormat::Markdown => TextMode::Markdown,
        FetchFormat::Text => TextMode::Formatted,
    };
    let cfg = Config {
        text_mode,
        ..Default::default()
    };
    let mut readability = Readability::new(html, Some(url), Some(cfg))
        .map_err(|e| WebError::Decode(e.to_string()))?;
    let article = readability
        .parse()
        .map_err(|e| WebError::Decode(e.to_string()))?;
    let content = article.text_content.trim().to_string();
    if content.is_empty() {
        return Err(WebError::Empty);
    }
    let title = {
        let t = article.title.to_string();
        if t.trim().is_empty() {
            None
        } else {
            Some(t)
        }
    };
    Ok(FetchedDoc {
        url: url.to_string(),
        title,
        content,
        provider: "local".to_string(),
    })
}
