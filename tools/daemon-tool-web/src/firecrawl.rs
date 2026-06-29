// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Firecrawl fetch backend (`POST https://api.firecrawl.dev/v1/scrape`, Bearer auth) — a hosted
//! scraper that renders JS and returns clean Markdown.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::backend::{FetchOpts, FetchedDoc, SecretSource, WebError, WebFetchBackend};

/// The default Firecrawl scrape endpoint.
pub const FIRECRAWL_ENDPOINT: &str = "https://api.firecrawl.dev/v1/scrape";

/// The Firecrawl-backed [`WebFetchBackend`]. Reads its API key live from the [`SecretSource`] under
/// `key_id` (default `"firecrawl"`); reports [`available`](WebFetchBackend::available) only when a
/// key is present so the extract tool can fall through to the local backend.
pub struct FirecrawlFetch {
    http: reqwest::Client,
    secrets: Arc<dyn SecretSource>,
    key_id: String,
    endpoint: String,
}

impl FirecrawlFetch {
    /// A Firecrawl backend reading its key from the `"firecrawl"` credential profile.
    pub fn new(secrets: Arc<dyn SecretSource>) -> Self {
        Self {
            http: reqwest::Client::new(),
            secrets,
            key_id: "firecrawl".to_string(),
            endpoint: FIRECRAWL_ENDPOINT.to_string(),
        }
    }

    /// Override the credential-profile id the key is read from.
    pub fn with_key_id(mut self, key_id: impl Into<String>) -> Self {
        self.key_id = key_id.into();
        self
    }

    /// Override the endpoint (used by tests to point at a mock server).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }
}

#[async_trait]
impl WebFetchBackend for FirecrawlFetch {
    fn name(&self) -> &str {
        "firecrawl"
    }

    fn available(&self) -> bool {
        self.secrets.secret(&self.key_id).is_some()
    }

    async fn fetch(&self, url: &str, _opts: &FetchOpts) -> Result<FetchedDoc, WebError> {
        let key = self
            .secrets
            .secret(&self.key_id)
            .ok_or_else(|| WebError::MissingKey(self.key_id.clone()))?;
        let body = json!({
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": true,
        });
        let resp = self
            .http
            .post(&self.endpoint)
            .bearer_auth(key)
            .json(&body)
            .send()
            .await
            .map_err(|e| WebError::Http(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(WebError::Http(format!(
                "firecrawl returned status {status}"
            )));
        }
        let parsed: FirecrawlResponse = resp
            .json()
            .await
            .map_err(|e| WebError::Decode(e.to_string()))?;
        let data = parsed.data.ok_or(WebError::Empty)?;
        let content = data.markdown.unwrap_or_default();
        if content.trim().is_empty() {
            return Err(WebError::Empty);
        }
        Ok(FetchedDoc {
            url: url.to_string(),
            title: data.metadata.and_then(|m| m.title),
            content,
            provider: "firecrawl".to_string(),
        })
    }
}

#[derive(Debug, Deserialize)]
struct FirecrawlResponse {
    #[serde(default)]
    data: Option<FirecrawlData>,
}

#[derive(Debug, Deserialize)]
struct FirecrawlData {
    #[serde(default)]
    markdown: Option<String>,
    #[serde(default)]
    metadata: Option<FirecrawlMeta>,
}

#[derive(Debug, Deserialize)]
struct FirecrawlMeta {
    #[serde(default)]
    title: Option<String>,
}
