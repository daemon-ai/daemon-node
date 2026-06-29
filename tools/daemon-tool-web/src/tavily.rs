// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Tavily search backend (`POST https://api.tavily.com/search`, Bearer auth) — an LLM-oriented
//! search API returning ranked, summarized hits and an optional direct answer.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::backend::{
    SearchHit, SearchOpts, SearchResults, SearchTopic, SecretSource, WebError, WebSearchBackend,
};

/// The default Tavily search endpoint.
pub const TAVILY_ENDPOINT: &str = "https://api.tavily.com/search";

/// The Tavily-backed [`WebSearchBackend`]. Reads its API key live from the [`SecretSource`] under
/// `key_id` (default `"tavily"`).
pub struct TavilySearch {
    http: reqwest::Client,
    secrets: Arc<dyn SecretSource>,
    key_id: String,
    endpoint: String,
}

impl TavilySearch {
    /// A Tavily backend reading its key from the `"tavily"` credential profile.
    pub fn new(secrets: Arc<dyn SecretSource>) -> Self {
        Self {
            http: reqwest::Client::new(),
            secrets,
            key_id: "tavily".to_string(),
            endpoint: TAVILY_ENDPOINT.to_string(),
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
impl WebSearchBackend for TavilySearch {
    fn name(&self) -> &str {
        "tavily"
    }

    async fn search(&self, query: &str, opts: &SearchOpts) -> Result<SearchResults, WebError> {
        let key = self
            .secrets
            .secret(&self.key_id)
            .ok_or_else(|| WebError::MissingKey(self.key_id.clone()))?;
        let topic = match opts.topic {
            SearchTopic::General => "general",
            SearchTopic::News => "news",
        };
        let body = json!({
            "query": query,
            "max_results": opts.max_results,
            "topic": topic,
            "search_depth": "basic",
            "include_answer": true,
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
            return Err(WebError::Http(format!("tavily returned status {status}")));
        }
        let parsed: TavilyResponse = resp
            .json()
            .await
            .map_err(|e| WebError::Decode(e.to_string()))?;
        Ok(SearchResults {
            query: parsed.query.unwrap_or_else(|| query.to_string()),
            answer: parsed.answer.filter(|a| !a.trim().is_empty()),
            hits: parsed
                .results
                .into_iter()
                .map(|r| SearchHit {
                    title: r.title,
                    url: r.url,
                    snippet: r.content,
                    score: r.score,
                })
                .collect(),
            provider: "tavily".to_string(),
        })
    }
}

#[derive(Debug, Deserialize)]
struct TavilyResponse {
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    results: Vec<TavilyResult>,
}

#[derive(Debug, Deserialize)]
struct TavilyResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    score: Option<f64>,
}
