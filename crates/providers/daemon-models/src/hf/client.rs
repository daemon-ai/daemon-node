// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! A thin async Hugging Face Hub HTTP client (read-only: search + tree).
//!
//! Wraps `reqwest` with the Hub endpoint, an optional bearer token, and status→[`ModelError`]
//! classification. The endpoint is overridable so the unit tests can point it at an in-process
//! `wiremock` server (no live network in the suite).

use serde::de::DeserializeOwned;

use crate::error::{ModelError, Result};

/// The default Hugging Face Hub endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// A read-only Hugging Face Hub client.
#[derive(Clone, Debug)]
pub struct HfClient {
    // Straggler (scoped): a raw reqwest client to the fixed Hugging Face Hub endpoint
    // (huggingface.co, or a test mock); not an agent-controlled URL. Dedupe into daemon-egress later.
    #[allow(clippy::disallowed_types)]
    client: reqwest::Client,
    endpoint: String,
    token: Option<String>,
}

/// A JSON body plus the `rel="next"` pagination URL parsed from the response `Link` header.
pub(crate) struct Paged<T> {
    /// The decoded body.
    pub body: T,
    /// The absolute URL of the next page, when the Hub advertised one.
    pub next: Option<String>,
}

impl HfClient {
    /// A client against the default Hub endpoint with an optional token.
    pub fn new(token: Option<String>) -> Self {
        Self::with_endpoint(DEFAULT_ENDPOINT, token)
    }

    /// A client against an explicit endpoint (used by tests to target a mock server).
    #[allow(clippy::disallowed_types)] // scoped straggler: fixed Hub endpoint (see struct)
    pub fn with_endpoint(endpoint: impl Into<String>, token: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            token,
        }
    }

    /// The configured endpoint (no trailing slash).
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// GET an endpoint-relative `path` with `query` params and decode the JSON body.
    pub(crate) async fn get_json<T: DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, String)],
    ) -> Result<T> {
        let url = format!("{}{}", self.endpoint, path);
        self.get_url(&url, query).await.map(|p| p.body)
    }

    /// GET an absolute `url` with `query` params, decoding the JSON body and parsing the `Link`
    /// header's `rel="next"` URL for cursor pagination.
    pub(crate) async fn get_url<T: DeserializeOwned>(
        &self,
        url: &str,
        query: &[(&str, String)],
    ) -> Result<Paged<T>> {
        let mut req = self.client.get(url);
        if !query.is_empty() {
            req = req.query(query);
        }
        if let Some(token) = &self.token {
            req = req.bearer_auth(token);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| ModelError::Http(e.to_string()))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let detail = format!("{status}: {}", body.chars().take(200).collect::<String>());
            return Err(match status.as_u16() {
                404 => ModelError::NotFound(detail),
                401 | 403 => ModelError::AccessDenied(detail),
                _ => ModelError::Http(detail),
            });
        }

        let next = resp
            .headers()
            .get(reqwest::header::LINK)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_next_link);
        let body = resp
            .text()
            .await
            .map_err(|e| ModelError::Http(e.to_string()))?;
        let decoded = serde_json::from_str::<T>(&body).map_err(|e| {
            ModelError::Decode(format!(
                "{e} (body: {})",
                body.chars().take(200).collect::<String>()
            ))
        })?;
        Ok(Paged {
            body: decoded,
            next,
        })
    }
}

/// Extract the `rel="next"` URL from an RFC-8288 `Link` header value.
fn parse_next_link(header: &str) -> Option<String> {
    for part in header.split(',') {
        let part = part.trim();
        if !part.contains("rel=\"next\"") {
            continue;
        }
        let start = part.find('<')?;
        let end = part.find('>')?;
        if end > start + 1 {
            return Some(part[start + 1..end].to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_next_link() {
        let h = "<https://huggingface.co/api/models/x/tree/main?cursor=abc>; rel=\"next\", <https://x>; rel=\"prev\"";
        assert_eq!(
            parse_next_link(h).as_deref(),
            Some("https://huggingface.co/api/models/x/tree/main?cursor=abc")
        );
        assert_eq!(parse_next_link("<https://x>; rel=\"prev\""), None);
    }
}
