//! Pluggable backend traits + shared types for the web tools.
//!
//! `web_search` resolves a query against a [`WebSearchBackend`] (hosted Tavily today; Exa et al.
//! drop in behind the same trait). `web_extract` fetches + cleans a page through one or more
//! [`WebFetchBackend`]s (a keyed hosted scraper like Firecrawl, falling back to the dependency-light
//! local reqwest+readability path). Credentials are read live from a host-provided [`SecretSource`]
//! so a GUI-set key applies without a restart.

use async_trait::async_trait;
use serde::Serialize;

/// A read-only secret provider keyed by credential-profile id (`"tavily"`, `"firecrawl"`, ...). The
/// host adapts its `CredentialStore` to this so the heavy substrate type never enters the tool crate
/// (mirrors the `daemon-tool-metta` `SemanticIndex` host-trait seam). Reads happen at call time, so a
/// key set later via `CredentialApi` takes effect immediately.
pub trait SecretSource: Send + Sync {
    /// The secret stored under `key`, if any.
    fn secret(&self, key: &str) -> Option<String>;
}

/// A [`SecretSource`] that never has any secret (key-less backends only).
pub struct NoSecrets;

impl SecretSource for NoSecrets {
    fn secret(&self, _key: &str) -> Option<String> {
        None
    }
}

/// What went wrong running a web backend.
#[derive(Debug, thiserror::Error)]
pub enum WebError {
    /// No credential is configured for the backend's key id (the tool reports how to set it).
    #[error("no credential configured for '{0}'")]
    MissingKey(String),
    /// The URL failed the egress safety policy.
    #[error("url rejected: {0}")]
    Rejected(String),
    /// A transport/HTTP-status failure talking to the backend.
    #[error("http error: {0}")]
    Http(String),
    /// The backend's response could not be decoded.
    #[error("decode error: {0}")]
    Decode(String),
    /// The backend returned no usable content.
    #[error("no content extracted")]
    Empty,
}

/// The topic hint for a search (maps to provider-specific knobs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SearchTopic {
    /// General web search.
    #[default]
    General,
    /// News-focused search.
    News,
}

/// Options carried from a `web_search` call into a [`WebSearchBackend`].
#[derive(Clone, Debug)]
pub struct SearchOpts {
    /// Maximum number of hits to return.
    pub max_results: u32,
    /// The topic hint.
    pub topic: SearchTopic,
}

/// One search hit.
#[derive(Clone, Debug, Serialize)]
pub struct SearchHit {
    /// The result title.
    pub title: String,
    /// The result URL.
    pub url: String,
    /// A query-relevant snippet/summary.
    pub snippet: String,
    /// The backend's relevance score (0-1), if provided.
    pub score: Option<f64>,
}

/// The result of a web search.
#[derive(Clone, Debug, Serialize)]
pub struct SearchResults {
    /// The (echoed) query.
    pub query: String,
    /// An optional LLM-generated direct answer (Tavily `include_answer`).
    pub answer: Option<String>,
    /// The ranked hits.
    pub hits: Vec<SearchHit>,
    /// The backend that served the search.
    pub provider: String,
}

/// The desired output format for an extraction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FetchFormat {
    /// Structured Markdown (the default).
    #[default]
    Markdown,
    /// Plain text.
    Text,
}

/// Options carried from a `web_extract` call into a [`WebFetchBackend`].
#[derive(Clone, Debug, Default)]
pub struct FetchOpts {
    /// The output format.
    pub format: FetchFormat,
}

/// A fetched + extracted document.
#[derive(Clone, Debug, Serialize)]
pub struct FetchedDoc {
    /// The source URL.
    pub url: String,
    /// The extracted title, if any.
    pub title: Option<String>,
    /// The extracted main content (Markdown or text per [`FetchOpts`]).
    pub content: String,
    /// The backend that served the fetch.
    pub provider: String,
}

/// A search provider (Tavily, Exa, ...).
#[async_trait]
pub trait WebSearchBackend: Send + Sync {
    /// The backend's stable name (for diagnostics + result provenance).
    fn name(&self) -> &str;
    /// Run a search.
    async fn search(&self, query: &str, opts: &SearchOpts) -> Result<SearchResults, WebError>;
}

/// A page-fetch + content-extraction provider (Firecrawl, local reqwest+readability, ...).
#[async_trait]
pub trait WebFetchBackend: Send + Sync {
    /// The backend's stable name (for diagnostics + result provenance).
    fn name(&self) -> &str;
    /// Whether this backend is currently usable (e.g. has its API key). The extract tool skips
    /// unavailable backends and falls through to the next in its chain.
    fn available(&self) -> bool {
        true
    }
    /// Fetch + extract `url`.
    async fn fetch(&self, url: &str, opts: &FetchOpts) -> Result<FetchedDoc, WebError>;
}
