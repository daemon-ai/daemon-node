//! `daemon-tool-web` — the `web_search` and `web_extract` chat tools (`daemon_core::Tool`s).
//!
//! Both are backed by pluggable traits ([`WebSearchBackend`], [`WebFetchBackend`]) so the concrete
//! provider is a wiring decision, not baked into the tool. The shipped backends are:
//!
//! - [`TavilySearch`] — hosted LLM-oriented search (`web_search`).
//! - [`FirecrawlFetch`] — hosted JS-rendering scraper -> Markdown (`web_extract`, keyed).
//! - [`LocalFetch`] — dependency-light `reqwest` + `dom_smoothie` readability extraction
//!   (`web_extract` fallback, key-less).
//!
//! API keys are read live from a host-provided [`SecretSource`] (the host adapts its
//! `CredentialStore`), so a GUI-set key applies without a restart. All returned page/search content
//! is marked **untrusted** so the §12 pipeline fences it before it reaches the model.

#![forbid(unsafe_code)]

mod backend;
mod extract_tool;
mod firecrawl;
mod local;
mod search_tool;
mod tavily;

pub use backend::{
    FetchFormat, FetchOpts, FetchedDoc, NoSecrets, SearchHit, SearchOpts, SearchResults,
    SearchTopic, SecretSource, WebError, WebFetchBackend, WebSearchBackend,
};
pub use extract_tool::WebExtractTool;
pub use firecrawl::{FirecrawlFetch, FIRECRAWL_ENDPOINT};
pub use local::LocalFetch;
pub use search_tool::WebSearchTool;
pub use tavily::{TavilySearch, TAVILY_ENDPOINT};
