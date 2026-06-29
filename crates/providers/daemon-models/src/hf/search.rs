// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Repo search over the Hub `/api/models` endpoint (step 1 of search→select→download).
//!
//! An idiomatic port of the old `HuggingFaceService` search: free-text query, engine-aware format
//! filter (`gguf` for llama), Hub `sort`, and page slicing. Returns a [`SearchPage`] of repos; the
//! client then lists a chosen repo's files via [`super::files`].

use daemon_common::{ModelEngine, SearchHit, SearchPage, SearchQuery};
use serde::Deserialize;

use crate::error::Result;
use crate::hf::client::HfClient;

/// One entry of the `/api/models` listing (only the fields we surface).
#[derive(Debug, Deserialize)]
struct RawModel {
    #[serde(default)]
    id: String,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default, rename = "pipeline_tag")]
    pipeline_tag: Option<String>,
    #[serde(default, rename = "lastModified")]
    last_modified: Option<String>,
    /// The Hub returns `false` or a string (`"auto"`/`"manual"`) here.
    #[serde(default)]
    gated: serde_json::Value,
    #[serde(default)]
    private: bool,
    #[serde(default)]
    safetensors: Option<SafeTensors>,
}

#[derive(Debug, Deserialize)]
struct SafeTensors {
    #[serde(default)]
    total: Option<u64>,
}

/// Run a repo search. The Hub listing endpoint has no offset param, so we over-fetch
/// `limit * (page + 1)` and slice the requested page — adequate for interactive browsing and
/// trivially mockable in tests.
pub async fn search(client: &HfClient, query: &SearchQuery) -> Result<SearchPage> {
    let limit = query.limit.max(1);
    let effective = (limit as u64 * (query.page as u64 + 1))
        .min(1000)
        .to_string();
    let mut params: Vec<(&str, String)> = vec![
        ("search", query.text.clone()),
        ("sort", query.sort.as_query().to_string()),
        ("direction", "-1".to_string()),
        ("limit", effective),
        ("full", "false".to_string()),
        ("config", "false".to_string()),
    ];
    // Engine-aware format filter: llama needs a GGUF in the repo.
    if matches!(query.engine, ModelEngine::Llama) {
        params.push(("filter", "gguf".to_string()));
    }

    let raw: Vec<RawModel> = client.get_json("/api/models", &params).await?;
    let fetched = raw.len();

    let start = (query.page as usize) * (limit as usize);
    let results: Vec<SearchHit> = raw
        .into_iter()
        .skip(start)
        .take(limit as usize)
        .map(to_hit)
        .collect();

    Ok(SearchPage {
        page: query.page,
        results,
        // Another page is plausible iff the upstream returned a full over-fetch.
        has_more: fetched as u64 >= limit as u64 * (query.page as u64 + 1),
    })
}

fn to_hit(m: RawModel) -> SearchHit {
    let author = m
        .author
        .or_else(|| m.id.split_once('/').map(|(a, _)| a.to_string()));
    let gated = match &m.gated {
        serde_json::Value::Bool(b) => *b,
        serde_json::Value::String(s) => !s.eq_ignore_ascii_case("false"),
        _ => false,
    };
    SearchHit {
        repo: m.id,
        author,
        downloads: m.downloads,
        likes: m.likes,
        num_parameters: m.safetensors.and_then(|s| s.total),
        pipeline_tag: m.pipeline_tag,
        last_modified: m.last_modified,
        gated,
        private: m.private,
    }
}
