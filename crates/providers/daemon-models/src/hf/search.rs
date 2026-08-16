// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Repo search (step 1 of search→select→download), wire v48.
//!
//! Primary endpoint: the Hub's website listing `GET /models-json` — the only endpoint that serves
//! the rich filters (`library=gguf`, `pipeline_tag`, `num_parameters` buckets, a total result
//! count via `withCount=true`, and page-number pagination). Fallback: the public `/api/models`
//! listing, **bounded** to genuine primary-endpoint breakage (a 5xx or response drift) — a 4xx
//! from a bad query surfaces as an error, never as silently-unfiltered results. A fallback page is
//! marked `degraded` (no total, no parameter filter).

use daemon_common::{ModelEngine, SearchHit, SearchPage, SearchQuery};
use serde::Deserialize;

use crate::error::Result;
use crate::hf::client::{FetchError, HfClient};

/// The Hub `num_parameters` bucket tokens the `models-json` endpoint accepts, as (parameter count,
/// token). The wire carries numeric bounds; the node maps them onto the nearest bucket that does
/// not exclude the requested range (min rounds down, max rounds up).
const PARAM_BUCKETS: &[(u64, &str)] = &[
    (0, "0"),
    (3_000_000_000, "3B"),
    (7_000_000_000, "7B"),
    (13_000_000_000, "13B"),
    (14_000_000_000, "14B"),
    (32_000_000_000, "32B"),
    (65_000_000_000, "65B"),
    (128_000_000_000, "128B"),
    (256_000_000_000, "256B"),
];

/// Encode numeric parameter bounds as the Hub's `min:X[,max:Y]` filter value. `None` when the
/// query carries no bounds (or the min rounds to 0 with an unbounded max — the "any size" case).
fn encode_params_filter(params_min: Option<u64>, params_max: Option<u64>) -> Option<String> {
    if params_min.is_none() && params_max.is_none() {
        return None;
    }
    // Min: the largest bucket at or below the requested minimum (never exclude requested models).
    let min_token = PARAM_BUCKETS
        .iter()
        .rev()
        .find(|(count, _)| *count <= params_min.unwrap_or(0))
        .map(|(_, token)| *token)
        .unwrap_or("0");
    // Max: the smallest bucket at or above the requested maximum; above the top bucket = unbounded.
    let max_token = params_max.and_then(|max| {
        PARAM_BUCKETS
            .iter()
            .find(|(count, _)| *count >= max)
            .map(|(_, token)| *token)
    });
    match (min_token, max_token) {
        ("0", None) => None,
        (min, None) => Some(format!("min:{min}")),
        (min, Some(max)) => Some(format!("min:{min},max:{max}")),
    }
}

/// One row of the `models-json` listing (tolerant: only the fields we surface, all defaulted).
#[derive(Debug, Deserialize)]
struct JsonRow {
    #[serde(default)]
    id: String,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default, rename = "numParameters")]
    num_parameters: Option<serde_json::Value>,
    #[serde(default, rename = "pipeline_tag")]
    pipeline_tag: Option<String>,
    #[serde(default, rename = "lastModified")]
    last_modified: Option<String>,
    /// The Hub returns `false` or a string (`"auto"`/`"manual"`) here.
    #[serde(default)]
    gated: serde_json::Value,
    #[serde(default)]
    private: bool,
}

/// The `models-json` response envelope. `models` is REQUIRED (no serde default): a payload
/// without it is response drift and must trip the classified decode error (→ bounded fallback),
/// not silently decode as an empty result set.
#[derive(Debug, Deserialize)]
struct JsonListing {
    #[serde(default, rename = "numTotalItems")]
    num_total_items: Option<u64>,
    models: Vec<JsonRow>,
}

/// One entry of the `/api/models` fallback listing (only the fields we surface).
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

/// The repo-format filter for an engine: llama needs a GGUF in the repo; mistral.rs loads
/// safetensors repos (ISQ from full precision).
fn library_for(engine: ModelEngine) -> &'static str {
    match engine {
        ModelEngine::Llama => "gguf",
        ModelEngine::MistralRs => "safetensors",
    }
}

/// Run a repo search for `engine` (resolved from the query's provider by the caller). Primary
/// `models-json`, bounded `/api/models` fallback on 5xx/decode drift only.
pub async fn search(
    client: &HfClient,
    engine: ModelEngine,
    query: &SearchQuery,
) -> Result<SearchPage> {
    match search_models_json(client, engine, query).await {
        Ok(page) => Ok(page),
        // Bounded fallback: the primary endpoint itself broke (server error or shape drift).
        Err(FetchError::Status { code, .. }) if code >= 500 => {
            search_api_models(client, engine, query).await
        }
        Err(FetchError::Decode(_)) => search_api_models(client, engine, query).await,
        // A 4xx (bad query) or a network failure is a real error — never silently-unfiltered
        // results from the fallback path.
        Err(e) => Err(e.into_model_error()),
    }
}

/// The primary `models-json` search.
async fn search_models_json(
    client: &HfClient,
    engine: ModelEngine,
    query: &SearchQuery,
) -> std::result::Result<SearchPage, FetchError> {
    let mut params: Vec<(&str, String)> = vec![
        ("library", library_for(engine).to_string()),
        ("pipeline_tag", "text-generation".to_string()),
        ("sort", query.sort.as_models_json_query().to_string()),
        ("p", query.page.to_string()),
        ("withCount", "true".to_string()),
    ];
    if !query.text.trim().is_empty() {
        params.push(("search", query.text.trim().to_string()));
    }
    let params_filter = encode_params_filter(query.params_min, query.params_max);
    let params_filter_applied = params_filter.is_some();
    if let Some(filter) = params_filter {
        params.push(("num_parameters", filter));
    }

    let listing: JsonListing = client.get_json_classified("/models-json", &params).await?;
    let total = listing.num_total_items;
    let page_len = listing.models.len() as u64;
    let results: Vec<SearchHit> = listing
        .models
        .into_iter()
        .map(|m| json_row_to_hit(client, m))
        .collect();

    // Another page is plausible while the cumulative lower bound stays under the total. On a
    // short final page this can over-report by one page; the client's next fetch then comes back
    // empty and terminates cleanly (a benign extra roundtrip, not duplicate rows).
    let has_more =
        page_len > 0 && total.is_none_or(|t| (query.page as u64 + 1).saturating_mul(page_len) < t);

    Ok(SearchPage {
        page: query.page,
        results,
        has_more,
        total,
        params_filter_applied,
        degraded: false,
        quant_filter_applied: false, // the manager's enrichment pass sets it
    })
}

/// The bounded `/api/models` fallback: page slicing over an over-fetch (the endpoint has no
/// offset param), no total, no parameter filter — the page is marked `degraded`.
async fn search_api_models(
    client: &HfClient,
    engine: ModelEngine,
    query: &SearchQuery,
) -> Result<SearchPage> {
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
    if matches!(engine, ModelEngine::Llama) {
        params.push(("filter", "gguf".to_string()));
    }

    let raw: Vec<RawModel> = client.get_json("/api/models", &params).await?;
    let fetched = raw.len();

    let start = (query.page as usize) * (limit as usize);
    let results: Vec<SearchHit> = raw
        .into_iter()
        .skip(start)
        .take(limit as usize)
        .map(|m| raw_to_hit(client, m))
        .collect();

    Ok(SearchPage {
        page: query.page,
        results,
        // Another page is plausible iff the upstream returned a full over-fetch.
        has_more: fetched as u64 >= limit as u64 * (query.page as u64 + 1),
        total: None,
        params_filter_applied: false,
        degraded: true,
        quant_filter_applied: false, // the manager's enrichment pass sets it
    })
}

/// The node-supplied canonical web page for a repo (clients never string-build Hub links).
fn web_url(client: &HfClient, repo: &str) -> Option<String> {
    if repo.is_empty() {
        return None;
    }
    Some(format!("{}/{}", client.endpoint(), repo))
}

fn json_row_to_hit(client: &HfClient, m: JsonRow) -> SearchHit {
    let author = m
        .author
        .or_else(|| m.id.split_once('/').map(|(a, _)| a.to_string()));
    // The endpoint has served this as an integer or a float; coerce either.
    let num_parameters = m.num_parameters.and_then(|v| {
        v.as_u64()
            .or_else(|| v.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64))
    });
    SearchHit {
        web_url: web_url(client, &m.id),
        author,
        downloads: m.downloads,
        likes: m.likes,
        num_parameters,
        pipeline_tag: m.pipeline_tag,
        last_modified: m.last_modified,
        gated: gated_flag(&m.gated),
        private: m.private,
        repo: m.id,
        quants: None, // enriched by the manager (cache-first / filter fetch)
    }
}

fn raw_to_hit(client: &HfClient, m: RawModel) -> SearchHit {
    let author = m
        .author
        .or_else(|| m.id.split_once('/').map(|(a, _)| a.to_string()));
    SearchHit {
        web_url: web_url(client, &m.id),
        author,
        downloads: m.downloads,
        likes: m.likes,
        num_parameters: m.safetensors.and_then(|s| s.total),
        pipeline_tag: m.pipeline_tag,
        last_modified: m.last_modified,
        gated: gated_flag(&m.gated),
        private: m.private,
        repo: m.id,
        quants: None, // enriched by the manager (cache-first / filter fetch)
    }
}

/// The Hub serves `gated` as `false` or a string (`"auto"`/`"manual"`).
fn gated_flag(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Bool(b) => *b,
        serde_json::Value::String(s) => !s.eq_ignore_ascii_case("false"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Numeric bounds map onto the nearest non-excluding bucket tokens: min rounds down, max
    /// rounds up, "any size" and above-top-bucket maxima drop their side of the filter.
    #[test]
    fn params_filter_encoding_rounds_outward() {
        // No bounds → no filter.
        assert_eq!(encode_params_filter(None, None), None);
        // Exact buckets pass through.
        assert_eq!(
            encode_params_filter(Some(7_000_000_000), Some(32_000_000_000)).as_deref(),
            Some("min:7B,max:32B")
        );
        // Min rounds DOWN (8B → 7B), max rounds UP (30B → 32B): never exclude requested models.
        assert_eq!(
            encode_params_filter(Some(8_000_000_000), Some(30_000_000_000)).as_deref(),
            Some("min:7B,max:32B")
        );
        // A min below the first non-zero bucket rounds to 0 → with no max that is "any size".
        assert_eq!(encode_params_filter(Some(1_000_000_000), None), None);
        // A max above the top bucket is unbounded (omit max).
        assert_eq!(
            encode_params_filter(Some(65_000_000_000), Some(999_000_000_000)).as_deref(),
            Some("min:65B")
        );
        // Max-only filters keep min at 0.
        assert_eq!(
            encode_params_filter(None, Some(3_000_000_000)).as_deref(),
            Some("min:0,max:3B")
        );
    }
}
