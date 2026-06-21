//! Per-repo file listing over the Hub `/api/models/{repo}/tree/{revision}` endpoint (step 2).
//!
//! Recursively pages the repo tree (following the `Link: rel="next"` cursor), then filters + labels
//! the artifacts the way the old `HFModelFilesVM` did: for llama, only `.gguf` files (with quant
//! label + split-shard detection); for mistral.rs, the repo's loadable siblings (config, tokenizer,
//! weights — `safetensors`/`uqff`/`gguf`).

use daemon_common::{ModelEngine, ModelFile};
use serde::Deserialize;

use crate::error::{ModelError, Result};
use crate::gguf;
use crate::hf::client::HfClient;

/// One node of the repo tree.
#[derive(Debug, Deserialize)]
struct RawTreeItem {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<RawLfs>,
}

#[derive(Debug, Deserialize)]
struct RawLfs {
    #[serde(default)]
    size: u64,
}

/// The maximum number of tree pages to follow (guards against a pathological repo).
const MAX_TREE_PAGES: usize = 50;

/// List the loadable files of `repo` at `revision` for `engine`.
pub async fn list_files(
    client: &HfClient,
    repo: &str,
    revision: &str,
    engine: ModelEngine,
) -> Result<Vec<ModelFile>> {
    if repo.trim().is_empty() {
        return Err(ModelError::Invalid("empty repo id".into()));
    }
    let items = fetch_tree(client, repo, revision).await?;
    let mut files: Vec<ModelFile> = items
        .into_iter()
        .filter(|it| it.kind == "file")
        .filter(|it| keep_for_engine(&it.path, engine))
        .map(|it| to_model_file(it, engine))
        .collect();
    // Stable, useful ordering: GGUF/weight files first, then by path.
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(files)
}

/// Every file in `repo` at `revision` as `(path, size_bytes)` — no engine filter. Used by the
/// mistral.rs acquisition planner, which warms the whole repo so the engine loads offline.
pub async fn list_all(client: &HfClient, repo: &str, revision: &str) -> Result<Vec<(String, u64)>> {
    if repo.trim().is_empty() {
        return Err(ModelError::Invalid("empty repo id".into()));
    }
    let items = fetch_tree(client, repo, revision).await?;
    Ok(items
        .into_iter()
        .filter(|it| it.kind == "file")
        .map(|it| {
            let size = if it.size > 0 {
                it.size
            } else {
                it.lfs.map(|l| l.size).unwrap_or(0)
            };
            (it.path, size)
        })
        .collect())
}

/// Detailed repo info: only the safetensors parameter total we use to size a mistral.rs ISQ
/// recommendation.
#[derive(Debug, Deserialize)]
struct RawRepoInfo {
    #[serde(default)]
    safetensors: Option<RawSafetensors>,
}

#[derive(Debug, Deserialize)]
struct RawSafetensors {
    #[serde(default)]
    total: Option<u64>,
}

/// The repo's total parameter count, when the Hub reports it (`safetensors.total`). Best-effort:
/// returns `None` on any lookup/parse miss rather than failing the caller.
pub async fn repo_param_count(client: &HfClient, repo: &str, revision: &str) -> Option<u64> {
    let path = format!("/api/models/{repo}/revision/{revision}");
    let query = [("expand", "safetensors".to_string())];
    let info: RawRepoInfo = client.get_json(&path, &query).await.ok()?;
    info.safetensors.and_then(|s| s.total)
}

/// Fetch (and follow pagination across) the full recursive tree.
async fn fetch_tree(client: &HfClient, repo: &str, revision: &str) -> Result<Vec<RawTreeItem>> {
    let first = format!(
        "{}/api/models/{repo}/tree/{revision}",
        client.endpoint()
    );
    let mut url = first;
    let mut query: Vec<(&str, String)> = vec![("recursive", "true".to_string())];
    let mut all = Vec::new();
    for _ in 0..MAX_TREE_PAGES {
        let page = client
            .get_url::<Vec<RawTreeItem>>(&url, &query)
            .await?;
        all.extend(page.body);
        match page.next {
            // The `next` URL already carries the cursor; don't re-append query params.
            Some(next) => {
                url = next;
                query = Vec::new();
            }
            None => break,
        }
    }
    Ok(all)
}

/// Whether a repo file is loadable by `engine`.
fn keep_for_engine(path: &str, engine: ModelEngine) -> bool {
    let lower = path.to_ascii_lowercase();
    match engine {
        ModelEngine::Llama => lower.ends_with(".gguf"),
        ModelEngine::MistralRs => {
            lower.ends_with(".safetensors")
                || lower.ends_with(".uqff")
                || lower.ends_with(".gguf")
                || lower.ends_with("config.json")
                || lower.contains("tokenizer")
        }
    }
}

fn to_model_file(item: RawTreeItem, _engine: ModelEngine) -> ModelFile {
    let size = if item.size > 0 {
        item.size
    } else {
        item.lfs.map(|l| l.size).unwrap_or(0)
    };
    let is_split = gguf::shard_spec(&item.path).is_some();
    ModelFile {
        quant: gguf::quant_label(&item.path),
        is_first_shard: gguf::is_first_shard(&item.path),
        is_split,
        path: item.path,
        size_bytes: size,
    }
}
