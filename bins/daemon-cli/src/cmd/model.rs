// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `model` subcommand: hub search/files, downloads, catalog, recommend/quantize/inspect, and
//! the `up` quickstart (recommend → pull → activate orchestrated client-side).

use daemon_api::{ApiRequest, ApiResponse};
use daemon_common::{
    DownloadId, ModelEngine, ModelId, ModelRef, ModelSource, SearchQuery, SearchSort,
};
use daemon_host::ApiClient;

use crate::cli::ModelCmd;
use crate::render::render;

/// Parse an engine selector, defaulting to llama.
fn parse_engine(s: &str) -> anyhow::Result<ModelEngine> {
    ModelEngine::parse(s)
        .ok_or_else(|| anyhow::anyhow!("unknown engine {s:?} (expected llama|mistralrs)"))
}

/// Build a [`ModelRef`] from CLI repo/file/revision/engine args.
fn build_model_ref(
    engine: ModelEngine,
    repo: String,
    file: Option<String>,
    revision: Option<String>,
) -> ModelRef {
    let revision = revision.unwrap_or_else(|| "main".to_string());
    ModelRef::new(
        engine,
        ModelSource::Hf {
            repo,
            file,
            revision,
        },
    )
}

/// GiB → bytes.
fn gib_to_bytes(gib: f64) -> u64 {
    (gib * 1024.0 * 1024.0 * 1024.0) as u64
}

/// Dispatch a `model` subcommand against the node.
pub(super) async fn run(client: &ApiClient, cmd: ModelCmd) -> anyhow::Result<()> {
    // The quickstart orchestrates several calls (recommend → pull → activate), so it is handled
    // out-of-band rather than as a single request mapping.
    if let ModelCmd::Up {
        repo,
        engine,
        vram,
        profile,
    } = cmd
    {
        return quickstart_up(client, repo, parse_engine(&engine)?, vram, profile).await;
    }

    let req = match cmd {
        ModelCmd::Search {
            query,
            engine,
            limit,
            page,
        } => ApiRequest::ModelSearch {
            query: SearchQuery {
                text: query,
                engine: parse_engine(&engine)?,
                sort: SearchSort::default(),
                page,
                limit,
            },
        },
        ModelCmd::Files {
            repo,
            revision,
            engine,
        } => ApiRequest::ModelFiles {
            repo,
            revision,
            engine: parse_engine(&engine)?,
            after: None,
        },
        ModelCmd::Pull {
            repo,
            file,
            revision,
            engine,
        } => ApiRequest::ModelDownload {
            model: build_model_ref(parse_engine(&engine)?, repo, file, revision),
        },
        ModelCmd::Downloads => ApiRequest::ModelDownloads,
        ModelCmd::Cancel { id } => ApiRequest::ModelCancel { id: DownloadId(id) },
        ModelCmd::Pause { id } => ApiRequest::ModelPause { id: DownloadId(id) },
        ModelCmd::Resume { id } => ApiRequest::ModelResume { id: DownloadId(id) },
        ModelCmd::Ls => ApiRequest::ModelCatalog,
        ModelCmd::Rm { id } => ApiRequest::ModelDelete {
            id: ModelId::new(id),
        },
        ModelCmd::Activate { id, profile } => ApiRequest::ModelActivate {
            id: ModelId::new(id),
            profile,
        },
        ModelCmd::Recommend {
            repo,
            engine,
            revision,
            vram,
        } => ApiRequest::ModelRecommend(daemon_api::ModelRecommendArgs {
            repo,
            revision,
            engine: parse_engine(&engine)?,
            budget_bytes: vram.map(gib_to_bytes),
        }),
        ModelCmd::Quantize {
            repo,
            ftype,
            source,
            revision,
        } => ApiRequest::ModelQuantize(daemon_api::ModelQuantizeArgs {
            repo,
            revision,
            target_quant: ftype,
            source_file: source,
        }),
        ModelCmd::Quantizes => ApiRequest::ModelQuantizes,
        ModelCmd::Inspect { id } => ApiRequest::ModelInspect {
            id: ModelId::new(id),
        },
        ModelCmd::Up { .. } => unreachable!("handled above"),
    };
    render(client.call(req).await?);
    Ok(())
}

/// The `model up` quickstart: recommend a quant for the hardware, download it, then activate it.
async fn quickstart_up(
    client: &ApiClient,
    repo: String,
    engine: ModelEngine,
    vram: Option<f64>,
    profile: Option<String>,
) -> anyhow::Result<()> {
    // 1. Recommend a quant for the detected (or overridden) budget.
    let rec = match client
        .call(ApiRequest::ModelRecommend(daemon_api::ModelRecommendArgs {
            repo: repo.clone(),
            revision: None,
            engine,
            budget_bytes: vram.map(gib_to_bytes),
        }))
        .await?
    {
        ApiResponse::ModelRecommend(rec) => rec,
        ApiResponse::Error(e) => anyhow::bail!("recommend failed: {e}"),
        other => anyhow::bail!("unexpected response to recommend: {other:?}"),
    };
    println!(
        "recommend: {} {} (fits={}) — {}",
        rec.repo, rec.quant, rec.fits, rec.reason
    );

    // 2. Build the model ref and start the download (llama needs the recommended file).
    let model = ModelRef::new(
        engine,
        ModelSource::Hf {
            repo: repo.clone(),
            file: rec.file.clone(),
            revision: "main".to_string(),
        },
    );
    if matches!(engine, ModelEngine::Llama) && rec.file.is_none() {
        anyhow::bail!(
            "no downloadable GGUF recommended for {repo}: {}",
            rec.reason
        );
    }
    let job = match client
        .call(ApiRequest::ModelDownload {
            model: model.clone(),
        })
        .await?
    {
        ApiResponse::ModelDownloadStarted(id) => id,
        ApiResponse::Error(e) => anyhow::bail!("download failed to start: {e}"),
        other => anyhow::bail!("unexpected response to download: {other:?}"),
    };
    println!("download: started {job}; waiting for completion…");

    // 3. Poll until the job reaches a terminal state.
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(700)).await;
        let jobs = match client.call(ApiRequest::ModelDownloads).await? {
            ApiResponse::ModelDownloads(jobs) => jobs,
            other => anyhow::bail!("unexpected response to downloads: {other:?}"),
        };
        let Some(status) = jobs.into_iter().find(|j| j.id == job) else {
            continue;
        };
        use daemon_common::DownloadState::*;
        match status.state {
            Completed => {
                println!("download: complete");
                break;
            }
            Failed | Cancelled => {
                anyhow::bail!(
                    "download {job} ended {:?}: {}",
                    status.state,
                    status.error.unwrap_or_default()
                );
            }
            _ => {
                let pct = if status.total_bytes > 0 {
                    (status.downloaded_bytes as f64 / status.total_bytes as f64 * 100.0) as u64
                } else {
                    0
                };
                println!(
                    "  … {pct}% ({}/{} bytes)",
                    status.downloaded_bytes, status.total_bytes
                );
            }
        }
    }

    // 4. Find the cataloged record for this model and activate it.
    let catalog = match client.call(ApiRequest::ModelCatalog).await? {
        ApiResponse::ModelCatalog(models) => models,
        other => anyhow::bail!("unexpected response to catalog: {other:?}"),
    };
    let record = catalog
        .into_iter()
        .find(|m| m.model == model)
        .ok_or_else(|| anyhow::anyhow!("downloaded model not found in catalog"))?;
    println!("activate: {} ({})", record.id, record.display_name);
    render(
        client
            .call(ApiRequest::ModelActivate {
                id: record.id,
                profile,
            })
            .await?,
    );
    Ok(())
}
