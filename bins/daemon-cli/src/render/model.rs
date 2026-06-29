// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Model-surface responses: hub search/files, downloads, installed catalog, recommendations,
//! quantizations, gguf inspection, and the runtime model roster/current selection.

use daemon_api::ApiResponse;

pub(super) fn try_render(resp: ApiResponse) -> Option<ApiResponse> {
    match resp {
        ApiResponse::ModelSearch(page) => {
            println!(
                "search: page={} results={} has_more={}",
                page.page,
                page.results.len(),
                page.has_more
            );
            for h in page.results {
                let gated = if h.gated { " [gated]" } else { "" };
                println!(
                    "  - {} downloads={} likes={} {}{}",
                    h.repo,
                    h.downloads,
                    h.likes,
                    h.pipeline_tag.as_deref().unwrap_or("-"),
                    gated
                );
            }
        }
        ApiResponse::ModelFiles(files) => {
            println!("files: {}", files.len());
            for f in files {
                let quant = f.quant.map(|q| format!(" quant={q}")).unwrap_or_default();
                let split = if f.is_split { " split" } else { "" };
                println!("  - {} ({} bytes){}{}", f.path, f.size_bytes, quant, split);
            }
        }
        ApiResponse::ModelDownloadStarted(id) => println!("download started: {id}"),
        ApiResponse::ModelDownloads(jobs) => {
            println!("downloads: {}", jobs.len());
            for j in jobs {
                let pct = if j.total_bytes > 0 {
                    (j.downloaded_bytes as f64 / j.total_bytes as f64 * 100.0) as u64
                } else {
                    0
                };
                let err = j.error.map(|e| format!(" error={e}")).unwrap_or_default();
                println!(
                    "  - {} {:?} {}% ({}/{} bytes, files {}/{}){}",
                    j.id,
                    j.state,
                    pct,
                    j.downloaded_bytes,
                    j.total_bytes,
                    j.files_done,
                    j.files_total,
                    err
                );
            }
        }
        ApiResponse::ModelCatalog(models) => {
            println!("installed: {}", models.len());
            for m in models {
                let quant = m.quant.map(|q| format!(" quant={q}")).unwrap_or_default();
                let arch = m.arch.map(|a| format!(" arch={a}")).unwrap_or_default();
                let ctx = m
                    .context_length
                    .map(|c| format!(" ctx={c}"))
                    .unwrap_or_default();
                println!(
                    "  - {} {} ({} bytes){}{}{} -> {}",
                    m.id,
                    m.display_name,
                    m.size_bytes,
                    quant,
                    arch,
                    ctx,
                    m.local_path.display()
                );
            }
        }
        ApiResponse::ModelRecommend(rec) => {
            println!(
                "recommend: {} engine={} quant={} fits={} budget={} bytes",
                rec.repo, rec.engine, rec.quant, rec.fits, rec.budget_bytes
            );
            if let Some(f) = &rec.file {
                println!("  file: {f}");
            }
            if let Some(s) = rec.size_bytes {
                println!("  size: {s} bytes");
            }
            println!("  reason: {}", rec.reason);
            for c in rec.candidates {
                let size = c
                    .size_bytes
                    .map(|s| format!("{s} bytes"))
                    .unwrap_or_else(|| "?".into());
                let file = c.file.map(|f| format!(" {f}")).unwrap_or_default();
                println!("    - {} ({}) fits={}{}", c.quant, size, c.fits, file);
            }
        }
        ApiResponse::ModelQuantizeStarted(id) => println!("quantize started: {id}"),
        ApiResponse::ModelQuantizes(jobs) => {
            println!("quantizations: {}", jobs.len());
            for j in jobs {
                let out = j
                    .output_path
                    .map(|p| format!(" -> {}", p.display()))
                    .unwrap_or_default();
                let err = j.error.map(|e| format!(" error={e}")).unwrap_or_default();
                println!(
                    "  - {} {} -> {} {:?}{}{}",
                    j.id, j.source_file, j.target_quant, j.state, out, err
                );
            }
        }
        ApiResponse::ModelInspect(info) => {
            println!("inspect:");
            println!(
                "  architecture: {}",
                info.architecture.as_deref().unwrap_or("-")
            );
            println!("  name: {}", info.name.as_deref().unwrap_or("-"));
            println!("  file_type: {}", info.file_type.as_deref().unwrap_or("-"));
            println!(
                "  context_length: {}",
                info.context_length
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "-".into())
            );
            println!(
                "  block_count: {}",
                info.block_count
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "-".into())
            );
            println!(
                "  parameters: {}",
                info.parameter_count
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "-".into())
            );
            println!("  size_bytes: {}", info.size_bytes);
        }
        ApiResponse::Models(models) => {
            println!("models: {}", models.len());
            for m in models {
                let ctx = m
                    .context_length
                    .map(|c| format!(" ctx={c}"))
                    .unwrap_or_default();
                let kind = if m.local { " [local]" } else { "" };
                println!("  - {} [{:?}]{}{}", m.id, m.provider, ctx, kind);
            }
        }
        ApiResponse::ModelCurrent(model) => match model {
            Some(m) => {
                let ctx = m
                    .context_length
                    .map(|c| format!(" ctx={c}"))
                    .unwrap_or_default();
                println!("current model: {} [{:?}]{}", m.id, m.provider, ctx);
            }
            None => println!("current model: none"),
        },
        other => return Some(other),
    }
    None
}
