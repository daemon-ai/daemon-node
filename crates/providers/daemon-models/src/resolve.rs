// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Turning a [`ModelRef`] into a concrete download plan: which repo files to fetch for each engine.
//!
//! - **llama**: a single named GGUF file (expanding to the full shard set when the named file is the
//!   first shard of a split model), plus — for *text* models — the repo's best-matching
//!   vision-projector (mmproj) companion appended to the same job when one scores above the
//!   pairing threshold. One artifact, `single_file = true`.
//! - **mistral.rs**: every file in the repo (config + tokenizer + weights), so the engine loads from
//!   the warmed cache offline. The artifact is the snapshot directory, `single_file = false`.

use daemon_common::{ModelEngine, ModelRef, ModelSource};

use crate::acquire::{plan_for, DownloadPlan, PlanFile};
use crate::error::{ModelError, Result};
use crate::hf::{files, HfClient};
use crate::{gguf, mmproj};

/// Build the download plan for a Hugging Face model reference (lists the repo to resolve sizes +
/// expand shard sets).
pub async fn plan(client: &HfClient, model: &ModelRef) -> Result<DownloadPlan> {
    let (repo, file, revision) = match &model.source {
        ModelSource::Hf {
            repo,
            file,
            revision,
        } => (repo, file, revision),
        ModelSource::Local { .. } => {
            return Err(ModelError::Invalid(
                "local models are already present; nothing to download".into(),
            ))
        }
    };

    match model.engine {
        ModelEngine::Llama => {
            let file = file.clone().ok_or_else(|| {
                ModelError::Invalid(format!(
                    "a llama.cpp download needs a specific .gguf file from {repo} — pick a \
                     quantization first (list the repo's files with ModelFiles, or take the \
                     ModelRecommend pick)"
                ))
            })?;
            let (listing, oids) =
                files::list_files_with_oids(client, repo, revision, ModelEngine::Llama).await?;
            let size_of = |p: &str| listing.iter().find(|f| f.path == p).map(|f| f.size_bytes);
            let oid_of = |p: &str| oids.get(p).cloned();

            let mut plan_files: Vec<PlanFile> = if gguf::is_first_shard(&file) {
                let set = gguf::shard_set(&file)
                    .ok_or_else(|| ModelError::Invalid("malformed shard name".into()))?;
                set.into_iter()
                    .map(|p| PlanFile {
                        size: size_of(&p).unwrap_or(0),
                        expected_sha256: oid_of(&p),
                        path: p,
                        is_mmproj_companion: false,
                    })
                    .collect()
            } else {
                vec![PlanFile {
                    size: size_of(&file).unwrap_or(0),
                    expected_sha256: oid_of(&file),
                    path: file.clone(),
                    is_mmproj_companion: false,
                }]
            };
            // Companion expansion: a *text* GGUF download also fetches the repo's best-matching
            // vision projector in the same job (skipped entirely when the requested file is
            // itself an mmproj — the reference rule). Best-effort: no match, no extra file.
            if let Some((proj_path, proj_size)) = mmproj::best_companion(
                &file,
                listing.iter().map(|f| (f.path.as_str(), f.size_bytes)),
            ) {
                if plan_files.iter().all(|f| f.path != proj_path) {
                    plan_files.push(PlanFile {
                        expected_sha256: oid_of(&proj_path),
                        path: proj_path,
                        size: proj_size,
                        is_mmproj_companion: true,
                    });
                }
            }
            plan_for(model, plan_files, true)
        }
        ModelEngine::MistralRs => {
            let all = files::list_all(client, repo, revision).await?;
            let plan_files = all
                .into_iter()
                .map(|(path, size)| PlanFile {
                    path,
                    size,
                    is_mmproj_companion: false,
                    // Directory (mistral.rs) artifacts are not pinned in this phase.
                    expected_sha256: None,
                })
                .collect();
            plan_for(model, plan_files, false)
        }
    }
}
