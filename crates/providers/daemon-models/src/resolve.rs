// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Turning a [`ModelRef`] into a concrete download plan: which repo files to fetch for each engine.
//!
//! - **llama**: a single named GGUF file (expanding to the full shard set when the named file is the
//!   first shard of a split model). One artifact, `single_file = true`.
//! - **mistral.rs**: every file in the repo (config + tokenizer + weights), so the engine loads from
//!   the warmed cache offline. The artifact is the snapshot directory, `single_file = false`.

use daemon_common::{ModelEngine, ModelRef, ModelSource};

use crate::acquire::{plan_for, DownloadPlan, PlanFile};
use crate::error::{ModelError, Result};
use crate::gguf;
use crate::hf::{files, HfClient};

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
                ModelError::Invalid("llama requires a specific .gguf file to download".into())
            })?;
            let listing = files::list_files(client, repo, revision, ModelEngine::Llama).await?;
            let size_of = |p: &str| listing.iter().find(|f| f.path == p).map(|f| f.size_bytes);

            let plan_files: Vec<PlanFile> = if gguf::is_first_shard(&file) {
                let set = gguf::shard_set(&file)
                    .ok_or_else(|| ModelError::Invalid("malformed shard name".into()))?;
                set.into_iter()
                    .map(|p| PlanFile {
                        size: size_of(&p).unwrap_or(0),
                        path: p,
                    })
                    .collect()
            } else {
                vec![PlanFile {
                    size: size_of(&file).unwrap_or(0),
                    path: file,
                }]
            };
            plan_for(model, plan_files, true)
        }
        ModelEngine::MistralRs => {
            let all = files::list_all(client, repo, revision).await?;
            let plan_files = all
                .into_iter()
                .map(|(path, size)| PlanFile { path, size })
                .collect();
            plan_for(model, plan_files, false)
        }
    }
}
