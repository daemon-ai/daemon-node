// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// Phase 4: fs here stats a resolved model artifact under the daemon-controlled cache, not
// attacker-influenced; raw fs allowed file-wide. No process spawns in this file.
#![allow(clippy::disallowed_methods)]

//! GGUF metadata introspection via `gguf-rs`.
//!
//! Reads the header key-values of a GGUF file (architecture, name, file-type, context length, block
//! count, parameter count) without loading the model for inference. Used to enrich catalog records
//! with authoritative metadata — more reliable than the filename-only quant guess — and to power the
//! `model inspect` surface.

use std::path::Path;

use daemon_common::{GgufInfo, InstalledModel};

use crate::error::{ModelError, Result};
use crate::gguf;

/// Read GGUF header metadata from `path`.
pub fn inspect(path: &Path) -> Result<GgufInfo> {
    let path_str = path
        .to_str()
        .ok_or_else(|| ModelError::Invalid("model path is not valid UTF-8".into()))?;
    let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    let mut container = gguf_rs::get_gguf_container(path_str)
        .map_err(|e| ModelError::Invalid(format!("not a readable GGUF: {e}")))?;
    let model = container
        .decode()
        .map_err(|e| ModelError::Invalid(format!("failed to decode GGUF header: {e}")))?;

    let architecture = match model.model_family().as_str() {
        "unknown" | "" => None,
        arch => Some(arch.to_string()),
    };
    let meta = model.metadata();
    let name = meta
        .get("general.name")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let file_type = clean_ftype(&model.file_type());
    let context_length = architecture
        .as_deref()
        .and_then(|arch| meta.get(&format!("{arch}.context_length")))
        .and_then(|v| v.as_u64())
        .and_then(|n| u32::try_from(n).ok());
    let block_count = architecture
        .as_deref()
        .and_then(|arch| meta.get(&format!("{arch}.block_count")))
        .and_then(|v| v.as_u64())
        .and_then(|n| u32::try_from(n).ok());
    let quantization_version = meta
        .get("general.quantization_version")
        .and_then(|v| v.as_u64())
        .and_then(|n| u32::try_from(n).ok());

    // The internal parameter total is private; reconstruct it by summing tensor element counts.
    let parameter_count = {
        let total: u64 = model
            .tensors()
            .iter()
            .map(|t| t.shape.iter().product::<u64>())
            .sum();
        (total > 0).then_some(total)
    };

    Ok(GgufInfo {
        architecture,
        name,
        file_type,
        context_length,
        block_count,
        quantization_version,
        parameter_count,
        size_bytes,
    })
}

/// Enrich a catalog record in place with GGUF metadata, when its local artifact is a single GGUF
/// file. Best-effort: a non-GGUF artifact (e.g. a mistral.rs repo dir) or an unreadable header
/// simply leaves the record's fields as-is.
pub fn enrich_installed(record: &mut InstalledModel) {
    let is_gguf_artifact = record
        .local_path
        .to_str()
        .map(gguf::is_gguf)
        .unwrap_or(false);
    if !is_gguf_artifact {
        return;
    }
    let Ok(info) = inspect(&record.local_path) else {
        return;
    };
    record.arch = info.architecture;
    record.context_length = info.context_length;
    if record.file_type.is_none() {
        record.file_type = info.file_type.clone();
    }
    // Prefer the precise filename quant (e.g. Q4_K_M); fall back to the metadata base type — this
    // covers local-only artifacts that never had a filename quant guess.
    if record.quant.is_none() {
        record.quant = info.file_type;
    }
}

/// Normalize `gguf-rs`'s human file-type string (`"Mostly Q4_K"`, `"All F32"`) to a bare quant
/// label (`"Q4_K"`, `"F32"`). Returns `None` for an unknown/absent type.
fn clean_ftype(raw: &str) -> Option<String> {
    let trimmed = raw
        .trim()
        .strip_prefix("Mostly ")
        .or_else(|| raw.trim().strip_prefix("All "))
        .unwrap_or(raw.trim());
    // Drop any trailing annotations such as " (UNSUPPORTED)" or " Some F16".
    let label = trimmed.split_whitespace().next().unwrap_or("");
    match label {
        "" | "unknown" => None,
        other => Some(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_ftype_strips_prefixes() {
        assert_eq!(clean_ftype("Mostly Q4_K").as_deref(), Some("Q4_K"));
        assert_eq!(clean_ftype("All F32").as_deref(), Some("F32"));
        assert_eq!(clean_ftype("Mostly BF16").as_deref(), Some("BF16"));
        assert_eq!(clean_ftype("Mostly Q4_1 Some F16").as_deref(), Some("Q4_1"));
        assert_eq!(clean_ftype("unknown"), None);
        assert_eq!(clean_ftype(""), None);
    }
}
