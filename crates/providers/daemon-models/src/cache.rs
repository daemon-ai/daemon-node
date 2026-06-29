// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Shared Hugging Face cache resolution + token discovery.
//!
//! Ported from `mistral.rs`' `pipeline/hf.rs` (`hf_hub_cache_dir`, token source) so the daemon's
//! `hf-hub` downloads land in the **same** cache layout the `mistralrs` engine reads. The daemon
//! warms this cache online, then launches the sidecar with `HF_HUB_OFFLINE=1` + `HF_HUB_CACHE`
//! pointing here, so the engine loads from the warmed cache without any network access.
//!
//! Precedence for the hub cache directory (highest first):
//! 1. an explicitly configured directory (the daemon's `cache_dir` config),
//! 2. `HF_HUB_CACHE`,
//! 3. `HF_HOME`/`hub`,
//! 4. `XDG_CACHE_HOME`/`huggingface`/`hub`,
//! 5. `~/.cache/huggingface/hub`.

use std::path::PathBuf;

/// The resolved Hugging Face cache + token the daemon shares with the engine sidecars.
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// The hub cache directory (where `models--org--name/…` trees live).
    pub hub_dir: PathBuf,
    /// The Hugging Face access token, when one is discoverable (for gated/private repos).
    pub token: Option<String>,
}

impl CacheConfig {
    /// Resolve the cache config, honoring an explicit `configured` hub directory then the standard
    /// `HF_*` environment precedence.
    pub fn resolve(configured: Option<PathBuf>) -> Self {
        let hub_dir = configured.unwrap_or_else(default_hub_dir);
        Self {
            hub_dir,
            token: resolve_token(),
        }
    }

    /// The environment a local-inference sidecar must run with to read this warmed cache **offline**
    /// (so a load never reaches the network — the daemon owns all acquisition).
    pub fn sidecar_env(&self) -> Vec<(String, String)> {
        vec![
            (
                "HF_HUB_CACHE".to_string(),
                self.hub_dir.to_string_lossy().into_owned(),
            ),
            ("HF_HUB_OFFLINE".to_string(), "1".to_string()),
        ]
    }
}

/// The default hub cache directory following the `HF_*` / XDG precedence.
fn default_hub_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE") {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("huggingface").join("hub");
    }
    home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

/// The Hugging Face token: `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` env, else the token file under the
/// configured `HF_HOME` (or `~/.cache/huggingface/token`).
fn resolve_token() -> Option<String> {
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(t) = std::env::var(key) {
            let t = t.trim().to_string();
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    let token_path = if let Some(home) = std::env::var_os("HF_HOME") {
        PathBuf::from(home).join("token")
    } else {
        home_dir()?.join(".cache").join("huggingface").join("token")
    };
    let raw = std::fs::read_to_string(token_path).ok()?;
    let trimmed = raw.trim().to_string();
    (!trimmed.is_empty()).then_some(trimmed)
}

/// The user's home directory (`$HOME`), if known.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_dir_wins() {
        let cfg = CacheConfig::resolve(Some(PathBuf::from("/tmp/some-cache")));
        assert_eq!(cfg.hub_dir, PathBuf::from("/tmp/some-cache"));
        let env = cfg.sidecar_env();
        assert!(env
            .iter()
            .any(|(k, v)| k == "HF_HUB_CACHE" && v == "/tmp/some-cache"));
        assert!(env.iter().any(|(k, v)| k == "HF_HUB_OFFLINE" && v == "1"));
    }
}
