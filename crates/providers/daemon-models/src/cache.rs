// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// Phase 4: fs here reads the daemon-controlled HF cache/token path under the node data root,
// not attacker-influenced; raw fs allowed file-wide. No process spawns in this file.
#![allow(clippy::disallowed_methods)]

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
//! 5. `~/.cache/huggingface/hub`,
//! 6. a caller-supplied last resort for HOME-less environments (containers/microvms) — the daemon
//!    passes a directory under its own data dir, so a missing `HOME` never breaks boot.

use std::ffi::OsString;
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
        Self::resolve_with_fallback(configured, None)
    }

    /// [`CacheConfig::resolve`] with an explicit **last-resort** hub directory for HOME-less
    /// environments (containers/microvms): it is used only when no directory is configured, none
    /// of the `HF_*`/XDG variables is set, and `HOME` is missing — every populated environment
    /// resolves exactly as before. `None` keeps the prior last resort (a cwd-relative
    /// `./.cache/huggingface/hub`).
    pub fn resolve_with_fallback(
        configured: Option<PathBuf>,
        last_resort: Option<PathBuf>,
    ) -> Self {
        let hub_dir = configured.unwrap_or_else(|| default_hub_dir(last_resort));
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
fn default_hub_dir(last_resort: Option<PathBuf>) -> PathBuf {
    default_hub_dir_from(|key| std::env::var_os(key), last_resort)
}

/// [`default_hub_dir`] over an injected environment lookup, so the precedence — including the
/// HOME-less container case — is unit-testable without mutating the process environment.
fn default_hub_dir_from(
    env: impl Fn(&str) -> Option<OsString>,
    last_resort: Option<PathBuf>,
) -> PathBuf {
    let set = |key: &str| env(key).filter(|v| !v.is_empty());
    if let Some(dir) = set("HF_HUB_CACHE") {
        return PathBuf::from(dir);
    }
    if let Some(home) = set("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    if let Some(xdg) = set("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("huggingface").join("hub");
    }
    if let Some(home) = set("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    last_resort.unwrap_or_else(|| {
        PathBuf::from(".")
            .join(".cache")
            .join("huggingface")
            .join("hub")
    })
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

/// The user's home directory (`$HOME`), if known. Empty counts as unset (containers often clear
/// rather than unset it); never consults the passwd database, so the answer matches what the
/// hub-dir precedence saw.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
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

    /// A synthetic environment for the injected-lookup precedence tests (no process-env mutation,
    /// so the tests are hermetic and parallel-safe).
    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |key| {
            pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| OsString::from(v))
        }
    }

    #[test]
    fn hub_dir_precedence_follows_hf_then_xdg_then_home() {
        let all = [
            ("HF_HUB_CACHE", "/hf-hub-cache"),
            ("HF_HOME", "/hf-home"),
            ("XDG_CACHE_HOME", "/xdg"),
            ("HOME", "/home/u"),
        ];
        assert_eq!(
            default_hub_dir_from(env_of(&all), None),
            PathBuf::from("/hf-hub-cache")
        );
        assert_eq!(
            default_hub_dir_from(env_of(&all[1..]), None),
            PathBuf::from("/hf-home/hub")
        );
        assert_eq!(
            default_hub_dir_from(env_of(&all[2..]), None),
            PathBuf::from("/xdg/huggingface/hub")
        );
        assert_eq!(
            default_hub_dir_from(env_of(&all[3..]), None),
            PathBuf::from("/home/u/.cache/huggingface/hub")
        );
    }

    /// The container case: nothing set at all. The caller-supplied last resort (the daemon's
    /// data-dir fallback) is used; without one the prior cwd-relative default remains.
    #[test]
    fn homeless_environment_uses_the_last_resort() {
        assert_eq!(
            default_hub_dir_from(env_of(&[]), Some(PathBuf::from("/data/huggingface/hub"))),
            PathBuf::from("/data/huggingface/hub")
        );
        assert_eq!(
            default_hub_dir_from(env_of(&[]), None),
            PathBuf::from("./.cache/huggingface/hub")
        );
    }

    /// Empty values count as unset (containers often clear rather than unset variables).
    #[test]
    fn empty_environment_values_count_as_unset() {
        let empties = [("HF_HUB_CACHE", ""), ("HF_HOME", ""), ("HOME", "")];
        assert_eq!(
            default_hub_dir_from(env_of(&empties), Some(PathBuf::from("/data/hub"))),
            PathBuf::from("/data/hub")
        );
    }
}
