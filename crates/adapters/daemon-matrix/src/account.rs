// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// Phase 4: the fs here creates the daemon-internal Matrix E2EE crypto/state store dir under the node
// data root (not attacker-influenced); raw fs allowed file-wide. No process spawns in this file.
#![allow(clippy::disallowed_methods)]

//! Per-account bring-up: the credential-store session blob, the on-disk E2EE client, and the
//! refresh write-back.
//!
//! Each Matrix **account** is a distinct transport instance (`matrix/@bot:hs.org`) owning its own
//! `matrix_sdk::Client`, its own state + E2EE crypto SQLite store, and its own sync loop (spec §2).
//! The credential subsystem is the system of record for the login material (spec §6.2); the crypto
//! store is matrix-sdk-owned on disk (spec §6.3). The store directory is keyed by the **credential
//! ref** (stable, known at both `login` and `serve` time) so the device the crypto store was created
//! for is always re-opened — never a fresh one (spec §6.3 device-id constraint).

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use matrix_sdk::authentication::matrix::MatrixSession;
use matrix_sdk::Client;
use serde::{Deserialize, Serialize};

use daemon_protocol::TransportId;

/// The credential-store blob for one Matrix account: the homeserver to dial plus the matrix-sdk
/// session (user/device ids + access/refresh tokens). Serialized as JSON under the account's
/// credential-ref. This is **not** the E2EE crypto store (that is the on-disk sqlite db).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredSession {
    /// The homeserver base URL the client dials.
    pub homeserver: String,
    /// The matrix-sdk native-auth session (meta + tokens).
    pub session: MatrixSession,
}

impl StoredSession {
    /// Serialize to the opaque credential blob.
    pub fn to_blob(&self) -> Result<String> {
        serde_json::to_string(self).context("serializing matrix session blob")
    }

    /// Parse from the opaque credential blob.
    pub fn from_blob(blob: &str) -> Result<Self> {
        serde_json::from_str(blob).context("parsing matrix session blob")
    }
}

/// One brought-up Matrix account: its instance-qualified transport id, the bare user id, and the
/// live client.
pub struct Account {
    /// The instance-qualified transport id (`matrix/@bot:hs.org`).
    pub transport: TransportId,
    /// The bare account user id (`@bot:hs.org`).
    pub bare: String,
    /// The live, session-restored client.
    pub client: Client,
}

/// The bare user id (`@bot:hs.org`) inside an instance-qualified `matrix/...` transport id.
pub fn bare_account(transport: &TransportId) -> &str {
    transport
        .as_str()
        .strip_prefix("matrix/")
        .unwrap_or_else(|| transport.as_str())
}

/// A filesystem-safe directory name for a credential ref / account handle (`matrix/alpha/a` ->
/// `matrix_alpha_a`).
pub fn store_dir_name(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// The per-account store directory under `store_root`, keyed by `credential_ref` so `login` and
/// `serve` always open the *same* on-disk state + crypto store (device-id stability, spec §6.3).
pub fn account_store_dir(store_root: &Path, credential_ref: &str) -> PathBuf {
    store_root.join(store_dir_name(credential_ref))
}

/// Build a client for `homeserver` with the per-account on-disk sqlite state + E2EE crypto store at
/// `store_dir`, with automatic access-token refresh. Does not log in or restore — the caller drives
/// `login` (one-shot) or `restore_session` (bring-up).
pub async fn build_client(homeserver: &str, store_dir: &Path) -> Result<Client> {
    std::fs::create_dir_all(store_dir)
        .with_context(|| format!("creating matrix store dir {}", store_dir.display()))?;
    Client::builder()
        .homeserver_url(homeserver)
        .sqlite_store(store_dir, None)
        .handle_refresh_tokens()
        .build()
        .await
        .map_err(|e| anyhow!("building matrix client for {homeserver}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_strips_family_prefix() {
        let t = TransportId::new("matrix/@bot:hs.org".to_string());
        assert_eq!(bare_account(&t), "@bot:hs.org");
    }

    #[test]
    fn store_dir_name_is_fs_safe() {
        assert_eq!(store_dir_name("matrix/@bot:hs.org"), "matrix__bot_hs.org");
        assert_eq!(store_dir_name("alpha-1.2"), "alpha-1.2");
    }
}
