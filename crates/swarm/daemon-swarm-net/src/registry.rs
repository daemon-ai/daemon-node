// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`RegistryClient`] — run discovery + envelope fetch against the coordinator registry (spec
//! §6.1/§11.1; A1).
//!
//! The cloud `apps/swarm` worker exposes a validation-only run registry:
//! `GET {base}/runs` (snapshot) and `GET {base}/runs/:id` (one descriptor), each wrapped in
//! `{ "data": … }` (`apps/swarm/src/registry.ts`, `index.ts`). The descriptors carry the frozen
//! envelope's blake3 (`envelope_hash`) + artifact manifest, never the module bytes — the cloud
//! never fetches/executes a module (spec §11.1/§12), and every peer re-derives eligibility at
//! assess (§6.5).
//!
//! A node discovers a run here, then [`fetch_envelope`](RegistryClient::fetch_envelope)s the frozen
//! envelope object (presigned `GET` of `runs/<run>/envelope.cbor`, §11.3) and **blake3-verifies** it
//! against the descriptor's `envelope_hash` before handing the bytes to the worker's `AssessRun`.
//! All outbound HTTP rides the SSRF-safe [`EgressClient`] (raw `reqwest::Client` is clippy-banned);
//! auth is the same `swarm:*` credential the WS client uses (Bearer for the gateway, or the internal
//! identity headers for a direct-to-worker dev target) — never hardcoded.

use daemon_egress::{EgressClient, EgressRequest, Redirects};
use daemon_swarm_proto::blake3_hash;
use serde::{Deserialize, Serialize};

use crate::presign::{PresignOp, PresignRequest, PresignResponse};
use crate::seam::RunId;
use crate::SwarmNetError;

/// One artifact the run references (name → pinned blake3 + size). Mirrors the cloud `ArtifactRef`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunArtifact {
    /// Run-relative artifact name/path (e.g. `experiment.wasm`).
    pub path: String,
    /// blake3 content hash, 64 lowercase hex chars.
    pub blake3: String,
    /// Declared size in bytes.
    pub size: u64,
}

/// A run descriptor from the registry (`apps/swarm` `RunDescriptor`). Experiment-opaque: it carries
/// the frozen envelope's hash + artifact manifest, never module bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunDescriptor {
    /// The run id (coordinator-assigned).
    pub run_id: String,
    /// Envelope schema major (spec §16).
    pub schema: u32,
    /// The swarm proto version the run is pinned to (§16).
    pub proto_version: u32,
    /// blake3 of the frozen envelope bytes (the signed hash), 64 lowercase hex chars.
    pub envelope_hash: String,
    /// ed25519 author public key (32 B), hex.
    pub author_pubkey: String,
    /// The envelope's artifact map (names + pinned hashes + sizes).
    #[serde(default)]
    pub artifacts: Vec<RunArtifact>,
    /// Per-peer round-payload cap in bytes.
    pub update_max_bytes: u64,
    /// Minimum roster size.
    pub min_peers: u32,
    /// Maximum roster size.
    pub max_peers: u32,
    /// Total rounds before the run finishes (`None` = driven elsewhere).
    #[serde(default)]
    pub rounds: Option<u64>,
    /// Creation time (unix seconds) stamped by the registry.
    #[serde(default)]
    pub created_at: u64,
    /// R2 key of the stored `envelope.cbor` (§11.3).
    #[serde(default)]
    pub envelope_key: String,
}

/// The `{ "data": T }` envelope every `apps/swarm` route wraps its success body in.
#[derive(Deserialize)]
struct DataEnvelope<T> {
    data: T,
}

/// The `swarm:*` credential the registry + presign requests carry (never hardcoded — sourced from
/// `JoinRun.credentials` / node config, mirroring [`crate::ws_client::WsAuth`]).
#[derive(Clone, Debug, Default)]
enum Auth {
    #[default]
    None,
    Bearer(String),
    Internal {
        org_id: String,
        actor: String,
    },
}

/// Discovery + envelope-fetch client against a coordinator registry base
/// (e.g. `https://api.daemon.ai/api/v1/swarm`).
pub struct RegistryClient {
    egress: EgressClient,
    base_url: String,
    auth: Auth,
}

impl RegistryClient {
    /// Build a client against `base_url` (a trailing `/` is trimmed).
    pub fn new(egress: EgressClient, base_url: impl Into<String>) -> Self {
        Self {
            egress,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            auth: Auth::None,
        }
    }

    /// Attach the `swarm:*`-scoped API-key bearer token (the gateway path).
    #[must_use]
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.auth = Auth::Bearer(token.into());
        self
    }

    /// Attach the internal identity headers (the direct-to-`apps/swarm` dev path).
    #[must_use]
    pub fn with_internal(mut self, org_id: impl Into<String>, actor: impl Into<String>) -> Self {
        self.auth = Auth::Internal {
            org_id: org_id.into(),
            actor: actor.into(),
        };
        self
    }

    /// The registry base URL.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Discover all runs (`GET {base}/runs`).
    pub async fn list_runs(&self) -> Result<Vec<RunDescriptor>, SwarmNetError> {
        let url = format!("{}/runs", self.base_url);
        let body = self.authed_get(&url).await?;
        let env: DataEnvelope<Vec<RunDescriptor>> = serde_json::from_slice(&body)
            .map_err(|e| SwarmNetError::Transport(format!("decode run list: {e}")))?;
        Ok(env.data)
    }

    /// Fetch one run descriptor (`GET {base}/runs/:id`); `Ok(None)` on a 404.
    pub async fn get_run(&self, run_id: &str) -> Result<Option<RunDescriptor>, SwarmNetError> {
        let url = format!("{}/runs/{run_id}", self.base_url);
        let req = self.authed_request(EgressRequest::get(&url));
        let resp = self
            .egress
            .execute(req, Redirects::None)
            .await
            .map_err(|e| SwarmNetError::Transport(format!("get run {run_id}: {e}")))?;
        let status = resp.status();
        if status.as_u16() == 404 {
            return Ok(None);
        }
        let body = resp
            .bytes()
            .await
            .map_err(|e| SwarmNetError::Transport(format!("read run {run_id}: {e}")))?;
        if !status.is_success() {
            return Err(SwarmNetError::Transport(format!(
                "get run {run_id} returned {status}: {}",
                String::from_utf8_lossy(&body)
            )));
        }
        let env: DataEnvelope<RunDescriptor> = serde_json::from_slice(&body)
            .map_err(|e| SwarmNetError::Transport(format!("decode run {run_id}: {e}")))?;
        Ok(Some(env.data))
    }

    /// Fetch the frozen envelope for `run` and **blake3-verify** it against `descriptor.envelope_hash`.
    ///
    /// Presigns a `GET` of the run-relative `envelope.cbor` artifact (§11.3), downloads the bytes via
    /// [`EgressClient`], and rejects a hash mismatch as [`SwarmNetError::HashMismatch`] (the tamper
    /// path, §12) — so a registry that served the wrong envelope can never reach `AssessRun`.
    pub async fn fetch_envelope(
        &self,
        run: &RunId,
        descriptor: &RunDescriptor,
    ) -> Result<Vec<u8>, SwarmNetError> {
        let presigned = self
            .presign(
                run,
                &PresignRequest::artifact(PresignOp::Get, "envelope.cbor"),
            )
            .await?;
        // The presigned URL carries its own credential (SigV4 query / object-proxy HMAC), so the
        // object GET needs no auth headers — just the bytes.
        let resp = self
            .egress
            .get(&presigned.url, Redirects::None)
            .await
            .map_err(|e| SwarmNetError::Fetch(format!("fetch envelope: {e}")))?;
        let status = resp.status();
        let body = resp
            .bytes()
            .await
            .map_err(|e| SwarmNetError::Fetch(format!("read envelope body: {e}")))?;
        if !status.is_success() {
            return Err(SwarmNetError::Fetch(format!(
                "envelope fetch returned {status}"
            )));
        }
        let got = blake3_hash(&body[..]).to_hex();
        if got != descriptor.envelope_hash {
            return Err(SwarmNetError::HashMismatch {
                expected: descriptor.envelope_hash.clone(),
                actual: got,
            });
        }
        Ok(body.to_vec())
    }

    /// Presign one object for `run` (`POST {base}/runs/:id/presign`) with the registry auth applied.
    async fn presign(
        &self,
        run: &RunId,
        req: &PresignRequest,
    ) -> Result<PresignResponse, SwarmNetError> {
        let url = format!("{}/runs/{}/presign", self.base_url, run.as_str());
        let ereq = EgressRequest::post_json(&url, req)
            .map_err(|e| SwarmNetError::Transport(format!("encode presign request: {e}")))?;
        let resp = self
            .egress
            .execute(self.authed_request(ereq), Redirects::DEFAULT)
            .await
            .map_err(|e| SwarmNetError::Transport(format!("presign request: {e}")))?;
        let status = resp.status();
        let body = resp
            .bytes()
            .await
            .map_err(|e| SwarmNetError::Transport(format!("read presign body: {e}")))?;
        if !status.is_success() {
            return Err(SwarmNetError::Transport(format!(
                "presign endpoint returned {status}: {}",
                String::from_utf8_lossy(&body)
            )));
        }
        serde_json::from_slice(&body)
            .map_err(|e| SwarmNetError::Transport(format!("decode presign response: {e}")))
    }

    /// Issue an authed GET and return the body bytes (2xx only).
    async fn authed_get(&self, url: &str) -> Result<Vec<u8>, SwarmNetError> {
        let req = self.authed_request(EgressRequest::get(url));
        let resp = self
            .egress
            .execute(req, Redirects::None)
            .await
            .map_err(|e| SwarmNetError::Transport(format!("registry GET {url}: {e}")))?;
        let status = resp.status();
        let body = resp
            .bytes()
            .await
            .map_err(|e| SwarmNetError::Transport(format!("read {url}: {e}")))?;
        if !status.is_success() {
            return Err(SwarmNetError::Transport(format!(
                "registry GET {url} returned {status}"
            )));
        }
        Ok(body.to_vec())
    }

    /// Apply the configured auth headers to an outbound request.
    fn authed_request(&self, req: EgressRequest) -> EgressRequest {
        match &self.auth {
            Auth::None => req,
            Auth::Bearer(token) => req.bearer_auth(token),
            Auth::Internal { org_id, actor } => req
                .header("x-daemon-org-id", org_id)
                .header("x-daemon-actor", actor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_descriptor_decodes_registry_shape() {
        // The `{ "data": … }`-wrapped shape `apps/swarm` returns, with optional fields present.
        let json = r#"{
            "run_id": "run-1", "schema": 1, "proto_version": 3,
            "envelope_hash": "aa", "author_pubkey": "bb",
            "artifacts": [{"path": "envelope.cbor", "blake3": "cc", "size": 12}],
            "update_max_bytes": 1048576, "min_peers": 1, "max_peers": 8,
            "rounds": 10, "created_at": 42, "envelope_key": "runs/run-1/envelope.cbor"
        }"#;
        let d: RunDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(d.run_id, "run-1");
        assert_eq!(d.proto_version, 3);
        assert_eq!(d.rounds, Some(10));
        assert_eq!(d.artifacts.len(), 1);
    }
}
