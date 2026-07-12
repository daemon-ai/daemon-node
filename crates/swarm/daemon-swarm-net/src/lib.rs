// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-swarm-net` — the swarm transport.
//!
//! `SwarmTransport`: the gossip control plane plus the `r2` / `iroh-blobs` payload planes
//! (swarm-training-spec.md §7.1), the presign client, and artifact fetch (`r2` / `hf` / `https`).
//! Engine-agnostic; consumed by [`daemon_swarm_run`](../daemon_swarm_run) (§10.1).
//!
//! Wave-0 scaffold: only the error type is present; the transport surface lands with lane **R**.
//! Outbound HTTP must route through `daemon_egress::EgressClient` (raw `reqwest::Client` is banned
//! workspace-wide by clippy).

#![forbid(unsafe_code)]

/// Errors surfaced by the swarm transport.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SwarmNetError {
    /// A control-plane or payload-plane transport step failed.
    #[error("swarm transport error: {0}")]
    Transport(String),
    /// An artifact fetch (`r2` / `hf` / `https`) failed.
    #[error("artifact fetch failed: {0}")]
    Fetch(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_renders() {
        assert!(SwarmNetError::Fetch("404".into())
            .to_string()
            .contains("artifact fetch"));
    }
}
