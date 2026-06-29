// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Resolved Rooms adapter configuration.
//!
//! Like the Matrix adapter's [`MatrixConfig`](daemon_matrix), the *shape* lives here and the binary
//! owns the parsing (`bins/daemon`'s `config.rs`). Rooms carry no account secrets: a Room's members
//! are profile-bound sessions resolved through the host routing registry, not credentials. The
//! durable Room entities themselves live in the store (`rooms` / `room_members`); this config only
//! toggles whether the loopback transport is spawned and sets the global floor-control defaults.

use daemon_ingest::IngestPolicy;
use daemon_protocol::IsolationPolicy;

/// The resolved `[rooms]` config the host hands to [`crate::serve`]. `enabled = false` (default)
/// leaves the Rooms loopback transport off, exactly like `[matrix].enabled`.
#[derive(Clone, Debug)]
pub struct RoomsConfig {
    /// Whether the Rooms loopback transport is spawned at all. `false` (default) leaves it off.
    pub enabled: bool,
    /// The per-Room turn budget cap (echo-storm prevention): the maximum number of re-injected turns
    /// a single post may fan into before the RoomRouter stops the cascade. `0` = unbounded.
    pub max_turns: u32,
}

impl Default for RoomsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_turns: 16,
        }
    }
}

impl RoomsConfig {
    /// The Room-level ingest policy. Isolation is pinned to `PerThread` to match the host's routing
    /// resolution, so each member session's gate keys busy-state on the same id the host resolves
    /// (the invariant the loopback outbound busy-tracking relies on — see the Matrix adapter).
    pub fn ingest_policy(&self) -> IngestPolicy {
        IngestPolicy {
            isolation: IsolationPolicy::PerThread,
            ..IngestPolicy::default()
        }
    }
}
