// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The swarm control-plane protocol version (spec §7.3, §16; TDD PROTO-13).
//!
//! `SwarmProtoVersion` governs the peer↔coordinator + peer↔peer control plane, **independent** of
//! the app↔node `WireVersion`. A run pins one version; peers with any other version cannot join
//! (exact match — no mid-run protocol drift). Bumps ship in this crate alongside fixtures.

use serde::{Deserialize, Serialize};

use crate::error::SwarmProtoError;

/// The swarm control-plane protocol version (a `u16`, spec §7.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SwarmProtoVersion(pub u16);

/// The version this build speaks.
///
/// Wave-1 ships version 1 (the scaffold placeholder was `0`); see `docs/specs/swarm-ledger-p1.md`.
pub const SWARM_PROTO_VERSION: SwarmProtoVersion = SwarmProtoVersion(1);

impl SwarmProtoVersion {
    /// Whether `peer` may join a run pinned to `self` — exact match only.
    #[must_use]
    pub fn accepts(self, peer: SwarmProtoVersion) -> bool {
        self == peer
    }

    /// Join predicate as a `Result`: `Ok` iff `peer` exactly matches the run's pinned `self`.
    pub fn check_join(self, peer: SwarmProtoVersion) -> Result<(), SwarmProtoError> {
        if self.accepts(peer) {
            Ok(())
        } else {
            Err(SwarmProtoError::Version(format!(
                "peer speaks swarm proto v{} but the run is pinned to v{}",
                peer.0, self.0
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_accepts() {
        assert!(SWARM_PROTO_VERSION.accepts(SWARM_PROTO_VERSION));
        assert!(SWARM_PROTO_VERSION.check_join(SWARM_PROTO_VERSION).is_ok());
    }

    #[test]
    fn mismatch_rejected() {
        let run = SwarmProtoVersion(1);
        assert!(!run.accepts(SwarmProtoVersion(2)));
        assert!(run.check_join(SwarmProtoVersion(0)).is_err());
        assert!(run.check_join(SwarmProtoVersion(2)).is_err());
    }
}
