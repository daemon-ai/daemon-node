// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The vhc control-plane protocol version (spec §7.3, §16; TDD PROTO-13).
//!
//! `VhcProtoVersion` governs the peer↔coordinator + peer↔peer control plane, **independent** of
//! the app↔node `WireVersion`. A run pins one version; peers with any other version cannot join
//! (exact match — no mid-run protocol drift). Bumps ship in this crate alongside fixtures.

use serde::{Deserialize, Serialize};

use crate::error::VhcProtoError;

/// The vhc control-plane protocol version (a `u16`, spec §7.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VhcProtoVersion(pub u16);

/// The version this build speaks.
///
/// Wave-1 ships version 1 (the scaffold placeholder was `0`); see `docs/specs/swarm-ledger-p1.md`.
pub const VHC_PROTO_VERSION: VhcProtoVersion = VhcProtoVersion(1);

impl VhcProtoVersion {
    /// Whether `peer` may join a run pinned to `self` — exact match only.
    #[must_use]
    pub fn accepts(self, peer: VhcProtoVersion) -> bool {
        self == peer
    }

    /// Join predicate as a `Result`: `Ok` iff `peer` exactly matches the run's pinned `self`.
    pub fn check_join(self, peer: VhcProtoVersion) -> Result<(), VhcProtoError> {
        if self.accepts(peer) {
            Ok(())
        } else {
            Err(VhcProtoError::Version(format!(
                "peer speaks vhc proto v{} but the run is pinned to v{}",
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
        assert!(VHC_PROTO_VERSION.accepts(VHC_PROTO_VERSION));
        assert!(VHC_PROTO_VERSION.check_join(VHC_PROTO_VERSION).is_ok());
    }

    #[test]
    fn mismatch_rejected() {
        let run = VhcProtoVersion(1);
        assert!(!run.accepts(VhcProtoVersion(2)));
        assert!(run.check_join(VhcProtoVersion(0)).is_err());
        assert!(run.check_join(VhcProtoVersion(2)).is_err());
    }
}
