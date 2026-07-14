// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The replay capture — the coordinator's reproducible `tick` driver trace (spec §14, §11.2).
//!
//! The node-visible [`MessageLog`](crate::MessageLog) records the **signed messages** any observer
//! sees on the wire, but the pure `tick` also consumes **clocks** (`Input::Clock`, the warmup /
//! cooldown / force-progress edges) — a driver detail that never crosses the wire (§14: clocks are
//! not signed messages). To make a recorded run *byte-reproducibly* replayable off disk (the
//! gate-ceremony `swarm-replay` step), the coordinator emits a [`RunCapture`]: its **initial**
//! [`CoordinatorState`] plus the exact ordered `Input` trace (messages **and** clocks) it fed
//! `tick`.
//!
//! [`crate::replay::replay_from_state`] re-runs `tick` from the captured initial state over that
//! trace and verifies the coordinator's [`RoundRecord`](daemon_swarm_proto::messages::RoundRecord)s
//! re-derive; a companion [`MessageLog`] supplies the **independent** wire-recorded records as the
//! oracle to compare against (PROTO-20 / §6.4 I1). Serializes as `magic + one canonical-CBOR blob`,
//! so two writes of one capture are byte-identical.

use std::io::{Read, Write};

use daemon_swarm_coordinator::{CoordinatorState, Input};
use daemon_swarm_proto::{from_canonical_slice, to_canonical_vec};

use crate::ObserveError;

/// Magic + version prefix of a serialized [`RunCapture`].
const MAGIC: &[u8; 8] = b"DSMCAP01";

/// The coordinator's reproducible `tick` driver trace: the initial state + the exact ordered inputs
/// (messages **and** clocks) it consumed. Feed to [`crate::replay::replay_from_state`] to re-derive
/// the run offline.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RunCapture {
    /// The `CoordinatorState` the run started from (`WaitingForMembers`, pre-genesis clock).
    pub initial: CoordinatorState,
    /// The exact ordered `tick` inputs the coordinator fed (driving messages + clocks), excluding the
    /// coordinator's own published outputs (`RoundOpen`/`RoundRecord`) — those are the oracle, taken
    /// from the wire [`MessageLog`](crate::MessageLog) at replay time.
    pub inputs: Vec<Input>,
}

impl RunCapture {
    /// A capture from an initial state + the driving input trace.
    #[must_use]
    pub fn new(initial: CoordinatorState, inputs: Vec<Input>) -> Self {
        Self { initial, inputs }
    }

    /// Serialize as `magic + canonical-CBOR(self)` (deterministic; two writes are byte-identical).
    pub fn write_to(&self, w: &mut impl Write) -> Result<(), ObserveError> {
        w.write_all(MAGIC)
            .map_err(|e| ObserveError::Store(e.to_string()))?;
        let bytes = to_canonical_vec(self).map_err(|e| ObserveError::Codec(e.to_string()))?;
        w.write_all(&bytes)
            .map_err(|e| ObserveError::Store(e.to_string()))?;
        Ok(())
    }

    /// Read a capture back from bytes produced by [`RunCapture::write_to`].
    pub fn read_from(r: &mut impl Read) -> Result<Self, ObserveError> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)
            .map_err(|e| ObserveError::Store(e.to_string()))?;
        if &magic != MAGIC {
            return Err(ObserveError::Store("bad run-capture magic".into()));
        }
        let mut rest = Vec::new();
        r.read_to_end(&mut rest)
            .map_err(|e| ObserveError::Store(e.to_string()))?;
        from_canonical_slice(&rest).map_err(|e| ObserveError::Codec(e.to_string()))
    }
}
