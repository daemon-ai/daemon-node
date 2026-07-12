// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The append-only, replayable message log (spec §14; TDD §3.9).
//!
//! Every swarm-visible transition is a signed control message (§6.4), so the per-run log is exactly a
//! sequence of [`SignedMessage`]s in **arrival order**, indexed by `(round, kind)`. It serializes as
//! canonical-CBOR **length-framed** records (a magic header + run id, then `u32`-LE length + the
//! canonical bytes of each message), so appends are O(1) and two writes of the same log are
//! byte-identical (canonical). The round messages being events in this log is what makes I1 replay a
//! debugging tool, not just a recovery mechanism (§14).

use std::io::{Read, Write};

use daemon_swarm_coordinator::Input;
use daemon_swarm_proto::messages::{SignedMessage, SwarmMessage};
use daemon_swarm_proto::{from_canonical_slice, to_canonical_vec};

use crate::ObserveError;

/// Magic + version prefix of a serialized [`MessageLog`].
const MAGIC: &[u8; 8] = b"DSMLOG01";

/// The kind of a swarm control message — the second half of the `(round, kind)` index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MessageKind {
    /// [`SwarmMessage::RoundOpen`].
    RoundOpen,
    /// [`SwarmMessage::Commitment`].
    Commitment,
    /// [`SwarmMessage::Attestation`].
    Attestation,
    /// [`SwarmMessage::StorageReceipt`].
    StorageReceipt,
    /// [`SwarmMessage::RoundRecord`].
    RoundRecord,
    /// [`SwarmMessage::Digest`].
    Digest,
    /// [`SwarmMessage::Straggle`].
    Straggle,
    /// [`SwarmMessage::Join`].
    Join,
    /// [`SwarmMessage::Heartbeat`].
    Heartbeat,
}

impl MessageKind {
    /// The kind of a payload.
    #[must_use]
    pub fn of(m: &SwarmMessage) -> Self {
        match m {
            SwarmMessage::RoundOpen(_) => Self::RoundOpen,
            SwarmMessage::Commitment(_) => Self::Commitment,
            SwarmMessage::Attestation(_) => Self::Attestation,
            SwarmMessage::StorageReceipt(_) => Self::StorageReceipt,
            SwarmMessage::RoundRecord(_) => Self::RoundRecord,
            SwarmMessage::Digest(_) => Self::Digest,
            SwarmMessage::Straggle(_) => Self::Straggle,
            SwarmMessage::Join(_) => Self::Join,
            SwarmMessage::Heartbeat(_) => Self::Heartbeat,
        }
    }
}

/// The round a payload pertains to, if any. `Join` is roster-scoped (no round).
#[must_use]
pub fn round_of(m: &SwarmMessage) -> Option<u64> {
    match m {
        SwarmMessage::RoundOpen(x) => Some(x.round),
        SwarmMessage::Commitment(x) => Some(x.round),
        SwarmMessage::Attestation(x) => Some(x.round),
        SwarmMessage::StorageReceipt(x) => Some(x.round),
        SwarmMessage::RoundRecord(x) => Some(x.round),
        SwarmMessage::Digest(x) => Some(x.round),
        SwarmMessage::Straggle(x) => Some(x.round),
        SwarmMessage::Heartbeat(x) => Some(x.round),
        SwarmMessage::Join(_) => None,
    }
}

/// An append-only, per-run log of signed control messages in arrival order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MessageLog {
    run_id: String,
    entries: Vec<SignedMessage>,
}

impl MessageLog {
    /// A new, empty log for `run_id`.
    #[must_use]
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            entries: Vec::new(),
        }
    }

    /// The run this log belongs to.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Append a message (arrival order). Verification is the caller's concern — the log is a faithful
    /// record of what arrived, including anything a replay/audit later rejects.
    pub fn append(&mut self, msg: SignedMessage) {
        self.entries.push(msg);
    }

    /// The number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The records in arrival order.
    #[must_use]
    pub fn entries(&self) -> &[SignedMessage] {
        &self.entries
    }

    /// Iterate the records in arrival order.
    pub fn iter(&self) -> impl Iterator<Item = &SignedMessage> {
        self.entries.iter()
    }

    /// Records pertaining to `round` (any kind; excludes roster-scoped `Join`s).
    pub fn by_round(&self, round: u64) -> impl Iterator<Item = &SignedMessage> {
        self.entries
            .iter()
            .filter(move |m| round_of(&m.payload) == Some(round))
    }

    /// Records of a given `kind` (any round).
    pub fn by_kind(&self, kind: MessageKind) -> impl Iterator<Item = &SignedMessage> {
        self.entries
            .iter()
            .filter(move |m| MessageKind::of(&m.payload) == kind)
    }

    /// Records matching both `round` and `kind` (the `(round, kind)` index).
    pub fn by_round_kind(
        &self,
        round: u64,
        kind: MessageKind,
    ) -> impl Iterator<Item = &SignedMessage> {
        self.entries.iter().filter(move |m| {
            MessageKind::of(&m.payload) == kind && round_of(&m.payload) == Some(round)
        })
    }

    /// The rounds present in the log, ascending + de-duplicated.
    #[must_use]
    pub fn rounds(&self) -> Vec<u64> {
        let mut rs: Vec<u64> = self
            .entries
            .iter()
            .filter_map(|m| round_of(&m.payload))
            .collect();
        rs.sort_unstable();
        rs.dedup();
        rs
    }

    /// Map the log's messages to `tick` inputs (arrival order). The coordinator's own published
    /// outputs (`RoundOpen`/`RoundRecord`) stay in the stream — [`crate::replay`] treats
    /// `RoundRecord`s as the oracle and skips feeding both back to `tick` (see its docs). Callers that
    /// need clock-driven transitions interleave `Input::Clock` themselves (clocks are not messages).
    pub fn replay_inputs(&self) -> impl Iterator<Item = Input> + '_ {
        self.entries.iter().cloned().map(Input::Message)
    }

    /// Write the log as canonical, length-framed records (magic + run id, then per-record frames).
    pub fn write_to(&self, w: &mut impl Write) -> Result<(), ObserveError> {
        w.write_all(MAGIC).map_err(store)?;
        write_frame(w, self.run_id.as_bytes())?;
        for msg in &self.entries {
            let bytes = to_canonical_vec(msg).map_err(|e| ObserveError::Codec(e.to_string()))?;
            write_frame(w, &bytes)?;
        }
        Ok(())
    }

    /// Read a log back from framed bytes produced by [`MessageLog::write_to`].
    pub fn read_from(r: &mut impl Read) -> Result<Self, ObserveError> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).map_err(store)?;
        if &magic != MAGIC {
            return Err(ObserveError::Store("bad message-log magic".into()));
        }
        let run_id_bytes = read_frame(r)?
            .ok_or_else(|| ObserveError::Store("truncated log: missing run id".into()))?;
        let run_id = String::from_utf8(run_id_bytes)
            .map_err(|e| ObserveError::Store(format!("run id is not utf-8: {e}")))?;
        let mut entries = Vec::new();
        while let Some(frame) = read_frame(r)? {
            let msg: SignedMessage =
                from_canonical_slice(&frame).map_err(|e| ObserveError::Codec(e.to_string()))?;
            entries.push(msg);
        }
        Ok(Self { run_id, entries })
    }
}

fn store(e: std::io::Error) -> ObserveError {
    ObserveError::Store(e.to_string())
}

fn write_frame(w: &mut impl Write, bytes: &[u8]) -> Result<(), ObserveError> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| ObserveError::Store("record exceeds u32 frame length".into()))?;
    w.write_all(&len.to_le_bytes()).map_err(store)?;
    w.write_all(bytes).map_err(store)?;
    Ok(())
}

/// Read one length-framed record, or `None` at a clean end-of-stream (frame boundary).
fn read_frame(r: &mut impl Read) -> Result<Option<Vec<u8>>, ObserveError> {
    let mut len_bytes = [0u8; 4];
    match r.read_exact(&mut len_bytes) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(store(e)),
    }
    let len = u32::from_le_bytes(len_bytes) as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(store)?;
    Ok(Some(buf))
}
