// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The coordinator-oracle migration onto the journal substrate (refactor §5 A1).
//!
//! A1 generalizes `daemon-vhc-observe`'s in-memory capture — the [`RunCapture`](crate::RunCapture)
//! (`initial` [`CoordinatorState`] + the exact ordered `tick` [`Input`] trace, messages **and**
//! clocks) — onto the crash-safe segmented [`Journal`]. The existing replay oracle
//! ([`crate::replay`]) and its tests are **pinned**: this module is an *adapter* that writes a
//! capture onto the journal and reads it back into the exact same driving-input sequence, so
//! [`replay_over_journal`] returns a byte-identical [`ReplayReport`] to
//! [`replay_capture`](crate::replay::replay_capture) over the in-memory path. The in-memory
//! `RunCapture`/`MessageLog` types are retained (their on-disk framing is separately pinned); the
//! journal is the durable substrate the same oracle now runs over.
//!
//! ### The mapping (documented — an ABI §8 record per coordinator input)
//!
//! The coordinator is a reactive state machine over its inputs (architecture §4.1: "the coordinator
//! is a module"), so each `tick` input is exactly a thing it could branch on — an ABI §8 journal
//! record:
//! * [`Input::Clock`] → a **clock record** (tag 3), per §6.5 / the coordinator-oracle lesson
//!   ("clocks are not messages but must be captured").
//! * [`Input::Message`] / [`Input::Control`] → an **event record** (tag 1) whose `frame` is the
//!   canonical CBOR of the `Input` (externally-tagged, so `Message` vs `Control` is unambiguous on
//!   read-back). `at` carries the input's position (the coordinator branches on `Clock`, not `at`).
//!
//! The initial [`CoordinatorState`] is journaled once as a **snapshot record** (tag 10, "verbatim
//! accepted state-manifest bytes") — it is the state the replay restores from, exactly §10.2's role.

use daemon_vhc_coordinator::{CoordinatorState, Input};
use daemon_vhc_proto::{from_canonical_slice, to_canonical_vec};

use crate::capture::RunCapture;
use crate::log::{MessageKind, MessageLog};
use crate::replay::{replay_from_state, ReplayError, ReplayReport};
use crate::ObserveError;

use super::record::{Body, ClockRec, EventRec, ExecIdentity, RunHeader, SnapshotRec};
use super::sidecar::KeyProvider;
use super::store::Journal;
use super::JournalError;

fn codec_err(e: impl core::fmt::Display) -> ObserveError {
    ObserveError::Codec(e.to_string())
}

fn journal_err(e: JournalError) -> ObserveError {
    ObserveError::Store(e.to_string())
}

/// Write a run-header (tag 0) describing this coordinator journal's execution identity (§8.1/§8.3).
///
/// The admitted-value byte fields carry the coordinator's envelope/config where available; the
/// oracle migration only requires the execution identity + format, so empty admitted bytes are
/// permitted for a pure in-process capture.
///
/// # Errors
/// [`ObserveError`] on encode/write failure.
pub fn record_run_header<K: KeyProvider + Clone>(
    journal: &mut Journal<K>,
    id: &ExecIdentity,
    config: Vec<u8>,
) -> Result<u64, ObserveError> {
    let header = RunHeader {
        run_id: id.run_id,
        epoch: id.epoch,
        role: id.role.clone(),
        instance: id.instance,
        module: id.module,
        abi: u64::from(daemon_vhc_abi::DA_ABI_VERSION),
        worlds: std::collections::BTreeMap::new(),
        bridge: false,
        manifest: Vec::new(),
        config,
        grants: Vec::new(),
        claim: Vec::new(),
        channels: Vec::new(),
        device: Vec::new(),
        format: u64::from(super::format_version()),
    };
    // The run header is a durable admission fact → cross a commit barrier (§8.4-style).
    journal
        .append_committed(Body::RunHeader(header))
        .map_err(journal_err)
}

/// Journal the initial [`CoordinatorState`] as a snapshot record (tag 10).
///
/// # Errors
/// [`ObserveError`] on encode/write failure.
pub fn record_initial_state<K: KeyProvider + Clone>(
    journal: &mut Journal<K>,
    initial: &CoordinatorState,
) -> Result<u64, ObserveError> {
    let manifest = to_canonical_vec(initial).map_err(codec_err)?;
    journal
        .append_committed(Body::Snapshot(SnapshotRec { manifest }))
        .map_err(journal_err)
}

/// Journal one coordinator `tick` input (clock → tag 3; message/control → tag 1). `at` is the
/// input's position in the trace.
///
/// # Errors
/// [`ObserveError`] on encode/write failure.
pub fn record_input<K: KeyProvider + Clone>(
    journal: &mut Journal<K>,
    at: u64,
    input: &Input,
) -> Result<u64, ObserveError> {
    let body = match input {
        Input::Clock(now) => Body::Clock(ClockRec { now: *now }),
        Input::Message(_) | Input::Control(_) => {
            let frame = to_canonical_vec(input).map_err(codec_err)?;
            Body::Event(EventRec { at, frame })
        }
    };
    journal.append(body).map_err(journal_err)
}

/// Record a whole [`RunCapture`] onto a fresh journal: the run header, the initial state, then every
/// driving input in order (the migration of `RunCapture` onto the §8 substrate).
///
/// # Errors
/// [`ObserveError`] on encode/write failure.
pub fn record_capture<K: KeyProvider + Clone>(
    journal: &mut Journal<K>,
    id: &ExecIdentity,
    capture: &RunCapture,
) -> Result<(), ObserveError> {
    record_run_header(journal, id, Vec::new())?;
    record_initial_state(journal, &capture.initial)?;
    for (i, input) in capture.inputs.iter().enumerate() {
        record_input(journal, i as u64, input)?;
    }
    // Durable end-of-capture barrier (the last inputs may otherwise be written-but-uncommitted).
    journal.commit().map_err(journal_err)?;
    Ok(())
}

/// Reconstruct the captured `(initial_state, driving_inputs)` from a journal — the inverse of
/// [`record_capture`], reading records back across every segment (chain-verified).
///
/// # Errors
/// [`ObserveError`] if the journal lacks a snapshot record or a record fails to decode.
pub fn recover_capture<K: KeyProvider + Clone>(
    journal: &Journal<K>,
) -> Result<(CoordinatorState, Vec<Input>), ObserveError> {
    let records = journal.read_all_records().map_err(journal_err)?;
    let mut initial: Option<CoordinatorState> = None;
    let mut inputs = Vec::new();
    for record in records {
        match record.body {
            Body::Snapshot(SnapshotRec { manifest }) => {
                if initial.is_none() {
                    initial = Some(from_canonical_slice(&manifest).map_err(codec_err)?);
                }
            }
            Body::Clock(ClockRec { now }) => inputs.push(Input::Clock(now)),
            Body::Event(EventRec { frame, .. }) => {
                let input: Input = from_canonical_slice(&frame).map_err(codec_err)?;
                inputs.push(input);
            }
            // The run header + any other records are not part of the driving trace.
            _ => {}
        }
    }
    let initial = initial.ok_or_else(|| {
        ObserveError::Replay("journal has no snapshot record (initial coordinator state)".into())
    })?;
    Ok((initial, inputs))
}

/// Replay the coordinator oracle over journal-backed capture (refactor §5 A1 acceptance).
///
/// Semantically identical to [`replay_capture`](crate::replay::replay_capture): reconstruct the
/// captured initial state + driving inputs from the journal, append the wire log's `RoundRecord`s as
/// the oracle, and re-run `tick`. Returns a byte-identical [`ReplayReport`], proving the pinned oracle
/// behavior is preserved over the new substrate.
///
/// # Errors
/// [`ReplayError`] on a divergence or setup failure.
pub fn replay_over_journal<K: KeyProvider + Clone>(
    journal: &Journal<K>,
    oracle: &MessageLog,
) -> Result<ReplayReport, ReplayError> {
    let (initial, inputs) = recover_capture(journal).map_err(ReplayError::Setup)?;
    let oracle_records: Vec<Input> = oracle
        .by_kind(MessageKind::RoundRecord)
        .cloned()
        .map(Input::Message)
        .collect();
    replay_from_state(initial, inputs.into_iter().chain(oracle_records))
}
