// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The major-2 (event-loop) driver surface — Phase A2.
//!
//! - [`event`] — the canonical event-frame codec (ABI §4.2/§5.1/§5.2) the driver, the session
//!   event pump, and the journal/replay verifier all consume.
//! - [`journal`] — the [`journal::JournalSink`] seam (§8): the driver is born audited through it;
//!   the concrete crash-safe store is A1's `daemon-vhc-observe::journal`, adapted by whoever links
//!   both (dependency direction: observe → host, never the reverse).
//! - [`driver`] — the event-loop driver itself: the dedicated guest thread, the blocking
//!   `next_event` bridge, `publish` under the §12.1 signed-frame envelope, timers, staging +
//!   `read_back`, per-slice budgets, and `da_init`/`da_run` dispatch.

//! - [`admission`] — `claim()` evaluation in the restricted assessment instance and the
//!   owner-bracketed five-stage funnel (architecture §3.5; ABI §9).

pub mod admission;
pub mod buffer;
pub mod completion;
pub mod custom_op;
pub mod driver;
pub mod event;
pub mod journal;
pub mod ops;
pub mod replay;
pub mod state_store;
pub mod streams;

pub use admission::{
    admit, apply_admitted_quotas, apply_state_grant_bounds, Admission, DeviceProfile,
    EnvelopeRoleGrants, FunnelRefusal, MemoryClaim, OwnerPolicy, ParticipationLane, TierBytes,
    STATE_RETAIN_ROOTS_GRANT,
};
pub use buffer::BufferTable;
pub use completion::{CompError, CompletionCodecError, CompletionResult, SuccessPayload};
pub use custom_op::CustomOpRegistry;
pub use driver::{
    start_run, start_run_migrating, DeliverVerdict, MigrationInput, OpOutcome, PumpHandle, Run,
    RunConfig, RunEnd, RunError, RunIdentity, SnapshotCapture, SpooledFrame,
};
pub use event::{decode_event_frame, encode_event_frame, EventCodecError, PayloadMeta, RunEvent};
pub use journal::{Dropped, JournalSink, MemorySink, SinkEntry, SinkError};
pub use ops::{OpRequest, OpTable};
pub use replay::{
    replay, replay_migrating, ReplayEnd, ReplayMigration, ReplayScript, ReplayedDecision,
    ReplayedRun,
};
pub use state_store::{
    SealedFold, StateStore, StateStoreConfig, StateStoreError, StateStoreStats,
    STATE_STREAM_ID_TOP_BIT,
};
pub use streams::StreamTable;
