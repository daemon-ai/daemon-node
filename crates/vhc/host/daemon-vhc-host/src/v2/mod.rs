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
pub mod driver;
pub mod event;
pub mod journal;
pub mod ops;
pub mod replay;

pub use admission::{
    admit_v2, AdmissionV2, DeviceProfile, FunnelRefusal, MemoryClaim, OwnerPolicy,
    ParticipationLane, TierBytes,
};
pub use buffer::BufferTable;
pub use completion::{CompError, CompletionCodecError, CompletionResult, SuccessPayload};
pub use driver::{
    start_run, DeliverVerdict, OpOutcome, PumpHandle, RunEnd, RunIdentity, V2Error, V2Run,
    V2RunConfig,
};
pub use event::{decode_event_frame, encode_event_frame, EventCodecError, EventV2, PayloadMeta};
pub use journal::{Dropped, JournalSink, MemorySink, SinkEntry, SinkError};
pub use ops::{OpRequest, OpTable};
pub use replay::{replay_v2, ReplayEnd, ReplayScript, ReplayedDecision, ReplayedRun};
