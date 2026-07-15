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

pub mod driver;
pub mod event;
pub mod journal;

pub use driver::{start_run, PumpHandle, RunEnd, RunIdentity, V2Error, V2Run, V2RunConfig};
pub use event::{decode_event_frame, encode_event_frame, EventCodecError, EventV2, PayloadMeta};
pub use journal::{JournalSink, MemorySink, SinkEntry, SinkError};
