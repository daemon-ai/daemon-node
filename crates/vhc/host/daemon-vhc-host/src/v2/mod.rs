// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The major-2 (event-loop) driver surface — Phase A2.
//!
//! Landed skeleton-first: [`event`] is the canonical event-frame codec (ABI §4.2/§5.1/§5.2) the
//! v2 driver, the session event pump, and the journal/replay verifier all consume — pinned by
//! tests before the driver that speaks it exists. The `da_run`/`next_event` driver itself wires in
//! next (it flips [`daemon_vhc_abi::HOST_IMPLEMENTED_MAJORS`] to `[1, 2]` when it is real; until
//! then a well-formed major-2 module keeps refusing `AbiUnsupportedMajor` at selection, unchanged
//! from A0).

pub mod event;

pub use event::{decode_event_frame, encode_event_frame, EventCodecError, EventV2, PayloadMeta};
