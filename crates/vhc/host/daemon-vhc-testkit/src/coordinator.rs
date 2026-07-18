// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Re-export of the production coordinator drive (`daemon_vhc_host::coordinator`).
//!
//! The configuration seat + in-process drive were LIFTED out of this harness into the host crate
//! when the worker's self-driven join re-seated onto the coordinator: the ratified rule is
//! that consensus never runs outside the sandboxed, content-addressed module — so the drive is
//! production machinery, and the harness reuses it rather than keeping a duplicate copy. The
//! testkit-facing paths (`daemon_vhc_testkit::coordinator::*`, the crate-root re-exports)
//! are unchanged.

pub use daemon_vhc_host::coordinator::{
    authorize_coordinator_frame, configure_coordinator, coordinator_state_from_capture,
    frame_sender, refuse_unconfigurable_envelope, CoordError, Coordinator, CoordinatorSpec,
};
