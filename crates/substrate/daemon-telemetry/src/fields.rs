// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Stable field and span names for operational tracing.
//!
//! These constants keep log pipelines and tests from depending on ad-hoc spelling at each call
//! site. The verifiable journal remains the audit source of truth; these names are for runtime
//! `tracing` spans/events.

/// The current [`daemon_common::TraceId`] rendered as fixed-width hex.
pub const TRACE_ID: &str = "trace_id";
/// The broad category of work represented by a span.
pub const SPAN_KIND: &str = "span.kind";
/// Durable session id.
pub const SESSION: &str = "session";
/// Supervision/unit id.
pub const UNIT: &str = "unit";
/// Request/correlation id on live protocols.
pub const REQ_ID: &str = "req_id";
/// Journal segment id.
pub const SEGMENT: &str = "segment";
/// Activation fence token.
pub const FENCE: &str = "fence";
/// Wire frame or enum variant name.
pub const FRAME: &str = "frame";
/// Wire/transport family.
pub const WIRE: &str = "wire";
/// Domain operation name.
pub const OPERATION: &str = "operation";
/// Lifecycle or recovery step.
pub const STEP: &str = "step";
/// Operation outcome.
pub const OUTCOME: &str = "outcome";

/// Common span names used by substrate/host boundaries.
pub mod span {
    pub const API_HTTP_REQUEST: &str = "api.http.request";
    pub const API_UNIX_REQUEST: &str = "api.unix.request";
    pub const CUT_RECV: &str = "cut.recv";
    pub const CUT_RUN_TURN: &str = "cut.run_turn";
    pub const CUT_STORE_BROKER: &str = "cut.store.broker";
    pub const CUT_CRED_BROKER: &str = "cut.cred.broker";
    pub const TRANSPORT_REQUEST: &str = "transport.request";
    pub const TRANSPORT_REPLY: &str = "transport.reply";
}

/// Common event names used by substrate/host boundaries.
pub mod event {
    pub const API_REQUEST: &str = "api.request";
    pub const CUT_FRAME_IN: &str = "cut.frame.in";
    pub const CUT_FRAME_OUT: &str = "cut.frame.out";
    pub const TRANSPORT_REQUEST: &str = "transport.request";
    pub const TRANSPORT_REPLY: &str = "transport.reply";
}
