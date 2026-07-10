// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-demo` — the in-process **demo** messaging transport (work package N5).
//!
//! A fully self-contained adapter + interactive-auth factories that exercise the ENTIRE v38
//! surface so the GUI/TUI apps and e2e scenarios can run the complete integrations experience
//! against a real node with **zero external network**:
//!
//! - **[`DemoAdapter`]** — a [`MessagingProtocol`](daemon_api::MessagingProtocol) presenting a
//!   deterministic seeded roster (varied presence) and the full conversation tree shape (a
//!   [`Space`](daemon_api::ConversationType::Space) with child channels, a standalone channel, DMs,
//!   a group DM), with live two-way chat: each `ConvSend` is journaled through the node's
//!   [`LifecycleSink`](daemon_api::LifecycleSink) and a scripted contact reply follows through the
//!   same seam. Its `account_schema` + `validate_account` exercise the N2 configure path.
//! - **[`demo_auth_factories`]** — one interactive-auth factory per
//!   [`AuthFlowKind`](daemon_api::AuthFlowKind) variant, covering every
//!   [`AuthChallenge`](daemon_api::AuthChallenge) shape (see [`auth`] for the full flow map + the
//!   documented demo credentials `demo`/`demo123`).
//!
//! No account secrets, no store, no credentials — everything is a pure function of in-crate seed
//! data. Depends only on the contracts (+ the host auth seam) + the async runtime, mirroring how
//! `daemon-rooms` stays a self-contained in-process adapter.
//!
//! ### The wire contract is untouched
//!
//! The demo uses only existing DTOs/ops. One observation (not a blocker): the wire
//! [`ContactInfo`](daemon_api::ContactInfo) models no avatar field, so a contact's "avatar-ish"
//! metadata is represented via its [`Presence`](daemon_api::Presence) decorations (status message +
//! mood emoji). This is cosmetic and does not warrant a wire change.

#![forbid(unsafe_code)]

pub mod adapter;
pub mod auth;
pub mod config;
pub mod seed;

pub use adapter::{DemoAdapter, FAMILY, VALIDATE_REJECT_VALUE};
pub use auth::{demo_auth_factories, DEMO_OTP_CODE, DEMO_PASSWORD, DEMO_USERNAME};
pub use config::DemoConfig;
