// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-transport` — wire transport for cross-node management traffic (phase 6c).
//!
//! When a placement cut puts a child host on another node, the management protocol runs over a real
//! socket rather than the in-process [`daemon_provision::CutChannel`]. This crate is the network
//! analogue of the placement-cut proxy: a length-framed TCP transport whose every frame is a
//! [`Wire`] envelope `{ wire_version, trace, body }`. The `trace` is stamped from the sender's
//! task-local [`TraceId`] and **restored** on decode (elfo's network path), so logs, spans, and the
//! verifiable journal correlate across the node boundary exactly as they do across a cut.
//!
//! Two roles (the `remote` feature):
//! - [`RemoteHost`] — the server. It holds the authoritative [`SessionStore`] and a hosted
//!   [`ManagedUnit`], and serves three things over the socket: a version handshake, *driving the
//!   unit through a turn*, and a **cross-node lease/fence handshake** (acquire a `FenceToken`, then
//!   commit under it). The authoritative store rejects a stale remote owner's commit — fencing
//!   holds across the node boundary, just as it holds across a cut.
//! - [`RemoteClient`] — the client. Sequential request/reply RPC; each call stamps the current
//!   trace and restores the peer's trace from the reply.
//!
//! True multi-node clustering / rebalancing stays deferred (doc: "when fleets-of-fleets is real");
//! this is the single process-pair proof that the trace and the fence both survive a network hop.

#![forbid(unsafe_code)]

#[cfg(feature = "remote")]
mod remote;

#[cfg(feature = "remote")]
pub use remote::{DriveOutcome, RemoteClient, RemoteHost, TransportError};
