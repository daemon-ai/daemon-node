// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-rooms` — the internal Rooms transport (see `daemon-rooms-spec.md`).
//!
//! A **Room/Chat** is a first-class N-participant conversation backed by an **internal loopback
//! transport**: structurally identical to [`daemon-matrix`](daemon_matrix) but the "homeserver" is
//! the daemon itself. A DM / session-to-session conversation is a 2-participant Room; a group chat is
//! an N-participant Room; the user observes (and may participate) as a `Spectator`. There is no second
//! engine and no DB-as-IPC — the [`adapter::RoomRuntime`] drives the same `Arc<dyn NodeApi>` as an
//! in-process client, exactly like the Matrix adapter, reusing the transport-agnostic halves:
//!
//! - **Inbound** ([`inbound`]): a post into a Room maps to `OriginScope::Group { chat: room_id }` under
//!   the room's loopback `TransportId("room/<id>")`; the runtime fans it out to each member session via
//!   `submit_from`, the command chosen per the floor-control decision (`StartTurn` if admitted, else
//!   `Observe`).
//! - **Outbound**: the runtime subscribes each member session's merged log; a member's
//!   `TurnFinished.final_text` is gated through the Room's [`policy`] floor control, appended to the
//!   merged Room transcript (a verifiable journal stream), and re-injected as the others' next post.
//!
//! The only genuinely novel logic is the [`policy`] floor control (whose turn it is) plus a turn
//! budget that prevents echo storms. Everything else — routing, session derivation, delivery targets,
//! handover, observability — is the existing substrate applied to a loopback transport.

#![forbid(unsafe_code)]

pub mod adapter;
pub mod config;
pub mod inbound;
pub mod policy;

pub use adapter::RoomsAdapter;
pub use config::RoomsConfig;
pub use inbound::{Membership, RoomInbound};
pub use policy::{FloorControl, TurnBudget};

// The live room loop (membership load, floor control, transcript, fan-out, re-injection) lives in
// [`adapter::RoomRuntime`], built inside [`RoomsAdapter::serve`] from the `api` handed to it (so the
// adapter struct itself never holds an `Arc<dyn NodeApi>`, avoiding a registry<->adapter cycle).
