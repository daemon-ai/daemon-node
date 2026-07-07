// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-telegram` — the Telegram chat transport (Phase 1 of the multi-protocol messaging plan).
//!
//! Structurally the twin of `daemon-matrix`: an in-process `NodeApi`-client adapter that isolates
//! the `grammers` MTProto SDK and drives *our* engine as a client. One instance hosts N accounts;
//! each account is a transport instance (`telegram/<id>`) with its own grammers client + on-disk
//! session store. Inbound: a Telegram message is normalised into an [`daemon_ingest::Reception`] and
//! handed to the reusable `daemon-ingest` gate; outbound: replies flow back through a
//! `daemon-delivery` [`daemon_delivery::Projector`]. No grammers/MTProto type leaves this crate —
//! every SDK use is confined to [`client`].

#![forbid(unsafe_code)]

mod account;
pub mod adapter;
pub mod auth;
pub mod client;
pub mod config;
mod inbound;
mod mapping;
mod outbound;

use std::collections::HashMap;
use std::sync::Arc;

use daemon_protocol::TransportId;

pub use account::{AccountMode, StoredSession};
pub use adapter::{TelegramAdapter, TelegramClient};
pub use auth::{LoginBackend, TelegramAuthFlowFactory};
pub use config::{TelegramConfig, TelegramRoute};

/// The transport family this adapter provisions (`AccountProvisioning::bound_accounts`).
pub(crate) const FAMILY: &str = "telegram";

/// The shared registry of live, session-restored clients keyed by their instance-qualified
/// transport id (`telegram/<id>`). Populated by [`serve`](client::serve) at bring-up and read by the
/// `MessagingProtocol` feature-trait method bodies (which only hold `&self`, so they recover the
/// account's client from here). A `tokio::sync::RwLock` (not std) so a read in an `async` verb body
/// never blocks the runtime, and the bring-up writer never holds a guard across an await.
pub(crate) type LiveClients =
    Arc<tokio::sync::RwLock<HashMap<TransportId, Arc<dyn TelegramClient>>>>;
