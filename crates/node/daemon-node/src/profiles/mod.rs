// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Role-profile construction + per-session engine resolution for the composition root.
//!
//! - [`dress`](dress::dress) applies the node's shared §10/§11 subsystem stores, credentials, and
//!   core toolset onto each role [`EngineProfile`](daemon_core::EngineProfile).
//! - [`registry`] builds the session/background tool registries and the §20 tunables overlay.
//! - [`resolve`] is the one [`SessionFactoryCtx`](resolve::SessionFactoryCtx) resolution path shared
//!   by the live session surface and the durable rehydration resolver.

pub(crate) mod dress;
pub(crate) mod registry;
pub(crate) mod resolve;
