// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE PHASE-8 CONTROL-SURFACE GATE (`daemon-workspace-layout.md` §7 phase-8 gate). The node is
//! assembled exactly as `bins/daemon` does (durable substrate + fleet-as-job-worker + the live
//! session surface) and driven through the one [`daemon_api`] surface over two transports: the
//! in-process trait call and the Unix socket. The gate proves the surface is transport-agnostic:
//! a session assigned over the socket is driven to `Completed` by the real `FleetRuntime` job
//! worker, the fleet usage folds in, and the in-process and socket reads agree.
//!
//! The session sub-surface's cross-language twin (the C FFI driving `StartTurn -> TurnFinished`)
//! is proven by the `bindings/daemon-core-ffi` C harness, not here.

mod auth_transport;
mod cron;
mod delivery_memory;
mod events_transport;
mod fs_content;
mod harness;
mod history;
mod ingest;
mod messaging;
mod profiles_skills;
mod routing_auth;
mod sessions;
mod tools_hitl;
mod tree_roster;
