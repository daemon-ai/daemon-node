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

mod access_control;
mod account_management;
mod auth_transport;
mod bootstrap_wiring;
mod cron;
mod daemon_cloud_e2e;
mod delivery_memory;
mod demo_transport;
mod detached_delegation;
mod events_transport;
mod f3f4_ownership;
mod feedback;
mod fs_content;
mod harness;
mod history;
mod http_ownership;
mod ingest;
mod ingress_governor;
mod live_agent_e2e;
mod messaging;
mod negative_auth;
mod operator_steer;
mod ownership;
mod ownership_matrix;
mod ownership_transport;
mod positive_e2e;
mod presence;
mod process_notify;
mod profiles_skills;
mod provider_discovery;
mod revocation_transport;
mod routing_auth;
mod routing_hot_reload;
mod rung1_revs;
mod rung2_deltas;
mod rung3_idempotency;
mod semantic_search;
mod session_recall;
mod sessions;
mod tools_hitl;
mod transport_configure;
mod tree_roster;
mod web_serve;
mod wire_client;
mod ws_transport;
