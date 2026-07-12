// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-swarm-e2e` — the swarm-training end-to-end test target (spec §6.4, §19.5).
//!
//! This crate exists only to host the integration test under `tests/`: N in-process peers driven
//! by the `daemon_swarm_run::harness` scripted coordinator through the full round protocol. It is
//! the **P0 milestone** test at Merge 2, at which point the TEST-ONLY scripted coordinator is
//! replaced by the real `daemon-swarm-coordinator` tick loop (`// MERGE-2`). The library target is
//! intentionally empty (a workspace-member package needs a target; `tests/*` is a workspace glob,
//! so this crate is picked up with no root `Cargo.toml` edit).
