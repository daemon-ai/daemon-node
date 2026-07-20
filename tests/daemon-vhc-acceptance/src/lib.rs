// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-acceptance` — the multi-process acceptance suite's crate root.
//!
//! The library target is intentionally empty (a workspace-member package needs a target;
//! `tests/*` is a workspace glob, so this crate is picked up with no root `Cargo.toml` edit).
//! Everything lives under `tests/`: the three-real-node-process harness and the required
//! acceptance gates, plus the vendored corpus fixture under `fixtures/`.
