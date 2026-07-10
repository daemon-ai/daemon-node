// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Resolved Demo adapter configuration.
//!
//! Like the [`RoomsConfig`](daemon_rooms) / [`MatrixConfig`](daemon_matrix) precedent the *shape*
//! lives here and the binary owns the parsing (`bins/daemon`'s `config.rs`, the `[demo]` table /
//! `DAEMON_DEMO__*` env). The demo transport reaches nothing external — no accounts, no secrets, no
//! store — so this config only toggles whether it is spawned and sets the scripted-reply cadence.

use serde::{Deserialize, Serialize};

/// The resolved `[demo]` config the host hands to [`crate::DemoAdapter`]. `enabled = false`
/// (default) leaves the demo transport off, exactly like `[rooms].enabled` / `[matrix].enabled`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DemoConfig {
    /// Whether the demo transport (adapter + auth factories) is registered at all. `false`
    /// (default) leaves it off.
    #[serde(with = "daemon_common::flex_bool")]
    pub enabled: bool,
    /// How long (milliseconds) after each `ConvSend` the scripted contact reply arrives (the live
    /// two-way traffic every chat UI sees). Small by default so an interactive demo feels live; the
    /// per-account `reply_delay_ms` setting (the `account_schema` field) is the operator-facing knob
    /// with the same meaning.
    pub reply_delay_ms: u64,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            reply_delay_ms: 40,
        }
    }
}
