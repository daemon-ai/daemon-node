// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-tool-browser` — the `browser` chat tool: a headless Chromium driven over the Chrome
//! DevTools Protocol (chromiumoxide), exposed as a single tagged-`op` `daemon_core::Tool`.
//!
//! The heavy CDP bindings (~60K generated LOC) live behind the `cdp` feature, which is **off by
//! default** so the standard workspace build/test never compiles them; `bins/daemon` turns it on via
//! its own `browser` feature. Without `cdp` this crate is empty (it compiles in a blink), so the
//! workspace gate stays cheap.
//!
//! A [`BrowserSupervisor`] owns the lazily-launched browser process (respawn on fault + crash-loop
//! breaker, mirroring the `daemon-metta` coprocessor); the [`BrowserTool`] drives one working page:
//! `navigate` (egress-checked + optionally approval-gated), `extract` (readability Markdown / text /
//! HTML, returned as untrusted data), `click`/`type`/`press_key`/`wait_for`, `screenshot`, `eval`,
//! `reload`/`back`/`close`.

#![forbid(unsafe_code)]
// Phase 4: test code may use raw fs/reqwest/Command; the --lib pass still guards production.
#![cfg_attr(test, allow(clippy::disallowed_methods, clippy::disallowed_types))]

#[cfg(feature = "cdp")]
mod error;
#[cfg(feature = "cdp")]
mod supervisor;
#[cfg(feature = "cdp")]
mod tool;

#[cfg(feature = "cdp")]
pub use error::BrowserError;
#[cfg(feature = "cdp")]
pub use supervisor::{BrowserSettings, BrowserSupervisor, ExtractFormat};
#[cfg(feature = "cdp")]
pub use tool::{BrowserTool, NavApproval};
