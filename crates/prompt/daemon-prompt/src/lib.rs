// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Prompt content + stores for the node-owned system-prompt composition (hermes parity, Phase 2).
//!
//! This crate is the *content* half of the prompt architecture: it produces plain `String` /
//! `Option<String>` blocks and owns the persona / user-profile file stores. It deliberately knows
//! nothing about prompt slots, composition order, or caching — binding these sources into the
//! engine's composed prompt is the integration lane's job. The only daemon-core surface consumed
//! is the §13 [`ExecutionEnvironment`](daemon_core::ExecutionEnvironment) trait (all workspace
//! context-file IO goes through it, so remote/sandboxed backends work unchanged).
//!
//! Modules (each a port of the corresponding hermes-agent seam):
//!
//! - [`scan`] — the shared threat-pattern library (`tools/threat_patterns.py`): scope lattice
//!   `all` ⊂ `context` ⊂ `strict`, invisible-unicode detection, and the whole-content
//!   `[BLOCKED: ...]` replacement applied wherever untrusted text enters the system prompt.
//! - [`truncate`] — the 20k-char head/tail cap (`agent/prompt_builder.py::_truncate_content`).
//!
//! Cache discipline: every producer here is deterministic from its inputs (no clocks, no env
//! vars, no ambient config), so a caller that snapshots the outputs once per session gets a
//! byte-stable system prefix for free.

#![forbid(unsafe_code)]

pub mod scan;
pub mod truncate;

pub use scan::{
    first_threat_message, scan_context_content, scan_for_threats, Scope, INVISIBLE_CHARS,
};
pub use truncate::{truncate_content, CONTEXT_FILE_MAX_CHARS};

/// Errors from the persona / user-profile stores and the revision log.
#[derive(Debug, thiserror::Error)]
pub enum PromptError {
    /// A write was rejected because the content is empty after trimming.
    #[error("content is empty")]
    Empty,
    /// A write was rejected by the threat scanner (the message names the pattern).
    #[error("{0}")]
    Blocked(String),
    /// A write was rejected because the content exceeds the store's character cap.
    #[error("content is {len} chars, over the {cap}-char cap")]
    OverCap {
        /// The rejected content's character count.
        len: usize,
        /// The store's configured cap.
        cap: usize,
    },
    /// An underlying filesystem failure.
    #[error("io: {0}")]
    Io(String),
    /// A revision-log line failed to (de)serialize.
    #[error("codec: {0}")]
    Codec(String),
}

impl From<std::io::Error> for PromptError {
    fn from(e: std::io::Error) -> Self {
        PromptError::Io(e.to_string())
    }
}
