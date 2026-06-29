// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The crate error type.

/// A `daemon-context-lcm` result.
pub type Result<T> = std::result::Result<T, Error>;

/// What can go wrong opening or driving the LCM context engine.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A SQLite error from the summary store.
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A filesystem error (e.g. creating the data dir).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// A JSON (de)serialization error (e.g. node `source_ids`).
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}
