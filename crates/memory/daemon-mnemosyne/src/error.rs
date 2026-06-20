//! Crate error type.

/// The crate result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// A Mnemosyne engine error.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A SQLite error from the storage layer.
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A JSON (de)serialization error.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// An I/O error (bank directory creation, blob spill).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// An embedding backend error.
    #[error("embedding: {0}")]
    Embedding(String),
    /// An invalid argument (e.g. a bad bank name).
    #[error("invalid: {0}")]
    Invalid(String),
}
