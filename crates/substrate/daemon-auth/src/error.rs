// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The crate error type.

/// A `daemon-auth` result.
pub type Result<T> = std::result::Result<T, Error>;

/// What can go wrong opening the identity store or authenticating a principal.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A SQLite error from the identity store.
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A schema-migration error (the `user_version` ladder).
    #[error("sqlite migrate: {0}")]
    Migrate(#[from] rusqlite_migration::Error),
    /// A password hash could not be produced or parsed.
    #[error("password hash: {0}")]
    PasswordHash(String),
    /// The OS CSPRNG failed while minting a token.
    #[error("entropy: {0}")]
    Entropy(String),
    /// No such user / token / row.
    #[error("not found")]
    NotFound,
    /// Username or password did not verify (kept deliberately opaque to the caller).
    #[error("invalid credentials")]
    InvalidCredentials,
    /// The account exists but has been disabled by an administrator.
    #[error("account disabled")]
    Disabled,
}
