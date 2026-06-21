//! The crate error type, classified so the node surface can map it onto [`daemon_common`] /
//! `daemon_api::ApiError` without leaking HTTP/IO specifics.

use std::path::PathBuf;

/// Why a model-management operation failed.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// A Hugging Face Hub HTTP request failed (network, DNS, TLS, non-2xx).
    #[error("hugging face request failed: {0}")]
    Http(String),

    /// The Hub response could not be parsed into the expected shape.
    #[error("decoding hugging face response: {0}")]
    Decode(String),

    /// The repo / revision / file does not exist (HTTP 404).
    #[error("not found: {0}")]
    NotFound(String),

    /// The repo is gated or requires authentication (HTTP 401/403) — set a token.
    #[error("access denied (gated or auth required): {0}")]
    AccessDenied(String),

    /// A download failed (transfer, integrity, or `hf-hub` error).
    #[error("download failed: {0}")]
    Download(String),

    /// A local filesystem operation failed.
    #[error("io error at {path}: {source}")]
    Io {
        /// The path the failing operation touched.
        path: PathBuf,
        /// The underlying IO error.
        #[source]
        source: std::io::Error,
    },

    /// The download/integrity preflight rejected the artifact (bad GGUF magic, too large for disk).
    #[error("integrity check failed: {0}")]
    Integrity(String),

    /// No such download job / installed model.
    #[error("unknown id: {0}")]
    Unknown(String),

    /// The request was malformed (e.g. an empty repo id, an unsupported source).
    #[error("invalid request: {0}")]
    Invalid(String),

    /// Any other failure.
    #[error("{0}")]
    Other(String),
}

impl ModelError {
    /// Wrap an IO error with the path it touched.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        ModelError::Io {
            path: path.into(),
            source,
        }
    }
}

/// Classify an `hf-hub` async API error into a [`ModelError`].
pub(crate) fn from_hf(err: hf_hub::api::tokio::ApiError) -> ModelError {
    use hf_hub::api::tokio::ApiError as E;
    match err {
        E::RequestError(e) => {
            if let Some(status) = e.status() {
                match status.as_u16() {
                    404 => return ModelError::NotFound(e.to_string()),
                    401 | 403 => return ModelError::AccessDenied(e.to_string()),
                    _ => {}
                }
            }
            ModelError::Http(e.to_string())
        }
        E::IoError(e) => ModelError::Download(format!("io: {e}")),
        E::LockAcquisition(p) => {
            ModelError::Download(format!("another download holds the lock on {}", p.display()))
        }
        other => ModelError::Download(other.to_string()),
    }
}

/// A convenient result alias.
pub type Result<T> = std::result::Result<T, ModelError>;
