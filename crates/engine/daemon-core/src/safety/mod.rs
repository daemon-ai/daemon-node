//! Tool safety primitives (§9) shared across the network-facing tool crates.
//!
//! Currently the [`url`] egress guard ([`check_url`](url::check_url)); future siblings (path
//! containment beyond [`crate::exec`], threat-pattern matching) land here so every tool crate
//! enforces the same policy rather than re-deriving it.

pub mod url;

pub use url::{check_url, CheckedUrl, UrlReject};
