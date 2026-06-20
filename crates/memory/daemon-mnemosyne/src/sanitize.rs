//! Content sanitizer — port of `content_sanitizer.py`.
//!
//! Detects oversized / base64 / high-entropy payloads and spills them to a blob store, returning a
//! stub + metadata. Scaffold: pass-through (no extraction) until the blob path is ported.

use serde_json::{json, Value};

/// Size hard cap in bytes (`content_sanitizer.py` L21).
pub const SIZE_HARD_CAP: usize = 1_000_000;

/// Sanitize content, returning `(content, metadata)` (`content_sanitizer.py` L103-L169).
/// Scaffold: returns the content unchanged with empty metadata.
pub fn sanitize_content(content: &str) -> (String, Value) {
    // TODO: data-URI base64 extraction, size cap, Shannon-entropy (>5.0) blob spill.
    (content.to_string(), json!({}))
}
