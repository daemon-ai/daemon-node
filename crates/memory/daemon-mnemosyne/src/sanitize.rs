//! Content sanitizer — port of `content_sanitizer.py`.
//!
//! Detects oversized / base64 / high-entropy payloads and spills them to a blob store, returning a
//! redacted placeholder plus metadata. The current Rust port is intentionally pass-through until the
//! Mnemosyne blob path is wired; callers must not assume `SIZE_HARD_CAP` is enforced here yet.

use serde_json::{json, Value};

/// Size hard cap in bytes (`content_sanitizer.py` L21).
pub const SIZE_HARD_CAP: usize = 1_000_000;

/// Sanitize content, returning `(content, metadata)` (`content_sanitizer.py` L103-L169).
/// Current behavior: returns the content unchanged with empty metadata.
pub fn sanitize_content(content: &str) -> (String, Value) {
    (content.to_string(), json!({}))
}
