//! Content sanitizer — port of `content_sanitizer.py`.
//!
//! Detects binary-shaped content (base64 data URIs, oversized payloads, high-entropy encoded blobs)
//! and spills it to a content-addressed blob store, replacing the in-row content with a stub and
//! recording a blob reference in metadata. Blobs live at
//! `{root}/{sha[..2]}/{sha[..4]}/{sha}` where `root` is `$MNEMOSYNE_BLOB_DIR` or
//! `~/.hermes/mnemosyne/blobs` (`content_sanitizer.py` L8, L32-L37).

use base64::Engine as _;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Size hard cap in bytes — always extract regardless of content type (`content_sanitizer.py` L22).
pub const SIZE_HARD_CAP: usize = 1_000_000;
/// Minimum size in bytes before the base64/entropy heuristic runs (`content_sanitizer.py` L23).
pub const SIZE_BASE64_CHECK: usize = 100_000;
/// Shannon entropy (bits/char) above which content is treated as an encoded blob
/// (`content_sanitizer.py` L24).
pub const ENTROPY_THRESHOLD: f64 = 5.0;

/// Inspect `content` for binary-shaped payloads and extract them to blob storage, returning
/// `(sanitized_content, blob_metadata)` — a verbatim port of `sanitize_content`
/// (`content_sanitizer.py` L103-L169). `blob_metadata` is an empty object when no extraction
/// occurred. Detection rules are checked in order: (1) `data:` URI, (2) size > 1 MB,
/// (3) size > 100 KB AND entropy > 5.0 bits/char.
///
/// On a blob-store I/O error the original content is returned unchanged (ingest is never broken by
/// a failed spill).
pub fn sanitize_content(content: &str) -> (String, Value) {
    let original_size = content.len();

    // Rule 1: data: URI -> decode the base64 payload and extract.
    if content.starts_with("data:") {
        if let Some((mime_type, raw_bytes)) = parse_data_uri(content) {
            if let Ok(sha256) = store_blob(&raw_bytes) {
                let meta = json!({
                    "blob_ref": format!("blob://sha256/{sha256}"),
                    "original_size": raw_bytes.len(),
                    "mime": mime_type,
                    "extraction_reason": "data_uri",
                });
                let placeholder = format!(
                    "[Binary content extracted: {mime_type}, {} bytes \u{2192} blob://sha256/{sha256}]",
                    thousands(raw_bytes.len()),
                );
                return (placeholder, meta);
            }
        }
    }

    // Rule 2: size hard cap.
    if original_size > SIZE_HARD_CAP {
        if let Ok(sha256) = store_blob(content.as_bytes()) {
            let meta = json!({
                "blob_ref": format!("blob://sha256/{sha256}"),
                "original_size": original_size,
                "extraction_reason": "size_cap",
            });
            let placeholder = format!(
                "[Large content extracted: {} bytes \u{2192} blob://sha256/{sha256}]",
                thousands(original_size),
            );
            return (placeholder, meta);
        }
    }

    // Rule 3: high-entropy (likely encoded blob).
    if original_size > SIZE_BASE64_CHECK && looks_like_base64_blob(content) {
        if let Ok(sha256) = store_blob(content.as_bytes()) {
            let entropy = round2(shannon_entropy(content));
            let meta = json!({
                "blob_ref": format!("blob://sha256/{sha256}"),
                "original_size": original_size,
                "entropy": entropy,
                "extraction_reason": "high_entropy",
            });
            let placeholder = format!(
                "[Encoded content extracted: {} bytes, entropy {entropy:.1} bits/char \u{2192} blob://sha256/{sha256}]",
                thousands(original_size),
            );
            return (placeholder, meta);
        }
    }

    (content.to_string(), json!({}))
}

/// Parse a `data:` URI into `(mime_type, raw_bytes)`, or `None` if it does not match / the payload
/// is not strict base64 (`content_sanitizer.py` L49-L60). Mirrors the regex
/// `^data:(mime)?(?:;base64)?,(payload)` (single line; `.` does not cross newlines).
fn parse_data_uri(content: &str) -> Option<(String, Vec<u8>)> {
    // Strip the `data:` scheme, then split header (mime[;base64]) from payload at the first comma,
    // restricted to the first line (the Python regex's `.` never matches a newline).
    let first_line = content.split('\n').next().unwrap_or(content);
    let rest = first_line.strip_prefix("data:")?;
    let (header, payload) = rest.split_once(',')?;
    let mime = header.split(';').next().unwrap_or("");
    let mime_type = if mime.is_empty() {
        "application/octet-stream".to_string()
    } else {
        mime.to_string()
    };
    // `base64::STANDARD` rejects non-alphabet characters, matching Python's `validate=True`.
    let raw = base64::engine::general_purpose::STANDARD
        .decode(payload.as_bytes())
        .ok()?;
    Some((mime_type, raw))
}

/// Whether `content` looks like a base64-encoded binary blob: at least `SIZE_BASE64_CHECK`
/// characters and Shannon entropy above `ENTROPY_THRESHOLD` (`content_sanitizer.py` L76-L88).
fn looks_like_base64_blob(content: &str) -> bool {
    if content.chars().count() < SIZE_BASE64_CHECK {
        return false;
    }
    shannon_entropy(content) > ENTROPY_THRESHOLD
}

/// Shannon entropy in bits per character over the string's code points
/// (`content_sanitizer.py` L63-L73).
fn shannon_entropy(text: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    let mut counts: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    let mut n = 0usize;
    for ch in text.chars() {
        *counts.entry(ch).or_insert(0) += 1;
        n += 1;
    }
    let n = n as f64;
    let mut entropy = 0.0;
    for &count in counts.values() {
        let p = count as f64 / n;
        entropy -= p * p.log2();
    }
    entropy
}

/// Store `raw_bytes` as a content-addressed blob, returning the sha256 hex digest
/// (`content_sanitizer.py` L91-L100). Idempotent: an existing blob is not rewritten.
fn store_blob(raw_bytes: &[u8]) -> std::io::Result<String> {
    let sha256 = sha256_hex(raw_bytes);
    let blob_dir = blob_root().join(&sha256[..2]).join(&sha256[..4]);
    std::fs::create_dir_all(&blob_dir)?;
    let blob_path = blob_dir.join(&sha256);
    if !blob_path.exists() {
        std::fs::write(&blob_path, raw_bytes)?;
    }
    Ok(sha256)
}

/// The content-addressed blob root: `$MNEMOSYNE_BLOB_DIR` or `~/.hermes/mnemosyne/blobs`
/// (`content_sanitizer.py` L32-L37).
fn blob_root() -> PathBuf {
    match std::env::var("MNEMOSYNE_BLOB_DIR") {
        Ok(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home)
                .join(".hermes")
                .join("mnemosyne")
                .join("blobs")
        }
    }
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Round to two decimal places (Python `round(x, 2)`).
fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Format an integer with `,` thousands separators (Python `f"{n:,}"`).
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let bytes = digits.as_bytes();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    let len = bytes.len();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_temp_blob_dir<T>(f: impl FnOnce() -> T) -> T {
        let dir = std::env::temp_dir().join(format!("mnemo-blob-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::env::set_var("MNEMOSYNE_BLOB_DIR", &dir);
        let out = f();
        std::env::remove_var("MNEMOSYNE_BLOB_DIR");
        let _ = std::fs::remove_dir_all(&dir);
        out
    }

    #[test]
    fn plain_text_passes_through() {
        let (content, meta) = sanitize_content("just a normal note about the deploy");
        assert_eq!(content, "just a normal note about the deploy");
        assert_eq!(meta, json!({}));
    }

    #[test]
    fn data_uri_is_extracted() {
        with_temp_blob_dir(|| {
            // "hello world" base64 -> aGVsbG8gd29ybGQ=
            let (content, meta) =
                sanitize_content("data:text/plain;base64,aGVsbG8gd29ybGQ=");
            assert!(content.starts_with("[Binary content extracted: text/plain, 11 bytes"));
            assert_eq!(meta["extraction_reason"], "data_uri");
            assert_eq!(meta["mime"], "text/plain");
            assert_eq!(meta["original_size"], 11);
            assert!(meta["blob_ref"].as_str().unwrap().starts_with("blob://sha256/"));
        });
    }

    #[test]
    fn invalid_data_uri_payload_falls_through() {
        // "hello" is not valid-length base64 -> rule 1 skipped, content unchanged.
        let (content, meta) = sanitize_content("data:text/plain,hello");
        assert_eq!(content, "data:text/plain,hello");
        assert_eq!(meta, json!({}));
    }

    #[test]
    fn oversized_content_spills_to_blob() {
        with_temp_blob_dir(|| {
            let big = "a".repeat(SIZE_HARD_CAP + 10);
            let (content, meta) = sanitize_content(&big);
            assert!(content.starts_with("[Large content extracted:"));
            assert_eq!(meta["extraction_reason"], "size_cap");
            assert_eq!(meta["original_size"], (SIZE_HARD_CAP + 10) as u64);
        });
    }

    #[test]
    fn high_entropy_blob_is_extracted() {
        with_temp_blob_dir(|| {
            // A long, near-uniform base64-ish string (> 100 KB) has entropy > 5.0.
            let alphabet: Vec<char> =
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
                    .chars()
                    .collect();
            let mut s = String::with_capacity(SIZE_BASE64_CHECK + 64);
            for i in 0..(SIZE_BASE64_CHECK + 64) {
                s.push(alphabet[i % alphabet.len()]);
            }
            let (content, meta) = sanitize_content(&s);
            assert!(content.starts_with("[Encoded content extracted:"), "got: {content}");
            assert_eq!(meta["extraction_reason"], "high_entropy");
            assert!(meta["entropy"].as_f64().unwrap() > ENTROPY_THRESHOLD);
        });
    }

    #[test]
    fn entropy_of_empty_is_zero() {
        assert_eq!(shannon_entropy(""), 0.0);
    }

    #[test]
    fn thousands_separator_matches_python() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(11), "11");
        assert_eq!(thousands(1_234), "1,234");
        assert_eq!(thousands(1_000_010), "1,000,010");
    }
}
