// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Small shared helpers (ids, timestamps).

use sha2::{Digest, Sha256};

/// Current time as an RFC3339 / ISO-8601 string (matches Mnemosyne's ISO timestamps).
pub fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Current date as `YYYY-MM-DD` (mirrors Python's `datetime.now().isoformat()[:10]`, used as the
/// temporal grain for `triples.valid_from` / `valid_until`).
pub fn today_iso() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}

/// A 16-char SHA-256 prefix memory id (mirrors `beam.py`'s id derivation).
pub fn memory_id(content: &str) -> String {
    let mut h = Sha256::new();
    h.update(content.as_bytes());
    let hex = format!("{:x}", h.finalize());
    hex[..16].to_string()
}

/// The time-salted memory id used for fresh `working_memory` rows (`beam.py` `_generate_id` L1122:
/// `sha256(content + now.isoformat())[:16]`). Non-deterministic on purpose — exact-content
/// idempotency is provided by the dedup lookup, not the id.
pub fn generate_id(content: &str) -> String {
    memory_id(&format!("{content}{}", now_iso()))
}

/// Python `str(float)` formatting for event-hash preimages (`0.5` -> `"0.5"`, `1.0` -> `"1.0"`).
pub(crate) fn py_float(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

/// Strip closed `<think>...</think>` blocks some LLMs emit, then trim (`beam.py`
/// `consolidate_to_episodic` L3991-L3993, `re.DOTALL`).
pub fn strip_think(text: &str) -> String {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"(?s)<think>.*?</think>").unwrap());
    re.replace_all(text, "").trim().to_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn strip_think_removes_closed_blocks_only() {
        assert_eq!(
            super::strip_think("<think>a\nb</think> summary <think>x</think>"),
            "summary"
        );
        // An unclosed block is left alone (Python's regex only matches closed pairs).
        assert_eq!(super::strip_think("<think>dangling"), "<think>dangling");
    }
}
