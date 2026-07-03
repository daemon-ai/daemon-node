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
