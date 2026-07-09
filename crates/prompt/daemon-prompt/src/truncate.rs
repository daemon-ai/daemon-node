// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Head/tail content truncation for prompt-bound sources — a port of hermes-agent
//! `agent/prompt_builder.py::_truncate_content`.
//!
//! Counts are Unicode scalar values (`chars`), not bytes, matching Python `len` semantics so the
//! ported cap/ratio assertions hold exactly.

/// The per-source character cap for context files (and the persona load path).
pub const CONTEXT_FILE_MAX_CHARS: usize = 20_000;
/// Fraction of the cap kept from the head on truncation.
pub const CONTEXT_TRUNCATE_HEAD_RATIO: f64 = 0.7;
/// Fraction of the cap kept from the tail on truncation.
pub const CONTEXT_TRUNCATE_TAIL_RATIO: f64 = 0.2;

/// Head/tail truncation with a marker in the middle.
///
/// Content at or under `max_chars` is returned unchanged. Over the cap, the head (70% of the
/// cap) and tail (20%) are kept around a marker naming `label` and the original size. The marker
/// text is byte-compatible with the hermes original so ported assertions transfer.
pub fn truncate_content(content: &str, label: &str, max_chars: usize) -> String {
    let total = content.chars().count();
    if total <= max_chars {
        return content.to_string();
    }
    let head_chars = (max_chars as f64 * CONTEXT_TRUNCATE_HEAD_RATIO) as usize;
    let tail_chars = (max_chars as f64 * CONTEXT_TRUNCATE_TAIL_RATIO) as usize;
    let head: String = content.chars().take(head_chars).collect();
    let tail: String = content.chars().skip(total - tail_chars).collect();
    format!(
        "{head}\n\n[...truncated {label}: kept {head_chars}+{tail_chars} of {total} chars. Use \
         file tools to read the full file.]\n\n{tail}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_content_unchanged() {
        let content = "short content";
        assert_eq!(
            truncate_content(content, "TEST.md", CONTEXT_FILE_MAX_CHARS),
            content
        );
    }

    #[test]
    fn exact_limit_unchanged() {
        let content = "x".repeat(CONTEXT_FILE_MAX_CHARS);
        assert_eq!(
            truncate_content(&content, "TEST.md", CONTEXT_FILE_MAX_CHARS),
            content
        );
    }

    #[test]
    fn long_content_truncated() {
        let content = "x".repeat(25_000);
        let result = truncate_content(&content, "TEST.md", CONTEXT_FILE_MAX_CHARS);
        assert!(result.chars().count() < content.chars().count());
        assert!(result.contains("truncated TEST.md"));
        assert!(result.contains("Use file tools to read the full file."));
    }

    #[test]
    fn truncation_keeps_head_and_tail() {
        let head_marker = "HEAD_MARKER_TEXT";
        let tail_marker = "TAIL_MARKER_TEXT";
        let content = format!("{head_marker}{}{tail_marker}", "x".repeat(25_000));
        let result = truncate_content(&content, "TEST.md", CONTEXT_FILE_MAX_CHARS);
        assert!(result.contains(head_marker));
        assert!(result.contains(tail_marker));
        // 70% head + 20% tail of the cap, plus the marker.
        assert!(result.contains("kept 14000+4000 of"));
    }

    #[test]
    fn multibyte_content_counts_chars_not_bytes() {
        // 'é' is 2 bytes; a byte-based cap would truncate this, a char-based one must not.
        let content = "é".repeat(CONTEXT_FILE_MAX_CHARS);
        assert_eq!(
            truncate_content(&content, "TEST.md", CONTEXT_FILE_MAX_CHARS),
            content
        );
        let over = "é".repeat(CONTEXT_FILE_MAX_CHARS + 1);
        let result = truncate_content(&over, "TEST.md", CONTEXT_FILE_MAX_CHARS);
        assert!(result.contains("truncated TEST.md"));
    }
}
