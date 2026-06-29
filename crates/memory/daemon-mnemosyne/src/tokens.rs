// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Token counting — port of `token_counter.py`.
//!
//! With the `tiktoken` feature, [`estimate_tokens`] returns exact `cl100k_base` token counts (via a
//! process-cached encoder); otherwise it falls back to the `len/4` heuristic. Mirrors
//! `token_counter.py` L20-L41 (tiktoken when available, else `len(text) // 4`).

/// Estimate the token count of `text` (`token_counter.py` L20-L41).
///
/// Empty input is `0`. With `tiktoken` the count is exact `cl100k_base`; without it (or on an
/// encoder error) the `len/4` byte heuristic is used.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    #[cfg(feature = "tiktoken")]
    {
        if let Some(count) = tiktoken_count(text) {
            return count;
        }
    }
    text.len() / 4
}

#[cfg(feature = "tiktoken")]
fn tiktoken_count(text: &str) -> Option<usize> {
    use std::sync::OnceLock;
    use tiktoken_rs::CoreBPE;
    static ENCODER: OnceLock<Option<CoreBPE>> = OnceLock::new();
    let encoder = ENCODER.get_or_init(|| tiktoken_rs::cl100k_base().ok());
    encoder
        .as_ref()
        .map(|bpe| bpe.encode_with_special_tokens(text).len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_zero() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn nonempty_is_counted() {
        // Both backends return a positive count for non-trivial text.
        assert!(estimate_tokens("the quick brown fox jumps over the lazy dog") > 0);
    }

    #[cfg(not(feature = "tiktoken"))]
    #[test]
    fn heuristic_is_len_over_four() {
        assert_eq!(estimate_tokens("abcdefgh"), 2);
    }

    #[cfg(feature = "tiktoken")]
    #[test]
    fn tiktoken_matches_cl100k_for_a_known_phrase() {
        // "hello world" is two tokens under cl100k_base.
        assert_eq!(estimate_tokens("hello world"), 2);
    }
}
