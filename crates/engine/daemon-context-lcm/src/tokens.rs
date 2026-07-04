// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Token counting (`daemon-context-lcm-port-spec.md` §6.1, §12.2).
//!
//! LCM sizes its compaction threshold and leaf/condense budgets in tokens, so it counts with the
//! model's real BPE when it can (`tiktoken-rs`, selected by model family) and falls back to the
//! Python `len/4 + 1` heuristic for unknown encodings (`count_tokens`, `LCM:tokens.py:32-42`).
//! Per-message overhead mirrors the OpenAI chat-format accounting (`4` tokens/message + `3`
//! reply-priming tokens — kept exact for parity).

use daemon_core::{Conversation, Turn};
use std::sync::Arc;
use tiktoken_rs::CoreBPE;

/// Per-message structural overhead (role/framing), OpenAI chat-format accounting.
pub(crate) const PER_MESSAGE_OVERHEAD: usize = 4;
/// Per-request reply-priming overhead.
const REPLY_PRIMING: usize = 3;

/// A model-aware token counter. Cheap to clone (the BPE is shared behind an `Arc`).
#[derive(Clone)]
pub struct Tokenizer {
    bpe: Option<Arc<CoreBPE>>,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::heuristic()
    }
}

impl Tokenizer {
    /// The always-available `len/4 + 1` heuristic (no BPE).
    pub fn heuristic() -> Self {
        Self { bpe: None }
    }

    /// Select a `tiktoken` encoding by model family, falling back to the heuristic if the encoding
    /// can't be constructed.
    pub fn for_model(model: &str) -> Self {
        Self {
            bpe: encoding_for_model(model).map(Arc::new),
        }
    }

    /// Whether this tokenizer is using a real BPE (vs the heuristic fallback).
    pub fn is_exact(&self) -> bool {
        self.bpe.is_some()
    }

    /// Count the tokens in a bare string (no message overhead). Empty text is `0`; the heuristic
    /// fallback is Python's `len(text) // 4 + 1` (`count_tokens`, `LCM:tokens.py:32-42`).
    pub fn count_text(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        match &self.bpe {
            Some(bpe) => bpe.encode_ordinary(text).len(),
            None => text.len() / 4 + 1,
        }
    }

    /// Count the tokens the whole conversation would occupy as a request (system + turns, with
    /// per-message overhead). Assistant `reasoning` is excluded (it is never sent — §6.2/§14.4).
    pub fn count_conversation(&self, conv: &Conversation) -> usize {
        let mut total = REPLY_PRIMING;
        if !conv.system.text.is_empty() {
            total += self.count_text(&conv.system.text) + PER_MESSAGE_OVERHEAD;
        }
        for turn in &conv.turns {
            total += self.count_turn(turn);
        }
        total
    }

    /// Count one turn's contribution (including per-message overhead).
    pub fn count_turn(&self, turn: &Turn) -> usize {
        match turn {
            Turn::User(u) => self.count_text(&u.text) + PER_MESSAGE_OVERHEAD,
            Turn::Assistant(a) => self.count_text(&a.text) + PER_MESSAGE_OVERHEAD,
            Turn::Tool(t) => {
                let mut n = self.count_text(&t.assistant.text) + PER_MESSAGE_OVERHEAD;
                for (call, result) in &t.calls {
                    n += self.count_text(&call.name)
                        + self.count_text(&call.args)
                        + PER_MESSAGE_OVERHEAD;
                    n += self.count_text(&result.content) + PER_MESSAGE_OVERHEAD;
                }
                n
            }
        }
    }
}

/// Pick the BPE encoding for a model family: `o200k_base` for the GPT-4o / o-series / GPT-4.1+ line,
/// `cl100k_base` for the GPT-3.5/4 line and as the default for unknown models.
fn encoding_for_model(model: &str) -> Option<CoreBPE> {
    let m = model.to_ascii_lowercase();
    let o200k = m.contains("gpt-4o")
        || m.contains("gpt-4.1")
        || m.contains("gpt-5")
        || m.contains("o1")
        || m.contains("o3")
        || m.contains("o4");
    let enc = if o200k {
        tiktoken_rs::o200k_base()
    } else {
        tiktoken_rs::cl100k_base()
    };
    enc.ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_core::{Conversation, SystemPrompt};
    use daemon_protocol::UserMsg;

    #[test]
    fn exact_counts_are_within_tolerance_of_a_fixture() {
        let tok = Tokenizer::for_model("gpt-4o-mini");
        assert!(tok.is_exact(), "tiktoken encoding should load");
        // "Hello, world!" is 4 tokens under o200k_base ("Hello" "," " world" "!").
        let n = tok.count_text("Hello, world!");
        assert!((3..=5).contains(&n), "expected ~4 tokens, got {n}");
    }

    #[test]
    fn heuristic_is_chars_over_four_plus_one() {
        let tok = Tokenizer::heuristic();
        assert!(!tok.is_exact());
        // Python: `len(text) // 4 + 1` for non-empty text, `0` for empty.
        assert_eq!(tok.count_text("abcdefgh"), 3);
        assert_eq!(tok.count_text("abc"), 1);
        assert_eq!(tok.count_text(""), 0);
    }

    #[test]
    fn conversation_counting_includes_overhead() {
        let tok = Tokenizer::heuristic();
        let mut c = Conversation::new(SystemPrompt::new(""));
        c.push_user(UserMsg::new("hello there"));
        // priming(3) + user(len/4 + 1 + 4)
        let expected = REPLY_PRIMING + ("hello there".len() / 4 + 1 + PER_MESSAGE_OVERHEAD);
        assert_eq!(tok.count_conversation(&c), expected);
    }
}
