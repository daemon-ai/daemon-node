// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Content scrubbing (§9) — the [`StreamingThinkScrubber`].
//!
//! Some models leak chain-of-thought into the *content* channel wrapped in `<think>`/`<thinking>`
//! spans instead of (or in addition to) a dedicated reasoning channel. Rendering that as assistant
//! output is wrong (§17.2). The scrubber is a small state machine that, fed text chunk-by-chunk as
//! they stream, routes `<think>…</think>` spans to the reasoning channel and everything else to the
//! text channel — crucially handling tags that straddle a chunk boundary by retaining the minimal
//! ambiguous tail until the next chunk resolves it.

/// One scrubbed chunk: text destined for the content channel and reasoning destined for the
/// reasoning channel (either may be empty).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScrubChunk {
    /// Content-channel text (outside any think span).
    pub text: String,
    /// Reasoning-channel text (inside a think span).
    pub reasoning: String,
}

/// The open think tags recognized (case-insensitive).
const OPEN: [&str; 2] = ["<think>", "<thinking>"];
/// The close think tags recognized (case-insensitive).
const CLOSE: [&str; 2] = ["</think>", "</thinking>"];

enum TagMatch {
    /// The buffer begins with a complete tag of this byte length.
    Full(usize),
    /// The buffer begins with a strict prefix of a tag — wait for more input.
    Partial,
    /// The `<` does not begin a tag.
    None,
}

/// A streaming scrubber that strips `<think>`/`<thinking>` spans out of the content channel and into
/// the reasoning channel, tolerant of tags split across [`StreamingThinkScrubber::push`] calls.
#[derive(Debug, Default)]
pub struct StreamingThinkScrubber {
    buf: String,
    inside: bool,
}

impl StreamingThinkScrubber {
    /// A fresh scrubber in the content (outside-think) state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the next text chunk, returning the text/reasoning resolvable so far. An ambiguous tail
    /// (a possible partial tag) is retained internally until the next `push`/`finish`.
    pub fn push(&mut self, chunk: &str) -> ScrubChunk {
        self.buf.push_str(chunk);
        self.drain(false)
    }

    /// Flush any retained tail at end of stream, treating an unresolved partial tag as literal text.
    pub fn finish(&mut self) -> ScrubChunk {
        self.drain(true)
    }

    fn drain(&mut self, final_flush: bool) -> ScrubChunk {
        let mut out = ScrubChunk::default();
        loop {
            let tags: &[&str] = if self.inside { &CLOSE } else { &OPEN };
            let Some(idx) = self.buf.find('<') else {
                // No tag start: the whole buffer is content for the current channel.
                self.route(&self.buf.clone(), &mut out);
                self.buf.clear();
                break;
            };
            // Everything before the '<' is plain content for the current channel.
            let before = self.buf[..idx].to_string();
            self.route(&before, &mut out);
            let rest = self.buf[idx..].to_string();
            match match_tag(&rest, tags) {
                TagMatch::Full(len) => {
                    // Consume the tag (never emitted) and toggle channel.
                    self.inside = !self.inside;
                    self.buf = self.buf[idx + len..].to_string();
                }
                TagMatch::Partial => {
                    if final_flush {
                        // End of stream with an unresolved partial: emit it literally.
                        self.route(&rest, &mut out);
                        self.buf.clear();
                    } else {
                        // Retain the ambiguous tail for the next chunk.
                        self.buf = rest;
                    }
                    break;
                }
                TagMatch::None => {
                    // A lone '<' that is not a tag: emit it and rescan after it.
                    self.route("<", &mut out);
                    self.buf = self.buf[idx + 1..].to_string();
                }
            }
        }
        out
    }

    fn route(&self, s: &str, out: &mut ScrubChunk) {
        if s.is_empty() {
            return;
        }
        if self.inside {
            out.reasoning.push_str(s);
        } else {
            out.text.push_str(s);
        }
    }
}

fn match_tag(rest: &str, tags: &[&str]) -> TagMatch {
    let lower = rest.to_ascii_lowercase();
    let mut partial = false;
    for tag in tags {
        if lower.starts_with(tag) {
            return TagMatch::Full(tag.len());
        }
        if lower.len() < tag.len() && tag.starts_with(&lower) {
            partial = true;
        }
    }
    if partial {
        TagMatch::Partial
    } else {
        TagMatch::None
    }
}

/// One-shot convenience: scrub a complete (non-streamed) content string.
pub fn scrub_content(text: &str) -> ScrubChunk {
    let mut scrubber = StreamingThinkScrubber::new();
    let mut out = scrubber.push(text);
    let tail = scrubber.finish();
    out.text.push_str(&tail.text);
    out.reasoning.push_str(&tail.reasoning);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_a_whole_span() {
        let out = scrub_content("hello <think>secret plan</think> world");
        assert_eq!(out.text, "hello  world");
        assert_eq!(out.reasoning, "secret plan");
    }

    #[test]
    fn handles_tag_split_across_chunks() {
        let mut s = StreamingThinkScrubber::new();
        let a = s.push("before <thi");
        assert_eq!(a.text, "before ");
        let b = s.push("nking>deep");
        // The open tag completed; "deep" is reasoning so far.
        assert_eq!(b.text, "");
        assert_eq!(b.reasoning, "deep");
        let c = s.push(" thoughts</thin");
        assert_eq!(c.reasoning, " thoughts");
        let d = s.push("king>after");
        assert_eq!(d.text, "after");
        let f = s.finish();
        assert_eq!(f.text, "");
    }

    #[test]
    fn non_tag_angle_brackets_are_literal() {
        let out = scrub_content("a < b and c > d <notathing>");
        assert_eq!(out.text, "a < b and c > d <notathing>");
        assert_eq!(out.reasoning, "");
    }

    #[test]
    fn unterminated_partial_at_eof_is_literal() {
        let mut s = StreamingThinkScrubber::new();
        let a = s.push("tail <thin");
        assert_eq!(a.text, "tail ");
        let f = s.finish();
        assert_eq!(f.text, "<thin");
    }
}
