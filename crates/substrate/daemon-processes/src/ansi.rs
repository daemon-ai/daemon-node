// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! ANSI escape stripping for model-facing output (hermes `tools/ansi_strip.py`): every poll/log/
//! wait tail and every notification body is stripped so terminal color/cursor noise never reaches
//! the model.

use regex::Regex;
use std::sync::OnceLock;

/// Strip ANSI escape sequences: CSI (`ESC [ ... cmd`), OSC (`ESC ] ... BEL`/`ESC \`), and the
/// remaining two-byte `ESC x` controls.
pub fn strip_ansi(s: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        // CSI: ESC [ params intermediates final; OSC: ESC ] ... (BEL | ST); else ESC single.
        Regex::new(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)|\x1b[@-Z\\-_]")
            .expect("static ANSI regex compiles")
    });
    re.replace_all(s, "").into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_color_cursor_and_osc_sequences() {
        assert_eq!(strip_ansi("\x1b[31mred\x1b[0m plain"), "red plain");
        assert_eq!(strip_ansi("\x1b[2K\x1b[1Gprogress 50%"), "progress 50%");
        assert_eq!(strip_ansi("\x1b]0;title\x07body"), "body");
        assert_eq!(strip_ansi("no escapes"), "no escapes");
    }
}
