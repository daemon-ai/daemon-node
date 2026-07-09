// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! HTML-markup stripping ported from libpurple (`purple_markup_strip_html`, `purplemarkup.c`).
//!
//! A single-pass, byte-level scanner that reduces a fragment of message HTML to plain text: it drops
//! tags, unescapes entities, collapses `<a href=…>` links to `text (href)`, turns block/table markup
//! into newlines/tabs, and discards `<script>`/`<style>` CDATA. This is node-authoritative text
//! normalization — adapters that receive formatted bodies (e.g. a Matrix `formatted_body`) hand the
//! HTML here rather than each re-deriving a stripper.

/// Case-insensitive ASCII prefix test (`g_ascii_strncasecmp(hay, needle, len) == 0`).
fn starts_with_ci(hay: &[u8], needle: &[u8]) -> bool {
    hay.len() >= needle.len() && hay[..needle.len()].eq_ignore_ascii_case(needle)
}

/// Decode a single HTML entity at the start of `text` (which must begin with `&`), returning the
/// decoded UTF-8 bytes and the number of input bytes consumed. Ports
/// `purple_markup_unescape_entity`: the named set plus numeric `&#dec;` / `&#xhex;`.
pub fn unescape_entity(_text: &[u8]) -> Option<(Vec<u8>, usize)> {
    None
}

/// Strip HTML markup to plain text (`purple_markup_strip_html`).
pub fn strip_html(_input: &str) -> String {
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- the libpurple /util/markup/strip-html data table -------------------

    #[test]
    fn strip_html_libpurple_matrix() {
        let cases: &[(&str, &str)] = &[
            ("", ""),
            (
                "<a href=\"https://example.com/\">https://example.com/</a>",
                "https://example.com/",
            ),
            (
                "<a href=\"https://example.com/\">example.com</a>",
                "example.com (https://example.com/)",
            ),
            ("<script>/* this should be ignored */</script>", ""),
            ("<style>/* this should be ignored */</style>", ""),
            (
                "<table><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table>",
                "1\t2\n3\t4\n",
            ),
            ("<p>foo</p><p>bar</p><p>baz</p>", "foo\nbar\nbaz"),
            ("<div><p>foo</p><p>bar</p></div>", "foo\nbar"),
            ("<hr>", ""),
            ("<br>", "\n"),
        ];
        for (markup, plaintext) in cases {
            assert_eq!(&strip_html(markup), plaintext, "markup: {markup:?}");
        }
    }

    // -- extra: named entities ---------------------------------------------

    #[test]
    fn strip_html_named_entities() {
        assert_eq!(strip_html("a &amp; b"), "a & b");
        assert_eq!(strip_html("&lt;tag&gt;"), "<tag>");
        assert_eq!(strip_html("&quot;q&quot;"), "\"q\"");
        assert_eq!(strip_html("it&apos;s"), "it's");
        assert_eq!(strip_html("a&nbsp;b"), "a b");
        assert_eq!(strip_html("&copy;"), "\u{a9}");
        assert_eq!(strip_html("&reg;"), "\u{ae}");
    }

    // -- extra: numeric entities -------------------------------------------

    #[test]
    fn strip_html_numeric_entities() {
        assert_eq!(strip_html("&#65;&#66;&#67;"), "ABC");
        assert_eq!(strip_html("&#x41;&#x42;"), "AB");
        // Invalid numeric entities are left verbatim.
        assert_eq!(strip_html("&#0;"), "&#0;");
        assert_eq!(strip_html("&#foo;"), "&#foo;");
        // Missing terminator is left verbatim.
        assert_eq!(strip_html("&#65"), "&#65");
    }

    // -- extra: nested + malformed -----------------------------------------

    #[test]
    fn strip_html_nested_and_malformed() {
        assert_eq!(strip_html("<b><i>hi</i></b>"), "hi");
        // Unclosed tag at end of input: the tag content is scanned and dropped.
        assert_eq!(strip_html("text<b"), "text");
        // A lone trailing '<' is emitted verbatim (libpurple falls through to the visible append).
        assert_eq!(strip_html("a<"), "a<");
        // Attributes are dropped along with the tag.
        assert_eq!(strip_html("<span class=\"x\">hi</span>"), "hi");
    }

    // -- unescape_entity unit ----------------------------------------------

    #[test]
    fn unescape_entity_units() {
        assert_eq!(unescape_entity(b"&amp;rest"), Some((b"&".to_vec(), 5)));
        assert_eq!(unescape_entity(b"&lt;"), Some((b"<".to_vec(), 4)));
        assert_eq!(unescape_entity(b"&#65;"), Some((b"A".to_vec(), 5)));
        assert_eq!(unescape_entity(b"&#x41;"), Some((b"A".to_vec(), 6)));
        assert_eq!(unescape_entity(b"&unknown;"), None);
        assert_eq!(unescape_entity(b"nope"), None);
    }
}
