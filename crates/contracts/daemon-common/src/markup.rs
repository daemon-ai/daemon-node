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
pub fn unescape_entity(text: &[u8]) -> Option<(Vec<u8>, usize)> {
    if text.first() != Some(&b'&') {
        return None;
    }

    // The named entities: (spelling, replacement).
    const NAMED: &[(&[u8], &[u8])] = &[
        (b"&amp;", b"&"),
        (b"&lt;", b"<"),
        (b"&gt;", b">"),
        (b"&nbsp;", b" "),
        (b"&copy;", "\u{a9}".as_bytes()),
        (b"&quot;", b"\""),
        (b"&reg;", "\u{ae}".as_bytes()),
        (b"&apos;", b"'"),
    ];
    for (needle, repl) in NAMED {
        if starts_with_ci(text, needle) {
            return Some((repl.to_vec(), needle.len()));
        }
    }

    // Numeric entities: `&#<dec>;` or `&#x<hex>;`. libpurple's guard is `text[1]=='#' &&
    // (isxdigit(text[2]) || text[2]=='x')`.
    if text.get(1) == Some(&b'#')
        && text
            .get(2)
            .is_some_and(|c| c.is_ascii_hexdigit() || *c == b'x' || *c == b'X')
    {
        let mut start = 2;
        let mut base = 10u32;
        if text.get(start) == Some(&b'x') || text.get(start) == Some(&b'X') {
            base = 16;
            start += 1;
        }
        // Read digits up to the terminating ';'.
        let mut end = start;
        while end < text.len() && text[end] != b';' {
            end += 1;
        }
        // A missing ';' (ran off the end) is invalid.
        if end >= text.len() {
            return None;
        }
        let digits = std::str::from_utf8(&text[start..end]).ok()?;
        let code = u32::from_str_radix(digits, base).ok()?;
        // libpurple rejects 0 and anything past INT_MAX.
        if code == 0 || code > i32::MAX as u32 {
            return None;
        }
        let ch = char::from_u32(code)?;
        let mut buf = [0u8; 4];
        let encoded = ch.encode_utf8(&mut buf).as_bytes().to_vec();
        // Consumed bytes: through the ';'.
        return Some((encoded, end + 1));
    }

    None
}

/// The newline-mapped block tags that only fire once output has started (guarded by `j &&` in the C).
fn is_leading_guarded_newline_tag(rest: &[u8]) -> bool {
    starts_with_ci(rest, b"<p>")
        || starts_with_ci(rest, b"<tr")
        || starts_with_ci(rest, b"<hr")
        || starts_with_ci(rest, b"<li")
        || starts_with_ci(rest, b"<div")
}

/// Strip HTML markup to plain text (`purple_markup_strip_html`, `purplemarkup.c`).
///
/// A faithful single-pass byte-level port: reads the input bytes, emits plain-text bytes. `<script>`/
/// `<style>` CDATA is dropped; `</td>`→`<td>` becomes a tab; block tags map to newlines; `<a href>`
/// links collapse to `text (href)` (unless the visible text already is the href); entities decode.
pub fn strip_html(input: &str) -> String {
    let s = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(s.len());

    let mut visible = true;
    let mut closing_td_p = false;
    let mut cdata_close_tag: Option<&[u8]> = None;
    let mut href: Option<Vec<u8>> = None;
    let mut href_st: usize = 0;

    let mut i = 0usize;
    while i < s.len() {
        let c = s[i];

        if c == b'<' {
            if let Some(tag) = cdata_close_tag {
                // Inside CDATA: nothing is a tag except the matching close tag.
                if starts_with_ci(&s[i..], tag) {
                    i += tag.len() - 1;
                    cdata_close_tag = None;
                }
                i += 1;
                continue;
            } else if starts_with_ci(&s[i..], b"<td") && closing_td_p {
                out.push(b'\t');
                visible = true;
            } else if starts_with_ci(&s[i..], b"</td>") {
                closing_td_p = true;
                visible = false;
            } else {
                closing_td_p = false;
                visible = true;
            }

            let k_start = i + 1;
            if k_start < s.len() && s[k_start].is_ascii_whitespace() {
                visible = true;
            } else if k_start < s.len() {
                // Scan to the end of the tag (sloppy: quoted `<`/`>` would confuse us, as in C).
                let mut k = k_start;
                while k < s.len() && s[k] != b'<' && s[k] != b'>' {
                    k += 1;
                }

                if starts_with_ci(&s[i..], b"<a")
                    && s.get(i + 2).is_some_and(u8::is_ascii_whitespace)
                {
                    // Find the href attribute and its (optionally quoted) value.
                    let mut st = i + 3;
                    let mut delim = b' ';
                    while st < k {
                        if starts_with_ci(&s[st..], b"href=") {
                            st += 5;
                            if st < s.len() && (s[st] == b'"' || s[st] == b'\'') {
                                delim = s[st];
                                st += 1;
                            }
                            break;
                        }
                        st += 1;
                    }
                    let mut end = st;
                    while end < k && s[end] != delim {
                        end += 1;
                    }
                    if st < k {
                        href = Some(unescape_html(&s[st..end]));
                        href_st = out.len();
                    }
                } else if href.is_some() && starts_with_ci(&s[i..], b"</a>") {
                    let h = href.as_ref().unwrap();
                    let hrlen = h.len();
                    let vislen = out.len() - href_st;
                    // Only insert the href if it differs from the visible CDATA (modulo a leading
                    // "http://", 7 chars) — mirrors the C double-condition.
                    let differs_plain = hrlen != vislen || out[href_st..] != h[..];
                    let differs_stripped =
                        hrlen < 7 || hrlen != vislen + 7 || out[href_st..] != h[7..];
                    if differs_plain && differs_stripped {
                        out.push(b' ');
                        out.push(b'(');
                        out.extend_from_slice(h);
                        out.push(b')');
                        href = None;
                    }
                } else if (!out.is_empty() && is_leading_guarded_newline_tag(&s[i..]))
                    || starts_with_ci(&s[i..], b"<br")
                    || starts_with_ci(&s[i..], b"</table>")
                {
                    out.push(b'\n');
                } else if starts_with_ci(&s[i..], b"<script") {
                    cdata_close_tag = Some(b"</script>");
                } else if starts_with_ci(&s[i..], b"<style") {
                    cdata_close_tag = Some(b"</style>");
                }

                // Continue scanning after the tag.
                i = if k >= s.len() || s[k] == b'<' {
                    // k-1 then +1 below = k; guard the k==0 impossibility (k>=1 here).
                    k
                } else {
                    k + 1
                };
                continue;
            }
        } else if cdata_close_tag.is_some() {
            i += 1;
            continue;
        } else if !c.is_ascii_whitespace() {
            visible = true;
        }

        if c == b'&' {
            if let Some((ent, entlen)) = unescape_entity(&s[i..]) {
                out.extend_from_slice(&ent);
                i += entlen;
                continue;
            }
        }

        if visible {
            out.push(if c.is_ascii_whitespace() { b' ' } else { c });
        }
        i += 1;
    }

    // `out` is built from valid UTF-8 input plus ASCII/UTF-8 entity replacements, so it is valid.
    String::from_utf8(out).unwrap_or_default()
}

/// Unescape HTML entities (and `<br>`) in a byte slice (`purple_unescape_html`). Used for the href
/// captured from an `<a>` tag before it is compared/emitted.
fn unescape_html(html: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(html.len());
    let mut i = 0;
    while i < html.len() {
        if let Some((ent, len)) = unescape_entity(&html[i..]) {
            out.extend_from_slice(&ent);
            i += len;
        } else if starts_with_ci(&html[i..], b"<br>") {
            out.push(b'\n');
            i += 4;
        } else {
            out.push(html[i]);
            i += 1;
        }
    }
    out
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
