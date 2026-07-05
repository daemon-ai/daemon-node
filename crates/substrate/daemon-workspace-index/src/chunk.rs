// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Heuristic, dependency-free text chunker (no tree-sitter).
//!
//! A file is split into *regions* at semantic boundaries — markdown headings and low-indentation
//! declaration lines (`fn`, `struct`, `impl`, `class`, `def`, …) — and each region is emitted as a
//! chunk, or split into fixed line-windows (with overlap) when it exceeds `chunk_lines`. Line spans
//! are 1-based and inclusive. The heuristic is intentionally language-agnostic and cheap: it favours
//! grouping a declaration with the lines that follow it, which is what an embedding recall wants.

/// One chunk of a file: an inclusive, 1-based line span plus its verbatim text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Chunk {
    /// The 1-based first line of the chunk (inclusive).
    pub start_line: usize,
    /// The 1-based last line of the chunk (inclusive).
    pub end_line: usize,
    /// The chunk's text (the joined source lines, newline-separated).
    pub text: String,
}

/// The low-indentation declaration prefixes that open a new region (checked against the
/// leading-whitespace-trimmed line). Kept small and language-agnostic.
const DECL_PREFIXES: &[&str] = &[
    "pub fn ",
    "fn ",
    "struct ",
    "impl ",
    "class ",
    "def ",
    "func ",
    "function ",
];

/// The maximum leading-whitespace width (spaces; a tab counts as one) at which a declaration line is
/// still treated as a region boundary. `4` catches top-level items and one level of nesting (e.g. a
/// Rust method inside an `impl`), which is the useful granularity for code recall.
const MAX_DECL_INDENT: usize = 4;

/// Split `content` into chunks. `chunk_lines` is the fixed-window height for regions without (or
/// larger than) a semantic anchor; `overlap` is how many lines consecutive windows share. Empty
/// input yields no chunks.
pub(crate) fn chunk_text(content: &str, chunk_lines: usize, overlap: usize) -> Vec<Chunk> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let chunk_lines = chunk_lines.max(1);
    // A window must advance by at least one line, so cap the overlap below the window height.
    let effective_overlap = overlap.min(chunk_lines - 1);
    let step = chunk_lines - effective_overlap;

    // Region starts: line 0 always opens the first region; every anchor line opens another.
    let mut starts = vec![0usize];
    for (i, line) in lines.iter().enumerate().skip(1) {
        if is_anchor(line) {
            starts.push(i);
        }
    }

    let mut chunks = Vec::new();
    for (w, &region_start) in starts.iter().enumerate() {
        let region_end = starts.get(w + 1).map(|&n| n - 1).unwrap_or(lines.len() - 1);
        window_region(
            &lines,
            region_start,
            region_end,
            chunk_lines,
            step,
            &mut chunks,
        );
    }
    chunks
}

/// Emit one or more chunks covering the inclusive line range `[region_start, region_end]`, splitting
/// into fixed windows of `chunk_lines` (advancing by `step`) when the region is larger.
fn window_region(
    lines: &[&str],
    region_start: usize,
    region_end: usize,
    chunk_lines: usize,
    step: usize,
    out: &mut Vec<Chunk>,
) {
    let mut pos = region_start;
    loop {
        let end = (pos + chunk_lines - 1).min(region_end);
        out.push(Chunk {
            start_line: pos + 1,
            end_line: end + 1,
            text: lines[pos..=end].join("\n"),
        });
        if end >= region_end {
            break;
        }
        pos += step;
    }
}

/// Whether `line` opens a new region: a markdown heading (`#`… followed by a space or nothing) or a
/// low-indentation declaration line.
fn is_anchor(line: &str) -> bool {
    let trimmed = line.trim_start();
    if is_heading(trimmed) {
        return true;
    }
    let indent = line.len() - trimmed.len();
    indent <= MAX_DECL_INDENT && DECL_PREFIXES.iter().any(|p| trimmed.starts_with(p))
}

/// A markdown ATX heading: one or more leading `#` followed by a space (or the whole line). This
/// deliberately excludes Rust attributes (`#[derive]`) and other `#`-prefixed non-headings.
fn is_heading(trimmed: &str) -> bool {
    let hashes = trimmed.chars().take_while(|&c| c == '#').count();
    if hashes == 0 {
        return false;
    }
    let rest = &trimmed[hashes..];
    rest.is_empty() || rest.starts_with(' ')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_no_chunks() {
        assert!(chunk_text("", 60, 10).is_empty());
    }

    #[test]
    fn single_line_is_one_chunk() {
        let chunks = chunk_text("only line", 60, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 1);
        assert_eq!(chunks[0].text, "only line");
    }

    #[test]
    fn declaration_lines_open_regions() {
        let src = "\
preamble
fn alpha() {
    let x = 1;
}
fn beta() {
    let y = 2;
}";
        let chunks = chunk_text(src, 60, 10);
        // Three regions: the preamble, `fn alpha`, `fn beta`.
        assert_eq!(chunks.len(), 3, "{chunks:#?}");
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (1, 1));
        assert!(chunks[1].text.starts_with("fn alpha"));
        assert_eq!(chunks[1].start_line, 2);
        assert!(chunks[2].text.starts_with("fn beta"));
        assert_eq!(chunks[2].start_line, 5);
    }

    #[test]
    fn nested_method_is_an_anchor_but_deep_indent_is_not() {
        let src = "\
impl Foo {
    fn method(&self) {
            deeply_nested();
    }
}";
        let chunks = chunk_text(src, 60, 10);
        // `impl Foo {` (indent 0) and `    fn method` (indent 4) are anchors; the 12-space
        // `deeply_nested()` line is not — it stays inside the method region.
        assert_eq!(chunks.len(), 2, "{chunks:#?}");
        assert_eq!(chunks[0].start_line, 1); // impl block header
        assert_eq!(chunks[1].start_line, 2); // the method
        assert!(chunks[1].text.contains("deeply_nested"));
    }

    #[test]
    fn markdown_headings_open_regions_but_attributes_do_not() {
        let src = "\
intro text
# Heading One
body
#[derive(Debug)]
struct S;";
        let chunks = chunk_text(src, 60, 10);
        // intro (1), `# Heading One`+body+attribute (2-4), `struct S` (5): the `#[derive]` attribute
        // is NOT a heading, so it rides with the preceding heading region rather than opening one.
        assert_eq!(chunks.len(), 3, "{chunks:#?}");
        assert_eq!(chunks[1].start_line, 2);
        assert!(chunks[1].text.starts_with("# Heading One"));
        assert!(chunks[1].text.contains("#[derive(Debug)]"));
        assert_eq!(chunks[2].start_line, 5);
        assert!(chunks[2].text.contains("struct S"));
    }

    #[test]
    fn oversized_region_splits_into_overlapping_windows() {
        // 10 anchorless lines, window 4, overlap 1 → step 3: [1..4], [4..7], [7..10] (the third
        // window already reaches the last line, so no trailing single-line window is emitted).
        let src = (1..=10)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_text(&src, 4, 1);
        assert_eq!(chunks.len(), 3, "{chunks:#?}");
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (1, 4));
        assert_eq!((chunks[1].start_line, chunks[1].end_line), (4, 7));
        assert_eq!((chunks[2].start_line, chunks[2].end_line), (7, 10));
        // The overlap line is shared between consecutive windows.
        assert!(chunks[0].text.ends_with("line4"));
        assert!(chunks[1].text.starts_with("line4"));
    }

    #[test]
    fn overlap_is_capped_below_window_height() {
        // overlap >= chunk_lines would stall; it is clamped so windows always advance.
        let src = (1..=6)
            .map(|i| format!("l{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_text(&src, 3, 9);
        assert!(chunks.len() >= 2);
        // Windows still make forward progress (no infinite loop, distinct starts).
        assert!(chunks[1].start_line > chunks[0].start_line);
    }
}
