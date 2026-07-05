// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Read-side helpers for the `fs` tool: line pagination with tail-reads, the compact
//! `LINE|content` gutter, per-line and whole-read character caps, and binary detection.
//!
//! The gutter is hermes' compact `<n>|<line>` (file_operations.py:784 `_add_line_numbers`) — no
//! zero/space padding, which measurably wastes tokens. Pagination follows the merged
//! hermes/Cursor semantics: `offset` is 1-indexed, a **negative** offset counts back from the end
//! (`-N` = the last N lines, the Cursor tail-read), and `limit` caps the returned line count.

/// One paginated read: the rendered window plus the bookkeeping the caller reports.
pub struct Page {
    /// The rendered `LINE|content` window.
    pub text: String,
    /// The 1-indexed first line of the window.
    pub start_line: usize,
    /// The 1-indexed last line of the window (0 when the window is empty).
    pub end_line: usize,
    /// Total lines in the content.
    pub total_lines: usize,
    /// Whether lines beyond `end_line` exist.
    pub truncated: bool,
}

/// The content's lines for pagination purposes: a trailing newline does not create a phantom
/// final empty line.
fn content_lines(content: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = content.split('\n').collect();
    if lines.last() == Some(&"") && content.ends_with('\n') {
        lines.pop();
    }
    lines
}

/// Paginate `content` into a numbered window. `offset` is 1-indexed (`<= 0` values other than
/// negative tail-reads clamp to 1); `offset = -N` starts N lines from the end. `limit` is clamped
/// to `[1, max_lines]`; individual lines longer than `max_line_chars` are cut with a
/// `... [truncated]` marker (hermes parity).
pub fn paginate(
    content: &str,
    offset: i64,
    limit: Option<usize>,
    max_lines: usize,
    max_line_chars: usize,
) -> Page {
    let lines = content_lines(content);
    let total = lines.len();
    let limit = limit.unwrap_or(max_lines).clamp(1, max_lines.max(1));

    let start_line = if offset < 0 {
        // Tail-read: `-N` = the last N lines.
        let back = usize::try_from(-offset).unwrap_or(usize::MAX);
        total.saturating_sub(back) + 1
    } else {
        (usize::try_from(offset).unwrap_or(1)).max(1)
    };

    if total == 0 || start_line > total {
        return Page {
            text: String::new(),
            start_line,
            end_line: 0,
            total_lines: total,
            truncated: false,
        };
    }

    let end_line = (start_line + limit - 1).min(total);
    let mut numbered: Vec<String> = Vec::with_capacity(end_line - start_line + 1);
    for (i, line) in lines[start_line - 1..end_line].iter().enumerate() {
        let line = truncate_line(line, max_line_chars);
        numbered.push(format!("{}|{}", start_line + i, line));
    }
    Page {
        text: numbered.join("\n"),
        start_line,
        end_line,
        total_lines: total,
        truncated: end_line < total,
    }
}

/// Cut a line at `max_chars` characters with hermes' `... [truncated]` marker.
fn truncate_line(line: &str, max_chars: usize) -> String {
    if line.chars().count() <= max_chars {
        return line.to_string();
    }
    let cut: String = line.chars().take(max_chars).collect();
    format!("{cut}... [truncated]")
}

/// File extensions treated as binary for text reads (hermes `tools/binary_extensions.py`,
/// itself ported from free-code). Extraction-capable document extensions stay listed here — the
/// read path attempts extraction *first*, so this guard only fires when extraction is
/// unavailable or the document is malformed (the graceful "binary file" fallback).
const BINARY_EXTENSIONS: &[&str] = &[
    // images
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "webp", "tiff", "tif", // videos
    "mp4", "mov", "avi", "mkv", "webm", "wmv", "flv", "m4v", "mpeg", "mpg", // audio
    "mp3", "wav", "ogg", "flac", "aac", "m4a", "wma", "aiff", "opus", // archives
    "zip", "tar", "gz", "bz2", "7z", "rar", "xz", "z", "tgz", "iso",
    // executables / binaries
    "exe", "dll", "so", "dylib", "bin", "o", "a", "obj", "lib", "app", "msi", "deb", "rpm",
    // documents (pdf/docx/xlsx are extraction-capable; see above)
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "ods", "odp", // fonts
    "ttf", "otf", "woff", "woff2", "eot", // bytecode / VM artifacts
    "pyc", "pyo", "class", "jar", "war", "ear", "node", "wasm", "rlib", // databases
    "sqlite", "sqlite3", "db", "mdb", "idx", // design / 3d
    "psd", "ai", "eps", "sketch", "fig", "xd", "blend", "3ds", "max", // flash
    "swf", "fla", // lock / profiling data
    "lockb", "dat", "data",
];

/// Whether `path` has a known-binary extension (pure string check, no I/O).
pub fn has_binary_extension(path: &str) -> bool {
    match path.rsplit_once('.') {
        Some((_, ext)) => BINARY_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()),
        None => false,
    }
}

/// Whether the leading bytes look like binary content (a NUL byte in the first 8 KiB) — the
/// content-sniff companion to the extension guard, for extension-less binaries.
pub fn looks_binary(bytes: &[u8]) -> bool {
    bytes[..bytes.len().min(8192)].contains(&0)
}

/// The structured refusal for a binary read (hermes' message shape, sans the tool names the
/// daemon does not have).
pub fn binary_refusal(path: &str) -> String {
    format!("fs read: cannot read binary file '{path}' as text; use a shell command to inspect binary content")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paginates_with_compact_gutter() {
        let content = "alpha\nbeta\ngamma\n";
        let page = paginate(content, 1, Some(2), 2000, 2000);
        assert_eq!(page.text, "1|alpha\n2|beta");
        assert_eq!(page.total_lines, 3);
        assert!(page.truncated);
        assert_eq!((page.start_line, page.end_line), (1, 2));
    }

    #[test]
    fn negative_offset_tail_reads() {
        let content = "a\nb\nc\nd";
        let page = paginate(content, -2, None, 2000, 2000);
        assert_eq!(page.text, "3|c\n4|d");
        assert!(!page.truncated);
        // A tail longer than the file starts at line 1.
        let page = paginate(content, -100, None, 2000, 2000);
        assert_eq!(page.start_line, 1);
    }

    #[test]
    fn offset_past_eof_is_empty_not_error() {
        let page = paginate("a\nb", 10, None, 2000, 2000);
        assert_eq!(page.text, "");
        assert_eq!(page.end_line, 0);
        assert_eq!(page.total_lines, 2);
    }

    #[test]
    fn long_lines_are_cut_with_marker() {
        let content = "x".repeat(10);
        let page = paginate(&content, 1, None, 2000, 4);
        assert_eq!(page.text, "1|xxxx... [truncated]");
    }

    #[test]
    fn binary_detection_by_extension_and_content() {
        assert!(has_binary_extension("a/b/photo.PNG"));
        assert!(has_binary_extension("doc.docx"));
        assert!(!has_binary_extension("src/main.rs"));
        assert!(!has_binary_extension("Makefile"));
        assert!(looks_binary(b"ELF\x00\x01\x02"));
        assert!(!looks_binary(b"plain text"));
    }
}
