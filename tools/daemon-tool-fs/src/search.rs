// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Native workspace search for the `fs` tool's `grep` and `glob` ops — Cursor tool semantics
//! (`output_mode: content|files_with_matches|count`, `-C` context, glob file filter, case
//! toggle, `head_limit`/`offset` pagination) built on the ripgrep component crates
//! (`ignore` walker + `grep-searcher`/`grep-regex`), so no external `rg` binary is needed.
//!
//! The walk is rooted at the session workspace (`cx.exec.cwd()`), respects `.gitignore` /
//! `.ignore` (even outside a git repo), skips hidden entries and binary files, and is sorted for
//! deterministic output. Synchronous by design — the caller runs it on a blocking thread.

use std::io;
use std::path::{Path, PathBuf};

use grep_searcher::{BinaryDetection, Searcher, SearcherBuilder, Sink, SinkContext, SinkMatch};

/// How `grep` renders its results (Cursor's `output_mode`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    /// Matching lines (`path:line:text`), with optional `-C` context lines.
    #[default]
    Content,
    /// Only the paths of files with at least one match.
    FilesWithMatches,
    /// Per-file match counts (`path:count`).
    Count,
}

/// One `grep` invocation, already resolved against the workspace root.
pub struct GrepRequest {
    /// The regex pattern.
    pub pattern: String,
    /// The directory (or single file) to search, absolute, already contained.
    pub root: PathBuf,
    /// The workspace root the walk is reported relative to.
    pub workspace: PathBuf,
    /// Optional file-glob filter (`*.rs` matches basenames; a pattern with `/` matches the
    /// workspace-relative path).
    pub glob: Option<String>,
    /// Case-insensitive matching (`-i`).
    pub case_insensitive: bool,
    /// Context lines before/after each match (`-C`).
    pub context: usize,
    /// Output shape.
    pub output_mode: OutputMode,
    /// Maximum result items (matches / files / count rows) after `offset`.
    pub head_limit: usize,
    /// Result items to skip (pagination).
    pub offset: usize,
}

/// One `glob` invocation.
pub struct GlobRequest {
    /// The glob pattern (auto-recursive: a bare `*.rs` is treated as `**/*.rs`).
    pub pattern: String,
    /// The directory to walk, absolute, already contained.
    pub root: PathBuf,
    /// The workspace root results are reported relative to.
    pub workspace: PathBuf,
    /// Maximum results after `offset`.
    pub head_limit: usize,
    /// Results to skip (pagination).
    pub offset: usize,
}

/// A single collected grep line.
enum Row {
    Match { line: u64, text: String },
    Context { line: u64, text: String },
    Break,
}

/// Per-file sink collecting matches + context rows, stopping once the file alone would overflow
/// the caller's window (`stop_after` matches).
struct CollectSink {
    rows: Vec<Row>,
    matches: usize,
    stop_after: usize,
}

impl Sink for CollectSink {
    type Error = io::Error;

    fn matched(&mut self, _searcher: &Searcher, mat: &SinkMatch<'_>) -> Result<bool, io::Error> {
        let text = String::from_utf8_lossy(mat.bytes());
        self.rows.push(Row::Match {
            line: mat.line_number().unwrap_or(0),
            text: text.trim_end_matches(['\n', '\r']).to_string(),
        });
        self.matches += 1;
        Ok(self.matches < self.stop_after)
    }

    fn context(&mut self, _searcher: &Searcher, ctx: &SinkContext<'_>) -> Result<bool, io::Error> {
        let text = String::from_utf8_lossy(ctx.bytes());
        self.rows.push(Row::Context {
            line: ctx.line_number().unwrap_or(0),
            text: text.trim_end_matches(['\n', '\r']).to_string(),
        });
        Ok(true)
    }

    fn context_break(&mut self, _searcher: &Searcher) -> Result<bool, io::Error> {
        self.rows.push(Row::Break);
        Ok(true)
    }
}

/// Build the gitignore-respecting, deterministic workspace walker.
fn walker(root: &Path) -> ignore::Walk {
    ignore::WalkBuilder::new(root)
        // Respect .gitignore/.ignore even when the workspace is not a git repo (session
        // sandboxes usually are not).
        .require_git(false)
        // Never traverse a symlink out of the (openat2-verified) search root — the walk stays within
        // the workspace even though it is path-based (Cluster C: the entry point is fd-verified, the
        // walk is non-following). This is the `ignore` default, made an explicit declared choice.
        .follow_links(false)
        .sort_by_file_path(std::cmp::Ord::cmp)
        .build()
}

/// The path rendered in results: workspace-relative when possible, absolute otherwise.
fn display_path(path: &Path, workspace: &Path) -> String {
    path.strip_prefix(workspace)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

/// Compile the optional file-glob filter. A pattern containing `/` matches the workspace-relative
/// path; otherwise it matches the basename (ripgrep `--glob` semantics).
fn file_filter(glob: Option<&str>) -> io::Result<Option<(globset::GlobMatcher, bool)>> {
    match glob {
        None => Ok(None),
        Some(pattern) => {
            let matcher = globset::Glob::new(pattern)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("bad glob: {e}")))?
                .compile_matcher();
            Ok(Some((matcher, pattern.contains('/'))))
        }
    }
}

fn filter_accepts(
    filter: &Option<(globset::GlobMatcher, bool)>,
    path: &Path,
    workspace: &Path,
) -> bool {
    match filter {
        None => true,
        Some((matcher, full_path)) => {
            if *full_path {
                matcher.is_match(path.strip_prefix(workspace).unwrap_or(path))
            } else {
                path.file_name().is_some_and(|name| matcher.is_match(name))
            }
        }
    }
}

/// Run a grep over the workspace. Returns the rendered, paginated result text — an empty match
/// set renders as a `"no matches"` line.
///
/// Pagination counts *result items* — match lines in `content` mode, files in the other two
/// modes. In `content` mode a file whose matches overlap the window is rendered as a whole block
/// (its context lines ride along); per-file collection is capped at the window end so one huge
/// file cannot flood the output.
pub fn grep(req: &GrepRequest) -> io::Result<String> {
    let matcher = grep_regex::RegexMatcherBuilder::new()
        .case_insensitive(req.case_insensitive)
        .line_terminator(Some(b'\n'))
        .build(&req.pattern)
        .map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("invalid pattern: {e}"))
        })?;
    let mut searcher = SearcherBuilder::new()
        .line_number(true)
        .before_context(req.context)
        .after_context(req.context)
        .binary_detection(BinaryDetection::quit(0))
        .build();
    let filter = file_filter(req.glob.as_deref())?;

    let window_end = req.offset.saturating_add(req.head_limit);
    let per_file_cap = match req.output_mode {
        OutputMode::FilesWithMatches => 1,
        OutputMode::Count => usize::MAX,
        OutputMode::Content => window_end.max(1),
    };

    let mut out: Vec<String> = Vec::new();
    let mut match_cursor = 0usize; // global match index (content-mode pagination)
    let mut file_cursor = 0usize; // global file index (files/count-mode pagination)
    let mut emitted = 0usize; // in-window items actually rendered
    let mut truncated = false;

    for entry in walker(&req.root) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let path = entry.path();
        if !filter_accepts(&filter, path, &req.workspace) {
            continue;
        }
        let mut sink = CollectSink {
            rows: Vec::new(),
            matches: 0,
            stop_after: per_file_cap,
        };
        if searcher.search_path(&matcher, path, &mut sink).is_err() {
            continue; // unreadable file: skip, like rg
        }
        if sink.matches == 0 {
            continue;
        }
        let hit_file_cap = sink.matches >= per_file_cap && req.output_mode == OutputMode::Content;
        let shown = display_path(path, &req.workspace);

        match req.output_mode {
            OutputMode::FilesWithMatches | OutputMode::Count => {
                if file_cursor >= req.offset {
                    if emitted >= req.head_limit {
                        truncated = true;
                        break;
                    }
                    match req.output_mode {
                        OutputMode::Count => out.push(format!("{shown}:{}", sink.matches)),
                        _ => out.push(shown),
                    }
                    emitted += 1;
                }
                file_cursor += 1;
            }
            OutputMode::Content => {
                let file_start = match_cursor;
                match_cursor += sink.matches;
                if match_cursor <= req.offset {
                    continue; // entirely before the window
                }
                if emitted >= req.head_limit || file_start >= window_end {
                    truncated = true;
                    break;
                }
                if !out.is_empty() && req.context > 0 {
                    out.push("--".to_string());
                }
                for row in &sink.rows {
                    match row {
                        Row::Match { line, text } => out.push(format!("{shown}:{line}:{text}")),
                        Row::Context { line, text } => {
                            out.push(format!("{shown}-{line}-{text}"));
                        }
                        Row::Break => out.push("--".to_string()),
                    }
                }
                emitted += sink.matches;
                if hit_file_cap {
                    truncated = true;
                }
            }
        }
    }

    if out.is_empty() {
        return Ok(format!("no matches for {:?}", req.pattern));
    }
    let summary = match req.output_mode {
        OutputMode::Content => format!("showing {emitted} matches"),
        OutputMode::FilesWithMatches => format!("{emitted} files with matches"),
        OutputMode::Count => format!("{emitted} files with matches (per-file counts below)"),
    };
    let mut rendered = format!("{summary}\n{}", out.join("\n"));
    if truncated {
        rendered.push_str(&format!(
            "\n[truncated: more results exist; continue with offset={}]",
            req.offset + emitted
        ));
    }
    Ok(rendered)
}

/// Run a filename glob over the workspace: matching files sorted by modification time
/// (most-recent first, Cursor semantics), workspace-relative, paginated.
pub fn glob(req: &GlobRequest) -> io::Result<String> {
    // Bare patterns are auto-recursive (Cursor: `*.js` becomes `**/*.js`).
    let pattern = if req.pattern.starts_with("**/") || req.pattern.starts_with('/') {
        req.pattern.clone()
    } else {
        format!("**/{}", req.pattern)
    };
    let matcher = globset::GlobBuilder::new(pattern.trim_start_matches('/'))
        .literal_separator(true)
        .build()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("bad glob: {e}")))?
        .compile_matcher();

    let mut hits: Vec<(std::time::SystemTime, String)> = Vec::new();
    for entry in walker(&req.root) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let rel = entry.path().strip_prefix(&req.root).unwrap_or(entry.path());
        if !matcher.is_match(rel) {
            continue;
        }
        let mtime = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(std::time::UNIX_EPOCH);
        hits.push((mtime, display_path(entry.path(), &req.workspace)));
    }
    // Most-recently modified first; path as the deterministic tiebreak.
    hits.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let total = hits.len();
    let page: Vec<String> = hits
        .into_iter()
        .skip(req.offset)
        .take(req.head_limit)
        .map(|(_, path)| path)
        .collect();
    if page.is_empty() {
        return Ok(format!("no files match {:?}", req.pattern));
    }
    let mut rendered = format!("{} of {total} files\n{}", page.len(), page.join("\n"));
    if req.offset + page.len() < total {
        rendered.push_str(&format!(
            "\n[truncated: continue with offset={}]",
            req.offset + page.len()
        ));
    }
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_tree(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("daemon-fs-search-{tag}-{nanos}"));
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("a.rs"), "fn alpha() {}\nfn beta() {}\n").unwrap();
        std::fs::write(root.join("sub/b.rs"), "fn alpha_two() {}\n").unwrap();
        std::fs::write(root.join("notes.txt"), "alpha note\n").unwrap();
        std::fs::write(root.join(".gitignore"), "ignored.rs\n").unwrap();
        std::fs::write(root.join("ignored.rs"), "fn alpha_ignored() {}\n").unwrap();
        root
    }

    fn grep_req(root: &Path, pattern: &str) -> GrepRequest {
        GrepRequest {
            pattern: pattern.into(),
            root: root.to_path_buf(),
            workspace: root.to_path_buf(),
            glob: None,
            case_insensitive: false,
            context: 0,
            output_mode: OutputMode::Content,
            head_limit: 50,
            offset: 0,
        }
    }

    #[test]
    fn grep_respects_gitignore_and_reports_matches() {
        let root = temp_tree("grep");
        let out = grep(&grep_req(&root, "alpha")).unwrap();
        assert!(out.contains("a.rs:1:fn alpha() {}"), "{out}");
        assert!(out.contains("sub/b.rs:1:fn alpha_two() {}"), "{out}");
        assert!(out.contains("notes.txt:1:alpha note"), "{out}");
        assert!(!out.contains("ignored.rs"), "gitignored file leaked: {out}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn grep_glob_filter_and_case_insensitive() {
        let root = temp_tree("glob-filter");
        let mut req = grep_req(&root, "ALPHA");
        req.case_insensitive = true;
        req.glob = Some("*.rs".into());
        let out = grep(&req).unwrap();
        assert!(out.contains("a.rs:1:"), "{out}");
        assert!(!out.contains("notes.txt"), "{out}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn grep_modes_and_pagination() {
        let root = temp_tree("modes");
        let mut req = grep_req(&root, "alpha");
        req.output_mode = OutputMode::FilesWithMatches;
        let out = grep(&req).unwrap();
        assert!(out.contains("a.rs") && out.contains("notes.txt"), "{out}");
        assert!(!out.contains(":1:"), "{out}");

        req.output_mode = OutputMode::Count;
        let out = grep(&req).unwrap();
        assert!(out.contains("a.rs:1"), "{out}");

        req.output_mode = OutputMode::Content;
        req.head_limit = 1;
        let out = grep(&req).unwrap();
        assert!(out.contains("truncated"), "{out}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn grep_context_lines_render_rg_style() {
        let root = temp_tree("ctx");
        let mut req = grep_req(&root, "beta");
        req.context = 1;
        let out = grep(&req).unwrap();
        assert!(out.contains("a.rs-1-fn alpha() {}"), "{out}");
        assert!(out.contains("a.rs:2:fn beta() {}"), "{out}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn grep_bad_pattern_is_an_input_error() {
        let root = temp_tree("badpat");
        let err = grep(&grep_req(&root, "(unclosed")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn glob_auto_recurses_sorts_and_paginates() {
        let root = temp_tree("globop");
        let req = GlobRequest {
            pattern: "*.rs".into(),
            root: root.clone(),
            workspace: root.clone(),
            head_limit: 10,
            offset: 0,
        };
        let out = glob(&req).unwrap();
        assert!(out.contains("a.rs") && out.contains("sub/b.rs"), "{out}");
        assert!(!out.contains("ignored.rs"), "{out}");
        assert!(!out.contains("notes.txt"), "{out}");

        let req = GlobRequest {
            pattern: "nomatch-*.zzz".into(),
            root: root.clone(),
            workspace: root.clone(),
            head_limit: 10,
            offset: 0,
        };
        assert!(glob(&req).unwrap().contains("no files match"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
