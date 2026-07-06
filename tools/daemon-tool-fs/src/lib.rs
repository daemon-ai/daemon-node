// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-tool-fs` — the coding-filesystem tool (§12/§13), a `daemon_core::Tool`.
//!
//! Seven ops over the session's contained workspace: `read` (paginated, `LINE|content` gutter,
//! tail-reads, document auto-extraction), `write` (atomic temp+rename), `list`, `edit` (the
//! hermes 9-strategy fuzzy find-and-replace with `replace_all` + "did you mean?" feedback),
//! `grep` + `glob` (native, gitignore-respecting workspace search), and `delete` — a
//! hermes/Cursor best-of-both port (hermes `tools/file_tools.py` + `fuzzy_match.py`; Cursor
//! `Read`/`StrReplace`/`Grep`/`Glob` semantics).
//!
//! Containment: file I/O goes through the engine's
//! [`ExecutionEnvironment`](daemon_core::ExecutionEnvironment) where the env exposes the
//! operation (`read`/`list`), and through [`daemon_core::exec::contain`] against the env's
//! workspace root for the operations the env does not expose (atomic rename, delete, directory
//! walking) — the same lexical containment floor either way, so an out-of-workspace path is
//! always a failed result, never an escape. Mutating ops (`write`/`edit`/`delete`) run the §12
//! edit-approval gate and the credential/system deny list before touching anything; post-edit
//! `[fs.lint]` diagnostics (delta-filtered) are appended to `write`/`edit` results.
//!
//! W5 executor seams: `concurrency_for` classifies `read`/`list`/`grep`/`glob` as
//! [`ToolConcurrency::Parallel`] (mutating ops stay exclusive), `mutates_for` scopes the §12
//! checkpoint to the mutating ops, and `parallel_scope_paths` declares each call's target path
//! for the batch path-overlap gate.

#![forbid(unsafe_code)]
// Phase 4: test code may use raw fs/reqwest/Command; the --lib pass still guards production.
#![cfg_attr(test, allow(clippy::disallowed_methods, clippy::disallowed_types))]

#[cfg(feature = "extract")]
pub mod extract;
pub mod fuzzy;
pub mod lint;
pub mod read;
pub mod search;

use async_trait::async_trait;
use daemon_core::exec::{contain, ContainedRoot};
use daemon_core::{
    approve_path, Effect, Gate, Tool, ToolCall, ToolConcurrency, ToolOutcome, TurnCx,
};
use daemon_protocol::ToolDetail;
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

pub use lint::{FsLintConfig, LintRule};

/// The `[fs]` config table: read caps, search caps, extra deny prefixes, and post-edit lint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FsConfig {
    /// Cap on the characters one `read` may return (hermes `file_read_max_chars`); an oversized
    /// read is rejected with an offset/limit hint instead of flooding the context.
    pub max_read_chars: usize,
    /// Cap on the lines one `read` may return (the default `limit`).
    pub max_read_lines: usize,
    /// Default result cap for `grep`/`glob` when the call passes no `head_limit`.
    pub search_result_cap: usize,
    /// Extra write-deny path prefixes (absolute, or `~/`-relative to the daemon's home) checked on
    /// every mutating op, on top of the built-in credential/system deny list.
    pub deny_paths: Vec<String>,
    /// Post-edit lint (`[fs.lint]`).
    pub lint: FsLintConfig,
}

impl Default for FsConfig {
    fn default() -> Self {
        Self {
            max_read_chars: 100_000,
            max_read_lines: 2_000,
            search_result_cap: 200,
            deny_paths: Vec::new(),
            lint: FsLintConfig::default(),
        }
    }
}

/// Characters a line keeps before the `... [truncated]` marker (hermes `MAX_LINE_LENGTH`).
const MAX_LINE_CHARS: usize = 2_000;
/// Cap on the unified diff attached to an `edit` result.
const MAX_DIFF_CHARS: usize = 4_000;
/// Cap on the `old_string`/`new_string` preview embedded in an approval prompt.
const PREVIEW_CHARS: usize = 80;

/// The filesystem operations the tool exposes.
#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
enum FsArgs {
    /// Read a file (paginated, numbered, document-extracting).
    Read {
        path: String,
        /// 1-indexed start line; negative = tail-read counted from the end (Cursor semantics).
        #[serde(default)]
        offset: Option<i64>,
        /// Max lines returned (clamped to the configured cap).
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Write (create/replace) a file atomically.
    Write { path: String, content: String },
    /// List a directory's entries.
    List {
        #[serde(default = "dot")]
        path: String,
    },
    /// Fuzzy find-and-replace (hermes 9-strategy chain + Cursor `StrReplace` semantics).
    Edit {
        path: String,
        /// The text to find (`find` accepted as a legacy alias).
        #[serde(alias = "find")]
        old_string: String,
        /// The replacement (`replace` accepted as a legacy alias); empty deletes the match.
        #[serde(alias = "replace")]
        new_string: String,
        /// Replace every occurrence instead of requiring a unique match.
        #[serde(default)]
        replace_all: bool,
    },
    /// Regex content search over the workspace (gitignore-respecting).
    Grep {
        pattern: String,
        #[serde(default = "dot")]
        path: String,
        /// Optional file filter (`*.rs`, or a `/`-containing path glob).
        #[serde(default)]
        glob: Option<String>,
        #[serde(default)]
        output_mode: search::OutputMode,
        /// Context lines before/after each match (`-C`).
        #[serde(default)]
        context: usize,
        #[serde(default)]
        case_insensitive: bool,
        /// Max result items (defaults to the configured cap).
        #[serde(default)]
        head_limit: Option<usize>,
        /// Result items to skip (pagination).
        #[serde(default)]
        search_offset: usize,
    },
    /// Find files by glob pattern, most-recently modified first.
    Glob {
        pattern: String,
        #[serde(default = "dot")]
        path: String,
        #[serde(default)]
        head_limit: Option<usize>,
        #[serde(default)]
        search_offset: usize,
    },
    /// Delete a file (or empty directory) — approval-gated like `write`/`edit`.
    Delete { path: String },
}

fn dot() -> String {
    ".".into()
}

impl FsArgs {
    /// Whether this op mutates the workspace.
    fn mutating(&self) -> bool {
        matches!(
            self,
            Self::Write { .. } | Self::Edit { .. } | Self::Delete { .. }
        )
    }

    /// The op's target path argument (the search root for `grep`/`glob`).
    fn target_path(&self) -> &str {
        match self {
            Self::Read { path, .. }
            | Self::Write { path, .. }
            | Self::List { path }
            | Self::Edit { path, .. }
            | Self::Grep { path, .. }
            | Self::Glob { path, .. }
            | Self::Delete { path } => path,
        }
    }
}

/// The structured detail attached to an fs result (opaque to the daemon; rendered by `kind`).
#[derive(Debug, Serialize)]
struct FsDetail<'a> {
    op: &'a str,
    path: &'a str,
    /// Bytes read/written, entry count for a list, or replacement/match count.
    count: usize,
    /// The fuzzy strategy that matched (edit only).
    #[serde(skip_serializing_if = "Option::is_none")]
    strategy: Option<&'a str>,
}

/// The filesystem tool.
#[derive(Default)]
pub struct FsTool {
    cfg: Arc<FsConfig>,
}

impl FsTool {
    /// A filesystem tool with default configuration (tests / minimal hosts).
    pub fn new() -> Self {
        Self::default()
    }

    /// A filesystem tool over an explicit `[fs]` configuration.
    pub fn with_config(cfg: FsConfig) -> Self {
        Self { cfg: Arc::new(cfg) }
    }

    fn detail(op: &str, path: &str, count: usize, strategy: Option<&str>) -> ToolDetail {
        let body = serde_json::to_vec(&FsDetail {
            op,
            path,
            count,
            strategy,
        })
        .unwrap_or_default();
        ToolDetail {
            kind: "fs".into(),
            body,
        }
    }

    /// The §12 edit-approval gate for a file-mutating op. `Ok(())` proceeds; `Err(outcome)` is the
    /// early-return result — a policy/operator rejection, or a durable-HITL defer that suspends the
    /// turn (carrying an [`Effect::AwaitDecision`] so the engine re-runs this call on approval).
    async fn gate(
        call: &ToolCall,
        cx: &TurnCx<'_>,
        path: &str,
        prompt: String,
    ) -> Result<(), ToolOutcome> {
        match approve_path(cx, path, prompt.clone()).await {
            // fs edits carry no command fingerprint, so permanence is never offered here.
            Gate::Proceed { .. } => Ok(()),
            Gate::Reject(reason) => Err(ToolOutcome::text(
                call.call_id.clone(),
                false,
                format!("fs: {reason}"),
            )),
            Gate::Defer(job_id) => Err(ToolOutcome::text(
                call.call_id.clone(),
                false,
                format!("awaiting-approval:{job_id}"),
            )
            .with_effects(vec![Effect::AwaitDecision {
                job_id,
                call: call.clone(),
                prompt,
                path: Some(path.to_string()),
            }])),
        }
    }

    /// The write-deny verdict for a contained absolute path: the built-in credential/system
    /// prefixes (hermes `agent/file_safety.py` write denylist) plus the configured extras.
    /// Defense-in-depth over the workspace containment — it matters when an operator binds a
    /// session workspace over a sensitive tree.
    fn write_denied(&self, resolved: &Path) -> Option<String> {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let mut prefixes: Vec<PathBuf> = vec![
            "/etc".into(),
            "/boot".into(),
            "/usr/lib/systemd".into(),
            "/private/etc".into(),
            "/private/var".into(),
            "/var/run/docker.sock".into(),
            "/run/docker.sock".into(),
        ];
        if let Some(home) = &home {
            for rel in [
                ".ssh",
                ".aws",
                ".gnupg",
                ".kube",
                ".docker",
                ".azure",
                ".config/gh",
                ".config/gcloud",
                ".netrc",
                ".pgpass",
                ".npmrc",
                ".pypirc",
                ".git-credentials",
            ] {
                prefixes.push(home.join(rel));
            }
        }
        for extra in &self.cfg.deny_paths {
            let path = match (extra.strip_prefix("~/"), &home) {
                (Some(rel), Some(home)) => home.join(rel),
                _ => PathBuf::from(extra),
            };
            prefixes.push(path);
        }
        prefixes
            .iter()
            .find(|prefix| resolved.starts_with(prefix))
            .map(|prefix| {
                format!(
                    "fs: write denied: '{}' is under the protected path '{}'",
                    resolved.display(),
                    prefix.display()
                )
            })
    }

    /// Atomically write `bytes` to the contained `path` under `workspace`: parent dirs created,
    /// content lands in a same-directory temp file first, then a rename swaps it into place — a
    /// crash mid-write can never leave a half-written target (hermes `_atomic_write` parity,
    /// tool-level per the approved design).
    async fn atomic_write(workspace: &Path, path: &str, bytes: &[u8]) -> std::io::Result<PathBuf> {
        // All fs access is fd-contained (openat2 RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS): a symlink at
        // any component of `path` — or of the same-dir temp — is rejected, never followed out of root.
        let cr = ContainedRoot::open(workspace)?;
        let rel = Path::new(path);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let file_name = rel
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let tmp_rel = match rel.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => {
                parent.join(format!(".daemon-fs-tmp.{nanos}.{file_name}"))
            }
            _ => PathBuf::from(format!(".daemon-fs-tmp.{nanos}.{file_name}")),
        };
        // Preserve an existing target's permissions across the swap (best-effort).
        let existing_mode = cr.symlink_metadata(rel).await.ok().map(|m| m.mode);
        // Content lands in a same-directory temp file first, then an atomic rename swaps it into
        // place — a crash mid-write can never leave a half-written target (both ops fd-contained).
        cr.write(&tmp_rel, bytes).await?;
        if let Some(mode) = existing_mode {
            let _ = cr.set_mode(&tmp_rel, mode).await;
        }
        match cr.rename(&tmp_rel, rel).await {
            Ok(()) => cr.resolve_display(rel),
            Err(e) => {
                let _ = cr.remove_file(&tmp_rel).await;
                Err(e)
            }
        }
    }

    /// Verify a grep/glob search root is contained (no symlink escape at any component) and return
    /// its absolute path for the `ignore` walker (which runs with `follow_links(false)`). A symlinked
    /// root is refused; a not-yet-existing path is tolerated so the caller reports "not found".
    async fn verify_search_root(workspace: &Path, path: &str) -> std::io::Result<PathBuf> {
        let cr = ContainedRoot::open(workspace)?;
        let rel = Path::new(path);
        match cr.symlink_metadata(rel).await {
            Ok(meta) if meta.is_symlink => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "refusing to search through a symlink",
                ))
            }
            Ok(_) => {}
            // A containment/symlink violation in the parent chain surfaces as PermissionDenied —
            // propagate it; a missing final component is fine (grep/glob report "not found").
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => return Err(e),
            Err(_) => {}
        }
        cr.resolve_display(rel)
    }

    /// Best-effort pre-edit content capture for the lint delta (only when a lint rule matches).
    async fn pre_content_for_lint(&self, cx: &TurnCx<'_>, path: &str) -> Option<String> {
        self.cfg.lint.rule_for(path)?;
        let bytes = cx.exec.read(Path::new(path)).await.ok()?;
        Some(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Run the post-edit lint delta and append its block to `content`.
    async fn append_lint(
        &self,
        cx: &TurnCx<'_>,
        path: &str,
        pre_content: Option<&str>,
        content: &mut String,
    ) {
        if let Some(block) =
            lint::lint_delta(cx, &self.cfg.lint, cx.exec.cwd(), path, pre_content).await
        {
            content.push_str("\n\n");
            content.push_str(&block);
        }
    }

    // -- op implementations ---------------------------------------------------------------------

    /// Attempt document extraction: `(extracted_text, failure_reason)`. With the `extract`
    /// feature off this is a constant no-op — reads of document formats then hit the binary
    /// guard (the graceful fallback).
    #[cfg(feature = "extract")]
    async fn try_extract(path: &str, bytes: &[u8]) -> (Option<String>, Option<String>) {
        let Some(kind) = extract::doc_kind(path) else {
            return (None, None);
        };
        let moved = bytes.to_vec();
        // A blocking thread for the CPU-heavy parse; a panicking parser (malformed PDFs can) is
        // contained by the task boundary and reads as an extraction failure.
        match tokio::task::spawn_blocking(move || extract::extract_document_text(kind, &moved))
            .await
        {
            Ok(Ok(rendered)) => (Some(rendered), None),
            Ok(Err(e)) => (None, Some(e.to_string())),
            Err(_) => (None, Some("document extraction panicked".to_string())),
        }
    }

    #[cfg(not(feature = "extract"))]
    async fn try_extract(_path: &str, _bytes: &[u8]) -> (Option<String>, Option<String>) {
        (None, None)
    }

    async fn op_read(
        &self,
        call: &ToolCall,
        cx: &TurnCx<'_>,
        path: String,
        offset: Option<i64>,
        limit: Option<usize>,
    ) -> ToolOutcome {
        let bytes = match cx.exec.read(Path::new(&path)).await {
            Ok(bytes) => bytes,
            Err(e) => {
                return ToolOutcome::text(call.call_id.clone(), false, format!("fs read: {e}"))
            }
        };
        let byte_len = bytes.len();

        // Document auto-extraction first (hermes order), so `.docx`/`.xlsx`/`.ipynb`/`.pdf`
        // render as text; a malformed document falls through to the binary guard below.
        let (text, extract_failure) = Self::try_extract(&path, &bytes).await;

        let text = match text {
            Some(text) => text,
            None => {
                // Binary guard: by extension, then by content sniff.
                if read::has_binary_extension(&path) || read::looks_binary(&bytes) {
                    let mut msg = read::binary_refusal(&path);
                    if let Some(reason) = extract_failure {
                        msg.push_str(&format!(" (document extraction failed: {reason})"));
                    }
                    return ToolOutcome::text(call.call_id.clone(), false, msg);
                }
                let mut text = String::from_utf8_lossy(&bytes).into_owned();
                // Strip a leading UTF-8 BOM so line 1 renders clean.
                if let Some(stripped) = text.strip_prefix('\u{feff}') {
                    text = stripped.to_string();
                }
                text
            }
        };

        let page = read::paginate(
            &text,
            offset.unwrap_or(1),
            limit,
            self.cfg.max_read_lines,
            MAX_LINE_CHARS,
        );
        if page.text.chars().count() > self.cfg.max_read_chars {
            return ToolOutcome::text(
                call.call_id.clone(),
                false,
                format!(
                    "fs read: output is {} characters, over the {} limit. Use offset and limit \
                     to read a smaller range; the file has {} lines.",
                    page.text.chars().count(),
                    self.cfg.max_read_chars,
                    page.total_lines
                ),
            );
        }
        let mut content = if page.total_lines == 0 {
            "(empty file)".to_string()
        } else if page.text.is_empty() {
            format!(
                "(no lines in range: the file has {} lines)",
                page.total_lines
            )
        } else {
            page.text
        };
        if page.truncated {
            content.push_str(&format!(
                "\n[showing lines {}-{} of {}; continue with offset={}]",
                page.start_line,
                page.end_line,
                page.total_lines,
                page.end_line + 1
            ));
        }
        ToolOutcome::text(call.call_id.clone(), true, content)
            .with_detail(Self::detail("read", &path, byte_len, None))
    }

    async fn op_write(
        &self,
        call: &ToolCall,
        cx: &TurnCx<'_>,
        path: String,
        content: String,
    ) -> ToolOutcome {
        let workspace = cx.exec.cwd().to_path_buf();
        let resolved = match contain(&workspace, Path::new(&path)) {
            Ok(resolved) => resolved,
            Err(e) => {
                return ToolOutcome::text(call.call_id.clone(), false, format!("fs write: {e}"))
            }
        };
        if let Some(denied) = self.write_denied(&resolved) {
            return ToolOutcome::text(call.call_id.clone(), false, denied);
        }
        let prompt = format!("approve write to {path} ({} bytes)", content.len());
        if let Err(out) = Self::gate(call, cx, &path, prompt).await {
            return out;
        }
        let pre_content = self.pre_content_for_lint(cx, &path).await;
        match Self::atomic_write(&workspace, &path, content.as_bytes()).await {
            Ok(_) => {
                let mut msg = format!("wrote {} bytes to {path}", content.len());
                self.append_lint(cx, &path, pre_content.as_deref(), &mut msg)
                    .await;
                ToolOutcome::text(call.call_id.clone(), true, msg).with_detail(Self::detail(
                    "write",
                    &path,
                    content.len(),
                    None,
                ))
            }
            Err(e) => ToolOutcome::text(call.call_id.clone(), false, format!("fs write: {e}")),
        }
    }

    #[allow(clippy::too_many_lines)] // one linear op pipeline; splitting would obscure the flow
    async fn op_edit(
        &self,
        call: &ToolCall,
        cx: &TurnCx<'_>,
        path: String,
        old_string: String,
        new_string: String,
        replace_all: bool,
    ) -> ToolOutcome {
        let bytes = match cx.exec.read(Path::new(&path)).await {
            Ok(bytes) => bytes,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("fs edit (read): {e}"),
                )
            }
        };
        let raw = String::from_utf8_lossy(&bytes).into_owned();
        // Strip a leading BOM before matching (a phantom U+FEFF defeats an exact first-line
        // match); restore it on the way back out.
        let (original, had_bom) = match raw.strip_prefix('\u{feff}') {
            Some(stripped) => (stripped.to_string(), true),
            None => (raw, false),
        };

        let replaced =
            match fuzzy::fuzzy_find_and_replace(&original, &old_string, &new_string, replace_all) {
                Ok(replaced) => replaced,
                Err(e) => {
                    let hint = fuzzy::format_no_match_hint(&e, &old_string, &original);
                    return ToolOutcome::text(
                        call.call_id.clone(),
                        false,
                        format!("fs edit: {e}{hint}"),
                    );
                }
            };

        // Line-ending preservation (hermes patch_replace parity): the substituted region arrives
        // LF through JSON; re-normalize the whole result to the file's dominant ending.
        let mut new_content = replaced.content;
        if detect_line_ending(&original) == Some("\r\n") {
            new_content = normalize_line_endings(&new_content, "\r\n");
        }

        let workspace = cx.exec.cwd().to_path_buf();
        let resolved = match contain(&workspace, Path::new(&path)) {
            Ok(resolved) => resolved,
            Err(e) => {
                return ToolOutcome::text(call.call_id.clone(), false, format!("fs edit: {e}"))
            }
        };
        if let Some(denied) = self.write_denied(&resolved) {
            return ToolOutcome::text(call.call_id.clone(), false, denied);
        }

        let prompt = format!(
            "approve edit to {path}: {} replacement(s) of {:?} (strategy: {})",
            replaced.count,
            preview(&old_string),
            replaced.strategy
        );
        if let Err(out) = Self::gate(call, cx, &path, prompt).await {
            return out;
        }

        let diff = unified_diff(&original, &new_content, &path);
        let write_back = if had_bom {
            format!("\u{feff}{new_content}")
        } else {
            new_content.clone()
        };
        match Self::atomic_write(&workspace, &path, write_back.as_bytes()).await {
            Ok(_) => {
                let mut msg = format!(
                    "edited {path}: {} replacement(s), strategy: {}\n\n{diff}",
                    replaced.count, replaced.strategy
                );
                self.append_lint(cx, &path, Some(&original), &mut msg).await;
                ToolOutcome::text(call.call_id.clone(), true, msg).with_detail(Self::detail(
                    "edit",
                    &path,
                    replaced.count,
                    Some(replaced.strategy),
                ))
            }
            Err(e) => {
                ToolOutcome::text(call.call_id.clone(), false, format!("fs edit (write): {e}"))
            }
        }
    }

    async fn op_grep(
        &self,
        call: &ToolCall,
        req: Box<search::GrepRequest>,
        path: &str,
    ) -> ToolOutcome {
        if !tokio::fs::try_exists(&req.root).await.unwrap_or(false) {
            return ToolOutcome::text(
                call.call_id.clone(),
                false,
                format!("fs grep: path not found: {path}"),
            );
        }
        match tokio::task::spawn_blocking(move || search::grep(&req)).await {
            Ok(Ok(text)) => ToolOutcome::text(call.call_id.clone(), true, text)
                .with_detail(Self::detail("grep", path, 0, None)),
            Ok(Err(e)) => ToolOutcome::text(call.call_id.clone(), false, format!("fs grep: {e}")),
            Err(_) => ToolOutcome::text(call.call_id.clone(), false, "fs grep: search panicked"),
        }
    }

    async fn op_delete(&self, call: &ToolCall, cx: &TurnCx<'_>, path: String) -> ToolOutcome {
        let workspace = cx.exec.cwd().to_path_buf();
        let resolved = match contain(&workspace, Path::new(&path)) {
            Ok(resolved) => resolved,
            Err(e) => {
                return ToolOutcome::text(call.call_id.clone(), false, format!("fs delete: {e}"))
            }
        };
        if let Some(denied) = self.write_denied(&resolved) {
            return ToolOutcome::text(call.call_id.clone(), false, denied);
        }
        let prompt = format!("approve delete of {path}");
        if let Err(out) = Self::gate(call, cx, &path, prompt).await {
            return out;
        }
        // fd-contained metadata + unlink (a symlinked component anywhere in `path` is rejected; a
        // symlinked entry is removed as the link itself, never followed out of the workspace).
        let cr = match ContainedRoot::open(&workspace) {
            Ok(cr) => cr,
            Err(e) => {
                return ToolOutcome::text(call.call_id.clone(), false, format!("fs delete: {e}"))
            }
        };
        let rel = Path::new(&path);
        let result = match cr.symlink_metadata(rel).await {
            Ok(meta) if meta.is_dir => cr.remove_dir(rel).await.map_err(|e| {
                if e.kind() == std::io::ErrorKind::DirectoryNotEmpty {
                    std::io::Error::new(
                        e.kind(),
                        "directory not empty (recursive delete is not offered; remove entries first)",
                    )
                } else {
                    e
                }
            }),
            Ok(_) => cr.remove_file(rel).await,
            Err(e) => Err(e),
        };
        match result {
            Ok(()) => ToolOutcome::text(call.call_id.clone(), true, format!("deleted {path}"))
                .with_detail(Self::detail("delete", &path, 1, None)),
            Err(e) => ToolOutcome::text(call.call_id.clone(), false, format!("fs delete: {e}")),
        }
    }
}

/// A short single-line preview of a string for approval prompts.
fn preview(s: &str) -> String {
    let flat: String = s.chars().take(PREVIEW_CHARS).collect();
    let flat = flat.replace('\n', "\\n");
    if s.chars().count() > PREVIEW_CHARS {
        format!("{flat}…")
    } else {
        flat
    }
}

/// The dominant line ending of `sample`: `"\r\n"` when any CRLF is present, `"\n"` when only
/// bare LFs are, `None` for a single-line sample (hermes `_detect_line_ending`).
fn detect_line_ending(sample: &str) -> Option<&'static str> {
    if sample.contains("\r\n") {
        Some("\r\n")
    } else if sample.contains('\n') {
        Some("\n")
    } else {
        None
    }
}

/// Normalize every line ending in `text` to `target`.
fn normalize_line_endings(text: &str, target: &str) -> String {
    let unified = text.replace("\r\n", "\n").replace('\r', "\n");
    if target == "\n" {
        unified
    } else {
        unified.replace('\n', target)
    }
}

/// A capped unified diff for an `edit` result.
fn unified_diff(original: &str, new: &str, path: &str) -> String {
    let diff = similar::TextDiff::from_lines(original, new);
    let rendered = diff
        .unified_diff()
        .context_radius(3)
        .header(&format!("a/{path}"), &format!("b/{path}"))
        .to_string();
    if rendered.chars().count() <= MAX_DIFF_CHARS {
        return rendered;
    }
    let cut: String = rendered.chars().take(MAX_DIFF_CHARS).collect();
    format!("{cut}\n... [diff truncated]")
}

/// Normalize a path argument for the batch overlap gate: `.`/`./` components collapsed so
/// `./a.txt` and `a.txt` compare equal under the engine's component-prefix check.
fn scope_path(path: &str) -> PathBuf {
    let normalized: PathBuf = Path::new(path)
        .components()
        .filter(|c| !matches!(c, Component::CurDir))
        .collect();
    if normalized.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        normalized
    }
}

#[async_trait]
impl Tool for FsTool {
    fn name(&self) -> &str {
        "fs"
    }

    fn schema(&self) -> &str {
        r#"{"type":"object","properties":{"op":{"type":"string","enum":["read","write","list","edit","grep","glob","delete"],"description":"read: paginated file read with LINE|content gutter (docx/xlsx/ipynb/pdf auto-extracted to text). write: create/replace a file atomically. list: directory entries. edit: fuzzy find-and-replace (whitespace/indentation/unicode drift tolerated). grep: regex content search, gitignore-respecting. glob: find files by pattern, most-recently modified first. delete: remove a file or empty directory."},"path":{"type":"string","description":"workspace-relative path (grep/glob: the directory to search, default '.')"},"offset":{"type":"integer","description":"read: 1-indexed start line; negative reads from the end (-10 = last 10 lines)"},"limit":{"type":"integer","description":"read: max lines returned"},"content":{"type":"string","description":"write: the complete file content"},"old_string":{"type":"string","description":"edit: the text to find; must be unique unless replace_all"},"new_string":{"type":"string","description":"edit: the replacement text ('' deletes the match)"},"replace_all":{"type":"boolean","description":"edit: replace every occurrence (default false)"},"pattern":{"type":"string","description":"grep: regex; glob: file pattern like '*.rs' or 'src/**/*.ts'"},"glob":{"type":"string","description":"grep: only search files matching this glob"},"output_mode":{"type":"string","enum":["content","files_with_matches","count"],"description":"grep output shape (default content)"},"context":{"type":"integer","description":"grep: context lines around each match"},"case_insensitive":{"type":"boolean","description":"grep: case-insensitive matching"},"head_limit":{"type":"integer","description":"grep/glob: max results"},"search_offset":{"type":"integer","description":"grep/glob: skip N results (pagination)"}},"required":["op"]}"#
    }

    fn mutates(&self) -> bool {
        // Conservative call-independent fallback; the pipeline consults `mutates_for` per call.
        true
    }

    fn mutates_for(&self, call: &ToolCall) -> bool {
        // Only write/edit/delete touch the workspace; reads skip the §12 checkpoint stage and
        // count as guardrail-idempotent. Unparseable args are treated as mutating (safe).
        serde_json::from_str::<FsArgs>(&call.args).map_or(true, |args| args.mutating())
    }

    fn concurrency_for(&self, call: &ToolCall) -> ToolConcurrency {
        match serde_json::from_str::<FsArgs>(&call.args) {
            Ok(args) if !args.mutating() => ToolConcurrency::Parallel,
            _ => ToolConcurrency::Exclusive,
        }
    }

    fn parallel_scope_paths(&self, call: &ToolCall) -> Option<Vec<PathBuf>> {
        serde_json::from_str::<FsArgs>(&call.args)
            .ok()
            .map(|args| vec![scope_path(args.target_path())])
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let args: FsArgs = match serde_json::from_str(&call.args) {
            Ok(args) => args,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("fs: invalid arguments: {e}"),
                )
            }
        };

        match args {
            FsArgs::Read {
                path,
                offset,
                limit,
            } => self.op_read(call, cx, path, offset, limit).await,
            FsArgs::Write { path, content } => self.op_write(call, cx, path, content).await,
            FsArgs::List { path } => match cx.exec.list(Path::new(&path)).await {
                Ok(entries) => {
                    let count = entries.len();
                    ToolOutcome::text(call.call_id.clone(), true, entries.join("\n"))
                        .with_detail(Self::detail("list", &path, count, None))
                }
                Err(e) => ToolOutcome::text(call.call_id.clone(), false, format!("fs list: {e}")),
            },
            FsArgs::Edit {
                path,
                old_string,
                new_string,
                replace_all,
            } => {
                self.op_edit(call, cx, path, old_string, new_string, replace_all)
                    .await
            }
            FsArgs::Grep {
                pattern,
                path,
                glob,
                output_mode,
                context,
                case_insensitive,
                head_limit,
                search_offset,
            } => {
                let workspace = cx.exec.cwd().to_path_buf();
                // openat2-verified entry point: proves no symlink escape to the subtree root. The
                // walker itself is `follow_links(false)`, so it never traverses a symlink thereafter.
                let root = match Self::verify_search_root(&workspace, &path).await {
                    Ok(root) => root,
                    Err(e) => {
                        return ToolOutcome::text(
                            call.call_id.clone(),
                            false,
                            format!("fs grep: {e}"),
                        )
                    }
                };
                let req = Box::new(search::GrepRequest {
                    pattern,
                    root,
                    workspace,
                    glob,
                    case_insensitive,
                    context,
                    output_mode,
                    head_limit: head_limit.unwrap_or(self.cfg.search_result_cap).max(1),
                    offset: search_offset,
                });
                self.op_grep(call, req, &path).await
            }
            FsArgs::Glob {
                pattern,
                path,
                head_limit,
                search_offset,
            } => {
                let workspace = cx.exec.cwd().to_path_buf();
                let root = match Self::verify_search_root(&workspace, &path).await {
                    Ok(root) => root,
                    Err(e) => {
                        return ToolOutcome::text(
                            call.call_id.clone(),
                            false,
                            format!("fs glob: {e}"),
                        )
                    }
                };
                if !tokio::fs::try_exists(&root).await.unwrap_or(false) {
                    return ToolOutcome::text(
                        call.call_id.clone(),
                        false,
                        format!("fs glob: path not found: {path}"),
                    );
                }
                let req = search::GlobRequest {
                    pattern,
                    root,
                    workspace,
                    head_limit: head_limit.unwrap_or(self.cfg.search_result_cap).max(1),
                    offset: search_offset,
                };
                match tokio::task::spawn_blocking(move || search::glob(&req)).await {
                    Ok(Ok(text)) => ToolOutcome::text(call.call_id.clone(), true, text)
                        .with_detail(Self::detail("glob", &path, 0, None)),
                    Ok(Err(e)) => {
                        ToolOutcome::text(call.call_id.clone(), false, format!("fs glob: {e}"))
                    }
                    Err(_) => {
                        ToolOutcome::text(call.call_id.clone(), false, "fs glob: search panicked")
                    }
                }
            }
            FsArgs::Delete { path } => self.op_delete(call, cx, path).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_common::{Budget, SessionId};
    use daemon_core::{ApprovalPolicy, EventSink, LocalEnvironment};
    use daemon_protocol::{HostRequest, HostRequestHandler, HostResponse, HostResponseBody};
    use tokio_util::sync::CancellationToken;

    struct FixedHost(bool);
    #[async_trait]
    impl HostRequestHandler for FixedHost {
        async fn request(&self, req: HostRequest) -> HostResponse {
            HostResponse {
                request_id: req.request_id,
                body: HostResponseBody::Approved {
                    approved: self.0,
                    allow_permanent: false,
                    reason: None,
                },
            }
        }
    }

    fn temp_root(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("daemon-tool-fs-test-{tag}-{nanos}"))
    }

    async fn run_with(
        tool: &FsTool,
        env: &LocalEnvironment,
        host: &dyn HostRequestHandler,
        args: &str,
    ) -> ToolOutcome {
        let cancel = CancellationToken::new();
        let events = EventSink::discarding();
        let cx = TurnCx {
            cancel,
            events: &events,
            host,
            session_id: SessionId::new("t"),
            profile: None,
            budget: Budget::unlimited(),
            exec: env,
            tool_result_budget: 0,
            approval_policy: ApprovalPolicy::AutoAllow,
            pre_approved: false,
            checkpoints: None,
            tool_timeout: None,
            session_allow: &[],
        };
        let call = ToolCall {
            call_id: "c1".into(),
            name: "fs".into(),
            args: args.into(),
        };
        tool.run(&call, &cx).await
    }

    async fn run(env: &LocalEnvironment, args: &str) -> ToolOutcome {
        run_with(&FsTool::new(), env, &FixedHost(true), args).await
    }

    #[tokio::test]
    async fn write_then_read_roundtrips_with_gutter_and_detail() {
        let root = temp_root("rw");
        let env = LocalEnvironment::new(&root);
        let w = run(
            &env,
            r#"{"op":"write","path":"a.txt","content":"hello\nworld"}"#,
        )
        .await;
        assert!(w.result.ok, "{}", w.result.content);
        assert!(w.detail.is_some());
        assert!(w.result.content.contains("wrote 11 bytes"));

        let r = run(&env, r#"{"op":"read","path":"a.txt"}"#).await;
        assert!(r.result.ok);
        assert_eq!(r.result.content, "1|hello\n2|world");
        let detail = r.detail.expect("read detail");
        assert_eq!(detail.kind, "fs");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn read_paginates_and_tail_reads() {
        let root = temp_root("page");
        let env = LocalEnvironment::new(&root);
        run(
            &env,
            r#"{"op":"write","path":"n.txt","content":"l1\nl2\nl3\nl4\nl5"}"#,
        )
        .await;
        let r = run(&env, r#"{"op":"read","path":"n.txt","offset":2,"limit":2}"#).await;
        assert!(r.result.ok);
        assert!(
            r.result.content.starts_with("2|l2\n3|l3"),
            "{}",
            r.result.content
        );
        assert!(r.result.content.contains("continue with offset=4"));

        let tail = run(&env, r#"{"op":"read","path":"n.txt","offset":-2}"#).await;
        assert_eq!(tail.result.content, "4|l4\n5|l5");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn read_caps_oversized_output() {
        let root = temp_root("cap");
        let env = LocalEnvironment::new(&root);
        let tool = FsTool::with_config(FsConfig {
            max_read_chars: 10,
            ..FsConfig::default()
        });
        run(
            &env,
            r#"{"op":"write","path":"big.txt","content":"0123456789abcdef"}"#,
        )
        .await;
        let r = run_with(
            &tool,
            &env,
            &FixedHost(true),
            r#"{"op":"read","path":"big.txt"}"#,
        )
        .await;
        assert!(!r.result.ok);
        assert!(
            r.result.content.contains("Use offset and limit"),
            "{}",
            r.result.content
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn read_refuses_binary_content() {
        let root = temp_root("bin");
        let env = LocalEnvironment::new(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("blob"), b"ab\x00cd").unwrap();
        let r = run(&env, r#"{"op":"read","path":"blob"}"#).await;
        assert!(!r.result.ok);
        assert!(r.result.content.contains("binary"), "{}", r.result.content);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn edit_exact_and_replace_all() {
        let root = temp_root("edit");
        let env = LocalEnvironment::new(&root);
        run(
            &env,
            r#"{"op":"write","path":"a.txt","content":"foo bar foo"}"#,
        )
        .await;
        // Two occurrences without replace_all: ambiguous.
        let e = run(
            &env,
            r#"{"op":"edit","path":"a.txt","old_string":"foo","new_string":"baz"}"#,
        )
        .await;
        assert!(!e.result.ok);
        assert!(
            e.result.content.contains("Found 2 matches"),
            "{}",
            e.result.content
        );

        let e = run(
            &env,
            r#"{"op":"edit","path":"a.txt","old_string":"foo","new_string":"baz","replace_all":true}"#,
        )
        .await;
        assert!(e.result.ok, "{}", e.result.content);
        assert!(e.result.content.contains("2 replacement(s)"));
        let r = run(&env, r#"{"op":"read","path":"a.txt"}"#).await;
        assert_eq!(r.result.content, "1|baz bar baz");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn edit_legacy_find_replace_aliases_still_work() {
        let root = temp_root("legacy");
        let env = LocalEnvironment::new(&root);
        run(&env, r#"{"op":"write","path":"a.txt","content":"foo bar"}"#).await;
        let e = run(
            &env,
            r#"{"op":"edit","path":"a.txt","find":"foo","replace":"qux"}"#,
        )
        .await;
        assert!(e.result.ok, "{}", e.result.content);
        let r = run(&env, r#"{"op":"read","path":"a.txt"}"#).await;
        assert_eq!(r.result.content, "1|qux bar");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn edit_fuzzy_indentation_and_did_you_mean() {
        let root = temp_root("fuzzy");
        let env = LocalEnvironment::new(&root);
        run(
            &env,
            r#"{"op":"write","path":"m.py","content":"def f():\n    if x:\n        go()\n"}"#,
        )
        .await;
        // The model sends 2-space indentation; the file uses 4: the fuzzy chain absorbs it and
        // the replacement is re-anchored to the file's indent.
        let e = run(
            &env,
            r#"{"op":"edit","path":"m.py","old_string":"  if x:\n      go()","new_string":"  if y:\n      stop()"}"#,
        )
        .await;
        assert!(e.result.ok, "{}", e.result.content);
        let r = run(&env, r#"{"op":"read","path":"m.py"}"#).await;
        assert!(
            r.result.content.contains("    if y:"),
            "reindented to the file's 4-space base: {}",
            r.result.content
        );

        // A no-match failure carries the "did you mean" snippet.
        let miss = run(
            &env,
            r#"{"op":"edit","path":"m.py","old_string":"if z and q:","new_string":"x"}"#,
        )
        .await;
        assert!(!miss.result.ok);
        assert!(
            miss.result.content.contains("Did you mean"),
            "{}",
            miss.result.content
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn grep_and_glob_search_the_workspace() {
        let root = temp_root("search");
        let env = LocalEnvironment::new(&root);
        run(
            &env,
            r#"{"op":"write","path":"src/lib.rs","content":"pub fn alpha() {}\n"}"#,
        )
        .await;
        run(
            &env,
            r#"{"op":"write","path":"notes.md","content":"alpha notes\n"}"#,
        )
        .await;
        let g = run(&env, r#"{"op":"grep","pattern":"alpha","glob":"*.rs"}"#).await;
        assert!(g.result.ok, "{}", g.result.content);
        assert!(g.result.content.contains("src/lib.rs:1:pub fn alpha() {}"));
        assert!(!g.result.content.contains("notes.md"));

        let f = run(&env, r#"{"op":"glob","pattern":"*.rs"}"#).await;
        assert!(f.result.ok);
        assert!(f.result.content.contains("src/lib.rs"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn delete_is_gated_and_removes_files() {
        let root = temp_root("del");
        let env = LocalEnvironment::new(&root);
        run(&env, r#"{"op":"write","path":"gone.txt","content":"x"}"#).await;
        let d = run(&env, r#"{"op":"delete","path":"gone.txt"}"#).await;
        assert!(d.result.ok, "{}", d.result.content);
        let r = run(&env, r#"{"op":"read","path":"gone.txt"}"#).await;
        assert!(!r.result.ok, "file is gone");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn delete_denied_by_operator_does_not_remove() {
        let root = temp_root("delete-gate");
        let env = LocalEnvironment::new(&root);
        run(&env, r#"{"op":"write","path":"keep.txt","content":"x"}"#).await;
        // Ask policy + denying host: the delete is rejected and the file survives.
        let cancel = CancellationToken::new();
        let events = EventSink::discarding();
        let host = FixedHost(false);
        let cx = TurnCx {
            cancel,
            events: &events,
            host: &host,
            session_id: SessionId::new("t"),
            profile: None,
            budget: Budget::unlimited(),
            exec: &env,
            tool_result_budget: 0,
            approval_policy: ApprovalPolicy::Ask,
            pre_approved: false,
            checkpoints: None,
            tool_timeout: None,
            session_allow: &[],
        };
        let call = ToolCall {
            call_id: "c1".into(),
            name: "fs".into(),
            args: r#"{"op":"delete","path":"keep.txt"}"#.into(),
        };
        let d = FsTool::new().run(&call, &cx).await;
        assert!(!d.result.ok);
        let r = run(&env, r#"{"op":"read","path":"keep.txt"}"#).await;
        assert!(r.result.ok, "denied delete must not remove the file");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn out_of_workspace_write_is_rejected() {
        let root = temp_root("escape");
        let env = LocalEnvironment::new(&root);
        let w = run(
            &env,
            r#"{"op":"write","path":"../escaped.txt","content":"x"}"#,
        )
        .await;
        assert!(!w.result.ok, "an out-of-workspace write must fail");
        assert!(w.result.content.contains("escapes the workspace"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn atomic_write_leaves_no_temp_files() {
        let root = temp_root("atomic");
        let env = LocalEnvironment::new(&root);
        run(&env, r#"{"op":"write","path":"sub/f.txt","content":"v1"}"#).await;
        run(&env, r#"{"op":"write","path":"sub/f.txt","content":"v2"}"#).await;
        let leftovers: Vec<_> = std::fs::read_dir(root.join("sub"))
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains("daemon-fs-tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temp files leaked: {leftovers:?}");
        let r = run(&env, r#"{"op":"read","path":"sub/f.txt"}"#).await;
        assert_eq!(r.result.content, "1|v2");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn crlf_files_keep_their_endings_through_edit() {
        let root = temp_root("crlf");
        let env = LocalEnvironment::new(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("w.txt"), "one\r\ntwo\r\nthree\r\n").unwrap();
        let e = run(
            &env,
            r#"{"op":"edit","path":"w.txt","old_string":"two","new_string":"TWO"}"#,
        )
        .await;
        assert!(e.result.ok, "{}", e.result.content);
        let bytes = std::fs::read(root.join("w.txt")).unwrap();
        assert_eq!(String::from_utf8_lossy(&bytes), "one\r\nTWO\r\nthree\r\n");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn lint_delta_reports_only_new_diagnostics() {
        let root = temp_root("lint");
        std::fs::create_dir_all(&root).unwrap();
        // A real linter shape: prints one line per `ERR` occurrence (grep -n's `line:text`) and
        // exits non-zero when any exist.
        std::fs::write(
            root.join("lintcheck.sh"),
            "#!/bin/sh\nif grep -n ERR \"$1\"; then exit 1; else exit 0; fi\n",
        )
        .unwrap();
        let env = LocalEnvironment::new(&root);
        let tool = FsTool::with_config(FsConfig {
            lint: FsLintConfig {
                enabled: true,
                timeout_ms: 5_000,
                output_cap: 4_096,
                commands: vec![LintRule {
                    globs: vec!["*.txt".into()],
                    command: "sh lintcheck.sh {file}".into(),
                }],
            },
            ..FsConfig::default()
        });
        let host = FixedHost(true);

        // A clean write is the hot path: no lint block at all.
        let w = run_with(
            &tool,
            &env,
            &host,
            r#"{"op":"write","path":"l.txt","content":"all good\n"}"#,
        )
        .await;
        assert!(w.result.ok);
        assert!(!w.result.content.contains("lint:"), "{}", w.result.content);

        // Plant a pre-existing diagnostic (no pre-content: the block carries the full output).
        let w = run_with(
            &tool,
            &env,
            &host,
            r#"{"op":"write","path":"l.txt","content":"ok\nERR one\n"}"#,
        )
        .await;
        assert!(w.result.ok);
        assert!(w.result.content.contains("lint:"), "{}", w.result.content);
        assert!(w.result.content.contains("ERR one"), "{}", w.result.content);

        // An edit that adds a SECOND diagnostic below the first (below, so pre-existing lines
        // keep their numbers — the delta is line-literal, hermes parity): the delta filters the
        // pre-existing diagnostic and surfaces only the new one.
        let e = run_with(
            &tool,
            &env,
            &host,
            r#"{"op":"edit","path":"l.txt","old_string":"ERR one","new_string":"ERR one\nERR two"}"#,
        )
        .await;
        assert!(e.result.ok, "{}", e.result.content);
        assert!(
            e.result.content.contains("pre-existing filtered out"),
            "{}",
            e.result.content
        );
        assert!(e.result.content.contains("ERR two"), "{}", e.result.content);
        let lint_block = e.result.content.split("lint:").nth(1).expect("lint block");
        assert!(
            !lint_block.contains("ERR one"),
            "pre-existing diagnostic leaked into the delta: {}",
            e.result.content
        );

        // An edit that changes nothing lint-relevant while the file stays broken: annotated as
        // pre-existing-only, never silently dropped.
        let e = run_with(
            &tool,
            &env,
            &host,
            r#"{"op":"edit","path":"l.txt","old_string":"ok","new_string":"fine"}"#,
        )
        .await;
        assert!(e.result.ok, "{}", e.result.content);
        assert!(
            e.result.content.contains("pre-existing problems only"),
            "{}",
            e.result.content
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn deny_paths_block_mutations() {
        let root = temp_root("deny");
        let env = LocalEnvironment::new(&root);
        let tool = FsTool::with_config(FsConfig {
            deny_paths: vec![root.join("secrets").to_string_lossy().into_owned()],
            ..FsConfig::default()
        });
        let w = run_with(
            &tool,
            &env,
            &FixedHost(true),
            r#"{"op":"write","path":"secrets/key.pem","content":"k"}"#,
        )
        .await;
        assert!(!w.result.ok);
        assert!(
            w.result.content.contains("write denied"),
            "{}",
            w.result.content
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn per_call_seams_classify_ops() {
        let tool = FsTool::new();
        let call = |args: &str| ToolCall {
            call_id: "c".into(),
            name: "fs".into(),
            args: args.into(),
        };
        let read = call(r#"{"op":"read","path":"a.txt"}"#);
        let grep = call(r#"{"op":"grep","pattern":"x"}"#);
        let write = call(r#"{"op":"write","path":"a.txt","content":"x"}"#);
        let bad = call("not json");

        assert_eq!(tool.concurrency_for(&read), ToolConcurrency::Parallel);
        assert_eq!(tool.concurrency_for(&grep), ToolConcurrency::Parallel);
        assert_eq!(tool.concurrency_for(&write), ToolConcurrency::Exclusive);
        assert_eq!(tool.concurrency_for(&bad), ToolConcurrency::Exclusive);

        assert!(!tool.mutates_for(&read));
        assert!(tool.mutates_for(&write));
        assert!(tool.mutates_for(&bad), "unparseable args stay conservative");

        assert_eq!(
            tool.parallel_scope_paths(&read),
            Some(vec![PathBuf::from("a.txt")])
        );
        assert_eq!(
            tool.parallel_scope_paths(&call(r#"{"op":"read","path":"./a.txt"}"#)),
            Some(vec![PathBuf::from("a.txt")]),
            "CurDir components collapse for the overlap gate"
        );
        assert_eq!(
            tool.parallel_scope_paths(&grep),
            Some(vec![PathBuf::from(".")])
        );
        assert_eq!(tool.parallel_scope_paths(&bad), None);
    }

    #[cfg(feature = "extract")]
    #[tokio::test]
    async fn notebook_reads_extract_to_text() {
        let root = temp_root("nb");
        let env = LocalEnvironment::new(&root);
        std::fs::create_dir_all(&root).unwrap();
        let nb = serde_json::json!({
            "cells": [{"cell_type": "code", "source": "print('extracted!')\n"}]
        });
        std::fs::write(root.join("demo.ipynb"), serde_json::to_vec(&nb).unwrap()).unwrap();
        let r = run(&env, r#"{"op":"read","path":"demo.ipynb"}"#).await;
        assert!(r.result.ok, "{}", r.result.content);
        assert!(r.result.content.contains("print('extracted!')"));
        assert!(r.result.content.contains("Code cell 1"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(feature = "extract")]
    #[tokio::test]
    async fn malformed_document_falls_back_to_binary_refusal() {
        let root = temp_root("badxlsx");
        let env = LocalEnvironment::new(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("fake.xlsx"), b"definitely not a zip").unwrap();
        let r = run(&env, r#"{"op":"read","path":"fake.xlsx"}"#).await;
        assert!(!r.result.ok);
        assert!(
            r.result.content.contains("binary") && r.result.content.contains("extraction failed"),
            "{}",
            r.result.content
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
