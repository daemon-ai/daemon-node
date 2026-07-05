// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Post-edit diagnostics for the `fs` tool (`[fs.lint]`): a configurable per-workspace lint
//! command run after `write`/`edit`, **delta-filtered** so only diagnostics newly introduced by
//! the edit reach the model (hermes `_check_lint_delta` / Cursor `ReadLints` parity).
//!
//! This is deliberately the command-runner seam only — no LSP client. The configured command runs
//! through the session's [`ExecutionEnvironment`](daemon_core::ExecutionEnvironment) (same
//! containment and scrubbed env as the `shell` tool) under a hard timeout and an output cap, so a
//! slow or chatty linter can never stall or flood the turn.
//!
//! Delta algorithm: lint the file post-write; if clean, done (the hot path is one run). If it
//! reports problems and the pre-edit content is available, materialize the pre-edit content into
//! a sibling temp file (its name *ends with* the original filename so extension-driven linters
//! engage), lint that, rewrite the temp path in its output back to the real path (so lines
//! compare), and set-difference the stripped output lines. Note hermes' own shell-linter tier
//! silently skips this baseline (it re-lints the on-disk file, which post-write *is* the new
//! content); the temp-file baseline here makes the delta actually correct for command runners.

use std::path::{Path, PathBuf};

use daemon_core::{Command, ExecCx, TurnCx};
use serde::{Deserialize, Serialize};

/// One lint rule: a command template gated on file patterns (`[[fs.lint.commands]]`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LintRule {
    /// File globs this rule applies to (`*.py` matches basenames; a pattern with `/` matches the
    /// workspace-relative path). Empty matches nothing.
    pub globs: Vec<String>,
    /// The command template, split on whitespace into argv (no shell). Every `{file}` token is
    /// replaced by the workspace-relative file path; without one, the path is appended as the
    /// final argument. Example: `"python -m py_compile {file}"`.
    pub command: String,
}

/// The `[fs.lint]` table: post-edit lint commands with delta filtering.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FsLintConfig {
    /// Whether post-edit lint runs at all (off by default).
    #[serde(with = "daemon_common::flex_bool")]
    pub enabled: bool,
    /// Hard wall-clock cap per lint invocation, in milliseconds.
    pub timeout_ms: u64,
    /// Cap on the lint output characters attached to a tool result.
    pub output_cap: usize,
    /// The rules, first match wins.
    pub commands: Vec<LintRule>,
}

impl Default for FsLintConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            timeout_ms: 5_000,
            output_cap: 4_096,
            commands: Vec::new(),
        }
    }
}

impl FsLintConfig {
    /// The first rule whose globs match `rel_path`, if any. Called before the write so the caller
    /// knows whether to capture pre-edit content.
    pub fn rule_for(&self, rel_path: &str) -> Option<&LintRule> {
        if !self.enabled {
            return None;
        }
        let path = Path::new(rel_path);
        let name = path.file_name().map(std::ffi::OsStr::to_string_lossy);
        self.commands.iter().find(|rule| {
            rule.globs.iter().any(|g| {
                let Ok(glob) = globset::Glob::new(g) else {
                    return false;
                };
                let matcher = glob.compile_matcher();
                if g.contains('/') {
                    matcher.is_match(path)
                } else {
                    name.as_deref().is_some_and(|n| matcher.is_match(n))
                }
            })
        })
    }
}

/// Split a command template into argv, substituting `{file}` (appending the path when the
/// template has no placeholder). Templates split on whitespace — no shell, no quoting.
fn render_command(template: &str, rel_path: &str) -> Option<Command> {
    let mut parts = template.split_whitespace();
    let program = parts.next()?.replace("{file}", rel_path);
    let mut cmd = Command::new(program);
    for part in parts {
        cmd = cmd.arg(part.replace("{file}", rel_path));
    }
    if !template.contains("{file}") {
        cmd = cmd.arg(rel_path);
    }
    Some(cmd)
}

/// One lint run: combined stdout+stderr plus whether the linter exited clean.
struct LintRun {
    ok: bool,
    output: String,
    timed_out: bool,
}

/// Run `cmd` through the execution environment under the configured hard timeout. On timeout the
/// child cancel token is fired so the environment kills the process (never leaks it), and the run
/// reports `timed_out`.
async fn run_linter(cx: &TurnCx<'_>, cfg: &FsLintConfig, cmd: Command) -> LintRun {
    let child = cx.cancel.child_token();
    let exec_cx = ExecCx { cancel: &child };
    let fut = cx.exec.run(cmd, &exec_cx);
    tokio::pin!(fut);
    let result = tokio::select! {
        r = &mut fut => r,
        () = tokio::time::sleep(std::time::Duration::from_millis(cfg.timeout_ms)) => {
            child.cancel();
            // Await the (now-cancelled) run so the environment reaps the process.
            let _ = fut.await;
            return LintRun { ok: true, output: String::new(), timed_out: true };
        }
    };
    match result {
        Ok(run) => LintRun {
            ok: run.exit_code == 0,
            output: format!("{}{}", run.stdout, run.stderr).trim().to_string(),
            timed_out: false,
        },
        // The linter could not run at all (missing binary, spawn failure): treat as skipped, not
        // as a diagnostic — a tooling gap is not the model's problem (hermes parity).
        Err(_) => LintRun {
            ok: true,
            output: String::new(),
            timed_out: false,
        },
    }
}

/// Run the post-edit lint with pre-edit delta filtering. Returns the block to append to the tool
/// result, or `None` when there is nothing worth surfacing (no rule, clean lint, tool gap).
///
/// `pre_content` is the file's content *before* the edit (`None` for a new file — every
/// diagnostic is then new by definition).
pub async fn lint_delta(
    cx: &TurnCx<'_>,
    cfg: &FsLintConfig,
    workspace: &Path,
    rel_path: &str,
    pre_content: Option<&str>,
) -> Option<String> {
    let rule = cfg.rule_for(rel_path)?;
    let cmd = render_command(&rule.command, rel_path)?;

    let post = run_linter(cx, cfg, cmd).await;
    if post.timed_out {
        return Some(format!(
            "lint: timed out after {}ms (skipped)",
            cfg.timeout_ms
        ));
    }
    if post.ok {
        return None; // hot path: clean post-edit lint.
    }
    if post.output.is_empty() {
        return Some("lint: linter exited non-zero with no output".to_string());
    }

    // Delta refinement: lint the pre-edit content from a sibling temp file, rewrite its path in
    // the output back to the real path, and drop every line that already existed.
    let mut new_lines: Vec<String> = Vec::new();
    let mut pre_available = false;
    if let Some(pre) = pre_content {
        if let Some(pre_output) = baseline_output(cx, cfg, rule, workspace, rel_path, pre).await {
            pre_available = true;
            let pre_set: std::collections::HashSet<&str> = pre_output
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .collect();
            new_lines = post
                .output
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !pre_set.contains(l))
                .map(str::to_string)
                .collect();
        }
    }

    let block = if pre_available && new_lines.is_empty() {
        format!(
            "lint: pre-existing problems only — this edit introduced no new diagnostics, but the \
             file still fails its linter:\n{}",
            cap(&post.output, cfg.output_cap)
        )
    } else if pre_available {
        format!(
            "lint: new diagnostics introduced by this edit (pre-existing filtered out):\n{}",
            cap(&new_lines.join("\n"), cfg.output_cap)
        )
    } else {
        format!("lint:\n{}", cap(&post.output, cfg.output_cap))
    };
    Some(block)
}

/// Lint the pre-edit content via a sibling temp file whose name ends with the original filename
/// (so extension-gated linters engage), then normalize its path mentions back to `rel_path`.
/// `None` when the baseline could not be produced (the caller then reports the full post output).
async fn baseline_output(
    cx: &TurnCx<'_>,
    cfg: &FsLintConfig,
    rule: &LintRule,
    workspace: &Path,
    rel_path: &str,
    pre_content: &str,
) -> Option<String> {
    let file_name = Path::new(rel_path).file_name()?.to_string_lossy();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let tmp_rel = match Path::new(rel_path).parent() {
        Some(parent) if parent != Path::new("") => {
            parent.join(format!(".fs-lint-pre.{nanos}.{file_name}"))
        }
        _ => PathBuf::from(format!(".fs-lint-pre.{nanos}.{file_name}")),
    };
    let tmp_abs = daemon_core::exec::contain(workspace, &tmp_rel).ok()?;
    tokio::fs::write(&tmp_abs, pre_content).await.ok()?;
    let tmp_rel_str = tmp_rel.to_string_lossy().into_owned();
    let cmd = render_command(&rule.command, &tmp_rel_str)?;
    let run = run_linter(cx, cfg, cmd).await;
    let _ = tokio::fs::remove_file(&tmp_abs).await;
    if run.timed_out {
        return None;
    }
    // Rewrite both the relative and absolute temp paths so lines compare against the post run.
    Some(
        run.output
            .replace(&tmp_abs.to_string_lossy().into_owned(), rel_path)
            .replace(&tmp_rel_str, rel_path),
    )
}

/// Cut a diagnostics block at `max_chars`, marking the cut.
fn cap(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let cut: String = text.chars().take(max_chars).collect();
    format!("{cut}\n... [lint output truncated]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_matching_gates_on_enabled_and_globs() {
        let cfg = FsLintConfig {
            enabled: true,
            commands: vec![
                LintRule {
                    globs: vec!["*.py".into()],
                    command: "python -m py_compile {file}".into(),
                },
                LintRule {
                    globs: vec!["src/**/*.rs".into()],
                    command: "rustc --emit=metadata {file}".into(),
                },
            ],
            ..FsLintConfig::default()
        };
        assert!(cfg.rule_for("pkg/mod.py").is_some());
        assert!(cfg.rule_for("src/deep/lib.rs").is_some());
        assert!(cfg.rule_for("lib.rs").is_none(), "path glob needs the dir");
        assert!(cfg.rule_for("notes.txt").is_none());

        let off = FsLintConfig {
            enabled: false,
            ..cfg
        };
        assert!(off.rule_for("pkg/mod.py").is_none());
    }

    #[test]
    fn command_template_substitutes_or_appends() {
        let cmd = render_command("python -m py_compile {file}", "a.py").unwrap();
        assert_eq!(cmd.program, "python");
        assert_eq!(cmd.args, vec!["-m", "py_compile", "a.py"]);

        let cmd = render_command("mylint --strict", "a.py").unwrap();
        assert_eq!(cmd.args, vec!["--strict", "a.py"]);

        assert!(render_command("", "a.py").is_none());
    }
}
