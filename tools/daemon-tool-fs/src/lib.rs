//! `daemon-tool-fs` — the filesystem tool (§12/§13), a `daemon_core::Tool`.
//!
//! Read, write, list, and edit files through the engine's [`ExecutionEnvironment`](daemon_core::ExecutionEnvironment)
//! — never the raw filesystem — so every path is resolved against and contained within the session's
//! workspace. The env is the sandbox boundary: an out-of-workspace path is rejected by
//! [`contain`](daemon_core::exec) and surfaced as a failed tool result, never an escape. Each result
//! also carries a structured [`ToolDetail`] envelope (`kind = "fs"`) for a rich transcript consumer.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use daemon_core::{Tool, ToolCall, ToolOutcome, TurnCx};
use daemon_protocol::ToolDetail;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The filesystem operations the tool exposes.
#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
enum FsArgs {
    /// Read a file's contents.
    Read { path: String },
    /// Write (create/replace) a file.
    Write { path: String, content: String },
    /// List a directory's entries.
    List {
        #[serde(default = "dot")]
        path: String,
    },
    /// Replace the first occurrence of `find` with `replace` in a file.
    Edit {
        path: String,
        find: String,
        replace: String,
    },
}

fn dot() -> String {
    ".".into()
}

/// The structured detail attached to an fs result (opaque to the daemon; rendered by `kind`).
#[derive(Debug, Serialize)]
struct FsDetail<'a> {
    op: &'a str,
    path: &'a str,
    /// Bytes read/written, or entry count for a list.
    count: usize,
}

/// The filesystem tool.
#[derive(Default)]
pub struct FsTool;

impl FsTool {
    /// A new filesystem tool.
    pub fn new() -> Self {
        Self
    }

    fn detail(op: &str, path: &str, count: usize) -> ToolDetail {
        let body = serde_json::to_vec(&FsDetail { op, path, count }).unwrap_or_default();
        ToolDetail {
            kind: "fs".into(),
            body,
        }
    }
}

#[async_trait]
impl Tool for FsTool {
    fn name(&self) -> &str {
        "fs"
    }

    fn schema(&self) -> &str {
        r#"{"type":"object","properties":{"op":{"type":"string","enum":["read","write","list","edit"]},"path":{"type":"string"},"content":{"type":"string"},"find":{"type":"string"},"replace":{"type":"string"}},"required":["op"]}"#
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
            FsArgs::Read { path } => match cx.exec.read(Path::new(&path)).await {
                Ok(bytes) => {
                    let len = bytes.len();
                    let content = String::from_utf8_lossy(&bytes).into_owned();
                    ToolOutcome::text(call.call_id.clone(), true, content)
                        .with_detail(Self::detail("read", &path, len))
                }
                Err(e) => ToolOutcome::text(call.call_id.clone(), false, format!("fs read: {e}")),
            },
            FsArgs::Write { path, content } => {
                match cx.exec.write(Path::new(&path), content.as_bytes()).await {
                    Ok(()) => ToolOutcome::text(
                        call.call_id.clone(),
                        true,
                        format!("wrote {} bytes to {path}", content.len()),
                    )
                    .with_detail(Self::detail("write", &path, content.len())),
                    Err(e) => {
                        ToolOutcome::text(call.call_id.clone(), false, format!("fs write: {e}"))
                    }
                }
            }
            FsArgs::List { path } => match cx.exec.list(Path::new(&path)).await {
                Ok(entries) => {
                    let count = entries.len();
                    ToolOutcome::text(call.call_id.clone(), true, entries.join("\n"))
                        .with_detail(Self::detail("list", &path, count))
                }
                Err(e) => ToolOutcome::text(call.call_id.clone(), false, format!("fs list: {e}")),
            },
            FsArgs::Edit {
                path,
                find,
                replace,
            } => {
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
                let original = String::from_utf8_lossy(&bytes).into_owned();
                if !original.contains(&find) {
                    return ToolOutcome::text(
                        call.call_id.clone(),
                        false,
                        format!("fs edit: `find` text not present in {path}"),
                    );
                }
                let edited = original.replacen(&find, &replace, 1);
                match cx.exec.write(Path::new(&path), edited.as_bytes()).await {
                    Ok(()) => {
                        ToolOutcome::text(call.call_id.clone(), true, format!("edited {path}"))
                            .with_detail(Self::detail("edit", &path, edited.len()))
                    }
                    Err(e) => ToolOutcome::text(
                        call.call_id.clone(),
                        false,
                        format!("fs edit (write): {e}"),
                    ),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_common::{Budget, SessionId};
    use daemon_core::{EventSink, LocalEnvironment};
    use daemon_protocol::{HostRequest, HostRequestHandler, HostResponse, HostResponseBody};
    use std::path::PathBuf;
    use tokio_util::sync::CancellationToken;

    struct NoopHost;
    #[async_trait]
    impl HostRequestHandler for NoopHost {
        async fn request(&self, req: HostRequest) -> HostResponse {
            HostResponse {
                request_id: req.request_id,
                body: HostResponseBody::Approved(true),
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

    async fn run(env: &LocalEnvironment, args: &str) -> ToolOutcome {
        let cancel = CancellationToken::new();
        let events = EventSink::discarding();
        let host = NoopHost;
        let cx = TurnCx {
            cancel,
            events: &events,
            host: &host,
            session_id: SessionId::new("t"),
            budget: Budget::unlimited(),
            exec: env,
            tool_result_budget: 0,
        };
        let call = ToolCall {
            call_id: "c1".into(),
            name: "fs".into(),
            args: args.into(),
        };
        FsTool::new().run(&call, &cx).await
    }

    #[tokio::test]
    async fn write_then_read_roundtrips_with_detail() {
        let root = temp_root("rw");
        let env = LocalEnvironment::new(&root);
        let w = run(&env, r#"{"op":"write","path":"a.txt","content":"hello"}"#).await;
        assert!(w.result.ok);
        assert!(w.detail.is_some());

        let r = run(&env, r#"{"op":"read","path":"a.txt"}"#).await;
        assert!(r.result.ok);
        assert_eq!(r.result.content, "hello");
        let detail = r.detail.expect("read detail");
        assert_eq!(detail.kind, "fs");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn edit_replaces_first_occurrence() {
        let root = temp_root("edit");
        let env = LocalEnvironment::new(&root);
        run(&env, r#"{"op":"write","path":"a.txt","content":"foo foo"}"#).await;
        let e = run(
            &env,
            r#"{"op":"edit","path":"a.txt","find":"foo","replace":"bar"}"#,
        )
        .await;
        assert!(e.result.ok);
        let r = run(&env, r#"{"op":"read","path":"a.txt"}"#).await;
        assert_eq!(r.result.content, "bar foo");
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
}
