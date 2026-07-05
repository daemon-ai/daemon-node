// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `session_search` tool coverage over a fake [`SessionArchive`]: the four inferred calling shapes
//! and their precedence, the hermes clamps, lineage dedupe + current-lineage exclusion, and the
//! soft-error paths.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use daemon_common::{Budget, SessionId};
use daemon_core::events::EventSink;
use daemon_core::exec::LocalEnvironment;
use daemon_core::{Tool, ToolCall, ToolConcurrency, TurnCx};
use daemon_tool_session_search::{
    ArchiveBrief, ArchiveHit, ArchiveSession, ArchiveTurn, SessionArchive, SessionSearchTool,
};
use serde_json::Value;

/// A canned archive: fixture sessions with parent links, briefs, and substring "FTS".
#[derive(Default)]
struct FakeArchive {
    sessions: HashMap<String, ArchiveSession>,
    parents: HashMap<String, String>,
}

impl FakeArchive {
    fn with_session(mut self, id: &str, turns: Vec<(&'static str, &str)>) -> Self {
        self.sessions.insert(
            id.to_string(),
            ArchiveSession {
                session_id: id.to_string(),
                title: Some(format!("title-{id}")),
                last_activity_ms: Some(1_000 + self.sessions.len() as u64),
                turns: turns
                    .into_iter()
                    .map(|(role, text)| ArchiveTurn {
                        role,
                        text: text.to_string(),
                        tools: vec![],
                    })
                    .collect(),
            },
        );
        self
    }

    fn with_parent(mut self, child: &str, parent: &str) -> Self {
        self.parents.insert(child.to_string(), parent.to_string());
        self
    }
}

#[async_trait]
impl SessionArchive for FakeArchive {
    async fn search(&self, query: &str, limit: u32) -> Vec<ArchiveHit> {
        let mut ids: Vec<&String> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.turns.iter().any(|t| t.text.contains(query)))
            .map(|(id, _)| id)
            .collect();
        ids.sort();
        ids.truncate(limit as usize);
        ids.into_iter()
            .map(|id| ArchiveHit {
                session_id: id.clone(),
                title: self.sessions[id].title.clone(),
                snippet: format!("…[{query}] fragment for anchor…"),
            })
            .collect()
    }

    async fn recent(&self, limit: u32) -> Vec<ArchiveBrief> {
        let mut briefs: Vec<ArchiveBrief> = self
            .sessions
            .values()
            .map(|s| ArchiveBrief {
                session_id: s.session_id.clone(),
                title: s.title.clone(),
                last_activity_ms: s.last_activity_ms,
            })
            .collect();
        briefs.sort_by_key(|b| std::cmp::Reverse(b.last_activity_ms));
        briefs.truncate(limit as usize);
        briefs
    }

    async fn turns(&self, session: &str) -> Option<ArchiveSession> {
        self.sessions.get(session).cloned()
    }

    async fn lineage_root(&self, session: &str) -> String {
        let mut cur = session.to_string();
        while let Some(parent) = self.parents.get(&cur) {
            cur = parent.clone();
        }
        cur
    }
}

/// Run the tool with `args` under `current` as the calling session; return the parsed JSON body.
async fn run(archive: FakeArchive, current: &str, args: &str) -> Value {
    let tool = SessionSearchTool::new(Arc::new(archive));
    assert_eq!(tool.concurrency(), ToolConcurrency::Parallel);
    assert!(!tool.mutates());
    let events = EventSink::discarding();
    let exec = LocalEnvironment::sandbox("session-search-test");
    let host = NoHost;
    let cx = TurnCx {
        cancel: tokio_util::sync::CancellationToken::new(),
        events: &events,
        host: &host,
        session_id: SessionId::new(current),
        profile: None,
        budget: Budget::unlimited(),
        exec: &exec,
        tool_result_budget: 0,
        approval_policy: daemon_core::ApprovalPolicy::AutoAllow,
        pre_approved: false,
        checkpoints: None,
        tool_timeout: None,
    };
    let call = ToolCall {
        call_id: "c1".into(),
        name: "session_search".into(),
        args: args.into(),
    };
    let out = tool.run(&call, &cx).await;
    assert!(out.result.ok, "transport-level ok: {}", out.result.content);
    serde_json::from_str(&out.result.content).expect("tool returns JSON")
}

fn corpus() -> FakeArchive {
    FakeArchive::default()
        .with_session(
            "s-docker",
            vec![
                ("user", "how do I fix docker networking on this host"),
                ("assistant", "docker networking fragment for anchor"),
                ("user", "thanks"),
                ("assistant", "anytime"),
            ],
        )
        .with_session(
            "s-docker-child",
            vec![("user", "child of the docker session: docker networking too")],
        )
        .with_parent("s-docker-child", "s-docker")
        .with_session(
            "s-rust",
            vec![
                ("user", "rust borrow checker question"),
                ("assistant", "borrows explained"),
            ],
        )
        .with_session(
            "s-current",
            vec![("user", "docker networking in my own chat")],
        )
}

#[tokio::test]
async fn browse_lists_recent_and_skips_the_current_session() {
    let body = run(corpus(), "s-current", "{}").await;
    assert_eq!(body["success"], true);
    assert_eq!(body["mode"], "browse");
    let ids: Vec<&str> = body["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["session_id"].as_str().unwrap())
        .collect();
    assert!(!ids.contains(&"s-current"), "current session excluded");
    assert!(ids.contains(&"s-rust"));
    let rust = body["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["session_id"] == "s-rust")
        .unwrap();
    assert_eq!(rust["message_count"], 2);
    assert_eq!(rust["preview"], "rust borrow checker question");
}

#[tokio::test]
async fn discovery_dedupes_lineage_and_anchors_the_match() {
    let body = run(corpus(), "s-current", r#"{"query": "docker networking"}"#).await;
    assert_eq!(body["mode"], "discover");
    let results = body["results"].as_array().unwrap();
    // s-docker and s-docker-child share a lineage -> one result; the current session's own
    // "docker networking" turn is excluded.
    assert_eq!(results.len(), 1, "lineage-deduped: {results:?}");
    let hit = &results[0];
    assert_eq!(hit["session_id"], "s-docker");
    assert_eq!(hit["turn_count"], 4);
    // The snippet fragment re-locates in turn 1 -> anchored window + bookends.
    assert_eq!(hit["match_message_id"], 1);
    assert!(hit["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m["anchor"] == true));
    assert_eq!(hit["bookend_start"].as_array().unwrap().len(), 3);
    assert_eq!(hit["bookend_end"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn read_returns_whole_small_sessions_and_truncates_large_ones() {
    let body = run(corpus(), "s-current", r#"{"session_id": "s-rust"}"#).await;
    assert_eq!(body["mode"], "read");
    assert_eq!(body["truncated"], false);
    assert_eq!(body["messages"].as_array().unwrap().len(), 2);

    // A 40-turn session truncates to head 20 + tail 10.
    let turns: Vec<(&'static str, String)> = (0..40)
        .map(|i| {
            (
                if i % 2 == 0 { "user" } else { "assistant" },
                format!("turn {i}"),
            )
        })
        .collect();
    let archive = FakeArchive::default().with_session(
        "s-long",
        turns.iter().map(|(r, t)| (*r, t.as_str())).collect(),
    );
    let body = run(archive, "s-current", r#"{"session_id": "s-long"}"#).await;
    assert_eq!(body["truncated"], true);
    assert_eq!(body["message_count"], 40);
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 30);
    assert_eq!(messages[0]["id"], 0);
    assert_eq!(messages[29]["id"], 39);
}

#[tokio::test]
async fn scroll_windows_around_the_anchor_and_wins_over_read() {
    let archive = corpus();
    let body = run(
        archive,
        "s-current",
        r#"{"session_id": "s-docker", "around_message_id": 2, "window": 1, "query": "ignored"}"#,
    )
    .await;
    assert_eq!(body["mode"], "scroll", "anchor beats query and read");
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3, "±1 around the anchor");
    assert_eq!(messages[0]["id"], 1);
    assert!(messages[1]["anchor"] == true);
    assert_eq!(body["messages_before"], 1);
    assert_eq!(body["messages_after"], 0);
}

#[tokio::test]
async fn scroll_clamps_window_and_rejects_bad_anchors() {
    // Window clamps to [1, 20]; an out-of-range anchor is a soft error.
    let body = run(
        corpus(),
        "s-current",
        r#"{"session_id": "s-docker", "around_message_id": 99, "window": 500}"#,
    )
    .await;
    assert_eq!(body["success"], false);
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("around_message_id 99 not in session_id"));
}

#[tokio::test]
async fn scroll_rejects_the_current_lineage() {
    // s-docker-child is in the CURRENT session's lineage when we run as s-docker.
    let body = run(
        corpus(),
        "s-docker",
        r#"{"session_id": "s-docker-child", "around_message_id": 0}"#,
    )
    .await;
    assert_eq!(body["success"], false);
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("current session lineage"));
}

#[tokio::test]
async fn discovery_clamps_limit_and_unknown_read_is_a_soft_error() {
    // limit clamps to [1, 10] (a 0 limit still returns one result).
    let body = run(
        corpus(),
        "s-current",
        r#"{"query": "docker networking", "limit": 0}"#,
    )
    .await;
    assert_eq!(body["count"], 1);

    let body = run(corpus(), "s-current", r#"{"session_id": "nope"}"#).await;
    assert_eq!(body["success"], false);
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("session_id not found"));
}

/// A host that must never be consulted (the tool is archive-only).
struct NoHost;

#[async_trait]
impl daemon_protocol::HostRequestHandler for NoHost {
    async fn request(&self, req: daemon_protocol::HostRequest) -> daemon_protocol::HostResponse {
        panic!("session_search must not raise host requests: {req:?}");
    }
}
