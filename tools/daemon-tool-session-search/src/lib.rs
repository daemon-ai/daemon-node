// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-tool-session-search` — the `session_search` chat tool (`daemon_core::Tool`): long-term
//! conversation recall over the node's durable session store (hermes `session_search_tool.py`
//! parity). **No LLM calls** — every shape returns actual conversation turns.
//!
//! One tool, four calling shapes inferred from the arguments (no explicit mode parameter, hermes'
//! exact precedence):
//!
//! 1. **SCROLL** — `session_id` + `around_message_id`: a ±`window` slice centered on an anchor.
//! 2. **READ** — `session_id` only: the whole session (head 20 + tail 10 when large).
//! 3. **BROWSE** — no args: recent sessions with titles/previews.
//! 4. **DISCOVERY** — `query`: full-text search, deduped by session lineage, each hit carrying a
//!    snippet, opening/closing bookends, and a best-effort context window around the match.
//!
//! The store side is injected as a [`SessionArchive`] (the `MnemosyneBanks` handle pattern): the
//! assembling binary implements it over the durable `SessionStore` (FTS + decoded snapshots + the
//! live actor's conversation view), keeping this crate free of store/protocol dependencies and the
//! node authoritative over how conversations are read.
//!
//! Divergences from hermes (documented in the schema so the model is never surprised):
//! - Anchors are **0-based conversation turn indexes** (`messages[].id`), not per-message DB row
//!   ids — the daemon store has no per-message identity.
//! - The FTS index is session-granular (one coalesced body per session), so discovery locates the
//!   match window by re-finding the snippet text in the transcript (best-effort) and there is no
//!   `role_filter` (the indexed body already carries user + assistant text only).
//! - Cross-profile reads (`profile=`) are out of scope for the daemon.

#![forbid(unsafe_code)]

use std::sync::Arc;

use async_trait::async_trait;
use daemon_core::{Tool, ToolCall, ToolConcurrency, ToolOutcome, TurnCx};
use serde::Deserialize;
use serde_json::{json, Value};

/// One full-text search hit (session-granular), as the archive returns it.
#[derive(Clone, Debug)]
pub struct ArchiveHit {
    /// The matching session id.
    pub session_id: String,
    /// The session's roster title, when one is set.
    pub title: Option<String>,
    /// The FTS-highlighted excerpt (matched terms wrapped in `[`…`]`, `…` elision).
    pub snippet: String,
}

/// One recent-session roster line for the browse shape.
#[derive(Clone, Debug)]
pub struct ArchiveBrief {
    /// The session id.
    pub session_id: String,
    /// The session's roster title, when one is set.
    pub title: Option<String>,
    /// Unix-millis of the last activity (the browse sort key), when stamped.
    pub last_activity_ms: Option<u64>,
}

/// One normalized conversation turn. `role` is `"user"` or `"assistant"` (a tool-calling turn is
/// assistant-authored with its tool names in `tools`).
#[derive(Clone, Debug)]
pub struct ArchiveTurn {
    /// Who spoke.
    pub role: &'static str,
    /// The turn's text (may be empty for a tool-call-only assistant turn).
    pub text: String,
    /// The names of tools invoked in this turn (empty for non-tool turns).
    pub tools: Vec<String>,
}

/// A session's readable conversation: meta + the full ordered turn list.
#[derive(Clone, Debug)]
pub struct ArchiveSession {
    /// The session id.
    pub session_id: String,
    /// The session's roster title, when one is set.
    pub title: Option<String>,
    /// Unix-millis of the last activity, when stamped.
    pub last_activity_ms: Option<u64>,
    /// The ordered turns (index = the anchor id the scroll shape addresses).
    pub turns: Vec<ArchiveTurn>,
}

/// The injected read surface over the node's session store — implemented by the assembling binary
/// (durable FTS + decoded snapshots + the live actor's conversation view). All methods are
/// read-only and best-effort.
#[async_trait]
pub trait SessionArchive: Send + Sync {
    /// Full-text search over the indexed session text, most-relevant first, capped at `limit`.
    async fn search(&self, query: &str, limit: u32) -> Vec<ArchiveHit>;

    /// Recent candidate sessions for the browse shape, newest-activity first, capped at `limit`.
    /// The implementation excludes subagent/child sessions (non-`Primary` roles) and archived ones.
    async fn recent(&self, limit: u32) -> Vec<ArchiveBrief>;

    /// A session's readable conversation, or `None` when the session is unknown or its transcript
    /// is not recoverable (e.g. a live-only session from before a daemon restart).
    async fn turns(&self, session: &str) -> Option<ArchiveSession>;

    /// The lineage root of a session (walk parent links to the top; self when parentless/unknown).
    async fn lineage_root(&self, session: &str) -> String;
}

/// The JSON-Schema advertised for the `session_search` tool.
const SESSION_SEARCH_SCHEMA: &str = r#"{
  "type": "object",
  "description": "Search past sessions in the local session store, or read/scroll inside one. FTS-backed retrieval over indexed conversations (user + assistant text + tool names). No LLM calls - every shape returns actual conversation turns. FOUR SHAPES inferred from args: 1) DISCOVERY - pass query: top sessions each with a snippet, bookend_start (first 3 turns: the goal), messages (a window around the match when locatable), bookend_end (last 3 turns: the resolution). 2) SCROLL - pass session_id + around_message_id (+ window): a slice centered on that turn; to scroll forward pass the last message id back, backward the first; ids are 0-based conversation turn indexes (NOT hermes-style message row ids). 3) READ - pass session_id only: the whole session, first 20 + last 10 turns when large. 4) BROWSE - no args: recent sessions with titles and previews. FTS5 syntax in query: AND is implicit, OR for broader recall, quoted phrases for exact match, prefix* wildcards. Reach for this on any 'what did we do about X' / 'where did we leave Y' question before external tools.",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query (discovery shape). Keywords, phrases, or FTS5 boolean expressions. Omit to browse recent sessions. Ignored when session_id is set."
    },
    "limit": {
      "type": "integer",
      "description": "Discovery/browse: max sessions to return (default 3, clamped to [1,10]). Bump to 5-10 when the topic likely spans several sessions.",
      "default": 3
    },
    "sort": {
      "type": "string",
      "enum": ["newest", "oldest"],
      "description": "Discovery only: temporal bias over the hits. Omit for relevance order ('what do we know about X'); 'newest' for 'where did we leave X'; 'oldest' for 'how did X start'."
    },
    "session_id": {
      "type": "string",
      "description": "Read/scroll target: a session_id from a prior discovery or browse result. Alone = read the whole session; with around_message_id = scroll."
    },
    "around_message_id": {
      "type": "integer",
      "description": "Scroll shape: the 0-based turn index to center the window on. From a discovery result use match_message_id or any messages[].id; to scroll forward pass the last window id, backward the first."
    },
    "window": {
      "type": "integer",
      "description": "Scroll shape: turns on each side of the anchor (anchor always included). Clamped to [1,20]. Default 5.",
      "default": 5
    }
  },
  "required": []
}"#;

/// How many turns open/close a discovery bookend.
const BOOKEND_TURNS: usize = 3;
/// The context radius around a located discovery match.
const DISCOVERY_WINDOW: usize = 5;
/// Read shape: how many opening turns a large session returns.
const READ_HEAD: usize = 20;
/// Read shape: how many closing turns a large session returns.
const READ_TAIL: usize = 10;
/// How wide discovery searches before lineage-deduping down to `limit`.
const DISCOVERY_FETCH: u32 = 50;
/// Browse preview length (chars, whitespace-collapsed).
const PREVIEW_CHARS: usize = 100;

/// The `session_search` tool: hermes' four-shape session recall over an injected [`SessionArchive`].
pub struct SessionSearchTool {
    archive: Arc<dyn SessionArchive>,
}

impl SessionSearchTool {
    /// A session-search tool over `archive`.
    pub fn new(archive: Arc<dyn SessionArchive>) -> Self {
        Self { archive }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Args {
    query: Option<String>,
    limit: Option<i64>,
    sort: Option<String>,
    session_id: Option<String>,
    around_message_id: Option<i64>,
    window: Option<i64>,
}

#[async_trait]
impl Tool for SessionSearchTool {
    fn name(&self) -> &str {
        "session_search"
    }

    fn schema(&self) -> &str {
        SESSION_SEARCH_SCHEMA
    }

    /// Read-only over the archive: safe alongside other parallel calls in a batch.
    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let args: Args = match serde_json::from_str(&call.args) {
            Ok(a) => a,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("session_search: invalid arguments: {e}"),
                )
            }
        };
        let body = self.dispatch(args, cx.session_id.as_str()).await;
        ToolOutcome::text(call.call_id.clone(), true, body)
    }
}

impl SessionSearchTool {
    /// Infer the calling shape from the args (hermes' exact precedence: scroll > read > browse >
    /// discovery) and serve it. `current_session` is excluded from every cross-session shape (its
    /// content is already in the model's context).
    async fn dispatch(&self, args: Args, current_session: &str) -> String {
        let session_id = args
            .session_id
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());

        // Scroll wins over read/discovery when an anchor is set.
        if let (Some(session), Some(anchor)) = (session_id, args.around_message_id) {
            let window = clamp(args.window.unwrap_or(5), 1, 20) as usize;
            return self.scroll(session, anchor, window, current_session).await;
        }
        if let Some(session) = session_id {
            return self.read(session).await;
        }

        let limit = clamp(args.limit.unwrap_or(3), 1, 10) as usize;
        let query = args
            .query
            .as_deref()
            .map(str::trim)
            .filter(|q| !q.is_empty());
        match query {
            None => self.browse(limit, current_session).await,
            Some(query) => {
                let sort = args
                    .sort
                    .as_deref()
                    .map(str::trim)
                    .map(str::to_lowercase)
                    .filter(|s| s == "newest" || s == "oldest");
                self.discover(query, limit, sort.as_deref(), current_session)
                    .await
            }
        }
    }

    /// DISCOVERY: FTS hits, deduped by lineage (one result per conversation tree), each with
    /// bookends + a best-effort window around the match.
    async fn discover(
        &self,
        query: &str,
        limit: usize,
        sort: Option<&str>,
        current_session: &str,
    ) -> String {
        let hits = self.archive.search(query, DISCOVERY_FETCH).await;
        if hits.is_empty() {
            return json!({
                "success": true,
                "mode": "discover",
                "query": query,
                "results": [],
                "count": 0,
                "message": "No matching sessions found.",
            })
            .to_string();
        }

        // Dedupe by lineage root; skip the current conversation tree (already in context).
        let current_root = self.archive.lineage_root(current_session).await;
        let mut kept: Vec<(String, ArchiveHit)> = Vec::new();
        for hit in hits {
            if hit.session_id == current_session {
                continue;
            }
            let root = self.archive.lineage_root(&hit.session_id).await;
            if root == current_root {
                continue;
            }
            if kept.iter().any(|(r, _)| *r == root) {
                continue;
            }
            kept.push((root, hit));
            if kept.len() >= limit {
                break;
            }
        }

        let mut results = Vec::new();
        for (_root, hit) in kept {
            let Some(session) = self.archive.turns(&hit.session_id).await else {
                // The FTS row survives a session whose transcript is no longer recoverable (e.g. a
                // pre-restart live-only session): still surface the hit, minus the turn content.
                results.push(json!({
                    "session_id": hit.session_id,
                    "title": hit.title,
                    "snippet": hit.snippet,
                    "message": "transcript not recoverable (no stored conversation for this session)",
                }));
                continue;
            };
            let total = session.turns.len();
            let anchor = locate_snippet(&hit.snippet, &session.turns);
            let (messages, before, after) = match anchor {
                Some(idx) => {
                    let lo = idx.saturating_sub(DISCOVERY_WINDOW);
                    let hi = (idx + DISCOVERY_WINDOW).min(total.saturating_sub(1));
                    (
                        shape_turns(&session.turns, lo, hi, Some(idx)),
                        lo,
                        total - 1 - hi,
                    )
                }
                None => (Vec::new(), 0, 0),
            };
            let mut entry = json!({
                "session_id": session.session_id,
                "title": session.title,
                "last_active_ms": session.last_activity_ms,
                "turn_count": total,
                "snippet": hit.snippet,
                "match_message_id": anchor,
                "bookend_start": shape_turns(&session.turns, 0, BOOKEND_TURNS.saturating_sub(1).min(total.saturating_sub(1)), None),
                "messages": messages,
                "bookend_end": shape_turns(&session.turns, total.saturating_sub(BOOKEND_TURNS), total.saturating_sub(1), None),
                "messages_before": before,
                "messages_after": after,
            });
            if total == 0 {
                entry["bookend_start"] = json!([]);
                entry["bookend_end"] = json!([]);
            }
            results.push(entry);
        }

        // Optional temporal bias over the deduped hits (relevance order otherwise).
        if let Some(sort) = sort {
            let key = |v: &Value| v.get("last_active_ms").and_then(Value::as_u64).unwrap_or(0);
            match sort {
                "newest" => results.sort_by_key(|v| std::cmp::Reverse(key(v))),
                _ => results.sort_by_key(key),
            }
        }

        json!({
            "success": true,
            "mode": "discover",
            "query": query,
            "count": results.len(),
            "results": results,
            "message": "Pass session_id + around_message_id (a messages[].id) to scroll for more context, or session_id alone to read the whole session.",
        })
        .to_string()
    }

    /// SCROLL: a ±window slice centered on a turn-index anchor.
    async fn scroll(
        &self,
        session_id: &str,
        anchor: i64,
        window: usize,
        current_session: &str,
    ) -> String {
        if anchor < 0 {
            return err_json("scroll requires a non-negative around_message_id (a turn index)");
        }
        // Reject scrolling inside the active conversation tree — already in the model's context.
        if session_id == current_session
            || self.archive.lineage_root(session_id).await
                == self.archive.lineage_root(current_session).await
        {
            return err_json(
                "scroll rejected: the anchor lives in the current session lineage (already in your active context)",
            );
        }
        let Some(session) = self.archive.turns(session_id).await else {
            return err_json(&format!("session_id not found: {session_id}"));
        };
        let total = session.turns.len();
        let anchor = anchor as usize;
        if anchor >= total {
            return err_json(&format!(
                "around_message_id {anchor} not in session_id {session_id} (turns: 0..{})",
                total.saturating_sub(1)
            ));
        }
        let lo = anchor.saturating_sub(window);
        let hi = (anchor + window).min(total - 1);
        json!({
            "success": true,
            "mode": "scroll",
            "session_id": session.session_id,
            "session_meta": { "title": session.title, "last_active_ms": session.last_activity_ms },
            "around_message_id": anchor,
            "window": window,
            "messages": shape_turns(&session.turns, lo, hi, Some(anchor)),
            "messages_before": lo,
            "messages_after": total - 1 - hi,
        })
        .to_string()
    }

    /// READ: the whole session, head+tail bounded when large.
    async fn read(&self, session_id: &str) -> String {
        let Some(session) = self.archive.turns(session_id).await else {
            return err_json(&format!("session_id not found: {session_id}"));
        };
        let total = session.turns.len();
        let truncated = total > READ_HEAD + READ_TAIL;
        let messages = if truncated {
            let mut m = shape_turns(&session.turns, 0, READ_HEAD - 1, None);
            m.extend(shape_turns(
                &session.turns,
                total - READ_TAIL,
                total - 1,
                None,
            ));
            m
        } else if total > 0 {
            shape_turns(&session.turns, 0, total - 1, None)
        } else {
            Vec::new()
        };
        let mut out = json!({
            "success": true,
            "mode": "read",
            "session_id": session.session_id,
            "session_meta": { "title": session.title, "last_active_ms": session.last_activity_ms },
            "message_count": total,
            "truncated": truncated,
            "messages": messages,
        });
        if truncated {
            out["message"] = json!(format!(
                "Session has {total} turns; showing first {READ_HEAD} + last {READ_TAIL}. \
                 Pass around_message_id (any id above) to scroll the middle."
            ));
        }
        out.to_string()
    }

    /// BROWSE: recent sessions with titles + previews (no FTS).
    async fn browse(&self, limit: usize, current_session: &str) -> String {
        // Fetch extra so skipping the current lineage still fills the page (hermes `limit + 5`).
        let briefs = self.archive.recent((limit + 5) as u32).await;
        let current_root = self.archive.lineage_root(current_session).await;
        let mut results = Vec::new();
        for brief in briefs {
            if brief.session_id == current_session
                || self.archive.lineage_root(&brief.session_id).await == current_root
            {
                continue;
            }
            let (preview, count) = match self.archive.turns(&brief.session_id).await {
                Some(session) => {
                    let preview = session
                        .turns
                        .iter()
                        .find(|t| t.role == "user" && !t.text.trim().is_empty())
                        .map(|t| collapse(&t.text, PREVIEW_CHARS));
                    (preview, Some(session.turns.len()))
                }
                None => (None, None),
            };
            results.push(json!({
                "session_id": brief.session_id,
                "title": brief.title,
                "last_active_ms": brief.last_activity_ms,
                "message_count": count,
                "preview": preview,
            }));
            if results.len() >= limit {
                break;
            }
        }
        json!({
            "success": true,
            "mode": "browse",
            "count": results.len(),
            "results": results,
            "message": format!(
                "Showing {} most recent sessions. Pass a query= to search, session_id= to read one, \
                 or session_id + around_message_id to scroll.",
                results.len()
            ),
        })
        .to_string()
    }
}

/// Shape `turns[lo..=hi]` for a response: `{id, role, content, tools?, anchor?}` per turn.
fn shape_turns(turns: &[ArchiveTurn], lo: usize, hi: usize, anchor: Option<usize>) -> Vec<Value> {
    if turns.is_empty() || lo > hi || lo >= turns.len() {
        return Vec::new();
    }
    let hi = hi.min(turns.len() - 1);
    turns[lo..=hi]
        .iter()
        .enumerate()
        .map(|(offset, turn)| {
            let id = lo + offset;
            let mut entry = json!({
                "id": id,
                "role": turn.role,
                "content": turn.text,
            });
            if !turn.tools.is_empty() {
                entry["tools"] = json!(turn.tools);
            }
            if anchor == Some(id) {
                entry["anchor"] = json!(true);
            }
            entry
        })
        .collect()
}

/// Best-effort: re-find an FTS snippet's text in the transcript to anchor the discovery window.
/// The snippet wraps matches in `[`…`]` with `…` elision; strip the markers, take the longest
/// elision-separated fragment, and find the LAST turn containing it (most recent occurrence).
/// `None` when the fragment is too short to be trustworthy or not found (session-granular FTS has
/// no per-message match position — this is the daemon-side approximation).
fn locate_snippet(snippet: &str, turns: &[ArchiveTurn]) -> Option<usize> {
    let cleaned = snippet.replace(['[', ']'], "");
    let fragment = cleaned
        .split('…')
        .map(str::trim)
        .max_by_key(|piece| piece.len())?;
    if fragment.len() < 8 {
        return None;
    }
    turns.iter().rposition(|turn| turn.text.contains(fragment))
}

/// Collapse whitespace runs and truncate to `limit` chars with an ellipsis.
fn collapse(text: &str, limit: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= limit {
        return collapsed;
    }
    let head: String = collapsed.chars().take(limit.saturating_sub(1)).collect();
    format!("{}…", head.trim_end())
}

/// A hermes-style soft error: transport-ok, `success: false` JSON body.
fn err_json(message: &str) -> String {
    json!({ "success": false, "error": message }).to_string()
}

/// Clamp an i64 arg into `[lo, hi]`.
fn clamp(value: i64, lo: i64, hi: i64) -> i64 {
    value.max(lo).min(hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_locator_finds_the_source_turn() {
        let turns = vec![
            ArchiveTurn {
                role: "user",
                text: "let's talk about docker networking today".into(),
                tools: vec![],
            },
            ArchiveTurn {
                role: "assistant",
                text: "sure — docker networking has bridge and overlay modes".into(),
                tools: vec![],
            },
        ];
        let idx = locate_snippet("…about [docker] [networking] today…", &turns);
        assert_eq!(idx, Some(0));
        assert_eq!(locate_snippet("…[db]…", &turns), None, "too short");
        assert_eq!(locate_snippet("…[completely absent text]…", &turns), None);
    }

    #[test]
    fn shaping_marks_the_anchor_and_bounds_the_slice() {
        let turns: Vec<ArchiveTurn> = (0..10)
            .map(|i| ArchiveTurn {
                role: if i % 2 == 0 { "user" } else { "assistant" },
                text: format!("turn {i}"),
                tools: vec![],
            })
            .collect();
        let shaped = shape_turns(&turns, 3, 5, Some(4));
        assert_eq!(shaped.len(), 3);
        assert_eq!(shaped[0]["id"], 3);
        assert!(shaped[1]["anchor"].as_bool().unwrap_or(false));
        assert!(shaped[0].get("anchor").is_none());
        assert!(shape_turns(&turns, 12, 15, None).is_empty(), "out of range");
    }

    #[test]
    fn collapse_truncates_on_char_boundaries() {
        assert_eq!(collapse("a  b\nc", 10), "a b c");
        let long = collapse(&"word ".repeat(50), 20);
        assert!(long.chars().count() <= 20);
        assert!(long.ends_with('…'));
    }
}
