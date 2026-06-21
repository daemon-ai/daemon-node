//! The `MemoryProvider` implementation — port of `hermes_memory_provider/__init__.py`.
//!
//! Maps Mnemosyne onto the daemon-core seam: `prompt_block` = memory-override instructions
//! (`system_prompt_block` L1437), `recall` = formatted BEAM recall block (`prefetch` L1474 / block
//! format L1645-L1659), `after_turn` = the `sync_turn` persist gates (L1668), and `tools`/`call_tool`
//! = the JSON tool dispatch (L1750). Scaffold: the core hooks are wired; the full 26-tool table and
//! identity-signal capture are TODO.

use crate::engine::{Engine, MemoryRow, RememberArgs};
use crate::MnemosyneConfig;
use daemon_core::conversation::{Conversation, Turn};
use daemon_core::memory::{MemoryProvider, PromptBlock, RecallQuery, RecalledBlock, SwitchReason};
use daemon_core::tools::ToolDef;
use serde_json::{json, Value};
use std::sync::Arc;

/// The Mnemosyne memory provider over a single bank engine.
pub struct MnemosyneProvider {
    engine: Arc<Engine>,
}

impl MnemosyneProvider {
    /// Wrap an existing engine.
    pub fn new(engine: Arc<Engine>) -> Self {
        Self { engine }
    }

    /// Open a provider for the configured bank.
    pub fn open(config: MnemosyneConfig) -> crate::Result<Self> {
        Ok(Self {
            engine: Arc::new(Engine::open(config)?),
        })
    }

    /// Format recall rows into a prompt block (`__init__.py` L1645-L1659):
    /// `  [ts] (importance X.XX[, source S])[ TRUST] content`.
    fn format_block(rows: &[MemoryRow]) -> String {
        let mut out = String::from("## Mnemosyne Context\n");
        for row in rows {
            let ts = row.timestamp.chars().take(16).collect::<String>();
            let mut meta = format!("importance {:.2}", row.importance);
            if row.source != "conversation" && !row.source.is_empty() {
                meta.push_str(&format!(", source {}", row.source));
            }
            let trust = if row.trust_tier != "STATED" && !row.trust_tier.is_empty() {
                format!(" [{}]", row.trust_tier)
            } else {
                String::new()
            };
            out.push_str(&format!("  [{}] ({}){} {}\n", ts, meta, trust, row.content));
        }
        out
    }
}

#[async_trait::async_trait]
impl MemoryProvider for MnemosyneProvider {
    fn name(&self) -> &str {
        "mnemosyne"
    }

    fn prompt_block(&self) -> Option<PromptBlock> {
        Some(PromptBlock {
            text: "You have a persistent memory (Mnemosyne). Recalled context is injected below; \
                   use the mnemosyne_* tools to remember, recall, and manage long-term memory."
                .to_string(),
        })
    }

    async fn recall(&self, q: &RecallQuery) -> Option<RecalledBlock> {
        let rows = self.engine.recall(&q.text, q.top_k).ok()?;
        if rows.is_empty() {
            return None;
        }
        Some(RecalledBlock {
            text: Self::format_block(&rows),
        })
    }

    async fn after_turn(&self, turn: &Turn, _conv: &Conversation) {
        // sync_turn gates (`__init__.py` L1668-L1692): persist the user text (>5 chars) and the
        // assistant text (>10 chars) with their respective importances.
        match turn {
            Turn::User(u) if u.text.len() > 5 => {
                let _ = self.engine.remember(
                    &format!("[USER] {}", u.text),
                    &RememberArgs { importance: 0.5, ..Default::default() },
                );
            }
            Turn::Assistant(a) if a.text.len() > 10 => {
                let _ = self.engine.remember(
                    &format!("[ASSISTANT] {}", a.text),
                    &RememberArgs { importance: 0.15, ..Default::default() },
                );
            }
            _ => {}
        }
    }

    async fn before_compact(&self, _conv: &Conversation) {
        // TODO: persist salient facts before the body is compacted (`on_pre_compress`).
    }

    async fn on_session_switch(&self, _reason: SwitchReason) {
        // TODO: consolidate via engine.sleep() on session end.
    }
}

impl MnemosyneProvider {
    /// The memory-management tools this backend exposes (`mnemosyne_remember`/`mnemosyne_recall`).
    ///
    /// These are *not* part of the §11 [`MemoryProvider`] seam — that seam is about context, not
    /// dispatch. A host that wants to expose them to the model registers them through the §12
    /// [`ToolRegistry`](daemon_core::tools) like any other tool, calling [`Self::call_tool`].
    pub fn tools(&self) -> Vec<ToolDef> {
        vec![
            ToolDef {
                name: "mnemosyne_remember".to_string(),
                schema: r#"{"type":"object","properties":{"content":{"type":"string"},"importance":{"type":"number"}},"required":["content"]}"#.to_string(),
            },
            ToolDef {
                name: "mnemosyne_recall".to_string(),
                schema: r#"{"type":"object","properties":{"query":{"type":"string"},"top_k":{"type":"integer"}},"required":["query"]}"#.to_string(),
            },
        ]
    }

    /// Dispatch one of [`Self::tools`] by name, returning a JSON string result.
    pub async fn call_tool(&self, name: &str, args: Value) -> String {
        match name {
            "mnemosyne_remember" => {
                let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let importance = args.get("importance").and_then(|v| v.as_f64()).unwrap_or(0.5);
                match self.engine.remember(content, &RememberArgs { importance, ..Default::default() }) {
                    Ok(id) => json!({"status": "ok", "memory_id": id}).to_string(),
                    Err(e) => json!({"status": "error", "error": e.to_string()}).to_string(),
                }
            }
            "mnemosyne_recall" => {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
                match self.engine.recall(query, top_k) {
                    Ok(rows) => {
                        let results: Vec<Value> = rows
                            .iter()
                            .map(|r| json!({"id": r.id, "content": r.content, "score": r.score}))
                            .collect();
                        json!({"query": query, "count": results.len(), "results": results}).to_string()
                    }
                    Err(e) => json!({"status": "error", "error": e.to_string()}).to_string(),
                }
            }
            _ => json!({"status": "unknown_tool", "tool": name}).to_string(),
        }
    }
}
