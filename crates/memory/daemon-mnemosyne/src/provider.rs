//! The `MemoryProvider` implementation — port of `hermes_memory_provider/__init__.py`.
//!
//! Maps Mnemosyne onto the daemon-core seam: `prompt_block` = memory-override instructions
//! (`system_prompt_block` L1437), `recall` = formatted BEAM recall block (`prefetch` L1474 / block
//! format L1645-L1659), `after_turn` = the `sync_turn` persist gates (L1668), and `tools`/`call_tool`
//! = the JSON tool dispatch (L1750). Scaffold: the core hooks are wired; the full 26-tool table and
//! identity-signal capture are TODO.

use crate::embeddings::Embedder;
use crate::engine::{Engine, MemoryRow, RememberArgs};
use crate::MnemosyneConfig;
use daemon_core::conversation::{Conversation, Turn};
use daemon_core::memory::{MemoryProvider, PromptBlock, RecallQuery, RecalledBlock, SwitchReason};
use daemon_core::tools::ToolDef;
use daemon_core::EmbeddingProvider;
use serde_json::{json, Value};
use std::sync::Arc;

/// The Mnemosyne memory provider over a single bank engine, with an optional embedding backend.
pub struct MnemosyneProvider {
    engine: Arc<Engine>,
    embedder: Embedder,
}

impl MnemosyneProvider {
    /// Wrap an existing engine in keyword-only mode (no embeddings).
    pub fn new(engine: Arc<Engine>) -> Self {
        Self {
            engine,
            embedder: Embedder::new(),
        }
    }

    /// Wrap an existing engine with an injected embedding provider (hybrid lexical + vector recall).
    pub fn with_embedder(engine: Arc<Engine>, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            engine,
            embedder: Embedder::with_provider(embedder),
        }
    }

    /// Open a provider for the configured bank in keyword-only mode.
    pub fn open(config: MnemosyneConfig) -> crate::Result<Self> {
        Ok(Self {
            engine: Arc::new(Engine::open(config)?),
            embedder: Embedder::new(),
        })
    }

    /// Open a provider for the configured bank with an injected embedding provider.
    pub fn open_with_embedder(
        config: MnemosyneConfig,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> crate::Result<Self> {
        Ok(Self {
            engine: Arc::new(Engine::open(config)?),
            embedder: Embedder::with_provider(embedder),
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
        // Embed the query at this async seam (keyword-only -> None) and pass the vector into the
        // synchronous hybrid recall so the engine never blocks on a model call.
        let query_vec = self.embedder.embed_query(&q.text).await;
        let rows = self
            .engine
            .recall_with_vector(&q.text, q.top_k, query_vec.as_deref())
            .ok()?;
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
        let (content, importance) = match turn {
            Turn::User(u) if u.text.len() > 5 => (format!("[USER] {}", u.text), 0.5),
            Turn::Assistant(a) if a.text.len() > 10 => (format!("[ASSISTANT] {}", a.text), 0.15),
            _ => return,
        };
        // Embed once at this async seam; the precomputed vector is persisted with the row.
        let vector = self.embedder.embed_query(&content).await;
        let model = self.embedder.model().unwrap_or("");
        let _ = self.engine.remember_with_vector(
            &content,
            &RememberArgs {
                importance,
                ..Default::default()
            },
            vector.as_deref(),
            model,
        );
    }

    async fn before_compact(&self, _conv: &Conversation) {
        // TODO: persist salient facts before the body is compacted (`on_pre_compress`).
    }

    async fn on_session_switch(&self, reason: SwitchReason) {
        // Promote unconsolidated working memory into the episodic tier at session boundaries (a
        // minimal slice of BEAM sleep/consolidation; full summarization/degradation is port-spec P1).
        if matches!(reason, SwitchReason::End | SwitchReason::Handoff) {
            let _ = self.engine.consolidate();
        }
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
                let importance = args
                    .get("importance")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5);
                let vector = self.embedder.embed_query(content).await;
                let model = self.embedder.model().unwrap_or("");
                match self.engine.remember_with_vector(
                    content,
                    &RememberArgs {
                        importance,
                        ..Default::default()
                    },
                    vector.as_deref(),
                    model,
                ) {
                    Ok(id) => json!({"status": "ok", "memory_id": id}).to_string(),
                    Err(e) => json!({"status": "error", "error": e.to_string()}).to_string(),
                }
            }
            "mnemosyne_recall" => {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
                let query_vec = self.embedder.embed_query(query).await;
                match self
                    .engine
                    .recall_with_vector(query, top_k, query_vec.as_deref())
                {
                    Ok(rows) => {
                        let results: Vec<Value> = rows
                            .iter()
                            .map(|r| json!({"id": r.id, "content": r.content, "score": r.score}))
                            .collect();
                        json!({"query": query, "count": results.len(), "results": results})
                            .to_string()
                    }
                    Err(e) => json!({"status": "error", "error": e.to_string()}).to_string(),
                }
            }
            _ => json!({"status": "unknown_tool", "tool": name}).to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use daemon_core::conversation::{Conversation, SystemPrompt, Turn, UserMsg};
    use daemon_core::memory::RecallQuery;
    use daemon_core::MockEmbedder;

    #[tokio::test]
    async fn vector_recall_surfaces_semantic_match_through_provider() {
        // Pin the stored memory and the query to the same vector, and a distractor orthogonal — all
        // with content that shares NO tokens with the query, so only the vector path can match.
        let stored = "[USER] the deployment uses a blue-green rollout";
        let distractor = "[USER] lunch yesterday was margherita pizza";
        let query = "shipping strategy";
        let embedder = Arc::new(MockEmbedder::scripted(
            3,
            [
                (stored.to_string(), vec![1.0, 0.0, 0.0]),
                (query.to_string(), vec![1.0, 0.0, 0.0]),
                (distractor.to_string(), vec![0.0, 1.0, 0.0]),
            ],
        ));
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::with_embedder(engine, embedder);
        let conv = Conversation::new(SystemPrompt::new(""));

        provider
            .after_turn(
                &Turn::User(UserMsg::new("the deployment uses a blue-green rollout")),
                &conv,
            )
            .await;
        provider
            .after_turn(
                &Turn::User(UserMsg::new("lunch yesterday was margherita pizza")),
                &conv,
            )
            .await;

        let block = provider
            .recall(&RecallQuery {
                text: query.to_string(),
                top_k: 5,
            })
            .await
            .expect("recall returns a block via the vector match");
        assert!(
            block.text.contains("blue-green"),
            "vector recall should surface the semantically-close memory; got: {}",
            block.text
        );
        assert!(
            !block.text.contains("pizza"),
            "the orthogonal distractor must not pass the vector gate; got: {}",
            block.text
        );
    }

    #[tokio::test]
    async fn session_end_promotes_working_memory_to_episodic() {
        let engine = Arc::new(Engine::open_in_memory(MnemosyneConfig::default()).unwrap());
        let provider = MnemosyneProvider::new(engine.clone());
        let conv = Conversation::new(SystemPrompt::new(""));

        provider
            .after_turn(
                &Turn::User(UserMsg::new("a memory worth keeping around")),
                &conv,
            )
            .await;
        assert!(
            !engine.recall("memory keeping", 5).unwrap().is_empty(),
            "the turn should have been stored"
        );

        // Ending the session should consolidate; a subsequent consolidate then finds nothing pending.
        provider.on_session_switch(SwitchReason::End).await;
        assert_eq!(
            engine.consolidate().unwrap(),
            0,
            "on_session_switch(End) should have already promoted the row"
        );
    }
}
