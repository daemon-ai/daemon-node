//! §4.3 attached, non-joining background spawn — host-side materialization of `Effect::Spawn`.
//!
//! The engine raises a fire-and-forget [`HostRequestKind::Spawn`](daemon_protocol::HostRequestKind)
//! when a post-turn review nudge fires; the host materializes an **attached, non-joining,
//! self-closing** child (a third delegation edge): recorded under the parent in the durable tree for
//! audit ([`SessionStore::record_child_edge`]), but bound to *no* parent job — so it runs to a
//! terminal state on its own and never wakes the parent. This is the general mechanism behind both
//! background skill review and background memory write (hermes `background_review.py`), differing
//! only in the per-`kind` constrained profile + review prompt registered here.

use daemon_common::{Epoch, PartitionId, SessionId};
use daemon_core::{Conversation, EngineProfile, Snapshot, SystemPrompt, UserMsg};
use daemon_protocol::SpawnSpec;
use daemon_store::SessionStore;
use std::collections::HashMap;
use std::sync::Arc;

/// One background-review profile: the constrained engine shape a spawned child runs under, plus the
/// review instruction seeded as the child's opening user message.
#[derive(Clone)]
pub struct BackgroundProfile {
    /// The constrained engine profile (skills-only / memory-only registry, bounded `max_iterations`,
    /// review nudges disabled to prevent recursion). It should inherit the parent's provider +
    /// credentials so the reviewer uses the same model (hermes forks the parent's provider/model).
    pub profile: EngineProfile,
    /// The review instruction appended as the child's opening user turn (hermes'
    /// `_SKILL_REVIEW_PROMPT` / memory-review analogue). The parent's system prompt + history seed
    /// the rest of the conversation.
    pub review_prompt: String,
}

impl BackgroundProfile {
    /// A background profile from a constrained engine profile + review prompt.
    pub fn new(profile: EngineProfile, review_prompt: impl Into<String>) -> Self {
        Self {
            profile,
            review_prompt: review_prompt.into(),
        }
    }
}

/// Maps a spawn `kind` to its [`BackgroundProfile`]. Unknown kinds are no-ops (the engine stays free
/// of the side-store/tool specifics — it only names a kind).
#[derive(Clone, Default)]
pub struct BackgroundProfileRegistry {
    by_kind: HashMap<String, BackgroundProfile>,
}

impl BackgroundProfileRegistry {
    /// An empty registry (every spawn is a no-op until a kind is registered).
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `profile` under `kind` (e.g. `"skill_review"`, `"memory_review"`).
    pub fn with(mut self, kind: impl Into<String>, profile: BackgroundProfile) -> Self {
        self.by_kind.insert(kind.into(), profile);
        self
    }

    /// The profile registered for `kind`, if any.
    pub fn get(&self, kind: &str) -> Option<&BackgroundProfile> {
        self.by_kind.get(kind)
    }

    /// Whether no kinds are registered (the spawn mechanism is effectively disabled).
    pub fn is_empty(&self) -> bool {
        self.by_kind.is_empty()
    }
}

/// The deterministic child id a background spawn of `kind` from `parent` at `epoch` materializes:
/// `<parent>/rv-<kind>-<epoch>`. Deterministic so a recovered/duplicate spawn dedupes onto the same
/// child; the `/rv-` segment lets the incarnation recover the kind to pick the constrained profile.
pub fn background_child_id(parent: &SessionId, kind: &str, epoch: Epoch) -> SessionId {
    SessionId::new(format!("{}/rv-{}-{}", parent.as_str(), kind, epoch.0))
}

/// Recover the spawn `kind` from a background child id (the inverse of [`background_child_id`]);
/// `None` for a non-background session id.
pub fn background_kind_of(id: &SessionId) -> Option<String> {
    let last = id.as_str().rsplit('/').next()?;
    let rest = last.strip_prefix("rv-")?;
    let (kind, _epoch) = rest.rsplit_once('-')?;
    Some(kind.to_string())
}

/// Materializes attached, non-joining background children (§4.3) from the durable store: build the
/// constrained seed snapshot, create the child row, record the **non-joining** parent->child edge
/// (no delegation, so the child self-closes), and enqueue a wake so the shared activation manager
/// drives it through the same `CoreIncarnation` path as any other session.
#[derive(Clone)]
pub struct BackgroundSpawner {
    store: Arc<dyn SessionStore>,
    partition: PartitionId,
    registry: BackgroundProfileRegistry,
}

impl BackgroundSpawner {
    /// A spawner over the durable `store`, seeding children into `partition` from `registry`.
    pub fn new(
        store: Arc<dyn SessionStore>,
        partition: PartitionId,
        registry: BackgroundProfileRegistry,
    ) -> Self {
        Self {
            store,
            partition,
            registry,
        }
    }

    /// The constrained profile a session id resolves to (when it is a background child), so the
    /// incarnation can run the review child under skills-only / memory-only tools + bounded budget.
    pub fn profile_for(&self, id: &SessionId) -> Option<EngineProfile> {
        background_kind_of(id).and_then(|k| self.registry.get(&k).map(|bg| bg.profile.clone()))
    }

    /// Materialize a child for `spec` spawned by `parent` (at `parent_epoch`). `seed_conversation` is
    /// the parent's live conversation when the spawn is raised mid-turn (durable path); when `None`
    /// the parent's last durable snapshot is read instead. Returns the child id, or `None` for an
    /// unknown kind (no-op). Idempotent: a re-raised spawn returns the existing child.
    pub async fn spawn(
        &self,
        parent: &SessionId,
        parent_epoch: Epoch,
        spec: &SpawnSpec,
        seed_conversation: Option<Conversation>,
    ) -> Option<SessionId> {
        let bg = self.registry.get(&spec.kind)?;
        let child = background_child_id(parent, &spec.kind, parent_epoch);
        // Idempotent: a recovered/duplicate spawn finds the child already present.
        if self.store.status(&child).await.is_some() {
            return Some(child);
        }

        // Seed the child's conversation. Every `SpawnSeed` today seeds `FromConversation`: keep the
        // parent's system prompt + history (so the reviewer sees exactly what happened), then append
        // the review instruction as the opening user turn. `seed_conversation` is the parent's live
        // conversation (durable path, captured mid-turn); otherwise read its last durable snapshot.
        let _ = &spec.seed;
        let mut conversation = match seed_conversation {
            Some(conv) => conv,
            None => self
                .store
                .peek_snapshot(parent)
                .await
                .and_then(|blob| Snapshot::decode(&blob).ok())
                .map(|snap| snap.conversation)
                .unwrap_or_else(|| Conversation::new(SystemPrompt::new("background review"))),
        };
        conversation.push_user(UserMsg::new(bg.review_prompt.clone()));

        let mut snapshot = Snapshot::fresh(child.clone());
        snapshot.conversation = conversation;
        let blob = snapshot.encode().ok()?;

        self.store
            .create_session(child.clone(), self.partition, blob)
            .await
            .ok()?;
        // The non-joining edge: tree-visible for audit, but no bound job — `mark_completed` finds no
        // delegation and never wakes the parent (self-close). Contrast `bind_delegation`.
        self.store
            .record_child_edge(parent.clone(), child.clone(), spec.kind.clone())
            .await
            .ok()?;
        self.store.enqueue_wake(child.clone()).await;
        Some(child)
    }
}
