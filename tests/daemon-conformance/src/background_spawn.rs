// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! §4.3 end-to-end: an attached, non-joining background child materialized by the host
//! `BackgroundSpawner` is driven to completion by the *same* `ActivationManager` as any session,
//! shows under its parent in the durable tree (audit), and self-closes without waking the parent.

use daemon_activation::ActivationManager;
use daemon_common::{Epoch, PartitionId, SessionId};
use daemon_core::{
    EngineProfile, MockProvider, Provider, ProviderBuilder, Snapshot, SystemPrompt, ToolRegistry,
};
use daemon_host::{
    background_kind_of, BackgroundProfile, BackgroundProfileRegistry, BackgroundSpawner,
    CoreEngineFactory,
};
use daemon_protocol::{SpawnSeed, SpawnSpec};
use daemon_store::{InMemoryStore, SessionStatus, SessionStore};
use std::sync::Arc;

const PARTITION: PartitionId = PartitionId::DEFAULT;

/// An engine profile whose model finishes every turn in one toolless call.
fn completing_profile(text: &str) -> EngineProfile {
    let text = text.to_string();
    let provider: ProviderBuilder =
        Arc::new(move || Arc::new(MockProvider::completing(text.clone())) as Arc<dyn Provider>);
    EngineProfile::new(
        provider,
        Arc::new(ToolRegistry::new()),
        SystemPrompt::new("reviewer"),
    )
}

#[tokio::test]
async fn background_child_is_attached_and_self_closing() {
    let store = Arc::new(InMemoryStore::new());

    // One review kind whose constrained child completes in a single model call.
    let registry = BackgroundProfileRegistry::new().with(
        "skill_review",
        BackgroundProfile::new(completing_profile("reviewed"), "Review the conversation."),
    );
    let spawner = Arc::new(BackgroundSpawner::new(store.clone(), PARTITION, registry));

    // The shared activation manager drives parent and child through one factory; the factory is
    // background-aware so the child hydrates under its constrained review profile.
    let factory = Arc::new(
        CoreEngineFactory::from_profile(completing_profile("parent"))
            .with_background(spawner.clone()),
    );
    let mgr = ActivationManager::new(store.clone(), factory, PARTITION);

    // Seed a parent, then materialize a background child (as a mid-turn `Effect::Spawn` would).
    let parent = SessionId::new("parent");
    let blob = Snapshot::fresh(parent.clone()).encode().unwrap();
    store
        .create_session(parent.clone(), PARTITION, blob)
        .await
        .unwrap();
    let child = spawner
        .spawn(
            &parent,
            Epoch::ZERO,
            &SpawnSpec {
                kind: "skill_review".into(),
                seed: SpawnSeed::FromConversation,
            },
            None,
        )
        .await
        .expect("kind is registered -> child materialized");

    // Attached + labeled for audit; the id round-trips the kind.
    assert_eq!(store.children_of(&parent).await, vec![child.clone()]);
    assert_eq!(
        store.delegation_work(&child).await.as_deref(),
        Some("skill_review")
    );
    assert_eq!(background_kind_of(&child).as_deref(), Some("skill_review"));

    // The spawner enqueued exactly the child's wake; drive it to terminal.
    let woken = store.dequeue_wake().await;
    assert_eq!(woken.as_ref(), Some(&child), "only the child was woken");
    mgr.wake(child.clone()).await.unwrap();

    assert_eq!(
        store.status(&child).await,
        Some(SessionStatus::Completed),
        "the background child self-closes"
    );
    assert!(
        store.dequeue_wake().await.is_none(),
        "an attached non-joining child must never wake its parent"
    );
    assert_eq!(
        store.status(&parent).await,
        Some(SessionStatus::Ready),
        "the parent is untouched by the child's completion"
    );
}

/// An unregistered kind is a host-side no-op: no child row, no edge, no wake.
#[tokio::test]
async fn unknown_kind_is_a_noop() {
    let store = Arc::new(InMemoryStore::new());
    let spawner =
        BackgroundSpawner::new(store.clone(), PARTITION, BackgroundProfileRegistry::new());
    let parent = SessionId::new("parent");
    let blob = Snapshot::fresh(parent.clone()).encode().unwrap();
    store
        .create_session(parent.clone(), PARTITION, blob)
        .await
        .unwrap();
    let out = spawner
        .spawn(
            &parent,
            Epoch::ZERO,
            &SpawnSpec {
                kind: "nope".into(),
                seed: SpawnSeed::FromConversation,
            },
            None,
        )
        .await;
    assert!(out.is_none(), "unknown kind -> no-op");
    assert!(store.children_of(&parent).await.is_empty());
    assert!(store.dequeue_wake().await.is_none());
}
