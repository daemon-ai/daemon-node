// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use daemon_activation::ActivationManager;
use daemon_common::{PartitionId, SessionId};
use daemon_core::Snapshot;
use daemon_host::CoreEngineFactory;
use daemon_store::{InMemoryStore, JobCompletion, SessionStatus, SessionStore};
use std::sync::Arc;

pub const PARTITION: PartitionId = PartitionId::DEFAULT;

/// A fresh store + a single activation manager owning the default partition.
pub fn new_world() -> (Arc<InMemoryStore>, ActivationManager) {
    let store = Arc::new(InMemoryStore::new());
    let mgr = manager(store.clone());
    (store, mgr)
}

/// An activation manager over an existing (possibly shared) store, driving the real engine.
pub fn manager(store: Arc<InMemoryStore>) -> ActivationManager {
    ActivationManager::new(store, Arc::new(CoreEngineFactory::delegating()), PARTITION)
}

/// Create a fresh `Ready` session with an encoded empty snapshot.
pub async fn seed(store: &InMemoryStore, id: &SessionId) {
    let blob = Snapshot::fresh(id.clone())
        .encode()
        .expect("encode fresh snapshot");
    store
        .create_session(id.clone(), PARTITION, blob)
        .await
        .expect("create session");
}

pub async fn status(store: &InMemoryStore, id: &SessionId) -> Option<SessionStatus> {
    store.status(id).await
}

pub async fn assert_completed(store: &InMemoryStore, id: &SessionId) {
    assert_eq!(
        status(store, id).await,
        Some(SessionStatus::Completed),
        "session {id} should be Completed"
    );
}

/// Build a completion for whatever job is sitting on the durable outbox.
pub async fn completion_for_next_job(store: &InMemoryStore) -> JobCompletion {
    let job = store.dequeue_job().await.expect("a job on the outbox");
    JobCompletion {
        session_id: job.session_id,
        epoch: job.epoch,
        job_id: job.job_id,
        payload: job.payload,
    }
}
