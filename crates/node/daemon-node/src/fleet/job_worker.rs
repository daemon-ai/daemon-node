// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The durable job worker: drives the durable job outbox by materializing each delegation as a
//! durable child session under the shared activation manager (recursive, crash-recoverable).

use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::{ManageEventView, SubagentPhase, TreeEvent};
use daemon_common::{PartitionId, SessionId, UnitId};
use daemon_core::EngineProfile;
use daemon_host::{BlobStore, JobWorker, ServiceError, WorkspaceRoots};

/// Drives the durable job outbox by materializing each delegation as a *durable child session*:
/// seed a fresh orchestrator-capable engine snapshot with the delegated work, create the child row,
/// bind it to the parent's job (so its terminal completion wakes the parent — store-parent-link),
/// and enqueue a wake. The one shared [`daemon_activation::ActivationManager`] then drives the child
/// through the same `CoreIncarnation` path as the top session; if the child itself delegates it
/// suspends and enqueues its own job (parent = child), so nesting is recursive and crash-recoverable
/// at every depth. The legacy synchronous `FleetRuntime::spawn_and_run` is retained only for the
/// foreign/ephemeral coarse lifecycle, not this path.
pub struct FleetJobWorker {
    store: Arc<dyn daemon_store::SessionStore>,
    partition: PartitionId,
    /// The orchestrator-capable profile every durable session (top and child) is built from — one
    /// engine shape at every level. Used here to seed a fresh child's first turn.
    profile: EngineProfile,
    /// The host fleet event bus (I4/I8). On a real durable child create the worker pushes a
    /// [`TreeEvent::Subagent`] spawn marker so `tree_subscribe` shows the new subagent row promptly
    /// (before any poll interval). `None` => no live push from the durable delegation seam.
    events: Option<tokio::sync::broadcast::Sender<TreeEvent>>,
    /// Monotonic sequence for the spawn markers the worker emits onto the bus.
    bus_seq: std::sync::atomic::AtomicU64,
    /// Workspace roots for materializing delegated attachments (parent -> child inbox/). `None`
    /// disables attachment transfer.
    workspace_roots: Option<Arc<WorkspaceRoots>>,
    /// The content store used to put/fetch delegated attachment bytes. `None` disables transfer.
    blobs: Option<Arc<dyn BlobStore>>,
}

impl FleetJobWorker {
    /// A durable job worker that seeds children from `profile` into `store` under `partition`.
    pub fn new(
        store: Arc<dyn daemon_store::SessionStore>,
        partition: PartitionId,
        profile: EngineProfile,
    ) -> Self {
        Self {
            store,
            partition,
            profile,
            events: None,
            bus_seq: std::sync::atomic::AtomicU64::new(0),
            workspace_roots: None,
            blobs: None,
        }
    }

    /// Give the worker the workspace roots + content store so it materializes a delegation's
    /// attachment paths (read from the parent's workspace, round-tripped through the content store)
    /// into the child's `inbox/` before the child's first turn. No-op transfer when unset.
    pub fn with_workspace(mut self, roots: Arc<WorkspaceRoots>, blobs: Arc<dyn BlobStore>) -> Self {
        self.workspace_roots = Some(roots);
        self.blobs = Some(blobs);
        self
    }

    /// Materialize a delegation's attachment paths from the parent workspace into the child's
    /// `inbox/`, round-tripping each through the content store (dedup + integrity; federation-ready).
    /// Best-effort: a missing/contained-rejected path or store error is skipped, never failing the
    /// job. No-op when no workspace/blob store is wired or there are no attachments.
    async fn materialize_attachments(
        &self,
        parent: &SessionId,
        child: &SessionId,
        paths: &[String],
    ) {
        let (Some(roots), Some(blobs)) = (&self.workspace_roots, &self.blobs) else {
            return;
        };
        if paths.is_empty() {
            return;
        }
        let parent_root = roots.session_root(parent.as_str());
        let inbox = roots.session_root(child.as_str()).join("inbox");
        if std::fs::create_dir_all(&inbox).is_err() {
            return;
        }
        for path in paths {
            let Ok(src) = daemon_core::exec::contain(&parent_root, std::path::Path::new(path))
            else {
                continue;
            };
            let Ok(bytes) = std::fs::read(&src) else {
                continue;
            };
            let Ok(blob_ref) = blobs.put(&bytes).await else {
                continue;
            };
            let Ok(out) = blobs.get(&blob_ref.hash, None).await else {
                continue;
            };
            let name = std::path::Path::new(path)
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("attachment"));
            let _ = std::fs::write(inbox.join(name), out);
        }
    }

    /// Inject the host fleet event bus so a durable child create pushes a live spawn delta. Call
    /// during assembly with the same sender wired into `NodeApiImpl`/`FleetRuntime`.
    pub fn with_event_sink(mut self, events: tokio::sync::broadcast::Sender<TreeEvent>) -> Self {
        self.events = Some(events);
        self
    }

    /// Push the spawn marker for a freshly-created durable child onto the fleet bus (role from the
    /// job's `ChildLifetime`, active count = the parent's current durable child total). A no-op when
    /// no bus is wired.
    async fn emit_spawn(
        &self,
        parent: &SessionId,
        child: &SessionId,
        role: daemon_api::SessionRole,
    ) {
        let Some(events) = &self.events else {
            return;
        };
        let active_children = self.store.children_of(parent).await.len() as u32;
        let seq = self
            .bus_seq
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _ = events.send(TreeEvent::Subagent(ManageEventView::Subagent {
            seq,
            child: UnitId::new(child.as_str()),
            role,
            phase: SubagentPhase::Spawned,
            active_children,
        }));
    }

    /// The deterministic id of the child session a delegation job materializes: the parent's id plus
    /// a `/c{epoch}` path segment. Deterministic so a re-enqueued/recovered job dedupes onto the same
    /// child, and the `/`-delimited path encodes the tree depth the orchestrate-tool guard reads.
    fn child_id(job: &daemon_store::JobCommand) -> SessionId {
        SessionId::new(format!("{}/c{}", job.session_id, job.epoch.0))
    }
}

/// Map a durable-store session role to its wire-surface equivalent (for the fleet bus markers).
fn map_store_role(role: daemon_store::SessionRole) -> daemon_api::SessionRole {
    match role {
        daemon_store::SessionRole::Primary => daemon_api::SessionRole::Primary,
        daemon_store::SessionRole::ManagedChild => daemon_api::SessionRole::ManagedChild,
        daemon_store::SessionRole::EphemeralSubagent => daemon_api::SessionRole::EphemeralSubagent,
    }
}

#[async_trait]
impl JobWorker for FleetJobWorker {
    async fn process_jobs_once(&self) -> Result<(), ServiceError> {
        while let Some(job) = self.store.dequeue_job().await {
            let child = Self::child_id(&job);
            // Create-if-absent: a fresh durable child session seeded with the delegated work as its
            // first turn (recovery-idempotent — a re-processed job finds the child already present).
            if self.store.status(&child).await.is_none() {
                // Decode the structured delegation (task + attachment paths), falling back to a
                // legacy plain-text task for pre-upgrade jobs. Seed the child with the real task and
                // materialize any attachments into its inbox/ before the first turn.
                let input = daemon_protocol::DelegationInput::decode(&job.payload);
                self.materialize_attachments(&job.session_id, &child, &input.attachments)
                    .await;
                let mut engine = self.profile.fresh(child.clone());
                engine.push_user(daemon_protocol::UserMsg::new(input.task));
                let blob = engine.snapshot().encode().map_err(ServiceError::new)?;
                self.store
                    .create_session(child.clone(), self.partition, blob)
                    .await
                    .map_err(ServiceError::new)?;
                // Stamp the hierarchy edge so the child is excluded from the `TopLevel` roster and
                // reached only by walking the tree: it is a non-`Primary` child of the delegating
                // session. Read-modify-write preserves any bound profile/overlay; the role is
                // derived from the job's declared `ChildLifetime` (managed vs ephemeral subagent).
                let mut meta = self.store.session_meta(&child).await.unwrap_or_default();
                meta.parent = Some(job.session_id.clone());
                let child_role = job.lifetime.role();
                meta.role = Some(child_role);
                self.store
                    .set_session_meta(&child, meta)
                    .await
                    .map_err(ServiceError::new)?;
                // Real topology change: push the spawn delta so a live `tree_subscribe` shows the new
                // subagent row promptly (the conformance "push before poll" guarantee).
                self.emit_spawn(&job.session_id, &child, map_store_role(child_role))
                    .await;
            }
            // Durable tree edge: the child's terminal completion fulfills this job and wakes the
            // parent (in the store's mark_completed transaction). Idempotent.
            self.store
                .bind_delegation(child.clone(), job.clone())
                .await
                .map_err(ServiceError::new)?;
            // Kick the child into its first turn via the shared wake dispatcher.
            self.store.enqueue_wake(child).await;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_core::{MockProvider, Provider, SystemPrompt, ToolRegistry};
    use daemon_host::FileBlobStore;

    /// The fleet job worker materializes a delegation's attachment paths from the parent's workspace
    /// into the child's `inbox/`, round-tripping through the content store (content-transfer Phase 2a,
    /// delegation-down).
    #[tokio::test]
    async fn worker_materializes_attachment_into_child_inbox() {
        let ws = std::env::temp_dir().join(format!("daemon-worker-ws-{}", std::process::id()));
        let cas = std::env::temp_dir().join(format!("daemon-worker-cas-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&cas);
        let roots = Arc::new(WorkspaceRoots::new(ws.clone()));
        let blobs: Arc<dyn BlobStore> = Arc::new(FileBlobStore::open(cas.clone()).unwrap());
        let store: Arc<dyn daemon_store::SessionStore> =
            Arc::new(daemon_store::InMemoryStore::new());
        let profile = EngineProfile::new(
            Arc::new(|| Arc::new(MockProvider::completing("x")) as Arc<dyn Provider>),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("t"),
        );
        let worker = FleetJobWorker::new(store, PartitionId::DEFAULT, profile)
            .with_workspace(roots.clone(), blobs);

        let parent = SessionId::new("parent");
        let child = SessionId::new("parent/c1");
        let pdir = roots.session_root(parent.as_str());
        std::fs::create_dir_all(&pdir).unwrap();
        std::fs::write(pdir.join("input.txt"), b"hand me down").unwrap();

        worker
            .materialize_attachments(&parent, &child, &["input.txt".to_string()])
            .await;

        let landed = roots.session_root(child.as_str()).join("inbox/input.txt");
        assert_eq!(std::fs::read(&landed).unwrap(), b"hand me down");

        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&cas);
    }
}
