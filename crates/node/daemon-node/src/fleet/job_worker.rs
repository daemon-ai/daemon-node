// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The durable job worker: drives the durable job outbox by materializing each delegation as a
//! durable child session under the shared activation manager (recursive, crash-recoverable).

use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::{ManageEventView, SubagentPhase, TreeEvent};
use daemon_common::{PartitionId, ProfileRef, SessionId, UnitId};
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
        let child_root = roots.session_root(child.as_str());
        // Both roots are opened as fd-contained boundaries (openat2 RESOLVE_BENEATH |
        // RESOLVE_NO_SYMLINKS): every attachment read/write below is symlink-escape-proof.
        let (Ok(parent_cr), Ok(child_cr)) = (
            daemon_core::exec::ContainedRoot::open(&parent_root),
            daemon_core::exec::ContainedRoot::open(&child_root),
        ) else {
            return;
        };
        if child_cr
            .create_dir_all(std::path::Path::new("inbox"))
            .await
            .is_err()
        {
            return;
        }
        for path in paths {
            // The attachment path is agent-influenced (the parent's workspace); read it fd-contained.
            let Ok(bytes) = parent_cr.read(std::path::Path::new(path)).await else {
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
            let _ = child_cr
                .write(&std::path::Path::new("inbox").join(name), &out)
                .await;
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
            // Decode the structured delegation (task + attachment paths + detached flag), falling back
            // to a legacy plain-text task for pre-upgrade jobs.
            let input = daemon_protocol::DelegationInput::decode(&job.payload);
            // A detached job carries a store-pre-minted `{parent}/dN` child id; a joining delegation
            // derives `{parent}/c{epoch}`.
            let child = job.child.clone().unwrap_or_else(|| Self::child_id(&job));
            // Create-if-absent: a fresh durable child session seeded with the delegated work as its
            // first turn (recovery-idempotent — a re-processed job finds the child already present).
            if self.store.status(&child).await.is_none() {
                // Seed the child with the real task and materialize any attachments into its inbox/
                // before the first turn.
                self.materialize_attachments(&job.session_id, &child, &input.attachments)
                    .await;
                // Captured before the task moves into the seed turn: the child's roster/tree title.
                let child_title: String = input.task.chars().take(80).collect();
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
                // Auth 4: a delegated child INHERITS the delegating (parent) session's owner — the
                // worker runs in a background service task with no request principal, so ownership
                // can only flow down the tree. Legacy/unowned parents leave the child unowned too.
                meta.owner = self
                    .store
                    .session_meta(&job.session_id)
                    .await
                    .and_then(|m| m.owner);
                // Per-child profile: a named profile in the delegation binds the child's engine
                // resolution — the durable resolver rehydrates it from `bound_profile` exactly like
                // an interactive session's binding. Unset keeps the one default engine shape, and
                // an unknown name silently falls back at resolve time (the resolver declines).
                if let Some(profile) = &input.profile {
                    meta.bound_profile = Some(ProfileRef::new(profile.clone()));
                }
                // Title the child from its task (truncated) so the parent's `status` verb and the
                // GUI tree show what each child is doing; never clobbers an existing title.
                if meta.title.is_none() && !child_title.is_empty() {
                    meta.title = Some(child_title);
                }
                self.store
                    .set_session_meta(&child, meta)
                    .await
                    .map_err(ServiceError::new)?;
                // Real topology change: push the spawn delta so a live `tree_subscribe` shows the new
                // subagent row promptly (the conformance "push before poll" guarantee).
                self.emit_spawn(&job.session_id, &child, map_store_role(child_role))
                    .await;
            }
            // Durable tree edge (idempotent). A DETACHED child binds a completion-notice edge: its
            // terminal completion delivers a notice to the parent (a fresh reactive turn), NOT a job
            // completion — the parent never suspended. A joining delegation binds the parent job, so
            // the child's terminal completion fulfills it and wakes the suspended parent.
            if input.detached {
                self.store
                    .bind_completion_notice(&child, &job.session_id)
                    .await
                    .map_err(ServiceError::new)?;
            } else {
                self.store
                    .bind_delegation(child.clone(), job.clone())
                    .await
                    .map_err(ServiceError::new)?;
            }
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

    /// Processing a delegation job stamps the child's host meta from the structured payload: the
    /// role derives from the declared `ChildLifetime`, a named profile binds `bound_profile` (the
    /// durable resolver's key), and the task becomes the child's title (the status/tree label).
    #[tokio::test]
    async fn worker_stamps_role_profile_and_title_from_the_delegation() {
        use daemon_common::{Epoch, FenceToken, JobId};
        use daemon_store::{Checkpoint, JobCommand};

        let store: Arc<dyn daemon_store::SessionStore> =
            Arc::new(daemon_store::InMemoryStore::new());
        let profile = EngineProfile::new(
            Arc::new(|| Arc::new(MockProvider::completing("x")) as Arc<dyn daemon_core::Provider>),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("t"),
        );
        let worker = FleetJobWorker::new(store.clone(), PartitionId::DEFAULT, profile);

        // A parent whose suspension enqueued an ephemeral, profile-bound delegation.
        let parent = SessionId::new("parent");
        store
            .create_session(
                parent.clone(),
                PartitionId::DEFAULT,
                daemon_common::SnapshotBlob::default(),
            )
            .await
            .unwrap();
        let payload = daemon_protocol::DelegationInput {
            task: "summarize the repo".into(),
            attachments: Vec::new(),
            lifetime: daemon_protocol::DelegationLifetime::Ephemeral,
            profile: Some("opus".into()),
            detached: false,
        }
        .encode();
        let job = JobCommand {
            job_id: JobId::new("parent:1:job"),
            session_id: parent.clone(),
            epoch: Epoch(1),
            payload,
            lifetime: daemon_store::ChildLifetime::Ephemeral,
            child: None,
        };
        store
            .checkpoint_and_enqueue(
                Checkpoint::new(
                    parent.clone(),
                    Epoch(1),
                    daemon_common::SnapshotBlob::default(),
                ),
                job,
                FenceToken::ZERO,
            )
            .await
            .unwrap();

        worker.process_jobs_once().await.unwrap();

        let child = SessionId::new("parent/c1");
        assert!(store.status(&child).await.is_some(), "child materialized");
        let meta = store.session_meta(&child).await.expect("child meta");
        assert_eq!(
            meta.role,
            Some(daemon_store::SessionRole::EphemeralSubagent),
            "role derives from the job's declared lifetime"
        );
        assert_eq!(
            meta.bound_profile.as_ref().map(|p| p.as_str()),
            Some("opus"),
            "the named profile binds the child's engine resolution"
        );
        assert_eq!(
            meta.title.as_deref(),
            Some("summarize the repo"),
            "the task titles the child for status/tree views"
        );
        assert_eq!(meta.parent, Some(parent));
    }

    /// A detached (`spawn wait:false`) job materializes the child at the store-pre-minted `{parent}/dN`
    /// id and binds a completion-notice edge (NOT a delegation edge): the child is tree-visible, and
    /// its terminal `mark_completed` pushes a `CompletionNotice` rather than a job completion.
    #[tokio::test]
    async fn worker_materializes_a_detached_child_with_a_notice_edge() {
        use daemon_common::{Epoch, JobId, SnapshotBlob};
        use daemon_store::{Checkpoint, JobCommand};

        let store: Arc<dyn daemon_store::SessionStore> =
            Arc::new(daemon_store::InMemoryStore::new());
        let profile = EngineProfile::new(
            Arc::new(|| Arc::new(MockProvider::completing("x")) as Arc<dyn daemon_core::Provider>),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("t"),
        );
        let worker = FleetJobWorker::new(store.clone(), PartitionId::DEFAULT, profile);

        let parent = SessionId::new("parent");
        let payload = daemon_protocol::DelegationInput {
            task: "background work".into(),
            attachments: Vec::new(),
            lifetime: daemon_protocol::DelegationLifetime::Persistent,
            profile: None,
            detached: true,
        }
        .encode();
        // The store mints the `{parent}/d1` child and stamps it onto the bare job.
        let child = store
            .enqueue_detached_job(JobCommand {
                job_id: JobId::new("parent:detached"),
                session_id: parent.clone(),
                epoch: Epoch::ZERO,
                payload,
                lifetime: daemon_store::ChildLifetime::Persistent,
                child: None,
            })
            .await
            .unwrap();
        assert_eq!(child.as_str(), "parent/d1");

        worker.process_jobs_once().await.unwrap();

        // The child materialized at the pre-minted id and is tree-visible under the parent.
        assert!(
            store.status(&child).await.is_some(),
            "child materialized at the pre-minted detached id"
        );
        assert!(store.children_of(&parent).await.contains(&child));

        // The notice edge (not a delegation edge): the child's terminal completion pushes a
        // CompletionNotice — a delegation edge would instead record a parent completion + wake.
        let fence = store.acquire_activation_lease(&child).await.unwrap();
        store
            .mark_completed(
                Checkpoint::new(child.clone(), Epoch(1), SnapshotBlob::default()),
                fence,
            )
            .await
            .unwrap();
        let notice = store
            .dequeue_completion_notice()
            .await
            .expect("a detached child fires a completion notice");
        assert_eq!(notice.parent, parent);
        assert_eq!(notice.child, child);
        // A joining delegation would have woken the parent; a notice edge must not.
        let mut woke_parent = false;
        while let Some(id) = store.dequeue_wake().await {
            if id == parent {
                woke_parent = true;
            }
        }
        assert!(
            !woke_parent,
            "a detached child never wakes its parent through the wake outbox"
        );
    }
}
