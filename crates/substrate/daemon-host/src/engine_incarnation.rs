// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The engine ⇄ activation-seam adapter (host-spec §3.1).
//!
//! `daemon-core` is deliberately free of the durable substrate (it depends only on
//! `daemon-protocol`). [`CoreIncarnation`] is the host-owned bridge that lets the activation layer
//! drive a real engine through the protocol-agnostic [`Incarnation`] seam: it decodes the durable
//! [`SnapshotBlob`] into the engine's typed [`Snapshot`], applies unapplied completions, runs one
//! turn, and maps the engine's terminal/suspension outcome back onto the seam's [`Step`].
//!
//! Background delegation on this path is resolved by [`DelegateResolver`], a built-in
//! [`HostRequestHandler`] that mints the deterministic durable `JobId` the activation outbox keys on
//! — the live management-protocol escalation path is the actor-backed `EngineUnit` (see
//! [`crate::unit`]).

use crate::background::BackgroundSpawner;
use crate::blob_store::BlobStore;
use crate::journal::{JournalFeeder, JournalSink};
use crate::node_api::{decode_overlay, DurableProfileResolver};
use crate::workspace_fs::WorkspaceRoots;
use async_trait::async_trait;
use daemon_activation::{EngineError, EngineFactory, Incarnation, SnapshotBlob, Step};
use daemon_common::{Epoch, JobId, JournalStreamId, ProfileRef, SessionId};
use daemon_core::{
    Completion, Conversation, DelegateTool, Engine, EngineProfile, EventSink, Failure,
    MockProvider, Provider, Snapshot, SystemPrompt, ToolRegistry, TurnControl, TurnOutcome,
};
use daemon_protocol::{
    HostRequest, HostRequestHandler, HostRequestKind, HostResponse, HostResponseBody, Outbound,
};
use daemon_store::{JobCommand, JobCompletion, ParkedApproval, SessionStore};
use daemon_telemetry::{TraceSigner, GENESIS_ROOT};
use std::sync::{Arc, Mutex};

/// The store + signer a durable incarnation journals into. Injected by the composition root; when
/// absent the durable path runs without journaling (e.g. the substrate conformance suite).
#[derive(Clone)]
pub struct JournalConfig {
    /// The authoritative store the journal is appended to + sealed in.
    pub store: Arc<dyn SessionStore>,
    /// The node's segment-root signer.
    pub signer: Arc<TraceSigner>,
}

// The provider/credential builder type aliases now live with the [`EngineProfile`] in `daemon-core`
// (the one composition seam); re-exported here for callers that still reference them by this path.
pub use daemon_core::{CredentialBuilder, ProviderBuilder};

/// Builds core-backed [`Incarnation`]s from a shared [`EngineProfile`] — the durable activation
/// path's view of the one engine-construction seam.
#[derive(Clone)]
pub struct CoreEngineFactory {
    profile: EngineProfile,
    journal: Option<JournalConfig>,
    /// The §4.3 background-spawn materializer, when configured. Threaded into every incarnation so
    /// (a) `Effect::Spawn` host requests materialize attached non-joining children, and (b) a
    /// background child session hydrates under its constrained review profile instead of `profile`.
    background: Option<Arc<BackgroundSpawner>>,
    /// The per-session profile resolver (bound profile ref + persisted overlay -> `EngineProfile`),
    /// injected by the node. When set, a durable session with a recorded bound profile rehydrates
    /// from *its own* profile + overlay (the unified resolution path) instead of pinning this
    /// factory's fixed `profile`; `None` (or no recorded binding) falls back to `profile`.
    resolver: Option<DurableProfileResolver>,
    /// The content store + workspace roots for node-mediated artifact transfer (content-transfer
    /// Phase 2a): at a child's terminal completion the incarnation captures its `outbox/` into the
    /// store; on a parent's hydrate it materializes a child's returned artifacts into the parent's
    /// `inbox/`. `None` disables artifact capture/materialization (the legacy `child:{id}` marker).
    content: Option<ContentTransfer>,
    /// The constrained profile a **cron-fired** session (`session_meta.scheduled_job.is_some()`)
    /// runs its turn under (I15/G3): an orchestrator-free, `cron`-tool-free toolset so a scheduled
    /// run cannot self-schedule or self-delegate (runaway prevention). When set it overrides the
    /// resolver/fallback for any scheduled session; `None` leaves cron sessions on the default path.
    cron_profile: Option<EngineProfile>,
}

/// The node-side content-transfer handles threaded into a durable incarnation (blob store +
/// workspace roots), used to capture/materialize delegated artifacts.
#[derive(Clone)]
struct ContentTransfer {
    blobs: Arc<dyn BlobStore>,
    roots: Arc<WorkspaceRoots>,
}

impl CoreEngineFactory {
    /// A factory whose engines delegate one unit of background work and then complete — the durable
    /// "delegate → suspend → resume → complete" cycle the substrate conformance suite drives.
    pub fn delegating() -> Self {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(DelegateTool::new("background-work")));
        let profile = EngineProfile::new(
            Arc::new(|| {
                Arc::new(MockProvider::delegating("delegate", "work complete")) as Arc<dyn Provider>
            }),
            Arc::new(registry),
            SystemPrompt::new("daemon-core conformance engine"),
        );
        Self {
            profile,
            journal: None,
            background: None,
            resolver: None,
            content: None,
            cron_profile: None,
        }
    }

    /// A factory over a custom provider builder, tool registry, and system prompt.
    pub fn with_provider(
        provider: ProviderBuilder,
        registry: Arc<ToolRegistry>,
        system: SystemPrompt,
    ) -> Self {
        Self {
            profile: EngineProfile::new(provider, registry, system),
            journal: None,
            background: None,
            resolver: None,
            content: None,
            cron_profile: None,
        }
    }

    /// A factory over an already-assembled [`EngineProfile`] (the binary's composition root).
    pub fn from_profile(profile: EngineProfile) -> Self {
        Self {
            profile,
            journal: None,
            background: None,
            resolver: None,
            content: None,
            cron_profile: None,
        }
    }

    /// Inject the verifiable-journal store + signer so every durable incarnation this factory builds
    /// seals its turn into the unified journal (the durable production journaling path).
    pub fn with_journal(mut self, store: Arc<dyn SessionStore>, signer: Arc<TraceSigner>) -> Self {
        self.journal = Some(JournalConfig { store, signer });
        self
    }

    /// Inject the §4.3 background-spawn materializer so this factory's incarnations can spawn
    /// attached, non-joining review children and hydrate them under their constrained profile.
    pub fn with_background(mut self, background: Arc<BackgroundSpawner>) -> Self {
        self.background = Some(background);
        self
    }

    /// Inject the per-session profile resolver so durable sessions rehydrate from their own bound
    /// profile + persisted overlay (unified live/durable resolution) instead of this factory's fixed
    /// profile. Requires the journal store (the source of the session metadata) to be set too.
    pub fn with_session_resolver(mut self, resolver: DurableProfileResolver) -> Self {
        self.resolver = Some(resolver);
        self
    }

    /// Inject the constrained cron-run profile (I15/G3): a cron-fired session
    /// (`session_meta.scheduled_job.is_some()`) hydrates under this orchestrator-free, `cron`-free
    /// toolset instead of the resolver/fallback profile, so a scheduled run cannot self-schedule or
    /// self-delegate. Leave unset to run cron sessions on the default profile path.
    pub fn with_cron_profile(mut self, profile: EngineProfile) -> Self {
        self.cron_profile = Some(profile);
        self
    }

    /// Inject the content store + workspace roots so this factory's incarnations capture a child's
    /// `outbox/` artifacts at completion and materialize a child's returned artifacts into a parent's
    /// `inbox/` on hydrate (daemon-content-transfer-spec.md Phase 2a, node-mediated).
    pub fn with_content(mut self, blobs: Arc<dyn BlobStore>, roots: Arc<WorkspaceRoots>) -> Self {
        self.content = Some(ContentTransfer { blobs, roots });
        self
    }

    /// Inject an authority-backed (or brokered) credential provider + profile into every engine
    /// this factory builds — the host bridge for the §7 port (host-spec §6).
    pub fn with_credentials(mut self, credentials: CredentialBuilder, profile: ProfileRef) -> Self {
        self.profile = self.profile.with_credentials(credentials, profile);
        self
    }
}

impl EngineFactory for CoreEngineFactory {
    fn create(&self) -> Box<dyn Incarnation> {
        Box::new(CoreIncarnation {
            profile: self.profile.clone(),
            engine: None,
            journal: self.journal.clone(),
            background: self.background.clone(),
            resolver: self.resolver.clone(),
            content: self.content.clone(),
            cron_profile: self.cron_profile.clone(),
            completion_payload: None,
        })
    }
}

/// One core-backed engine incarnation driven through the activation seam.
pub struct CoreIncarnation {
    profile: EngineProfile,
    engine: Option<Engine>,
    journal: Option<JournalConfig>,
    /// The §4.3 background-spawn materializer (when configured): drives `Effect::Spawn` requests and
    /// selects the constrained review profile when *this* incarnation is itself a background child.
    background: Option<Arc<BackgroundSpawner>>,
    /// The per-session profile resolver (when configured): re-resolves this session's `EngineProfile`
    /// from its bound profile + persisted overlay at hydrate, so a durable session honors its own
    /// profile + restored session override instead of the factory's fixed profile.
    resolver: Option<DurableProfileResolver>,
    /// Node-side content transfer (blob store + workspace roots) for capturing/materializing
    /// delegated artifacts; `None` disables it.
    content: Option<ContentTransfer>,
    /// The constrained cron-run profile (I15/G3): used in place of the resolver/fallback when this
    /// incarnation hydrates a cron-fired session, so the run carries no `cron`/`orchestrate` tools.
    cron_profile: Option<EngineProfile>,
    /// The structured completion payload captured at `Step::Completed` (a CBOR `DelegationResult`
    /// over the child's `outbox/`), surfaced via [`Incarnation::completion_payload`]. `None` => no
    /// artifacts captured (the store falls back to the legacy `child:{id}` marker).
    completion_payload: Option<Vec<u8>>,
}

fn map_failure(failure: Failure) -> EngineError {
    EngineError::Other(failure.to_string())
}

impl CoreIncarnation {
    /// Re-resolve `session`'s effective [`EngineProfile`] from its host-level metadata (the bound
    /// profile plus the persisted overlay) via the injected resolver. Returns `None` when no
    /// resolver/journal store is wired or no profile binding is recorded, so the caller then falls
    /// back to the factory's default profile. This is the durable half of the one resolution path
    /// shared with the live surface, so a durable session honors its own profile and restored
    /// session override.
    async fn resolve_session_profile(&self, session: &SessionId) -> Option<EngineProfile> {
        let resolver = self.resolver.as_ref()?;
        let store = self.journal.as_ref().map(|cfg| &cfg.store)?;
        let meta = store.session_meta(session).await.unwrap_or_default();
        let overlay = decode_overlay(&meta.overlay);
        resolver(meta.bound_profile, &overlay)
    }

    /// Capture this (child) session's `outbox/` into the content store as a structured
    /// [`DelegationResult`](daemon_protocol::DelegationResult): each regular file is `blob_put` and
    /// referenced by name. Returns `None` (legacy marker) when content transfer is unwired or the
    /// `outbox/` is absent/empty. Best-effort: an unreadable file or store error is skipped.
    async fn capture_outbox(&self, session: &SessionId) -> Option<Vec<u8>> {
        let content = self.content.as_ref()?;
        let outbox = content.roots.session_root(session.as_str()).join("outbox");
        let mut artifacts = Vec::new();
        for entry in std::fs::read_dir(&outbox).ok()?.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let Ok(mut blob_ref) = content.blobs.put(&bytes).await else {
                continue;
            };
            blob_ref.name = path.file_name().map(|n| n.to_string_lossy().into_owned());
            artifacts.push(blob_ref);
        }
        if artifacts.is_empty() {
            return None;
        }
        let summary = format!("completed with {} artifact(s)", artifacts.len());
        Some(daemon_protocol::DelegationResult { summary, artifacts }.encode())
    }

    /// Materialize the artifacts a delegated child returned (decoded from its completion payload)
    /// into this (parent) session's `inbox/`, fetching each from the content store. Best-effort and a
    /// no-op when content transfer is unwired, the payload is legacy/structureless, or there are no
    /// artifacts. The basename guards against a name escaping `inbox/`.
    async fn materialize_artifacts(&self, session: &SessionId, payload: &[u8]) {
        let Some(content) = &self.content else {
            return;
        };
        let result = daemon_protocol::DelegationResult::decode(payload);
        if result.artifacts.is_empty() {
            return;
        }
        let inbox = content.roots.session_root(session.as_str()).join("inbox");
        if std::fs::create_dir_all(&inbox).is_err() {
            return;
        }
        for art in &result.artifacts {
            let Ok(bytes) = content.blobs.get(&art.hash, None).await else {
                continue;
            };
            let name = art
                .name
                .clone()
                .unwrap_or_else(|| format!("{}.bin", art.hash.to_hex()));
            let base = std::path::Path::new(&name)
                .file_name()
                .map(|n| n.to_owned())
                .unwrap_or_else(|| std::ffi::OsStr::new("artifact").to_owned());
            let _ = std::fs::write(inbox.join(base), bytes);
        }
    }
}

#[async_trait]
impl Incarnation for CoreIncarnation {
    async fn hydrate(
        &mut self,
        snapshot: SnapshotBlob,
        unapplied: Vec<JobCompletion>,
    ) -> Result<(), EngineError> {
        if snapshot.is_empty() {
            return Err(EngineError::Other(
                "core incarnation hydrated from an empty snapshot".into(),
            ));
        }
        let snap = Snapshot::decode(&snapshot)?;
        let session_id = snap.session_id.clone();
        // I15: read the cron origin once — it drives both the profile selection here (a cron-fired
        // session runs under the constrained cron profile) and the `TurnTrigger::Scheduled` arming
        // below, so we avoid a second `session_meta` round-trip.
        let scheduled_job = if let Some(store) = self.journal.as_ref().map(|cfg| &cfg.store) {
            store
                .session_meta(&session_id)
                .await
                .and_then(|m| m.scheduled_job)
        } else {
            None
        };
        // A background child (§4.3) hydrates under its constrained review profile (skills-only /
        // memory-only tools + bounded budget + nudges off), not the parent's full profile. A
        // cron-fired session (I15/G3) hydrates under the constrained cron profile (no `cron`/
        // `orchestrate` tools) so it cannot self-schedule. Otherwise, when a per-session resolver +
        // journal store are wired, re-resolve this session's profile from its persisted bound profile
        // + overlay (unified resolution: a durable session honors its own model/tools/approval
        // override). Falls back to the factory's fixed profile when no binding is recorded (e.g.
        // delegated orchestrator children).
        let profile = if let Some(bg_profile) = self
            .background
            .as_ref()
            .and_then(|bg| bg.profile_for(&snap.session_id))
        {
            bg_profile
        } else if scheduled_job.is_some() {
            // I15/G3 + Phase 2 shaping: a cron-fired session resolves its bound profile overlaid with
            // the run's persisted `SessionOverlay` (model/provider/tool-allowlist/workdir) through the
            // SAME unified resolver the live/durable paths use. That resolver is G3-safe **by
            // construction** — it builds the session tool registry from fs+shell+node-extras+skills and
            // never wires the `cron`/`orchestrate` tools — so honoring the overlay cannot let a
            // scheduled run self-schedule or self-delegate. Falls back to the explicitly-constrained
            // `cron_profile` (then the factory default) when no resolver/binding is wired.
            if let Some(resolved) = self.resolve_session_profile(&snap.session_id).await {
                resolved
            } else if let Some(cron_profile) = &self.cron_profile {
                cron_profile.clone()
            } else {
                self.profile.clone()
            }
        } else if let Some(resolved) = self.resolve_session_profile(&snap.session_id).await {
            resolved
        } else {
            self.profile.clone()
        };
        let mut engine = profile.from_snapshot(snap);
        // Node-side: materialize any artifacts the completed children returned into this (parent)
        // session's `inbox/` before the engine folds the completions (the engine sees only the
        // summary text; the files land on disk). Best-effort; no-op without content transfer.
        for completion in &unapplied {
            self.materialize_artifacts(&session_id, &completion.payload)
                .await;
        }
        let completions = unapplied
            .into_iter()
            .map(|c| Completion {
                job_id: c.job_id,
                payload: c.payload,
            })
            .collect();
        engine.apply_completions(completions);
        // Durable inbound-input seam: drain any host-enqueued pending inputs (a background
        // process-exit notification, a message to a delegated child) into the conversation as user
        // messages before the turn runs. This is how content reaches an activation-lifecycle session
        // that `SessionApi::submit` cannot drive (the one-lifecycle-owner guard-rail); the enqueuer
        // pairs it with `enqueue_wake` so this hydrate happens. Payloads are CBOR `UserMsg`s; an
        // undecodable payload is dropped with a warning rather than failing the activation.
        if let Some(store) = self.journal.as_ref().map(|cfg| &cfg.store) {
            for payload in store.take_session_inputs(&session_id).await {
                match ciborium::from_reader::<daemon_protocol::UserMsg, _>(payload.as_slice()) {
                    Ok(msg) => engine.push_user(msg),
                    Err(e) => tracing::warn!(
                        session = %session_id,
                        error = %e,
                        "dropping undecodable pending session input"
                    ),
                }
            }
        }
        // I15: a cron-fired session carries `SessionMeta::scheduled_job` (read above). Arm the next
        // turn's trigger as `TurnTrigger::Scheduled { job }` so the fired turn reports its scheduled
        // origin instead of the durable wake path's default `User`. One-shot (consumed by `run_turn`).
        if let Some(job) = scheduled_job {
            engine.set_next_trigger(daemon_protocol::TurnTrigger::Scheduled { job });
        }
        self.engine = Some(engine);
        Ok(())
    }

    async fn run(&mut self) -> Result<Step, EngineError> {
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| EngineError::Other("run before hydrate".into()))?;
        let session_id = engine.snapshot().session_id.clone();
        let segment = engine.epoch().0;
        // When background spawn is enabled, capture a clone of the parent's live conversation so a
        // mid-turn `Effect::Spawn` can seed the review child `FromConversation` without a store read.
        let seed_conversation = self
            .background
            .as_ref()
            .map(|_| engine.snapshot().conversation.clone());
        let host = DelegateResolver {
            session_id: session_id.clone(),
            epoch: engine.epoch(),
            background: self.background.clone(),
            seed_conversation,
            approval_seq: Mutex::new(0),
        };
        // When journaling, capture the engine's events so they can be coalesced into finished blocks
        // and sealed after the turn, and so the turn's token usage can be folded into the durable
        // per-session usage surface (the tree projection's usage source); otherwise discard.
        let captured: Arc<Mutex<Vec<daemon_protocol::AgentEvent>>> =
            Arc::new(Mutex::new(Vec::new()));
        let sink = if self.journal.is_some() {
            let cap = captured.clone();
            EventSink::new(move |ev| cap.lock().unwrap().push(ev))
        } else {
            EventSink::discarding()
        };
        let control = TurnControl::new();
        let outcome = engine
            .run_turn(&host, &sink, &control)
            .await
            .map_err(map_failure)?;

        // Fold this turn's token usage into the durable per-session usage surface so the management
        // tree projects real, recovery-survivable usage at every node (replacing the in-memory fleet
        // fan-in for durable sessions).
        if let Some(cfg) = &self.journal {
            let mut delta = daemon_common::UsageDelta::default();
            for ev in captured.lock().unwrap().iter() {
                if let daemon_protocol::AgentEvent::Usage { delta: d, .. } = ev {
                    delta.add(d);
                }
            }
            if delta != daemon_common::UsageDelta::default() {
                cfg.store.record_usage(&session_id, delta).await;
            }
        }

        // Seal this incarnation's turn into the unified verifiable journal (unfenced on the durable
        // path: the snapshot chain fences durable state, the ed25519 signature seals the transcript).
        if let Some(cfg) = &self.journal {
            let stream = JournalStreamId::session(&session_id);
            let prior = if segment == 0 {
                GENESIS_ROOT
            } else {
                cfg.store
                    .load_trace_segment(&stream, segment - 1)
                    .await
                    .and_then(|s| s.committed.map(|c| c.root))
                    .unwrap_or(GENESIS_ROOT)
            };
            let jsink = Arc::new(JournalSink::with_segment(
                cfg.store.clone(),
                cfg.signer.clone(),
                stream,
                None,
                segment,
                prior,
            ));
            let feeder = JournalFeeder::new(jsink);
            let events = std::mem::take(&mut *captured.lock().unwrap());
            for ev in events {
                feeder.feed(&Outbound::Event(ev)).await;
            }
        }

        // Re-index this session's searchable text at the turn boundary (the durable half of the
        // `session_search` FTS surface; the live pump indexes interactive sessions): the coalesced
        // full conversation (user + assistant text + tool names) replaces the prior row, so search
        // reflects the whole conversation, not just the opening turn. Best-effort by construction
        // (`index_session_text` swallows store errors).
        if let Some(cfg) = &self.journal {
            let title = cfg
                .store
                .session_meta(&session_id)
                .await
                .and_then(|m| m.title);
            let turns =
                crate::session_index::turns_from_conversation(&engine.snapshot().conversation);
            let body = crate::session_index::coalesce_body(&turns);
            if !body.trim().is_empty() {
                cfg.store
                    .index_session_text(&session_id, title, &body)
                    .await;
            }
        }

        match outcome {
            TurnOutcome::Completed(_) => {
                // Terminal deactivation (§10/§11): `Step::Completed` marks the session `Completed`
                // in the store (never re-activated), so flush the context engine + memory providers
                // (LCM final ingest + lifecycle finalize) before the final checkpoint is taken.
                engine.end_session().await;
                // Terminal: capture this child's `outbox/` artifacts into the content store as the
                // structured completion payload (the parent materializes them on its wake). `None`
                // when content transfer is unwired or no artifacts were produced (legacy marker).
                self.completion_payload = self.capture_outbox(&session_id).await;
                Ok(Step::Completed)
            }
            // §12 HITL: an approval park records its parked rows for the operator surface and enqueues
            // no runnable job (the activation layer routes it to `park_approval`). The snapshot keeps
            // the typed `PendingApproval`s (with the deferred `ToolCall`); these are the store rows.
            TurnOutcome::Suspended(suspension)
                if suspension.payload == daemon_core::APPROVAL_SUSPEND_PAYLOAD =>
            {
                let approvals = engine
                    .snapshot()
                    .pending_approvals
                    .iter()
                    .map(|p| ParkedApproval {
                        session_id: session_id.clone(),
                        job_id: p.job_id.clone(),
                        epoch: suspension.epoch,
                        prompt: p.prompt.clone(),
                        path: p.path.clone(),
                        decision: None,
                    })
                    .collect();
                Ok(Step::ParkApproval { approvals })
            }
            TurnOutcome::Suspended(suspension) => Ok(Step::Suspended {
                job: JobCommand {
                    job_id: suspension.job_id,
                    session_id,
                    epoch: suspension.epoch,
                    payload: suspension.payload,
                    // The orchestrate path delegates long-lived managed children; the ephemeral
                    // subagent producer is forward-looking, so default to `Persistent` here.
                    lifetime: daemon_store::ChildLifetime::default(),
                },
            }),
        }
    }

    fn checkpoint(&self) -> Result<SnapshotBlob, EngineError> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| EngineError::Other("checkpoint before hydrate".into()))?;
        Ok(engine.snapshot().encode()?)
    }

    fn epoch(&self) -> Epoch {
        self.engine.as_ref().map(|e| e.epoch()).unwrap_or_default()
    }

    fn completion_payload(&self) -> Option<Vec<u8>> {
        self.completion_payload.clone()
    }
}

/// The substrate-path host handler: resolves a delegation to the deterministic durable `JobId` the
/// activation outbox dedupes on, materializes an attached non-joining background child for a
/// `Spawn` (§4.3, fire-and-forget — never suspends the parent), and trivially answers the other §17
/// request kinds.
struct DelegateResolver {
    session_id: SessionId,
    epoch: Epoch,
    /// The §4.3 background-spawn materializer, when configured.
    background: Option<Arc<BackgroundSpawner>>,
    /// The parent's live conversation snapshot, captured before the turn so a `Spawn` seeds the
    /// review child `FromConversation` without a store round-trip (only `Some` when spawn is on).
    seed_conversation: Option<Conversation>,
    /// A per-run counter minting a deterministic `JobId` for each §12 edit-approval ask in turn
    /// order, so a gated tool on the durable path defers to a parked operator decision. Deterministic
    /// per `(session, post-bump epoch, ordinal)` so a recovery re-park reuses the same id (dedupe).
    approval_seq: Mutex<u32>,
}

#[async_trait]
impl HostRequestHandler for DelegateResolver {
    async fn request(&self, req: HostRequest) -> HostResponse {
        let body = match req.kind {
            HostRequestKind::Delegate { .. } => {
                // Deterministic per (session, post-bump epoch) so a recovery re-enqueue dedupes.
                let job_id = JobId::new(format!("{}:{}:job", self.session_id, self.epoch.next().0));
                HostResponseBody::Delegated(job_id)
            }
            HostRequestKind::Spawn { spec } => {
                // Fire-and-forget: materialize the attached non-joining child now and return its id;
                // the parent neither suspends nor waits. Unknown kind / no spawner -> no-op.
                let child = match &self.background {
                    Some(bg) => bg
                        .spawn(
                            &self.session_id,
                            self.epoch,
                            &spec,
                            self.seed_conversation.clone(),
                        )
                        .await
                        .unwrap_or_else(|| self.session_id.clone()),
                    None => self.session_id.clone(),
                };
                HostResponseBody::Spawned(child)
            }
            HostRequestKind::Approval { .. } => {
                // §12 durable HITL: a gated tool on the durable path asks only when its policy is
                // `Ask` (the engine already auto-allowed/denied otherwise). There is no synchronous
                // operator on this headless path, so defer: mint the deterministic parked `JobId` the
                // engine records + suspends on, to be answered later by `ApprovalDecide`.
                let mut seq = self.approval_seq.lock().unwrap();
                let job_id = JobId::new(format!(
                    "{}:{}:approval:{}",
                    self.session_id,
                    self.epoch.next().0,
                    *seq
                ));
                *seq += 1;
                HostResponseBody::Deferred(job_id)
            }
            HostRequestKind::Input { .. } => HostResponseBody::Input(String::new()),
            HostRequestKind::Choice { .. } => HostResponseBody::Chosen(0),
            _ => HostResponseBody::Approved(true),
        };
        HostResponse {
            request_id: req.request_id,
            body,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob_store::FileBlobStore;
    use daemon_core::{MockProvider, Provider, SystemPrompt, ToolRegistry};

    fn unique_dir(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("daemon-inc-{tag}-{}-{n}", std::process::id()))
    }

    fn incarnation_with_content(
        blobs: Arc<dyn BlobStore>,
        roots: Arc<WorkspaceRoots>,
    ) -> CoreIncarnation {
        let profile = EngineProfile::new(
            Arc::new(|| Arc::new(MockProvider::completing("done")) as Arc<dyn Provider>),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("test"),
        );
        CoreIncarnation {
            profile,
            engine: None,
            journal: None,
            background: None,
            resolver: None,
            content: Some(ContentTransfer { blobs, roots }),
            cron_profile: None,
            completion_payload: None,
        }
    }

    /// A child's `outbox/` is captured into the content store as a structured `DelegationResult`,
    /// and that result materializes back into a (parent) session's `inbox/` — the node-mediated
    /// artifact round-trip (daemon-content-transfer-spec.md Phase 2a, completion-up).
    #[tokio::test]
    async fn outbox_capture_round_trips_into_parent_inbox() {
        let ws = unique_dir("ws");
        let cas = unique_dir("cas");
        let roots = Arc::new(WorkspaceRoots::new(ws.clone()));
        let blobs: Arc<dyn BlobStore> =
            Arc::new(FileBlobStore::open(cas.clone()).expect("open blob store"));
        let inc = incarnation_with_content(blobs.clone(), roots.clone());

        let child = SessionId::new("parent/c1");
        let parent = SessionId::new("parent");

        // The child writes an artifact into its outbox/.
        let outbox = roots.session_root(child.as_str()).join("outbox");
        std::fs::create_dir_all(&outbox).unwrap();
        std::fs::write(outbox.join("report.txt"), b"final report").unwrap();

        // Capture: the outbox is folded into a DelegationResult referencing the stored blob.
        let payload = inc
            .capture_outbox(&child)
            .await
            .expect("a non-empty outbox yields a structured payload");
        let result = daemon_protocol::DelegationResult::decode(&payload);
        assert_eq!(result.artifacts.len(), 1);
        assert_eq!(result.artifacts[0].name.as_deref(), Some("report.txt"));

        // Materialize: the parent's inbox/ receives the artifact bytes fetched from the store.
        inc.materialize_artifacts(&parent, &payload).await;
        let landed = roots.session_root(parent.as_str()).join("inbox/report.txt");
        assert_eq!(std::fs::read(&landed).unwrap(), b"final report");

        // An empty outbox captures nothing (the store falls back to the legacy marker).
        let empty_child = SessionId::new("parent/c2");
        assert!(inc.capture_outbox(&empty_child).await.is_none());

        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&cas);
    }

    /// The durable inbound-input seam: pending inputs enqueued on the store
    /// (`enqueue_session_input`) are drained at hydrate — decoded as `UserMsg`s and appended to the
    /// conversation in FIFO order, before the turn runs — and the store side is emptied (a second
    /// hydrate sees nothing). An undecodable payload is dropped without failing the activation.
    #[tokio::test]
    async fn hydrate_drains_pending_session_inputs_into_conversation() {
        use daemon_store::{InMemoryStore, SessionStore};

        let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
        let session = SessionId::new("notify-target");
        let snapshot = daemon_core::Snapshot::fresh(session.clone());
        store
            .create_session(session.clone(), daemon_common::PartitionId::DEFAULT, {
                snapshot.encode().unwrap()
            })
            .await
            .unwrap();

        // Two well-formed inputs (FIFO) plus one undecodable payload (dropped with a warning).
        let encode = |text: &str| {
            let mut buf = Vec::new();
            ciborium::into_writer(&daemon_protocol::UserMsg::new(text), &mut buf).unwrap();
            buf
        };
        store
            .enqueue_session_input(&session, encode("[proc done] first"))
            .await;
        store
            .enqueue_session_input(&session, b"not-cbor".to_vec())
            .await;
        store
            .enqueue_session_input(&session, encode("[proc done] second"))
            .await;

        let profile = EngineProfile::new(
            Arc::new(|| Arc::new(MockProvider::completing("ok")) as Arc<dyn Provider>),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("test"),
        );
        let factory = CoreEngineFactory::from_profile(profile)
            .with_journal(store.clone(), Arc::new(TraceSigner::generate()));
        let mut inc = factory.create();
        let blob = store.peek_snapshot(&session).await.unwrap();
        inc.hydrate(blob, Vec::new()).await.unwrap();

        // The checkpointed conversation carries both decoded inputs, in enqueue order.
        let snap = Snapshot::decode(&inc.checkpoint().unwrap()).unwrap();
        let users: Vec<String> = snap
            .conversation
            .turns
            .iter()
            .filter_map(|t| match t {
                daemon_core::Turn::User(msg) => Some(msg.text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            users,
            vec![
                "[proc done] first".to_string(),
                "[proc done] second".to_string()
            ]
        );
        // Drained: a second hydrate sees no pending inputs.
        assert!(store.take_session_inputs(&session).await.is_empty());
    }
}
