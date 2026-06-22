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
use crate::journal::{JournalFeeder, JournalSink};
use crate::node_api::{decode_overlay, DurableProfileResolver};
use async_trait::async_trait;
use daemon_activation::{EngineError, EngineFactory, Incarnation, SnapshotBlob, Step};
use daemon_common::{Epoch, JobId, JournalStreamId, ProfileRef, SessionId};
use daemon_core::{
    Completion, Conversation, DelegateTool, Engine, EngineProfile, EventSink, Failure, MockProvider,
    Provider, Snapshot, SystemPrompt, ToolRegistry, TurnControl, TurnOutcome,
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
        }
    }

    /// A factory over an already-assembled [`EngineProfile`] (the binary's composition root).
    pub fn from_profile(profile: EngineProfile) -> Self {
        Self {
            profile,
            journal: None,
            background: None,
            resolver: None,
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
        // A background child (§4.3) hydrates under its constrained review profile (skills-only /
        // memory-only tools + bounded budget + nudges off), not the parent's full profile. Otherwise,
        // when a per-session resolver + journal store are wired, re-resolve this session's profile
        // from its persisted bound profile + overlay (unified resolution: a durable session honors
        // its own model/tools/approval override, restored on rehydration). Falls back to the
        // factory's fixed profile when no binding is recorded (e.g. delegated orchestrator children).
        let profile = if let Some(bg_profile) = self
            .background
            .as_ref()
            .and_then(|bg| bg.profile_for(&snap.session_id))
        {
            bg_profile
        } else if let Some(resolved) = self.resolve_session_profile(&snap.session_id).await {
            resolved
        } else {
            self.profile.clone()
        };
        let mut engine = profile.from_snapshot(snap);
        let completions = unapplied
            .into_iter()
            .map(|c| Completion {
                job_id: c.job_id,
                payload: c.payload,
            })
            .collect();
        engine.apply_completions(completions);
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

        match outcome {
            TurnOutcome::Completed(_) => Ok(Step::Completed),
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
