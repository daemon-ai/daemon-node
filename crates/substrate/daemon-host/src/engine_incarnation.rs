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

use crate::journal::{JournalFeeder, JournalSink};
use async_trait::async_trait;
use daemon_activation::{EngineError, EngineFactory, Incarnation, SnapshotBlob, Step};
use daemon_common::{Epoch, JobId, JournalStreamId, ProfileRef, SessionId};
use daemon_core::{
    Completion, DelegateTool, Engine, EngineProfile, EventSink, Failure, MockProvider, Provider,
    Snapshot, SystemPrompt, ToolRegistry, TurnControl, TurnOutcome,
};
use daemon_protocol::{
    HostRequest, HostRequestHandler, HostRequestKind, HostResponse, HostResponseBody, Outbound,
};
use daemon_store::{JobCommand, JobCompletion, SessionStore};
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
        }
    }

    /// A factory over an already-assembled [`EngineProfile`] (the binary's composition root).
    pub fn from_profile(profile: EngineProfile) -> Self {
        Self {
            profile,
            journal: None,
        }
    }

    /// Inject the verifiable-journal store + signer so every durable incarnation this factory builds
    /// seals its turn into the unified journal (the durable production journaling path).
    pub fn with_journal(mut self, store: Arc<dyn SessionStore>, signer: Arc<TraceSigner>) -> Self {
        self.journal = Some(JournalConfig { store, signer });
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
        })
    }
}

/// One core-backed engine incarnation driven through the activation seam.
pub struct CoreIncarnation {
    profile: EngineProfile,
    engine: Option<Engine>,
    journal: Option<JournalConfig>,
}

fn map_failure(failure: Failure) -> EngineError {
    EngineError::Other(failure.to_string())
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
        let mut engine = self.profile.from_snapshot(snap);
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
        let host = DelegateResolver {
            session_id: session_id.clone(),
            epoch: engine.epoch(),
        };
        // When journaling, capture the engine's events so they can be coalesced into finished blocks
        // and sealed after the turn, and so the turn's token usage can be folded into the durable
        // per-session usage surface (the tree projection's usage source); otherwise discard.
        let captured: Arc<Mutex<Vec<daemon_protocol::AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
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
/// activation outbox dedupes on, and trivially answers the other §17 request kinds.
struct DelegateResolver {
    session_id: SessionId,
    epoch: Epoch,
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
            HostRequestKind::Approval { .. } => HostResponseBody::Approved(true),
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
