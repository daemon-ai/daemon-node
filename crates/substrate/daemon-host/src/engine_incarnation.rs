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

use async_trait::async_trait;
use daemon_activation::{EngineError, EngineFactory, Incarnation, SnapshotBlob, Step};
use daemon_common::{Epoch, JobId, ProfileRef, SessionId};
use daemon_core::{
    Completion, CredentialProvider, DelegateTool, Engine, EventSink, Failure, MockProvider,
    Provider, Snapshot, SystemPrompt, ToolRegistry, TurnOutcome,
};
use daemon_protocol::{
    HostRequest, HostRequestHandler, HostRequestKind, HostResponse, HostResponseBody,
};
use daemon_store::{JobCommand, JobCompletion};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// A builder for the model [`Provider`] each incarnation drives (a fresh provider per activation).
pub type ProviderBuilder = Arc<dyn Fn() -> Arc<dyn Provider> + Send + Sync>;

/// A builder for the [`CredentialProvider`] each incarnation acquires capabilities from. A fresh
/// handle per activation lets a brokered client be re-bound to the live cut (host-spec §6).
pub type CredentialBuilder = Arc<dyn Fn() -> Arc<dyn CredentialProvider> + Send + Sync>;

/// Builds core-backed [`Incarnation`]s — the phase-3 replacement for the retired stub engine.
#[derive(Clone)]
pub struct CoreEngineFactory {
    provider: ProviderBuilder,
    registry: Arc<ToolRegistry>,
    system: SystemPrompt,
    /// The credential provider + profile injected into each engine; `None` keeps the engine's
    /// embedded L1 default.
    credentials: Option<(CredentialBuilder, ProfileRef)>,
}

impl CoreEngineFactory {
    /// A factory whose engines delegate one unit of background work and then complete — the durable
    /// "delegate → suspend → resume → complete" cycle the substrate conformance suite drives.
    pub fn delegating() -> Self {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(DelegateTool::new("background-work")));
        Self {
            provider: Arc::new(|| {
                Arc::new(MockProvider::delegating("delegate", "work complete")) as Arc<dyn Provider>
            }),
            registry: Arc::new(registry),
            system: SystemPrompt::new("daemon-core conformance engine"),
            credentials: None,
        }
    }

    /// A factory over a custom provider builder, tool registry, and system prompt.
    pub fn with_provider(
        provider: ProviderBuilder,
        registry: Arc<ToolRegistry>,
        system: SystemPrompt,
    ) -> Self {
        Self {
            provider,
            registry,
            system,
            credentials: None,
        }
    }

    /// Inject an authority-backed (or brokered) credential provider + profile into every engine
    /// this factory builds — the host bridge for the §7 port (host-spec §6).
    pub fn with_credentials(mut self, credentials: CredentialBuilder, profile: ProfileRef) -> Self {
        self.credentials = Some((credentials, profile));
        self
    }
}

impl EngineFactory for CoreEngineFactory {
    fn create(&self) -> Box<dyn Incarnation> {
        Box::new(CoreIncarnation {
            provider: self.provider.clone(),
            registry: self.registry.clone(),
            system: self.system.clone(),
            credentials: self.credentials.clone(),
            engine: None,
        })
    }
}

/// One core-backed engine incarnation driven through the activation seam.
pub struct CoreIncarnation {
    provider: ProviderBuilder,
    registry: Arc<ToolRegistry>,
    #[allow(dead_code)]
    system: SystemPrompt,
    credentials: Option<(CredentialBuilder, ProfileRef)>,
    engine: Option<Engine>,
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
        let provider = (self.provider)();
        let mut engine = Engine::from_snapshot(snap, provider, self.registry.clone());
        if let Some((build, profile)) = &self.credentials {
            engine = engine.with_credentials(build(), profile.clone());
        }
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
        let host = DelegateResolver {
            session_id: session_id.clone(),
            epoch: engine.epoch(),
        };
        let sink = EventSink::discarding();
        let cancel = CancellationToken::new();
        match engine
            .run_turn(&host, &sink, cancel)
            .await
            .map_err(map_failure)?
        {
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
