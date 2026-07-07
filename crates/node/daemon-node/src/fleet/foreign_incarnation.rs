// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The durable activation path for a delegated child whose bound profile is a foreign engine.
//!
//! A delegated child suspends its parent, materializes as a durable child session, and is driven by
//! the shared [`daemon_activation::ActivationManager`] through the protocol-agnostic
//! [`Incarnation`] seam. For a `Core` bound profile that is `daemon_host::CoreIncarnation`; this
//! module adds the `Foreign{agent}` counterpart:
//!
//! - [`ForeignIncarnation`] spawns the profile's ACP / stream-json backend (via the same
//!   [`spawn_foreign_session`](crate::fleet::foreign_live::spawn_foreign_session) helper the live
//!   session builder uses), drains the durable task input, runs the turn to completion, journals the
//!   transcript, and returns a `DelegationResult` summary so the parent's join is fulfilled exactly
//!   as a Core child's is.
//! - [`DispatchingEngineFactory`] is the node's durable [`EngineFactory`]: each incarnation inspects
//!   its session's bound profile at hydrate and routes `Core` -> `CoreIncarnation` /
//!   `Foreign{agent}` -> [`ForeignIncarnation`], so a foreign-profile child no longer silently falls
//!   back to Core.
//!
//! Foreign conversation state lives in the agent process, not in the durable CBOR snapshot, so the
//! checkpoint is a minimal id-carrying snapshot and each activation spawns a fresh backend.

use std::sync::Arc;

use async_trait::async_trait;
use daemon_activation::{EngineError, EngineFactory, Incarnation, SnapshotBlob, Step};
use daemon_api::{EngineSelector, ForeignBackend};
use daemon_common::{Epoch, JobId, JournalStreamId, ReqId, SessionId};
use daemon_core::Snapshot;
use daemon_host::{AgentSession, CoreEngineFactory, JournalFeeder, JournalSink, ProfileStore};
use daemon_protocol::{
    AgentCommand, AgentEvent, DelegationResult, HostRequest, HostRequestHandler, HostRequestKind,
    HostResponse, HostResponseBody, Outbound, UserMsg,
};
use daemon_store::SessionStore;
use daemon_telemetry::{TraceSigner, GENESIS_ROOT};
use tokio::sync::broadcast;

use crate::fleet::foreign_live::spawn_foreign_session;
use crate::GatewayCoords;

/// The node subsystems a durable foreign incarnation needs (beyond the Core factory): the profile
/// store (to resolve the child's bound `Foreign{agent}` spec), the session store (durable inputs +
/// usage + journal seal), the journal signer, and the optional node gateway (NodeProvider routing).
#[derive(Clone)]
pub(crate) struct ForeignConfig {
    /// Resolves a bound profile name to its [`daemon_api::ProfileSpec`] (engine + foreign backend).
    pub(crate) profiles: Arc<dyn ProfileStore>,
    /// The durable store: read `session_meta`/`take_session_inputs`, seal the journal, fold usage.
    pub(crate) store: Arc<dyn SessionStore>,
    /// The node's verifiable-journal signer (the durable/live/fleet paths share this key).
    pub(crate) signer: Arc<TraceSigner>,
    /// The node gateway coordinates for a `NodeProvider`-routed foreign backend; `None` disables it.
    pub(crate) gateway: Option<GatewayCoords>,
}

/// The node's durable [`EngineFactory`]: builds a [`DispatchingIncarnation`] that resolves Core vs
/// Foreign per session at hydrate, so one factory drives both engine kinds on the durable path.
pub(crate) struct DispatchingEngineFactory {
    core: CoreEngineFactory,
    foreign: ForeignConfig,
}

impl DispatchingEngineFactory {
    /// A factory that dispatches to `core` (the fully-wired [`CoreEngineFactory`]) for Core-profile
    /// sessions and to a [`ForeignIncarnation`] (over `foreign`) for `Foreign{agent}`-profile ones.
    pub(crate) fn new(core: CoreEngineFactory, foreign: ForeignConfig) -> Self {
        Self { core, foreign }
    }
}

impl EngineFactory for DispatchingEngineFactory {
    fn create(&self) -> Box<dyn Incarnation> {
        Box::new(DispatchingIncarnation {
            core: self.core.clone(),
            foreign: self.foreign.clone(),
            inner: None,
        })
    }
}

/// One durable incarnation that decides its concrete backend at hydrate: a session whose bound
/// profile is `Foreign{agent}` is driven by a [`ForeignIncarnation`], every other session by the
/// factory's [`CoreEngineFactory`] incarnation (the unchanged Core path). All seam methods delegate
/// to the resolved inner incarnation.
struct DispatchingIncarnation {
    core: CoreEngineFactory,
    foreign: ForeignConfig,
    inner: Option<Box<dyn Incarnation>>,
}

impl DispatchingIncarnation {
    /// Resolve the session's foreign binding from `snapshot`: decode the session id, resolve its
    /// effective spec (an INLINE sub-agent spec from `SessionMeta.inline_profile` takes precedence
    /// over a bound profile name — Phase 1), and — when that spec's engine is `Foreign{agent}` —
    /// return the id + agent name + foreign backend. `None` means "run this session on the Core
    /// path" (no binding/inline, an unknown profile, or a Core-engine spec). `resolve_effective` on
    /// the Core resolver therefore never sees a foreign spec. Static over `&ForeignConfig` (not
    /// `&self`) so the future stays `Send` (`self` also holds a non-`Sync` `Box<dyn Incarnation>`).
    async fn resolve_foreign(
        foreign: &ForeignConfig,
        snapshot: &SnapshotBlob,
    ) -> Option<(SessionId, String, ForeignBackend)> {
        let session_id = Snapshot::decode(snapshot).ok()?.session_id;
        let meta = foreign.store.session_meta(&session_id).await?;
        let spec = if !meta.inline_profile.is_empty() {
            daemon_api::from_cbor::<daemon_api::ProfileSpec>(&meta.inline_profile).ok()?
        } else {
            let bound = meta.bound_profile?;
            foreign.profiles.get(bound.as_str()).ok().flatten()?
        };
        match spec.engine {
            EngineSelector::Foreign { agent } => Some((session_id, agent, spec.foreign_backend)),
            EngineSelector::Core => None,
        }
    }
}

#[async_trait]
impl Incarnation for DispatchingIncarnation {
    async fn hydrate(
        &mut self,
        snapshot: SnapshotBlob,
        unapplied: Vec<daemon_store::JobCompletion>,
    ) -> Result<(), EngineError> {
        let foreign = self.foreign.clone();
        let mut inner: Box<dyn Incarnation> = match Self::resolve_foreign(&foreign, &snapshot).await
        {
            Some((session_id, agent, backend)) => Box::new(ForeignIncarnation::new(
                session_id,
                agent,
                backend,
                foreign.store.clone(),
                foreign.signer.clone(),
                foreign.gateway.clone(),
            )),
            None => self.core.create(),
        };
        inner.hydrate(snapshot, unapplied).await?;
        self.inner = Some(inner);
        Ok(())
    }

    async fn run(&mut self) -> Result<Step, EngineError> {
        self.inner
            .as_mut()
            .ok_or_else(|| EngineError::Other("dispatching run before hydrate".into()))?
            .run()
            .await
    }

    fn checkpoint(&self) -> Result<SnapshotBlob, EngineError> {
        self.inner
            .as_ref()
            .ok_or_else(|| EngineError::Other("dispatching checkpoint before hydrate".into()))?
            .checkpoint()
    }

    fn epoch(&self) -> Epoch {
        self.inner.as_ref().map(|i| i.epoch()).unwrap_or_default()
    }

    fn completion_payload(&self) -> Option<Vec<u8>> {
        self.inner.as_ref().and_then(|i| i.completion_payload())
    }
}

/// A durable incarnation that runs a delegated child as its bound `Foreign{agent}` backend: spawn
/// the ACP / stream-json session, run the drained task to a terminal turn, and return a
/// `DelegationResult` so the parent's join resolves. Single-activation-to-completion: a foreign
/// child does not suspend (its conversation state lives in the agent process), so `run` always
/// reaches `Step::Completed`.
pub(crate) struct ForeignIncarnation {
    session_id: SessionId,
    agent: String,
    backend: ForeignBackend,
    store: Arc<dyn SessionStore>,
    signer: Arc<TraceSigner>,
    gateway: Option<GatewayCoords>,
    /// The spawned foreign backend, materialized at hydrate.
    session: Option<Arc<dyn AgentSession>>,
    /// The durable pending inputs drained at hydrate (the delegated task + any queued `send`s).
    inputs: Vec<UserMsg>,
    /// The structured completion payload captured at terminal (a CBOR `DelegationResult`).
    completion_payload: Option<Vec<u8>>,
}

impl ForeignIncarnation {
    /// Construct over a resolved foreign binding; hydrate spawns the backend and drains inputs.
    pub(crate) fn new(
        session_id: SessionId,
        agent: String,
        backend: ForeignBackend,
        store: Arc<dyn SessionStore>,
        signer: Arc<TraceSigner>,
        gateway: Option<GatewayCoords>,
    ) -> Self {
        Self {
            session_id,
            agent,
            backend,
            store,
            signer,
            gateway,
            session: None,
            inputs: Vec::new(),
            completion_payload: None,
        }
    }

    /// Seal this activation's transcript into the unified verifiable journal (segment 0: a foreign
    /// child runs a single activation to completion, so there is no epoch chain) and fold its token
    /// usage into the durable per-session usage surface — mirroring the Core incarnation's journaling.
    async fn journal_turn(&self, events: &[AgentEvent]) {
        let mut delta = daemon_common::UsageDelta::default();
        for ev in events {
            if let AgentEvent::Usage { delta: d, .. } = ev {
                delta.add(d);
            }
        }
        if delta != daemon_common::UsageDelta::default() {
            self.store.record_usage(&self.session_id, delta).await;
        }
        let stream = JournalStreamId::session(&self.session_id);
        let jsink = Arc::new(JournalSink::with_segment(
            self.store.clone(),
            self.signer.clone(),
            stream,
            None,
            0,
            GENESIS_ROOT,
        ));
        let feeder = JournalFeeder::new(jsink);
        for ev in events {
            feeder.feed(&Outbound::Event(ev.clone())).await;
        }
    }
}

#[async_trait]
impl Incarnation for ForeignIncarnation {
    async fn hydrate(
        &mut self,
        _snapshot: SnapshotBlob,
        _unapplied: Vec<daemon_store::JobCompletion>,
    ) -> Result<(), EngineError> {
        // The agent answers its own blocking §17 requests (ACP permission prompts) against an
        // auto-allow host: a durable child runs headless with no operator to prompt.
        let host: Arc<dyn HostRequestHandler> = Arc::new(AutoAllowHost {
            session_id: self.session_id.clone(),
        });
        let session = spawn_foreign_session(
            self.agent.clone(),
            self.backend.clone(),
            None,
            self.session_id.clone(),
            self.store.clone(),
            self.gateway.clone(),
            host,
        )
        .await
        .map_err(|e| EngineError::Other(format!("spawn foreign session `{}`: {e}", self.agent)))?;
        // Drain the durable pending inputs (the delegated task seeded by the job worker, plus any
        // `send`s queued while dehydrated) into this activation's turn queue; each decodes as a
        // `UserMsg` (bare text falls back).
        self.inputs = self
            .store
            .take_session_inputs(&self.session_id)
            .await
            .iter()
            .map(|raw| UserMsg::decode(raw))
            .collect();
        self.session = Some(session);
        Ok(())
    }

    async fn run(&mut self) -> Result<Step, EngineError> {
        let session = self
            .session
            .clone()
            .ok_or_else(|| EngineError::Other("foreign run before hydrate".into()))?;
        // At least one turn: the drained task(s), or an empty prompt if the queue was empty.
        let inputs = if self.inputs.is_empty() {
            vec![UserMsg::new(String::new())]
        } else {
            std::mem::take(&mut self.inputs)
        };
        let mut captured: Vec<AgentEvent> = Vec::new();
        let mut final_text = String::new();
        for input in inputs {
            // Subscribe before submitting so no event between submit and the first recv is missed.
            let mut rx = session.subscribe();
            session
                .submit(AgentCommand::StartTurn {
                    input,
                    request_id: ReqId(0),
                })
                .await;
            loop {
                match rx.recv().await {
                    Ok(AgentEvent::TurnFinished { seq, summary }) => {
                        if let Some(text) = &summary.final_text {
                            final_text = text.clone();
                        }
                        captured.push(AgentEvent::TurnFinished { seq, summary });
                        break;
                    }
                    Ok(ev) => captured.push(ev),
                    Err(broadcast::error::RecvError::Closed) => break,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                }
            }
        }
        self.journal_turn(&captured).await;
        // The structured result the parent materializes on its wake: a foreign child produces no
        // node-captured `outbox/` artifacts (its work lives in the agent process), so the payload is
        // the agent's final summary text.
        let summary = if final_text.trim().is_empty() {
            format!("foreign agent `{}` completed", self.agent)
        } else {
            final_text
        };
        self.completion_payload = Some(
            DelegationResult {
                summary,
                artifacts: Vec::new(),
            }
            .encode(),
        );
        Ok(Step::Completed)
    }

    fn checkpoint(&self) -> Result<SnapshotBlob, EngineError> {
        // A minimal id-carrying snapshot: the foreign conversation state lives in the agent process,
        // not in CBOR, so each activation spawns a fresh backend from this.
        Ok(Snapshot::fresh(self.session_id.clone()).encode()?)
    }

    fn epoch(&self) -> Epoch {
        // A foreign child completes in one activation with no suspension bump.
        Epoch::default()
    }

    fn completion_payload(&self) -> Option<Vec<u8>> {
        self.completion_payload.clone()
    }
}

/// A headless host handler for a durable foreign incarnation: it auto-allows the agent's blocking
/// §17 requests (ACP `session/request_permission` -> `Approval`) so a delegated child runs to
/// completion without an operator, and answers the other kinds trivially.
struct AutoAllowHost {
    session_id: SessionId,
}

#[async_trait]
impl HostRequestHandler for AutoAllowHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
        let body = match req.kind {
            HostRequestKind::Delegate { .. } => {
                HostResponseBody::Delegated(JobId::new(format!("{}:noop", self.session_id)))
            }
            HostRequestKind::Input { .. } => HostResponseBody::Input(String::new()),
            HostRequestKind::Choice { .. } => HostResponseBody::Chosen(0),
            _ => HostResponseBody::Approved {
                approved: true,
                allow_permanent: false,
                reason: None,
            },
        };
        HostResponse {
            request_id: req.request_id,
            body,
        }
    }
}
