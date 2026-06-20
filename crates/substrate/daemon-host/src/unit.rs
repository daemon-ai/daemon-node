//! The in-process §17 session backend and the `EngineUnit` constructor.
//!
//! [`LiveSection17`] is the [`Section17Session`] backed by a `daemon-core` engine running on the
//! in-process actor ([`spawn_agent_session`]). [`EngineUnit`] wires it into the shared
//! [`AgentUnit`] adapter, so a `daemon-core` engine is presented to its supervisor as a
//! `UnitKind::Engine` [`ManagedUnit`](daemon_supervision::ManagedUnit) — identically to a foreign
//! brain ([`crate::process_agent::ProcessAgentUnit`]). The §17 ⇄ management translation lives in
//! [`crate::section17`]; this module only supplies the in-process transport.

use crate::journal::JournalFeeder;
use crate::section17::{AgentUnit, Section17Session};
use async_trait::async_trait;
use daemon_common::UnitId;
use daemon_core::{spawn_agent_session, AgentHandle, Engine};
use daemon_protocol::{AgentCommand, AgentEvent, HostRequestHandler};
use std::sync::Arc;
use tokio::sync::broadcast;

/// A [`Section17Session`] over a `daemon-core` engine on the in-process actor.
struct LiveSection17 {
    handle: AgentHandle,
}

#[async_trait]
impl Section17Session for LiveSection17 {
    async fn submit(&self, cmd: AgentCommand) {
        match cmd {
            AgentCommand::StartTurn { input, .. } => {
                // Background the turn so `command` returns promptly and progress streams as events.
                let handle = self.handle.clone();
                tokio::spawn(async move {
                    let _ = handle.start_turn(input).await;
                });
            }
            AgentCommand::Steer { text, request_id } => self.handle.steer(request_id, text).await,
            AgentCommand::Snapshot { request_id } => self.handle.snapshot(request_id).await,
            AgentCommand::Interrupt { reason } => self.handle.interrupt(reason).await,
            AgentCommand::Shutdown => self.handle.shutdown().await,
            _ => {}
        }
    }

    fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.handle.subscribe()
    }
}

/// An engine presented to its supervisor as a `UnitKind::Engine` [`ManagedUnit`] (host-spec §9).
pub struct EngineUnit;

impl EngineUnit {
    /// Spawn an engine session and present it as a managed unit identified by `id`.
    pub fn spawn(id: UnitId, engine: Engine) -> AgentUnit {
        Self::spawn_journaled(id, engine, None)
    }

    /// As [`Self::spawn`], but durably journals the unit's transcript (finished blocks + lifecycle,
    /// sealed per turn) into `journal` when provided — the fleet/live production journaling path.
    pub fn spawn_journaled(id: UnitId, engine: Engine, journal: Option<Arc<JournalFeeder>>) -> AgentUnit {
        AgentUnit::start_journaled(id, journal, |host: Arc<dyn HostRequestHandler>| {
            Arc::new(LiveSection17 {
                handle: spawn_agent_session(engine, host),
            }) as Arc<dyn Section17Session>
        })
    }
}
