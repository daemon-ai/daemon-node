// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The in-process §17 session backend and the `EngineUnit` constructor.
//!
//! [`LiveAgentSession`] is the [`AgentSession`] backed by a `daemon-core` engine running on the
//! in-process actor ([`spawn_agent_session`]). [`EngineUnit`] wires it into the shared
//! [`AgentUnit`] adapter, so a `daemon-core` engine is presented to its supervisor as a
//! `UnitKind::Engine` [`ManagedUnit`](daemon_supervision::ManagedUnit) — identically to a foreign
//! brain ([`crate::process_agent::ProcessAgentUnit`]). The §17 ⇄ management translation lives in
//! [`crate::agent_session`]; this module only supplies the in-process transport.

use crate::agent_session::{AgentSession, AgentUnit};
use crate::journal::JournalFeeder;
use async_trait::async_trait;
use daemon_common::{SessionId, UnitId};
use daemon_core::{spawn_agent_session, AgentHandle, CheckpointStore, Engine};
use daemon_protocol::{AgentCommand, AgentEvent, HostRequestHandler};
use daemon_store::SessionStore;
use std::sync::Arc;
use tokio::sync::broadcast;

/// The durable-side handles a managed engine needs to apply a conversation rewind's seal + workspace
/// rollback (conversation-rewind spec §6) — the exact side-effects the live-session path applies.
/// Threaded into [`LiveAgentSession`] so the managed/fleet path no longer diverges from the live one
/// (it previously skipped both). `None` (e.g. a node without a journal/checkpoint store) leaves the
/// managed rewind as an engine-only truncate, matching the live path's behavior under the same node.
pub struct RewindHooks {
    /// The durable session store the journal seal is recorded against.
    pub store: Arc<dyn SessionStore>,
    /// The §12 tool-checkpoint store, when wired (drives the workspace rollback).
    pub checkpoints: Option<Arc<dyn CheckpointStore>>,
    /// Whether this unit journals its transcript (a seal is only meaningful when journaled).
    pub journaled: bool,
    /// The session id (journal stream + checkpoint scope) of this managed engine.
    pub session: SessionId,
}

/// A [`AgentSession`] over a `daemon-core` engine on the in-process actor.
struct LiveAgentSession {
    handle: AgentHandle,
    /// The durable seal/rollback handles, when this managed engine should apply them on rewind.
    rewind: Option<RewindHooks>,
}

#[async_trait]
impl AgentSession for LiveAgentSession {
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
            // Context-only append (no turn): folds in when idle, lands in the following turn if busy.
            AgentCommand::Observe { input, request_id } => {
                self.handle.observe(request_id, input).await
            }
            AgentCommand::Snapshot { request_id } => self.handle.snapshot(request_id).await,
            AgentCommand::Interrupt { reason } => self.handle.interrupt(reason).await,
            // Conversation rewind on the managed daemon-core path: truncate + reconstruct + emit
            // `Rewound`, then apply the *same* durable seal + workspace rollback the live path does,
            // via the shared `apply_rewind_side_effects` helper (so the two paths stay consistent).
            AgentCommand::RewindTo { anchor, request_id } => {
                if let Ok(outcome) = self.handle.rewind_to(request_id, anchor).await {
                    if let Some(hooks) = &self.rewind {
                        crate::node_api::apply_rewind_side_effects(
                            crate::node_api::RewindSideEffects {
                                store: &hooks.store,
                                checkpoints: hooks.checkpoints.as_ref(),
                                journaled: hooks.journaled,
                                session: &hooks.session,
                                outcome: &outcome,
                                restore_workspace: true,
                            },
                        )
                        .await;
                    }
                }
            }
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
    pub fn spawn_journaled(
        id: UnitId,
        engine: Engine,
        journal: Option<Arc<JournalFeeder>>,
    ) -> AgentUnit {
        Self::spawn_rewindable(id, engine, journal, None)
    }

    /// As [`Self::spawn_journaled`], but threads in the durable [`RewindHooks`] so a conversation
    /// rewind on this managed engine applies the same journal seal + workspace rollback the live
    /// path applies (conversation-rewind spec §6). `None` keeps the engine-only truncate.
    pub fn spawn_rewindable(
        id: UnitId,
        engine: Engine,
        journal: Option<Arc<JournalFeeder>>,
        rewind: Option<RewindHooks>,
    ) -> AgentUnit {
        AgentUnit::start_journaled(id, journal, move |host: Arc<dyn HostRequestHandler>| {
            Arc::new(LiveAgentSession {
                handle: spawn_agent_session(engine, host),
                rewind,
            }) as Arc<dyn AgentSession>
        })
    }
}
