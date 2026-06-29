// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The profile-driven placement seam: materialize each fleet child as the configured
//! [`AgentBackend`] (in-process reference engine or a launched foreign agent), uniformly presented
//! up the tree as a `ManagedUnit`.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use daemon_common::{JournalStreamId, SessionId, UnitId};
use daemon_core::EngineProfile;
use daemon_host::{
    AgentSession, AgentUnit, CodecSession, EngineUnit, JournalConfig, JournalFeeder, JournalSink,
    ProcessAgentUnit, StreamJsonCodec,
};
use daemon_orchestration::ChildSpawner;
use daemon_protocol::HostRequestHandler;
use daemon_provision::{PlacementSpec, ProcessProvisioner, Provisioner};
use daemon_supervision::{DelegationSpec, ManagedUnit};

/// A foreign agent launch profile: how to start a non-`daemon-core` brain that speaks §17 over a
/// process cut (mirrors [`daemon_provision::PlacementSpec`]). The reference brain needs none of this;
/// it is the home for "manage the foreign process's environment" the way `EngineProfile` is for ours.
pub struct LaunchProfile {
    /// The program to exec.
    pub program: PathBuf,
    /// Its CLI arguments.
    pub args: Vec<String>,
    /// Environment overrides applied to the child.
    pub env: Vec<(String, String)>,
    /// Which foreign wire protocol the agent speaks (selects the transport + codec / adapter).
    pub protocol: ForeignProtocol,
}

/// The wire protocol a foreign agent speaks — the selector that decides which transport + codec (or
/// out-of-tree adapter) materializes the child. All three present up the tree as a
/// `UnitKind::Engine` `ManagedUnit` and journal identically; only the bytes on the cut differ.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ForeignProtocol {
    /// The native `daemon` cut: CBOR §17 frames over the length-framed transport (our own placed
    /// `daemon-core` children, or any brain that speaks the native dialect).
    #[default]
    NativeCut,
    /// Claude-Code `stream-json`: NDJSON event envelope over the line transport (also Amp, Cursor).
    StreamJson,
    /// Agent Client Protocol: symmetric JSON-RPC 2.0 over stdio, via the `daemon-acp` adapter.
    Acp,
}

/// How to construct a child brain. `Core` is the in-process reference engine; `Foreign` launches an
/// external agent process. Both are presented up the tree as a `UnitKind::Engine` `ManagedUnit`, so
/// the fleet/orchestrator (and the GUI above it) cannot tell them apart.
// Built once per child at placement time, not stored in bulk - the variant size delta is irrelevant
// here, and boxing would leak into this pub enum's construction/match sites for no real benefit.
#[allow(clippy::large_enum_variant)]
pub enum AgentBackend {
    /// The in-process reference engine, built from a shared [`EngineProfile`].
    Core(EngineProfile),
    /// An external agent process launched from a [`LaunchProfile`].
    Foreign(LaunchProfile),
}

/// The profile-driven placement seam: materialize each child as the configured [`AgentBackend`],
/// uniformly presented as a `ManagedUnit`.
pub struct ProfileChildSpawner {
    backend: AgentBackend,
    provisioner: Arc<dyn Provisioner>,
    /// The verifiable-journal store + signer; when set, each spawned child journals its transcript
    /// (finished blocks + lifecycle) sealed per turn into the shared store, keyed by its `UnitId`.
    journal: Option<JournalConfig>,
    /// The durable session store + §12 checkpoint store, threaded into a `Core` child's managed
    /// engine so a conversation rewind on it applies the same journal seal + workspace rollback the
    /// live-session path applies (conversation-rewind spec §6). `None` keeps the engine-only truncate.
    rewind_store: Option<Arc<dyn daemon_store::SessionStore>>,
    rewind_checkpoints: Option<Arc<dyn daemon_core::CheckpointStore>>,
}

impl ProfileChildSpawner {
    /// A spawner that materializes children from the in-process reference engine profile.
    pub fn core(profile: EngineProfile) -> Self {
        Self {
            backend: AgentBackend::Core(profile),
            provisioner: Arc::new(ProcessProvisioner::new()),
            journal: None,
            rewind_store: None,
            rewind_checkpoints: None,
        }
    }

    /// A spawner that materializes children by launching a foreign agent process.
    pub fn foreign(launch: LaunchProfile) -> Self {
        Self {
            backend: AgentBackend::Foreign(launch),
            provisioner: Arc::new(ProcessProvisioner::new()),
            journal: None,
            rewind_store: None,
            rewind_checkpoints: None,
        }
    }

    /// Thread the durable seal + workspace-rollback handles into the spawned `Core` children so a
    /// conversation rewind on a managed engine matches the live path (conversation-rewind spec §6).
    pub fn with_rewind(
        mut self,
        store: Arc<dyn daemon_store::SessionStore>,
        checkpoints: Option<Arc<dyn daemon_core::CheckpointStore>>,
    ) -> Self {
        self.rewind_store = Some(store);
        self.rewind_checkpoints = checkpoints;
        self
    }

    /// Journal every spawned child into the unified verifiable journal (keyed by `UnitId`).
    pub fn with_journal(mut self, journal: JournalConfig) -> Self {
        self.journal = Some(journal);
        self
    }

    /// Build a per-child journal feeder keyed by `id`, when journaling is configured.
    fn feeder(&self, id: &UnitId) -> Option<Arc<JournalFeeder>> {
        self.journal.as_ref().map(|cfg| {
            let sink = JournalSink::new(
                cfg.store.clone(),
                cfg.signer.clone(),
                JournalStreamId::unit(id),
            );
            Arc::new(JournalFeeder::new(Arc::new(sink)))
        })
    }
}

#[async_trait]
impl ChildSpawner for ProfileChildSpawner {
    async fn spawn(&self, id: UnitId, _spec: &DelegationSpec) -> Arc<dyn ManagedUnit> {
        let feeder = self.feeder(&id);
        match &self.backend {
            AgentBackend::Core(profile) => {
                let session = SessionId::new(id.as_str());
                let engine = profile.fresh(session.clone());
                // Thread the durable seal/rollback handles into the managed engine so a rewind on it
                // matches the live path (the §17⇄management seam fix); `None` store => engine-only.
                let rewind = self
                    .rewind_store
                    .clone()
                    .map(|store| daemon_host::RewindHooks {
                        store,
                        checkpoints: self.rewind_checkpoints.clone(),
                        journaled: feeder.is_some(),
                        session,
                    });
                Arc::new(EngineUnit::spawn_rewindable(id, engine, feeder, rewind))
            }
            AgentBackend::Foreign(launch) => {
                let session = SessionId::new(id.as_str());
                let spec = PlacementSpec {
                    program: launch.program.clone(),
                    args: launch.args.clone(),
                    env: launch.env.clone(),
                };
                match launch.protocol {
                    ForeignProtocol::NativeCut => {
                        let placement = self
                            .provisioner
                            .place(&session, spec)
                            .await
                            .expect("place native-cut foreign agent");
                        Arc::new(ProcessAgentUnit::start_journaled(id, placement, feeder))
                    }
                    ForeignProtocol::StreamJson => {
                        // NDJSON over the line transport, driven by the generic codec session driver.
                        let placement = self
                            .provisioner
                            .place_lines(&session, spec)
                            .await
                            .expect("place stream-json foreign agent");
                        let daemon_provision::Placement { channel, child } = placement;
                        Arc::new(AgentUnit::start_journaled(
                            id,
                            feeder,
                            move |host: Arc<dyn HostRequestHandler>| {
                                Arc::new(CodecSession::from_channel(
                                    channel,
                                    Some(child),
                                    host,
                                    StreamJsonCodec::new(),
                                )) as Arc<dyn AgentSession>
                            },
                        ))
                    }
                    ForeignProtocol::Acp => {
                        // The ACP adapter owns its own subprocess + stdio (it does not use the cut).
                        let acp = daemon_acp::AcpLaunch::new(launch.program.clone())
                            .args(launch.args.clone())
                            .env(launch.env.clone());
                        Arc::new(daemon_acp::acp_unit(id, acp, feeder))
                    }
                }
            }
        }
    }
}
