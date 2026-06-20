//! `daemon-node` — the single host-composition root.
//!
//! Phases 1-11 grew the node's wiring (durable store + resident services, the orchestration fleet as
//! the real job worker, the credential broker, and the live session surface) inline in `bins/daemon`,
//! with a near-identical copy in the conformance harness. [`assemble`] collapses that into one place:
//! both the binary and the gate build their node through it, so there is exactly one composition to
//! keep correct. It lives above `daemon-host` because the fleet + orchestrate-tool glue is
//! composition-layer policy — `daemon-host` deliberately does not depend on `daemon-orchestration`.
//!
//! Callers supply only *policy*: the store, the [`ProviderRegistry`] (provider selection seam),
//! optional brokered credentials, the session/credential [`ProfileRef`], and the engine
//! [`Config`](daemon_core::Config). [`assemble`] does the standard plumbing (three role
//! `EngineProfile`s, the fleet, the durable factory, the host, and the [`NodeApiImpl`]).

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::FleetReport;
use daemon_common::{JournalStreamId, PartitionId, ProfileRef, SessionId, UnitId};
use daemon_core::{
    Config, CredentialBuilder, EngineProfile, ProviderRegistry, SystemPrompt, ToolRegistry,
};
use daemon_host::{
    AgentUnit, CodecSection17, CoreEngineFactory, EngineUnit, FleetControl, Host, HostConfig,
    JobWorker, JournalConfig, JournalFeeder, JournalSink, NodeApiImpl, ProcessAgentUnit,
    Section17Session, ServiceError, SessionEngineBuilder, StreamJsonCodec, SupervisorHandle,
};
use daemon_protocol::HostRequestHandler;
use daemon_telemetry::TraceSigner;
use daemon_orchestration::{ChildSpawner, DefaultAnswerPolicy, FleetRuntime, OrchestratorSpawner};
use daemon_provision::{PlacementSpec, ProcessProvisioner, Provisioner};
use daemon_supervision::{DelegationSpec, ManageRequestHandler, ManagedUnit};

/// The provider-registry profile name the orchestrator (parent) engine resolves to.
const ORCHESTRATOR_PROFILE: &str = "orchestrator";
/// The provider-registry profile name the fleet-child engine resolves to.
const CHILD_PROFILE: &str = "child";
/// The id of the node's synthetic tree root (the top fleet as the GUI's root node).
const NODE_ROOT: &str = "node";

/// The policy inputs for [`assemble`]: everything that varies between a production node and a test
/// node. The standard plumbing (role profiles, fleet, factory, host, session surface) is derived.
pub struct NodeAssembly {
    /// The durable store backend (shared by the host, fleet, and control surface).
    pub store: Arc<dyn daemon_store::SessionStore>,
    /// The partition this node owns.
    pub partition: PartitionId,
    /// Resident-service cadence + supervision policy.
    pub host_config: HostConfig,
    /// The provider *selection* seam: the orchestrator/child engines resolve `"orchestrator"`/
    /// `"child"`, the session engine resolves `profile` (falling back to the registry default).
    pub providers: ProviderRegistry,
    /// The brokered credential builder applied uniformly to every engine (durable, live, child);
    /// `None` leaves engines on their embedded L1 pool (tests).
    pub credentials: Option<CredentialBuilder>,
    /// The session + credential profile name.
    pub profile: ProfileRef,
    /// The engine tunables (§20) every engine this node builds runs under.
    pub engine_config: Config,
    /// The 32-byte seed for the node's verifiable-journal signer, so its verifying key is stable
    /// across restarts (auditors keep verifying old segments). `None` generates an ephemeral key
    /// (fine for tests; a fresh key each boot otherwise).
    pub journal_seed: Option<[u8; 32]>,
    /// How many orchestrator levels the top fleet materializes before its leaves. `0` (default) is a
    /// flat fleet of engine leaves; `1` makes every top child an orchestrator owning a sub-fleet of
    /// leaves (fleets-of-fleets), `n` nests `n` deep — the tree the GUI projects and addresses.
    pub nesting_depth: usize,
}

/// The assembled node: the bound surface, its started resident-service handle, and the fleet handle.
pub struct AssembledNode {
    /// The one [`daemon_api`] surface (control + session + fleet sub-surfaces).
    pub node: Arc<NodeApiImpl>,
    /// The started resident-service tree; drive shutdown via [`SupervisorHandle::shutdown`].
    pub handle: SupervisorHandle,
    /// The orchestration fleet handle (e.g. for inspection in tests).
    pub fleet: FleetRuntime,
    /// The node's verifiable-journal signer — its verifying key is published so auditors can verify
    /// sealed history (`ControlApi::verifying_key`).
    pub signer: Arc<TraceSigner>,
}

/// Apply the optional brokered credentials to a role profile.
fn dress(profile: EngineProfile, a: &NodeAssembly) -> EngineProfile {
    let profile = profile.with_config(a.engine_config);
    match &a.credentials {
        Some(credentials) => profile.with_credentials(credentials.clone(), a.profile.clone()),
        None => profile,
    }
}

/// Resolve a provider builder for `name`, falling back to the registry default.
fn provider_for(providers: &ProviderRegistry, name: &str) -> daemon_core::ProviderBuilder {
    providers
        .builder_for(&ProfileRef::new(name))
        .unwrap_or_else(|| panic!("no provider registered for {name:?} and no default set"))
}

/// Assemble and start the default host node: durable substrate + resident services, the
/// orchestration fleet as the real job worker, the credential seam, and the live session surface,
/// all built from one shared [`EngineProfile`] per role so the durable, live, and fleet-child paths
/// share provider/credential/tunable policy.
pub fn assemble(a: NodeAssembly) -> AssembledNode {
    // The node's one verifiable-journal signer: every engine path (durable, live, fleet child) seals
    // its per-stream chain with this key, and the control surface publishes the verifying half.
    let signer = Arc::new(
        a.journal_seed
            .map(|seed| TraceSigner::from_seed(&seed))
            .unwrap_or_else(TraceSigner::generate),
    );
    let journal = JournalConfig {
        store: a.store.clone(),
        signer: signer.clone(),
    };

    // The fleet child: one shared profile, driven as the real job worker so every child gets the same
    // provider + brokered credentials. Each child journals into the shared store keyed by its UnitId.
    let child_profile = dress(
        EngineProfile::new(
            provider_for(&a.providers, CHILD_PROFILE),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("fleet child"),
        ),
        &a,
    );
    // The leaf placement seam (in-process engine children). When `nesting_depth > 0` the top fleet
    // spawns orchestrator children instead, each owning a sub-fleet that bottoms out in these leaves
    // — a real, addressable fleets-of-fleets tree.
    let leaf_spawner: Arc<dyn ChildSpawner> =
        Arc::new(ProfileChildSpawner::core(child_profile).with_journal(journal.clone()));
    let spawner: Arc<dyn ChildSpawner> = if a.nesting_depth == 0 {
        leaf_spawner
    } else {
        Arc::new(OrchestratorSpawner::new(
            a.store.clone(),
            a.partition,
            Arc::new(DefaultAnswerPolicy),
            leaf_spawner,
            a.nesting_depth,
        ))
    };
    let fleet = FleetRuntime::new(
        a.store.clone(),
        a.partition,
        spawner,
        Arc::new(DefaultAnswerPolicy),
        None::<Arc<dyn ManageRequestHandler>>,
    )
    // The top fleet projects a rooted tree (the GUI's tree root is the node itself).
    .with_root_id(UnitId::new(NODE_ROOT));

    // The parent orchestrator: an engine that delegates through the orchestrate tool, then completes.
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(daemon_tool_orchestrate::OrchestrateTool::new(
        fleet.clone(),
    )));
    let parent_profile = dress(
        EngineProfile::new(
            provider_for(&a.providers, ORCHESTRATOR_PROFILE),
            Arc::new(registry),
            SystemPrompt::new("daemon host node"),
        ),
        &a,
    );
    // The durable path journals too: replace the discarding sink with one sealing per turn into the
    // shared store, keyed by the durable `SessionId`.
    let factory =
        CoreEngineFactory::from_profile(parent_profile).with_journal(a.store.clone(), signer.clone());

    let host = Host::new(a.store.clone(), Arc::new(factory), a.host_config)
        .with_job_worker(Arc::new(FleetJobWorker(fleet.clone())));
    let handle = host.start();

    // The interactive (session sub-surface) engines: built from the same seam (resolved provider +
    // brokered credentials), so the live path is not credential-asymmetric with the durable one.
    let session_profile = dress(
        EngineProfile::new(
            provider_for(&a.providers, a.profile.as_str()),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("interactive session"),
        ),
        &a,
    );
    let session_builder: SessionEngineBuilder = {
        let profile = session_profile;
        Arc::new(move |id: SessionId| profile.fresh(id))
    };

    let node = Arc::new(
        NodeApiImpl::new(
            handle.observer(),
            a.store.clone(),
            host.manager().clone(),
            a.partition,
            session_builder,
            Some(Arc::new(FleetViewImpl(fleet.clone())) as Arc<dyn FleetControl>),
        )
        // Live interactive sessions journal per turn; also records the signer so history reads verify.
        .with_journal(a.store.clone(), signer.clone()),
    );

    AssembledNode {
        node,
        handle,
        fleet,
        signer,
    }
}

// ---------------------------------------------------------------------------
// Composition-layer glue (moved here from `bins/daemon` so the binary and the
// conformance harness share one implementation).
// ---------------------------------------------------------------------------

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
}

impl ProfileChildSpawner {
    /// A spawner that materializes children from the in-process reference engine profile.
    pub fn core(profile: EngineProfile) -> Self {
        Self {
            backend: AgentBackend::Core(profile),
            provisioner: Arc::new(ProcessProvisioner::new()),
            journal: None,
        }
    }

    /// A spawner that materializes children by launching a foreign agent process.
    pub fn foreign(launch: LaunchProfile) -> Self {
        Self {
            backend: AgentBackend::Foreign(launch),
            provisioner: Arc::new(ProcessProvisioner::new()),
            journal: None,
        }
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
                let engine = profile.fresh(SessionId::new(id.as_str()));
                Arc::new(EngineUnit::spawn_journaled(id, engine, feeder))
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
                                Arc::new(CodecSection17::from_channel(
                                    channel,
                                    Some(child),
                                    host,
                                    StreamJsonCodec::new(),
                                )) as Arc<dyn Section17Session>
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

/// Drives the durable job outbox with the real fleet (spawn + run a child per delegation job).
pub struct FleetJobWorker(pub FleetRuntime);

#[async_trait]
impl JobWorker for FleetJobWorker {
    async fn process_jobs_once(&self) -> Result<(), ServiceError> {
        self.0
            .process_jobs_once()
            .await
            .map(|_| ())
            .map_err(ServiceError::new)
    }
}

/// Projects the fleet for the node control surface: the flat roster + the tree the GUI/TUI drives.
pub struct FleetViewImpl(pub FleetRuntime);

#[async_trait]
impl FleetControl for FleetViewImpl {
    async fn report(&self) -> FleetReport {
        FleetReport {
            children: self.0.children(),
            usage: self.0.fleet_usage(),
        }
    }

    async fn cancel(&self, child: &UnitId) -> bool {
        self.0.cancel_child(child).await
    }

    async fn tree(&self) -> daemon_api::TreeReport {
        self.0.tree()
    }

    async fn unit(&self, id: &UnitId) -> Option<daemon_api::UnitNode> {
        self.0.unit(id)
    }

    async fn unit_events(&self, id: &UnitId, max: u32) -> Vec<daemon_api::ManageEventView> {
        self.0.unit_events(id, max)
    }

    async fn unit_outbound(&self, id: &UnitId, max: u32) -> Vec<daemon_api::Outbound> {
        self.0.unit_outbound(id, max)
    }

    async fn pause(&self, id: &UnitId) -> bool {
        self.0.pause(id).await
    }

    async fn resume(&self, id: &UnitId) -> bool {
        self.0.resume(id).await
    }

    async fn scale(&self, id: &UnitId, n: u32) -> bool {
        self.0.scale(id, n).await
    }
}
