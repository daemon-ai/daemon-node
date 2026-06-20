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
use daemon_common::{PartitionId, ProfileRef, SessionId, UnitId};
use daemon_core::{
    Config, CredentialBuilder, EngineProfile, ProviderRegistry, SystemPrompt, ToolRegistry,
};
use daemon_host::{
    CoreEngineFactory, EngineUnit, FleetControl, Host, HostConfig, JobWorker, NodeApiImpl,
    ProcessAgentUnit, ServiceError, SessionEngineBuilder, SupervisorHandle,
};
use daemon_orchestration::{ChildSpawner, DefaultAnswerPolicy, FleetRuntime};
use daemon_provision::{PlacementSpec, ProcessProvisioner, Provisioner};
use daemon_supervision::{DelegationSpec, ManageRequestHandler, ManagedUnit};

/// The provider-registry profile name the orchestrator (parent) engine resolves to.
const ORCHESTRATOR_PROFILE: &str = "orchestrator";
/// The provider-registry profile name the fleet-child engine resolves to.
const CHILD_PROFILE: &str = "child";

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
}

/// The assembled node: the bound surface, its started resident-service handle, and the fleet handle.
pub struct AssembledNode {
    /// The one [`daemon_api`] surface (control + session + fleet sub-surfaces).
    pub node: Arc<NodeApiImpl>,
    /// The started resident-service tree; drive shutdown via [`SupervisorHandle::shutdown`].
    pub handle: SupervisorHandle,
    /// The orchestration fleet handle (e.g. for inspection in tests).
    pub fleet: FleetRuntime,
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
    // The fleet child: one shared profile, driven as the real job worker so every child gets the same
    // provider + brokered credentials.
    let child_profile = dress(
        EngineProfile::new(
            provider_for(&a.providers, CHILD_PROFILE),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("fleet child"),
        ),
        &a,
    );
    let fleet = FleetRuntime::new(
        a.store.clone(),
        a.partition,
        Arc::new(ProfileChildSpawner::core(child_profile)),
        Arc::new(DefaultAnswerPolicy),
        None::<Arc<dyn ManageRequestHandler>>,
    );

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
    let factory = CoreEngineFactory::from_profile(parent_profile);

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

    let node = Arc::new(NodeApiImpl::new(
        handle.observer(),
        a.store.clone(),
        host.manager().clone(),
        a.partition,
        session_builder,
        Some(Arc::new(FleetViewImpl(fleet.clone())) as Arc<dyn FleetControl>),
    ));

    AssembledNode { node, handle, fleet }
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
}

impl ProfileChildSpawner {
    /// A spawner that materializes children from the in-process reference engine profile.
    pub fn core(profile: EngineProfile) -> Self {
        Self {
            backend: AgentBackend::Core(profile),
            provisioner: Arc::new(ProcessProvisioner::new()),
        }
    }

    /// A spawner that materializes children by launching a foreign agent process.
    pub fn foreign(launch: LaunchProfile) -> Self {
        Self {
            backend: AgentBackend::Foreign(launch),
            provisioner: Arc::new(ProcessProvisioner::new()),
        }
    }
}

#[async_trait]
impl ChildSpawner for ProfileChildSpawner {
    async fn spawn(&self, id: UnitId, _spec: &DelegationSpec) -> Arc<dyn ManagedUnit> {
        match &self.backend {
            AgentBackend::Core(profile) => {
                let engine = profile.fresh(SessionId::new(id.as_str()));
                Arc::new(EngineUnit::spawn(id, engine))
            }
            AgentBackend::Foreign(launch) => {
                let placement = self
                    .provisioner
                    .place(
                        &SessionId::new(id.as_str()),
                        PlacementSpec {
                            program: launch.program.clone(),
                            args: launch.args.clone(),
                            env: launch.env.clone(),
                        },
                    )
                    .await
                    .expect("place foreign agent");
                Arc::new(ProcessAgentUnit::start(id, placement))
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
