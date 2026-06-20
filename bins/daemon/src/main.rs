//! `daemon` — the host binary that assembles an engine, its host, tools, and orchestration.
//!
//! It is the role-by-config node (workspace-layout §6):
//! - default **host** role: assemble the durable substrate (store + resident services), the
//!   orchestration fleet (as the real job worker), credentials, and the live session surface, then
//!   serve the one [`daemon_api`] surface over a Unix socket ([`daemon_host::serve_api_unix`]).
//! - **placed-child** role (`DAEMON_PLACED_CHILD`): the far side of a placement cut, driving an
//!   engine whose durable state is brokered back to the parent's store.
//! - **transport-server** role (`DAEMON_TRANSPORT_SERVER=<addr>`): host a unit + authoritative
//!   store reached over a socket ([`daemon_transport::RemoteHost`]).

#![forbid(unsafe_code)]

mod config;

use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::FleetReport;
use daemon_common::{CredMode, CredScope, PartitionId, ProfileRef, SessionId, UnitId};
use daemon_core::{
    CredentialBuilder, CredentialProvider, Engine, EngineProfile, MockProvider, Provider,
    ProviderRegistry, SystemPrompt, ToolRegistry,
};
use daemon_credentials::{CapabilitySigner, CredentialAuthority, StubCredentialSource};
use daemon_host::{
    run_placed_child, serve_api_unix, BrokeredCredentialProvider, CoreEngineFactory,
    CredentialBroker, EngineUnit, FleetControl, Host, HostConfig, JobWorker, NodeApiImpl,
    OwnerBroker, ProcessAgentUnit, ServiceError, SessionEngineBuilder,
};
use daemon_orchestration::{ChildSpawner, DefaultAnswerPolicy, FleetRuntime};
use daemon_provision::{CutChannel, PlacementSpec, ProcessProvisioner, Provisioner};
use std::path::PathBuf;
use daemon_store::{InMemoryStore, SessionStore};
use daemon_supervision::{DelegationSpec, ManageRequestHandler, ManagedUnit};
use daemon_transport::RemoteHost;

use config::{NodeConfig, StoreBackend};

/// The environment variable that selects the placed-child role.
const PLACED_CHILD_ENV: &str = "DAEMON_PLACED_CHILD";
/// The environment variable that selects the transport-server role (its value is the bind address).
const TRANSPORT_SERVER_ENV: &str = "DAEMON_TRANSPORT_SERVER";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Stderr-only structured logging (stdout is the cut transport in the child role).
    daemon_telemetry::init_subscriber();

    if std::env::var_os(PLACED_CHILD_ENV).is_some() {
        run_as_placed_child().await;
        return Ok(());
    }

    if let Some(addr) = std::env::var_os(TRANSPORT_SERVER_ENV) {
        run_as_transport_server(addr.to_string_lossy().into_owned()).await?;
        return Ok(());
    }

    run_as_host(NodeConfig::load()?).await
}

/// Build the durable store backend the config selected.
fn build_store(backend: &StoreBackend) -> anyhow::Result<Arc<dyn SessionStore>> {
    match backend {
        StoreBackend::Memory => Ok(Arc::new(InMemoryStore::new())),
        StoreBackend::Sqlite { path } => {
            let store = daemon_store::SqliteStore::open(path)
                .map_err(|e| anyhow::anyhow!("opening sqlite store at {}: {e}", path.display()))?;
            Ok(Arc::new(store))
        }
    }
}

/// Assemble and run the default host node, serving the unified surface over a Unix socket until
/// `ctrl_c` trips a graceful shutdown.
async fn run_as_host(cfg: NodeConfig) -> anyhow::Result<()> {
    let store = build_store(&cfg.store)?;

    // Credentials: an owner authority brokered into *every* engine, uniformly across the durable,
    // interactive, and fleet-child construction paths (host-spec §6).
    let owner = build_owner_broker(&cfg.profile, &cfg.credential_key);
    let cred_profile = ProfileRef::new(cfg.profile.clone());
    let credentials: CredentialBuilder = {
        let owner = owner.clone();
        Arc::new(move || {
            Arc::new(BrokeredCredentialProvider::new(owner.clone(), None))
                as Arc<dyn CredentialProvider>
        })
    };

    // Provider selection seam: Mock is the default; a real networked provider drops in via a single
    // `register(...)` / `set_default(...)` without touching the engine or the construction sites.
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(MockProvider::completing("session done")) as Arc<dyn Provider>
    }));
    providers.register(
        "orchestrator",
        Arc::new(|| Arc::new(MockProvider::delegating("orchestrate", "fleet done")) as Arc<dyn Provider>),
    );
    providers.register(
        "child",
        Arc::new(|| Arc::new(MockProvider::completing("child done")) as Arc<dyn Provider>),
    );

    // Orchestration fleet: children built from one shared child profile, driven as the real job
    // worker (so every child gets the same provider + brokered credentials).
    let child_profile = EngineProfile::new(
        providers
            .builder_for(&ProfileRef::new("child"))
            .expect("child provider registered"),
        Arc::new(ToolRegistry::new()),
        SystemPrompt::new("fleet child"),
    )
    .with_credentials(credentials.clone(), cred_profile.clone());
    let fleet = FleetRuntime::new(
        store.clone(),
        cfg.partition,
        Arc::new(ProfileChildSpawner::core(child_profile)),
        Arc::new(DefaultAnswerPolicy),
        None::<Arc<dyn ManageRequestHandler>>,
    );

    // The parent orchestrator profile: an engine that delegates once through the orchestrate tool,
    // then completes — the durable delegate -> suspend -> resume -> complete cycle.
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(daemon_tool_orchestrate::OrchestrateTool::new(
        fleet.clone(),
    )));
    let parent_profile = EngineProfile::new(
        providers
            .builder_for(&ProfileRef::new("orchestrator"))
            .expect("orchestrator provider registered"),
        Arc::new(registry),
        SystemPrompt::new("daemon host node"),
    )
    .with_credentials(credentials.clone(), cred_profile.clone());
    let factory = CoreEngineFactory::from_profile(parent_profile);

    let config = HostConfig {
        partition: cfg.partition,
        dispatch_interval: cfg.dispatch_interval,
        scan_interval: cfg.scan_interval,
        ..HostConfig::default()
    };
    let host = Host::new(store.clone(), Arc::new(factory), config)
        .with_job_worker(Arc::new(FleetJobWorker(fleet.clone())));

    let handle = host.start();
    tracing::info!("daemon host node started");

    // The interactive (session sub-surface) engines: built from the same seam (default provider +
    // brokered credentials), so the live path is no longer credential-asymmetric with the durable one.
    let session_profile = EngineProfile::new(
        providers
            .builder_for(&cred_profile)
            .expect("default session provider"),
        Arc::new(ToolRegistry::new()),
        SystemPrompt::new("interactive session"),
    )
    .with_credentials(credentials.clone(), cred_profile.clone());
    let session_builder: SessionEngineBuilder = {
        let profile = session_profile;
        Arc::new(move |id: SessionId| profile.fresh(id))
    };

    let node = Arc::new(NodeApiImpl::new(
        handle.observer(),
        store.clone(),
        host.manager().clone(),
        cfg.partition,
        session_builder,
        Some(Arc::new(FleetViewImpl(fleet)) as Arc<dyn FleetControl>),
    ));

    // Bind the api socket (fresh) and serve the unified surface over it.
    let _ = std::fs::remove_file(&cfg.socket_path);
    let listener = tokio::net::UnixListener::bind(&cfg.socket_path)?;
    tracing::info!(socket = %cfg.socket_path.display(), "serving daemon-api over unix socket");
    let server = tokio::spawn(serve_api_unix(listener, node));

    tokio::signal::ctrl_c().await?;
    tracing::info!("ctrl_c received; shutting down");
    server.abort();
    handle.shutdown().await;
    let _ = std::fs::remove_file(&cfg.socket_path);
    Ok(())
}

/// Build the owner end of the credential brokering chain over a stub source (host-spec §7), minting
/// the configured key for the configured profile.
fn build_owner_broker(profile: &str, key: &str) -> Arc<dyn CredentialBroker> {
    let signer = Arc::new(CapabilitySigner::generate());
    let source = Arc::new(StubCredentialSource::minting(profile, key));
    let scope = CredScope::new([profile], ["chat"], Some(1_000));
    let authority = Arc::new(CredentialAuthority::new(
        scope,
        CredMode::Native,
        60_000,
        signer,
        source,
    ));
    Arc::new(OwnerBroker::new(authority))
}

/// A foreign agent launch profile: how to start a non-`daemon-core` brain that speaks §17 over a
/// process cut (mirrors [`daemon_provision::PlacementSpec`]). The reference brain needs none of this;
/// it is the home for "manage the foreign process's environment" the way `EngineProfile` is for ours.
#[allow(dead_code)]
struct LaunchProfile {
    program: PathBuf,
    args: Vec<String>,
    env: Vec<(String, String)>,
}

/// How to construct a child brain. `Core` is the in-process reference engine; `Foreign` launches an
/// external agent process. Both are presented up the tree as a `UnitKind::Engine` `ManagedUnit`, so
/// the fleet/orchestrator (and the GUI above it) cannot tell them apart. The default host wires
/// `Core`; the foreign path is exercised by the conformance gate.
#[allow(dead_code)]
enum AgentBackend {
    Core(EngineProfile),
    Foreign(LaunchProfile),
}

/// The profile-driven placement seam: materialize each child as the configured [`AgentBackend`],
/// uniformly presented as a `ManagedUnit`.
struct ProfileChildSpawner {
    backend: AgentBackend,
    provisioner: Arc<dyn Provisioner>,
}

impl ProfileChildSpawner {
    /// A spawner that materializes children from the in-process reference engine profile.
    fn core(profile: EngineProfile) -> Self {
        Self {
            backend: AgentBackend::Core(profile),
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
struct FleetJobWorker(FleetRuntime);

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
struct FleetViewImpl(FleetRuntime);

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

/// Run as a transport server: host a completing engine unit + an authoritative store, reachable as
/// a `ManagedUnit` over a socket (with the cross-node lease/fence handshake).
async fn run_as_transport_server(addr: String) -> anyhow::Result<()> {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let engine = Engine::fresh(
        SessionId::new("u1"),
        SystemPrompt::new("transport-hosted unit"),
        Arc::new(MockProvider::completing("transport done")) as Arc<dyn Provider>,
        Arc::new(ToolRegistry::new()),
    );
    let unit: Arc<dyn ManagedUnit> = Arc::new(EngineUnit::spawn(UnitId::new("u1"), engine));
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(%addr, "transport server listening");
    Arc::new(RemoteHost::new(store, unit)).serve(listener).await?;
    Ok(())
}

/// Run as the far side of a placement cut: a completing engine driven over the brokered store.
async fn run_as_placed_child() {
    let factory = CoreEngineFactory::with_provider(
        Arc::new(|| Arc::new(MockProvider::completing("placed child done")) as Arc<dyn Provider>),
        Arc::new(ToolRegistry::new()),
        SystemPrompt::new("placed child"),
    );
    run_placed_child(
        CutChannel::from_stdio(),
        Arc::new(factory),
        PartitionId::DEFAULT,
    )
    .await;
}
