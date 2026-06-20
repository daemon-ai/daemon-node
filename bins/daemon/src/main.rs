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

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::FleetReport;
use daemon_common::{CredMode, CredScope, PartitionId, ProfileRef, SessionId, UnitId};
use daemon_core::{
    CredentialProvider, Engine, MockProvider, Provider, SystemPrompt, ToolRegistry,
};
use daemon_credentials::{CapabilitySigner, CredentialAuthority, StubCredentialSource};
use daemon_host::{
    run_placed_child, serve_api_unix, BrokeredCredentialProvider, CoreEngineFactory,
    CredentialBroker, EngineUnit, FleetView, Host, HostConfig, JobWorker, NodeApiImpl, OwnerBroker,
    ServiceError, SessionEngineBuilder,
};
use daemon_orchestration::{ChildSpawner, DefaultAnswerPolicy, FleetRuntime};
use daemon_provision::CutChannel;
use daemon_store::{InMemoryStore, SessionStore};
use daemon_supervision::{DelegationSpec, ManageRequestHandler, ManagedUnit};
use daemon_transport::RemoteHost;

/// The environment variable that selects the placed-child role.
const PLACED_CHILD_ENV: &str = "DAEMON_PLACED_CHILD";
/// The environment variable that selects the transport-server role (its value is the bind address).
const TRANSPORT_SERVER_ENV: &str = "DAEMON_TRANSPORT_SERVER";
/// The environment variable overriding the host role's api socket path.
const API_SOCKET_ENV: &str = "DAEMON_API_SOCKET";

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

    run_as_host(NodeConfig::from_env()).await
}

/// The host role's env-first configuration.
struct NodeConfig {
    /// The partition this node owns.
    partition: PartitionId,
    /// The Unix socket the node serves its [`daemon_api`] surface on.
    socket_path: PathBuf,
}

impl NodeConfig {
    fn from_env() -> Self {
        let socket_path = std::env::var_os(API_SOCKET_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let dir = std::env::var_os("TMPDIR").unwrap_or_else(|| "/tmp".into());
                PathBuf::from(dir).join("daemon-api.sock")
            });
        Self {
            partition: PartitionId::DEFAULT,
            socket_path,
        }
    }
}

/// Assemble and run the default host node, serving the unified surface over a Unix socket until
/// `ctrl_c` trips a graceful shutdown.
async fn run_as_host(cfg: NodeConfig) -> anyhow::Result<()> {
    let store = Arc::new(InMemoryStore::new());

    // Orchestration fleet: a completing-child spawner, driven as the node's real job worker.
    let fleet = FleetRuntime::new(
        store.clone(),
        cfg.partition,
        Arc::new(EngineChildSpawner),
        Arc::new(DefaultAnswerPolicy),
        None::<Arc<dyn ManageRequestHandler>>,
    );

    // Credentials: an owner authority brokered into every engine the factory builds (host-spec §6).
    let owner = build_owner_broker();
    let credential_owner = owner.clone();
    let credentials: daemon_host::engine_incarnation::CredentialBuilder = Arc::new(move || {
        Arc::new(BrokeredCredentialProvider::new(credential_owner.clone(), None))
            as Arc<dyn CredentialProvider>
    });

    // The parent factory: an engine that delegates once through the orchestrate tool, then
    // completes — the durable delegate -> suspend -> resume -> complete cycle.
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(daemon_tool_orchestrate::OrchestrateTool::new(
        fleet.clone(),
    )));
    let factory = CoreEngineFactory::with_provider(
        Arc::new(|| {
            Arc::new(MockProvider::delegating("orchestrate", "fleet done")) as Arc<dyn Provider>
        }),
        Arc::new(registry),
        SystemPrompt::new("daemon host node"),
    )
    .with_credentials(credentials, ProfileRef::new("openai"));

    let config = HostConfig {
        partition: cfg.partition,
        ..HostConfig::default()
    };
    let host = Host::new(store.clone(), Arc::new(factory), config)
        .with_job_worker(Arc::new(FleetJobWorker(fleet.clone())));

    let handle = host.start();
    tracing::info!("daemon host node started");

    // The interactive (session sub-surface) engines complete in one turn.
    let session_builder: SessionEngineBuilder = Arc::new(|id: SessionId| {
        Engine::fresh(
            id,
            SystemPrompt::new("interactive session"),
            Arc::new(MockProvider::completing("session done")) as Arc<dyn Provider>,
            Arc::new(ToolRegistry::new()),
        )
    });

    let node = Arc::new(NodeApiImpl::new(
        handle.observer(),
        store.clone() as Arc<dyn SessionStore>,
        host.manager().clone(),
        cfg.partition,
        session_builder,
        Some(Arc::new(FleetViewImpl(fleet)) as Arc<dyn FleetView>),
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

/// Build the owner end of the credential brokering chain over a stub source (host-spec §7).
fn build_owner_broker() -> Arc<dyn CredentialBroker> {
    let signer = Arc::new(CapabilitySigner::generate());
    let source = Arc::new(StubCredentialSource::minting("openai", "sk-configured"));
    let scope = CredScope::new(["openai"], ["chat"], Some(1_000));
    let authority = Arc::new(CredentialAuthority::new(
        scope,
        CredMode::Native,
        60_000,
        signer,
        source,
    ));
    Arc::new(OwnerBroker::new(authority))
}

/// The injected placement seam: materialize a child as an engine-backed `ManagedUnit`. A completing
/// provider finishes the child in one turn (no further delegation).
struct EngineChildSpawner;

#[async_trait]
impl ChildSpawner for EngineChildSpawner {
    async fn spawn(&self, id: UnitId, _spec: &DelegationSpec) -> Arc<dyn ManagedUnit> {
        let engine = Engine::fresh(
            SessionId::new(id.as_str()),
            SystemPrompt::new("fleet child"),
            Arc::new(MockProvider::completing("child done")) as Arc<dyn Provider>,
            Arc::new(ToolRegistry::new()),
        );
        Arc::new(EngineUnit::spawn(id, engine))
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

/// Projects the fleet for the node control surface (`ControlApi::fleet`/`cancel`).
struct FleetViewImpl(FleetRuntime);

#[async_trait]
impl FleetView for FleetViewImpl {
    async fn report(&self) -> FleetReport {
        FleetReport {
            children: self.0.children(),
            usage: self.0.fleet_usage(),
        }
    }

    async fn cancel(&self, child: &UnitId) -> bool {
        self.0.cancel_child(child).await
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
