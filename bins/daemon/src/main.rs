//! `daemon` — the host binary that assembles an engine, its host, tools, and orchestration.
//!
//! It is the role-by-config node (workspace-layout §6):
//! - default **host** role: build the policy inputs (store, credentials, provider registry, engine
//!   tunables) and hand them to [`daemon_node::assemble`] — the single host-composition root shared
//!   with the conformance harness — then serve the one [`daemon_api`] surface over a Unix socket.
//! - **placed-child** role (`DAEMON_PLACED_CHILD`): the far side of a placement cut, driving an
//!   engine whose durable state is brokered back to the parent's store.
//! - **transport-server** role (`DAEMON_TRANSPORT_SERVER=<addr>`): host a unit + authoritative
//!   store reached over a socket ([`daemon_transport::RemoteHost`]).

#![forbid(unsafe_code)]

mod config;

use std::sync::Arc;

use daemon_common::{CredMode, CredScope, JournalStreamId, ProfileRef, SessionId, UnitId};
use daemon_core::{
    CredentialBuilder, CredentialProvider, EngineProfile, MockProvider, Provider, ProviderRegistry,
    SystemPrompt, ToolRegistry,
};
use daemon_credentials::{CapabilitySigner, CredentialAuthority, StubCredentialSource};
use daemon_host::{
    run_placed_child, run_placed_child_journaled, serve_api_unix, BrokeredCredentialProvider,
    CoreEngineFactory, CredentialBroker, EngineUnit, HostConfig, JournalFeeder, JournalSink,
    OwnerBroker,
};
use daemon_node::{assemble, AssembledNode, NodeAssembly};
use daemon_provision::CutChannel;
use daemon_store::{InMemoryStore, SessionStore};
use daemon_supervision::ManagedUnit;
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
/// `ctrl_c` trips a graceful shutdown. The wiring itself lives in [`daemon_node::assemble`]; this
/// role only builds the policy inputs (store, credentials, provider registry, engine tunables).
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
        Arc::new(|| {
            Arc::new(MockProvider::delegating("orchestrate", "fleet done")) as Arc<dyn Provider>
        }),
    );
    providers.register(
        "child",
        Arc::new(|| Arc::new(MockProvider::completing("child done")) as Arc<dyn Provider>),
    );

    let host_config = HostConfig {
        partition: cfg.partition,
        dispatch_interval: cfg.dispatch_interval,
        scan_interval: cfg.scan_interval,
        ..HostConfig::default()
    };

    let AssembledNode { node, handle, .. } = assemble(NodeAssembly {
        store,
        partition: cfg.partition,
        host_config,
        providers,
        credentials: Some(credentials),
        profile: cred_profile,
        engine_config: cfg.engine,
        journal_seed: cfg.journal_seed,
    });
    tracing::info!("daemon host node started");

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

/// Run as a transport server: host a completing engine unit + an authoritative store, reachable as
/// a `ManagedUnit` over a socket (with the cross-node lease/fence handshake). The engine is built
/// through a *dressed* [`EngineProfile`] (engine tunables + a local owner-broker credential seam,
/// since a transport node is its own authority over its own store) and journals its transcript per
/// turn under a seed-derived signer, so its construction matches the host path.
async fn run_as_transport_server(addr: String) -> anyhow::Result<()> {
    let cfg = NodeConfig::load()?;
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());

    // A transport node owns its store, so it mints its own credentials (the host path's owner
    // broker) rather than brokering from a parent — the engine is therefore not credential-less.
    let owner = build_owner_broker(&cfg.profile, &cfg.credential_key);
    let credentials: CredentialBuilder = {
        let owner = owner.clone();
        Arc::new(move || {
            Arc::new(BrokeredCredentialProvider::new(owner.clone(), None))
                as Arc<dyn CredentialProvider>
        })
    };
    let profile = EngineProfile::new(
        Arc::new(|| Arc::new(MockProvider::completing("transport done")) as Arc<dyn Provider>),
        Arc::new(ToolRegistry::new()),
        SystemPrompt::new("transport-hosted unit"),
    )
    .with_config(cfg.engine)
    .with_credentials(credentials, ProfileRef::new(cfg.profile.clone()));

    // The unit journals per turn into the local store, keyed by its UnitId, sealed under the
    // config-seeded signer (or an ephemeral key when no seed is configured).
    let unit_id = UnitId::new("u1");
    let signer = Arc::new(
        cfg.journal_seed
            .map(|seed| daemon_telemetry::TraceSigner::from_seed(&seed))
            .unwrap_or_else(daemon_telemetry::TraceSigner::generate),
    );
    let sink = JournalSink::new(store.clone(), signer, JournalStreamId::unit(&unit_id));
    let feeder = Arc::new(JournalFeeder::new(Arc::new(sink)));

    let unit: Arc<dyn ManagedUnit> = Arc::new(EngineUnit::spawn_journaled(
        unit_id.clone(),
        profile.fresh(SessionId::new(unit_id.as_str())),
        Some(feeder),
    ));
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(%addr, "transport server listening");
    Arc::new(RemoteHost::new(store, unit)).serve(listener).await?;
    Ok(())
}

/// Run as the far side of a placement cut: a completing engine driven over the brokered store. The
/// engine is built from a *dressed* [`EngineProfile`] (engine tunables applied, via
/// [`CoreEngineFactory::from_profile`]) so it shares the host's construction seam rather than a
/// bespoke literal. When the node's journal seed is configured (passed down via `DAEMON_JOURNAL_SEED`
/// by the spawning parent), the child journals its durable transcript **through the parent's brokered
/// store**, sealed under the node's seed-derived signer so the chain verifies under the node's
/// published verifying key. Credentials stay on the embedded L1 pool — brokering them over the cut
/// is a separate channel, deferred.
async fn run_as_placed_child() {
    let cfg = match NodeConfig::load() {
        Ok(cfg) => cfg,
        Err(e) => {
            tracing::error!(error = %e, "placed child failed to load config");
            return;
        }
    };
    let profile = EngineProfile::new(
        Arc::new(|| Arc::new(MockProvider::completing("placed child done")) as Arc<dyn Provider>),
        Arc::new(ToolRegistry::new()),
        SystemPrompt::new("placed child"),
    )
    .with_config(cfg.engine);
    let factory = CoreEngineFactory::from_profile(profile);
    let channel = CutChannel::from_stdio();

    match cfg.journal_seed {
        Some(seed) => {
            let signer = Arc::new(daemon_telemetry::TraceSigner::from_seed(&seed));
            run_placed_child_journaled(channel, factory, cfg.partition, signer).await;
        }
        None => run_placed_child(channel, Arc::new(factory), cfg.partition).await,
    }
}
