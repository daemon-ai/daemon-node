//! `daemon` — the host binary that assembles an engine, its host, tools, and orchestration.
//!
//! It is the role-by-config node (workspace-layout §6). Phase 5 adds the *placed-child* role: when
//! `DAEMON_PLACED_CHILD` is set, the process is the far side of a placement cut — it runs
//! [`daemon_host::run_placed_child`] over its stdio, driving an engine whose durable state is
//! brokered back to the parent's store. Phase 6 adds the *transport-server* role: when
//! `DAEMON_TRANSPORT_SERVER=<addr>` is set, the process hosts a unit + authoritative store reached
//! over a socket ([`daemon_transport::RemoteHost`]). The full host-assembly role is wired later.

#![forbid(unsafe_code)]

use std::sync::Arc;

use daemon_common::{PartitionId, SessionId, UnitId};
use daemon_core::{Engine, MockProvider, Provider, SystemPrompt, ToolRegistry};
use daemon_host::{run_placed_child, CoreEngineFactory, EngineUnit};
use daemon_provision::CutChannel;
use daemon_store::{InMemoryStore, SessionStore};
use daemon_supervision::ManagedUnit;
use daemon_transport::RemoteHost;

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

    // TODO: full host assembly (engine + host + tools + orchestration) lands in a later phase.
    Ok(())
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
