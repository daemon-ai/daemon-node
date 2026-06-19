//! `daemon` — the host binary that assembles an engine, its host, tools, and orchestration.
//!
//! It is the role-by-config node (workspace-layout §6). Phase 5 adds the *placed-child* role: when
//! `DAEMON_PLACED_CHILD` is set, the process is the far side of a placement cut — it runs
//! [`daemon_host::run_placed_child`] over its stdio, driving an engine whose durable state is
//! brokered back to the parent's store. The full host-assembly role is wired in a later phase.

#![forbid(unsafe_code)]

use std::sync::Arc;

use daemon_common::PartitionId;
use daemon_core::{MockProvider, Provider, SystemPrompt, ToolRegistry};
use daemon_host::{run_placed_child, CoreEngineFactory};
use daemon_provision::CutChannel;

/// The environment variable that selects the placed-child role.
const PLACED_CHILD_ENV: &str = "DAEMON_PLACED_CHILD";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if std::env::var_os(PLACED_CHILD_ENV).is_some() {
        run_as_placed_child().await;
        return Ok(());
    }

    // TODO: full host assembly (engine + host + tools + orchestration) lands in a later phase.
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
