// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `curator` subcommand: per-profile skill curation (`curator_*` ops).

use daemon_api::ApiRequest;
use daemon_host::ApiClient;

use crate::cli::CuratorCmd;
use crate::render::render;

/// Dispatch a `curator` subcommand over the api mirror.
pub(super) async fn run(client: &ApiClient, cmd: CuratorCmd) -> anyhow::Result<()> {
    let req = match cmd {
        CuratorCmd::List { profile } => ApiRequest::CuratorList { profile },
        CuratorCmd::Pin { name, profile } => ApiRequest::CuratorPin { profile, name },
        CuratorCmd::Unpin { name, profile } => ApiRequest::CuratorUnpin { profile, name },
        CuratorCmd::Archive { name, profile } => ApiRequest::CuratorArchive { profile, name },
        CuratorCmd::Restore { name, profile } => ApiRequest::CuratorRestore { profile, name },
        CuratorCmd::Run { profile } => ApiRequest::CuratorRun { profile },
    };
    render(client.call(req).await?);
    Ok(())
}
