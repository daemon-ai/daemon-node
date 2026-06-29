// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `directory` subcommand: contact/user-directory search over the messaging-adapter interface
//! (`directory_search`).

use daemon_api::ApiRequest;
use daemon_host::ApiClient;
use daemon_protocol::TransportId;

use crate::cli::DirectoryCmd;
use crate::render::render;

/// Dispatch a `directory` subcommand over the api mirror.
pub(super) async fn run(client: &ApiClient, cmd: DirectoryCmd) -> anyhow::Result<()> {
    let req = match cmd {
        DirectoryCmd::Search { transport, query } => ApiRequest::DirectorySearch {
            transport: TransportId::new(transport),
            query,
        },
    };
    render(client.call(req).await?);
    Ok(())
}
