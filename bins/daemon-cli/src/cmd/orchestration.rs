// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The node/orchestration flat commands (`status`, `health`, `stats`, `telemetry`, `fleet`,
//! `tree`, `unit`, `unit-outbound`, `unit-events`, `pause`, `resume`, `scale`, `unit-history`,
//! `verifying-key`, `transports`). This is the exhaustive tail of the flat-command chain: the
//! nested subcommand groups are peeled off by [`super::dispatch`] before it runs, so they are
//! `unreachable!` here while the compiler still proves every flat command is handled.

use daemon_api::ApiRequest;
use daemon_common::UnitId;
use daemon_host::ApiClient;

use crate::cli::Command;
use crate::render::render;

/// Handle the remaining flat node/orchestration commands.
pub(super) async fn run(client: &ApiClient, command: Command) -> anyhow::Result<()> {
    match command {
        Command::Status => {
            render(client.call(ApiRequest::Health).await?);
            render(client.call(ApiRequest::Stats).await?);
        }
        Command::Health => render(client.call(ApiRequest::Health).await?),
        Command::Stats => render(client.call(ApiRequest::Stats).await?),
        Command::Telemetry => render(client.call(ApiRequest::Telemetry).await?),
        Command::Fleet => render(client.call(ApiRequest::Fleet).await?),
        Command::Tree => render(client.call(ApiRequest::Tree).await?),
        Command::Unit { id } => render(
            client
                .call(ApiRequest::Unit {
                    unit: UnitId::new(id),
                })
                .await?,
        ),
        Command::UnitOutbound { id, max } => render(
            client
                .call(ApiRequest::UnitOutbound {
                    unit: UnitId::new(id),
                    max,
                })
                .await?,
        ),
        Command::UnitEvents { id, max } => render(
            client
                .call(ApiRequest::UnitEvents {
                    unit: UnitId::new(id),
                    max,
                })
                .await?,
        ),
        Command::Pause { id } => render(
            client
                .call(ApiRequest::Pause {
                    unit: UnitId::new(id),
                })
                .await?,
        ),
        Command::Resume { id } => render(
            client
                .call(ApiRequest::Resume {
                    unit: UnitId::new(id),
                })
                .await?,
        ),
        Command::Scale { id, n } => render(
            client
                .call(ApiRequest::Scale {
                    unit: UnitId::new(id),
                    n,
                })
                .await?,
        ),
        Command::UnitHistory { id, after, max } => render(
            client
                .call(ApiRequest::UnitHistory {
                    unit: UnitId::new(id),
                    after_cursor: after,
                    max,
                })
                .await?,
        ),
        Command::VerifyingKey => render(client.call(ApiRequest::VerifyingKey).await?),
        Command::Transports => render(client.call(ApiRequest::TransportInstances).await?),
        _ => unreachable!("nested subcommands are routed by the dispatcher"),
    }
    Ok(())
}
