// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `cron` subcommand: scheduled-job management + the consent-first suggestion catalog
//! (`cron_*` ops).

use daemon_api::{ApiRequest, CronSpec};
use daemon_host::ApiClient;

use crate::cli::{CronCmd, CronSuggestCmd};
use crate::render::render;

/// Dispatch a `cron` subcommand over the api mirror.
pub(super) async fn run(client: &ApiClient, cmd: CronCmd) -> anyhow::Result<()> {
    let req = match cmd {
        CronCmd::Create {
            name,
            schedule,
            prompt,
            timezone,
            repeat,
            disabled,
        } => ApiRequest::CronCreate {
            spec: CronSpec {
                name,
                schedule,
                payload: prompt.into_bytes(),
                enabled: !disabled,
                timezone,
                repeat,
                ..CronSpec::default()
            },
        },
        CronCmd::List => ApiRequest::CronList,
        CronCmd::Update {
            id,
            name,
            schedule,
            prompt,
        } => ApiRequest::CronUpdate {
            id,
            spec: CronSpec {
                name,
                schedule,
                payload: prompt.into_bytes(),
                enabled: true,
                ..CronSpec::default()
            },
        },
        CronCmd::Pause { id } => ApiRequest::CronPause { id, paused: true },
        CronCmd::Resume { id } => ApiRequest::CronPause { id, paused: false },
        CronCmd::Run { id } => ApiRequest::CronTrigger { id },
        CronCmd::Remove { id } => ApiRequest::CronDelete { id },
        CronCmd::Runs { id } => ApiRequest::CronRuns { id },
        CronCmd::Suggest { cmd } => match cmd {
            CronSuggestCmd::List => ApiRequest::CronSuggestions,
            CronSuggestCmd::Accept { id } => ApiRequest::CronAcceptSuggestion { id },
            CronSuggestCmd::Dismiss { id } => ApiRequest::CronDismissSuggestion { id },
        },
    };
    render(client.call(req).await?);
    Ok(())
}
