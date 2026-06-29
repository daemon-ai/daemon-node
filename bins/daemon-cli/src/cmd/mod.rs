// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Command handlers: each maps a parsed [`crate::cli`] command into one (or, for `model up`, a
//! short sequence of) [`daemon_api::ApiRequest`] and renders the reply. [`dispatch`] is the thin
//! router — it peels the nested subcommand groups off to their modules and chains the flat
//! node/session/orchestration commands the way `render` chains its domain renderers.

use daemon_api::{ContactInfo, Participant};
use daemon_common::ProfileRef;
use daemon_host::ApiClient;

use crate::cli::Command;

mod contact;
mod conv;
mod cron;
mod curator;
mod directory;
mod member;
mod model;
mod orchestration;
mod session;

/// Route a parsed top-level command to its handler.
pub(crate) async fn dispatch(client: &ApiClient, command: Command) -> anyhow::Result<()> {
    match command {
        Command::Model { cmd } => model::run(client, cmd).await,
        Command::Curator { cmd } => curator::run(client, cmd).await,
        Command::Cron { cmd } => cron::run(client, cmd).await,
        Command::Conv { cmd } => conv::run(client, cmd).await,
        Command::Member { cmd } => member::run(client, cmd).await,
        Command::Contact { cmd } => contact::run(client, cmd).await,
        Command::Directory { cmd } => directory::run(client, cmd).await,
        flat => {
            let Some(flat) = session::try_run(client, flat).await? else {
                return Ok(());
            };
            orchestration::run(client, flat).await
        }
    }
}

/// Build a [`Participant`] from a member handle and an optional profile: with a profile it is an
/// agent participant (`Participant::Agent`), otherwise a contact.
pub(super) fn participant(member: String, profile: Option<String>) -> Participant {
    match profile {
        Some(p) => Participant::Agent {
            profile: ProfileRef::new(p),
            member,
        },
        None => Participant::Contact(ContactInfo {
            id: member,
            ..ContactInfo::default()
        }),
    }
}
