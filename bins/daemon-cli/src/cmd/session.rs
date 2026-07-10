// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The durable/interactive-session flat commands (`sessions`, `assign`, `cancel`, `rename`, `pin`,
//! `archive`, `submit`, `poll`, `history`). Like the `render` chain, this peels its own variants
//! and hands anything else back as `Some(command)` for the next handler.

use daemon_api::ApiRequest;
use daemon_common::{ReqId, SessionId};
use daemon_host::ApiClient;
use daemon_protocol::{AgentCommand, UserMsg};

use crate::cli::Command;
use crate::render::render;

/// Handle a session command, or return it unhandled for the next handler in the chain.
pub(super) async fn try_run(
    client: &ApiClient,
    command: Command,
) -> anyhow::Result<Option<Command>> {
    let req = match command {
        Command::Sessions => ApiRequest::Sessions,
        Command::Assign { id } => ApiRequest::Assign {
            session: SessionId::new(id),
        },
        Command::Cancel { id } => ApiRequest::Cancel {
            session: SessionId::new(id),
        },
        Command::Rename { id, title } => ApiRequest::SessionUpdateMeta {
            session: SessionId::new(id),
            patch: daemon_api::SessionMetaPatch {
                title: Some(title),
                ..Default::default()
            },
            op_id: None,
        },
        Command::Pin { id, off } => ApiRequest::SessionUpdateMeta {
            session: SessionId::new(id),
            patch: daemon_api::SessionMetaPatch {
                pinned: Some(!off),
                ..Default::default()
            },
            op_id: None,
        },
        Command::Archive { id, off } => ApiRequest::SessionUpdateMeta {
            session: SessionId::new(id),
            patch: daemon_api::SessionMetaPatch {
                archived: Some(!off),
                ..Default::default()
            },
            op_id: None,
        },
        Command::Submit { id, text } => ApiRequest::Submit {
            session: SessionId::new(id),
            command: AgentCommand::StartTurn {
                input: UserMsg::new(text),
                request_id: ReqId(1),
            },
            origin: None,
            profile: None,
        },
        Command::Poll { id, max } => ApiRequest::Poll {
            session: SessionId::new(id),
            max,
        },
        Command::History { id, after, max } => ApiRequest::SessionHistory {
            session: SessionId::new(id),
            after_cursor: after,
            before_cursor: None,
            max,
        },
        other => return Ok(Some(other)),
    };
    render(client.call(req).await?);
    Ok(None)
}
