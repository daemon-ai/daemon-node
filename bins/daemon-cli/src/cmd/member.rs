// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `member` subcommand: membership administration over the messaging-adapter interface
//! (`member_*` ops).

use daemon_api::{ApiRequest, MemberRole};
use daemon_host::ApiClient;
use daemon_protocol::TransportId;

use crate::cli::MemberCmd;
use crate::render::render;

/// Parse a [`MemberRole`] from its CLI label.
fn parse_role(role: &str) -> anyhow::Result<MemberRole> {
    Ok(match role.to_ascii_lowercase().as_str() {
        "none" => MemberRole::None,
        "voice" => MemberRole::Voice,
        "halfop" => MemberRole::HalfOp,
        "op" => MemberRole::Op,
        "founder" => MemberRole::Founder,
        other => anyhow::bail!("unknown role {other:?} (none|voice|halfop|op|founder)"),
    })
}

/// Dispatch a `member` subcommand over the api mirror.
pub(super) async fn run(client: &ApiClient, cmd: MemberCmd) -> anyhow::Result<()> {
    let req = match cmd {
        MemberCmd::Invite {
            transport,
            conv,
            member,
            profile,
            message,
        } => ApiRequest::MemberInvite(daemon_api::MemberInviteArgs {
            transport: TransportId::new(transport),
            conv,
            who: super::participant(member, profile),
            message,
            op_id: None,
        }),
        MemberCmd::Remove {
            transport,
            conv,
            member,
            profile,
            reason,
        } => ApiRequest::MemberRemove(daemon_api::MemberRemoveArgs {
            transport: TransportId::new(transport),
            conv,
            who: super::participant(member, profile),
            reason,
            op_id: None,
        }),
        MemberCmd::Ban {
            transport,
            conv,
            member,
            profile,
            reason,
        } => ApiRequest::MemberBan(daemon_api::MemberBanArgs {
            transport: TransportId::new(transport),
            conv,
            who: super::participant(member, profile),
            reason,
            op_id: None,
        }),
        MemberCmd::SetRole {
            transport,
            conv,
            member,
            role,
            profile,
        } => ApiRequest::MemberSetRole(daemon_api::MemberSetRoleArgs {
            transport: TransportId::new(transport),
            conv,
            who: super::participant(member, profile),
            role: parse_role(&role)?,
            op_id: None,
        }),
    };
    render(client.call(req).await?);
    Ok(())
}
