// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `conv` subcommand: conversation management over the messaging-adapter interface
//! (`conv_*` ops).

use daemon_api::{
    AccountSettingsValues, ApiRequest, ChannelJoinDetails, CreateConversationDetails,
};
use daemon_host::ApiClient;
use daemon_protocol::{TransportId, UserMsg};

use crate::cli::ConvCmd;
use crate::render::render;

/// Dispatch a `conv` subcommand over the api mirror.
pub(super) async fn run(client: &ApiClient, cmd: ConvCmd) -> anyhow::Result<()> {
    let req = match cmd {
        ConvCmd::List { transport } => ApiRequest::ConvList {
            transport: TransportId::new(transport),
            after: None,
            since_rev: None,
        },
        ConvCmd::Get { transport, conv } => ApiRequest::ConvGet {
            transport: TransportId::new(transport),
            conv,
        },
        ConvCmd::Create {
            transport,
            id,
            name,
            policy,
            kind,
        } => {
            let mut values = AccountSettingsValues::default();
            if let Some(id) = id {
                values.values.insert("id".into(), id);
            }
            if let Some(name) = name {
                values.values.insert("name".into(), name);
            }
            if let Some(policy) = policy {
                values.values.insert("policy".into(), policy);
            }
            if let Some(kind) = kind {
                values.values.insert("kind".into(), kind);
            }
            ApiRequest::ConvCreate {
                transport: TransportId::new(transport),
                details: CreateConversationDetails {
                    extras: values,
                    ..CreateConversationDetails::default()
                },
            }
        }
        ConvCmd::Join { transport, name } => ApiRequest::ConvJoin {
            transport: TransportId::new(transport),
            details: ChannelJoinDetails {
                name: Some(name),
                ..ChannelJoinDetails::default()
            },
        },
        ConvCmd::Leave { transport, conv } => ApiRequest::ConvLeave {
            transport: TransportId::new(transport),
            conv,
        },
        ConvCmd::Send {
            transport,
            conv,
            text,
            from,
            from_profile,
        } => ApiRequest::ConvSend(daemon_api::ConvSendArgs {
            transport: TransportId::new(transport),
            conv,
            from: from.map(|m| super::participant(m, from_profile)),
            message: UserMsg::new(text),
        }),
        ConvCmd::Topic {
            transport,
            conv,
            topic,
        } => ApiRequest::ConvSetTopic {
            transport: TransportId::new(transport),
            conv,
            topic,
        },
        ConvCmd::Title {
            transport,
            conv,
            title,
        } => ApiRequest::ConvSetTitle {
            transport: TransportId::new(transport),
            conv,
            title,
        },
        ConvCmd::Describe {
            transport,
            conv,
            description,
        } => ApiRequest::ConvSetDescription {
            transport: TransportId::new(transport),
            conv,
            description,
        },
        ConvCmd::Delete { transport, conv } => ApiRequest::ConvDelete {
            transport: TransportId::new(transport),
            conv,
        },
        ConvCmd::History {
            transport,
            conv,
            after,
            max,
        } => ApiRequest::ConvHistory(daemon_api::ConvHistoryArgs {
            transport: TransportId::new(transport),
            conv,
            after_cursor: after,
            before_cursor: None,
            max,
        }),
    };
    render(client.call(req).await?);
    Ok(())
}
