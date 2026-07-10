// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `contact` subcommand: remote-contact operations over the messaging-adapter interface
//! (`contact_*` ops). The `contact` arg is the contact id (an MXID on Matrix), wrapped as a
//! `Participant::Contact` carrier.

use daemon_api::{ApiRequest, ContactInfo};
use daemon_host::ApiClient;
use daemon_protocol::TransportId;

use crate::cli::ContactCmd;
use crate::render::render;

/// Dispatch a `contact` subcommand over the api mirror.
pub(super) async fn run(client: &ApiClient, cmd: ContactCmd) -> anyhow::Result<()> {
    let req = match cmd {
        ContactCmd::GetProfile { transport, contact } => ApiRequest::ContactGetProfile {
            transport: TransportId::new(transport),
            contact: ContactInfo {
                id: contact,
                ..ContactInfo::default()
            },
        },
        ContactCmd::SetAlias {
            transport,
            contact,
            alias,
        } => ApiRequest::ContactSetAlias {
            transport: TransportId::new(transport),
            contact: ContactInfo {
                id: contact,
                ..ContactInfo::default()
            },
            alias,
            op_id: None,
        },
    };
    render(client.call(req).await?);
    Ok(())
}
