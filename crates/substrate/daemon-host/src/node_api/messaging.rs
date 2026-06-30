// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Messaging-adapter feature resolution (daemon-messaging-adapter-spec.md §6): map a `TransportId`
//! through the adapter registry to the `MessagingProtocol` feature trait a management op needs.

use super::*;

impl NodeApiImpl {
    /// Resolve the conversation-management feature for `transport` through the adapter registry
    /// (`adapter_for_transport -> messaging -> conversations`).
    pub(crate) fn conversations_for(
        &self,
        transport: &TransportId,
    ) -> Result<Arc<dyn SupportsConversations>, ApiError> {
        self.adapters
            .load_full()
            .adapter_for_transport(transport)
            .and_then(|a| a.messaging())
            .and_then(|m| m.conversations())
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "transport {} has no conversation support",
                    transport.as_str()
                ))
            })
    }

    /// Resolve the membership-administration feature for `transport`.
    pub(crate) fn membership_for(
        &self,
        transport: &TransportId,
    ) -> Result<Arc<dyn SupportsMembership>, ApiError> {
        self.adapters
            .load_full()
            .adapter_for_transport(transport)
            .and_then(|a| a.messaging())
            .and_then(|m| m.membership())
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "transport {} has no membership support",
                    transport.as_str()
                ))
            })
    }

    /// Resolve the remote-contacts feature for `transport`.
    pub(crate) fn contacts_for(
        &self,
        transport: &TransportId,
    ) -> Result<Arc<dyn SupportsContacts>, ApiError> {
        self.adapters
            .load_full()
            .adapter_for_transport(transport)
            .and_then(|a| a.messaging())
            .and_then(|m| m.contacts())
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "transport {} has no contacts support",
                    transport.as_str()
                ))
            })
    }

    /// Resolve the contact/user-directory feature for `transport`.
    pub(crate) fn directory_for(
        &self,
        transport: &TransportId,
    ) -> Result<Arc<dyn SupportsDirectory>, ApiError> {
        self.adapters
            .load_full()
            .adapter_for_transport(transport)
            .and_then(|a| a.messaging())
            .and_then(|m| m.directory())
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "transport {} has no directory support",
                    transport.as_str()
                ))
            })
    }
}

/// A human label for a [`Participant`] (the management-audit detail; never a secret payload).
pub(crate) fn participant_label(who: &Participant) -> String {
    match who {
        Participant::Contact(c) => c.id.clone(),
        Participant::Agent { member, profile } => {
            format!("{member} (profile {})", profile.as_str())
        }
    }
}
