// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Authorization requests, add-contact requests, and typed notifications ported from libpurple
//! (work package W2-G). Unlike the Wave-1 DTO-logic module, this package **touches the wire**: the
//! types here are reachable from [`ApiResponse::Notifications`](crate::ApiResponse) and
//! [`NodeEvent::NotificationsChanged`](crate::NodeEvent), so they are serde types mirrored in
//! `daemon-api.cddl` and derive feature-gated [`arbitrary::Arbitrary`].
//!
//! - [`AuthorizationRequest`] ← `purpleauthorizationrequest.c` (accept/deny idempotency + coupling).
//! - [`AddContactRequest`] ← `purpleaddcontactrequest.c` (single-shot `add`).
//! - [`NotificationInfo`] + [`NotificationKind`] ← `purplenotification.c` and its subclasses
//!   (`purplenotification{addcontact,authorizationrequest,link,connectionerror}.c`): id,
//!   created-timestamp, read state, optional account/transport binding, and the typed kinds.
//!
//! The host [`NotificationManager`](../../daemon_host/notifications/struct.NotificationManager.html)
//! (in `daemon-host`) owns the live collection + unread accounting; this module is the pure DTO layer
//! the node is authoritative over and the thin clients render.

use crate::ContactInfo;
use daemon_protocol::TransportId;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Shared request-handling error
// ---------------------------------------------------------------------------

/// The programming-error a second `accept`/`deny`/`add` on an already-handled request maps to
/// (libpurple's `g_return_if_fail(handled == FALSE)` no-op / CRITICAL, modeled as a `Result`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestError {
    /// The request was already accepted/denied/added; a second decision is rejected.
    AlreadyHandled,
}

// ---------------------------------------------------------------------------
// AuthorizationRequest  (← purpleauthorizationrequest.c)
// ---------------------------------------------------------------------------

/// A remote party's request to be authorized to contact the user (← `PurpleAuthorizationRequest`).
/// `accept()` and `deny(message)` are coupled by a single `handled` flag: the first of either wins
/// and records the decision; any subsequent call is rejected.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizationRequest {
    /// The contact the request is for (the remote requester).
    pub contact: ContactInfo,
    /// The optional message the remote user sent with the request.
    #[serde(default)]
    pub message: Option<String>,
    /// Whether the UI should offer to add the remote user back after accepting.
    #[serde(default)]
    pub add: bool,
    /// Whether `accept`/`deny` has already been called (node-authoritative; a client renders it).
    #[serde(default)]
    handled: bool,
}

impl AuthorizationRequest {
    /// A fresh request for `contact` (`purple_authorization_request_new`): no message, `add` false,
    /// not yet handled.
    pub fn new(contact: ContactInfo) -> Self {
        // RED stub.
        let _ = contact;
        Self::default()
    }

    /// Whether the request has already been accepted or denied.
    pub fn is_handled(&self) -> bool {
        // RED stub.
        false
    }

    /// Accept the request (`purple_authorization_request_accept`). The first `accept`/`deny` wins:
    /// returns `Ok(())` and marks it handled; a subsequent call is `Err(RequestError::AlreadyHandled)`.
    pub fn accept(&mut self) -> Result<(), RequestError> {
        // RED stub.
        Err(RequestError::AlreadyHandled)
    }

    /// Deny the request with an optional message (`purple_authorization_request_deny`). The first
    /// `accept`/`deny` wins: returns `Ok(message)` (echoing the argument, matching the C `denied`
    /// signal's `message` parameter) and marks it handled; a subsequent call is
    /// `Err(RequestError::AlreadyHandled)`.
    pub fn deny(&mut self, message: Option<String>) -> Result<Option<String>, RequestError> {
        // RED stub.
        let _ = message;
        Err(RequestError::AlreadyHandled)
    }
}

// ---------------------------------------------------------------------------
// AddContactRequest  (← purpleaddcontactrequest.c)
// ---------------------------------------------------------------------------

/// A request notifying the user that a remote contact added them, offering to add back
/// (← `PurpleAddContactRequest`). `add()` is single-shot (idempotency like [`AuthorizationRequest`]).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AddContactRequest {
    /// The remote contact that added the user.
    pub contact: ContactInfo,
    /// The optional message the remote user sent.
    #[serde(default)]
    pub message: Option<String>,
    /// Whether `add` has already been called.
    #[serde(default)]
    handled: bool,
}

impl AddContactRequest {
    /// A fresh request for `contact` (`purple_add_contact_request_new`).
    pub fn new(contact: ContactInfo) -> Self {
        // RED stub.
        let _ = contact;
        Self::default()
    }

    /// Whether `add` has already been called.
    pub fn is_handled(&self) -> bool {
        // RED stub.
        false
    }

    /// Tell the UI to add the contact (`purple_add_contact_request_add`). Single-shot: the first
    /// call returns `Ok(())` and marks it handled; a subsequent call is
    /// `Err(RequestError::AlreadyHandled)`.
    pub fn add(&mut self) -> Result<(), RequestError> {
        // RED stub.
        Err(RequestError::AlreadyHandled)
    }
}

// ---------------------------------------------------------------------------
// Notification + typed kinds  (← purplenotification*.c)
// ---------------------------------------------------------------------------

/// The typed payload of a [`NotificationInfo`] (← the `PurpleNotification` subclass hierarchy).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotificationKind {
    /// A plain notification with no typed payload (`purple_notification_new`).
    #[default]
    Generic,
    /// A remote contact added the user (`PurpleNotificationAddContact`).
    AddContact(AddContactRequest),
    /// A remote party requests authorization (`PurpleNotificationAuthorizationRequest`).
    Authorization(AuthorizationRequest),
    /// A clickable link (`PurpleNotificationLink`).
    Link {
        /// The text to show instead of the URI; falls back to `link_uri` when empty
        /// (`purple_notification_link_get_link_text`).
        #[serde(default)]
        link_text: Option<String>,
        /// The link target URI.
        link_uri: String,
    },
    /// An account's connection failed (`PurpleNotificationConnectionError`); account-bound and
    /// transient in the manager's account-scoped removal.
    ConnectionError,
}

/// One notification as the host/GUI sees it (← `PurpleNotification`): a stable id, a creation
/// timestamp, read state, an optional account/transport binding, and a typed [`NotificationKind`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotificationInfo {
    /// The stable id (auto-generated by [`NotificationInfo::new`] when not supplied).
    pub id: String,
    /// The optional account/transport this notification is bound to.
    #[serde(default)]
    pub account: Option<TransportId>,
    /// Unix-millis creation time (`created-timestamp`).
    #[serde(default)]
    pub created_ms: u64,
    /// Whether the notification has been read.
    #[serde(default)]
    pub read: bool,
    /// An optional title (a translated, UI-facing string).
    #[serde(default)]
    pub title: Option<String>,
    /// An optional subtitle.
    #[serde(default)]
    pub subtitle: Option<String>,
    /// The icon-name hint for the UI.
    #[serde(default)]
    pub icon_name: Option<String>,
    /// Whether the notification can be interacted with.
    #[serde(default)]
    pub interactive: bool,
    /// Whether the notification is persistent (not user-dismissable).
    #[serde(default)]
    pub persistent: bool,
    /// The typed payload.
    pub kind: NotificationKind,
    /// Whether `delete` has been called (node-authoritative; drives remove-on-delete).
    #[serde(default)]
    deleted: bool,
}

/// Compute the title for an add-contact notification (`purple_notification_add_contact_update`):
/// includes the remote contact's display name.
pub fn add_contact_title(request: &AddContactRequest) -> String {
    // RED stub.
    let _ = request;
    String::new()
}

/// Compute the title for an authorization-request notification
/// (`purple_notification_authorization_request_update`): includes the remote contact's display name.
pub fn authorization_title(request: &AuthorizationRequest) -> String {
    // RED stub.
    let _ = request;
    String::new()
}

impl NotificationInfo {
    /// A generic notification (`purple_notification_new`): a supplied or auto-generated non-empty
    /// `id`, `created_ms` stamped to now, and [`NotificationKind::Generic`].
    pub fn new(id: Option<String>, title: Option<String>) -> Self {
        // RED stub.
        let _ = (id, title);
        Self::default()
    }

    /// An add-contact notification (`purple_notification_add_contact_new`): the title is derived
    /// from the request's contact and the icon is `contact-new-symbolic`.
    pub fn new_add_contact(id: Option<String>, request: AddContactRequest) -> Self {
        // RED stub.
        let _ = (id, request);
        Self::default()
    }

    /// An authorization-request notification (`purple_notification_authorization_request_new`).
    pub fn new_authorization(id: Option<String>, request: AuthorizationRequest) -> Self {
        // RED stub.
        let _ = (id, request);
        Self::default()
    }

    /// A link notification (`purple_notification_link_new`).
    pub fn new_link(
        id: Option<String>,
        title: impl Into<String>,
        link_text: Option<String>,
        link_uri: impl Into<String>,
    ) -> Self {
        // RED stub.
        let _ = (id, title.into(), link_text, link_uri.into());
        Self::default()
    }

    /// A connection-error notification (`purple_notification_connection_error_new`): account-bound.
    pub fn new_connection_error(id: Option<String>, account: TransportId) -> Self {
        // RED stub.
        let _ = (id, account);
        Self::default()
    }

    /// The link text of a [`NotificationKind::Link`] with null-text fallback
    /// (`purple_notification_link_get_link_text`): `link_text` when non-empty, else `link_uri`.
    /// `None` for a non-link notification.
    pub fn link_text(&self) -> Option<&str> {
        // RED stub.
        None
    }

    /// Sort order by creation time (`purple_notification_compare`).
    pub fn compare(&self, other: &NotificationInfo) -> Ordering {
        // RED stub.
        let _ = other;
        Ordering::Equal
    }

    /// Whether `delete` has been called.
    pub fn is_deleted(&self) -> bool {
        // RED stub.
        false
    }

    /// Mark the notification deleted (`purple_notification_delete`). Single-shot: the first call
    /// returns `Ok(())`; a subsequent call is `Err(RequestError::AlreadyHandled)`.
    pub fn delete(&mut self) -> Result<(), RequestError> {
        // RED stub.
        Err(RequestError::AlreadyHandled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contact(id: &str, display_name: Option<&str>) -> ContactInfo {
        ContactInfo {
            id: id.to_string(),
            display_name: display_name.map(str::to_string),
            ..Default::default()
        }
    }

    // -- AuthorizationRequest (test_authorization_request.c) ----------------

    #[test]
    fn authz_new() {
        // /request-authorization/new: the contact is stored.
        let c = contact("remote", None);
        let request = AuthorizationRequest::new(c.clone());
        assert_eq!(request.contact, c);
        assert_eq!(request.message, None);
        assert!(!request.add);
        assert!(!request.is_handled());
    }

    #[test]
    fn authz_properties() {
        // /request-authorization/properties: contact + message set at construction.
        let c = contact("remote", None);
        let mut request = AuthorizationRequest::new(c.clone());
        request.message = Some("hello friend".into());
        assert_eq!(request.contact, c);
        assert_eq!(request.message.as_deref(), Some("hello friend"));
    }

    #[test]
    fn authz_accept_idempotent() {
        // /request-authorization/accept: first accept Ok; a second is rejected (counter stays 1).
        let mut request = AuthorizationRequest::new(contact("remote", None));
        assert_eq!(request.accept(), Ok(()));
        assert!(request.is_handled());
        assert_eq!(request.accept(), Err(RequestError::AlreadyHandled));
    }

    #[test]
    fn authz_accept_then_deny_rejected() {
        // /request-authorization/accept-deny: after accept, deny is rejected.
        let mut request = AuthorizationRequest::new(contact("remote", None));
        assert_eq!(request.accept(), Ok(()));
        assert_eq!(request.deny(None), Err(RequestError::AlreadyHandled));
    }

    #[test]
    fn authz_deny_idempotent() {
        // /request-authorization/deny: first deny Ok; a second is rejected.
        let mut request = AuthorizationRequest::new(contact("remote", None));
        assert_eq!(request.deny(None), Ok(None));
        assert!(request.is_handled());
        assert_eq!(request.deny(None), Err(RequestError::AlreadyHandled));
    }

    #[test]
    fn authz_deny_then_accept_rejected() {
        // /request-authorization/deny-accept: after deny, accept is rejected.
        let mut request = AuthorizationRequest::new(contact("remote", None));
        assert_eq!(request.deny(None), Ok(None));
        assert_eq!(request.accept(), Err(RequestError::AlreadyHandled));
    }

    #[test]
    fn authz_deny_message_null() {
        // /request-authorization/deny-message/null: the denial message is None.
        let mut request = AuthorizationRequest::new(contact("remote", None));
        assert_eq!(request.deny(None), Ok(None));
    }

    #[test]
    fn authz_deny_message_non_null() {
        // /request-authorization/deny-message/non-null: the denial carries the argument message.
        let mut request = AuthorizationRequest::new(contact("remote", None));
        assert_eq!(
            request.deny(Some("this is a message".into())),
            Ok(Some("this is a message".to_string()))
        );
    }

    // -- AddContactRequest (derived from purpleaddcontactrequest.c) ---------

    #[test]
    fn add_contact_new() {
        let c = contact("remote", None);
        let request = AddContactRequest::new(c.clone());
        assert_eq!(request.contact, c);
        assert_eq!(request.message, None);
        assert!(!request.is_handled());
    }

    #[test]
    fn add_contact_properties() {
        let c = contact("username", None);
        let mut request = AddContactRequest::new(c.clone());
        request.message = Some("hi".into());
        assert_eq!(request.contact, c);
        assert_eq!(request.message.as_deref(), Some("hi"));
    }

    #[test]
    fn add_contact_set_message() {
        let mut request = AddContactRequest::new(contact("remote", None));
        assert_eq!(request.message, None);
        request.message = Some("added you".into());
        assert_eq!(request.message.as_deref(), Some("added you"));
    }

    #[test]
    fn add_contact_add_idempotent() {
        let mut request = AddContactRequest::new(contact("remote", None));
        assert_eq!(request.add(), Ok(()));
        assert!(request.is_handled());
        assert_eq!(request.add(), Err(RequestError::AlreadyHandled));
    }

    // -- NotificationInfo generic (test_notification.c) --------------------

    #[test]
    fn notification_new_generates_id() {
        // /notification/new: id auto-generated (non-empty) + created_ms stamped.
        let n = NotificationInfo::new(None, None);
        assert!(!n.id.is_empty(), "id must be auto-generated");
        assert!(n.created_ms > 0, "created_ms must be stamped");
        assert_eq!(n.kind, NotificationKind::Generic);
    }

    #[test]
    fn notification_properties() {
        // /notification/properties: the modeled property subset round-trips.
        let n = NotificationInfo {
            id: "n1".into(),
            created_ms: 1_700_000_000_000,
            icon_name: Some("icon-name".into()),
            interactive: true,
            persistent: true,
            read: true,
            subtitle: Some("subtitle".into()),
            title: Some("title".into()),
            ..Default::default()
        };
        assert_eq!(n.created_ms, 1_700_000_000_000);
        assert_eq!(n.icon_name.as_deref(), Some("icon-name"));
        assert!(n.interactive);
        assert!(n.persistent);
        assert!(n.read);
        assert_eq!(n.subtitle.as_deref(), Some("subtitle"));
        assert_eq!(n.title.as_deref(), Some("title"));
    }

    #[test]
    fn notification_compare_by_created() {
        // purple_notification_compare: order by created timestamp.
        let a = NotificationInfo {
            id: "a".into(),
            created_ms: 10,
            ..Default::default()
        };
        let b = NotificationInfo {
            id: "b".into(),
            created_ms: 20,
            ..Default::default()
        };
        assert_eq!(a.compare(&b), Ordering::Less);
        assert_eq!(b.compare(&a), Ordering::Greater);
        assert_eq!(a.compare(&a.clone()), Ordering::Equal);
    }

    #[test]
    fn notification_delete_idempotent() {
        let mut n = NotificationInfo::new(None, Some("title".into()));
        assert!(!n.is_deleted());
        assert_eq!(n.delete(), Ok(()));
        assert!(n.is_deleted());
        assert_eq!(n.delete(), Err(RequestError::AlreadyHandled));
    }

    // -- add-contact notification (test_notification_add_contact.c) --------

    #[test]
    fn notification_add_contact_new() {
        let request = AddContactRequest::new(contact("remote", None));
        let n = NotificationInfo::new_add_contact(None, request.clone());
        assert!(matches!(n.kind, NotificationKind::AddContact(_)));
        assert!(!n.id.is_empty());
        if let NotificationKind::AddContact(r) = &n.kind {
            assert_eq!(r, &request);
        }
    }

    #[test]
    fn notification_add_contact_properties() {
        let request = AddContactRequest::new(contact("username", None));
        let n = NotificationInfo::new_add_contact(Some("notification1".into()), request.clone());
        assert_eq!(n.id, "notification1");
        assert!(matches!(&n.kind, NotificationKind::AddContact(r) if r == &request));
    }

    #[test]
    fn notification_add_contact_title() {
        // /notification/add-contact/updates-title (title-derivation half): the remote name is in
        // the title, and recomputing after the contact name changes yields the new name.
        let mut request = AddContactRequest::new(contact("remote-username", None));
        let n = NotificationInfo::new_add_contact(None, request.clone());
        let title = n.title.clone().unwrap_or_default();
        assert!(
            title.contains("remote-username"),
            "title {title:?} must contain the remote name"
        );
        // Alias/display-name change is reflected on recompute.
        request.contact.display_name = Some("test-alias".into());
        let title2 = add_contact_title(&request);
        assert!(
            title2.contains("test-alias"),
            "recomputed title {title2:?} must contain the new name"
        );
    }

    // -- authorization notification (test_notification_authorization_request.c) --

    #[test]
    fn notification_authz_new() {
        let request = AuthorizationRequest::new(contact("remote", None));
        let n = NotificationInfo::new_authorization(Some("id".into()), request.clone());
        assert_eq!(n.id, "id");
        assert!(matches!(&n.kind, NotificationKind::Authorization(r) if r == &request));
    }

    #[test]
    fn notification_authz_properties() {
        let request = AuthorizationRequest::new(contact("remote", None));
        let n = NotificationInfo::new_authorization(Some("id1".into()), request.clone());
        assert_eq!(n.id, "id1");
        assert!(matches!(&n.kind, NotificationKind::Authorization(r) if r == &request));
    }

    #[test]
    fn notification_authz_title() {
        let mut request = AuthorizationRequest::new(contact("remote", Some("Remote User")));
        let n = NotificationInfo::new_authorization(None, request.clone());
        let title = n.title.clone().unwrap_or_default();
        assert!(
            title.contains("Remote User"),
            "title {title:?} must contain the remote name"
        );
        request.contact.display_name = Some("foo".into());
        let title2 = authorization_title(&request);
        assert!(title2.contains("foo"), "recomputed title {title2:?}");
    }

    // -- link notification (test_notification_link.c) ----------------------

    #[test]
    fn notification_link_new() {
        let n = NotificationInfo::new_link(None, "insert title", None, "https://pidgin.im/");
        assert!(matches!(n.kind, NotificationKind::Link { .. }));
        assert_eq!(n.title.as_deref(), Some("insert title"));
    }

    #[test]
    fn notification_link_properties() {
        let n = NotificationInfo::new_link(
            Some("notification1".into()),
            "title1",
            Some("pidgin.im".into()),
            "https://pidgin.im/",
        );
        assert_eq!(n.id, "notification1");
        assert_eq!(n.title.as_deref(), Some("title1"));
        assert_eq!(n.link_text(), Some("pidgin.im"));
        if let NotificationKind::Link { link_uri, .. } = &n.kind {
            assert_eq!(link_uri, "https://pidgin.im/");
        } else {
            panic!("expected a link kind");
        }
    }

    #[test]
    fn notification_link_null_link_text() {
        // /notification/link/null-link-text: null link_text falls back to link_uri.
        let n = NotificationInfo::new_link(None, "title1", None, "https://pidgin.im/");
        assert_eq!(n.link_text(), Some("https://pidgin.im/"));
    }

    // -- connection-error notification (derived) ---------------------------

    #[test]
    fn notification_connection_error_new() {
        let account = TransportId::new("matrix/@me:hs.org");
        let n = NotificationInfo::new_connection_error(None, account.clone());
        assert_eq!(n.kind, NotificationKind::ConnectionError);
        assert_eq!(n.account, Some(account));
    }

    // -- wire round-trip (daemon-native) -----------------------------------

    #[test]
    fn notification_info_cbor_round_trips() {
        let n = NotificationInfo::new_authorization(
            Some("n1".into()),
            AuthorizationRequest::new(contact("remote", Some("Remote"))),
        );
        let bytes = crate::to_cbor(&n);
        let back: NotificationInfo = crate::from_cbor(&bytes).unwrap();
        assert_eq!(n, back);
    }
}
