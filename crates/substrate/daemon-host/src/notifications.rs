// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The node-side notification manager — the single live collection behind the notification surface
//! ([`ControlApi::notification_list`](daemon_api::ControlApi::notification_list) /
//! the [`NodeEvent::NotificationsChanged`](daemon_api::NodeEvent) pointer), ported from libpurple's
//! `purplenotificationmanager.c`.
//!
//! It owns an ordered collection of [`NotificationInfo`]s (newest first, like the C
//! prepend-and-let-the-UI-sort model) plus the unread-count accounting. Mutations return a typed
//! outcome the caller acts on (the daemon analog of the C `added`/`removed`/`read`/`unread` signals
//! + the `unread-count` property notify): there are no GObject signals, so the manager reports
//! transitions by value.
//!
//! Faithful-to-C accounting note: only `add`, single `remove`, and `set_read` adjust `unread_count`
//! (mirroring `purplenotificationmanager.c`); `remove_with_account` and `clear` intentionally do
//! **not** touch it, matching the C implementation.

use daemon_api::{NotificationInfo, NotificationKind};
use daemon_protocol::TransportId;

/// The outcome of [`NotificationManager::add`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddOutcome {
    /// The notification was added (prepended).
    Added,
    /// A notification with the same id already exists; the add was rejected (the C double-add
    /// `g_warning`, modeled as a rejected no-op).
    DuplicateRejected,
}

/// The transition [`NotificationManager::set_read`] reports (the daemon analog of the C
/// `read`/`unread` signals + the `unread-count` property notify).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadChange {
    /// The notification went unread → read (unread count decremented).
    MarkedRead,
    /// The notification went read → unread (unread count incremented).
    MarkedUnread,
    /// No state change (the read flag already matched, or the id was unknown).
    Unchanged,
}

/// The live notification collection + unread accounting (← `PurpleNotificationManager`).
#[derive(Debug, Default)]
pub struct NotificationManager {
    /// Newest first (index 0), matching the C prepend model.
    notifications: Vec<NotificationInfo>,
    unread_count: u32,
}

impl NotificationManager {
    /// An empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of notifications.
    pub fn len(&self) -> usize {
        self.notifications.len()
    }

    /// Whether the manager holds no notifications.
    pub fn is_empty(&self) -> bool {
        self.notifications.is_empty()
    }

    /// The current unread count.
    pub fn unread_count(&self) -> u32 {
        self.unread_count
    }

    /// A snapshot of the notifications (newest first) — what the `notification_list` op returns.
    pub fn list(&self) -> Vec<NotificationInfo> {
        self.notifications.clone()
    }

    /// Add a notification (`purple_notification_manager_add`): prepend it (newest first), reject a
    /// double-add by id, and increment `unread_count` when it is unread.
    pub fn add(&mut self, notification: NotificationInfo) -> AddOutcome {
        if self.notifications.iter().any(|n| n.id == notification.id) {
            return AddOutcome::DuplicateRejected;
        }
        if !notification.read {
            self.unread_count = self.unread_count.saturating_add(1);
        }
        self.notifications.insert(0, notification);
        AddOutcome::Added
    }

    /// Remove a notification by id (`purple_notification_manager_remove`): decrement `unread_count`
    /// when the removed item was unread. Returns whether one was removed (a second remove is a no-op).
    pub fn remove(&mut self, id: &str) -> bool {
        let Some(pos) = self.notifications.iter().position(|n| n.id == id) else {
            return false;
        };
        let removed = self.notifications.remove(pos);
        if !removed.read {
            self.unread_count = self.unread_count.saturating_sub(1);
        }
        true
    }

    /// Remove every notification bound to `account`
    /// (`purple_notification_manager_remove_with_account`). A [`NotificationKind::ConnectionError`]
    /// notification is transient — removed only when `all == true`. Returns the removed count.
    /// (Faithful to C: does not adjust `unread_count`.)
    pub fn remove_with_account(&mut self, account: &TransportId, all: bool) -> usize {
        let before = self.notifications.len();
        self.notifications.retain(|n| {
            let matches = n.account.as_ref() == Some(account);
            let can_remove = if is_transient(n) { all } else { true };
            !(matches && can_remove)
        });
        before - self.notifications.len()
    }

    /// Remove every notification (`purple_notification_manager_clear`). (Faithful to C: does not
    /// adjust `unread_count`.)
    pub fn clear(&mut self) {
        self.notifications.clear();
    }

    /// Set a notification's read flag by id, updating `unread_count` and reporting the transition
    /// (the daemon analog of the manager's `notify::read` callback firing `read`/`unread`).
    pub fn set_read(&mut self, id: &str, read: bool) -> ReadChange {
        let Some(notification) = self.notifications.iter_mut().find(|n| n.id == id) else {
            return ReadChange::Unchanged;
        };
        if notification.read == read {
            return ReadChange::Unchanged;
        }
        notification.read = read;
        if read {
            self.unread_count = self.unread_count.saturating_sub(1);
            ReadChange::MarkedRead
        } else {
            self.unread_count = self.unread_count.saturating_add(1);
            ReadChange::MarkedUnread
        }
    }

    /// Remove a notification as if it were deleted (`purple_notification_delete` → the manager's
    /// `deleted` callback → `remove`). Returns whether one was removed.
    pub fn delete(&mut self, id: &str) -> bool {
        self.remove(id)
    }
}

/// Whether `notification` is transient for account-scoped removal (a connection error is only
/// removed when `all` is requested; everything else is always removable).
fn is_transient(notification: &NotificationInfo) -> bool {
    matches!(notification.kind, NotificationKind::ConnectionError)
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_api::{AuthorizationRequest, ContactInfo};

    fn generic() -> NotificationInfo {
        NotificationInfo::new(None, None)
    }

    fn account(name: &str) -> TransportId {
        TransportId::new(name)
    }

    fn with_account(acct: &TransportId) -> NotificationInfo {
        let mut n = NotificationInfo::new(None, None);
        n.account = Some(acct.clone());
        n
    }

    fn connection_error(acct: &TransportId) -> NotificationInfo {
        NotificationInfo::new_connection_error(None, acct.clone())
    }

    // /notification-manager/add-remove
    #[test]
    fn manager_add_remove() {
        let mut m = NotificationManager::new();
        let n = generic();
        let id = n.id.clone();
        assert_eq!(m.add(n), AddOutcome::Added);
        assert_eq!(m.unread_count(), 1);
        assert!(m.remove(&id));
        assert_eq!(m.unread_count(), 0);
        assert!(m.is_empty());
    }

    // /notification-manager/double-add
    #[test]
    fn manager_double_add_rejected() {
        let mut m = NotificationManager::new();
        let n = generic();
        let dup = n.clone();
        assert_eq!(m.add(n), AddOutcome::Added);
        assert_eq!(m.add(dup), AddOutcome::DuplicateRejected);
        assert_eq!(m.len(), 1);
    }

    // /notification-manager/double-remove
    #[test]
    fn manager_double_remove() {
        let mut m = NotificationManager::new();
        let n = generic();
        let id = n.id.clone();
        m.add(n);
        assert!(m.remove(&id));
        assert!(!m.remove(&id));
        assert!(m.is_empty());
    }

    // /notification-manager/remove-with-account/simple
    #[test]
    fn manager_remove_with_account_simple() {
        let mut m = NotificationManager::new();
        let acct = account("test");

        // Nothing happens on an empty list.
        assert_eq!(m.remove_with_account(&acct, true), 0);
        assert_eq!(m.len(), 0);

        // A single notification WITHOUT the account is not removed.
        m.add(generic());
        assert_eq!(m.remove_with_account(&acct, true), 0);
        assert_eq!(m.len(), 1);
        m.clear();

        // A single notification WITH the account is removed.
        m.add(with_account(&acct));
        assert_eq!(m.remove_with_account(&acct, true), 1);
        assert_eq!(m.len(), 0);
    }

    // /notification-manager/remove-with-account/mixed
    #[test]
    fn manager_remove_with_account_mixed() {
        let mut m = NotificationManager::new();
        let accounts = [
            account("account1"),
            account("account2"),
            account("account3"),
        ];
        let pattern = [0, 0, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 0, 0];
        for &p in &pattern {
            m.add(with_account(&accounts[p]));
        }
        assert_eq!(m.len(), 14);

        assert_eq!(m.remove_with_account(&accounts[0], true), 8);
        assert_eq!(m.len(), 6);
        assert_eq!(m.remove_with_account(&accounts[1], true), 4);
        assert_eq!(m.len(), 2);
        assert_eq!(m.remove_with_account(&accounts[2], true), 2);
        assert_eq!(m.len(), 0);
    }

    // /notification-manager/remove-with-account/all
    #[test]
    fn manager_remove_with_account_all() {
        let mut m = NotificationManager::new();
        let acct = account("test");

        assert_eq!(m.remove_with_account(&acct, true), 0);
        assert_eq!(m.len(), 0);

        // In order: generic+acct, connection-error+acct, generic+acct, generic (no account).
        m.add(with_account(&acct));
        m.add(connection_error(&acct));
        m.add(with_account(&acct));
        m.add(generic());
        assert_eq!(m.len(), 4);

        // all=false leaves the transient connection-error (and the no-account generic).
        assert_eq!(m.remove_with_account(&acct, false), 2);
        assert_eq!(m.len(), 2);
        // The second item is the connection error.
        let list = m.list();
        assert_eq!(list[1].kind, NotificationKind::ConnectionError);

        // all=true removes the connection error too.
        assert_eq!(m.remove_with_account(&acct, true), 1);
        assert_eq!(m.len(), 1);

        m.clear();
        assert_eq!(m.len(), 0);
    }

    // /notification-manager/read-propagation
    #[test]
    fn manager_read_propagation() {
        let mut m = NotificationManager::new();
        let n = generic();
        let id = n.id.clone();
        m.add(n);
        assert_eq!(m.unread_count(), 1);

        // Mark read.
        assert_eq!(m.set_read(&id, true), ReadChange::MarkedRead);
        assert_eq!(m.unread_count(), 0);
        // Idempotent (already read).
        assert_eq!(m.set_read(&id, true), ReadChange::Unchanged);
        assert_eq!(m.unread_count(), 0);

        // Mark unread.
        assert_eq!(m.set_read(&id, false), ReadChange::MarkedUnread);
        assert_eq!(m.unread_count(), 1);
    }

    // /notification-manager/remove-on-delete
    #[test]
    fn manager_remove_on_delete() {
        let mut m = NotificationManager::new();
        let n = NotificationInfo::new(None, Some("title".into()));
        let id = n.id.clone();
        m.add(n);
        assert_eq!(m.len(), 1);
        assert!(m.delete(&id));
        assert_eq!(m.len(), 0);
    }

    // A typed (authorization) notification is carried through list() intact.
    #[test]
    fn manager_lists_typed_notifications() {
        let mut m = NotificationManager::new();
        let request = AuthorizationRequest::new(ContactInfo {
            id: "remote".into(),
            ..Default::default()
        });
        m.add(NotificationInfo::new_authorization(None, request));
        let list = m.list();
        assert_eq!(list.len(), 1);
        assert!(matches!(list[0].kind, NotificationKind::Authorization(_)));
    }
}
