// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The live-connection revocation registry (Cluster F, Part A).
//!
//! A store mutation (`session_revoke`, `user_disable`, role/password change) invalidates the
//! *reconnect* fast-path (the deleted `auth_sessions` row), but a mux connection already open
//! snapshots its [`Principal`](daemon_auth::Principal) for its whole lifetime — so it would keep
//! issuing `Call`/`Open` (and streaming another owner's transcript) with the revoked identity. This
//! registry closes that: each authenticated connection captures a per-principal *revocation epoch*
//! at auth time ([`SessionRevocations::guard`]); an admin op bumps that principal's epoch
//! ([`SessionRevocations::revoke`]); the transport observes the mismatch and tears the live
//! connection (and its live stream pumps) down.
//!
//! **In-memory by design.** A live connection is a this-process artifact; the durable store stays
//! the source of truth for reconnect. After a restart there are no live connections, so an epoch
//! reset to 0 is correct.
//!
//! **Per-principal, never global.** The counter is keyed by `user_id`, so revoking user X never
//! tears down user Y's connections.
//!
//! **Local trust is deliberately non-revocable.** The unix local-trust / FFI path captures no
//! guard (see the transport), so [`RevocationGuard::is_revoked`] is only ever consulted for
//! network-authenticated connections.
//!
//! Concurrency: the `users` map lock is a **leaf** — held only for the get-or-create of a per-user
//! cell, never across an `.await`, and never while any other lock (notably the `AuthStore` SQLite
//! connection mutex) is held. The admin handler performs the store mutation first (releasing the
//! store lock) and *then* calls [`revoke`](SessionRevocations::revoke), so the epoch bump can never
//! be observed while a store lock is held. The epoch itself is an atomic; the `Notify` is only a
//! prompt-wake optimization (the atomic is authoritative, so a missed wake still tears down at the
//! connection's next keepalive tick).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

/// Invalidate every outstanding credential lease for a profile (Cluster F, Part B). Implemented by
/// the credential broker ([`MultiProfileStoreBroker`](crate::credentials::MultiProfileStoreBroker)):
/// bumps the profile authority's lease epoch (so a stale-epoch lease is refused at
/// `use_capability`) and drops any retained `Proxied` key. The `CredentialApi` handlers call this
/// after mutating the credential store, so removing/replacing a credential tears down the leases
/// minted against the old material.
pub trait CredentialRevoker: Send + Sync {
    /// Revoke all outstanding leases the authority for `profile` has minted.
    fn revoke_profile(&self, profile: &str);
}

/// One principal's revocation state: a monotonically-increasing epoch plus a wake channel for the
/// connections/pumps currently bound to it.
struct UserRevocation {
    epoch: AtomicU64,
    notify: Notify,
}

/// The node-wide, per-principal revocation registry. Cheap to share (`Arc`); one instance is wired
/// into both the [`Authenticator`](crate::authn::Authenticator) (connections *capture* an epoch)
/// and the [`NodeApiImpl`](crate::node_api::NodeApiImpl) admin handlers (which *bump* it).
#[derive(Default)]
pub struct SessionRevocations {
    users: Mutex<HashMap<String, Arc<UserRevocation>>>,
}

impl SessionRevocations {
    /// An empty registry.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Get (or lazily create) the cell for `user_id`. Holds the leaf `users` lock only for the
    /// map lookup/insert.
    fn cell(&self, user_id: &str) -> Arc<UserRevocation> {
        let mut map = self.users.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cell) = map.get(user_id) {
            return cell.clone();
        }
        let cell = Arc::new(UserRevocation {
            epoch: AtomicU64::new(0),
            notify: Notify::new(),
        });
        map.insert(user_id.to_string(), cell.clone());
        cell
    }

    /// Capture `user_id`'s current epoch as a [`RevocationGuard`]. The connection holds this for its
    /// lifetime and clones it into each stream pump it spawns.
    pub fn guard(&self, user_id: &str) -> RevocationGuard {
        let cell = self.cell(user_id);
        let at = cell.epoch.load(Ordering::Acquire);
        RevocationGuard { cell, at }
    }

    /// Revoke every live connection/lease for `user_id`: bump the epoch (so captured guards read
    /// stale) and wake anyone parked on it. Idempotent-safe to call repeatedly.
    ///
    /// MUST be called *after* the caller's store mutation has committed and released its lock — the
    /// bump takes only the leaf `users` lock, never nested under the store lock.
    pub fn revoke(&self, user_id: &str) {
        let cell = self.cell(user_id);
        cell.epoch.fetch_add(1, Ordering::AcqRel);
        cell.notify.notify_waiters();
    }

    /// The current epoch for `user_id` (test/observability).
    #[cfg(test)]
    pub fn epoch(&self, user_id: &str) -> u64 {
        self.cell(user_id).epoch.load(Ordering::Acquire)
    }
}

/// A captured revocation epoch for one principal. Cheap to clone (`Arc` + `u64`) so a connection can
/// hand a copy to each stream pump it spawns; all clones observe the same cell.
#[derive(Clone)]
pub struct RevocationGuard {
    cell: Arc<UserRevocation>,
    at: u64,
}

impl RevocationGuard {
    /// Whether this principal has been revoked since the guard was captured.
    pub fn is_revoked(&self) -> bool {
        self.cell.epoch.load(Ordering::Acquire) != self.at
    }

    /// Resolve once this principal is revoked. Lost-wakeup-safe: the `Notified` future is created
    /// before each re-check of the authoritative atomic, so a bump racing the `.await` is never
    /// missed.
    pub async fn revoked(&self) {
        loop {
            if self.is_revoked() {
                return;
            }
            let notified = self.cell.notify.notified();
            if self.is_revoked() {
                return;
            }
            notified.await;
            if self.is_revoked() {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_flips_after_revoke_for_that_user_only() {
        let reg = SessionRevocations::new();
        let alice = reg.guard("alice");
        let bob = reg.guard("bob");
        assert!(!alice.is_revoked());
        assert!(!bob.is_revoked());

        reg.revoke("alice");
        assert!(alice.is_revoked(), "alice's guard reads revoked");
        assert!(
            !bob.is_revoked(),
            "bob is unaffected (per-principal, not global)"
        );

        // A guard captured *after* the bump is fresh again (a re-login is not pre-revoked).
        let alice2 = reg.guard("alice");
        assert!(!alice2.is_revoked());
    }

    #[test]
    fn repeated_revoke_keeps_prior_guards_stale() {
        let reg = SessionRevocations::new();
        let g = reg.guard("carol");
        reg.revoke("carol");
        reg.revoke("carol");
        assert!(g.is_revoked());
        assert_eq!(reg.epoch("carol"), 2);
    }

    #[tokio::test]
    async fn revoked_future_completes_after_a_concurrent_revoke() {
        let reg = SessionRevocations::new();
        let g = reg.guard("dave");
        let reg2 = reg.clone();
        let waiter = tokio::spawn(async move { g.revoked().await });
        // Give the waiter a chance to park on the notify, then revoke.
        tokio::task::yield_now().await;
        reg2.revoke("dave");
        tokio::time::timeout(std::time::Duration::from_secs(5), waiter)
            .await
            .expect("revoked() must complete promptly after revoke")
            .expect("waiter task");
    }

    #[tokio::test]
    async fn revoked_future_returns_immediately_when_already_stale() {
        let reg = SessionRevocations::new();
        let g = reg.guard("erin");
        reg.revoke("erin");
        // Already stale: must not block.
        tokio::time::timeout(std::time::Duration::from_secs(5), g.revoked())
            .await
            .expect("an already-revoked guard resolves immediately");
    }
}
