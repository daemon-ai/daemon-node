// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `PresenceManager` — the host-side saved-presence surface (work package W2-F).
//!
//! Ported from libpurple's `PurplePresenceManager` (`purplepresencemanager.c`): an ordered
//! collection of [`SavedPresence`]s that always guarantees a default **Offline** and **Available**
//! presence, supports add / remove / lookup (by id or name) / set-active / list, and bumps the
//! active presence's use-count + last-used on activation.
//!
//! libpurple's manager is a `GListModel` backed by an async GObject backend (bound to GSettings for
//! the active id). The daemon's backend seam is the durable [`SessionStore`]: `add`→
//! `saved_presence_set`, `remove`→`saved_presence_remove`, load→`saved_presence_list` + default
//! seeding, and the active id is a single-row store setting. The store keeps the presence as opaque
//! CBOR (protocol-free, mirroring [`CronOps`](crate::cron::CronOps)); this surface owns the
//! wire<->store (de)serialization and the id/default discipline.

use daemon_api::{from_cbor, to_cbor, ApiError, PresencePrimitive, SavedPresence};
use daemon_store::{SessionStore, StoredSavedPresence};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// The stable id of the always-present default **Offline** saved presence
/// (← `PURPLE_PRESENCE_MANAGER_DEFAULT_OFFLINE_ID`).
pub const DEFAULT_OFFLINE_ID: &str = "00000000-0000-0000-0000-000000000000";
/// The stable id of the always-present default **Available** saved presence
/// (← `PURPLE_PRESENCE_MANAGER_DEFAULT_AVAILABLE_ID`).
pub const DEFAULT_AVAILABLE_ID: &str = "ffffffff-ffff-ffff-ffff-ffffffffffff";

/// The host-side saved-presence manager over a durable [`SessionStore`].
#[derive(Clone)]
pub struct PresenceManager {
    store: Arc<dyn SessionStore>,
}

impl PresenceManager {
    /// A manager over `store`. Call [`PresenceManager::load`] once to seed the default presences
    /// (the async analogue of libpurple's backend load).
    pub fn new(store: Arc<dyn SessionStore>) -> Self {
        Self { store }
    }

    fn now_unix() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Load from the backend, seeding the default Offline + Available presences when absent
    /// (← the manager's backend-load callback + `add_default_{offline,available}`). Idempotent.
    pub async fn load(&self) {
        // TDD red stub: no seeding.
    }

    /// Every saved presence, in insertion order (← the `GListModel`).
    pub async fn list(&self) -> Vec<SavedPresence> {
        // TDD red stub.
        Vec::new()
    }

    /// Add a saved presence (← `purple_presence_manager_add`): mints an id when unset, rejects a
    /// duplicate id (returns `Ok(false)`), else persists it and returns `Ok(true)`.
    pub async fn add(&self, _presence: SavedPresence) -> Result<bool, ApiError> {
        // TDD red stub.
        Ok(false)
    }

    /// Upsert a saved presence (the wire `PresenceSave` seam): mints an id when unset, then
    /// persists — creating a new presence or replacing an existing one by id.
    pub async fn save(&self, _presence: SavedPresence) -> Result<(), ApiError> {
        // TDD red stub.
        Err(ApiError::Unsupported("presence_save".into()))
    }

    /// Remove a saved presence by id (← `purple_presence_manager_remove`); returns whether one was
    /// removed.
    pub async fn remove(&self, _id: &str) -> Result<bool, ApiError> {
        // TDD red stub.
        Ok(false)
    }

    /// Find a saved presence by id (← `purple_presence_manager_find_with_id`).
    pub async fn find_with_id(&self, _id: &str) -> Option<SavedPresence> {
        // TDD red stub.
        None
    }

    /// Find the first saved presence whose name equals `name` (daemon-native lookup-by-name).
    pub async fn find_with_name(&self, _name: &str) -> Option<SavedPresence> {
        // TDD red stub.
        None
    }

    /// Set the active saved presence by id (← `purple_presence_manager_set_active_from_id`), bumping
    /// its use-count and stamping its last-used time (daemon-native activation bookkeeping).
    pub async fn set_active(&self, _id: &str) -> Result<(), ApiError> {
        // TDD red stub.
        Err(ApiError::Unsupported("presence_set_active".into()))
    }

    /// The active saved presence, if one has been set (← `purple_presence_manager_get_active`).
    pub async fn active(&self) -> Option<SavedPresence> {
        // TDD red stub.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_store::InMemoryStore;

    async fn manager() -> PresenceManager {
        let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
        let m = PresenceManager::new(store);
        m.load().await;
        m
    }

    // -- /presence-manager/new ---------------------------------------------

    #[tokio::test]
    #[ignore = "TDD red: pending impl"]
    async fn mgr_new_has_two_defaults() {
        // ← /presence-manager/new: a fresh manager has exactly the 2 default presences.
        let m = manager().await;
        let list = m.list().await;
        assert_eq!(list.len(), 2, "the two default presences are seeded");
        let offline = m.find_with_id(DEFAULT_OFFLINE_ID).await.expect("offline");
        assert_eq!(offline.primitive, PresencePrimitive::Offline);
        let available = m
            .find_with_id(DEFAULT_AVAILABLE_ID)
            .await
            .expect("available");
        assert_eq!(available.primitive, PresencePrimitive::Available);
    }

    // -- /presence-manager/add-remove --------------------------------------

    #[tokio::test]
    #[ignore = "TDD red: pending impl"]
    async fn mgr_add_remove() {
        // ← /presence-manager/add-remove.
        let m = manager().await;
        let mut p = SavedPresence::new(PresencePrimitive::Idle);
        p.name = Some("test presence".into());
        let id = p.id.clone();

        assert!(m.add(p).await.expect("add"), "a fresh id is added");
        assert_eq!(m.list().await.len(), 3);
        assert!(m.find_with_id(&id).await.is_some());

        assert!(
            m.remove(&id).await.expect("remove"),
            "an existing id removes"
        );
        assert_eq!(m.list().await.len(), 2);
        assert!(m.find_with_id(&id).await.is_none());
        // Double remove is a no-op (returns false).
        assert!(!m.remove(&id).await.expect("remove again"));
    }

    #[tokio::test]
    #[ignore = "TDD red: pending impl"]
    async fn mgr_add_rejects_duplicate() {
        // ← purple_presence_manager_add returning FALSE on a duplicate id.
        let m = manager().await;
        let p = SavedPresence::new(PresencePrimitive::Away);
        assert!(m.add(p.clone()).await.expect("first add"));
        assert!(
            !m.add(p).await.expect("dup add"),
            "a duplicate id is rejected"
        );
    }

    // -- backend-normal ports (the store persistence seam) -----------------

    #[tokio::test]
    #[ignore = "TDD red: pending impl"]
    async fn mgr_add_persists_to_store() {
        // ← /presence-manager-backend-normal/save-saved-presence: add persists to the store backend.
        let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
        let m = PresenceManager::new(store.clone());
        m.load().await;
        let p = SavedPresence::new(PresencePrimitive::Streaming);
        let id = p.id.clone();
        m.add(p).await.expect("add");
        assert!(
            store.saved_presence_list().await.iter().any(|s| s.id == id),
            "the added presence is durably persisted"
        );
    }

    #[tokio::test]
    #[ignore = "TDD red: pending impl"]
    async fn mgr_remove_deletes_from_store() {
        // ← /presence-manager-backend-normal/delete-saved-presence: remove deletes from the backend.
        let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
        let m = PresenceManager::new(store.clone());
        m.load().await;
        let p = SavedPresence::new(PresencePrimitive::Streaming);
        let id = p.id.clone();
        m.add(p).await.expect("add");
        m.remove(&id).await.expect("remove");
        assert!(
            !store.saved_presence_list().await.iter().any(|s| s.id == id),
            "the removed presence is gone from the store"
        );
    }

    #[tokio::test]
    #[ignore = "TDD red: pending impl"]
    async fn mgr_new_loads_from_store() {
        // ← /presence-manager-backend-normal/load-saved-presences: a new manager loads persisted
        // presences from the backend (plus seeds the two defaults).
        let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
        let seeded = SavedPresence::new(PresencePrimitive::DoNotDisturb);
        let seeded_id = seeded.id.clone();
        store
            .saved_presence_set(StoredSavedPresence {
                id: seeded_id.clone(),
                payload: to_cbor(&seeded),
            })
            .await
            .expect("seed");

        let m = PresenceManager::new(store);
        m.load().await;
        // The seeded presence plus the two defaults.
        assert_eq!(m.list().await.len(), 3);
        assert!(m.find_with_id(&seeded_id).await.is_some());
    }

    // -- derived: lookup-by-name + activation bookkeeping ------------------

    #[tokio::test]
    #[ignore = "TDD red: pending impl"]
    async fn mgr_find_by_name() {
        let m = manager().await;
        let mut p = SavedPresence::new(PresencePrimitive::Away);
        p.name = Some("Lunch".into());
        m.add(p).await.expect("add");
        assert!(m.find_with_name("Lunch").await.is_some());
        assert!(m.find_with_name("nope").await.is_none());
    }

    #[tokio::test]
    #[ignore = "TDD red: pending impl"]
    async fn mgr_set_active_bumps_use_count_and_last_used() {
        let m = manager().await;
        // Unknown id errors.
        assert!(m.set_active("nope").await.is_err());

        m.set_active(DEFAULT_AVAILABLE_ID).await.expect("activate");
        let active = m.active().await.expect("active");
        assert_eq!(active.id, DEFAULT_AVAILABLE_ID);
        assert_eq!(active.use_count, 1, "first activation bumps the use-count");
        assert!(active.last_used.is_some(), "activation stamps last-used");

        // A second activation bumps again.
        m.set_active(DEFAULT_AVAILABLE_ID).await.expect("activate");
        assert_eq!(m.active().await.expect("active").use_count, 2);
    }

    #[tokio::test]
    #[ignore = "TDD red: pending impl"]
    async fn mgr_active_persists_across_reload() {
        let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
        let m = PresenceManager::new(store.clone());
        m.load().await;
        m.set_active(DEFAULT_OFFLINE_ID).await.expect("activate");

        // A fresh manager over the same store recovers the active id.
        let m2 = PresenceManager::new(store);
        m2.load().await;
        assert_eq!(m2.active().await.expect("active").id, DEFAULT_OFFLINE_ID);
    }
}
