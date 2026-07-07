// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `ProfileOps` — the shared profile create/validate/persist/version surface behind both
//! [`NodeApiImpl`](crate::NodeApiImpl)'s operator `ProfileApi` create/update ops and the agent-facing
//! `profile_manage` tool.
//!
//! It owns the durable [`ProfileStore`] CRUD plus the revision-log recording, and defers the
//! engine/inference validation to an injected [`ProfileValidator`] — the exact `validate_engine` +
//! `validate_inference` seam the operator create path already runs (implemented by `NodeApiImpl`,
//! which owns the agent catalog / model / credential handles validation consults). A single surface
//! means the operator (`profile_create`/`profile_update`) and the agent tool author profiles through
//! **one** validation + persistence + revision path — "one engine, not two" — mirroring how
//! [`CronOps`](crate::CronOps) backs both the operator cron ops and the agent `cron` tool.
//!
//! The provenance differs only in the [`Author`] the caller passes: the operator path records
//! [`Author::Operator`]; the tool records `Author::Agent("profile_manage")`, exactly as
//! `SkillStore.default_author` records agent-authored skill edits.

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use daemon_api::{ApiError, ProfileSpec};
use daemon_common::{Author, RevisionKind, RevisionLog};

use crate::node_api::profile_err;
use crate::profiles::ProfileStore;

/// The engine + inference validation a profile spec must pass before it is persisted (create/update).
/// Injected into [`ProfileOps`] so the shared surface runs the **same** validation the operator path
/// runs, without `ProfileOps` re-plumbing the agent-catalog / model / credential handles the checks
/// consult. Implemented by [`NodeApiImpl`](crate::NodeApiImpl) (`validate_engine` +
/// `validate_inference`); late-bound after the node is `Arc`-wrapped (the validator IS the node).
#[async_trait]
pub trait ProfileValidator: Send + Sync {
    /// Validate `spec`'s engine binding (a `Foreign{agent}` must name an installed catalog agent; a
    /// `NodeProvider` backend must name a serveable model/credential) and its inference selector (a
    /// local provider must name a model). Returns the wire [`ApiError`] on rejection.
    async fn validate_profile(&self, spec: &ProfileSpec) -> Result<(), ApiError>;
}

/// The node-wide "profiles changed" pointer sink (Phase 3): a successful create/update/delete emits
/// through it so a thin client refetches the profile list without polling. Injected so [`ProfileOps`]
/// (which backs BOTH the operator ops and the agent tool) is the single emit point for profile
/// authoring; implemented by the node's [`NodeEventFeed`](crate::NodeEventFeed) (emits a coalesced
/// `NodeEvent::ProfilesChanged`). Unset => no feed wired (tests / a minimal node) — a no-op.
pub trait ProfileEvents: Send + Sync {
    /// Note a profile author/delete happened and emit the node-wide `ProfilesChanged` pointer.
    fn profiles_changed(&self);
}

/// Recover the owning agent SESSION id from an agent-namespaced profile id (`agent/{session}/{name}`):
/// the session is everything between the `agent/` prefix and the final `/{name}` segment (a session
/// id may itself embed lineage, e.g. `s1/c2`). `None` for an operator profile (no `agent/` prefix) or
/// a malformed id — the stamped `owner`. Mirrors the tool's `parse_authoring_session` (the id
/// convention is the shared seam), kept here so provenance is stamped authoritatively at the store
/// layer regardless of caller.
fn owner_from_id(id: &str) -> Option<String> {
    let rest = id.strip_prefix("agent/")?;
    let (session, name) = rest.rsplit_once('/')?;
    if session.is_empty() || name.is_empty() {
        return None;
    }
    Some(session.to_string())
}

/// The shared profile operations surface over a durable [`ProfileStore`] (+ optional revision log).
pub struct ProfileOps {
    store: Arc<dyn ProfileStore>,
    /// The append-only revision history; a create/update appends a snapshot attributed to the
    /// caller's [`Author`]. `None` => versioning is off (the mutation still persists).
    revisions: Option<Arc<dyn RevisionLog>>,
    /// The injected engine/inference validation seam. Late-bound (the validator is the assembled
    /// `NodeApiImpl`, built after this surface). Unset => validation is skipped (store-only), so a
    /// minimal node that wires no validator still persists — never a *second* validation path.
    validator: OnceLock<Arc<dyn ProfileValidator>>,
    /// The node-wide `ProfilesChanged` emit sink (Phase 3). `None` => no feed wired (a no-op).
    events: Option<Arc<dyn ProfileEvents>>,
}

impl ProfileOps {
    /// A profile surface over `store` with no revision log and no validator wired.
    pub fn new(store: Arc<dyn ProfileStore>) -> Self {
        Self {
            store,
            revisions: None,
            validator: OnceLock::new(),
            events: None,
        }
    }

    /// Attach the append-only revision log so a create/update records a versioned snapshot.
    pub fn with_revisions(mut self, revisions: Arc<dyn RevisionLog>) -> Self {
        self.revisions = Some(revisions);
        self
    }

    /// Attach the node-wide `ProfilesChanged` sink so a successful create/update/delete emits the
    /// profile-list-changed pointer (Phase 3). Unset leaves emission off (tests / a minimal node).
    pub fn with_events(mut self, events: Arc<dyn ProfileEvents>) -> Self {
        self.events = Some(events);
        self
    }

    /// Late-bind the engine/inference validator (the assembled node). Idempotent: the first set wins.
    pub fn set_validator(&self, validator: Arc<dyn ProfileValidator>) {
        let _ = self.validator.set(validator);
    }

    /// Fetch one profile by id (the agent tool's subtree-scoped view read).
    pub fn get(&self, id: &str) -> Result<Option<ProfileSpec>, ApiError> {
        self.store.get(id).map_err(profile_err)
    }

    /// All known profiles (the agent tool's subtree-scoped listing filters these).
    pub fn list(&self) -> Result<Vec<ProfileSpec>, ApiError> {
        self.store.list().map_err(profile_err)
    }

    /// Run the injected validator, if any (unset => store-only, no validation).
    async fn validate(&self, spec: &ProfileSpec) -> Result<(), ApiError> {
        match self.validator.get() {
            Some(validator) => validator.validate_profile(spec).await,
            None => Ok(()),
        }
    }

    /// Create a new profile: stamp its provenance authoritatively (never from the incoming spec),
    /// validate the engine/inference binding, persist it (errors if the id exists), record a
    /// `create` revision attributed to `author`, and emit the `ProfilesChanged` pointer. The
    /// provenance is `created_by = author` and, for an agent author, `owner = {session}` parsed from
    /// the `agent/{session}/{name}` id (operator profiles have no owner — they are node-wide).
    pub async fn create(&self, mut spec: ProfileSpec, author: Author) -> Result<(), ApiError> {
        spec.created_by = Some(author.clone());
        spec.owner = match &author {
            Author::Operator => None,
            Author::Agent(_) => owner_from_id(&spec.id),
        };
        self.validate(&spec).await?;
        let id = spec.id.clone();
        self.store.create(spec).map_err(profile_err)?;
        self.record(&id, author, "create");
        self.emit_changed();
        Ok(())
    }

    /// Replace an existing profile: validate, PRESERVE the stored provenance (an update never
    /// rewrites who created/owns the profile), persist (errors if the id is absent), record an
    /// `update` revision attributed to `author`, and emit the `ProfilesChanged` pointer.
    pub async fn update(&self, mut spec: ProfileSpec, author: Author) -> Result<(), ApiError> {
        if let Ok(Some(existing)) = self.store.get(&spec.id) {
            spec.created_by = existing.created_by;
            spec.owner = existing.owner;
        }
        self.validate(&spec).await?;
        let id = spec.id.clone();
        self.store.update(spec).map_err(profile_err)?;
        self.record(&id, author, "update");
        self.emit_changed();
        Ok(())
    }

    /// Delete a profile by id (idempotent; no revision — a deletion has no snapshot to record). Emits
    /// the `ProfilesChanged` pointer so a client drops the row.
    pub fn delete(&self, id: &str) -> Result<(), ApiError> {
        self.store.delete(id).map_err(profile_err)?;
        self.emit_changed();
        Ok(())
    }

    /// Emit the node-wide `ProfilesChanged` pointer when a feed is wired (a no-op otherwise).
    fn emit_changed(&self) {
        if let Some(events) = &self.events {
            events.profiles_changed();
        }
    }

    /// Record a profile revision of `id`'s current on-disk spec under `author`/`reason`. Best-effort:
    /// only when a revision log is wired, and a hiccup never fails the underlying mutation (mirrors
    /// `NodeApiImpl::record_profile`).
    fn record(&self, id: &str, author: Author, reason: &str) {
        let Some(log) = self.revisions.as_ref() else {
            return;
        };
        let Ok(Some(spec)) = self.store.get(id) else {
            return;
        };
        let mut blob = Vec::new();
        if ciborium::into_writer(&spec, &mut blob).is_ok() {
            let _ = log.append(RevisionKind::Profile, id, &blob, author, reason);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::MemProfileStore;
    use daemon_api::ProviderSelector;

    fn ops() -> (ProfileOps, Arc<dyn ProfileStore>) {
        let store: Arc<dyn ProfileStore> = Arc::new(MemProfileStore::new());
        (ProfileOps::new(store.clone()), store)
    }

    #[test]
    fn owner_parsing() {
        assert_eq!(owner_from_id("agent/s1/helper").as_deref(), Some("s1"));
        assert_eq!(
            owner_from_id("agent/s1/c2/helper").as_deref(),
            Some("s1/c2")
        );
        assert_eq!(owner_from_id("opus"), None);
        assert_eq!(owner_from_id("agent/only"), None);
        assert_eq!(owner_from_id("agent/s1/"), None);
    }

    #[tokio::test]
    async fn agent_create_stamps_provenance_and_persists() {
        let (ops, store) = ops();
        ops.create(
            ProfileSpec::new("agent/s1/helper", ProviderSelector::Mock, "m"),
            Author::Agent("profile_manage".into()),
        )
        .await
        .expect("create");
        // Provenance is stamped authoritatively and durably persisted (survives a round-trip).
        let got = store.get("agent/s1/helper").unwrap().unwrap();
        assert_eq!(got.created_by, Some(Author::Agent("profile_manage".into())));
        assert_eq!(got.owner.as_deref(), Some("s1"));
    }

    #[tokio::test]
    async fn operator_create_has_no_owner() {
        let (ops, store) = ops();
        ops.create(
            ProfileSpec::new("opus", ProviderSelector::Mock, "m"),
            Author::Operator,
        )
        .await
        .expect("create");
        let got = store.get("opus").unwrap().unwrap();
        assert_eq!(got.created_by, Some(Author::Operator));
        assert_eq!(
            got.owner, None,
            "an operator profile is node-wide (no owner)"
        );
    }

    #[tokio::test]
    async fn incoming_provenance_is_ignored_on_create() {
        let (ops, store) = ops();
        // A caller cannot spoof provenance: whatever is on the incoming spec is overwritten.
        let spoofed = ProfileSpec {
            created_by: Some(Author::Operator),
            owner: Some("someone-else".into()),
            ..ProfileSpec::new("agent/s1/helper", ProviderSelector::Mock, "m")
        };
        ops.create(spoofed, Author::Agent("profile_manage".into()))
            .await
            .expect("create");
        let got = store.get("agent/s1/helper").unwrap().unwrap();
        assert_eq!(got.created_by, Some(Author::Agent("profile_manage".into())));
        assert_eq!(got.owner.as_deref(), Some("s1"));
    }

    #[tokio::test]
    async fn update_preserves_original_provenance() {
        let (ops, store) = ops();
        ops.create(
            ProfileSpec::new("agent/s1/helper", ProviderSelector::Mock, "m"),
            Author::Agent("profile_manage".into()),
        )
        .await
        .expect("create");
        // An update (even attributed to the operator, with blank incoming provenance) never rewrites
        // who created/owns the profile.
        let mut edit = store.get("agent/s1/helper").unwrap().unwrap();
        edit.created_by = None;
        edit.owner = None;
        edit.model = "m2".into();
        ops.update(edit, Author::Operator).await.expect("update");
        let got = store.get("agent/s1/helper").unwrap().unwrap();
        assert_eq!(got.model, "m2");
        assert_eq!(got.created_by, Some(Author::Agent("profile_manage".into())));
        assert_eq!(got.owner.as_deref(), Some("s1"));
    }
}
