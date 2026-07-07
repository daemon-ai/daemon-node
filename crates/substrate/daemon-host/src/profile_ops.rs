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
}

impl ProfileOps {
    /// A profile surface over `store` with no revision log and no validator wired.
    pub fn new(store: Arc<dyn ProfileStore>) -> Self {
        Self {
            store,
            revisions: None,
            validator: OnceLock::new(),
        }
    }

    /// Attach the append-only revision log so a create/update records a versioned snapshot.
    pub fn with_revisions(mut self, revisions: Arc<dyn RevisionLog>) -> Self {
        self.revisions = Some(revisions);
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

    /// Create a new profile: validate the engine/inference binding, persist it (errors if the id
    /// exists), and record a `create` revision attributed to `author`.
    pub async fn create(&self, spec: ProfileSpec, author: Author) -> Result<(), ApiError> {
        self.validate(&spec).await?;
        let id = spec.id.clone();
        self.store.create(spec).map_err(profile_err)?;
        self.record(&id, author, "create");
        Ok(())
    }

    /// Replace an existing profile: validate, persist (errors if the id is absent), and record an
    /// `update` revision attributed to `author`.
    pub async fn update(&self, spec: ProfileSpec, author: Author) -> Result<(), ApiError> {
        self.validate(&spec).await?;
        let id = spec.id.clone();
        self.store.update(spec).map_err(profile_err)?;
        self.record(&id, author, "update");
        Ok(())
    }

    /// Delete a profile by id (idempotent; no revision — a deletion has no snapshot to record).
    pub fn delete(&self, id: &str) -> Result<(), ApiError> {
        self.store.delete(id).map_err(profile_err)
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
