// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::*;

#[async_trait]
impl ProfileApi for NodeApiImpl {
    fn supports_versioning(&self) -> bool {
        self.revisions.is_some()
    }

    async fn profile_list(&self) -> Vec<ProfileInfo> {
        match self.profile_store() {
            Err(_) => Vec::new(),
            Ok(store) => {
                let active = store.active().ok().flatten();
                match store.list() {
                    Ok(specs) => {
                        let mut out: Vec<ProfileInfo> = specs
                            .iter()
                            .map(|s| {
                                ProfileInfo::from_spec(s, active.as_deref() == Some(s.id.as_str()))
                            })
                            .collect();
                        out.sort_by(|a, b| a.id.cmp(&b.id));
                        out
                    }
                    Err(_) => Vec::new(),
                }
            }
        }
    }

    async fn profile_get(&self, id: String) -> Result<Option<ProfileSpec>, ApiError> {
        self.profile_store()?.get(&id).map_err(profile_err)
    }

    async fn profile_create(&self, spec: ProfileSpec) -> Result<(), ApiError> {
        // Author through the shared `ProfileOps` (one validation + persistence + revision path with
        // the agent `profile_manage` tool). The inline fallback is the same three operations for a
        // minimal node that wires no shared facade.
        match &self.profile_ops {
            // The shared facade stamps provenance + emits `ProfilesChanged` on its own.
            Some(ops) => ops.create(spec, daemon_common::Author::Operator).await?,
            None => {
                self.validate_engine(&spec).await?;
                validate_inference(&spec)?;
                let mut spec = spec;
                spec.created_by = Some(daemon_common::Author::Operator);
                spec.owner = None;
                let id = spec.id.clone();
                self.profile_store()?.create(spec).map_err(profile_err)?;
                self.record_profile(&id, daemon_common::Author::Operator, "create");
                self.emit_profiles_changed();
            }
        }
        // A created profile can declare `bound_accounts`: rebuild the live routing table so its
        // account baseline takes effect without a restart (§5.9 hot-reload).
        self.rebuild_routing();
        Ok(())
    }

    async fn profile_update(&self, spec: ProfileSpec) -> Result<(), ApiError> {
        match &self.profile_ops {
            // The shared facade preserves provenance + emits `ProfilesChanged` on its own.
            Some(ops) => ops.update(spec, daemon_common::Author::Operator).await?,
            None => {
                self.validate_engine(&spec).await?;
                validate_inference(&spec)?;
                let mut spec = spec;
                if let Ok(Some(existing)) = self.profile_store()?.get(&spec.id) {
                    spec.created_by = existing.created_by;
                    spec.owner = existing.owner;
                }
                let id = spec.id.clone();
                self.profile_store()?.update(spec).map_err(profile_err)?;
                self.record_profile(&id, daemon_common::Author::Operator, "update");
                self.emit_profiles_changed();
            }
        }
        // A profile change can alter routing (agent selection / transport patterns): rebuild the
        // live table so routed submits pick up the change without a restart (§5.9 hot-reload).
        self.rebuild_routing();
        Ok(())
    }

    async fn profile_delete(&self, id: String) -> Result<(), ApiError> {
        // Route through the shared facade when wired (it emits `ProfilesChanged`); otherwise delete
        // store-direct + emit here. An operator may `ProfileDelete` ANY profile (agent-authored
        // included) — the subtree scoping is only on the agent tool path.
        match &self.profile_ops {
            Some(ops) => ops.delete(&id)?,
            None => {
                self.profile_store()?.delete(&id).map_err(profile_err)?;
                self.emit_profiles_changed();
            }
        }
        // A deleted profile takes its `bound_accounts` baseline with it (§5.9 hot-reload).
        self.rebuild_routing();
        Ok(())
    }

    async fn profile_select(&self, id: String) -> Result<(), ApiError> {
        self.profile_store()?.set_active(&id).map_err(profile_err)
    }

    async fn profile_clone(&self, source: String, new_id: String) -> Result<(), ApiError> {
        let store = self.profile_store()?;
        let mut spec = store
            .get(&source)
            .map_err(profile_err)?
            .ok_or_else(|| ApiError::UnknownSession(source.clone()))?;
        spec.id = new_id.clone();
        // A clone is a new operator-authored profile: it inherits none of the source's provenance.
        spec.created_by = Some(daemon_common::Author::Operator);
        spec.owner = None;
        store.create(spec).map_err(profile_err)?;
        self.record_profile(
            &new_id,
            daemon_common::Author::Operator,
            &format!("clone of {source}"),
        );
        self.emit_profiles_changed();
        Ok(())
    }

    async fn profile_export(&self, id: String) -> Result<Distribution, ApiError> {
        let spec = self
            .profile_store()?
            .get(&id)
            .map_err(profile_err)?
            .ok_or_else(|| ApiError::UnknownSession(id.clone()))?;
        // A profile distribution carries *that profile's* local (non-bundled) skills, resolved from
        // its own per-profile store; otherwise just the spec. credential_ref is kept (a name).
        let skills = match self.skills.as_ref() {
            Some(provider) => provider
                .for_profile(&id)
                .export_local()
                .map_err(|e| ApiError::Other(format!("skill export: {e}")))?,
            None => Vec::new(),
        };
        let head_seq = self
            .revisions
            .as_ref()
            .and_then(|log| {
                log.head(daemon_common::RevisionKind::Profile, &id)
                    .ok()
                    .flatten()
            })
            .map(|r| r.seq);
        Ok(Distribution {
            wire_version: daemon_common::WireVersion::CURRENT,
            profile: spec,
            skills,
            head_seq,
            source: None,
        })
    }

    async fn profile_import(
        &self,
        dist: Distribution,
        new_id: Option<String>,
    ) -> Result<String, ApiError> {
        if dist.wire_version != daemon_common::WireVersion::CURRENT {
            return Err(ApiError::Other(format!(
                "incompatible distribution wire version {} (node is {})",
                dist.wire_version.0,
                daemon_common::WireVersion::CURRENT.0
            )));
        }
        let store = self.profile_store()?;
        let mut spec = dist.profile;
        if let Some(id) = new_id {
            spec.id = id;
        }
        // An imported distribution's engine binding is validated against THIS node's catalog: the
        // exporting node's ACP agents do not necessarily exist here.
        self.validate_engine(&spec).await?;
        validate_inference(&spec)?;
        // An imported profile is operator-authored on THIS node (its origin provenance does not
        // travel — a distribution carries no owner/created_by that this node would honor).
        spec.created_by = Some(daemon_common::Author::Operator);
        spec.owner = None;
        let id = spec.id.clone();
        store.create(spec).map_err(profile_err)?;
        self.record_profile(&id, daemon_common::Author::Operator, "import");
        self.emit_profiles_changed();
        // Materialize the distribution's local skills into the *imported profile's own* skills dir
        // (so a session that resolves this agent actually sees them), attributed to the operator. A
        // skill that already exists is left as-is rather than clobbered.
        if let Some(provider) = self.skills.as_ref() {
            let skill_store = provider.for_profile(&id);
            for bundle in &dist.skills {
                if skill_store.is_bundled(&bundle.name) {
                    continue;
                }
                skill_store
                    .import_bundle(
                        bundle,
                        daemon_common::Author::Operator,
                        &format!("import via {id}"),
                    )
                    .map_err(|e| ApiError::Other(format!("skill import: {e}")))?;
            }
        }
        // An imported profile can declare `bound_accounts` (§5.9 hot-reload).
        self.rebuild_routing();
        Ok(id)
    }

    async fn profile_history(
        &self,
        id: String,
        after: Option<String>,
    ) -> Result<daemon_api::WirePage<daemon_common::Revision>, ApiError> {
        let history = self
            .revision_log()?
            .history(daemon_common::RevisionKind::Profile, &id)
            .map_err(revision_err)?;
        Ok(paginate_revisions(history, after))
    }

    async fn profile_at(&self, id: String, seq: u64) -> Result<ProfileSpec, ApiError> {
        let blob = self
            .revision_log()?
            .get_at(daemon_common::RevisionKind::Profile, &id, seq)
            .map_err(revision_err)?;
        ciborium::from_reader(blob.as_slice())
            .map_err(|e| ApiError::Other(format!("decode profile revision: {e}")))
    }

    async fn profile_revert(&self, id: String, seq: u64) -> Result<(), ApiError> {
        let spec = self.profile_at(id.clone(), seq).await?;
        self.profile_store()?.update(spec).map_err(profile_err)?;
        self.record_profile(
            &id,
            daemon_common::Author::Operator,
            &format!("revert to {seq}"),
        );
        self.emit_profiles_changed();
        Ok(())
    }

    async fn soul_get(&self, id: String) -> Result<String, ApiError> {
        let persona = self.persona_backend()?;
        // Fetch-before-delegate: the persona backend seeds SOUL.md on a miss, so an unknown
        // profile id must fail here (the same not-found the other profile ops raise) rather than
        // materialize an orphan persona doc.
        let spec = self.profile_store()?.get(&id).map_err(profile_err)?;
        crate::persona_ops::soul_get_guarded(persona.as_ref(), spec.as_ref(), &id).await
    }

    async fn soul_set(&self, id: String, text: String) -> Result<(), ApiError> {
        let persona = self.persona_backend()?;
        let spec = self.profile_store()?.get(&id).map_err(profile_err)?;
        crate::persona_ops::soul_set_guarded(persona.as_ref(), spec.as_ref(), &id, &text).await?;
        // A persona edit changes what a client renders for the profile: ping the node-wide
        // pointer so thin clients refetch. The backend owns validation + the revision log
        // (PersonaStore::set is the single SOUL.md revision writer), so nothing is recorded here.
        self.emit_profiles_changed();
        Ok(())
    }

    async fn skill_history(
        &self,
        name: String,
        after: Option<String>,
    ) -> Result<daemon_api::WirePage<daemon_common::Revision>, ApiError> {
        let history = self
            .revision_log()?
            .history(daemon_common::RevisionKind::Skill, &name)
            .map_err(revision_err)?;
        Ok(paginate_revisions(history, after))
    }

    async fn skill_at(
        &self,
        name: String,
        seq: u64,
    ) -> Result<daemon_common::SkillBundle, ApiError> {
        let blob = self
            .revision_log()?
            .get_at(daemon_common::RevisionKind::Skill, &name, seq)
            .map_err(revision_err)?;
        ciborium::from_reader(blob.as_slice())
            .map_err(|e| ApiError::Other(format!("decode skill revision: {e}")))
    }

    async fn skill_revert(&self, name: String, seq: u64) -> Result<(), ApiError> {
        let skills = self.active_skills_store()?;
        if skills.is_bundled(&name) {
            return Err(ApiError::Conflict(format!(
                "skill `{name}` is binary-bundled and cannot be reverted"
            )));
        }
        let bundle = self.skill_at(name.clone(), seq).await?;
        skills
            .import_bundle(
                &bundle,
                daemon_common::Author::Operator,
                &format!("revert to {seq}"),
            )
            .map_err(|e| ApiError::Other(format!("skill revert: {e}")))
    }

    async fn skill_get(&self, name: String) -> Result<daemon_common::SkillBundle, ApiError> {
        self.active_skills_store()?
            .export_bundle(&name)
            .map_err(|e| ApiError::Other(format!("skill get: {e}")))
    }

    async fn skill_put(&self, bundle: daemon_common::SkillBundle) -> Result<(), ApiError> {
        let skills = self.active_skills_store()?;
        if skills.is_bundled(&bundle.name) {
            return Err(ApiError::Conflict(format!(
                "skill `{}` is binary-bundled and cannot be edited",
                bundle.name
            )));
        }
        skills
            .import_bundle(&bundle, daemon_common::Author::Operator, "skill_put")
            .map_err(|e| ApiError::Other(format!("skill put: {e}")))
    }

    async fn curator_list(
        &self,
        profile: Option<String>,
    ) -> Result<Vec<daemon_api::CuratorEntry>, ApiError> {
        let store = self.curator_store(profile)?;
        let usage = store.usage();
        let mut entries = Vec::new();
        // Live (discovered) skills, with their usage record (defaulting when untracked).
        for item in store.list() {
            let record = usage.and_then(|u| u.get(&item.name)).unwrap_or_default();
            entries.push(daemon_api::CuratorEntry {
                name: item.name.clone(),
                category: item.category,
                is_bundled: store.is_bundled(&item.name),
                usage: record,
            });
        }
        // Archived skills (out of discovery): surfaced with their archived-state record so an
        // operator can see + restore them.
        for name in store.archived() {
            let mut record = usage.and_then(|u| u.get(&name)).unwrap_or_default();
            record.state = daemon_common::SkillState::Archived;
            entries.push(daemon_api::CuratorEntry {
                name: name.clone(),
                category: None,
                is_bundled: store.is_bundled(&name),
                usage: record,
            });
        }
        Ok(entries)
    }

    async fn curator_pin(&self, profile: Option<String>, name: String) -> Result<(), ApiError> {
        let store = self.curator_store(profile)?;
        self.curator_usage(&store)?.set_pinned(&name, true);
        Ok(())
    }

    async fn curator_unpin(&self, profile: Option<String>, name: String) -> Result<(), ApiError> {
        let store = self.curator_store(profile)?;
        self.curator_usage(&store)?.set_pinned(&name, false);
        Ok(())
    }

    async fn curator_archive(&self, profile: Option<String>, name: String) -> Result<(), ApiError> {
        self.curator_store(profile)?
            .archive(&name)
            .map_err(skill_err)
    }

    async fn curator_restore(&self, profile: Option<String>, name: String) -> Result<(), ApiError> {
        self.curator_store(profile)?
            .restore(&name)
            .map_err(skill_err)
    }

    async fn curator_run(
        &self,
        profile: Option<String>,
    ) -> Result<Vec<daemon_api::CuratorChange>, ApiError> {
        let store = self.curator_store(profile)?;
        let usage = self.curator_usage(&store)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let transitions = daemon_skills::apply_automatic_transitions(
            &usage.all(),
            now,
            daemon_skills::CuratorConfig::default(),
        );
        let mut changes = Vec::new();
        for t in transitions {
            match t.to {
                // Physically archive (also flips usage state to Archived). A skill already gone from
                // discovery (race) yields a not-found we tolerate.
                daemon_common::SkillState::Archived => {
                    if store.archive(&t.name).is_err() {
                        continue;
                    }
                }
                // Stale / reactivation are soft (the body stays discoverable): just flip the record.
                state => usage.set_state(&t.name, state),
            }
            changes.push(daemon_api::CuratorChange {
                name: t.name,
                from: t.from,
                to: t.to,
            });
        }
        Ok(changes)
    }
}

impl NodeApiImpl {
    /// Emit the node-wide `ProfilesChanged` pointer (Phase 3) for an operator write that does NOT
    /// route through the shared [`ProfileOps`] (clone/import/revert, and the no-facade fallback
    /// create/update/delete): those store-direct paths do not go through the facade's emit, so they
    /// ping the feed here. The `ProfileOps` path emits on its own, so this is never double-called.
    /// A no-op when no node-event feed is wired.
    pub(crate) fn emit_profiles_changed(&self) {
        if let Some(feed) = &self.node_events {
            crate::profile_ops::ProfileEvents::profiles_changed(feed.as_ref());
        }
    }

    /// The profile store, or [`ApiError::Unsupported`] when this node hosts no profile management.
    pub(crate) fn profile_store(&self) -> Result<&Arc<dyn ProfileStore>, ApiError> {
        self.profiles
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("profile management not available".into()))
    }

    /// The persona (SOUL.md) backend, or [`ApiError::Unsupported`] when this node hosts no persona
    /// management (no [`PersonaOps`](crate::persona_ops::PersonaOps) bound at assembly).
    fn persona_backend(&self) -> Result<&Arc<dyn crate::persona_ops::PersonaOps>, ApiError> {
        self.persona_ops
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("persona management not available".into()))
    }

    /// Resolve an agent-catalog entry by `name`: the merged catalog (durable manual registrations +
    /// the last discovery scan) first, then the curated builtin recipe table via the injected
    /// [`AgentDiscovery`](crate::AgentDiscovery) hook (a cheap PATH check, never an `initialize`
    /// probe — validation must not spawn candidate processes). `None` when the node knows no such
    /// agent. Profiles reference agents BY NAME ONLY, so this lookup is the sole recipe source.
    pub(crate) async fn resolve_agent_entry(&self, name: &str) -> Option<daemon_api::AgentEntry> {
        if let Some(entry) = self
            .agent_catalog()
            .await
            .into_iter()
            .find(|e| e.name == name)
        {
            return Some(entry);
        }
        self.agents.as_ref().and_then(|agents| agents.builtin(name))
    }

    /// Validate a profile spec's engine selector before it is persisted (create/update/import):
    /// a `Foreign { agent }` binding must name an agent the node's catalog knows AND that is
    /// currently installed — otherwise the mutation fails fast with a clear error instead of
    /// minting a profile whose sessions can never spawn. (Spawn re-checks installed-ness too,
    /// since it can change after validation.) `Core` always passes.
    ///
    /// A `Foreign` profile additionally validates its [`daemon_api::ForeignBackend`]: `AgentNative`
    /// always passes (the agent owns its backend); `NodeProvider` requires a non-empty model that
    /// is an installed local model (for a local provider) and a resolvable credential (for a cloud
    /// provider) — so a routed foreign session can never mint on an un-serveable provider/model.
    pub(crate) async fn validate_engine(&self, spec: &ProfileSpec) -> Result<(), ApiError> {
        let daemon_api::EngineSelector::Foreign { agent } = &spec.engine else {
            return Ok(());
        };
        let entry = self.resolve_agent_entry(agent).await.ok_or_else(|| {
            ApiError::Other(format!(
                "profile engine references unknown agent `{agent}` — register it via \
                 agent_register or run AgentDiscover first"
            ))
        })?;
        if !entry.installed {
            return Err(ApiError::Other(format!(
                "agent `{agent}` is not installed (catalog entry present, binary/endpoint \
                 missing)"
            )));
        }
        if let daemon_api::ForeignBackend::NodeProvider {
            provider,
            model,
            credential_ref,
        } = &spec.foreign_backend
        {
            self.validate_node_provider(spec, *provider, model, credential_ref.as_deref())
                .await?;
        }
        Ok(())
    }

    /// Validate a `NodeProvider` foreign backend (the gateway-routed arm): the model must be
    /// non-empty and, for a local provider, an installed model in the node catalog; a cloud provider
    /// must have a resolvable credential. The membership/credential checks degrade gracefully when
    /// the backing surface is absent (no model manager / no credential store) — validation never
    /// spawns or probes, mirroring [`validate_inference`].
    async fn validate_node_provider(
        &self,
        spec: &ProfileSpec,
        provider: ProviderSelector,
        model: &str,
        credential_ref: Option<&str>,
    ) -> Result<(), ApiError> {
        if model.trim().is_empty() {
            return Err(ApiError::Unsupported(format!(
                "profile `{}` routes a NodeProvider foreign backend but names no model — set \
                 `foreign_backend.model`",
                spec.id
            )));
        }
        if provider.is_local() {
            // A local routed model must be installed (the same latitude `validate_inference` gives:
            // only enforced when a model catalog is available to verify against).
            if self.models.is_some() {
                let installed = self
                    .models_all()
                    .await
                    .into_iter()
                    .any(|m| m.local && m.id == model);
                if !installed {
                    return Err(ApiError::Unsupported(format!(
                        "NodeProvider model `{model}` is not an installed local model — download \
                         one (ModelSearch/ModelDownload) and name its installed catalog id"
                    )));
                }
            }
        } else if !matches!(provider, ProviderSelector::Mock) {
            // A cloud provider needs a resolvable credential (the profile's own `credential_ref`, or
            // the backend override). Only enforced when a credential store is wired.
            let cref = credential_ref
                .map(str::to_string)
                .unwrap_or_else(|| spec.credential_profile().to_string());
            if let Some(store) = &self.credentials {
                if store.get(&cref).is_none() {
                    return Err(ApiError::Unsupported(format!(
                        "NodeProvider cloud backend for profile `{}` needs a resolvable credential \
                         `{cref}` — provision it via CredentialSet",
                        spec.id
                    )));
                }
            }
        }
        Ok(())
    }

    /// Resolve the spec for an explicit id, or the active default when `profile` is `None`.
    pub(crate) fn resolve_profile(
        &self,
        profile: Option<String>,
    ) -> Result<Option<ProfileSpec>, ApiError> {
        let store = self.profile_store()?;
        let id = match profile {
            Some(id) => Some(id),
            None => store.active().map_err(profile_err)?,
        };
        match id {
            Some(id) => store.get(&id).map_err(profile_err),
            None => Ok(None),
        }
    }

    /// The revision log, or [`ApiError::Unsupported`] when this node hosts no versioning.
    fn revision_log(&self) -> Result<&Arc<dyn daemon_common::RevisionLog>, ApiError> {
        self.revisions
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("versioning not available".into()))
    }

    /// The skills provider, or [`ApiError::Unsupported`] when this node hosts no skills.
    fn skills_provider(&self) -> Result<&Arc<daemon_skills::SkillsProvider>, ApiError> {
        self.skills
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("skills not available".into()))
    }

    /// Resolve the [`SkillStore`](daemon_skills::SkillStore) for an explicit profile `id`.
    fn skills_store_for(&self, id: &str) -> Result<Arc<daemon_skills::SkillStore>, ApiError> {
        Ok(self.skills_provider()?.for_profile(id))
    }

    /// Resolve the skills store for the profile a skill *versioning* op targets: the node's active
    /// default profile (falling back to the node's `default_local_profile` when no profile store /
    /// active default is set). The skill revision history is keyed by bare skill name, so this picks
    /// the library a name-keyed `skill_revert` writes back into.
    fn active_skills_store(&self) -> Result<Arc<daemon_skills::SkillStore>, ApiError> {
        let id = self
            .profiles
            .as_ref()
            .and_then(|p| p.active().ok().flatten())
            .unwrap_or_else(|| self.default_local_profile.clone());
        self.skills_store_for(&id)
    }

    /// Resolve the skills store a curator op targets: an explicit `profile`, else the active default.
    fn curator_store(
        &self,
        profile: Option<String>,
    ) -> Result<Arc<daemon_skills::SkillStore>, ApiError> {
        match profile {
            Some(id) => self.skills_store_for(&id),
            None => self.active_skills_store(),
        }
    }

    /// The per-profile usage sidecar for a curator op, or [`ApiError::Unsupported`] when usage
    /// tracking is off (no `.usage.json` factory wired).
    fn curator_usage(
        &self,
        store: &Arc<daemon_skills::SkillStore>,
    ) -> Result<Arc<dyn daemon_common::SkillUsageLog>, ApiError> {
        store
            .usage()
            .cloned()
            .ok_or_else(|| ApiError::Unsupported("skill usage tracking not available".into()))
    }

    /// Record a profile revision of `id`'s current on-disk spec under `author`/`reason`. Best-effort:
    /// only when both a profile store and a revision log are wired, and a revision-log hiccup never
    /// fails the underlying profile mutation.
    fn record_profile(&self, id: &str, author: daemon_common::Author, reason: &str) {
        let (Some(store), Some(log)) = (self.profiles.as_ref(), self.revisions.as_ref()) else {
            return;
        };
        let Ok(Some(spec)) = store.get(id) else {
            return;
        };
        let mut blob = Vec::new();
        if ciborium::into_writer(&spec, &mut blob).is_ok() {
            let _ = log.append(
                daemon_common::RevisionKind::Profile,
                id,
                &blob,
                author,
                reason,
            );
        }
    }
}

/// The node IS the profile validator the shared [`ProfileOps`](crate::profile_ops::ProfileOps) runs:
/// it owns the agent catalog / model / credential handles [`validate_engine`](NodeApiImpl::validate_engine)
/// consults. Late-bound into `ProfileOps` after the node is `Arc`-wrapped, so the operator ops and
/// the agent `profile_manage` tool share the exact same `validate_engine` + `validate_inference`.
#[async_trait]
impl crate::profile_ops::ProfileValidator for NodeApiImpl {
    async fn validate_profile(&self, spec: &ProfileSpec) -> Result<(), ApiError> {
        self.validate_engine(spec).await?;
        validate_inference(spec)
    }
}

/// Fail fast on a LOCAL-provider profile with an empty model (create/update/import): a
/// llama.cpp / mistral.rs session cannot resolve any artifact to load, so it would only fail
/// later — at first turn, far from the mistake. Cloud selectors keep the empty-model latitude
/// (the unconfigured boot placeholder is exactly that, and it is seeded, not created here).
fn validate_inference(spec: &ProfileSpec) -> Result<(), ApiError> {
    let local = matches!(
        spec.provider,
        ProviderSelector::LlamaCpp | ProviderSelector::MistralRs
    );
    if local && spec.model.trim().is_empty() {
        return Err(ApiError::Unsupported(format!(
            "profile `{}` selects a local provider but names no model — download one \
             (ModelSearch/ModelDownload) and set `model` to its installed catalog id",
            spec.id
        )));
    }
    Ok(())
}

/// Map a profile-store error onto the wire [`ApiError`].
pub(crate) fn profile_err(e: crate::profiles::ProfileError) -> ApiError {
    use crate::profiles::ProfileError;
    match e {
        ProfileError::NotFound(id) => ApiError::UnknownSession(id),
        ProfileError::Exists(id) => ApiError::Conflict(format!("profile exists: {id}")),
        other => ApiError::Other(other.to_string()),
    }
}

/// Page an oldest-first (seq-ascending) revision history under the uniform wire envelope. The
/// cursor is the stringified `seq`, resumed NUMERICALLY rather than through the generic
/// string-keyed [`daemon_api::paginate`]: a lexicographic compare would mis-order multi-digit
/// seqs ("10" < "9"), and the append-only log makes seq-resume exact anyway. An unparseable
/// cursor restarts from the first revision.
fn paginate_revisions(
    history: Vec<daemon_common::Revision>,
    after: Option<String>,
) -> daemon_api::WirePage<daemon_common::Revision> {
    let start = match after.as_deref().and_then(|s| s.parse::<u64>().ok()) {
        None => 0,
        Some(seq) => history.partition_point(|r| r.seq <= seq),
    };
    let mut items: Vec<daemon_common::Revision> = history.into_iter().skip(start).collect();
    let next = if items.len() > daemon_api::WIRE_PAGE_MAX {
        items.truncate(daemon_api::WIRE_PAGE_MAX);
        items.last().map(|r| r.seq.to_string())
    } else {
        None
    };
    daemon_api::WirePage { items, next }
}

/// Map a revision-log error onto the wire [`ApiError`].
fn revision_err(e: daemon_common::RevisionError) -> ApiError {
    use daemon_common::RevisionError;
    match e {
        RevisionError::NotFound { kind, id, seq } => {
            ApiError::UnknownSession(format!("{kind}/{id}@{seq}"))
        }
        other => ApiError::Other(other.to_string()),
    }
}

fn skill_err(e: daemon_skills::SkillError) -> ApiError {
    use daemon_skills::SkillError;
    match e {
        SkillError::NotFound(id) => ApiError::UnknownSession(format!("skill/{id}")),
        SkillError::Exists(id) => ApiError::Conflict(format!("skill exists: {id}")),
        other => ApiError::Other(other.to_string()),
    }
}
