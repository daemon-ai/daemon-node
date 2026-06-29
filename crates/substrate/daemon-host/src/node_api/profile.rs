// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::*;

#[async_trait]
impl ProfileApi for NodeApiImpl {
    fn supports_versioning(&self) -> bool {
        self.revisions.is_some()
    }

    async fn profile_list(&self) -> Vec<ProfileInfo> {
        let Ok(store) = self.profile_store() else {
            return Vec::new();
        };
        let active = store.active().ok().flatten();
        match store.list() {
            Ok(specs) => {
                let mut out: Vec<ProfileInfo> = specs
                    .iter()
                    .map(|s| ProfileInfo::from_spec(s, active.as_deref() == Some(s.id.as_str())))
                    .collect();
                out.sort_by(|a, b| a.id.cmp(&b.id));
                out
            }
            Err(_) => Vec::new(),
        }
    }

    async fn profile_get(&self, id: String) -> Result<Option<ProfileSpec>, ApiError> {
        self.profile_store()?.get(&id).map_err(profile_err)
    }

    async fn profile_create(&self, spec: ProfileSpec) -> Result<(), ApiError> {
        let id = spec.id.clone();
        self.profile_store()?.create(spec).map_err(profile_err)?;
        self.record_profile(&id, daemon_common::Author::Operator, "create");
        Ok(())
    }

    async fn profile_update(&self, spec: ProfileSpec) -> Result<(), ApiError> {
        let id = spec.id.clone();
        self.profile_store()?.update(spec).map_err(profile_err)?;
        self.record_profile(&id, daemon_common::Author::Operator, "update");
        // A profile change can alter routing (agent selection / transport patterns): rebuild the
        // live table so routed submits pick up the change without a restart (§5.9 hot-reload).
        self.rebuild_routing();
        Ok(())
    }

    async fn profile_delete(&self, id: String) -> Result<(), ApiError> {
        self.profile_store()?.delete(&id).map_err(profile_err)
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
        store.create(spec).map_err(profile_err)?;
        self.record_profile(
            &new_id,
            daemon_common::Author::Operator,
            &format!("clone of {source}"),
        );
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
        let id = spec.id.clone();
        store.create(spec).map_err(profile_err)?;
        self.record_profile(&id, daemon_common::Author::Operator, "import");
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
        Ok(id)
    }

    async fn profile_history(&self, id: String) -> Result<Vec<daemon_common::Revision>, ApiError> {
        self.revision_log()?
            .history(daemon_common::RevisionKind::Profile, &id)
            .map_err(revision_err)
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
        Ok(())
    }

    async fn skill_history(&self, name: String) -> Result<Vec<daemon_common::Revision>, ApiError> {
        self.revision_log()?
            .history(daemon_common::RevisionKind::Skill, &name)
            .map_err(revision_err)
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
