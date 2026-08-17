// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::*;

#[async_trait]
impl CredentialApi for NodeApiImpl {
    async fn credential_set(&self, profile: String, secret: String) -> Result<(), ApiError> {
        let store = self
            .credentials
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("credential management not available".into()))?;
        // Paste-for-profile redirect (plan Phase 2): when the named string is a profile whose
        // credential ref is a node-managed PROVIDER-GLOBAL ref (`provider/<vendor>`), the secret
        // lands under THAT ref — the one the broker's single-ref lookup reads — not under the
        // profile id (where nothing would ever read it). Any other target (a legacy profile-id
        // key, an explicit operator ref like `oauth2/<label>`, a direct store row) is stored
        // verbatim; only the node-minted namespace is redirected.
        let key = self
            .profile_store()
            .ok()
            .and_then(|profiles| profiles.get(&profile).ok().flatten())
            .map(|spec| spec.credential_profile().to_string())
            .filter(|r| daemon_api::is_provider_credential_ref(r))
            .unwrap_or(profile);
        store
            .set(&key, &secret)
            .map_err(|e| ApiError::Other(format!("credential set: {e}")))?;
        // Cluster F (Part B): replacing the profile's credential must invalidate leases minted
        // against the OLD material — bump the authority's lease epoch after the store write.
        if let Some(revoker) = &self.credential_revoker {
            revoker.revoke_profile(&key);
        }
        Ok(())
    }

    async fn credential_list(&self) -> Vec<CredentialInfo> {
        let mut list = match &self.credentials {
            Some(store) => store.list_redacted(),
            None => Vec::new(),
        };
        // Overlay the node-owned human labels (wire v35) from the durable store onto the redacted
        // rows — the credential store itself holds only the secret material.
        let labels: std::collections::HashMap<String, String> =
            self.store.credential_labels().await.into_iter().collect();
        // The typed manager overlay (wire v50, plan Phase 5): one profile scan shared by every
        // row — which profiles reference each ref, and which refs are bound as channel accounts.
        let profiles = self
            .profile_store()
            .ok()
            .and_then(|s| s.list().ok())
            .unwrap_or_default();
        for info in &mut list {
            if let Some(label) = labels.get(&info.profile) {
                info.label = Some(label.clone());
            }
            self.enrich_credential_info(info, &profiles);
        }
        list
    }

    async fn credential_remove(&self, profile: String, force: bool) -> Result<(), ApiError> {
        let store = self
            .credentials
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("credential management not available".into()))?;
        // The guarded mode (wire v50): removal of an in-use credential is rejected WITH the
        // dependent list — node-authoritative, an app-side check would race profile edits. The
        // node's own teardown paths (TransportRemove) pass `force` because the referencing
        // resource is being removed in the same operation.
        if !force {
            let used_by = self.credential_used_by(&profile);
            if !used_by.is_empty() {
                return Err(ApiError::Other(format!(
                    "credential in use by profile(s): {}; remove with force to delete anyway",
                    used_by.join(", ")
                )));
            }
        }
        store
            .remove(&profile)
            .map_err(|e| ApiError::Other(format!("credential remove: {e}")))?;
        // Cluster F (Part B): removing the profile's credential must invalidate any outstanding
        // lease minted against it — bump the authority's lease epoch (and drop retained proxied
        // keys) after the store delete.
        if let Some(revoker) = &self.credential_revoker {
            revoker.revoke_profile(&profile);
        }
        // Drop any human label for this credential too, so a later re-add starts clean (wire v35).
        let _ = self.store.set_credential_label(&profile, None).await;
        Ok(())
    }

    async fn credential_set_label(
        &self,
        profile: String,
        label: Option<String>,
    ) -> Result<(), ApiError> {
        // Persist the human label (wire v35); it is overlaid onto `CredentialInfo` in
        // `credential_list()`. Node-owned in the durable store (the credential store holds only
        // secret material), so this is available even without a credential store bound.
        self.store
            .set_credential_label(&profile, label)
            .await
            .map_err(|e| ApiError::Other(format!("set_credential_label: {e}")))?;
        Ok(())
    }
}

#[async_trait]
impl AuthApi for NodeApiImpl {
    async fn auth_begin(&self, req: AuthBeginRequest) -> Result<AuthBeginResponse, ApiError> {
        let flows = self
            .auth_flows
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("interactive auth not available".into()))?;
        flows.begin(req).await
    }

    async fn auth_step(&self, req: AuthStepRequest) -> Result<AuthStepResult, ApiError> {
        let flows = self
            .auth_flows
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("interactive auth not available".into()))?;
        // Advance the parked flow one step (the registry awaits the family's step outside its lock).
        match flows.step(&req.flow_id, req.input).await? {
            crate::auth::FlowStep::Challenge(challenge) => Ok(AuthStepResult::Challenge(challenge)),
            crate::auth::FlowStep::Completed { outcome, bind } => Ok(AuthStepResult::Completed(
                self.finish_auth_outcome(outcome, bind)?,
            )),
        }
    }

    async fn auth_cancel(&self, flow_id: String) -> Result<(), ApiError> {
        if let Some(flows) = self.auth_flows.as_ref() {
            flows.cancel(&flow_id);
        }
        Ok(())
    }

    async fn auth_providers(&self) -> Vec<AuthProviderInfo> {
        self.auth_flows
            .as_ref()
            .map(|f| f.providers())
            .unwrap_or_default()
    }
}

/// Profiles referencing `credential_ref` — via `credential_ref` / profile-id keying
/// ([`ProfileSpec::credential_profile`]), the configured fallback ref, or a bound account. One
/// definition shared by the guarded `CredentialRemove` and the `used_by` overlay, so the error a
/// removal returns names exactly the rows the manager displays.
fn credential_used_by_in(credential_ref: &str, profiles: &[ProfileSpec]) -> Vec<String> {
    profiles
        .iter()
        .filter(|spec| {
            spec.credential_profile() == credential_ref
                || spec.fallback_credential_profile() == Some(credential_ref)
                || spec
                    .bound_accounts
                    .iter()
                    .any(|a| a.credential_ref == credential_ref)
        })
        .map(|spec| spec.id.clone())
        .collect()
}

impl NodeApiImpl {
    /// The dependency list for one credential ref (a fresh profile scan — remove-time accuracy).
    fn credential_used_by(&self, credential_ref: &str) -> Vec<String> {
        let profiles = self
            .profile_store()
            .ok()
            .and_then(|s| s.list().ok())
            .unwrap_or_default();
        credential_used_by_in(credential_ref, &profiles)
    }

    /// The typed manager overlay (wire v50, plan Phase 5) for one redacted row: scope and
    /// classification from the ref shape + profile table, material kind / expiry / refresh posture
    /// from decoding the stored envelope, `used_by` from the shared dependency scan. All
    /// node-derived — the client renders, never re-derives.
    fn enrich_credential_info(&self, info: &mut CredentialInfo, profiles: &[ProfileSpec]) {
        let credential_ref = info.profile.clone();
        info.used_by = credential_used_by_in(&credential_ref, profiles);
        let is_provider_global = daemon_api::is_provider_credential_ref(&credential_ref);
        info.scope = Some(
            if is_provider_global {
                "global"
            } else {
                "profile"
            }
            .to_string(),
        );
        // Section: agent-gate refs by namespace; channel refs by being bound as a transport
        // account (or living in the operator `oauth2/` namespace); everything else is a
        // model-provider credential (provider-global or a legacy profile-keyed key).
        let bound_as_account = profiles.iter().any(|spec| {
            spec.bound_accounts
                .iter()
                .any(|a| a.credential_ref == credential_ref)
        });
        info.classification = Some(
            if credential_ref.starts_with(crate::agent_auth::AGENT_FAMILY_PREFIX) {
                "agent"
            } else if bound_as_account || credential_ref.starts_with("oauth2/") {
                "channel"
            } else {
                "provider"
            }
            .to_string(),
        );
        if is_provider_global {
            info.provider = credential_ref
                .strip_prefix(daemon_api::vendor::PROVIDER_REF_PREFIX)
                .map(str::to_string);
        }
        // Material facts from the stored blob. An undecodable envelope leaves `kind` unset —
        // honest "unknown", matching the projector's fail-closed posture.
        let Some(blob) = self
            .credentials
            .as_ref()
            .and_then(|s| s.get(&credential_ref))
        else {
            return;
        };
        match daemon_common::CredentialEnvelope::parse(&blob) {
            Ok(daemon_common::CredentialEnvelope::Key(_)) => info.kind = Some("api_key".into()),
            Ok(daemon_common::CredentialEnvelope::OAuthTokenSet(ts)) => {
                info.kind = Some("oauth_token".into());
                // The envelope's provider identity is TRUSTED (node-written at mint time) and
                // more specific than the ref shape — it wins.
                info.provider = Some(ts.provider_id.clone());
                info.expires_at = ts.expires_at;
                info.refresh_status = if ts.refresh_token.is_some() {
                    Some("refreshable".into())
                } else if ts.expires_at.is_some() {
                    Some("reauth_required".into())
                } else {
                    None // no refresh token but also no expiry: nothing to refresh, ever
                };
                // The store's redaction masked the envelope JSON; re-mask over the ACCESS TOKEN so
                // the hint is honest material, not serialization tail.
                info.hint = CredentialInfo::redacted(&credential_ref, Some(&ts.access_token)).hint;
            }
            Err(_) => {}
        }
    }

    /// Persist a completed flow's [`AuthOutcome`] into the credential store and honor any bind,
    /// returning the wire [`AuthCompleteResponse`]. The single credential-slot resolution path shared
    /// by `auth_step` completion and (via it) the `auth_complete` compatibility wrapper.
    fn finish_auth_outcome(
        &self,
        outcome: crate::auth::AuthOutcome,
        bind: Option<AuthBindRequest>,
    ) -> Result<AuthCompleteResponse, ApiError> {
        let store = self
            .credentials
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("credential management not available".into()))?;

        // Slot mapping (the node decides, never the client): a provider-bound family (OpenRouter /
        // GitHub Copilot / Hugging Face) mints a MODEL-PROVIDER API key stored under the
        // PROVIDER-GLOBAL ref the family derived (`provider/<vendor>`, plan Phase 2) — one shared
        // credential per vendor, flowing downstream exactly like a pasted key. The bind still
        // names the target profile: its `credential_ref` is pointed at the shared ref (a
        // reference, not a copy) so the broker's single-ref lookup finds the key even on a
        // pre-Phase-2 profile that was still profile-id-keyed. No `BoundAccount` is attached (a
        // provider key is not a transport account).
        if outcome.slot == crate::auth::CredentialSlotKind::ProviderKeyForProfile {
            let bind = bind.ok_or_else(|| {
                ApiError::Other(
                    "provider-key auth requires a bind naming the target profile: the profile \
                     must adopt the minted key's credential ref for the model broker to use it"
                        .into(),
                )
            })?;
            let profile = bind.profile.clone();
            let credential_ref = outcome.credential_ref.clone();
            store
                .set(&credential_ref, &outcome.credential_blob)
                .map_err(|e| ApiError::Other(format!("credential set: {e}")))?;
            // Replacing the vendor's provider key must invalidate leases minted against any prior
            // material — bump the authority's lease epoch after the store write (as `credential_set`).
            if let Some(revoker) = &self.credential_revoker {
                revoker.revoke_profile(&credential_ref);
            }
            // Point the bound profile at the shared ref. Best-effort by design: a bind naming a
            // not-yet-created profile still lands the key (creation will mint the same ref from
            // the vendor's model namespace).
            let profiles = self.profile_store()?;
            if let Ok(Some(mut spec)) = profiles.get(profile.as_str()) {
                if spec.credential_ref.as_deref() != Some(credential_ref.as_str()) {
                    spec.credential_ref = Some(credential_ref.clone());
                    profiles.update(spec).map_err(profile_err)?;
                }
            }
            return Ok(AuthCompleteResponse {
                credential_ref,
                account_label: outcome.account_label,
                transport_instance: outcome.transport_instance,
                bound_profile: Some(profile),
            });
        }

        // The credential ref: a bind-supplied override wins over the family-derived default, so an
        // operator can pin where the blob lands; otherwise the family names it (e.g. by resolved user).
        let credential_ref = bind
            .as_ref()
            .and_then(|b| b.credential_ref.clone())
            .unwrap_or_else(|| outcome.credential_ref.clone());

        store
            .set(&credential_ref, &outcome.credential_blob)
            .map_err(|e| ApiError::Other(format!("credential set: {e}")))?;

        // Optional account→profile bind: attach (or replace) the BoundAccount on the target profile so
        // the transport's account bring-up (`AccountProvisioning::bound_accounts`) discovers it.
        let mut bound_profile = None;
        if let Some(bind) = bind {
            let profiles = self.profile_store()?;
            let mut spec = profiles
                .get(bind.profile.as_str())
                .map_err(profile_err)?
                .ok_or_else(|| ApiError::Other(format!("unknown profile: {}", bind.profile)))?;
            let transport_instance = bind
                .transport_instance
                .clone()
                .unwrap_or_else(|| outcome.transport_instance.clone());
            spec.bound_accounts
                .retain(|a| a.transport_instance != transport_instance.as_str());
            spec.bound_accounts.push(BoundAccount::new(
                transport_instance.as_str(),
                &credential_ref,
            ));
            profiles.update(spec).map_err(profile_err)?;
            bound_profile = Some(bind.profile.clone());
        }

        // A completed account bind can change which transport instances route to which profile:
        // rebuild routing so the freshly-authenticated account is reachable without a restart.
        if bound_profile.is_some() {
            self.rebuild_routing();
        }

        // An agent-scoped credential (wire v47: `agent/<name>/...` from the dynamic `agent/*`
        // families) flips that agent's derived auth verdict — nudge subscribers to refetch the
        // catalog, exactly as a registration or discovery pass would. A completed re-auth also
        // clears the runtime rejection fact (A6): the fresh credential supersedes the stale one
        // the rejection proved.
        if let Some(rest) = credential_ref.strip_prefix(crate::agent_auth::AGENT_FAMILY_PREFIX) {
            if let Some((name, _)) = rest.split_once('/') {
                if let Some(store) = &self.credentials {
                    let _ = store.remove(&format!("agent/{name}/rejected"));
                }
                // The credential only reaches an agent process at spawn-time materialization, so
                // resident sessions for this agent still run with the stale (or absent) secret —
                // the very state the operator just fixed. Evict them (durable sessions survive in
                // the store); the next turn re-`ensure`s a fresh spawn that carries the new
                // credential. Best-effort async: completion must not block on process teardown.
                let live = self.live.clone();
                let agent = name.to_string();
                tokio::spawn(async move {
                    live.evict_foreign_for_agent(&agent).await;
                });
            }
            self.emit_agents_changed();
        }

        Ok(AuthCompleteResponse {
            credential_ref,
            account_label: outcome.account_label,
            transport_instance: outcome.transport_instance,
            bound_profile,
        })
    }
}
