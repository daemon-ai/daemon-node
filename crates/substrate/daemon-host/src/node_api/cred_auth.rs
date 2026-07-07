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
        store
            .set(&profile, &secret)
            .map_err(|e| ApiError::Other(format!("credential set: {e}")))?;
        // Cluster F (Part B): replacing the profile's credential must invalidate leases minted
        // against the OLD material — bump the authority's lease epoch after the store write.
        if let Some(revoker) = &self.credential_revoker {
            revoker.revoke_profile(&profile);
        }
        Ok(())
    }

    async fn credential_list(&self) -> Vec<CredentialInfo> {
        match &self.credentials {
            Some(store) => store.list_redacted(),
            None => Vec::new(),
        }
    }

    async fn credential_remove(&self, profile: String) -> Result<(), ApiError> {
        let store = self
            .credentials
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("credential management not available".into()))?;
        store
            .remove(&profile)
            .map_err(|e| ApiError::Other(format!("credential remove: {e}")))?;
        // Cluster F (Part B): removing the profile's credential must invalidate any outstanding
        // lease minted against it — bump the authority's lease epoch (and drop retained proxied
        // keys) after the store delete.
        if let Some(revoker) = &self.credential_revoker {
            revoker.revoke_profile(&profile);
        }
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

impl NodeApiImpl {
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
        // Hugging Face) mints a MODEL-PROVIDER API key, which must ride the target profile's
        // credential slot — the id the credential broker reads — so it flows downstream exactly like
        // a pasted API key. It requires a bind naming the profile (a key with no bind target would be
        // stranded where no broker reads it); no `BoundAccount` is attached (a provider key is not a
        // transport account).
        if outcome.slot == crate::auth::CredentialSlotKind::ProviderKeyForProfile {
            let bind = bind.ok_or_else(|| {
                ApiError::Other(
                    "provider-key auth requires a bind naming the target profile: the minted key \
                     must land in that profile's credential slot for the model broker to use it"
                        .into(),
                )
            })?;
            let profile = bind.profile.clone();
            let credential_ref = profile.as_str().to_string();
            store
                .set(&credential_ref, &outcome.credential_blob)
                .map_err(|e| ApiError::Other(format!("credential set: {e}")))?;
            // Replacing the profile's provider key must invalidate leases minted against any prior
            // material — bump the authority's lease epoch after the store write (as `credential_set`).
            if let Some(revoker) = &self.credential_revoker {
                revoker.revoke_profile(&credential_ref);
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

        Ok(AuthCompleteResponse {
            credential_ref,
            account_label: outcome.account_label,
            transport_instance: outcome.transport_instance,
            bound_profile,
        })
    }
}
