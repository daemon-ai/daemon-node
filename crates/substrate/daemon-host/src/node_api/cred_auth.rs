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

    async fn auth_complete(
        &self,
        req: AuthCompleteRequest,
    ) -> Result<AuthCompleteResponse, ApiError> {
        let flows = self
            .auth_flows
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("interactive auth not available".into()))?;
        // Pull the parked flow out (and its bind request) *before* awaiting the family completion, so
        // the registry lock is never held across the network round-trip the family performs.
        let (flow, bind) = flows.take(&req.flow_id)?;
        let outcome = flow.complete(&req.callback).await?;

        // The credential ref: a bind-supplied override wins over the family-derived default, so an
        // operator can pin where the blob lands; otherwise the family names it (e.g. by resolved user).
        let credential_ref = bind
            .as_ref()
            .and_then(|b| b.credential_ref.clone())
            .unwrap_or_else(|| outcome.credential_ref.clone());

        let store = self
            .credentials
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("credential management not available".into()))?;
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
