// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The in-process account bring-up seam (daemon-event-io-spec §5.9.4): the read side of the
//! account→profile binding a chat-transport adapter uses to enumerate the accounts it owns and
//! resolve / write back their credential material (never over the wire).

use super::*;

/// One transport-instance account a profile is bound to (daemon-event-io-spec §5.9.4) — the read
/// side of [`ProfileSpec::bound_accounts`](daemon_api::ProfileSpec). It names *which* profile owns
/// the account, the instance-qualified [`TransportId`] (`matrix/@bot:hs.org`), and the
/// `credential_ref` of its opaque session blob in the `CredentialStore`. The secret itself is *not*
/// carried here — resolving it is a separate, in-process-only call ([`AccountProvisioning::account_credential`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvisionedAccount {
    /// The profile that declared (and runs) this account.
    pub profile: ProfileRef,
    /// The instance-qualified transport id (`matrix/@bot:hs.org`).
    pub transport_instance: TransportId,
    /// The `CredentialStore` key naming this account's opaque session blob.
    pub credential_ref: String,
}

/// The **in-process** account bring-up seam (daemon-event-io-spec §5.9.4): a chat-transport adapter
/// that holds the live [`NodeApiImpl`] uses this to discover the accounts it owns (across every
/// profile, by transport *family*), resolve each account's credential material, and write back a
/// refreshed blob after a token refresh. It is the read side of the account→profile binding the host
/// already derives routing from.
///
/// This is deliberately **not** part of the wire [`daemon_api::NodeApi`] surface — like
/// [`DeliveryHost`], it is a live in-process handle. [`Self::account_credential`] returns the *full*
/// secret blob, which never crosses the wire (the wire `CredentialApi` only lists redacted metadata);
/// enumeration ([`Self::bound_accounts`]) is kept separate from secret resolution so an adapter (or a
/// status view) can list accounts without touching secrets (least-privilege).
pub trait AccountProvisioning: Send + Sync {
    /// Every bound account whose `transport_instance` is in `transport_family` (the segment before the
    /// first `/`, e.g. `"matrix"` matches `matrix/@a:hs` and `matrix/@b:hs` but not `slack/…`), across
    /// all profiles. Empty if no profile store is wired or no account matches.
    fn bound_accounts(&self, transport_family: &str) -> Vec<ProvisionedAccount>;

    /// Resolve an account's full credential blob by its `credential_ref` (in-process only; the secret
    /// never crosses the wire). `None` if no credential store is wired or the ref is unknown.
    fn account_credential(&self, credential_ref: &str) -> Option<String>;

    /// Persist a refreshed credential `blob` under `credential_ref` — the token-refresh write-back
    /// seam (the `CredentialStore` is the system of record; e.g. driven from a `set_session_callback`).
    fn store_account_credential(&self, credential_ref: &str, blob: &str) -> Result<(), ApiError>;
}

impl AccountProvisioning for NodeApiImpl {
    fn bound_accounts(&self, transport_family: &str) -> Vec<ProvisionedAccount> {
        let Some(profiles) = self.profiles.as_ref() else {
            return Vec::new();
        };
        let Ok(specs) = profiles.list() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for spec in specs {
            for account in &spec.bound_accounts {
                // Family = the segment before the first `/` (the instance-qualified TransportId
                // convention, matching `routing::TransportPattern::Family`).
                if account.transport_instance.split('/').next() == Some(transport_family) {
                    out.push(ProvisionedAccount {
                        profile: ProfileRef::new(&spec.id),
                        transport_instance: TransportId::new(account.transport_instance.clone()),
                        credential_ref: account.credential_ref.clone(),
                    });
                }
            }
        }
        out
    }

    fn account_credential(&self, credential_ref: &str) -> Option<String> {
        self.credentials.as_ref()?.get(credential_ref)
    }

    fn store_account_credential(&self, credential_ref: &str, blob: &str) -> Result<(), ApiError> {
        let store = self
            .credentials
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("credential management not available".into()))?;
        store
            .set(credential_ref, blob)
            .map_err(|e| ApiError::Other(format!("credential set: {e}")))
    }
}
