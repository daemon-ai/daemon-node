// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Canonical provider-vendor identity (credential plan Phase 2).
//!
//! One vendor historically carries three spellings: the catalog/discovery id (`open_router` —
//! genai's `AdapterKind::as_lower_str()`, which is also the model-id namespace `open_router::…`),
//! the interactive-auth family (`provider/openrouter`), and the account label (`openrouter`).
//! This module is the single reconciliation point:
//!
//! - The **canonical vendor id** is the catalog id / model-namespace segment. It is what profile
//!   creation can derive mechanically from any namespaced model id — no table required — so every
//!   vendor (including registered custom providers) scopes correctly without registration here.
//! - The **provider-global credential ref** is [`provider_credential_ref`]:
//!   `provider/<canonical>` (e.g. `provider/open_router`). One shared credential per vendor;
//!   profiles point at it via `ProfileSpec::credential_ref` (a reference, not a copy).
//! - The [`PROVIDER_VENDORS`] table maps each curated interactive-auth family to its canonical id
//!   so a sign-in completion mints into the SAME ref a profile resolution reads. Only families
//!   that mint provider keys need an entry; key-in-field vendors never consult the table.

/// The node-managed credential-ref namespace for provider-global (cross-profile) credentials.
/// Refs under this prefix are minted by the node (profile creation, sign-in completion, and the
/// `CredentialSet` paste redirect) — clients never invent them.
pub const PROVIDER_REF_PREFIX: &str = "provider/";

/// One curated vendor's reconciled identities (see the module docs for which is canonical).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderVendor {
    /// The canonical vendor id: the catalog/discovery id and model-id namespace (`open_router`).
    pub canonical: &'static str,
    /// The interactive-auth family serving this vendor's sign-in (`provider/openrouter`).
    pub auth_family: &'static str,
    /// The fixed account label its completions carry (`openrouter`).
    pub account_label: &'static str,
}

/// Curated vendors with an interactive sign-in family. Key-in-field vendors are absent by design:
/// their canonical id comes straight from the model namespace and needs no reconciliation.
pub const PROVIDER_VENDORS: &[ProviderVendor] = &[
    ProviderVendor {
        canonical: "open_router",
        auth_family: "provider/openrouter",
        account_label: "openrouter",
    },
    ProviderVendor {
        canonical: "huggingface",
        auth_family: "provider/huggingface",
        account_label: "huggingface",
    },
    ProviderVendor {
        canonical: "github_copilot",
        auth_family: "provider/github_copilot",
        account_label: "github_copilot",
    },
];

/// The provider-global credential ref for a canonical vendor id: `provider/<canonical>`.
pub fn provider_credential_ref(canonical: &str) -> String {
    format!("{PROVIDER_REF_PREFIX}{canonical}")
}

/// Look up a curated vendor by its interactive-auth family (`provider/openrouter` →
/// `open_router`). `None` for non-provider families (`oauth2`, `matrix/*`, `agent/*`, …).
pub fn vendor_for_auth_family(family: &str) -> Option<&'static ProviderVendor> {
    PROVIDER_VENDORS.iter().find(|v| v.auth_family == family)
}

/// Whether `credential_ref` lives in the node-managed provider-global namespace.
pub fn is_provider_credential_ref(credential_ref: &str) -> bool {
    credential_ref.starts_with(PROVIDER_REF_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ref_uses_the_canonical_id() {
        assert_eq!(
            provider_credential_ref("open_router"),
            "provider/open_router"
        );
    }

    #[test]
    fn family_lookup_reconciles_the_three_spellings() {
        let v = vendor_for_auth_family("provider/openrouter").expect("curated");
        assert_eq!(v.canonical, "open_router");
        assert_eq!(v.account_label, "openrouter");
        assert!(vendor_for_auth_family("oauth2").is_none());
        assert!(vendor_for_auth_family("agent/claude/login").is_none());
    }

    #[test]
    fn provider_ref_namespace_detection() {
        assert!(is_provider_credential_ref("provider/open_router"));
        assert!(!is_provider_credential_ref("oauth2/someone"));
        assert!(!is_provider_credential_ref("my-profile"));
    }

    #[test]
    fn table_families_all_live_under_the_provider_prefix() {
        for v in PROVIDER_VENDORS {
            assert!(v.auth_family.starts_with(PROVIDER_REF_PREFIX), "{v:?}");
        }
    }
}
