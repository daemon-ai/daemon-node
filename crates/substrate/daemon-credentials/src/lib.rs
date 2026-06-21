//! `daemon-credentials` — the credential authority.
//!
//! The host-owned authority backing the engine's §7 `CredentialProvider` port (host-spec §6). It
//! owns secret material and governance; it brokers **capability leases** — scoped, TTL-bounded,
//! signed [`CapabilityLease`](daemon_common::CapabilityLease)s — down the supervision tree, never
//! the long-lived key (except the `Bearer` mode, which hands a key over and is compensated by the
//! audit trail rather than the TTL).
//!
//! Three modes trade isolation against cost (`CredMode::{Native, Bearer, Proxied}`): a short-lived
//! provider token, a handed-over key, or proxied use where the key never leaves the owner. The
//! [`CredentialAuthority`] mints and verifies capabilities (ed25519 / Gordian Envelope, the same
//! stack as the phase-6 trace journal), governs fleet cost, and records every step to an audit log
//! the host journals into the verifiable trace. The recursive *serve-or-forward* brokering across
//! placement cuts lives in `daemon-host`; this crate is the owner endpoint and depends only on
//! `daemon-common` (+ the BC crypto crates).

#![forbid(unsafe_code)]

pub mod audit;
pub mod authority;
pub mod capability;
pub mod source;

pub use audit::{CredAuditKind, CredentialAuditEvent};
pub use authority::{AcquireCtx, CredentialAuthority};
pub use capability::{CapabilitySigner, CapabilityVerifyingKey};
pub use source::{CredentialSource, Provisioned, StubCredentialSource};

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_common::{CredError, CredId, CredMode, CredScope, ProfileRef};
    use std::sync::Arc;

    fn authority(mode: CredMode) -> CredentialAuthority {
        let signer = Arc::new(CapabilitySigner::generate());
        let source = Arc::new(StubCredentialSource::minting("openai", "sk-configured"));
        CredentialAuthority::new(
            CredScope::new(["openai"], ["chat", "embed"], Some(1_000)),
            mode,
            60_000,
            signer,
            source,
        )
    }

    #[test]
    fn native_lease_carries_token_and_verifies() {
        let auth = authority(CredMode::Native);
        let ctx = AcquireCtx::default();
        let lease = auth
            .acquire(
                &ctx,
                &ProfileRef::new("openai"),
                &CredScope::new(["openai"], ["chat"], None),
            )
            .unwrap();
        assert_eq!(lease.mode, CredMode::Native);
        assert!(lease.secret.is_some(), "native carries a token");
        auth.verify(&lease)
            .expect("a freshly minted capability verifies");
    }

    #[test]
    fn proxied_lease_hides_secret_until_use() {
        let auth = authority(CredMode::Proxied);
        let ctx = AcquireCtx::default();
        let lease = auth
            .acquire(
                &ctx,
                &ProfileRef::new("openai"),
                &CredScope::new(["openai"], ["chat"], None),
            )
            .unwrap();
        assert_eq!(lease.mode, CredMode::Proxied);
        assert!(
            lease.secret.is_none(),
            "proxied exposes no secret in the lease"
        );
        // The owner performs the call and returns only a result — never the raw key.
        let result = auth.use_capability(&ctx, &lease).unwrap();
        assert_ne!(
            result.expose(),
            "sk-configured",
            "the raw key must never cross back"
        );
        assert!(result.expose().starts_with("proxied-result:"));
    }

    #[test]
    fn bearer_mints_a_fresh_key_when_the_source_can() {
        let auth = authority(CredMode::Bearer);
        let ctx = AcquireCtx::default();
        let lease = auth
            .acquire(
                &ctx,
                &ProfileRef::new("openai"),
                &CredScope::new(["openai"], ["chat"], None),
            )
            .unwrap();
        assert_eq!(lease.mode, CredMode::Bearer);
        let secret = lease.secret.as_ref().unwrap().expose().to_string();
        assert!(
            secret.starts_with("sk-fresh-"),
            "fresh per-grant key, got {secret}"
        );
        // The issuance is recorded — the audit trail is the compensating control for bearer.
        let granted = auth
            .audit_log()
            .into_iter()
            .any(|e| e.kind == CredAuditKind::Grant);
        assert!(granted, "a bearer issuance must be audited");
    }

    #[test]
    fn attenuation_narrows_to_grant() {
        let auth = authority(CredMode::Native);
        let ctx = AcquireCtx::default();
        // Request more than the grant: extra profile/action and a bigger ceiling.
        let req = CredScope::new(["openai", "anthropic"], ["chat", "admin"], Some(10_000));
        let lease = auth
            .acquire(&ctx, &ProfileRef::new("openai"), &req)
            .unwrap();
        assert!(lease.scope.profiles.contains("openai"));
        assert!(!lease.scope.profiles.contains("anthropic"));
        assert!(lease.scope.actions.contains("chat"));
        assert!(
            !lease.scope.actions.contains("admin"),
            "admin is not in the grant"
        );
        assert_eq!(
            lease.scope.max_tokens,
            Some(1_000),
            "ceiling clamps to the grant"
        );
    }

    #[test]
    fn edited_capability_is_refused() {
        let auth = authority(CredMode::Native);
        let ctx = AcquireCtx::default();
        let mut lease = auth
            .acquire(
                &ctx,
                &ProfileRef::new("openai"),
                &CredScope::new(["openai"], ["chat"], None),
            )
            .unwrap();
        // Tamper with the scope after signing.
        lease.scope.actions.insert("admin".into());
        assert_eq!(auth.verify(&lease).unwrap_err(), CredError::BadSignature);
    }

    #[test]
    fn expired_capability_is_refused() {
        let signer = Arc::new(CapabilitySigner::generate());
        let source = Arc::new(StubCredentialSource::new("openai", "sk"));
        // Zero TTL: the lease is already expired when minted.
        let auth = CredentialAuthority::new(
            CredScope::new(["openai"], ["chat"], None),
            CredMode::Native,
            0,
            signer,
            source,
        );
        let ctx = AcquireCtx::default();
        let lease = auth
            .acquire(
                &ctx,
                &ProfileRef::new("openai"),
                &CredScope::new(["openai"], ["chat"], None),
            )
            .unwrap();
        assert_eq!(auth.verify(&lease).unwrap_err(), CredError::Expired);
    }

    #[test]
    fn revoke_blocks_use_and_revokes_at_source() {
        let signer = Arc::new(CapabilitySigner::generate());
        let source = Arc::new(StubCredentialSource::minting("openai", "sk"));
        let auth = CredentialAuthority::new(
            CredScope::new(["openai"], ["chat"], None),
            CredMode::Bearer,
            60_000,
            signer,
            source.clone(),
        );
        let ctx = AcquireCtx::default();
        let lease = auth
            .acquire(
                &ctx,
                &ProfileRef::new("openai"),
                &CredScope::new(["openai"], ["chat"], None),
            )
            .unwrap();
        auth.revoke(&ctx, &lease.cap_id);
        assert!(
            source.is_revoked(&lease.cap_id),
            "fresh key revoked at the source"
        );
        assert!(
            auth.use_capability(&ctx, &lease).is_err(),
            "revoked capability cannot be used"
        );
    }

    #[test]
    fn cost_ceiling_throttles_budget() {
        let auth = authority(CredMode::Native).with_cost_ceiling(100);
        // Under the ceiling: headroom remains.
        let b1 = auth.charge(60);
        assert_eq!(b1.tokens, Some(40));
        // Over the ceiling: throttled to zero.
        let b2 = auth.charge(60);
        assert_eq!(b2.tokens, Some(0));
        assert_eq!(auth.spent_tokens(), 120);
    }

    #[test]
    fn unknown_cap_id_for_proxied_use_is_unavailable() {
        let auth = authority(CredMode::Proxied);
        let ctx = AcquireCtx::default();
        let mut lease = auth
            .acquire(
                &ctx,
                &ProfileRef::new("openai"),
                &CredScope::new(["openai"], ["chat"], None),
            )
            .unwrap();
        // Forge a different cap_id (still must fail signature, but ensure no panic path).
        lease.cap_id = CredId::new("forged");
        assert!(auth.use_capability(&ctx, &lease).is_err());
    }
}
