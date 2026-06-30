// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The authentication / authorization audit sink (Auth 5).
//!
//! Authn/authz events — login success/failure, permission denials, and the admin access-control
//! mutations (user CRUD, role changes, session revocation) — are recorded into the node's existing
//! verifiable journal ([`crate::journal`]) on a dedicated `node-auth` stream, append-only and
//! tamper-evident (each event seals its segment, chaining onto the prior). The same handle is shared
//! by the transport (login + denial events; see [`crate::authn`]/[`crate::socket`]) and the node
//! interface (admin events; see `node_api::access`), so every auth event rides one chain.
//!
//! **No credential material ever enters a payload.** The `detail` of every record is a plain,
//! non-secret string (`user_id=…`, `username=…`, `method=…`, `roles=[…]`, `op=…`, `cap=…`). Passwords,
//! session tokens, and PHC/SCRAM material are NEVER formatted into an audit record — callers pass only
//! identifiers and outcomes. This is the auth analogue of the credential-audit pattern already in
//! [`crate::journal`].

use std::sync::{Arc, Mutex};

use daemon_common::{JournalStreamId, UnitId};
use daemon_store::SessionStore;
use daemon_telemetry::TraceSigner;

use crate::journal::JournalSink;

/// The journal stream (unit) name the auth audit chain is recorded under.
pub const AUTH_JOURNAL_UNIT: &str = "node-auth";

/// A shared, lazily-opened sink that records authn/authz events onto the verifiable `node-auth`
/// journal stream. Cheap to clone-share via `Arc`. Best-effort: a journaling hiccup is logged and
/// swallowed so it never blocks login or an admin op.
pub struct AuthAudit {
    store: Arc<dyn SessionStore>,
    signer: Arc<TraceSigner>,
    /// One long-lived sink so the chain links per event (each `seal` advances the segment). Opened
    /// on first use.
    sink: Mutex<Option<Arc<JournalSink>>>,
}

impl AuthAudit {
    /// Build an auth-audit sink over the node store + journal signer.
    pub fn new(store: Arc<dyn SessionStore>, signer: Arc<TraceSigner>) -> Self {
        Self {
            store,
            signer,
            sink: Mutex::new(None),
        }
    }

    /// Build it as an `Arc` (the shared shape every holder keeps).
    pub fn shared(store: Arc<dyn SessionStore>, signer: Arc<TraceSigner>) -> Arc<Self> {
        Arc::new(Self::new(store, signer))
    }

    fn sink(&self) -> Arc<JournalSink> {
        let mut guard = self.sink.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(Arc::new(JournalSink::new(
                self.store.clone(),
                self.signer.clone(),
                JournalStreamId::unit(&UnitId::new(AUTH_JOURNAL_UNIT)),
            )));
        }
        guard.as_ref().unwrap().clone()
    }

    /// Record one auth event (kind + non-secret detail) and seal the segment so it is durable and
    /// verifiable. Best-effort.
    pub async fn record(&self, kind: &str, detail: String) {
        let sink = self.sink();
        if let Err(e) = sink.record_management(kind.to_string(), detail).await {
            tracing::warn!(error = %e, kind, "auth audit: record failed");
            return;
        }
        if let Err(e) = sink.seal().await {
            tracing::warn!(error = %e, kind, "auth audit: seal failed");
        }
    }

    // --- typed helpers (the only place the audit `kind`s + detail formats are defined) ----------
    // Each detail carries identifiers/outcomes only — never a password, token, or PHC/SCRAM blob.

    /// A successful login (mechanism resolved a principal + minted a token).
    pub async fn login_ok(&self, user_id: &str, method: &str) {
        self.record(
            "auth.login_ok",
            format!("user_id={user_id} method={method}"),
        )
        .await;
    }

    /// A failed login. `username` is the *attempted* identity when the mechanism exposes it
    /// (PLAIN/EXTERNAL); SCRAM failures record the mechanism only (the username is not proven). Never
    /// records the supplied password.
    pub async fn login_fail(&self, mechanism: &str, username: Option<&str>) {
        let detail = match username {
            Some(u) => format!("mechanism={mechanism} username={u}"),
            None => format!("mechanism={mechanism}"),
        };
        self.record("auth.login_fail", detail).await;
    }

    /// A permission denial at the capability gate (authenticated-but-forbidden, or unauthenticated).
    pub async fn permission_denied(&self, op: &str, conn_id: Option<u64>, reason: &str) {
        let detail = match conn_id {
            Some(c) => format!("op={op} conn_id={c} reason={reason}"),
            None => format!("op={op} reason={reason}"),
        };
        self.record("auth.permission_denied", detail).await;
    }

    /// A user was created (records id/username/roles — never the password).
    pub async fn user_created(&self, user_id: &str, username: &str, roles: &[String]) {
        self.record(
            "auth.user_created",
            format!("user_id={user_id} username={username} roles={roles:?}"),
        )
        .await;
    }

    /// A user was disabled or re-enabled.
    pub async fn user_disabled(&self, user_id: &str, disabled: bool) {
        let kind = if disabled {
            "auth.user_disabled"
        } else {
            "auth.user_enabled"
        };
        self.record(kind, format!("user_id={user_id}")).await;
    }

    /// A user's role set was replaced.
    pub async fn roles_changed(&self, user_id: &str, roles: &[String]) {
        self.record(
            "auth.roles_changed",
            format!("user_id={user_id} roles={roles:?}"),
        )
        .await;
    }

    /// A user's password was reset (records the id only — never the new password or its hash).
    pub async fn password_reset(&self, user_id: &str) {
        self.record("auth.password_reset", format!("user_id={user_id}"))
            .await;
    }

    /// A user's sessions were revoked.
    pub async fn sessions_revoked(&self, user_id: &str) {
        self.record("auth.sessions_revoked", format!("user_id={user_id}"))
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_common::ContentHash;
    use daemon_store::{InMemoryStore, SessionStore};
    use daemon_telemetry::{
        decode_entry, verify_segment, JournalPayload, SegmentInput, GENESIS_ROOT,
    };

    fn stream() -> JournalStreamId {
        JournalStreamId::unit(&UnitId::new(AUTH_JOURNAL_UNIT))
    }

    /// Recompute + signature-verify one sealed `(stream, segment)` against the signer's key, chaining
    /// onto the prior segment's root (the same check the production history reader performs).
    async fn segment_verifies(
        store: &dyn SessionStore,
        signer: &TraceSigner,
        segment: u64,
    ) -> bool {
        let s = stream();
        let Some(seg) = store.load_trace_segment(&s, segment).await else {
            return false;
        };
        let Some(committed) = seg.committed else {
            return false;
        };
        let prior = if segment == 0 {
            GENESIS_ROOT
        } else {
            match store
                .load_trace_segment(&s, segment - 1)
                .await
                .and_then(|p| p.committed)
            {
                Some(c) => c.root,
                None => return false,
            }
        };
        let entries: Vec<(u64, Vec<u8>, ContentHash)> = seg
            .entries
            .into_iter()
            .map(|e| (e.seq, e.bytes, e.content_hash))
            .collect();
        let input = SegmentInput {
            stream: &s,
            segment,
            prior,
            entries: &entries,
        };
        verify_segment(
            &input,
            &committed.root,
            &committed.signature,
            &signer.verifying_key(),
        )
        .is_ok()
    }

    #[tokio::test]
    async fn records_events_and_journal_verifies() {
        let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
        let signer = Arc::new(TraceSigner::generate());
        let audit = AuthAudit::new(store.clone(), signer.clone());

        // Each helper records one event and seals its segment (chaining onto the prior).
        audit.login_ok("u-1", "scram").await;
        audit.login_fail("PLAIN", Some("mallory")).await;
        audit
            .user_created("u-2", "bob", &["user".to_string()])
            .await;
        audit.password_reset("u-2").await;
        audit
            .permission_denied("UserCreate", Some(7), "forbidden: requires AccessAdmin")
            .await;

        // Five sealed, individually-verifiable segments (the chain holds).
        for seg in 0..5u64 {
            assert!(
                segment_verifies(store.as_ref(), &signer, seg).await,
                "segment {seg} must verify"
            );
        }

        // The recorded kinds + details are present and well-formed.
        let page = store.load_journal(&stream(), 0, 100).await;
        let decoded: Vec<_> = page
            .entries
            .iter()
            .filter_map(|je| decode_entry(&je.entry.bytes).ok())
            .collect();
        assert_eq!(decoded.len(), 5, "all five events landed");
        let kinds: Vec<&str> = decoded.iter().map(|v| v.kind.as_str()).collect();
        assert!(kinds.contains(&"auth.login_ok"));
        assert!(kinds.contains(&"auth.login_fail"));
        assert!(kinds.contains(&"auth.user_created"));
        assert!(kinds.contains(&"auth.password_reset"));
        assert!(kinds.contains(&"auth.permission_denied"));
    }

    #[tokio::test]
    async fn audit_helpers_carry_no_secret_material() {
        // The typed helpers structurally cannot receive a password/token; assert their *details*
        // carry only identifiers/outcomes (a regression guard for the formats).
        let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
        let signer = Arc::new(TraceSigner::generate());
        let audit = AuthAudit::new(store.clone(), signer);
        audit.login_ok("u-1", "scram").await;
        audit
            .user_created("u-2", "bob", &["admin".to_string()])
            .await;
        audit.password_reset("u-2").await;

        let page = store.load_journal(&stream(), 0, 100).await;
        for je in &page.entries {
            let view = decode_entry(&je.entry.bytes).expect("decode");
            let detail = match view.payload {
                JournalPayload::Management { detail } => detail,
                _ => panic!("auth audit records are management payloads"),
            };
            for marker in ["$argon2", "$scram", "stored_key", "server_key"] {
                assert!(
                    !detail.contains(marker),
                    "audit detail must not contain credential material: {detail}"
                );
            }
        }
    }
}
