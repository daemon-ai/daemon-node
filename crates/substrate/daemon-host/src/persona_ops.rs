// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `PersonaOps` — the narrow persona (SOUL.md) seam behind the wire `SoulGet`/`SoulSet` ops
//! (wire v36).
//!
//! The node interface ([`NodeApiImpl`](crate::NodeApiImpl)) resolves the target profile spec and
//! enforces the wire-level guards here — an unknown profile id fails with the same not-found error
//! the other profile ops raise, and a `SoulSet` against a Foreign-engine profile is rejected typed
//! (an ACP agent owns its own prompt; there is no persona to set) — then delegates the actual
//! persona IO to an injected [`PersonaOps`]. The real implementation (the `PersonaStore` over
//! per-profile SOUL.md files) owns validation, threat-scanning, the size cap, atomic writes, and
//! the revision log; it is bound at node assembly via
//! [`NodeApiImpl::with_persona_ops`](crate::NodeApiImpl::with_persona_ops). A node assembled
//! without a persona backend resolves both ops to [`ApiError::Unsupported`].
//!
//! Guard order matters: the persona backend seeds a default SOUL.md on a miss, so the profile
//! existence check MUST run before any delegation — otherwise a read of a bogus id would
//! materialize an orphan persona doc for a profile that does not exist.

use async_trait::async_trait;
use daemon_api::{ApiError, EngineSelector, ProfileSpec};

/// The persona (SOUL.md) backend the wire `SoulGet`/`SoulSet` ops delegate to. Implemented by the
/// node-side `PersonaStore` adapter (bound at assembly); the implementation owns scan/cap/atomic
/// write + revision logging — the handlers never re-implement any of it.
#[async_trait]
pub trait PersonaOps: Send + Sync {
    /// Read the persona (SOUL.md) text for `profile_id` (seeding a default on first read is the
    /// implementation's prerogative — the caller has already proven the profile exists).
    async fn soul_get(&self, profile_id: &str) -> Result<String, ApiError>;
    /// Replace the persona (SOUL.md) text for `profile_id` (validate, scan, cap, atomic-write,
    /// revision-log — all implementation-owned).
    async fn soul_set(&self, profile_id: &str, text: &str) -> Result<(), ApiError>;
}

/// The `SoulGet` flow over a pre-fetched profile lookup: an unknown id fails with the same
/// not-found error the other profile ops raise ([`ApiError::UnknownSession`], the `profile_err`
/// mapping of a store miss) WITHOUT touching the persona backend (which seeds SOUL.md on a miss);
/// a known profile delegates the read. Reads are not Foreign-gated: only `SoulSet` rejects.
pub(crate) async fn soul_get_guarded(
    persona: &dyn PersonaOps,
    spec: Option<&ProfileSpec>,
    id: &str,
) -> Result<String, ApiError> {
    if spec.is_none() {
        return Err(ApiError::UnknownSession(id.to_string()));
    }
    persona.soul_get(id).await
}

/// The `SoulSet` flow over a pre-fetched profile lookup: an unknown id fails not-found (as
/// [`soul_get_guarded`]); a Foreign-engine profile is rejected typed (its agent owns its own
/// prompt — there is no persona to set); a Core profile delegates the write to the backend
/// (which validates/scans/caps and revision-logs).
pub(crate) async fn soul_set_guarded(
    persona: &dyn PersonaOps,
    spec: Option<&ProfileSpec>,
    id: &str,
    text: &str,
) -> Result<(), ApiError> {
    let Some(spec) = spec else {
        return Err(ApiError::UnknownSession(id.to_string()));
    };
    if matches!(spec.engine, EngineSelector::Foreign { .. }) {
        return Err(ApiError::Unsupported(format!(
            "profile `{id}` runs a Foreign engine (its agent owns its prompt); no persona to set"
        )));
    }
    persona.soul_set(id, text).await
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use daemon_api::ProviderSelector;

    use super::*;

    /// A recording mock backend: `soul_get`/`soul_set` log every call, back a plain map, and
    /// never guard anything themselves (the guards under test live in this module).
    #[derive(Default)]
    struct MockPersonaOps {
        souls: Mutex<HashMap<String, String>>,
        calls: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl PersonaOps for MockPersonaOps {
        async fn soul_get(&self, profile_id: &str) -> Result<String, ApiError> {
            self.calls.lock().unwrap().push(format!("get {profile_id}"));
            Ok(self
                .souls
                .lock()
                .unwrap()
                .get(profile_id)
                .cloned()
                .unwrap_or_else(|| "seeded default persona".to_string()))
        }

        async fn soul_set(&self, profile_id: &str, text: &str) -> Result<(), ApiError> {
            self.calls.lock().unwrap().push(format!("set {profile_id}"));
            self.souls
                .lock()
                .unwrap()
                .insert(profile_id.to_string(), text.to_string());
            Ok(())
        }
    }

    fn core_spec(id: &str) -> ProfileSpec {
        ProfileSpec::new(id, ProviderSelector::Mock, "m")
    }

    fn foreign_spec(id: &str) -> ProfileSpec {
        ProfileSpec {
            engine: EngineSelector::Foreign {
                agent: "gemini".into(),
            },
            ..ProfileSpec::new(id, ProviderSelector::Mock, "")
        }
    }

    #[tokio::test]
    async fn soul_get_unknown_profile_is_not_found_and_never_delegates() {
        // The backend seeds SOUL.md on a miss, so a bogus id must fail BEFORE delegation —
        // otherwise the read would materialize an orphan persona doc for a nonexistent profile.
        let mock = MockPersonaOps::default();
        let err = soul_get_guarded(&mock, None, "ghost").await.unwrap_err();
        assert!(
            matches!(&err, ApiError::UnknownSession(id) if id == "ghost"),
            "unknown id must raise the profile-op not-found error, got: {err:?}"
        );
        assert!(
            mock.calls.lock().unwrap().is_empty(),
            "the persona backend must never be touched for an unknown profile"
        );
    }

    #[tokio::test]
    async fn soul_get_known_profile_delegates_the_read() {
        let mock = MockPersonaOps::default();
        mock.souls
            .lock()
            .unwrap()
            .insert("work".into(), "You are a focused work assistant.".into());
        let spec = core_spec("work");
        let text = soul_get_guarded(&mock, Some(&spec), "work").await.unwrap();
        assert_eq!(text, "You are a focused work assistant.");
        assert_eq!(mock.calls.lock().unwrap().as_slice(), ["get work"]);
    }

    #[tokio::test]
    async fn soul_get_foreign_profile_is_not_rejected() {
        // Only SoulSet Foreign-gates: a read on a Foreign profile delegates (clients hide the
        // persona UI for foreign profiles; the backend decides what a read yields).
        let mock = MockPersonaOps::default();
        let spec = foreign_spec("acp");
        assert!(soul_get_guarded(&mock, Some(&spec), "acp").await.is_ok());
        assert_eq!(mock.calls.lock().unwrap().as_slice(), ["get acp"]);
    }

    #[tokio::test]
    async fn soul_set_unknown_profile_is_not_found_and_never_delegates() {
        let mock = MockPersonaOps::default();
        let err = soul_set_guarded(&mock, None, "ghost", "text")
            .await
            .unwrap_err();
        assert!(
            matches!(&err, ApiError::UnknownSession(id) if id == "ghost"),
            "unknown id must raise the profile-op not-found error, got: {err:?}"
        );
        assert!(mock.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn soul_set_foreign_profile_is_rejected_typed_and_never_delegates() {
        // ACP agents own their prompts: a Foreign-engine profile has no persona to set, and the
        // rejection must be typed (Unsupported) and must never reach the backend.
        let mock = MockPersonaOps::default();
        let spec = foreign_spec("acp");
        let err = soul_set_guarded(&mock, Some(&spec), "acp", "nope")
            .await
            .unwrap_err();
        assert!(
            matches!(&err, ApiError::Unsupported(msg) if msg.contains("Foreign engine")),
            "a Foreign-engine profile must reject SoulSet typed, got: {err:?}"
        );
        assert!(
            mock.calls.lock().unwrap().is_empty(),
            "the persona backend must never be touched for a Foreign profile"
        );
    }

    #[tokio::test]
    async fn soul_set_core_profile_delegates_the_write() {
        let mock = MockPersonaOps::default();
        let spec = core_spec("work");
        soul_set_guarded(&mock, Some(&spec), "work", "Be terse.")
            .await
            .unwrap();
        assert_eq!(mock.calls.lock().unwrap().as_slice(), ["set work"]);
        assert_eq!(
            mock.souls.lock().unwrap().get("work").map(String::as_str),
            Some("Be terse.")
        );
    }
}
