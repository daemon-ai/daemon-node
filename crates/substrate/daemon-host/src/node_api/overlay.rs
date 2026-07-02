// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Per-session [`SessionOverlay`] resolution: the persisted override codec, the read-modify-write
//! persistence path, and the live-actor apply (model/provider/approval) of an override.

use super::*;

/// Encode a [`SessionOverlay`] to the opaque CBOR blob the store persists (host-level metadata).
pub fn encode_overlay(overlay: &SessionOverlay) -> Vec<u8> {
    let mut buf = Vec::new();
    // A SessionOverlay is always serializable; a failure here is a bug, not a runtime condition.
    ciborium::into_writer(overlay, &mut buf).expect("encode SessionOverlay");
    buf
}

/// Decode a [`SessionOverlay`] from its persisted blob; an empty/malformed blob is the empty
/// (all-inherit) overlay, so a session with no recorded override resolves straight from its profile.
pub fn decode_overlay(bytes: &[u8]) -> SessionOverlay {
    if bytes.is_empty() {
        return SessionOverlay::default();
    }
    ciborium::from_reader(bytes).unwrap_or_default()
}

/// Translate a wire-level [`daemon_api::ApprovalMode`] into the engine's
/// [`daemon_core::ApprovalPolicy`].
pub(crate) fn approval_mode_to_policy(
    mode: daemon_api::ApprovalMode,
) -> daemon_core::ApprovalPolicy {
    match mode {
        daemon_api::ApprovalMode::Ask => daemon_core::ApprovalPolicy::Ask,
        daemon_api::ApprovalMode::AcceptEdits => daemon_core::ApprovalPolicy::AcceptEdits,
        daemon_api::ApprovalMode::AutoAllow => daemon_core::ApprovalPolicy::AutoAllow,
        daemon_api::ApprovalMode::Deny => daemon_core::ApprovalPolicy::Deny,
    }
}

impl NodeApiImpl {
    /// Resolve the [`ProfileSpec`] a session resolves its engine from: the session's persisted
    /// bound profile, falling back to the node's active default. The base for a live override apply.
    async fn session_spec(&self, session: &SessionId) -> Result<Option<ProfileSpec>, ApiError> {
        let bound = self
            .store
            .session_meta(session)
            .await
            .and_then(|m| m.bound_profile);
        match bound {
            Some(r) => self.resolve_profile(Some(r.as_str().to_string())),
            None => self.resolve_profile(None),
        }
    }

    /// Read-modify-write a session's persisted [`SessionOverlay`] (preserving its bound profile),
    /// returning the updated overlay. This is the single persistence path for every per-session
    /// override (model/provider/tools/approval), so an override is restored on rehydration.
    pub(crate) async fn update_overlay<F: FnOnce(&mut SessionOverlay)>(
        &self,
        session: &SessionId,
        f: F,
    ) -> SessionOverlay {
        let mut meta = self.store.session_meta(session).await.unwrap_or_default();
        let mut overlay = decode_overlay(&meta.overlay);
        f(&mut overlay);
        meta.overlay = encode_overlay(&overlay);
        let _ = self.store.set_session_meta(session, meta).await;
        overlay
    }

    /// Apply a session's overlay to a live (resident) actor in place: rebuild the provider for a
    /// model/provider override and switch the edit-approval policy for a mode override. A
    /// non-resident (durable) session is a no-op here — it picks the overlay up at its next
    /// (re)hydration. Tool-allowlist overrides are *not* hot-applied (the live registry is fixed for
    /// the actor's lifetime); they take effect on the next (re)hydration.
    ///
    /// A resident FOREIGN (ACP) session has no model provider to swap — a model/provider override
    /// is refused explicitly (the profile's engine owns its own model). Its approval-mode override
    /// IS honored: the shared `session_modes` map is what the ParkingHandler consults, for both
    /// backend kinds.
    pub(crate) async fn apply_overlay_live(
        &self,
        session: &SessionId,
        overlay: &SessionOverlay,
    ) -> Result<(), ApiError> {
        let foreign = self.live.resident_is_foreign(session) == Some(true);
        if foreign && (overlay.model.is_some() || overlay.provider.is_some()) {
            return Err(ApiError::Unsupported(
                "a foreign-engine (ACP) session has no model provider to override".into(),
            ));
        }
        if let Some(mode) = overlay.approval_mode {
            let policy = approval_mode_to_policy(mode);
            if let Some(handle) = self.live.handle_if_live(session) {
                handle.set_approval_policy(policy).await;
            }
            if self.live.is_resident(session) {
                self.session_modes.insert(session.clone(), policy);
            }
        }
        let Some(handle) = self.live.handle_if_live(session) else {
            return Ok(());
        };
        if overlay.model.is_some() || overlay.provider.is_some() {
            let factory = self.model_factory.as_ref().ok_or_else(|| {
                ApiError::Unsupported("per-session model switch is not available".into())
            })?;
            let mut spec = self.session_spec(session).await?.ok_or_else(|| {
                ApiError::Unsupported("no profile to derive a provider from".into())
            })?;
            overlay.apply_to(&mut spec);
            handle.set_provider((factory)(&spec)).await;
        }
        Ok(())
    }
}
