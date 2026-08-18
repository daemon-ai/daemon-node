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

    /// Apply a session's overlay to a live (resident, Foreign-only post-retire) backend in place.
    /// A non-resident (durable Core) session is a no-op here — the overlay changes its profile
    /// inputs (`ProfileKey`), so the next hydrate rebuilds the engine under the new resolution
    /// (including a lingering resident incarnation, which rebuilds at its very next turn).
    ///
    /// A resident FOREIGN (ACP) session has no model provider to swap — a provider override is
    /// refused explicitly (the profile's engine owns its own model). Its model override routes to
    /// the live backend (`set_foreign_model`), and its approval-mode override IS honored: the
    /// shared `session_modes` map is what the ParkingHandler consults.
    pub(crate) async fn apply_overlay_live(
        &self,
        session: &SessionId,
        overlay: &SessionOverlay,
    ) -> Result<(), ApiError> {
        let foreign = self.live.resident_is_foreign(session) == Some(true);
        // A foreign engine has no genai provider knob (its provider is fixed by the profile's
        // foreign backend); a provider override is meaningless. A model override IS honored below,
        // routed to the live foreign backend rather than the provider factory.
        if foreign && overlay.provider.is_some() {
            return Err(ApiError::Unsupported(
                "a foreign-engine (ACP) session has no model provider to override".into(),
            ));
        }
        // Edit-approval mode: the shared `session_modes` map is what the live ParkingHandler
        // consults (Foreign residencies); durable Core turns read the persisted overlay.
        if let Some(mode) = overlay.approval_mode {
            if self.live.is_resident(session) {
                self.session_modes
                    .insert(session.clone(), approval_mode_to_policy(mode));
            }
        }
        // Foreign model override (Phase 3): route the change to the live backend — a foreign ACP
        // `AgentNative` session issues a `set_config_option`; a gateway-routed `NodeProvider` session
        // re-binds its per-session token. The persisted `overlay.model` already re-steers the backend
        // on the next (re)open, so this is the *live* half.
        if foreign {
            if let Some(model) = &overlay.model {
                self.live.set_foreign_model(session, model.clone()).await?;
            }
        }
        Ok(())
    }
}
