// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::*;

#[async_trait]
impl SessionApi for NodeApiImpl {
    async fn submit(&self, session: SessionId, command: AgentCommand) -> Result<(), ApiError> {
        // Guard-rail: claim the session for the live lifecycle (rejects an id already durable-managed).
        self.claim(&session, Lifecycle::Live)?;
        // Auth 4: own-or-`SessionControlAny`. An `Absent` (brand-new) session passes here, then
        // `note_activity` stamps the caller as owner — checked BEFORE `note_activity` so a foreign
        // caller never mutates last-activity / the FTS index.
        self.require_session_access(&session, true).await?;
        self.note_activity(&session, &command).await;
        self.live.submit(session, command).await
    }

    async fn submit_from(
        &self,
        session: SessionId,
        origin: Origin,
        command: AgentCommand,
    ) -> Result<(), ApiError> {
        self.claim(&session, Lifecycle::Live)?;
        self.require_session_access(&session, true).await?;
        self.note_activity(&session, &command).await;
        self.live.submit_from(session, origin, command).await
    }

    async fn session_create(
        &self,
        session: Option<SessionId>,
        profile: Option<ProfileRef>,
    ) -> Result<SessionId, ApiError> {
        // Node-authoritative creation of a blank, profile-bound, UN-RUN session: the create-if-absent
        // body of `assign` (durable row + fresh snapshot + owner stamp) enriched with `bound_profile`,
        // MINUS `manager.wake()` — no turn runs and no engine is woken.
        let session = session.unwrap_or_else(mint_session_id);
        // Reserve the id for the live lifecycle: the GUI binds its composer to this id and opens it
        // with a live `StartTurn`, so claiming `Live` keeps that subsequent submit idempotent (a
        // `Durable` claim would make the first turn conflict with the guard-rail).
        self.claim(&session, Lifecycle::Live)?;
        // Auth 4: an `Absent` session passes; the durable-create + meta stamp below fixes ownership.
        self.require_session_access(&session, true).await?;
        // Resolve the profile to bind: an explicit ref, else the node's active default — so a blank
        // session still lands under an agent in the ByProfile roster.
        let bound = match profile {
            Some(p) => Some(p),
            None => self
                .profile_store()
                .ok()
                .and_then(|s| s.active().ok().flatten())
                .map(ProfileRef::new),
        };
        // Create-if-absent durable row with the engine's initial snapshot (the `assign` body).
        let created = if self.store.status(&session).await.is_none() {
            let blob = Snapshot::fresh(session.clone())
                .encode()
                .map_err(|e| ApiError::Other(format!("encode initial snapshot: {e}")))?;
            self.store
                .create_session(session.clone(), self.partition, blob)
                .await
                .map_err(|e| ApiError::Other(format!("create session: {e}")))?;
            true
        } else {
            false
        };
        // Bind `bound_profile` + stamp the owner on the durable host meta (read-modify-write, so a
        // pre-existing overlay/title is preserved and a re-create never clobbers an existing binding).
        let mut meta = self.store.session_meta(&session).await.unwrap_or_default();
        if meta.bound_profile.is_none() {
            meta.bound_profile = bound.clone();
        }
        if meta.owner.is_none() {
            meta.owner = current_principal().map(|p| p.user_id);
        }
        let _ = self.store.set_session_meta(&session, meta).await;
        // L3: the roster *set* changed — a client refetches the roster + the ByProfile query. This is
        // the existing `RosterChanged` the live `ensure()` path also emits.
        if let Some(feed) = self.node_feed() {
            let rev = feed.note_roster_change(&session);
            feed.emit(NodeEvent::RosterChanged { rev });
        }
        // #region agent log
        {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("/home/j/experiments/daemon/.cursor/debug-96b7ad.log")
            {
                let _ = writeln!(
                    f,
                    "{{\"sessionId\":\"96b7ad\",\"hypothesisId\":\"SESSION-CREATE\",\"location\":\"node:session_create\",\"message\":\"node created blank session\",\"data\":{{\"session\":\"{}\",\"profile\":\"{}\",\"created_row\":{}}},\"timestamp\":0}}",
                    session.as_str(),
                    bound.as_ref().map(|p| p.as_str()).unwrap_or(""),
                    created
                );
            }
        }
        // #endregion
        Ok(session)
    }

    async fn submit_as(&self, args: SubmitAsArgs) -> Result<(), ApiError> {
        let SubmitAsArgs {
            session,
            origin,
            command,
            profile,
        } = args;
        self.claim(&session, Lifecycle::Live)?;
        self.require_session_access(&session, true).await?;
        // Bind the explicit profile sticky-on-first-open (the same `ensure` seam `submit_routed`
        // uses), so a GUI can "open this chat as agent X" before the first turn submits.
        if profile.is_some() {
            self.live.ensure(&session, profile).await;
        }
        self.note_activity(&session, &command).await;
        match origin {
            Some(origin) => self.live.submit_from(session, origin, command).await,
            None => self.live.submit(session, command).await,
        }
    }

    async fn submit_routed(
        &self,
        origin: Origin,
        command: AgentCommand,
    ) -> Result<SessionId, ApiError> {
        // Resolve the origin through the §5.9 routing registry: session name, the profile that runs
        // it (agent selection), and where its replies post.
        let routing = self.routing.load();
        let resolved = routing.resolve(&origin);
        self.claim(&resolved.session, Lifecycle::Live)?;
        // Auth 4: own-or-`SessionControlAny` on the resolved session (new sessions pass and are
        // stamped by `note_activity`).
        self.require_session_access(&resolved.session, true).await?;
        // For session-opening commands, bind the resolved profile (sticky on first `ensure`) and seed
        // the resolved `Primary` before submitting, so routing owns agent-selection + delivery. Other
        // commands act on an already-open session whose profile/Primary were bound when it opened.
        if matches!(
            command,
            AgentCommand::StartTurn { .. }
                | AgentCommand::Steer { .. }
                | AgentCommand::Observe { .. }
        ) {
            self.live
                .ensure(&resolved.session, resolved.profile.clone())
                .await;
            self.live
                .seed_primary_target(&resolved.session, resolved.delivery.clone());
        }
        self.note_activity(&resolved.session, &command).await;
        self.live
            .submit_from(resolved.session.clone(), origin, command)
            .await?;
        Ok(resolved.session)
    }

    async fn poll(&self, session: SessionId, max: u32) -> Result<Vec<Outbound>, ApiError> {
        // Auth 4: own-or-`SessionControlAny` (the task's named control ops include `poll`).
        self.require_session_access(&session, true).await?;
        self.live.poll(&session, max)
    }

    async fn respond(&self, session: SessionId, response: HostResponse) -> Result<(), ApiError> {
        self.require_session_access(&session, true).await?;
        self.live.respond(&session, response)
    }

    async fn session_history(
        &self,
        session: SessionId,
        after_cursor: u64,
        max: u32,
    ) -> JournalPageView {
        // Auth 4 (read-of-one): own-or-`SessionSeeAll`. The wire return is non-fallible, so an
        // unauthorized read yields an empty page (no transcript leak) rather than an error.
        if self.require_session_access(&session, false).await.is_err() {
            return JournalPageView::default();
        }
        self.read_history(JournalStreamId::session(&session), after_cursor, max)
            .await
    }

    async fn log_after(
        &self,
        session: SessionId,
        after_seq: u64,
        max: u32,
    ) -> Result<LogPageView, ApiError> {
        Ok(self.live.log_after(&session, after_seq, max))
    }

    async fn subscribe(&self, session: SessionId, after_seq: u64) -> Result<LogStream, ApiError> {
        // Auth 4: own-or-`SessionControlAny` (a live subscription is a session-interaction op).
        self.require_session_access(&session, true).await?;
        Ok(self.live.subscribe(&session, after_seq))
    }

    async fn log_epoch(&self, session: SessionId) -> u64 {
        self.live.log_epoch(&session)
    }

    async fn delivery_targets(&self, session: SessionId) -> Vec<DeliveryTarget> {
        self.live.delivery_targets(&session)
    }

    async fn delivery_sessions(&self, transport: TransportId) -> Vec<SessionId> {
        self.live.delivery_sessions(&transport)
    }

    async fn handover(&self, session: SessionId, target: DeliveryTarget) -> Result<(), ApiError> {
        // Auth 4: own-or-`SessionControlAny`.
        self.require_session_access(&session, true).await?;
        self.live.handover(&session, target)
    }

    async fn record_meta(&self, args: RecordMetaArgs) -> Result<(), ApiError> {
        // Auth 4: own-or-`SessionControlAny` (writes into the session's live log).
        self.require_session_access(&args.session, true).await?;
        self.live.record_meta(args)
    }

    async fn set_session_model(
        &self,
        session: SessionId,
        model: String,
        provider: Option<ProviderSelector>,
    ) -> Result<(), ApiError> {
        // Auth 4: own-or-`SessionControlAny` (a per-session override write).
        self.require_session_access(&session, true).await?;
        // Persist the model/provider override on the session overlay (durable host-level metadata),
        // then apply it to the live actor in place when resident. A non-resident session picks it up
        // at its next (re)hydration via the overlay — so a switch is no longer lost on restart.
        let overlay = self
            .update_overlay(&session, |o| {
                o.model = Some(model.clone());
                if let Some(p) = provider {
                    o.provider = Some(p);
                }
            })
            .await;
        self.apply_overlay_live(&session, &overlay).await?;
        self.session_models.insert(session, model);
        Ok(())
    }

    async fn set_session_mode(
        &self,
        session: SessionId,
        mode: ApprovalMode,
    ) -> Result<(), ApiError> {
        // Auth 4: own-or-`SessionControlAny`.
        self.require_session_access(&session, true).await?;
        // Persist the edit-approval override on the overlay, then switch the live actor's policy in
        // place when resident (the live ParkingHandler reads `session_modes` to auto-allow vs park).
        let overlay = self
            .update_overlay(&session, |o| o.approval_mode = Some(mode))
            .await;
        self.apply_overlay_live(&session, &overlay).await?;
        // Keep the live mode cache populated even when not resident, so a freshly-resident actor's
        // ParkingHandler sees the persisted policy until `apply_overlay_live` refreshes it.
        self.session_modes
            .insert(session, approval_mode_to_policy(mode));
        Ok(())
    }

    async fn set_session_overlay(
        &self,
        session: SessionId,
        overlay: SessionOverlay,
    ) -> Result<(), ApiError> {
        // Auth 4: own-or-`SessionControlAny`.
        self.require_session_access(&session, true).await?;
        // The unified per-session override write: persist the whole overlay, then apply what can be
        // hot-applied to a resident actor (model/provider/approval). A tool-allowlist change takes
        // effect on the next (re)hydration (the live registry is fixed for an actor's lifetime).
        let persisted = self
            .update_overlay(&session, |o| *o = overlay.clone())
            .await;
        self.apply_overlay_live(&session, &persisted).await?;
        if let Some(model) = &persisted.model {
            self.session_models.insert(session, model.clone());
        }
        Ok(())
    }
}

/// Mint a fresh node-authoritative session id: the `s-<32 hex>` shape the GUI historically minted
/// client-side, now produced on the node from 16 random bytes so nothing is client-minted. A
/// getrandom failure is astronomically unlikely; fall back to a time-seeded id rather than panicking.
fn mint_session_id() -> SessionId {
    let mut bytes = [0u8; 16];
    if getrandom::getrandom(&mut bytes).is_err() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        bytes[..16].copy_from_slice(&nanos.to_le_bytes());
    }
    let mut hex = String::with_capacity(2 + bytes.len() * 2);
    hex.push_str("s-");
    for b in bytes {
        hex.push_str(&format!("{b:02x}"));
    }
    SessionId::new(hex)
}
