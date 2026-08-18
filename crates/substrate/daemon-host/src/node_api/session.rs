// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::*;

#[async_trait]
impl SessionApi for NodeApiImpl {
    async fn submit(&self, session: SessionId, command: AgentCommand) -> Result<(), ApiError> {
        // Stage-5 cutover (session-unification §8): a non-resident Core session's commands are
        // homed on the durable rail — splice + wake for input, the AttachmentHub for control.
        if self.cutover_routes(&session).await {
            let _auth = self.require_session_access(&session, true).await?;
            self.bind_profile_on_first_open(&session, None).await;
            // Live parity: an opening StartTurn seeds the Primary reply sink from the generic
            // `api` origin (the same seam `LiveSessions::submit` rides); handover re-points later.
            if matches!(command, AgentCommand::StartTurn { .. }) {
                self.attach_hub(&session)
                    .await
                    .seed_primary_target(internals::api_origin().primary_target());
            }
            self.note_activity(&session, &command).await;
            return self.submit_attached(&session, command, None).await;
        }
        // F4 durable-resume: a `StartTurn`/`Steer` at a PARKED-DURABLE session rides the durable
        // inbox rail (typed splice into the durable transcript + wake) instead of opening a
        // divergent fresh live incarnation. Auth 4 is enforced before enqueuing (own-or-operator,
        // the same gate the live path uses); a settled/absent session falls through to live.
        if let Some((kind, msg)) = self.durable_resume_input(&session, &command).await {
            self.require_session_access(&session, true).await?;
            return self
                .enqueue_durable_input(&session, kind, &msg, "wire-submit")
                .await;
        }
        // Auth 4: own-or-`SessionControlAny`. An `Absent` (brand-new) session passes here, then
        // `note_activity` stamps the caller as owner — checked BEFORE `note_activity` so a foreign
        // caller never mutates last-activity / the FTS index.
        let auth = self.require_session_access(&session, true).await?;
        self.note_activity(&session, &command).await;
        self.live.submit(&auth, command).await
    }

    async fn submit_from(
        &self,
        session: SessionId,
        origin: Origin,
        command: AgentCommand,
    ) -> Result<(), ApiError> {
        // Stage-5 cutover (§8): a non-resident Core session routes durable (origin attribution is
        // delivery metadata the durable session already owns).
        if self.cutover_routes(&session).await {
            self.require_session_access(&session, true).await?;
            self.bind_profile_on_first_open(&session, None).await;
            // Live parity: an opening StartTurn seeds the Primary from the submitting origin.
            if matches!(command, AgentCommand::StartTurn { .. }) {
                self.attach_hub(&session)
                    .await
                    .seed_primary_target(origin.primary_target());
            }
            self.note_activity(&session, &command).await;
            return self.submit_attached(&session, command, Some(&origin)).await;
        }
        // F4 durable-resume: a parked-durable `StartTurn`/`Steer` folds into the durable transcript
        // (the origin is delivery attribution the durable session already owns) rather than opening
        // a fresh live incarnation.
        if let Some((kind, msg)) = self.durable_resume_input(&session, &command).await {
            self.require_session_access(&session, true).await?;
            return self
                .enqueue_durable_input(&session, kind, &msg, "wire-submit")
                .await;
        }
        let auth = self.require_session_access(&session, true).await?;
        self.note_activity(&session, &command).await;
        self.live.submit_from(&auth, origin, command).await
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
        // Create-if-absent durable row with the engine's initial snapshot — in `Idle`, NOT `Ready`
        // (session-unification §2/§3): a blank un-run session is not scanner work. The old `Ready`
        // write was the incident's root cause — the recovery scanner activated the blank row while
        // the GUI's first live turn ran the same session, and the failed blank activation's
        // terminal snapshot clobbered the conversation. A concurrent duplicate create is benign
        // (`AlreadyExists` = the row is there, which is all this path needs).
        if self.store.status(&session).await.is_none() {
            let blob = Snapshot::fresh(session.clone())
                .encode()
                .map_err(|e| ApiError::Other(format!("encode initial snapshot: {e}")))?;
            match self
                .store
                .create_idle(session.clone(), self.partition, blob)
                .await
            {
                Ok(()) | Err(daemon_store::StoreError::AlreadyExists(_)) => {}
                Err(e) => return Err(ApiError::Other(format!("create session: {e}"))),
            }
        }
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
        Ok(session)
    }

    async fn submit_as(&self, args: SubmitAsArgs) -> Result<(), ApiError> {
        let SubmitAsArgs {
            session,
            origin,
            command,
            profile,
        } = args;
        // Stage-5 cutover (§8): a non-resident Core session routes durable. An explicitly-passed
        // FOREIGN profile keeps the live actor rail (the probe can only inspect the durable
        // binding, which does not exist yet on a first open); a Core profile binds
        // sticky-on-first-open onto the durable meta (the same rule `session_create` uses), so
        // "open this chat as agent X" works without forcing a live residency.
        let explicit_foreign = match (&profile, &self.foreign_probe) {
            (Some(p), Some(probe)) => probe(p),
            _ => false,
        };
        if !explicit_foreign && self.cutover_routes(&session).await {
            self.require_session_access(&session, true).await?;
            self.bind_profile_on_first_open(&session, profile).await;
            // Live parity: an opening StartTurn seeds the Primary from the submitting origin
            // (the generic `api` origin when the surface passed none).
            if matches!(command, AgentCommand::StartTurn { .. }) {
                let target = origin
                    .as_ref()
                    .map(|o| o.primary_target())
                    .unwrap_or_else(|| internals::api_origin().primary_target());
                self.attach_hub(&session).await.seed_primary_target(target);
            }
            self.note_activity(&session, &command).await;
            return self
                .submit_attached(&session, command, origin.as_ref())
                .await;
        }
        // F4 durable-resume: a parked-durable `StartTurn`/`Steer` folds into the durable transcript
        // (its engine profile is already bound durably) rather than opening a fresh live incarnation.
        if let Some((kind, msg)) = self.durable_resume_input(&session, &command).await {
            self.require_session_access(&session, true).await?;
            return self
                .enqueue_durable_input(&session, kind, &msg, "wire-submit")
                .await;
        }
        let auth = self.require_session_access(&session, true).await?;
        // Bind the explicit profile sticky-on-first-open (the same `ensure` seam `submit_routed`
        // uses), so a GUI can "open this chat as agent X" before the first turn submits.
        if profile.is_some() {
            self.live.ensure(&session, profile).await?;
        }
        self.note_activity(&session, &command).await;
        match origin {
            Some(origin) => self.live.submit_from(&auth, origin, command).await,
            None => self.live.submit(&auth, command).await,
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
        // Stage-5 cutover (§8): a Core-resolved origin routes durable, exactly like a direct
        // `submit_as` — bind the RESOLVED profile sticky-on-first-open (routing owns agent
        // selection) and seed the resolved `Primary` (routing owns delivery). A Foreign-resolved
        // profile keeps the live actor rail (the probe inspects the resolution the registry
        // already made, since a first open has no durable binding yet to probe).
        let resolved_foreign = match (&resolved.profile, &self.foreign_probe) {
            (Some(p), Some(probe)) => probe(p),
            _ => false,
        };
        if !resolved_foreign && self.cutover_routes(&resolved.session).await {
            self.require_session_access(&resolved.session, true).await?;
            self.bind_profile_on_first_open(&resolved.session, resolved.profile.clone())
                .await;
            if matches!(
                command,
                AgentCommand::StartTurn { .. }
                    | AgentCommand::Steer { .. }
                    | AgentCommand::Observe { .. }
            ) {
                self.attach_hub(&resolved.session)
                    .await
                    .seed_primary_target(resolved.delivery.clone());
            }
            self.note_activity(&resolved.session, &command).await;
            self.submit_attached(&resolved.session, command, Some(&origin))
                .await?;
            return Ok(resolved.session);
        }
        // F4 durable-resume: if the origin resolves to a parked-durable session, a `StartTurn`/
        // `Steer` folds into the durable transcript + wakes it, rather than opening a fresh live
        // incarnation over the durable state.
        if let Some((kind, msg)) = self.durable_resume_input(&resolved.session, &command).await {
            self.require_session_access(&resolved.session, true).await?;
            self.enqueue_durable_input(&resolved.session, kind, &msg, "wire-submit")
                .await?;
            return Ok(resolved.session);
        }
        // Auth 4: own-or-`SessionControlAny` on the resolved session (new sessions pass and are
        // stamped by `note_activity`).
        let auth = self.require_session_access(&resolved.session, true).await?;
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
                .await?;
            self.live
                .seed_primary_target(&resolved.session, resolved.delivery.clone());
        }
        self.note_activity(&resolved.session, &command).await;
        self.live.submit_from(&auth, origin, command).await?;
        Ok(resolved.session)
    }

    async fn poll(&self, session: SessionId, max: u32) -> Result<Vec<Outbound>, ApiError> {
        // Auth 4: own-or-`SessionControlAny` (the task's named control ops include `poll`).
        let auth = self.require_session_access(&session, true).await?;
        // Stage-5 cutover (§8): a non-resident Core session's drain is its AttachmentHub's.
        if self.cutover_routes(&session).await {
            return Ok(self.attach_hub(&session).await.poll(max));
        }
        self.live.poll(&auth, max)
    }

    async fn respond(&self, session: SessionId, response: HostResponse) -> Result<(), ApiError> {
        let auth = self.require_session_access(&session, true).await?;
        // Stage-5 cutover (§8): a durable session's parked Input/Choice/Approval is answered on
        // its hub (the request parked there via the incarnation's HubParkingResolver).
        if self.cutover_routes(&session).await {
            return self.attach_hub(&session).await.respond(response);
        }
        self.live.respond(&auth, response)
    }

    async fn session_history(
        &self,
        session: SessionId,
        after_cursor: u64,
        before_cursor: Option<u64>,
        max: u32,
    ) -> JournalPageView {
        // Auth 4 (read-of-one): own-or-`SessionSeeAll`. The wire return is non-fallible, so an
        // unauthorized read yields an empty page (no transcript leak) rather than an error.
        if self.require_session_access(&session, false).await.is_err() {
            return JournalPageView::default();
        }
        let stream = JournalStreamId::session(&session);
        match before_cursor {
            // rung 2: the newest-anchored backward window (before_cursor wins over after_cursor).
            Some(before) => self.read_history_before(stream, before, max).await,
            None => self.read_history(stream, after_cursor, max).await,
        }
    }

    async fn log_after(
        &self,
        session: SessionId,
        after_seq: u64,
        max: u32,
    ) -> Result<LogPageView, ApiError> {
        // Auth 4: own-or-`SessionControlAny`. This is the one-shot / long-poll form of the live
        // `Subscribe` op (the wire `Subscribe` `Call` routes here); it must enforce the SAME
        // ownership check as the streaming `subscribe` below (both are `control = true`, so the
        // `Call` and `Open` forms of one op deny identically). Previously unguarded — the gap that
        // let a non-owner read another user's live transcript.
        let auth = self.require_session_access(&session, true).await?;
        // Stage-5 cutover (§8): a non-resident Core session's merged log is its AttachmentHub's.
        if self.cutover_routes(&session).await {
            return Ok(self.attach_hub(&session).await.log_after(after_seq, max));
        }
        Ok(self.live.log_after(&auth, after_seq, max))
    }

    async fn subscribe(&self, session: SessionId, after_seq: u64) -> Result<LogStream, ApiError> {
        // Auth 4: own-or-`SessionControlAny` (a live subscription is a session-interaction op).
        let auth = self.require_session_access(&session, true).await?;
        // Stage-5 cutover (§8): subscribe to the hub's merged log (backfill + live continuation).
        if self.cutover_routes(&session).await {
            return Ok(self.attach_hub(&session).await.subscribe(after_seq));
        }
        Ok(self.live.subscribe(&auth, after_seq))
    }

    async fn log_epoch(&self, session: SessionId) -> u64 {
        // Auth 4 (read-of-one, non-fallible): deny → 0. Not wire-reachable on its own (the mux pump
        // reads it before `subscribe`, which now enforces ownership under the same bound principal),
        // so this is defense-in-depth for any future caller.
        let Ok(auth) = self.require_session_access(&session, false).await else {
            return 0;
        };
        // Stage-5 cutover (§8): the hub's merged-log generation (L2 resync) for durable sessions.
        if self.cutover_routes(&session).await {
            return self.attach_hub(&session).await.log_epoch();
        }
        self.live.log_epoch(&auth)
    }

    async fn delivery_targets(&self, session: SessionId) -> Vec<DeliveryTarget> {
        // Auth 4 (read-of-one, non-fallible): a peer must not read another user's reply-routing —
        // deny → empty (no existence oracle). Previously unguarded.
        let Ok(auth) = self.require_session_access(&session, false).await else {
            return Vec::new();
        };
        // Stage-5 cutover (§8): a durable session's delivery roster is homed on its hub.
        if self.cutover_routes(&session).await {
            return match self.attachments.as_ref().and_then(|h| h.get(&session)) {
                Some(hub) => hub.delivery_targets(),
                None => Vec::new(),
            };
        }
        self.live.delivery_targets(&auth)
    }

    async fn delivery_sessions(
        &self,
        transport: TransportId,
        after: Option<String>,
    ) -> daemon_api::WirePage<SessionId> {
        // The live registry is a DashMap scan with no stable order; sort by session id (the
        // cursor key) before slicing. Stage-5 cutover (§8): the durable half of the roster lives
        // on the attachment hubs — union both (a session is only ever homed on one).
        let mut sessions = self.live.delivery_sessions(&transport);
        if let Some(hubs) = &self.attachments {
            for s in hubs.delivery_sessions(&transport) {
                if !sessions.contains(&s) {
                    sessions.push(s);
                }
            }
        }
        let sessions = sessions;
        // Auth 4 (F4): a non-owner must not enumerate another owner's sessions on a shared transport
        // (own sessions only unless SessionSeeAll) — per-row owner_visible, mirroring the roster /
        // checkpoints filter. The internal delivery bridge (daemon-http `serve_delivery_scoped`) runs
        // under a SessionSeeAll `system` scope, so it still discovers the transport's full owned set.
        let principal = current_principal();
        let mut visible = Vec::with_capacity(sessions.len());
        for s in sessions {
            let owner = self.store.session_meta(&s).await.and_then(|m| m.owner);
            if owner_visible(&principal, &owner) {
                visible.push(s);
            }
        }
        let mut sessions = visible;
        sessions.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        daemon_api::paginate(sessions, after.as_deref(), daemon_api::WIRE_PAGE_MAX, |s| {
            s.as_str().to_string()
        })
    }

    async fn handover(&self, session: SessionId, target: DeliveryTarget) -> Result<(), ApiError> {
        // Auth 4: own-or-`SessionControlAny`.
        let auth = self.require_session_access(&session, true).await?;
        // Stage-5 cutover (§8): a durable session's delivery roster is homed on its hub.
        if self.cutover_routes(&session).await {
            self.attach_hub(&session).await.handover(target);
            return Ok(());
        }
        self.live.handover(&auth, target)
    }

    async fn record_meta(&self, args: RecordMetaArgs) -> Result<(), ApiError> {
        // Auth 4: own-or-`SessionControlAny` (writes into the session's live log).
        let auth = self.require_session_access(&args.session, true).await?;
        // Stage-5 cutover (§8): a durable session's merged log is its hub's.
        if self.cutover_routes(&args.session).await {
            let session = args.session.clone();
            let RecordMetaArgs {
                origin, kind, body, ..
            } = args;
            self.attach_hub(&session)
                .await
                .record_meta(origin, kind, body);
            return Ok(());
        }
        self.live.record_meta(&auth, args)
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
        // Cluster E: widening a session's autonomy (`AcceptEdits`/`AutoAllow`) is operator-tier — a
        // non-operator owner may narrow (`Ask`/`Deny`) but not widen its own approval posture.
        if mode.widens_autonomy() {
            self.require_operator("widening the session approval mode")?;
        }
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
        // Cluster E: the security-widening subset of an overlay (autonomy-widening approval mode, or
        // `FullToolset`) is operator-tier; the rest (model/provider/workspace/`Allowlist`/`Ask`/
        // `Deny`) stays owner-allowed. A non-operator owner cannot widen its own approval posture or
        // tool surface through the unified overlay write.
        if overlay.widens_security_posture() {
            self.require_operator("widening the session approval mode or tool surface")?;
        }
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
