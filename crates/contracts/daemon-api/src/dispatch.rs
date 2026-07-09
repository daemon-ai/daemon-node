// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The shared dispatch core every non-in-process transport calls (request -> interface -> response).
//!
//! `dispatch` fans out by surface: each `serve_<surface>` handles its own `ApiRequest` variants and
//! returns `None` for the rest, so `dispatch` is a thin chain. NOTE: a new `ApiRequest` variant MUST
//! be routed by exactly one `serve_*` helper below; the `daemon-conformance` suite routes every
//! variant through `dispatch`, so an unrouted variant fails there (and hits the final `unreachable!`).

use crate::*;

// ---------------------------------------------------------------------------
// Dispatch — the shared core every non-in-process transport calls
// ---------------------------------------------------------------------------

fn unit_or_err(r: Result<(), ApiError>) -> ApiResponse {
    match r {
        Ok(()) => ApiResponse::Ok,
        Err(e) => ApiResponse::Error(e),
    }
}

/// Map a fallible interface result onto the wire: apply `ok` to the value (usually the matching
/// `ApiResponse` variant constructor), or fold the error into `ApiResponse::Error`. The value-bearing
/// sibling of [`unit_or_err`]; collapses the repetitive `match { Ok(v) => Variant(v), Err(e) => Error }`
/// arms in the `serve_*` helpers.
fn ok_or_err<T>(r: Result<T, ApiError>, ok: impl FnOnce(T) -> ApiResponse) -> ApiResponse {
    match r {
        Ok(v) => ok(v),
        Err(e) => ApiResponse::Error(e),
    }
}

async fn serve_session(api: &dyn SessionApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::Submit {
            session,
            command,
            origin,
            profile,
        } => {
            if profile.is_some() {
                unit_or_err(
                    api.submit_as(SubmitAsArgs {
                        session,
                        origin,
                        command,
                        profile,
                    })
                    .await,
                )
            } else {
                match origin {
                    Some(origin) => unit_or_err(api.submit_from(session, origin, command).await),
                    None => unit_or_err(api.submit(session, command).await),
                }
            }
        }
        ApiRequest::SubmitRouted { origin, command } => {
            ok_or_err(api.submit_routed(origin, command).await, |session| {
                ApiResponse::Routed { session }
            })
        }
        ApiRequest::SessionCreate { session, profile } => {
            ok_or_err(api.session_create(session, profile).await, |session| {
                ApiResponse::SessionCreated { session }
            })
        }
        ApiRequest::Poll { session, max } => ok_or_err(
            // Clamp to the wire page bound (0 previously meant "everything"): the client codec
            // decodes into fixed WIRE_PAGE_MAX buffers. The drain leaves un-returned items queued,
            // so the next poll picks them up.
            api.poll(session, clamp_page_max(max)).await,
            ApiResponse::Drained,
        ),
        ApiRequest::Respond { session, response } => {
            unit_or_err(api.respond(session, response).await)
        }
        ApiRequest::SessionHistory {
            session,
            after_cursor,
            max,
        } => ApiResponse::Journal(api.session_history(session, after_cursor, max).await),
        ApiRequest::Subscribe {
            session,
            after_seq,
            max,
        } => ok_or_err(
            // Clamp to the wire page bound: max == 0 previously flowed to the merged log ring
            // uncapped (CursoredRing::page treats 0 as "no cap"), which can exceed the client
            // codec's fixed array buffers. The page is cursored, so the client loops.
            api.log_after(session, after_seq, clamp_page_max(max)).await,
            ApiResponse::LogPage,
        ),
        ApiRequest::DeliveryTargets { session } => {
            ApiResponse::DeliveryTargets(api.delivery_targets(session).await)
        }
        ApiRequest::DeliverySessions { transport, after } => {
            ApiResponse::DeliverySessions(api.delivery_sessions(transport, after).await)
        }
        ApiRequest::Handover { session, target } => {
            unit_or_err(api.handover(session, target).await)
        }
        ApiRequest::RecordMeta(args) => unit_or_err(api.record_meta(args).await),
        ApiRequest::SetSessionModel {
            session,
            model,
            provider,
        } => unit_or_err(api.set_session_model(session, model, provider).await),
        ApiRequest::SetSessionMode { session, mode } => {
            unit_or_err(api.set_session_mode(session, mode).await)
        }
        ApiRequest::SetSessionOverlay { session, overlay } => {
            unit_or_err(api.set_session_overlay(session, overlay).await)
        }
        _ => return None,
    })
}

/// Control plane: health/stats/telemetry, the session roster, approvals, checkpoints, durable
/// lifecycle (assign/cancel/rewind), and the node verifying key.
async fn serve_control(api: &dyn NodeApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::Health => ApiResponse::Health(api.health().await),
        ApiRequest::Stats => ApiResponse::Stats(api.stats().await),
        ApiRequest::Telemetry => ApiResponse::Telemetry(api.telemetry().await),
        ApiRequest::Sessions => ApiResponse::Sessions(api.sessions().await),
        ApiRequest::ApprovalsPending { session, after } => {
            ApiResponse::Approvals(api.approvals_pending(session, after).await)
        }
        ApiRequest::ApprovalDecide {
            session,
            request_id,
            allow,
            allow_permanent,
            reason,
        } => unit_or_err(
            api.approval_decide(session, request_id, allow, allow_permanent, reason)
                .await,
        ),
        ApiRequest::FingerprintList { session } => ok_or_err(
            api.fingerprint_list(session).await,
            ApiResponse::Fingerprints,
        ),
        ApiRequest::FingerprintRevoke {
            session,
            fingerprint,
        } => unit_or_err(api.fingerprint_revoke(session, fingerprint).await),
        ApiRequest::CheckpointList { session, after } => {
            ApiResponse::Checkpoints(api.checkpoints(session, after).await)
        }
        // L3 node-wide event feed: the one-shot/long-poll page (the push form rides `Open` ->
        // `events_subscribe` in the socket pump, not `dispatch`). Bounded at the wire page max
        // (previously 0 = up to the whole ring); the page is cursored, so the client loops.
        ApiRequest::EventsSince { cursor, .. } => {
            ApiResponse::EventsPage(api.events_page(cursor, clamp_page_max(0)).await)
        }
        ApiRequest::CheckpointRewind {
            session,
            checkpoint_id,
        } => unit_or_err(api.checkpoint_rewind(session, checkpoint_id).await),
        ApiRequest::Assign { session } => unit_or_err(api.assign(session).await),
        ApiRequest::Cancel { session } => unit_or_err(api.cancel(session).await),
        ApiRequest::VerifyingKey => ApiResponse::VerifyingKey(api.verifying_key().await),
        ApiRequest::SessionsQuery { query } => {
            ApiResponse::SessionPage(api.sessions_query(query).await)
        }
        ApiRequest::SessionGet { session } => {
            ApiResponse::SessionDetail(api.session_get(session).await)
        }
        ApiRequest::SessionSearch { query, limit } => {
            ApiResponse::SessionSearch(api.session_search(query, limit).await)
        }
        ApiRequest::SessionRecap { session } => {
            ApiResponse::SessionRecap(api.session_recap(session).await)
        }
        ApiRequest::SessionUpdateMeta { session, patch } => {
            unit_or_err(api.session_update_meta(session, patch).await)
        }
        ApiRequest::Rewind { session, point } => unit_or_err(api.rewind(session, point).await),
        // -- user feedback + node-owned telemetry consent (N1; wire v31) ------------------------
        ApiRequest::FeedbackSubmit {
            kind,
            target,
            rating,
            comment,
            include_content,
            diagnostics,
            surface,
        } => ok_or_err(
            api.feedback_submit(FeedbackSubmitArgs {
                kind,
                target,
                rating,
                comment,
                include_content,
                diagnostics,
                surface,
            })
            .await,
            ApiResponse::FeedbackAck,
        ),
        ApiRequest::TelemetryConsentGet => {
            ok_or_err(api.telemetry_consent_get().await, |enabled| {
                ApiResponse::TelemetryConsent { enabled }
            })
        }
        ApiRequest::TelemetryConsentSet { enabled } => {
            ok_or_err(api.telemetry_consent_set(enabled).await, |enabled| {
                ApiResponse::TelemetryConsent { enabled }
            })
        }
        // -- saved presences (W2-F; wire v37) -------------------------------------------------
        ApiRequest::PresenceList => ApiResponse::SavedPresences(api.presence_list().await),
        ApiRequest::PresenceSave { presence } => unit_or_err(api.presence_save(presence).await),
        ApiRequest::PresenceDelete { id } => unit_or_err(api.presence_delete(id).await),
        ApiRequest::PresenceSetActive { id } => unit_or_err(api.presence_set_active(id).await),
        // -- notifications (W2-G; wire v37) ---------------------------------------------------
        ApiRequest::NotificationList => ApiResponse::Notifications(api.notification_list().await),
        // -- persons / metacontacts (W3-J; wire v37) ------------------------------------------
        ApiRequest::PersonList => ApiResponse::Persons(api.person_list().await),
        _ => return None,
    })
}

/// Orchestration fleet/tree: unit projection, history, and lifecycle (pause/resume/scale).
async fn serve_fleet(api: &dyn NodeApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::Fleet => ApiResponse::Fleet(api.fleet().await),
        ApiRequest::Tree { after } => ApiResponse::Tree(api.tree(after).await),
        ApiRequest::Unit { unit } => ApiResponse::Unit(api.unit(unit).await),
        ApiRequest::UnitEvents { unit, max } => {
            // Clamp to the wire page bound (0 previously meant the store's whole snapshot).
            ApiResponse::UnitEvents(api.unit_events(unit, clamp_page_max(max)).await)
        }
        ApiRequest::UnitOutbound { unit, max } => {
            // Reuses the `Drained(Vec<Outbound>)` response — the same rich §17 drain shape as
            // `poll`, with the same wire-bound clamp (leftovers stay queued for the next drain).
            ApiResponse::Drained(api.unit_outbound(unit, clamp_page_max(max)).await)
        }
        ApiRequest::UnitHistory {
            unit,
            after_cursor,
            max,
        } => ApiResponse::Journal(api.unit_history(unit, after_cursor, max).await),
        ApiRequest::Pause { unit } => unit_or_err(api.pause(unit).await),
        ApiRequest::Resume { unit } => unit_or_err(api.resume(unit).await),
        ApiRequest::Scale { unit, n } => unit_or_err(api.scale(unit, n).await),
        _ => return None,
    })
}

/// Model management: search/files/download/quantize/recommend/inspect + the active model.
async fn serve_models(api: &dyn NodeApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::ModelSearch { query } => {
            ok_or_err(api.model_search(query).await, ApiResponse::ModelSearch)
        }
        ApiRequest::ModelFiles {
            repo,
            revision,
            engine,
            after,
        } => ok_or_err(
            api.model_files(repo, revision, engine, after).await,
            ApiResponse::ModelFiles,
        ),
        ApiRequest::ModelDownload { model } => ok_or_err(
            api.model_download(model).await,
            ApiResponse::ModelDownloadStarted,
        ),
        ApiRequest::ModelDownloads => ApiResponse::ModelDownloads(api.model_downloads().await),
        ApiRequest::ModelCancel { id } => unit_or_err(api.model_cancel(id).await),
        ApiRequest::ModelPause { id } => unit_or_err(api.model_pause(id).await),
        ApiRequest::ModelResume { id } => unit_or_err(api.model_resume(id).await),
        ApiRequest::ModelCatalog => ApiResponse::ModelCatalog(api.model_catalog().await),
        ApiRequest::ModelDelete { id } => unit_or_err(api.model_delete(id).await),
        ApiRequest::ModelActivate { id, profile } => {
            unit_or_err(api.model_activate(id, profile).await)
        }
        ApiRequest::ModelRecommend(args) => {
            ok_or_err(api.model_recommend(args).await, ApiResponse::ModelRecommend)
        }
        ApiRequest::ModelQuantize(args) => ok_or_err(
            api.model_quantize(args).await,
            ApiResponse::ModelQuantizeStarted,
        ),
        ApiRequest::ModelQuantizes => ApiResponse::ModelQuantizes(api.model_quantizes().await),
        ApiRequest::ModelInspect { id } => {
            ok_or_err(api.model_inspect(id).await, ApiResponse::ModelInspect)
        }
        ApiRequest::Models { after } => ApiResponse::Models(api.models(after).await),
        ApiRequest::ModelCurrent { profile } => {
            ok_or_err(api.model_current(profile).await, ApiResponse::ModelCurrent)
        }
        ApiRequest::ProviderCatalog => ApiResponse::ProviderCatalog(api.provider_catalog().await),
        ApiRequest::ProviderModels {
            provider,
            credential_ref,
            transient_key,
            after,
        } => ApiResponse::ProviderModels(
            api.provider_models(provider, credential_ref, transient_key, after)
                .await,
        ),
        ApiRequest::CustomProviderList => {
            ApiResponse::CustomProviders(api.custom_provider_list().await)
        }
        ApiRequest::CustomProviderSet { provider } => {
            unit_or_err(api.custom_provider_set(provider).await)
        }
        ApiRequest::CustomProviderRemove { id } => {
            unit_or_err(api.custom_provider_remove(id).await)
        }
        _ => return None,
    })
}

/// Profiles + skills (versioned): CRUD, history/at/revert, distribution import/export.
async fn serve_profile(api: &dyn NodeApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::ProfileList => ApiResponse::Profiles(api.profile_list().await),
        ApiRequest::ProfileGet { id } => ok_or_err(api.profile_get(id).await, ApiResponse::Profile),
        ApiRequest::ProfileCreate { spec } => unit_or_err(api.profile_create(spec).await),
        ApiRequest::ProfileUpdate { spec } => unit_or_err(api.profile_update(spec).await),
        ApiRequest::ProfileDelete { id } => unit_or_err(api.profile_delete(id).await),
        ApiRequest::ProfileSelect { id } => unit_or_err(api.profile_select(id).await),
        ApiRequest::ProfileClone { source, new_id } => {
            unit_or_err(api.profile_clone(source, new_id).await)
        }
        ApiRequest::ProfileExport { id } => {
            ok_or_err(api.profile_export(id).await, ApiResponse::Distribution)
        }
        ApiRequest::ProfileImport { dist, new_id } => ok_or_err(
            api.profile_import(dist, new_id).await,
            ApiResponse::ProfileId,
        ),
        ApiRequest::ProfileHistory { id, after } => {
            ok_or_err(api.profile_history(id, after).await, ApiResponse::Revisions)
        }
        ApiRequest::ProfileAt { id, seq } => ok_or_err(api.profile_at(id, seq).await, |spec| {
            ApiResponse::Profile(Some(spec))
        }),
        ApiRequest::ProfileRevert { id, seq } => unit_or_err(api.profile_revert(id, seq).await),
        ApiRequest::SoulGet { id } => ok_or_err(api.soul_get(id).await, ApiResponse::SoulText),
        ApiRequest::SoulSet { id, text } => unit_or_err(api.soul_set(id, text).await),
        ApiRequest::SkillHistory { name, after } => {
            ok_or_err(api.skill_history(name, after).await, ApiResponse::Revisions)
        }
        ApiRequest::SkillAt { name, seq } => {
            ok_or_err(api.skill_at(name, seq).await, ApiResponse::SkillBundle)
        }
        ApiRequest::SkillRevert { name, seq } => unit_or_err(api.skill_revert(name, seq).await),
        ApiRequest::SkillGet { name } => {
            ok_or_err(api.skill_get(name).await, ApiResponse::SkillBundle)
        }
        ApiRequest::SkillPut { bundle } => unit_or_err(api.skill_put(bundle).await),
        _ => return None,
    })
}

/// Per-profile skill curator: list/pin/unpin/archive/restore/run.
async fn serve_curator(api: &dyn NodeApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::CuratorList { profile } => {
            ok_or_err(api.curator_list(profile).await, ApiResponse::CuratorSkills)
        }
        ApiRequest::CuratorPin { profile, name } => {
            unit_or_err(api.curator_pin(profile, name).await)
        }
        ApiRequest::CuratorUnpin { profile, name } => {
            unit_or_err(api.curator_unpin(profile, name).await)
        }
        ApiRequest::CuratorArchive { profile, name } => {
            unit_or_err(api.curator_archive(profile, name).await)
        }
        ApiRequest::CuratorRestore { profile, name } => {
            unit_or_err(api.curator_restore(profile, name).await)
        }
        ApiRequest::CuratorRun { profile } => {
            ok_or_err(api.curator_run(profile).await, ApiResponse::CuratorRun)
        }
        _ => return None,
    })
}

/// Interactive auth flows + credential store.
async fn serve_auth(api: &dyn NodeApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::AuthBegin(req) => ok_or_err(api.auth_begin(req).await, ApiResponse::AuthBegun),
        ApiRequest::AuthStep(req) => ok_or_err(api.auth_step(req).await, ApiResponse::AuthStepped),
        ApiRequest::AuthComplete(req) => {
            ok_or_err(api.auth_complete(req).await, ApiResponse::AuthCompleted)
        }
        ApiRequest::AuthCancel { flow_id } => unit_or_err(api.auth_cancel(flow_id).await),
        ApiRequest::AuthProviders => ApiResponse::AuthProviders(api.auth_providers().await),
        ApiRequest::CredentialSet { profile, secret } => {
            unit_or_err(api.credential_set(profile, secret).await)
        }
        ApiRequest::CredentialList => ApiResponse::Credentials(api.credential_list().await),
        ApiRequest::CredentialRemove { profile } => {
            unit_or_err(api.credential_remove(profile).await)
        }
        ApiRequest::CredentialSetLabel { profile, label } => {
            unit_or_err(api.credential_set_label(profile, label).await)
        }
        _ => return None,
    })
}

/// Scheduled jobs (cron) + suggestions.
async fn serve_cron(api: &dyn NodeApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::CronList => ApiResponse::CronJobs(api.cron_list().await),
        ApiRequest::CronCreate { spec } => {
            ok_or_err(api.cron_create(spec).await, ApiResponse::CronId)
        }
        ApiRequest::CronUpdate { id, spec } => unit_or_err(api.cron_update(id, spec).await),
        ApiRequest::CronDelete { id } => unit_or_err(api.cron_delete(id).await),
        ApiRequest::CronTrigger { id } => unit_or_err(api.cron_trigger(id).await),
        ApiRequest::CronRuns { id } => ApiResponse::CronRuns(api.cron_runs(id).await),
        ApiRequest::CronPause { id, paused } => unit_or_err(api.cron_pause(id, paused).await),
        ApiRequest::CronSuggestions => ApiResponse::CronSuggestions(api.cron_suggestions().await),
        ApiRequest::CronAcceptSuggestion { id } => {
            ok_or_err(api.cron_accept_suggestion(id).await, ApiResponse::CronId)
        }
        ApiRequest::CronDismissSuggestion { id } => {
            unit_or_err(api.cron_dismiss_suggestion(id).await)
        }
        _ => return None,
    })
}

/// Chat routing + transport adapter/instance registry.
async fn serve_routing(api: &dyn NodeApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::RoutingListChats { after } => {
            ApiResponse::ChatRoutes(api.routing_list_chats(after).await)
        }
        ApiRequest::RoutingGet { origin } => ApiResponse::ChatRoute(api.routing_get(origin).await),
        ApiRequest::RoutingSet { route } => unit_or_err(api.routing_set(route).await),
        ApiRequest::RoutingBindChat {
            origin,
            session,
            profile,
        } => unit_or_err(api.routing_bind_chat(origin, session, profile).await),
        ApiRequest::RoutingUnbindChat { origin } => {
            unit_or_err(api.routing_unbind_chat(origin).await)
        }
        ApiRequest::TransportRooms { transport, after } => {
            ApiResponse::Rooms(api.transport_rooms(transport, after).await)
        }
        ApiRequest::TransportAdapters => ApiResponse::Adapters(api.transport_adapters().await),
        ApiRequest::TransportInstances => {
            ApiResponse::TransportInstances(api.transport_instances().await)
        }
        ApiRequest::TransportDisconnect { transport } => {
            unit_or_err(api.transport_disconnect(transport).await)
        }
        ApiRequest::TransportRemove { transport } => {
            unit_or_err(api.transport_remove(transport).await)
        }
        ApiRequest::TransportConnect { transport } => {
            unit_or_err(api.transport_connect(transport).await)
        }
        ApiRequest::TransportSetEnabled { transport, enabled } => {
            unit_or_err(api.transport_set_enabled(transport, enabled).await)
        }
        ApiRequest::TransportSetLabel { transport, label } => {
            unit_or_err(api.transport_set_label(transport, label).await)
        }
        _ => return None,
    })
}

/// Messaging surface: conversations, membership, contacts, directory.
async fn serve_messaging(api: &dyn NodeApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::ConvList { transport, after } => {
            ApiResponse::Conversations(api.conv_list(transport, after).await)
        }
        ApiRequest::ConvGet { transport, conv } => {
            ApiResponse::Conversation(api.conv_get(transport, conv).await)
        }
        ApiRequest::ConvCreateDetails { transport } => {
            ApiResponse::ConvCreateDetails(api.conv_create_details(transport).await)
        }
        ApiRequest::ConvCreate { transport, details } => {
            ok_or_err(api.conv_create(transport, details).await, |info| {
                ApiResponse::Conversation(Some(info))
            })
        }
        ApiRequest::ConvJoinDetails { transport } => {
            ApiResponse::ConvJoinDetails(api.conv_join_details(transport).await)
        }
        ApiRequest::ConvJoin { transport, details } => {
            ok_or_err(api.conv_join(transport, details).await, |info| {
                ApiResponse::Conversation(Some(info))
            })
        }
        ApiRequest::ConvLeave { transport, conv } => {
            unit_or_err(api.conv_leave(transport, conv).await)
        }
        ApiRequest::ConvSend(args) => unit_or_err(api.conv_send(args).await),
        ApiRequest::ConvSetTopic {
            transport,
            conv,
            topic,
        } => unit_or_err(api.conv_set_topic(transport, conv, topic).await),
        ApiRequest::ConvSetTitle {
            transport,
            conv,
            title,
        } => unit_or_err(api.conv_set_title(transport, conv, title).await),
        ApiRequest::ConvSetDescription {
            transport,
            conv,
            description,
        } => unit_or_err(api.conv_set_description(transport, conv, description).await),
        ApiRequest::ConvDelete { transport, conv } => {
            unit_or_err(api.conv_delete(transport, conv).await)
        }
        ApiRequest::ConvHistory(args) => ApiResponse::Journal(api.conv_history(args).await),
        ApiRequest::MemberInvite(args) => unit_or_err(api.member_invite(args).await),
        ApiRequest::MemberRemove(args) => unit_or_err(api.member_remove(args).await),
        ApiRequest::MemberBan(args) => unit_or_err(api.member_ban(args).await),
        ApiRequest::MemberSetRole(args) => unit_or_err(api.member_set_role(args).await),
        ApiRequest::ContactGetProfile { transport, contact } => ok_or_err(
            api.contact_get_profile(transport, contact).await,
            ApiResponse::ContactProfile,
        ),
        ApiRequest::ContactSetAlias {
            transport,
            contact,
            alias,
        } => unit_or_err(api.contact_set_alias(transport, contact, alias).await),
        ApiRequest::ContactActionMenu { transport, contact } => {
            ApiResponse::ActionMenu(api.contact_action_menu(transport, contact).await)
        }
        ApiRequest::DirectorySearch { transport, query } => ok_or_err(
            api.directory_search(transport, query).await,
            ApiResponse::Contacts,
        ),
        ApiRequest::RosterList { transport, after } => {
            ApiResponse::ContactPage(api.roster_list(transport, after).await)
        }
        ApiRequest::RosterAdd { transport, contact } => {
            unit_or_err(api.roster_add(transport, contact).await)
        }
        ApiRequest::RosterUpdate { transport, contact } => {
            unit_or_err(api.roster_update(transport, contact).await)
        }
        ApiRequest::RosterRemove { transport, contact } => {
            unit_or_err(api.roster_remove(transport, contact).await)
        }
        ApiRequest::FtSend {
            transport,
            transfer,
        } => unit_or_err(api.ft_send(transport, transfer).await),
        ApiRequest::FtReceive {
            transport,
            transfer,
        } => unit_or_err(api.ft_receive(transport, transfer).await),
        _ => return None,
    })
}

/// Extension registry + node config: foreign agents, providers, tools, commands, config.
async fn serve_registry(api: &dyn NodeApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::AgentDiscover => ApiResponse::AgentCatalog(api.agent_discover().await),
        ApiRequest::AgentCatalog => ApiResponse::AgentCatalog(api.agent_catalog().await),
        ApiRequest::AgentRegister { entry } => unit_or_err(api.agent_register(entry).await),
        ApiRequest::AgentRemove { name } => unit_or_err(api.agent_remove(name).await),
        ApiRequest::ProviderList => ApiResponse::Providers(api.provider_list().await),
        ApiRequest::ProviderRegister { provider } => {
            unit_or_err(api.provider_register(provider).await)
        }
        ApiRequest::ToolList => ApiResponse::Tools(api.tool_list().await),
        ApiRequest::ToolRegister { tool } => unit_or_err(api.tool_register(tool).await),
        ApiRequest::ToolSetEnabled { tool, enabled } => {
            unit_or_err(api.tool_set_enabled(tool, enabled).await)
        }
        ApiRequest::CommandList => ApiResponse::Commands(api.command_list().await),
        ApiRequest::CommandInvoke { invocation } => ok_or_err(
            api.command_invoke(invocation).await,
            ApiResponse::CommandOutput,
        ),
        ApiRequest::Caps => ApiResponse::Caps(api.caps().await),
        ApiRequest::ConfigGet => ok_or_err(api.config_get().await, ApiResponse::Config),
        ApiRequest::ConfigSet { config } => unit_or_err(api.config_set(config).await),
        ApiRequest::GatewayGet => ok_or_err(api.gateway_get().await, ApiResponse::GatewayStatus),
        ApiRequest::GatewaySet { enabled, addr } => ok_or_err(
            api.gateway_set(enabled, addr).await,
            ApiResponse::GatewayStatus,
        ),
        _ => return None,
    })
}

/// Filesystem surface + content-addressed blob store.
async fn serve_fs(api: &dyn NodeApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::FsRoots => ApiResponse::FsRoots(api.fs_roots().await),
        ApiRequest::FsList {
            root,
            dir,
            show_ignored,
            after,
        } => ok_or_err(
            api.fs_list(root, dir, show_ignored, after).await,
            ApiResponse::FsList,
        ),
        ApiRequest::FsStat { root, path } => {
            ok_or_err(api.fs_stat(root, path).await, ApiResponse::FsStat)
        }
        ApiRequest::FsRead {
            root,
            path,
            max_bytes,
        } => ok_or_err(
            api.fs_read(root, path, max_bytes).await,
            ApiResponse::FsRead,
        ),
        ApiRequest::FsWrite(args) => ok_or_err(api.fs_write(args).await, ApiResponse::FsWrite),
        ApiRequest::FsSearch { root, query } => {
            ok_or_err(api.fs_search(root, query).await, ApiResponse::FsSearch)
        }
        ApiRequest::FsWatchPoll(args) => {
            ok_or_err(api.fs_watch_after(args).await, ApiResponse::FsWatch)
        }
        ApiRequest::BlobPut { bytes } => ok_or_err(api.blob_put(bytes).await, ApiResponse::BlobPut),
        ApiRequest::BlobGet { hash, range } => {
            ok_or_err(api.blob_get(hash, range).await, ApiResponse::BlobGet)
        }
        ApiRequest::BlobStat { hash } => ApiResponse::BlobStat(api.blob_stat(hash).await),
        ApiRequest::FsWriteFromBlob(args) => {
            ok_or_err(api.fs_write_from_blob(args).await, ApiResponse::FsWrite)
        }
        _ => return None,
    })
}

/// Admin access control (Auth 5): user/role/session administration + reserved per-resource grants.
async fn serve_access(api: &dyn NodeApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::UserCreate {
            username,
            password,
            roles,
        } => ok_or_err(
            api.user_create(username, password, roles).await,
            ApiResponse::AccessUser,
        ),
        ApiRequest::UserList => ok_or_err(api.user_list().await, ApiResponse::AccessUsers),
        ApiRequest::UserDisable { user_id, disabled } => {
            unit_or_err(api.user_disable(user_id, disabled).await)
        }
        ApiRequest::UserSetRoles { user_id, roles } => {
            unit_or_err(api.user_set_roles(user_id, roles).await)
        }
        ApiRequest::UserSetPassword { user_id, password } => {
            unit_or_err(api.user_set_password(user_id, password).await)
        }
        ApiRequest::RoleList => ok_or_err(api.role_list().await, ApiResponse::AccessRoles),
        ApiRequest::WhoAmI => ok_or_err(api.who_am_i().await, ApiResponse::WhoAmI),
        ApiRequest::SessionRevoke { user_id } => unit_or_err(api.session_revoke(user_id).await),
        ApiRequest::ResourceGrantCreate {
            user_id,
            resource_kind,
            resource_id,
            capability,
        } => unit_or_err(
            api.resource_grant_create(user_id, resource_kind, resource_id, capability)
                .await,
        ),
        ApiRequest::ResourceGrantList { user_id } => {
            unit_or_err(api.resource_grant_list(user_id).await)
        }
        ApiRequest::ResourceGrantRevoke { id } => unit_or_err(api.resource_grant_revoke(id).await),
        _ => return None,
    })
}

/// Dispatch a request against a full [`NodeApi`] — the entry point the socket/TCP/JSON-RPC node
/// transports call. Fans out to the per-surface `serve_*` helpers; every `ApiRequest` variant is
/// routed by exactly one of them (verified by the `daemon-conformance` suite, which exercises the
/// whole surface through `dispatch`).
pub async fn dispatch(api: &dyn NodeApi, req: ApiRequest) -> ApiResponse {
    if let Some(resp) = serve_session(api, req.clone()).await {
        return resp;
    }
    if let Some(resp) = serve_control(api, req.clone()).await {
        return resp;
    }
    if let Some(resp) = serve_fleet(api, req.clone()).await {
        return resp;
    }
    if let Some(resp) = serve_models(api, req.clone()).await {
        return resp;
    }
    if let Some(resp) = serve_profile(api, req.clone()).await {
        return resp;
    }
    if let Some(resp) = serve_curator(api, req.clone()).await {
        return resp;
    }
    if let Some(resp) = serve_auth(api, req.clone()).await {
        return resp;
    }
    if let Some(resp) = serve_cron(api, req.clone()).await {
        return resp;
    }
    if let Some(resp) = serve_routing(api, req.clone()).await {
        return resp;
    }
    if let Some(resp) = serve_messaging(api, req.clone()).await {
        return resp;
    }
    if let Some(resp) = serve_registry(api, req.clone()).await {
        return resp;
    }
    if let Some(resp) = serve_fs(api, req.clone()).await {
        return resp;
    }
    if let Some(resp) = serve_access(api, req).await {
        return resp;
    }
    unreachable!("every ApiRequest variant is routed by exactly one serve_* helper")
}

/// Dispatch against a **session-only** surface — the entry point the `daemon-core-ffi` transport
/// calls. Control-surface requests resolve to [`ApiError::Unsupported`] (this transport is the §17
/// brain seam, not the node control plane).
pub async fn dispatch_session(api: &dyn SessionApi, req: ApiRequest) -> ApiResponse {
    match serve_session(api, req).await {
        Some(resp) => resp,
        None => ApiResponse::Error(ApiError::Unsupported(
            "control surface is not available on this transport".into(),
        )),
    }
}
