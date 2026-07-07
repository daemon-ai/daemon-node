// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Inbound: LINE webhook events → `Origin` + `Reception` → the `daemon-ingest` gate.
//!
//! LINE is **webhook-push**: the platform POSTs a batch of events to a public HTTP endpoint, signing
//! the raw request body with the channel secret (`x-line-signature`: base64(HMAC-SHA256(secret,
//! body))). This module owns that inbound seam:
//!
//! 1. [`verify_and_parse`] — the security boundary: verify the signature (constant-time, via the
//!    SDK's [`validate_signature`]) *before* trusting the body, then decode the [`CallbackRequest`]
//!    and project each text message event into a transport-agnostic [`InboundMessage`].
//! 2. [`to_reception`] — apply the route table (engagement + addressing classification) and build a
//!    [`Reception`] for the reusable [`Ingestor`].
//! 3. [`handle_webhook`] — glue the two, gate through the ingestor under `RequestContext::internal()`,
//!    and ensure the opened session has an outbound delivery subscription.
//!
//! ## HTTP wiring (webhook ingress)
//!
//! The listener is a small adapter-owned axum [`Router`] ([`webhook_router`]) with one route,
//! `POST {webhook_path}/{instance}`, where `{instance}` is the account handle (the segment after
//! `line/`). Keying by path segment lets one listener host N channels and verify each event against
//! the *right* channel secret. [`crate::serve`] binds + serves it itself when
//! [`LineConfig::webhook_bind`](crate::config::LineConfig::webhook_bind) is set.
//!
//! **Phase 2 must connect the public ingress**: expose `{webhook_path}/{handle}` on a public URL
//! (typically via a reverse proxy / tunnel), register that URL as the channel's webhook in the LINE
//! Developers console, and either set `webhook_bind` (adapter-owned) or mount [`webhook_router`] into
//! a shared node ingress. The channel secret is resolved from the account credential blob at bring-up.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use axum::Router;

use line_bot_sdk_rust::line_webhook::models::{CallbackRequest, Event, MessageContent, Source};
use line_bot_sdk_rust::parser::signature::validate_signature;

use daemon_host::{with_request_context, RequestContext};
use daemon_ingest::{Ingestor, Reception};
use daemon_protocol::{Origin, SenderId, TransportId, UserMsg};

use crate::config::{self, LineRoute};
use crate::mapping::scope_for;
use crate::outbound::DeliveryManager;

/// The HTTP header LINE signs the raw request body with.
pub const SIGNATURE_HEADER: &str = "x-line-signature";

/// Why a webhook request was rejected before it could be ingested.
#[derive(Debug, PartialEq, Eq)]
pub enum WebhookError {
    /// No `x-line-signature` header was present.
    MissingSignature,
    /// The signature did not verify against the channel secret (spoofed / tampered).
    BadSignature,
    /// The body was not valid UTF-8 / not a parseable LINE callback.
    BadBody,
}

impl WebhookError {
    /// The HTTP status a handler returns for this rejection.
    pub fn status(&self) -> StatusCode {
        match self {
            WebhookError::MissingSignature | WebhookError::BadSignature => StatusCode::UNAUTHORIZED,
            WebhookError::BadBody => StatusCode::BAD_REQUEST,
        }
    }
}

/// A normalised inbound LINE message, decoded from a verified webhook batch. Transport-agnostic (no
/// SDK types) so the routing/gating logic and its tests never touch `line-bot-sdk-rust`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InboundMessage {
    /// The conversation id to route + reply to (LINE user / group / room id).
    pub target: String,
    /// Whether the source is a 1:1 user conversation (drives the `addressed` classification).
    pub is_dm: bool,
    /// The immutable sender user id (may be empty for group/room events without a resolvable user).
    pub sender: String,
    /// The message text.
    pub text: String,
}

/// The security boundary: verify the `x-line-signature` over the raw `body` against `channel_secret`
/// (constant-time), then decode the callback and project each **text** message event into an
/// [`InboundMessage`]. Non-text events (media, follow, join, …) are skipped in v1. Returns an error
/// (never a partial batch) when the signature is missing/invalid or the body cannot be parsed — so a
/// caller never ingests an unverified event.
pub fn verify_and_parse(
    channel_secret: &str,
    signature: Option<&str>,
    body: &[u8],
) -> Result<Vec<InboundMessage>, WebhookError> {
    let signature = signature.ok_or(WebhookError::MissingSignature)?;
    let body_str = std::str::from_utf8(body).map_err(|_| WebhookError::BadBody)?;
    if !validate_signature(channel_secret, signature, body_str) {
        return Err(WebhookError::BadSignature);
    }
    let callback: CallbackRequest =
        serde_json::from_str(body_str).map_err(|_| WebhookError::BadBody)?;

    let mut out = Vec::new();
    for event in callback.events {
        let Event::MessageEvent(message_event) = event else {
            continue;
        };
        let text = match *message_event.message {
            MessageContent::TextMessageContent(t) => t.text,
            _ => continue,
        };
        let (target, is_dm, sender) = match message_event.source.as_deref() {
            Some(Source::UserSource(u)) => {
                let uid = u.user_id.clone().unwrap_or_default();
                (uid.clone(), true, uid)
            }
            Some(Source::GroupSource(g)) => (
                g.group_id.clone(),
                false,
                g.user_id.clone().unwrap_or_default(),
            ),
            Some(Source::RoomSource(r)) => (
                r.room_id.clone(),
                false,
                r.user_id.clone().unwrap_or_default(),
            ),
            _ => continue,
        };
        if target.is_empty() {
            continue;
        }
        out.push(InboundMessage {
            target,
            is_dm,
            sender,
            text,
        });
    }
    Ok(out)
}

/// Apply the route table to a decoded [`InboundMessage`] and build a [`Reception`], or `None` when a
/// configured route table does not engage this conversation. `bare` is the account handle (route
/// matcher key); `transport` is the instance-qualified id whose scope seeds the reply route.
pub fn to_reception(
    transport: &TransportId,
    routes: &[LineRoute],
    bare: &str,
    msg: &InboundMessage,
) -> Option<Reception> {
    let route = config::route_for(routes, bare, &msg.target, msg.is_dm)?;
    // LINE has no reliable bot-mention marker in group text, so mention-gating leans on DM + the
    // `!command` convention. A non-gating route treats every message as addressed.
    let addressed = if route.mention_gating {
        msg.is_dm || msg.text.trim_start().starts_with('!')
    } else {
        true
    };
    let sender = if msg.sender.is_empty() {
        msg.target.clone()
    } else {
        msg.sender.clone()
    };
    // Attribution (who spoke) rides inside the text, adapter-formatted (ingest treats input opaquely).
    let attributed = format!("{sender}: {}", msg.text);
    Some(Reception {
        origin: Origin::new(transport.clone(), scope_for(&msg.target)),
        sender: SenderId::new(sender),
        input: UserMsg::new(attributed),
        addressed,
    })
}

/// One account served by the webhook listener: the transport id to stamp receptions with, the bare
/// handle (route matcher key), and the channel secret its signatures are verified against.
#[derive(Clone)]
pub struct WebhookAccount {
    /// The instance-qualified transport id (`line/<handle>`).
    pub transport: TransportId,
    /// The bare account handle (route matcher key).
    pub bare: String,
    /// The channel secret the `x-line-signature` is verified against.
    pub channel_secret: String,
}

/// The shared, cloneable state the webhook route handler resolves everything from.
#[derive(Clone)]
pub struct WebhookState {
    /// The reusable inbound gate (shared with the outbound projector that drives its busy state).
    pub ingestor: Arc<Ingestor>,
    /// Ensures an outbound delivery subscription exists for any session an inbound message opens.
    pub delivery: Arc<DeliveryManager>,
    /// The engaged-conversation / addressing route table.
    pub routes: Arc<Vec<LineRoute>>,
    /// The served accounts, keyed by handle (the `{instance}` path segment).
    pub accounts: Arc<HashMap<String, WebhookAccount>>,
}

/// Handle one webhook POST for account `handle`: resolve the account, verify + parse the batch, and
/// gate each message through the ingestor (opening/steering a session and ensuring its outbound
/// delivery). Returns the HTTP status to reply. Unknown account → 404; bad/missing signature → 401;
/// unparseable body → 400; otherwise 200 (even if all events were skipped — LINE expects 200 so it
/// does not retry a well-formed delivery).
pub async fn handle_webhook(
    state: &WebhookState,
    handle: &str,
    signature: Option<&str>,
    body: &[u8],
) -> StatusCode {
    let Some(account) = state.accounts.get(handle) else {
        return StatusCode::NOT_FOUND;
    };
    let messages = match verify_and_parse(&account.channel_secret, signature, body) {
        Ok(messages) => messages,
        Err(e) => return e.status(),
    };
    for msg in messages {
        let Some(reception) = to_reception(&account.transport, &state.routes, &account.bare, &msg)
        else {
            continue;
        };
        // Bind the in-process `internal` principal: this handler runs in an axum task with no request
        // context, so it supplies the trusted embedded-caller identity explicitly (a fresh chat
        // session is then stamped `owner = "internal"`).
        match with_request_context(
            RequestContext::internal(),
            state.ingestor.receive(reception),
        )
        .await
        {
            Ok(session) => state.delivery.ensure(session, account.transport.clone()),
            Err(e) => tracing::warn!(error = %e, "line: ingest receive failed"),
        }
    }
    StatusCode::OK
}

/// The adapter-owned inbound axum router: `POST /{instance}` per account handle. Mount it under the
/// configured `webhook_path` (adapter-owned via [`serve_webhook`], or by an external ingress in
/// Phase 2). Signature verification happens inside [`handle_webhook`], per account.
pub fn webhook_router(state: WebhookState) -> Router {
    Router::new()
        .route("/{instance}", post(webhook_route))
        .with_state(state)
}

/// The axum handler: pull the raw body + signature header and dispatch to [`handle_webhook`]. The
/// raw [`Bytes`] body (not a parsed JSON extractor) is required because the signature is computed
/// over the exact bytes LINE sent.
async fn webhook_route(
    State(state): State<WebhookState>,
    Path(instance): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> StatusCode {
    let signature = headers.get(SIGNATURE_HEADER).and_then(|v| v.to_str().ok());
    handle_webhook(&state, &instance, signature, &body).await
}

/// Bind + serve the [`webhook_router`] on `listener` until it errors — the self-contained
/// adapter-owned ingress used when `webhook_bind` is configured.
pub async fn serve_webhook(
    listener: tokio::net::TcpListener,
    state: WebhookState,
) -> std::io::Result<()> {
    axum::serve(listener, webhook_router(state)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    use base64::Engine as _;
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    /// Compute a valid `x-line-signature` (base64(HMAC-SHA256(secret, body))) for a test fixture — the
    /// exact scheme the SDK's `validate_signature` checks, so no real secret or network is needed.
    fn sign(secret: &str, body: &str) -> String {
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("hmac key");
        mac.update(body.as_bytes());
        base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes())
    }

    const SECRET: &str = "fake-channel-secret-for-tests";

    fn user_text_body() -> String {
        // A representative LINE webhook batch: one text message from a 1:1 user.
        serde_json::json!({
            "destination": "Ubotdestination",
            "events": [{
                "type": "message",
                "mode": "active",
                "timestamp": 1_700_000_000_000i64,
                "webhookEventId": "01FAKEEVENTID",
                "deliveryContext": { "isRedelivery": false },
                "replyToken": "faketoken",
                "source": { "type": "user", "userId": "Ualice" },
                "message": {
                    "type": "text",
                    "id": "100001",
                    "text": "hello there",
                    "quoteToken": "qtoken"
                }
            }]
        })
        .to_string()
    }

    fn group_text_body() -> String {
        serde_json::json!({
            "destination": "Ubotdestination",
            "events": [{
                "type": "message",
                "mode": "active",
                "timestamp": 1_700_000_000_000i64,
                "webhookEventId": "01FAKEEVENTID2",
                "deliveryContext": { "isRedelivery": false },
                "replyToken": "faketoken2",
                "source": { "type": "group", "groupId": "Cteam", "userId": "Ubob" },
                "message": {
                    "type": "text",
                    "id": "100002",
                    "text": "!status please",
                    "quoteToken": "qtoken2"
                }
            }]
        })
        .to_string()
    }

    #[test]
    fn missing_signature_is_rejected() {
        let body = user_text_body();
        let err = verify_and_parse(SECRET, None, body.as_bytes()).expect_err("no signature");
        assert_eq!(err, WebhookError::MissingSignature);
        assert_eq!(err.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn bad_signature_is_rejected() {
        let body = user_text_body();
        let err = verify_and_parse(SECRET, Some("not-the-real-signature"), body.as_bytes())
            .expect_err("bad signature");
        assert_eq!(err, WebhookError::BadSignature);
    }

    #[test]
    fn wrong_secret_does_not_verify() {
        let body = user_text_body();
        // A signature computed with a *different* secret must not verify against SECRET.
        let sig = sign("some-other-secret", &body);
        let err = verify_and_parse(SECRET, Some(&sig), body.as_bytes())
            .expect_err("signature from wrong secret");
        assert_eq!(err, WebhookError::BadSignature);
    }

    #[test]
    fn valid_signature_parses_user_message() {
        let body = user_text_body();
        let sig = sign(SECRET, &body);
        let msgs =
            verify_and_parse(SECRET, Some(&sig), body.as_bytes()).expect("verified + parsed");
        assert_eq!(msgs.len(), 1);
        let m = &msgs[0];
        assert_eq!(m.target, "Ualice");
        assert!(m.is_dm);
        assert_eq!(m.sender, "Ualice");
        assert_eq!(m.text, "hello there");
    }

    #[test]
    fn valid_signature_parses_group_message() {
        let body = group_text_body();
        let sig = sign(SECRET, &body);
        let msgs =
            verify_and_parse(SECRET, Some(&sig), body.as_bytes()).expect("verified + parsed");
        assert_eq!(msgs.len(), 1);
        let m = &msgs[0];
        assert_eq!(m.target, "Cteam");
        assert!(!m.is_dm);
        assert_eq!(m.sender, "Ubob");
        assert_eq!(m.text, "!status please");
    }

    #[test]
    fn non_text_events_are_skipped() {
        let body = serde_json::json!({
            "destination": "Ubot",
            "events": [{
                "type": "message",
                "mode": "active",
                "timestamp": 1_700_000_000_000i64,
                "webhookEventId": "01STICKER",
                "deliveryContext": { "isRedelivery": false },
                "source": { "type": "user", "userId": "Ualice" },
                "message": {
                    "type": "sticker",
                    "id": "1",
                    "quoteToken": "q",
                    "stickerId": "52002734",
                    "packageId": "11537",
                    "stickerResourceType": "STATIC"
                }
            }]
        })
        .to_string();
        let sig = sign(SECRET, &body);
        let msgs = verify_and_parse(SECRET, Some(&sig), body.as_bytes()).expect("parsed");
        assert!(msgs.is_empty(), "sticker event yields no text message");
    }

    #[test]
    fn to_reception_dm_is_addressed_and_scoped() {
        let transport = TransportId::new("line/acme");
        let msg = InboundMessage {
            target: "Ualice".to_string(),
            is_dm: true,
            sender: "Ualice".to_string(),
            text: "hi".to_string(),
        };
        let reception = to_reception(&transport, &[], "acme", &msg).expect("engaged");
        assert!(reception.addressed, "a DM is always addressed");
        assert_eq!(reception.sender.as_str(), "Ualice");
        assert_eq!(
            reception.origin.scope,
            daemon_protocol::OriginScope::Dm {
                user: "Ualice".to_string()
            }
        );
        assert!(reception.input.text.contains("hi"));
    }

    #[test]
    fn to_reception_group_requires_command_under_gating() {
        let transport = TransportId::new("line/acme");
        let ambient = InboundMessage {
            target: "Cteam".to_string(),
            is_dm: false,
            sender: "Ubob".to_string(),
            text: "just chatting".to_string(),
        };
        let r = to_reception(&transport, &[], "acme", &ambient).expect("engaged");
        assert!(
            !r.addressed,
            "ambient group chatter is not addressed under gating"
        );

        let command = InboundMessage {
            text: "!help".to_string(),
            ..ambient
        };
        let r = to_reception(&transport, &[], "acme", &command).expect("engaged");
        assert!(r.addressed, "!command addresses the agent");
    }

    #[test]
    fn to_reception_none_when_route_table_excludes() {
        let transport = TransportId::new("line/acme");
        let routes = vec![LineRoute {
            target_glob: Some("C*".into()),
            ..Default::default()
        }];
        let dm = InboundMessage {
            target: "Ualice".to_string(),
            is_dm: true,
            sender: "Ualice".to_string(),
            text: "hi".to_string(),
        };
        assert!(
            to_reception(&transport, &routes, "acme", &dm).is_none(),
            "a group-only route table ignores DMs"
        );
    }
}
