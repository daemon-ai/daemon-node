// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The WhatsApp Web user backend (`whatsapp-rust`).
//!
//! A `mode = "user"` account is a linked "companion" device. The whatsapp-rust `Client` is built over
//! an in-memory storage backend seeded from the persisted `Device` snapshot (the credential store is
//! the session-of-record; there is no on-disk SQLite store — see the crate `Cargo.toml`). Outbound
//! send + group participant add/remove are wired; inbound messages are forwarded off the client's
//! event bus into an mpsc channel that [`crate::serve`] drains into the ingest gate.
//!
//! Pairing ([`Pairing`]) drives the QR-linking flow for the interactive-auth factory: it starts a
//! client, captures the QR payload from the event bus, and — once the phone links — hands back the
//! `Device` snapshot to persist as the session blob.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;

use daemon_api::{ApiError, MembershipOps};

use whatsapp_rust::bot::{Bot, BotBuilder, BotHandle, Provided};
use whatsapp_rust::store::traits::Backend;
use whatsapp_rust::transport::{TokioWebSocketTransportFactory, UreqHttpClient};
use whatsapp_rust::waproto::whatsapp as wa;
use whatsapp_rust::{Client, Jid, TokioRuntime};

use wacore::store::traits::DeviceStore;
use wacore::store::{Device, InMemoryBackend};
use wacore::types::events::Event;

use crate::backend::{WaBackend, WaInbound};

/// How long to wait for the first pairing QR to be produced before giving up (bounded polling).
const QR_WAIT_ATTEMPTS: u32 = 40;
/// Poll cadence while waiting for the first QR.
const QR_WAIT_STEP: Duration = Duration::from_millis(500);

/// Build a fully-provided `whatsapp-rust` bot builder (in-memory backend + Tokio transport / ureq
/// HTTP / Tokio runtime), seeding the restored `device` when reconnecting a paired account.
async fn base_builder(
    device: Option<Device>,
) -> Result<BotBuilder<Provided, Provided, Provided, Provided>, ApiError> {
    let backend = InMemoryBackend::new();
    if let Some(d) = device {
        backend
            .save(&d)
            .await
            .map_err(|e| ApiError::Other(format!("whatsapp: restoring device: {e}")))?;
    }
    let backend: Arc<dyn Backend> = Arc::new(backend);
    Ok(Bot::builder()
        .with_backend(backend)
        .with_transport_factory(TokioWebSocketTransportFactory::new())
        .with_http_client(UreqHttpClient::new())
        .with_runtime(TokioRuntime))
}

/// Resolve a membership target string to a WhatsApp `Jid`: a full JID is parsed as-is; a bare number
/// becomes a phone JID.
fn who_to_jid(who: &str) -> Result<Jid, ApiError> {
    if who.contains('@') {
        who.parse::<Jid>()
            .map_err(|e| ApiError::Other(format!("invalid whatsapp jid {who}: {e}")))
    } else {
        Ok(Jid::pn(who))
    }
}

/// A live WhatsApp Web account: the running whatsapp-rust client used for send + group ops.
pub struct UserBackend {
    client: Arc<Client>,
}

impl UserBackend {
    /// Connect a paired account from its persisted `device` snapshot, forwarding inbound messages to
    /// `inbound_tx`. Returns the backend + the run-loop handle (kept alive by the caller).
    pub async fn connect(
        device: serde_json::Value,
        inbound_tx: tokio::sync::mpsc::Sender<WaInbound>,
    ) -> Result<(Arc<UserBackend>, BotHandle), ApiError> {
        let device: Device = serde_json::from_value(device)
            .map_err(|e| ApiError::Other(format!("whatsapp: device blob: {e}")))?;
        let mut bot = base_builder(Some(device))
            .await?
            .on_event(move |event, _client| {
                let tx = inbound_tx.clone();
                async move {
                    forward_inbound(&event, &tx).await;
                }
            })
            .build()
            .await
            .map_err(|e| ApiError::Other(format!("whatsapp: bot build: {e}")))?;
        let client = bot.client();
        let handle = bot
            .run()
            .await
            .map_err(|e| ApiError::Other(format!("whatsapp: bot run: {e}")))?;
        Ok((Arc::new(UserBackend { client }), handle))
    }
}

/// Forward a `Message` event to the ingest channel (skipping our own posts and empty/non-text bodies).
async fn forward_inbound(event: &Event, tx: &tokio::sync::mpsc::Sender<WaInbound>) {
    if let Event::Message(msg, info) = event {
        if info.source.is_from_me {
            return;
        }
        let Some(text) = msg.conversation.clone() else {
            return;
        };
        if text.trim().is_empty() {
            return;
        }
        let _ = tx
            .send(WaInbound {
                chat: info.source.chat.to_string(),
                sender: info.source.sender.to_string(),
                text,
                is_group: info.source.is_group,
            })
            .await;
    }
}

#[async_trait]
impl WaBackend for UserBackend {
    async fn send_text(&self, to: &str, text: &str) -> Result<(), ApiError> {
        let jid = to
            .parse::<Jid>()
            .map_err(|e| ApiError::Other(format!("invalid whatsapp jid {to}: {e}")))?;
        let message = wa::Message {
            conversation: Some(text.to_string()),
            ..Default::default()
        };
        self.client
            .send_message(jid, message)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("whatsapp send: {e}")))
    }

    fn membership(&self) -> MembershipOps {
        // WhatsApp Web exposes group participant add/remove; there is no ban/role analogue we map.
        MembershipOps {
            invite: true,
            remove: true,
            ..MembershipOps::default()
        }
    }

    async fn invite(&self, conv: &str, who: &str) -> Result<(), ApiError> {
        let group = conv
            .parse::<Jid>()
            .map_err(|e| ApiError::Other(format!("invalid whatsapp group jid {conv}: {e}")))?;
        let user = who_to_jid(who)?;
        self.client
            .groups()
            .add_participants(&group, &[user])
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("whatsapp add_participants: {e}")))
    }

    async fn remove(&self, conv: &str, who: &str) -> Result<(), ApiError> {
        let group = conv
            .parse::<Jid>()
            .map_err(|e| ApiError::Other(format!("invalid whatsapp group jid {conv}: {e}")))?;
        let user = who_to_jid(who)?;
        self.client
            .groups()
            .remove_participants(&group, &[user])
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("whatsapp remove_participants: {e}")))
    }
}

/// The mutable state a running pairing client publishes: the latest QR payload and whether the phone
/// has linked the device.
#[derive(Default)]
struct QrState {
    qr: Option<String>,
    linked: bool,
}

/// A live QR-pairing session held across the interactive-auth poll steps: the running client, its
/// run-loop handle (kept alive), and the shared QR/link state its event bus feeds.
pub struct Pairing {
    client: Arc<Client>,
    // Held only to keep the run loop from being aborted; behind a Mutex so `Pairing` stays `Sync`.
    _handle: Mutex<BotHandle>,
    state: Arc<Mutex<QrState>>,
}

impl Pairing {
    /// Start a fresh pairing client and wait (bounded) for the first QR payload (or an immediate
    /// link, if a session was somehow already present).
    pub async fn start() -> Result<Pairing, ApiError> {
        let state = Arc::new(Mutex::new(QrState::default()));
        let ev_state = state.clone();
        let mut bot = base_builder(None)
            .await?
            .on_event(move |event, _client| {
                let state = ev_state.clone();
                async move {
                    match &*event {
                        Event::PairingQrCode { code, .. } => {
                            state.lock().unwrap().qr = Some(code.clone());
                        }
                        Event::PairSuccess(_) | Event::Connected(_) => {
                            state.lock().unwrap().linked = true;
                        }
                        _ => {}
                    }
                }
            })
            .build()
            .await
            .map_err(|e| ApiError::Other(format!("whatsapp: pairing bot build: {e}")))?;
        let client = bot.client();
        let handle = bot
            .run()
            .await
            .map_err(|e| ApiError::Other(format!("whatsapp: pairing bot run: {e}")))?;

        for _ in 0..QR_WAIT_ATTEMPTS {
            if client.is_logged_in() {
                state.lock().unwrap().linked = true;
                break;
            }
            if state.lock().unwrap().qr.is_some() {
                break;
            }
            tokio::time::sleep(QR_WAIT_STEP).await;
        }

        let ready = {
            let guard = state.lock().unwrap();
            guard.qr.is_some() || guard.linked
        };
        if !ready {
            return Err(ApiError::Other(
                "whatsapp: no pairing QR was produced".into(),
            ));
        }

        Ok(Pairing {
            client,
            _handle: Mutex::new(handle),
            state,
        })
    }

    /// The latest QR payload the peer device should scan.
    pub fn current_qr(&self) -> Option<String> {
        self.state.lock().unwrap().qr.clone()
    }

    /// Whether the phone has linked this device.
    pub fn is_linked(&self) -> bool {
        self.state.lock().unwrap().linked || self.client.is_logged_in()
    }

    /// The linked account's `(jid, serialized-device-snapshot)` to persist as the session blob. The
    /// runtime wrapper's `to_serializable()` yields the plain `wacore` `Device` we round-trip on
    /// restore (see [`UserBackend::connect`]).
    pub async fn device_blob(&self) -> Result<(String, serde_json::Value), ApiError> {
        let core: Device = self
            .client
            .persistence_manager()
            .get_device_snapshot()
            .await
            .to_serializable();
        let jid = core
            .pn
            .as_ref()
            .map(|j| j.to_string())
            .or_else(|| core.lid.as_ref().map(|j| j.to_string()))
            .unwrap_or_else(|| "unknown".to_string());
        let value = serde_json::to_value(&core)
            .map_err(|e| ApiError::Other(format!("whatsapp: device snapshot: {e}")))?;
        Ok((jid, value))
    }
}
