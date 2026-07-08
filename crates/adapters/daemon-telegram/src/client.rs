// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// This module is the SOLE home of every grammers / MTProto type in the crate — the confinement seam
// that keeps the SDK out of the adapter, auth, inbound, outbound, and mapping modules (mirrors how
// `daemon-matrix` isolates `matrix-sdk`). It opens the per-account on-disk session store (a daemon
// data-root path, not attacker-influenced), so raw fs is allowed file-wide as `daemon-matrix`'s
// `account.rs` does; production egress still routes through grammers' own MTProto networking.
#![allow(clippy::disallowed_methods)]

//! The confined grammers backend: the real implementations of the crate's two SDK-agnostic seams —
//! [`LoginBackend`](crate::auth::LoginBackend) (interactive login) and
//! [`TelegramClient`](crate::adapter::TelegramClient) (conversation/membership/contacts/directory
//! verbs + outbound send) — plus the multi-account [`serve`] bring-up + per-account update loop.
//!
//! grammers 0.10's peer model is object-capability: an operation needs a [`PeerRef`] (identity +
//! authority), which is obtained from the session cache — not a bare id. The update loop therefore
//! caches each seen chat/sender [`Peer`] keyed by its Bot-API id, and the verb bodies resolve the
//! `PeerRef` from that cache. A verb on a peer the account has never seen returns `Unsupported`
//! rather than fabricating authority — the honest limit of the friendly API without raw MTProto.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use daemon_api::{ApiError, ContactInfo, ConversationInfo, NodeApi};
use daemon_host::AccountProvisioning;
use daemon_protocol::TransportId;

use grammers_client::client::UpdatesConfiguration;
use grammers_client::peer::{Peer, User};
use grammers_client::tl;
use grammers_client::update::Update;
use grammers_client::{Client, SenderPool};
use grammers_session::storages::SqliteSession;

/// The update receiver half of a [`SenderPool`] — the sequential stream of network updates the
/// per-account loop drains via [`Client::stream_updates`]. Aliased so the bring-up plumbing needn't
/// spell the (re-exported) grammers types at every use.
type UpdatesRx =
    tokio::sync::mpsc::UnboundedReceiver<grammers_client::session::updates::UpdatesLike>;

use crate::account::{account_session_path, bare_account, AccountMode, StoredSession};
use crate::adapter::TelegramClient;
use crate::auth::{CodeStep, LoginBackend, LoginIdentity};
use crate::config::TelegramConfig;
use crate::inbound::{self, InboundCtx, InboundEvent};
use crate::mapping::{contact_from, conversation_from};
use crate::outbound::{DeliveryManager, TelegramProjector};
use crate::{LiveClients, FAMILY};

/// The Bot-API dialog id of a peer — the stable `i64` the daemon-opaque conversation id renders.
fn peer_i64(peer: &Peer) -> i64 {
    peer.id().bot_api_dialog_id_unchecked()
}

/// Whether a peer is a 1:1 private chat (a user) vs. a group/channel.
fn peer_is_dm(peer: &Peer) -> bool {
    matches!(peer, Peer::User(_))
}

/// Open (creating if needed) the per-account SQLite session store, keyed by `credential_ref`.
async fn open_session(
    store_root: &Path,
    credential_ref: &str,
) -> Result<Arc<SqliteSession>, ApiError> {
    let path = account_session_path(store_root, credential_ref);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| ApiError::Other(format!("telegram: creating session dir: {e}")))?;
    }
    let session = SqliteSession::open(&path)
        .await
        .map_err(|e| ApiError::Other(format!("telegram: opening session store: {e}")))?;
    Ok(Arc::new(session))
}

/// Build a connected grammers [`Client`] over a freshly-opened session store, spawning the sender
/// pool runner (the runner drives I/O until every handle drops). Returns the client + its store.
async fn connect_client(
    store_root: &Path,
    api_id: i32,
    credential_ref: &str,
) -> Result<(Client, Arc<SqliteSession>), ApiError> {
    let session = open_session(store_root, credential_ref).await?;
    let SenderPool {
        runner,
        updates: _updates,
        handle,
    } = SenderPool::new(session.clone(), api_id);
    let client = Client::new(handle);
    tokio::spawn(runner.run());
    Ok((client, session))
}

// ---------------------------------------------------------------------------
// Interactive login (LoginBackend)
// ---------------------------------------------------------------------------

/// A grammers-backed [`LoginBackend`]: holds a connected client + the continuation tokens across the
/// challenge/response steps (behind `Mutex`es, since the flow steps in place under `&self`).
struct GrammersLogin {
    client: Client,
    api_hash: String,
    mode: AccountMode,
    login_token: Mutex<Option<grammers_client::client::LoginToken>>,
    password_token: Mutex<Option<grammers_client::client::PasswordToken>>,
    bot_token: Mutex<Option<String>>,
}

/// Connect a grammers client and wrap it as the [`LoginBackend`] the auth flow drives.
pub(crate) async fn connect_login(
    store_root: &Path,
    api_id: i32,
    api_hash: &str,
    credential_ref: &str,
    mode: AccountMode,
) -> Result<Arc<dyn LoginBackend>, ApiError> {
    let (client, _session) = connect_client(store_root, api_id, credential_ref).await?;
    Ok(Arc::new(GrammersLogin {
        client,
        api_hash: api_hash.to_string(),
        mode,
        login_token: Mutex::new(None),
        password_token: Mutex::new(None),
        bot_token: Mutex::new(None),
    }))
}

#[async_trait]
impl LoginBackend for GrammersLogin {
    async fn request_code(&self, phone: &str) -> Result<(), ApiError> {
        let token = self
            .client
            .request_login_code(phone, &self.api_hash)
            .await
            .map_err(|e| ApiError::Other(format!("telegram request_login_code: {e}")))?;
        *self.login_token.lock().unwrap() = Some(token);
        Ok(())
    }

    async fn submit_code(&self, code: &str) -> Result<CodeStep, ApiError> {
        let token = self
            .login_token
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| ApiError::Other("telegram: no login code requested yet".into()))?;
        match self.client.sign_in(&token, code).await {
            Ok(_user) => Ok(CodeStep::Done),
            Err(grammers_client::SignInError::PasswordRequired(pt)) => {
                *self.password_token.lock().unwrap() = Some(pt);
                Ok(CodeStep::PasswordRequired)
            }
            Err(e) => Err(ApiError::Other(format!("telegram sign_in: {e}"))),
        }
    }

    async fn submit_password(&self, password: &str) -> Result<(), ApiError> {
        let pt = self
            .password_token
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| ApiError::Other("telegram: no 2FA password expected".into()))?;
        self.client
            .check_password(pt, password.as_bytes())
            .await
            .map(|_user| ())
            .map_err(|e| ApiError::Other(format!("telegram check_password: {e}")))
    }

    async fn bot_sign_in(&self, token: &str) -> Result<(), ApiError> {
        self.client
            .bot_sign_in(token, &self.api_hash)
            .await
            .map_err(|e| ApiError::Other(format!("telegram bot_sign_in: {e}")))?;
        *self.bot_token.lock().unwrap() = Some(token.to_string());
        Ok(())
    }

    async fn finish(&self) -> Result<LoginIdentity, ApiError> {
        let me = self
            .client
            .get_me()
            .await
            .map_err(|e| ApiError::Other(format!("telegram get_me: {e}")))?;
        let account_id = me.id().bot_api_dialog_id_unchecked();
        let label = me
            .username()
            .map(|u| u.to_string())
            .unwrap_or_else(|| me.full_name());
        // The on-disk SQLite session store now holds the authorization key (grammers persists it via
        // the sender pool); the credential blob records only the mode + (for a bot) its token.
        let blob = match self.mode {
            AccountMode::Bot => {
                let token = self
                    .bot_token
                    .lock()
                    .unwrap()
                    .clone()
                    .ok_or_else(|| ApiError::Other("telegram: bot token missing".into()))?;
                StoredSession::bot(token, account_id)
            }
            AccountMode::User => StoredSession::user(account_id),
        }
        .to_blob()
        .map_err(|e| ApiError::Other(format!("telegram: serializing session blob: {e}")))?;
        Ok(LoginIdentity {
            account_id,
            label,
            credential_blob: blob,
        })
    }
}

// ---------------------------------------------------------------------------
// Verb + outbound seam (TelegramClient)
// ---------------------------------------------------------------------------

/// A grammers-backed [`TelegramClient`]: the live client plus a peer cache (Bot-API id -> [`Peer`])
/// the update loop fills and the verb bodies resolve their ocap [`PeerRef`] from.
pub(crate) struct GrammersTelegramClient {
    client: Client,
    transport: TransportId,
    /// Whether this account is a user or a bot. Contacts/roster MTProto calls are user-only, so the
    /// roster verbs reject a bot session with a clean `Unsupported` rather than surfacing the raw
    /// `BOT_METHOD_INVALID` RPC error.
    mode: AccountMode,
    peers: Mutex<HashMap<i64, Peer>>,
}

impl GrammersTelegramClient {
    fn new(client: Client, transport: TransportId, mode: AccountMode) -> Arc<Self> {
        Arc::new(Self {
            client,
            transport,
            mode,
            peers: Mutex::new(HashMap::new()),
        })
    }

    /// Guard the user-only contact roster: a bot account has no server-side contact list.
    fn ensure_user_roster(&self) -> Result<(), ApiError> {
        match self.mode {
            AccountMode::User => Ok(()),
            AccountMode::Bot => Err(ApiError::Unsupported(
                "telegram bot accounts have no server-side contact roster".into(),
            )),
        }
    }

    /// Cache a seen `peer` (chat or sender) so later verbs can resolve its ocap reference.
    fn cache_peer(&self, peer: &Peer) {
        self.peers
            .lock()
            .unwrap()
            .insert(peer_i64(peer), peer.clone());
    }

    /// A cached copy of the peer for `id`, if seen.
    fn cached(&self, id: i64) -> Option<Peer> {
        self.peers.lock().unwrap().get(&id).cloned()
    }

    /// Resolve the ocap reference for a previously-seen peer id (`Unsupported` if not cached, or if
    /// the cached peer carries no usable authority).
    async fn peer_ref(&self, id: i64) -> Result<grammers_session::types::PeerRef, ApiError> {
        let peer = self
            .cached(id)
            .ok_or_else(|| ApiError::Unsupported(format!("telegram peer {id} not in cache")))?;
        peer.to_ref()
            .await
            .map_err(|e| ApiError::Other(format!("telegram peer ref: {e}")))?
            .ok_or_else(|| {
                ApiError::Unsupported(format!("telegram peer {id} has no usable reference"))
            })
    }

    fn project(&self, peer: &Peer) -> ConversationInfo {
        conversation_from(
            &self.transport,
            peer_i64(peer),
            peer_is_dm(peer),
            peer.name().map(str::to_string),
            Vec::new(),
        )
    }
}

#[async_trait]
impl TelegramClient for GrammersTelegramClient {
    async fn send_text(&self, chat_id: i64, text: &str) -> Result<(), ApiError> {
        let peer = self.peer_ref(chat_id).await?;
        self.client
            .send_message(peer, text)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("telegram send: {e}")))
    }

    async fn list_conversations(&self, _transport: &TransportId) -> Vec<ConversationInfo> {
        let peers: Vec<Peer> = self.peers.lock().unwrap().values().cloned().collect();
        peers.iter().map(|p| self.project(p)).collect()
    }

    async fn get_conversation(
        &self,
        _transport: &TransportId,
        chat_id: i64,
    ) -> Option<ConversationInfo> {
        self.cached(chat_id).map(|p| self.project(&p))
    }

    async fn join_channel(
        &self,
        _transport: &TransportId,
        target: &str,
    ) -> Result<ConversationInfo, ApiError> {
        let username = target.trim().trim_start_matches('@');
        let peer = self
            .client
            .resolve_username(username)
            .await
            .map_err(|e| ApiError::Other(format!("telegram resolve_username: {e}")))?
            .ok_or_else(|| ApiError::Other(format!("telegram: no such public chat @{username}")))?;
        let peer_ref = peer
            .to_ref()
            .await
            .map_err(|e| ApiError::Other(format!("telegram peer ref: {e}")))?
            .ok_or_else(|| ApiError::Other("telegram: chat has no usable reference".into()))?;
        self.client
            .join_chat(peer_ref)
            .await
            .map_err(|e| ApiError::Other(format!("telegram join_chat: {e}")))?;
        self.cache_peer(&peer);
        Ok(self.project(&peer))
    }

    async fn leave(&self, chat_id: i64) -> Result<(), ApiError> {
        let peer = self.peer_ref(chat_id).await?;
        self.client
            .delete_dialog(peer)
            .await
            .map_err(|e| ApiError::Other(format!("telegram leave: {e}")))
    }

    async fn remove(&self, chat_id: i64, user_id: i64) -> Result<(), ApiError> {
        let chat = self.peer_ref(chat_id).await?;
        let user = self.peer_ref(user_id).await?;
        self.client
            .kick_participant(chat, user)
            .await
            .map_err(|e| ApiError::Other(format!("telegram remove: {e}")))
    }

    async fn ban(&self, chat_id: i64, user_id: i64) -> Result<(), ApiError> {
        let chat = self.peer_ref(chat_id).await?;
        let user = self.peer_ref(user_id).await?;
        // Revoking `view_messages` bans the user from the chat (the builder default grants all
        // rights, i.e. un-bans; we revoke the base right to ban).
        self.client
            .set_banned_rights(chat, user)
            .view_messages(false)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("telegram ban: {e}")))
    }

    async fn get_profile(&self, user_id: i64) -> Result<String, ApiError> {
        let peer = self.cached(user_id).ok_or_else(|| {
            ApiError::Unsupported(format!("telegram user {user_id} not in cache"))
        })?;
        let mut lines = vec![format!("user_id: {user_id}")];
        if let Some(name) = peer.name() {
            lines.push(format!("name: {name}"));
        }
        if let Some(username) = peer.username() {
            lines.push(format!("username: @{username}"));
        }
        Ok(lines.join("\n"))
    }

    async fn search_contacts(&self, query: &str) -> Result<Vec<ContactInfo>, ApiError> {
        let username = query.trim().trim_start_matches('@');
        if username.is_empty() {
            return Ok(Vec::new());
        }
        match self
            .client
            .resolve_username(username)
            .await
            .map_err(|e| ApiError::Other(format!("telegram directory search: {e}")))?
        {
            Some(peer) => {
                // Cache the resolved peer so a follow-up `roster_add` can build its ocap `InputUser`
                // from the found contact (the resolve→add flow, mirroring the inbound peer cache).
                self.cache_peer(&peer);
                Ok(vec![contact_from(
                    peer_i64(&peer),
                    peer.name().map(str::to_string),
                )])
            }
            None => Ok(Vec::new()),
        }
    }

    async fn roster_list(&self, _transport: &TransportId) -> Result<Vec<ContactInfo>, ApiError> {
        self.ensure_user_roster()?;
        let contacts = self
            .client
            .invoke(&tl::functions::contacts::GetContacts { hash: 0 })
            .await
            .map_err(|e| ApiError::Other(format!("telegram getContacts: {e}")))?;
        let users = match contacts {
            tl::enums::contacts::Contacts::Contacts(c) => c.users,
            tl::enums::contacts::Contacts::NotModified => Vec::new(),
        };
        let mut out = Vec::with_capacity(users.len());
        for user in users {
            // Cache each roster user so a follow-up add/update/remove resolves its ocap `InputUser`.
            let peer = Peer::User(User::from_raw(&self.client, user));
            self.cache_peer(&peer);
            out.push(contact_from(
                peer_i64(&peer),
                peer.name().map(str::to_string),
            ));
        }
        Ok(out)
    }

    async fn roster_add(&self, user_id: i64, first_name: &str) -> Result<(), ApiError> {
        self.ensure_user_roster()?;
        // `contacts.addContact` upserts: it also refreshes the first/last name for an existing
        // contact, so the same call backs both `roster.add` and `roster.update`. It needs an ocap
        // `InputUser` (id + access hash), resolved from the cached peer for `user_id`.
        let peer_ref = self.peer_ref(user_id).await?;
        let id: tl::enums::InputUser = (&peer_ref).into();
        self.client
            .invoke(&tl::functions::contacts::AddContact {
                add_phone_privacy_exception: false,
                id,
                first_name: first_name.to_string(),
                last_name: String::new(),
                phone: String::new(),
                note: None,
            })
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("telegram addContact: {e}")))
    }

    async fn roster_remove(&self, user_id: i64) -> Result<(), ApiError> {
        self.ensure_user_roster()?;
        let peer_ref = self.peer_ref(user_id).await?;
        let id: tl::enums::InputUser = (&peer_ref).into();
        self.client
            .invoke(&tl::functions::contacts::DeleteContacts { id: vec![id] })
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("telegram deleteContacts: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Multi-account bring-up
// ---------------------------------------------------------------------------

/// One brought-up account: its transport id, live client (for the update loop), verb/outbound seam,
/// and the update receiver the loop consumes.
struct BroughtUp {
    transport: TransportId,
    bare: String,
    client: Client,
    gclient: Arc<GrammersTelegramClient>,
    updates: UpdatesRx,
}

/// Bring up every credential-bound Telegram account and run the inbound update loop + outbound
/// delivery for each until its stream ends (or the task is aborted). Spawned in-process at host
/// launch. Mirrors the Matrix adapter's `serve` shape.
pub async fn serve(
    api: Arc<dyn NodeApi>,
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: TelegramConfig,
    live_clients: LiveClients,
) {
    if !cfg.enabled {
        return;
    }
    let accounts = provisioning.bound_accounts(FAMILY);
    if accounts.is_empty() {
        tracing::info!("telegram: enabled but no bound telegram accounts; nothing to do");
        return;
    }

    let ingestor = Arc::new(daemon_ingest::Ingestor::with_policy(
        api.clone(),
        cfg.ingest_policy(),
    ));
    let routes = Arc::new(cfg.routes.clone());

    let mut brought: Vec<BroughtUp> = Vec::new();
    for acct in &accounts {
        let Some(blob) = provisioning.account_credential(&acct.credential_ref) else {
            tracing::warn!(
                instance = %acct.transport_instance.as_str(),
                "telegram: no stored session; run `telegram login` first — skipping"
            );
            continue;
        };
        let stored = match StoredSession::from_blob(&blob) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "telegram: bad session blob; skipping");
                continue;
            }
        };

        let session = match open_session(&cfg.store_root, &acct.credential_ref).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, instance = %acct.transport_instance.as_str(), "telegram: session open failed; skipping");
                continue;
            }
        };
        let SenderPool {
            runner,
            updates,
            handle,
        } = SenderPool::new(session, cfg.api_id);
        let client = Client::new(handle);
        tokio::spawn(runner.run());

        match client.is_authorized().await {
            Ok(true) => {}
            Ok(false) => match stored.mode {
                AccountMode::Bot => {
                    let Some(token) = stored.bot_token.as_deref() else {
                        tracing::warn!(instance = %acct.transport_instance.as_str(), "telegram: bot account has no stored token; skipping");
                        continue;
                    };
                    if let Err(e) = client.bot_sign_in(token, &cfg.api_hash).await {
                        tracing::warn!(error = %e, instance = %acct.transport_instance.as_str(), "telegram: bot sign-in failed; skipping");
                        continue;
                    }
                }
                AccountMode::User => {
                    tracing::warn!(instance = %acct.transport_instance.as_str(), "telegram: user session not authorized; run `telegram login` first — skipping");
                    continue;
                }
            },
            Err(e) => {
                tracing::warn!(error = %e, instance = %acct.transport_instance.as_str(), "telegram: authorization check failed; skipping");
                continue;
            }
        }

        let gclient = GrammersTelegramClient::new(
            client.clone(),
            acct.transport_instance.clone(),
            stored.mode,
        );
        brought.push(BroughtUp {
            transport: acct.transport_instance.clone(),
            bare: bare_account(&acct.transport_instance).to_string(),
            client,
            gclient,
            updates,
        });
        tracing::info!(instance = %acct.transport_instance.as_str(), "telegram: account brought up");
    }

    if brought.is_empty() {
        tracing::warn!("telegram: no accounts could be brought up; exiting");
        return;
    }

    // Publish the live clients so the adapter's feature-trait method bodies (which only have `&self`)
    // can resolve the per-account verb seam.
    {
        let mut guard = live_clients.write().await;
        for acct in &brought {
            guard.insert(
                acct.transport.clone(),
                acct.gclient.clone() as Arc<dyn TelegramClient>,
            );
        }
    }

    let client_map: HashMap<TransportId, Arc<dyn TelegramClient>> = brought
        .iter()
        .map(|a| {
            (
                a.transport.clone(),
                a.gclient.clone() as Arc<dyn TelegramClient>,
            )
        })
        .collect();
    let projector = Arc::new(TelegramProjector::new(
        api.clone(),
        ingestor.clone(),
        client_map,
    ));
    let delivery = Arc::new(DeliveryManager::new(api.clone(), projector));

    // Resume delivery for any sessions this transport already owns (reconnect / restart), walking the
    // wire pages so every owned session resumes.
    for acct in &brought {
        let mut after: Option<String> = None;
        loop {
            let page = api
                .delivery_sessions(acct.transport.clone(), after.take())
                .await;
            for session in page.items {
                delivery.ensure(session, acct.transport.clone());
            }
            match page.next {
                Some(next) => after = Some(next),
                None => break,
            }
        }
    }

    // Run each account's update loop until its stream ends.
    let mut tasks = Vec::new();
    for acct in brought {
        let BroughtUp {
            transport,
            bare,
            client,
            gclient,
            updates,
        } = acct;
        let ctx = InboundCtx {
            ingestor: ingestor.clone(),
            delivery: delivery.clone(),
            routes: routes.clone(),
            bare,
            transport: transport.clone(),
        };
        tasks.push(tokio::spawn(run_update_loop(client, gclient, updates, ctx)));
    }

    for task in tasks {
        let _ = task.await;
    }
}

/// One account's inbound update loop: stream updates, cache seen peers (so outbound + verbs can
/// resolve them), normalise each incoming text message into an [`InboundEvent`], and gate it through
/// [`inbound::handle`].
async fn run_update_loop(
    client: Client,
    gclient: Arc<GrammersTelegramClient>,
    updates: UpdatesRx,
    ctx: InboundCtx,
) {
    let mut stream = match client
        .stream_updates(
            updates,
            UpdatesConfiguration {
                catch_up: true,
                ..Default::default()
            },
        )
        .await
    {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, instance = %ctx.transport.as_str(), "telegram: stream_updates failed");
            return;
        }
    };
    tracing::info!(instance = %ctx.transport.as_str(), "telegram: update loop started");
    loop {
        let update = match stream.next().await {
            Ok(u) => u,
            Err(e) => {
                tracing::warn!(error = %e, instance = %ctx.transport.as_str(), "telegram: update loop ended");
                break;
            }
        };
        if let Some(ev) = event_from_update(&gclient, update) {
            inbound::handle(&ctx, ev).await;
        }
    }
}

/// Normalise one grammers [`Update`] into the transport-neutral [`InboundEvent`], caching the chat +
/// sender peers along the way. `None` for anything that is not an incoming text message.
fn event_from_update(gclient: &GrammersTelegramClient, update: Update) -> Option<InboundEvent> {
    let Update::NewMessage(message) = update else {
        return None;
    };
    // Never react to our own posts (the outbound reply path would otherwise loop).
    if message.outgoing() {
        return None;
    }
    let text = message.text().to_string();
    if text.is_empty() {
        return None;
    }
    let chat_id = message.peer_id().bot_api_dialog_id_unchecked();
    let (is_dm, chat_peer) = match message.peer() {
        Some(p) => (peer_is_dm(p), Some(p)),
        None => (false, None),
    };
    if let Some(p) = chat_peer {
        gclient.cache_peer(p);
    }
    let sender = message.sender();
    if let Some(p) = sender {
        gclient.cache_peer(p);
    }
    let sender_id = message
        .sender_id()
        .map(|p| p.bot_api_dialog_id_unchecked())
        .unwrap_or(chat_id);
    let sender_display = sender.and_then(|p| p.name()).map(str::to_string);

    Some(InboundEvent {
        chat_id,
        sender_id,
        sender_display,
        text,
        is_dm,
        mentioned: message.mentioned(),
    })
}
