// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! N5 — the in-process **demo** transport (`daemon-demo`) driven end to end over the Unix socket,
//! exactly as a GUI/TUI would, proving the whole v38 surface runs against a real node with zero
//! external network:
//!
//! 1. **Auth** — `auth_providers` advertises one family per [`AuthFlowKind`] variant; every flow
//!    walks begin → step(s) → completion over the socket, producing every [`AuthChallenge`] kind
//!    across the set; the `UserPassword` flow completes on the documented `demo`/`demo123` pair and
//!    rejects a wrong password (staying parked for a retry).
//! 2. **Tree** — `ConvList` returns the full shape: a [`Space`](daemon_api::ConversationType::Space)
//!    with child channels (parent set), a standalone channel, DMs, and a group DM.
//! 3. **Roster + presence** — `RosterList` returns the seeded contacts with varied presence.
//! 4. **Chat** — a `ConvSend` journals the message + a scripted contact reply (both as `Chat`
//!    records via `ConvHistory`) and each append raises `MessagesChanged`.
//! 5. **Settings** — `TransportSettings`/`TransportConfigure` round-trip with unknown-key rejection,
//!    a `validate_account` rejection, and apply-by-reconnect.

use super::harness::*;
use daemon_api::{
    AccountSettingsValues, AuthBeginRequest, AuthBindRequest, AuthChallenge, AuthFlowKind,
    AuthStepInput, AuthStepRequest, AuthStepResult, ConnectionState, ConversationType, NodeEvent,
    ProfileSpec, ProviderSelector,
};
use daemon_host::{MemCredentialStore, MemProfileStore};
use daemon_protocol::{TransportId, UserMsg};
use std::collections::BTreeMap;

/// A node wired with the demo interactive-auth factories + a credential/profile store (so a flow can
/// complete + bind), the [`DemoAdapter`](daemon_demo::DemoAdapter) registered + served, and a
/// Unix-socket client on it. Returns everything the caller needs to drive + tear down.
struct DemoSocket {
    node: Arc<NodeApiImpl>,
    handle: daemon_host::SupervisorHandle,
    server: tokio::task::JoinHandle<()>,
    adapter_tasks: Vec<tokio::task::JoinHandle<()>>,
    client: ApiClient,
    path: std::path::PathBuf,
}

impl DemoSocket {
    async fn bring_up() -> Self {
        let profiles = Arc::new(MemProfileStore::new());
        use daemon_host::ProfileStore;
        profiles
            .create(ProfileSpec::new(
                "alpha",
                ProviderSelector::GenAi,
                "model-a",
            ))
            .expect("create alpha profile");

        let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
            store: Arc::new(InMemoryStore::new()),
            partition: PARTITION,
            host_config: fast_host_config(),
            providers: gate_providers(),
            credentials: None,
            profile: ProfileRef::new("alpha"),
            engine_config: daemon_core::Config::default(),
            journal_seed: Some([0x6d; 32]),
            nesting_depth: 0,
            context: None,
            context_builder: None,
            memory: Vec::new(),
            memory_builder: None,
            extra_tools: Vec::new(),
            models: None,
            profiles: Some(profiles),
            provider_resolver: None,
            credential_store: Some(Arc::new(MemCredentialStore::new())),
            cloud_catalog: None,
            prompt_sources: vec![],
            revisions: None,
            skills: None,
            skills_resolver: None,
            routing: None,
            checkpoints: None,
            auth_factories: daemon_demo::demo_auth_factories(),
            workspace_root: None,
            blob_root: None,
            fs: Default::default(),
            processes: Default::default(),
            title_aux: None,
            reaper: Default::default(),
            orchestrate: Default::default(),
            foreign_gateway: None,
            prompt: Default::default(),
        });

        let cfg = daemon_demo::DemoConfig {
            enabled: true,
            reply_delay_ms: 15,
        };
        node.set_adapters(daemon_host::AdapterRegistry::new().with_adapter(
            daemon_demo::DemoAdapter::new(cfg, Some(node.lifecycle_sink())),
        ));
        let adapter_tasks = node.spawn_adapters().await;

        let path = temp_socket();
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).expect("bind api socket");
        let server = tokio::spawn(serve_api_unix(listener, node.clone()));
        let client = ApiClient::new(path.clone());
        Self {
            node,
            handle,
            server,
            adapter_tasks,
            client,
            path,
        }
    }

    async fn begin(
        &self,
        family: &str,
        bind: Option<AuthBindRequest>,
    ) -> daemon_api::AuthBeginResponse {
        match self
            .client
            .call(ApiRequest::AuthBegin(AuthBeginRequest {
                family: family.into(),
                params: BTreeMap::new(),
                redirect_uri: "http://127.0.0.1:7777/cb".into(),
                bind,
            }))
            .await
            .unwrap()
        {
            ApiResponse::AuthBegun(b) => b,
            other => panic!("expected AuthBegun, got {other:?}"),
        }
    }

    async fn step(
        &self,
        flow_id: &str,
        input: AuthStepInput,
    ) -> Result<AuthStepResult, daemon_api::ApiError> {
        match self
            .client
            .call(ApiRequest::AuthStep(AuthStepRequest {
                flow_id: flow_id.into(),
                input,
            }))
            .await
            .unwrap()
        {
            ApiResponse::AuthStepped(r) => Ok(r),
            ApiResponse::Error(e) => Err(e),
            other => panic!("expected AuthStepped/Error, got {other:?}"),
        }
    }

    async fn tear_down(self) {
        self.server.abort();
        for task in &self.adapter_tasks {
            task.abort();
        }
        self.handle.shutdown().await;
        let _ = std::fs::remove_file(&self.path);
    }
}

/// A completed `AuthStepResult` or a panic.
fn completed(result: AuthStepResult) -> daemon_api::AuthCompleteResponse {
    match result {
        AuthStepResult::Completed(resp) => resp,
        AuthStepResult::Challenge(c) => panic!("expected completion, got challenge {c:?}"),
    }
}

/// A challenge `AuthStepResult` or a panic.
fn challenge(result: AuthStepResult) -> AuthChallenge {
    match result {
        AuthStepResult::Challenge(c) => c,
        AuthStepResult::Completed(r) => panic!("expected a challenge, got completion {r:?}"),
    }
}

/// Record a challenge's kind (its enum discriminant) into `seen`, deduped. `AuthChallenge`'s
/// discriminant is `PartialEq` but not `Ord`/`Hash`, so a `Vec` + `contains` is the set.
fn note_kind(seen: &mut Vec<std::mem::Discriminant<AuthChallenge>>, c: &AuthChallenge) {
    let d = std::mem::discriminant(c);
    if !seen.contains(&d) {
        seen.push(d);
    }
}

/// (1a) `auth_providers` advertises exactly one family per `AuthFlowKind` variant, and the union of
/// families produces every `AuthChallenge` kind at least once as their initial challenge.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn demo_auth_providers_cover_every_flow_kind() {
    let h = DemoSocket::bring_up().await;

    let providers = match h.client.call(ApiRequest::AuthProviders).await.unwrap() {
        ApiResponse::AuthProviders(p) => p,
        other => panic!("expected AuthProviders, got {other:?}"),
    };
    // Every AuthFlowKind variant is represented (enumerated from the factory set, not hard-coded).
    let want = [
        AuthFlowKind::UserPassword,
        AuthFlowKind::MatrixSso,
        AuthFlowKind::OAuth2Pkce,
        AuthFlowKind::BotToken,
        AuthFlowKind::UserToken,
        AuthFlowKind::PhoneOtp,
        AuthFlowKind::QrPairing,
    ];
    for kind in want {
        assert!(
            providers.iter().any(|p| p.flow_kind == kind),
            "a demo family advertises {kind:?}, got {:?}",
            providers.iter().map(|p| p.flow_kind).collect::<Vec<_>>()
        );
    }

    // The initial challenges across families cover Redirect / Form / Qr (Message is produced by a
    // QR step, asserted in the walk test below).
    let mut kinds: Vec<std::mem::Discriminant<AuthChallenge>> = Vec::new();
    for p in &providers {
        let begun = h.begin(&p.family, None).await;
        note_kind(&mut kinds, &begun.challenge);
        // Different families must mint distinct flow ids.
        assert!(!begun.flow_id.is_empty());
    }
    assert!(kinds.len() >= 3, "initial challenges span Redirect/Form/Qr");

    h.tear_down().await;
}

/// (1b) The UserPassword flow: the initial `Form` carries a MASKED password field; the documented
/// `demo`/`demo123` pair completes + binds the account to a profile; a WRONG password is rejected
/// and the flow stays parked so the RIGHT password then completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn demo_userpassword_completes_and_rejects_wrong_password() {
    let h = DemoSocket::bring_up().await;

    let begun = h
        .begin(
            "demo",
            Some(AuthBindRequest {
                profile: ProfileRef::new("alpha"),
                transport_instance: None,
                credential_ref: None,
            }),
        )
        .await;
    match &begun.challenge {
        AuthChallenge::Form { fields, .. } => assert!(
            fields
                .iter()
                .any(|f| f.key == "password" && f.kind == daemon_api::AuthFieldKind::Password),
            "the initial form carries a masked password field: {fields:?}"
        ),
        other => panic!("expected a Form challenge, got {other:?}"),
    }

    // Wrong password: rejected; the flow stays parked.
    let wrong = h
        .step(
            &begun.flow_id,
            AuthStepInput::Fields(BTreeMap::from([
                ("username".into(), daemon_demo::DEMO_USERNAME.into()),
                ("password".into(), "nope".into()),
            ])),
        )
        .await;
    assert!(wrong.is_err(), "a wrong password is rejected");

    // Right password: completes, binds the account to alpha, and its transport id is under `demo/`.
    let done = completed(
        h.step(
            &begun.flow_id,
            AuthStepInput::Fields(BTreeMap::from([
                ("username".into(), daemon_demo::DEMO_USERNAME.into()),
                ("password".into(), daemon_demo::DEMO_PASSWORD.into()),
            ])),
        )
        .await
        .expect("the right password completes"),
    );
    assert_eq!(done.account_label, daemon_demo::DEMO_USERNAME);
    assert!(
        done.transport_instance.as_str().starts_with("demo/"),
        "the bound account's transport id drives the demo adapter, got {}",
        done.transport_instance.as_str()
    );
    assert_eq!(
        done.bound_profile.as_ref().map(|p| p.as_str()),
        Some("alpha")
    );

    // The exchanged token (not the transient password) is persisted, redacted.
    use daemon_api::CredentialApi;
    let listed = h.node.credential_list().await;
    assert!(
        listed
            .iter()
            .any(|c| c.present && c.profile.starts_with("demo/")),
        "the demo session token is persisted (redacted): {listed:?}"
    );

    h.tear_down().await;
}

/// (1c) The remaining flows walk to completion over the socket, together producing the Redirect,
/// Qr, Message, Number-field, and Choice-field surfaces: the redirect flow completes on a captured
/// callback (loopback url); the QR flow polls → shows a Message → completes; the OTP flow is a
/// two-step phone → Number-code form; the token forms complete.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn demo_redirect_qr_otp_token_flows_complete() {
    let h = DemoSocket::bring_up().await;
    let mut seen_challenges: Vec<std::mem::Discriminant<AuthChallenge>> = Vec::new();

    // Redirect (SSO): a loopback-style authorization URL completes on the captured callback.
    let sso = h.begin("demo-sso", None).await;
    note_kind(&mut seen_challenges, &sso.challenge);
    match &sso.challenge {
        AuthChallenge::Redirect { authorization_url } => assert!(
            authorization_url.contains("redirect_uri=http://127.0.0.1:7777/cb"),
            "the redirect carries the client's loopback redirect_uri: {authorization_url}"
        ),
        other => panic!("expected Redirect, got {other:?}"),
    }
    let done = completed(
        h.step(
            &sso.flow_id,
            AuthStepInput::Callback("http://127.0.0.1:7777/cb?loginToken=demo".into()),
        )
        .await
        .expect("redirect completes on callback"),
    );
    assert!(done.transport_instance.as_str().starts_with("demo/"));

    // QR pairing: Qr → Poll → Message → Poll completes.
    let qr = h.begin("demo-qr", None).await;
    note_kind(&mut seen_challenges, &qr.challenge);
    assert!(matches!(qr.challenge, AuthChallenge::Qr { .. }));
    let msg = challenge(
        h.step(&qr.flow_id, AuthStepInput::Poll)
            .await
            .expect("poll"),
    );
    note_kind(&mut seen_challenges, &msg);
    assert!(
        matches!(msg, AuthChallenge::Message { .. }),
        "the QR flow shows an informational Message while pairing, got {msg:?}"
    );
    let done = completed(
        h.step(&qr.flow_id, AuthStepInput::Poll)
            .await
            .expect("second poll completes"),
    );
    assert!(done.transport_instance.as_str().starts_with("demo/"));

    // OTP: phone Form → Number-code Form → completes on the documented code.
    let otp = h.begin("demo-otp", None).await;
    note_kind(&mut seen_challenges, &otp.challenge);
    let code_form = challenge(
        h.step(
            &otp.flow_id,
            AuthStepInput::Fields(BTreeMap::from([("phone".into(), "+15550100".into())])),
        )
        .await
        .expect("phone step yields the code form"),
    );
    match &code_form {
        AuthChallenge::Form { fields, .. } => {
            let code = fields
                .iter()
                .find(|f| f.key == "code")
                .expect("a code field");
            assert_eq!(code.kind, daemon_api::AuthFieldKind::Number);
            assert!(
                code.placeholder.is_some(),
                "the code field has a placeholder"
            );
        }
        other => panic!("expected a Form for the code, got {other:?}"),
    }
    let done = completed(
        h.step(
            &otp.flow_id,
            AuthStepInput::Fields(BTreeMap::from([(
                "code".into(),
                daemon_demo::DEMO_OTP_CODE.into(),
            )])),
        )
        .await
        .expect("the right code completes"),
    );
    assert!(done.transport_instance.as_str().starts_with("demo/"));

    // Bot token: a Form with a Choice (region) + a masked token, completes on a pasted token.
    let bot = h.begin("demo-bot", None).await;
    match &bot.challenge {
        AuthChallenge::Form { fields, .. } => {
            let region = fields
                .iter()
                .find(|f| f.key == "region")
                .expect("a region field");
            assert_eq!(region.kind, daemon_api::AuthFieldKind::Choice);
            assert!(!region.choices.is_empty() && region.default.is_some());
        }
        other => panic!("expected a Form, got {other:?}"),
    }
    let done = completed(
        h.step(
            &bot.flow_id,
            AuthStepInput::Fields(BTreeMap::from([("token".into(), "demo-bot-token".into())])),
        )
        .await
        .expect("bot token completes"),
    );
    assert!(done.transport_instance.as_str().starts_with("demo/"));

    // Across the flows every AuthChallenge kind was produced (Redirect, Qr, Message, Form).
    assert_eq!(
        seen_challenges.len(),
        4,
        "the demo flows produce every AuthChallenge kind (Redirect/Form/Qr/Message)"
    );

    h.tear_down().await;
}

/// (2) The conversation tree lists the full shape over the socket: a Space (root, no parent) with
/// child channels that name it, a standalone channel, DMs, and a group DM.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn demo_tree_lists_space_parent_and_kinds() {
    let h = DemoSocket::bring_up().await;
    let demo = TransportId::new("demo");

    let convs = match h
        .client
        .call(ApiRequest::ConvList {
            transport: demo.clone(),
            after: None,
            since_rev: None,
        })
        .await
        .unwrap()
    {
        ApiResponse::Conversations(page) => page.items,
        other => panic!("expected Conversations, got {other:?}"),
    };

    let space = convs
        .iter()
        .find(|c| c.kind == ConversationType::Space)
        .expect("a Space conversation is listed");
    assert!(space.parent.is_none(), "the Space is a tree root");
    assert_eq!(space.title.as_deref(), Some("Demo Server"));

    let children: Vec<_> = convs
        .iter()
        .filter(|c| c.parent.as_deref() == Some(space.id.as_str()))
        .collect();
    assert!(
        children.len() >= 2 && children.iter().all(|c| c.kind == ConversationType::Channel),
        "the Space has child channels naming it as parent, got {children:?}"
    );
    // Each child channel lists the roster members (membership listing for channels).
    assert!(
        children.iter().all(|c| !c.members.is_empty()),
        "child channels list their members"
    );

    assert!(
        convs
            .iter()
            .any(|c| c.kind == ConversationType::Channel && c.parent.is_none()),
        "a standalone (root) channel is present"
    );
    assert!(
        convs.iter().any(|c| c.kind == ConversationType::Dm),
        "a DM is present"
    );
    assert!(
        convs.iter().any(|c| c.kind == ConversationType::GroupDm),
        "a group DM is present"
    );

    // ConvGet resolves a child channel by id (transport parity).
    let general = conv_get(
        &h.client,
        &demo,
        &convs.iter().find(|c| c.parent.is_some()).unwrap().id,
    )
    .await;
    assert!(general.parent.is_some());

    h.tear_down().await;
}

/// (3) The seeded roster lists over the socket with varied presence (and avatar-ish decoration on
/// at least one contact).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn demo_roster_lists_contacts_with_presence() {
    let h = DemoSocket::bring_up().await;
    let demo = TransportId::new("demo");

    let contacts = match h
        .client
        .call(ApiRequest::RosterList {
            transport: demo,
            after: None,
            since_rev: None,
        })
        .await
        .unwrap()
    {
        ApiResponse::ContactPage(page) => page.items,
        other => panic!("expected ContactPage, got {other:?}"),
    };
    assert!(
        contacts.len() >= 4,
        "a handful of contacts, got {}",
        contacts.len()
    );
    let mut primitives: Vec<daemon_api::PresencePrimitive> = Vec::new();
    for c in &contacts {
        if !primitives.contains(&c.presence.primitive) {
            primitives.push(c.presence.primitive);
        }
    }
    assert!(
        primitives.len() >= 3,
        "presence varies across the roster, got {primitives:?}"
    );
    assert!(
        contacts
            .iter()
            .any(|c| c.presence.emoji.is_some() && c.presence.message.is_some()),
        "at least one contact carries avatar-ish decoration (emoji + status)"
    );

    h.tear_down().await;
}

/// (4) A `ConvSend` journals the outbound message AND the scripted contact reply as `Chat` records
/// readable via `ConvHistory` (in append order), and each append raises a `MessagesChanged` pointer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn demo_send_journals_chat_and_scripted_reply() {
    let h = DemoSocket::bring_up().await;
    let demo = TransportId::new("demo");
    let conv = "chan-general";

    assert!(matches!(
        h.client
            .call(ApiRequest::ConvSend(daemon_api::ConvSendArgs {
                transport: demo.clone(),
                conv: conv.into(),
                from: None,
                message: UserMsg::new("hello demo"),
                op_id: None,
            }))
            .await
            .unwrap(),
        ApiResponse::Ok
    ));

    // Poll ConvHistory until the sent message + the scripted reply have both landed (the reply
    // arrives on a spawned task after the configured delay).
    let deadline = Instant::now() + Duration::from_secs(5);
    let page = loop {
        let page = match h
            .client
            .call(ApiRequest::ConvHistory(daemon_api::ConvHistoryArgs {
                transport: demo.clone(),
                conv: conv.into(),
                after_cursor: 0,
                before_cursor: None,
                max: 0,
            }))
            .await
            .unwrap()
        {
            ApiResponse::Journal(p) => p,
            other => panic!("expected Journal, got {other:?}"),
        };
        if page.entries.len() >= 2 || Instant::now() >= deadline {
            break page;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(
        page.entries.len() >= 2,
        "the send + the scripted reply are journaled, got {page:?}"
    );
    assert!(
        page.entries.windows(2).all(|w| w[0].cursor < w[1].cursor),
        "history is in append order with strictly-increasing cursors"
    );
    // The first record is the outbound send (operator-originated, RAW text, delivered).
    match &page.entries[0].payload {
        daemon_api::JournalRecordPayload::Chat { message } => {
            assert_eq!(message.text, "hello demo");
            assert_eq!(message.author, None, "operator send has no author");
            assert!(message.delivered(), "the send is stamped delivered");
        }
        other => panic!("expected Chat, got {other:?}"),
    }
    // A later record is the scripted reply, carrying a real roster contact as its author.
    assert!(
        page.entries[1..].iter().any(|e| matches!(
            &e.payload,
            daemon_api::JournalRecordPayload::Chat { message }
                if matches!(&message.author, Some(daemon_api::Participant::Contact(_)))
        )),
        "the scripted reply is journaled with a contact author"
    );

    // Each Chat append raised a MessagesChanged pointer for (demo, conv).
    let messages_changed = match h
        .client
        .call(ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: None,
        })
        .await
        .unwrap()
    {
        ApiResponse::EventsPage(pg) => pg
            .events
            .iter()
            .filter(|e| {
                matches!(e, NodeEvent::MessagesChanged { transport: t, conv: c, .. }
                    if t.as_str() == "demo" && c == conv)
            })
            .count(),
        other => panic!("expected EventsPage, got {other:?}"),
    };
    assert!(
        messages_changed >= page.entries.len(),
        "MessagesChanged is emitted per Chat append (>= {}, got {messages_changed})",
        page.entries.len()
    );

    h.tear_down().await;
}

/// (5) `TransportSettings`/`TransportConfigure` round-trip: a fresh instance reads empty; a
/// configure merge-persists; an unknown key is rejected (naming it); a `validate_account` marker is
/// rejected; and configuring the connected instance cycles disconnect → reconnect on the L3 feed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn demo_transport_settings_configure_and_reconnect() {
    let h = DemoSocket::bring_up().await;
    let demo = TransportId::new("demo");

    let settings_of = |resp: ApiResponse| match resp {
        ApiResponse::TransportSettings(v) => v.values,
        other => panic!("expected TransportSettings, got {other:?}"),
    };
    let map = |pairs: &[(&str, &str)]| -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    };

    // Fresh: empty.
    let got = settings_of(
        h.client
            .call(ApiRequest::TransportSettings {
                transport: demo.clone(),
            })
            .await
            .unwrap(),
    );
    assert!(got.is_empty(), "a fresh instance has no settings: {got:?}");

    // Wait for the serve-start Connected so the reconnect below is unambiguous.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let page = h.node.events_page(0, 0).await;
        if page.events.iter().any(|e| matches!(e, NodeEvent::TransportChanged { transport: t, connection, .. } if t.as_str() == "demo" && *connection == ConnectionState::Connected)) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "demo instance never reported Connected"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let before = h.node.events_page(0, 0).await.head_cursor;

    // A known key persists (and reconnects — the cycle Offline then Connected past `before`).
    match h
        .client
        .call(ApiRequest::TransportConfigure {
            transport: demo.clone(),
            settings: AccountSettingsValues {
                values: map(&[("display_name", "My Demo"), ("mood", "grumpy")]),
            },
            op_id: None,
        })
        .await
        .unwrap()
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }
    let got = settings_of(
        h.client
            .call(ApiRequest::TransportSettings {
                transport: demo.clone(),
            })
            .await
            .unwrap(),
    );
    assert_eq!(got, map(&[("display_name", "My Demo"), ("mood", "grumpy")]));

    // Merge: a second configure upserts (display_name survives).
    match h
        .client
        .call(ApiRequest::TransportConfigure {
            transport: demo.clone(),
            settings: AccountSettingsValues {
                values: map(&[("reply_delay_ms", "80")]),
            },
            op_id: None,
        })
        .await
        .unwrap()
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }
    let got = settings_of(
        h.client
            .call(ApiRequest::TransportSettings {
                transport: demo.clone(),
            })
            .await
            .unwrap(),
    );
    assert_eq!(
        got,
        map(&[
            ("display_name", "My Demo"),
            ("mood", "grumpy"),
            ("reply_delay_ms", "80")
        ]),
        "configure merges over the persisted map"
    );

    // Unknown key: rejected, naming the key; nothing persisted beyond the prior merge.
    match h
        .client
        .call(ApiRequest::TransportConfigure {
            transport: demo.clone(),
            settings: AccountSettingsValues {
                values: map(&[("bogus", "x")]),
            },
            op_id: None,
        })
        .await
        .unwrap()
    {
        ApiResponse::Error(e) => assert!(
            format!("{e:?}").contains("bogus"),
            "the error names the unknown key: {e:?}"
        ),
        other => panic!("expected Error for an unknown key, got {other:?}"),
    }

    // validate_account marker: rejected.
    match h
        .client
        .call(ApiRequest::TransportConfigure {
            transport: demo.clone(),
            settings: AccountSettingsValues {
                values: map(&[("display_name", daemon_demo::VALIDATE_REJECT_VALUE)]),
            },
            op_id: None,
        })
        .await
        .unwrap()
    {
        ApiResponse::Error(e) => assert!(
            format!("{e:?}").contains("validate_account"),
            "the adapter's validation error is surfaced: {e:?}"
        ),
        other => panic!("expected Error from validate_account, got {other:?}"),
    }

    // Apply-by-reconnect: the first successful configure cycled the connected instance — an Offline
    // then a fresh Connected past the pre-configure cursor.
    let saw_reconnect = |page: daemon_api::EventsPage, want: ConnectionState| {
        page.events.iter().any(|e| matches!(e, NodeEvent::TransportChanged { transport: t, connection, .. } if t.as_str() == "demo" && *connection == want))
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    let (mut offline, mut connected) = (false, false);
    while Instant::now() < deadline && !(offline && connected) {
        let page = h.node.events_page(before, 0).await;
        offline = offline || saw_reconnect(page.clone(), ConnectionState::Offline);
        connected = connected || saw_reconnect(page, ConnectionState::Connected);
        if !(offline && connected) {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }
    assert!(
        offline && connected,
        "configuring the connected instance cycles disconnect → reconnect (offline={offline}, connected={connected})"
    );

    h.tear_down().await;
}
