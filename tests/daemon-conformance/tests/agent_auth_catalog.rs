// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The foreign-agent auth surface at the catalog (wire v47): `AgentRegister` validates the
//! name slug + any manually supplied `auth_descriptor`, and `AgentCatalog` serves a node-DERIVED
//! `auth` verdict — descriptor + credential/marker presence folded in ONE place at assembly, so
//! no client re-implements the rule and no caller-supplied verdict is ever trusted.

use std::sync::Arc;

use std::time::{Duration, Instant};

use daemon_api::{
    AgentAuth, AgentAuthDescriptor, AgentAuthScheme, AgentAuthState, AgentEntry, AgentProtocol,
    AgentRecipe, AgentSource, AgentVerification, AuthApi, AuthBeginRequest, AuthChallenge,
    AuthFlowKind, AuthStepInput, AuthStepRequest, AuthStepResult, ControlApi, EngineSelector,
    Outbound, ProfileApi, ProfileSpec, ProviderSelector, RejectionClassifier, SessionApi,
};
use daemon_common::{ProfileRef, ReqId};
use daemon_core::{MockProvider, Provider, ProviderBuilder, ProviderRegistry};
use daemon_host::{CredentialStore, HostConfig, MemCredentialStore, MemProfileStore, NodeApiImpl};
use daemon_node::{assemble, AssembledNode, NodeAssembly, ProviderResolver};
use daemon_protocol::{AgentCommand, AgentEvent, HostResponse, HostResponseBody, UserMsg};
use daemon_store::InMemoryStore;

/// Assemble a node with an in-memory credential store (the derive input) + the standard mock
/// provider plumbing. Returns the credential store handle so tests can plant secrets/markers.
fn assemble_auth_node() -> (
    Arc<NodeApiImpl>,
    Arc<dyn CredentialStore>,
    daemon_host::SupervisorHandle,
) {
    assemble_auth_node_with(HostConfig::default())
}

fn assemble_auth_node_with(
    host_config: HostConfig,
) -> (
    Arc<NodeApiImpl>,
    Arc<dyn CredentialStore>,
    daemon_host::SupervisorHandle,
) {
    let credentials: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(MockProvider::completing("session done")) as Arc<dyn Provider>
    }));
    // A provider resolver so the profile-aware session builder is installed (the foreign spawn
    // materialization leg opens real sessions); foreign engines never consult it.
    let resolver: ProviderResolver = Arc::new(move |_spec: &ProfileSpec| {
        let builder: ProviderBuilder =
            Arc::new(|| Arc::new(MockProvider::completing("native reply")) as Arc<dyn Provider>);
        builder
    });
    let AssembledNode { node, handle, .. } = assemble(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: daemon_common::PartitionId::DEFAULT,
        host_config,
        providers,
        credentials: None,
        profile: ProfileRef::new("default"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x66; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(Arc::new(MemProfileStore::new())),
        provider_resolver: Some(resolver),
        credential_store: Some(credentials.clone()),
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: None,
        checkpoints: None,
        auth_factories: vec![],
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
    (node, credentials, handle)
}

/// A registrable stream-json entry (no handshake, program need not exist) with an optional
/// manual auth descriptor and an optional caller-supplied `auth` claim.
fn stream_json_entry(
    name: &str,
    descriptor: Option<AgentAuthDescriptor>,
    claimed_auth: Option<AgentAuth>,
) -> AgentEntry {
    AgentEntry {
        name: name.into(),
        recipe: AgentRecipe {
            program: Some("/nonexistent-stream-json-agent".into()),
            args: Vec::new(),
            env: Vec::new(),
            endpoint: None,
        },
        source: AgentSource::Manual,
        protocol: AgentProtocol::StreamJson,
        installed: false,
        version: None,
        capabilities: Vec::new(),
        verification: AgentVerification::NotInstalled,
        auth_descriptor: descriptor,
        auth: claimed_auth,
    }
}

fn api_key_descriptor(var: &str, label: &str) -> AgentAuthDescriptor {
    AgentAuthDescriptor {
        scheme: AgentAuthScheme::ApiKeyEnv {
            var: var.into(),
            label: label.into(),
        },
        rejection: None,
    }
}

async fn catalog_entry(node: &Arc<NodeApiImpl>, name: &str) -> AgentEntry {
    node.agent_catalog()
        .await
        .into_iter()
        .find(|e| e.name == name)
        .unwrap_or_else(|| panic!("catalog serves `{name}`"))
}

/// Agent names fan out into auth family ids (`agent/<name>`), credential refs and state-home
/// path segments, so `AgentRegister` enforces a strict slug before a name can reach any of them.
#[tokio::test]
async fn register_rejects_a_non_slug_agent_name() {
    let (node, _creds, _handle) = assemble_auth_node();
    for bad in [
        "Bad Name",
        "UPPER",
        "a/b",
        "-leading-dash",
        ".hidden",
        "dots..inside",
        &"x".repeat(65),
    ] {
        let err = node
            .agent_register(stream_json_entry(bad, None, None))
            .await
            .expect_err("a non-slug name must fail agent_register");
        assert!(
            err.to_string().contains("not a valid slug"),
            "the error names the rule for `{bad}`: {err}"
        );
    }
    assert!(
        node.agent_catalog().await.iter().all(|e| e.name != "UPPER"),
        "a refused registration never reaches the catalog"
    );
}

/// A manually supplied descriptor is a caller claim that fans out into env injection and runtime
/// classification, so its fields are validated at ingress.
#[tokio::test]
async fn register_validates_a_manual_auth_descriptor() {
    let (node, _creds, _handle) = assemble_auth_node();
    let cases: Vec<(AgentAuthDescriptor, &str)> = vec![
        (
            api_key_descriptor("lower_case", "Key"),
            "not a valid environment variable",
        ),
        (
            api_key_descriptor("1STARTS_WITH_DIGIT", "Key"),
            "not a valid environment variable",
        ),
        (
            api_key_descriptor("", "Key"),
            "not a valid environment variable",
        ),
        (
            api_key_descriptor("GOOD_KEY", "   "),
            "label must not be empty",
        ),
        (
            AgentAuthDescriptor {
                scheme: AgentAuthScheme::OAuthFamily { family: " ".into() },
                rejection: None,
            },
            "family must not be empty",
        ),
        (
            AgentAuthDescriptor {
                scheme: AgentAuthScheme::None,
                rejection: Some(RejectionClassifier {
                    field: "".into(),
                    code: "auth_required".into(),
                }),
            },
            "rejection classifier",
        ),
    ];
    for (descriptor, want) in cases {
        let err = node
            .agent_register(stream_json_entry("badauth", Some(descriptor), None))
            .await
            .expect_err("a malformed descriptor must fail agent_register");
        assert!(
            err.to_string().contains(want),
            "the error names the offending field (want `{want}`): {err}"
        );
    }
    assert!(
        node.agent_catalog()
            .await
            .iter()
            .all(|e| e.name != "badauth"),
        "a refused registration never reaches the catalog"
    );
}

/// The ApiKeyEnv derive: no credential -> `Required` with one runnable UserToken method; the
/// credential landing under the agent-scoped ref flips the SAME catalog row to `Authenticated`
/// (configured, not vendor-verified). A caller-supplied `auth` claim is discarded on the way in.
#[tokio::test]
async fn api_key_verdict_follows_credential_presence() {
    let (node, creds, _handle) = assemble_auth_node();
    // The caller "claims" Authenticated — the node must not care.
    let claimed = AgentAuth {
        state: AgentAuthState::Authenticated,
        methods: Vec::new(),
    };
    node.agent_register(stream_json_entry(
        "keyed",
        Some(api_key_descriptor("KEYED_API_KEY", "Keyed API key")),
        Some(claimed),
    ))
    .await
    .expect("registration with a valid descriptor succeeds");

    let auth = catalog_entry(&node, "keyed")
        .await
        .auth
        .expect("a descriptor-carrying row gets a derived verdict");
    assert_eq!(
        auth.state,
        AgentAuthState::Required,
        "no credential stored -> Required (the caller's Authenticated claim was discarded)"
    );
    assert_eq!(auth.methods.len(), 1, "one runnable method is offered");
    assert_eq!(auth.methods[0].kind, AuthFlowKind::UserToken);
    assert_eq!(auth.methods[0].family, "agent/keyed");

    creds
        .set("agent/keyed/KEYED_API_KEY", "sk-verify")
        .expect("plant the agent-scoped credential");
    let auth = catalog_entry(&node, "keyed").await.auth.expect("verdict");
    assert_eq!(
        auth.state,
        AgentAuthState::Authenticated,
        "the stored credential flips the derived verdict"
    );
}

/// The research-gated vendor schemes (`OAuthFamily`/`DeviceCode`): `Required`, but with NO
/// runnable method until the dedicated auth family actually lands — a client renders "sign-in
/// not available yet", never a dead button. (This is the curated `claude` shape.)
#[tokio::test]
async fn oauth_family_is_required_with_no_runnable_method() {
    let (node, _creds, _handle) = assemble_auth_node();
    node.agent_register(stream_json_entry(
        "vendorish",
        Some(AgentAuthDescriptor {
            scheme: AgentAuthScheme::OAuthFamily {
                family: "agent/vendorish".into(),
            },
            rejection: None,
        }),
        None,
    ))
    .await
    .expect("registration succeeds");
    let auth = catalog_entry(&node, "vendorish")
        .await
        .auth
        .expect("verdict");
    assert_eq!(auth.state, AgentAuthState::Required);
    assert!(
        auth.methods.is_empty(),
        "no runnable method until the vendor family is registered"
    );
}

/// The ACP shape: `authenticate` returns no credential and agent dotfile state is opaque, so the
/// node-owned success marker (`agent/<name>/acp:<method_id>` in the credential store) is the only
/// derivable fact — absent -> `Required` (with a runnable method), present -> `Authenticated`.
#[tokio::test]
async fn acp_marker_flips_the_verdict_to_authenticated() {
    let (node, creds, _handle) = assemble_auth_node();
    let mut entry = stream_json_entry(
        "acpish",
        Some(AgentAuthDescriptor {
            scheme: AgentAuthScheme::AcpAuthenticate,
            rejection: None,
        }),
        None,
    );
    entry.protocol = AgentProtocol::Acp;
    node.agent_register(entry)
        .await
        .expect("registration succeeds");

    let auth = catalog_entry(&node, "acpish").await.auth.expect("verdict");
    assert_eq!(auth.state, AgentAuthState::Required);
    assert_eq!(
        auth.methods.len(),
        1,
        "a manual AcpAuthenticate descriptor without captured methods still offers one runnable method"
    );
    assert_eq!(auth.methods[0].kind, AuthFlowKind::AcpAuthenticate);

    creds
        .set("agent/acpish/acp:oauth", "completed")
        .expect("plant the node-owned success marker");
    let auth = catalog_entry(&node, "acpish").await.auth.expect("verdict");
    assert_eq!(
        auth.state,
        AgentAuthState::Authenticated,
        "the ACP success marker flips the derived verdict"
    );
}

/// A verified ACP handshake that advertises NO authMethods derives scheme `None` ->
/// `NotRequired`: the strongest statement discovery can make without vendor knowledge.
#[tokio::test]
async fn a_quiet_acp_probe_derives_not_required() {
    let (node, _creds, _handle) = assemble_auth_node();
    node.agent_register(AgentEntry {
        name: "mock-acp".into(),
        recipe: AgentRecipe {
            program: Some(env!("CARGO_BIN_EXE_mock_acp_agent").to_string()),
            args: Vec::new(),
            env: Vec::new(),
            endpoint: None,
        },
        source: AgentSource::Manual,
        protocol: AgentProtocol::Acp,
        installed: false,
        version: None,
        capabilities: Vec::new(),
        verification: AgentVerification::NotInstalled,
        auth_descriptor: None,
        auth: None,
    })
    .await
    .expect("the mock ACP agent registers");

    let entry = catalog_entry(&node, "mock-acp").await;
    assert!(
        entry.version.is_some(),
        "the register probe completed the initialize handshake"
    );
    assert_eq!(
        entry.auth_descriptor,
        Some(AgentAuthDescriptor {
            scheme: AgentAuthScheme::None,
            rejection: None,
        }),
        "a confirmed handshake with no advertised authMethods derives scheme None"
    );
    let auth = entry.auth.expect("verdict");
    assert_eq!(auth.state, AgentAuthState::NotRequired);
    assert!(auth.methods.is_empty());
}

/// The dynamic `agent/*` ApiKeyEnv flow end to end over the wire surface (A3): `auth_begin`
/// resolves the namespace family on demand, the Form challenge collects the key, and completion
/// stores it under the agent-scoped ref — flipping the SAME agent's derived catalog verdict.
#[tokio::test]
async fn api_key_flow_authenticates_the_agent_over_the_auth_api() {
    let (node, creds, _handle) = assemble_auth_node();
    node.agent_register(stream_json_entry(
        "keyflow",
        Some(api_key_descriptor("KEYFLOW_API_KEY", "Keyflow API key")),
        None,
    ))
    .await
    .expect("registration succeeds");

    // An unknown agent inside the namespace still refuses cleanly.
    assert!(node
        .auth_begin(AuthBeginRequest {
            family: "agent/nonexistent".into(),
            params: Default::default(),
            redirect_uri: "http://127.0.0.1:0/cb".into(),
            bind: None,
        })
        .await
        .is_err());

    let begun = node
        .auth_begin(AuthBeginRequest {
            family: "agent/keyflow".into(),
            params: Default::default(),
            redirect_uri: "http://127.0.0.1:0/cb".into(),
            bind: None,
        })
        .await
        .expect("the agent/* namespace resolves a registered ApiKeyEnv agent");
    match &begun.challenge {
        AuthChallenge::Form { fields, .. } => {
            assert_eq!(fields.len(), 1, "one pasted-key field");
        }
        other => panic!("ApiKeyEnv begins with a Form challenge, got {other:?}"),
    }

    let mut fields = std::collections::BTreeMap::new();
    fields.insert("key".to_string(), "sk-keyflow-e2e".to_string());
    let result = node
        .auth_step(AuthStepRequest {
            flow_id: begun.flow_id,
            input: AuthStepInput::Fields(fields),
        })
        .await
        .expect("the pasted key completes the flow");
    match result {
        AuthStepResult::Completed(done) => {
            assert_eq!(done.credential_ref, "agent/keyflow/KEYFLOW_API_KEY");
        }
        AuthStepResult::Challenge(c) => panic!("expected completion, got challenge {c:?}"),
    }
    assert_eq!(
        creds.get("agent/keyflow/KEYFLOW_API_KEY").as_deref(),
        Some("sk-keyflow-e2e"),
        "the key landed under the agent-scoped credential ref"
    );
    let auth = catalog_entry(&node, "keyflow").await.auth.expect("verdict");
    assert_eq!(
        auth.state,
        AgentAuthState::Authenticated,
        "the completed flow flipped the derived catalog verdict"
    );
}

/// The dynamic `agent/*` ACP flow end to end (A3): the register probe captures the advertised
/// `authMethods` from a REAL agent binary, `auth_begin` auto-selects the single method, the poll
/// step drives the agent's own `authenticate` through the gateway, and completion writes the
/// node-owned success marker (`agent/<name>/acp:<method>`) that flips the verdict. The agent's
/// own login side effect lands where the AGENT puts it — proving the marker (not agent state) is
/// what the node derives from.
// Harness-owned temp-file bookkeeping (the mock agent's side-effect marker), not node code —
// the ContainedRoot fs ban targets production write paths.
#[allow(clippy::disallowed_methods)]
#[tokio::test]
async fn acp_authenticate_flow_runs_the_real_agent_and_marks_success() {
    let (node, creds, _handle) = assemble_auth_node();
    let mark = std::env::temp_dir().join(format!("mock-acp-auth-mark-{}", std::process::id()));
    let _ = std::fs::remove_file(&mark);
    node.agent_register(AgentEntry {
        name: "mockauth".into(),
        recipe: AgentRecipe {
            program: Some(env!("CARGO_BIN_EXE_mock_acp_auth_agent").to_string()),
            args: Vec::new(),
            env: vec![(
                "MOCK_ACP_AUTH_MARK".to_string(),
                mark.to_string_lossy().into_owned(),
            )],
            endpoint: None,
        },
        source: AgentSource::Manual,
        protocol: AgentProtocol::Acp,
        installed: false,
        version: None,
        capabilities: Vec::new(),
        verification: AgentVerification::NotInstalled,
        auth_descriptor: None,
        auth: None,
    })
    .await
    .expect("the auth-gated mock ACP agent registers");

    // The probe captured the advertised method and derived the AcpAuthenticate descriptor.
    let entry = catalog_entry(&node, "mockauth").await;
    let auth = entry.auth.clone().expect("verdict");
    assert_eq!(auth.state, AgentAuthState::Required);
    assert_eq!(auth.methods.len(), 1);
    assert_eq!(auth.methods[0].id, "mock-login");
    assert_eq!(auth.methods[0].label, "Mock login");

    let begun = node
        .auth_begin(AuthBeginRequest {
            family: "agent/mockauth".into(),
            params: Default::default(),
            redirect_uri: "http://127.0.0.1:0/cb".into(),
            bind: None,
        })
        .await
        .expect("one advertised method auto-selects");
    assert!(
        matches!(begun.challenge, AuthChallenge::Message { .. }),
        "the ACP ceremony begins with an informational message"
    );

    let result = node
        .auth_step(AuthStepRequest {
            flow_id: begun.flow_id,
            input: AuthStepInput::Poll,
        })
        .await
        .expect("the poll drives the agent's authenticate to completion");
    match result {
        AuthStepResult::Completed(done) => {
            assert_eq!(done.credential_ref, "agent/mockauth/acp:mock-login");
        }
        AuthStepResult::Challenge(c) => panic!("expected completion, got challenge {c:?}"),
    }

    // The node-owned marker exists, the agent wrote its own side effect, and the verdict flipped.
    assert!(
        creds.get("agent/mockauth/acp:mock-login").is_some(),
        "the node-owned success marker is stored"
    );
    assert_eq!(
        std::fs::read_to_string(&mark).ok().as_deref(),
        Some("mock-login"),
        "the agent ran its own login ceremony (side effect landed where the AGENT put it)"
    );
    let _ = std::fs::remove_file(&mark);
    let auth = catalog_entry(&node, "mockauth")
        .await
        .auth
        .expect("verdict");
    assert_eq!(auth.state, AgentAuthState::Authenticated);
}

/// Drive the polled outbound stream until `TurnFinished`, answering any parked permission
/// request affirmatively. Returns every drained item (mirrors the stream-json profile e2e).
async fn drain_turn(node: &Arc<NodeApiImpl>, session: &daemon_common::SessionId) -> Vec<Outbound> {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut acc = Vec::new();
    while Instant::now() < deadline {
        let items = node.poll(session.clone(), 0).await.expect("poll");
        for item in items {
            if let Outbound::Request(req) = &item {
                node.respond(
                    session.clone(),
                    HostResponse {
                        request_id: req.request_id,
                        body: HostResponseBody::Approved {
                            approved: true,
                            allow_permanent: false,
                            reason: None,
                        },
                    },
                )
                .await
                .expect("answer the parked permission request");
            }
            let terminal = matches!(item, Outbound::Event(AgentEvent::TurnFinished { .. }));
            acc.push(item);
            if terminal {
                return acc;
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("timed out waiting for TurnFinished; drained: {acc:?}");
}

/// The streamed text of a drained turn (all TextDelta payloads concatenated).
fn streamed_text(items: &[Outbound]) -> String {
    items
        .iter()
        .filter_map(|o| match o {
            Outbound::Event(AgentEvent::TextDelta { text, .. }) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// Spawn-time credential materialization (A4): the node-held `agent/<name>/<VAR>` credential is
/// injected into the foreign child's environment at the ONE shared spawn seam — proven by the
/// spawned agent echoing the value it actually sees — and an explicit recipe env for the same
/// var always wins (the node never silently overrides an operator's recipe).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_materializes_the_agent_credential_into_the_child_env() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        spawn_materializes_the_agent_credential_into_the_child_env_impl(),
    )
    .await;
}
async fn spawn_materializes_the_agent_credential_into_the_child_env_impl() {
    let (node, creds, _handle) = assemble_auth_node();
    let descriptor = api_key_descriptor("MOCK_AGENT_KEY", "Mock agent key");

    // Two catalog identities over the same mock binary: one materializes from the node store,
    // one carries an explicit recipe env that must win over the stored credential.
    for (agent, recipe_env) in [
        ("keyed-agent", Vec::new()),
        (
            "explicit-agent",
            vec![("MOCK_AGENT_KEY".to_string(), "recipe-wins".to_string())],
        ),
    ] {
        node.agent_register(AgentEntry {
            name: agent.into(),
            recipe: AgentRecipe {
                program: Some(env!("CARGO_BIN_EXE_mock_stream_json_agent").to_string()),
                args: Vec::new(),
                env: recipe_env,
                endpoint: None,
            },
            source: AgentSource::Manual,
            protocol: AgentProtocol::StreamJson,
            installed: false,
            version: None,
            capabilities: Vec::new(),
            verification: AgentVerification::NotInstalled,
            auth_descriptor: Some(descriptor.clone()),
            auth: None,
        })
        .await
        .expect("register the mock agent");
        creds
            .set(&format!("agent/{agent}/MOCK_AGENT_KEY"), "sk-materialized")
            .expect("plant the agent credential");
        node.profile_create(ProfileSpec {
            engine: EngineSelector::Foreign {
                agent: agent.into(),
            },
            ..ProfileSpec::new(format!("p-{agent}"), ProviderSelector::Mock, "")
        })
        .await
        .expect("create the foreign profile");
    }

    let session = node
        .session_create(None, Some(ProfileRef::new("p-keyed-agent")))
        .await
        .expect("open a session bound to the keyed agent");
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit a turn");
    let text = streamed_text(&drain_turn(&node, &session).await);
    assert!(
        text.contains("key=sk-materialized"),
        "the spawned agent sees the node-held credential in its env: {text:?}"
    );

    let session = node
        .session_create(None, Some(ProfileRef::new("p-explicit-agent")))
        .await
        .expect("open a session bound to the explicit-env agent");
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello"),
            request_id: ReqId(2),
        },
    )
    .await
    .expect("submit a turn");
    let text = streamed_text(&drain_turn(&node, &session).await);
    assert!(
        text.contains("key=recipe-wins"),
        "an explicit recipe env wins over the stored credential: {text:?}"
    );
}

/// The config-gated isolated spawn (wire v47 A4 second half): with `agent_state_root` set, a
/// stream-json agent runs under a `Clean` scrubbed environment with `HOME` repointed into the
/// node-owned `<root>/<agent>` state home — proven by the child echoing the HOME it actually
/// sees, by a daemon-ambient canary variable NOT leaking through the scrub, and by the injected
/// credential still arriving (materialization and isolation compose).
// Harness-owned temp-dir bookkeeping (the sandboxed state-home root), not node code.
#[allow(clippy::disallowed_methods)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn isolated_spawn_confines_home_and_scrubs_the_ambient_env() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        isolated_spawn_confines_home_and_scrubs_the_ambient_env_impl(),
    )
    .await;
}
#[allow(clippy::disallowed_methods)] // harness-owned temp-dir cleanup, not node code
async fn isolated_spawn_confines_home_and_scrubs_the_ambient_env_impl() {
    let state_root = std::env::temp_dir().join(format!(
        "daemon-agent-homes-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    // A daemon-ambient canary: under InheritFull it would reach the child; the Clean scrub must
    // drop it. Set before assembly so the spawned child could only see it by inheritance.
    std::env::set_var("MOCK_AGENT_CANARY", "daemon-secret");
    let (node, creds, _handle) = assemble_auth_node_with(HostConfig {
        agent_state_root: Some(state_root.clone()),
        ..HostConfig::default()
    });

    node.agent_register(AgentEntry {
        name: "iso-agent".into(),
        recipe: AgentRecipe {
            program: Some(env!("CARGO_BIN_EXE_mock_stream_json_agent").to_string()),
            args: Vec::new(),
            env: vec![("MOCK_AGENT_ECHO_HOME".to_string(), "1".to_string())],
            endpoint: None,
        },
        source: AgentSource::Manual,
        protocol: AgentProtocol::StreamJson,
        installed: false,
        version: None,
        capabilities: Vec::new(),
        verification: AgentVerification::NotInstalled,
        auth_descriptor: Some(api_key_descriptor("MOCK_AGENT_KEY", "Mock agent key")),
        auth: None,
    })
    .await
    .expect("register the mock agent");
    creds
        .set("agent/iso-agent/MOCK_AGENT_KEY", "sk-isolated")
        .expect("plant the agent credential");
    node.profile_create(ProfileSpec {
        engine: EngineSelector::Foreign {
            agent: "iso-agent".into(),
        },
        ..ProfileSpec::new("p-iso-agent", ProviderSelector::Mock, "")
    })
    .await
    .expect("create the foreign profile");

    let session = node
        .session_create(None, Some(ProfileRef::new("p-iso-agent")))
        .await
        .expect("open a session bound to the isolated agent");
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit a turn");
    let text = streamed_text(&drain_turn(&node, &session).await);

    let expected_home = state_root.join("iso-agent");
    assert!(
        text.contains(&format!("home={}", expected_home.display())),
        "the child's HOME is the node-owned state home: {text:?}"
    );
    assert!(
        !text.contains("canary=leaked"),
        "the daemon-ambient canary must not survive the Clean scrub: {text:?}"
    );
    assert!(
        text.contains("key=sk-isolated"),
        "credential materialization still lands under the Clean policy: {text:?}"
    );
    assert!(
        expected_home.is_dir(),
        "the node created the state home directory"
    );
    let _ = std::fs::remove_dir_all(&state_root);
}

/// The A6 runtime feedback loop, stream-json leg: a failed result frame matching the agent's
/// descriptor-declared rejection classifier (1) reaches the client as a structured
/// `Error{kind: "auth_required", family}` event, (2) stamps the node-owned rejection fact that
/// flips the derived verdict from Authenticated back to Required, and (3) a completed re-auth
/// through the `agent/<name>` flow clears the fact and restores Authenticated.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_classified_stream_json_rejection_flips_the_verdict() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        a_classified_stream_json_rejection_flips_the_verdict_impl(),
    )
    .await;
}
async fn a_classified_stream_json_rejection_flips_the_verdict_impl() {
    let (node, creds, _handle) = assemble_auth_node();
    node.agent_register(AgentEntry {
        name: "rej-agent".into(),
        recipe: AgentRecipe {
            program: Some(env!("CARGO_BIN_EXE_mock_stream_json_agent").to_string()),
            args: Vec::new(),
            env: vec![("MOCK_AGENT_AUTH_FAIL".to_string(), "1".to_string())],
            endpoint: None,
        },
        source: AgentSource::Manual,
        protocol: AgentProtocol::StreamJson,
        installed: false,
        version: None,
        capabilities: Vec::new(),
        verification: AgentVerification::NotInstalled,
        auth_descriptor: Some(AgentAuthDescriptor {
            scheme: AgentAuthScheme::ApiKeyEnv {
                var: "MOCK_AGENT_KEY".into(),
                label: "Mock agent key".into(),
            },
            rejection: Some(RejectionClassifier {
                field: "subtype".into(),
                code: "error_auth".into(),
            }),
        }),
        auth: None,
    })
    .await
    .expect("register the rejecting mock agent");
    creds
        .set("agent/rej-agent/MOCK_AGENT_KEY", "sk-stale")
        .expect("plant the (stale) agent credential");
    assert_eq!(
        catalog_entry(&node, "rej-agent").await.auth.unwrap().state,
        AgentAuthState::Authenticated,
        "credential configured -> Authenticated before the runtime disproves it"
    );

    node.profile_create(ProfileSpec {
        engine: EngineSelector::Foreign {
            agent: "rej-agent".into(),
        },
        ..ProfileSpec::new("p-rej-agent", ProviderSelector::Mock, "")
    })
    .await
    .expect("create the foreign profile");
    let session = node
        .session_create(None, Some(ProfileRef::new("p-rej-agent")))
        .await
        .expect("open a session bound to the rejecting agent");
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit a turn");
    let drained = drain_turn(&node, &session).await;
    assert!(
        drained.iter().any(|o| matches!(
            o,
            Outbound::Event(daemon_protocol::AgentEvent::Error { kind: Some(k), family: Some(f), .. })
                if k == daemon_protocol::ERROR_KIND_AUTH_REQUIRED && f == "agent/rej-agent"
        )),
        "the structured auth_required error reaches the client: {drained:?}"
    );

    assert!(
        creds.get("agent/rej-agent/rejected").is_some(),
        "the rejection fact is stamped node-side"
    );
    assert_eq!(
        catalog_entry(&node, "rej-agent").await.auth.unwrap().state,
        AgentAuthState::Required,
        "the runtime rejection refines the configured verdict back to Required"
    );

    // Re-auth through the ordinary agent/* flow: the fresh credential clears the fact.
    let begun = node
        .auth_begin(AuthBeginRequest {
            family: "agent/rej-agent".into(),
            params: Default::default(),
            redirect_uri: "http://127.0.0.1:0/cb".into(),
            bind: None,
        })
        .await
        .expect("the re-auth flow begins");
    let mut fields = std::collections::BTreeMap::new();
    fields.insert("key".to_string(), "sk-fresh".to_string());
    let result = node
        .auth_step(AuthStepRequest {
            flow_id: begun.flow_id,
            input: AuthStepInput::Fields(fields),
        })
        .await
        .expect("the pasted key completes the flow");
    assert!(matches!(result, AuthStepResult::Completed(_)));
    assert!(
        creds.get("agent/rej-agent/rejected").is_none(),
        "a completed re-auth clears the rejection fact"
    );
    assert_eq!(
        catalog_entry(&node, "rej-agent").await.auth.unwrap().state,
        AgentAuthState::Authenticated,
        "the fresh credential restores the Authenticated verdict"
    );
}

/// The A6 runtime feedback loop, ACP leg: an agent refusing `session/new` with the structured
/// `auth_required` error (-32000) surfaces the classified event, clears the now-disproven ACP
/// success marker, and flips the derived verdict back to Required.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_acp_auth_required_refusal_clears_the_marker() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        an_acp_auth_required_refusal_clears_the_marker_impl(),
    )
    .await;
}
async fn an_acp_auth_required_refusal_clears_the_marker_impl() {
    let (node, creds, _handle) = assemble_auth_node();
    node.agent_register(AgentEntry {
        name: "gated".into(),
        recipe: AgentRecipe {
            program: Some(env!("CARGO_BIN_EXE_mock_acp_auth_agent").to_string()),
            args: Vec::new(),
            env: vec![("MOCK_ACP_AUTH_GATE".to_string(), "1".to_string())],
            endpoint: None,
        },
        source: AgentSource::Manual,
        protocol: AgentProtocol::Acp,
        installed: false,
        version: None,
        capabilities: Vec::new(),
        verification: AgentVerification::NotInstalled,
        auth_descriptor: None,
        auth: None,
    })
    .await
    .expect("register the gated mock ACP agent");
    // A previously-completed node-run flow left its success marker; the runtime will disprove it.
    creds
        .set("agent/gated/acp:mock-login", "1234567890")
        .expect("plant the ACP success marker");
    assert_eq!(
        catalog_entry(&node, "gated").await.auth.unwrap().state,
        AgentAuthState::Authenticated,
        "marker present -> Authenticated before the runtime disproves it"
    );

    node.profile_create(ProfileSpec {
        engine: EngineSelector::Foreign {
            agent: "gated".into(),
        },
        ..ProfileSpec::new("p-gated", ProviderSelector::Mock, "")
    })
    .await
    .expect("create the foreign profile");
    let session = node
        .session_create(None, Some(ProfileRef::new("p-gated")))
        .await
        .expect("session open succeeds (the ACP driver fails asynchronously at session/new)");
    // Submitting materializes the live session (ensure) and spawns the driver; the refusal then
    // happens at session/new inside the driver — no turn ever opens; drain until the classified
    // error surfaces.
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit a turn");
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut seen = Vec::new();
    let classified = loop {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the classified auth_required error; drained: {seen:?}"
        );
        let items = node.poll(session.clone(), 0).await.expect("poll");
        let hit = items.iter().any(|o| {
            matches!(
                o,
                Outbound::Event(daemon_protocol::AgentEvent::Error { kind: Some(k), family: Some(f), .. })
                    if k == daemon_protocol::ERROR_KIND_AUTH_REQUIRED && f == "agent/gated"
            )
        });
        seen.extend(items);
        if hit {
            break true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    assert!(classified);
    assert!(
        creds.get("agent/gated/acp:mock-login").is_none(),
        "the disproven ACP success marker is cleared"
    );
    assert_eq!(
        catalog_entry(&node, "gated").await.auth.unwrap().state,
        AgentAuthState::Required,
        "the verdict flips back to Required"
    );
}

/// A descriptor-less row (a stream-json agent with no static claim) serves NO auth block: the
/// wire default decodes to `Unknown`, so an absent verdict can never be mistaken for a claim.
#[tokio::test]
async fn a_descriptorless_row_serves_no_auth_block() {
    let (node, _creds, _handle) = assemble_auth_node();
    node.agent_register(stream_json_entry("plain", None, None))
        .await
        .expect("registration succeeds");
    let entry = catalog_entry(&node, "plain").await;
    assert_eq!(entry.auth_descriptor, None);
    assert_eq!(entry.auth, None, "no facts -> no verdict block on the wire");
}
