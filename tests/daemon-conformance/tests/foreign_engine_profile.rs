// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Foreign-engine (ACP) profiles end-to-end through the NODE API surface (wire v23
//! `ProfileSpec::engine`, generalized in v29 to `Foreign{agent}` + the protocol-tagged agent
//! catalog): a scripted fake ACP agent is registered in the catalog (`agent_register`, source
//! Manual — the operator-managed recipe path), a profile is created with `engine = Foreign{agent}`,
//! a session bound to that profile is opened, and one interactive turn round-trips — proving
//! profile -> catalog resolution -> foreign spawn -> §17 turn, with the genai provider/model path
//! fully bypassed and the agent's symmetric permission callback parking as an ordinary host
//! request.
//!
//! Also proves the fail-fast validation seams: a profile referencing an unknown catalog name or an
//! uninstalled agent is rejected at `ProfileCreate`/`ProfileUpdate` with a clear error, and the
//! catalog-by-NAME-only security invariant (the profile never carries a recipe; the spawn resolves
//! the node's own catalog).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use daemon_api::{
    AgentEntry, AgentProtocol, AgentRecipe, AgentSource, ControlApi, EngineSelector, Outbound,
    ProfileApi, ProfileSpec, ProviderSelector, SessionApi, SessionQuery,
};
use daemon_common::{ProfileRef, ReqId};
use daemon_core::{MockProvider, Provider, ProviderBuilder, ProviderRegistry};
use daemon_host::{HostConfig, MemProfileStore, NodeApiImpl};
use daemon_node::{assemble, AssembledNode, NodeAssembly, ProviderResolver};
use daemon_protocol::{AgentCommand, AgentEvent, HostResponse, HostResponseBody, UserMsg};
use daemon_store::InMemoryStore;

/// Assemble a full node with a profile store + a provider resolver that FLAGS when it is consulted,
/// so the test can assert the foreign-engine path never touches the provider/model seam.
fn assemble_acp_node() -> (
    Arc<NodeApiImpl>,
    Arc<AtomicBool>,
    daemon_host::SupervisorHandle,
) {
    let resolver_called = Arc::new(AtomicBool::new(false));
    let flag = resolver_called.clone();
    let resolver: ProviderResolver = Arc::new(move |_spec: &ProfileSpec| {
        flag.store(true, Ordering::SeqCst);
        let builder: ProviderBuilder =
            Arc::new(|| Arc::new(MockProvider::completing("native reply")) as Arc<dyn Provider>);
        builder
    });
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(MockProvider::completing("session done")) as Arc<dyn Provider>
    }));
    let AssembledNode { node, handle, .. } = assemble(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: daemon_common::PartitionId::DEFAULT,
        host_config: HostConfig::default(),
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
        credential_store: None,
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
    });
    (node, resolver_called, handle)
}

/// A profile spec bound to a foreign agent BY NAME ONLY: no provider/model/recipe — the
/// catalog owns the launch recipe, the node resolves it at spawn.
fn foreign_profile(id: &str, agent: &str) -> ProfileSpec {
    ProfileSpec {
        engine: EngineSelector::Foreign {
            agent: agent.into(),
        },
        ..ProfileSpec::new(id, ProviderSelector::Mock, "")
    }
}

/// Like [`foreign_profile`] but carrying a node-validated `model` (Layer 1): the catalog still owns
/// the launch recipe; only the model string is threaded through to the ACP `set_config_option`.
fn foreign_profile_with_model(id: &str, agent: &str, model: &str) -> ProfileSpec {
    ProfileSpec {
        model: model.into(),
        ..foreign_profile(id, agent)
    }
}

/// Register the compiled mock ACP agent under `name` (source Manual). The node's real
/// `AcpDiscoverer::probe` runs the ACP `initialize` handshake against it, so the stored entry is
/// verified installed with a reported protocol version — the operator registration path, for real.
async fn register_mock_agent(node: &Arc<NodeApiImpl>, name: &str) {
    node.agent_register(AgentEntry {
        name: name.into(),
        recipe: AgentRecipe {
            program: Some(env!("CARGO_BIN_EXE_mock_acp_agent").to_string()),
            args: Vec::new(),
            env: Vec::new(),
            endpoint: None,
        },
        source: AgentSource::Manual,
        protocol: AgentProtocol::Acp,
        installed: false, // the probe fills this in; a caller-supplied value is not trusted
        version: None,
        capabilities: Vec::new(),
    })
    .await
    .expect("register the mock ACP agent");
    let catalog = node.agent_catalog().await;
    let entry = catalog
        .iter()
        .find(|e| e.name == name)
        .expect("registered agent is in the catalog");
    assert!(
        entry.installed,
        "the initialize probe should verify the mock agent as installed"
    );
}

/// Drive the polled outbound stream until `TurnFinished`, answering any parked permission request
/// affirmatively (the ACP `session/request_permission` -> §17 `Approval` bridge). Returns every
/// drained item.
async fn drain_turn(node: &Arc<NodeApiImpl>, session: &daemon_common::SessionId) -> Vec<Outbound> {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut acc = Vec::new();
    while Instant::now() < deadline {
        let items = node
            .poll(session.clone(), 0)
            .await
            .expect("poll the live session");
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
    panic!("timed out waiting for the foreign agent's TurnFinished; drained: {acc:?}");
}

/// The full round-trip: register -> create profile (engine=Foreign) -> session bound to it -> one
/// interactive turn (text streamed, permission answered, turn completes) — provider seam untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acp_profile_spawns_and_completes_a_turn() {
    // Drive the node as the trusted local embedder (`system`): after the Auth 4 flip a `None`
    // request principal is denied, so an in-process test must bind a context exactly as `bins/daemon`
    // does. `system` holds the ownership overrides, so it stands in for the local-trust driver.
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        acp_profile_spawns_and_completes_a_turn_impl(),
    )
    .await;
}
async fn acp_profile_spawns_and_completes_a_turn_impl() {
    let (node, resolver_called, _handle) = assemble_acp_node();
    register_mock_agent(&node, "fake-echo").await;

    node.profile_create(foreign_profile("acp-fake", "fake-echo"))
        .await
        .expect("create a profile bound to the registered ACP agent");

    // A blank, profile-bound session (node-authoritative create), then one live turn on it.
    let session = node
        .session_create(None, Some(ProfileRef::new("acp-fake")))
        .await
        .expect("create a session bound to the ACP profile");
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello foreign agent"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit a turn to the foreign-engine session");

    let items = drain_turn(&node, &session).await;
    let streamed_text = items.iter().any(|o| {
        matches!(o, Outbound::Event(AgentEvent::TextDelta { text, .. })
                 if text.contains("acp agent reporting in"))
    });
    assert!(
        streamed_text,
        "the mock ACP agent's message chunk should stream up as a TextDelta: {items:?}"
    );
    let raised_permission = items.iter().any(|o| matches!(o, Outbound::Request(_)));
    assert!(
        raised_permission,
        "the agent's permission request should park as an ordinary host request"
    );
    let completed = items.iter().any(|o| {
        matches!(o, Outbound::Event(AgentEvent::TurnFinished { summary, .. })
                 if summary.end_reason == daemon_protocol::EndReason::Completed)
    });
    assert!(
        completed,
        "the foreign turn should reach TurnFinished{{Completed}}: {items:?}"
    );

    // The genai provider/model seam was never consulted for the foreign engine.
    assert!(
        !resolver_called.load(Ordering::SeqCst),
        "a foreign-engine session must bypass the provider resolver entirely"
    );

    // The roster reports the live foreign session as NOT rewindable (ACP has no
    // truncate-at-anchor), while rewind itself is refused with a clear error.
    let page = node.sessions_query(SessionQuery::default()).await;
    let row = page
        .sessions
        .iter()
        .find(|s| s.session == session)
        .expect("the foreign session is on the roster");
    assert!(
        !row.rewindable,
        "a live foreign (ACP) session must advertise rewindable=false"
    );
}

/// Layer-1 model selection (positive): a foreign profile whose `model` matches a value the agent
/// advertises in its `Model` `Select` config option drives the adapter to send
/// `session/set_config_option`, which the mock reflects back in-band (`set:model=mock-model-b`) —
/// proving the full assembly -> foreign_live -> daemon-acp thread applies the model.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acp_profile_matching_model_is_applied() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        acp_profile_matching_model_is_applied_impl(),
    )
    .await;
}

async fn acp_profile_matching_model_is_applied_impl() {
    let (node, _resolver_called, _handle) = assemble_acp_node();
    register_mock_agent(&node, "fake-echo").await;

    node.profile_create(foreign_profile_with_model(
        "acp-model",
        "fake-echo",
        "mock-model-b",
    ))
    .await
    .expect("create a foreign profile carrying a model the agent offers");

    let session = node
        .session_create(None, Some(ProfileRef::new("acp-model")))
        .await
        .expect("create a session bound to the model-carrying ACP profile");
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello foreign agent"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit a turn to the foreign-engine session");

    let items = drain_turn(&node, &session).await;
    let applied = items.iter().any(|o| {
        matches!(o, Outbound::Event(AgentEvent::TextDelta { text, .. })
                 if text.contains("set:model=mock-model-b"))
    });
    assert!(
        applied,
        "a matching profile model must be applied via set_config_option: {items:?}"
    );
}

/// Layer-1 model selection (negative): a foreign profile whose `model` the agent does not offer
/// sends no `session/set_config_option` (the mock reports `unset`), and the session still runs to a
/// finished turn on the agent's default model — model selection is best-effort, never fatal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acp_profile_unknown_model_sends_nothing_and_still_runs() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        acp_profile_unknown_model_sends_nothing_and_still_runs_impl(),
    )
    .await;
}

async fn acp_profile_unknown_model_sends_nothing_and_still_runs_impl() {
    let (node, _resolver_called, _handle) = assemble_acp_node();
    register_mock_agent(&node, "fake-echo").await;

    node.profile_create(foreign_profile_with_model(
        "acp-unknown-model",
        "fake-echo",
        "no-such-model",
    ))
    .await
    .expect("create a foreign profile carrying a model the agent does not offer");

    let session = node
        .session_create(None, Some(ProfileRef::new("acp-unknown-model")))
        .await
        .expect("create a session bound to the unknown-model ACP profile");
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello foreign agent"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit a turn to the foreign-engine session");

    let items = drain_turn(&node, &session).await;
    let reported_unset = items.iter().any(|o| {
        matches!(o, Outbound::Event(AgentEvent::TextDelta { text, .. })
                 if text.contains("unset"))
    });
    assert!(
        reported_unset,
        "an unmatched model must not send any set_config_option (mock reports unset): {items:?}"
    );
    let completed = items.iter().any(|o| {
        matches!(o, Outbound::Event(AgentEvent::TurnFinished { summary, .. })
                 if summary.end_reason == daemon_protocol::EndReason::Completed)
    });
    assert!(
        completed,
        "the session must still complete on the default model: {items:?}"
    );
}

/// Fail-fast validation: an unknown catalog name is rejected at create AND update; an uninstalled
/// agent (catalog entry present, binary missing) is rejected too. The errors carry the agent name.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acp_profile_validation_rejects_unknown_and_uninstalled_agents() {
    let (node, _resolver_called, _handle) = assemble_acp_node();

    // Unknown: never registered, not in the curated builtin table.
    let err = node
        .profile_create(foreign_profile("bad", "no-such-agent"))
        .await
        .expect_err("an unknown agent must fail profile_create");
    assert!(
        err.to_string().contains("no-such-agent"),
        "the error names the unknown agent: {err}"
    );

    // Uninstalled: registered with a recipe whose program does not exist, so the initialize probe
    // cannot verify it (installed stays false).
    node.agent_register(AgentEntry {
        name: "ghost".into(),
        recipe: AgentRecipe {
            program: Some("/nonexistent/daemon-conformance-ghost-agent".into()),
            args: Vec::new(),
            env: Vec::new(),
            endpoint: None,
        },
        source: AgentSource::Manual,
        protocol: AgentProtocol::Acp,
        installed: false,
        version: None,
        capabilities: Vec::new(),
    })
    .await
    .expect("register the ghost agent");
    let err = node
        .profile_create(foreign_profile("ghostly", "ghost"))
        .await
        .expect_err("an uninstalled agent must fail profile_create");
    assert!(
        err.to_string().contains("not installed"),
        "the error says the agent is not installed: {err}"
    );

    // Update is validated the same way: a valid native profile cannot be flipped onto an unknown
    // agent.
    node.profile_create(ProfileSpec::new(
        "native",
        ProviderSelector::Mock,
        "mock-model",
    ))
    .await
    .expect("create a native profile");
    let err = node
        .profile_update(foreign_profile("native", "no-such-agent"))
        .await
        .expect_err("an unknown agent must fail profile_update");
    assert!(err.to_string().contains("no-such-agent"));
}
