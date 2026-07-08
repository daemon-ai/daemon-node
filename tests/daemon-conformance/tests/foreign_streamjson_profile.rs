// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Foreign-engine STREAM-JSON profiles end-to-end through the NODE API surface (wire v29
//! `AgentProtocol::StreamJson`): a scripted fake Claude-Code-style stream-json agent is registered
//! in the catalog (`agent_register`, `protocol = StreamJson` — probed installed-on-PATH only, no
//! `initialize` handshake), a profile is created with `engine = Foreign{agent}`, a session bound to
//! that profile is opened, and one interactive turn round-trips — proving the live profile spawn
//! resolves the catalog entry's PROTOCOL and drives the NDJSON child through the generic
//! `CodecSession` + `StreamJsonCodec`, with the agent's `control_request` parking as an ordinary
//! host approval request and the genai provider/model path fully bypassed.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use daemon_api::{
    AgentEntry, AgentProtocol, AgentRecipe, AgentSource, AgentVerification, ControlApi,
    EngineSelector, Outbound, ProfileApi, ProfileSpec, ProviderSelector, SessionApi,
};
use daemon_common::{ProfileRef, ReqId};
use daemon_core::{MockProvider, Provider, ProviderBuilder, ProviderRegistry};
use daemon_host::{HostConfig, MemProfileStore, NodeApiImpl};
use daemon_node::{assemble, AssembledNode, NodeAssembly, ProviderResolver};
use daemon_protocol::{AgentCommand, AgentEvent, HostResponse, HostResponseBody, UserMsg};
use daemon_store::InMemoryStore;

/// Assemble a full node with a profile store + a provider resolver that FLAGS when it is consulted,
/// so the test can assert the foreign-engine path never touches the provider/model seam.
fn assemble_node() -> (
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
        journal_seed: Some([0x67; 32]),
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

/// Drive the polled outbound stream until `TurnFinished`, answering any parked permission request
/// affirmatively (the stream-json `control_request` -> §17 `Approval` bridge). Returns every
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
    panic!("timed out waiting for the stream-json agent's TurnFinished; drained: {acc:?}");
}

/// The full round-trip: register (protocol=StreamJson) -> create profile (engine=Foreign) ->
/// session bound to it -> one interactive turn (permission answered, text streamed, turn
/// completes) — provider seam untouched, no `initialize` metadata on the entry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streamjson_profile_spawns_and_completes_a_turn() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        streamjson_profile_spawns_and_completes_a_turn_impl(),
    )
    .await;
}
async fn streamjson_profile_spawns_and_completes_a_turn_impl() {
    let (node, resolver_called, _handle) = assemble_node();

    // Register the compiled mock stream-json agent (source Manual, protocol StreamJson). The
    // probe is a PATH/file presence check ONLY — no `initialize` handshake exists for this
    // protocol, so version/capabilities must stay empty.
    node.agent_register(AgentEntry {
        name: "fake-claude".into(),
        recipe: AgentRecipe {
            program: Some(env!("CARGO_BIN_EXE_mock_stream_json_agent").to_string()),
            args: Vec::new(),
            env: Vec::new(),
            endpoint: None,
        },
        source: AgentSource::Manual,
        protocol: AgentProtocol::StreamJson,
        installed: false, // the probe fills this in; a caller-supplied value is not trusted
        version: None,
        capabilities: Vec::new(),
        verification: AgentVerification::NotInstalled, // untrusted; the node re-derives on register
    })
    .await
    .expect("register the mock stream-json agent");
    let catalog = node.agent_catalog().await;
    let entry = catalog
        .iter()
        .find(|e| e.name == "fake-claude")
        .expect("registered agent is in the catalog");
    assert!(
        entry.installed,
        "the PATH probe should verify the mock agent as installed"
    );
    assert_eq!(entry.protocol, AgentProtocol::StreamJson);
    assert!(
        entry.version.is_none() && entry.capabilities.is_empty(),
        "a stream-json entry has no initialize handshake, so version/capabilities stay empty: \
         {entry:?}"
    );

    node.profile_create(ProfileSpec {
        engine: EngineSelector::Foreign {
            agent: "fake-claude".into(),
        },
        ..ProfileSpec::new("sj-fake", ProviderSelector::Mock, "")
    })
    .await
    .expect("create a profile bound to the registered stream-json agent");

    // A blank, profile-bound session (node-authoritative create), then one live turn on it.
    let session = node
        .session_create(None, Some(ProfileRef::new("sj-fake")))
        .await
        .expect("create a session bound to the stream-json profile");
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello stream-json agent"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit a turn to the foreign-engine session");

    let items = drain_turn(&node, &session).await;
    let raised_permission = items.iter().any(|o| matches!(o, Outbound::Request(_)));
    assert!(
        raised_permission,
        "the agent's control_request should park as an ordinary host approval request: {items:?}"
    );
    let streamed_text = items.iter().any(|o| {
        matches!(o, Outbound::Event(AgentEvent::TextDelta { text, .. })
                 if text.contains("stream-json agent reporting in"))
    });
    assert!(
        streamed_text,
        "the mock agent's message should stream up as a TextDelta: {items:?}"
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
}
