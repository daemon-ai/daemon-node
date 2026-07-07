// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE PHASE-0 FOREIGN-DELEGATION GATE: a delegated child bound to a `Foreign{agent}` profile runs
//! on the DURABLE activation path as its ACP agent — not as a silent Core fallback — and its
//! terminal completion wakes the suspended parent through the ordinary delegation/join protocol.
//!
//! This drives the REAL assembled node (dispatching engine factory + `ForeignIncarnation` + durable
//! job worker + wake dispatcher): a Core orchestrator parent delegates (joining) a child whose
//! `orchestrate spawn { profile }` names a foreign profile, the job worker seeds the child on the
//! foreign path (empty snapshot + task on the durable input seam), the `ForeignIncarnation` spawns
//! the mock ACP agent and runs the turn to completion, and the child's `mark_completed` wakes the
//! parent to completion. The child's journaled transcript carries the ACP agent's own output, which
//! a Core fallback could never produce — the positive proof it ran as ACP.

use std::sync::Arc;
use std::time::{Duration, Instant};

use daemon_api::{
    AgentEntry, AgentProtocol, AgentRecipe, AgentSource, ControlApi, EngineSelector, ProfileApi,
    ProfileSpec, ProviderSelector,
};
use daemon_common::{JournalStreamId, PartitionId, ProfileRef, SessionId};
use daemon_core::{
    Capabilities, Failure, MockProvider, ModelOutput, Provider, ProviderBuilder, ProviderRegistry,
    Request, ToolCall, ToolCallFormat,
};
use daemon_host::{HostConfig, MemProfileStore, NodeApiImpl, SupervisorHandle};
use daemon_node::{assemble, AssembledNode, NodeAssembly, ProviderResolver};
use daemon_store::{InMemoryStore, SessionStatus, SessionStore};

/// A Core orchestrator provider that delegates (joining) exactly once to a child bound to
/// `profile`, then completes on the resume once the child's result is folded back in. This is the
/// scripted parent driving the foreign-delegation cycle.
struct DelegateToProfileProvider {
    profile: String,
}

#[async_trait::async_trait]
impl Provider for DelegateToProfileProvider {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_native_tools: true,
            supports_streaming: false,
            tool_call_format: ToolCallFormat::Native,
            max_context: Some(8192),
        }
    }

    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
        if req.has_tool_result() {
            // The resume after the foreign child completed and woke us: finish.
            return Ok(ModelOutput {
                text: "parent done".into(),
                ..Default::default()
            });
        }
        // First round: a joining `orchestrate spawn` naming the foreign profile.
        Ok(ModelOutput {
            text: String::new(),
            tool_calls: vec![ToolCall {
                call_id: "spawn-foreign".into(),
                name: "orchestrate".into(),
                args: format!(
                    r#"{{"verb":"spawn","source":{{"profile":"{}"}},"task":"do the foreign work"}}"#,
                    self.profile
                ),
            }],
            ..Default::default()
        })
    }
}

/// Assemble a full node wired for foreign delegation: a profile store + a provider resolver (so the
/// dispatching engine factory + foreign path are active) and an `orchestrator` provider that
/// delegates to `child_profile`. The provider resolver is only consulted for Core bound-profile
/// sessions (never the foreign child, which bypasses the genai seam entirely).
fn assemble_foreign_delegation_node(
    child_profile: &str,
) -> (Arc<NodeApiImpl>, Arc<dyn SessionStore>, SupervisorHandle) {
    let orchestrator: ProviderBuilder = {
        let profile = child_profile.to_string();
        Arc::new(move || {
            Arc::new(DelegateToProfileProvider {
                profile: profile.clone(),
            }) as Arc<dyn Provider>
        })
    };
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(MockProvider::completing("session done")) as Arc<dyn Provider>
    }));
    providers.register("orchestrator", orchestrator.clone());
    providers.register("child", orchestrator);
    // A completing resolver (satisfies the profiles-path requirement; the foreign child never uses it).
    let resolver: ProviderResolver = Arc::new(|_spec: &ProfileSpec| {
        let builder: ProviderBuilder =
            Arc::new(|| Arc::new(MockProvider::completing("resolved")) as Arc<dyn Provider>);
        builder
    });

    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } = assemble(NodeAssembly {
        store: store.clone(),
        partition: PartitionId::DEFAULT,
        host_config: HostConfig::default(),
        providers,
        credentials: None,
        profile: ProfileRef::new("orchestrator"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x77; 32]),
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
    (node, store, handle)
}

/// Register the compiled mock ACP agent under `name` (source Manual). The node's real
/// `AcpDiscoverer::probe` runs the ACP `initialize` handshake, so the stored entry is verified
/// installed — the operator registration path, for real.
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
        installed: false,
        version: None,
        capabilities: Vec::new(),
    })
    .await
    .expect("register the mock ACP agent");
}

/// A profile bound to a foreign agent by NAME ONLY (no provider/model/recipe).
fn foreign_profile(id: &str, agent: &str) -> ProfileSpec {
    ProfileSpec {
        engine: EngineSelector::Foreign {
            agent: agent.into(),
        },
        ..ProfileSpec::new(id, ProviderSelector::Mock, "")
    }
}

async fn wait_for_status(
    store: &Arc<dyn SessionStore>,
    session: &SessionId,
    want: SessionStatus,
    what: &str,
) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if store.status(session).await.as_ref() == Some(&want) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {what} ({session}) to reach {want:?}; got {:?}",
            store.status(session).await
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// A joining delegation to a `Foreign{agent}` profile runs the child as its ACP agent on the durable
/// path and wakes the suspended parent to completion.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreign_delegated_child_runs_as_acp_and_wakes_parent() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        foreign_delegated_child_runs_as_acp_and_wakes_parent_impl(),
    )
    .await;
}

async fn foreign_delegated_child_runs_as_acp_and_wakes_parent_impl() {
    let (node, store, handle) = assemble_foreign_delegation_node("acp-fake");
    register_mock_agent(&node, "fake-echo").await;
    node.profile_create(foreign_profile("acp-fake", "fake-echo"))
        .await
        .expect("create a profile bound to the registered ACP agent");

    // Drive the durable orchestrator parent: it delegates (joining) to the foreign profile and
    // suspends; the resident services materialize + run the foreign child and wake the parent.
    let parent = SessionId::new("orch-parent");
    node.assign(parent.clone())
        .await
        .expect("assign the durable orchestrator parent");

    // The child materializes at the joining child id and completes as an ACP agent.
    let child = SessionId::new("orch-parent/c1");
    wait_for_status(
        &store,
        &child,
        SessionStatus::Completed,
        "the foreign child",
    )
    .await;

    // It bound the foreign profile (the durable resolver's key for the dispatching factory).
    let meta = store.session_meta(&child).await.expect("child meta");
    assert_eq!(
        meta.bound_profile.as_ref().map(|p| p.as_str()),
        Some("acp-fake"),
        "the child bound the foreign profile"
    );

    // Positive proof it ran as ACP (not a silent Core fallback): the child's sealed transcript
    // carries the mock ACP agent's own streamed output — a Core child could never produce it.
    let seg = store
        .load_trace_segment(&JournalStreamId::session(&child), 0)
        .await
        .expect("the foreign child journaled a sealed segment");
    // The coalesced message block frames each character with a control byte, so keep only printable
    // ASCII across all entries and search that: the ACP agent's text run survives contiguously
    // (a Core child would journal "resolved"/"session done", never the ACP agent's own words).
    let printable: String = seg
        .entries
        .iter()
        .flat_map(|e| e.bytes.iter().copied())
        .filter(|b| (0x20..0x7f).contains(b))
        .map(|b| b as char)
        .collect();
    assert!(
        printable.contains("acp agent reporting in"),
        "the child's transcript must carry the ACP agent's output (proving foreign execution)"
    );

    // The join protocol fired end-to-end: the child's completion woke the parent to completion.
    wait_for_status(&store, &parent, SessionStatus::Completed, "the parent").await;

    handle.shutdown().await;
}
