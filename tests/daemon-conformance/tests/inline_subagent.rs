// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE PHASE-1 INLINE-SUBAGENT GATE: an `orchestrate spawn { source: Inline { … } }` materializes a
//! transient sub-agent from an AD-HOC spec (persona/toolset/engine) with NO saved profile, and the
//! child runs with that inline config on the real durable path.
//!
//! Three cases against the REAL assembled node (dispatching engine factory + durable job worker +
//! the ephemeral reaper):
//!
//! - a **Core** inline sub-agent (custom `system_prompt` + a restricted `tool_allowlist`): the
//!   durable resolver rebuilds the child's engine from the persisted inline `ProfileSpec` — proven
//!   by capturing the exact spec the provider resolver is handed at resolution — and the child
//!   (ephemeral) is reaped (archived) after it completes;
//! - a **Foreign** inline sub-agent (`engine = Foreign { agent }`): the child runs as its ACP agent
//!   via the `ForeignIncarnation` (its journaled transcript carries the ACP agent's own output);
//! - a **posture-widening** inline spec (no `tool_allowlist` = the full node toolset): the in-turn
//!   agent's spawn is REJECTED (operator-only), so no child is ever materialized.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use daemon_api::{
    AgentEntry, AgentProtocol, AgentRecipe, AgentSource, AgentVerification, ControlApi, ProfileSpec,
};
use daemon_common::{JournalStreamId, PartitionId, ProfileRef, SessionId};
use daemon_core::{
    Capabilities, Failure, MockProvider, ModelOutput, Provider, ProviderBuilder, ProviderRegistry,
    Request, ToolCall, ToolCallFormat,
};
use daemon_host::{HostConfig, MemProfileStore, NodeApiImpl, SupervisorHandle};
use daemon_node::{assemble, AssembledNode, NodeAssembly, ProviderResolver, ReaperConfig};
use daemon_store::{InMemoryStore, SessionStatus, SessionStore};

/// A Core orchestrator provider that emits ONE `orchestrate spawn` carrying the caller-supplied
/// `source` JSON (a `{"inline":{…}}` or `{"profile":…}` object) + an ephemeral lifetime, then
/// completes on the resume. The scripted parent driving an inline-delegation cycle.
struct SpawnSourceProvider {
    /// The JSON value for the spawn's `source` field.
    source_json: String,
}

#[async_trait::async_trait]
impl Provider for SpawnSourceProvider {
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
            return Ok(ModelOutput {
                text: "parent done".into(),
                ..Default::default()
            });
        }
        Ok(ModelOutput {
            text: String::new(),
            tool_calls: vec![ToolCall {
                call_id: "spawn-inline".into(),
                name: "orchestrate".into(),
                args: format!(
                    r#"{{"verb":"spawn","lifetime":"ephemeral","source":{},"task":"do the inline work"}}"#,
                    self.source_json
                ),
            }],
            ..Default::default()
        })
    }
}

/// The captured `(id, system_prompt, tool_allowlist)` the provider resolver is handed — the proof an
/// inline sub-agent's spec reached engine resolution.
type CapturedSpecs = Arc<Mutex<Vec<(String, String, Option<Vec<String>>)>>>;

/// Assemble a full node wired for inline delegation: a profile store + a capturing provider resolver
/// (so the dispatching factory + Core resolution are active) and an `orchestrator` provider that
/// spawns an inline sub-agent from `source_json`. The reaper is set to a short grace/interval so an
/// ephemeral child is archived promptly after it completes.
fn assemble_inline_node(
    source_json: &str,
) -> (
    Arc<NodeApiImpl>,
    Arc<dyn SessionStore>,
    SupervisorHandle,
    CapturedSpecs,
) {
    let orchestrator: ProviderBuilder = {
        let source_json = source_json.to_string();
        Arc::new(move || {
            Arc::new(SpawnSourceProvider {
                source_json: source_json.clone(),
            }) as Arc<dyn Provider>
        })
    };
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(MockProvider::completing("session done")) as Arc<dyn Provider>
    }));
    providers.register("orchestrator", orchestrator);

    // The capturing resolver: records every spec it is handed (the inline child's spec carries the
    // ad-hoc persona + allowlist), and returns a completing mock so a Core child finishes its turn.
    let captured: CapturedSpecs = Arc::new(Mutex::new(Vec::new()));
    let resolver: ProviderResolver = {
        let captured = captured.clone();
        Arc::new(move |spec: &ProfileSpec| {
            captured.lock().unwrap().push((
                spec.id.clone(),
                spec.system_prompt.clone(),
                spec.tool_allowlist.clone(),
            ));
            let builder: ProviderBuilder =
                Arc::new(|| Arc::new(MockProvider::completing("inline done")) as Arc<dyn Provider>);
            builder
        })
    };

    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } = assemble(NodeAssembly {
        store: store.clone(),
        partition: PartitionId::DEFAULT,
        host_config: HostConfig::default(),
        providers,
        credentials: None,
        profile: ProfileRef::new("orchestrator"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x91; 32]),
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
        // A short grace/interval so an ephemeral child is reaped soon after it completes.
        reaper: ReaperConfig {
            enabled: true,
            grace: Duration::from_millis(10),
            interval: Duration::from_millis(40),
        },
        orchestrate: Default::default(),
        foreign_gateway: None,
    });
    (node, store, handle, captured)
}

/// Register the compiled mock ACP agent under `name` (source Manual), verified installed by the
/// node's real `AcpDiscoverer::probe`.
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
        verification: AgentVerification::NotInstalled, // untrusted; the node re-derives on register
    })
    .await
    .expect("register the mock ACP agent");
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

/// A Core inline sub-agent runs with its ad-hoc persona + restricted toolset (no saved profile) and
/// is reaped once it completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn core_inline_subagent_runs_with_inline_config_and_is_reaped() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        core_inline_subagent_runs_with_inline_config_and_is_reaped_impl(),
    )
    .await;
}

async fn core_inline_subagent_runs_with_inline_config_and_is_reaped_impl() {
    // A Core inline sub-agent: a custom persona + a restricted single-tool allowlist, no engine
    // (defaults to Core), no saved profile.
    let source = r#"{"inline":{"system_prompt":"you are a haiku bot","tool_allowlist":["fs"],"model":"mock-model"}}"#;
    let (node, store, handle, captured) = assemble_inline_node(source);

    let parent = SessionId::new("orch-parent");
    node.assign(parent.clone())
        .await
        .expect("assign the durable orchestrator parent");

    let child = SessionId::new("orch-parent/c1");
    wait_for_status(&store, &child, SessionStatus::Completed, "the inline child").await;

    // The child bound NO profile (inline is `bound_profile = None`) but persisted the inline spec.
    let meta = store.session_meta(&child).await.expect("child meta");
    assert!(
        meta.bound_profile.is_none(),
        "an inline sub-agent binds no saved profile"
    );
    assert!(
        !meta.inline_profile.is_empty(),
        "the inline spec is persisted on the child (for durable re-resolution)"
    );
    assert_eq!(
        meta.role,
        Some(daemon_store::SessionRole::EphemeralSubagent),
        "an ephemeral lifetime yields an EphemeralSubagent (the reaped role)"
    );

    // Positive proof it ran with the INLINE config: the provider resolver was handed a spec keyed by
    // the child's id carrying the inline persona + the restricted allowlist (the durable resolver
    // rebuilt the engine from the persisted inline spec).
    let saw_inline = captured.lock().unwrap().iter().any(|(id, prompt, allow)| {
        id == child.as_str()
            && prompt == "you are a haiku bot"
            && allow.as_deref() == Some(&["fs".to_string()][..])
    });
    assert!(
        saw_inline,
        "the inline persona + restricted allowlist reached engine resolution; captured: {:?}",
        captured.lock().unwrap()
    );

    // Ephemeral reaping: the completed ephemeral child is archived after the (short) grace elapses.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if store.session_meta(&child).await.is_some_and(|m| m.archived) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the ephemeral inline child to be reaped (archived)"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    handle.shutdown().await;
}

/// A Foreign inline sub-agent (`engine = Foreign { agent }`, no saved profile) runs as its ACP agent
/// on the durable path — its journaled transcript carries the ACP agent's own output.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreign_inline_subagent_runs_as_acp() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        foreign_inline_subagent_runs_as_acp_impl(),
    )
    .await;
}

async fn foreign_inline_subagent_runs_as_acp_impl() {
    // A Foreign inline sub-agent: engine names the ACP agent by catalog name; an explicit (empty)
    // allowlist keeps it out of the posture-widening gate (a foreign agent uses its own tools).
    let source = r#"{"inline":{"engine":{"Foreign":{"agent":"fake-echo"}},"tool_allowlist":[]}}"#;
    let (node, store, handle, _captured) = assemble_inline_node(source);
    register_mock_agent(&node, "fake-echo").await;

    let parent = SessionId::new("orch-parent");
    node.assign(parent.clone())
        .await
        .expect("assign the durable orchestrator parent");

    let child = SessionId::new("orch-parent/c1");
    wait_for_status(
        &store,
        &child,
        SessionStatus::Completed,
        "the foreign inline child",
    )
    .await;

    // The child bound NO profile and persisted the inline (Foreign) spec.
    let meta = store.session_meta(&child).await.expect("child meta");
    assert!(
        meta.bound_profile.is_none(),
        "inline binds no saved profile"
    );
    assert!(!meta.inline_profile.is_empty(), "inline spec persisted");

    // Positive proof it ran as ACP (not a Core fallback): the child's sealed transcript carries the
    // mock ACP agent's own streamed output — routed via the ForeignIncarnation from the inline spec.
    let seg = store
        .load_trace_segment(&JournalStreamId::session(&child), 0)
        .await
        .expect("the foreign inline child journaled a sealed segment");
    let printable: String = seg
        .entries
        .iter()
        .flat_map(|e| e.bytes.iter().copied())
        .filter(|b| (0x20..0x7f).contains(b))
        .map(|b| b as char)
        .collect();
    assert!(
        printable.contains("acp agent reporting in"),
        "the inline foreign child's transcript must carry the ACP agent's output (proving foreign \
         execution from an inline spec)"
    );

    handle.shutdown().await;
}

/// A posture-widening inline spec (no `tool_allowlist` = the full node toolset) is operator-only, so
/// an in-turn agent's spawn is rejected — no child is ever materialized.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn posture_widening_inline_spec_is_rejected() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        posture_widening_inline_spec_is_rejected_impl(),
    )
    .await;
}

async fn posture_widening_inline_spec_is_rejected_impl() {
    // No tool_allowlist -> the full node toolset -> a security-widening only an operator may grant.
    let source = r#"{"inline":{"system_prompt":"unrestricted"}}"#;
    let (node, store, handle, _captured) = assemble_inline_node(source);

    let parent = SessionId::new("orch-parent");
    node.assign(parent.clone())
        .await
        .expect("assign the durable orchestrator parent");

    // The tool rejects the widening spawn, so the parent never suspends — it completes without
    // delegating. Wait for the parent to complete, then assert no child was materialized.
    wait_for_status(&store, &parent, SessionStatus::Completed, "the parent").await;
    let child = SessionId::new("orch-parent/c1");
    assert!(
        store.status(&child).await.is_none(),
        "a rejected posture-widening inline spawn materializes no child"
    );
    assert!(
        store.children_of(&parent).await.is_empty(),
        "no child edge is recorded for a rejected inline spawn"
    );

    handle.shutdown().await;
}
