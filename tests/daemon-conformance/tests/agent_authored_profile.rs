// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE PHASE-2 AGENT-AUTHORED-PROFILE GATE: an in-turn orchestrator agent authors a reusable profile
//! through the `profile_manage` tool, the write is recorded on the shared revision log as
//! `Author::Agent("profile_manage")` under the `agent/{session}/{name}` namespace, and a LATER
//! `orchestrate spawn { source: Profile(that id) }` binds and runs a child from it.
//!
//! This drives the REAL assembled node (the orchestrator profile carries both `profile_manage` and
//! `orchestrate`; the tool wraps the SAME `ProfileOps` the operator create path uses, validated by
//! the SAME node). It also proves the operator path still records `Author::Operator` and that the
//! shared validation engine rejects a foreign profile naming an unknown agent — from BOTH the
//! operator op and (by construction) the agent tool, since they share one `ProfileOps`/validator.

use std::sync::Arc;
use std::time::{Duration, Instant};

use daemon_api::{ControlApi, EngineSelector, ProfileApi, ProfileSpec, ProviderSelector};
use daemon_common::{Author, PartitionId, ProfileRef, SessionId};
use daemon_core::{
    Capabilities, Failure, MockProvider, ModelOutput, Provider, ProviderBuilder, ProviderRegistry,
    Request, ToolCall, ToolCallFormat,
};
use daemon_host::{FileRevisionLog, HostConfig, MemProfileStore, NodeApiImpl, SupervisorHandle};
use daemon_node::{assemble, AssembledNode, NodeAssembly, ProviderResolver};
use daemon_store::{InMemoryStore, SessionStatus, SessionStore};

/// A scripted orchestrator, STATELESS so it survives suspend/resume incarnation rebuilds (the
/// durable factory rebuilds the provider fresh each activation). It decides the round from the
/// conversation itself — the number of tool results present: 0 => author a profile via
/// `profile_manage`; 1 => delegate (joining) to a child bound to that authored profile via
/// `orchestrate spawn { source: Profile }`; >=2 (the resume once the child completes) => finish.
struct AuthorThenSpawnProvider {
    authored_id: String,
}

#[async_trait::async_trait]
impl Provider for AuthorThenSpawnProvider {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_native_tools: true,
            supports_streaming: false,
            tool_call_format: ToolCallFormat::Native,
            max_context: Some(8192),
        }
    }

    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
        match req.messages.iter().filter(|m| m.role == "tool").count() {
            // Round 1: author a reusable Core profile from building blocks.
            0 => Ok(ModelOutput {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "author".into(),
                    name: "profile_manage".into(),
                    args: r#"{"action":"create","name":"helper","model":"m","system_prompt":"a focused helper"}"#.into(),
                }],
                ..Default::default()
            }),
            // Round 2: delegate (joining) to a child bound to the authored profile.
            1 => Ok(ModelOutput {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "spawn".into(),
                    name: "orchestrate".into(),
                    args: format!(
                        r#"{{"verb":"spawn","source":{{"profile":"{}"}},"task":"go"}}"#,
                        self.authored_id
                    ),
                }],
                ..Default::default()
            }),
            // Round 3 (resume after the child completed and woke us): finish.
            _ => Ok(ModelOutput {
                text: "parent done".into(),
                ..Default::default()
            }),
        }
    }
}

/// Assemble a full node with a profile store + revision log (so authoring is versioned) and a scripted
/// orchestrator that authors then spawns. The provider resolver builds a completing Core child for
/// the authored profile.
fn assemble_authoring_node(
    authored_id: &str,
    revisions_dir: std::path::PathBuf,
) -> (Arc<NodeApiImpl>, Arc<dyn SessionStore>, SupervisorHandle) {
    let orchestrator: ProviderBuilder = {
        let authored_id = authored_id.to_string();
        Arc::new(move || {
            Arc::new(AuthorThenSpawnProvider {
                authored_id: authored_id.clone(),
            }) as Arc<dyn Provider>
        })
    };
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(MockProvider::completing("session done")) as Arc<dyn Provider>
    }));
    providers.register("orchestrator", orchestrator);
    // The authored Core child resolves a completing provider.
    let resolver: ProviderResolver = Arc::new(|_spec: &ProfileSpec| {
        let builder: ProviderBuilder =
            Arc::new(|| Arc::new(MockProvider::completing("child resolved")) as Arc<dyn Provider>);
        builder
    });

    let revisions: Arc<dyn daemon_common::RevisionLog> =
        Arc::new(FileRevisionLog::open(revisions_dir).unwrap());
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } = assemble(NodeAssembly {
        store: store.clone(),
        partition: PartitionId::DEFAULT,
        host_config: HostConfig::default(),
        providers,
        credentials: None,
        profile: ProfileRef::new("orchestrator"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x88; 32]),
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
        revisions: Some(revisions),
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

/// An agent authors a profile (recorded `Author::Agent`, `agent/{session}/{name}` namespace) and a
/// later joining `orchestrate spawn { source: Profile }` binds + runs a child from it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn agent_authored_profile_is_recorded_and_spawned() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        agent_authored_profile_is_recorded_and_spawned_impl(),
    )
    .await;
}

async fn agent_authored_profile_is_recorded_and_spawned_impl() {
    let parent = SessionId::new("author-parent");
    let authored_id = format!("agent/{}/helper", parent.as_str());
    // A self-cleaning temp dir (dropped at test end) backs the file revision log.
    let dir = tempfile::tempdir().unwrap();
    let (node, store, handle) = assemble_authoring_node(&authored_id, dir.path().join("revisions"));

    // Drive the durable orchestrator: it authors the profile, then delegates to a child bound to it.
    node.assign(parent.clone())
        .await
        .expect("assign the durable orchestrator parent");

    // The child materializes at the joining child id, bound to the AGENT-AUTHORED profile.
    let child = SessionId::new("author-parent/c1");
    wait_for_status(
        &store,
        &child,
        SessionStatus::Completed,
        "the authored-profile child",
    )
    .await;
    let meta = store.session_meta(&child).await.expect("child meta");
    assert_eq!(
        meta.bound_profile.as_ref().map(|p| p.as_str()),
        Some(authored_id.as_str()),
        "the spawn bound the agent-authored profile as its source"
    );

    // The parent's turn resumes and completes once the child wakes it.
    wait_for_status(&store, &parent, SessionStatus::Completed, "the parent").await;

    // The authored profile persists under the `agent/{session}/{name}` namespace...
    let spec = node
        .profile_get(authored_id.clone())
        .await
        .expect("profile_get")
        .expect("the agent-authored profile is persisted");
    assert!(authored_id.starts_with("agent/author-parent/"));
    assert_eq!(spec.system_prompt, "a focused helper");

    // ...and it is recorded as `Author::Agent("profile_manage")` on the shared revision log.
    let hist = node
        .profile_history(authored_id.clone(), None)
        .await
        .expect("profile_history");
    assert_eq!(hist.items.len(), 1, "one create revision");
    assert_eq!(
        hist.items[0].author,
        Author::Agent("profile_manage".to_string()),
        "an agent-authored profile is attributed to the tool, not the operator"
    );

    handle.shutdown().await;
}

/// The operator create path still records `Author::Operator`, and the SHARED validation engine (the
/// one `ProfileOps`/validator both the operator op and the agent tool run) rejects a foreign profile
/// naming an unknown agent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn operator_create_records_operator_and_shares_validation() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        operator_create_records_operator_and_shares_validation_impl(),
    )
    .await;
}

async fn operator_create_records_operator_and_shares_validation_impl() {
    let authored_id = "agent/unused/x";
    let dir = tempfile::tempdir().unwrap();
    let (node, _store, handle) = assemble_authoring_node(authored_id, dir.path().join("revisions"));

    // Operator create of a valid Core profile: persisted + recorded as `Author::Operator`.
    node.profile_create(ProfileSpec::new(
        "op-core",
        ProviderSelector::Mock,
        "mock-model",
    ))
    .await
    .expect("operator create of a valid Core profile");
    let hist = node
        .profile_history("op-core".into(), None)
        .await
        .expect("profile_history");
    assert_eq!(hist.items.len(), 1);
    assert_eq!(
        hist.items[0].author,
        Author::Operator,
        "the operator create path records Author::Operator through the shared ProfileOps"
    );

    // Shared validation: a foreign profile naming an unknown catalog agent is rejected at create
    // (the same `validate_engine` the agent `profile_manage` tool runs through the shared ProfileOps).
    let bad = ProfileSpec {
        engine: EngineSelector::Foreign {
            agent: "no-such-agent".into(),
        },
        ..ProfileSpec::new("op-foreign", ProviderSelector::Mock, "")
    };
    let err = node
        .profile_create(bad)
        .await
        .expect_err("an unknown foreign agent must be rejected by the shared validator");
    assert!(
        err.to_string().contains("no-such-agent"),
        "the shared validation error names the unknown agent: {err}"
    );

    handle.shutdown().await;
}
