// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE PHASE-3 PROVENANCE + CAPS GATE: an in-turn orchestrator agent composes profiles through the
//! `profile_manage` tool on the REAL assembled node, and:
//!
//! - the node's `[orchestrate].max_composed_profiles` cap declines the N+1-th `create` tool-side
//!   (the composed-profiles guardrail), so only N profiles persist;
//! - each authored profile carries PROVENANCE — `created_by = Author::Agent("profile_manage")` and
//!   `owner = {authoring session}` — surfaced on both `ProfileGet` (the spec) and `ProfileList`
//!   (the redacted `ProfileInfo`);
//! - an operator sees every profile via `ProfileList` and may `ProfileDelete` an agent-authored one
//!   (the operator surface is un-scoped, unlike the subtree-scoped agent tool).

use std::sync::Arc;
use std::time::{Duration, Instant};

use daemon_api::{ControlApi, NodeEvent, ProfileApi, ProfileSpec};
use daemon_common::{Author, PartitionId, ProfileRef, SessionId};
use daemon_core::{
    Capabilities, Failure, MockProvider, ModelOutput, Provider, ProviderBuilder, ProviderRegistry,
    Request, ToolCall, ToolCallFormat,
};
use daemon_host::{FileRevisionLog, HostConfig, MemProfileStore, NodeApiImpl, SupervisorHandle};
use daemon_node::{assemble, AssembledNode, NodeAssembly, OrchestrateCaps, ProviderResolver};
use daemon_store::{InMemoryStore, SessionStatus, SessionStore};

/// A stateless orchestrator that authors one profile per ReAct iteration (deciding the round from
/// the tool-result count in the conversation) until it has issued `authors` create calls, then
/// finishes. Stateless so it survives suspend/resume incarnation rebuilds.
struct ComposeProvider {
    authors: usize,
}

#[async_trait::async_trait]
impl Provider for ComposeProvider {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_native_tools: true,
            supports_streaming: false,
            tool_call_format: ToolCallFormat::Native,
            max_context: Some(8192),
        }
    }

    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
        let n = req.messages.iter().filter(|m| m.role == "tool").count();
        if n < self.authors {
            Ok(ModelOutput {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: format!("author-{n}"),
                    name: "profile_manage".into(),
                    args: format!(
                        r#"{{"action":"create","name":"p{n}","model":"m","persona":"helper {n}"}}"#
                    ),
                }],
                ..Default::default()
            })
        } else {
            Ok(ModelOutput {
                text: "composed".into(),
                ..Default::default()
            })
        }
    }
}

/// Assemble a full node with a profile store + revision log and a scripted orchestrator that authors
/// `authors` profiles, under a composed-profiles cap of `max_composed`.
fn assemble_compose_node(
    authors: usize,
    max_composed: usize,
    revisions_dir: std::path::PathBuf,
) -> (Arc<NodeApiImpl>, Arc<dyn SessionStore>, SupervisorHandle) {
    let orchestrator: ProviderBuilder =
        Arc::new(move || Arc::new(ComposeProvider { authors }) as Arc<dyn Provider>);
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(MockProvider::completing("session done")) as Arc<dyn Provider>
    }));
    providers.register("orchestrator", orchestrator);
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
        journal_seed: Some([0x99; 32]),
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
        orchestrate: OrchestrateCaps {
            max_composed_profiles: max_composed,
            ..Default::default()
        },
        foreign_gateway: None,
        prompt: Default::default(),
    });
    (node, store, handle)
}

async fn wait_for_status(store: &Arc<dyn SessionStore>, session: &SessionId, want: SessionStatus) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if store.status(session).await.as_ref() == Some(&want) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {session} to reach {want:?}; got {:?}",
            store.status(session).await
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn agent_profiles_carry_provenance_and_respect_the_composed_cap() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        agent_profiles_carry_provenance_and_respect_the_composed_cap_impl(),
    )
    .await;
}

async fn agent_profiles_carry_provenance_and_respect_the_composed_cap_impl() {
    let parent = SessionId::new("compose-parent");
    // The agent tries to author 3 profiles; the cap of 2 declines the third tool-side.
    let dir = tempfile::tempdir().unwrap();
    let (node, store, handle) = assemble_compose_node(3, 2, dir.path().join("revisions"));

    node.assign(parent.clone())
        .await
        .expect("assign the durable orchestrator parent");
    wait_for_status(&store, &parent, SessionStatus::Completed).await;

    // Exactly TWO profiles persisted (the third `create` hit the composed-profiles cap tool-side).
    let p0 = format!("agent/{}/p0", parent.as_str());
    let p1 = format!("agent/{}/p1", parent.as_str());
    let p2 = format!("agent/{}/p2", parent.as_str());
    assert!(
        node.profile_get(p0.clone()).await.unwrap().is_some(),
        "the first authored profile persists"
    );
    assert!(
        node.profile_get(p1.clone()).await.unwrap().is_some(),
        "the second authored profile persists"
    );
    assert!(
        node.profile_get(p2.clone()).await.unwrap().is_none(),
        "the third `create` was declined by the composed-profiles cap"
    );

    // PROVENANCE on the spec (ProfileGet): agent-authored, owned by the authoring session.
    let spec = node.profile_get(p0.clone()).await.unwrap().unwrap();
    assert_eq!(
        spec.created_by,
        Some(Author::Agent("profile_manage".into()))
    );
    assert_eq!(spec.owner.as_deref(), Some(parent.as_str()));

    // PROVENANCE on the listing view (ProfileList): an operator sees every profile, provenance and
    // all — the operator surface is un-scoped.
    let infos = node.profile_list().await;
    let info = infos
        .iter()
        .find(|i| i.id == p0)
        .expect("the agent profile is visible to the operator");
    assert_eq!(
        info.created_by,
        Some(Author::Agent("profile_manage".into()))
    );
    assert_eq!(info.owner.as_deref(), Some(parent.as_str()));

    // The node emitted a `ProfilesChanged` pointer so a thin client refetches the profile list
    // (the node-authoritative GUI/TUI surface — no client-side domain logic).
    let page = node.events_page(0, 64).await;
    assert!(
        page.events
            .iter()
            .any(|e| matches!(e, NodeEvent::ProfilesChanged { .. })),
        "authoring a profile must emit a ProfilesChanged pointer: {:?}",
        page.events
    );

    // An operator may DELETE an agent-authored profile (no subtree scoping on the operator surface).
    node.profile_delete(p1.clone())
        .await
        .expect("operator deletes an agent-authored profile");
    assert!(
        node.profile_get(p1).await.unwrap().is_none(),
        "the operator delete removed the agent profile"
    );

    handle.shutdown().await;
}
