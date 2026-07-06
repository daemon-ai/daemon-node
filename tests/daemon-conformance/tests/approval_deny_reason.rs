// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Deny-with-reason (wire v29, E3) end-to-end through the NODE API surface: a durable session
//! parks a gated fs write (§12 HITL), the operator denies it via `ApprovalDecide` WITH a reason,
//! and the woken engine injects that reason into the agent's conversation as the gated tool's
//! error content — asserted from the MODEL's side (a recording provider proves the next request
//! it receives carries the operator's words), which is exactly the "model can adapt its next
//! attempt" contract.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use daemon_api::{
    from_cbor, to_cbor, ApiRequest, ApprovalMode, ControlApi, ProfileApi, ProfileSpec,
    ProviderSelector,
};
use daemon_common::{PartitionId, ProfileRef, SessionId, UsageDelta};
use daemon_core::{
    Capabilities, Failure, ModelOutput, Provider, ProviderBuilder, ProviderRegistry, Request,
    ToolCall, ToolCallFormat,
};
use daemon_host::{HostConfig, MemProfileStore, NodeApiImpl};
use daemon_node::{assemble, AssembledNode, NodeAssembly, ProviderResolver};
use daemon_store::{InMemoryStore, SessionStatus, SessionStore};

/// A conversation-aware deterministic provider: it emits a single gated fs `write` until the
/// conversation carries a tool result, then records the tool-result content it was shown and
/// completes. The recording IS the assertion surface: whatever lands here is what a real model
/// would see on its next attempt.
struct WriteThenRecord {
    seen_tool_results: Arc<Mutex<Vec<String>>>,
}

#[async_trait::async_trait]
impl Provider for WriteThenRecord {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_native_tools: true,
            supports_streaming: false,
            tool_call_format: ToolCallFormat::Native,
            max_context: Some(8192),
        }
    }

    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
        let usage = UsageDelta {
            input_tokens: 8,
            output_tokens: 4,
            api_calls: 1,
            ..Default::default()
        };
        if req.has_tool_result() {
            let mut seen = self.seen_tool_results.lock().unwrap();
            seen.extend(
                req.messages
                    .iter()
                    .filter(|m| m.role == "tool")
                    .map(|m| m.content.clone()),
            );
            Ok(ModelOutput {
                text: "adapting after the deny".into(),
                reasoning: None,
                tool_calls: Vec::new(),
                usage,
                ..Default::default()
            })
        } else {
            Ok(ModelOutput {
                text: String::new(),
                reasoning: None,
                tool_calls: vec![ToolCall {
                    call_id: "call-0".into(),
                    name: "fs".into(),
                    args: r#"{"op":"write","path":"gated.txt","content":"hi"}"#.into(),
                }],
                usage,
                ..Default::default()
            })
        }
    }
}

/// Assemble a full node with a profile store + a provider resolver that materializes the recording
/// provider for profile-bound sessions. The durable HITL park needs the per-session resolution
/// path: autonomous durable engines force `AutoAllow` by design, so the gate comes from a
/// profile-bound session whose overlay selects `Ask` (the operator-narrowed posture).
fn assemble_recording_node(
    store: Arc<dyn SessionStore>,
    seen: Arc<Mutex<Vec<String>>>,
) -> (Arc<NodeApiImpl>, daemon_host::SupervisorHandle) {
    let resolver: ProviderResolver = Arc::new(move |_spec: &ProfileSpec| {
        let seen = seen.clone();
        let builder: ProviderBuilder = Arc::new(move || {
            Arc::new(WriteThenRecord {
                seen_tool_results: seen.clone(),
            }) as Arc<dyn Provider>
        });
        builder
    });
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(daemon_core::MockProvider::completing("session done")) as Arc<dyn Provider>
    }));
    let AssembledNode { node, handle, .. } = assemble(NodeAssembly {
        store,
        partition: PartitionId::DEFAULT,
        host_config: HostConfig::default(),
        providers,
        credentials: None,
        profile: ProfileRef::new("default"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x44; 32]),
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
    });
    (node, handle)
}

/// The wire shape round-trips: `ApprovalDecide` carries the optional operator reason (absent
/// encodings still decode — the field is `#[serde(default)]`).
#[test]
fn approval_decide_reason_round_trips() {
    let req = ApiRequest::ApprovalDecide {
        session: SessionId::new("s1"),
        request_id: "r1".into(),
        allow: false,
        allow_permanent: false,
        reason: Some("wrong branch — rebase first".into()),
    };
    assert_eq!(req, from_cbor::<ApiRequest>(&to_cbor(&req)).unwrap());
}

/// The full durable deny-with-reason cycle: park -> `ApprovalDecide { allow: false, reason }` ->
/// wake -> the engine splices `denied ... : {reason}` into the gated tool's result -> the model's
/// NEXT request carries the operator's words.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deny_reason_reaches_the_models_next_request() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        deny_reason_reaches_the_models_next_request_impl(),
    )
    .await;
}
async fn deny_reason_reaches_the_models_next_request_impl() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (node, _handle) = assemble_recording_node(store.clone(), seen.clone());

    // A DURABLE profile-bound session whose overlay narrows the posture to `Ask` — the durable
    // HITL shape (autonomous durable engines otherwise force AutoAllow so they never gate on a
    // human). Stamp the binding + overlay exactly as the node's own child-creation paths do
    // (`session_create` claims the LIVE lifecycle, which would conflict with the durable drive).
    node.profile_create(ProfileSpec::new(
        "recorder",
        ProviderSelector::Mock,
        "mock-model",
    ))
    .await
    .expect("create the recording profile");
    let session = SessionId::new("deny-reason-1");
    let blob = daemon_core::Snapshot::fresh(session.clone())
        .encode()
        .expect("encode fresh snapshot");
    store
        .create_session(session.clone(), PartitionId::DEFAULT, blob)
        .await
        .expect("create the durable session");
    let mut meta = store.session_meta(&session).await.unwrap_or_default();
    meta.bound_profile = Some(ProfileRef::new("recorder"));
    meta.overlay = daemon_host::encode_overlay(&daemon_api::SessionOverlay {
        approval_mode: Some(ApprovalMode::Ask),
        ..Default::default()
    });
    store
        .set_session_meta(&session, meta)
        .await
        .expect("bind profile + Ask overlay");

    // Durable drive: wake; the scripted fs write gates under Ask and parks.
    node.assign(session.clone()).await.expect("assign");
    let deadline = Instant::now() + Duration::from_secs(10);
    let request_id = loop {
        let pending = node
            .approvals_pending(Some(session.clone()), None)
            .await
            .items;
        if let Some(first) = pending.first() {
            break first.request_id.clone();
        }
        if Instant::now() >= deadline {
            panic!(
                "timed out waiting for the parked approval; status={:?} store_pending={:?}",
                store.status(&session).await,
                store.pending_approvals_of(Some(&session)).await,
            );
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    };

    // Operator denies WITH a reason over the wire op.
    let reason = "gated.txt is generated — write to scratch/notes.txt instead";
    node.approval_decide(
        session.clone(),
        request_id.clone(),
        false,
        false,
        Some(reason.into()),
    )
    .await
    .expect("deny with reason");

    // The session resumes and completes (the deny never strands it).
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if store.status(&session).await == Some(SessionStatus::Completed) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the denied session to complete"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    // The model's next request carried the operator's words in the gated tool's result slot.
    let seen = seen.lock().unwrap();
    let denial = seen
        .iter()
        .find(|content| content.contains("denied"))
        .unwrap_or_else(|| panic!("no denial tool-result reached the provider: {seen:?}"));
    assert!(
        denial.contains(reason),
        "the operator's reason must reach the model verbatim: {denial:?}"
    );
    assert!(
        denial.contains(&format!("(request {request_id})")),
        "the denial still names the request id: {denial:?}"
    );
}
