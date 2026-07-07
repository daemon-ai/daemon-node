// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Approval-fingerprint management (wire v29, D4) end-to-end through the NODE API surface:
//! an operator's "Allow permanently" records the exec-approval command fingerprint on the durable
//! session allow-list; `FingerprintList` surfaces it; `FingerprintRevoke` drops it from the
//! dormant snapshot; and — the acceptance bar — the next IDENTICAL command RE-PROMPTS (parks a
//! fresh approval), proven with no test backdoors: every step drives the real wire ops and the
//! real engine gate.
//!
//! Script (one durable session, tool-result count N drives the provider):
//!   N=0: shell A  -> parks #1 (Ask policy) -> ApprovalDecide(allow, allow_permanent) remembers fpA
//!   N=1: shell A  -> auto-approved by the remembered fingerprint (no park)
//!   N=2: shell B  -> parks #2 (different argv = different fingerprint) — the session now sits
//!                    DORMANT (Suspended) with fpA on its allow-list: list -> [fpA]; revoke fpA
//!   N=3: shell A  -> after ApprovalDecide(#2 allow) resumes the turn, the SAME command that was
//!                    permanently allowed now RE-PROMPTS: parks #3 with fingerprint fpA.

use std::sync::Arc;
use std::time::{Duration, Instant};

use daemon_api::{from_cbor, to_cbor, ApiRequest, ApprovalMode, ControlApi, ProfileApi};
use daemon_api::{ProfileSpec, ProviderSelector, RememberedFingerprint};
use daemon_common::{PartitionId, ProfileRef, SessionId, UsageDelta};
use daemon_core::{
    Capabilities, Failure, ModelOutput, Provider, ProviderBuilder, ProviderRegistry, Request,
    ToolCall, ToolCallFormat,
};
use daemon_host::{HostConfig, MemProfileStore, NodeApiImpl};
use daemon_node::{assemble, AssembledNode, NodeAssembly, ProviderResolver};
use daemon_store::{InMemoryStore, SessionStore};

// Background shell-string commands: the ALWAYS-GATED §12 surface (benign foreground argv runs
// unattended even under `Ask`; the background/pty shell-string tier never rides a fast path), so
// the script parks deterministically and each distinct line owns a distinct fingerprint.
const CMD_A: &str = r#"{"command":"printf","args":["fp-%s","alpha"],"background":true}"#;
const CMD_B: &str = r#"{"command":"printf","args":["fp-%s","beta"],"background":true}"#;

/// A conversation-keyed scripted provider: the number of tool results already in the conversation
/// picks the next action, so it is correct across durable incarnations (like a real model, it only
/// ever sees the conversation).
struct FingerprintScript;

#[async_trait::async_trait]
impl Provider for FingerprintScript {
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
        let n = req.messages.iter().filter(|m| m.role == "tool").count();
        let args = match n {
            0 | 1 => Some(CMD_A),
            2 => Some(CMD_B),
            3 => Some(CMD_A),
            _ => None,
        };
        match args {
            Some(args) => Ok(ModelOutput {
                text: String::new(),
                reasoning: None,
                tool_calls: vec![ToolCall {
                    call_id: format!("call-{n}"),
                    name: "shell".into(),
                    args: args.into(),
                }],
                usage,
                ..Default::default()
            }),
            None => Ok(ModelOutput {
                text: "done".into(),
                reasoning: None,
                tool_calls: Vec::new(),
                usage,
                ..Default::default()
            }),
        }
    }
}

fn assemble_node(
    store: Arc<dyn SessionStore>,
) -> (Arc<NodeApiImpl>, daemon_host::SupervisorHandle) {
    let resolver: ProviderResolver = Arc::new(move |_spec: &ProfileSpec| {
        let builder: ProviderBuilder =
            Arc::new(|| Arc::new(FingerprintScript) as Arc<dyn Provider>);
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
        journal_seed: Some([0x45; 32]),
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
    (node, handle)
}

/// Poll the pending-approval inbox until exactly one entry is parked, and return it.
async fn wait_for_park(
    node: &Arc<NodeApiImpl>,
    session: &SessionId,
    _store_dbg: &Arc<dyn SessionStore>,
) -> daemon_api::ApprovalInfo {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let pending = node
            .approvals_pending(Some(session.clone()), None)
            .await
            .items;
        if let Some(first) = pending.first() {
            return first.clone();
        }
        if Instant::now() >= deadline {
            panic!(
                "timed out waiting for a parked approval; status={:?} snapshot_turns={:?}",
                daemon_store::SessionStore::status(&**_store_dbg, session).await,
                _store_dbg
                    .peek_snapshot(session)
                    .await
                    .and_then(|b| daemon_core::Snapshot::decode(&b).ok())
                    .map(|s| format!("{:?}", s.conversation)),
            );
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// The wire shapes round-trip (the new ops + the list DTO).
#[test]
fn fingerprint_ops_round_trip() {
    let reqs = [
        ApiRequest::FingerprintList {
            session: SessionId::new("s1"),
        },
        ApiRequest::FingerprintRevoke {
            session: SessionId::new("s1"),
            fingerprint: "ab12".into(),
        },
    ];
    for req in reqs {
        assert_eq!(req, from_cbor::<ApiRequest>(&to_cbor(&req)).unwrap());
    }
    let dto = RememberedFingerprint {
        fingerprint: "ab12".into(),
        label: None,
        remembered_at_ms: 0,
    };
    assert_eq!(
        dto,
        from_cbor::<RememberedFingerprint>(&to_cbor(&dto)).unwrap()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remembered_fingerprint_lists_revokes_and_reprompts() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        remembered_fingerprint_lists_revokes_and_reprompts_impl(),
    )
    .await;
}
async fn remembered_fingerprint_lists_revokes_and_reprompts_impl() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let (node, _handle) = assemble_node(store.clone());

    // A durable profile-bound session narrowed to Ask (the HITL posture) — stamped exactly as the
    // node's own child-creation paths do.
    node.profile_create(ProfileSpec::new(
        "fp-script",
        ProviderSelector::Mock,
        "mock-model",
    ))
    .await
    .expect("create profile");
    let session = SessionId::new("fp-manage-1");
    let blob = daemon_core::Snapshot::fresh(session.clone())
        .encode()
        .expect("encode fresh snapshot");
    store
        .create_session(session.clone(), PartitionId::DEFAULT, blob)
        .await
        .expect("create session");
    let mut meta = store.session_meta(&session).await.unwrap_or_default();
    meta.bound_profile = Some(ProfileRef::new("fp-script"));
    meta.overlay = daemon_host::encode_overlay(&daemon_api::SessionOverlay {
        approval_mode: Some(ApprovalMode::Ask),
        ..Default::default()
    });
    store
        .set_session_meta(&session, meta)
        .await
        .expect("bind profile + Ask overlay");

    // Park #1: command A gates under Ask; the parked row carries its resolved fingerprint.
    node.assign(session.clone()).await.expect("assign");
    let park1 = wait_for_park(&node, &session, &store).await;
    let fp_a = park1
        .fingerprint
        .clone()
        .expect("a shell approval parks with its command fingerprint");

    // Allow PERMANENTLY: the engine re-runs A, remembers fpA, auto-approves the second identical A
    // (no park), then command B (different argv => different fingerprint) parks #2.
    node.approval_decide(session.clone(), park1.request_id.clone(), true, true, None)
        .await
        .expect("allow permanently");
    let park2 = wait_for_park(&node, &session, &store).await;
    assert_ne!(
        park2.request_id, park1.request_id,
        "park #2 is a fresh approval (command B)"
    );
    let fp_b = park2
        .fingerprint
        .clone()
        .expect("command B parks with its fingerprint");
    assert_ne!(fp_a, fp_b, "different argv => different fingerprint");

    // The session is now DORMANT (suspended on #2) with fpA remembered: list shows exactly it.
    let listed = node
        .fingerprint_list(session.clone())
        .await
        .expect("fingerprint_list");
    assert_eq!(
        listed.len(),
        1,
        "the allow-list holds exactly the permanently-allowed fingerprint"
    );
    assert_eq!(listed[0].fingerprint, fp_a);
    assert_eq!(listed[0].label, None);
    // Provenance (wire v30): the remembered-at timestamp is stamped at the decide path.
    assert!(
        listed[0].remembered_at_ms > 0,
        "provenance timestamp is captured"
    );

    // Revoke it (the session is dormant, so the compare-and-swap applies cleanly), and prove the
    // list is empty afterwards. A second revoke of the same fingerprint errors (nothing to drop).
    node.fingerprint_revoke(session.clone(), fp_a.clone())
        .await
        .expect("revoke fpA");
    assert!(
        node.fingerprint_list(session.clone())
            .await
            .expect("fingerprint_list after revoke")
            .is_empty(),
        "the revoked fingerprint left the allow-list"
    );
    assert!(
        node.fingerprint_revoke(session.clone(), fp_a.clone())
            .await
            .is_err(),
        "revoking an absent fingerprint reports an error"
    );

    // Resume by answering #2 (single allow): the turn re-runs B, then issues the SAME command A —
    // which must now RE-PROMPT (park #3 carrying fpA) instead of auto-approving.
    node.approval_decide(session.clone(), park2.request_id.clone(), true, false, None)
        .await
        .expect("allow #2");
    let park3 = wait_for_park(&node, &session, &store).await;
    assert_ne!(park3.request_id, park2.request_id);
    assert_eq!(
        park3.fingerprint.as_deref(),
        Some(fp_a.as_str()),
        "the revoked command re-prompts with the same fingerprint"
    );
}
