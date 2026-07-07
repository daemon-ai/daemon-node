// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Orchestrate-tool coverage: the `spawn` verb plumbs the agent's task/lifetime/profile into the
//! delegation payload, `send` queues its `message` into an owned child (and only an owned child),
//! `status` reports per-child durable state, the serde-derived args reject unknown verbs/fields and
//! missing required fields as failed results, and the per-verb concurrency/mutation classes hold.
//! The args are strict JSON now (no `delegate` alias, no bare-`{}` label fallback, no word forms,
//! no stringly-typed `wait`).

use std::sync::Arc;

use async_trait::async_trait;
use daemon_common::{Budget, PartitionId, SessionId, SnapshotBlob, UnitId};
use daemon_core::events::EventSink;
use daemon_core::exec::LocalEnvironment;
use daemon_core::{Effect, Tool, ToolCall, ToolConcurrency, ToolOutcome, TurnCx};
use daemon_orchestration::{ChildSpawner, DefaultAnswerPolicy, FleetRuntime};
use daemon_protocol::{
    ChildSource, DelegationInput, DelegationLifetime, HostRequest, HostRequestHandler,
    HostRequestKind, HostResponse, HostResponseBody, InlineProfileSpec, UserMsg,
};
use daemon_store::{InMemoryStore, SessionMeta, SessionRole, SessionStore};
use daemon_supervision::{DelegationSpec, ManagedUnit};
use daemon_tool_orchestrate::OrchestrateTool;

/// The tool never drives the legacy synchronous placement path in these tests; the seam only has
/// to exist for `FleetRuntime` construction.
struct UnusedSpawner;

#[async_trait]
impl ChildSpawner for UnusedSpawner {
    async fn spawn(&self, _id: UnitId, _spec: &DelegationSpec) -> Arc<dyn ManagedUnit> {
        panic!("the orchestrate tool tests never spawn through the legacy sync path")
    }
}

/// A host that answers every `Delegate` with a fixed durable job id.
struct DelegatingHost;

#[async_trait]
impl HostRequestHandler for DelegatingHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
        let body = match req.kind {
            HostRequestKind::Delegate { .. } => {
                HostResponseBody::Delegated(daemon_common::JobId::new("job-1"))
            }
            _ => HostResponseBody::Approved {
                approved: true,
                allow_permanent: false,
                reason: None,
            },
        };
        HostResponse {
            request_id: req.request_id,
            body,
        }
    }
}

fn fleet(store: Arc<InMemoryStore>) -> FleetRuntime {
    FleetRuntime::new(
        store,
        PartitionId::DEFAULT,
        Arc::new(UnusedSpawner),
        Arc::new(DefaultAnswerPolicy),
        None,
    )
}

async fn run_as(tool: &dyn Tool, session: &str, args: &str) -> ToolOutcome {
    let events = EventSink::discarding();
    let exec = LocalEnvironment::sandbox("orchestrate-test");
    let host = DelegatingHost;
    let cx = TurnCx {
        cancel: tokio_util::sync::CancellationToken::new(),
        events: &events,
        host: &host,
        session_id: SessionId::new(session),
        profile: None,
        budget: Budget::unlimited(),
        exec: &exec,
        tool_result_budget: 0,
        approval_policy: daemon_core::ApprovalPolicy::AutoAllow,
        pre_approved: false,
        checkpoints: None,
        tool_timeout: None,
        session_allow: &[],
    };
    let call = ToolCall {
        call_id: "c1".into(),
        name: "orchestrate".into(),
        args: args.into(),
    };
    tool.run(&call, &cx).await
}

/// The delegation payload carried on `Effect::Delegate`, decoded.
fn delegation_payload(out: &ToolOutcome) -> DelegationInput {
    let Some(Effect::Delegate { payload, .. }) = out.effects.first() else {
        panic!("expected an Effect::Delegate, got ok={}", out.result.ok);
    };
    DelegationInput::decode(payload)
}

#[tokio::test]
async fn spawn_plumbs_task_lifetime_and_source_into_the_payload() {
    let store = Arc::new(InMemoryStore::new());
    let tool = OrchestrateTool::new(fleet(store));
    let out = run_as(
        &tool,
        "parent",
        r#"{"verb":"spawn","task":"summarize the repo","lifetime":"ephemeral","source":{"profile":"opus"},"attachments":["notes.md"]}"#,
    )
    .await;
    assert!(out.result.ok);
    assert!(out.result.content.starts_with("spawned:"));
    let input = delegation_payload(&out);
    assert_eq!(input.task, "summarize the repo");
    assert_eq!(input.lifetime, DelegationLifetime::Ephemeral);
    assert_eq!(input.source, ChildSource::Profile("opus".into()));
    assert_eq!(input.attachments, vec!["notes.md".to_string()]);

    // The joining spawn carries a `delegation-spawn` detail whose body names the durable `job`
    // handle (the child id is minted node-side later) and is flagged non-detached.
    let detail = out
        .detail
        .as_ref()
        .expect("a joining spawn carries a delegation-spawn detail");
    assert_eq!(detail.kind, "delegation-spawn");
    let body: serde_json::Value = serde_json::from_slice(&detail.body).expect("JSON body");
    assert_eq!(body["job"], "job-1");
    assert_eq!(body["detached"], false);
    assert!(body.get("child").is_none(), "joining arm has no child yet");
}

#[tokio::test]
async fn spawn_defaults_lifetime_source_and_joining_wait() {
    // A spawn with only the required `task` defaults to persistent lifetime, the default engine
    // source, and a joining (wait:true) delegation.
    let store = Arc::new(InMemoryStore::new());
    let tool = OrchestrateTool::new(fleet(store));
    let out = run_as(&tool, "parent", r#"{"verb":"spawn","task":"do the work"}"#).await;
    assert!(out.result.ok, "{}", out.result.content);
    assert!(out.result.content.starts_with("spawned:"));
    assert!(
        matches!(out.effects.first(), Some(Effect::Delegate { .. })),
        "the default (wait omitted) is a joining delegation"
    );
    let input = delegation_payload(&out);
    assert_eq!(input.task, "do the work");
    assert_eq!(input.lifetime, DelegationLifetime::Persistent);
    assert_eq!(input.source, ChildSource::Default);
    assert!(!input.detached, "the default payload is joining");
}

#[tokio::test]
async fn spawn_carries_an_inline_source_and_rejects_posture_widening() {
    let store = Arc::new(InMemoryStore::new());
    let tool = OrchestrateTool::new(fleet(store));
    // An inline sub-agent with an explicit (restricting) tool_allowlist is accepted and its spec
    // rides the delegation payload verbatim.
    let out = run_as(
        &tool,
        "parent",
        r#"{"verb":"spawn","task":"inline work","source":{"inline":{"system_prompt":"be terse","tool_allowlist":["fs"],"model":"mock-model"}}}"#,
    )
    .await;
    assert!(out.result.ok, "{}", out.result.content);
    let input = delegation_payload(&out);
    assert_eq!(
        input.source,
        ChildSource::Inline(InlineProfileSpec {
            system_prompt: "be terse".into(),
            tool_allowlist: Some(vec!["fs".into()]),
            model: "mock-model".into(),
            ..InlineProfileSpec::default()
        })
    );

    // An inline spec with NO tool_allowlist requests the full node toolset — a security-widening
    // that is operator-only, so the in-turn agent's spawn is rejected.
    let denied = run_as(
        &tool,
        "parent",
        r#"{"verb":"spawn","task":"inline work","source":{"inline":{"system_prompt":"anything"}}}"#,
    )
    .await;
    assert!(
        !denied.result.ok,
        "a posture-widening inline spec is denied"
    );
    assert!(
        denied.result.content.contains("operator-only"),
        "the decline explains the operator gate: {}",
        denied.result.content
    );
    assert!(
        denied.effects.is_empty(),
        "a denied spawn delegates nothing"
    );
}

#[tokio::test]
async fn depth_guard_still_caps_nested_spawns() {
    let store = Arc::new(InMemoryStore::new());
    let tool = OrchestrateTool::new(fleet(store)).with_max_depth(2);
    let out = run_as(&tool, "top/c1/c2", r#"{"verb":"spawn","task":"deeper"}"#).await;
    assert!(out.result.ok);
    assert_eq!(out.result.content, "depth-limit:2");
    assert!(out.effects.is_empty(), "no delegation past the cap");
    // v29: the decline also carries the structured guardrail detail a rich client renders.
    let detail = out.detail.expect("a guardrail decline carries a detail");
    assert_eq!(detail.kind, "guardrail");
    let body: serde_json::Value = serde_json::from_slice(&detail.body).expect("JSON body");
    assert_eq!(body["kind"], "depth");
    assert_eq!(body["limit"], 2);
    assert!(body["reason"].as_str().is_some_and(|r| !r.is_empty()));
}

#[tokio::test]
async fn invalid_calls_fail_with_reasons() {
    let store = Arc::new(InMemoryStore::new());
    let tool = OrchestrateTool::new(fleet(store));
    // The serde-derived args surface unknown verbs (`unknown variant`), typo'd/unknown keys
    // (`unknown field`, from `deny_unknown_fields`), missing required fields (`missing field`), and
    // wrong value types (`invalid type`) — all as failed results.
    for (args, needle) in [
        // Unknown verb tag.
        (r#"{"verb":"bogus"}"#, "unknown variant"),
        // Retired back-compat forms are now just unknown verbs.
        (r#"{"verb":"delegate","task":"t"}"#, "unknown variant"),
        // spawn requires `task`.
        (r#"{"verb":"spawn"}"#, "missing field `task`"),
        // An unknown lifetime value.
        (
            r#"{"verb":"spawn","task":"t","lifetime":"forever"}"#,
            "unknown variant",
        ),
        // deny_unknown_fields rejects a stray/typo'd key.
        (r#"{"verb":"spawn","task":"t","bogus":1}"#, "unknown field"),
        // `wait` must be a JSON bool (no stringly-typed coercion).
        (
            r#"{"verb":"spawn","task":"t","wait":"true"}"#,
            "invalid type",
        ),
        // send requires both `target` and `message`.
        (
            r#"{"verb":"send","message":"hi"}"#,
            "missing field `target`",
        ),
        (
            r#"{"verb":"send","target":"parent/c1"}"#,
            "missing field `message`",
        ),
        // send's message field is `message`, not the spawn `task`.
        (
            r#"{"verb":"send","target":"parent/c1","task":"hi"}"#,
            "unknown field",
        ),
        // cancel requires `target`.
        (r#"{"verb":"cancel"}"#, "missing field `target`"),
    ] {
        let out = run_as(&tool, "parent", args).await;
        assert!(!out.result.ok, "args {args:?} must fail");
        assert!(
            out.result.content.contains(needle),
            "args {args:?} => {}",
            out.result.content
        );
    }
}

/// Seed a durable child row + meta under `parent` so send/status have a real target.
async fn seed_child(
    store: &Arc<InMemoryStore>,
    parent: &str,
    child: &str,
    role: SessionRole,
    title: &str,
) {
    let id = SessionId::new(child);
    store
        .create_session(id.clone(), PartitionId::DEFAULT, SnapshotBlob::default())
        .await
        .unwrap();
    let meta = SessionMeta {
        role: Some(role),
        parent: Some(SessionId::new(parent)),
        title: Some(title.into()),
        bound_profile: Some(daemon_common::ProfileRef::new("opus")),
        ..SessionMeta::default()
    };
    store.set_session_meta(&id, meta).await.unwrap();
}

#[tokio::test]
async fn send_queues_input_and_wakes_an_owned_child() {
    let store = Arc::new(InMemoryStore::new());
    seed_child(
        &store,
        "parent",
        "parent/c1",
        SessionRole::ManagedChild,
        "t",
    )
    .await;
    let tool = OrchestrateTool::new(fleet(store.clone())).with_store(store.clone());

    let out = run_as(
        &tool,
        "parent",
        r#"{"verb":"send","target":"parent/c1","message":"also check the docs"}"#,
    )
    .await;
    assert!(out.result.ok, "{}", out.result.content);
    assert_eq!(out.result.content, "sent:parent/c1");

    // The message landed on the durable pending-input queue (a CBOR UserMsg)...
    let child = SessionId::new("parent/c1");
    let queued = store.take_session_inputs(&child).await;
    assert_eq!(queued.len(), 1);
    assert_eq!(UserMsg::decode(&queued[0]).text, "also check the docs");
    // ...and a wake was enqueued so the next dispatch activates the child.
    assert_eq!(store.dequeue_wake().await, Some(child));
}

#[tokio::test]
async fn send_rejects_targets_outside_the_callers_subtree() {
    let store = Arc::new(InMemoryStore::new());
    // A child of a DIFFERENT parent.
    seed_child(&store, "other", "other/c1", SessionRole::ManagedChild, "t").await;
    let tool = OrchestrateTool::new(fleet(store.clone())).with_store(store.clone());

    let out = run_as(
        &tool,
        "parent",
        r#"{"verb":"send","target":"other/c1","message":"pssst"}"#,
    )
    .await;
    assert!(!out.result.ok);
    assert!(out.result.content.contains("not a child"));
    assert!(
        store
            .take_session_inputs(&SessionId::new("other/c1"))
            .await
            .is_empty(),
        "nothing may be queued on a foreign session"
    );
}

#[tokio::test]
async fn send_authorizes_via_the_durable_parent_link_when_ids_do_not_nest() {
    let store = Arc::new(InMemoryStore::new());
    // The child's id does NOT prefix-match the caller, but its durable meta parent chain does.
    seed_child(
        &store,
        "parent",
        "detached-child",
        SessionRole::ManagedChild,
        "t",
    )
    .await;
    let tool = OrchestrateTool::new(fleet(store.clone())).with_store(store.clone());

    let out = run_as(
        &tool,
        "parent",
        r#"{"verb":"send","target":"detached-child","message":"hello"}"#,
    )
    .await;
    assert!(out.result.ok, "{}", out.result.content);
}

#[tokio::test]
async fn send_to_an_unknown_child_fails() {
    let store = Arc::new(InMemoryStore::new());
    let tool = OrchestrateTool::new(fleet(store.clone())).with_store(store);
    // Prefix-authorized but no such session row.
    let out = run_as(
        &tool,
        "parent",
        r#"{"verb":"send","target":"parent/c9","message":"hi"}"#,
    )
    .await;
    assert!(!out.result.ok);
    assert!(out.result.content.contains("unknown child"));
}

#[tokio::test]
async fn status_reports_per_child_state_from_the_durable_graph() {
    let store = Arc::new(InMemoryStore::new());
    let parent = SessionId::new("parent");
    // Two durable children bound to the parent via the delegation edge the tree walks.
    for (child, role, title) in [
        ("parent/c1", SessionRole::ManagedChild, "index the code"),
        ("parent/c2", SessionRole::EphemeralSubagent, "quick check"),
    ] {
        seed_child(&store, "parent", child, role, title).await;
        store
            .bind_delegation(
                SessionId::new(child),
                daemon_store::JobCommand {
                    job_id: daemon_common::JobId::new(format!("{child}:job")),
                    session_id: parent.clone(),
                    epoch: daemon_common::Epoch(1),
                    payload: Vec::new(),
                    lifetime: daemon_store::ChildLifetime::Persistent,
                    child: None,
                },
            )
            .await
            .unwrap();
    }
    let tool = OrchestrateTool::new(fleet(store.clone())).with_store(store.clone());

    let out = run_as(&tool, "parent", r#"{"verb":"status"}"#).await;
    assert!(out.result.ok);
    let content = &out.result.content;
    assert!(content.contains("children: 2"), "{content}");
    assert!(
        content.contains("parent/c1 [managed] ready profile=opus — index the code"),
        "{content}"
    );
    assert!(
        content.contains("parent/c2 [ephemeral] ready profile=opus — quick check"),
        "{content}"
    );

    // A target filter narrows to one child; an unknown target errors.
    let one = run_as(&tool, "parent", r#"{"verb":"status","target":"parent/c2"}"#).await;
    assert!(one.result.ok);
    assert!(one.result.content.contains("children: 1"));
    assert!(!one.result.content.contains("parent/c1 "));
    let missing = run_as(&tool, "parent", r#"{"verb":"status","target":"nope"}"#).await;
    assert!(!missing.result.ok);
}

#[tokio::test]
async fn status_without_children_or_store_still_answers() {
    let store = Arc::new(InMemoryStore::new());
    let with_store = OrchestrateTool::new(fleet(store.clone())).with_store(store.clone());
    let out = run_as(&with_store, "parent", r#"{"verb":"status"}"#).await;
    assert!(out.result.ok);
    assert_eq!(out.result.content, "no children");

    // No store wired: the legacy fleet-wide counts.
    let legacy = OrchestrateTool::new(fleet(store));
    let out = run_as(&legacy, "parent", r#"{"verb":"status"}"#).await;
    assert!(out.result.ok);
    assert!(out.result.content.starts_with("fleet: 0 children"));
}

#[tokio::test]
async fn cancel_reports_unknown_children_as_false() {
    let store = Arc::new(InMemoryStore::new());
    let tool = OrchestrateTool::new(fleet(store));
    let out = run_as(&tool, "parent", r#"{"verb":"cancel","target":"ghost"}"#).await;
    assert!(out.result.ok);
    assert_eq!(out.result.content, "cancel:ghost:false");
}

#[tokio::test]
async fn detached_spawn_enqueues_a_bare_job_and_notice_edge_without_an_effect() {
    let store = Arc::new(InMemoryStore::new());
    let tool = OrchestrateTool::new(fleet(store.clone())).with_store(store.clone());
    let out = run_as(
        &tool,
        "parent",
        r#"{"verb":"spawn","wait":false,"task":"bg work","lifetime":"ephemeral","source":{"profile":"opus"}}"#,
    )
    .await;
    assert!(out.result.ok, "{}", out.result.content);
    assert_eq!(out.result.content, "spawned-detached:parent/d1");
    assert!(
        out.effects.is_empty(),
        "a detached spawn never emits Effect::Delegate (the parent does not suspend)"
    );

    // The detached spawn carries a `delegation-spawn` detail whose body names the concrete `child`
    // session id the store minted, flagged detached.
    let detail = out
        .detail
        .as_ref()
        .expect("a detached spawn carries a delegation-spawn detail");
    assert_eq!(detail.kind, "delegation-spawn");
    let body: serde_json::Value = serde_json::from_slice(&detail.body).expect("JSON body");
    assert_eq!(body["child"], "parent/d1");
    assert_eq!(body["detached"], true);

    // The bare job landed on the durable outbox with the store-minted child and detached payload.
    let job = store
        .dequeue_job()
        .await
        .expect("a detached job on the outbox");
    assert_eq!(job.child.as_ref().map(|c| c.as_str()), Some("parent/d1"));
    let input = DelegationInput::decode(&job.payload);
    assert_eq!(input.task, "bg work");
    assert!(input.detached, "the payload carries detached=true");
    assert_eq!(input.lifetime, DelegationLifetime::Ephemeral);
    assert_eq!(input.source, ChildSource::Profile("opus".into()));

    // The completion-notice edge makes the child tree-visible under the parent (not a delegation).
    assert!(
        store
            .children_of(&SessionId::new("parent"))
            .await
            .contains(&SessionId::new("parent/d1")),
        "the detached child shows up in the parent's child index"
    );
}

#[tokio::test]
async fn wait_true_and_default_stay_joining_delegations() {
    let store = Arc::new(InMemoryStore::new());
    let tool = OrchestrateTool::new(fleet(store));
    // Explicit wait:true and the default (wait omitted) both emit a joining Effect::Delegate with a
    // non-detached payload; only a JSON `wait:false` bool detaches.
    for args in [
        r#"{"verb":"spawn","wait":true,"task":"t"}"#,
        r#"{"verb":"spawn","task":"t"}"#,
    ] {
        let out = run_as(&tool, "parent", args).await;
        assert!(out.result.ok, "args {args:?}: {}", out.result.content);
        assert!(
            out.result.content.starts_with("spawned:"),
            "args {args:?} => {}",
            out.result.content
        );
        assert!(
            matches!(out.effects.first(), Some(Effect::Delegate { .. })),
            "args {args:?} must still delegate"
        );
        assert!(
            !delegation_payload(&out).detached,
            "args {args:?} payload must be joining"
        );
    }
}

#[tokio::test]
async fn detached_spawn_requires_a_durable_store() {
    let store = Arc::new(InMemoryStore::new());
    let tool = OrchestrateTool::new(fleet(store)); // no store wired
    let out = run_as(
        &tool,
        "parent",
        r#"{"verb":"spawn","wait":false,"task":"t"}"#,
    )
    .await;
    assert!(!out.result.ok);
    assert!(
        out.result
            .content
            .contains("detached spawn requires a durable session store"),
        "{}",
        out.result.content
    );
}

#[tokio::test]
async fn detached_fanout_cap_declines_at_the_limit() {
    let store = Arc::new(InMemoryStore::new());
    let tool = OrchestrateTool::new(fleet(store.clone()))
        .with_store(store.clone())
        .with_max_fanout(2);
    // The first two detached spawns succeed (they count as active children even before materializing).
    for i in 1..=2 {
        let out = run_as(
            &tool,
            "parent",
            r#"{"verb":"spawn","wait":false,"task":"x"}"#,
        )
        .await;
        assert!(out.result.ok);
        assert_eq!(out.result.content, format!("spawned-detached:parent/d{i}"));
    }
    // The third is declined at the cap (mirroring the depth guard: ok result, no job).
    let capped = run_as(
        &tool,
        "parent",
        r#"{"verb":"spawn","wait":false,"task":"x"}"#,
    )
    .await;
    assert!(capped.result.ok);
    assert_eq!(capped.result.content, "fanout-limit:2");
    assert!(capped.effects.is_empty());
    // v29: the decline also carries the structured guardrail detail a rich client renders.
    let detail = capped
        .detail
        .as_ref()
        .expect("a guardrail decline carries a detail");
    assert_eq!(detail.kind, "guardrail");
    let body: serde_json::Value = serde_json::from_slice(&detail.body).expect("JSON body");
    assert_eq!(body["kind"], "fanout");
    assert_eq!(body["limit"], 2);
    // Only two jobs were enqueued.
    assert!(store.dequeue_job().await.is_some());
    assert!(store.dequeue_job().await.is_some());
    assert!(
        store.dequeue_job().await.is_none(),
        "the capped spawn enqueued nothing"
    );
}

#[test]
fn per_verb_concurrency_and_mutation_classes() {
    let call = |args: &str| ToolCall {
        call_id: "c".into(),
        name: "orchestrate".into(),
        args: args.into(),
    };
    let store = Arc::new(InMemoryStore::new());
    let tool = OrchestrateTool::new(fleet(store));
    // status: read-only, batch-parallel.
    assert_eq!(
        tool.concurrency_for(&call(r#"{"verb":"status"}"#)),
        ToolConcurrency::Parallel
    );
    assert!(!tool.mutates_for(&call(r#"{"verb":"status"}"#)));
    // spawn/send/cancel: exclusive + mutating.
    for args in [
        r#"{"verb":"spawn","task":"t"}"#,
        r#"{"verb":"send","target":"parent/c1","message":"t"}"#,
        r#"{"verb":"cancel","target":"parent/c1"}"#,
    ] {
        assert_eq!(
            tool.concurrency_for(&call(args)),
            ToolConcurrency::Exclusive,
            "{args}"
        );
        assert!(tool.mutates_for(&call(args)), "{args}");
    }
}
