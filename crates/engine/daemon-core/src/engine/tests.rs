use super::support::*;
use super::*;
use crate::provider::{Capabilities, ModelOutput, Request};
use daemon_common::{CredScope, ReqId, UsageDelta};
use std::sync::atomic::{AtomicU64, Ordering};

/// A provider that fails the first call with a rotatable error, then completes.
struct FlakyProvider {
    calls: AtomicU64,
}

#[async_trait::async_trait]
impl Provider for FlakyProvider {
    fn capabilities(&self) -> Capabilities {
        test_caps()
    }

    async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
        let n = self.calls.fetch_add(1, Ordering::Relaxed);
        if n == 0 {
            Err(Failure::Rotatable("quota exceeded (429)".into()))
        } else {
            Ok(ok_output("done after rotation"))
        }
    }
}

/// A rotatable provider failure rotates the credential and retries on a second key — the turn
/// completes, the provider was called twice, and one key is now cooling down.
#[tokio::test]
async fn rotatable_failure_rotates_credential_and_retries() {
    let provider = Arc::new(FlakyProvider {
        calls: AtomicU64::new(0),
    });
    let pool = Arc::new(EmbeddedCredentialPool::new(
        "openai",
        CredScope::new(["openai"], ["chat"], None),
        [
            ("key-a".to_string(), "secret-a".to_string()),
            ("key-b".to_string(), "secret-b".to_string()),
        ],
        60_000,
        30_000,
    ));
    let mut engine = Engine::fresh(
        SessionId::new("rotate"),
        SystemPrompt::new("test"),
        provider.clone(),
        Arc::new(ToolRegistry::new()),
    )
    .with_credentials(pool.clone(), ProfileRef::new("openai"));
    engine.push_user(UserMsg::new("hello"));

    let outcome = engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .expect("turn completes after a single rotation");
    assert!(matches!(outcome, TurnOutcome::Completed(_)));
    assert_eq!(provider.calls.load(Ordering::Relaxed), 2, "retried once");
    assert_eq!(pool.live_count(), 1, "the rotated key is cooling down");
}

/// A provider that records the system prompt of the request it receives, then completes.
struct SystemRecordingProvider {
    seen: std::sync::Mutex<String>,
}

#[async_trait::async_trait]
impl Provider for SystemRecordingProvider {
    fn capabilities(&self) -> Capabilities {
        test_caps()
    }
    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
        *self.seen.lock().unwrap() = req.system.clone();
        Ok(ok_output("ok"))
    }
}

/// A stable-tier source emitting a fixed block.
struct FixedBlock(&'static str);
impl crate::context::StablePromptSource for FixedBlock {
    fn block(&self) -> Option<String> {
        Some(self.0.to_string())
    }
}

/// A registered [`StablePromptSource`] is folded into the request's system preamble each turn.
#[tokio::test]
async fn prompt_source_block_is_injected_into_system() {
    let provider = Arc::new(SystemRecordingProvider {
        seen: std::sync::Mutex::new(String::new()),
    });
    let mut engine = Engine::fresh(
        SessionId::new("ps"),
        SystemPrompt::new("base system"),
        provider.clone(),
        Arc::new(ToolRegistry::new()),
    )
    .with_prompt_sources(vec![Arc::new(FixedBlock(
        "<available_skills>\n  x\n</available_skills>",
    ))]);
    engine.push_user(UserMsg::new("hi"));
    engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .unwrap();
    let seen = provider.seen.lock().unwrap().clone();
    assert!(seen.contains("base system"), "keeps the base system prompt");
    assert!(
        seen.contains("<available_skills>"),
        "folds the stable block in"
    );
}

/// With `memory_review_interval = 1`, a completed turn (no memory write) fires exactly one
/// fire-and-forget `memory_review` spawn and the turn still completes normally (no suspend).
#[tokio::test]
async fn memory_review_nudge_emits_spawn_on_threshold() {
    let host = SpawnRecordingHost::default();
    let mut engine = Engine::fresh(
        SessionId::new("nudge"),
        SystemPrompt::new("test"),
        Arc::new(TextProvider),
        Arc::new(ToolRegistry::new()),
    )
    .with_config(Config {
        memory_review_interval: 1,
        ..Config::default()
    });
    engine.push_user(UserMsg::new("hi"));

    let outcome = engine
        .run_turn(&host, &EventSink::discarding(), &TurnControl::new())
        .await
        .expect("turn completes");
    assert!(matches!(outcome, TurnOutcome::Completed(_)));
    assert!(
        engine.snapshot().waiting_for.is_empty(),
        "spawn does not suspend"
    );
    assert_eq!(
        *host.spawns.lock().unwrap(),
        vec!["memory_review".to_string()],
    );
    assert_eq!(engine.snapshot().turns_since_memory, 0, "counter reset");
}

/// The default `0` intervals disable the engine-native trigger entirely.
#[tokio::test]
async fn review_nudges_disabled_by_default() {
    let host = SpawnRecordingHost::default();
    let mut engine = Engine::fresh(
        SessionId::new("no-nudge"),
        SystemPrompt::new("test"),
        Arc::new(TextProvider),
        Arc::new(ToolRegistry::new()),
    );
    engine.push_user(UserMsg::new("hi"));
    engine
        .run_turn(&host, &EventSink::discarding(), &TurnControl::new())
        .await
        .expect("turn completes");
    assert!(
        host.spawns.lock().unwrap().is_empty(),
        "no spawns when disabled"
    );
}

/// A credential provider serving two profiles with distinct secrets, recording each acquisition
/// — so a test can observe the engine hopping from the primary to the fallback profile.
struct TwoProfileCreds {
    acquired: std::sync::Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl crate::credentials::CredentialProvider for TwoProfileCreds {
    async fn acquire(
        &self,
        profile: &ProfileRef,
        scope: &CredScope,
    ) -> Result<daemon_common::CapabilityLease, daemon_common::CredError> {
        self.acquired.lock().unwrap().push(profile.to_string());
        Ok(daemon_common::CapabilityLease {
            cap_id: daemon_common::CredId::new(format!("{profile}-cap")),
            profile: profile.clone(),
            scope: scope.clone(),
            mode: daemon_common::CredMode::Native,
            expires_at_ms: crate::credentials::now_ms() + 60_000,
            epoch: 0,
            secret: Some(daemon_common::LeaseSecret::new(format!("sk-{profile}"))),
            signature: Vec::new(),
        })
    }
    async fn release(&self, _lease: &daemon_common::CapabilityLease) {}
    async fn rotate(&self, _profile: &ProfileRef, _cap_id: &daemon_common::CredId) {}
}

/// A provider that rejects every credential with a content-policy failure *except* the fallback
/// profile's secret — so the turn only completes once the engine has hopped profiles.
struct PolicyGatedProvider {
    ok_secret: String,
}

#[async_trait::async_trait]
impl Provider for PolicyGatedProvider {
    fn capabilities(&self) -> Capabilities {
        test_caps()
    }

    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
        if req.auth.as_deref() == Some(self.ok_secret.as_str()) {
            Ok(ModelOutput {
                text: "ok on fallback profile".into(),
                reasoning: None,
                tool_calls: Vec::new(),
                usage: UsageDelta::default(),
                ..Default::default()
            })
        } else {
            Err(Failure::ContentPolicy("blocked on primary".into()))
        }
    }
}

/// A persistent content-policy failure hops once to the configured fallback profile (wired via
/// `EngineProfile::with_fallback_profile`); the engine then completes on the fallback's
/// credential, having acquired the primary first and the fallback second.
#[tokio::test]
async fn fallback_profile_hops_credential_profile() {
    use crate::EngineProfile;
    use std::sync::Arc;

    let creds = Arc::new(TwoProfileCreds {
        acquired: std::sync::Mutex::new(Vec::new()),
    });
    let creds_for_builder = creds.clone();
    let profile = EngineProfile::new(
        Arc::new(|| {
            Arc::new(PolicyGatedProvider {
                ok_secret: "sk-fallback".into(),
            }) as Arc<dyn Provider>
        }),
        Arc::new(ToolRegistry::new()),
        SystemPrompt::new("test"),
    )
    .with_credentials(
        Arc::new(move || {
            creds_for_builder.clone() as Arc<dyn crate::credentials::CredentialProvider>
        }),
        ProfileRef::new("primary"),
    )
    .with_fallback_profile(ProfileRef::new("fallback"));

    let mut engine = profile.fresh(SessionId::new("fallback-hop"));
    engine.push_user(UserMsg::new("hello"));

    let outcome = engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .expect("turn completes after hopping to the fallback profile");
    assert!(matches!(outcome, TurnOutcome::Completed(_)));
    let acquired = creds.acquired.lock().unwrap().clone();
    assert_eq!(
        acquired,
        vec!["primary".to_string(), "fallback".to_string()],
        "acquired the primary first, then hopped to the fallback profile"
    );
}

/// A provider that always fails with a non-rotatable error.
struct FailingProvider;

#[async_trait::async_trait]
impl Provider for FailingProvider {
    fn capabilities(&self) -> Capabilities {
        test_caps()
    }

    async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
        Err(Failure::Provider("model exploded".into()))
    }
}

/// An interrupt observed at the opening phase boundary finalizes the turn as `Interrupted`.
#[tokio::test]
async fn interrupt_at_boundary_finalizes_interrupted() {
    let mut engine = completing_engine("int");
    engine.push_user(UserMsg::new("hello"));
    let control = TurnControl::new();
    control.cancel();
    let (sink, log) = collecting();

    let outcome = engine.run_turn(&NoopHost, &sink, &control).await.unwrap();
    match outcome {
        TurnOutcome::Completed(s) => assert_eq!(s.end_reason, EndReason::Interrupted),
        _ => panic!("expected a completed-but-interrupted outcome"),
    }
    assert!(log.lock().unwrap().iter().any(|e| matches!(
        e,
        AgentEvent::TurnFinished { summary, .. } if summary.end_reason == EndReason::Interrupted
    )));
}

/// A provider failure ends the turn as `Failed`, after an `Error` event.
#[tokio::test]
async fn provider_failure_emits_error_and_failed() {
    let mut engine = Engine::fresh(
        SessionId::new("fail"),
        SystemPrompt::new("test"),
        Arc::new(FailingProvider),
        Arc::new(ToolRegistry::new()),
    );
    engine.push_user(UserMsg::new("hello"));
    let (sink, log) = collecting();

    let outcome = engine
        .run_turn(&NoopHost, &sink, &TurnControl::new())
        .await
        .unwrap();
    match outcome {
        TurnOutcome::Completed(s) => assert_eq!(s.end_reason, EndReason::Failed),
        _ => panic!("expected a failed outcome"),
    }
    let log = log.lock().unwrap();
    assert!(log.iter().any(|e| matches!(e, AgentEvent::Error { .. })));
    assert!(log.iter().any(|e| matches!(
        e,
        AgentEvent::TurnFinished { summary, .. } if summary.end_reason == EndReason::Failed
    )));
}

/// A pending snapshot request is served at a phase boundary with a `ConvView` reflecting the
/// conversation.
#[tokio::test]
async fn snapshot_request_served_with_conv_view() {
    let mut engine = completing_engine("snap");
    engine.push_user(UserMsg::new("question"));
    let control = TurnControl::new();
    control.push_snapshot(ReqId(7));
    let (sink, log) = collecting();

    engine.run_turn(&NoopHost, &sink, &control).await.unwrap();
    let log = log.lock().unwrap();
    let (request_id, view) = log
        .iter()
        .find_map(|e| match e {
            AgentEvent::Snapshot {
                request_id, view, ..
            } => Some((*request_id, view.clone())),
            _ => None,
        })
        .expect("a snapshot event");
    assert_eq!(request_id, ReqId(7));
    assert!(view
        .turns
        .iter()
        .any(|t| t.role == "user" && t.text == "question"));
}

/// The snapshot view is wire-bounded: a conversation past `WIRE_PAGE_MAX` turns projects only the
/// LAST `WIRE_PAGE_MAX` turns, in order (the fixed-buffer client codec cannot decode more; the
/// durable journal serves scroll-back).
#[tokio::test]
async fn conv_view_truncates_to_the_last_wire_page_of_turns() {
    let mut engine = completing_engine("snap-cap");
    for i in 0..70 {
        engine.push_observe(UserMsg::new(format!("turn-{i}")));
    }
    let view = engine.conv_view();
    assert_eq!(view.turns.len(), daemon_common::WIRE_PAGE_MAX);
    // The window keeps the TAIL: turns 6..=69, oldest-first.
    assert_eq!(view.turns.first().unwrap().text, "turn-6");
    assert_eq!(view.turns.last().unwrap().text, "turn-69");
}

/// A queued steer is drained at a boundary: the marker lands in the conversation and a
/// `Steered` ack is emitted.
#[tokio::test]
async fn steer_drained_appends_marker_and_acks() {
    let mut engine = completing_engine("steer");
    engine.push_user(UserMsg::new("hi"));
    let control = TurnControl::new();
    control.push_steer(SteerReq {
        request_id: ReqId(3),
        text: "focus".into(),
    });
    let (sink, log) = collecting();

    engine.run_turn(&NoopHost, &sink, &control).await.unwrap();
    assert!(log.lock().unwrap().iter().any(|e| matches!(
        e,
        AgentEvent::Steered { request_id, accepted, .. } if *request_id == ReqId(3) && *accepted
    )));
    assert!(engine
        .snapshot()
        .conversation
        .turns
        .iter()
        .any(|t| matches!(t, Turn::User(u) if u.text.contains("[steer] focus"))));
}

// --- ReAct loop (§4.2) ---

use crate::conversation::ToolCall;
use crate::provider::{ScriptStep, ScriptedProvider};

/// The model calls a tool twice across two rounds, then returns final text — one activation runs
/// the whole multi-round loop and completes (no suspension).
#[tokio::test]
async fn multi_round_loop_runs_tools_then_completes() {
    let runs = Arc::new(AtomicU64::new(0));
    let provider = Arc::new(ScriptedProvider::new(
        vec![
            ScriptStep::Call {
                name: "counter".into(),
                args: "{}".into(),
            },
            ScriptStep::Call {
                name: "counter".into(),
                args: "{}".into(),
            },
        ],
        "all done",
    ));
    let mut engine = looping_engine(provider.clone(), runs.clone(), 90);
    engine.push_user(UserMsg::new("do work"));

    let outcome = engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .unwrap();
    match outcome {
        TurnOutcome::Completed(s) => {
            assert_eq!(s.end_reason, EndReason::Completed);
            assert_eq!(s.final_text.as_deref(), Some("all done"));
        }
        _ => panic!("expected completion after the loop"),
    }
    assert_eq!(
        runs.load(Ordering::Relaxed),
        2,
        "the tool ran on both rounds"
    );
    // 3 model rounds: two tool rounds + the final-text round.
    assert_eq!(provider.call_count(), 3);
    // Two tool turns are recorded in the durable conversation.
    let tool_turns = engine
        .snapshot()
        .conversation
        .turns
        .iter()
        .filter(|t| matches!(t, Turn::Tool(_)))
        .count();
    assert_eq!(tool_turns, 2);
}

/// A batch of two `Parallel` tool calls runs concurrently: the peak observed in-flight count is 2.
#[tokio::test]
async fn parallel_tool_batch_runs_concurrently() {
    let active = Arc::new(AtomicU64::new(0));
    let max_seen = Arc::new(AtomicU64::new(0));
    let provider = Arc::new(ScriptedProvider::new(
        vec![ScriptStep::Calls(vec![
            ("para".into(), "{}".into()),
            ("para".into(), "{}".into()),
        ])],
        "done",
    ));
    let tool = probe_tool(
        "para",
        crate::tools::ToolConcurrency::Parallel,
        &active,
        &max_seen,
    );
    let mut engine = probe_engine(provider, vec![tool]);
    engine.push_user(UserMsg::new("go"));

    let outcome = engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .unwrap();
    assert!(matches!(outcome, TurnOutcome::Completed(_)));
    assert_eq!(
        max_seen.load(Ordering::SeqCst),
        2,
        "both parallel calls should be in flight at once"
    );
}

/// A mixed batch (one `Parallel`, one `Exclusive`) is serialized all-or-nothing: peak in-flight is 1.
#[tokio::test]
async fn exclusive_call_serializes_the_batch() {
    let active = Arc::new(AtomicU64::new(0));
    let max_seen = Arc::new(AtomicU64::new(0));
    let provider = Arc::new(ScriptedProvider::new(
        vec![ScriptStep::Calls(vec![
            ("para".into(), "{}".into()),
            ("excl".into(), "{}".into()),
        ])],
        "done",
    ));
    let para = probe_tool(
        "para",
        crate::tools::ToolConcurrency::Parallel,
        &active,
        &max_seen,
    );
    let excl = probe_tool(
        "excl",
        crate::tools::ToolConcurrency::Exclusive,
        &active,
        &max_seen,
    );
    let mut engine = probe_engine(provider, vec![para, excl]);
    engine.push_user(UserMsg::new("go"));

    let outcome = engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .unwrap();
    assert!(matches!(outcome, TurnOutcome::Completed(_)));
    assert_eq!(
        max_seen.load(Ordering::SeqCst),
        1,
        "an exclusive call must force the whole batch to run sequentially"
    );
}

/// A provider that never stops calling a (non-delegating) tool exhausts the iteration budget; the
/// turn ends `BudgetExhausted` after one final toolless summary call.
#[tokio::test]
async fn iteration_budget_exhaustion_ends_with_summary() {
    let runs = Arc::new(AtomicU64::new(0));
    let provider = looping_call_provider("counter");
    let mut engine = looping_engine(provider.clone(), runs.clone(), 4);
    engine.push_user(UserMsg::new("loop forever"));

    let outcome = engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .unwrap();
    match outcome {
        TurnOutcome::Completed(s) => assert_eq!(s.end_reason, EndReason::BudgetExhausted),
        _ => panic!("expected budget exhaustion"),
    }
    // The tool ran exactly `max_iterations` times; the budget is the hard stop.
    assert_eq!(runs.load(Ordering::Relaxed), 4);
    // 4 loop rounds + 1 toolless summary round.
    assert_eq!(provider.call_count(), 5);
}

/// A model that re-issues the identical tool call every round, where the tool returns the
/// identical result, is looping: the §4.2 no-progress guard ends the turn `NoProgress` after
/// `max_repeated_rounds` identical rounds — well before the (much larger) iteration budget.
#[tokio::test]
async fn no_progress_guard_ends_repeated_identical_rounds() {
    let runs = Arc::new(AtomicU64::new(0));
    let provider = looping_call_provider("constant");
    let mut engine = constant_engine(provider.clone(), runs.clone(), 90, 3);
    engine.push_user(UserMsg::new("loop"));

    let outcome = engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .unwrap();
    match outcome {
        TurnOutcome::Completed(s) => assert_eq!(s.end_reason, EndReason::NoProgress),
        _ => panic!("expected a no-progress stop"),
    }
    // Stopped after 3 identical rounds, not the 90-round iteration budget.
    assert_eq!(runs.load(Ordering::Relaxed), 3);
    // 3 loop rounds + 1 toolless summary round.
    assert_eq!(provider.call_count(), 4);
}

/// With the guard disabled (`max_repeated_rounds = 0`), the same looping/constant scenario runs
/// all the way to the iteration cap — proving the guard, not the budget, is what stops it above.
#[tokio::test]
async fn no_progress_guard_disabled_runs_to_iteration_budget() {
    let runs = Arc::new(AtomicU64::new(0));
    let provider = looping_call_provider("constant");
    let mut engine = constant_engine(provider, runs.clone(), 4, 0);
    engine.push_user(UserMsg::new("loop"));

    let outcome = engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .unwrap();
    match outcome {
        TurnOutcome::Completed(s) => assert_eq!(s.end_reason, EndReason::BudgetExhausted),
        _ => panic!("expected budget exhaustion"),
    }
    assert_eq!(runs.load(Ordering::Relaxed), 4);
}

/// Cancellation observed mid-loop (after a tool runs) finalizes the turn as `Interrupted` rather
/// than looping back to the model.
#[tokio::test]
async fn cancel_mid_loop_finalizes_interrupted() {
    let runs = Arc::new(AtomicU64::new(0));
    let provider = looping_call_provider("counter");
    let mut engine = looping_engine(provider, runs, 90);
    engine.push_user(UserMsg::new("go"));
    let control = TurnControl::new();
    // Cancel before the first boundary: the turn finalizes interrupted immediately.
    control.cancel();

    let outcome = engine
        .run_turn(&NoopHost, &EventSink::discarding(), &control)
        .await
        .unwrap();
    match outcome {
        TurnOutcome::Completed(s) => assert_eq!(s.end_reason, EndReason::Interrupted),
        _ => panic!("expected interruption"),
    }
}

// --- §8 recovery + §10/§11 hooks ---

use std::collections::VecDeque;

/// A provider that replays a scripted sequence of results (failures then success), defaulting to
/// a completing response once the script is exhausted.
struct FaultProvider {
    script: std::sync::Mutex<VecDeque<Result<ModelOutput, Failure>>>,
    calls: AtomicU64,
}

impl FaultProvider {
    fn new(seq: Vec<Result<ModelOutput, Failure>>) -> Self {
        Self {
            script: std::sync::Mutex::new(seq.into_iter().collect()),
            calls: AtomicU64::new(0),
        }
    }
}

#[async_trait::async_trait]
impl Provider for FaultProvider {
    fn capabilities(&self) -> Capabilities {
        test_caps()
    }
    async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.script
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Ok(ok_output("default")))
    }
}

/// A 429 with backoff retries (emitting `RateLimit`) and then completes on a fresh attempt.
#[tokio::test]
async fn rate_limit_retries_with_backoff_then_completes() {
    let provider = Arc::new(FaultProvider::new(vec![
        Err(Failure::RateLimit {
            retry_after: None,
            message: "slow down".into(),
        }),
        Err(Failure::RateLimit {
            retry_after: None,
            message: "slow down".into(),
        }),
        Ok(ok_output("recovered")),
    ]));
    // Tiny backoff so the test is fast.
    let config = Config {
        model_backoff_base_ms: 1,
        model_backoff_max_ms: 2,
        model_max_retries: 3,
        ..Config::default()
    };
    let mut engine = Engine::fresh(
        SessionId::new("rl"),
        SystemPrompt::new("test"),
        provider.clone(),
        Arc::new(ToolRegistry::new()),
    )
    .with_config(config);
    engine.push_user(UserMsg::new("hello"));
    let (sink, log) = collecting();

    let outcome = engine
        .run_turn(&NoopHost, &sink, &TurnControl::new())
        .await
        .unwrap();
    match outcome {
        TurnOutcome::Completed(s) => {
            assert_eq!(s.end_reason, EndReason::Completed);
            assert_eq!(s.final_text.as_deref(), Some("recovered"));
        }
        _ => panic!("expected completion after backoff retries"),
    }
    assert_eq!(
        provider.calls.load(Ordering::Relaxed),
        3,
        "two retries then success"
    );
    assert!(
        log.lock()
            .unwrap()
            .iter()
            .any(|e| matches!(e, AgentEvent::RateLimit { .. })),
        "a RateLimit event was emitted during backoff"
    );
}

/// A `ContextOverflow` compacts the conversation once (the §8 -> §10 tie-in) then retries and
/// completes; the conversation is shorter afterwards.
#[tokio::test]
async fn context_overflow_compacts_then_retries() {
    let provider = Arc::new(FaultProvider::new(vec![
        Err(Failure::ContextOverflow("too long".into())),
        Ok(ok_output("after compact")),
    ]));
    let mut engine = Engine::fresh(
        SessionId::new("overflow"),
        SystemPrompt::new("test"),
        provider.clone(),
        Arc::new(ToolRegistry::new()),
    );
    // Enough turns that drop-oldest frees > 10%.
    for i in 0..8 {
        engine.push_user(UserMsg::new(format!("message {i} ").repeat(20)));
    }
    let before = engine.snapshot().conversation.turns.len();

    let outcome = engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .unwrap();
    match outcome {
        TurnOutcome::Completed(s) => assert_eq!(s.final_text.as_deref(), Some("after compact")),
        _ => panic!("expected completion after compaction"),
    }
    assert_eq!(
        provider.calls.load(Ordering::Relaxed),
        2,
        "overflow then retry"
    );
    assert!(
        engine.snapshot().conversation.turns.len() < before,
        "the conversation was compacted"
    );
}

/// An always-overflowing provider compacts once then aborts (no infinite loop): the turn ends
/// `Failed` rather than hanging.
#[tokio::test]
async fn unrecoverable_overflow_aborts() {
    let provider = Arc::new(FaultProvider::new(vec![
        Err(Failure::ContextOverflow("a".into())),
        Err(Failure::ContextOverflow("b".into())),
        Err(Failure::ContextOverflow("c".into())),
    ]));
    let mut engine = Engine::fresh(
        SessionId::new("overflow2"),
        SystemPrompt::new("test"),
        provider.clone(),
        Arc::new(ToolRegistry::new()),
    );
    for i in 0..8 {
        engine.push_user(UserMsg::new(format!("message {i} ").repeat(20)));
    }
    let outcome = engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .unwrap();
    match outcome {
        TurnOutcome::Completed(s) => assert_eq!(s.end_reason, EndReason::Failed),
        _ => panic!("expected a failed outcome, not a hang"),
    }
}

/// A context engine that always reports over-budget and drops one turn on compaction — used to
/// prove the §10 compaction signal reaches the §17 stream as an `AgentEvent::Context`.
struct DroppingContext;
#[async_trait::async_trait]
impl ContextEngine for DroppingContext {
    fn before_turn(
        &self,
        _conv: &mut Conversation,
        budget: Option<usize>,
    ) -> crate::context::Pressure {
        over_budget_pressure(budget)
    }
    async fn compact(&self, mut conv: Conversation, _budget: usize) -> Conversation {
        // Drop the oldest turn so the engine observes a non-zero `dropped_turns`.
        if !conv.turns.is_empty() {
            conv.turns.remove(0);
        }
        conv
    }
}

/// A context engine that reports over-budget but whose `compact` frees **nothing** (returns the
/// conversation unchanged) — used to prove the C6 hard last-resort truncation in
/// `prepare_turn_context` reduces the context even when the engine's own compaction is a no-op.
struct StubbornContext;
#[async_trait::async_trait]
impl ContextEngine for StubbornContext {
    fn before_turn(
        &self,
        _conv: &mut Conversation,
        budget: Option<usize>,
    ) -> crate::context::Pressure {
        over_budget_pressure(budget)
    }
    async fn compact(&self, conv: Conversation, _budget: usize) -> Conversation {
        // Stubborn: frees nothing. Any reduction the engine observes is the C6 hard cap.
        conv
    }
}

/// When the §10 context engine reports over-budget but its `compact` frees nothing, the C6 hard
/// last-resort cap deterministically drops oldest turns so the turn does not proceed over budget
/// — observable as a non-zero `dropped_turns` on the `AgentEvent::Context` and a shorter
/// conversation, despite the engine's compaction being a no-op.
#[tokio::test]
async fn hard_cap_truncates_when_engine_compaction_frees_nothing() {
    let mut engine = Engine::fresh(
        SessionId::new("hard-cap"),
        SystemPrompt::new("test"),
        Arc::new(crate::provider::MockProvider::completing("done")),
        Arc::new(ToolRegistry::new()),
    )
    .with_config(Config {
        context_budget_tokens: Some(1),
        ..Config::default()
    })
    .with_context_engine(Arc::new(StubbornContext));
    engine.push_user(UserMsg::new("first"));
    engine.push_user(UserMsg::new("second"));
    engine.push_user(UserMsg::new("third"));
    let before = engine.snapshot().conversation.turns.len();
    let (sink, log) = collecting();

    engine
        .run_turn(&NoopHost, &sink, &TurnControl::new())
        .await
        .unwrap();

    let log = log.lock().unwrap();
    assert!(
        log.iter().any(|e| matches!(
            e,
            AgentEvent::Context { status, .. } if status.compacted && status.dropped_turns >= 1
        )),
        "expected the hard cap to drop turns, got: {log:?}"
    );
    // The conversation the turn ran on was truncated below the pre-turn turn count even though
    // the engine's `compact` returned everything unchanged.
    assert!(
        engine
            .snapshot()
            .conversation
            .turns
            .iter()
            .filter(|t| matches!(t, Turn::User(_)))
            .count()
            < before
    );
}

/// Compaction at the pre-turn pressure check emits an `AgentEvent::Context { compacted: true,
/// dropped_turns >= 1 }` on the stream — the data a GUI renders as a "compacted" toast.
#[tokio::test]
async fn compaction_surfaces_context_event() {
    let mut engine = Engine::fresh(
        SessionId::new("compact-evt"),
        SystemPrompt::new("test"),
        Arc::new(crate::provider::MockProvider::completing("done")),
        Arc::new(ToolRegistry::new()),
    )
    .with_config(Config {
        context_budget_tokens: Some(1),
        ..Config::default()
    })
    .with_context_engine(Arc::new(DroppingContext));
    engine.push_user(UserMsg::new("first"));
    engine.push_user(UserMsg::new("second"));
    let (sink, log) = collecting();

    engine
        .run_turn(&NoopHost, &sink, &TurnControl::new())
        .await
        .unwrap();

    let log = log.lock().unwrap();
    assert!(
        log.iter().any(|e| matches!(
            e,
            AgentEvent::Context { status, .. } if status.compacted && status.dropped_turns >= 1
        )),
        "expected a compaction Context event, got: {log:?}"
    );
}

// §10/§11 hook-order instrumentation.
struct RecordingContext {
    log: Arc<std::sync::Mutex<Vec<&'static str>>>,
}
#[async_trait::async_trait]
impl ContextEngine for RecordingContext {
    fn on_model(&self, _model: &crate::context::ModelInfo) {
        self.log.lock().unwrap().push("on_model");
    }
    fn before_turn(
        &self,
        _conv: &mut Conversation,
        budget: Option<usize>,
    ) -> crate::context::Pressure {
        self.log.lock().unwrap().push("before_turn");
        // Force over-budget so the compaction hooks fire.
        crate::context::Pressure {
            used_tokens: 10_000,
            budget_tokens: budget,
        }
    }
    async fn compact(&self, conv: Conversation, _budget: usize) -> Conversation {
        self.log.lock().unwrap().push("compact");
        conv
    }
    fn after_response(&self, _usage: &UsageDelta) {
        self.log.lock().unwrap().push("after_response");
    }
    fn on_session_start(&self, _session: &SessionId) {
        self.log.lock().unwrap().push("session_start");
    }
    fn on_session_end(&self, _session: &SessionId, _conv: &Conversation) {
        self.log.lock().unwrap().push("session_end");
    }
    fn on_session_reset(&self, _session: &SessionId) {
        self.log.lock().unwrap().push("session_reset");
    }
}

struct RecordingMemory {
    log: Arc<std::sync::Mutex<Vec<&'static str>>>,
}
#[async_trait::async_trait]
impl MemoryProvider for RecordingMemory {
    fn name(&self) -> &str {
        "rec"
    }
    fn prompt_block(&self) -> Option<crate::memory::PromptBlock> {
        self.log.lock().unwrap().push("prompt_block");
        None
    }
    async fn recall(&self, _q: &RecallQuery) -> Option<crate::memory::RecalledBlock> {
        self.log.lock().unwrap().push("recall");
        None
    }
    async fn after_turn(&self, _turn: &Turn, _conv: &Conversation) {
        self.log.lock().unwrap().push("after_turn");
    }
    async fn before_compact(&self, _conv: &Conversation) {
        self.log.lock().unwrap().push("before_compact");
    }
    async fn on_session_switch(&self, reason: SwitchReason) {
        let label = match reason {
            SwitchReason::Start => "switch:start",
            SwitchReason::Compaction => "switch:compaction",
            SwitchReason::Handoff => "switch:handoff",
            SwitchReason::Resume => "switch:resume",
            SwitchReason::End => "switch:end",
            SwitchReason::Manual => "switch:manual",
        };
        self.log.lock().unwrap().push(label);
    }
}

/// The §10/§11 hooks fire in spec order across an incarnation: the once-per-incarnation
/// lifecycle hooks (`on_model -> session_start -> switch:start`) precede the per-turn hooks
/// (`recall -> prompt_block -> before_turn -> before_compact -> compact -> switch:compaction ->
/// after_turn -> after_response`), and `end_session` fires the teardown hooks
/// (`session_end -> switch:end`).
#[tokio::test]
async fn memory_and_context_hooks_fire_in_spec_order() {
    let log = Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
    let config = Config {
        context_budget_tokens: Some(1),
        ..Config::default()
    };
    let mut engine = Engine::fresh(
        SessionId::new("hooks"),
        SystemPrompt::new("test"),
        Arc::new(crate::provider::MockProvider::completing("done")),
        Arc::new(ToolRegistry::new()),
    )
    .with_config(config)
    .with_context_engine(Arc::new(RecordingContext { log: log.clone() }))
    .with_memory(vec![Arc::new(RecordingMemory { log: log.clone() })]);
    engine.push_user(UserMsg::new("hello"));

    engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .unwrap();
    engine.end_session().await;

    let order = log.lock().unwrap().clone();
    assert_eq!(
        order,
        vec![
            "on_model",
            "session_start",
            "switch:start",
            "recall",
            "prompt_block",
            "before_turn",
            "before_compact",
            "compact",
            "switch:compaction",
            "after_turn",
            "after_response",
            "session_end",
            "switch:end",
        ]
    );
}

/// `end_session` is balanced with `ensure_session_started`: a session that never started fires no
/// teardown hooks, a started one fires them exactly once, and a repeat call is a no-op.
#[tokio::test]
async fn end_session_is_guarded_and_idempotent() {
    let log = Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
    let mut engine = Engine::fresh(
        SessionId::new("end-guard"),
        SystemPrompt::new("test"),
        Arc::new(crate::provider::MockProvider::completing("done")),
        Arc::new(ToolRegistry::new()),
    )
    .with_context_engine(Arc::new(RecordingContext { log: log.clone() }));

    // Never started: teardown is a no-op.
    engine.end_session().await;
    assert!(
        !log.lock().unwrap().contains(&"session_end"),
        "no session_end before the lifecycle started"
    );

    engine.push_user(UserMsg::new("hello"));
    engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .unwrap();
    engine.end_session().await;
    engine.end_session().await; // idempotent: the second call is a no-op

    let ends = log
        .lock()
        .unwrap()
        .iter()
        .filter(|e| **e == "session_end")
        .count();
    assert_eq!(ends, 1, "teardown fired exactly once");
}

// ---- conversation rewind (conversation-rewind spec) -----------------------------------------

use crate::conversation::ToolResult;

/// Build an engine whose conversation is `[User, Assistant, User, Tool(call_id=cp-1), Assistant]`.
fn rewind_engine() -> Engine {
    let mut engine = completing_engine("rw");
    let conv = &mut engine.snapshot.conversation;
    conv.push_user(UserMsg::new("u0"));
    conv.push_assistant(AssistantMsg::text("a1"));
    conv.push_user(UserMsg::new("u2"));
    conv.push_tool(ToolTurn {
        assistant: AssistantMsg::text("a3 (tool)"),
        calls: vec![(
            ToolCall {
                call_id: "cp-1".into(),
                name: "write".into(),
                args: "{}".into(),
            },
            ToolResult {
                call_id: "cp-1".into(),
                ok: true,
                content: "ok".into(),
            },
        )],
    });
    conv.push_assistant(AssistantMsg::text("a4"));
    engine
}

/// `UserTurn { ordinal }` keeps `[0, ordinal)`, bumps the epoch, and emits `Rewound`.
#[tokio::test]
async fn rewind_user_turn_truncates_and_bumps_epoch() {
    let mut engine = rewind_engine();
    let epoch_before = engine.epoch();
    let (sink, log) = collecting();

    let outcome = engine
        .rewind_to(&RewindAnchor::UserTurn { ordinal: 2 }, ReqId(7), &sink)
        .expect("user-turn rewind resolves");

    assert_eq!(outcome.retained_turns, 2);
    assert_eq!(engine.snapshot.conversation.turns.len(), 2);
    assert_eq!(engine.epoch(), epoch_before.next());
    // The sealed-off tail held one tool call.
    assert_eq!(outcome.dropped_call_ids, vec!["cp-1".to_string()]);
    let events = log.lock().unwrap();
    assert!(events.iter().any(|e| matches!(
        e,
        AgentEvent::Rewound { to_cursor, epoch, request_id, .. }
            if *to_cursor == 2 && *epoch == engine.epoch().0 && *request_id == ReqId(7)
    )));
}

/// `ReplyAfter { ordinal }` keeps the user turn and drops its reply: keeps `[0, ordinal]`.
#[tokio::test]
async fn rewind_reply_after_keeps_user_turn() {
    let mut engine = rewind_engine();
    let outcome = engine
        .rewind_to(
            &RewindAnchor::ReplyAfter { ordinal: 2 },
            ReqId(1),
            &EventSink::discarding(),
        )
        .expect("reply-after rewind resolves");
    assert_eq!(outcome.retained_turns, 3);
    assert!(matches!(
        engine.snapshot.conversation.turns.last(),
        Some(Turn::User(u)) if u.text == "u2"
    ));
}

/// `Cursor { seq }` maps 1:1 to a retained turn count.
#[tokio::test]
async fn rewind_cursor_truncates_to_count() {
    let mut engine = rewind_engine();
    let outcome = engine
        .rewind_to(
            &RewindAnchor::Cursor { seq: 1 },
            ReqId(1),
            &EventSink::discarding(),
        )
        .expect("cursor rewind resolves");
    assert_eq!(outcome.retained_turns, 1);
    assert_eq!(engine.snapshot.conversation.turns.len(), 1);
}

/// An out-of-range anchor is rejected (the actor maps this to an error rather than truncating).
#[tokio::test]
async fn rewind_out_of_range_is_rejected() {
    let mut engine = rewind_engine();
    let err = engine
        .rewind_to(
            &RewindAnchor::UserTurn { ordinal: 99 },
            ReqId(1),
            &EventSink::discarding(),
        )
        .unwrap_err();
    assert_eq!(err, RewindError::OutOfRange);
    // The conversation is untouched on rejection.
    assert_eq!(engine.snapshot.conversation.turns.len(), 5);
}

/// A `UserTurn`/`ReplyAfter` anchor that does not point at a user turn is rejected.
#[tokio::test]
async fn rewind_non_user_anchor_is_rejected() {
    let mut engine = rewind_engine();
    // Ordinal 1 is the assistant turn `a1`.
    let err = engine
        .rewind_to(
            &RewindAnchor::UserTurn { ordinal: 1 },
            ReqId(1),
            &EventSink::discarding(),
        )
        .unwrap_err();
    assert_eq!(err, RewindError::NotAUserTurn);
}

/// Rewind clears the awaited-job set, so a late completion for a now-unawaited job is fenced.
#[tokio::test]
async fn rewind_fences_late_completion() {
    let mut engine = rewind_engine();
    engine.snapshot.waiting_for.push(JobId::new("job-1"));
    engine
        .rewind_to(
            &RewindAnchor::UserTurn { ordinal: 0 },
            ReqId(1),
            &EventSink::discarding(),
        )
        .expect("rewind resolves");
    assert!(engine.snapshot.waiting_for.is_empty());
    // A completion for the abandoned job is dropped (not stashed) post-rewind.
    engine.apply_completions(vec![Completion {
        job_id: JobId::new("job-1"),
        payload: b"late".to_vec(),
    }]);
    assert!(
        engine.pending.is_empty(),
        "late completion fenced by rewind"
    );
}

/// A full-clear rewind (retained == 0) is the daemon's `/new` analog: the §10 context engine gets
/// `on_session_reset` so a stateful engine (LCM) resets in step with the emptied conversation.
#[tokio::test]
async fn rewind_to_root_fires_context_session_reset() {
    let log = Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
    let mut engine =
        rewind_engine().with_context_engine(Arc::new(RecordingContext { log: log.clone() }));
    let outcome = engine
        .rewind_to(
            &RewindAnchor::UserTurn { ordinal: 0 },
            ReqId(1),
            &EventSink::discarding(),
        )
        .expect("root rewind resolves");
    assert_eq!(outcome.retained_turns, 0);
    assert_eq!(
        log.lock().unwrap().as_slice(),
        ["session_reset"],
        "a full clear fires exactly the reset hook"
    );
}

/// A partial rewind is not a reset: the context engine re-measures the shortened body next turn.
#[tokio::test]
async fn partial_rewind_does_not_fire_session_reset() {
    let log = Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
    let mut engine =
        rewind_engine().with_context_engine(Arc::new(RecordingContext { log: log.clone() }));
    let outcome = engine
        .rewind_to(
            &RewindAnchor::UserTurn { ordinal: 2 },
            ReqId(1),
            &EventSink::discarding(),
        )
        .expect("partial rewind resolves");
    assert_eq!(outcome.retained_turns, 2);
    assert!(
        !log.lock().unwrap().contains(&"session_reset"),
        "no reset on a partial rewind"
    );
}

// --- Cluster B: exec-approval fingerprint gate on the durable re-run ------------------------------

/// A tool that records whether it ran and reports a fixed [`CommandFingerprint`] — lets the
/// approval-resolve tests drive a match / mismatch against a stored fingerprint deterministically.
struct FingerprintProbeTool {
    ran: Arc<AtomicU64>,
    resolved: crate::exec::CommandFingerprint,
}

#[async_trait::async_trait]
impl crate::tools::Tool for FingerprintProbeTool {
    fn name(&self) -> &str {
        "fp_probe"
    }
    fn schema(&self) -> &str {
        "{}"
    }
    async fn run(
        &self,
        call: &crate::conversation::ToolCall,
        _cx: &crate::turn::TurnCx<'_>,
    ) -> crate::tools::ToolOutcome {
        self.ran.fetch_add(1, Ordering::SeqCst);
        crate::tools::ToolOutcome::text(call.call_id.clone(), true, "fp_probe RAN")
    }
    async fn resolved_fingerprint(
        &self,
        _call: &crate::conversation::ToolCall,
        _cx: &crate::turn::TurnCx<'_>,
    ) -> Option<crate::exec::CommandFingerprint> {
        Some(self.resolved.clone())
    }
}

/// Seed an engine with one parked approval for `fp_probe` (stored fingerprint = `stored`), an
/// `awaiting-approval` marker slot in the conversation, and an `"allow"` completion. The probe tool
/// reports `resolved` as its current fingerprint. Returns the engine + the run counter after
/// `resolve_approvals`.
async fn drive_approval_resolution(
    stored: crate::exec::CommandFingerprint,
    resolved: crate::exec::CommandFingerprint,
) -> (Engine, Arc<AtomicU64>) {
    use crate::conversation::{AssistantMsg, ToolCall, ToolResult, ToolTurn, Turn};

    let ran = Arc::new(AtomicU64::new(0));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(FingerprintProbeTool {
        ran: ran.clone(),
        resolved,
    }));
    let mut engine = test_engine("fp", Arc::new(TextProvider), registry);

    let job = JobId::new("job-fp");
    let call = ToolCall {
        call_id: "call-fp".into(),
        name: "fp_probe".into(),
        args: "{}".into(),
    };
    // The parked tool left an `awaiting-approval:{job}` marker slot the resolver splices into.
    engine
        .snapshot
        .conversation
        .turns
        .push(Turn::Tool(ToolTurn {
            assistant: AssistantMsg::text(""),
            calls: vec![(
                call.clone(),
                ToolResult {
                    call_id: "call-fp".into(),
                    ok: false,
                    content: format!("awaiting-approval:{job}"),
                },
            )],
        }));
    engine
        .snapshot
        .pending_approvals
        .push(crate::snapshot::PendingApproval {
            job_id: job.clone(),
            call,
            prompt: "approve fp_probe".into(),
            path: None,
            fingerprint: Some(stored),
        });
    // The parked job must be awaited or the completion is fenced (rewind/epoch guard).
    engine.snapshot.waiting_for.push(job.clone());
    engine.apply_completions(vec![Completion {
        job_id: job,
        payload: b"allow".to_vec(),
    }]);
    engine
        .resolve_approvals(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await;
    (engine, ran)
}

/// The content spliced into the `fp_probe` result slot after resolution.
fn spliced_result(engine: &Engine) -> (bool, String) {
    for turn in &engine.snapshot.conversation.turns {
        if let crate::conversation::Turn::Tool(tt) = turn {
            for (_c, r) in &tt.calls {
                if r.call_id == "call-fp" {
                    return (r.ok, r.content.clone());
                }
            }
        }
    }
    panic!("no fp_probe result slot found");
}

/// THE Cluster B TOCTOU gate: an operator-approved command whose resolved-at-exec fingerprint no
/// longer matches what was approved is REFUSED — the tool never runs, and a refusal is spliced in.
#[tokio::test]
async fn approved_command_refused_when_fingerprint_changed() {
    let stored = crate::exec::CommandFingerprint::compute(
        "exec.argv",
        std::path::Path::new("/usr/bin/approved"),
        &[],
        &[],
        std::path::Path::new("/ws"),
    );
    // At re-run the command resolves to a DIFFERENT absolute binary (the approve-then-swap).
    let resolved = crate::exec::CommandFingerprint::compute(
        "exec.argv",
        std::path::Path::new("/tmp/evil"),
        &[],
        &[],
        std::path::Path::new("/ws"),
    );
    let (engine, ran) = drive_approval_resolution(stored, resolved).await;

    assert_eq!(
        ran.load(Ordering::SeqCst),
        0,
        "a fingerprint mismatch must NOT run the tool"
    );
    let (ok, content) = spliced_result(&engine);
    assert!(!ok, "a refused approval is not ok");
    assert!(
        content.contains("refused") && !content.contains("RAN"),
        "refusal spliced, command not run: {content}"
    );
}

/// The matching case: when the resolved-at-exec fingerprint equals what was approved, the command
/// runs as before (proves the gate does not refuse legitimate re-runs).
#[tokio::test]
async fn approved_command_runs_when_fingerprint_matches() {
    let fp = crate::exec::CommandFingerprint::compute(
        "exec.argv",
        std::path::Path::new("/usr/bin/approved"),
        &[],
        &[],
        std::path::Path::new("/ws"),
    );
    let (engine, ran) = drive_approval_resolution(fp.clone(), fp).await;

    assert_eq!(
        ran.load(Ordering::SeqCst),
        1,
        "a matching fingerprint runs the approved command exactly once"
    );
    let (ok, content) = spliced_result(&engine);
    assert!(ok, "the matched command's result is spliced in");
    assert!(
        content.contains("fp_probe RAN"),
        "command output: {content}"
    );
}
