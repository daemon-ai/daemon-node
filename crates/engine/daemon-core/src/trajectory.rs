// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! B2 — the adversarial provider testkit: a scripted **streaming** [`Provider`] that can fail,
//! disconnect mid-stream, hang, and verify request digests, so the §8 recovery machinery
//! (`ModelCallPolicy`, `drive_model_call`, the `call_model` loop) has deterministic adversarial
//! coverage.
//!
//! [`ScriptedProvider`](crate::provider::ScriptedProvider) never returns `Err` and never streams;
//! every retry/rotate/compact/fallback/watchdog path it leaves dark is exactly where production
//! incidents live. A [`TrajectoryProvider`] plays a fixed script of [`TrajectoryCall`]s — one per
//! expected provider invocation, each bundling an optional request expectation with its outcome
//! (review correction: expectation and outcome ride together, not as a standalone expect step).
//!
//! This is an in-memory testkit, not yet a replay lane: recorded fixtures, fault-override files,
//! and the socket-level `daemon_node::assemble()` scenarios are the B2b follow-up.
//!
//! # Model Experience
//!
//! None — this provider never runs in production. In tests the "model" is the script: deltas are
//! streamed exactly as written, so host-visible event ordering matches a real streaming provider.
//!
//! # KV Cache Effect
//!
//! None. The optional [`TrajectoryCall::expect`] digest ties B2 to B1a: a scripted call can assert
//! the engine sent exactly the request its assembly inputs imply, which is the same invariant that
//! protects the provider prefix cache.
//!
//! # Durability/Replay Effect
//!
//! The testkit is what engine-level recovery tests drive `run_turn` against, so the durability
//! invariants (partial deltas before a failure leave no durable assistant message; a retried call
//! does not duplicate conversation results) are asserted against real streams, not chat shims.
//!
//! # Security Boundary
//!
//! [`TrajectoryProvider`] strips `Request::auth` before digesting (the digest is secret-free by
//! exclusion); the script never sees or stores the credential.
//!
//! # Known Limitations and Deferred Work
//!
//! - Scripts are Rust-constructed in tests; serde fixtures + recorder/normalizer are B2b.
//! - A `Hang`/`HangAfter` outcome relies on the test terminating it (cancel or watchdog); scripts
//!   must place such an outcome only as the **final** call, so no deliberately unreachable suffix
//!   exists for [`TrajectoryProvider::assert_exhausted`] to mis-assert.
//! - Chunk accounting assumes the engine consumes a non-hanging stream to completion (true of
//!   `drive_model_call`); a `HangAfter` test must cancel only after observing the emitted deltas.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures::stream::{self, BoxStream, StreamExt};

use crate::provider::{
    Capabilities, Failure, ModelOutput, Provider, Request, StreamEvent, ToolCallFormat,
};
use crate::request_digest::ModelRequestDigest;

/// One scripted provider invocation: an optional B1a request expectation bundled with the outcome
/// the provider plays for it.
pub struct TrajectoryCall {
    /// When set, the incoming request (auth stripped) must digest to exactly this value; a
    /// mismatch fails the call with [`Failure::Fatal`] and poisons
    /// [`TrajectoryProvider::assert_exhausted`].
    pub expect: Option<ModelRequestDigest>,
    /// What the provider does for this call.
    pub outcome: TrajectoryOutcome,
}

impl TrajectoryCall {
    /// A call with no request expectation.
    pub fn new(outcome: TrajectoryOutcome) -> Self {
        Self {
            expect: None,
            outcome,
        }
    }

    /// Attach a B1a digest expectation to this call.
    pub fn expecting(mut self, digest: ModelRequestDigest) -> Self {
        self.expect = Some(digest);
        self
    }
}

/// The scripted behaviour of one [`TrajectoryCall`].
pub enum TrajectoryOutcome {
    /// A scripted delta sequence, normally ending in [`StreamEvent::Done`].
    Chunks(Vec<StreamEvent>),
    /// An error before the first chunk.
    Fail(Failure),
    /// A mid-stream disconnect/ratelimit: the chunks, then the failure.
    FailAfter {
        /// The deltas emitted before the stream breaks.
        chunks: Vec<StreamEvent>,
        /// The failure terminating the stream.
        failure: Failure,
    },
    /// Pending before any output, until cancel or the stale-stream watchdog fires.
    Hang,
    /// Emit the chunks, then pend forever — the deterministic MID-STREAM cancellation point
    /// (review round 2): `Chunks` can complete before an async cancel arrives, so a test cancels
    /// only after observing these deltas, and the stream then pends until the cancel token fires.
    HangAfter {
        /// The deltas emitted before the stream goes silent.
        chunks: Vec<StreamEvent>,
    },
}

impl TrajectoryOutcome {
    /// A successful streamed completion: one `TextDelta` per chunk, then a terminal `Done`
    /// carrying the concatenated text.
    pub fn streamed_text(chunks: &[&str]) -> Self {
        let full: String = chunks.concat();
        let mut events: Vec<StreamEvent> = chunks
            .iter()
            .map(|c| StreamEvent::TextDelta((*c).to_string()))
            .collect();
        events.push(StreamEvent::Done(ModelOutput {
            text: full,
            ..Default::default()
        }));
        TrajectoryOutcome::Chunks(events)
    }

    /// How many scripted events this outcome is expected to deliver to the engine.
    fn chunk_count(&self) -> u64 {
        match self {
            TrajectoryOutcome::Chunks(events) => events.len() as u64,
            TrajectoryOutcome::Fail(_) => 0,
            TrajectoryOutcome::FailAfter { chunks, .. } => chunks.len() as u64,
            TrajectoryOutcome::Hang => 0,
            TrajectoryOutcome::HangAfter { chunks } => chunks.len() as u64,
        }
    }
}

/// A streaming [`Provider`] that plays a fixed [`TrajectoryCall`] script and then fails loudly:
/// an unscripted call, an unmet digest expectation, an unissued scripted call, or an undelivered
/// scripted chunk all fail the test via [`TrajectoryProvider::assert_exhausted`].
pub struct TrajectoryProvider {
    calls: Mutex<VecDeque<TrajectoryCall>>,
    total: usize,
    chunks_expected: AtomicU64,
    chunks_emitted: Arc<AtomicU64>,
    problem: Mutex<Option<String>>,
}

impl TrajectoryProvider {
    /// A provider that plays `calls` in order.
    pub fn new(calls: Vec<TrajectoryCall>) -> Self {
        let total = calls.len();
        Self {
            calls: Mutex::new(calls.into_iter().collect()),
            total,
            chunks_expected: AtomicU64::new(0),
            chunks_emitted: Arc::new(AtomicU64::new(0)),
            problem: Mutex::new(None),
        }
    }

    /// The first scripting violation observed (unscripted call or digest mismatch), if any.
    pub fn first_problem(&self) -> Option<String> {
        self.problem.lock().unwrap().clone()
    }

    /// Every scripted call was issued, every scripted chunk was delivered, and no violation was
    /// recorded — or panic (fail the test). A call terminated by *intentional* cancellation
    /// (`Hang`/`HangAfter` cancelled by the test) counts as consumed; its pre-hang chunks are
    /// still accounted, so a test must cancel only after observing them.
    pub fn assert_exhausted(&self) {
        if let Some(problem) = self.first_problem() {
            panic!("trajectory violation: {problem}");
        }
        let unissued = self.calls.lock().unwrap().len();
        assert_eq!(
            unissued, 0,
            "trajectory not exhausted: {unissued} of {} scripted calls never issued",
            self.total
        );
        let expected = self.chunks_expected.load(Ordering::SeqCst);
        let emitted = self.chunks_emitted.load(Ordering::SeqCst);
        assert_eq!(
            emitted, expected,
            "trajectory not exhausted: {emitted} of {expected} scripted chunks delivered"
        );
    }

    fn record_problem(&self, problem: String) {
        let mut slot = self.problem.lock().unwrap();
        if slot.is_none() {
            *slot = Some(problem);
        }
    }
}

#[async_trait::async_trait]
impl Provider for TrajectoryProvider {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_native_tools: true,
            supports_streaming: true,
            tool_call_format: ToolCallFormat::Native,
            max_context: Some(8192),
        }
    }

    /// Fold the scripted stream into a terminal output (a `Hang` outcome pends here too). The
    /// engine always drives [`Provider::stream`]; this exists for completeness.
    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
        let mut stream = self.stream(req);
        let mut done: Option<ModelOutput> = None;
        while let Some(event) = stream.next().await {
            if let StreamEvent::Done(out) = event? {
                done = Some(out);
            }
        }
        done.ok_or_else(|| Failure::Provider("trajectory stream closed without Done".into()))
    }

    fn stream(&self, req: Request) -> BoxStream<'_, Result<StreamEvent, Failure>> {
        let call = self.calls.lock().unwrap().pop_front();
        let Some(call) = call else {
            let msg = format!(
                "unscripted provider call: all {} scripted calls already consumed",
                self.total
            );
            self.record_problem(msg.clone());
            return Box::pin(stream::once(async move { Err(Failure::Fatal(msg)) }));
        };

        if let Some(expected) = &call.expect {
            // Secret-free by exclusion: the engine attaches the lease secret after assemble(), so
            // strip it before digesting — the expectation binds the request shape, never the auth.
            let mut sanitized = req;
            sanitized.auth = None;
            let got = ModelRequestDigest::of_request(&sanitized);
            if &got != expected {
                let msg =
                    format!("request digest mismatch: expected {expected}, engine sent {got}");
                self.record_problem(msg.clone());
                return Box::pin(stream::once(async move { Err(Failure::Fatal(msg)) }));
            }
        }

        self.chunks_expected
            .fetch_add(call.outcome.chunk_count(), Ordering::SeqCst);
        let emitted = self.chunks_emitted.clone();
        let count_ok = move |event: &Result<StreamEvent, Failure>| {
            if event.is_ok() {
                emitted.fetch_add(1, Ordering::SeqCst);
            }
        };
        match call.outcome {
            TrajectoryOutcome::Chunks(events) => {
                Box::pin(stream::iter(events.into_iter().map(Ok)).inspect(count_ok))
            }
            TrajectoryOutcome::Fail(failure) => Box::pin(stream::once(async move { Err(failure) })),
            TrajectoryOutcome::FailAfter { chunks, failure } => Box::pin(
                stream::iter(chunks.into_iter().map(Ok))
                    .chain(stream::once(async move { Err(failure) }))
                    .inspect(count_ok),
            ),
            TrajectoryOutcome::Hang => Box::pin(stream::pending()),
            TrajectoryOutcome::HangAfter { chunks } => Box::pin(
                stream::iter(chunks.into_iter().map(Ok))
                    .inspect(count_ok)
                    .chain(stream::pending()),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn plays_chunks_in_order_and_exhausts() {
        let provider = TrajectoryProvider::new(vec![TrajectoryCall::new(
            TrajectoryOutcome::streamed_text(&["a", "b"]),
        )]);
        let mut stream = provider.stream(Request::default());
        let mut texts = Vec::new();
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::TextDelta(t) => texts.push(t),
                StreamEvent::Done(out) => assert_eq!(out.text, "ab"),
                other => panic!("unexpected event: {other:?}"),
            }
        }
        assert_eq!(texts, vec!["a", "b"]);
        provider.assert_exhausted();
    }

    #[tokio::test]
    async fn fail_after_emits_chunks_then_the_failure() {
        let provider =
            TrajectoryProvider::new(vec![TrajectoryCall::new(TrajectoryOutcome::FailAfter {
                chunks: vec![StreamEvent::TextDelta("partial".into())],
                failure: Failure::TransientTransport("connection reset".into()),
            })]);
        let mut stream = provider.stream(Request::default());
        assert!(matches!(
            stream.next().await,
            Some(Ok(StreamEvent::TextDelta(t))) if t == "partial"
        ));
        assert!(matches!(
            stream.next().await,
            Some(Err(Failure::TransientTransport(_)))
        ));
        provider.assert_exhausted();
    }

    #[tokio::test]
    #[should_panic(expected = "unscripted provider call")]
    async fn an_unscripted_call_poisons_exhaustion() {
        let provider = TrajectoryProvider::new(vec![]);
        let mut stream = provider.stream(Request::default());
        assert!(matches!(stream.next().await, Some(Err(Failure::Fatal(_)))));
        provider.assert_exhausted();
    }

    #[test]
    #[should_panic(expected = "never issued")]
    fn an_unissued_call_fails_exhaustion() {
        let provider = TrajectoryProvider::new(vec![TrajectoryCall::new(TrajectoryOutcome::Fail(
            Failure::Fatal("unreached".into()),
        ))]);
        provider.assert_exhausted();
    }

    #[tokio::test]
    #[should_panic(expected = "scripted chunks delivered")]
    async fn an_undelivered_chunk_fails_exhaustion() {
        let provider = TrajectoryProvider::new(vec![TrajectoryCall::new(
            TrajectoryOutcome::streamed_text(&["never read"]),
        )]);
        // Open the stream (consumes the call) but drop it without polling any chunk.
        let stream = provider.stream(Request::default());
        drop(stream);
        provider.assert_exhausted();
    }

    #[tokio::test]
    async fn digest_mismatch_fails_the_call_and_records_the_problem() {
        use crate::conversation::Conversation;
        use crate::request_digest::AssemblyInputs;
        let other = Conversation::default();
        let injection = crate::context::TurnInjection::default();
        let wrong = ModelRequestDigest::of_assembly(&AssemblyInputs {
            conversation: &other,
            composed: None,
            injection: &injection,
            tools: &[],
            cache_ttl: crate::provider::CacheTtl::FiveMin,
        });
        let provider = TrajectoryProvider::new(vec![TrajectoryCall::new(
            TrajectoryOutcome::streamed_text(&["unreachable"]),
        )
        .expecting(wrong)]);
        let mut req = Request::default();
        req.messages.push(crate::provider::RequestMsg {
            role: "user".into(),
            content: "something else".into(),
            ..Default::default()
        });
        let mut stream = provider.stream(req);
        assert!(matches!(stream.next().await, Some(Err(Failure::Fatal(_)))));
        assert!(provider
            .first_problem()
            .expect("mismatch recorded")
            .contains("digest mismatch"));
    }
}
