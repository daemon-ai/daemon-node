//! Model-call recovery middleware (§8).
//!
//! A model call is the most failure-prone step in a turn: providers rate-limit, time out, overload,
//! reject oversized context, and rotate credentials out from under us. This module concentrates the
//! recovery policy that wraps every call:
//!
//! - [`classify_api_error`] maps an HTTP status/headers/body into the precise [`Failure`] taxonomy
//!   (used by the networked providers in `daemon-providers`).
//! - [`Failure::recovery`] maps a failure to a [`Recovery`] action; [`ModelCallPolicy::decide`]
//!   bounds that action by the retry budget (so a recoverable failure eventually aborts).
//! - [`ModelCallPolicy::backoff`] computes a jittered exponential backoff (2-120s) honoring a
//!   server `Retry-After`.
//! - [`drive_model_call`] consumes a [`Provider::stream`], streaming `TextDelta`/`ReasoningDelta`
//!   to the host as chunks arrive, guarded by a stale-stream watchdog and the turn's cancel token.
//!
//! The retry/rotate/compact/fallback *loop* itself lives in `engine.rs` (`call_model`) because it
//! owns the credential provider and the conversation; this module is the policy + the stream driver.

use crate::config::Config;
use crate::events::EventSink;
use crate::provider::{Failure, ModelOutput, Provider, Recovery, Request, StreamEvent};
use daemon_common::UsageDelta;
use daemon_protocol::AgentEvent;
use futures::StreamExt;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// What the recovery loop should do next for a given failure, *after* accounting for the retry
/// budget — the actionable refinement of [`Recovery`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecoveryStep {
    /// Sleep `after` then retry the same profile.
    Retry {
        /// The backoff to wait before retrying.
        after: Duration,
    },
    /// Rotate the credential then retry (quota/auth).
    Rotate,
    /// Compact the context then retry once (context/payload overflow).
    Compact,
    /// Hop to the single fallback profile.
    Fallback,
    /// Give up — surface the failure.
    Abort,
}

/// The §8 recovery policy: retry budget + backoff bounds + stream watchdog, derived from [`Config`].
#[derive(Clone, Copy, Debug)]
pub struct ModelCallPolicy {
    /// Maximum recoverable retries before aborting.
    pub max_retries: u32,
    /// The backoff floor.
    pub backoff_base: Duration,
    /// The backoff ceiling.
    pub backoff_max: Duration,
    /// The stale-stream watchdog (zero disables it).
    pub watchdog: Duration,
}

impl ModelCallPolicy {
    /// Build the policy from the engine tunables (§20).
    pub fn from_config(config: &Config) -> Self {
        Self {
            max_retries: config.model_max_retries,
            backoff_base: Duration::from_millis(config.model_backoff_base_ms),
            backoff_max: Duration::from_millis(config.model_backoff_max_ms),
            watchdog: Duration::from_millis(config.model_stream_watchdog_ms),
        }
    }

    /// Decide the next step for `failure` on retry `attempt` (0-based). The failure's intrinsic
    /// [`Recovery`] is bounded by the retry budget: a retryable/rotatable failure that has exhausted
    /// the budget aborts (or, for content/billing, hops to the fallback profile).
    pub fn decide(&self, failure: &Failure, attempt: u32) -> RecoveryStep {
        let exhausted = attempt >= self.max_retries;
        match failure.recovery() {
            Recovery::Retry { after } => {
                if exhausted {
                    RecoveryStep::Abort
                } else {
                    RecoveryStep::Retry {
                        after: self.backoff(attempt, after),
                    }
                }
            }
            Recovery::Rotate => {
                if exhausted {
                    // A persistently rotatable failure on the last attempt: try the fallback profile.
                    RecoveryStep::Fallback
                } else {
                    RecoveryStep::Rotate
                }
            }
            // Compaction is attempted at most once (the engine guards re-entry); past that, abort.
            Recovery::Compact => {
                if exhausted {
                    RecoveryStep::Abort
                } else {
                    RecoveryStep::Compact
                }
            }
            Recovery::Fallback => RecoveryStep::Fallback,
            Recovery::Abort => RecoveryStep::Abort,
        }
    }

    /// A jittered exponential backoff for `attempt` (0-based), honoring a server `retry_after`
    /// floor: `min(base * 2^attempt, max)` plus up to 25% jitter, never below `retry_after`.
    pub fn backoff(&self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        let base = self.backoff_base.as_millis() as u64;
        let cap = self.backoff_max.as_millis() as u64;
        let exp = base.saturating_mul(1u64.checked_shl(attempt).unwrap_or(u64::MAX));
        let capped = exp.min(cap).max(1);
        // Deterministic-ish jitter from a cheap clock read — adds spread without an rng dep.
        let jitter = jitter_ms(capped);
        let computed = Duration::from_millis(capped.saturating_add(jitter));
        match retry_after {
            Some(ra) if ra > computed => ra.min(self.backoff_max.max(ra)),
            _ => computed,
        }
    }
}

/// Up to ~25% of `ceiling_ms` of pseudo-random jitter, seeded from the wall clock (no rng dep).
fn jitter_ms(ceiling_ms: u64) -> u64 {
    if ceiling_ms == 0 {
        return 0;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    let span = (ceiling_ms / 4).max(1);
    nanos % span
}

/// Parse a `Retry-After` header value (delta-seconds form) into a [`Duration`]. The HTTP-date form
/// is not honored (providers use delta-seconds); returns `None` for anything unparseable.
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    value.trim().parse::<u64>().ok().map(Duration::from_secs)
}

/// Classify an HTTP error response into the §8 [`Failure`] taxonomy.
///
/// `header` looks up a (case-insensitive by convention) response header; `body` is the (possibly
/// truncated) response body used to disambiguate `400`s (context-overflow vs content-policy vs a
/// plain format error). This is the single place the networked providers turn transport errors into
/// recoverable failures, so recovery behaviour is provider-independent and unit-testable.
pub fn classify_api_error(
    status: u16,
    header: impl Fn(&str) -> Option<String>,
    body: &str,
) -> Failure {
    let retry_after = header("retry-after").and_then(|v| parse_retry_after(&v));
    let lower = body.to_ascii_lowercase();
    match status {
        429 => Failure::RateLimit {
            retry_after,
            message: snippet(body),
        },
        401 | 403 => Failure::Auth(snippet(body)),
        402 => Failure::Billing(snippet(body)),
        413 => Failure::PayloadTooLarge(snippet(body)),
        400 | 422 => {
            if mentions_context_overflow(&lower) {
                Failure::ContextOverflow(snippet(body))
            } else if mentions_content_policy(&lower) {
                Failure::ContentPolicy(snippet(body))
            } else {
                Failure::FormatError(snippet(body))
            }
        }
        408 | 409 | 425 => Failure::TransientTransport(snippet(body)),
        503 | 529 => Failure::ProviderOverloaded(snippet(body)),
        500 | 502 | 504 => Failure::TransientTransport(snippet(body)),
        s if (500..600).contains(&s) => Failure::ProviderOverloaded(snippet(body)),
        _ => Failure::Provider(format!("http {status}: {}", snippet(body))),
    }
}

fn mentions_context_overflow(lower: &str) -> bool {
    lower.contains("context length")
        || lower.contains("maximum context")
        || lower.contains("context_length_exceeded")
        || lower.contains("too many tokens")
        || lower.contains("prompt is too long")
        || lower.contains("reduce the length")
}

fn mentions_content_policy(lower: &str) -> bool {
    lower.contains("content_policy")
        || lower.contains("content policy")
        || lower.contains("content_filter")
        || lower.contains("safety")
}

/// A short, single-line snippet of a response body for error messages (never the whole body).
fn snippet(body: &str) -> String {
    let one_line: String = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.len() > 200 {
        format!("{}…", &one_line[..200])
    } else {
        one_line
    }
}

/// Consume a single [`Provider::stream`] call under the stale-stream watchdog and the turn's cancel
/// token, streaming `TextDelta`/`ReasoningDelta` to `events` as chunks arrive and returning the
/// assembled canonical [`ModelOutput`].
///
/// Emission contract: incremental text/reasoning deltas are forwarded live; if the provider did not
/// stream a given channel (e.g. the `chat()`-adapting default that emits one terminal `Done`), the
/// channel is emitted once from the final output so the host always sees the content. A single
/// `Usage` event is emitted from the authoritative final usage (per-chunk `Usage` deltas are folded,
/// not re-emitted). A stream silent for longer than `watchdog` is a recoverable transport failure.
pub async fn drive_model_call(
    provider: &dyn Provider,
    req: Request,
    cancel: &CancellationToken,
    watchdog: Duration,
    events: &EventSink,
) -> Result<ModelOutput, Failure> {
    let mut stream = provider.stream(req);
    let mut streamed_text = false;
    let mut streamed_reasoning = false;
    let mut streamed_usage = UsageDelta::default();
    let mut done: Option<ModelOutput> = None;

    loop {
        let next = if watchdog.is_zero() {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(Failure::Cancelled),
                n = stream.next() => Ok(n),
            }
        } else {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(Failure::Cancelled),
                n = tokio::time::timeout(watchdog, stream.next()) => n,
            }
        };
        let event = match next {
            Err(_elapsed) => {
                return Err(Failure::TransientTransport(
                    "model stream stalled (watchdog elapsed)".into(),
                ))
            }
            Ok(None) => break,
            Ok(Some(Err(failure))) => return Err(failure),
            Ok(Some(Ok(event))) => event,
        };
        match event {
            StreamEvent::TextDelta(text) => {
                if !text.is_empty() {
                    streamed_text = true;
                    events.emit(|seq| AgentEvent::TextDelta { seq, text });
                }
            }
            StreamEvent::ReasoningDelta(text) => {
                if !text.is_empty() {
                    streamed_reasoning = true;
                    events.emit(|seq| AgentEvent::ReasoningDelta { seq, text });
                }
            }
            StreamEvent::Usage(delta) => streamed_usage.add(&delta),
            StreamEvent::Done(output) => done = Some(output),
        }
    }

    // The terminal `Done` carries the authoritative output; fall back to streamed usage if a
    // provider somehow closed the stream without one.
    let mut out = done.unwrap_or_default();
    if out.usage == UsageDelta::default() && streamed_usage != UsageDelta::default() {
        out.usage = streamed_usage;
    }

    // Emit any channel the provider did not stream incrementally, exactly once.
    if !streamed_text && !out.text.is_empty() {
        let text = out.text.clone();
        events.emit(|seq| AgentEvent::TextDelta { seq, text });
    }
    if !streamed_reasoning {
        if let Some(reasoning) = &out.reasoning {
            if !reasoning.is_empty() {
                let text = reasoning.clone();
                events.emit(|seq| AgentEvent::ReasoningDelta { seq, text });
            }
        }
    }
    events.emit(|seq| AgentEvent::Usage {
        seq,
        delta: out.usage,
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{Capabilities, ModelOutput, ToolCallFormat};
    use crate::Provider;
    use daemon_common::UsageDelta;
    use futures::stream::{self, BoxStream};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    fn policy() -> ModelCallPolicy {
        ModelCallPolicy::from_config(&Config::default())
    }

    fn caps() -> Capabilities {
        Capabilities {
            supports_native_tools: true,
            supports_streaming: true,
            tool_call_format: ToolCallFormat::Native,
            max_context: None,
        }
    }

    /// A provider whose stream never yields — exercises the stale-stream watchdog.
    struct HungProvider;
    #[async_trait::async_trait]
    impl Provider for HungProvider {
        fn capabilities(&self) -> Capabilities {
            caps()
        }
        async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
            Ok(ModelOutput::default())
        }
        fn stream(&self, _req: Request) -> BoxStream<'_, Result<StreamEvent, Failure>> {
            Box::pin(stream::pending())
        }
    }

    /// A provider that streams two text deltas then a terminal `Done` with the full text.
    struct StreamingProvider {
        calls: AtomicU64,
    }
    #[async_trait::async_trait]
    impl Provider for StreamingProvider {
        fn capabilities(&self) -> Capabilities {
            caps()
        }
        async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
            Ok(ModelOutput::default())
        }
        fn stream(&self, _req: Request) -> BoxStream<'_, Result<StreamEvent, Failure>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let done = ModelOutput {
                text: "Hello".into(),
                reasoning: None,
                tool_calls: Vec::new(),
                usage: UsageDelta {
                    input_tokens: 5,
                    output_tokens: 2,
                    api_calls: 1,
                },
            };
            Box::pin(stream::iter(vec![
                Ok(StreamEvent::TextDelta("Hel".into())),
                Ok(StreamEvent::TextDelta("lo".into())),
                Ok(StreamEvent::Done(done)),
            ]))
        }
    }

    fn collecting() -> (EventSink, Arc<std::sync::Mutex<Vec<AgentEvent>>>) {
        let log = Arc::new(std::sync::Mutex::new(Vec::<AgentEvent>::new()));
        let l = log.clone();
        (EventSink::new(move |ev| l.lock().unwrap().push(ev)), log)
    }

    #[tokio::test]
    async fn watchdog_classifies_hung_stream_as_transport() {
        let (sink, _log) = collecting();
        let cancel = CancellationToken::new();
        let out = drive_model_call(
            &HungProvider,
            Request::default(),
            &cancel,
            Duration::from_millis(40),
            &sink,
        )
        .await;
        assert!(matches!(out, Err(Failure::TransientTransport(_))));
    }

    #[tokio::test]
    async fn cancel_aborts_stream() {
        let (sink, _log) = collecting();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let out = drive_model_call(
            &HungProvider,
            Request::default(),
            &cancel,
            Duration::from_secs(60),
            &sink,
        )
        .await;
        assert!(matches!(out, Err(Failure::Cancelled)));
    }

    #[tokio::test]
    async fn streams_deltas_live_and_assembles_done() {
        let (sink, log) = collecting();
        let cancel = CancellationToken::new();
        let provider = StreamingProvider {
            calls: AtomicU64::new(0),
        };
        let out = drive_model_call(
            &provider,
            Request::default(),
            &cancel,
            Duration::from_secs(60),
            &sink,
        )
        .await
        .expect("stream completes");
        assert_eq!(out.text, "Hello");
        let log = log.lock().unwrap();
        let text_deltas = log
            .iter()
            .filter(|e| matches!(e, AgentEvent::TextDelta { .. }))
            .count();
        // Exactly the two streamed chunks (no extra re-emit from Done).
        assert_eq!(text_deltas, 2);
        assert!(log.iter().any(|e| matches!(e, AgentEvent::Usage { .. })));
    }

    #[test]
    fn classify_maps_status_codes() {
        let no_hdr = |_: &str| None;
        assert!(matches!(
            classify_api_error(429, |h| (h == "retry-after").then(|| "30".into()), ""),
            Failure::RateLimit { retry_after: Some(d), .. } if d == Duration::from_secs(30)
        ));
        assert!(matches!(classify_api_error(401, no_hdr, ""), Failure::Auth(_)));
        assert!(matches!(classify_api_error(402, no_hdr, ""), Failure::Billing(_)));
        assert!(matches!(
            classify_api_error(413, no_hdr, ""),
            Failure::PayloadTooLarge(_)
        ));
        assert!(matches!(
            classify_api_error(400, no_hdr, "maximum context length is 8192 tokens"),
            Failure::ContextOverflow(_)
        ));
        assert!(matches!(
            classify_api_error(400, no_hdr, "request blocked by content_policy"),
            Failure::ContentPolicy(_)
        ));
        assert!(matches!(
            classify_api_error(400, no_hdr, "bad json"),
            Failure::FormatError(_)
        ));
        assert!(matches!(
            classify_api_error(503, no_hdr, ""),
            Failure::ProviderOverloaded(_)
        ));
        assert!(matches!(
            classify_api_error(500, no_hdr, ""),
            Failure::TransientTransport(_)
        ));
    }

    #[test]
    fn backoff_is_bounded_and_honors_retry_after() {
        let p = policy();
        // Attempt 0 backoff is >= base and <= max(+jitter).
        let b0 = p.backoff(0, None);
        assert!(b0 >= p.backoff_base);
        // A large attempt is capped near the ceiling (allowing jitter).
        let big = p.backoff(40, None);
        assert!(big <= p.backoff_max + p.backoff_max / 4 + Duration::from_millis(1));
        // A server Retry-After larger than the computed backoff wins.
        let ra = p.backoff(0, Some(Duration::from_secs(300)));
        assert!(ra >= Duration::from_secs(300));
    }

    #[test]
    fn decide_bounds_by_budget() {
        let p = policy();
        // Rate limit retries while budget remains, then aborts.
        let rl = Failure::RateLimit {
            retry_after: None,
            message: String::new(),
        };
        assert!(matches!(p.decide(&rl, 0), RecoveryStep::Retry { .. }));
        assert_eq!(p.decide(&rl, p.max_retries), RecoveryStep::Abort);
        // Auth rotates while budget remains, then falls back.
        let auth = Failure::Auth("nope".into());
        assert_eq!(p.decide(&auth, 0), RecoveryStep::Rotate);
        assert_eq!(p.decide(&auth, p.max_retries), RecoveryStep::Fallback);
        // Context overflow compacts; content policy always falls back; fatal always aborts.
        assert_eq!(
            p.decide(&Failure::ContextOverflow("x".into()), 0),
            RecoveryStep::Compact
        );
        assert_eq!(
            p.decide(&Failure::ContentPolicy("x".into()), 0),
            RecoveryStep::Fallback
        );
        assert_eq!(p.decide(&Failure::Fatal("x".into()), 0), RecoveryStep::Abort);
    }
}
