// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! User-feedback export as an OpenTelemetry LOG EVENT (feature `otel`).
//!
//! Users submit thumbs up/down on agent responses and general app feedback; this module turns one
//! such record into an OTel log event (`event.name = "app.feedback"`, severity INFO) and ships it
//! over OTLP/HTTP+protobuf on the same rustls `reqwest` stack as the trace layer ([`crate::otel`]).
//! Unlike the trace layer — which is gated on the operator env var `OTEL_EXPORTER_OTLP_ENDPOINT` —
//! the feedback endpoint is *product* configuration (`telemetry.feedback_endpoint` in the node
//! config); the two are deliberately independent.
//!
//! Consent shapes the lifetime of the exporter:
//! - **opted-in** — telemetry is enabled, so a long-lived [`FeedbackExporter`] may be built once and
//!   reused ([`FeedbackExporter::emit`] per submission, [`FeedbackExporter::flush`] opportunistically).
//! - **explicit-one-shot** — telemetry is otherwise disabled, but the user explicitly submitted, so
//!   [`emit_one_shot`] constructs a scoped provider, emits, flushes, and shuts it down, leaving
//!   nothing persistent enabled.
//!
//! When the `otel` feature is OFF the whole surface degrades to a no-op stub (same signatures) so
//! callers compile unchanged.

/// A plain-data user-feedback record — no `daemon-api` dependency, so the sibling integration phase
/// can map its store/outbox record into this without a cycle. Field spellings mirror the wire
/// vocabulary the sibling workstream produces; this module only knows how to render them as OTel
/// log attributes (see [`crate::fields::daemon::feedback`] + the reused `gen_ai.*` names).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FeedbackEvent {
    /// The feedback category: `"response"` (on a specific agent turn) or `"app"` (general).
    pub kind: String,
    /// The thumbs rating, when given: `"up"` or `"down"`.
    pub rating: Option<String>,
    /// A free-form comment body, when supplied.
    pub comment: Option<String>,
    /// The conversation/session id the feedback relates to (rendered as `gen_ai.conversation.id`).
    pub session_id: Option<String>,
    /// The daemon trace id correlating the feedback with spans/journal (rendered as `trace_id`).
    pub trace_id: Option<String>,
    /// The UI surface the feedback was submitted from.
    pub surface: String,
    /// The consent posture: `"opted-in"` or `"explicit-one-shot"`.
    pub consent: String,
    /// The submitting client/app version, when known.
    pub app_version: Option<String>,
    /// The submitting client operating system, when known.
    pub os: Option<String>,
    /// The exporting `daemon-node` version.
    pub node_version: String,
    /// When the feedback was created, in Unix epoch milliseconds.
    pub created_at_ms: u64,

    // -- optional turn descriptor (present for `kind = "response"`) ------------------------------
    /// The model of the turn the feedback is about (rendered as `gen_ai.request.model`).
    pub model: Option<String>,
    /// The provider of the turn (rendered as `gen_ai.provider.name`).
    pub provider: Option<String>,
    /// The turn's stop/finish reason (rendered as `gen_ai.response.finish_reasons`).
    pub end_reason: Option<String>,
    /// The turn's input/prompt tokens (rendered as `gen_ai.usage.input_tokens`).
    pub input_tokens: Option<u64>,
    /// The turn's output/completion tokens (rendered as `gen_ai.usage.output_tokens`).
    pub output_tokens: Option<u64>,
    /// The rated response text, present only when the submitter consented via `include_content`
    /// (rendered as `daemon.feedback.content`). This is what makes a response thumb self-describing
    /// rather than a bare `(session, cursor)` anchor the consumer cannot resolve.
    pub response_content: Option<String>,
}

/// The OTel log-event name stamped on every exported feedback record (`LogRecord::event_name`).
pub const EVENT_NAME: &str = "app.feedback";

/// A feedback-export error. Kept string-backed (no OTel types leak into the public surface) so the
/// error type is identical whether or not the `otel` feature is compiled in.
#[derive(Debug, thiserror::Error)]
pub enum FeedbackError {
    /// Building the OTLP log exporter / logger provider failed.
    #[error("building the OTLP feedback log exporter failed: {0}")]
    Build(String),
    /// Flushing or shutting down the logger provider failed.
    #[error("exporting the feedback log event failed: {0}")]
    Export(String),
}

/// Result alias for the feedback export surface.
pub type Result<T> = std::result::Result<T, FeedbackError>;

#[cfg(feature = "otel")]
pub use imp::{feedback_log_attributes, FeedbackExporter};
#[cfg(feature = "otel")]
mod imp {
    use super::{FeedbackError, FeedbackEvent, Result, EVENT_NAME};
    use crate::fields;
    use opentelemetry::logs::{
        AnyValue, LogRecord as _, Logger as _, LoggerProvider as _, Severity,
    };
    use opentelemetry::KeyValue;
    use opentelemetry_otlp::{LogExporter, WithExportConfig as _};
    use opentelemetry_sdk::logs::SdkLoggerProvider;
    use opentelemetry_sdk::Resource;

    /// The OTLP resource `service.name` (matches the trace layer's default).
    const SERVICE_NAME: &str = "daemon-node";
    /// The instrumentation scope name for the feedback logger.
    const SCOPE_NAME: &str = "daemon-node";

    /// An OTLP/HTTP log-event exporter for user feedback. Owns an [`SdkLoggerProvider`] with a
    /// batching OTLP log processor; kept alive it exports one event per [`emit`](Self::emit).
    pub struct FeedbackExporter {
        provider: SdkLoggerProvider,
    }

    impl FeedbackExporter {
        /// Build an exporter shipping to `endpoint` (an OTLP/HTTP base URL, e.g.
        /// `http://localhost:4318`). The signal path suffix is appended per the OTLP spec by the
        /// underlying exporter.
        pub fn new(endpoint: &str) -> Result<Self> {
            let exporter = LogExporter::builder()
                .with_http()
                .with_endpoint(endpoint)
                .build()
                .map_err(|err| FeedbackError::Build(err.to_string()))?;

            let resource = Resource::builder()
                .with_service_name(SERVICE_NAME)
                .with_attribute(KeyValue::new(
                    "service.version",
                    daemon_common::VERSION.to_string(),
                ))
                .build();

            let provider = SdkLoggerProvider::builder()
                .with_batch_exporter(exporter)
                .with_resource(resource)
                .build();

            Ok(Self { provider })
        }

        /// Emit one feedback record as an `app.feedback` INFO log event. Non-blocking: the batch
        /// processor exports on its own thread; call [`flush`](Self::flush) to force delivery.
        pub fn emit(&self, event: &FeedbackEvent) {
            let logger = self.provider.logger(SCOPE_NAME);
            let mut record = logger.create_log_record();
            record.set_event_name(EVENT_NAME);
            record.set_severity_number(Severity::Info);
            record.set_severity_text("INFO");
            for (key, value) in feedback_log_attributes(event) {
                record.add_attribute(key, value);
            }
            logger.emit(record);
        }

        /// Force-flush any buffered log events to the collector.
        pub fn flush(&self) -> Result<()> {
            self.provider
                .force_flush()
                .map_err(|err| FeedbackError::Export(err.to_string()))
        }

        /// Flush and shut the provider down, consuming the exporter (the end of the opted-in
        /// exporter's life, or the tail of the one-shot path).
        pub fn shutdown(self) -> Result<()> {
            self.provider
                .shutdown()
                .map_err(|err| FeedbackError::Export(err.to_string()))
        }
    }

    /// The consent-off explicit path: construct a scoped exporter, emit, flush, and shut it down so
    /// nothing persistent is left enabled.
    pub fn emit_one_shot(endpoint: &str, event: &FeedbackEvent) -> Result<()> {
        let exporter = FeedbackExporter::new(endpoint)?;
        exporter.emit(event);
        exporter.flush()?;
        exporter.shutdown()
    }

    /// Map a [`FeedbackEvent`] to its OTel log-record attribute list. Factored out of [`emit`] so
    /// the mapping is unit-testable without a live exporter or a runtime. Semconv-aligned names are
    /// reused where they exist (`gen_ai.*`, `trace_id`); the rest live under `daemon.feedback.*`.
    /// Optional fields are omitted entirely when unset (never emitted as null/empty).
    pub fn feedback_log_attributes(event: &FeedbackEvent) -> Vec<(&'static str, AnyValue)> {
        // Always-present feedback-specific fields.
        let mut attrs: Vec<(&'static str, AnyValue)> = vec![
            (fields::daemon::feedback::KIND, event.kind.clone().into()),
            (
                fields::daemon::feedback::SURFACE,
                event.surface.clone().into(),
            ),
            (
                fields::daemon::feedback::CONSENT,
                event.consent.clone().into(),
            ),
            (
                fields::daemon::feedback::NODE_VERSION,
                event.node_version.clone().into(),
            ),
            (
                fields::daemon::feedback::CREATED_AT_MS,
                AnyValue::Int(i64::try_from(event.created_at_ms).unwrap_or(i64::MAX)),
            ),
        ];

        // Optional feedback-specific fields.
        if let Some(rating) = &event.rating {
            attrs.push((fields::daemon::feedback::RATING, rating.clone().into()));
        }
        if let Some(comment) = &event.comment {
            attrs.push((fields::daemon::feedback::COMMENT, comment.clone().into()));
        }
        if let Some(app_version) = &event.app_version {
            attrs.push((
                fields::daemon::feedback::APP_VERSION,
                app_version.clone().into(),
            ));
        }
        if let Some(os) = &event.os {
            attrs.push((fields::daemon::feedback::OS, os.clone().into()));
        }

        // Semconv-aligned correlation + turn descriptor.
        if let Some(session_id) = &event.session_id {
            attrs.push((fields::gen_ai::CONVERSATION_ID, session_id.clone().into()));
        }
        if let Some(trace_id) = &event.trace_id {
            attrs.push((fields::TRACE_ID, trace_id.clone().into()));
        }
        if let Some(model) = &event.model {
            attrs.push((fields::gen_ai::REQUEST_MODEL, model.clone().into()));
        }
        if let Some(provider) = &event.provider {
            attrs.push((fields::gen_ai::PROVIDER_NAME, provider.clone().into()));
        }
        if let Some(end_reason) = &event.end_reason {
            attrs.push((
                fields::gen_ai::RESPONSE_FINISH_REASONS,
                end_reason.clone().into(),
            ));
        }
        if let Some(input_tokens) = event.input_tokens {
            attrs.push((
                fields::gen_ai::USAGE_INPUT_TOKENS,
                AnyValue::Int(i64::try_from(input_tokens).unwrap_or(i64::MAX)),
            ));
        }
        if let Some(output_tokens) = event.output_tokens {
            attrs.push((
                fields::gen_ai::USAGE_OUTPUT_TOKENS,
                AnyValue::Int(i64::try_from(output_tokens).unwrap_or(i64::MAX)),
            ));
        }
        if let Some(response_content) = &event.response_content {
            attrs.push((
                fields::daemon::feedback::CONTENT,
                response_content.clone().into(),
            ));
        }

        attrs
    }
}

#[cfg(not(feature = "otel"))]
pub use stub::FeedbackExporter;
#[cfg(not(feature = "otel"))]
mod stub {
    use super::{FeedbackEvent, Result};

    /// No-op feedback exporter compiled when the `otel` feature is off, so callers link and run
    /// unchanged with feedback export inert (the config endpoint is ignored).
    pub struct FeedbackExporter {
        _private: (),
    }

    impl FeedbackExporter {
        /// No-op: succeeds without building anything.
        pub fn new(_endpoint: &str) -> Result<Self> {
            Ok(Self { _private: () })
        }

        /// No-op: drops the event.
        pub fn emit(&self, _event: &FeedbackEvent) {}

        /// No-op: succeeds.
        pub fn flush(&self) -> Result<()> {
            Ok(())
        }

        /// No-op: succeeds.
        pub fn shutdown(self) -> Result<()> {
            Ok(())
        }
    }
}

/// No-op one-shot path when the `otel` feature is off (mirrors [`imp::emit_one_shot`]).
#[cfg(not(feature = "otel"))]
pub fn emit_one_shot(_endpoint: &str, _event: &FeedbackEvent) -> Result<()> {
    Ok(())
}
#[cfg(feature = "otel")]
pub use imp::emit_one_shot;

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_response_event() -> FeedbackEvent {
        FeedbackEvent {
            kind: "response".to_string(),
            rating: Some("up".to_string()),
            comment: Some("great answer".to_string()),
            session_id: Some("sess-123".to_string()),
            trace_id: Some("0123abcd".to_string()),
            surface: "transcript".to_string(),
            consent: "opted-in".to_string(),
            app_version: Some("1.2.3".to_string()),
            os: Some("linux".to_string()),
            node_version: "0.0.1".to_string(),
            created_at_ms: 1_700_000_000_000,
            model: Some("claude-opus-4-8".to_string()),
            provider: Some("anthropic".to_string()),
            end_reason: Some("stop".to_string()),
            input_tokens: Some(42),
            output_tokens: Some(7),
            response_content: Some("the rated reply text".to_string()),
        }
    }

    // Without the `otel` feature only the no-op stub exists; prove it constructs, emits, and the
    // one-shot path returns Ok (the whole surface compiles + links inert).
    #[cfg(not(feature = "otel"))]
    #[test]
    fn stub_surface_is_a_noop() {
        let event = sample_response_event();
        let exporter = FeedbackExporter::new("http://localhost:4318").expect("stub new is Ok");
        exporter.emit(&event);
        exporter.flush().expect("stub flush is Ok");
        exporter.shutdown().expect("stub shutdown is Ok");
        emit_one_shot("http://localhost:4318", &event).expect("stub one-shot is Ok");
    }

    #[cfg(feature = "otel")]
    mod otel_tests {
        use super::super::{feedback_log_attributes, FeedbackEvent, EVENT_NAME};
        use crate::fields;
        use opentelemetry::logs::AnyValue;
        use std::collections::HashMap;

        fn attrs_map(event: &FeedbackEvent) -> HashMap<&'static str, AnyValue> {
            feedback_log_attributes(event).into_iter().collect()
        }

        fn as_str(value: &AnyValue) -> Option<&str> {
            match value {
                AnyValue::String(s) => Some(s.as_str()),
                _ => None,
            }
        }

        fn as_int(value: &AnyValue) -> Option<i64> {
            match value {
                AnyValue::Int(i) => Some(*i),
                _ => None,
            }
        }

        #[test]
        fn event_name_is_stable() {
            assert_eq!(EVENT_NAME, "app.feedback");
        }

        #[test]
        fn response_event_maps_all_fields_with_semconv_names() {
            let event = super::sample_response_event();
            let map = attrs_map(&event);

            // Feedback-specific `daemon.feedback.*` names.
            assert_eq!(
                as_str(&map[fields::daemon::feedback::KIND]),
                Some("response")
            );
            assert_eq!(
                as_str(&map[fields::daemon::feedback::SURFACE]),
                Some("transcript")
            );
            assert_eq!(
                as_str(&map[fields::daemon::feedback::CONSENT]),
                Some("opted-in")
            );
            assert_eq!(
                as_str(&map[fields::daemon::feedback::NODE_VERSION]),
                Some("0.0.1")
            );
            assert_eq!(
                as_int(&map[fields::daemon::feedback::CREATED_AT_MS]),
                Some(1_700_000_000_000)
            );
            assert_eq!(as_str(&map[fields::daemon::feedback::RATING]), Some("up"));
            assert_eq!(
                as_str(&map[fields::daemon::feedback::COMMENT]),
                Some("great answer")
            );
            assert_eq!(
                as_str(&map[fields::daemon::feedback::APP_VERSION]),
                Some("1.2.3")
            );
            assert_eq!(as_str(&map[fields::daemon::feedback::OS]), Some("linux"));

            // Semconv-aligned names reused from the trace vocabulary.
            assert_eq!(
                as_str(&map[fields::gen_ai::CONVERSATION_ID]),
                Some("sess-123")
            );
            assert_eq!(as_str(&map[fields::TRACE_ID]), Some("0123abcd"));
            assert_eq!(
                as_str(&map[fields::gen_ai::REQUEST_MODEL]),
                Some("claude-opus-4-8")
            );
            assert_eq!(
                as_str(&map[fields::gen_ai::PROVIDER_NAME]),
                Some("anthropic")
            );
            assert_eq!(
                as_str(&map[fields::gen_ai::RESPONSE_FINISH_REASONS]),
                Some("stop")
            );
            assert_eq!(as_int(&map[fields::gen_ai::USAGE_INPUT_TOKENS]), Some(42));
            assert_eq!(as_int(&map[fields::gen_ai::USAGE_OUTPUT_TOKENS]), Some(7));
            assert_eq!(
                as_str(&map[fields::daemon::feedback::CONTENT]),
                Some("the rated reply text")
            );
        }

        #[test]
        fn optional_fields_are_omitted_when_unset() {
            let event = FeedbackEvent {
                kind: "app".to_string(),
                surface: "settings".to_string(),
                consent: "explicit-one-shot".to_string(),
                node_version: "0.0.1".to_string(),
                created_at_ms: 1,
                ..FeedbackEvent::default()
            };
            let map = attrs_map(&event);

            // Present: the required fields.
            assert!(map.contains_key(fields::daemon::feedback::KIND));
            assert!(map.contains_key(fields::daemon::feedback::SURFACE));
            assert!(map.contains_key(fields::daemon::feedback::CONSENT));
            assert!(map.contains_key(fields::daemon::feedback::NODE_VERSION));
            assert!(map.contains_key(fields::daemon::feedback::CREATED_AT_MS));

            // Absent: every optional field.
            assert!(!map.contains_key(fields::daemon::feedback::RATING));
            assert!(!map.contains_key(fields::daemon::feedback::COMMENT));
            assert!(!map.contains_key(fields::daemon::feedback::APP_VERSION));
            assert!(!map.contains_key(fields::daemon::feedback::OS));
            assert!(!map.contains_key(fields::gen_ai::CONVERSATION_ID));
            assert!(!map.contains_key(fields::TRACE_ID));
            assert!(!map.contains_key(fields::gen_ai::REQUEST_MODEL));
            assert!(!map.contains_key(fields::gen_ai::PROVIDER_NAME));
            assert!(!map.contains_key(fields::gen_ai::RESPONSE_FINISH_REASONS));
            assert!(!map.contains_key(fields::gen_ai::USAGE_INPUT_TOKENS));
            assert!(!map.contains_key(fields::gen_ai::USAGE_OUTPUT_TOKENS));
            assert!(!map.contains_key(fields::daemon::feedback::CONTENT));
        }
    }
}
