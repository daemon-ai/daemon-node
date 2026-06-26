//! `TraceId` helpers for structured `tracing` spans/events.
//!
//! The task-local [`TraceId`](daemon_common::TraceId) remains the source of truth for wire
//! propagation and journal stamping. These helpers make the operational span layer follow that same
//! context without forcing every caller to repeat the `trace_id = %current_trace()` boilerplate.

use crate::trace::{current_trace, set_trace, with_trace};
use daemon_common::TraceId;
use std::future::Future;
use tracing::{Instrument, Level, Span};

/// Broad category of span being opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanKind {
    /// External API or wire boundary.
    Boundary,
    /// Host/substrate lifecycle work.
    Lifecycle,
    /// Engine turn or model/tool work.
    Execution,
    /// Durable store operation.
    Store,
    /// Credential acquire/use/deny path.
    Credential,
    /// Network or placement transport.
    Transport,
    /// Resident service tick.
    Resident,
    /// Verifiable journal append/seal path.
    Journal,
}

impl SpanKind {
    /// Stable lowercase label for span fields.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Boundary => "boundary",
            Self::Lifecycle => "lifecycle",
            Self::Execution => "execution",
            Self::Store => "store",
            Self::Credential => "credential",
            Self::Transport => "transport",
            Self::Resident => "resident",
            Self::Journal => "journal",
        }
    }
}

/// Return `incoming` when it is nonzero, otherwise mint a fresh ingress trace.
pub fn ingress_trace(incoming: Option<TraceId>) -> TraceId {
    incoming
        .filter(|trace| !trace.is_none())
        .unwrap_or_else(TraceId::generate)
}

/// Open a span carrying the current task-local trace id.
pub fn trace_span(level: Level, name: &'static str, kind: SpanKind) -> Span {
    match level {
        Level::ERROR => tracing::error_span!(
            "trace.scope",
            span.name = name,
            trace_id = %current_trace(),
            span.kind = kind.as_str()
        ),
        Level::WARN => tracing::warn_span!(
            "trace.scope",
            span.name = name,
            trace_id = %current_trace(),
            span.kind = kind.as_str()
        ),
        Level::INFO => tracing::info_span!(
            "trace.scope",
            span.name = name,
            trace_id = %current_trace(),
            span.kind = kind.as_str()
        ),
        Level::DEBUG => tracing::debug_span!(
            "trace.scope",
            span.name = name,
            trace_id = %current_trace(),
            span.kind = kind.as_str()
        ),
        Level::TRACE => tracing::trace_span!(
            "trace.scope",
            span.name = name,
            trace_id = %current_trace(),
            span.kind = kind.as_str()
        ),
    }
}

/// Restore the task-local trace id and return a boundary span carrying it.
///
/// This is intended for receive/decode sites that already run inside a `with_trace` scope. Outside
/// a scope, the task-local restore is a no-op, but the returned span still carries the supplied id.
pub fn restore_trace_span(trace: TraceId, name: &'static str, kind: SpanKind) -> Span {
    set_trace(trace);
    tracing::debug_span!(
        "trace.restore",
        span.name = name,
        trace_id = %trace,
        span.kind = kind.as_str()
    )
}

/// Run `fut` inside both a task-local trace scope and a `tracing` span.
pub async fn with_trace_span<F>(
    trace: TraceId,
    name: &'static str,
    kind: SpanKind,
    fut: F,
) -> F::Output
where
    F: Future,
{
    let span = tracing::info_span!(
        "trace.scope",
        span.name = name,
        trace_id = %trace,
        span.kind = kind.as_str()
    );
    with_trace(trace, fut.instrument(span)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::current_trace;
    use std::sync::{Arc, Mutex};
    use tracing::field::{Field, Visit};
    use tracing::{span, Id, Subscriber};
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;
    use tracing_subscriber::{Layer, Registry};

    #[test]
    fn ingress_trace_preserves_nonzero() {
        let trace = TraceId(0xCAFE);
        assert_eq!(ingress_trace(Some(trace)), trace);
    }

    #[test]
    fn ingress_trace_replaces_none() {
        let generated = ingress_trace(Some(TraceId::NONE));
        assert!(!generated.is_none());
        let generated = ingress_trace(None);
        assert!(!generated.is_none());
    }

    #[tokio::test]
    async fn with_trace_span_sets_task_local_trace() {
        let trace = TraceId(0x1234);
        with_trace_span(trace, "test.trace_scope", SpanKind::Boundary, async {
            assert_eq!(current_trace(), trace);
        })
        .await;
        assert_eq!(current_trace(), TraceId::NONE);
    }

    #[tokio::test]
    async fn restore_trace_span_updates_task_local_trace() {
        let first = TraceId(0x1111);
        let second = TraceId(0x2222);
        with_trace_span(first, "test.restore_scope", SpanKind::Boundary, async {
            let _span = restore_trace_span(second, "test.restore", SpanKind::Boundary);
            assert_eq!(current_trace(), second);
        })
        .await;
    }

    #[derive(Default)]
    struct CapturedFields {
        span_name: Option<String>,
        trace_id: Option<String>,
    }

    impl Visit for CapturedFields {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            let rendered = format!("{value:?}").trim_matches('"').to_string();
            match field.name() {
                "span.name" => self.span_name = Some(rendered),
                "trace_id" => self.trace_id = Some(rendered),
                _ => {}
            }
        }
    }

    #[derive(Clone)]
    struct CaptureLayer {
        // Test-only capture buffer; the tuple shape is local and self-explanatory.
        #[allow(clippy::type_complexity)]
        spans: Arc<Mutex<Vec<(String, Option<String>, Option<String>)>>>,
    }

    impl<S> Layer<S> for CaptureLayer
    where
        S: Subscriber,
        S: for<'lookup> LookupSpan<'lookup>,
    {
        fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
            let mut fields = CapturedFields::default();
            attrs.record(&mut fields);
            if let Some(span) = ctx.span(id) {
                self.spans.lock().unwrap().push((
                    span.metadata().name().to_string(),
                    fields.span_name,
                    fields.trace_id,
                ));
            }
        }
    }

    #[tokio::test]
    async fn with_trace_span_records_trace_id_field() {
        let spans = Arc::new(Mutex::new(Vec::new()));
        let subscriber = Registry::default().with(CaptureLayer {
            spans: spans.clone(),
        });
        let _guard = tracing::subscriber::set_default(subscriber);

        let trace = TraceId(0x1234);
        with_trace_span(trace, "test.captured", SpanKind::Boundary, async {}).await;

        let spans = spans.lock().unwrap();
        assert!(spans.iter().any(|(metadata_name, span_name, trace_id)| {
            metadata_name == "trace.scope"
                && span_name.as_deref() == Some("test.captured")
                && trace_id.as_deref() == Some(&trace.to_string())
        }));
    }
}
