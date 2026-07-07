// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The user-feedback outbox drain (N1 → N2 integration): the piece both feature branches
//! deliberately left unwired.
//!
//! N1 (`feat/feedback-node-api`) enqueues a [`FeedbackRecord`] onto the durable outbox on every
//! `FeedbackSubmit`, tagging each with a consent provenance string (`"opted-in"` when the global
//! telemetry toggle was on at submit time, else `"explicit-one-shot"`). N2 (`feat/feedback-otel`)
//! renders one [`FeedbackEvent`] as an `app.feedback` OTLP log event. This module is the bridge:
//! it maps a stored record into a [`FeedbackEvent`], ships it through the configured
//! `telemetry.feedback_endpoint`, and marks the record delivered on success.
//!
//! Export policy:
//! - No endpoint configured (or the `otel` feature is off): the drain is inert — records simply
//!   stay queued, no error spam. This is the default workspace build.
//! - `"explicit-one-shot"` records use [`emit_one_shot`](daemon_telemetry::feedback::emit_one_shot)
//!   (a scoped provider, nothing persistent left enabled).
//! - `"opted-in"` records reuse a lazily-built, long-lived
//!   [`FeedbackExporter`](daemon_telemetry::feedback::FeedbackExporter).
//!
//! On export success a record is [`feedback_mark_delivered`](daemon_store::SessionStore::feedback_mark_delivered);
//! on failure it is left queued (retried on the next drain trigger) with a rate-limited warn.
//!
//! Triggers: a best-effort drain after each successful `FeedbackSubmit` enqueue
//! ([`NodeApiImpl::spawn_feedback_drain`]) and once at node startup (leftover records from a
//! previous run).

use super::*;

use daemon_telemetry::feedback::FeedbackEvent;

/// The seam the drain ships one mapped feedback event through. Production wires the OTLP exporter
/// (see [`drain_for_endpoint`]); tests inject a stub. `Ok(())` means "ship succeeded — mark the
/// record delivered"; `Err(reason)` means "leave it queued for the next drain".
pub(crate) trait FeedbackExport: Send + Sync {
    /// Ship one feedback event to the collector.
    fn export(&self, event: &FeedbackEvent) -> Result<(), String>;
}

/// The wired feedback-outbox drain: the export seam plus a rate-limit clock for the failure warn.
/// `None` on [`NodeApiImpl`] means export is inert (no endpoint / the `otel` feature is off).
pub(crate) struct FeedbackDrain {
    /// How one mapped event reaches the collector.
    export: Arc<dyn FeedbackExport>,
    /// Epoch-ms of the last emitted failure warn (`0` = never), so a burst of failing drains logs
    /// at most once per [`WARN_WINDOW_MS`] window.
    last_warn_ms: std::sync::atomic::AtomicI64,
}

/// The minimum gap between two feedback-export failure warns (rate-limit window).
const WARN_WINDOW_MS: i64 = 60_000;

impl FeedbackDrain {
    /// Build a drain over an export seam (production: the OTLP seam; tests: a stub).
    // Only constructed behind the `otel` feature (via `drain_for_endpoint`) or in unit tests; the
    // default build wires no exporter, so the constructor is unreachable there.
    #[cfg_attr(not(feature = "otel"), allow(dead_code))]
    pub(crate) fn new(export: Arc<dyn FeedbackExport>) -> Self {
        Self {
            export,
            last_warn_ms: std::sync::atomic::AtomicI64::new(0),
        }
    }

    /// Emit a rate-limited warn for a failed export (at most once per [`WARN_WINDOW_MS`]).
    fn note_failure(&self, reason: &str) {
        use std::sync::atomic::Ordering::Relaxed;
        let now = now_ms();
        let last = self.last_warn_ms.load(Relaxed);
        if last == 0 || now.saturating_sub(last) >= WARN_WINDOW_MS {
            self.last_warn_ms.store(now, Relaxed);
            tracing::warn!(
                error = %reason,
                "feedback export failed; leaving records queued for the next drain"
            );
        }
    }
}

/// Map a stored [`FeedbackRecord`] into the telemetry [`FeedbackEvent`] N2 renders as an
/// `app.feedback` log event. The consent provenance string is carried through verbatim (the
/// exporter branches on it), the trace id is rendered as fixed-width hex (matching
/// [`daemon_common::TraceId`]'s `Display`), and the node version is filled from the record (which
/// stamped it from [`daemon_common::VERSION`] at accept time), falling back to the live
/// [`daemon_common::VERSION`] for any record that stored none. The turn-descriptor enrichment
/// (model/provider/end-reason/usage from the journal at the stored cursor) is out of scope for this
/// drain and left unset.
pub(crate) fn feedback_event_from(record: &FeedbackRecord) -> FeedbackEvent {
    FeedbackEvent {
        kind: record.kind.clone(),
        rating: record.rating.clone(),
        comment: record.comment.clone(),
        session_id: record.session.clone(),
        trace_id: record.trace.map(|t| format!("{t:016x}")),
        surface: record.surface.clone(),
        consent: record.consent.clone(),
        app_version: record.app_version.clone(),
        os: record.os.clone(),
        node_version: if record.node_version.is_empty() {
            daemon_common::VERSION.to_string()
        } else {
            record.node_version.clone()
        },
        created_at_ms: u64::try_from(record.created_at_ms).unwrap_or(0),
        model: None,
        provider: None,
        end_reason: None,
        input_tokens: None,
        output_tokens: None,
    }
}

/// Drain the durable feedback outbox once through `drain`: map each queued record to a
/// [`FeedbackEvent`], export it, and mark it delivered on success. A `None` drain (no endpoint / the
/// `otel` feature off) is a no-op — records stay queued. A single export failure stops the pass
/// (the remaining records stay queued and retry on the next trigger). Factored out of
/// [`NodeApiImpl::drain_feedback_outbox`] so it is unit-testable with an in-memory store + a stub
/// export seam, no live collector required.
pub(crate) async fn drain_outbox(store: &Arc<dyn SessionStore>, drain: Option<&FeedbackDrain>) {
    let Some(drain) = drain else {
        tracing::debug!(
            "feedback export unavailable (no endpoint or the `otel` feature is off); records stay queued"
        );
        return;
    };
    // `0` = every pending record, oldest first.
    for record in store.feedback_pending(0).await {
        let event = feedback_event_from(&record);
        match drain.export.export(&event) {
            Ok(()) => {
                if let Err(err) = store.feedback_mark_delivered(&record.id).await {
                    tracing::warn!(id = %record.id, %err, "feedback mark-delivered failed after a successful export");
                }
            }
            Err(reason) => {
                drain.note_failure(&reason);
                break;
            }
        }
    }
}

/// Build the drain for a configured endpoint, gating the real OTLP exporter behind the `otel`
/// feature. `None` endpoint — or a build without `otel` — yields `None` (export inert), so the
/// default workspace build queues records and never ships them.
pub(crate) fn drain_for_endpoint(endpoint: Option<String>) -> Option<Arc<FeedbackDrain>> {
    let endpoint = endpoint?;
    #[cfg(feature = "otel")]
    {
        Some(Arc::new(FeedbackDrain::new(Arc::new(
            OtlpFeedbackExport::new(endpoint),
        ))))
    }
    #[cfg(not(feature = "otel"))]
    {
        let _ = endpoint;
        tracing::debug!(
            "telemetry.feedback_endpoint is set but the `otel` feature is off; feedback export is inert (records stay queued)"
        );
        None
    }
}

/// The OTLP-backed export seam (compiled only with the `otel` feature). Holds the endpoint and, for
/// opted-in records, a lazily-built reusable exporter; explicit-one-shot records take the scoped
/// one-shot path so nothing persistent is left enabled.
#[cfg(feature = "otel")]
struct OtlpFeedbackExport {
    endpoint: String,
    reusable: std::sync::Mutex<Option<daemon_telemetry::feedback::FeedbackExporter>>,
}

#[cfg(feature = "otel")]
impl OtlpFeedbackExport {
    fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            reusable: std::sync::Mutex::new(None),
        }
    }
}

#[cfg(feature = "otel")]
impl FeedbackExport for OtlpFeedbackExport {
    fn export(&self, event: &FeedbackEvent) -> Result<(), String> {
        use daemon_telemetry::feedback::{emit_one_shot, FeedbackExporter};
        // Explicit-one-shot: telemetry is otherwise off, but the user explicitly submitted — emit
        // through a scoped provider that is torn down immediately.
        if event.consent == "explicit-one-shot" {
            return emit_one_shot(&self.endpoint, event).map_err(|e| e.to_string());
        }
        // Opted-in (or any other provenance): reuse a long-lived exporter, built on first use.
        let mut guard = self.reusable.lock().unwrap();
        if guard.is_none() {
            *guard = Some(FeedbackExporter::new(&self.endpoint).map_err(|e| e.to_string())?);
        }
        let exporter = guard.as_ref().expect("exporter just built");
        exporter.emit(event);
        exporter.flush().map_err(|e| e.to_string())
    }
}

/// Unix epoch milliseconds (saturating; a clock before the epoch reads as `0`).
fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

impl NodeApiImpl {
    /// Drain the durable feedback outbox once to the configured OTLP endpoint (N1 → N2). A no-op
    /// when export is inert (no `telemetry.feedback_endpoint` / the `otel` feature is off): records
    /// stay queued.
    pub(crate) async fn drain_feedback_outbox(&self) {
        drain_outbox(&self.store, self.feedback_drain.as_deref()).await;
    }

    /// Spawn a detached, best-effort feedback-outbox drain — the trigger the `FeedbackSubmit`
    /// handler fires after each enqueue and the binary fires once at startup. Cheap no-op when
    /// export is inert (nothing is spawned), so it never floods the runtime on a node without a
    /// feedback endpoint.
    pub fn spawn_feedback_drain(&self) {
        if self.feedback_drain.is_none() {
            return;
        }
        let this = self.clone();
        tokio::spawn(async move { this.drain_feedback_outbox().await });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_store::InMemoryStore;

    fn record(id: &str, consent: &str) -> FeedbackRecord {
        FeedbackRecord {
            id: id.into(),
            created_at_ms: 1,
            kind: "app".into(),
            rating: Some("up".into()),
            comment: Some("hi".into()),
            include_content: false,
            session: None,
            cursor: None,
            trace: Some(0xABCD),
            surface: "settings".into(),
            app_version: Some("1.0".into()),
            os: Some("linux".into()),
            consent: consent.into(),
            node_version: "test-ver".into(),
            delivered: false,
        }
    }

    #[test]
    fn mapping_carries_provenance_and_renders_trace_as_hex() {
        let event = feedback_event_from(&record("fb-1", "opted-in"));
        assert_eq!(
            event.consent, "opted-in",
            "consent provenance is carried through"
        );
        assert_eq!(
            event.node_version, "test-ver",
            "the record's node version is used"
        );
        assert_eq!(
            event.trace_id.as_deref(),
            Some("000000000000abcd"),
            "trace id renders as fixed-width hex (TraceId::Display parity)"
        );
        assert_eq!(event.kind, "app");
        assert_eq!(event.created_at_ms, 1);
        // Turn-descriptor enrichment is out of scope for the drain.
        assert!(event.model.is_none() && event.input_tokens.is_none());
    }

    #[test]
    fn mapping_falls_back_to_common_version_when_record_stored_none() {
        let mut r = record("fb-x", "opted-in");
        r.node_version = String::new();
        let event = feedback_event_from(&r);
        assert_eq!(event.node_version, daemon_common::VERSION);
    }

    /// A stub export seam: records the consent of every exported event; optionally fails.
    struct StubExport {
        fail: bool,
        calls: std::sync::Mutex<Vec<String>>,
    }

    impl FeedbackExport for StubExport {
        fn export(&self, event: &FeedbackEvent) -> Result<(), String> {
            self.calls.lock().unwrap().push(event.consent.clone());
            if self.fail {
                Err("stub failure".into())
            } else {
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn drain_marks_delivered_when_export_succeeds() {
        let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
        store
            .feedback_enqueue(record("fb-a", "explicit-one-shot"))
            .await
            .expect("enqueue a");
        store
            .feedback_enqueue(record("fb-b", "opted-in"))
            .await
            .expect("enqueue b");

        let stub = Arc::new(StubExport {
            fail: false,
            calls: Default::default(),
        });
        let drain = FeedbackDrain::new(stub.clone());
        drain_outbox(&store, Some(&drain)).await;

        assert!(
            store.feedback_pending(0).await.is_empty(),
            "every successfully-exported record drops out of pending"
        );
        assert_eq!(
            stub.calls.lock().unwrap().as_slice(),
            ["explicit-one-shot", "opted-in"],
            "both records were exported, oldest first"
        );
    }

    #[tokio::test]
    async fn drain_is_a_noop_when_export_is_inert() {
        // Endpoint unset / otel off => `None` drain: records must simply stay queued.
        let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
        store
            .feedback_enqueue(record("fb-a", "opted-in"))
            .await
            .expect("enqueue");
        drain_outbox(&store, None).await;
        assert_eq!(
            store.feedback_pending(0).await.len(),
            1,
            "records stay queued when no exporter is wired"
        );
    }

    #[tokio::test]
    async fn drain_leaves_records_queued_on_export_failure() {
        let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
        store
            .feedback_enqueue(record("fb-a", "opted-in"))
            .await
            .expect("enqueue");
        let drain = FeedbackDrain::new(Arc::new(StubExport {
            fail: true,
            calls: Default::default(),
        }));
        drain_outbox(&store, Some(&drain)).await;
        assert_eq!(
            store.feedback_pending(0).await.len(),
            1,
            "a failed export keeps the record queued for the next drain"
        );
    }
}
