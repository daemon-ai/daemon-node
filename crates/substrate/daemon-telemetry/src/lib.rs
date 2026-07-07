// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-telemetry` — the operational + verifiable-trace surface.
//!
//! Three concerns live here, layered from cheap to cryptographic:
//! - [`trace`] — an elfo-style task-local *trace scope*. A [`TraceId`](daemon_common::TraceId) is
//!   stamped onto every outbound frame from the sender's [`current_trace`](trace::current_trace)
//!   and *restored* on the receiver via [`set_trace`](trace::set_trace), so logs, spans, and the
//!   journal correlate across a placement cut or a network hop ("context rides every message").
//! - [`metrics`] — a lightweight in-tree aggregator that folds `UsageDelta` up the tree and
//!   renders a serializable [`Dump`](metrics::Dump) (the resident health/metrics surface).
//! - [`journal`] — the verifiable trace journal: each event becomes a Gordian Envelope encoded as
//!   deterministic CBOR, folded into a per-`(session, epoch)` Merkle root, chained across epochs,
//!   and signed (theater-style hash chain upgraded to a real digest tree + ed25519).
//!
//! Layering: this crate owns the whole crypto stack (`dcbor` + `bc-envelope` + `bc-components`).
//! `daemon-common` (DAG root) holds only the opaque value types (`TraceId`, `ContentHash`,
//! `MerkleRoot`); `daemon-store` persists roots as bytes without a crypto dependency.

#![forbid(unsafe_code)]

pub mod feedback;
pub mod fields;
pub mod journal;
pub mod metrics;
#[cfg(feature = "otel")]
mod otel;
pub mod spans;
pub mod trace;

pub use journal::{
    decode_entry, encode_entry, segment_root, verify_segment, JournalEntryView, JournalPayload,
    SegmentInput, TraceSigner, VerifyError, VerifyingKey, GENESIS_ROOT,
};
pub use metrics::{Dump, Metrics};
pub use spans::{ingress_trace, restore_trace_span, trace_span, with_trace_span, SpanKind};
pub use trace::{current_trace, set_trace, with_trace};

use std::sync::Once;

static SUBSCRIBER: Once = Once::new();

/// Install the process tracing subscriber (idempotent; safe to call from every binary role and
/// from tests). A `fmt` layer with an `RUST_LOG` env filter (defaulting to `info`); trace context
/// is surfaced by opening spans with `trace_id = %current_trace()` at message boundaries (see
/// [`trace`]). Calling more than once — or after another subscriber is set — is a no-op.
///
/// Logs are written to **stderr**: a placement-cut child uses stdout as its framed transport, so
/// the diagnostic stream must not collide with it.
pub fn init_subscriber() {
    SUBSCRIBER.call_once(|| {
        use tracing_subscriber::{fmt, EnvFilter};
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        // `try_init` returns Err if a global subscriber is already installed; ignore it so this
        // composes with hosts/tests that set their own.
        let _ = fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_writer(std::io::stderr)
            .try_init();
    });
}

/// Holds process-lifetime telemetry resources; flushing them on drop.
///
/// When the `otel` feature is on and an OTLP endpoint is configured, this owns the
/// `SdkTracerProvider` and calls `shutdown()` on drop so the batch exporter flushes any buffered
/// spans before the process exits. Without the feature (or without an endpoint) it is a zero-sized
/// no-op. Hold it for the whole process (bind it in `main`, e.g. `let _telemetry = init_telemetry();`).
#[derive(Default)]
pub struct TelemetryGuard {
    #[cfg(feature = "otel")]
    provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
}

impl TelemetryGuard {
    /// Whether an OpenTelemetry exporter is actually installed (feature on + endpoint configured).
    /// The host uses this to switch on GenAI span content capture only when export is live.
    pub fn is_exporting(&self) -> bool {
        #[cfg(feature = "otel")]
        {
            self.provider.is_some()
        }
        #[cfg(not(feature = "otel"))]
        {
            false
        }
    }
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        #[cfg(feature = "otel")]
        if let Some(provider) = self.provider.take() {
            // Flush buffered spans; a shutdown error at exit is not actionable, so it is ignored.
            let _ = provider.shutdown();
        }
    }
}

/// Install the process tracing subscriber and (when built with `--features otel` and
/// `OTEL_EXPORTER_OTLP_ENDPOINT` is set) an OpenTelemetry OTLP export layer alongside the `fmt`
/// layer. Idempotent via the same [`Once`] as [`init_subscriber`] (whichever runs first wins), so a
/// binary that wants OTLP export should call this once at startup **before** any other telemetry
/// init. Returns a [`TelemetryGuard`] the caller must keep alive to flush the exporter on shutdown.
pub fn init_telemetry() -> TelemetryGuard {
    #[allow(unused_mut)]
    let mut guard = TelemetryGuard::default();
    SUBSCRIBER.call_once(|| {
        use tracing_subscriber::prelude::*;
        use tracing_subscriber::{fmt, EnvFilter};
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let fmt_layer = fmt::layer().with_target(true).with_writer(std::io::stderr);

        #[cfg(feature = "otel")]
        {
            if let Some((otel_layer, provider)) = otel::otel_layer() {
                let _ = tracing_subscriber::registry()
                    .with(filter)
                    .with(fmt_layer)
                    .with(otel_layer)
                    .try_init();
                guard.provider = Some(provider);
                return;
            }
        }

        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .try_init();
    });
    guard
}
