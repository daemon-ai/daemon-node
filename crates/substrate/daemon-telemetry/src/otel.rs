// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! OpenTelemetry OTLP trace export (feature `otel`).
//!
//! Builds an OTLP/HTTP+protobuf span exporter (over the tree's existing rustls `reqwest` — no
//! gRPC/tonic), wraps it in a batching [`SdkTracerProvider`], and returns a
//! [`tracing-opentelemetry`](tracing_opentelemetry) layer that maps the engine's `tracing` spans
//! (including the `gen_ai.*` attributes) to OTel spans.
//!
//! Export is two-gated: the `otel` build feature must be on **and** `OTEL_EXPORTER_OTLP_ENDPOINT`
//! must be set at runtime; otherwise [`otel_layer`] returns `None` and only the `fmt` layer runs.
//! The returned [`SdkTracerProvider`] must be kept alive for the process lifetime and
//! `shutdown()`-flushed on exit (the caller holds it in the telemetry guard).

use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_otlp::SpanExporter;
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// Build the OTLP export layer + its owning provider, or `None` when no endpoint is configured (so
/// a build with `--features otel` still runs as a plain `fmt`-only process until an endpoint is set).
///
/// Endpoint/headers/timeout are read from the standard `OTEL_EXPORTER_OTLP_*` environment variables
/// (default `http://localhost:4318` for HTTP); the service name comes from `OTEL_SERVICE_NAME`
/// (default `daemon-node`).
pub(crate) fn otel_layer<S>() -> Option<(impl Layer<S>, SdkTracerProvider)>
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    // Runtime gate: no endpoint => do not install the exporter.
    std::env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT")?;

    let exporter = match SpanExporter::builder().with_http().build() {
        Ok(exporter) => exporter,
        Err(err) => {
            // The subscriber is not installed yet, so log to stderr directly (matching the fmt sink).
            eprintln!("daemon-telemetry: OTLP span exporter init failed: {err}");
            return None;
        }
    };

    let service_name =
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "daemon-node".to_string());
    let resource = Resource::builder()
        .with_service_name(service_name)
        .with_attribute(KeyValue::new(
            "service.version",
            daemon_common::VERSION.to_string(),
        ))
        .build();

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();

    let tracer = provider.tracer("daemon-node");
    let layer = tracing_opentelemetry::layer().with_tracer(tracer);
    Some((layer, provider))
}
