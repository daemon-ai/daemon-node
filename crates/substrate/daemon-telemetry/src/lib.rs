//! `daemon-telemetry` — operational surface: tracing, metrics, health.
//!
//! Shared `tracing` setup and health/metrics hooks consumed by the host and bins. Depends only on
//! `daemon-common`.

#![forbid(unsafe_code)]

// TODO: tracing/metrics init + health reporting hooks.
