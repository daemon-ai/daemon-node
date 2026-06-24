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

pub mod fields;
pub mod journal;
pub mod metrics;
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
