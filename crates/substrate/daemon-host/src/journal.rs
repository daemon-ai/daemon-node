//! Wiring the `ManageEvent` stream into the verifiable trace journal (host-spec §5; phase 6b).
//!
//! A [`JournalSink`] binds a `(session, epoch)` segment to the authoritative [`SessionStore`] and a
//! [`TraceSigner`]. As a unit's events stream by, [`JournalSink::record`] turns each into a Gordian
//! Envelope (via [`daemon_telemetry`]) and **appends** it durably. At the suspension/completion
//! boundary [`JournalSink::seal`] folds the segment into a signed [`MerkleRoot`] and commits it
//! **under the same fence the checkpoint commits under** — so the trace root is bound to exactly
//! that durable incarnation, and a stale incarnation can neither append progress nor seal a root.
//!
//! Epochs chain: a sink for epoch *N+1* is created [`JournalSink::chained`] onto epoch *N*'s sealed
//! root, giving a rolling, tamper-evident hash chain across a session's incarnations.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use daemon_common::{Epoch, FenceToken, MerkleRoot, SessionId, TraceId};
use daemon_credentials::CredentialAuditEvent;
use daemon_store::SessionStore;
use daemon_supervision::ManageEvent;
use daemon_telemetry::{
    current_trace, encode_entry, segment_root, SegmentInput, TraceRecord, TraceSigner, GENESIS_ROOT,
};

/// Map a `ManageEvent` to a `(kind, detail)` pair for the journal record.
fn classify(event: &ManageEvent) -> (&'static str, String) {
    match event {
        ManageEvent::Started { trigger, .. } => ("started", format!("{trigger:?}")),
        ManageEvent::Progress { delta, .. } => ("progress", format!("{delta:?}")),
        ManageEvent::Usage { delta, .. } => ("usage", format!("{delta:?}")),
        ManageEvent::RateLimit { snapshot, .. } => ("ratelimit", format!("{snapshot:?}")),
        ManageEvent::Health { status, .. } => ("health", format!("{status:?}")),
        ManageEvent::Finished { outcome, .. } => ("finished", format!("{outcome:?}")),
        ManageEvent::Error { failure, .. } => ("error", format!("{failure:?}")),
        // `ManageEvent` is `#[non_exhaustive]`; journal any future variant generically.
        other => ("event", format!("{other:?}")),
    }
}

/// Whether an event terminates the segment (the epoch boundary at which the root is sealed).
fn is_terminal(event: &ManageEvent) -> bool {
    matches!(event, ManageEvent::Finished { .. } | ManageEvent::Error { .. })
}

/// A per-`(session, epoch)` verifiable-journal writer bound to the authoritative store.
pub struct JournalSink {
    store: Arc<dyn SessionStore>,
    signer: Arc<TraceSigner>,
    session: SessionId,
    epoch: Epoch,
    fence: FenceToken,
    prior: MerkleRoot,
    seq: AtomicU64,
}

impl JournalSink {
    /// Open the journal for a session's first incarnation (chains onto [`GENESIS_ROOT`]).
    pub fn new(
        store: Arc<dyn SessionStore>,
        signer: Arc<TraceSigner>,
        session: SessionId,
        epoch: Epoch,
        fence: FenceToken,
    ) -> Self {
        Self::chained(store, signer, session, epoch, fence, GENESIS_ROOT)
    }

    /// Open the journal for an incarnation chained onto `prior` (the previous epoch's sealed root).
    pub fn chained(
        store: Arc<dyn SessionStore>,
        signer: Arc<TraceSigner>,
        session: SessionId,
        epoch: Epoch,
        fence: FenceToken,
        prior: MerkleRoot,
    ) -> Self {
        Self {
            store,
            signer,
            session,
            epoch,
            fence,
            prior,
            seq: AtomicU64::new(0),
        }
    }

    /// Append one event to the durable segment as a Gordian Envelope. Idempotent per `seq`.
    pub async fn record(&self, event: &ManageEvent) -> Result<(), daemon_store::StoreError> {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let (kind, detail) = classify(event);
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let record = TraceRecord {
            session: self.session.clone(),
            epoch: self.epoch,
            seq,
            // The trace context active on this task (restored from the cut/transport).
            trace: trace_or(current_trace(), TraceId::NONE),
            kind: kind.to_string(),
            detail,
            timestamp_ms: now_ms,
        };
        let (bytes, content_hash) = encode_entry(&record);
        self.store
            .append_trace(
                &self.session,
                self.epoch,
                daemon_store::TraceEntry {
                    seq,
                    bytes,
                    content_hash,
                },
            )
            .await
    }

    /// Append a **credential audit** event to the same durable segment (host-spec §6). The
    /// credential lifecycle thus rides the identical verifiable trace as the unit's `ManageEvent`s:
    /// after [`JournalSink::seal`] the sealed, signed root is the tamper-evident answer to "who
    /// requested which credential when," and it verifies end-to-end by the audit's own `trace_id`.
    pub async fn record_credential(
        &self,
        event: &CredentialAuditEvent,
    ) -> Result<(), daemon_store::StoreError> {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let record = TraceRecord {
            session: self.session.clone(),
            epoch: self.epoch,
            seq,
            // Prefer the event's own correlation trace (captured at the requesting hop), falling
            // back to this task's restored trace.
            trace: trace_or(event.trace, current_trace()),
            kind: event.kind.label().to_string(),
            detail: event.summary(),
            timestamp_ms: event.timestamp_ms,
        };
        let (bytes, content_hash) = encode_entry(&record);
        self.store
            .append_trace(
                &self.session,
                self.epoch,
                daemon_store::TraceEntry {
                    seq,
                    bytes,
                    content_hash,
                },
            )
            .await
    }

    /// Seal the segment: recompute the Merkle root from the durable entries, sign it, and commit it
    /// under the fence. Fenced exactly like the checkpoint — a stale incarnation is rejected.
    pub async fn seal(&self) -> Result<MerkleRoot, daemon_store::StoreError> {
        let segment = self
            .store
            .load_trace_segment(&self.session, self.epoch)
            .await
            .unwrap_or_else(|| daemon_store::TraceSegment {
                session_id: self.session.clone(),
                epoch: self.epoch,
                entries: Vec::new(),
                committed: None,
            });
        let entries: Vec<(u64, Vec<u8>, daemon_common::ContentHash)> = segment
            .entries
            .iter()
            .map(|e| (e.seq, e.bytes.clone(), e.content_hash))
            .collect();
        let input = SegmentInput {
            session: &self.session,
            epoch: self.epoch,
            prior: self.prior,
            entries: &entries,
        };
        // Root computation is over our own freshly-built entries; it cannot fail to decode.
        let root = segment_root(&input).expect("recompute segment root from durable entries");
        let signature = self.signer.sign_root(&root);
        self.store
            .commit_trace_segment(&self.session, self.epoch, root, signature, self.fence)
            .await?;
        Ok(root)
    }

    /// The fence this sink commits under (the durable incarnation's lease).
    pub fn fence(&self) -> FenceToken {
        self.fence
    }
}

fn trace_or(t: TraceId, fallback: TraceId) -> TraceId {
    if t.is_none() {
        fallback
    } else {
        t
    }
}

/// Drain a unit's event stream into `sink`, recording each event and sealing the segment at the
/// terminal boundary. Returns the sealed root, or `None` if the stream closed without a terminal
/// event. This is the "host feeds the `ManageEvent` stream into the journal" path (host-spec §5).
pub async fn journal_stream(
    sink: &JournalSink,
    events: &mut daemon_supervision::EventStream<ManageEvent>,
) -> Option<MerkleRoot> {
    while let Ok(event) = events.recv().await {
        let _ = sink.record(&event).await;
        if is_terminal(&event) {
            return sink.seal().await.ok();
        }
    }
    None
}
