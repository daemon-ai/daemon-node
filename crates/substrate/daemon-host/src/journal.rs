//! Wiring an agent's stream into the unified verifiable journal (host-spec §5).
//!
//! A [`JournalSink`] binds a `(stream, segment)` chain to the authoritative [`SessionStore`] and a
//! [`TraceSigner`]. It records two kinds of entry into one hash-linked chain: coarse **management**
//! records (lifecycle / credential-audit) and coalesced finished **chat blocks**
//! ([`TranscriptBlock`]). Each entry becomes a Gordian Envelope (via [`daemon_telemetry`]) and is
//! **appended** durably; at a turn boundary [`JournalSink::seal`] folds the open segment into a
//! signed [`MerkleRoot`], commits it, and advances to the next segment chained onto that root.
//!
//! Keyed `(stream, segment)` decouples the journal from the durable `(session, epoch)` identity, so
//! a live session, a fleet child, or a foreign agent journals exactly like the durable path. The
//! fence is `Some` only on the durable path (the seal is bound to the incarnation lease); a
//! non-durable stream seals unfenced (the ed25519 signature is the integrity primitive).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use daemon_common::{FenceToken, JournalStreamId, MerkleRoot, TraceId};
use daemon_credentials::CredentialAuditEvent;
use daemon_protocol::TranscriptBlock;
use daemon_store::SessionStore;
use daemon_supervision::ManageEvent;
use daemon_telemetry::{
    current_trace, encode_entry, segment_root, JournalEntryView, JournalPayload, SegmentInput,
    TraceSigner, GENESIS_ROOT,
};

/// Map a `ManageEvent` to a `(kind, detail)` pair for a management journal record.
fn classify(event: &ManageEvent) -> (&'static str, String) {
    match event {
        ManageEvent::Started { trigger, .. } => ("mgmt.started", format!("{trigger:?}")),
        ManageEvent::Progress { delta, .. } => ("mgmt.progress", format!("{delta:?}")),
        ManageEvent::Usage { delta, .. } => ("mgmt.usage", format!("{delta:?}")),
        ManageEvent::RateLimit { snapshot, .. } => ("mgmt.ratelimit", format!("{snapshot:?}")),
        ManageEvent::Health { status, .. } => ("mgmt.health", format!("{status:?}")),
        ManageEvent::Finished { outcome, .. } => ("mgmt.finished", format!("{outcome:?}")),
        ManageEvent::Error { failure, .. } => ("mgmt.error", format!("{failure:?}")),
        // `ManageEvent` is `#[non_exhaustive]`; journal any future variant generically.
        other => ("mgmt.event", format!("{other:?}")),
    }
}

/// Whether a management event terminates the segment (the boundary at which the root is sealed).
fn is_terminal(event: &ManageEvent) -> bool {
    matches!(
        event,
        ManageEvent::Finished { .. } | ManageEvent::Error { .. }
    )
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn trace_or(t: TraceId, fallback: TraceId) -> TraceId {
    if t.is_none() {
        fallback
    } else {
        t
    }
}

/// A per-stream verifiable-journal writer bound to the authoritative store. Records management
/// entries and chat blocks into one chain, sealing per turn.
pub struct JournalSink {
    store: Arc<dyn SessionStore>,
    signer: Arc<TraceSigner>,
    stream: JournalStreamId,
    /// `Some` on the durable path (sealed under the incarnation lease); `None` otherwise.
    fence: Option<FenceToken>,
    /// The open segment being appended to (a turn for streaming units, an incarnation for durable).
    segment: AtomicU64,
    /// The prior segment's committed root (the rolling chain link).
    prior: Mutex<MerkleRoot>,
    /// Monotonic per-segment sequence number, reset on each seal.
    seq: AtomicU64,
}

impl JournalSink {
    /// Open a journal for a non-durable stream (live session / fleet / foreign): segment 0 chaining
    /// onto [`GENESIS_ROOT`], unfenced.
    pub fn new(
        store: Arc<dyn SessionStore>,
        signer: Arc<TraceSigner>,
        stream: JournalStreamId,
    ) -> Self {
        Self::with_segment(store, signer, stream, None, 0, GENESIS_ROOT)
    }

    /// Open a journal at an explicit starting segment + prior root, with an optional fence.
    pub fn with_segment(
        store: Arc<dyn SessionStore>,
        signer: Arc<TraceSigner>,
        stream: JournalStreamId,
        fence: Option<FenceToken>,
        start_segment: u64,
        prior: MerkleRoot,
    ) -> Self {
        Self {
            store,
            signer,
            stream,
            fence,
            segment: AtomicU64::new(start_segment),
            prior: Mutex::new(prior),
            seq: AtomicU64::new(0),
        }
    }

    /// Open a journal for one durable incarnation: segment = `epoch`, chained onto the prior epoch's
    /// sealed root (loaded from the store; [`GENESIS_ROOT`] for epoch 0), fenced by the lease.
    pub async fn for_incarnation(
        store: Arc<dyn SessionStore>,
        signer: Arc<TraceSigner>,
        stream: JournalStreamId,
        fence: FenceToken,
        epoch: u64,
    ) -> Self {
        let prior = if epoch == 0 {
            GENESIS_ROOT
        } else {
            store
                .load_trace_segment(&stream, epoch - 1)
                .await
                .and_then(|seg| seg.committed.map(|c| c.root))
                .unwrap_or(GENESIS_ROOT)
        };
        Self::with_segment(store, signer, stream, Some(fence), epoch, prior)
    }

    /// The current open segment.
    pub fn segment(&self) -> u64 {
        self.segment.load(Ordering::Relaxed)
    }

    async fn append(
        &self,
        kind: String,
        payload: JournalPayload,
    ) -> Result<(), daemon_store::StoreError> {
        let segment = self.segment.load(Ordering::Relaxed);
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let view = JournalEntryView {
            stream: self.stream.clone(),
            segment,
            seq,
            epoch: segment,
            trace: trace_or(current_trace(), TraceId::NONE).0,
            kind,
            timestamp_ms: now_ms(),
            payload,
        };
        let (bytes, content_hash) = encode_entry(&view);
        self.store
            .append_trace(
                &self.stream,
                segment,
                daemon_store::TraceEntry {
                    seq,
                    bytes,
                    content_hash,
                },
            )
            .await
    }

    /// Append a management lifecycle record (kind + human/structured detail).
    pub async fn record_management(
        &self,
        kind: impl Into<String>,
        detail: String,
    ) -> Result<(), daemon_store::StoreError> {
        self.append(kind.into(), JournalPayload::Management { detail })
            .await
    }

    /// Append one management event (the coarse `ManageEvent` projection). Convenience over
    /// [`Self::record_management`] used by the management-stream driver and tests.
    pub async fn record(&self, event: &ManageEvent) -> Result<(), daemon_store::StoreError> {
        let (kind, detail) = classify(event);
        self.record_management(kind, detail).await
    }

    /// Append a **credential audit** event to the same chain (host-spec §6), so "who requested which
    /// credential when" rides the identical tamper-evident journal as the unit's lifecycle.
    pub async fn record_credential(
        &self,
        event: &CredentialAuditEvent,
    ) -> Result<(), daemon_store::StoreError> {
        let segment = self.segment.load(Ordering::Relaxed);
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let view = JournalEntryView {
            stream: self.stream.clone(),
            segment,
            seq,
            epoch: segment,
            trace: trace_or(event.trace, current_trace()).0,
            kind: format!("cred.{}", event.kind.label()),
            timestamp_ms: event.timestamp_ms,
            payload: JournalPayload::Management {
                detail: event.summary(),
            },
        };
        let (bytes, content_hash) = encode_entry(&view);
        self.store
            .append_trace(
                &self.stream,
                segment,
                daemon_store::TraceEntry {
                    seq,
                    bytes,
                    content_hash,
                },
            )
            .await
    }

    /// Append one coalesced finished chat block to the chain. The block is the durable, signable
    /// unit of transcript history (streaming deltas are not individually journaled).
    pub async fn record_block(
        &self,
        block: &TranscriptBlock,
    ) -> Result<(), daemon_store::StoreError> {
        let mut body = Vec::new();
        ciborium::into_writer(block, &mut body).expect("encode transcript block to CBOR");
        self.append(
            block.kind_label().to_string(),
            JournalPayload::Block { body },
        )
        .await
    }

    /// Seal the open segment: recompute the Merkle root from the durable entries, sign it, and
    /// commit it (fenced on the durable path). Then advance to the next segment, chaining onto this
    /// root and resetting the per-segment sequence — so the next turn is the next link.
    pub async fn seal(&self) -> Result<MerkleRoot, daemon_store::StoreError> {
        let segment = self.segment.load(Ordering::Relaxed);
        let prior = *self.prior.lock().unwrap();
        let loaded = self.store.load_trace_segment(&self.stream, segment).await;
        let entries: Vec<(u64, Vec<u8>, daemon_common::ContentHash)> = loaded
            .map(|seg| {
                seg.entries
                    .iter()
                    .map(|e| (e.seq, e.bytes.clone(), e.content_hash))
                    .collect()
            })
            .unwrap_or_default();
        let input = SegmentInput {
            stream: &self.stream,
            segment,
            prior,
            entries: &entries,
        };
        let root = segment_root(&input).expect("recompute segment root from durable entries");
        let signature = self.signer.sign_root(&root);
        self.store
            .commit_trace_segment(&self.stream, segment, root, signature, self.fence)
            .await?;
        // Advance the chain: next turn is the next segment, chained onto this root.
        self.segment.fetch_add(1, Ordering::Relaxed);
        *self.prior.lock().unwrap() = root;
        self.seq.store(0, Ordering::Relaxed);
        Ok(root)
    }
}

/// Drives a [`JournalSink`] from the rich `Outbound` stream through a [`BlockCoalescer`]: feed every
/// upbound frame, and the feeder appends finished blocks / management records and seals at turn
/// boundaries. The shared journaling tap for the streaming paths (live session, fleet, foreign).
pub struct JournalFeeder {
    sink: Arc<JournalSink>,
    coalescer: tokio::sync::Mutex<crate::transcript::BlockCoalescer>,
}

impl JournalFeeder {
    /// Wrap a sink with a fresh coalescer.
    pub fn new(sink: Arc<JournalSink>) -> Self {
        Self {
            sink,
            coalescer: tokio::sync::Mutex::new(crate::transcript::BlockCoalescer::new()),
        }
    }

    /// The underlying sink (e.g. to inspect the current segment).
    pub fn sink(&self) -> &Arc<JournalSink> {
        &self.sink
    }

    /// Fold one upbound frame and apply the resulting journal actions durably (best-effort: a store
    /// error is swallowed so journaling never blocks the live path).
    pub async fn feed(&self, frame: &daemon_protocol::Outbound) {
        let actions = {
            let mut c = self.coalescer.lock().await;
            c.push(frame)
        };
        for action in actions {
            let _ = match action {
                crate::transcript::JournalAction::Management { kind, detail } => {
                    self.sink.record_management(kind, detail).await
                }
                crate::transcript::JournalAction::Block(block) => {
                    self.sink.record_block(&block).await
                }
                crate::transcript::JournalAction::Seal => self.sink.seal().await.map(|_| ()),
            };
        }
    }
}

/// Drain a unit's **management** event stream into `sink`, recording each event and sealing the
/// segment at the terminal boundary (the management-only journaling path; the rich transcript path
/// drives the sink through a [`crate::transcript::BlockCoalescer`]).
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
