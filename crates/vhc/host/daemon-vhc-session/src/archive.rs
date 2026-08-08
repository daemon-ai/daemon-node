// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Sealed record-archive publication (architecture §4.4/§7.4; runbook §3.4) — the durable
//! journal's sealed segments published to the content plane, each covered by an attested
//! [`ArchiveHeadRecord`].
//!
//! A journal (ABI §8) is a chain of segments; a SEALED segment is immutable and
//! content-addressed by its complete-file blake3 (the §8.2 chain link). Two publication paths:
//!
//! * **The product path** — [`spawn_archive_publisher`]: the incremental per-seal publisher a
//!   role session runs. It reconciles the on-disk chain against the head store at startup
//!   (crash-safe idempotence: anything sealed-but-unpublished republishes), then consumes the
//!   journal's seal hook — on every seal it uploads exactly the newly sealed segment, then
//!   publishes its attested head (signed by the sealing span's per-run key). Never a
//!   whole-directory scan per seal.
//! * **The backfill/test path** — [`publish_journal_archive`]: the one-shot whole-directory
//!   sweep (segments only, no heads); harness + operator backfill tooling.
//!
//! Schema-free by construction: this reads the segment SUBSTRATE (`daemon-vhc-journal`) and moves
//! opaque bytes to a [`ContentStore`] — it never decodes a round message (the oracle that
//! interprets the archive is harness tooling, not this production path).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use daemon_vhc_journal::{scan_file, JournalPaths, SealedSegment};
use daemon_vhc_net::{ArchiveHeadStore, ContentHash, ContentStore};
use daemon_vhc_proto::domains::ARCHIVE_HEAD_DOMAIN;
use daemon_vhc_proto::{
    ArchiveHeadBody, ArchiveHeadDecision, ArchiveHeadRecord, Hash, RunKeyCertificate, SigningKey,
};

/// The published archive: the sealed segments' content addresses in chain order, and the head
/// (the last sealed segment) — the pointer an archive reader starts from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PublishedArchive {
    /// The sealed segments' content addresses, ascending by ordinal (chain order).
    pub segments: Vec<ContentHash>,
    /// The head address (the last sealed segment), or `None` when nothing is sealed yet.
    pub head: Option<ContentHash>,
}

/// An archive-publication failure.
#[derive(Debug, thiserror::Error)]
pub enum ArchiveError {
    /// A journal substrate error (scan / read).
    #[error("journal: {0}")]
    Journal(String),
    /// A content-store put failure.
    #[error("content store: {0}")]
    Store(String),
    /// A published segment's store address did not match its complete-file blake3 (the store is
    /// broken — the addresses must agree by construction).
    #[error("archive address mismatch: segment complete-file {expected} != store {actual}")]
    AddressMismatch {
        /// The segment's complete-file blake3 (hex).
        expected: String,
        /// The address the store returned (hex).
        actual: String,
    },
}

/// Publish every SEALED segment of the journal at `journal_dir` to `store`, returning their
/// content addresses in chain order + the head. The unsealed (active) tail is NOT published: only
/// immutable segments are archive material. Idempotent — re-publishing puts identical bytes at
/// identical addresses.
///
/// # Errors
/// A journal read/scan failure, a store put failure, or an address disagreement.
pub async fn publish_journal_archive(
    journal_dir: &std::path::Path,
    store: &dyn ContentStore,
) -> Result<PublishedArchive, ArchiveError> {
    let paths =
        JournalPaths::open(journal_dir).map_err(|e| ArchiveError::Journal(e.to_string()))?;
    let ordinals = paths
        .existing_segments()
        .map_err(|e| ArchiveError::Journal(e.to_string()))?;

    let mut segments = Vec::new();
    for ord in ordinals {
        let path = paths.segment(ord);
        let scan = scan_file(&path).map_err(|e| ArchiveError::Journal(e.to_string()))?;
        // Only immutable (sealed) segments are archive material; a torn/active tail is skipped.
        if !scan.sealed {
            continue;
        }
        let bytes = read_segment(&path).map_err(|e| ArchiveError::Journal(e.to_string()))?;
        let address = store
            .put_content(&bytes)
            .await
            .map_err(|e| ArchiveError::Store(e.to_string()))?;
        let expected = ContentHash(scan.complete_file_blake3);
        if address != expected {
            return Err(ArchiveError::AddressMismatch {
                expected: expected.to_hex(),
                actual: address.to_hex(),
            });
        }
        segments.push(address);
    }
    let head = segments.last().copied();
    Ok(PublishedArchive { segments, head })
}

// The journal root is a host-owned, node-chosen directory (never attacker-influenced); the
// segment file read mirrors the journal substrate's own sanctioned raw-fs discipline.
#[allow(clippy::disallowed_methods)]
fn read_segment(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    std::fs::read(path)
}

// ---- the incremental per-seal publisher (the product path) --------------------------------------

/// One identity span's head-signing material: the per-run key seed and its certificate. The
/// session seeds the list with its own binding and pushes the successor's at a live module
/// switch, so the publisher can attest any span the series seals (the head's certificate must
/// bind the SEALING span, not whichever span is current).
pub struct SignerBinding {
    /// The span's certified per-run signing seed.
    pub signing_seed: [u8; 32],
    /// The certificate binding that key to `(run, epoch, role, instance, module)`.
    pub certificate: RunKeyCertificate,
}

/// The archive half a durable journal home hands the session: the seal-hook stream plus the
/// chain coordinates ([`crate::journal_home::DurableSink::arm_seal_hook`] /
/// [`crate::journal_home::DurableSink::founding_instance`]).
pub struct ArchiveSpec {
    /// The seal-hook stream (closed when the sink drops — the publisher drains, then exits).
    pub seals: tokio::sync::mpsc::UnboundedReceiver<SealedSegment>,
    /// The journal home directory (the startup-reconciliation scan root).
    pub journal_dir: PathBuf,
    /// The series' founding incarnation — the chain scope's `chain_instance`.
    pub chain_instance: u64,
    /// The committed-round watermark cell (`0` = none yet, else `round + 1`): allocated beside
    /// the chain coordinates, ADVANCED by the session's egress relay on every published
    /// `RoundRecord` (structural probe — no schema linked), and stamped by the publisher into
    /// each head as the ABI §8.8 freshness claim ([`daemon_vhc_proto::ArchiveHeadBody::round`]).
    pub round_claim: Arc<AtomicU64>,
    /// The ARCHIVE-TIP round watermark (same `0`/`round + 1` encoding): advanced by the
    /// publisher on every acknowledged head to the claim it stamped. The session's seal pacer
    /// reads `round_claim - archived_round` as the live archive/ring overlap lag (Gate B') and
    /// requests a recovery point when it drifts.
    pub archived_round: Arc<AtomicU64>,
}

/// Capped exponential backoff for the publisher's retry loops: transient store/registry faults
/// retry forever (the run does not depend on publication liveness; the channel buffers), typed
/// refusals never retry.
async fn backoff(attempt: u32) {
    let secs = 1u64 << attempt.min(6); // 1s .. 64s cap
    tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
}

/// Spawn the incremental archive publisher for one journal chain. Runs until the seal stream
/// closes and the backlog is drained (or a typed non-extending refusal — fork evidence — aborts
/// publication for operator attention).
///
/// The publisher never blocks the session: seals arrive over the unbounded hook channel, uploads
/// and head publishes happen here with capped-backoff retries.
///
/// [`ArchiveSpec::round_claim`] is the session-maintained committed-round watermark (`0` = none
/// yet, else `round + 1` — see [`crate::role_session`]'s egress relay): each published head
/// stamps the claim current at publish time as [`daemon_vhc_proto::ArchiveHeadBody::round`].
/// Sampling at publish (not seal) can only OVER-state a segment's span — the staleness judgment
/// reading the claim compares against a horizon that absorbs the skew, and freshness evidence
/// that errs fresh never admits a staler joiner.
pub fn spawn_archive_publisher(
    run_label: String,
    run_id: Hash,
    role: String,
    spec: ArchiveSpec,
    heads: Arc<dyn ArchiveHeadStore>,
    segments: Arc<dyn ContentStore>,
    bindings: Arc<Mutex<Vec<SignerBinding>>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // The persisted custody ledger (Phase 6): archive facts recorded per acknowledged head,
        // consumed by archive-then-prune here and by the node's orphan reconciliation. A ledger
        // that cannot load leaves custody bookkeeping off for the session (publication itself is
        // unaffected) — pruning without its facts would be guessing.
        let ledger = match crate::custody::CustodyLedger::load(&spec.journal_dir) {
            Ok(l) => Some(l),
            Err(e) => {
                tracing::error!(
                    dir = %spec.journal_dir.display(),
                    error = %e,
                    "custody ledger unreadable; archive-then-prune disabled for this session"
                );
                None
            }
        };
        let mut publisher = Publisher {
            run_label,
            run_id,
            role,
            chain_instance: spec.chain_instance,
            journal_dir: spec.journal_dir,
            heads,
            segments,
            bindings,
            round_claim: spec.round_claim,
            archived_round: spec.archived_round,
            published_tip: None,
            predecessor: None,
            ledger,
            prune_horizon: daemon_vhc_custody::prune_horizon_from_env(),
        };
        let mut seals = spec.seals;
        if !publisher.reconcile().await {
            return;
        }
        while let Some(sealed) = seals.recv().await {
            if !publisher.publish_sealed(&sealed).await {
                return;
            }
        }
        tracing::debug!(
            run = publisher.run_label,
            role = publisher.role,
            chain_instance = publisher.chain_instance,
            tip = ?publisher.published_tip,
            "archive publisher drained; chain is current"
        );
    })
}

struct Publisher {
    run_label: String,
    run_id: Hash,
    role: String,
    chain_instance: u64,
    journal_dir: PathBuf,
    heads: Arc<dyn ArchiveHeadStore>,
    segments: Arc<dyn ContentStore>,
    bindings: Arc<Mutex<Vec<SignerBinding>>>,
    /// The committed-round watermark (`0` = none, else `round + 1`) stamped on each head.
    round_claim: Arc<AtomicU64>,
    /// The archive-tip watermark (same encoding), advanced on each ACKNOWLEDGED head to the
    /// claim it stamped — the seal pacer's overlap-lag input.
    archived_round: Arc<AtomicU64>,
    /// The highest head ordinal the store holds for this chain (`None` = nothing published).
    published_tip: Option<u64>,
    /// The predecessor chain's terminal head address (segment 0's succession link), resolved
    /// during reconciliation.
    predecessor: Option<Hash>,
    /// The persisted custody ledger (Phase 6): archive facts + prune facts for this chain.
    /// `None` = an unreadable ledger; bookkeeping disabled, publication unaffected.
    ledger: Option<crate::custody::CustodyLedger>,
    /// The archive-then-prune recovery horizon (segments; `0` = never prune).
    prune_horizon: u64,
}

impl Publisher {
    /// The chain scope's base identity (the certificate issuer — every binding shares it).
    fn own_base(&self) -> Option<daemon_vhc_proto::PeerId> {
        self.bindings
            .lock()
            .expect("signer bindings mutex")
            .first()
            .map(|b| b.certificate.base_identity)
    }

    /// Startup reconciliation: resolve the published tip + the succession link from the head
    /// store, then republish every sealed-but-unpublished segment from the on-disk chain (the
    /// crash between a seal and its publish acknowledgment re-sends here; the store's fold is
    /// idempotent). Returns `false` on an abort-worthy refusal.
    async fn reconcile(&mut self) -> bool {
        let Some(base) = self.own_base() else {
            tracing::error!(
                run = self.run_label,
                "archive publisher spawned with no signer binding; publication disabled"
            );
            return false;
        };

        // The stored view of this run's chains, with unbounded capped-backoff (publication has
        // no deadline; the backlog waits).
        let stored = {
            let mut attempt = 0u32;
            loop {
                match self.heads.fetch_heads().await {
                    Ok(heads) => break heads,
                    Err(e) => {
                        tracing::warn!(
                            run = self.run_label,
                            error = %e,
                            attempt,
                            "archive head store unreachable at reconciliation; retrying"
                        );
                        backoff(attempt).await;
                        attempt += 1;
                    }
                }
            }
        };

        let own = |h: &&ArchiveHeadRecord| {
            h.body.run_id == self.run_id
                && h.body.role == self.role
                && h.certificate.base_identity == base
        };
        let own_tip = stored
            .iter()
            .filter(own)
            .filter(|h| h.body.chain_instance == self.chain_instance)
            .max_by_key(|h| h.body.segment);
        self.published_tip = own_tip.map(|h| h.body.segment);
        // Seed the archive-tip watermark from the stored tip's stamped claim, so a restarted
        // session's seal pacer measures lag against what is actually archived rather than 0.
        if let Some(round) = own_tip.and_then(|h| h.body.round) {
            self.archived_round
                .fetch_max(round.saturating_add(1), Ordering::Relaxed);
        }
        // The succession link: our founding head names the predecessor chain's last published
        // head by content address (`None` when this base+role has no earlier chain).
        if self.published_tip.is_none() {
            let predecessor_tip = stored
                .iter()
                .filter(own)
                .filter(|h| h.body.chain_instance < self.chain_instance)
                .max_by_key(|h| (h.body.chain_instance, h.body.segment));
            self.predecessor = match predecessor_tip.map(ArchiveHeadRecord::content_address) {
                None => None,
                Some(Ok(address)) => Some(address),
                Some(Err(e)) => {
                    tracing::error!(
                        run = self.run_label,
                        error = %e,
                        "predecessor head does not re-encode; publication disabled"
                    );
                    return false;
                }
            };
        }

        // The on-disk sealed chain above the published tip.
        let paths = match JournalPaths::open(&self.journal_dir) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(
                    run = self.run_label,
                    error = %e,
                    "archive reconciliation cannot open the journal home; publication disabled"
                );
                return false;
            }
        };
        let ordinals = match paths.existing_segments() {
            Ok(o) => o,
            Err(e) => {
                tracing::error!(
                    run = self.run_label,
                    error = %e,
                    "archive reconciliation cannot list segments; publication disabled"
                );
                return false;
            }
        };
        for ord in ordinals {
            if self.published_tip.is_some_and(|tip| ord <= tip) {
                continue;
            }
            let path = paths.segment(ord);
            let scan = match scan_file(&path) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(
                        run = self.run_label,
                        segment = ord,
                        error = %e,
                        "archive reconciliation cannot scan a segment; publication disabled"
                    );
                    return false;
                }
            };
            if !scan.sealed {
                continue; // the active tail is not archive material
            }
            let records = scan
                .records
                .iter()
                .filter(|r| !matches!(r.body, daemon_vhc_journal::Body::Seal(_)))
                .count() as u64;
            let sealed = SealedSegment {
                id: scan.header.id.clone(),
                segment: ord,
                path,
                segment_blake3: scan.complete_file_blake3,
                prev_blake3: scan.header.prev_blake3,
                records,
            };
            if !self.publish_sealed(&sealed).await {
                return false;
            }
        }
        true
    }

    /// Publish one sealed segment: upload the bytes (content-addressed), then the attested head.
    /// Returns `false` on an abort-worthy condition (corruption, no attesting binding, a typed
    /// non-extending refusal — fork evidence).
    async fn publish_sealed(&mut self, sealed: &SealedSegment) -> bool {
        if self.published_tip.is_some_and(|tip| sealed.segment <= tip) {
            return true; // reconciliation/hook overlap — already published
        }

        // -- the segment bytes, verified against the seal-time content address ------------------
        let bytes = match read_segment(&sealed.path) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!(
                    run = self.run_label,
                    segment = sealed.segment,
                    error = %e,
                    "sealed segment unreadable; archive publication disabled"
                );
                return false;
            }
        };
        let content = daemon_vhc_proto::blake3_hash(&bytes);
        if content.0 != sealed.segment_blake3 {
            tracing::error!(
                run = self.run_label,
                segment = sealed.segment,
                expected = %Hash(sealed.segment_blake3).to_hex(),
                actual = %content.to_hex(),
                "sealed segment bytes do not match the seal-time hash; archive publication disabled"
            );
            return false;
        }
        let mut attempt = 0u32;
        loop {
            match self.segments.put_content(&bytes).await {
                Ok(address) if address.0 == sealed.segment_blake3 => break,
                Ok(address) => {
                    tracing::error!(
                        run = self.run_label,
                        segment = sealed.segment,
                        store_address = %address.to_hex(),
                        "content store returned a foreign address; archive publication disabled"
                    );
                    return false;
                }
                Err(e) => {
                    tracing::warn!(
                        run = self.run_label,
                        segment = sealed.segment,
                        error = %e,
                        attempt,
                        "segment upload failed; retrying"
                    );
                    backoff(attempt).await;
                    attempt += 1;
                }
            }
        }

        // -- the attested head, signed by the SEALING span's per-run key ------------------------
        let claim_at_stamp = self.round_claim.load(Ordering::Relaxed);
        let body = ArchiveHeadBody {
            domain: ARCHIVE_HEAD_DOMAIN.into(),
            run_id: self.run_id,
            role: self.role.clone(),
            chain_instance: self.chain_instance,
            segment: sealed.segment,
            segment_hash: Hash(sealed.segment_blake3),
            prev_hash: Hash(sealed.prev_blake3),
            records: sealed.records,
            instance: sealed.id.instance,
            epoch: sealed.id.epoch,
            module: sealed.id.module,
            predecessor: (sealed.segment == 0).then_some(self.predecessor).flatten(),
            round: claim_at_stamp.checked_sub(1),
        };
        let record = {
            let bindings = self.bindings.lock().expect("signer bindings mutex");
            let scope = body.cert_scope();
            let Some(binding) = bindings.iter().find(|b| {
                let c = &b.certificate.body.scope;
                c.run_id == scope.run_id
                    && c.epoch == scope.epoch
                    && c.role == scope.role
                    && c.instance == scope.instance
                    && c.module_hash == scope.module_hash
            }) else {
                tracing::error!(
                    run = self.run_label,
                    segment = sealed.segment,
                    instance = sealed.id.instance,
                    epoch = sealed.id.epoch,
                    "no signer binding covers the sealing span; archive publication disabled"
                );
                return false;
            };
            match ArchiveHeadRecord::publish(
                &SigningKey::from_bytes(&binding.signing_seed),
                binding.certificate.clone(),
                body,
            ) {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!(
                        run = self.run_label,
                        segment = sealed.segment,
                        error = %e,
                        "archive head does not author; publication disabled"
                    );
                    return false;
                }
            }
        };
        let mut attempt = 0u32;
        loop {
            match self.heads.put_head(&record).await {
                Ok(ArchiveHeadDecision::Accepted | ArchiveHeadDecision::AlreadyStored) => {
                    tracing::debug!(
                        run = self.run_label,
                        role = self.role,
                        segment = sealed.segment,
                        records = sealed.records,
                        address = %Hash(sealed.segment_blake3).to_hex(),
                        "archive segment + attested head published"
                    );
                    self.published_tip = Some(sealed.segment);
                    // The acknowledged head carries `claim_at_stamp` as its freshness claim —
                    // advance the archive-tip watermark the seal pacer measures lag against.
                    self.archived_round
                        .fetch_max(claim_at_stamp, Ordering::Relaxed);
                    self.record_and_prune(sealed);
                    return true;
                }
                Ok(ArchiveHeadDecision::RejectedNonExtending {
                    stored_segment,
                    stored_segment_hash,
                }) => {
                    // Fork evidence: a stored head at this height disagrees. Never retried —
                    // this is the operator-attention surface (two signed heads that do not
                    // extend one another are portable evidence).
                    tracing::error!(
                        run = self.run_label,
                        segment = sealed.segment,
                        stored_segment,
                        stored_hash = %stored_segment_hash.to_hex(),
                        ours = %Hash(sealed.segment_blake3).to_hex(),
                        "archive head refused as non-extending (fork evidence); publication disabled"
                    );
                    return false;
                }
                Ok(ArchiveHeadDecision::RejectedStructural { reason }) => {
                    tracing::error!(
                        run = self.run_label,
                        segment = sealed.segment,
                        reason,
                        "archive head refused structurally; publication disabled"
                    );
                    return false;
                }
                Err(e) => {
                    tracing::warn!(
                        run = self.run_label,
                        segment = sealed.segment,
                        error = %e,
                        attempt,
                        "archive head publish failed; retrying"
                    );
                    backoff(attempt).await;
                    attempt += 1;
                }
            }
        }
    }

    /// Phase 6 bookkeeping after an acknowledged publish: record the archive facts in the
    /// persisted custody ledger, then run one archive-then-prune pass (the dependency closure +
    /// recovery horizon live in [`crate::custody`]). Best-effort: a bookkeeping failure is loud
    /// but never blocks publication — the ledger is re-derivable from the head store, and an
    /// unpruned segment is a bounded cost, not a correctness fault.
    fn record_and_prune(&mut self, sealed: &SealedSegment) {
        let Some(ledger) = self.ledger.as_mut() else {
            return;
        };
        if let Err(e) =
            ledger.record_archived(sealed.segment, Hash(sealed.segment_blake3), sealed.records)
        {
            tracing::warn!(
                run = self.run_label,
                segment = sealed.segment,
                error = %e,
                "custody ledger persist failed; archive facts not recorded"
            );
            return;
        }
        match crate::custody::prune_archived(&self.journal_dir, ledger, self.prune_horizon) {
            Ok(outcome) if outcome.bytes > 0 => {
                tracing::info!(
                    run = self.run_label,
                    role = self.role,
                    segments = outcome.segments,
                    sidecars = outcome.sidecars,
                    bytes = outcome.bytes,
                    horizon = self.prune_horizon,
                    "archive-then-prune reclaimed local journal bytes"
                );
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(
                    run = self.run_label,
                    error = %e,
                    "archive-then-prune pass failed; local bytes retained"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_host::run::{JournalSink, RunIdentity};
    use daemon_vhc_net::MemoryContentStore;

    use crate::journal_home::{journal_dir, DurableSink};

    fn identity() -> RunIdentity {
        RunIdentity {
            run_id: [0x11; 32],
            epoch: 0,
            role: "coordinator".into(),
            instance: 1,
            module: [0x22; 32],
        }
    }

    #[tokio::test]
    async fn sealed_segments_publish_content_addressed_and_refetch() {
        let dir = tempfile::tempdir().unwrap();
        let jdir = journal_dir(dir.path(), "archive-run", "coordinator", 1);
        // Write enough records to seal at least one segment (RotatePolicy default rolls at a
        // record threshold), then leave the journal.
        {
            let mut sink = DurableSink::open(&jdir, &identity(), [0x5C; 32]).unwrap();
            sink.run_header(
                2 << 16,
                &[("vhc".into(), 2)],
                false,
                b"m",
                b"c",
                b"g",
                daemon_vhc_host::run::RunHeaderResources::Declared(b"cl"),
                b"ch",
                b"d",
            )
            .unwrap();
            for i in 0..2048u64 {
                sink.event(i, b"opaque-record").unwrap();
            }
            sink.terminal(0, Some(0), None).unwrap();
        }

        let store = MemoryContentStore::new();
        let archive = publish_journal_archive(&jdir, &store).await.unwrap();
        assert!(
            !archive.segments.is_empty(),
            "at least one segment sealed + published"
        );
        assert_eq!(archive.head, archive.segments.last().copied());
        // Every published address re-fetches (content-addressed, verified).
        for addr in &archive.segments {
            let bytes = store.get_content(addr).await.expect("archived segment");
            assert_eq!(
                daemon_vhc_proto::blake3_hash(&bytes),
                *addr,
                "the archived segment is addressed by its own bytes"
            );
        }
        // Idempotent re-publish yields the same addresses.
        let again = publish_journal_archive(&jdir, &store).await.unwrap();
        assert_eq!(again.segments, archive.segments);
    }

    // ---- the incremental publisher --------------------------------------------------------------

    use daemon_vhc_net::FsArchiveHeadStore;
    use daemon_vhc_proto::CertScope;

    use crate::identity::issue_run_key;

    const RUN: [u8; 32] = [0x11; 32];
    const MODULE: [u8; 32] = [0x22; 32];

    fn scope(epoch: u64, instance: u64) -> CertScope {
        CertScope {
            run_id: Hash(RUN),
            epoch,
            role: "coordinator".into(),
            instance,
            module_hash: Hash(MODULE),
        }
    }

    fn write_span(sink: &mut DurableSink, records: u64) {
        for i in 0..records {
            sink.event(i, b"opaque-record").unwrap();
        }
    }

    /// The product path end to end: a live seal streams through the hook and lands as segment
    /// bytes + an attested, reader-verifiable head; a seal a crashed publisher never saw is
    /// republished by the next publisher's startup reconciliation — dense, idempotent, and
    /// without touching the already-published prefix.
    #[tokio::test]
    async fn the_publisher_streams_each_seal_and_reconciles_after_a_crash() {
        let dir = tempfile::tempdir().unwrap();
        let jdir = journal_dir(dir.path(), "pub-run", "coordinator", 1);
        let heads_dir = dir.path().join("heads");
        let base = SigningKey::from_bytes(&[7u8; 32]);
        let trusted = vec![daemon_vhc_proto::peer_id(&base)];
        let certified = issue_run_key(&base, scope(0, 1)).unwrap();
        let bindings = Arc::new(Mutex::new(vec![SignerBinding {
            signing_seed: certified.key.to_bytes(),
            certificate: certified.cert.clone(),
        }]));
        let segments = Arc::new(MemoryContentStore::new());

        // -- phase 1: live streaming ------------------------------------------------------------
        let heads: Arc<dyn ArchiveHeadStore> =
            Arc::new(FsArchiveHeadStore::open(&heads_dir).await.unwrap());
        let mut sink = DurableSink::open(&jdir, &identity(), [0x5C; 32]).unwrap();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        sink.arm_seal_hook(tx);
        let publisher = spawn_archive_publisher(
            "pub-run".into(),
            Hash(RUN),
            "coordinator".into(),
            ArchiveSpec {
                seals: rx,
                journal_dir: jdir.clone(),
                chain_instance: sink.founding_instance(),
                round_claim: Arc::new(AtomicU64::new(0)),
                archived_round: Arc::new(AtomicU64::new(0)),
            },
            heads.clone(),
            segments.clone(),
            bindings.clone(),
        );
        write_span(&mut sink, 16);
        sink.terminal(0, Some(0), None).unwrap(); // rolls: segment 0 seals + streams
        drop(sink);
        DurableSink::release_seal_stream(&jdir); // the end-of-session half: closes the stream
        publisher.await.unwrap();

        let stored = heads.fetch_heads().await.unwrap();
        assert_eq!(stored.len(), 1, "the live seal published exactly one head");
        let head0 = &stored[0];
        head0.authorize(&trusted).expect("reader-verifiable head");
        assert_eq!(head0.body.segment, 0);
        assert_eq!(head0.body.chain_instance, 1);
        assert_eq!(
            head0.body.predecessor, None,
            "a founding chain links nothing"
        );
        let bytes = segments
            .get_content(&head0.body.segment_hash)
            .await
            .expect("the segment bytes are on the content plane");
        assert_eq!(
            daemon_vhc_proto::blake3_hash(&bytes),
            head0.body.segment_hash
        );

        // -- phase 2: a seal the (crashed) publisher never saw ------------------------------------
        {
            let mut sink = DurableSink::open(&jdir, &identity(), [0x5C; 32]).unwrap();
            write_span(&mut sink, 8);
            sink.terminal(0, Some(0), None).unwrap(); // seals segment 1; nothing is listening
        }
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SealedSegment>();
        drop(tx); // reconciliation-only: the stream is already closed
        let publisher = spawn_archive_publisher(
            "pub-run".into(),
            Hash(RUN),
            "coordinator".into(),
            ArchiveSpec {
                seals: rx,
                journal_dir: jdir.clone(),
                chain_instance: 1,
                round_claim: Arc::new(AtomicU64::new(0)),
                archived_round: Arc::new(AtomicU64::new(0)),
            },
            heads.clone(),
            segments.clone(),
            bindings,
        );
        publisher.await.unwrap();

        let stored = heads.fetch_heads().await.unwrap();
        assert_eq!(
            stored.len(),
            2,
            "reconciliation republished the missed seal"
        );
        let head1 = stored
            .iter()
            .find(|h| h.body.segment == 1)
            .expect("segment 1's head");
        head1.authorize(&trusted).unwrap();
        assert_eq!(
            head1.body.prev_hash, head0.body.segment_hash,
            "the chain links densely across the crash boundary"
        );
        assert!(
            segments.get_content(&head1.body.segment_hash).await.is_ok(),
            "the reconciled segment's bytes are on the content plane"
        );
    }

    /// The live-upgrade seam: the retiring span's final seal lands INSIDE `open_continuation`
    /// (before any caller could re-arm a hook), and the successor's seals follow — both must
    /// reach the SAME publisher, each attested under its own sealing span's certificate.
    #[tokio::test]
    async fn the_seam_seal_streams_and_each_span_attests_under_its_own_certificate() {
        let dir = tempfile::tempdir().unwrap();
        let jdir = journal_dir(dir.path(), "seam-run", "coordinator", 1);
        let heads_dir = dir.path().join("heads");
        let base = SigningKey::from_bytes(&[9u8; 32]);
        let trusted = vec![daemon_vhc_proto::peer_id(&base)];
        let span_a = issue_run_key(&base, scope(0, 1)).unwrap();
        let span_b = issue_run_key(&base, scope(1, 2)).unwrap();
        let bindings = Arc::new(Mutex::new(vec![SignerBinding {
            signing_seed: span_a.key.to_bytes(),
            certificate: span_a.cert.clone(),
        }]));
        let segments = Arc::new(MemoryContentStore::new());
        let heads: Arc<dyn ArchiveHeadStore> =
            Arc::new(FsArchiveHeadStore::open(&heads_dir).await.unwrap());

        let mut sink = DurableSink::open(&jdir, &identity(), [0x5C; 32]).unwrap();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        sink.arm_seal_hook(tx);
        let publisher = spawn_archive_publisher(
            "seam-run".into(),
            Hash(RUN),
            "coordinator".into(),
            ArchiveSpec {
                seals: rx,
                journal_dir: jdir.clone(),
                chain_instance: sink.founding_instance(),
                round_claim: Arc::new(AtomicU64::new(0)),
                archived_round: Arc::new(AtomicU64::new(0)),
            },
            heads.clone(),
            segments.clone(),
            bindings.clone(),
        );
        write_span(&mut sink, 16);
        drop(sink); // the retiring instance's sink drops FIRST (one writer per series)

        // The switch activates: the successor's signer binding lands before its seals do.
        bindings.lock().unwrap().push(SignerBinding {
            signing_seed: span_b.key.to_bytes(),
            certificate: span_b.cert.clone(),
        });
        let successor = RunIdentity {
            epoch: 1,
            instance: 2,
            ..identity()
        };
        let mut sink = DurableSink::open_continuation(&jdir, &successor, [0x5C; 32]).unwrap();
        write_span(&mut sink, 8);
        sink.terminal(0, Some(0), None).unwrap();
        drop(sink);
        DurableSink::release_seal_stream(&jdir);
        publisher.await.unwrap();

        let stored = heads.fetch_heads().await.unwrap();
        assert_eq!(
            stored.len(),
            2,
            "the seam seal AND the successor's terminal seal both published"
        );
        let head0 = stored.iter().find(|h| h.body.segment == 0).unwrap();
        let head1 = stored.iter().find(|h| h.body.segment == 1).unwrap();
        head0.authorize(&trusted).unwrap();
        head1.authorize(&trusted).unwrap();
        assert_eq!(
            (head0.body.instance, head0.body.epoch),
            (1, 0),
            "the seam-rolled segment attests under the RETIRING span"
        );
        assert_eq!(head0.certificate, span_a.cert);
        assert_eq!(
            (head1.body.instance, head1.body.epoch),
            (2, 1),
            "the successor's segment attests under the NEW span"
        );
        assert_eq!(head1.certificate, span_b.cert);
        assert_eq!(head0.body.chain_instance, head1.body.chain_instance);
        assert_eq!(head1.body.prev_hash, head0.body.segment_hash);
    }
}
