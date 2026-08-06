// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`ArchiveHeadStore`] — the archive-head half of incremental authenticated journal-archive
//! publication (architecture §4.4; runbook §3.4).
//!
//! Sealed segment BYTES ride the existing content-addressed content plane
//! ([`crate::transport::ContentStore`]: presigned R2 or the run's filesystem store) — a segment
//! object needs no new surface, its BLAKE3 is its address. The HEADS need a slot with structure:
//! a conforming store applies the normative structural fold
//! ([`daemon_vhc_proto::ArchiveChainSlot::fold`]) — dense ordinals, `prev_hash` linkage,
//! byte-identical idempotent republish, typed refusal of everything else (the fork-evidence
//! surface). Like the roster registry, the store is UNTRUSTED STORAGE: it never verifies
//! signatures and never judges authority — readers verify every fetched head themselves
//! ([`daemon_vhc_proto::ArchiveHeadRecord::authorize`] against the genesis-trusted bases).
//!
//! Two production planes, mirroring the content-store duality:
//! * [`HttpArchiveHeadStore`] — the cloud registry (`PUT {base}/runs/:id/archive/head`,
//!   `GET {base}/runs/:id/archive/heads`; canonical CBOR both ways, 200/409 + decision).
//! * [`FsArchiveHeadStore`] — the filesystem plane rooted in the run's state dir
//!   (`<run state dir>/archive/heads/`); the local / single-host / acceptance-baseline store.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use daemon_core::ContainedRoot;
use daemon_egress::{EgressClient, EgressRequest, Redirects};
use daemon_vhc_proto::{
    from_canonical_slice, to_canonical_vec, ArchiveChainSlot, ArchiveHeadDecision,
    ArchiveHeadRecord, PeerId,
};
use serde::{Deserialize, Serialize};

use crate::seam::RunId;
use crate::VhcNetError;

/// The archive-head slot store one run's publishers and assemblers bind.
#[async_trait]
pub trait ArchiveHeadStore: Send + Sync {
    /// Publish one attested head; the store answers with the structural fold's decision.
    /// `Accepted` / `AlreadyStored` are the publisher's success cases; a refusal is typed and
    /// final for these bytes (a non-extending head at a stored height is fork evidence — the
    /// publisher must not retry it).
    async fn put_head(
        &self,
        record: &ArchiveHeadRecord,
    ) -> Result<ArchiveHeadDecision, VhcNetError>;

    /// Every stored head across every chain of the run, in slot storage order. The caller MUST
    /// verify each record itself (`authorize` against the genesis-trusted bases) — the store
    /// stores, it never vouches.
    async fn fetch_heads(&self) -> Result<Vec<ArchiveHeadRecord>, VhcNetError>;
}

/// The canonical-CBOR mutation response body (`{decision, tip}`; 200 accepted / 409 refused) —
/// the frozen node↔cloud shape, mirrored by the TS side.
#[derive(Debug, Serialize, Deserialize)]
struct HeadMutationResponse {
    decision: ArchiveHeadDecision,
}

/// The canonical-CBOR `GET .../archive/heads` snapshot body: `{heads: [...]}` (storage order).
#[derive(Debug, Serialize, Deserialize)]
struct HeadsSnapshot {
    heads: Vec<ArchiveHeadRecord>,
}

// ---- the cloud registry plane -------------------------------------------------------------------

/// The registry-backed [`ArchiveHeadStore`]: the gateway's archive-head routes, carrying the
/// same credential the presign/WS planes use.
pub struct HttpArchiveHeadStore {
    egress: EgressClient,
    base_url: String,
    run: RunId,
    bearer: Option<String>,
    internal: Option<(String, String)>,
}

impl HttpArchiveHeadStore {
    /// A store against `base_url` (the coordinator/gateway base) for `run`.
    #[must_use]
    pub fn new(egress: EgressClient, base_url: impl Into<String>, run: RunId) -> Self {
        Self {
            egress,
            base_url: base_url.into(),
            run,
            bearer: None,
            internal: None,
        }
    }

    /// Attach a bearer credential (mirrors [`crate::HttpPresignClient::with_bearer`]).
    #[must_use]
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.bearer = Some(token.into());
        self
    }

    /// Attach the internal identity headers (the direct-to-`apps/vhc` dev path).
    #[must_use]
    pub fn with_internal(mut self, org_id: impl Into<String>, actor: impl Into<String>) -> Self {
        self.internal = Some((org_id.into(), actor.into()));
        self
    }

    fn authed(&self, mut req: EgressRequest) -> EgressRequest {
        if let Some(token) = &self.bearer {
            req = req.bearer_auth(token);
        }
        if let Some((org_id, actor)) = &self.internal {
            req = req
                .header("x-daemon-org-id", org_id)
                .header("x-daemon-actor", actor);
        }
        req
    }
}

#[async_trait]
impl ArchiveHeadStore for HttpArchiveHeadStore {
    async fn put_head(
        &self,
        record: &ArchiveHeadRecord,
    ) -> Result<ArchiveHeadDecision, VhcNetError> {
        let url = format!("{}/runs/{}/archive/head", self.base_url, self.run.as_str());
        let bytes = to_canonical_vec(record)
            .map_err(|e| VhcNetError::Transport(format!("encode archive head: {e}")))?;
        let req =
            self.authed(EgressRequest::put(&url, bytes).header("content-type", "application/cbor"));
        let resp = self
            .egress
            .execute(req, Redirects::None)
            .await
            .map_err(|e| VhcNetError::Transport(format!("publish archive head: {e}")))?;
        let status = resp.status();
        let body = resp
            .bytes()
            .await
            .map_err(|e| VhcNetError::Transport(format!("read archive head response: {e}")))?;
        if status.is_success() || status.as_u16() == 409 {
            let decoded: HeadMutationResponse = from_canonical_slice(&body).map_err(|e| {
                VhcNetError::Transport(format!("decode archive head decision ({status}): {e}"))
            })?;
            return Ok(decoded.decision);
        }
        Err(VhcNetError::Transport(format!(
            "publish archive head returned {status}: {}",
            String::from_utf8_lossy(&body)
        )))
    }

    async fn fetch_heads(&self) -> Result<Vec<ArchiveHeadRecord>, VhcNetError> {
        let url = format!("{}/runs/{}/archive/heads", self.base_url, self.run.as_str());
        let resp = self
            .egress
            .execute(self.authed(EgressRequest::get(&url)), Redirects::None)
            .await
            .map_err(|e| VhcNetError::Transport(format!("fetch archive heads: {e}")))?;
        let status = resp.status();
        if status.as_u16() == 404 {
            return Ok(Vec::new());
        }
        let body = resp
            .bytes()
            .await
            .map_err(|e| VhcNetError::Transport(format!("read archive heads body: {e}")))?;
        if !status.is_success() {
            return Err(VhcNetError::Transport(format!(
                "fetch archive heads returned {status}"
            )));
        }
        let snapshot: HeadsSnapshot = from_canonical_slice(&body)
            .map_err(|e| VhcNetError::Transport(format!("decode archive heads snapshot: {e}")))?;
        Ok(snapshot.heads)
    }
}

// ---- the filesystem plane -------------------------------------------------------------------

/// The chain slot's directory name: `<base_hex>-<role>-<chain_instance>` (the base identity hex
/// is derived from the record's certificate, never from a peer-supplied path component).
fn chain_dir(role: &str, base: &PeerId, chain_instance: u64) -> String {
    let base_hex: String = base.0.iter().map(|b| format!("{b:02x}")).collect();
    format!("{base_hex}-{role}-{chain_instance}")
}

/// A filesystem [`ArchiveHeadStore`]: accepted heads live under a [`ContainedRoot`] at
/// `<root>/<chain dir>/<segment>.head` (canonical CBOR). The normative fold runs in memory over
/// slots loaded from disk at open; an accepted head is durable before the decision returns.
pub struct FsArchiveHeadStore {
    root: ContainedRoot,
    slots: Arc<tokio::sync::Mutex<BTreeMap<String, ArchiveChainSlot>>>,
    /// The ambient disk custodian + charge scope (Phase 6). Heads are tiny and
    /// consensus-critical — [`daemon_vhc_custody::WriteClass::Critical`]: refusing head
    /// publication at quota would wedge the very archive-then-prune reclaim that relieves the
    /// pressure.
    custody: Option<(Arc<daemon_vhc_custody::DiskCustodian>, String)>,
}

impl FsArchiveHeadStore {
    /// Open a store rooted at `root` (created if missing), loading every persisted head.
    ///
    /// # Errors
    /// [`VhcNetError::Transport`] if the root cannot be opened or a persisted head is unreadable
    /// (a corrupt store must refuse loudly, not silently fork a chain).
    pub async fn open(root: &Path) -> Result<Self, VhcNetError> {
        let custody = daemon_vhc_custody::ambient_for(root);
        let contained = ContainedRoot::open(root)
            .map_err(|e| VhcNetError::Transport(format!("open archive head root: {e}")))?;
        let mut slots: BTreeMap<String, ArchiveChainSlot> = BTreeMap::new();
        let chain_entries = contained
            .read_dir(Path::new(""))
            .await
            .map_err(|e| VhcNetError::Transport(format!("list archive head root: {e}")))?;
        for chain_entry in chain_entries {
            if !chain_entry.meta.is_dir {
                continue;
            }
            let chain = chain_entry.name;
            let head_entries = contained
                .read_dir(Path::new(&chain))
                .await
                .map_err(|e| VhcNetError::Transport(format!("list archive chain {chain}: {e}")))?;
            let mut heads: Vec<(u64, ArchiveHeadRecord)> = Vec::new();
            for head_entry in head_entries {
                let file = head_entry.name;
                let Some(stem) = file.strip_suffix(".head") else {
                    continue;
                };
                let Ok(segment) = stem.parse::<u64>() else {
                    continue;
                };
                let bytes = contained
                    .read(Path::new(&format!("{chain}/{file}")))
                    .await
                    .map_err(|e| {
                        VhcNetError::Transport(format!("read archive head {chain}/{file}: {e}"))
                    })?;
                let record: ArchiveHeadRecord = from_canonical_slice(&bytes).map_err(|e| {
                    VhcNetError::Transport(format!("decode archive head {chain}/{file}: {e}"))
                })?;
                heads.push((segment, record));
            }
            heads.sort_by_key(|(seg, _)| *seg);
            let mut slot = ArchiveChainSlot::new();
            for (_, record) in heads {
                if !matches!(
                    slot.fold(record),
                    ArchiveHeadDecision::Accepted | ArchiveHeadDecision::AlreadyStored
                ) {
                    return Err(VhcNetError::Transport(format!(
                        "archive head store {chain}: persisted heads do not fold to one chain"
                    )));
                }
            }
            slots.insert(chain, slot);
        }
        Ok(Self {
            root: contained,
            slots: Arc::new(tokio::sync::Mutex::new(slots)),
            custody,
        })
    }
}

#[async_trait]
impl ArchiveHeadStore for FsArchiveHeadStore {
    async fn put_head(
        &self,
        record: &ArchiveHeadRecord,
    ) -> Result<ArchiveHeadDecision, VhcNetError> {
        let (role, base, chain_instance) = record.chain_key();
        let chain = chain_dir(&role, &base, chain_instance);
        let mut slots = self.slots.lock().await;
        let slot = slots.entry(chain.clone()).or_default();
        let decision = slot.fold(record.clone());
        if matches!(decision, ArchiveHeadDecision::Accepted) {
            let bytes = to_canonical_vec(record)
                .map_err(|e| VhcNetError::Transport(format!("encode archive head: {e}")))?;
            let reservation = match &self.custody {
                None => None,
                Some((custodian, scope)) => {
                    match custodian.reserve(
                        scope,
                        bytes.len() as u64,
                        daemon_vhc_custody::WriteClass::Critical,
                    ) {
                        Ok(r) => Some(r),
                        Err(refusal) => {
                            slot.heads.pop();
                            return Err(VhcNetError::Transport(format!(
                                "archive head custody: {}",
                                refusal.to_io()
                            )));
                        }
                    }
                }
            };
            let rel = format!("{chain}/{}.head", record.body.segment);
            if let Err(e) = self.root.write(Path::new(&rel), &bytes).await {
                // The fold advanced but the durable write failed: roll the slot back so the
                // in-memory state never claims a head the disk does not hold.
                slot.heads.pop();
                return Err(VhcNetError::Transport(format!(
                    "write archive head {rel}: {e}"
                )));
            }
            if let Some(r) = reservation {
                r.commit();
            }
        }
        Ok(decision)
    }

    async fn fetch_heads(&self) -> Result<Vec<ArchiveHeadRecord>, VhcNetError> {
        let slots = self.slots.lock().await;
        Ok(slots
            .values()
            .flat_map(|slot| slot.heads.iter().cloned())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_proto::domains::ARCHIVE_HEAD_DOMAIN;
    use daemon_vhc_proto::{
        peer_id, ArchiveHeadBody, CertScope, Hash, RunKeyCertificate, SigningKey,
    };

    fn base_key() -> SigningKey {
        SigningKey::from_bytes(&[0xB0; 32])
    }

    fn run_key() -> SigningKey {
        SigningKey::from_bytes(&[0x4A; 32])
    }

    fn head(segment: u64, prev: Hash, seg_hash: Hash) -> ArchiveHeadRecord {
        let cert = RunKeyCertificate::issue(
            &base_key(),
            CertScope {
                run_id: Hash([0x1D; 32]),
                epoch: 0,
                role: "coordinator".into(),
                instance: 1,
                module_hash: Hash([0x2A; 32]),
            },
            peer_id(&run_key()),
        )
        .expect("cert");
        ArchiveHeadRecord::publish(
            &run_key(),
            cert,
            ArchiveHeadBody {
                domain: ARCHIVE_HEAD_DOMAIN.into(),
                run_id: Hash([0x1D; 32]),
                role: "coordinator".into(),
                chain_instance: 1,
                segment,
                segment_hash: seg_hash,
                prev_hash: prev,
                records: 4,
                instance: 1,
                epoch: 0,
                module: Hash([0x2A; 32]),
                predecessor: None,
                round: None,
            },
        )
        .expect("publish")
    }

    /// The fs plane persists what it accepts: a fresh open over the same root reloads the chain,
    /// republish stays idempotent, and a fork is still refused after the reload — the disk is
    /// the slot, not the process.
    #[tokio::test]
    async fn the_fs_store_survives_reopen_with_fold_semantics_intact() {
        let dir = tempfile::tempdir().unwrap();
        let h0 = head(0, Hash([0; 32]), Hash([0xA0; 32]));
        let h1 = head(1, Hash([0xA0; 32]), Hash([0xA1; 32]));

        {
            let store = FsArchiveHeadStore::open(dir.path()).await.unwrap();
            assert_eq!(
                store.put_head(&h0).await.unwrap(),
                ArchiveHeadDecision::Accepted
            );
            assert_eq!(
                store.put_head(&h1).await.unwrap(),
                ArchiveHeadDecision::Accepted
            );
        }

        let store = FsArchiveHeadStore::open(dir.path()).await.unwrap();
        assert_eq!(
            store.put_head(&h1).await.unwrap(),
            ArchiveHeadDecision::AlreadyStored,
            "republish after reopen is idempotent"
        );
        let fork = head(1, Hash([0xA0; 32]), Hash([0xFF; 32]));
        assert_eq!(
            store.put_head(&fork).await.unwrap(),
            ArchiveHeadDecision::RejectedNonExtending {
                stored_segment: 1,
                stored_segment_hash: Hash([0xA1; 32]),
            },
            "a conflicting head at a stored height is fork evidence after reopen too"
        );
        let heads = store.fetch_heads().await.unwrap();
        assert_eq!(heads.len(), 2);
        assert_eq!(heads[0], h0);
        assert_eq!(heads[1], h1);
    }
}
