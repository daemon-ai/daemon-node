// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **Archive-then-prune with dependency closure** + the persisted per-chain custody ledger
//! (Phase 6; architecture resource doctrine).
//!
//! The invariant: VHC retains local bytes only while required by an active run, the recovery
//! horizon, or explicit retention policy. A sealed journal segment becomes locally PRUNABLE only
//! when its full dependency closure is satisfied:
//!
//! 1. the segment OBJECT is durably archived (published to the content plane, address-verified),
//! 2. its authenticated HEAD is stored (the store's fold accepted the attested record),
//! 3. it sits outside the RECOVERY HORIZON (the newest `horizon` sealed segments stay local so a
//!    coordinator reconstruction / replay starts warm — ABI §8.8 [AR-8] falls back to the
//!    content plane, hash-verified, for anything pruned),
//! 4. every SIDECAR it references is either pruned with it or still referenced by a RETAINED
//!    segment (content addresses can repeat across records; a sidecar is deleted only when no
//!    retained segment references its address).
//!
//! Facts 1+2 are recorded in the **persisted custody ledger** (`custody.cbor` beside the
//! segments, canonical CBOR, atomic tmp→rename) the moment the publisher's head publish is
//! acknowledged — the ledger is what the node's startup orphan reconciliation trusts ("reclaim
//! only what the metadata proves archived", never merely "not running"), and what makes prune
//! decisions crash-safe (a crash between archive and prune re-reads the ledger and resumes).
//!
//! The unsealed tail is NEVER prune material (it is the reconstruction input for the chain's
//! successor), and nothing here touches the payload plane (content-addressed, run-scoped — its
//! retention is the cache bound's business).

// Sanctioned raw-fs home (like the journal substrate it prunes): the ledger + segment files live
// in the host-owned journal root; paths derive from ordinals/content hashes, never from
// attacker-influenced input.
#![allow(clippy::disallowed_methods)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use daemon_vhc_journal::{scan_file, JournalPaths};
use daemon_vhc_proto::{from_canonical_slice, to_canonical_vec, Hash};
use serde::{Deserialize, Serialize};

/// The custody ledger's file name, beside the segment files in the journal home.
pub const CUSTODY_LEDGER_FILE: &str = "custody.cbor";

/// One sealed segment's custody facts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentCustody {
    /// The segment ordinal.
    pub segment: u64,
    /// The segment's complete-file blake3 (its content-plane address).
    pub blake3: Hash,
    /// Records the segment carries (excluding the seal frame) — reporting.
    pub records: u64,
    /// The archive facts hold: bytes at the content plane + attested head stored.
    pub archived: bool,
    /// The local segment file (and its exclusively-referenced sidecars) were reclaimed.
    pub pruned: bool,
}

/// The persisted per-chain custody ledger (see the module docs).
#[derive(Debug, Default)]
pub struct CustodyLedger {
    path: PathBuf,
    entries: BTreeMap<u64, SegmentCustody>,
}

impl CustodyLedger {
    /// Load the ledger beside the journal at `journal_dir` (empty when absent — a chain that
    /// never archived has no custody facts).
    ///
    /// # Errors
    /// [`std::io::Error`] on an unreadable or undecodable ledger file (a corrupt ledger must
    /// refuse loudly — silently starting empty would let reconciliation "prove" nothing was
    /// archived and never reclaim, or worse, let a prune re-run without its facts).
    pub fn load(journal_dir: &Path) -> std::io::Result<Self> {
        let path = journal_dir.join(CUSTODY_LEDGER_FILE);
        let entries = match std::fs::read(&path) {
            Ok(bytes) => {
                let list: Vec<SegmentCustody> = from_canonical_slice(&bytes)
                    .map_err(|e| std::io::Error::other(format!("custody ledger decode: {e}")))?;
                list.into_iter().map(|e| (e.segment, e)).collect()
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(e),
        };
        Ok(Self { path, entries })
    }

    /// The entries, ascending by ordinal.
    pub fn entries(&self) -> impl Iterator<Item = &SegmentCustody> {
        self.entries.values()
    }

    /// One segment's custody facts.
    #[must_use]
    pub fn get(&self, segment: u64) -> Option<&SegmentCustody> {
        self.entries.get(&segment)
    }

    /// Record that `segment`'s archive facts hold (bytes published + head stored) and persist.
    ///
    /// # Errors
    /// [`std::io::Error`] on a persist failure.
    pub fn record_archived(
        &mut self,
        segment: u64,
        blake3: Hash,
        records: u64,
    ) -> std::io::Result<()> {
        self.entries
            .entry(segment)
            .and_modify(|e| e.archived = true)
            .or_insert(SegmentCustody {
                segment,
                blake3,
                records,
                archived: true,
                pruned: false,
            });
        self.save()
    }

    /// Record that `segment` was locally pruned and persist.
    ///
    /// # Errors
    /// [`std::io::Error`] on a persist failure.
    pub fn record_pruned(&mut self, segment: u64) -> std::io::Result<()> {
        if let Some(e) = self.entries.get_mut(&segment) {
            e.pruned = true;
        }
        self.save()
    }

    /// Atomic persist (tmp → rename): a torn ledger is never observed.
    fn save(&self) -> std::io::Result<()> {
        let list: Vec<&SegmentCustody> = self.entries.values().collect();
        let bytes = to_canonical_vec(&list)
            .map_err(|e| std::io::Error::other(format!("custody ledger encode: {e}")))?;
        let tmp = self.path.with_extension("cbor.tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }
}

/// What one prune pass reclaimed (logging/reporting).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct PruneOutcome {
    /// Segment files deleted.
    pub segments: u64,
    /// Sidecar files deleted.
    pub sidecars: u64,
    /// Total bytes reclaimed (segments + sidecars).
    pub bytes: u64,
}

/// One archive-then-prune pass over the chain at `journal_dir` (see the module docs for the
/// dependency closure). `horizon` is the recovery horizon: the newest `horizon` ARCHIVED sealed
/// segments stay local; `0` disables pruning entirely (the explicit "retain everything" policy).
///
/// Idempotent and crash-safe: every fact consulted is either on disk (segment files) or in the
/// persisted ledger; a re-run after a crash converges to the same state. Reclaimed bytes are
/// discharged from the ambient disk custodian's ledger (the same scope the writes charged).
///
/// # Errors
/// [`std::io::Error`] on a ledger persist failure or an unlistable journal home. Individual
/// file deletions are best-effort (a vanished file is already pruned).
pub fn prune_archived(
    journal_dir: &Path,
    ledger: &mut CustodyLedger,
    horizon: u64,
) -> std::io::Result<PruneOutcome> {
    let mut outcome = PruneOutcome::default();
    if horizon == 0 {
        return Ok(outcome);
    }
    let paths = JournalPaths::open(journal_dir)
        .map_err(|e| std::io::Error::other(format!("journal home: {e}")))?;
    let ordinals = paths
        .existing_segments()
        .map_err(|e| std::io::Error::other(format!("list segments: {e}")))?;

    // The archived tip judges the horizon; segments the ledger cannot prove archived are never
    // prune material, however old.
    let Some(archived_tip) = ledger
        .entries()
        .filter(|e| e.archived)
        .map(|e| e.segment)
        .max()
    else {
        return Ok(outcome);
    };
    // Only a contiguous PREFIX is ever prune material: recovery walks one unbroken hash chain
    // from the (anchored) first retained segment, so pruning past a mid-chain hole — an
    // unarchived or out-of-horizon segment — would leave a locally unverifiable chain. The
    // first segment that fails the closure stops the pass.
    let prunable: Vec<u64> = ordinals
        .iter()
        .copied()
        .take_while(|ord| {
            archived_tip.checked_sub(*ord).is_some_and(|d| d >= horizon)
                && ledger.get(*ord).is_some_and(|e| e.archived && !e.pruned)
        })
        .collect();
    if prunable.is_empty() {
        return Ok(outcome);
    }

    // The sidecar closure: collect the content addresses every RETAINED segment still
    // references; a pruned segment's sidecar is deleted only when retained references are gone.
    // (Sealed segments are immutable and few post-prune, so the scan is cheap; the unsealed
    // tail is scanned too — its references are as retained as it is.)
    let sidecar_refs = |ord: u64| -> Vec<daemon_vhc_journal::record::SidecarRef> {
        scan_file(paths.segment(ord)).map_or_else(
            |_| Vec::new(),
            |scan| {
                scan.records
                    .iter()
                    .filter_map(|r| match &r.body {
                        daemon_vhc_journal::Body::ReadBack(rb) => rb.sidecar.clone(),
                        _ => None,
                    })
                    .collect()
            },
        )
    };
    let retained: BTreeSet<Hash> = ordinals
        .iter()
        .filter(|ord| !prunable.contains(ord))
        .flat_map(|ord| sidecar_refs(*ord))
        .map(|sref| sref.hash)
        .collect();

    let custody = daemon_vhc_custody::ambient_for(journal_dir);
    let discharge = |path: &Path| {
        let bytes = std::fs::symlink_metadata(path)
            .map(|m| m.len())
            .unwrap_or(0);
        if std::fs::remove_file(path).is_ok() {
            if let Some((custodian, scope)) = &custody {
                custodian.discharge(scope, bytes);
            }
            return bytes;
        }
        0
    };

    for ord in prunable {
        // The chain anchor FIRST (atomic, fsynced): once it names `ord + 1` as the chain's first
        // retained segment (carrying the archived `ord`'s complete-file hash for `prev_blake3`
        // verification), recovery treats a still-present `ord` as skippable prune debris — so a
        // crash anywhere below leaves a re-openable journal.
        let blake3 = ledger
            .get(ord)
            .map(|e| e.blake3)
            .expect("prunable implies a ledger entry");
        daemon_vhc_journal::ChainAnchor {
            first_ord: ord + 1,
            prev_blake3: blake3.0,
        }
        .store(&paths)
        .map_err(|e| std::io::Error::other(format!("chain anchor: {e}")))?;
        // Sidecars next (they reference the segment, not vice versa): while the segment file
        // still exists, a crash mid-prune leaves a re-scannable chain.
        for sref in sidecar_refs(ord) {
            if retained.contains(&sref.hash) {
                continue;
            }
            let path = paths
                .sidecars()
                .join(format!("{}.dvhcsc", sref.hash.to_hex()));
            let bytes = discharge(&path);
            if bytes > 0 {
                outcome.sidecars += 1;
                outcome.bytes += bytes;
            }
        }
        let bytes = discharge(&paths.segment(ord));
        if bytes > 0 {
            outcome.segments += 1;
            outcome.bytes += bytes;
        }
        ledger.record_pruned(ord)?;
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_journal::record::ExecIdentity;
    use daemon_vhc_journal::{Journal, RotatePolicy, StaticKey};

    use crate::journal_home::journal_dir;

    fn identity() -> ExecIdentity {
        ExecIdentity {
            run_id: Hash([0x71; 32]),
            epoch: 0,
            role: "coordinator".into(),
            instance: 1,
            module: Hash([0x72; 32]),
        }
    }

    /// Build a chain with several sealed segments (each carrying a sidecar read-back), mark a
    /// prefix archived, and prune at a horizon: only the archived segments outside the horizon
    /// go, their exclusively-referenced sidecars go with them, the ledger records the prune, and
    /// a re-run is a no-op. Unarchived segments never prune, however old.
    #[test]
    fn prune_respects_the_dependency_closure_and_the_horizon() {
        let dir = tempfile::tempdir().unwrap();
        let jdir = journal_dir(dir.path(), "prune-run", "coordinator", 1);
        {
            let mut journal = Journal::create(
                &jdir,
                identity(),
                StaticKey::new([0x5C; 32]),
                RotatePolicy::default(),
            )
            .unwrap();
            // Four sealed segments, each with one oversize read-back (a sidecar lands). The
            // values DIFFER per segment (distinct content addresses) — an identical value would
            // rightly share one sidecar with the retained segments and never prune.
            for seg in 0u64..4 {
                let oversize = vec![seg as u8; daemon_vhc_abi::READBACK_INLINE_MAX + 1];
                journal.read_back(seg, 0, 0, &oversize).unwrap();
                journal
                    .append(daemon_vhc_journal::Body::Event(
                        daemon_vhc_journal::record::EventRec {
                            at: seg,
                            frame: b"frame".to_vec(),
                        },
                    ))
                    .unwrap();
                journal.roll().unwrap();
            }
        }
        let paths = JournalPaths::open(&jdir).unwrap();
        let scan0 = scan_file(paths.segment(0)).unwrap();
        assert!(scan0.sealed);

        // The ledger proves segments 0..=2 archived; 3 sealed-but-unarchived.
        let mut ledger = CustodyLedger::load(&jdir).unwrap();
        for ord in 0..3u64 {
            let scan = scan_file(paths.segment(ord)).unwrap();
            ledger
                .record_archived(
                    ord,
                    Hash(scan.complete_file_blake3),
                    scan.records.len() as u64,
                )
                .unwrap();
        }

        // Horizon 1: archived tip is 2, so only segments 0 and 1 are outside the horizon.
        let outcome = prune_archived(&jdir, &mut ledger, 1).unwrap();
        assert_eq!(outcome.segments, 2, "0 and 1 pruned; 2 inside the horizon");
        assert!(outcome.sidecars >= 2, "their exclusive sidecars went too");
        assert!(!paths.segment(0).exists() && !paths.segment(1).exists());
        assert!(paths.segment(2).exists(), "the horizon keeps 2 local");
        assert!(
            paths.segment(3).exists(),
            "unarchived 3 is never prune material"
        );

        // The ledger persisted the prune facts; a reload + re-run is a no-op.
        let mut reloaded = CustodyLedger::load(&jdir).unwrap();
        assert!(reloaded.get(0).unwrap().pruned && reloaded.get(1).unwrap().pruned);
        assert!(!reloaded.get(2).unwrap().pruned);
        let rerun = prune_archived(&jdir, &mut reloaded, 1).unwrap();
        assert_eq!(rerun, PruneOutcome::default());

        // Horizon 0 = pruning disabled.
        let none = prune_archived(&jdir, &mut reloaded, 0).unwrap();
        assert_eq!(none, PruneOutcome::default());
    }

    /// The regression behind the live module-switch failure: a chain whose archived prefix was
    /// pruned must still RE-OPEN — plain crash recovery AND the switch seam's continuation
    /// (`open_continuation` rolls the same file series to the successor identity). The pruner's
    /// chain anchor carries the pruned predecessor's hash, so recovery verifies from the first
    /// retained segment instead of refusing at genesis; a leftover pre-anchor segment (crash
    /// between the anchor write and the file delete) is skipped as prune debris.
    #[test]
    fn a_pruned_chain_reopens_and_continues_across_the_seam() {
        let dir = tempfile::tempdir().unwrap();
        let jdir = journal_dir(dir.path(), "prune-reopen", "trainer", 1);
        let key = StaticKey::new([0x5D; 32]);
        {
            let mut journal =
                Journal::create(&jdir, identity(), key.clone(), RotatePolicy::default()).unwrap();
            for seg in 0u64..4 {
                journal
                    .append(daemon_vhc_journal::Body::Event(
                        daemon_vhc_journal::record::EventRec {
                            at: seg,
                            frame: b"frame".to_vec(),
                        },
                    ))
                    .unwrap();
                journal.roll().unwrap();
            }
        }
        let paths = JournalPaths::open(&jdir).unwrap();
        let mut ledger = CustodyLedger::load(&jdir).unwrap();
        for ord in 0..3u64 {
            let scan = scan_file(paths.segment(ord)).unwrap();
            ledger
                .record_archived(
                    ord,
                    Hash(scan.complete_file_blake3),
                    scan.records.len() as u64,
                )
                .unwrap();
        }
        let outcome = prune_archived(&jdir, &mut ledger, 1).unwrap();
        assert_eq!(outcome.segments, 2, "0 and 1 pruned");

        // Plain re-open (crash recovery over the pruned chain) appends and rolls.
        {
            let mut journal =
                Journal::open(&jdir, identity(), key.clone(), RotatePolicy::default())
                    .expect("a pruned chain re-opens from its anchor");
            journal
                .append(daemon_vhc_journal::Body::Event(
                    daemon_vhc_journal::record::EventRec {
                        at: 99,
                        frame: b"post-prune".to_vec(),
                    },
                ))
                .unwrap();
        }

        // The switch seam: the successor incarnation continues the SAME file series.
        let successor = ExecIdentity {
            instance: 2,
            ..identity()
        };
        Journal::open_continuation(&jdir, successor, key.clone(), RotatePolicy::default(), None)
            .expect("the seam continuation opens a pruned chain");

        // Crash window: the anchor advanced past a segment whose file still exists (prune died
        // between the anchor write and the delete). Recovery skips the debris.
        let first_retained = *paths.existing_segments().unwrap().first().unwrap();
        let scan = scan_file(paths.segment(first_retained)).unwrap();
        daemon_vhc_journal::ChainAnchor {
            first_ord: first_retained + 1,
            prev_blake3: scan.complete_file_blake3,
        }
        .store(&paths)
        .unwrap();
        Journal::open(&jdir, identity(), key, RotatePolicy::default())
            .expect("a leftover pre-anchor segment is skipped as prune debris");
    }
}
