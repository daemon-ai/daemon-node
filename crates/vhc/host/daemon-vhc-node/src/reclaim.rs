// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **Manifest-driven orphan reconciliation** (Phase 6): reclaim a dead incarnation's run-state
//! directory at startup — but only when the persisted custody metadata PROVES it, never merely
//! because it is "not running" (a dead incarnation may hold the only reconstruction journal).
//!
//! The rules, per incarnation directory `<runs root>/<blake3(label)>/<role>-<incarnation>/`:
//!
//! - **Superseded**: a HIGHER incarnation directory for the same role exists in the same run
//!   scope. The newest incarnation is NEVER reclaimed here, whatever the run state — it is the
//!   next join's reconstruction input (ABI §8.8 [AR-8] reads its local unsealed tail) and the
//!   replay oracle's warmest source. Full reclamation of a finished run is the operator's
//!   explicit safe-wipe, not startup housekeeping.
//! - **Archived, by the ledger**: every segment of the incarnation's journal is recorded
//!   archived (or already pruned) in the chain's persisted custody ledger
//!   ([`daemon_vhc_session::custody::CustodyLedger`]) — with one shape-exception: a trailing
//!   UNSEALED, EMPTY segment (the successor file a terminal tail-seal opened) holds no records
//!   and needs no archive. An unarchived record anywhere ⇒ the directory is retained, loudly.
//! - **The spill is instance garbage**: a superseded incarnation's `state/` spill is
//!   reclaimable UNCONDITIONALLY ([SF-4]: a torn/unsealed fold is never durable; restores come
//!   from the checkpoint payload plane, never a dead spill).
//! - **Unknown scopes are untouchable**: a directory under the runs root with no matching run
//!   row is never reclaimed automatically (it may belong to a run whose row lives in another
//!   node's store — adjudicate by hand).
//!
//! The run scope's `payload/` and `archive/` directories are ARCHIVE PLANES on a filesystem
//! deployment — they hold the published copies the pruned journals point at — and are never
//! touched here.

// Sanctioned raw-fs home: reconciliation walks the node-owned runs root; every path derives
// from run labels the store persisted and directory names the node itself created.
#![allow(clippy::disallowed_methods)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use daemon_vhc_custody::DiskCustodian;
use daemon_vhc_session::custody::CustodyLedger;

/// What one reconciliation pass did (logging/reporting).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ReclaimReport {
    /// Superseded incarnation directories fully reclaimed (journal proven archived).
    pub incarnations: u64,
    /// Superseded incarnations whose SPILL was reclaimed while the journal was retained
    /// (unproven archive facts — the journal stays, loudly).
    pub spills_only: u64,
    /// Total bytes reclaimed.
    pub bytes: u64,
    /// Directories retained with the reason (label-hash/dir, reason) — the loud trail.
    pub retained: Vec<(String, String)>,
}

/// One startup reconciliation pass over `runs_root` (see the module docs for the rules).
/// `known_scopes` is the store's truth: the `blake3(label)` hex of every persisted run row.
/// Reclaimed bytes are discharged from `custodian`'s ledger when one is wired.
///
/// # Errors
/// [`std::io::Error`] if the runs root cannot be listed. Per-directory failures retain the
/// directory and record the reason — one broken scope never blocks the pass.
pub fn reconcile_orphans(
    runs_root: &Path,
    known_scopes: &BTreeSet<String>,
    custodian: Option<&std::sync::Arc<DiskCustodian>>,
) -> std::io::Result<ReclaimReport> {
    let mut report = ReclaimReport::default();
    let entries = match std::fs::read_dir(runs_root) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(report),
        Err(e) => return Err(e),
    };
    for entry in entries.flatten() {
        let scope = entry.file_name().to_string_lossy().into_owned();
        let scope_path = entry.path();
        if !scope_path.is_dir() {
            continue;
        }
        if !known_scopes.contains(&scope) {
            report.retained.push((
                scope.clone(),
                "no run row matches this scope; never reclaimed automatically".into(),
            ));
            continue;
        }
        reconcile_scope(&scope, &scope_path, custodian, &mut report);
    }
    Ok(report)
}

/// Reconcile one run scope: reclaim superseded incarnation directories the ledger proves out.
fn reconcile_scope(
    scope: &str,
    scope_path: &Path,
    custodian: Option<&std::sync::Arc<DiskCustodian>>,
    report: &mut ReclaimReport,
) {
    // Collect incarnation dirs per role; everything else (payload/, archive/) is plane storage.
    let mut by_role: BTreeMap<String, Vec<(u64, PathBuf)>> = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(scope_path) else {
        report
            .retained
            .push((scope.into(), "scope unlistable".into()));
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !entry.path().is_dir() {
            continue;
        }
        let Some((role, inc)) = name.rsplit_once('-') else {
            continue;
        };
        let Ok(inc) = inc.parse::<u64>() else {
            continue;
        };
        by_role
            .entry(role.to_string())
            .or_default()
            .push((inc, entry.path()));
    }

    for (role, mut incs) in by_role {
        incs.sort_by_key(|(inc, _)| *inc);
        let Some(newest) = incs.last().map(|(inc, _)| *inc) else {
            continue;
        };
        for (inc, dir) in incs {
            if inc == newest {
                continue; // the newest incarnation is the reconstruction input — never here
            }
            let tag = format!("{scope}/{role}-{inc}");
            // The spill is instance garbage for any superseded incarnation.
            let spill_bytes = remove_tree(&dir.join("state"));
            if spill_bytes > 0 {
                report.bytes += spill_bytes;
            }
            match journal_fully_archived(&dir.join("journal")) {
                Ok(true) => {
                    let bytes = remove_tree(&dir);
                    report.incarnations += 1;
                    report.bytes += bytes;
                    if let Some(c) = custodian {
                        c.discharge(scope, bytes + spill_bytes);
                    }
                }
                Ok(false) => {
                    if spill_bytes > 0 {
                        report.spills_only += 1;
                        if let Some(c) = custodian {
                            c.discharge(scope, spill_bytes);
                        }
                    }
                    report.retained.push((
                        tag,
                        "custody ledger does not prove every record archived".into(),
                    ));
                }
                Err(e) => {
                    if spill_bytes > 0 {
                        report.spills_only += 1;
                        if let Some(c) = custodian {
                            c.discharge(scope, spill_bytes);
                        }
                    }
                    report
                        .retained
                        .push((tag, format!("journal unreadable: {e}")));
                }
            }
        }
    }
}

/// The archive proof: every segment present in the journal directory is recorded archived (or
/// pruned) in the persisted custody ledger — except a trailing unsealed EMPTY segment (the
/// successor file a terminal tail-seal opened), which holds no records. An absent journal
/// directory is vacuously archived (a seat that never journaled durably).
fn journal_fully_archived(journal_dir: &Path) -> std::io::Result<bool> {
    if !journal_dir.exists() {
        return Ok(true);
    }
    let paths = daemon_vhc_journal::JournalPaths::open(journal_dir)
        .map_err(|e| std::io::Error::other(format!("journal home: {e}")))?;
    let ordinals = paths
        .existing_segments()
        .map_err(|e| std::io::Error::other(format!("list segments: {e}")))?;
    let ledger = CustodyLedger::load(journal_dir)?;
    let last = ordinals.last().copied();
    for ord in ordinals {
        if ledger.get(ord).is_some_and(|e| e.archived) {
            continue;
        }
        // The one sanctioned exception: a trailing unsealed segment with zero records.
        if Some(ord) == last {
            let scan = daemon_vhc_journal::scan_file(paths.segment(ord))
                .map_err(|e| std::io::Error::other(format!("scan tail: {e}")))?;
            if !scan.sealed && scan.records.is_empty() {
                continue;
            }
        }
        return Ok(false);
    }
    Ok(true)
}

/// One run scope's bytes split by reclaim class (the `vhc disk` row): `payload/` + `archive/`
/// are the ARCHIVE PLANES (published evidence); everything else — incarnation journals, spills,
/// working files — is recoverable state, rebuildable from that archive.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ScopeBreakdown {
    /// Recoverable-state bytes (journal + spill + working files).
    pub recoverable: u64,
    /// Archived-evidence bytes (payload + archive planes).
    pub evidence: u64,
}

/// Whether a scope entry name is an archive plane (published evidence) rather than
/// recoverable run state.
fn is_evidence_plane(name: &str) -> bool {
    name == "payload" || name == "archive"
}

/// Every scope directory under the runs root with its [`ScopeBreakdown`] (the `vhc disk` rows),
/// in directory order. An absent/unlistable root reads empty.
pub fn scope_rows(runs_root: &Path) -> Vec<(String, ScopeBreakdown)> {
    let Ok(entries) = std::fs::read_dir(runs_root) else {
        return Vec::new();
    };
    let mut rows: Vec<(String, ScopeBreakdown)> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| {
            (
                e.file_name().to_string_lossy().into_owned(),
                scope_breakdown(&e.path()),
            )
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    rows
}

/// Classify one run scope directory's bytes ([`ScopeBreakdown`]). Absent scope ⇒ all zeros.
pub fn scope_breakdown(scope_path: &Path) -> ScopeBreakdown {
    let mut out = ScopeBreakdown::default();
    let Ok(entries) = std::fs::read_dir(scope_path) else {
        return out;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let bytes = tree_bytes(&entry.path());
        if is_evidence_plane(&name) {
            out.evidence = out.evidence.saturating_add(bytes);
        } else {
            out.recoverable = out.recoverable.saturating_add(bytes);
        }
    }
    out
}

/// What one safe wipe removed.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct WipeOutcome {
    /// Bytes reclaimed.
    pub bytes: u64,
    /// Whether the archive planes went too (`include_evidence` requested AND bytes existed).
    pub wiped_evidence: bool,
}

/// The operator's safe wipe of ONE run scope: recoverable state (incarnation journals, spills,
/// working files) always goes; the archive planes (`payload/`, `archive/`) go only when
/// `include_evidence`. The identity keystore lives OUTSIDE the runs root and is never in
/// reach. Idempotent: an absent scope wipes to a zero outcome.
///
/// # Errors
/// [`std::io::Error`] if the scope directory exists but cannot be listed.
pub fn wipe_scope(scope_path: &Path, include_evidence: bool) -> std::io::Result<WipeOutcome> {
    let mut out = WipeOutcome::default();
    let entries = match std::fs::read_dir(scope_path) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e),
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if is_evidence_plane(&name) {
            if !include_evidence {
                continue;
            }
            let bytes = remove_tree(&entry.path());
            out.bytes = out.bytes.saturating_add(bytes);
            out.wiped_evidence |= bytes > 0;
        } else {
            out.bytes = out.bytes.saturating_add(remove_tree(&entry.path()));
        }
    }
    // Drop the scope shell itself once nothing (or only nothing) remains.
    let _ = std::fs::remove_dir(scope_path);
    Ok(out)
}

/// Best-effort recursive removal, returning the bytes it reclaimed (0 for an absent tree).
fn remove_tree(path: &Path) -> u64 {
    let bytes = tree_bytes(path);
    if std::fs::remove_dir_all(path).is_ok() {
        bytes
    } else {
        0
    }
}

/// Total regular-file bytes under `path` (advisory accounting for the discharge).
fn tree_bytes(path: &Path) -> u64 {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return 0;
    };
    if meta.is_file() {
        return meta.len();
    }
    if !meta.is_dir() {
        return 0;
    }
    std::fs::read_dir(path).map_or(0, |entries| {
        entries
            .flatten()
            .map(|e| tree_bytes(&e.path()))
            .fold(0u64, u64::saturating_add)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_journal::record::{EventRec, ExecIdentity};
    use daemon_vhc_journal::{Body, Journal, RotatePolicy, StaticKey};
    use daemon_vhc_proto::Hash;

    fn identity(instance: u64) -> ExecIdentity {
        ExecIdentity {
            run_id: Hash([0x1D; 32]),
            epoch: 0,
            role: "trainer".into(),
            instance,
            module: Hash([0x2A; 32]),
        }
    }

    /// One sealed segment (+ the empty successor the roll opens) at `dir`; returns the sealed
    /// segment's complete-file blake3 + record count for the ledger.
    fn seed_journal(dir: &Path, instance: u64) -> (Hash, u64) {
        let mut journal = Journal::create(
            dir,
            identity(instance),
            StaticKey::new([0x5C; 32]),
            RotatePolicy::default(),
        )
        .unwrap();
        journal
            .append(Body::Event(EventRec {
                at: 1,
                frame: b"frame".to_vec(),
            }))
            .unwrap();
        journal.roll().unwrap();
        let paths = daemon_vhc_journal::JournalPaths::open(dir).unwrap();
        let scan = daemon_vhc_journal::scan_file(paths.segment(0)).unwrap();
        (Hash(scan.complete_file_blake3), scan.records.len() as u64)
    }

    /// Superseded + ledger-proven ⇒ reclaimed; superseded + unproven ⇒ retained (spill still
    /// reclaimed); the newest incarnation and unknown scopes are never touched.
    #[test]
    fn reconciliation_reclaims_only_proven_superseded_incarnations() {
        let root = tempfile::tempdir().unwrap();
        let scope = blake3::hash(b"run-x").to_hex().to_string();
        let scope_path = root.path().join(&scope);

        // trainer-1: superseded, archived (ledger proof) → fully reclaimed.
        let (hash1, records1) = seed_journal(&scope_path.join("trainer-1/journal"), 1);
        let mut ledger = CustodyLedger::load(&scope_path.join("trainer-1/journal")).unwrap();
        ledger.record_archived(0, hash1, records1).unwrap();

        // trainer-2: superseded, NO archive proof → journal retained, spill reclaimed.
        seed_journal(&scope_path.join("trainer-2/journal"), 2);
        std::fs::create_dir_all(scope_path.join("trainer-2/state")).unwrap();
        std::fs::write(scope_path.join("trainer-2/state/chunk"), vec![0u8; 64]).unwrap();

        // trainer-3: the newest incarnation → never touched, however unproven.
        seed_journal(&scope_path.join("trainer-3/journal"), 3);

        // An unknown scope → never touched.
        std::fs::create_dir_all(root.path().join("deadbeef/trainer-1/journal")).unwrap();

        // The archive planes → never touched.
        std::fs::create_dir_all(scope_path.join("payload")).unwrap();
        std::fs::create_dir_all(scope_path.join("archive/heads")).unwrap();

        let known: BTreeSet<String> = [scope.clone()].into();
        let report = reconcile_orphans(root.path(), &known, None).unwrap();

        assert_eq!(
            report.incarnations, 1,
            "only the proven superseded dir goes"
        );
        assert!(!scope_path.join("trainer-1").exists());
        assert!(
            scope_path.join("trainer-2/journal").exists(),
            "unproven journal retained"
        );
        assert!(
            !scope_path.join("trainer-2/state").exists(),
            "dead spill reclaimed [SF-4]"
        );
        assert_eq!(report.spills_only, 1);
        assert!(
            scope_path.join("trainer-3").exists(),
            "the newest is the recovery input"
        );
        assert!(
            root.path().join("deadbeef").exists(),
            "unknown scopes untouchable"
        );
        assert!(scope_path.join("payload").exists() && scope_path.join("archive/heads").exists());
        assert!(
            report.retained.iter().any(|(d, _)| d.contains("trainer-2")),
            "the retained journal is loud: {:?}",
            report.retained
        );
        assert!(report.bytes > 0);

        // Idempotent: a second pass reclaims nothing further.
        let rerun = reconcile_orphans(root.path(), &known, None).unwrap();
        assert_eq!(rerun.incarnations, 0);
        assert_eq!(rerun.spills_only, 0);
    }

    /// The safe wipe: recoverable state always goes; the archive planes survive the default
    /// wipe and go only on the explicit evidence wipe; absent scopes wipe to a zero outcome.
    #[test]
    fn the_safe_wipe_spares_the_archive_planes_unless_evidence_is_included() {
        let root = tempfile::tempdir().unwrap();
        let scope = root.path().join("scopehash");
        std::fs::create_dir_all(scope.join("trainer-1/journal")).unwrap();
        std::fs::write(scope.join("trainer-1/journal/seg-0.vjl"), vec![1u8; 256]).unwrap();
        std::fs::create_dir_all(scope.join("trainer-1/state")).unwrap();
        std::fs::write(scope.join("trainer-1/state/chunk"), vec![2u8; 128]).unwrap();
        std::fs::create_dir_all(scope.join("payload")).unwrap();
        std::fs::write(scope.join("payload/blob"), vec![3u8; 512]).unwrap();
        std::fs::create_dir_all(scope.join("archive/heads")).unwrap();
        std::fs::write(scope.join("archive/heads/head"), vec![4u8; 64]).unwrap();

        let split = scope_breakdown(&scope);
        assert_eq!(split.recoverable, 256 + 128);
        assert_eq!(split.evidence, 512 + 64);

        let default_wipe = wipe_scope(&scope, false).unwrap();
        assert_eq!(default_wipe.bytes, 256 + 128, "recoverable state only");
        assert!(!default_wipe.wiped_evidence);
        assert!(!scope.join("trainer-1").exists());
        assert!(
            scope.join("payload/blob").exists(),
            "evidence survives the default wipe"
        );

        let evidence_wipe = wipe_scope(&scope, true).unwrap();
        assert_eq!(evidence_wipe.bytes, 512 + 64);
        assert!(evidence_wipe.wiped_evidence);
        assert!(!scope.exists(), "an emptied scope shell is dropped");

        let rerun = wipe_scope(&scope, true).unwrap();
        assert_eq!(
            rerun,
            WipeOutcome::default(),
            "idempotent on an absent scope"
        );
    }
}
