// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The durable coprocessor state: spaces of data records, versioned procedures, an append-only
//! journal with monotonic snapshot ids, and the candidate -> active -> rollback lifecycle.
//!
//! This layer is **engine-independent** (no `hyperon`): it owns the SKILL semantics that are pure
//! bookkeeping — provenance, idempotency, snapshot CAS, supersession, procedure versioning, the
//! promotion gate, and rollback. The engine ([`crate::engine`]) only evaluates/matches atoms; it
//! reads the records this layer holds.
//!
//! Persistence is an append-only `journal.jsonl` of [`Mutation`]s under a state dir. The live state
//! is the replay of that journal, so a restart reconstructs the exact store; the snapshot id is the
//! count of applied mutations, which doubles as the optimistic-concurrency (CAS) token.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::protocol::{Provenance, RetractTarget, Space, Status};

/// One data record in a space (the SKILL canonical memory shape, stored as MeTTa source text plus
/// the side metadata the engine does not need to parse).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Record {
    /// The stable record id.
    pub id: String,
    /// The space the record lives in.
    pub space: Space,
    /// The MeTTa atom source (the data atom; never an executable `=` rule — those are procedures).
    pub text: String,
    /// Where the record came from.
    pub provenance: Provenance,
    /// The lifecycle status.
    pub status: Status,
    /// The snapshot at which the record was created.
    pub created_at_snapshot: u64,
    /// The id of a record this one supersedes (correction without deletion), if any.
    #[serde(default)]
    pub supersedes: Option<String>,
}

/// One immutable version of a procedure (SKILL §Procedural memory).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcedureVersion {
    /// The 1-based version number.
    pub version: u64,
    /// The executable MeTTa program source.
    pub program: String,
    /// Opaque metadata (intent, preconditions, ...), as a JSON string.
    pub metadata: Option<String>,
    /// The declared test sources.
    pub tests: Vec<String>,
    /// This version's status (`Candidate`, `Active`, `Retired`).
    pub status: Status,
    /// Promotion evidence (trajectory ids, metrics) recorded at promotion time.
    pub evidence: Vec<String>,
}

/// A versioned procedure: an ordered history of versions plus the currently-active one (if any).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Procedure {
    /// The stable procedure id.
    pub id: String,
    /// Every version, in creation order.
    pub versions: Vec<ProcedureVersion>,
    /// The active version number, if a version has been promoted.
    pub active_version: Option<u64>,
}

impl Procedure {
    /// The latest version (highest version number).
    pub fn latest(&self) -> Option<&ProcedureVersion> {
        self.versions.iter().max_by_key(|v| v.version)
    }

    /// The active version, if any.
    pub fn active(&self) -> Option<&ProcedureVersion> {
        let v = self.active_version?;
        self.versions.iter().find(|pv| pv.version == v)
    }
}

/// A single committed mutation. The live state is the replay of the journal of these, so the same
/// [`MettaState::apply`] drives both live commits and recovery.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mutation {
    /// Insert a data record.
    Assert(Record),
    /// Mark a record `Superseded` (supersession) or remove it (destructive).
    Retract {
        /// The record id.
        id: String,
        /// `true` removes the row; `false` marks it `Superseded`.
        hard: bool,
    },
    /// Append a new candidate version to a procedure (creating it if new).
    DefineProcedure {
        /// The procedure id.
        id: String,
        /// The new version.
        version: ProcedureVersion,
    },
    /// Promote a candidate version to active.
    Promote {
        /// The procedure id.
        id: String,
        /// The version promoted.
        version: u64,
        /// Promotion evidence.
        evidence: Vec<String>,
    },
    /// Roll back to a prior active version, retiring the failed one.
    Rollback {
        /// The procedure id.
        id: String,
        /// The version restored to active.
        target_version: u64,
        /// The version retired.
        retired_version: u64,
    },
}

/// A CAS / validation error from a mutating op.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StateError {
    /// The caller's `expected_snapshot` did not match the current snapshot.
    #[error("snapshot mismatch: expected {expected}, current {current}")]
    SnapshotMismatch {
        /// The snapshot the caller expected.
        expected: u64,
        /// The actual current snapshot.
        current: u64,
    },
    /// A referenced id does not exist.
    #[error("not found: {0}")]
    NotFound(String),
    /// The op is invalid (e.g. promoting an already-active version, rollback with no prior active).
    #[error("invalid: {0}")]
    Invalid(String),
    /// Persisting the journal failed.
    #[error("persist: {0}")]
    Persist(String),
}

/// The full coprocessor state.
pub struct MettaState {
    /// Data records by id.
    records: BTreeMap<String, Record>,
    /// Procedures by id.
    procedures: BTreeMap<String, Procedure>,
    /// Idempotency keys already committed -> the ids they produced (assert retries are no-ops).
    idempotency: BTreeMap<String, Vec<String>>,
    /// The monotonic snapshot id (count of applied mutations) — also the CAS token.
    snapshot: u64,
    /// Monotonic id allocator for generated record ids.
    next_id: u64,
    /// The append-only journal file (None = ephemeral / in-memory).
    journal_path: Option<PathBuf>,
}

impl MettaState {
    /// An ephemeral, in-memory state (no persistence) — used for tests and ephemeral nodes.
    pub fn in_memory() -> Self {
        Self {
            records: BTreeMap::new(),
            procedures: BTreeMap::new(),
            idempotency: BTreeMap::new(),
            snapshot: 0,
            next_id: 0,
            journal_path: None,
        }
    }

    /// Open (and replay) the durable state rooted at `state_dir`, creating it if absent.
    pub fn open(state_dir: &Path) -> Result<Self, StateError> {
        std::fs::create_dir_all(state_dir).map_err(|e| StateError::Persist(e.to_string()))?;
        let journal_path = state_dir.join("journal.jsonl");
        let mut state = Self::in_memory();
        if journal_path.exists() {
            let text = std::fs::read_to_string(&journal_path)
                .map_err(|e| StateError::Persist(e.to_string()))?;
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<Mutation>(line) {
                    Ok(mutation) => state.apply_replay(mutation),
                    Err(e) => {
                        tracing::warn!(error = %e, "daemon-metta: skipping corrupt journal entry");
                    }
                }
            }
        }
        state.journal_path = Some(journal_path);
        Ok(state)
    }

    /// The current snapshot id (the CAS token / count of applied mutations).
    pub fn snapshot(&self) -> u64 {
        self.snapshot
    }

    /// All records in `space` (for the engine to load into an atomspace, and for `match`/`explain`).
    pub fn records_in(&self, space: Space) -> impl Iterator<Item = &Record> {
        self.records.values().filter(move |r| r.space == space)
    }

    /// A record by id.
    pub fn record(&self, id: &str) -> Option<&Record> {
        self.records.get(id)
    }

    /// A procedure by id.
    pub fn procedure(&self, id: &str) -> Option<&Procedure> {
        self.procedures.get(id)
    }

    /// Per-space record counts (for `inspect`).
    pub fn space_counts(&self) -> BTreeMap<Space, usize> {
        let mut counts: BTreeMap<Space, usize> = BTreeMap::new();
        for space in Space::all() {
            counts.insert(space, 0);
        }
        for r in self.records.values() {
            *counts.entry(r.space).or_insert(0) += 1;
        }
        counts
    }

    /// Check the optimistic-concurrency token before a mutation.
    fn check_cas(&self, expected: Option<u64>) -> Result<(), StateError> {
        match expected {
            Some(e) if e != self.snapshot => Err(StateError::SnapshotMismatch {
                expected: e,
                current: self.snapshot,
            }),
            _ => Ok(()),
        }
    }

    /// Commit a mutation: append it to the journal (if durable), then apply it (bumping snapshot).
    fn commit(&mut self, mutation: Mutation) -> Result<(), StateError> {
        if let Some(path) = &self.journal_path {
            let line =
                serde_json::to_string(&mutation).map_err(|e| StateError::Persist(e.to_string()))?;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| StateError::Persist(e.to_string()))?;
            writeln!(file, "{line}").map_err(|e| StateError::Persist(e.to_string()))?;
            file.flush()
                .map_err(|e| StateError::Persist(e.to_string()))?;
        }
        self.apply(mutation);
        Ok(())
    }

    /// Apply a mutation to the in-memory state, bumping the snapshot. Shared by commit + replay.
    fn apply(&mut self, mutation: Mutation) {
        self.apply_inner(&mutation);
        self.snapshot += 1;
    }

    /// Replay a journal entry (same as [`Self::apply`], used at open time).
    fn apply_replay(&mut self, mutation: Mutation) {
        self.apply(mutation);
    }

    fn apply_inner(&mut self, mutation: &Mutation) {
        match mutation {
            Mutation::Assert(record) => {
                // Keep `next_id` ahead of any replayed `rec-N` id so live ids never collide.
                if let Some(n) = record
                    .id
                    .strip_prefix("rec-")
                    .and_then(|s| s.parse::<u64>().ok())
                {
                    self.next_id = self.next_id.max(n + 1);
                }
                self.records.insert(record.id.clone(), record.clone());
            }
            Mutation::Retract { id, hard } => {
                if *hard {
                    self.records.remove(id);
                } else if let Some(r) = self.records.get_mut(id) {
                    r.status = Status::Superseded;
                }
            }
            Mutation::DefineProcedure { id, version } => {
                let proc = self
                    .procedures
                    .entry(id.clone())
                    .or_insert_with(|| Procedure {
                        id: id.clone(),
                        versions: Vec::new(),
                        active_version: None,
                    });
                proc.versions.push(version.clone());
            }
            Mutation::Promote {
                id,
                version,
                evidence,
            } => {
                if let Some(proc) = self.procedures.get_mut(id) {
                    if let Some(v) = proc.versions.iter_mut().find(|v| v.version == *version) {
                        v.status = Status::Active;
                        v.evidence = evidence.clone();
                    }
                    // Demote any other active version to superseded.
                    for v in proc.versions.iter_mut() {
                        if v.version != *version && v.status == Status::Active {
                            v.status = Status::Superseded;
                        }
                    }
                    proc.active_version = Some(*version);
                }
            }
            Mutation::Rollback {
                id,
                target_version,
                retired_version,
            } => {
                if let Some(proc) = self.procedures.get_mut(id) {
                    for v in proc.versions.iter_mut() {
                        if v.version == *retired_version {
                            v.status = Status::Retired;
                        }
                        if v.version == *target_version {
                            v.status = Status::Active;
                        }
                    }
                    proc.active_version = Some(*target_version);
                }
            }
        }
    }

    /// The result of an assert: the committed ids and whether the call was an idempotent no-op.
    pub fn assert_atoms(
        &mut self,
        atoms: &[String],
        space: Space,
        provenance: &Provenance,
        idempotency_key: Option<&str>,
        expected_snapshot: Option<u64>,
    ) -> Result<(Vec<String>, bool), StateError> {
        if space == Space::Governance {
            return Err(StateError::Invalid(
                "governance space is read-only unless explicitly authorized".into(),
            ));
        }
        if let Some(key) = idempotency_key {
            if let Some(existing) = self.idempotency.get(key) {
                return Ok((existing.clone(), true));
            }
        }
        self.check_cas(expected_snapshot)?;

        let mut ids = Vec::with_capacity(atoms.len());
        for atom in atoms {
            let id = format!("rec-{:06}", self.next_id);
            self.next_id += 1;
            let record = Record {
                id: id.clone(),
                space,
                text: atom.clone(),
                provenance: provenance.clone(),
                status: Status::Active,
                created_at_snapshot: self.snapshot,
                supersedes: None,
            };
            self.commit(Mutation::Assert(record))?;
            ids.push(id);
        }
        if let Some(key) = idempotency_key {
            self.idempotency.insert(key.to_string(), ids.clone());
        }
        Ok((ids, false))
    }

    /// Retract by exact ids or matching ids. With `dry_run`, no mutation is committed — the would-be
    /// targets are returned. Supersession (soft) is the default; `hard = true` removes rows.
    pub fn retract(
        &mut self,
        targets: &[String],
        hard: bool,
        dry_run: bool,
        expected_snapshot: Option<u64>,
    ) -> Result<Vec<String>, StateError> {
        if dry_run {
            return Ok(targets
                .iter()
                .filter(|id| self.records.contains_key(*id))
                .cloned()
                .collect());
        }
        self.check_cas(expected_snapshot)?;
        let mut removed = Vec::new();
        for id in targets {
            if self.records.contains_key(id) {
                self.commit(Mutation::Retract {
                    id: id.clone(),
                    hard,
                })?;
                removed.push(id.clone());
            }
        }
        Ok(removed)
    }

    /// Resolve a [`RetractTarget`] to concrete ids the caller's `matcher` selected (for `Pattern`).
    pub fn resolve_retract_ids<F>(
        &self,
        target: &RetractTarget,
        space: Space,
        matcher: F,
    ) -> Vec<String>
    where
        F: Fn(&Record) -> bool,
    {
        match target {
            RetractTarget::Ids(ids) => ids.clone(),
            RetractTarget::Pattern(_) => self
                .records_in(space)
                .filter(|r| matcher(r))
                .map(|r| r.id.clone())
                .collect(),
        }
    }

    /// Define a new candidate version of a procedure (always `Candidate`; never active implicitly).
    pub fn define_procedure(
        &mut self,
        id: &str,
        program: &str,
        metadata: Option<&str>,
        tests: &[String],
    ) -> Result<u64, StateError> {
        let next_version = self
            .procedures
            .get(id)
            .and_then(|p| p.versions.iter().map(|v| v.version).max())
            .unwrap_or(0)
            + 1;
        let version = ProcedureVersion {
            version: next_version,
            program: program.to_string(),
            metadata: metadata.map(str::to_string),
            tests: tests.to_vec(),
            status: Status::Candidate,
            evidence: Vec::new(),
        };
        self.commit(Mutation::DefineProcedure {
            id: id.to_string(),
            version,
        })?;
        Ok(next_version)
    }

    /// Promote a candidate version to active after the gate passed. `expected_version` is a CAS on
    /// the version being promoted.
    pub fn promote(
        &mut self,
        id: &str,
        evidence: &[String],
        expected_version: Option<u64>,
    ) -> Result<u64, StateError> {
        let proc = self
            .procedures
            .get(id)
            .ok_or_else(|| StateError::NotFound(id.to_string()))?;
        // Promote the latest candidate (or the explicitly-expected version).
        let version = match expected_version {
            Some(v) => {
                let pv = proc
                    .versions
                    .iter()
                    .find(|pv| pv.version == v)
                    .ok_or_else(|| StateError::NotFound(format!("{id} v{v}")))?;
                if pv.status == Status::Active {
                    return Err(StateError::Invalid(format!("{id} v{v} already active")));
                }
                v
            }
            None => proc
                .versions
                .iter()
                .filter(|pv| pv.status == Status::Candidate)
                .map(|pv| pv.version)
                .max()
                .ok_or_else(|| StateError::Invalid(format!("{id}: no candidate to promote")))?,
        };
        self.commit(Mutation::Promote {
            id: id.to_string(),
            version,
            evidence: evidence.to_vec(),
        })?;
        Ok(version)
    }

    /// Roll back to the last-good active version (or an explicit `target_version`), retiring the
    /// currently-active one. Returns `(restored_version, retired_version)`.
    pub fn rollback(
        &mut self,
        id: &str,
        target_version: Option<u64>,
    ) -> Result<(u64, u64), StateError> {
        let proc = self
            .procedures
            .get(id)
            .ok_or_else(|| StateError::NotFound(id.to_string()))?;
        let retired = proc
            .active_version
            .ok_or_else(|| StateError::Invalid(format!("{id}: no active version to roll back")))?;
        let target = match target_version {
            Some(v) => v,
            None => proc
                .versions
                .iter()
                .filter(|pv| pv.version != retired && pv.version < retired)
                .map(|pv| pv.version)
                .max()
                .ok_or_else(|| StateError::Invalid(format!("{id}: no prior version to restore")))?,
        };
        if target == retired {
            return Err(StateError::Invalid(format!(
                "{id}: rollback target equals the active version"
            )));
        }
        self.commit(Mutation::Rollback {
            id: id.to_string(),
            target_version: target,
            retired_version: retired,
        })?;
        Ok((target, retired))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prov() -> Provenance {
        Provenance {
            source: Some("test".into()),
            ..Default::default()
        }
    }

    #[test]
    fn assert_is_idempotent_by_key() {
        let mut s = MettaState::in_memory();
        let (ids1, replay1) = s
            .assert_atoms(&["(a)".into()], Space::Semantic, &prov(), Some("k1"), None)
            .unwrap();
        assert!(!replay1);
        let snap_after_first = s.snapshot();
        let (ids2, replay2) = s
            .assert_atoms(&["(a)".into()], Space::Semantic, &prov(), Some("k1"), None)
            .unwrap();
        assert!(replay2, "second assert with same key is a no-op");
        assert_eq!(ids1, ids2);
        assert_eq!(
            s.snapshot(),
            snap_after_first,
            "no new snapshot on idempotent retry"
        );
    }

    #[test]
    fn snapshot_cas_rejects_stale_writes() {
        let mut s = MettaState::in_memory();
        s.assert_atoms(&["(a)".into()], Space::Semantic, &prov(), None, None)
            .unwrap();
        let current = s.snapshot();
        let err = s
            .assert_atoms(
                &["(b)".into()],
                Space::Semantic,
                &prov(),
                None,
                Some(current - 1),
            )
            .unwrap_err();
        assert!(matches!(err, StateError::SnapshotMismatch { .. }));
        // The correct snapshot succeeds.
        s.assert_atoms(
            &["(b)".into()],
            Space::Semantic,
            &prov(),
            None,
            Some(current),
        )
        .unwrap();
    }

    #[test]
    fn governance_is_read_only() {
        let mut s = MettaState::in_memory();
        let err = s
            .assert_atoms(&["(policy)".into()], Space::Governance, &prov(), None, None)
            .unwrap_err();
        assert!(matches!(err, StateError::Invalid(_)));
    }

    #[test]
    fn retract_dry_run_does_not_mutate() {
        let mut s = MettaState::in_memory();
        let (ids, _) = s
            .assert_atoms(&["(a)".into()], Space::Semantic, &prov(), None, None)
            .unwrap();
        let snap = s.snapshot();
        let preview = s.retract(&ids, false, true, None).unwrap();
        assert_eq!(preview, ids);
        assert_eq!(s.snapshot(), snap, "dry run does not bump the snapshot");
        assert_eq!(s.record(&ids[0]).unwrap().status, Status::Active);
        // A real soft retract supersedes.
        s.retract(&ids, false, false, None).unwrap();
        assert_eq!(s.record(&ids[0]).unwrap().status, Status::Superseded);
    }

    #[test]
    fn procedure_lifecycle_define_promote_rollback() {
        let mut s = MettaState::in_memory();
        let v1 = s
            .define_procedure("proc-1", "(= (f) 1)", None, &[])
            .unwrap();
        assert_eq!(v1, 1);
        assert_eq!(s.procedure("proc-1").unwrap().active_version, None);
        s.promote("proc-1", &["traj-1".into()], Some(1)).unwrap();
        assert_eq!(s.procedure("proc-1").unwrap().active_version, Some(1));
        // A second candidate, promoted, then rolled back to v1.
        let v2 = s
            .define_procedure("proc-1", "(= (f) 2)", None, &[])
            .unwrap();
        assert_eq!(v2, 2);
        s.promote("proc-1", &[], Some(2)).unwrap();
        assert_eq!(s.procedure("proc-1").unwrap().active_version, Some(2));
        let (restored, retired) = s.rollback("proc-1", None).unwrap();
        assert_eq!((restored, retired), (1, 2));
        let proc = s.procedure("proc-1").unwrap();
        assert_eq!(proc.active_version, Some(1));
        assert_eq!(
            proc.versions
                .iter()
                .find(|v| v.version == 2)
                .unwrap()
                .status,
            Status::Retired
        );
    }

    #[test]
    fn journal_replay_reconstructs_state() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut s = MettaState::open(dir.path()).unwrap();
            s.assert_atoms(
                &["(a)".into(), "(b)".into()],
                Space::Semantic,
                &prov(),
                None,
                None,
            )
            .unwrap();
            s.define_procedure("proc-1", "(= (f) 1)", None, &[])
                .unwrap();
            s.promote("proc-1", &["e".into()], Some(1)).unwrap();
        }
        // Reopen: the replay must reproduce the snapshot, records, and active procedure version.
        let s2 = MettaState::open(dir.path()).unwrap();
        assert_eq!(s2.records_in(Space::Semantic).count(), 2);
        assert_eq!(s2.procedure("proc-1").unwrap().active_version, Some(1));
        assert_eq!(s2.snapshot(), 4); // 2 asserts + 1 define + 1 promote
    }
}
