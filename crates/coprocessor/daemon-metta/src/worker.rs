// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The coprocessor worker core: maps one [`Command`] to one [`Event`] against the [`MettaState`]
//! and the [`MettaEngine`].
//!
//! This is the synchronous heart of the worker. It is driven by the actor thread in
//! [`crate::bin`](../bin/daemon-metta.rs) — the part that owns the (`!Send`) engine — while the
//! async stdio loop bridges frames to it over channels. Keeping it a plain `&mut self` function
//! makes the whole op surface unit-testable without any process/thread machinery.

use std::time::Instant;

use crate::engine::{default_engine, structural_match, EngineError, MettaEngine};
use crate::protocol::{
    BudgetUsed, Command, ErrorClass, Event, OpResponse, Provenance, RetractTarget, Space, Status,
    TestOutcome,
};
use crate::state::{MettaState, StateError};

/// The worker: the durable state plus the (possibly `!Send`) evaluation engine.
pub struct Worker {
    state: MettaState,
    engine: Box<dyn MettaEngine>,
}

impl Worker {
    /// A worker over `state` using the compilation's default engine (hyperon or fallback).
    pub fn new(state: MettaState) -> Self {
        Self {
            state,
            engine: default_engine(),
        }
    }

    /// A worker with an explicit engine (tests).
    pub fn with_engine(state: MettaState, engine: Box<dyn MettaEngine>) -> Self {
        Self { state, engine }
    }

    /// The engine identifier (for [`Event::Ready`]).
    pub fn engine_name(&self) -> &'static str {
        self.engine.name()
    }

    /// Handle one command, producing its reply event. `Shutdown` returns `None` (the loop exits);
    /// `Ping` returns `Pong`.
    pub fn handle(&mut self, cmd: Command) -> Option<Event> {
        match cmd {
            Command::Ping => Some(Event::Pong),
            Command::Shutdown => None,
            Command::Inspect { request_id } => Some(self.inspect(request_id)),
            Command::SemanticCandidates { request_id, .. } => {
                // The worker has no embeddings; the daemon serves candidates and the agent then
                // hydrates the ids via `match` (SKILL §Separate the semantic index from memory).
                let mut resp = OpResponse::ok(request_id, Vec::new());
                resp.snapshot = self.state.snapshot();
                resp.warnings.push(
                    "semantic_candidates is served by the host embedding/memory provider".into(),
                );
                Some(Event::Reply(resp))
            }
            Command::Match {
                request_id,
                pattern,
                space,
                limit,
                ..
            } => Some(self.do_match(request_id, &pattern, space, limit as usize)),
            Command::Eval {
                request_id,
                expression,
                space,
                bounds,
                allow_grounded,
            } => Some(self.eval(request_id, &expression, space, bounds, allow_grounded)),
            Command::Assert {
                request_id,
                atoms,
                space,
                provenance,
                idempotency_key,
                expected_snapshot,
            } => Some(self.assert(
                request_id,
                &atoms,
                space,
                provenance,
                idempotency_key.as_deref(),
                expected_snapshot,
            )),
            Command::Retract {
                request_id,
                target,
                space,
                dry_run,
                expected_snapshot,
            } => Some(self.retract(request_id, target, space, dry_run, expected_snapshot)),
            Command::Define {
                request_id,
                program,
                metadata,
                tests,
                status,
            } => Some(self.define(request_id, &program, metadata.as_deref(), &tests, status)),
            Command::Test {
                request_id,
                program_or_id,
                cases,
                bounds,
            } => Some(self.test(request_id, &program_or_id, &cases, bounds)),
            Command::Explain {
                request_id, target, ..
            } => Some(self.explain(request_id, &target)),
            Command::Promote {
                request_id,
                candidate_id,
                evidence,
                expected_version,
            } => Some(self.promote(request_id, &candidate_id, &evidence, expected_version)),
            Command::Rollback {
                request_id,
                procedure_id,
                target_version,
                ..
            } => Some(self.rollback(request_id, &procedure_id, target_version)),
        }
    }

    fn inspect(&self, request_id: u64) -> Event {
        let mut results = Vec::new();
        results.push(format!("engine: {}", self.engine.name()));
        results.push(format!("snapshot: {}", self.state.snapshot()));
        for (space, count) in self.state.space_counts() {
            results.push(format!("space {}: {} records", space.as_str(), count));
        }
        results.push(
            "ops: inspect, semantic_candidates, match, eval, assert, retract, define, test, \
             explain, promote, rollback"
                .into(),
        );
        let mut resp = OpResponse::ok(request_id, results);
        resp.snapshot = self.state.snapshot();
        Event::Reply(resp)
    }

    fn do_match(&mut self, request_id: u64, pattern: &str, space: Space, limit: usize) -> Event {
        let records: Vec<&_> = self.state.records_in(space).collect();
        let start = Instant::now();
        match structural_match(pattern, &records, limit) {
            Ok(m) => {
                let results = m
                    .record_ids
                    .iter()
                    .zip(&m.bindings)
                    .map(|(id, b)| format!("{id} {b}"))
                    .collect();
                let provenance = m
                    .record_ids
                    .iter()
                    .filter_map(|id| self.state.record(id))
                    .map(|r| r.provenance.clone())
                    .collect();
                let mut resp = OpResponse::ok(request_id, results);
                resp.truncated = m.truncated;
                resp.snapshot = self.state.snapshot();
                resp.provenance = provenance;
                resp.budget_used = BudgetUsed {
                    steps: records.len() as u64,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                };
                Event::Reply(resp)
            }
            Err(e) => engine_error(request_id, e),
        }
    }

    fn eval(
        &mut self,
        request_id: u64,
        expression: &str,
        space: Space,
        bounds: crate::protocol::Bounds,
        allow_grounded: bool,
    ) -> Event {
        let records: Vec<&_> = self.state.records_in(space).collect();
        let start = Instant::now();
        match self
            .engine
            .eval(expression, &records, bounds, allow_grounded)
        {
            Ok(r) => {
                let mut resp = OpResponse::ok(request_id, r.results);
                resp.truncated = r.truncated;
                resp.snapshot = self.state.snapshot();
                resp.budget_used = BudgetUsed {
                    steps: r.steps,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                };
                Event::Reply(resp)
            }
            Err(e) => engine_error(request_id, e),
        }
    }

    fn assert(
        &mut self,
        request_id: u64,
        atoms: &[String],
        space: Space,
        provenance: Provenance,
        idempotency_key: Option<&str>,
        expected_snapshot: Option<u64>,
    ) -> Event {
        match self.state.assert_atoms(
            atoms,
            space,
            &provenance,
            idempotency_key,
            expected_snapshot,
        ) {
            Ok((ids, replayed)) => {
                let mut resp = OpResponse::ok(request_id, ids.clone());
                resp.committed_ids = ids;
                resp.snapshot = self.state.snapshot();
                resp.provenance = vec![provenance];
                if replayed {
                    resp.warnings.push("idempotent replay: no new write".into());
                }
                Event::Reply(resp)
            }
            Err(e) => state_error(request_id, e),
        }
    }

    fn retract(
        &mut self,
        request_id: u64,
        target: RetractTarget,
        space: Space,
        dry_run: bool,
        expected_snapshot: Option<u64>,
    ) -> Event {
        // Resolve a pattern target to ids via structural match (read-only) before mutating.
        let ids = match &target {
            RetractTarget::Ids(ids) => ids.clone(),
            RetractTarget::Pattern(pattern) => {
                let records: Vec<&_> = self.state.records_in(space).collect();
                match structural_match(pattern, &records, 0) {
                    Ok(m) => m.record_ids,
                    Err(e) => return engine_error(request_id, e),
                }
            }
        };
        // Pattern retraction is soft (supersession); explicit-id retraction is destructive (SKILL).
        let hard = matches!(target, RetractTarget::Ids(_));
        match self.state.retract(&ids, hard, dry_run, expected_snapshot) {
            Ok(affected) => {
                let mut resp = OpResponse::ok(request_id, affected.clone());
                resp.committed_ids = if dry_run { Vec::new() } else { affected };
                resp.snapshot = self.state.snapshot();
                if dry_run {
                    resp.warnings.push("dry-run: no records removed".into());
                }
                Event::Reply(resp)
            }
            Err(e) => state_error(request_id, e),
        }
    }

    fn define(
        &mut self,
        request_id: u64,
        program: &str,
        metadata: Option<&str>,
        tests: &[String],
        status: Status,
    ) -> Event {
        if status != Status::Candidate {
            return Event::Error {
                request_id: Some(request_id),
                class: ErrorClass::BadRequest,
                message: "define may only write status=candidate (promotion is a separate op)"
                    .into(),
            };
        }
        // An explicit id may be supplied in metadata `{"id": "..."}`; else generate one.
        let id = metadata
            .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
            .and_then(|v| v.get("id").and_then(|i| i.as_str().map(str::to_string)))
            .unwrap_or_else(|| format!("proc-{:06}", self.state.snapshot()));
        match self.state.define_procedure(&id, program, metadata, tests) {
            Ok(version) => {
                let mut resp =
                    OpResponse::ok(request_id, vec![format!("{id} v{version} (candidate)")]);
                resp.committed_ids = vec![id];
                resp.snapshot = self.state.snapshot();
                Event::Reply(resp)
            }
            Err(e) => state_error(request_id, e),
        }
    }

    fn test(
        &mut self,
        request_id: u64,
        program_or_id: &str,
        cases: &[crate::protocol::TestCase],
        bounds: crate::protocol::Bounds,
    ) -> Event {
        // Resolve a procedure id to its latest program; otherwise treat the arg as program source.
        let program = match self.state.procedure(program_or_id) {
            Some(proc) => match proc.latest() {
                Some(v) => v.program.clone(),
                None => {
                    return Event::Error {
                        request_id: Some(request_id),
                        class: ErrorClass::BadRequest,
                        message: format!("procedure {program_or_id} has no versions"),
                    }
                }
            },
            None => program_or_id.to_string(),
        };
        // The program is loaded as the sole space content for the sandboxed evaluation.
        let program_record = crate::state::Record {
            id: "__test_program__".into(),
            space: Space::Working,
            text: program,
            provenance: Provenance::default(),
            status: Status::Candidate,
            created_at_snapshot: 0,
            supersedes: None,
        };
        let records = [&program_record];
        let mut outcomes = Vec::new();
        let mut all_passed = true;
        for case in cases {
            let outcome = match self.engine.eval(&case.input, &records, bounds, false) {
                Ok(r) => {
                    let detail = r.results.join(" | ");
                    let contains_ok = case.expect_contains.iter().all(|s| detail.contains(s));
                    let absent_ok = case.expect_absent.iter().all(|s| !detail.contains(s));
                    TestOutcome {
                        name: case.name.clone(),
                        passed: contains_ok && absent_ok,
                        detail,
                    }
                }
                Err(e) => TestOutcome {
                    name: case.name.clone(),
                    passed: false,
                    detail: format!("eval error: {}", describe_engine_error(&e)),
                },
            };
            all_passed &= outcome.passed;
            outcomes.push(outcome);
        }
        let mut resp = OpResponse::ok(request_id, Vec::new());
        resp.ok = all_passed;
        resp.tests = outcomes;
        resp.snapshot = self.state.snapshot();
        Event::Reply(resp)
    }

    fn explain(&mut self, request_id: u64, target: &str) -> Event {
        let mut results = Vec::new();
        if let Some(record) = self.state.record(target) {
            results.push(format!("record {}: {}", record.id, record.text));
            results.push(format!("status: {:?}", record.status));
            if let Some(src) = &record.provenance.source {
                results.push(format!("source: {src}"));
            }
            if !record.provenance.derived_from.is_empty() {
                results.push(format!(
                    "derived-from: {}",
                    record.provenance.derived_from.join(", ")
                ));
            }
        } else if let Some(proc) = self.state.procedure(target) {
            results.push(format!(
                "procedure {}: active={:?}",
                proc.id, proc.active_version
            ));
            for v in &proc.versions {
                results.push(format!("v{} [{:?}]: {}", v.version, v.status, v.program));
            }
        } else {
            // SKILL: never invent a proof when the tool cannot provide one.
            let mut resp = OpResponse::ok(request_id, Vec::new());
            resp.warnings
                .push(format!("no record or procedure '{target}' to explain"));
            resp.snapshot = self.state.snapshot();
            return Event::Reply(resp);
        }
        let mut resp = OpResponse::ok(request_id, results);
        resp.snapshot = self.state.snapshot();
        Event::Reply(resp)
    }

    fn promote(
        &mut self,
        request_id: u64,
        candidate_id: &str,
        evidence: &[String],
        expected_version: Option<u64>,
    ) -> Event {
        match self.state.promote(candidate_id, evidence, expected_version) {
            Ok(version) => {
                let mut resp = OpResponse::ok(
                    request_id,
                    vec![format!("{candidate_id} v{version} active")],
                );
                resp.committed_ids = vec![candidate_id.to_string()];
                resp.snapshot = self.state.snapshot();
                Event::Reply(resp)
            }
            Err(e) => state_error(request_id, e),
        }
    }

    fn rollback(
        &mut self,
        request_id: u64,
        procedure_id: &str,
        target_version: Option<u64>,
    ) -> Event {
        match self.state.rollback(procedure_id, target_version) {
            Ok((restored, retired)) => {
                let mut resp = OpResponse::ok(
                    request_id,
                    vec![format!(
                        "{procedure_id}: restored v{restored}, retired v{retired}"
                    )],
                );
                resp.committed_ids = vec![procedure_id.to_string()];
                resp.snapshot = self.state.snapshot();
                Event::Reply(resp)
            }
            Err(e) => state_error(request_id, e),
        }
    }
}

fn describe_engine_error(e: &EngineError) -> String {
    match e {
        EngineError::Parse(m) => format!("parse: {m}"),
        EngineError::Unsupported(m) => format!("unsupported: {m}"),
        EngineError::Internal(m) => format!("internal: {m}"),
    }
}

fn engine_error(request_id: u64, e: EngineError) -> Event {
    let class = match e {
        EngineError::Parse(_) => ErrorClass::BadRequest,
        EngineError::Unsupported(_) => ErrorClass::Unsupported,
        EngineError::Internal(_) => ErrorClass::Transient,
    };
    Event::Error {
        request_id: Some(request_id),
        class,
        message: describe_engine_error(&e),
    }
}

fn state_error(request_id: u64, e: StateError) -> Event {
    let class = match e {
        StateError::SnapshotMismatch { .. } | StateError::NotFound(_) | StateError::Invalid(_) => {
            ErrorClass::BadRequest
        }
        StateError::Persist(_) => ErrorClass::Fatal,
    };
    Event::Error {
        request_id: Some(request_id),
        class,
        message: e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Bounds;

    fn worker() -> Worker {
        Worker::new(MettaState::in_memory())
    }

    fn reply(ev: Option<Event>) -> OpResponse {
        match ev {
            Some(Event::Reply(r)) => r,
            other => panic!("expected reply, got {other:?}"),
        }
    }

    #[test]
    fn assert_then_match_roundtrip() {
        let mut w = worker();
        let assert = reply(w.handle(Command::Assert {
            request_id: 1,
            atoms: vec!["(owns alice artifact-42)".into()],
            space: Space::Semantic,
            provenance: Provenance::default(),
            idempotency_key: None,
            expected_snapshot: None,
        }));
        assert!(assert.ok);
        assert_eq!(assert.committed_ids.len(), 1);

        let m = reply(w.handle(Command::Match {
            request_id: 2,
            pattern: "(owns $p artifact-42)".into(),
            space: Space::Semantic,
            limit: 0,
            cursor: None,
        }));
        assert!(m.ok);
        assert_eq!(m.results.len(), 1);
        assert!(m.results[0].contains("$p = alice"));
    }

    #[test]
    fn eval_arithmetic_via_fallback() {
        let mut w = worker();
        let r = reply(w.handle(Command::Eval {
            request_id: 1,
            expression: "(+ 2 3)".into(),
            space: Space::Working,
            bounds: Bounds::default(),
            allow_grounded: false,
        }));
        assert_eq!(r.results, vec!["5".to_string()]);
    }

    #[test]
    fn ping_pong_and_shutdown() {
        let mut w = worker();
        assert_eq!(w.handle(Command::Ping), Some(Event::Pong));
        assert_eq!(w.handle(Command::Shutdown), None);
    }

    #[test]
    fn inspect_reports_engine_and_spaces() {
        let mut w = worker();
        let r = reply(w.handle(Command::Inspect { request_id: 1 }));
        assert!(r.results.iter().any(|l| l.starts_with("engine:")));
        assert!(r.results.iter().any(|l| l.contains("space semantic")));
    }
}
