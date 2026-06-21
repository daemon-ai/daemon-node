//! The MeTTa coprocessor wire protocol — [`Command`]/[`Event`] frames + a CBOR codec.
//!
//! The daemon (`daemon-metta-client`'s `MettaCoprocessor`) and the `daemon-metta` worker exchange
//! these frames over a length-framed stdio cut ([`daemon_provision::CutChannel`],
//! [`Framing::Length`]) — exactly the transport the inference worker uses. Each frame body is CBOR;
//! the `u32`-length prefix is handled by the channel, so this module only owns the body
//! [`encode`]/[`decode`].
//!
//! These are standalone wire types: `serde` + `serde_json` + `ciborium` only, no `daemon-core` and
//! no `hyperon`. A consumer that needs only the protocol (the client + tool) depends on the crate
//! with `default-features = false`, so the engine is never dragged in.
//!
//! The op set mirrors the `metta-symbolic-coprocessor` SKILL contract: `inspect`,
//! `semantic_candidates`, `match`, `eval`, `assert`, `retract`, `define`, `test`, `explain`,
//! `promote`, `rollback`.
//!
//! [`Framing::Length`]: daemon_provision::Framing::Length

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// A logical memory space (SKILL §Spaces). A single worker hosts all of them; an atom always
/// carries its space so the separation is preserved even over one backing store.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Space {
    /// Disposable hypotheses, temporary bindings, query programs for the current task.
    Working,
    /// Immutable task events, tool observations, decisions, outcomes.
    Episodic,
    /// Durable facts, claims, preferences, entities, relations, constraints, artifact refs.
    Semantic,
    /// Draft and shadow procedures, rules, tests, metrics.
    ProceduralCandidates,
    /// Promoted procedures and their active versions.
    ProceduralActive,
    /// Schemas, capability policies, promotion thresholds, protected invariants (read-only).
    Governance,
}

impl Space {
    /// The canonical kebab-case name.
    pub fn as_str(self) -> &'static str {
        match self {
            Space::Working => "working",
            Space::Episodic => "episodic",
            Space::Semantic => "semantic",
            Space::ProceduralCandidates => "procedural-candidates",
            Space::ProceduralActive => "procedural-active",
            Space::Governance => "governance",
        }
    }

    /// Parse a space name (kebab or snake), `None` if unknown.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "working" => Some(Space::Working),
            "episodic" => Some(Space::Episodic),
            "semantic" => Some(Space::Semantic),
            "procedural-candidates" | "candidates" => Some(Space::ProceduralCandidates),
            "procedural-active" | "active" => Some(Space::ProceduralActive),
            "governance" => Some(Space::Governance),
            _ => None,
        }
    }

    /// Every space, in a stable order (used by `inspect`).
    pub fn all() -> [Space; 6] {
        [
            Space::Working,
            Space::Episodic,
            Space::Semantic,
            Space::ProceduralCandidates,
            Space::ProceduralActive,
            Space::Governance,
        ]
    }
}

impl Default for Space {
    fn default() -> Self {
        Space::Working
    }
}

/// A record / procedure lifecycle status (SKILL §Separate salience, confidence, and lifecycle).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    /// A draft procedure/rule — never active implicitly (the only status `define` may write).
    #[default]
    Candidate,
    /// A promoted, active record/procedure.
    Active,
    /// A record replaced by a newer one (kept for audit).
    Superseded,
    /// A retired procedure version (kept as evidence after a rollback).
    Retired,
}

/// Where a write came from / how a conclusion was derived (SKILL §Canonical memory records).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// The originating source (e.g. `user-message msg-184`, a tool id, a rule id).
    #[serde(default)]
    pub source: Option<String>,
    /// ISO-8601 UTC capture time (the worker stamps this when absent).
    #[serde(default)]
    pub recorded_at: Option<String>,
    /// Premise / rule ids a derived conclusion was produced from.
    #[serde(default)]
    pub derived_from: Vec<String>,
    /// A short, audit-only note (never private chain-of-thought).
    #[serde(default)]
    pub note: Option<String>,
}

/// Evaluation/test resource bounds (SKILL §eval/§test).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bounds {
    /// Max interpreter steps before truncation (`0` = the worker default).
    pub max_steps: u64,
    /// Wall-clock deadline in milliseconds (`0` = the worker default).
    pub timeout_ms: u64,
    /// Max results returned before truncation (`0` = the worker default).
    pub max_results: u64,
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            max_steps: 1_000,
            timeout_ms: 1_000,
            max_results: 100,
        }
    }
}

/// How much budget an op consumed (returned in every [`OpResponse`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetUsed {
    /// Interpreter steps actually executed.
    pub steps: u64,
    /// Wall-clock milliseconds elapsed.
    pub elapsed_ms: u64,
}

/// The target of a [`Command::Retract`]: exact ids (preferred for destructive change) or a pattern.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetractTarget {
    /// Retract these exact record ids.
    Ids(Vec<String>),
    /// Retract every record matching this MeTTa pattern.
    Pattern(String),
}

/// A `parent -> worker` command frame. Every op carries a `request_id` correlating its reply.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Command {
    /// Read-only discovery of spaces, schema, limits, capabilities, snapshot ids.
    Inspect { request_id: u64 },
    /// Embedding/lexical candidate discovery. The worker has no embeddings; it returns a clear
    /// "served by host" marker so the client delegates to the daemon's memory/embedding provider.
    SemanticCandidates {
        request_id: u64,
        query: String,
        #[serde(default)]
        space: Space,
        #[serde(default)]
        k: u32,
        #[serde(default)]
        filters: Vec<String>,
    },
    /// Read-only structural matching; returns bindings + record ids.
    Match {
        request_id: u64,
        pattern: String,
        #[serde(default)]
        space: Space,
        #[serde(default)]
        limit: u64,
        #[serde(default)]
        cursor: Option<String>,
    },
    /// Bounded evaluation / rewriting; pure by default (`allow_grounded = false`).
    Eval {
        request_id: u64,
        expression: String,
        #[serde(default)]
        space: Space,
        #[serde(default)]
        bounds: Bounds,
        #[serde(default)]
        allow_grounded: bool,
    },
    /// Atomic, versioned insertion of data atoms (the `=`-rule-free path).
    Assert {
        request_id: u64,
        atoms: Vec<String>,
        #[serde(default)]
        space: Space,
        #[serde(default)]
        provenance: Provenance,
        #[serde(default)]
        idempotency_key: Option<String>,
        #[serde(default)]
        expected_snapshot: Option<u64>,
    },
    /// Preview-or-commit removal. Prefer supersession; use exact ids for destructive change.
    Retract {
        request_id: u64,
        target: RetractTarget,
        #[serde(default)]
        space: Space,
        #[serde(default)]
        dry_run: bool,
        #[serde(default)]
        expected_snapshot: Option<u64>,
    },
    /// Create a versioned rule/procedure in the candidate space. Never active implicitly.
    Define {
        request_id: u64,
        program: String,
        #[serde(default)]
        metadata: Option<String>,
        #[serde(default)]
        tests: Vec<String>,
        #[serde(default)]
        status: Status,
    },
    /// Run deterministic/edge/negative/safety/termination tests in a sandbox (no side effects).
    Test {
        request_id: u64,
        program_or_id: String,
        #[serde(default)]
        cases: Vec<TestCase>,
        #[serde(default)]
        bounds: Bounds,
    },
    /// Return the rules, premises, source records, and rewrite path for a result/query.
    Explain {
        request_id: u64,
        target: String,
        #[serde(default)]
        max_depth: u32,
    },
    /// Move a tested candidate to active status after the promotion gate passes.
    Promote {
        request_id: u64,
        candidate_id: String,
        #[serde(default)]
        evidence: Vec<String>,
        #[serde(default)]
        expected_version: Option<u64>,
    },
    /// Restore the last-good active version, retaining the failed version + evidence.
    Rollback {
        request_id: u64,
        procedure_id: String,
        #[serde(default)]
        target_version: Option<u64>,
        #[serde(default)]
        reason: Option<String>,
    },
    /// Liveness probe (answered with [`Event::Pong`]).
    Ping,
    /// Ask the worker to exit cleanly.
    Shutdown,
}

impl Command {
    /// The request id this command correlates on (`None` for `Ping`/`Shutdown`).
    pub fn request_id(&self) -> Option<u64> {
        match self {
            Command::Inspect { request_id }
            | Command::SemanticCandidates { request_id, .. }
            | Command::Match { request_id, .. }
            | Command::Eval { request_id, .. }
            | Command::Assert { request_id, .. }
            | Command::Retract { request_id, .. }
            | Command::Define { request_id, .. }
            | Command::Test { request_id, .. }
            | Command::Explain { request_id, .. }
            | Command::Promote { request_id, .. }
            | Command::Rollback { request_id, .. } => Some(*request_id),
            Command::Ping | Command::Shutdown => None,
        }
    }

    /// Overwrite the correlating request id (the supervised client assigns ids centrally; `Ping`
    /// and `Shutdown` carry none, so this is a no-op for them).
    pub fn set_request_id(&mut self, id: u64) {
        match self {
            Command::Inspect { request_id }
            | Command::SemanticCandidates { request_id, .. }
            | Command::Match { request_id, .. }
            | Command::Eval { request_id, .. }
            | Command::Assert { request_id, .. }
            | Command::Retract { request_id, .. }
            | Command::Define { request_id, .. }
            | Command::Test { request_id, .. }
            | Command::Explain { request_id, .. }
            | Command::Promote { request_id, .. }
            | Command::Rollback { request_id, .. } => *request_id = id,
            Command::Ping | Command::Shutdown => {}
        }
    }
}

/// One deterministic test case (SKILL §Test before trust): an input expression and the substrings
/// the result must (or must not) contain.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestCase {
    /// A human label (`nominal`, `edge`, `negative`, `safety`, `termination`).
    #[serde(default)]
    pub name: String,
    /// The expression to evaluate.
    pub input: String,
    /// Substrings the result must contain (all of them).
    #[serde(default)]
    pub expect_contains: Vec<String>,
    /// Substrings the result must NOT contain (none of them).
    #[serde(default)]
    pub expect_absent: Vec<String>,
}

/// The result of a single test case.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestOutcome {
    /// The case label.
    pub name: String,
    /// Whether the case passed.
    pub passed: bool,
    /// The rendered result the assertions were checked against.
    pub detail: String,
}

/// A classified worker failure (mirrors the inference worker's taxonomy so the client maps it onto
/// the daemon `Failure` taxonomy for supervision/recovery).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorClass {
    /// A bad request (unparseable atom, unknown id, CAS mismatch) — the caller must fix it.
    BadRequest,
    /// The engine is not compiled / the op needs the `hyperon` feature.
    Unsupported,
    /// A transient/internal evaluation error — retry may help.
    Transient,
    /// Unrecoverable: internal bug, corrupt state — abort.
    Fatal,
    /// The op was cancelled / timed out cooperatively.
    Cancelled,
}

/// The generic op response carried by [`Event::Reply`]. Every op fills the shared envelope (SKILL:
/// always check `ok`, `results`, `truncated`, `warnings`, `budget_used`, `snapshot`, `provenance`);
/// op-specific payloads ride `results` (binding strings, candidate ids, explanation lines, ...).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OpResponse {
    /// The request this answers.
    pub request_id: u64,
    /// Whether the op succeeded. An empty `results` with `ok = true` means "not found", not "false".
    pub ok: bool,
    /// The op payload: bindings, ids, candidate ids, explanation lines, eval results, ...
    #[serde(default)]
    pub results: Vec<String>,
    /// Whether the results were truncated by a bound (not exhaustive).
    #[serde(default)]
    pub truncated: bool,
    /// Non-fatal warnings (e.g. "served by host", "dry-run", "stale snapshot").
    #[serde(default)]
    pub warnings: Vec<String>,
    /// How much budget the op consumed.
    #[serde(default)]
    pub budget_used: BudgetUsed,
    /// The store snapshot after the op (monotonic; mutations bump it).
    #[serde(default)]
    pub snapshot: u64,
    /// The ids committed by a mutating op (assert/define/promote/...).
    #[serde(default)]
    pub committed_ids: Vec<String>,
    /// Provenance for the records the op touched/produced.
    #[serde(default)]
    pub provenance: Vec<Provenance>,
    /// Per-case test outcomes (only for `test`).
    #[serde(default)]
    pub tests: Vec<TestOutcome>,
}

impl OpResponse {
    /// A successful response carrying `results`.
    pub fn ok(request_id: u64, results: Vec<String>) -> Self {
        Self {
            request_id,
            ok: true,
            results,
            ..Default::default()
        }
    }
}

/// A `worker -> parent` event frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Event {
    /// The worker is up; reports the compiled engine and the spaces it hosts.
    Ready {
        /// The compiled engine identifier (`"hyperon"` or `"fallback"`).
        engine: String,
        /// The spaces available.
        spaces: Vec<String>,
    },
    /// An op reply.
    Reply(OpResponse),
    /// A classified failure for `request_id` (or worker-level when `None`).
    Error {
        request_id: Option<u64>,
        class: ErrorClass,
        message: String,
    },
    /// Liveness reply to [`Command::Ping`].
    Pong,
}

/// A CBOR codec error.
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    /// Encoding a frame to CBOR failed.
    #[error("cbor encode: {0}")]
    Encode(String),
    /// Decoding a frame from CBOR failed.
    #[error("cbor decode: {0}")]
    Decode(String),
}

/// Encode a frame body to CBOR bytes (the [`CutChannel`](daemon_provision::CutChannel) adds the
/// length prefix).
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, CodecError> {
    let mut buf = Vec::new();
    ciborium::into_writer(value, &mut buf).map_err(|e| CodecError::Encode(e.to_string()))?;
    Ok(buf)
}

/// Decode a CBOR frame body.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, CodecError> {
    ciborium::from_reader(bytes).map_err(|e| CodecError::Decode(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip_command(cmd: Command) {
        let bytes = encode(&cmd).expect("encode command");
        let back: Command = decode(&bytes).expect("decode command");
        assert_eq!(cmd, back);
    }

    fn round_trip_event(ev: Event) {
        let bytes = encode(&ev).expect("encode event");
        let back: Event = decode(&bytes).expect("decode event");
        assert_eq!(ev, back);
    }

    #[test]
    fn commands_round_trip() {
        round_trip_command(Command::Inspect { request_id: 1 });
        round_trip_command(Command::SemanticCandidates {
            request_id: 2,
            query: "release notes".into(),
            space: Space::Semantic,
            k: 8,
            filters: vec!["kind:preference".into()],
        });
        round_trip_command(Command::Match {
            request_id: 3,
            pattern: "(owns $p artifact-42)".into(),
            space: Space::Semantic,
            limit: 20,
            cursor: None,
        });
        round_trip_command(Command::Eval {
            request_id: 4,
            expression: "(+ 1 2)".into(),
            space: Space::Working,
            bounds: Bounds::default(),
            allow_grounded: false,
        });
        round_trip_command(Command::Assert {
            request_id: 5,
            atoms: vec!["(prefers user direct)".into()],
            space: Space::Semantic,
            provenance: Provenance {
                source: Some("user-message msg-1".into()),
                ..Default::default()
            },
            idempotency_key: Some("idem-1".into()),
            expected_snapshot: Some(0),
        });
        round_trip_command(Command::Retract {
            request_id: 6,
            target: RetractTarget::Ids(vec!["mem-1".into()]),
            space: Space::Semantic,
            dry_run: true,
            expected_snapshot: None,
        });
        round_trip_command(Command::Define {
            request_id: 7,
            program: "(= (double $x) (* 2 $x))".into(),
            metadata: Some("{\"intent\":\"double\"}".into()),
            tests: vec!["(double 2)".into()],
            status: Status::Candidate,
        });
        round_trip_command(Command::Test {
            request_id: 8,
            program_or_id: "proc-1".into(),
            cases: vec![TestCase {
                name: "nominal".into(),
                input: "(double 2)".into(),
                expect_contains: vec!["4".into()],
                expect_absent: vec![],
            }],
            bounds: Bounds::default(),
        });
        round_trip_command(Command::Explain {
            request_id: 9,
            target: "mem-1".into(),
            max_depth: 3,
        });
        round_trip_command(Command::Promote {
            request_id: 10,
            candidate_id: "proc-1".into(),
            evidence: vec!["traj-1".into()],
            expected_version: Some(1),
        });
        round_trip_command(Command::Rollback {
            request_id: 11,
            procedure_id: "proc-1".into(),
            target_version: Some(1),
            reason: Some("regression".into()),
        });
        round_trip_command(Command::Ping);
        round_trip_command(Command::Shutdown);
    }

    #[test]
    fn events_round_trip() {
        round_trip_event(Event::Ready {
            engine: "fallback".into(),
            spaces: Space::all().iter().map(|s| s.as_str().to_string()).collect(),
        });
        round_trip_event(Event::Reply(OpResponse::ok(1, vec!["3".into()])));
        round_trip_event(Event::Error {
            request_id: Some(1),
            class: ErrorClass::BadRequest,
            message: "bad atom".into(),
        });
        round_trip_event(Event::Pong);
    }

    #[test]
    fn space_parse_roundtrips() {
        for s in Space::all() {
            assert_eq!(Space::parse(s.as_str()), Some(s));
        }
        assert_eq!(Space::parse("candidates"), Some(Space::ProceduralCandidates));
        assert_eq!(Space::parse("nope"), None);
    }
}
