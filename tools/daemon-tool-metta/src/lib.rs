//! `daemon-tool-metta` — the single `metta` symbolic-coprocessor tool (a `daemon_core::Tool`).
//!
//! One tool, an `op` field, the full `metta-symbolic-coprocessor` SKILL contract. `run()` parses the
//! op + its arguments, dispatches to the supervised [`MettaCoprocessor`] (which owns the worker
//! lifecycle), and returns a [`ToolOutcome`] carrying a human summary plus a structured
//! [`ToolDetail`] (`kind = "metta"`) of the raw [`OpResponse`] for rich GUI rendering.
//!
//! The 11 ops are `inspect`, `semantic_candidates`, `match`, `eval`, `assert`, `retract`, `define`,
//! `test`, `explain`, `promote`, `rollback`. `semantic_candidates` is the one op the worker cannot
//! answer (it has no embeddings): the tool delegates it to an optional host [`SemanticIndex`] (the
//! daemon's memory/embedding provider) and the agent then hydrates the returned ids via `match` —
//! the SKILL "candidates, never authoritative" rule.

#![forbid(unsafe_code)]

use std::sync::Arc;

use async_trait::async_trait;
use daemon_core::provider::{GrammarConstraint, Request, RequestMsg};
use daemon_core::{Tool, ToolCall, ToolOutcome, TurnCx};
use daemon_infer::grammar::{METTA_GBNF, METTA_LARK};
use daemon_metta::protocol::{
    Bounds, Command as MettaCommand, Provenance, RetractTarget, Space, Status, TestCase,
};
use daemon_metta_client::{MettaCoprocessor, MettaError};
use daemon_protocol::ToolDetail;
use serde::Deserialize;

/// A host-side candidate index (embeddings / lexical recall). The daemon's memory provider
/// implements this so `op=semantic_candidates` can return candidate record ids that the agent then
/// hydrates via `op=match`. Optional: when unset, the tool reports "no host index configured".
#[async_trait]
pub trait SemanticIndex: Send + Sync {
    /// Return up to `k` candidate record ids (ranked) for `query`, optionally narrowed by `filters`.
    async fn candidates(&self, query: &str, k: u32, filters: &[String]) -> Vec<String>;
}

/// The `metta` tool.
pub struct MettaTool {
    copro: Arc<MettaCoprocessor>,
    index: Option<Arc<dyn SemanticIndex>>,
    default_bounds: Bounds,
}

impl MettaTool {
    /// A tool over `copro` with no host semantic index (`semantic_candidates` returns a clear note)
    /// and the protocol default bounds.
    pub fn new(copro: Arc<MettaCoprocessor>) -> Self {
        Self {
            copro,
            index: None,
            default_bounds: Bounds::default(),
        }
    }

    /// Attach a host semantic index for `semantic_candidates` delegation.
    pub fn with_index(mut self, index: Arc<dyn SemanticIndex>) -> Self {
        self.index = Some(index);
        self
    }

    /// Set the default eval/test bounds applied when a call omits (zeroes) them.
    pub fn with_default_bounds(mut self, default_bounds: Bounds) -> Self {
        self.default_bounds = default_bounds;
        self
    }
}

/// The tool's argument envelope: a tagged `op` plus that op's fields. Mirrors the protocol ops and
/// is the contract the JSON-Schema advertises.
#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum MettaArgs {
    Inspect,
    SemanticCandidates {
        query: String,
        #[serde(default)]
        space: Option<String>,
        #[serde(default)]
        k: u32,
        #[serde(default)]
        filters: Vec<String>,
    },
    Match {
        pattern: String,
        #[serde(default)]
        space: Option<String>,
        #[serde(default)]
        limit: u64,
        #[serde(default)]
        cursor: Option<String>,
    },
    Eval {
        expression: String,
        #[serde(default)]
        space: Option<String>,
        #[serde(default)]
        max_steps: u64,
        #[serde(default)]
        timeout_ms: u64,
        #[serde(default)]
        max_results: u64,
        #[serde(default)]
        allow_grounded: bool,
    },
    Assert {
        atoms: Vec<String>,
        #[serde(default)]
        space: Option<String>,
        #[serde(default)]
        provenance: Provenance,
        #[serde(default)]
        idempotency_key: Option<String>,
        #[serde(default)]
        expected_snapshot: Option<u64>,
    },
    Retract {
        #[serde(default)]
        ids: Vec<String>,
        #[serde(default)]
        pattern: Option<String>,
        #[serde(default)]
        space: Option<String>,
        #[serde(default)]
        dry_run: bool,
        #[serde(default)]
        expected_snapshot: Option<u64>,
    },
    Define {
        program: String,
        #[serde(default)]
        metadata: Option<String>,
        #[serde(default)]
        tests: Vec<String>,
    },
    Test {
        program_or_id: String,
        #[serde(default)]
        cases: Vec<TestCase>,
        #[serde(default)]
        max_steps: u64,
        #[serde(default)]
        timeout_ms: u64,
        #[serde(default)]
        max_results: u64,
    },
    Explain {
        target: String,
        #[serde(default)]
        max_depth: u32,
    },
    Promote {
        candidate_id: String,
        #[serde(default)]
        evidence: Vec<String>,
        #[serde(default)]
        expected_version: Option<u64>,
    },
    Rollback {
        procedure_id: String,
        #[serde(default)]
        target_version: Option<u64>,
        #[serde(default)]
        reason: Option<String>,
    },
}

/// Resolve an optional space name to a [`Space`], defaulting to `Working` when absent.
fn space_or_default(name: Option<String>) -> Result<Space, String> {
    match name {
        None => Ok(Space::Working),
        Some(s) => Space::parse(&s).ok_or_else(|| format!("unknown space '{s}'")),
    }
}

/// Build [`Bounds`] from the loose per-op fields, filling any `0` (omitted) field from `defaults`.
fn bounds(max_steps: u64, timeout_ms: u64, max_results: u64, defaults: Bounds) -> Bounds {
    Bounds {
        max_steps: if max_steps == 0 { defaults.max_steps } else { max_steps },
        timeout_ms: if timeout_ms == 0 { defaults.timeout_ms } else { timeout_ms },
        max_results: if max_results == 0 { defaults.max_results } else { max_results },
    }
}

impl MettaArgs {
    /// Convert to a protocol [`MettaCommand`] (the client assigns the request id), using `defaults`
    /// for any omitted eval/test bound.
    fn into_command(self, defaults: Bounds) -> Result<MettaCommand, String> {
        Ok(match self {
            MettaArgs::Inspect => MettaCommand::Inspect { request_id: 0 },
            MettaArgs::SemanticCandidates {
                query,
                space,
                k,
                filters,
            } => MettaCommand::SemanticCandidates {
                request_id: 0,
                query,
                space: space_or_default(space)?,
                k,
                filters,
            },
            MettaArgs::Match {
                pattern,
                space,
                limit,
                cursor,
            } => MettaCommand::Match {
                request_id: 0,
                pattern,
                space: space_or_default(space)?,
                limit,
                cursor,
            },
            MettaArgs::Eval {
                expression,
                space,
                max_steps,
                timeout_ms,
                max_results,
                allow_grounded,
            } => MettaCommand::Eval {
                request_id: 0,
                expression,
                space: space_or_default(space)?,
                bounds: bounds(max_steps, timeout_ms, max_results, defaults),
                allow_grounded,
            },
            MettaArgs::Assert {
                atoms,
                space,
                provenance,
                idempotency_key,
                expected_snapshot,
            } => MettaCommand::Assert {
                request_id: 0,
                atoms,
                space: space_or_default(space)?,
                provenance,
                idempotency_key,
                expected_snapshot,
            },
            MettaArgs::Retract {
                ids,
                pattern,
                space,
                dry_run,
                expected_snapshot,
            } => {
                let target = match (ids.is_empty(), pattern) {
                    (false, _) => RetractTarget::Ids(ids),
                    (true, Some(p)) => RetractTarget::Pattern(p),
                    (true, None) => {
                        return Err("retract needs either `ids` or `pattern`".into())
                    }
                };
                MettaCommand::Retract {
                    request_id: 0,
                    target,
                    space: space_or_default(space)?,
                    dry_run,
                    expected_snapshot,
                }
            }
            MettaArgs::Define {
                program,
                metadata,
                tests,
            } => MettaCommand::Define {
                request_id: 0,
                program,
                metadata,
                tests,
                status: Status::Candidate,
            },
            MettaArgs::Test {
                program_or_id,
                cases,
                max_steps,
                timeout_ms,
                max_results,
            } => MettaCommand::Test {
                request_id: 0,
                program_or_id,
                cases,
                bounds: bounds(max_steps, timeout_ms, max_results, defaults),
            },
            MettaArgs::Explain { target, max_depth } => MettaCommand::Explain {
                request_id: 0,
                target,
                max_depth,
            },
            MettaArgs::Promote {
                candidate_id,
                evidence,
                expected_version,
            } => MettaCommand::Promote {
                request_id: 0,
                candidate_id,
                evidence,
                expected_version,
            },
            MettaArgs::Rollback {
                procedure_id,
                target_version,
                reason,
            } => MettaCommand::Rollback {
                request_id: 0,
                procedure_id,
                target_version,
                reason,
            },
        })
    }
}

/// The JSON-Schema advertised for the tool (covers all 11 ops via the `op` discriminator).
const METTA_SCHEMA: &str = r#"{
  "type": "object",
  "required": ["op"],
  "properties": {
    "op": {
      "type": "string",
      "enum": ["inspect","semantic_candidates","match","eval","assert","retract","define","test","explain","promote","rollback"],
      "description": "The coprocessor operation to perform."
    },
    "space": {
      "type": "string",
      "enum": ["working","episodic","semantic","procedural-candidates","procedural-active","governance"],
      "description": "Target space (default: working)."
    },
    "pattern": {"type": "string", "description": "A MeTTa pattern with $variables (match/retract)."},
    "expression": {"type": "string", "description": "A MeTTa expression to evaluate (eval)."},
    "atoms": {"type": "array", "items": {"type": "string"}, "description": "MeTTa data atoms to assert."},
    "ids": {"type": "array", "items": {"type": "string"}, "description": "Exact record ids to retract."},
    "query": {"type": "string", "description": "A natural-language query (semantic_candidates)."},
    "k": {"type": "integer", "description": "Max candidates to return (semantic_candidates)."},
    "filters": {"type": "array", "items": {"type": "string"}},
    "program": {"type": "string", "description": "An executable MeTTa program/rule (define)."},
    "metadata": {"type": "string", "description": "JSON metadata; an `id` field names the procedure."},
    "tests": {"type": "array", "items": {"type": "string"}},
    "cases": {"type": "array", "items": {"type": "object"}, "description": "Test cases (test)."},
    "target": {"type": "string", "description": "A record/procedure id to explain (explain)."},
    "candidate_id": {"type": "string", "description": "Candidate procedure id to promote (promote)."},
    "procedure_id": {"type": "string", "description": "Procedure id to roll back (rollback)."},
    "evidence": {"type": "array", "items": {"type": "string"}},
    "max_steps": {"type": "integer"},
    "timeout_ms": {"type": "integer"},
    "max_results": {"type": "integer"},
    "max_depth": {"type": "integer"},
    "limit": {"type": "integer"},
    "dry_run": {"type": "boolean"},
    "allow_grounded": {"type": "boolean", "description": "Permit grounded/side-effecting ops (default false)."},
    "idempotency_key": {"type": "string"},
    "expected_snapshot": {"type": "integer", "description": "Optimistic-concurrency CAS token."},
    "expected_version": {"type": "integer"},
    "target_version": {"type": "integer"},
    "provenance": {"type": "object", "description": "Source/derivation metadata for a write."}
  }
}"#;

#[async_trait]
impl Tool for MettaTool {
    fn name(&self) -> &str {
        "metta"
    }

    fn schema(&self) -> &str {
        METTA_SCHEMA
    }

    async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> ToolOutcome {
        let args: MettaArgs = match serde_json::from_str(&call.args) {
            Ok(args) => args,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("metta: invalid arguments: {e}"),
                )
            }
        };

        // `semantic_candidates` is served by the host index (the worker has no embeddings).
        if let MettaArgs::SemanticCandidates {
            query, k, filters, ..
        } = &args
        {
            return self
                .semantic_candidates(&call.call_id, query, *k, filters)
                .await;
        }

        let cmd = match args.into_command(self.default_bounds) {
            Ok(cmd) => cmd,
            Err(e) => return ToolOutcome::text(call.call_id.clone(), false, format!("metta: {e}")),
        };

        match self.copro.request(cmd).await {
            Ok(resp) => {
                let content = render_response(&resp);
                let detail = ToolDetail {
                    kind: "metta".into(),
                    body: serde_json::to_vec(&resp).unwrap_or_default(),
                };
                ToolOutcome::text(call.call_id.clone(), resp.ok, content).with_detail(detail)
            }
            Err(e) => ToolOutcome::text(call.call_id.clone(), false, render_error(&e)),
        }
    }
}

impl MettaTool {
    async fn semantic_candidates(
        &self,
        call_id: &str,
        query: &str,
        k: u32,
        filters: &[String],
    ) -> ToolOutcome {
        match &self.index {
            Some(index) => {
                let k = if k == 0 { 8 } else { k };
                let ids = index.candidates(query, k, filters).await;
                let content = if ids.is_empty() {
                    "metta semantic_candidates: no candidates (hydrate nothing)".to_string()
                } else {
                    format!(
                        "metta semantic_candidates: {} candidate(s) — hydrate via op=match:\n{}",
                        ids.len(),
                        ids.join("\n")
                    )
                };
                ToolOutcome::text(call_id.to_string(), true, content)
            }
            None => ToolOutcome::text(
                call_id.to_string(),
                true,
                "metta semantic_candidates: no host semantic index configured; \
                 use op=match for structural lookup"
                    .to_string(),
            ),
        }
    }
}

/// Render an [`OpResponse`](daemon_metta::protocol::OpResponse) as a concise text summary.
fn render_response(resp: &daemon_metta::protocol::OpResponse) -> String {
    let mut out = String::new();
    out.push_str(&format!("ok={} snapshot={}", resp.ok, resp.snapshot));
    if resp.truncated {
        out.push_str(" (truncated)");
    }
    if !resp.committed_ids.is_empty() {
        out.push_str(&format!("\ncommitted: {}", resp.committed_ids.join(", ")));
    }
    if !resp.results.is_empty() {
        out.push_str("\nresults:\n");
        out.push_str(&resp.results.join("\n"));
    }
    if !resp.tests.is_empty() {
        out.push_str("\ntests:");
        for t in &resp.tests {
            out.push_str(&format!(
                "\n  [{}] {}: {}",
                if t.passed { "pass" } else { "FAIL" },
                t.name,
                t.detail
            ));
        }
    }
    if !resp.warnings.is_empty() {
        out.push_str(&format!("\nwarnings: {}", resp.warnings.join("; ")));
    }
    out
}

fn render_error(e: &MettaError) -> String {
    format!("metta: {e}")
}

/// The MeTTa grammar constraint (both dialects) for grammar-bounded generation. Pair it with
/// [`Request::with_constraint`] so a local engine emits only well-formed MeTTa.
pub fn metta_grammar_constraint() -> GrammarConstraint {
    GrammarConstraint {
        lark: Some(METTA_LARK.to_string()),
        gbnf: Some(METTA_GBNF.to_string()),
    }
}

/// Build a "draft MeTTa" generation request: the model is asked (via `instruction`) to produce a
/// MeTTa program/atom and its output is *constrained* to the MeTTa grammar. The drafted text is
/// then handed to the `metta` tool (`op=eval`/`define`/...). This is the constrained generation path
/// the coprocessor flow uses to keep LLM-authored MeTTa syntactically valid by construction.
pub fn draft_metta_request(system: impl Into<String>, instruction: impl Into<String>) -> Request {
    Request {
        system: system.into(),
        messages: vec![RequestMsg {
            role: "user".into(),
            content: instruction.into(),
            ..Default::default()
        }],
        ..Default::default()
    }
    .with_constraint(metta_grammar_constraint())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_each_op_into_a_command() {
        let cases = [
            (r#"{"op":"inspect"}"#, "Inspect"),
            (r#"{"op":"match","pattern":"(p $x)","space":"semantic"}"#, "Match"),
            (r#"{"op":"eval","expression":"(+ 1 2)"}"#, "Eval"),
            (
                r#"{"op":"assert","atoms":["(p a)"],"space":"semantic","idempotency_key":"k"}"#,
                "Assert",
            ),
            (r#"{"op":"retract","ids":["rec-1"],"space":"semantic"}"#, "Retract"),
            (r#"{"op":"define","program":"(= (f) 1)"}"#, "Define"),
            (r#"{"op":"test","program_or_id":"proc-1"}"#, "Test"),
            (r#"{"op":"explain","target":"rec-1"}"#, "Explain"),
            (r#"{"op":"promote","candidate_id":"proc-1"}"#, "Promote"),
            (r#"{"op":"rollback","procedure_id":"proc-1"}"#, "Rollback"),
        ];
        for (json, label) in cases {
            let args: MettaArgs = serde_json::from_str(json).unwrap_or_else(|e| {
                panic!("parse {label} ({json}): {e}");
            });
            args.into_command(Bounds::default())
                .unwrap_or_else(|e| panic!("convert {label}: {e}"));
        }
    }

    #[test]
    fn retract_requires_ids_or_pattern() {
        let args: MettaArgs = serde_json::from_str(r#"{"op":"retract","space":"semantic"}"#).unwrap();
        assert!(args.into_command(Bounds::default()).is_err());
    }

    #[test]
    fn draft_request_is_grammar_constrained() {
        let req = draft_metta_request("system", "draft a rule for doubling");
        let c = req.constraint.expect("draft request must carry a grammar constraint");
        assert!(c.lark.is_some() && c.gbnf.is_some(), "both dialects present");
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn unknown_space_is_rejected() {
        let args: MettaArgs =
            serde_json::from_str(r#"{"op":"match","pattern":"(p $x)","space":"bogus"}"#).unwrap();
        assert!(args.into_command(Bounds::default()).is_err());
    }
}
