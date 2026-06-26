//! The evaluation/matching engine seam.
//!
//! A [`MettaEngine`] evaluates a MeTTa expression and structurally matches a pattern against a set
//! of records. There are two implementations:
//!
//! - [`FallbackEngine`] (always compiled): a pure-Rust S-expression matcher + a small arithmetic
//!   reducer. It needs no `hyperon`, so the default workspace gate (and `cargo test --workspace`)
//!   exercises the full protocol/state/lifecycle without the engine. Anything beyond the arithmetic
//!   subset reports [`ErrorClass::Unsupported`](crate::protocol::ErrorClass::Unsupported).
//! - `HyperonEngine` (behind the `hyperon` feature): a real MeTTa runner with **bounded** stepped
//!   evaluation. Built only for the dedicated worker output, never in the default gate.
//!
//! The engine is owned by the worker's single actor thread, so — like `hyperon`'s `Rc`-based runner
//! — it is **not** required to be `Send`/`Sync`.

use crate::protocol::Bounds;
use crate::state::Record;

/// A classified engine failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// The input could not be parsed.
    Parse(String),
    /// The op is not supported by this engine build (needs the `hyperon` feature).
    Unsupported(String),
    /// An internal evaluation error.
    Internal(String),
}

/// The result of a bounded evaluation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvalResult {
    /// The rendered result atoms.
    pub results: Vec<String>,
    /// Interpreter steps consumed.
    pub steps: u64,
    /// Whether a bound (steps/time/results) truncated the output.
    pub truncated: bool,
}

/// The result of a structural match.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MatchResult {
    /// Rendered variable bindings, one entry per matching record.
    pub bindings: Vec<String>,
    /// The ids of the records that matched (aligned with `bindings`).
    pub record_ids: Vec<String>,
    /// Whether the `limit` truncated the output.
    pub truncated: bool,
}

/// The engine seam. Not `Send`/`Sync`: it lives on the worker's dedicated actor thread.
pub trait MettaEngine {
    /// The compiled engine identifier (reported in [`crate::protocol::Event::Ready`]).
    fn name(&self) -> &'static str;

    /// Bounded evaluation of `expression` with `records` loaded as the space contents.
    fn eval(
        &mut self,
        expression: &str,
        records: &[&Record],
        bounds: Bounds,
        allow_grounded: bool,
    ) -> Result<EvalResult, EngineError>;

    /// Structural match of `pattern` against `records`, capped at `limit` (0 = engine default).
    fn match_pattern(
        &mut self,
        pattern: &str,
        records: &[&Record],
        limit: usize,
    ) -> Result<MatchResult, EngineError>;
}

// ---------------------------------------------------------------------------
// A tiny S-expression parser + unifier (shared by the fallback engine).
// ---------------------------------------------------------------------------

/// A parsed S-expression atom (the subset the fallback understands).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sexpr {
    /// A symbol / grounded token (`foo`, `42`, `"a string"`).
    Sym(String),
    /// A variable (`$x`).
    Var(String),
    /// An expression `( ... )`.
    List(Vec<Sexpr>),
}

impl Sexpr {
    /// Render back to MeTTa source text.
    pub fn render(&self) -> String {
        match self {
            Sexpr::Sym(s) | Sexpr::Var(s) => s.clone(),
            Sexpr::List(items) => {
                let inner: Vec<String> = items.iter().map(Sexpr::render).collect();
                format!("({})", inner.join(" "))
            }
        }
    }
}

/// Parse exactly one S-expression from `src` (ignoring trailing whitespace/comments).
pub fn parse_sexpr(src: &str) -> Result<Sexpr, EngineError> {
    let tokens = tokenize(src)?;
    let mut pos = 0;
    let expr = parse_tokens(&tokens, &mut pos)?;
    // Allow trailing tokens to be ignored (a record may have side metadata); we only need the head.
    Ok(expr)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Open,
    Close,
    Atom(String),
}

fn tokenize(src: &str) -> Result<Vec<Token>, EngineError> {
    let mut tokens = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '(' => {
                tokens.push(Token::Open);
                chars.next();
            }
            ')' => {
                tokens.push(Token::Close);
                chars.next();
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            ';' => {
                // line comment to EOL
                for nc in chars.by_ref() {
                    if nc == '\n' {
                        break;
                    }
                }
            }
            '"' => {
                // a quoted string token, kept verbatim (including quotes)
                let mut s = String::from("\"");
                chars.next();
                let mut closed = false;
                for nc in chars.by_ref() {
                    s.push(nc);
                    if nc == '"' {
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    return Err(EngineError::Parse("unterminated string".into()));
                }
                tokens.push(Token::Atom(s));
            }
            _ => {
                let mut s = String::new();
                while let Some(&nc) = chars.peek() {
                    if nc.is_whitespace() || nc == '(' || nc == ')' || nc == ';' {
                        break;
                    }
                    s.push(nc);
                    chars.next();
                }
                tokens.push(Token::Atom(s));
            }
        }
    }
    Ok(tokens)
}

fn parse_tokens(tokens: &[Token], pos: &mut usize) -> Result<Sexpr, EngineError> {
    let tok = tokens
        .get(*pos)
        .ok_or_else(|| EngineError::Parse("unexpected end of input".into()))?;
    match tok {
        Token::Open => {
            *pos += 1;
            let mut items = Vec::new();
            loop {
                match tokens.get(*pos) {
                    Some(Token::Close) => {
                        *pos += 1;
                        break;
                    }
                    Some(_) => items.push(parse_tokens(tokens, pos)?),
                    None => return Err(EngineError::Parse("unclosed '('".into())),
                }
            }
            Ok(Sexpr::List(items))
        }
        Token::Close => Err(EngineError::Parse("unexpected ')'".into())),
        Token::Atom(s) => {
            *pos += 1;
            if let Some(rest) = s.strip_prefix('$') {
                Ok(Sexpr::Var(format!("${rest}")))
            } else {
                Ok(Sexpr::Sym(s.clone()))
            }
        }
    }
}

/// Bindings produced by a unification (`$x` -> rendered value).
type Bindings = std::collections::BTreeMap<String, String>;

/// Unify `pattern` against `target`, accumulating variable bindings. Variables only appear in the
/// pattern (the target is ground data).
fn unify(pattern: &Sexpr, target: &Sexpr, bindings: &mut Bindings) -> bool {
    match (pattern, target) {
        (Sexpr::Var(v), t) => {
            let rendered = t.render();
            match bindings.get(v) {
                Some(existing) => *existing == rendered,
                None => {
                    bindings.insert(v.clone(), rendered);
                    true
                }
            }
        }
        (Sexpr::Sym(a), Sexpr::Sym(b)) => a == b,
        (Sexpr::List(a), Sexpr::List(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(pa, ta)| unify(pa, ta, bindings))
        }
        _ => false,
    }
}

/// Render a binding set deterministically (`{$x = foo, $y = bar}`; `{}` for a variable-free match).
fn render_bindings(bindings: &Bindings) -> String {
    if bindings.is_empty() {
        return "{}".to_string();
    }
    let parts: Vec<String> = bindings.iter().map(|(k, v)| format!("{k} = {v}")).collect();
    format!("{{{}}}", parts.join(", "))
}

/// Structural match shared by both engines: parse each record, unify with the pattern.
pub fn structural_match(
    pattern: &str,
    records: &[&Record],
    limit: usize,
) -> Result<MatchResult, EngineError> {
    let pat = parse_sexpr(pattern)?;
    let cap = if limit == 0 { usize::MAX } else { limit };
    let mut out = MatchResult::default();
    for record in records {
        let Ok(target) = parse_sexpr(&record.text) else {
            continue; // skip records that aren't well-formed S-expressions
        };
        let mut bindings = Bindings::new();
        if unify(&pat, &target, &mut bindings) {
            if out.record_ids.len() >= cap {
                out.truncated = true;
                break;
            }
            out.bindings.push(render_bindings(&bindings));
            out.record_ids.push(record.id.clone());
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Fallback engine (always compiled)
// ---------------------------------------------------------------------------

/// The pure-Rust engine: structural match + a small arithmetic reducer. No `hyperon`.
#[derive(Default)]
pub struct FallbackEngine;

impl FallbackEngine {
    /// A new fallback engine.
    pub fn new() -> Self {
        Self
    }
}

/// Reduce the small arithmetic subset (`+ - * /` over integer literals), MeTTa-style: a reducible
/// expression yields its value; an irreducible one yields itself (identity). Returns `None` for
/// anything the fallback cannot evaluate (the caller reports `Unsupported`).
fn reduce_arith(expr: &Sexpr) -> Option<i64> {
    match expr {
        Sexpr::Sym(s) => s.parse::<i64>().ok(),
        Sexpr::List(items) => {
            let [Sexpr::Sym(op), a, b] = items.as_slice() else {
                return None;
            };
            let (x, y) = (reduce_arith(a)?, reduce_arith(b)?);
            match op.as_str() {
                "+" => Some(x + y),
                "-" => Some(x - y),
                "*" => Some(x * y),
                "/" if y != 0 => Some(x / y),
                _ => None,
            }
        }
        Sexpr::Var(_) => None,
    }
}

impl MettaEngine for FallbackEngine {
    fn name(&self) -> &'static str {
        "fallback"
    }

    fn eval(
        &mut self,
        expression: &str,
        _records: &[&Record],
        _bounds: Bounds,
        _allow_grounded: bool,
    ) -> Result<EvalResult, EngineError> {
        let expr = parse_sexpr(expression)?;
        match reduce_arith(&expr) {
            Some(value) => Ok(EvalResult {
                results: vec![value.to_string()],
                steps: 1,
                truncated: false,
            }),
            None => Err(EngineError::Unsupported(
                "the fallback engine evaluates only the integer arithmetic subset; \
                 build the worker with --features hyperon for full MeTTa evaluation"
                    .into(),
            )),
        }
    }

    fn match_pattern(
        &mut self,
        pattern: &str,
        records: &[&Record],
        limit: usize,
    ) -> Result<MatchResult, EngineError> {
        structural_match(pattern, records, limit)
    }
}

// ---------------------------------------------------------------------------
// Hyperon engine (behind the `hyperon` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "hyperon")]
mod hyperon_engine {
    use super::*;
    use hyperon::metta::runner::{Metta, RunnerState};
    use hyperon::metta::text::SExprParser;

    /// Symbols whose presence makes an expression side-effecting; rejected when `allow_grounded` is
    /// false (SKILL §Separate external action from symbolic evaluation).
    const GROUNDED_DENYLIST: &[&str] = &[
        "fileio",
        "read-file",
        "write-file",
        "println!",
        "print",
        "system",
        "shell",
        "http",
    ];

    /// The real MeTTa engine. Holds one runner; records are (re)loaded into a fresh runner per eval
    /// so evaluation is pure with respect to the durable store (the store is the source of truth).
    pub struct HyperonEngine;

    impl HyperonEngine {
        pub fn new() -> Self {
            Self
        }

        fn load_records(metta: &Metta, records: &[&Record]) {
            for record in records {
                // Non-`!` top-level atoms are added to the space when run.
                let _ = metta.run(SExprParser::new(record.text.as_str()));
            }
        }
    }

    impl MettaEngine for HyperonEngine {
        fn name(&self) -> &'static str {
            "hyperon"
        }

        fn eval(
            &mut self,
            expression: &str,
            records: &[&Record],
            bounds: Bounds,
            allow_grounded: bool,
        ) -> Result<EvalResult, EngineError> {
            if !allow_grounded {
                let lower = expression.to_ascii_lowercase();
                if let Some(bad) = GROUNDED_DENYLIST.iter().find(|s| lower.contains(*s)) {
                    return Err(EngineError::Unsupported(format!(
                        "grounded/side-effecting symbol '{bad}' is denied (allow_grounded = false)"
                    )));
                }
            }
            let metta = Metta::new(None);
            Self::load_records(&metta, records);

            let max_steps = if bounds.max_steps == 0 {
                1_000
            } else {
                bounds.max_steps
            };
            let max_results = if bounds.max_results == 0 {
                100
            } else {
                bounds.max_results
            } as usize;
            let deadline = (bounds.timeout_ms > 0).then(|| {
                std::time::Instant::now() + std::time::Duration::from_millis(bounds.timeout_ms)
            });

            // A bare top-level atom is *added* to the space by the runner; only an atom marked with
            // the `!` "evaluate" prefix is reduced and its result returned. Prepend `!` when the
            // caller did not, so `(+ 2 3)` evaluates to `5` rather than silently being stored.
            let trimmed = expression.trim_start();
            let source = if trimmed.starts_with('!') {
                expression.to_string()
            } else {
                format!("!{trimmed}")
            };
            let mut state =
                RunnerState::new_with_parser(&metta, Box::new(SExprParser::new(source.as_str())));
            let mut steps = 0u64;
            let mut truncated = false;
            while !state.is_complete() {
                if steps >= max_steps {
                    truncated = true;
                    break;
                }
                if let Some(deadline) = deadline {
                    if std::time::Instant::now() >= deadline {
                        truncated = true;
                        break;
                    }
                }
                state.run_step().map_err(|e| EngineError::Internal(e))?;
                steps += 1;
            }

            let mut results: Vec<String> = state
                .into_results()
                .into_iter()
                .flatten()
                .map(|atom| atom.to_string())
                .collect();
            if results.len() > max_results {
                results.truncate(max_results);
                truncated = true;
            }
            Ok(EvalResult {
                results,
                steps,
                truncated,
            })
        }

        fn match_pattern(
            &mut self,
            pattern: &str,
            records: &[&Record],
            limit: usize,
        ) -> Result<MatchResult, EngineError> {
            // Structural matching over the stored records is engine-independent and deterministic;
            // it mirrors the SKILL `match` contract (bindings + record ids) without depending on a
            // particular runner's query internals.
            structural_match(pattern, records, limit)
        }
    }
}

#[cfg(feature = "hyperon")]
pub use hyperon_engine::HyperonEngine;

/// Build the default engine for this compilation: hyperon when the feature is on, else the fallback.
pub fn default_engine() -> Box<dyn MettaEngine> {
    #[cfg(feature = "hyperon")]
    {
        Box::new(HyperonEngine::new())
    }
    #[cfg(not(feature = "hyperon"))]
    {
        Box::new(FallbackEngine::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Provenance, Space, Status};

    fn rec(id: &str, text: &str) -> Record {
        Record {
            id: id.into(),
            space: Space::Semantic,
            text: text.into(),
            provenance: Provenance::default(),
            status: Status::Active,
            created_at_snapshot: 0,
            supersedes: None,
        }
    }

    #[test]
    fn parses_nested_sexpr() {
        let e = parse_sexpr("(owns $p (artifact 42))").unwrap();
        assert_eq!(e.render(), "(owns $p (artifact 42))");
    }

    #[test]
    fn structural_match_binds_variables() {
        let r1 = rec("r1", "(owns alice artifact-42)");
        let r2 = rec("r2", "(owns bob artifact-7)");
        let records = [&r1, &r2];
        let m = structural_match("(owns $p artifact-42)", &records, 0).unwrap();
        assert_eq!(m.record_ids, vec!["r1".to_string()]);
        assert_eq!(m.bindings, vec!["{$p = alice}".to_string()]);
    }

    #[test]
    fn structural_match_respects_limit() {
        let r1 = rec("r1", "(p a)");
        let r2 = rec("r2", "(p b)");
        let records = [&r1, &r2];
        let m = structural_match("(p $x)", &records, 1).unwrap();
        assert_eq!(m.record_ids.len(), 1);
        assert!(m.truncated);
    }

    #[test]
    fn fallback_evaluates_arithmetic() {
        let mut e = FallbackEngine::new();
        let out = e.eval("(+ 1 2)", &[], Bounds::default(), false).unwrap();
        assert_eq!(out.results, vec!["3".to_string()]);
        let nested = e
            .eval("(* (+ 1 2) 4)", &[], Bounds::default(), false)
            .unwrap();
        assert_eq!(nested.results, vec!["12".to_string()]);
    }

    #[test]
    fn fallback_rejects_non_arithmetic() {
        let mut e = FallbackEngine::new();
        let err = e
            .eval("(double 2)", &[], Bounds::default(), false)
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)));
    }
}
