// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! B1a — the request-consistency digest: proof that the dispatched provider request is a pure
//! function of the turn's *assembly inputs*.
//!
//! The invariant, stated precisely: the [`Request`](crate::provider::Request) the engine hands a
//! provider must be a deterministic function of the snapshot-durable `conversation` and
//! `composed_prompt` plus the turn-ephemeral [`TurnInjection`] (rebuilt each turn, never
//! persisted), the offered tool set, and the configured cache TTL — nothing else. Two digests are
//! computed at the end of `Engine::assemble` and compared:
//!
//! - [`ModelRequestDigest::of_request`] folds the **assembled request itself**, via exhaustive
//!   struct destructuring (a newly added `Request`/`RequestMsg` field breaks compilation here and
//!   forces an explicit digest decision).
//! - [`ModelRequestDigest::of_assembly`] folds an explicit [`AssemblyInputs`] projection built by
//!   this module's **own** flattening fold — it does not call `build_context` or `assemble`, so a
//!   flattening bug or a post-assemble mutation of the request shows up as a mismatch rather than
//!   being laundered through the same code path twice.
//!
//! This is a *consistency digest*, not historical reconstruction: nothing is durably recorded
//! (durable request recording is the B1b follow-up).
//!
//! # Model Experience
//!
//! None. The digest never reaches the model: a mismatch is a `debug_assert` in debug/test lanes
//! and an error-level trace in release. A user turn is never aborted for a digest mismatch.
//!
//! # KV Cache Effect
//!
//! None directly — the digest observes the request, it never mutates it. Indirectly it guards the
//! cache: the byte-stability the provider prefix cache depends on (composed prompt restored
//! byte-identical, injection confined to the last user message) is exactly what the two-sided
//! digest asserts every turn.
//!
//! # Durability/Replay Effect
//!
//! The digest proves representational agreement between the assembled request and its durable +
//! per-turn inputs, which is the precondition for replay: when B1b records requests durably, this
//! digest is the identity under which they are recorded. The engine-side context tuple (profile,
//! model, injection provenance) rides tracing only in B1a and is NOT persisted.
//!
//! # Security Boundary
//!
//! The digest is secret-free **by exclusion and by assertion**: `Request::auth` is attached later
//! (in `call_model`, after `assemble` returns), the fold destructures it into an ignored binding,
//! and `of_request` `debug_assert!`s it is `None` at digest time. The digest is safe to log.
//!
//! # Known Limitations and Deferred Work
//!
//! - The expected-side fold shares [`repair_message_sequence`](crate::repair) with the real
//!   assembly: repair is the provider structural contract (leading-user, tool pairing), not
//!   assembly, and duplicating it would create drift risk — but a repair bug is consequently
//!   invisible to this digest.
//! - Durable request recording (with attempt/retry provenance and blob refs) is B1b.
//! - The composition-generation id joins the traced context tuple once generations exist (Item 5).

use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use crate::context::{ComposedPrompt, TurnInjection};
use crate::conversation::{Conversation, ToolCall, Turn};
use crate::provider::{
    CacheTtl, GrammarConstraint, Request, RequestImage, RequestMsg, RequestParams,
};
use crate::tools::ToolDef;

/// A canonical SHA-256 digest of one assembled provider request (hex). Secret-free: `auth` never
/// contributes. Distinct from [`CommandFingerprint`](crate::exec::CommandFingerprint) — that name
/// is taken by command approvals; this is a consistency *digest*.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelRequestDigest(String);

impl std::fmt::Display for ModelRequestDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The explicit assembly inputs the expected-side fold digests — everything `assemble()` consumes
/// (review round 3: `cache_ttl` is assigned by `assemble` and belongs here). Main-turn defaults
/// for the remaining request fields (`constraint`/`params`/`task` unset) are fixed inside the fold
/// so both sides agree on unset values.
pub struct AssemblyInputs<'a> {
    /// The snapshot-durable conversation body.
    pub conversation: &'a Conversation,
    /// The snapshot-durable composed prompt; `None` folds an empty system string, matching
    /// `assemble`'s `unwrap_or_default`.
    pub composed: Option<&'a ComposedPrompt>,
    /// The turn-ephemeral injection (rebuilt each turn, never persisted).
    pub injection: &'a TurnInjection,
    /// The tools offered this turn, in offer order.
    pub tools: &'a [ToolDef],
    /// The configured cache TTL `assemble` stamps on the request.
    pub cache_ttl: CacheTtl,
}

/// The canonical encoding (fixed, not ad hoc): every field label-tagged and length-prefixed into
/// one SHA-256 stream (the `CommandFingerprint::compute` idiom); counts and integers little-endian;
/// bools one byte; `Option` a presence tag + value; floats via `to_bits()`; enums explicit stable
/// tags; message and tool order preserved as-is (order is provider-visible — never sorted). No
/// serde/JSON anywhere in the digest path.
struct Enc(Sha256);

impl Enc {
    fn new() -> Self {
        Enc(Sha256::new())
    }

    fn bytes(&mut self, label: &str, bytes: &[u8]) {
        self.0.update((label.len() as u64).to_le_bytes());
        self.0.update(label.as_bytes());
        self.0.update((bytes.len() as u64).to_le_bytes());
        self.0.update(bytes);
    }

    fn str(&mut self, label: &str, s: &str) {
        self.bytes(label, s.as_bytes());
    }

    fn u64(&mut self, label: &str, v: u64) {
        self.bytes(label, &v.to_le_bytes());
    }

    fn bool(&mut self, label: &str, v: bool) {
        self.bytes(label, &[u8::from(v)]);
    }

    fn opt_str(&mut self, label: &str, v: Option<&str>) {
        match v {
            None => self.bytes(label, &[0]),
            Some(s) => {
                self.bytes(label, &[1]);
                self.str(label, s);
            }
        }
    }

    fn opt_f64(&mut self, label: &str, v: Option<f64>) {
        match v {
            None => self.bytes(label, &[0]),
            Some(f) => {
                self.bytes(label, &[1]);
                self.bytes(label, &f.to_bits().to_le_bytes());
            }
        }
    }

    fn opt_u64(&mut self, label: &str, v: Option<u64>) {
        match v {
            None => self.bytes(label, &[0]),
            Some(n) => {
                self.bytes(label, &[1]);
                self.u64(label, n);
            }
        }
    }

    fn finish(self) -> ModelRequestDigest {
        let digest = self.0.finalize();
        let mut hex = String::with_capacity(digest.len() * 2);
        for b in digest {
            let _ = write!(hex, "{b:02x}");
        }
        ModelRequestDigest(hex)
    }
}

impl ModelRequestDigest {
    /// Digest the assembled request via exhaustive destructuring — a new `Request` field breaks
    /// compilation here and forces a digest decision. `auth` is excluded explicitly and by
    /// assertion: `assemble` computes this digest *before* `call_model` attaches the lease secret.
    pub fn of_request(req: &Request) -> Self {
        let Request {
            system,
            messages,
            tools,
            auth,
            constraint,
            cache_system,
            cache_ttl,
            params,
            task,
        } = req;
        debug_assert!(
            auth.is_none(),
            "request digest must be computed before the lease secret is attached"
        );
        digest_shape(
            system,
            messages,
            tools,
            constraint.as_ref(),
            *cache_system,
            *cache_ttl,
            params,
            task.as_deref(),
        )
    }

    /// Digest the expected request shape by this module's own fold over the explicit
    /// [`AssemblyInputs`] — the anti-tautology side. It flattens the conversation itself (a
    /// `Turn::Tool` becomes an assistant message carrying its calls plus one `tool` message per
    /// result), applies the shared structural repair, appends the injection to the last user
    /// message, takes the system string from the composed prompt (empty when absent), and marks
    /// the `system_and_3` cache breakpoints — all without calling `build_context` or `assemble`.
    /// Main-turn defaults are fixed here: no constraint, default params, no task, no auth.
    pub fn of_assembly(inputs: &AssemblyInputs<'_>) -> Self {
        let AssemblyInputs {
            conversation,
            composed,
            injection,
            tools,
            cache_ttl,
        } = inputs;

        // The documented flattening contract, folded independently of `build_context`.
        let mut messages: Vec<RequestMsg> = Vec::new();
        for turn in &conversation.turns {
            match turn {
                Turn::User(u) => messages.push(RequestMsg {
                    role: "user".into(),
                    content: u.text.clone(),
                    ..Default::default()
                }),
                Turn::Assistant(a) => messages.push(RequestMsg {
                    role: "assistant".into(),
                    content: a.text.clone(),
                    ..Default::default()
                }),
                Turn::Tool(t) => {
                    messages.push(RequestMsg {
                        role: "assistant".into(),
                        content: t.assistant.text.clone(),
                        tool_calls: t.calls.iter().map(|(call, _)| call.clone()).collect(),
                        ..Default::default()
                    });
                    for (_call, result) in &t.calls {
                        messages.push(RequestMsg {
                            role: "tool".into(),
                            content: result.content.clone(),
                            tool_call_id: Some(result.call_id.clone()),
                            ..Default::default()
                        });
                    }
                }
            }
        }
        // Shared with the real assembly deliberately (see module docs, Known Limitations): repair
        // is the provider structural contract, not assembly.
        let mut messages = crate::repair::repair_message_sequence(messages);

        // Injection appends to the last user message of the outgoing request only.
        if !injection.is_empty() {
            if let Some(last_user) = messages.iter_mut().rev().find(|m| m.role == "user") {
                if !last_user.content.is_empty() {
                    last_user.content.push_str("\n\n");
                }
                last_user.content.push_str(&injection.render());
            }
        }

        // Composed prompt overrides the system string; absent composition folds an empty system.
        let system = composed.map(ComposedPrompt::render).unwrap_or_default();

        // `system_and_3` breakpoints: the system prefix (when non-empty) plus the trailing
        // messages up to four total.
        let cache_system = !system.is_empty();
        let remaining = 4 - usize::from(cache_system);
        let skip = messages.len().saturating_sub(remaining);
        for msg in messages.iter_mut().skip(skip) {
            msg.cache_breakpoint = true;
        }

        digest_shape(
            &system,
            &messages,
            tools,
            None,
            cache_system,
            *cache_ttl,
            &RequestParams::default(),
            None,
        )
    }
}

/// The shared canonical fold both sides feed — same encoding, independently constructed inputs.
#[allow(clippy::too_many_arguments)]
fn digest_shape(
    system: &str,
    messages: &[RequestMsg],
    tools: &[ToolDef],
    constraint: Option<&GrammarConstraint>,
    cache_system: bool,
    cache_ttl: CacheTtl,
    params: &RequestParams,
    task: Option<&str>,
) -> ModelRequestDigest {
    let mut e = Enc::new();
    e.str("system", system);

    e.u64("message_count", messages.len() as u64);
    for msg in messages {
        let RequestMsg {
            role,
            content,
            tool_calls,
            tool_call_id,
            cache_breakpoint,
            images,
        } = msg;
        e.str("role", role);
        e.str("content", content);
        e.u64("tool_call_count", tool_calls.len() as u64);
        for call in tool_calls {
            let ToolCall {
                call_id,
                name,
                args,
            } = call;
            e.str("call_id", call_id);
            e.str("call_name", name);
            e.str("call_args", args);
        }
        e.opt_str("tool_call_id", tool_call_id.as_deref());
        e.bool("cache_breakpoint", *cache_breakpoint);
        e.u64("image_count", images.len() as u64);
        for image in images {
            let RequestImage { mime, data_base64 } = image;
            e.str("image_mime", mime);
            e.str("image_data", data_base64);
        }
    }

    e.u64("tool_count", tools.len() as u64);
    for tool in tools {
        let ToolDef { name, schema } = tool;
        e.str("tool_name", name);
        e.str("tool_schema", schema);
    }

    // `auth` is excluded by construction — it never reaches this fold.
    match constraint {
        None => e.bool("constraint", false),
        Some(GrammarConstraint { lark, gbnf }) => {
            e.bool("constraint", true);
            e.opt_str("constraint_lark", lark.as_deref());
            e.opt_str("constraint_gbnf", gbnf.as_deref());
        }
    }
    e.bool("cache_system", cache_system);
    e.bytes(
        "cache_ttl",
        match cache_ttl {
            CacheTtl::FiveMin => &[0],
            CacheTtl::OneHour => &[1],
        },
    );
    let RequestParams {
        temperature,
        top_p,
        top_k,
        max_tokens,
        seed,
    } = params;
    e.opt_f64("temperature", *temperature);
    e.opt_f64("top_p", *top_p);
    e.opt_u64("top_k", top_k.map(u64::from));
    e.opt_u64("max_tokens", max_tokens.map(u64::from));
    e.opt_u64("seed", *seed);
    e.opt_str("task", task);
    e.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::SlotKind;
    use crate::conversation::{AssistantMsg, SystemPrompt, ToolResult, ToolTurn, UserMsg};
    use crate::provider::{build_context, mark_cache_breakpoints};

    /// Replicate the REAL assembly (`Engine::assemble`) from the same inputs: `build_context`,
    /// composed-prompt system override, injection, cache TTL, breakpoints — the production code
    /// path the independent fold must agree with.
    fn real_assemble(inputs: &AssemblyInputs<'_>) -> Request {
        let mut req = build_context(inputs.conversation, inputs.tools);
        req.system = inputs
            .composed
            .map(ComposedPrompt::render)
            .unwrap_or_default();
        inputs.injection.apply_to_last_user(&mut req);
        req.cache_ttl = inputs.cache_ttl;
        mark_cache_breakpoints(&mut req);
        req
    }

    fn composed(text: &str) -> ComposedPrompt {
        let mut b = ComposedPrompt::builder();
        b.push(SlotKind::Identity, text.to_string());
        b.build()
    }

    fn tool_turn_conversation() -> Conversation {
        let mut conv = Conversation::new(SystemPrompt::new("sys"));
        conv.push_user(UserMsg::new("run the check"));
        conv.push_tool(ToolTurn {
            assistant: AssistantMsg::text("checking"),
            calls: vec![
                (
                    ToolCall {
                        call_id: "c1".into(),
                        name: "shell".into(),
                        args: r#"{"command":"true"}"#.into(),
                    },
                    ToolResult {
                        call_id: "c1".into(),
                        ok: true,
                        content: "ok".into(),
                    },
                ),
                (
                    ToolCall {
                        call_id: "c2".into(),
                        name: "fs".into(),
                        args: r#"{"op":"read","path":"a"}"#.into(),
                    },
                    ToolResult {
                        call_id: "c2".into(),
                        ok: true,
                        content: "contents".into(),
                    },
                ),
            ],
        });
        conv.push_user(UserMsg::new("and now?"));
        conv
    }

    fn defs() -> Vec<ToolDef> {
        vec![
            ToolDef {
                name: "shell".into(),
                schema: "{}".into(),
            },
            ToolDef {
                name: "fs".into(),
                schema: r#"{"op":{}}"#.into(),
            },
        ]
    }

    #[test]
    fn assembled_request_agrees_with_assembly_fold() {
        let conv = tool_turn_conversation();
        let comp = composed("persona text");
        let injection = TurnInjection {
            recalled: vec!["remembered fact".into()],
            nudges: vec!["nudge".into()],
        };
        let tools = defs();
        let inputs = AssemblyInputs {
            conversation: &conv,
            composed: Some(&comp),
            injection: &injection,
            tools: &tools,
            cache_ttl: CacheTtl::OneHour,
        };
        let req = real_assemble(&inputs);
        assert_eq!(
            ModelRequestDigest::of_request(&req),
            ModelRequestDigest::of_assembly(&inputs),
            "real assembly and the independent fold must agree (tool linkage + injection + composition)"
        );
    }

    #[test]
    fn agreement_holds_without_injection_and_without_composition() {
        let conv = tool_turn_conversation();
        let injection = TurnInjection::default();
        let tools = defs();
        let inputs = AssemblyInputs {
            conversation: &conv,
            composed: None,
            injection: &injection,
            tools: &tools,
            cache_ttl: CacheTtl::FiveMin,
        };
        let req = real_assemble(&inputs);
        assert!(
            req.system.is_empty(),
            "no composition folds an empty system"
        );
        assert_eq!(
            ModelRequestDigest::of_request(&req),
            ModelRequestDigest::of_assembly(&inputs)
        );
    }

    #[test]
    fn digest_is_stable_across_recomputation_and_snapshot_round_trip() {
        let conv = tool_turn_conversation();
        let comp = composed("persona");
        let injection = TurnInjection::default();
        let tools = defs();
        let inputs = AssemblyInputs {
            conversation: &conv,
            composed: Some(&comp),
            injection: &injection,
            tools: &tools,
            cache_ttl: CacheTtl::FiveMin,
        };
        let first = ModelRequestDigest::of_assembly(&inputs);
        assert_eq!(first, ModelRequestDigest::of_assembly(&inputs));

        // Snapshot resume: the conversation survives a serde round-trip (the durable form) and the
        // digest is unchanged — the byte-identical restore invariant.
        let json = serde_json::to_string(&conv).unwrap();
        let restored: Conversation = serde_json::from_str(&json).unwrap();
        let resumed = AssemblyInputs {
            conversation: &restored,
            composed: Some(&comp),
            injection: &injection,
            tools: &tools,
            cache_ttl: CacheTtl::FiveMin,
        };
        assert_eq!(first, ModelRequestDigest::of_assembly(&resumed));
    }

    #[test]
    fn deliberate_mutations_each_break_equality() {
        let conv = tool_turn_conversation();
        let comp = composed("persona");
        let injection = TurnInjection::default();
        let tools = defs();
        let inputs = AssemblyInputs {
            conversation: &conv,
            composed: Some(&comp),
            injection: &injection,
            tools: &tools,
            cache_ttl: CacheTtl::FiveMin,
        };
        let baseline = real_assemble(&inputs);
        let expected = ModelRequestDigest::of_assembly(&inputs);
        assert_eq!(ModelRequestDigest::of_request(&baseline), expected);

        // An extra message (a post-assemble mutation the tautological check would miss).
        let mut extra = baseline.clone();
        extra.messages.push(RequestMsg {
            role: "user".into(),
            content: "smuggled".into(),
            ..Default::default()
        });
        assert_ne!(ModelRequestDigest::of_request(&extra), expected);

        // A changed sampling param.
        let mut warmed = baseline.clone();
        warmed.params.temperature = Some(0.7);
        assert_ne!(ModelRequestDigest::of_request(&warmed), expected);

        // Reordered tools (order is provider-visible; the digest must not sort it away).
        let mut reordered = baseline.clone();
        reordered.tools.reverse();
        assert_ne!(ModelRequestDigest::of_request(&reordered), expected);

        // A changed cache TTL.
        let mut ttl = baseline.clone();
        ttl.cache_ttl = CacheTtl::OneHour;
        assert_ne!(ModelRequestDigest::of_request(&ttl), expected);
    }

    #[test]
    #[should_panic(expected = "before the lease secret")]
    fn digest_refuses_an_auth_bearing_request() {
        // `auth` is excluded by assertion: digesting after the secret is attached is a misuse of
        // the API (the digest must run inside `assemble`, before `call_model` threads the lease).
        let conv = tool_turn_conversation();
        let tools = defs();
        let mut req = build_context(&conv, &tools);
        req.auth = Some("bearer-secret".into());
        let _ = ModelRequestDigest::of_request(&req);
    }

    #[test]
    fn empty_and_boundary_shapes_agree() {
        // Empty conversation (repair drops everything), no tools, no composition.
        let conv = Conversation::default();
        let injection = TurnInjection::default();
        let inputs = AssemblyInputs {
            conversation: &conv,
            composed: None,
            injection: &injection,
            tools: &[],
            cache_ttl: CacheTtl::FiveMin,
        };
        let req = real_assemble(&inputs);
        assert_eq!(
            ModelRequestDigest::of_request(&req),
            ModelRequestDigest::of_assembly(&inputs)
        );

        // A single user message with injection: the breakpoint lands on the injected message and
        // both sides must still agree byte-for-byte.
        let mut conv = Conversation::new(SystemPrompt::new("s"));
        conv.push_user(UserMsg::new("hi"));
        let injection = TurnInjection {
            recalled: vec!["recall".into()],
            nudges: vec![],
        };
        let comp = composed("sys");
        let tools = defs();
        let inputs = AssemblyInputs {
            conversation: &conv,
            composed: Some(&comp),
            injection: &injection,
            tools: &tools,
            cache_ttl: CacheTtl::FiveMin,
        };
        let req = real_assemble(&inputs);
        assert_eq!(
            ModelRequestDigest::of_request(&req),
            ModelRequestDigest::of_assembly(&inputs)
        );
    }
}
