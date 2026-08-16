// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! HF chat-template rendering + capability probing (engine-independent).
//!
//! A GGUF's `tokenizer.chat_template` is the authoritative statement of how the model was trained
//! to see conversations — including whether it was trained to see *tools*. This module compiles
//! that Jinja source with minijinja and answers two questions the worker needs:
//!
//! 1. **Capabilities** ([`TemplateCaps`], probed once at load): does the template consume a `tools`
//!    input, can it render `tool_calls`/`tool` history turns, does it honor a `system` role?
//!    Mirrors llama.cpp's `jinja::caps_get` (`common/jinja/caps.cpp`), which trial-renders probe
//!    conversations — we use black-box canary/diff checks instead of its value-usage stats.
//! 2. **Rendering** ([`ChatTemplate::render`]): the prompt string for a conversation, with tools
//!    passed natively through the template (llama-server parity) instead of a hand-rolled text
//!    preamble — the preamble is exactly what tiny instruct models parrot into loops.
//!
//! Pure Rust (no engine dependency), so the probe and renderer are unit-tested in the default stub
//! build against real template fixtures, like [`crate::tooling`].

use minijinja::value::Value;
use minijinja::{context, Environment, Error, ErrorKind};

use crate::protocol::{Msg, ToolDef};

/// Canary strings the probes look for in rendered output. Improbable in any real template text.
const SYSTEM_CANARY: &str = "daemon-sys-canary-7f3a";
const TOOL_CANARY: &str = "daemon_tool_canary_7f3a";
const CALL_CANARY: &str = "daemon_call_canary_7f3a";

/// What the template can express — probed once at construction, cached for the model's lifetime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TemplateCaps {
    /// The template consumes a `tools` input (tool advertisement can go through it natively).
    pub supports_tools: bool,
    /// The template renders assistant `tool_calls` + `tool`-role turns (tool history round-trips).
    pub supports_tool_calls: bool,
    /// `tool_calls[].function.arguments` renders as a JSON *object* (vs a pre-serialized string).
    pub arguments_as_object: bool,
    /// The template honors a `system`-role message (otherwise it is folded into the first user turn).
    pub supports_system_role: bool,
}

/// A compiled chat template plus its probed capabilities.
pub struct ChatTemplate {
    env: Environment<'static>,
    caps: TemplateCaps,
    eos: String,
}

impl ChatTemplate {
    /// Compile `source` and probe its capabilities. `eos` is the model's end-of-turn token text
    /// (templates like Mistral's interleave `{{ eos_token }}` between history turns); `bos_token`
    /// is always rendered empty — the tokenizer adds BOS itself (`AddBos::Always`), so a template
    /// echoing `{{ bos_token }}` must not duplicate it.
    ///
    /// Returns `Err` only when the source fails to *compile*; a template that compiles but cannot
    /// render a plain conversation is caught later (render errors fall back to the C-API path).
    pub fn new(source: &str, eos: &str) -> Result<Self, Error> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        minijinja_contrib::add_to_environment(&mut env);
        // HF-convention helpers: `raise_exception` aborts a render (templates use it to reject
        // configurations they were not trained for — the abort IS the capability signal), and
        // `strftime_now` stamps "today" into date-aware system preambles (SmolLM/Granite family).
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |fmt: String| -> String {
            chrono::Utc::now().format(&fmt).to_string()
        });
        env.add_template_owned("chat".to_string(), source.to_string())?;

        let mut this = Self {
            env,
            caps: TemplateCaps::default(),
            eos: eos.to_string(),
        };
        this.caps = this.probe();
        Ok(this)
    }

    /// The probed capabilities.
    pub fn caps(&self) -> TemplateCaps {
        self.caps
    }

    /// Render the conversation into a prompt, passing `tools` through the template when it
    /// supports them (callers gate on [`TemplateCaps::supports_tools`] — an unsupported `tools`
    /// input is silently unused by the template, so passing it is harmless but pointless).
    pub fn render(
        &self,
        system: &str,
        messages: &[Msg],
        tools: &[ToolDef],
    ) -> Result<String, Error> {
        let mut msgs: Vec<serde_json::Value> = Vec::with_capacity(messages.len() + 1);
        if !system.is_empty() && self.caps.supports_system_role {
            msgs.push(serde_json::json!({"role": "system", "content": system}));
        }
        let mut fold_system =
            (!system.is_empty() && !self.caps.supports_system_role).then(|| system.to_string());
        for m in messages {
            let mut msg = self.message_value(m);
            // No system role: fold the system text into the first user turn (llama.cpp parity).
            if fold_system.is_some() && m.role == "user" {
                let sys = fold_system.take().unwrap_or_default();
                msg["content"] = serde_json::Value::String(format!(
                    "{sys}\n\n{}",
                    msg["content"].as_str().unwrap_or_default()
                ));
            }
            msgs.push(msg);
        }
        // System text with no user turn to fold into: degrade to a leading user message.
        if let Some(sys) = fold_system {
            msgs.insert(0, serde_json::json!({"role": "user", "content": sys}));
        }

        let tools_value: Option<serde_json::Value> =
            (!tools.is_empty()).then(|| tools.iter().map(tool_value).collect());

        self.render_context(&msgs, tools_value.as_ref())
    }

    /// One conversation message as the HF-convention template input.
    fn message_value(&self, m: &Msg) -> serde_json::Value {
        let mut msg = serde_json::json!({"role": m.role, "content": m.content});
        if !m.tool_calls.is_empty() {
            let calls: Vec<serde_json::Value> = m
                .tool_calls
                .iter()
                .map(|c| {
                    let arguments = if self.caps.arguments_as_object {
                        serde_json::from_str::<serde_json::Value>(&c.args)
                            .unwrap_or(serde_json::Value::String(c.args.clone()))
                    } else {
                        serde_json::Value::String(c.args.clone())
                    };
                    serde_json::json!({
                        "id": c.call_id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": arguments},
                    })
                })
                .collect();
            msg["tool_calls"] = serde_json::Value::Array(calls);
        }
        if let Some(id) = &m.tool_call_id {
            msg["tool_call_id"] = serde_json::Value::String(id.clone());
        }
        msg
    }

    /// Render with the standard HF globals. `tools` is absent (undefined) when `None` — templates
    /// guard with `{% if tools %}`, and an empty list would wrongly render "no tools available"
    /// scaffolding on some templates.
    fn render_context(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
    ) -> Result<String, Error> {
        let tmpl = self.env.get_template("chat")?;
        match tools {
            Some(tools) => tmpl.render(context! {
                messages => messages,
                tools => tools,
                add_generation_prompt => true,
                bos_token => "",
                eos_token => self.eos.as_str(),
            }),
            None => tmpl.render(context! {
                messages => messages,
                add_generation_prompt => true,
                bos_token => "",
                eos_token => self.eos.as_str(),
            }),
        }
    }

    // --- capability probes (llama.cpp `caps_get` parity, black-box) ---------------------------

    /// Run all probes. Each is a trial render of a fixed probe conversation; a failed render is
    /// itself a signal (templates `raise_exception` on inputs they were not trained for).
    fn probe(&self) -> TemplateCaps {
        let user = serde_json::json!({"role": "user", "content": "User message"});

        // System role: does canary system content survive into the output?
        let supports_system_role = self
            .render_context(
                &[
                    serde_json::json!({"role": "system", "content": SYSTEM_CANARY}),
                    user.clone(),
                ],
                None,
            )
            .is_ok_and(|out| out.contains(SYSTEM_CANARY));

        // Tools input: does a canary tool's name surface in the rendered prompt?
        let canary_tool = serde_json::json!([{
            "type": "function",
            "function": {
                "name": TOOL_CANARY,
                "description": "Tool description",
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string", "description": "Arg description"}},
                    "required": ["arg"],
                },
            },
        }]);
        let supports_tools = self
            .render_context(std::slice::from_ref(&user), Some(&canary_tool))
            .is_ok_and(|out| out.contains(TOOL_CANARY));

        // Tool-call history: can an assistant tool_calls turn + tool-role result render — and does
        // the call actually SURFACE (a role+content-only template "renders" the turn while
        // silently dropping the calls, which is non-support — the call canary must appear in the
        // output)? Object arguments first (the modern convention), then a pre-serialized string
        // (llama.cpp probes in the same order). The probes carry the advertised canary tool too:
        // templates commonly render history inside their tool scaffolding, so a bare tool_calls
        // turn without a tools input is an unrealistic shape.
        let history = |arguments: serde_json::Value| -> Vec<serde_json::Value> {
            vec![
                user.clone(),
                serde_json::json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call00001",
                        "type": "function",
                        "function": {"name": CALL_CANARY, "arguments": arguments},
                    }],
                }),
                serde_json::json!({
                    "role": "tool",
                    "content": "Tool response",
                    "tool_call_id": "call00001",
                }),
                serde_json::json!({"role": "assistant", "content": "Done"}),
                user.clone(),
            ]
        };
        let probe_tools = supports_tools.then_some(&canary_tool);
        let surfaced = |msgs: &[serde_json::Value]| {
            self.render_context(msgs, probe_tools)
                .is_ok_and(|out| out.contains(CALL_CANARY))
        };
        let (supports_tool_calls, arguments_as_object) =
            if surfaced(&history(serde_json::json!({"arg": "value"}))) {
                (true, true)
            } else if surfaced(&history(serde_json::json!(r#"{"arg": "value"}"#))) {
                (true, false)
            } else {
                (false, false)
            };

        TemplateCaps {
            supports_tools,
            supports_tool_calls,
            arguments_as_object,
            supports_system_role,
        }
    }
}

/// One offered tool as the HF-convention `tools` entry. [`ToolDef::schema`] is the argument
/// JSON-Schema; its `description` doubles as the tool description when present.
fn tool_value(t: &ToolDef) -> serde_json::Value {
    let parameters = serde_json::from_str::<serde_json::Value>(&t.schema)
        .unwrap_or(serde_json::json!({"type": "object"}));
    let description = parameters
        .get("description")
        .and_then(|d| d.as_str())
        .unwrap_or_default()
        .to_string();
    serde_json::json!({
        "type": "function",
        "function": {"name": t.name, "description": description, "parameters": parameters},
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ToolCall;

    // Real chat templates, vendored from llama.cpp's template corpus (models/templates/) and the
    // SmolLM2-135M-Instruct tokenizer_config — the model whose degeneration motivated this module.
    const SMOLLM2: &str = include_str!("../tests/templates/smollm2-135m-instruct.jinja");
    const HERMES_2_PRO: &str = include_str!("../tests/templates/hermes-2-pro-tool-use.jinja");
    const LLAMA_3_2: &str = include_str!("../tests/templates/llama-3.2-instruct.jinja");
    const GEMMA_2: &str = include_str!("../tests/templates/gemma-2-it.jinja");

    fn user(text: &str) -> Msg {
        Msg {
            role: "user".into(),
            content: text.into(),
            ..Default::default()
        }
    }

    fn tool(name: &str) -> ToolDef {
        ToolDef {
            name: name.into(),
            schema: r#"{"type":"object","description":"Recall a memory.","properties":{"query":{"type":"string"}},"required":["query"]}"#.into(),
        }
    }

    /// SmolLM2-135M-Instruct is plain ChatML with NO tools branch — the probe must say so. This is
    /// the capability gate that stops the node advertising tools at this model at all.
    #[test]
    fn smollm2_template_has_no_tool_support() {
        let t = ChatTemplate::new(SMOLLM2, "<|im_end|>").expect("compiles");
        let caps = t.caps();
        assert!(!caps.supports_tools);
        assert!(!caps.supports_tool_calls);
        assert!(caps.supports_system_role);

        // Rendering with tools passed anyway must NOT leak tool names into the prompt.
        let out = t
            .render("Be helpful.", &[user("hi")], &[tool("mnemosyne_recall")])
            .expect("renders");
        assert!(!out.contains("mnemosyne_recall"));
        assert!(out.contains("<|im_start|>system\nBe helpful.<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    /// Hermes 2 Pro's tool_use template consumes `tools` and renders tool_calls/tool history.
    #[test]
    fn hermes_template_supports_tools_natively() {
        let t = ChatTemplate::new(HERMES_2_PRO, "<|im_end|>").expect("compiles");
        let caps = t.caps();
        assert!(caps.supports_tools);
        assert!(caps.supports_tool_calls);
        assert!(caps.supports_system_role);

        let out = t
            .render(
                "Be helpful.",
                &[user("recall x")],
                &[tool("mnemosyne_recall")],
            )
            .expect("renders");
        // The template advertises the tool inside its own <tools> scaffolding.
        assert!(out.contains("mnemosyne_recall"));
        assert!(out.contains("<tools>"));
    }

    /// A tool round-trip (assistant tool_calls turn + tool result) renders through Hermes 2 Pro.
    #[test]
    fn hermes_template_renders_tool_history() {
        let t = ChatTemplate::new(HERMES_2_PRO, "<|im_end|>").expect("compiles");
        let messages = vec![
            user("recall x"),
            Msg {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call-1".into(),
                    name: "mnemosyne_recall".into(),
                    args: r#"{"query":"x"}"#.into(),
                }],
                tool_call_id: None,
            },
            Msg {
                role: "tool".into(),
                content: "x is 42".into(),
                tool_calls: Vec::new(),
                tool_call_id: Some("call-1".into()),
            },
        ];
        let out = t
            .render("", &messages, &[tool("mnemosyne_recall")])
            .expect("renders");
        assert!(out.contains("<tool_call>"));
        assert!(out.contains("x is 42"));
    }

    /// Llama 3.2's template consumes `tools` (its JSON-function advertisement path).
    #[test]
    fn llama32_template_supports_tools() {
        let t = ChatTemplate::new(LLAMA_3_2, "<|eot_id|>").expect("compiles");
        assert!(t.caps().supports_tools);
        assert!(t.caps().supports_system_role);
    }

    /// Gemma 2 raises on a system role; the renderer folds the system text into the first user turn.
    #[test]
    fn gemma_template_folds_system_into_user_turn() {
        let t = ChatTemplate::new(GEMMA_2, "<end_of_turn>").expect("compiles");
        assert!(!t.caps().supports_system_role);
        assert!(!t.caps().supports_tools);

        let out = t.render("Be terse.", &[user("hi")], &[]).expect("renders");
        assert!(out.contains("Be terse.\n\nhi"));
    }

    /// A template that never mentions `tools` must not fail when tools are passed — they are
    /// simply unused (the render gate upstream is advisory, not load-bearing).
    #[test]
    fn tools_are_inert_on_tool_blind_templates() {
        let t = ChatTemplate::new(SMOLLM2, "<|im_end|>").expect("compiles");
        let with = t
            .render("", &[user("hi")], &[tool("canary_zzz")])
            .expect("renders");
        let without = t.render("", &[user("hi")], &[]).expect("renders");
        assert_eq!(with, without);
    }

    /// Invalid Jinja fails compilation (the caller falls back to the C-API path).
    #[test]
    fn broken_template_fails_to_compile() {
        assert!(ChatTemplate::new("{% for m in messages %}{{ m.role }", "").is_err());
    }
}
