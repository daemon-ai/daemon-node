// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Engine-independent tool-calling helpers (Phase 1b): a tool-advertisement preamble, a tool-call
//! parser, and a JSON-Schema → GBNF grammar builder.
//!
//! These are pure functions with no engine dependency, so they compile in the default (stub) build
//! and are unit-tested there — the trickiest logic (parsing the model's tool-call output and the
//! grammar shape) is verified without cmake/llama.cpp. The engine backends use them as building
//! blocks: the llama backend injects [`tool_preamble`], optionally constrains output with
//! [`tools_to_gbnf`], and decodes the result with [`extract_tool_calls`]; mistral.rs uses its native
//! tool API instead and needs none of this.

use crate::protocol::{ToolCall, ToolDef};

/// Build a system-prompt preamble advertising the offered tools and the expected call format.
///
/// Local chat models are steered to emit Hermes-style `<tool_call>{json}</tool_call>` blocks, which
/// [`extract_tool_calls`] decodes. Returns `None` when no tools are offered.
pub fn tool_preamble(tools: &[ToolDef]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    let mut s = String::from(
        "You can call tools. When a tool is needed, emit one or more blocks of the form\n\
         <tool_call>{\"name\": \"<tool>\", \"arguments\": { ... }}</tool_call>\n\
         Use only these tools (name: JSON-Schema):\n",
    );
    for tool in tools {
        s.push_str("- ");
        s.push_str(&tool.name);
        s.push_str(": ");
        s.push_str(&tool.schema);
        s.push('\n');
    }
    Some(s)
}

/// Decode tool calls from generated `text`, returning the text with the tool-call markup removed and
/// the extracted calls (call ids synthesized positionally).
///
/// Recognizes Hermes `<tool_call>…</tool_call>` blocks; if none are present but the whole trimmed
/// text is a single `{"name":…,"arguments":…}` object, that is treated as one call. Names are passed
/// through verbatim — the daemon's §9 repair fuzzy-matches them against the offered tools.
pub fn extract_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut cleaned = String::new();
    let mut rest = text;

    while let Some(start) = find_ci(rest, "<tool_call>") {
        cleaned.push_str(&rest[..start]);
        let after = &rest[start + "<tool_call>".len()..];
        let Some(end) = find_ci(after, "</tool_call>") else {
            // Unterminated block: keep the literal text and stop scanning.
            cleaned.push_str(&rest[start..]);
            rest = "";
            break;
        };
        let body = after[..end].trim();
        if let Some(call) = parse_call_json(body, calls.len()) {
            calls.push(call);
        }
        rest = &after[end + "</tool_call>".len()..];
    }
    cleaned.push_str(rest);

    if calls.is_empty() {
        if let Some(call) = parse_call_json(text.trim(), 0) {
            return (String::new(), vec![call]);
        }
    }

    (cleaned.trim().to_string(), calls)
}

/// Parse a single `{"name": ..., "arguments": ...}` object into a [`ToolCall`].
fn parse_call_json(body: &str, index: usize) -> Option<ToolCall> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let name = value.get("name")?.as_str()?.to_string();
    // Accept "arguments" (Hermes) or "parameters"; default to an empty object.
    let args = value
        .get("arguments")
        .or_else(|| value.get("parameters"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    Some(ToolCall {
        call_id: format!("call-{index}"),
        name,
        args: args.to_string(),
    })
}

/// Case-insensitive substring search returning the byte offset of the first match.
fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    let (h, n) = (haystack.to_ascii_lowercase(), needle.to_ascii_lowercase());
    h.find(&n)
}

/// Build a GBNF grammar constraining output to a Hermes tool-call block whose `name` is one of the
/// offered tools and whose `arguments` is any JSON object.
///
/// This is the JSON-Schema → GBNF seam: Phase 1b constrains the *shape* (a tool-call wrapper over an
/// arbitrary JSON object) and the tool-name alternation; translating each tool's parameter schema
/// into per-field grammar rules is a Phase-2 refinement. Returns `None` when no tools are offered.
/// The engine applies this only when a tool call is *required* (forcing it unconditionally would
/// preclude a plain-text answer).
pub fn tools_to_gbnf(tools: &[ToolDef]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    let names = tools
        .iter()
        .map(|t| format!("\"\\\"{}\\\"\"", escape_gbnf(&t.name)))
        .collect::<Vec<_>>()
        .join(" | ");

    Some(format!(
        r#"root ::= "<tool_call>" ws "{{" ws "\"name\"" ws ":" ws name ws "," ws "\"arguments\"" ws ":" ws object ws "}}" ws "</tool_call>"
name ::= {names}
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{{" ws ( string ws ":" ws value ( ws "," ws string ws ":" ws value )* )? ws "}}"
array ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
string ::= "\"" ( [^"\\] | "\\" . )* "\""
number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?
ws ::= [ \t\n]*
"#
    ))
}

/// Escape a tool name for inclusion in a GBNF double-quoted literal.
fn escape_gbnf(name: &str) -> String {
    name.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> ToolDef {
        ToolDef {
            name: name.to_string(),
            schema: r#"{"type":"object"}"#.to_string(),
        }
    }

    #[test]
    fn preamble_lists_tools_or_none() {
        assert!(tool_preamble(&[]).is_none());
        let p = tool_preamble(&[tool("read_file")]).unwrap();
        assert!(p.contains("read_file"));
        assert!(p.contains("<tool_call>"));
    }

    #[test]
    fn extracts_hermes_block_and_cleans_text() {
        let text = "Sure.<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}</tool_call> done";
        let (cleaned, calls) = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].args, r#"{"path":"a"}"#);
        assert_eq!(cleaned, "Sure. done");
    }

    #[test]
    fn extracts_multiple_blocks() {
        let text = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>";
        let (cleaned, calls) = extract_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].call_id, "call-1");
        assert!(cleaned.is_empty());
    }

    #[test]
    fn bare_json_object_is_a_call() {
        let (cleaned, calls) =
            extract_tool_calls(r#"{"name": "shell", "arguments": {"cmd": "ls"}}"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert!(cleaned.is_empty());
    }

    #[test]
    fn plain_text_yields_no_calls() {
        let (cleaned, calls) = extract_tool_calls("just a normal answer");
        assert!(calls.is_empty());
        assert_eq!(cleaned, "just a normal answer");
    }

    #[test]
    fn unterminated_block_is_kept_literal() {
        let (cleaned, calls) = extract_tool_calls("a <tool_call>{\"name\":\"x\"");
        assert!(calls.is_empty());
        assert!(cleaned.contains("<tool_call>"));
    }

    #[test]
    fn gbnf_lists_names_or_none() {
        assert!(tools_to_gbnf(&[]).is_none());
        let g = tools_to_gbnf(&[tool("read_file"), tool("shell")]).unwrap();
        assert!(g.contains("root ::="));
        assert!(g.contains(r#"\"read_file\""#));
        assert!(g.contains(r#"\"shell\""#));
        assert!(g.contains("name ::="));
    }
}
