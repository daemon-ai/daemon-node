//! Output/tool repair (§9) — run **once at the decode boundary**.
//!
//! A typed [`Conversation`](crate::conversation) means the engine never re-parses its own history
//! (hermes' preflight conversation sanitizer is unnecessary here). What *does* need repair is the
//! freshly decoded model output, which is the most adversarial input in the loop: providers emit
//! truncated/loosely-quoted tool-call JSON, slightly-wrong tool names, leaked `<think>` spans in the
//! content channel, and tool errors carrying control bytes / untrusted instructions.
//!
//! These modules concentrate that repair so a networked provider can call them at decode and the
//! tool pipeline can reuse arg-repair at its validate stage:
//!
//! - [`tool_arg`] — multi-pass JSON repair + truncation detection + canonicalization.
//! - [`tool_name`] — normalize + fuzzy-match a tool name against the registry, else a protocol-valid
//!   error listing the valid names.
//! - [`content`] — the [`StreamingThinkScrubber`], which strips `<think>`/`<thinking>` spans out of
//!   the content channel and into the reasoning channel across chunk boundaries.
//! - [`tool_error`] — sanitize a tool error string and wrap untrusted tool output so the model reads
//!   it as data, not instructions.

pub mod content;
pub mod tool_arg;
pub mod tool_error;
pub mod tool_name;

pub use content::{scrub_content, ScrubChunk, StreamingThinkScrubber};
pub use tool_arg::{repair_tool_args, ArgRepair};
pub use tool_error::{sanitize_tool_error, wrap_untrusted_tool_result};
pub use tool_name::{repair_tool_name, NameRepairError};

use crate::conversation::ToolCall;

/// Repair one decoded tool call: fuzzy-resolve its `name` against `valid`, then JSON-repair and
/// canonicalize its `args`. Returns the repaired call, or a [`NameRepairError`] when the name cannot
/// be resolved (the provider turns that into a protocol-valid error the model can correct).
pub fn repair_tool_call(call: ToolCall, valid: &[String]) -> Result<ToolCall, NameRepairError> {
    let name = repair_tool_name(&call.name, valid)?;
    let args = repair_tool_args(&call.args).args;
    Ok(ToolCall {
        call_id: call.call_id,
        name,
        args,
    })
}
