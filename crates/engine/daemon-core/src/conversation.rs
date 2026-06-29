// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The typed conversation (§5) — the engine's source of truth.
//!
//! A [`Conversation`] is a system prompt plus an ordered list of [`Turn`]s. The key invariant is
//! that a tool call cannot exist without its result slot: assistant tool calls and their results
//! live together inside a [`ToolTurn`], so an orphaned tool result is unrepresentable and
//! compaction operating on `Turn`s cannot split a pair (§5).

pub use daemon_protocol::UserMsg;
use serde::{Deserialize, Serialize};

/// The system prompt that opens every model context.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemPrompt {
    /// The system instruction text.
    pub text: String,
}

impl SystemPrompt {
    /// A system prompt from text.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// One assistant message — text plus optional reasoning, no tool calls (those live in [`ToolTurn`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantMsg {
    /// The assistant's text.
    pub text: String,
    /// The assistant's reasoning (a separate channel; never rendered as output — §17.2).
    pub reasoning: Option<String>,
}

impl AssistantMsg {
    /// An assistant message carrying only text.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            reasoning: None,
        }
    }
}

/// A model tool invocation (the canonical decoded form).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Correlates the call with its result.
    pub call_id: String,
    /// The tool's stable name.
    pub name: String,
    /// The (canonical) argument payload.
    pub args: String,
}

/// The result slot for a [`ToolCall`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    /// Correlates back to the originating [`ToolCall`].
    pub call_id: String,
    /// Whether the tool succeeded.
    pub ok: bool,
    /// The textual result content.
    pub content: String,
}

/// A tool turn: an assistant tool-call message AND its result slots, together (§5). A `ToolCall`
/// cannot exist without its `ToolResult` slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolTurn {
    /// The assistant message that issued the calls.
    pub assistant: AssistantMsg,
    /// Each call paired with its result slot.
    pub calls: Vec<(ToolCall, ToolResult)>,
}

/// One conversational turn.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Turn {
    /// A user message.
    User(UserMsg),
    /// An assistant message with no tool calls.
    Assistant(AssistantMsg),
    /// An assistant tool-call message and its results.
    Tool(ToolTurn),
}

/// The typed conversation body — the engine's durable source of truth (§5).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conversation {
    /// The system prompt.
    pub system: SystemPrompt,
    /// The ordered turns.
    pub turns: Vec<Turn>,
}

impl Conversation {
    /// A fresh conversation with the given system prompt.
    pub fn new(system: SystemPrompt) -> Self {
        Self {
            system,
            turns: Vec::new(),
        }
    }

    /// Append a user message.
    pub fn push_user(&mut self, msg: UserMsg) {
        self.turns.push(Turn::User(msg));
    }

    /// Append an assistant message.
    pub fn push_assistant(&mut self, msg: AssistantMsg) {
        self.turns.push(Turn::Assistant(msg));
    }

    /// Append a tool turn.
    pub fn push_tool(&mut self, turn: ToolTurn) {
        self.turns.push(Turn::Tool(turn));
    }

    /// Number of turns recorded so far.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Whether the conversation has no turns.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Whether the conversation contains any completed tool turn (used to detect a resolved
    /// delegation when finalizing a resumed turn).
    pub fn has_tool_turn(&self) -> bool {
        self.turns.iter().any(|t| matches!(t, Turn::Tool(_)))
    }
}
