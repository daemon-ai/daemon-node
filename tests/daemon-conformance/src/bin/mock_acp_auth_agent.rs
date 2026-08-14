// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! A minimal **auth-gated ACP** agent: `initialize` advertises one agent-handled auth method
//! (`mock-login`), and `authenticate` answers it — recording the ceremony as a side effect (a
//! file named by `MOCK_ACP_AUTH_MARK`, when set) exactly the way a real agent persists its own
//! login state outside the node's sight. Backs the wire-v47 `agent/*` auth conformance: the
//! register probe must capture the advertised method, `auth_begin`/`auth_step` must drive this
//! binary's `authenticate` through the ACP gateway, and only the node-owned success marker (not
//! this file) may feed the derived catalog verdict.
//!
//! With `MOCK_ACP_AUTH_GATE` set, `session/new` refuses with the structured ACP `auth_required`
//! error (-32000) — the runtime signal the A6 feedback loop classifies. When
//! `MOCK_ACP_AUTH_MARK` is ALSO set, the gate is stateful across processes exactly like a real
//! vendor login: `session/new` refuses only while the mark file is absent, and a successful
//! `authenticate` writes it — so the NEXT spawned process (sessions are fresh processes) starts
//! authenticated. That mark lives wherever the harness points it (e.g. `$HOME/.mock-acp/…` under
//! a sandboxed HOME) — agent-owned dotfile state, invisible to the node, which is precisely the
//! documented ACP-isolation gap the A9 e2e journey asserts around.
//!
//! For the A9 runtime journey the agent also behaves like a live session once unlocked:
//! `session/new` pushes an `available_commands_update` (the A10 pipeline), and `session/prompt`
//! echoes the prompt text back in one message chunk and ends the turn.

use agent_client_protocol::schema::v1::{
    AuthMethod, AuthMethodAgent, AuthenticateRequest, AuthenticateResponse, AvailableCommand,
    AvailableCommandInput, AvailableCommandsUpdate, ContentBlock, ContentChunk, InitializeRequest,
    InitializeResponse, NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse,
    SessionNotification, SessionUpdate, StopReason, TextContent, UnstructuredCommandInput,
};
use agent_client_protocol::{Agent, ConnectionTo, Responder, Result, Stdio};

/// The one advertised auth method id.
const METHOD_ID: &str = "mock-login";

/// The harness-chosen agent-side login marker path, when set.
fn mark_path() -> Option<String> {
    std::env::var("MOCK_ACP_AUTH_MARK")
        .ok()
        .filter(|mark| !mark.is_empty())
}

#[tokio::main]
async fn main() -> Result<()> {
    Agent
        .builder()
        .name("mock-acp-auth-agent")
        .on_receive_request(
            async move |init: InitializeRequest, responder: Responder<InitializeResponse>, _cx| {
                responder.respond(InitializeResponse::new(init.protocol_version).auth_methods(
                    vec![AuthMethod::Agent(AuthMethodAgent::new(
                        METHOD_ID,
                        "Mock login",
                    ))],
                ))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |req: AuthenticateRequest,
                        responder: Responder<AuthenticateResponse>,
                        _cx| {
                if req.method_id.0.as_ref() != METHOD_ID {
                    return Err(agent_client_protocol::Error::method_not_found());
                }
                // The agent-side login side effect (what a real agent writes under its own HOME):
                // observable by the harness, invisible to the node.
                if let Some(mark) = mark_path() {
                    // Test-harness side-effect marker at a harness-chosen path (a mock
                    // agent binary, not node code — ContainedRoot does not apply here).
                    #[allow(clippy::disallowed_methods)]
                    {
                        if let Some(parent) = std::path::Path::new(&mark).parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        let _ = std::fs::write(&mark, METHOD_ID);
                    }
                }
                responder.respond(AuthenticateResponse::new())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |_req: NewSessionRequest, responder: Responder<NewSessionResponse>, cx| {
                // The A6 runtime signal: an unauthenticated agent refuses session open with the
                // structured, machine-classifiable `auth_required` error. With a mark path
                // configured the refusal is stateful (a completed `authenticate` unlocks every
                // later process); without one the gate is absolute (the conformance shape).
                if std::env::var("MOCK_ACP_AUTH_GATE").is_ok() {
                    // A mock binary reading its own harness-owned marker; ContainedRoot governs
                    // node code, not test doubles.
                    #[allow(clippy::disallowed_methods)]
                    let unlocked =
                        mark_path().is_some_and(|mark| std::path::Path::new(&mark).exists());
                    if !unlocked {
                        return Err(agent_client_protocol::Error::auth_required());
                    }
                }
                responder.respond(NewSessionResponse::new("mock-session"))?;
                // A10: advertise one command post-open so the e2e journey can assert the
                // adapter -> sidecar -> completion-popover pipeline on a real GUI.
                cx.send_notification(SessionNotification::new(
                    "mock-session",
                    SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(vec![
                        AvailableCommand::new("ping", "Echo a ping").input(
                            AvailableCommandInput::Unstructured(UnstructuredCommandInput::new(
                                "text",
                            )),
                        ),
                    ])),
                ))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |prompt: PromptRequest,
                        responder: Responder<PromptResponse>,
                        cx: ConnectionTo<agent_client_protocol::Client>| {
                // Echo the prompt text back in one chunk and end the turn — enough for the e2e
                // journey to assert a round-trip (including a selected slash command's text).
                let text = prompt
                    .prompt
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text(t) => Some(t.text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                cx.send_notification(SessionNotification::new(
                    prompt.session_id.clone(),
                    SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                        TextContent::new(format!("echo:{text}")),
                    ))),
                ))?;
                responder.respond(PromptResponse::new(StopReason::EndTurn))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_to(Stdio::new())
        .await
}
