// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! A minimal **ACP** agent (JSON-RPC 2.0 over stdio), built on the published
//! `agent-client-protocol` crate's agent side.
//!
//! It answers `initialize` / `session/new`, and on `session/prompt` streams one
//! `agent_message_chunk`, raises one `session/request_permission` back to the client, and (once
//! answered) finishes the turn with `EndTurn`. This proves the `daemon-acp` adapter drives a real
//! ACP agent up the tree as an ordinary `Engine` leaf and round-trips the symmetric permission
//! callback. The prompt handler defers its work onto a spawned task (per the crate's ordering rules)
//! so blocking on the permission round-trip never stalls the dispatch loop.

use agent_client_protocol::schema::v1::{
    AgentCapabilities, ContentBlock, ContentChunk, InitializeRequest, InitializeResponse,
    NewSessionRequest, NewSessionResponse, PermissionOption, PermissionOptionKind, PromptRequest,
    PromptResponse, RequestPermissionRequest, SessionNotification, SessionUpdate, StopReason,
    TextContent, ToolCallUpdate, ToolCallUpdateFields,
};
use agent_client_protocol::{Agent, ConnectionTo, Responder, Result, Stdio};

#[tokio::main]
async fn main() -> Result<()> {
    Agent
        .builder()
        .name("mock-acp-agent")
        .on_receive_request(
            async move |init: InitializeRequest, responder: Responder<InitializeResponse>, _cx| {
                responder.respond(
                    InitializeResponse::new(init.protocol_version)
                        .agent_capabilities(AgentCapabilities::new()),
                )
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |_req: NewSessionRequest, responder: Responder<NewSessionResponse>, _cx| {
                responder.respond(NewSessionResponse::new("mock-acp-session"))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |prompt: PromptRequest,
                        responder: Responder<PromptResponse>,
                        cx: ConnectionTo<agent_client_protocol::Client>| {
                // Defer onto a spawned task: blocking on the permission round-trip inside the
                // dispatch-loop handler would deadlock (the response can't be processed).
                let cx2 = cx.clone();
                cx.spawn(async move {
                    let sid = prompt.session_id.clone();
                    cx2.send_notification(SessionNotification::new(
                        sid.clone(),
                        SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                            TextContent::new("acp agent reporting in"),
                        ))),
                    ))?;
                    let _ = cx2
                        .send_request(RequestPermissionRequest::new(
                            sid.clone(),
                            ToolCallUpdate::new("tool-1", ToolCallUpdateFields::new()),
                            vec![
                                PermissionOption::new(
                                    "allow",
                                    "Allow",
                                    PermissionOptionKind::AllowOnce,
                                ),
                                PermissionOption::new(
                                    "reject",
                                    "Reject",
                                    PermissionOptionKind::RejectOnce,
                                ),
                            ],
                        ))
                        .block_task()
                        .await?;
                    responder.respond(PromptResponse::new(StopReason::EndTurn))?;
                    Ok(())
                })?;
                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_to(Stdio::new())
        .await
}
