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
//!
//! For the Layer-1 model-selection conformance it also advertises a `Model`-category `Select`
//! config option (two values) on `session/new`, and answers `session/set_config_option` — recording
//! the received `configId`/`value` in-process and streaming it back as a second message chunk of the
//! following turn (`set:<configId>=<value>`, or `unset` when none was received). A test can then
//! observe that message text to assert exactly which selection the adapter sent (or that it sent
//! none) without any side channel.

use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v1::{
    AgentCapabilities, ContentBlock, ContentChunk, InitializeRequest, InitializeResponse,
    NewSessionRequest, NewSessionResponse, PermissionOption, PermissionOptionKind, PromptRequest,
    PromptResponse, RequestPermissionRequest, SessionConfigOption, SessionConfigOptionCategory,
    SessionConfigSelectOption, SessionConfigValueId, SessionNotification, SessionUpdate,
    SetSessionConfigOptionRequest, SetSessionConfigOptionResponse, StopReason, TextContent,
    ToolCallUpdate, ToolCallUpdateFields,
};
use agent_client_protocol::{Agent, ConnectionTo, Responder, Result, Stdio};

/// The advertised Model selector's config-option id (what `set_config_option` targets).
const MODEL_CONFIG_ID: &str = "model";
/// The first (default / current) advertised model value id.
const MODEL_A: &str = "mock-model-a";
/// The second advertised model value id (a switch target that differs from the current value).
const MODEL_B: &str = "mock-model-b";

/// The `Model`-category `Select` config option the mock advertises, with `current` selected.
fn model_option(current: impl Into<SessionConfigValueId>) -> SessionConfigOption {
    SessionConfigOption::select(
        MODEL_CONFIG_ID,
        "Model",
        current,
        vec![
            SessionConfigSelectOption::new(MODEL_A, "Mock Model A"),
            SessionConfigSelectOption::new(MODEL_B, "Mock Model B"),
        ],
    )
    .category(Some(SessionConfigOptionCategory::Model))
}

#[tokio::main]
async fn main() -> Result<()> {
    // The last `session/set_config_option` the agent received (`Some((config_id, value))`), shared
    // between the set-config handler that records it and the prompt handler that streams it back.
    let last_set: Arc<Mutex<Option<(String, String)>>> = Arc::new(Mutex::new(None));
    let last_set_config = last_set.clone();
    let last_set_prompt = last_set.clone();

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
                responder.respond(
                    NewSessionResponse::new("mock-acp-session")
                        .config_options(vec![model_option(MODEL_A)]),
                )
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |req: SetSessionConfigOptionRequest,
                        responder: Responder<SetSessionConfigOptionResponse>,
                        _cx| {
                // Record the received selection so the next turn can stream it back for the test.
                *last_set_config.lock().unwrap() =
                    Some((req.config_id.0.to_string(), req.value.0.to_string()));
                // Reflect the new current value back in the option set.
                responder.respond(SetSessionConfigOptionResponse::new(vec![model_option(
                    req.value.clone(),
                )]))
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
                let last = last_set_prompt.clone();
                cx.spawn(async move {
                    let sid = prompt.session_id.clone();
                    cx2.send_notification(SessionNotification::new(
                        sid.clone(),
                        SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                            TextContent::new("acp agent reporting in"),
                        ))),
                    ))?;
                    // A second chunk reports the model selection the adapter applied (if any), so a
                    // test can assert the exact `set_config_option` over the ordinary event stream.
                    let observed = match last.lock().unwrap().clone() {
                        Some((config_id, value)) => format!("set:{config_id}={value}"),
                        None => "unset".to_string(),
                    };
                    cx2.send_notification(SessionNotification::new(
                        sid.clone(),
                        SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                            TextContent::new(observed),
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
