// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Hermetic MCP client<->server roundtrip conformance.
//!
//! Stands up a minimal in-process MCP **server** (rmcp `server` feature, dev-only) over a duplex
//! transport, points an [`McpClient`] at the client half, and asserts the full client path the daemon
//! relies on: **discover** (`list_tools`) -> **register** an [`McpToolProxy`] into a real
//! [`ToolRegistry`] -> **run_tool** through the engine pipeline -> a fenced, untrusted [`ToolOutcome`].
//! Using a real rmcp server (not a hand-rolled JSON mock) is what makes this a protocol conformance
//! test rather than a unit test of our own serialization.

use std::sync::Arc;

use daemon_common::{Budget, SessionId};
use daemon_core::events::EventSink;
use daemon_core::exec::LocalEnvironment;
use daemon_core::{run_tool, ApprovalPolicy, ToolCall, ToolRegistry, TurnCx};
use daemon_mcp_client::{McpClient, McpServerConfig, McpToolProxy};
use daemon_protocol::{HostRequest, HostRequestHandler, HostResponse, HostResponseBody};

use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, ListToolsResult, PaginatedRequestParams,
    ServerInfo,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt};

/// A one-tool MCP server: advertises `echo` and replies with `echo: {message}`.
#[derive(Clone)]
struct EchoServer;

impl ServerHandler for EchoServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::default()
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _cx: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let mut schema = serde_json::Map::new();
        schema.insert(
            "type".to_string(),
            serde_json::Value::String("object".to_string()),
        );
        let mut props = serde_json::Map::new();
        let mut msg = serde_json::Map::new();
        msg.insert(
            "type".to_string(),
            serde_json::Value::String("string".to_string()),
        );
        props.insert("message".to_string(), serde_json::Value::Object(msg));
        schema.insert("properties".to_string(), serde_json::Value::Object(props));
        let tool = rmcp::model::Tool::new("echo", "echo back the message", Arc::new(schema));
        Ok(ListToolsResult::with_all_items(vec![tool]))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _cx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let message = request
            .arguments
            .as_ref()
            .and_then(|a| a.get("message"))
            .and_then(|v| v.as_str())
            .unwrap_or("(none)");
        Ok(CallToolResult::success(vec![Content::text(format!(
            "echo: {message}"
        ))]))
    }
}

/// A host that auto-approves anything the pipeline asks about (no HITL in this test).
struct AutoApproveHost;

#[async_trait::async_trait]
impl HostRequestHandler for AutoApproveHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
        HostResponse {
            request_id: req.request_id,
            body: HostResponseBody::Approved {
                approved: true,
                allow_permanent: false,
                reason: None,
            },
        }
    }
}

#[tokio::test]
async fn mcp_discover_register_run_roundtrip() {
    // A bidirectional in-process pipe: one end serves the MCP server, the other the client.
    let (server_io, client_io) = tokio::io::duplex(64 * 1024);

    let server = tokio::spawn(async move {
        let running = EchoServer.serve(server_io).await.expect("server handshake");
        running.waiting().await.expect("server run");
    });

    // Client half: complete the rmcp handshake, then wrap in our McpClient over the live connection.
    let connection = ().serve(client_io).await.expect("client handshake");
    let client = Arc::new(McpClient::from_connection(
        McpServerConfig::stdio("echo-srv", "true", Vec::new()),
        connection,
    ));

    // Discover: the real MCP `tools/list`.
    let tools = client.list_tools().await.expect("list_tools");
    assert_eq!(tools.len(), 1, "server advertises exactly one tool");
    assert_eq!(tools[0].name.as_ref(), "echo");

    // Register: build a proxy and put it in a registry exactly as the host would.
    let mut registry = ToolRegistry::new();
    registry.register_deferrable(Arc::new(McpToolProxy::new(client.clone(), &tools[0])));
    assert!(
        registry.get("mcp__echo-srv__echo").is_some(),
        "namespaced MCP tool is registered"
    );

    // Run: drive the namespaced tool through the engine pipeline (the path call_model would take).
    let events = EventSink::discarding();
    let exec = LocalEnvironment::sandbox("mcp-roundtrip");
    let cx = TurnCx {
        cancel: tokio_util::sync::CancellationToken::new(),
        events: &events,
        host: &AutoApproveHost,
        session_id: SessionId::new("s-mcp"),
        profile: None,
        budget: Budget::unlimited(),
        exec: &exec,
        tool_result_budget: 0,
        approval_policy: ApprovalPolicy::AutoAllow,
        pre_approved: false,
        checkpoints: None,
        tool_timeout: None,
        session_allow: &[],
    };
    let call = ToolCall {
        call_id: "c-echo".into(),
        name: "mcp__echo-srv__echo".into(),
        args: r#"{"message":"hello daemon"}"#.into(),
    };
    let outcome = run_tool(&call, &registry, &cx).await;

    assert!(outcome.result.ok, "echo call succeeded end-to-end");
    assert!(
        outcome.result.content.contains("echo: hello daemon"),
        "round-tripped the server's reply: {:?}",
        outcome.result.content
    );
    // MCP output is external content: the pipeline must have fenced it as untrusted.
    assert!(
        outcome.result.content.contains("UNTRUSTED_TOOL_OUTPUT"),
        "external MCP output is untrusted-fenced: {:?}",
        outcome.result.content
    );

    drop(cx);
    drop(client);
    server.abort();
}
