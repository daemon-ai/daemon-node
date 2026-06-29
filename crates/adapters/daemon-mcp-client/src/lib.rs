// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-mcp-client` — an MCP client exposed through the engine's [`ToolProvider`] seam.
//!
//! This adapter is to an external **MCP server** what [`PyToolProvider`](daemon_pytool_client) is to
//! the Python tool worker: it owns a connection's lifecycle (lazy connect, reconnect-on-fault) and
//! returns one [`McpToolProxy`] (`impl `[`daemon_core::Tool`]) per remote tool, so MCP tools register
//! into the ordinary [`ToolRegistry`](daemon_core::ToolRegistry) and the engine never knows they are
//! remote. A proxy's `run()` issues a `call_tool` round-trip and maps the reply onto a
//! [`ToolOutcome`](daemon_core::ToolOutcome) — always **untrusted-fenced**, since MCP results are
//! external content outside the agent's trust boundary.
//!
//! Two transports are supported (the two rmcp client transports): a **stdio child process** (the
//! daemon spawns the server binary) and **streamable HTTP** (the daemon connects to a URL). The host
//! wires a `Vec<Arc<dyn ToolProvider>>` uniformly across Python and MCP sources.
//!
//! Tool names are namespaced `mcp__{server}__{tool}` (double-underscore, not `:` — provider tool-name
//! grammars are typically `^[a-zA-Z0-9_-]+$`, so a colon would be rejected by Anthropic/OpenAI). The
//! provider's diagnostic [`label`](ToolProvider::label) keeps the human-readable `mcp:{server}` form.

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use daemon_core::{Tool, ToolCall, ToolOutcome, ToolProvider, ToolProviderError, TurnCx};
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{Peer, ServiceExt};
use tokio::sync::Mutex as AsyncMutex;

/// How a [`McpServerConfig`] reaches its MCP server.
#[derive(Clone, Debug)]
pub enum McpTransport {
    /// Spawn a local server binary and speak MCP over its stdio (the daemon owns the child process).
    Stdio {
        /// The program to exec (e.g. `npx`, or a server binary).
        command: String,
        /// Arguments passed to the program.
        args: Vec<String>,
        /// Extra environment variables set on the child.
        env: Vec<(String, String)>,
    },
    /// Connect to a remote server over streamable HTTP (the daemon is a client of an existing URL).
    Http {
        /// The base MCP endpoint, e.g. `http://localhost:8000/mcp`.
        url: String,
    },
}

/// One MCP server the daemon connects to and surfaces tools from.
#[derive(Clone, Debug)]
pub struct McpServerConfig {
    /// A short, stable server name used for tool namespacing (`mcp__{name}__{tool}`) and diagnostics.
    pub name: String,
    /// How to reach the server.
    pub transport: McpTransport,
    /// Per-operation timeout (discovery / a tool call) before declaring a transport fault.
    pub op_timeout: Duration,
}

impl McpServerConfig {
    /// A config for a stdio server `name` spawning `command` with `args`.
    pub fn stdio(name: impl Into<String>, command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            name: name.into(),
            transport: McpTransport::Stdio {
                command: command.into(),
                args,
                env: Vec::new(),
            },
            op_timeout: Duration::from_secs(60),
        }
    }

    /// A config for an HTTP server `name` at `url`.
    pub fn http(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            transport: McpTransport::Http { url: url.into() },
            op_timeout: Duration::from_secs(60),
        }
    }
}

/// A classified MCP-client failure.
#[derive(Debug, thiserror::Error)]
pub enum McpClientError {
    /// The connection could not be established (spawn failure, handshake error, bad URL).
    #[error("mcp connect failed: {0}")]
    Connect(String),
    /// A request failed at the protocol/transport layer (the connection is torn down + retried next).
    #[error("mcp request failed: {0}")]
    Request(String),
    /// The op exceeded its timeout.
    #[error("mcp request timed out")]
    Timeout,
}

/// The MCP connection backing all proxies for one server. Holds at most one live
/// [`RunningService`]; a transport fault clears it so the next op reconnects (mirrors `PyToolHost`).
pub struct McpClient {
    config: McpServerConfig,
    conn: AsyncMutex<Option<RunningService<RoleClient, ()>>>,
}

impl McpClient {
    /// Build a (not-yet-connected) client for `config`.
    pub fn new(config: McpServerConfig) -> Self {
        Self {
            config,
            conn: AsyncMutex::new(None),
        }
    }

    /// Build a client over an already-established connection (the seam for tests and for callers
    /// that own the rmcp handshake themselves). The `config` is retained only for reconnect-on-fault;
    /// the supplied connection is used until it faults.
    pub fn from_connection(config: McpServerConfig, conn: RunningService<RoleClient, ()>) -> Self {
        Self {
            config,
            conn: AsyncMutex::new(Some(conn)),
        }
    }

    /// The server's namespacing/diagnostic name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Establish a fresh connection from the configured transport.
    async fn connect(&self) -> Result<RunningService<RoleClient, ()>, McpClientError> {
        match &self.config.transport {
            McpTransport::Stdio { command, args, env } => {
                let mut cmd = tokio::process::Command::new(command);
                cmd.args(args);
                for (k, v) in env {
                    cmd.env(k, v);
                }
                let transport = TokioChildProcess::new(cmd)
                    .map_err(|e| McpClientError::Connect(e.to_string()))?;
                ().serve(transport)
                    .await
                    .map_err(|e| McpClientError::Connect(e.to_string()))
            }
            McpTransport::Http { url } => {
                let transport = StreamableHttpClientTransport::from_uri(url.clone());
                ().serve(transport)
                    .await
                    .map_err(|e| McpClientError::Connect(e.to_string()))
            }
        }
    }

    /// A cloned [`Peer`] over the (lazily established) connection. The peer is a cheap, `Send + Sync`
    /// request sink, so concurrent calls share one connection.
    async fn peer(&self) -> Result<Peer<RoleClient>, McpClientError> {
        let mut guard = self.conn.lock().await;
        if guard.is_none() {
            *guard = Some(self.connect().await?);
        }
        Ok(guard
            .as_ref()
            .expect("connection just established")
            .peer()
            .clone())
    }

    /// Tear down the current connection so the next op reconnects.
    async fn reset(&self) {
        if let Some(svc) = self.conn.lock().await.take() {
            let _ = svc.cancel().await;
        }
    }

    /// List the server's tools (used by [`McpClientProvider::discover`]).
    pub async fn list_tools(&self) -> Result<Vec<rmcp::model::Tool>, McpClientError> {
        let peer = self.peer().await?;
        let fut = peer.list_tools(Default::default());
        match tokio::time::timeout(self.config.op_timeout, fut).await {
            Ok(Ok(res)) => Ok(res.tools),
            Ok(Err(e)) => {
                self.reset().await;
                Err(McpClientError::Request(e.to_string()))
            }
            Err(_) => {
                self.reset().await;
                Err(McpClientError::Timeout)
            }
        }
    }

    /// Invoke `remote_name` with the given JSON argument string.
    async fn call(
        &self,
        remote_name: &str,
        args_json: &str,
    ) -> Result<CallToolResult, McpClientError> {
        let arguments = match serde_json::from_str::<serde_json::Value>(args_json) {
            Ok(serde_json::Value::Object(map)) => Some(map),
            _ => None,
        };
        let peer = self.peer().await?;
        let mut param = CallToolRequestParams::new(remote_name.to_string());
        if let Some(args) = arguments {
            param = param.with_arguments(args);
        }
        let fut = peer.call_tool(param);
        match tokio::time::timeout(self.config.op_timeout, fut).await {
            Ok(Ok(res)) => Ok(res),
            Ok(Err(e)) => {
                self.reset().await;
                Err(McpClientError::Request(e.to_string()))
            }
            Err(_) => {
                self.reset().await;
                Err(McpClientError::Timeout)
            }
        }
    }
}

/// Render a [`CallToolResult`]'s content blocks into one text body. Text blocks are concatenated;
/// non-text blocks (image/audio/resource) are summarized so the model at least sees their presence.
fn render_content(result: &CallToolResult) -> String {
    let mut parts: Vec<String> = Vec::new();
    for c in &result.content {
        if let Some(text) = c.as_text() {
            parts.push(text.text.clone());
        } else if c.as_image().is_some() {
            parts.push("[mcp image content omitted]".to_string());
        } else if c.as_resource().is_some() {
            parts.push("[mcp resource content omitted]".to_string());
        } else {
            parts.push("[mcp non-text content omitted]".to_string());
        }
    }
    if parts.is_empty() {
        if let Some(sc) = &result.structured_content {
            return serde_json::to_string(sc).unwrap_or_default();
        }
    }
    parts.join("\n")
}

/// A [`daemon_core::Tool`] that forwards a call to an MCP server. Shares one [`McpClient`] with every
/// other proxy from the same server.
pub struct McpToolProxy {
    client: Arc<McpClient>,
    /// The engine-facing name: `mcp__{server}__{tool}`.
    full_name: String,
    /// The server-local tool name passed back over MCP.
    remote_name: String,
    /// The remote tool's JSON-Schema (serialized).
    schema: String,
}

impl McpToolProxy {
    /// Build a proxy for `tool` advertised by `client`.
    pub fn new(client: Arc<McpClient>, tool: &rmcp::model::Tool) -> Self {
        let remote_name = tool.name.to_string();
        let full_name = format!("mcp__{}__{}", client.name(), remote_name);
        let schema =
            serde_json::to_string(&*tool.input_schema).unwrap_or_else(|_| "{}".to_string());
        Self {
            client,
            full_name,
            remote_name,
            schema,
        }
    }
}

#[async_trait]
impl Tool for McpToolProxy {
    fn name(&self) -> &str {
        &self.full_name
    }

    fn schema(&self) -> &str {
        &self.schema
    }

    fn deferrable(&self) -> bool {
        // MCP tools are the dynamic breadth: hide them behind `tool_search` once the surface is large.
        true
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let call_id = call.call_id.clone();
        let call_fut = self.client.call(&self.remote_name, &call.args);
        tokio::select! {
            biased;
            _ = cx.cancel.cancelled() => {
                ToolOutcome::text(call_id, false, "mcp tool call cancelled")
            }
            reply = call_fut => match reply {
                Ok(result) => {
                    // MCP output is external/untrusted content: fence it (the §12 pipeline wraps it).
                    let ok = !result.is_error.unwrap_or(false);
                    ToolOutcome::untrusted_text(call_id, ok, render_content(&result))
                }
                Err(e) => ToolOutcome::text(call_id, false, format!("mcp tool error: {e}")),
            }
        }
    }
}

/// An MCP server surfaced as a [`daemon_core::ToolProvider`] — the shared discovery seam the host
/// uses for every dynamic tool source (Python today, MCP here). Wraps one [`McpClient`] (one
/// connection, reconnected lazily); [`discover`](ToolProvider::discover) returns an [`McpToolProxy`]
/// per advertised tool.
pub struct McpClientProvider {
    client: Arc<McpClient>,
    label: String,
}

impl McpClientProvider {
    /// Build a provider for `config`. The connection is established lazily on the first `discover`.
    pub fn new(config: McpServerConfig) -> Self {
        let label = format!("mcp:{}", config.name);
        Self {
            client: Arc::new(McpClient::new(config)),
            label,
        }
    }

    /// The shared client backing this provider's proxies.
    pub fn client(&self) -> Arc<McpClient> {
        self.client.clone()
    }
}

#[async_trait]
impl ToolProvider for McpClientProvider {
    fn label(&self) -> &str {
        &self.label
    }

    async fn discover(&self) -> Result<Vec<Arc<dyn Tool>>, ToolProviderError> {
        let tools = self.client.list_tools().await?;
        Ok(tools
            .iter()
            .map(|t| Arc::new(McpToolProxy::new(self.client.clone(), t)) as Arc<dyn Tool>)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;

    fn schema_obj() -> StdArc<rmcp::model::JsonObject> {
        let mut m = serde_json::Map::new();
        m.insert(
            "type".to_string(),
            serde_json::Value::String("object".to_string()),
        );
        StdArc::new(m)
    }

    #[test]
    fn proxy_namespaces_tool_name_and_serializes_schema() {
        let client = Arc::new(McpClient::new(McpServerConfig::stdio(
            "github",
            "true",
            Vec::new(),
        )));
        let tool = rmcp::model::Tool::new("create_issue", "make an issue", schema_obj());
        let proxy = McpToolProxy::new(client, &tool);
        // API-safe double-underscore namespacing, not `mcp:server:tool` (colons fail provider grammars).
        assert_eq!(proxy.name(), "mcp__github__create_issue");
        assert_eq!(proxy.remote_name, "create_issue");
        assert!(proxy.schema().contains("\"type\":\"object\""));
    }

    #[test]
    fn provider_label_is_human_readable_form() {
        let p = McpClientProvider::new(McpServerConfig::http("docs", "http://localhost:9/mcp"));
        assert_eq!(p.label(), "mcp:docs");
    }

    #[test]
    fn render_content_joins_text_blocks() {
        let result = CallToolResult::success(vec![
            rmcp::model::Content::text("alpha"),
            rmcp::model::Content::text("beta"),
        ]);
        assert_eq!(render_content(&result), "alpha\nbeta");
        assert_eq!(result.is_error, Some(false));
    }
}
