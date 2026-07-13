// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-cli` — the operator surface over the node's [`daemon_api`] interface.
//!
//! A thin client: every subcommand marshals one [`daemon_api::ApiRequest`] over the Unix-socket
//! transport ([`daemon_host::ApiClient`]) and renders the [`daemon_api::ApiResponse`]. It reaches
//! the *same* surface the in-process caller and the C FFI reach — only the transport differs.
//!
//! The command grammar lives in [`cli`], the per-surface handlers in [`cmd`], and the response
//! printers in [`render`]; `main` is just parse → connect → dispatch.

#![forbid(unsafe_code)]

use std::path::PathBuf;

use clap::Parser;
use daemon_host::ApiClient;

mod cli;
mod cmd;
mod render;

use cli::Cli;

fn default_socket() -> PathBuf {
    if let Some(p) = std::env::var_os("DAEMON_SOCKET_PATH") {
        return PathBuf::from(p);
    }
    let dir = std::env::var_os("TMPDIR").unwrap_or_else(|| "/tmp".into());
    PathBuf::from(dir).join("daemon-api.sock")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Panic-only crash reporting (component = cli). No minidump monitor (hence no re-exec), so
    // ordering vs. clap does not matter. A no-op unless a DSN + `DAEMON_CRASH_CONSENT=1` are set.
    let _crash = daemon_telemetry::init_panic_reporting("cli");
    let cli = Cli::parse();
    let socket = cli.socket.clone().unwrap_or_else(default_socket);
    let client = ApiClient::new(socket);
    cmd::dispatch(&client, cli.command).await
}
