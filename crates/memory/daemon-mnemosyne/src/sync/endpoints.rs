// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Sync protocol endpoints — port of `mnemosyne/core/sync_server.py`, **transport-free**.
//!
//! Python ships a stdlib HTTP listener; this crate deliberately does not. The daemon node owns
//! transport (AGENTS.md: the node decides), so the port keeps only the protocol decisions as
//! pure functions: [`route`] maps a parsed [`SyncRequest`] to `(status, JSON)` — auth check,
//! endpoint dispatch, error shapes — and a host that wants to *be* a sync target mounts it (or
//! the individual handlers) behind its own router/mux. The only socket code is a test fixture
//! (`sync::tests`) that drives these functions over real HTTP to exercise
//! [`SyncEngine::sync_with`] end-to-end. The wire shapes stay compatible with Python peers:
//!
//! - `POST /sync/pull` — page events from this bank's log (`{since, limit, device_id}`)
//! - `POST /sync/push` — accept and apply events (`{events: [...]}`)
//! - `GET  /sync/status` — sync statistics
//!
//! Divergences: success responses additionally carry `"status": "ok"` (Python's tool adapter
//! expected it while its own server never sent it — the Rust pair heals that mismatch, and
//! Python clients ignore the extra key); the unauthenticated-JWT branch is not ported (it
//! base64-decoded the payload without verifying the signature — Bearer `api_key` is the
//! supported auth); TLS termination is the host's job.

use super::SyncEngine;
use crate::engine::Engine;
use serde_json::{json, Value};

/// Handle `POST /sync/pull` (`_handle_pull`): `{since, limit (≤10000), device_id}` ->
/// [`SyncEngine::pull_changes`].
pub fn handle_pull(se: &SyncEngine<'_>, body: &Value) -> (u16, Value) {
    let since = body.get("since").and_then(|v| v.as_str());
    let limit = body
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(1000)
        .min(10_000) as usize;
    match se.pull_changes(since, limit) {
        Ok(mut result) => {
            result["status"] = json!("ok");
            (200, result)
        }
        Err(e) => (500, json!({"error": format!("Pull failed: {e}")})),
    }
}

/// Handle `POST /sync/push` (`_handle_push`): `{events: [...]}` ->
/// [`SyncEngine::push_changes`] + a fresh `next_cursor` for the caller to persist.
pub fn handle_push(se: &SyncEngine<'_>, body: &Value) -> (u16, Value) {
    let Some(events) = body.get("events").and_then(|v| v.as_array()) else {
        return (400, json!({"error": "'events' must be a list"}));
    };
    match se.push_changes(events) {
        Ok(mut result) => {
            result["status"] = json!("ok");
            result["next_cursor"] = json!(crate::util::now_iso());
            (200, result)
        }
        Err(e) => (500, json!({"error": format!("Push failed: {e}")})),
    }
}

/// Handle `GET /sync/status` (`_handle_status`).
pub fn handle_status(se: &SyncEngine<'_>) -> (u16, Value) {
    match se.get_status(None) {
        Ok(mut status) => {
            status["status"] = json!("ok");
            (200, status)
        }
        Err(e) => (500, json!({"error": format!("Status failed: {e}")})),
    }
}

/// One parsed sync request, transport already stripped away (`SyncHTTPHandler`'s view after
/// `_parse_path` + `_read_body`): `path` is query-less with the trailing slash normalized,
/// `bearer` is the token from an `Authorization: Bearer ...` header, and `body` is the parsed
/// JSON (`Value::Null` = unparseable, mapped to a 400).
pub struct SyncRequest {
    /// The HTTP method (`GET`/`POST`/`OPTIONS`).
    pub method: String,
    /// The normalized path (`/sync/pull`, `/sync/push`, `/sync/status`).
    pub path: String,
    /// The Bearer token, when the transport carried one.
    pub bearer: Option<String>,
    /// The parsed JSON body (`json!({})` when the request had none).
    pub body: Value,
}

impl SyncRequest {
    /// Normalize a raw request target into the matchable path: strip the query string and any
    /// trailing slash (`_parse_path`).
    pub fn normalize_path(target: &str) -> String {
        let path = target.split('?').next().unwrap_or("/");
        if path.len() > 1 {
            path.trim_end_matches('/').to_string()
        } else {
            path.to_string()
        }
    }
}

/// Route one sync request over `engine`'s bank: CORS preflight, Bearer auth (`_check_auth`,
/// `api_key` branch), then endpoint dispatch. Returns `(http_status, response_json)`;
/// a 204 carries `Value::Null` (no body).
pub fn route(
    engine: &Engine,
    device_id: Option<String>,
    api_key: Option<&str>,
    request: &SyncRequest,
) -> (u16, Value) {
    // CORS preflight (`do_OPTIONS`).
    if request.method == "OPTIONS" {
        return (204, Value::Null);
    }
    if let Some(expected) = api_key {
        if request.bearer.as_deref() != Some(expected) {
            return (401, json!({"error": "Invalid or missing API key"}));
        }
    }
    if request.body.is_null() {
        return (400, json!({"error": "Invalid JSON body"}));
    }
    let se = match SyncEngine::new(engine, device_id, None) {
        Ok(se) => se,
        Err(e) => {
            return (
                500,
                json!({"error": format!("Sync engine not initialized: {e}")}),
            )
        }
    };
    match (request.method.as_str(), request.path.as_str()) {
        ("POST", "/sync/pull") => handle_pull(&se, &request.body),
        ("POST", "/sync/push") => handle_push(&se, &request.body),
        ("GET", "/sync/status") => handle_status(&se),
        _ => (
            404,
            json!({"error": format!("Not found: {}", request.path)}),
        ),
    }
}
