// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! A minimal **Claude-Code `stream-json`** foreign agent (newline-delimited JSON over stdio).
//!
//! It has no `daemon-core` (or `daemon-host`) dependency, yet a `daemon-host` `CodecSession`
//! driving the [`StreamJsonCodec`](daemon_host::StreamJsonCodec) over a line-framed cut presents it
//! up the tree as an ordinary `Engine` leaf. On a user turn it raises one permission
//! (`control_request`) and, once the host approves it (`control_response`), emits a canned assistant
//! message + a `result`, proving the line transport + codec map a real CLI dialect up like an engine
//! and round-trip a blocking permission request.

use std::io::{BufRead, Write};

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    // Session preamble (the codec ignores `system`/`init`).
    let _ = writeln!(
        out,
        r#"{{"type":"system","subtype":"init","model":"mock"}}"#
    );
    let _ = out.flush();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => break,
        };
        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(_) => continue,
        };
        match value.get("type").and_then(|t| t.as_str()) {
            // A user turn: gate it behind a permission request first. In MOCK_AGENT_AUTH_FAIL
            // mode the agent instead rejects the turn outright with the machine-readable frame a
            // vendor CLI emits when its credential is bad (`subtype` carries the code the A6
            // rejection classifier matches on) — no permission round-trip.
            //
            // MOCK_AGENT_REQUIRE_AUTH=<expected> (the A9 e2e knob, set in the registered recipe
            // env) turns the mock into a credential-checking vendor: the turn succeeds only when
            // the SPAWN env carries MOCK_AGENT_TOKEN equal to <expected> — i.e. only after the
            // node's flow engine stored the token AND spawn-time materialization injected it.
            // The authorized path answers directly (no permission round-trip): the e2e journey
            // proves auth plumbing, not the approval flow, which conformance covers elsewhere.
            Some("user") => {
                let required = std::env::var("MOCK_AGENT_REQUIRE_AUTH").ok();
                let rejected = std::env::var("MOCK_AGENT_AUTH_FAIL").is_ok()
                    || required.as_deref().is_some_and(|expected| {
                        std::env::var("MOCK_AGENT_TOKEN").ok().as_deref() != Some(expected)
                    });
                if rejected {
                    let _ = writeln!(
                        out,
                        r#"{{"type":"result","subtype":"error_auth","is_error":true,"result":"Not logged in"}}"#
                    );
                    let _ = out.flush();
                    continue;
                }
                if required.is_some() {
                    let _ = writeln!(
                        out,
                        r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"authorized: stream-json agent reporting in"}}]}}}}"#
                    );
                    let _ = writeln!(
                        out,
                        r#"{{"type":"result","subtype":"success","is_error":false,"result":"done"}}"#
                    );
                    let _ = out.flush();
                    continue;
                }
                let _ = writeln!(
                    out,
                    r#"{{"type":"control_request","request_id":"perm-1","request":{{"subtype":"can_use_tool","tool_name":"Bash"}}}}"#
                );
                let _ = out.flush();
            }
            // Permission granted: emit the assistant message and finish the turn. When the
            // harness sets MOCK_AGENT_KEY (the wire-v47 A4 materialization proof), the message
            // reports the value this PROCESS actually sees — so a test can assert the node
            // injected the credential into the spawn env, over the ordinary event stream.
            // MOCK_AGENT_ECHO_HOME likewise makes it report its HOME (the A4 isolated
            // state-home proof) and whether a daemon-ambient canary variable leaked through.
            Some("control_response") => {
                let mut text = String::from("stream-json agent reporting in");
                if let Ok(key) = std::env::var("MOCK_AGENT_KEY") {
                    text.push_str(&format!(" key={key}"));
                }
                if std::env::var("MOCK_AGENT_ECHO_HOME").is_ok() {
                    let home = std::env::var("HOME").unwrap_or_else(|_| "<unset>".into());
                    text.push_str(&format!(" home={home}"));
                    if std::env::var("MOCK_AGENT_CANARY").is_ok() {
                        text.push_str(" canary=leaked");
                    }
                }
                let _ = writeln!(
                    out,
                    r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"{text}"}}]}}}}"#
                );
                let _ = writeln!(
                    out,
                    r#"{{"type":"result","subtype":"success","is_error":false,"result":"done"}}"#
                );
                let _ = out.flush();
            }
            _ => {}
        }
    }
}
