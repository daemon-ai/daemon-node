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
    let _ = writeln!(out, r#"{{"type":"system","subtype":"init","model":"mock"}}"#);
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
            // A user turn: gate it behind a permission request first.
            Some("user") => {
                let _ = writeln!(
                    out,
                    r#"{{"type":"control_request","request_id":"perm-1","request":{{"subtype":"can_use_tool","tool_name":"Bash"}}}}"#
                );
                let _ = out.flush();
            }
            // Permission granted: emit the assistant message and finish the turn.
            Some("control_response") => {
                let _ = writeln!(
                    out,
                    r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"stream-json agent reporting in"}}]}}}}"#
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
