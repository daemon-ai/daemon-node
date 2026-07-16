// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `pipeline-stage` — the Phase-B two-stage pipeline acceptance toy (refactor §6 acceptance;
//! architecture §9 "SWARM pipeline stage" row, the net-buffer version — the tensor
//! `export`/`import` version is Phase C's gate).
//!
//! Two instances of THIS module exchange **opaque net buffers over credit-controlled streams**:
//!
//! - **Stage A (producer, role 0)**: `stream_open(peer)` → on `Completion(Ok(stream))` seals its
//!   deterministic chunks (`create_from`, the budgeted OUT path) and issues ALL the
//!   `stream_write`s back to back — deliberately exceeding the transport's credit window, so the
//!   surplus writes are HELD host-side and complete only as the consumer's reads replenish credit
//!   (§3.3 flow control, exercised structurally) → publishes `b"sent"` after the last write
//!   completion.
//! - **Stage B (consumer, role 1)**: `stream_accept()` → on `Completion(Ok(stream))` issues
//!   `stream_read`s; each completes with a buffer whose bytes it reads back (`read_into`) and
//!   accumulates; after the last chunk it seals the concatenation and `payload_put`s it,
//!   publishing the completion hash — the received content, committed content-addressed.
//!
//! Config (raw bytes): `[role, n_chunks, chunk_len, peer_id[32]]`.

use daemon_vhc_sdk_v2::{ModuleDecl, V2Module};

const EV_STOP: u64 = 4;
const EV_COMPLETION: u64 = 6;

struct Pipeline {
    role: u8,
    n_chunks: u8,
    chunk_len: u8,
    peer: [u8; 32],
}

/// Chunk `i`'s deterministic content.
fn chunk(i: u8, len: u8) -> Vec<u8> {
    (0..len)
        .map(|j| i.wrapping_mul(31).wrapping_add(j))
        .collect()
}

impl V2Module for Pipeline {
    fn decl() -> ModuleDecl {
        ModuleDecl {
            name: "pipeline-stage",
            version: env!("CARGO_PKG_VERSION"),
            abi_minor: 1,
            channels: vec![0],
            host_state_bytes: 1 << 16,
            host_scratch_bytes: 1 << 16,
            device_state_bytes: 0,
            device_scratch_bytes: 0,
        }
    }

    fn init(config: &[u8], _grants: &[u8]) -> Result<Self, u32> {
        if config.len() < 35 {
            return Err(16);
        }
        let mut peer = [0u8; 32];
        peer.copy_from_slice(&config[3..35]);
        Ok(Self {
            role: config[0],
            n_chunks: config[1].max(1),
            chunk_len: config[2].max(1),
            peer,
        })
    }

    fn run(&mut self) -> u32 {
        if self.role == 0 {
            produce(self)
        } else {
            consume(self)
        }
    }
}

daemon_vhc_sdk_v2::main!(Pipeline);

/// Decode a completion frame's `(op, ok, uint-or-unit payload)`.
fn completion_parts(ev: &daemon_vhc_sdk_v2::Event) -> (u64, bool, u64) {
    let op = ev.uint(1);
    let (mut ok, mut value) = (false, 0u64);
    if let Some(ciborium::value::Value::Array(result)) = ev.items.get(2) {
        ok = result
            .first()
            .and_then(|v| v.as_integer())
            .is_some_and(|n| i128::from(n) == 0);
        value = result
            .get(1)
            .and_then(|v| v.as_integer())
            .map(|n| u64::try_from(i128::from(n)).unwrap_or(0))
            .unwrap_or(0);
    }
    (op, ok, value)
}

/// Stage A: open → write everything (over-credit on purpose) → publish "sent".
fn produce(st: &Pipeline) -> u32 {
    let open_op = daemon_vhc_sdk_v2::stream_open(&st.peer);
    let mut buf: Vec<u8> = Vec::with_capacity(64);
    let mut writes_pending: Vec<u64> = Vec::new();
    let mut done = false;
    loop {
        let ev = daemon_vhc_sdk_v2::next_event(&mut buf);
        match ev.tag {
            EV_STOP => return 0,
            EV_COMPLETION => {
                let (op, ok, value) = completion_parts(&ev);
                if op == open_op {
                    if !ok {
                        daemon_vhc_sdk_v2::publish(0, b"open-failed");
                        continue;
                    }
                    let stream = value;
                    // Seal + write ALL chunks now: the surplus beyond the credit window is held
                    // host-side and completes as the consumer reads (§3.3).
                    for i in 0..st.n_chunks {
                        let bytes = chunk(i, st.chunk_len);
                        let buffer = daemon_vhc_sdk_v2::create_from(&bytes);
                        writes_pending.push(daemon_vhc_sdk_v2::stream_write(stream, buffer));
                        daemon_vhc_sdk_v2::buffer_release(buffer);
                    }
                } else if let Some(pos) = writes_pending.iter().position(|w| *w == op) {
                    if !ok {
                        daemon_vhc_sdk_v2::publish(0, b"write-failed");
                        continue;
                    }
                    writes_pending.remove(pos);
                    if writes_pending.is_empty() && !done {
                        done = true;
                        daemon_vhc_sdk_v2::publish(0, b"sent");
                    }
                }
            }
            _ => {}
        }
    }
}

/// Stage B: accept → read n chunks → seal the concatenation → payload_put → publish its hash.
fn consume(st: &Pipeline) -> u32 {
    let accept_op = daemon_vhc_sdk_v2::stream_accept();
    let mut buf: Vec<u8> = Vec::with_capacity(64);
    let mut stream = 0u64;
    let mut read_op = 0u64;
    let mut put_op = 0u64;
    let mut received: Vec<u8> = Vec::new();
    let mut chunks_read = 0u8;
    loop {
        let ev = daemon_vhc_sdk_v2::next_event(&mut buf);
        match ev.tag {
            EV_STOP => return 0,
            EV_COMPLETION => {
                let (op, ok, value) = completion_parts(&ev);
                if op == accept_op {
                    if !ok {
                        daemon_vhc_sdk_v2::publish(0, b"accept-failed");
                        continue;
                    }
                    stream = value;
                    read_op = daemon_vhc_sdk_v2::stream_read(stream);
                } else if op == read_op {
                    if !ok {
                        daemon_vhc_sdk_v2::publish(0, b"read-failed");
                        continue;
                    }
                    // The received chunk: an opaque net buffer — read it back and accumulate.
                    received.extend(daemon_vhc_sdk_v2::read_buffer(value));
                    daemon_vhc_sdk_v2::buffer_release(value);
                    chunks_read += 1;
                    if chunks_read < st.n_chunks {
                        read_op = daemon_vhc_sdk_v2::stream_read(stream);
                    } else {
                        // Commit the received content, content-addressed (the B1 seal walk).
                        let sealed = daemon_vhc_sdk_v2::create_from(&received);
                        put_op = daemon_vhc_sdk_v2::payload_put(sealed);
                        daemon_vhc_sdk_v2::buffer_release(sealed);
                    }
                } else if op == put_op {
                    // Publish the commitment hash of everything received.
                    let hash = match ev.items.get(2) {
                        Some(ciborium::value::Value::Array(result)) => match result.get(1) {
                            Some(ciborium::value::Value::Bytes(b)) => b.clone(),
                            _ => Vec::new(),
                        },
                        _ => Vec::new(),
                    };
                    daemon_vhc_sdk_v2::publish(0, &hash);
                }
            }
            _ => {}
        }
    }
}
