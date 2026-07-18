// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `pipeline-stage` — the two-stage pipeline toy, **graduated to tensor buffers** (Phase C,
//! refactor §7: "the pipeline toy graduates to tensor buffers here"; architecture §9 "SWARM
//! pipeline stage" row, tensor `export`/`import` version — the Phase-B net-buffer version is in
//! this crate's history).
//!
//! Two instances exchange **exported device tensors over credit-controlled streams**
//! (architecture §3.4: `compute.export(tensor) → Completion(BufferHandle)` →
//! `net.stream_write(stream, buffer)`; the receiver re-imports each buffer as a device tensor):
//!
//! - **Stage A (producer, role 0)**: `stream_open(peer)` → on `Ok(stream)` builds its
//!   deterministic chunk tensors (each crosses to the device via `compute@2::import`), exports
//!   ALL of them (device → sealed buffer), then issues ALL the `stream_write`s back to back —
//!   deliberately exceeding the transport's credit window, so the surplus writes are HELD
//!   host-side and complete only as the consumer's reads replenish credit (§3.3 flow control,
//!   the Phase-B pin, unchanged) → publishes `b"sent"` after the last write completion.
//! - **Stage B (consumer, role 1)**: `stream_accept()` → per received buffer: re-imports it as a
//!   device tensor (`compute@2::import`), **doubles it on-device** (`t * 2`), exports the result,
//!   accumulates the doubled `CBOR(TensorData)` bytes; after the last chunk seals the
//!   concatenation and `payload_put`s it, publishing the completion hash — device-transformed
//!   content, committed content-addressed.
//!
//! Config (raw bytes): `[role, n_chunks, chunk_len, peer_id[32]]` (`chunk_len` = f32 elements).

use daemon_vhc_sdk::{GuestModule, ModuleDecl};
use daemon_vhc_sdk_compute::{
    decode_tensor_data, export_tensor, import_buffer_as_tensor, tensor_from_floats,
};

const EV_STOP: u64 = 4;
const EV_COMPLETION: u64 = 6;

struct Pipeline {
    role: u8,
    n_chunks: u8,
    chunk_len: u8,
    peer: [u8; 32],
}

/// Chunk `i`'s deterministic content, as f32 elements (the Phase-B byte recipe, widened).
fn chunk_floats(i: u8, len: u8) -> Vec<f32> {
    (0..len)
        .map(|j| f32::from(i.wrapping_mul(31).wrapping_add(j)))
        .collect()
}

impl GuestModule for Pipeline {
    fn decl() -> ModuleDecl {
        ModuleDecl {
            name: "pipeline-stage",
            version: env!("CARGO_PKG_VERSION"),
            abi_minor: 2, // compute@2 imports force the Phase-C minor (ABI §1.3 step 5)
            channels: vec![0],
            host_state_bytes: 1 << 16,
            host_scratch_bytes: 1 << 16,
            device_state_bytes: 1 << 16,
            device_scratch_bytes: 1 << 16,
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

daemon_vhc_sdk::main!(Pipeline);

/// Decode a completion frame's `(op, ok, uint-or-unit payload)`.
fn completion_parts(ev: &daemon_vhc_sdk::Event) -> (u64, bool, u64) {
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

/// Stage A: open → export every chunk tensor → write ALL exported buffers (over-credit on
/// purpose) → publish "sent".
fn produce(st: &Pipeline) -> u32 {
    let open_op = daemon_vhc_sdk::stream_open(&st.peer);
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    let mut stream = 0u64;
    // export op → chunk index; exported buffers by chunk index (write order must be chunk order).
    let mut export_ops: Vec<u64> = Vec::new();
    let mut exported: Vec<Option<u64>> = vec![None; usize::from(st.n_chunks)];
    let mut writes_pending: Vec<u64> = Vec::new();
    let mut done = false;
    loop {
        let ev = daemon_vhc_sdk::next_event(&mut buf);
        match ev.tag {
            EV_STOP => return 0,
            EV_COMPLETION => {
                let (op, ok, value) = completion_parts(&ev);
                if op == open_op {
                    if !ok {
                        daemon_vhc_sdk::publish(0, b"open-failed");
                        continue;
                    }
                    stream = value;
                    // The tensor path (§3.4): each chunk crosses guest → device (import inside
                    // tensor_from_floats) and device → sealed buffer (export). The export
                    // completions carry the buffers the stream writes will send.
                    for i in 0..st.n_chunks {
                        let t = tensor_from_floats(
                            chunk_floats(i, st.chunk_len),
                            [usize::from(st.chunk_len)],
                        );
                        export_ops.push(export_tensor(t));
                    }
                } else if let Some(idx) = export_ops.iter().position(|e| *e == op) {
                    if !ok {
                        daemon_vhc_sdk::publish(0, b"export-failed");
                        continue;
                    }
                    exported[idx] = Some(value);
                    if exported.iter().all(Option::is_some) {
                        // ALL writes back to back: the surplus beyond the credit window is held
                        // host-side and completes as the consumer reads (§3.3 — the pin).
                        for b in exported.iter().flatten() {
                            writes_pending.push(daemon_vhc_sdk::stream_write(stream, *b));
                            daemon_vhc_sdk::buffer_release(*b);
                        }
                    }
                } else if let Some(pos) = writes_pending.iter().position(|w| *w == op) {
                    if !ok {
                        daemon_vhc_sdk::publish(0, b"write-failed");
                        continue;
                    }
                    writes_pending.remove(pos);
                    if writes_pending.is_empty() && !done {
                        done = true;
                        daemon_vhc_sdk::publish(0, b"sent");
                    }
                }
            }
            _ => {}
        }
    }
}

/// Stage B: accept → per chunk: read → re-import as a device tensor → double on-device →
/// export → accumulate the doubled bytes → seal + payload_put → publish its hash.
fn consume(st: &Pipeline) -> u32 {
    let accept_op = daemon_vhc_sdk::stream_accept();
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    let mut stream = 0u64;
    let mut read_op = 0u64;
    let mut export_op = 0u64;
    let mut put_op = 0u64;
    let mut received: Vec<u8> = Vec::new();
    let mut chunks_done = 0u8;
    loop {
        let ev = daemon_vhc_sdk::next_event(&mut buf);
        match ev.tag {
            EV_STOP => return 0,
            EV_COMPLETION => {
                let (op, ok, value) = completion_parts(&ev);
                if op == accept_op {
                    if !ok {
                        daemon_vhc_sdk::publish(0, b"accept-failed");
                        continue;
                    }
                    stream = value;
                    read_op = daemon_vhc_sdk::stream_read(stream);
                } else if op == read_op {
                    if !ok {
                        daemon_vhc_sdk::publish(0, b"read-failed");
                        continue;
                    }
                    // The received buffer IS an exported tensor: re-import it as a device
                    // tensor, transform on-device, and export the result (§3.4).
                    let data = decode_tensor_data(&daemon_vhc_sdk::read_buffer(value));
                    let t = import_buffer_as_tensor::<1>(value, &data);
                    daemon_vhc_sdk::buffer_release(value);
                    export_op = export_tensor(t.mul_scalar(2.0_f32));
                } else if op == export_op {
                    if !ok {
                        daemon_vhc_sdk::publish(0, b"re-export-failed");
                        continue;
                    }
                    received.extend(daemon_vhc_sdk::read_buffer(value));
                    daemon_vhc_sdk::buffer_release(value);
                    chunks_done += 1;
                    if chunks_done < st.n_chunks {
                        read_op = daemon_vhc_sdk::stream_read(stream);
                    } else {
                        // Commit the device-transformed content, content-addressed.
                        let sealed = daemon_vhc_sdk::create_from(&received);
                        put_op = daemon_vhc_sdk::payload_put(sealed);
                        daemon_vhc_sdk::buffer_release(sealed);
                    }
                } else if op == put_op {
                    // Publish the commitment hash of everything received-and-doubled.
                    let hash = match ev.items.get(2) {
                        Some(ciborium::value::Value::Array(result)) => match result.get(1) {
                            Some(ciborium::value::Value::Bytes(b)) => b.clone(),
                            _ => Vec::new(),
                        },
                        _ => Vec::new(),
                    };
                    daemon_vhc_sdk::publish(0, &hash);
                }
            }
            _ => {}
        }
    }
}
