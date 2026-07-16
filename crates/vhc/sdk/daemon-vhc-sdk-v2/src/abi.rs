// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The raw `vhc@2`/`net@2`/`sys@2` externs + the small safe wrappers (ABI §4.1/§6): the ABI-floor
//! tier of the SDK table (architecture §6) for the event loop, exactly as the v1 SDK's `abi.rs`
//! is for the frozen tensor vocabulary (which major-2 modules keep linking as the §2.5 bridge).
//!
//! wasm32-only: the native path virtualizes these worlds in Phase B's `vhc-sim`.

use ciborium::value::Value;

#[link(wasm_import_module = "vhc@2")]
extern "C" {
    #[link_name = "next_event"]
    fn abi_next_event(buf_ptr: u32, buf_cap: u32) -> u64;
    #[link_name = "read_back"]
    fn abi_read_back(src: u64, kind: u32, out_ptr: u32, out_cap: u32) -> u64;
    // -- minor 1 (Phase B, track B1): the buffer layer + cancellation (§3.4/§7.5) --------------
    #[link_name = "create_from"]
    fn abi_create_from(ptr: u32, len: u32) -> u64;
    #[link_name = "read_into"]
    fn abi_read_into(buffer: u64, offset: u64, out_ptr: u32, out_cap: u32) -> u64;
    #[link_name = "buffer_len"]
    fn abi_buffer_len(buffer: u64) -> u64;
    #[link_name = "buffer_release"]
    fn abi_buffer_release(buffer: u64);
    #[link_name = "cancel"]
    fn abi_cancel(op: u64) -> u32;
}

#[link(wasm_import_module = "net@2")]
extern "C" {
    #[link_name = "publish"]
    fn abi_publish(channel_id: u32, payload_ptr: u32, payload_len: u32) -> u64;
    // -- minor 1: content-addressed payloads by handle (§3.4) — complete via Event::Completion --
    #[link_name = "payload_put"]
    fn abi_payload_put(buffer: u64) -> u64;
    #[link_name = "payload_get"]
    fn abi_payload_get(hash_ptr: u32) -> u64;
    // -- minor 1: direct peer streams under credit flow control (§3.3/§3.4) ---------------------
    #[link_name = "stream_open"]
    fn abi_stream_open(peer_ptr: u32) -> u64;
    #[link_name = "stream_accept"]
    fn abi_stream_accept() -> u64;
    #[link_name = "stream_write"]
    fn abi_stream_write(stream: u64, buffer: u64) -> u64;
    #[link_name = "stream_read"]
    fn abi_stream_read(stream: u64) -> u64;
}

#[link(wasm_import_module = "sys@2")]
extern "C" {
    #[link_name = "set_timer"]
    fn abi_set_timer(delay_ms: u64) -> u64;
}

/// One decoded event: the §4.2 tag plus the positional fields (tag included at index 0).
pub struct Event {
    /// The leading event tag (`EV_TAG_*`).
    pub tag: u64,
    /// The full positional array, tag included.
    pub items: Vec<Value>,
}

impl Event {
    /// Positional field `i` as a u64 (0 when absent/mistyped — callers know their tag's shape).
    #[must_use]
    pub fn uint(&self, i: usize) -> u64 {
        self.items
            .get(i)
            .and_then(Value::as_integer)
            .map(|n| u64::try_from(i128::from(n)).unwrap_or(u64::MAX))
            .unwrap_or(0)
    }

    /// Positional field `i` as bytes (empty when absent/mistyped).
    #[must_use]
    pub fn bytes(&self, i: usize) -> Vec<u8> {
        match self.items.get(i) {
            Some(Value::Bytes(b)) => b.clone(),
            _ => Vec::new(),
        }
    }
}

/// Pull the next event, honoring the mandatory `NeedCapacity` retry (§4.1). Fails closed
/// (`unreachable` → `GuestPanic`) on an unknown status or a malformed frame (§5.2).
#[must_use]
pub fn next_event(buf: &mut Vec<u8>) -> Event {
    loop {
        // SAFETY: the span handed to the host is exactly `buf`'s live allocation.
        let packed = unsafe { abi_next_event(buf.as_mut_ptr() as u32, buf.capacity() as u32) };
        let (status, len) = (packed >> 32, (packed & 0xffff_ffff) as usize);
        match status {
            0 => {
                // SAFETY: the host wrote exactly `len` bytes (§4.1).
                unsafe { buf.set_len(len) };
                let v: Value = ciborium::from_reader(buf.as_slice())
                    .unwrap_or_else(|_| unreachable!("malformed event frame"));
                let Value::Array(items) = v else {
                    unreachable!("event frame is not a positional array")
                };
                let tag = items
                    .first()
                    .and_then(Value::as_integer)
                    .map(|n| u64::try_from(i128::from(n)).unwrap_or(u64::MAX))
                    .unwrap_or_else(|| unreachable!("missing event tag"));
                return Event { tag, items };
            }
            1 => buf.reserve(len),
            _ => unreachable!("unknown next_event status (fail closed, §5.2)"),
        }
    }
}

/// `read_back` a staged item that resolves to a CBOR uint — the bridge kinds (1 = batch handle,
/// 2 = update staging index, §6.4) — honoring the mandatory retry.
#[must_use]
pub fn read_back_uint(src: u64, kind: u32) -> u64 {
    let mut buf = vec![0u8; 16];
    loop {
        // SAFETY: `buf` is a live guest span for the call's duration.
        let packed = unsafe { abi_read_back(src, kind, buf.as_mut_ptr() as u32, buf.len() as u32) };
        let (status, len) = (packed >> 32, (packed & 0xffff_ffff) as usize);
        match status {
            0 => {
                let v: Value = ciborium::from_reader(&buf[..len])
                    .unwrap_or_else(|_| unreachable!("read_back uint cbor"));
                return v
                    .as_integer()
                    .map(|n| u64::try_from(i128::from(n)).unwrap_or(u64::MAX))
                    .unwrap_or_else(|| unreachable!("read_back uint"));
            }
            1 => buf.resize(len, 0),
            _ => unreachable!("unknown read_back status (fail closed, §5.2)"),
        }
    }
}

/// `read_back` a staged item's raw bytes (kind 0 verbatim; kind 3 state-section during
/// `da_migrate`, §6.6/§10.2) — honoring the mandatory retry.
#[must_use]
pub fn read_back_bytes(src: u64, kind: u32) -> Vec<u8> {
    let mut buf = vec![0u8; 256];
    loop {
        // SAFETY: `buf` is a live guest span for the call's duration.
        let packed = unsafe { abi_read_back(src, kind, buf.as_mut_ptr() as u32, buf.len() as u32) };
        let (status, len) = (packed >> 32, (packed & 0xffff_ffff) as usize);
        match status {
            0 => {
                buf.truncate(len);
                return buf;
            }
            1 => buf.resize(len, 0),
            _ => unreachable!("unknown read_back status (fail closed, §5.2)"),
        }
    }
}

/// Publish opaque payload bytes on `channel` (§6.2); returns the durable channel-scoped seq.
pub fn publish(channel: u32, payload: &[u8]) -> u64 {
    // SAFETY: `payload` is a live guest span for the call's duration.
    unsafe { abi_publish(channel, payload.as_ptr() as u32, payload.len() as u32) }
}

/// Arm a one-shot logical-clock timer (§6.3); returns the timer id.
pub fn set_timer(delay_ms: u64) -> u64 {
    // SAFETY: plain-value import.
    unsafe { abi_set_timer(delay_ms) }
}

// -- minor 1 (Phase B, track B1): buffers + the completion protocol (§3.4/§7.5) -------------------

/// Seal `bytes` into a host buffer (the budgeted linear-memory OUT path; sealed at creation).
/// Returns the kind-8 `BufferHandle`. Requires declaring abi minor ≥ 1.
pub fn create_from(bytes: &[u8]) -> u64 {
    // SAFETY: `bytes` is a live guest span for the call's duration.
    unsafe { abi_create_from(bytes.as_ptr() as u32, bytes.len() as u32) }
}

/// Read a sealed buffer back into guest memory in full (the budgeted linear-memory IN path,
/// charged against the per-slice readback allowance).
#[must_use]
pub fn read_buffer(buffer: u64) -> Vec<u8> {
    // SAFETY: plain-value import.
    let len = unsafe { abi_buffer_len(buffer) } as usize;
    let mut out = vec![0u8; len];
    if len > 0 {
        // SAFETY: `out` is a live guest span for the call's duration.
        let n = unsafe { abi_read_into(buffer, 0, out.as_mut_ptr() as u32, len as u32) };
        out.truncate(n as usize);
    }
    out
}

/// The sealed length of a buffer (deterministic bookkeeping).
#[must_use]
pub fn buffer_len(buffer: u64) -> u64 {
    // SAFETY: plain-value import.
    unsafe { abi_buffer_len(buffer) }
}

/// Release the guest's hold on a buffer (frees its quota; §3.4 ownership).
pub fn buffer_release(buffer: u64) {
    // SAFETY: plain-value import.
    unsafe { abi_buffer_release(buffer) };
}

/// Cancel an outstanding op (§7.5): `0` = accepted (its completion will report `Cancelled`),
/// `1` = already completed/cancelled or unknown.
pub fn cancel(op: u64) -> u32 {
    // SAFETY: plain-value import.
    unsafe { abi_cancel(op) }
}

/// Store a sealed buffer on the run's payload plane (§3.4). Returns the `OpId`; completes with
/// `Ok(hash)` — the content commitment computed host-side over exactly the sealed bytes.
pub fn payload_put(buffer: u64) -> u64 {
    // SAFETY: plain-value import.
    unsafe { abi_payload_put(buffer) }
}

/// Fetch content-addressed bytes (§3.4). Returns the `OpId`; completes with `Ok(BufferHandle)`
/// after host-side hash verification.
pub fn payload_get(hash: &[u8; 32]) -> u64 {
    // SAFETY: `hash` is a live 32-byte guest span for the call's duration.
    unsafe { abi_payload_get(hash.as_ptr() as u32) }
}

/// Open a direct stream to `peer` (§3.3). Returns the `OpId`; completes with `Ok(StreamHandle)`.
pub fn stream_open(peer: &[u8; 32]) -> u64 {
    // SAFETY: `peer` is a live 32-byte guest span for the call's duration.
    unsafe { abi_stream_open(peer.as_ptr() as u32) }
}

/// Stand an accept for an incoming stream (§3.3). Returns the `OpId`; completes with
/// `Ok(StreamHandle)` when a peer opens.
pub fn stream_accept() -> u64 {
    // SAFETY: plain-value import.
    unsafe { abi_stream_accept() }
}

/// Write a sealed buffer to a stream (§3.4): consumes writable credit; a write beyond the window
/// is held host-side and completes when the receiver's reads replenish credit — the completion IS
/// the credit signal. Returns the `OpId`; completes `Ok(())`.
pub fn stream_write(stream: u64, buffer: u64) -> u64 {
    // SAFETY: plain-value import.
    unsafe { abi_stream_write(stream, buffer) }
}

/// Read the next chunk from a stream (§3.4). Returns the `OpId`; completes with
/// `Ok(BufferHandle)` of the received opaque bytes.
pub fn stream_read(stream: u64) -> u64 {
    // SAFETY: plain-value import.
    unsafe { abi_stream_read(stream) }
}
