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
}

#[link(wasm_import_module = "sys@2")]
extern "C" {
    #[link_name = "set_timer"]
    fn abi_set_timer(delay_ms: u64) -> u64;
    #[link_name = "emit_metric"]
    fn abi_emit_metric(name_ptr: u32, name_len: u32, value: f64);
    #[link_name = "rng_seed"]
    fn abi_rng_seed(out_ptr: u32) -> u32;
    #[link_name = "device_profile"]
    fn abi_device_profile(out_ptr: u32, out_cap: u32) -> u64;
    #[link_name = "hash"]
    fn abi_hash(in_ptr: u32, in_len: u32, out_ptr: u32) -> u32;
    #[link_name = "verify_sig"]
    fn abi_verify_sig(pk_ptr: u32, sig_ptr: u32, msg_ptr: u32, msg_len: u32) -> u32;
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

/// Emit an advisory metric (§6.5 — egress only, rate-limited host-side, never journaled).
pub fn emit_metric(name: &str, value: f64) {
    // SAFETY: `name` is a live guest span for the call's duration.
    unsafe { abi_emit_metric(name.as_ptr() as u32, name.len() as u32, value) }
}

/// The run-scoped deterministic RNG seed (architecture §3.2 "seeded randomness"): a pure
/// function of the execution identity, identical across trap-restarts of the same incarnation
/// and at replay (§2.7 dc class — never journaled, always re-derivable).
#[must_use]
pub fn rng_seed() -> [u8; 32] {
    let mut seed = [0u8; 32];
    // SAFETY: `seed` is a live 32-byte guest span for the call's duration.
    let status = unsafe { abi_rng_seed(seed.as_mut_ptr() as u32) };
    if status != 0 {
        unreachable!("unknown rng_seed status (fail closed, §5.2)");
    }
    seed
}

/// The `sys@2::hash` crypto acceleration: blake3-256 of `data` (the det-lane pattern — the
/// in-guest fallback is `daemon_vhc_proto::crypto::hash`, this is the host fast path; the two
/// are bit-identical by construction and gated in tier-1). Deterministic — never journaled.
#[must_use]
pub fn hash_accel(data: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    // SAFETY: `data`/`out` are live guest spans for the call's duration.
    let status = unsafe {
        abi_hash(
            data.as_ptr() as u32,
            data.len() as u32,
            out.as_mut_ptr() as u32,
        )
    };
    if status != 0 {
        unreachable!("unknown hash status (fail closed, §5.2)");
    }
    out
}

/// The `sys@2::verify_sig` crypto acceleration: ed25519 verify, returning the tri-state
/// `daemon_vhc_proto::crypto::VerifyOutcome` code (0 valid / 1 invalid / 2 malformed — unknown
/// codes fail closed). In-guest fallback: `daemon_vhc_proto::crypto::verify_sig`.
#[must_use]
pub fn verify_sig_accel(public_key: &[u8; 32], signature: &[u8; 64], message: &[u8]) -> u32 {
    // SAFETY: all three are live guest spans for the call's duration.
    let code = unsafe {
        abi_verify_sig(
            public_key.as_ptr() as u32,
            signature.as_ptr() as u32,
            message.as_ptr() as u32,
            message.len() as u32,
        )
    };
    if code > 2 {
        unreachable!("unknown verify_sig outcome (fail closed, §5.2)");
    }
    code
}

/// The device profile the probe measures (architecture §3.5 — what makes module autotune
/// possible), as canonical-CBOR bytes; honors the mandatory NeedCapacity retry. The host
/// journals every delivery (tag 15) — it is a nondeterministic input.
#[must_use]
pub fn device_profile() -> Vec<u8> {
    let mut buf = vec![0u8; 128];
    loop {
        // SAFETY: `buf` is a live guest span for the call's duration.
        let packed = unsafe { abi_device_profile(buf.as_mut_ptr() as u32, buf.len() as u32) };
        let (status, len) = (packed >> 32, (packed & 0xffff_ffff) as usize);
        match status {
            0 => {
                buf.truncate(len);
                return buf;
            }
            1 => buf.resize(len, 0),
            _ => unreachable!("unknown device_profile status (fail closed, §5.2)"),
        }
    }
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
