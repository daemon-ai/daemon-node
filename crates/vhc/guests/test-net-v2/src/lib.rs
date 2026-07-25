// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `test-net-v2` — the B1 buffer + completion-protocol conformance guest (declares abi **2.1**).
//!
//! Exercises the whole minor-1 surface end to end, raw-ABI (no SDK), driven by its config:
//!
//! 1. **The guest-authored sealing shape**: `create_from(config payload)` (the budgeted OUT path,
//!    sealed at creation) → `payload_put(buffer)` → on `Completion(op, Ok(hash))` publish the
//!    hash — the commitment-authorship pattern that retires the plumbing-sealed interim.
//! 2. **The verified fetch**: `payload_get(hash)` → on `Completion(op, Ok(BufferHandle))` →
//!    `buffer_len` + `read_into` (the budgeted IN path) → verify the bytes round-tripped →
//!    publish `b"roundtrip-ok"` / `b"roundtrip-bad"` → `buffer_release` both buffers.
//! 3. **Cancellation**: `payload_get(unknown hash)` → `cancel(op)` (expect 0 = accepted) → on
//!    `Completion(op, Err(Cancelled))` publish `b"cancelled"`.
//! 4. **Failure as completion**: `payload_get(another unknown hash)`, which the harness answers
//!    `Failed(NetUnreachable)` → on `Completion(op, Err(code 1))` publish `b"unreachable"`.
//!
//! Then it keeps pulling until `Stop` → Outcome `Ok`. Unknown event tags fail closed (§5.2).

use std::alloc::{alloc, dealloc, Layout};

// ---- raw ABI imports (minor 0 + the B1 minor-1 surface) ------------------------------------------

#[link(wasm_import_module = "vhc@2")]
extern "C" {
    #[link_name = "next_event"]
    fn abi_next_event(buf_ptr: u32, buf_cap: u32) -> u64;
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
    #[link_name = "payload_put"]
    fn abi_payload_put(buffer: u64) -> u64;
    #[link_name = "payload_get"]
    fn abi_payload_get(hash_ptr: u32) -> u64;
}

// ---- guest allocator (ABI §2.4) -------------------------------------------------------------------

fn layout(size: u32, align: u32) -> Layout {
    Layout::from_size_align(size as usize, (align as usize).max(1)).expect("valid layout")
}

/// Host-requested guest buffer (config/grants spans, ABI §2.4).
#[no_mangle]
pub extern "C" fn da_alloc(size: u32, align: u32) -> u32 {
    if size == 0 {
        return 0;
    }
    // SAFETY: non-zero validated layout; the host pairs with `da_free` (ABI §2.4).
    unsafe { alloc(layout(size, align)) as u32 }
}

/// Paired release for a `da_alloc` span (ABI §2.4).
#[no_mangle]
pub extern "C" fn da_free(ptr: u32, size: u32, align: u32) {
    if ptr == 0 || size == 0 {
        return;
    }
    // SAFETY: matches a prior `da_alloc` triple (host obligation, ABI §2.4).
    unsafe { dealloc(ptr as *mut u8, layout(size, align)) };
}

fn emit_cbor(bytes: &[u8]) -> u64 {
    if bytes.is_empty() {
        return 0;
    }
    let ptr = da_alloc(bytes.len() as u32, 1);
    // SAFETY: fresh len-byte allocation; no overlap.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len()) };
    (u64::from(ptr) << 32) | bytes.len() as u64
}

// ---- required exports (ABI §2.1) -------------------------------------------------------------------

/// `(major << 16) | minor` — major 2, **minor 1**: this module consumes completions (ABI §1.1).
#[no_mangle]
pub extern "C" fn da_abi() -> u32 {
    (2 << 16) | 1
}

/// The static-requirements manifest (ABI §2.3).
#[no_mangle]
pub extern "C" fn da_manifest(_cfg_ptr: u32, _cfg_len: u32) -> u64 {
    let v = ciborium::value::Value::Map(vec![
        (
            ciborium::value::Value::Text("name".into()),
            ciborium::value::Value::Text("test-net-v2".into()),
        ),
        (
            ciborium::value::Value::Text("version".into()),
            ciborium::value::Value::Text(env!("CARGO_PKG_VERSION").into()),
        ),
        (
            ciborium::value::Value::Text("sdk".into()),
            ciborium::value::Value::Text("raw-abi".into()),
        ),
        (
            ciborium::value::Value::Text("abi".into()),
            ciborium::value::Value::Integer(da_abi().into()),
        ),
        (
            ciborium::value::Value::Text("channels".into()),
            ciborium::value::Value::Array(vec![ciborium::value::Value::Integer(0.into())]),
        ),
    ]);
    let mut bytes = Vec::new();
    ciborium::into_writer(&v, &mut bytes).expect("manifest cbor");
    emit_cbor(&bytes)
}

/// The tiered memory claim (ABI §9.1) — small honest constants.
#[no_mangle]
pub extern "C" fn da_claim(_c: u32, _cl: u32, _g: u32, _gl: u32) -> u64 {
    let tier = |device: u64, host: u64| {
        ciborium::value::Value::Map(vec![
            (
                ciborium::value::Value::Text("device".into()),
                ciborium::value::Value::Integer(device.into()),
            ),
            (
                ciborium::value::Value::Text("host".into()),
                ciborium::value::Value::Integer(host.into()),
            ),
        ])
    };
    let claim = ciborium::value::Value::Map(vec![
        (
            ciborium::value::Value::Text("hard_accountable".into()),
            // The wasm32 Rust `cdylib` linear-memory floor: this tier is what the host enforces
            // as the sandbox cap, and the toolchain floor (shadow stack + data + first heap
            // pages) is beneath any module state — measured at 4 MiB in
            // `daemon_vhc_sdk::module::WASM_LINEAR_MEMORY_FLOOR_BYTES`, restated here because
            // this guest hand-authors its claim.
            tier(0, 4 << 20),
        ),
        (
            ciborium::value::Value::Text("declared_peak".into()),
            tier(0, 5 << 20),
        ),
        (
            ciborium::value::Value::Text("workspace".into()),
            tier(0, 1 << 16),
        ),
        (
            ciborium::value::Value::Text("under_pressure".into()),
            ciborium::value::Value::Array(vec![
                ciborium::value::Value::Integer(0.into()),
                ciborium::value::Value::Integer(1.into()),
            ]),
        ),
    ]);
    let mut bytes = Vec::new();
    ciborium::into_writer(&claim, &mut bytes).expect("claim cbor");
    emit_cbor(&bytes)
}

// ---- module state -----------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    /// `payload_put` issued; awaiting its hash completion.
    AwaitPutHash,
    /// `payload_get(hash)` issued; awaiting its buffer completion.
    AwaitGetBuffer,
    /// `payload_get(unknown)` issued and cancelled; awaiting the Cancelled completion.
    AwaitCancelled,
    /// `payload_get(unknown-2)` issued; awaiting the harness's NetUnreachable completion.
    AwaitUnreachable,
    /// Everything proven; pulling until Stop.
    Done,
}

struct State {
    payload: Vec<u8>,
    sealed: u64,
    put_op: u64,
    get_op: u64,
    cancel_op: u64,
    fail_op: u64,
    put_hash: [u8; 32],
    phase: Phase,
}

static mut STATE: State = State {
    payload: Vec::new(),
    sealed: 0,
    put_op: 0,
    get_op: 0,
    cancel_op: 0,
    fail_op: 0,
    put_hash: [0u8; 32],
    phase: Phase::AwaitPutHash,
};

/// Initialize: the config bytes ARE the payload this module seals + round-trips (ABI §2.1).
///
/// # Safety
/// Called exactly once by the host before `da_run`; `cfg_ptr` is a host-written span.
#[no_mangle]
pub unsafe extern "C" fn da_init(cfg_ptr: u32, cfg_len: u32, _g: u32, _gl: u32) -> u32 {
    let payload = (0..cfg_len)
        .map(|i| *((cfg_ptr + i) as *const u8))
        .collect::<Vec<u8>>();
    if payload.is_empty() {
        return 16; // module-defined: an empty payload proves nothing
    }
    STATE = State {
        payload,
        sealed: 0,
        put_op: 0,
        get_op: 0,
        cancel_op: 0,
        fail_op: 0,
        put_hash: [0u8; 32],
        phase: Phase::AwaitPutHash,
    };
    0
}

// ---- the loop (ABI §3.1) -------------------------------------------------------------------------------

const EV_STOP: u64 = 4;
const EV_COMPLETION: u64 = 6;

fn publish(bytes: &[u8]) {
    // SAFETY: `bytes` is a live guest span for the call's duration.
    unsafe { abi_publish(0, bytes.as_ptr() as u32, bytes.len() as u32) };
}

fn pull_event(buf: &mut Vec<u8>) -> Vec<u8> {
    loop {
        // SAFETY: the span handed to the host is exactly `buf`'s live allocation.
        let packed = unsafe { abi_next_event(buf.as_mut_ptr() as u32, buf.capacity() as u32) };
        let (status, len) = (packed >> 32, (packed & 0xffff_ffff) as usize);
        match status {
            0 => {
                // SAFETY: the host wrote exactly `len` bytes (§4.1).
                unsafe { buf.set_len(len) };
                return buf.clone();
            }
            1 => buf.reserve(len),
            _ => unreachable!("unknown next_event status (fail closed, §5.2)"),
        }
    }
}

/// Decode a completion frame `[6, op, [variant, payload]]` → `(op, variant, payload)`.
fn decode_completion(
    items: &[ciborium::value::Value],
) -> (u64, u64, Option<ciborium::value::Value>) {
    let uint = |v: &ciborium::value::Value| -> u64 {
        v.as_integer()
            .map(|n| u64::try_from(i128::from(n)).unwrap_or(u64::MAX))
            .unwrap_or(u64::MAX)
    };
    let op = items.get(1).map(&uint).unwrap_or(u64::MAX);
    let Some(ciborium::value::Value::Array(result)) = items.get(2) else {
        unreachable!("completion result is [variant, payload]");
    };
    let variant = result.first().map(&uint).unwrap_or(u64::MAX);
    (op, variant, result.get(1).cloned())
}

/// The module main loop: the four-phase completion-protocol proof (module docs).
#[no_mangle]
pub extern "C" fn da_run() -> u32 {
    // SAFETY: wasm is single-threaded; the host calls da_run exactly once (ABI §3.1).
    let st = unsafe { &mut *core::ptr::addr_of_mut!(STATE) };

    // Phase 1 opening move: seal the payload (budgeted OUT path) and put it.
    st.sealed = unsafe { abi_create_from(st.payload.as_ptr() as u32, st.payload.len() as u32) };
    st.put_op = unsafe { abi_payload_put(st.sealed) };

    let mut buf: Vec<u8> = Vec::with_capacity(16); // deliberately small: exercises NeedCapacity
    loop {
        let frame = pull_event(&mut buf);
        let v: ciborium::value::Value =
            ciborium::from_reader(frame.as_slice()).unwrap_or_else(|_| unreachable!("frame cbor"));
        let ciborium::value::Value::Array(items) = v else {
            unreachable!("event frame is not an array");
        };
        let tag = items
            .first()
            .and_then(|t| t.as_integer())
            .map(|i| u64::try_from(i128::from(i)).unwrap_or(u64::MAX))
            .unwrap_or_else(|| unreachable!("missing event tag"));
        match tag {
            EV_STOP => return 0,
            EV_COMPLETION => {
                let (op, variant, payload) = decode_completion(&items);
                match st.phase {
                    Phase::AwaitPutHash if op == st.put_op && variant == 0 => {
                        // Ok(hash): the guest-authored commitment — publish the hash it was told.
                        let Some(ciborium::value::Value::Bytes(hash)) = payload else {
                            unreachable!("put completion carries a 32-byte hash");
                        };
                        st.put_hash.copy_from_slice(&hash);
                        publish(&hash);
                        // Phase 2: fetch the content back by hash.
                        st.get_op = unsafe { abi_payload_get(st.put_hash.as_ptr() as u32) };
                        st.phase = Phase::AwaitGetBuffer;
                    }
                    Phase::AwaitGetBuffer if op == st.get_op && variant == 0 => {
                        // Ok(BufferHandle): read it back through the budgeted IN path + verify.
                        let Some(pv) = payload else {
                            unreachable!("get completion carries a handle");
                        };
                        let handle = pv
                            .as_integer()
                            .map(|n| u64::try_from(i128::from(n)).unwrap_or(0))
                            .unwrap_or(0);
                        let len = unsafe { abi_buffer_len(handle) } as usize;
                        let mut got = vec![0u8; len];
                        let n = unsafe {
                            abi_read_into(handle, 0, got.as_mut_ptr() as u32, len as u32)
                        };
                        let ok = n as usize == len && got == st.payload;
                        publish(if ok {
                            b"roundtrip-ok"
                        } else {
                            b"roundtrip-bad"
                        });
                        unsafe {
                            abi_buffer_release(handle);
                            abi_buffer_release(st.sealed);
                        }
                        // Phase 3: cancel a fetch of content nobody has.
                        let unknown = [0xEEu8; 32];
                        st.cancel_op = unsafe { abi_payload_get(unknown.as_ptr() as u32) };
                        let status = unsafe { abi_cancel(st.cancel_op) };
                        if status != 0 {
                            publish(b"cancel-not-accepted");
                        }
                        st.phase = Phase::AwaitCancelled;
                    }
                    Phase::AwaitCancelled if op == st.cancel_op => {
                        // Err(Cancelled): variant 1, comp-error code 0.
                        let cancelled = variant == 1
                            && matches!(
                                &payload,
                                Some(ciborium::value::Value::Map(m))
                                    if m.iter().any(|(k, val)| {
                                        matches!(k, ciborium::value::Value::Text(t) if t == "code")
                                            && val.as_integer().is_some_and(|n| i128::from(n) == 0)
                                    })
                            );
                        publish(if cancelled {
                            b"cancelled"
                        } else {
                            b"cancel-wrong-result"
                        });
                        // Phase 4: a fetch the harness fails with NetUnreachable (code 1).
                        let unknown2 = [0xDDu8; 32];
                        st.fail_op = unsafe { abi_payload_get(unknown2.as_ptr() as u32) };
                        st.phase = Phase::AwaitUnreachable;
                    }
                    Phase::AwaitUnreachable if op == st.fail_op => {
                        let unreachable_err = variant == 1
                            && matches!(
                                &payload,
                                Some(ciborium::value::Value::Map(m))
                                    if m.iter().any(|(k, val)| {
                                        matches!(k, ciborium::value::Value::Text(t) if t == "code")
                                            && val.as_integer().is_some_and(|n| i128::from(n) == 1)
                                    })
                            );
                        publish(if unreachable_err {
                            b"unreachable"
                        } else {
                            b"fail-wrong-result"
                        });
                        st.phase = Phase::Done;
                    }
                    _ => publish(b"unexpected-completion"),
                }
            }
            // Frame / PayloadReady / Timer / Budget / Quiesce: ignored (module policy; only
            // unknown TAGS fail closed, §5.2).
            _ => {}
        }
    }
}
