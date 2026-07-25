// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `test-migrate-old` — the Phase-E upgrade-transaction drill pair, OLD side (ABI §10.2/§10.3).
//!
//! A counter module: `da_init` seeds the counter from config byte 0; every delivered `Frame`
//! increments it and publishes the new value (8-byte LE u64) on the `control` channel. On
//! `Quiesce{Upgrade}` it performs the §10.2 producing protocol — `stage_state` the 8-byte counter,
//! author the state-manifest (one `"counter"` section, schema 1, consensus-canonical class),
//! `snapshot_state`, assert `Accepted` — and returns `QuiesceReady`. On `Stop` it returns `Ok`.
//!
//! Config byte 1 (nonzero) flips the module into its misbehaving twin: it IGNORES `Quiesce` —
//! no snapshot, no return — and keeps pulling. The drain-deadline drill drives this knob: the
//! host's §4.4/§11.3 forced interruption (`QuiesceDeadlineExceeded`) is the only way such a
//! module leaves the drain.
//!
//! Raw-ABI on purpose (no SDK link — the pin-stability rule the sibling `toy-averager` follows).

use std::alloc::{alloc, dealloc, Layout};

#[link(wasm_import_module = "vhc@2")]
extern "C" {
    #[link_name = "next_event"]
    fn abi_next_event(buf_ptr: u32, buf_cap: u32) -> u64;
    #[link_name = "stage_state"]
    fn abi_stage_state(ptr: u32, len: u32) -> u64;
    #[link_name = "snapshot_state"]
    fn abi_snapshot_state(manifest_ptr: u32, manifest_len: u32) -> u32;
}

#[link(wasm_import_module = "net@2")]
extern "C" {
    #[link_name = "publish"]
    fn abi_publish(channel_id: u32, payload_ptr: u32, payload_len: u32) -> u64;
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

fn text(s: &str) -> ciborium::value::Value {
    ciborium::value::Value::Text(s.into())
}

fn uint(v: u64) -> ciborium::value::Value {
    ciborium::value::Value::Integer(v.into())
}

// ---- required exports (ABI §2.1) -------------------------------------------------------------------

/// `(major << 16) | minor` — major 2, minor 0 (the Phase-A closed subset).
#[no_mangle]
pub extern "C" fn da_abi() -> u32 {
    2 << 16
}

/// The static-requirements manifest (ABI §2.3): the `control` channel, migratable.
#[no_mangle]
pub extern "C" fn da_manifest(_cfg_ptr: u32, _cfg_len: u32) -> u64 {
    let manifest = ciborium::value::Value::Map(vec![
        (text("name"), text("test-migrate-old")),
        (text("version"), text(env!("CARGO_PKG_VERSION"))),
        (text("sdk"), text("raw-abi")),
        (text("abi"), uint(u64::from(2u32 << 16))),
        (
            text("channels"),
            ciborium::value::Value::Array(vec![uint(0)]),
        ),
        (text("migratable"), ciborium::value::Value::Bool(true)),
    ]);
    let mut bytes = Vec::new();
    ciborium::into_writer(&manifest, &mut bytes).expect("manifest cbor");
    emit_cbor(&bytes)
}

/// The tiered memory claim (ABI §9.1) — tiny, honest constants.
#[no_mangle]
pub extern "C" fn da_claim(_c: u32, _cl: u32, _g: u32, _gl: u32) -> u64 {
    let tier = |device: u64, host: u64| {
        ciborium::value::Value::Map(vec![
            (text("device"), uint(device)),
            (text("host"), uint(host)),
        ])
    };
    let claim = ciborium::value::Value::Map(vec![
        // The wasm32 Rust `cdylib` linear-memory floor: this tier is what the host enforces as
        // the sandbox cap, and the toolchain floor (shadow stack + data + first heap pages) is
        // beneath any module state — measured at 4 MiB in
        // `daemon_vhc_sdk::module::WASM_LINEAR_MEMORY_FLOOR_BYTES`, restated here because this
        // guest hand-authors its claim.
        (text("hard_accountable"), tier(0, 4 << 20)),
        (text("declared_peak"), tier(0, 5 << 20)),
        (text("workspace"), tier(0, 1 << 16)),
        (
            text("under_pressure"),
            ciborium::value::Value::Array(vec![uint(0), uint(1)]),
        ),
    ]);
    let mut bytes = Vec::new();
    ciborium::into_writer(&claim, &mut bytes).expect("claim cbor");
    emit_cbor(&bytes)
}

// ---- module state -----------------------------------------------------------------------------------

static mut COUNTER: u64 = 0;
static mut IGNORE_QUIESCE: bool = false;

/// Initialize: the counter seeds from config byte 0 (0 when absent); config byte 1 (nonzero)
/// arms the ignore-quiesce misbehavior knob. No imports here (§6.6).
///
/// # Safety
/// Called exactly once by the host before `da_run`; `cfg_ptr` is a host-written span.
#[no_mangle]
pub unsafe extern "C" fn da_init(cfg_ptr: u32, cfg_len: u32, _g: u32, _gl: u32) -> u32 {
    let byte = |i: u32| -> u8 {
        if i < cfg_len {
            *((cfg_ptr + i) as *const u8)
        } else {
            0
        }
    };
    COUNTER = u64::from(byte(0));
    IGNORE_QUIESCE = byte(1) != 0;
    0
}

// ---- the loop (ABI §3.1) -------------------------------------------------------------------------------

const EV_FRAME: u64 = 0;
const EV_STOP: u64 = 4;
const EV_QUIESCE: u64 = 7;
const OUTCOME_OK: u32 = 0;
const OUTCOME_QUIESCE_READY: u32 = 2;

fn pull_event(buf: &mut Vec<u8>) -> Vec<u8> {
    loop {
        // SAFETY: the span handed to the host is exactly `buf`'s live allocation.
        let packed = unsafe { abi_next_event(buf.as_mut_ptr() as u32, buf.capacity() as u32) };
        let (status, len) = (packed >> 32, (packed & 0xffff_ffff) as usize);
        match status {
            0 => {
                // SAFETY: the host wrote exactly `len` bytes.
                unsafe { buf.set_len(len) };
                return buf.clone();
            }
            1 => buf.reserve(len),
            _ => unreachable!("unknown next_event status (fail closed, §5.2)"),
        }
    }
}

fn decode_tag(bytes: &[u8]) -> u64 {
    let v: ciborium::value::Value =
        ciborium::from_reader(bytes).unwrap_or_else(|_| unreachable!("malformed event frame"));
    let ciborium::value::Value::Array(items) = v else {
        unreachable!("event frame is not an array");
    };
    items
        .first()
        .and_then(|t| t.as_integer())
        .map(|i| u64::try_from(i128::from(i)).unwrap_or(u64::MAX))
        .unwrap_or_else(|| unreachable!("missing event tag"))
}

/// The §10.2 producing protocol: stage the counter section, author + submit the state-manifest.
fn snapshot_counter(counter: u64) -> u32 {
    let section = counter.to_le_bytes();
    // SAFETY: `section` is a live 8-byte stack span for the call's duration.
    let _staging_id = unsafe { abi_stage_state(section.as_ptr() as u32, section.len() as u32) };
    let manifest = ciborium::value::Value::Map(vec![
        (text("schema"), uint(1)),
        // The producing module hash: not knowable in-guest (a module cannot hash its own
        // bytes); zeroed here — the HOST verifies sections by content hash, and E1's typed
        // manifest work owns the field's final discipline.
        (text("module"), ciborium::value::Value::Bytes(vec![0u8; 32])),
        (
            text("sections"),
            ciborium::value::Value::Array(vec![ciborium::value::Value::Map(vec![
                (text("name"), text("counter")),
                (text("schema"), uint(1)),
                (
                    text("hash"),
                    ciborium::value::Value::Bytes(blake3::hash(&section).as_bytes().to_vec()),
                ),
                (text("size"), uint(section.len() as u64)),
                (text("class"), uint(0)), // consensus-canonical (architecture §5.3)
            ])]),
        ),
    ]);
    let mut manifest_bytes = Vec::new();
    ciborium::into_writer(&manifest, &mut manifest_bytes).expect("manifest cbor");
    // SAFETY: `manifest_bytes` is a live guest span for the call's duration.
    unsafe { abi_snapshot_state(manifest_bytes.as_ptr() as u32, manifest_bytes.len() as u32) }
}

/// The module main loop: count frames, publish the counter; snapshot + `QuiesceReady` on drain
/// (or, with the misbehavior knob armed, ignore the drain and keep pulling).
#[no_mangle]
pub extern "C" fn da_run() -> u32 {
    // SAFETY: wasm is single-threaded; the host calls da_run exactly once (ABI §3.1).
    let counter = unsafe { &mut *core::ptr::addr_of_mut!(COUNTER) };
    let ignore_quiesce = unsafe { *core::ptr::addr_of!(IGNORE_QUIESCE) };
    let mut buf: Vec<u8> = Vec::with_capacity(64);
    loop {
        let frame = pull_event(&mut buf);
        match decode_tag(&frame) {
            EV_FRAME => {
                *counter += 1;
                let payload = counter.to_le_bytes();
                // SAFETY: `payload` is a live 8-byte stack span for the call's duration.
                unsafe { abi_publish(0, payload.as_ptr() as u32, payload.len() as u32) };
            }
            EV_STOP => return OUTCOME_OK,
            EV_QUIESCE if ignore_quiesce => {
                // The misbehaving twin: no snapshot, no return — the host's drain deadline is
                // the only exit (§4.4/§11.3 forced interruption).
            }
            EV_QUIESCE => {
                // One successful submission before QuiesceReady (§10.2) — fail loud otherwise.
                let status = snapshot_counter(*counter);
                if status != 0 {
                    unreachable!("snapshot_state rejected the drill manifest");
                }
                return OUTCOME_QUIESCE_READY;
            }
            _ => {}
        }
    }
}
