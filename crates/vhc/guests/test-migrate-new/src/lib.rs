// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `test-migrate-new` — the Phase-E upgrade-transaction drill pair, NEW side (ABI §10.2/§10.3).
//!
//! `da_migrate` decodes the host-built migration descriptor, restores the `"counter"` section
//! through the explicitly granted restore capability (`read_back(staging_id, kind = 3)` — legal
//! exactly during `da_migrate`, §6.6), validates it, and returns `Ready`. `da_run` then announces
//! the restored counter by publishing it immediately (the drill's migrate-succeeded signal) and
//! resumes counting: every delivered `Frame` increments and publishes — continuity across the
//! epoch fence, without a process restart.
//!
//! Raw-ABI on purpose (no SDK link — the pin-stability rule the sibling guests follow).

use std::alloc::{alloc, dealloc, Layout};

#[link(wasm_import_module = "vhc@2")]
extern "C" {
    #[link_name = "next_event"]
    fn abi_next_event(buf_ptr: u32, buf_cap: u32) -> u64;
    #[link_name = "read_back"]
    fn abi_read_back(src: u64, kind: u32, out_ptr: u32, out_cap: u32) -> u64;
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

/// Host-requested guest buffer (config/grants/descriptor spans, ABI §2.4).
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
        (text("name"), text("test-migrate-new")),
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
        (text("hard_accountable"), tier(0, 1 << 16)),
        (text("declared_peak"), tier(0, 1 << 20)),
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
static mut MIGRATED: bool = false;

/// Initialize: a fresh instance starts at 0; `da_migrate` (if the host runs one) restores the
/// real value before `da_run`. No imports here (§6.6).
///
/// # Safety
/// Called exactly once by the host before `da_run`.
#[no_mangle]
pub unsafe extern "C" fn da_init(_c: u32, _cl: u32, _g: u32, _gl: u32) -> u32 {
    COUNTER = 0;
    MIGRATED = false;
    0
}

/// Consume the migration descriptor (§10.2): find the `"counter"` binding, restore its 8 bytes
/// through `read_back(kind = 3)`, validate, `Ready`.
///
/// # Safety
/// The host writes the descriptor span before the call (ABI §10.2).
#[no_mangle]
pub unsafe extern "C" fn da_migrate(descriptor_ptr: u32, descriptor_len: u32) -> u32 {
    const INCOMPATIBLE: u32 = 1;
    let bytes =
        std::slice::from_raw_parts(descriptor_ptr as *const u8, descriptor_len as usize).to_vec();
    let Ok(v) = ciborium::from_reader::<ciborium::value::Value, _>(bytes.as_slice()) else {
        return INCOMPATIBLE;
    };
    let ciborium::value::Value::Map(entries) = v else {
        return INCOMPATIBLE;
    };
    let field = |name: &str| {
        entries.iter().find_map(|(k, val)| match k {
            ciborium::value::Value::Text(t) if t == name => Some(val),
            _ => None,
        })
    };
    let Some(ciborium::value::Value::Array(sections)) = field("sections") else {
        return INCOMPATIBLE;
    };
    // The restore staging ID travels IN the descriptor (§10.2) — find the counter binding.
    let mut staging_id: Option<u64> = None;
    for s in sections {
        let ciborium::value::Value::Map(fields) = s else {
            return INCOMPATIBLE;
        };
        let get = |name: &str| {
            fields.iter().find_map(|(k, val)| match k {
                ciborium::value::Value::Text(t) if t == name => Some(val),
                _ => None,
            })
        };
        let is_counter =
            matches!(get("name"), Some(ciborium::value::Value::Text(t)) if t == "counter");
        if is_counter {
            staging_id = get("staging_id")
                .and_then(|v| v.as_integer())
                .map(|i| u64::try_from(i128::from(i)).unwrap_or(0));
        }
    }
    let Some(id) = staging_id else {
        return INCOMPATIBLE; // this module needs its counter section
    };
    // The explicitly granted restore capability: read_back(kind = 3), honoring the mandatory
    // NeedCapacity retry (§6.4).
    let mut out = vec![0u8; 8];
    loop {
        // SAFETY: `out` is a live guest span for the call's duration.
        let packed = abi_read_back(id, 3, out.as_mut_ptr() as u32, out.len() as u32);
        let (status, len) = (packed >> 32, (packed & 0xffff_ffff) as usize);
        match status {
            0 => {
                out.truncate(len);
                break;
            }
            1 => out.resize(len, 0),
            _ => unreachable!("unknown read_back status (fail closed, §5.2)"),
        }
    }
    let Ok(section) = <[u8; 8]>::try_from(out.as_slice()) else {
        return INCOMPATIBLE; // wrong section shape — this module cannot consume it
    };
    COUNTER = u64::from_le_bytes(section);
    MIGRATED = true;
    0 // Ready — state reconstructed and validated (§10.2)
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

/// Resume the loop from the migrated state: announce the restored counter first (the drill's
/// migrate-succeeded signal), then count frames exactly as the old module did.
#[no_mangle]
pub extern "C" fn da_run() -> u32 {
    // SAFETY: wasm is single-threaded; the host calls da_run exactly once (ABI §3.1).
    let counter = unsafe { &mut *core::ptr::addr_of_mut!(COUNTER) };
    {
        let payload = counter.to_le_bytes();
        // SAFETY: `payload` is a live 8-byte stack span for the call's duration.
        unsafe { abi_publish(0, payload.as_ptr() as u32, payload.len() as u32) };
    }
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
            EV_QUIESCE => return OUTCOME_QUIESCE_READY,
            _ => {}
        }
    }
}
