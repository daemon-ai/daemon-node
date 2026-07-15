// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `test-claim-v2` — the claim-path exerciser (ABI §9; refactor §5 A2 claim acceptance).
//!
//! Config bytes `[mode, param]` select the behavior (a claim MUST be a pure function of
//! (config, grants) — every mode is; mode 1 deliberately breaks purity to pin the
//! `ClaimInconsistent` refusal):
//!
//! | mode | `da_claim` | `da_manifest` | `da_run` |
//! |---|---|---|---|
//! | 0 | honest tiny claim | channel 0 | pull until `Stop` → `Ok` |
//! | 1 | **invocation counter in the bytes** (impure) | channel 0 | unreached (refused at assess) |
//! | 2 | honest tiny claim | **names channel 9** (unadmitted) | unreached |
//! | 3 | hard host cap = 512 B | channel 0 | `stage_state` 4096 B → the attributable cap trap |
//! | 4 | hard device = `param` GiB | channel 0 | pull until `Stop` → `Ok` |

use std::alloc::{alloc, dealloc, Layout};

#[link(wasm_import_module = "vhc@2")]
extern "C" {
    #[link_name = "next_event"]
    fn abi_next_event(buf_ptr: u32, buf_cap: u32) -> u64;
    #[link_name = "stage_state"]
    fn abi_stage_state(ptr: u32, len: u32) -> u64;
}

// ---- guest allocator (ABI §2.4) -------------------------------------------------------------------

fn layout(size: u32, align: u32) -> Layout {
    Layout::from_size_align(size as usize, (align as usize).max(1)).expect("valid layout")
}

/// Host-requested guest buffer (ABI §2.4).
#[no_mangle]
pub extern "C" fn da_alloc(size: u32, align: u32) -> u32 {
    if size == 0 {
        return 0;
    }
    // SAFETY: non-zero validated layout; paired with `da_free` by the host.
    unsafe { alloc(layout(size, align)) as u32 }
}

/// Paired release (ABI §2.4).
#[no_mangle]
pub extern "C" fn da_free(ptr: u32, size: u32, align: u32) {
    if ptr == 0 || size == 0 {
        return;
    }
    // SAFETY: matches a prior `da_alloc` triple.
    unsafe { dealloc(ptr as *mut u8, layout(size, align)) };
}

fn emit_cbor(bytes: &[u8]) -> u64 {
    if bytes.is_empty() {
        return 0;
    }
    let ptr = da_alloc(bytes.len() as u32, 1);
    // SAFETY: fresh allocation, no overlap.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len()) };
    (u64::from(ptr) << 32) | bytes.len() as u64
}

/// major 2, minor 0 (ABI §1.1).
#[no_mangle]
pub extern "C" fn da_abi() -> u32 {
    2 << 16
}

// ---- config plumbing --------------------------------------------------------------------------------

/// Read `[mode, param]` from a raw config span (shared by the CBOR-returning exports, which all
/// receive the same admitted config bytes).
///
/// # Safety
/// `cfg_ptr` is a host-written in-bounds span (ABI §2.4/§9.4).
unsafe fn mode_param(cfg_ptr: u32, cfg_len: u32) -> (u8, u8) {
    let byte = |i: u32| -> u8 {
        if i < cfg_len {
            *((cfg_ptr + i) as *const u8)
        } else {
            0
        }
    };
    (byte(0), byte(1))
}

fn text(s: &str) -> ciborium::value::Value {
    ciborium::value::Value::Text(s.into())
}

fn uint(v: u64) -> ciborium::value::Value {
    ciborium::value::Value::Integer(v.into())
}

// ---- da_manifest (ABI §2.3) ---------------------------------------------------------------------------

/// Mode 2 names an unadmitted channel (9) so the funnel's manifest-vs-table check refuses
/// `GrantsExceedLane`; every other mode names only the admitted `control` channel 0.
#[no_mangle]
pub extern "C" fn da_manifest(cfg_ptr: u32, cfg_len: u32) -> u64 {
    // SAFETY: host-written span.
    let (mode, _) = unsafe { mode_param(cfg_ptr, cfg_len) };
    let channels = if mode == 2 {
        vec![uint(0), uint(9)]
    } else {
        vec![uint(0)]
    };
    let manifest = ciborium::value::Value::Map(vec![
        (text("name"), text("test-claim-v2")),
        (text("version"), text(env!("CARGO_PKG_VERSION"))),
        (text("sdk"), text("raw-abi")),
        (text("abi"), uint(u64::from(2u32 << 16))),
        (text("channels"), ciborium::value::Value::Array(channels)),
    ]);
    let mut bytes = Vec::new();
    ciborium::into_writer(&manifest, &mut bytes).expect("manifest cbor");
    emit_cbor(&bytes)
}

// ---- da_claim (ABI §9.1) -------------------------------------------------------------------------------

/// How many times `da_claim` has run in THIS instance — mode 1's deliberate impurity.
static mut CLAIM_CALLS: u64 = 0;

#[no_mangle]
pub extern "C" fn da_claim(cfg_ptr: u32, cfg_len: u32, _g: u32, _gl: u32) -> u64 {
    // SAFETY: host-written span; wasm is single-threaded for the counter.
    let (mode, param) = unsafe { mode_param(cfg_ptr, cfg_len) };
    let calls = unsafe {
        CLAIM_CALLS += 1;
        CLAIM_CALLS
    };
    let tier = |device: u64, host: u64| {
        ciborium::value::Value::Map(vec![
            (text("device"), uint(device)),
            (text("host"), uint(host)),
        ])
    };
    let (hard, peak, workspace) = match mode {
        // Under-claimer: a 512-byte hard host cap it will breach at run time.
        3 => (tier(0, 512), tier(0, 1 << 16), tier(0, 1 << 12)),
        // Param-GiB device claimer (lane-bounds / owner-cap refusal lanes).
        4 => (
            tier(u64::from(param) << 30, 1 << 16),
            tier(0, 0),
            tier(0, 0),
        ),
        // Honest tiny claim (modes 0/1/2).
        _ => (tier(0, 1 << 16), tier(0, 1 << 20), tier(0, 1 << 12)),
    };
    let mut fields = vec![
        (text("hard_accountable"), hard),
        (text("declared_peak"), peak),
        (text("workspace"), workspace),
        (
            text("under_pressure"),
            ciborium::value::Value::Array(vec![uint(0), uint(1)]),
        ),
    ];
    if mode == 1 {
        // Impure on purpose: the bytes change per invocation → ClaimInconsistent (§9.2/§9.4-7).
        fields.push((text("notes"), text(&format!("invocation-{calls}"))));
    }
    let mut bytes = Vec::new();
    ciborium::into_writer(&ciborium::value::Value::Map(fields), &mut bytes).expect("claim cbor");
    emit_cbor(&bytes)
}

// ---- da_init / da_run -----------------------------------------------------------------------------------

static mut MODE: u8 = 0;

/// Store the mode; no imports here (§6.6).
///
/// # Safety
/// Called once by the host before `da_run`.
#[no_mangle]
pub unsafe extern "C" fn da_init(cfg_ptr: u32, cfg_len: u32, _g: u32, _gl: u32) -> u32 {
    let (mode, _) = mode_param(cfg_ptr, cfg_len);
    MODE = mode;
    0
}

const EV_STOP: u64 = 4;

fn pull_event(buf: &mut Vec<u8>) -> u64 {
    loop {
        // SAFETY: the span is `buf`'s live allocation.
        let packed = unsafe { abi_next_event(buf.as_mut_ptr() as u32, buf.capacity() as u32) };
        let (status, len) = (packed >> 32, (packed & 0xffff_ffff) as usize);
        match status {
            0 => {
                // SAFETY: the host wrote exactly `len` bytes.
                unsafe { buf.set_len(len) };
                // Minimal decode: the leading tag of the positional array (§5.1). A full decode
                // is unnecessary here — this guest only distinguishes Stop.
                let v: ciborium::value::Value =
                    ciborium::from_reader(buf.as_slice()).expect("event frame");
                let ciborium::value::Value::Array(items) = v else {
                    unreachable!("frame shape")
                };
                return items
                    .first()
                    .and_then(|t| t.as_integer())
                    .map(|i| u64::try_from(i128::from(i)).unwrap_or(u64::MAX))
                    .expect("tag");
            }
            1 => buf.reserve(len.saturating_sub(buf.capacity())),
            _ => unreachable!("unknown next_event status"),
        }
    }
}

/// Mode 3 stages 4096 bytes against its self-claimed 512-byte hard cap — the host must trap the
/// breach ATTRIBUTABLY (the under-claim acceptance); other modes pull until `Stop` → `Ok`.
#[no_mangle]
pub extern "C" fn da_run() -> u32 {
    // SAFETY: single-threaded wasm; da_init ran first.
    let mode = unsafe { MODE };
    if mode == 3 {
        let block = vec![0xEEu8; 4096];
        // SAFETY: `block` is a live span for the call's duration. This call must trap.
        unsafe { abi_stage_state(block.as_ptr() as u32, block.len() as u32) };
        // Unreachable when the cap is enforced; returning Left here would fail the test loudly.
        return 1;
    }
    let mut buf: Vec<u8> = Vec::with_capacity(64);
    loop {
        if pull_event(&mut buf) == EV_STOP {
            return 0;
        }
    }
}
