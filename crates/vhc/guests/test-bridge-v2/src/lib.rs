// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `test-bridge-v2` — the `tabi@1`-bridge-under-major-2 conformance guest (ABI §2.5).
//!
//! `da_init` registers a param through the bridge (legal only there, §2.5 rule 1); `da_run` arms
//! a timer, and inside the delivered `Timer` slice runs frozen v1 tensor ops
//! (`ones@1` → `add@1` → `scalar@1`) and publishes the read-out scalar — the whole point: the
//! event loop replaced the lifecycle while the math runs on the IDENTICAL frozen op surface.
//! Modes (config byte 0): 0 happy path; 1 registration inside a slice (PhaseViolation);
//! 2 a slice-class handle used across a slice boundary (StaleHandle).

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

#[link(wasm_import_module = "sys@2")]
extern "C" {
    #[link_name = "set_timer"]
    fn abi_set_timer(delay_ms: u64) -> u64;
}

#[link(wasm_import_module = "tabi@1")]
extern "C" {
    #[link_name = "param@1"]
    fn tabi_param(np: u32, nl: u32, dp: u32, dr: u32, dt: u32, init: u32, p0: f64, p1: f64) -> u64;
    #[link_name = "ones@1"]
    fn tabi_ones(dp: u32, dr: u32, dt: u32) -> u64;
    #[link_name = "add@1"]
    fn tabi_add(a: u64, b: u64) -> u64;
    #[link_name = "scalar@1"]
    fn tabi_scalar(x: u64) -> f64;
    #[link_name = "batch_size@1"]
    fn tabi_batch_size(b: u64) -> u32;
    #[link_name = "upd_sections@1"]
    fn tabi_upd_sections(i: u32) -> u32;
}

/// `read_back` a staged item as a CBOR uint (bridge kinds 1/2 — the handle / staging index).
fn read_back_uint(src: u64, kind: u32) -> u64 {
    let mut buf = [0u8; 16];
    // SAFETY: `buf` is a live guest span for the call's duration.
    let packed = unsafe { abi_read_back(src, kind, buf.as_mut_ptr() as u32, buf.len() as u32) };
    let (status, len) = (packed >> 32, (packed & 0xffff_ffff) as usize);
    assert!(status == 0 && len <= buf.len(), "tiny CBOR uint fits");
    let v: ciborium::value::Value = ciborium::from_reader(&buf[..len]).expect("uint cbor");
    v.as_integer()
        .map(|i| u64::try_from(i128::from(i)).unwrap_or(u64::MAX))
        .expect("uint")
}

// ---- allocator + CBOR-return glue (ABI §2.4/§2.1, same as the sibling raw-ABI guests) ------------

fn layout(size: u32, align: u32) -> Layout {
    Layout::from_size_align(size as usize, (align as usize).max(1)).expect("layout")
}

/// Host-requested guest buffer (ABI §2.4).
#[no_mangle]
pub extern "C" fn da_alloc(size: u32, align: u32) -> u32 {
    if size == 0 {
        return 0;
    }
    // SAFETY: validated non-zero layout; host pairs with da_free.
    unsafe { alloc(layout(size, align)) as u32 }
}

/// Paired release (ABI §2.4).
#[no_mangle]
pub extern "C" fn da_free(ptr: u32, size: u32, align: u32) {
    if ptr == 0 || size == 0 {
        return;
    }
    // SAFETY: matches a prior da_alloc triple.
    unsafe { dealloc(ptr as *mut u8, layout(size, align)) };
}

fn emit_cbor(bytes: &[u8]) -> u64 {
    if bytes.is_empty() {
        return 0;
    }
    let ptr = da_alloc(bytes.len() as u32, 1);
    // SAFETY: fresh allocation; no overlap.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len()) };
    (u64::from(ptr) << 32) | bytes.len() as u64
}

/// major 2, minor 0.
#[no_mangle]
pub extern "C" fn da_abi() -> u32 {
    2 << 16
}

fn text(s: &str) -> ciborium::value::Value {
    ciborium::value::Value::Text(s.into())
}
fn uint(v: u64) -> ciborium::value::Value {
    ciborium::value::Value::Integer(v.into())
}

/// Manifest: the `control` channel + the bridge world (ABI §2.3).
#[no_mangle]
pub extern "C" fn da_manifest(_c: u32, _cl: u32) -> u64 {
    let m = ciborium::value::Value::Map(vec![
        (text("name"), text("test-bridge-v2")),
        (text("version"), text(env!("CARGO_PKG_VERSION"))),
        (text("sdk"), text("raw-abi")),
        (text("abi"), uint(u64::from(2u32 << 16))),
        (
            text("channels"),
            ciborium::value::Value::Array(vec![uint(0)]),
        ),
    ]);
    let mut b = Vec::new();
    ciborium::into_writer(&m, &mut b).expect("cbor");
    emit_cbor(&b)
}

/// A tiny honest claim (ABI §9.1).
#[no_mangle]
pub extern "C" fn da_claim(_c: u32, _cl: u32, _g: u32, _gl: u32) -> u64 {
    let tier = |d: u64, h: u64| {
        ciborium::value::Value::Map(vec![(text("device"), uint(d)), (text("host"), uint(h))])
    };
    let claim = ciborium::value::Value::Map(vec![
        (text("hard_accountable"), tier(0, 1 << 16)),
        (text("declared_peak"), tier(0, 1 << 20)),
        (text("workspace"), tier(0, 1 << 12)),
        (
            text("under_pressure"),
            ciborium::value::Value::Array(vec![uint(0), uint(1)]),
        ),
    ]);
    let mut b = Vec::new();
    ciborium::into_writer(&claim, &mut b).expect("cbor");
    emit_cbor(&b)
}

// ---- state ------------------------------------------------------------------------------------------

static mut MODE: u8 = 0;
static mut W: u64 = 0;

/// Register the param through the bridge — legal exactly here (§2.5 rule 1).
///
/// # Safety
/// Called once by the host; `cfg_ptr` is a host-written span.
#[no_mangle]
pub unsafe extern "C" fn da_init(cfg_ptr: u32, cfg_len: u32, _g: u32, _gl: u32) -> u32 {
    MODE = if cfg_len >= 1 {
        *(cfg_ptr as *const u8)
    } else {
        0
    };
    let dims: [u32; 1] = [1];
    let name = b"w";
    W = tabi_param(
        name.as_ptr() as u32,
        name.len() as u32,
        dims.as_ptr() as u32,
        1,
        0, // dtype f32
        1, // init = ones
        0.0,
        0.0,
    );
    0
}

const EV_PAYLOAD_READY: u64 = 1;
const EV_TIMER: u64 = 2;
const EV_STOP: u64 = 4;

/// Pull one event; returns `(tag, staging_id-if-PayloadReady)`.
fn pull_tag2(buf: &mut Vec<u8>) -> (u64, u64) {
    loop {
        // SAFETY: the span is buf's live allocation.
        let packed = unsafe { abi_next_event(buf.as_mut_ptr() as u32, buf.capacity() as u32) };
        let (status, len) = (packed >> 32, (packed & 0xffff_ffff) as usize);
        match status {
            0 => {
                // SAFETY: host wrote exactly len bytes.
                unsafe { buf.set_len(len) };
                let v: ciborium::value::Value =
                    ciborium::from_reader(buf.as_slice()).expect("frame");
                let ciborium::value::Value::Array(items) = v else {
                    unreachable!()
                };
                let uint = |i: usize| {
                    items
                        .get(i)
                        .and_then(|t| t.as_integer())
                        .map(|n| u64::try_from(i128::from(n)).unwrap_or(u64::MAX))
                        .unwrap_or(u64::MAX)
                };
                return (uint(0), uint(1));
            }
            1 => buf.reserve(len.saturating_sub(buf.capacity())),
            _ => unreachable!(),
        }
    }
}

/// The loop: bridge math inside Timer slices, per mode; Stop → Ok.
#[no_mangle]
pub extern "C" fn da_run() -> u32 {
    // SAFETY: single-threaded wasm; da_init ran.
    let mode = unsafe { MODE };
    // SAFETY: plain-value import.
    unsafe { abi_set_timer(3) };
    let mut buf: Vec<u8> = Vec::with_capacity(64);
    let mut stale_probe: u64 = 0;
    let mut slices = 0u32;
    loop {
        let (tag, staging_id) = pull_tag2(&mut buf);
        if tag == EV_STOP {
            return 0;
        }
        // Staging modes (§2.5 rule 2): consume a host-staged batch (3) / update container (4)
        // through read_back kinds 1/2 and publish the bridge readout as the proof.
        if tag == EV_PAYLOAD_READY {
            // SAFETY: plain-value imports over guest-owned spans.
            unsafe {
                match mode {
                    3 => {
                        let handle = read_back_uint(staging_id, 1);
                        let n = tabi_batch_size(handle);
                        let p = n.to_le_bytes();
                        abi_publish(0, p.as_ptr() as u32, p.len() as u32);
                    }
                    4 => {
                        let idx = read_back_uint(staging_id, 2);
                        let n = tabi_upd_sections(idx as u32);
                        let p = n.to_le_bytes();
                        abi_publish(0, p.as_ptr() as u32, p.len() as u32);
                    }
                    _ => {}
                }
            }
            continue;
        }
        if tag != EV_TIMER {
            continue;
        }
        slices += 1;
        // SAFETY: plain-value imports over guest-owned spans.
        unsafe {
            match (mode, slices) {
                // Happy path: ones([1]) + w (ones param) → scalar 2.0 → publish LE bytes.
                (0, 1) => {
                    let dims: [u32; 1] = [1];
                    let o = tabi_ones(dims.as_ptr() as u32, 1, 0);
                    let s = tabi_add(o, W);
                    let v = tabi_scalar(s);
                    let payload = v.to_le_bytes();
                    abi_publish(0, payload.as_ptr() as u32, payload.len() as u32);
                }
                // Registration inside a slice: MUST trap PhaseViolation (§2.5 rule 1).
                (1, 1) => {
                    let dims: [u32; 1] = [1];
                    let n = b"late";
                    tabi_param(
                        n.as_ptr() as u32,
                        n.len() as u32,
                        dims.as_ptr() as u32,
                        1,
                        0,
                        1,
                        0.0,
                        0.0,
                    );
                }
                // Slice-class retention: create in slice 1, use in slice 2 → StaleHandle (§7.1).
                (2, 1) => {
                    let dims: [u32; 1] = [1];
                    stale_probe = tabi_ones(dims.as_ptr() as u32, 1, 0);
                    abi_set_timer(3);
                }
                (2, 2) => {
                    let _ = tabi_scalar(stale_probe); // MUST trap StaleHandle
                }
                _ => {}
            }
        }
    }
}
