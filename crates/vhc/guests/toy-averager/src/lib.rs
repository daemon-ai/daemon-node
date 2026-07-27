// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `toy-averager` — the Phase-A expressiveness proof (refactor §5 A2 acceptance).
//!
//! A **non-round** topology from the architecture's catalog (§9, the SPARTA-shaped row): a
//! timer-driven averager that uses only A2's declared closed subset — `set_timer` + `publish`
//! (+ the mandatory `next_event` pull and slice-constant `now`). Behavior:
//!
//! - `da_init` parses its config (`[n: u8]` — how many timer ticks to average over) and arms
//!   nothing (imports are illegal during `da_init`, ABI §6.6).
//! - `da_run` first performs **module autotune** (architecture §3.5, Phase B): it reads the
//!   host's device profile (`sys@2::device_profile` — a journaled tag-15 nondeterministic input)
//!   and picks its own micro-batch size from `vram_bytes`, and reads the identity-derived
//!   deterministic seed (`sys@2::rng_seed` — §2.7 dc class, replay re-derives it), voicing both
//!   as advisory metrics the harness pins.
//! - It then arms the first timer and pulls events. On each `Timer` event it folds `fired_at`
//!   into a running mean, publishes the mean (8-byte LE f64 payload — opaque bytes to every layer
//!   below) on the `control` channel, and re-arms until `n` ticks have been averaged.
//! - It then keeps pulling until `Stop` and returns Outcome `Ok` (0).
//!
//! It exercises the `next_event` `NeedCapacity` protocol deliberately: the first pull uses an
//! 8-byte buffer, so the host must answer `NeedCapacity` with the exact required length and the
//! guest must retry enlarged (ABI §4.1).
//!
//! Raw-ABI on purpose: no SDK. The extern blocks below are the wire-true `vhc@2`/`net@2`/`sys@2`
//! signatures (ABI §6.1); the event frame is decoded from its canonical-CBOR positional-array
//! form (ABI §4.2/§5.1) with a fail-closed unknown-tag trap (§5.2).

use std::alloc::{alloc, dealloc, Layout};

// ---- raw ABI imports (ABI §6.1) -----------------------------------------------------------------

#[link(wasm_import_module = "vhc@2")]
extern "C" {
    #[link_name = "next_event"]
    fn abi_next_event(buf_ptr: u32, buf_cap: u32) -> u64;
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
    #[link_name = "emit_metric"]
    fn abi_emit_metric(name_ptr: u32, name_len: u32, value: f64);
    #[link_name = "rng_seed"]
    fn abi_rng_seed(out_ptr: u32) -> u32;
    #[link_name = "device_profile"]
    fn abi_device_profile(out_ptr: u32, out_cap: u32) -> u64;
}

// ---- guest allocator (ABI §2.4, the retained v1 convention) ---------------------------------------

fn layout(size: u32, align: u32) -> Layout {
    Layout::from_size_align(size as usize, (align as usize).max(1)).expect("valid layout")
}

/// The host requests a guest buffer through this (config/grants spans, ABI §2.4).
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

/// `(major << 16) | minor` — major 2, minor 1 (ABI §1.1): since B2 the module consumes the
/// minor-1 `sys@2` ambient surface (`rng_seed`/`device_profile`), so it declares the minor its
/// imports require (declaring below them is a typed `AbiDeclarationMismatch`, §1.3 step 5).
#[no_mangle]
pub extern "C" fn da_abi() -> u32 {
    (2 << 16) | 1
}

/// The static-requirements manifest (ABI §2.3): worlds + the single `control` channel.
#[no_mangle]
pub extern "C" fn da_manifest(_cfg_ptr: u32, _cfg_len: u32) -> u64 {
    // Canonical-enough CBOR authored by hand via ciborium (a map with the §2.3 keys this module
    // needs; full manifest/grants evaluation is the admission funnel's job).
    let manifest = ciborium::value::Value::Map(vec![
        (
            ciborium::value::Value::Text("name".into()),
            ciborium::value::Value::Text("toy-averager".into()),
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
            ciborium::value::Value::Integer(((2u32 << 16) | 1).into()),
        ),
        (
            ciborium::value::Value::Text("channels".into()),
            ciborium::value::Value::Array(vec![ciborium::value::Value::Integer(0.into())]),
        ),
    ]);
    let mut bytes = Vec::new();
    ciborium::into_writer(&manifest, &mut bytes).expect("manifest cbor");
    emit_cbor(&bytes)
}

/// The tiered memory claim (ABI §9.1) — tiny, honest constants for a module that owns no tensors.
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

struct State {
    /// Ticks to average over (config byte 0).
    n: u32,
    /// The channel to publish on (config byte 1, default 0 = `control`). A test knob: an
    /// undeclared channel makes the host trap `GrantViolation` typed (§6.2) — proving the
    /// channel table, not the guest, owns routing.
    channel: u32,
    /// How many timers to arm UP FRONT (config byte 2, default 1). A burst > the host's
    /// declared advisory `Timer` depth makes the queue coalesce (drop-oldest, journaled §4.7)
    /// deterministically — the replay-under-coalescing conformance knob.
    burst: u32,
    /// Ticks folded so far.
    ticks: u32,
    /// Running mean of `fired_at`.
    mean: f64,
}

static mut STATE: State = State {
    n: 0,
    channel: 0,
    burst: 1,
    ticks: 0,
    mean: 0.0,
};

/// Initialize with the admitted config + grants (ABI §2.1/§9.4 step 11). No imports here (§6.6).
///
/// # Safety
/// Called exactly once by the host before `da_run`; `cfg_ptr` is a host-written span.
#[no_mangle]
pub unsafe extern "C" fn da_init(cfg_ptr: u32, cfg_len: u32, _g: u32, _gl: u32) -> u32 {
    let byte = |i: u32| -> Option<u8> {
        if i < cfg_len {
            Some(*((cfg_ptr + i) as *const u8))
        } else {
            None
        }
    };
    STATE = State {
        n: u32::from(byte(0).unwrap_or(3)).max(1),
        channel: u32::from(byte(1).unwrap_or(0)),
        burst: u32::from(byte(2).unwrap_or(1)).max(1),
        ticks: 0,
        mean: 0.0,
    };
    0
}

// ---- the loop (ABI §3.1) -------------------------------------------------------------------------------

const EV_TIMER: u64 = 2;
const EV_STOP: u64 = 4;
const EV_QUIESCE: u64 = 7;
const TIMER_STEP_MS: u64 = 5;

/// Pull one event, honoring the mandatory `NeedCapacity` retry protocol (ABI §4.1). Returns the
/// frame bytes.
fn pull_event(buf: &mut Vec<u8>) -> Vec<u8> {
    loop {
        // SAFETY: the span handed to the host is exactly `buf`'s live allocation.
        let packed = unsafe { abi_next_event(buf.as_mut_ptr() as u32, buf.capacity() as u32) };
        let (status, len) = (packed >> 32, (packed & 0xffff_ffff) as usize);
        match status {
            0 => {
                // SAFETY: the host wrote exactly `len` bytes into the buffer.
                unsafe { buf.set_len(len) };
                return buf.clone();
            }
            1 => {
                // Immediate mandatory retry with an enlarged buffer (§4.1).
                buf.reserve(len);
            }
            // Unknown status: fail closed (§5.2).
            _ => unreachable!("unknown next_event status"),
        }
    }
}

/// Decode `[tag, ...]` and return `(tag, fields)` — fail closed on anything malformed (§5.2).
fn decode_frame(bytes: &[u8]) -> (u64, Vec<ciborium::value::Value>) {
    let v: ciborium::value::Value =
        ciborium::from_reader(bytes).unwrap_or_else(|_| unreachable!("malformed event frame"));
    let ciborium::value::Value::Array(items) = v else {
        unreachable!("event frame is not an array");
    };
    let tag = items
        .first()
        .and_then(|t| t.as_integer())
        .map(|i| u64::try_from(i128::from(i)).unwrap_or(u64::MAX))
        .unwrap_or_else(|| unreachable!("missing event tag"));
    (tag, items)
}

/// Pull the device profile honoring the mandatory NeedCapacity retry (ABI §6.4 discipline).
fn read_device_profile() -> Vec<u8> {
    let mut buf = vec![0u8; 64];
    loop {
        // SAFETY: `buf` is a live guest span for the call's duration.
        let packed = unsafe { abi_device_profile(buf.as_mut_ptr() as u32, buf.len() as u32) };
        let (status, len) = (packed >> 32, (packed & 0xffff_ffff) as usize);
        match status {
            0 => {
                // SAFETY: the host wrote exactly `len` bytes.
                buf.truncate(len);
                return buf;
            }
            1 => buf.resize(len, 0),
            _ => unreachable!("unknown device_profile status (fail closed, §5.2)"),
        }
    }
}

/// **Module autotune** (architecture §3.5: "micro-batch autotune moves inside the module"):
/// the module — not the host — picks its batch size from the profile's `vram_bytes`. The exact
/// ladder is arbitrary module policy; what the harness pins is that the choice is a function of
/// the host-journaled profile.
fn autotune_micro(profile: &[u8]) -> u64 {
    if profile.is_empty() {
        return 1; // no profile delivered (e.g. a harness without one): a safe floor
    }
    let v: ciborium::value::Value =
        ciborium::from_reader(profile).unwrap_or_else(|_| unreachable!("malformed profile"));
    let vram = match &v {
        ciborium::value::Value::Map(m) => m
            .iter()
            .find_map(|(k, val)| match k {
                ciborium::value::Value::Text(t) if t == "vram_bytes" => val.as_integer(),
                _ => None,
            })
            .map(|i| u64::try_from(i128::from(i)).unwrap_or(0))
            .unwrap_or(0),
        _ => 0,
    };
    if vram >= 8 << 30 {
        8
    } else if vram >= 2 << 30 {
        4
    } else {
        1
    }
}

fn emit_metric(name: &str, value: f64) {
    // SAFETY: `name` is a live guest span for the call's duration.
    unsafe { abi_emit_metric(name.as_ptr() as u32, name.len() as u32, value) };
}

/// The module main loop (ABI §3.1): arm → pull → fold → publish → re-arm → … → `Stop` → `Ok`.
#[no_mangle]
pub extern "C" fn da_run() -> u32 {
    // SAFETY: wasm is single-threaded; the host calls da_run exactly once (ABI §3.1).
    let st = unsafe { &mut *core::ptr::addr_of_mut!(STATE) };

    // Module autotune (architecture §3.5) + seeded determinism, voiced as advisory metrics the
    // harness pins: the batch size is a guest decision over the journaled (tag-15) profile, and
    // the seed is the identity-derived `rng_seed` (deterministic — replay re-derives it).
    let micro = autotune_micro(&read_device_profile());
    #[allow(clippy::cast_precision_loss)]
    emit_metric("autotune.micro_batch", micro as f64);
    let mut seed = [0u8; 32];
    // SAFETY: `seed` is a live 32-byte guest span for the call's duration.
    let rng_status = unsafe { abi_rng_seed(seed.as_mut_ptr() as u32) };
    if rng_status != 0 {
        unreachable!("unknown rng_seed status (fail closed, §5.2)");
    }
    emit_metric(
        "rng.seed0",
        f64::from(u32::from_le_bytes([seed[0], seed[1], seed[2], seed[3]])),
    );

    // Arm the opening tick(s) (legal before the first slice — §6.6 rule 2). A burst > 1 arms
    // them all at the same deadline, so they fire in ONE pump batch — the deterministic
    // queue-coalescing driver for the replay-under-coalescing conformance lane.
    for _ in 0..st.burst {
        // SAFETY: plain-value import call.
        unsafe { abi_set_timer(TIMER_STEP_MS) };
    }

    // Start deliberately undersized to exercise the NeedCapacity round-trip (§4.1).
    let mut buf: Vec<u8> = Vec::with_capacity(8);
    loop {
        let frame = pull_event(&mut buf);
        let (tag, items) = decode_frame(&frame);
        match tag {
            EV_TIMER => {
                let fired_at = items
                    .get(2)
                    .and_then(|v| v.as_integer())
                    .map(|i| u64::try_from(i128::from(i)).unwrap_or(0))
                    .unwrap_or(0);
                if st.ticks < st.n {
                    // Fold the observed logical fire time into the running mean.
                    st.ticks += 1;
                    #[allow(clippy::cast_precision_loss)]
                    {
                        st.mean += (fired_at as f64 - st.mean) / f64::from(st.ticks);
                    }
                    let payload = st.mean.to_le_bytes();
                    // SAFETY: `payload` is a live 8-byte stack span for the call's duration.
                    unsafe {
                        abi_publish(st.channel, payload.as_ptr() as u32, payload.len() as u32)
                    };
                    if st.ticks < st.n {
                        // SAFETY: plain-value import call.
                        unsafe { abi_set_timer(TIMER_STEP_MS) };
                    }
                }
            }
            EV_STOP => return 0, // Outcome Ok — return promptly, no further imports (§4.4).
            EV_QUIESCE => return 2, // QuiesceReady (nothing to snapshot — no durable state).
            // Frame / PayloadReady / Budget: this module has no use for them; ignoring a
            // DELIVERED event is module policy (only unknown TAGS must fail closed, §5.2).
            _ => {}
        }
    }
}

// ---- da_resource_plan (the certification rung's assessment export) ------------------------------

/// This module's Logical Resource Plan. Its algorithm holds nothing device-resident, so the
/// canonical trivial plan IS its plan: the module's linear-memory floor, and no device tensor, no
/// operation family and no bounded transfer.
///
/// It is emitted here rather than written down beside the module because authoring consumes module
/// output with no fallback — a plan that exists anywhere except as this export's result is a second
/// source that can drift from the module it claims to describe.
#[no_mangle]
pub extern "C" fn da_resource_plan(_c: u32, _cl: u32, _g: u32, _gl: u32) -> u64 {
    let plan = daemon_vhc_proto::resource_plan::LogicalResourcePlan::trivial(
        daemon_vhc_proto::resource_plan::WASM_GUEST_LINEAR_FLOOR_BYTES,
    );
    match plan.to_canonical_bytes() {
        Ok(bytes) => emit_cbor(&bytes),
        Err(_) => 0,
    }
}
