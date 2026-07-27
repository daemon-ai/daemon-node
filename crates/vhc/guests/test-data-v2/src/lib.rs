// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `test-data-v2` — the B2 `data@2` fetch conformance guest (declares abi **2.1**).
//!
//! The end-to-end shape architecture §3.2 promises: **windowing is module policy, fetch is host
//! mechanism**. Driven by its config `mode`:
//!
//! - **0 — the corpus-shard window fetch ("adapts")**: parse the corpus manifest (the SDK corpus
//!   policy layer), `locate(seq)` to decide WHICH byte range of WHICH shard the module needs (a
//!   policy decision made in-guest), `data_fetch(shard_hash, range)` → on
//!   `Completion(op, Ok(BufferHandle))` read the window back, decode the u16 tokens, and publish
//!   them LE-encoded (the fed batch) — then release the buffer and pull until `Stop`.
//! - **1 — the grant negative**: fetch a hash outside the admitted artifact set. The host traps
//!   `GrantViolation` before any op is issued (the run dies typed; nothing to publish).
//! - **2 — the range negative**: fetch a granted artifact with an absurd `range_off`; the
//!   completion is `Err(StoreRefused)` — publish `[code]` so the harness pins the code.
//! - **3 — the tamper negative**: fetch a granted artifact the harness services with WRONG
//!   bytes; the pump's whole-artifact verification completes `Err(HashMismatch)` — publish
//!   `[code]`. This is "hosts fetch-and-verify against the committed hash" made falsifiable.
//!
//! The **chunk-addressed** modes (the corpus contract — shard identity is the chunk fold, the
//! host serves verified covering-chunk ranges, `register_chunks` first):
//!
//! - **4 — the chunked range fetch**: `register_chunks(desc)` then `fetch(shard, off, len)`;
//!   on `Ok(BufferHandle)` publish the window bytes (the host verified the covering chunks and
//!   sliced the exact range — the embedder never moved the whole shard).
//! - **5 — the chunked tamper negative**: same calls; the harness lies in one covering chunk;
//!   the pump completes `Err(HashMismatch)` — publish `[code]`.
//! - **6 — the read budget**: register, then the same fetch twice; the harness admits a budget
//!   of exactly one window, so the first completes `Ok` (window published) and the second
//!   completes `Err(GrantExhausted)` — publish `[code]`.
//! - **7 — the registered-bounds negative**: register, then fetch an absurd `off`; bounds are
//!   knowable at the call on a registered shard — `Err(StoreRefused)`, publish `[code]`.
//! - **8 — the registration grant negative**: `register_chunks` with a descriptor whose fold is
//!   NOT a granted artifact — the host traps `GrantViolation` at the call (nothing published).
//!
//! The guest names artifacts by content hash only — it has no URL, locator, or credential
//! surface to even express (snapshot pinning at the edge, architecture §5.1).

use daemon_vhc_sdk::corpus::Manifest;

// ---- required exports (ABI §2.1) — the sdk-v2 rt helpers back the allocator ---------------------

#[no_mangle]
pub extern "C" fn da_alloc(size: u32, align: u32) -> u32 {
    daemon_vhc_sdk::module::rt::da_alloc(size, align)
}

#[no_mangle]
pub extern "C" fn da_free(ptr: u32, size: u32, align: u32) {
    daemon_vhc_sdk::module::rt::da_free(ptr, size, align);
}

/// `(major << 16) | minor` — major 2, **minor 2**: this module consumes completions + data@2
/// including `register_chunks` (introduced at minor 2 — declaring lower would be
/// `AbiDeclarationMismatch`, §1.3 step 5).
#[no_mangle]
pub extern "C" fn da_abi() -> u32 {
    (2 << 16) | 2
}

fn text(s: &str) -> ciborium::value::Value {
    ciborium::value::Value::Text(s.into())
}

fn uint(v: u64) -> ciborium::value::Value {
    ciborium::value::Value::Integer(v.into())
}

#[no_mangle]
pub extern "C" fn da_manifest(_cfg_ptr: u32, _cfg_len: u32) -> u64 {
    let v = ciborium::value::Value::Map(vec![
        (text("name"), text("test-data-v2")),
        (text("version"), text(env!("CARGO_PKG_VERSION"))),
        (text("sdk"), text("daemon-vhc-sdk")),
        (text("abi"), uint(u64::from(da_abi()))),
        (
            text("channels"),
            ciborium::value::Value::Array(vec![uint(0)]),
        ),
    ]);
    let mut bytes = Vec::new();
    ciborium::into_writer(&v, &mut bytes).expect("manifest cbor");
    daemon_vhc_sdk::module::rt::emit_cbor(&bytes)
}

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
    daemon_vhc_sdk::module::rt::emit_cbor(&bytes)
}

// ---- module state -----------------------------------------------------------------------------------

struct State {
    mode: u64,
    manifest_json: String,
    shard: [u8; 32],
    seq: u64,
    /// The chunk-map descriptor for the chunked modes (canonical CBOR, register_chunks input).
    desc: Vec<u8>,
    /// The chunked modes' requested range.
    off: u64,
    len: u64,
}

static mut STATE: State = State {
    mode: 0,
    manifest_json: String::new(),
    shard: [0u8; 32],
    seq: 0,
    desc: Vec::new(),
    off: 0,
    len: 0,
};

/// Config: canonical CBOR `{"mode": uint, "manifest": tstr, "shard": bstr32, "seq": uint,
/// "desc": bstr, "off": uint, "len": uint}` (the last three feed the chunked modes). In a
/// real run the manifest is itself a pinned artifact the module fetches first; inline here
/// keeps the conformance focused on the shard fetch path.
///
/// # Safety
/// Called exactly once by the host before `da_run`; `cfg_ptr` is a host-written span.
#[no_mangle]
pub unsafe extern "C" fn da_init(cfg_ptr: u32, cfg_len: u32, _g: u32, _gl: u32) -> u32 {
    let bytes = core::slice::from_raw_parts(cfg_ptr as *const u8, cfg_len as usize);
    let Ok(ciborium::value::Value::Map(entries)) = ciborium::from_reader(bytes) else {
        return 16;
    };
    let field = |name: &str| -> Option<ciborium::value::Value> {
        entries.iter().find_map(|(k, v)| match k {
            ciborium::value::Value::Text(t) if t == name => Some(v.clone()),
            _ => None,
        })
    };
    let as_uint = |v: &ciborium::value::Value| -> u64 {
        v.as_integer()
            .map(|n| u64::try_from(i128::from(n)).unwrap_or(0))
            .unwrap_or(0)
    };
    let mode = field("mode").map(|v| as_uint(&v)).unwrap_or(0);
    let manifest_json = match field("manifest") {
        Some(ciborium::value::Value::Text(t)) => t,
        _ => String::new(),
    };
    let shard: [u8; 32] = match field("shard") {
        Some(ciborium::value::Value::Bytes(b)) if b.len() == 32 => {
            b.as_slice().try_into().expect("32 bytes")
        }
        _ => return 17,
    };
    let seq = field("seq").map(|v| as_uint(&v)).unwrap_or(0);
    let desc = match field("desc") {
        Some(ciborium::value::Value::Bytes(b)) => b,
        _ => Vec::new(),
    };
    let off = field("off").map(|v| as_uint(&v)).unwrap_or(0);
    let len = field("len").map(|v| as_uint(&v)).unwrap_or(0);
    STATE = State {
        mode,
        manifest_json,
        shard,
        seq,
        desc,
        off,
        len,
    };
    0
}

// ---- the loop (ABI §3.1) -------------------------------------------------------------------------------

const EV_STOP: u64 = 4;
const EV_COMPLETION: u64 = 6;

/// Decode a completion frame `[6, op, [variant, payload]]`.
fn decode_completion(ev: &daemon_vhc_sdk::Event) -> (u64, u64, Option<ciborium::value::Value>) {
    let op = ev.uint(1);
    let Some(ciborium::value::Value::Array(result)) = ev.items.get(2) else {
        unreachable!("completion result is [variant, payload]");
    };
    let variant = result
        .first()
        .and_then(ciborium::value::Value::as_integer)
        .map(|n| u64::try_from(i128::from(n)).unwrap_or(u64::MAX))
        .unwrap_or(u64::MAX);
    (op, variant, result.get(1).cloned())
}

/// The comp-error code of an `Err` payload map.
fn err_code(payload: &Option<ciborium::value::Value>) -> u64 {
    match payload {
        Some(ciborium::value::Value::Map(m)) => m
            .iter()
            .find_map(|(k, v)| match k {
                ciborium::value::Value::Text(t) if t == "code" => v.as_integer(),
                _ => None,
            })
            .map(|n| u64::try_from(i128::from(n)).unwrap_or(u64::MAX))
            .unwrap_or(u64::MAX),
        _ => u64::MAX,
    }
}

#[no_mangle]
pub extern "C" fn da_run() -> u32 {
    // SAFETY: wasm is single-threaded; the host calls da_run exactly once (ABI §3.1).
    let st = unsafe { &mut *core::ptr::addr_of_mut!(STATE) };

    // The chunked modes register their shard's chunk map FIRST (the corpus contract: a
    // fold-identity shard has no whole-object hash — an unregistered fetch can never verify).
    // Mode 8 registers an UNGRANTED fold: the host traps GrantViolation on this very call.
    if matches!(st.mode, 4..=8) {
        let status = daemon_vhc_sdk::data_register_chunks(&st.desc);
        if status != 0 {
            return 20;
        }
    }

    // The opening move per mode: issue exactly one fetch.
    let mut op = match st.mode {
        // Mode 1: an UNGRANTED hash — the host traps GrantViolation on this very call.
        1 => daemon_vhc_sdk::data_fetch(&[0xAB; 32], 0, 0),
        // Mode 2: a granted artifact, absurd range — completes Err(StoreRefused).
        2 => daemon_vhc_sdk::data_fetch(&st.shard, u64::MAX / 2, 1),
        // Mode 3: a granted artifact the harness tampers — completes Err(HashMismatch).
        3 => daemon_vhc_sdk::data_fetch(&st.shard, 0, 0),
        // Modes 4/5/6: a registered chunked shard's byte range (POLICY chose it; the host
        // serves + verifies the covering chunks and slices exactly this window).
        4 | 5 | 6 => daemon_vhc_sdk::data_fetch(&st.shard, st.off, st.len),
        // Mode 7: registered bounds are knowable at the call — absurd off refuses typed.
        7 => daemon_vhc_sdk::data_fetch(&st.shard, u64::MAX / 2, 1),
        // Mode 8: unreachable (register_chunks trapped above).
        8 => return 21,
        // Mode 0: the corpus window ("adapts"): POLICY decides which bytes it needs — the SDK
        // corpus layer locates the sequence, and the module fetches exactly that byte range.
        _ => {
            let Ok(manifest) = Manifest::from_json(&st.manifest_json) else {
                return 18;
            };
            let Ok(loc) = manifest.locate(st.seq) else {
                return 19;
            };
            let width = manifest.token_width.bytes();
            let range_off = loc.token_offset * width;
            let range_len = u64::from(manifest.seq_len) * width;
            daemon_vhc_sdk::data_fetch(&st.shard, range_off, range_len)
        }
    };

    // Mode 6 drives a SECOND identical fetch after the first window lands (the budget covers
    // exactly one window, so the second completes Err(GrantExhausted)).
    let mut second_pending = st.mode == 6;

    let mut buf: Vec<u8> = Vec::with_capacity(16); // deliberately small: exercises NeedCapacity
    loop {
        let ev = daemon_vhc_sdk::next_event(&mut buf);
        match ev.tag {
            EV_STOP => return 0,
            EV_COMPLETION => {
                let (cop, variant, payload) = decode_completion(&ev);
                if cop != op {
                    let _ = daemon_vhc_sdk::publish(0, b"unexpected-completion");
                    continue;
                }
                match (st.mode, variant) {
                    (0 | 4 | 6, 0) => {
                        // Ok(BufferHandle): the fetched window IS the batch — publish it.
                        let handle = payload
                            .as_ref()
                            .and_then(ciborium::value::Value::as_integer)
                            .map(|n| u64::try_from(i128::from(n)).unwrap_or(0))
                            .unwrap_or(0);
                        let window = daemon_vhc_sdk::read_buffer(handle);
                        let _ = daemon_vhc_sdk::publish(0, &window);
                        daemon_vhc_sdk::buffer_release(handle);
                        if second_pending {
                            second_pending = false;
                            op = daemon_vhc_sdk::data_fetch(&st.shard, st.off, st.len);
                        }
                    }
                    // The negative modes publish the comp-error code byte for the harness.
                    (2 | 3 | 5 | 6 | 7, 1) => {
                        let code = err_code(&payload);
                        let _ = daemon_vhc_sdk::publish(0, &[u8::try_from(code).unwrap_or(0xFF)]);
                    }
                    _ => {
                        let _ = daemon_vhc_sdk::publish(0, b"wrong-variant");
                    }
                }
            }
            // Frame / PayloadReady / Timer / Budget / Quiesce: ignored (module policy).
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
        Ok(bytes) => {
            let ptr = da_alloc(bytes.len() as u32, 1);
            // SAFETY: fresh allocation of exactly `len` bytes; the regions do not overlap.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len()) };
            (u64::from(ptr) << 32) | bytes.len() as u64
        }
        Err(_) => 0,
    }
}
