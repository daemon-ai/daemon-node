// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The module layer under [`crate::main!`]: the [`GuestModule`] trait, the author's declaration,
//! and the SDK-side **claim generation** (ABI §9.1, deferred from the admission sitting) —
//! authors declare raw capacities, the SDK derives the tiered claim wire form. Hand-authored
//! manifests/claims stay legal: the funnel judges bytes, not their author.
//!
//! Native-visible on purpose: sim tests drive [`GuestModule`] methods and the derivations directly
//! (the `main!` exports are wasm32-only, exactly like the v1 `experiment!` macro).

use crate::migrate::{MigrationDescriptor, SectionReader};

/// What a module declares about itself — the input the SDK derives the manifest + claim from.
#[derive(Debug, Clone)]
pub struct ModuleDecl {
    /// Module name (manifest `name`).
    pub name: &'static str,
    /// Module version (manifest `version`).
    pub version: &'static str,
    /// The declared major-2 ABI **minor** (ABI §1.1): MUST be ≥ the highest introducing minor of
    /// the module's static imports (§1.3 step 5) and ≤ the host's implemented minor. `0` = the
    /// Phase-A closed subset; `1` = the B1 buffer/completion surface.
    pub abi_minor: u32,
    /// Channels the module publishes/subscribes on (manifest `channels`, §6.2).
    pub channels: Vec<u32>,
    /// Long-lived host-accountable state bytes — for a wasm module, its **guest linear-memory
    /// ceiling** (params, queues, config, fold-window buffers, decoded payload rows).
    ///
    /// Linear memory never shrinks, so a module's transient peak IS its long-lived floor: whatever
    /// it touches once, the host reserves for its lifetime. This tier therefore accounts the
    /// module's peak linear memory, and it is the tier the host ENFORCES as the sandbox's memory
    /// cap (`EngineConfig::with_claimed_memory` — the ABI §9.1 "resources the host meters exactly",
    /// metered by the pooling allocator itself). Declare it from a measurement, at the geometry the
    /// run admits under ([`GuestModule::decl_for_config`]).
    pub host_state_bytes: u64,
    /// Transient host scratch ABOVE the linear-memory floor: the host-side bytes the module stages
    /// but never holds in linear memory — staged sections and buffers it seals (`create_from`,
    /// `buffer_append`), e.g. an outgoing committed update built append-by-append. Metered exactly
    /// too (against the peak tier), just against a different resource.
    pub host_scratch_bytes: u64,
    /// Long-lived device-resident bytes (0 for a pure-bridge or host-only module — device
    /// residency is host-mechanism under the §2.5 bridge).
    pub device_state_bytes: u64,
    /// Transient device scratch above the device state floor.
    pub device_scratch_bytes: u64,
}

/// A major-2 module under [`crate::main!`]: the v2 analogue of the v1 SDK's `Experiment`.
pub trait GuestModule: Sized {
    /// The static declaration the manifest + claim derive from.
    fn decl() -> ModuleDecl;

    /// The **config-dependent** declaration (ABI §9.1): the honest manifest/claim derive from the
    /// run config, not a fixed constant. A module whose host/device footprint is a function of the
    /// geometry it is admitted for — the streamed trainer's working set is O(window buffers +
    /// resident payload sections + bookkeeping), derived from the model layout + profile — makes
    /// its claim honest here, so the fleet preflight's capability-fit line item is checkable
    /// (production program §9). The default is config-independent (`Self::decl()`), so modules
    /// with a fixed footprint need not override it. `da_manifest`/`da_claim` are handed the config
    /// span the host wrote (both run before `da_init`).
    fn decl_for_config(config: &[u8]) -> ModuleDecl {
        let _ = config;
        Self::decl()
    }

    /// Build state from config + grants (`da_init`, ABI §2.3) — bridge registration is legal
    /// exactly here (§2.5). A nonzero code refuses the join (module-defined detail ≥ 16).
    fn init(config: &[u8], grants: &[u8]) -> Result<Self, u32>;

    /// The inverted event loop (`da_run`, §3.1). Returns the module outcome code.
    fn run(&mut self) -> u32;

    /// Consume a migration descriptor (`da_migrate`, §10.2): reconstruct state from the staged
    /// sections and return `0` (`Ready`) or `1`/`≥16` (`Incompatible` + detail). The default is
    /// an honest `Incompatible`: `main!` always exports `da_migrate` (so the manifest's
    /// `migratable: true` echo is truthful, §6.2), and a module that does not override this
    /// simply cannot consume any descriptor — the host rolls back (§10.3 step 7), the ratified
    /// recoverable outcome.
    fn migrate(&mut self, descriptor: &MigrationDescriptor, reader: &mut dyn SectionReader) -> u32 {
        let _ = (descriptor, reader);
        1
    }
}

/// Guest-memory page granularity for claim rounding (wasm32 page = 64 KiB is the growth unit,
/// but claims account bytes; 4 KiB keeps tiers honest without gifting slack).
const CLAIM_PAGE: u64 = 4096;

/// The **linear-memory floor of a wasm32 Rust `cdylib`** — the bytes the module's image needs
/// before any declared state exists: the linker's initial memory (the 1 MiB default shadow stack,
/// the data segments, the first heap pages) plus the allocator arena the SDK's own event-frame and
/// CBOR scratch touches on the first slice.
///
/// It is a floor on [`ModuleDecl::host_state_bytes`], applied here rather than left to each author,
/// because that tier is what the host ENFORCES as the sandbox's linear-memory cap
/// (`EngineConfig::with_claimed_memory`): a module that declares only its own working set is not
/// making a smaller claim, it is making a false one — the runtime beneath it does not become
/// smaller for having gone undeclared, and the instance simply fails to come up.
///
/// **Measured, not guessed**: every guest in `crates/vhc/guests` fails to instantiate at a 1 MiB
/// cap ("module memory does not fit in pooling allocator requirements") and comes up at 2 MiB,
/// independent of its size — it is the toolchain's floor, not the module's. The declared figure is
/// double the measured minimum so a toy module has real working margin. A floor can only ever
/// RAISE a claim, so it cannot hide an over-run, and any module whose own derived figure is larger
/// (the trainer's is ~59 MiB at the ceremony geometry) is unaffected.
pub const WASM_LINEAR_MEMORY_FLOOR_BYTES: u64 = 4 << 20;

fn round_page(v: u64) -> u64 {
    v.div_ceil(CLAIM_PAGE) * CLAIM_PAGE
}

fn text(s: &str) -> ciborium::value::Value {
    ciborium::value::Value::Text(s.into())
}

fn uint(v: u64) -> ciborium::value::Value {
    ciborium::value::Value::Integer(v.into())
}

/// Derive the tiered memory claim (ABI §9.1) from the declaration:
///
/// - `hard_accountable` = the declared state floor, lifted to [`WASM_LINEAR_MEMORY_FLOOR_BYTES`]
///   and page-rounded — what the host must reserve, and for a wasm module what it enforces;
/// - `workspace`       = the declared transient scratch (page-rounded);
/// - `declared_peak`   = state + scratch (page-rounded) — one extra transient copy of
///   everything, the conservative headroom a module may briefly hold beyond the accounted
///   tiers (the three tiers are disjoint; the host reserves their sum);
/// - `under_pressure`  = `[0, 1]` — deny new buffers, then trap the slice (§9.1's ratified
///   pressure ladder for modules without a custom shedding strategy).
///
/// Returns the claim's CBOR wire bytes (the exact form `da_claim` emits).
#[must_use]
pub fn derive_claim(decl: &ModuleDecl) -> Vec<u8> {
    let tier = |d: u64, h: u64| {
        ciborium::value::Value::Map(vec![(text("device"), uint(d)), (text("host"), uint(h))])
    };
    let host_state = decl.host_state_bytes.max(WASM_LINEAR_MEMORY_FLOOR_BYTES);
    let claim = ciborium::value::Value::Map(vec![
        (
            text("hard_accountable"),
            tier(round_page(decl.device_state_bytes), round_page(host_state)),
        ),
        (
            text("declared_peak"),
            tier(
                round_page(decl.device_state_bytes + decl.device_scratch_bytes),
                round_page(host_state + decl.host_scratch_bytes),
            ),
        ),
        (
            text("workspace"),
            tier(
                round_page(decl.device_scratch_bytes),
                round_page(decl.host_scratch_bytes),
            ),
        ),
        (
            text("under_pressure"),
            ciborium::value::Value::Array(vec![uint(0), uint(1)]),
        ),
    ]);
    let mut b = Vec::new();
    ciborium::into_writer(&claim, &mut b).expect("claim cbor");
    b
}

/// Derive the manifest wire bytes (ABI §2.3/§6.2 Phase-A fields). `migratable` is always `true`
/// under `main!` — the macro always exports `da_migrate` (§6.2: "true iff exported"); whether a
/// descriptor is consumable is [`GuestModule::migrate`]'s runtime answer.
#[must_use]
pub fn manifest_bytes(decl: &ModuleDecl) -> Vec<u8> {
    let m = ciborium::value::Value::Map(vec![
        (text("name"), text(decl.name)),
        (text("version"), text(decl.version)),
        (text("sdk"), text("daemon-vhc-sdk")),
        (text("abi"), uint(u64::from((2u32 << 16) | decl.abi_minor))),
        (
            text("channels"),
            ciborium::value::Value::Array(
                decl.channels.iter().map(|c| uint(u64::from(*c))).collect(),
            ),
        ),
        (text("migratable"), ciborium::value::Value::Bool(true)),
    ]);
    let mut b = Vec::new();
    ciborium::into_writer(&m, &mut b).expect("manifest cbor");
    b
}

/// The guest-side runtime shims the `main!` exports delegate to (mirrors the v1 SDK's `rt`
/// byte-for-byte in semantics: raw `std::alloc` spans the host pairs with `da_free`).
pub mod rt {
    use std::alloc::{alloc, dealloc, Layout};

    fn layout(size: u32, align: u32) -> Layout {
        Layout::from_size_align(size as usize, (align.max(1)) as usize).expect("layout")
    }

    /// `da_alloc` body: a fresh guest span for host writes (ABI §2.4).
    #[must_use]
    pub fn da_alloc(size: u32, align: u32) -> u32 {
        if size == 0 {
            return 0;
        }
        // SAFETY: layout is non-zero-sized and validity-checked; the host pairs with `da_free`.
        let ptr = unsafe { alloc(layout(size, align)) };
        ptr as u32
    }

    /// `da_free` body: paired release (ABI §2.4).
    pub fn da_free(ptr: u32, size: u32, align: u32) {
        if ptr == 0 || size == 0 {
            return;
        }
        // SAFETY: `ptr`/`size`/`align` match a prior `da_alloc` (host obligation).
        unsafe { dealloc(ptr as *mut u8, layout(size, align)) };
    }

    /// Copy `bytes` into a fresh guest span and pack `(ptr << 32) | len` — the return form of
    /// `da_manifest`/`da_claim`; the host copies out then calls `da_free(ptr, len, 1)`.
    #[must_use]
    pub fn emit_cbor(bytes: &[u8]) -> u64 {
        let len = bytes.len();
        if len == 0 {
            return 0;
        }
        let ptr = da_alloc(len as u32, 1);
        // SAFETY: `ptr` is a fresh `len`-byte allocation; regions don't overlap.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, len) };
        ((ptr as u64) << 32) | len as u64
    }

    /// The longest forwarded panic line; a payload past this is truncated with an ellipsis. A
    /// panic message is a developer string, not a data channel — the cap keeps a runaway
    /// `{:?}` payload from turning the log sink into one.
    #[cfg(target_arch = "wasm32")]
    const PANIC_LINE_MAX: usize = 4096;

    /// The global allocator `main!` installs: [`std::alloc::System`], plus a report on the way out
    /// when it comes back empty-handed.
    ///
    /// An out-of-memory abort is the guest failure mode a panic hook CANNOT see. Rust routes a
    /// failed allocation through `handle_alloc_error`, which prints to a stderr the sandbox does
    /// not have and calls `abort()` — on wasm that is an `unreachable`, so the host gets a
    /// `GuestPanic` with no message and no location, indistinguishable from a failed assertion.
    /// It is also the failure a guest is MOST likely to hit: linear memory is capped at the
    /// module's own admitted claim (ABI §9.1), so any accidentally geometry-scaled buffer ends
    /// here.
    ///
    /// So the size is reported at the one moment it is still known: when the underlying allocator
    /// returns null, before the abort. The report allocates NOTHING — it formats into a stack
    /// buffer — because the one thing known for certain at that point is that allocation is
    /// failing.
    #[cfg(target_arch = "wasm32")]
    pub struct ReportingAlloc;

    #[cfg(target_arch = "wasm32")]
    // SAFETY: every method delegates to `System`, which is a valid `GlobalAlloc`; the wrapper adds
    // only a null check and a log call, and never touches the returned pointer.
    unsafe impl std::alloc::GlobalAlloc for ReportingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = std::alloc::System.alloc(layout);
            if ptr.is_null() {
                report_alloc_failure(layout);
            }
            ptr
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let ptr = std::alloc::System.alloc_zeroed(layout);
            if ptr.is_null() {
                report_alloc_failure(layout);
            }
            ptr
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let out = std::alloc::System.realloc(ptr, layout, new_size);
            if out.is_null() {
                report_alloc_failure(
                    Layout::from_size_align(new_size, layout.align()).unwrap_or(layout),
                );
            }
            out
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            std::alloc::System.dealloc(ptr, layout);
        }
    }

    /// Forward one exhausted allocation through `sys@2::log`, formatting into a stack buffer so
    /// the report itself needs no heap. Reports once: a guest whose allocator is failing will fail
    /// it again on the way to the abort, and one line is the diagnosis.
    #[cfg(target_arch = "wasm32")]
    fn report_alloc_failure(layout: Layout) {
        use std::sync::atomic::{AtomicBool, Ordering};

        static REPORTED: AtomicBool = AtomicBool::new(false);
        if REPORTED.swap(true, Ordering::Relaxed) {
            return;
        }
        let mut line = [0u8; 160];
        let mut n = 0;
        let mut put = |src: &[u8], n: &mut usize| {
            let room = line.len() - *n;
            let take = src.len().min(room);
            line[*n..*n + take].copy_from_slice(&src[..take]);
            *n += take;
        };
        put(daemon_vhc_abi::GUEST_PANIC_LOG_PREFIX.as_bytes(), &mut n);
        put(b"memory allocation of ", &mut n);
        put(decimal(layout.size() as u64).as_slice_of(), &mut n);
        put(
            b" bytes failed (linear memory is capped at the module's admitted claim)",
            &mut n,
        );
        // SAFETY: every byte written above came from ASCII sources.
        crate::abi::log(daemon_vhc_abi::LOG_LEVEL_ERROR, unsafe {
            std::str::from_utf8_unchecked(&line[..n])
        });
    }

    /// A `u64` rendered into a fixed 20-byte buffer — `u64::to_string` would allocate, and this
    /// runs when allocation has just failed.
    #[cfg(target_arch = "wasm32")]
    struct Decimal {
        buf: [u8; 20],
        start: usize,
    }

    #[cfg(target_arch = "wasm32")]
    impl Decimal {
        fn as_slice_of(&self) -> &[u8] {
            &self.buf[self.start..]
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn decimal(mut v: u64) -> Decimal {
        let mut buf = [0u8; 20];
        let mut start = buf.len();
        loop {
            start -= 1;
            buf[start] = b'0' + (v % 10) as u8;
            v /= 10;
            if v == 0 {
                break;
            }
        }
        Decimal { buf, start }
    }

    /// Arm panic forwarding for this instance: install a panic hook that pushes the message and
    /// its `file:line:col` out through `sys@2::log` before the panic runtime aborts the guest
    /// (ABI [`daemon_vhc_abi::GUEST_PANIC_LOG_PREFIX`]).
    ///
    /// Without this, a guest panic reaches the host as a bare wasm `unreachable` — a `GuestPanic`
    /// trap with no message, since the payload dies with the linear memory. The hook is the only
    /// moment the message is both formed and still reachable.
    ///
    /// `main!` calls this at the top of `da_run` and nowhere else, deliberately: capability
    /// imports are illegal during `da_init`/`da_migrate` (ABI §6.6), so logging from a hook armed
    /// there would re-class the guest's panic as a `PhaseViolation` and destroy the classification
    /// the forwarding exists to preserve.
    #[cfg(target_arch = "wasm32")]
    pub fn forward_panics() {
        use std::sync::Once;

        static ARMED: Once = Once::new();
        ARMED.call_once(|| {
            std::panic::set_hook(Box::new(|info| {
                let mut line = String::from(daemon_vhc_abi::GUEST_PANIC_LOG_PREFIX);
                if let Some(loc) = info.location() {
                    line.push_str(&format!("{}:{}:{}: ", loc.file(), loc.line(), loc.column()));
                }
                // The payload is `&str` for a literal panic and `String` for a formatted one;
                // anything else is a `panic_any` payload no message can be recovered from.
                let payload = info.payload();
                let message = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("<non-string panic payload>");
                // One log line: a host log record is line-oriented, and a panic message may carry
                // newlines (a multi-line `assert_eq!` is the common case).
                line.extend(message.chars().map(|c| if c == '\n' { ' ' } else { c }));
                if line.len() > PANIC_LINE_MAX {
                    line.truncate(
                        (0..=PANIC_LINE_MAX)
                            .rev()
                            .find(|&i| line.is_char_boundary(i))
                            .unwrap_or(0),
                    );
                    line.push('…');
                }
                crate::abi::log(daemon_vhc_abi::LOG_LEVEL_ERROR, &line);
            }));
        });
    }
}
