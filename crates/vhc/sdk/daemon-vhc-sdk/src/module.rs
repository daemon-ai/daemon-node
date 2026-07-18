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
    /// Long-lived host-accountable state bytes (params, queues, config) — the module's floor.
    pub host_state_bytes: u64,
    /// Transient host scratch above the state floor (decode buffers, staging copies).
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
/// - `hard_accountable` = the declared state floor (page-rounded) — what the host must reserve;
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
    let claim = ciborium::value::Value::Map(vec![
        (
            text("hard_accountable"),
            tier(
                round_page(decl.device_state_bytes),
                round_page(decl.host_state_bytes),
            ),
        ),
        (
            text("declared_peak"),
            tier(
                round_page(decl.device_state_bytes + decl.device_scratch_bytes),
                round_page(decl.host_state_bytes + decl.host_scratch_bytes),
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
}
