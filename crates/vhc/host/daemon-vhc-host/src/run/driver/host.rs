// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The guest-side store data + per-slice budget/legality state: the wasmtime `Store` payload
//! ([`Host`]), the §6.6 temporal-legality gate every import enters, the guest-memory borrow
//! helpers, the §12.1 signed-frame envelope builder, and the deterministic `sys@2` bodies
//! (identity-derived RNG seed, crypto accelerations) the linker and the replay engine share.

use std::sync::Arc;
use std::time::Duration;

use ciborium::value::Value;
use wasmtime::{Caller, Memory, StoreLimits};

use daemon_vhc_abi::FRAME_ENVELOPE_DOMAIN_V2;
use daemon_vhc_proto::{sign_canonical, to_canonical_vec, SigningKey};

use crate::run::driver::config::RunIdentity;
use crate::run::driver::pump::PumpShared;
use crate::trap::{Trap, TrapCode};

/// Per-slice budget/legality state (guest-thread-local — never behind the pump lock).
pub(crate) struct SliceState {
    /// Inside `da_init` (imports illegal, §6.6 rule 1).
    pub(crate) in_init: bool,
    /// Inside `da_migrate` (every import illegal EXCEPT `read_back(kind = 3)` — the one §6.6
    /// exception, §10.2).
    pub(crate) in_migrate: bool,
    /// `Stop` has been consumed (every import traps `PhaseViolation`, §4.4).
    pub(crate) stopped: bool,
    /// A `Quiesce` drain is open (freezes some behaviors, §4.4).
    pub(crate) draining: bool,
    /// The slice-constant logical `now()` (§6.5): the current slice's delivery timestamp.
    pub(crate) now: u64,
    /// Op-budget consumed this slice.
    pub(crate) op_calls: u64,
    /// Readback bytes consumed this slice.
    pub(crate) readback_bytes: u64,
    /// A pending mandatory `next_event` retry: the required capacity (§4.1).
    pub(crate) pending_next: Option<u64>,
    /// A pending mandatory `read_back` retry: `(src, kind, required)` (§6.4).
    pub(crate) pending_readback: Option<(u64, u32, u64)>,
    /// A pending mandatory `device_profile` retry: the required capacity (same §4.1/§6.4
    /// mandatory-retry discipline — the profile is delivered, and journaled, exactly once).
    pub(crate) pending_device: Option<u64>,
    /// The already-computed value behind a pending `read_back` retry (§6.4 "the staged value
    /// remains available"): the retry re-delivers the SAME value.
    pub(crate) pending_readback_value: Option<Vec<u8>>,
    /// Inside `da_run` (set once the run loop is entered).
    pub(crate) in_run: bool,
    /// The ordinal of the slice currently active, or `None` between slices.
    ///
    /// The four `da_run` states are distinct and a trap that is not inside a slice MUST NOT invent
    /// one: "before the first event", "between slices" and "after the last slice" are each different
    /// from "in slice n", and attributing a between-slices trap to the last slice that happened to
    /// run would point a reader at code that had already returned.
    pub(crate) slice_ordinal: Option<u64>,
    /// How many event slices have been delivered. Zero means the first event has not arrived, which
    /// is what distinguishes "before the first slice" from "between slices".
    pub(crate) slices_delivered: u64,
    /// Accepted `sys@2::log` calls in the current exempt phase (`[LX-6]`). Each exempt phase has its
    /// own independent budget, reset when the phase is entered.
    pub(crate) log_calls_this_phase: u64,
    /// Accepted `sys@2::log` bytes in the current exempt phase (`[LX-6]`).
    pub(crate) log_bytes_this_phase: u64,
    /// Monotonic count of import entries — the guest's **liveness signal** for the epoch watchdog.
    ///
    /// Every host-import entry increments it (in [`Host::enter`], the one seam every import
    /// passes). The run store's epoch-deadline callback compares it against
    /// [`Self::import_calls_at_epoch_check`]: a guest that has made host calls since the last
    /// expiry is WORKING (device compute happens inside imports and its wall belongs to the
    /// device, not to a wedge), so the deadline extends; a guest that spun a full epoch budget in
    /// pure wasm without one import is wedged and traps `BudgetEpoch`. The deterministic budgets
    /// (fuel, ops) are untouched — this only stops the WALL watchdog from killing a long
    /// device-lane slice that is demonstrably alive (the same principle as the `next_event`
    /// park re-arm, §5.6: never epoch-kill a guest for time it did not burn).
    pub(crate) import_calls: u64,
    /// The [`Self::import_calls`] value the epoch-deadline callback last observed.
    pub(crate) import_calls_at_epoch_check: u64,
}

impl SliceState {
    /// The execution context this state represents.
    ///
    /// One derivation, at the single point where lifecycle state is known, reused by the diagnostic
    /// tagging, the panic-detail lift, the trap classification and the journal — because a context
    /// derived twice is a context that can disagree with itself.
    pub(crate) fn execution_context(&self) -> daemon_vhc_abi::ExecutionContext {
        use daemon_vhc_abi::ExecutionContext;
        if self.in_init {
            return ExecutionContext::Init;
        }
        if self.in_migrate {
            return ExecutionContext::Migrate;
        }
        if !self.in_run {
            // The run instance exists but its loop has not been entered: the phase it is about to
            // enter is initialization.
            return ExecutionContext::Init;
        }
        if let Some(ordinal) = self.slice_ordinal {
            return ExecutionContext::RunSlice(ordinal);
        }
        // A consumed stop means no further slice can begin, so this is after the last one.
        if self.stopped {
            return ExecutionContext::RunAfterLastSlice;
        }
        if self.slices_delivered == 0 {
            ExecutionContext::RunBeforeFirstSlice
        } else {
            ExecutionContext::RunBetweenSlices
        }
    }
}

/// The wasmtime `Store` data for a v2 run instance.
pub(crate) struct Host {
    pub(crate) shared: Arc<PumpShared>,
    pub(crate) limits: StoreLimits,
    pub(crate) trap: Option<Trap>,
    pub(crate) slice: SliceState,
    // budgets (per-slice allowances)
    pub(crate) fuel_per_slice: u64,
    pub(crate) op_budget: u64,
    pub(crate) epoch_ticks: u64,
    pub(crate) max_readback_bytes: u64,
    pub(crate) max_frame_bytes: u32,
    // the claim's hard-accountable host-tier cap (standing, not per-slice — ABI §9.1/§5.5)
    pub(crate) hard_accountable_host_bytes: u64,
    pub(crate) accountable_staged_bytes: u64,
    // the migration grant (ABI §2.6): snapshot bounds on the producing side; the restore bit on
    // the consuming (migrating) side (`read_back(kind = 3)` legality, §10.2)
    pub(crate) migration_max_sections: u64,
    pub(crate) migration_max_section_bytes: u64,
    pub(crate) migration_restore: bool,
    // signing (§12.1)
    pub(crate) signing: SigningKey,
    pub(crate) identity: RunIdentity,
    pub(crate) sender: [u8; 32],
    // sys@2 ambient inputs: the admitted device-profile bytes (nondeterministic input — journaled
    // tag 15 per delivery) and the identity-derived RNG seed (deterministic — never journaled).
    pub(crate) device_bytes: Vec<u8>,
    pub(crate) rng_seed: [u8; 32],
    // data@2: the admitted artifact set ("which artifacts a module may touch is a grant") — the
    // envelope's edge-pinned artifact map ∩ the role's grants. Fail closed when empty.
    pub(crate) granted_artifacts: std::collections::BTreeSet<[u8; 32]>,
    // compute@2 (track C1, ABI §15): the per-instance command-queue runner over the ADMITTED
    // backend (`EngineConfig.backend` → ndarray/wgpu/cuda), guest-thread-local (device work
    // belongs to the guest thread, §11.1/§11.3 — the runner drops with the Store; GPU backends
    // additionally REQUIRE single-thread driving, which this placement provides by
    // construction). `None` when the module imports no compute@2 symbol.
    pub(crate) compute: Option<crate::compute::HostCompute>,
    // The queue-depth grant + its ledger: ops enqueued since the last successful fence.
    pub(crate) compute_queue_depth: u64,
    pub(crate) compute_ops_since_fence: u64,
    // The deferred-fault injection seam (see `RunConfig::compute_fault_after_ops`).
    pub(crate) compute_fault_after_ops: Option<u64>,
    pub(crate) compute_ops_total: u64,
}

impl Host {
    fn charge_op(&mut self, import: &'static str) -> Result<(), Trap> {
        self.slice.op_calls += 1;
        if self.slice.op_calls > self.op_budget {
            return Err(Trap::new(
                TrapCode::BudgetOps,
                import,
                None,
                "per-slice op budget exhausted",
            ));
        }
        Ok(())
    }

    /// The §6.6 temporal-legality gate + §4.1/§6.4 mandatory-retry enforcement, shared by every
    /// import. `is_next_event`/`is_read_back` let the two blocking imports pass their own retry.
    pub(crate) fn enter(&mut self, import: &'static str) -> Result<(), Trap> {
        // Liveness for the epoch watchdog (see [`SliceState::import_calls`]): counted BEFORE any
        // legality verdict — an import that is about to trap still proves the guest is executing,
        // and the trap it earns is the answer, not an epoch kill racing it.
        self.slice.import_calls = self.slice.import_calls.wrapping_add(1);
        if self.slice.stopped {
            return Err(Trap::new(
                TrapCode::PhaseViolation,
                import,
                None,
                "import after Stop was consumed (§4.4)",
            ));
        }
        // The §6.6 temporal-legality rules admit exactly ONE exemption: `sys@2::log`, in `da_init`
        // and `da_migrate`, on the run instance (`[LX-1]`/`[LX-2]`). It is observational — no
        // journal record, no state, no digest, no decision — so it introduces nothing the guest
        // could branch on. Every other capability import remains illegal in those phases.
        //
        // The reason this exemption exists at all: a guest that panics or exhausts its linear memory
        // during initialization or migration otherwise reaches the host as an ANONYMOUS trap, and
        // arming the forwarding without this exemption would convert the panic into a phase
        // violation and destroy the very classification the forwarding preserves.
        let log_exempt = import == "log";
        if self.slice.in_init && !log_exempt {
            return Err(Trap::new(
                TrapCode::PhaseViolation,
                import,
                None,
                "capability import during da_init (§6.6)",
            ));
        }
        if self.slice.in_migrate && !log_exempt && import != "read_back" {
            return Err(Trap::new(
                TrapCode::PhaseViolation,
                import,
                None,
                "only read_back(kind = state-section) is legal during da_migrate (§6.6/§10.2)",
            ));
        }
        // [LX-9]: a pending mandatory retry MUST NOT convert a `log` call into a protocol violation,
        // in ANY phase, and a `log` call never satisfies or clears a pending retry. A guest that
        // panics inside a retry window must still be able to name itself — and a capacity problem is
        // exactly when a panic is likely.
        if log_exempt {
            // [LX-7]: exempt-phase log calls are charged against the per-phase counters only, never
            // against the per-event-slice operation budget, which is a `da_run` concept with no
            // meaning before the first slice.
            if self.slice.in_init || self.slice.in_migrate {
                return Ok(());
            }
            return self.charge_op(import);
        }
        if self.slice.pending_next.is_some() && import != "next_event" {
            return Err(Trap::new(
                TrapCode::BadEvent,
                import,
                None,
                "NeedCapacity from next_event requires an immediate retry (§4.1)",
            ));
        }
        if self.slice.pending_readback.is_some() && import != "read_back" {
            return Err(Trap::new(
                TrapCode::BadEvent,
                import,
                None,
                "NeedCapacity from read_back requires an immediate retry (§6.4)",
            ));
        }
        if self.slice.pending_device.is_some() && import != "device_profile" {
            return Err(Trap::new(
                TrapCode::BadEvent,
                import,
                None,
                "NeedCapacity from device_profile requires an immediate retry (§6.4)",
            ));
        }
        self.charge_op(import)
    }
}

// -- memory helpers (Caller<Host>) ---------------------------------------------------------------

fn mem_of(caller: &mut Caller<'_, Host>) -> Result<Memory, Trap> {
    caller
        .get_export("memory")
        .and_then(wasmtime::Extern::into_memory)
        .ok_or_else(|| Trap::bare(TrapCode::BadModule, "module has no exported memory"))
}

pub(crate) fn read_guest(
    caller: &mut Caller<'_, Host>,
    ptr: u32,
    len: u32,
) -> Result<Vec<u8>, Trap> {
    let mem = mem_of(caller)?;
    let (start, end) = (ptr as usize, ptr as usize + len as usize);
    mem.data(&caller)
        .get(start..end)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| Trap::bare(TrapCode::MemOob, "guest span out of bounds"))
}

pub(crate) fn write_guest(
    caller: &mut Caller<'_, Host>,
    ptr: u32,
    bytes: &[u8],
) -> Result<(), Trap> {
    let mem = mem_of(caller)?;
    let start = ptr as usize;
    let data = mem.data_mut(caller);
    let end = start + bytes.len();
    data.get_mut(start..end)
        .ok_or_else(|| Trap::bare(TrapCode::MemOob, "guest span out of bounds"))?
        .copy_from_slice(bytes);
    Ok(())
}

pub(crate) fn stash<T>(
    caller: &mut Caller<'_, Host>,
    r: Result<T, Trap>,
) -> Result<T, wasmtime::Error> {
    r.map_err(|t| {
        let msg = t.to_string();
        caller.data_mut().trap = Some(t);
        wasmtime::Error::msg(msg)
    })
}

// -- the signed-frame envelope (§12.1) --------------------------------------------------------------

/// Build + sign the §12.1 domain-separated frame: `[envelope, payload, sig]` canonical CBOR, the
/// signature over the canonical envelope (which commits to the payload via `payload_hash`).
pub(crate) fn build_signed_frame(
    host: &Host,
    channel: u64,
    seq: u64,
    payload: &[u8],
) -> Result<Vec<u8>, Trap> {
    let payload_hash = blake3::hash(payload);
    let envelope = Value::Map(vec![
        (Value::from("domain"), Value::from(FRAME_ENVELOPE_DOMAIN_V2)),
        (
            Value::from("run_id"),
            Value::Bytes(host.identity.run_id.to_vec()),
        ),
        (Value::from("epoch"), Value::from(host.identity.epoch)),
        (
            Value::from("role"),
            Value::from(host.identity.role.as_str()),
        ),
        (Value::from("instance"), Value::from(host.identity.instance)),
        (
            Value::from("module"),
            Value::Bytes(host.identity.module.to_vec()),
        ),
        (Value::from("sender"), Value::Bytes(host.sender.to_vec())),
        (Value::from("channel"), Value::from(channel)),
        (Value::from("seq"), Value::from(seq)),
        (
            Value::from("payload_hash"),
            Value::Bytes(payload_hash.as_bytes().to_vec()),
        ),
    ]);
    let sig = sign_canonical(&host.signing, &envelope)
        .map_err(|e| Trap::bare(TrapCode::BadModule, format!("frame signing: {e}")))?;
    let frame = Value::Array(vec![
        envelope,
        Value::Bytes(payload.to_vec()),
        Value::Bytes(sig.0.to_vec()),
    ]);
    to_canonical_vec(&frame)
        .map_err(|e| Trap::bare(TrapCode::BadModule, format!("frame encoding: {e}")))
}

// -- sys@2 seeded deterministic randomness (architecture §3.2 "seeded randomness") ------------------

/// Derive the run-scoped RNG seed for one execution identity (`sys@2::rng_seed`): a **pure
/// function of the frozen §8.1 identity**, domain-separated under
/// [`daemon_vhc_abi::RNG_SEED_DOMAIN_V2`]. Deterministic per the §2.7 `dc` class — the import
/// carries **no journal record**; replay re-derives the identical seed from the run header's
/// identity (see [`crate::run::replay`]). Two role-instances never share a seed; a trap-restart of the
/// same incarnation reproduces it (the seed is an *identity* property, not an *instantiation*
/// property — restarted policy must be able to re-derive its own randomness).
#[must_use]
pub fn derive_rng_seed(identity: &RunIdentity) -> [u8; 32] {
    // Unambiguous concatenation: fixed-width fields + a length prefix on the one variable field.
    let mut material = Vec::with_capacity(32 + 8 + 4 + identity.role.len() + 8 + 32);
    material.extend_from_slice(&identity.run_id);
    material.extend_from_slice(&identity.epoch.to_le_bytes());
    material.extend_from_slice(&(identity.role.len() as u32).to_le_bytes());
    material.extend_from_slice(identity.role.as_bytes());
    material.extend_from_slice(&identity.instance.to_le_bytes());
    material.extend_from_slice(&identity.module);
    blake3::derive_key(daemon_vhc_abi::RNG_SEED_DOMAIN_V2, &material)
}

// -- sys@2 crypto accelerations (the det/crypto-lane fast path, §3.2/§3.7) --------------------------

/// The host `sys@2::hash` acceleration body: blake3-256 over `data`, pinned by the dual-compiled
/// [`daemon_vhc_proto::crypto`] contract. Because the in-guest fallback is that *same* contract
/// compiled to wasm, host-op ≡ in-guest-op is bit-exact **by construction** (architecture §3.2, the
/// det-lane pattern). Exposed (crate-public) so the tier-1 conformance gate exercises the exact
/// body the live import runs. Deterministic → the import carries **no journal record** (§2.7 `dc`
/// class); replay re-executes it (see [`crate::run::replay`]).
#[must_use]
pub fn host_crypto_hash(data: &[u8]) -> [u8; daemon_vhc_proto::HASH_LEN] {
    daemon_vhc_proto::crypto_hash(data)
}

/// The host `sys@2::verify_sig` acceleration body: the ABI status code of the tri-state
/// [`daemon_vhc_proto::VerifyOutcome`] (0 = valid, 1 = invalid, 2 = malformed). Same
/// dual-compiled-contract / by-construction-parity story as [`host_crypto_hash`]; deterministic,
/// not journaled.
#[must_use]
pub fn host_crypto_verify(public_key: &[u8], signature: &[u8], message: &[u8]) -> u32 {
    daemon_vhc_proto::verify_sig(public_key, signature, message).code()
}

/// How long a parked `next_event` waits between wake checks when no timer bounds the wait.
pub(crate) const PARK_RECHECK: Duration = Duration::from_millis(50);

/// The pump shared state behind a caller (borrow helper for the import bodies).
pub(crate) fn shared_of(c: &Caller<'_, Host>) -> Arc<PumpShared> {
    c.data().shared.clone()
}
