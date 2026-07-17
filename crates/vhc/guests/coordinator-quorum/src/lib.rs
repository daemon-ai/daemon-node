// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `coordinator-quorum` — the launch coordinator module (architecture §4.1; refactor §8/D2).
//!
//! Consensus is a wasm module, not a native host service. This guest is a **deterministic reactive
//! state machine over signed control frames and a logical clock** — exactly the shape the
//! architecture (§4.1) and Psyche's re-hostable coordinator prove. Its body is
//! [`daemon_vhc_sdk_consensus::coordinator::tick`] — the SAME state machine the native
//! dual-compilation reference runs, relocated into the consensus SDK at D2 when the native
//! `daemon-vhc-coordinator` crate dissolved. Running the identical `tick` under wasm32 and native
//! on identical inputs is the D2 dual-compilation identity gate.
//!
//! ## The event loop (ABI §3.1, Phase-A closed subset)
//!
//! - **`Frame`** — a worker's control message (`Join`/`Commitment`/`StorageReceipt`/…), delivered
//!   host-verified (the host authenticated the §12 signed-frame envelope above the sandbox; the
//!   guest sees the authenticated `sender` + opaque payload, never a re-checkable signature). The
//!   guest accepts frames **only on the declared authoritative records channel**, mints D1's
//!   [`Authorized`] token for that host-verified delivery
//!   (`Authorized::from_authoritative_channel` — the bridge path of the D1 contract), and feeds
//!   the decoded [`SwarmMessage`] through [`tick_authenticated`] with it.
//! - **`Timer`** — drives the logical clock for deadline-based transitions (warmup / round
//!   timeouts) in a live run.
//!
//! ## Deterministic logical clock
//!
//! `tick`'s phase transitions split into two classes: **event-driven fast paths**
//! (all-committed → witness, all-evidenced → record, ready-heartbeat → open) that need no clock,
//! and **timeout paths** that compare `now` against a phase deadline. Time enters `tick` only as
//! `Input::Clock`, so this loop advances a **synthetic monotonic clock one tick per delivered
//! event** (`tick_period_ms == 0`, the deterministic default the D2 gate uses) — the value is a
//! pure function of the event count, never wall-clock, so wasm ≡ native holds. A live deployment
//! sets `tick_period_ms > 0` to also arm real deadline timers.

use daemon_vhc_abi::{EV_TAG_FRAME, EV_TAG_QUIESCE, EV_TAG_STOP, EV_TAG_TIMER};
use daemon_vhc_proto::{from_canonical_slice, to_canonical_vec, Hash, PeerId, SwarmMessage};
use daemon_vhc_sdk_consensus::coordinator::{
    tick, tick_authenticated, CoordinatorState, Input, Output,
};
use daemon_vhc_sdk_consensus::{Authorized, DEFAULT_RECORDS_CHANNEL};
use daemon_vhc_sdk_v2::migrate::{
    build_manifest, MigrationDescriptor, OwnedSection, SectionReader,
};
use daemon_vhc_sdk_v2::module::{ModuleDecl, V2Module};
use serde::Deserialize;

/// The one state-manifest section a coordinator snapshot declares: its whole `CoordinatorState`,
/// canonical-CBOR, consensus-canonical (`class` 0). Named so `da_migrate` can bind it on restore.
const STATE_SECTION: &str = "consensus";

/// The coordinator state-schema version (ABI §10.2 `state-section-decl.schema`).
const STATE_SCHEMA: u64 = 1;

/// `da_run` outcome for a completed `Quiesce` drain (ABI §4.5): the module snapshotted and is ready
/// to be torn down / migrated from.
const OUTCOME_QUIESCE_READY: u32 = 2;

/// `da_migrate` status for a descriptor the coordinator cannot consume (ABI §10.2 `Incompatible`).
const MIGRATE_INCOMPATIBLE: u32 = 1;

/// The module's `da_init` config (canonical CBOR): the resolved initial coordinator state plus the
/// two guest-runtime knobs. The native reference consumes only `state` — the knobs are wasm-loop
/// concerns the pure `tick` never sees, so identity is preserved.
#[derive(Deserialize)]
struct CoordinatorInit {
    /// The resolved initial coordinator state (config + genesis roster/seed/clock).
    state: CoordinatorState,
    /// Real-timer period in logical ms; `0` = the deterministic event-driven clock (the D2 gate).
    #[serde(default)]
    tick_period_ms: u64,
    /// The control channel `RoundOpen`/`RoundRecord` are published on (§6.2). Default `0`.
    #[serde(default)]
    control_channel: u32,
}

/// The coordinator module: the pure `tick` state machine plus the wasm event-loop plumbing.
struct Coordinator {
    state: CoordinatorState,
    now_s: u64,
    tick_period_ms: u64,
    control_channel: u32,
}

impl Coordinator {
    /// Advance the synthetic logical clock one tick and run the timeout-checking `tick(Clock)`.
    fn advance_clock(&mut self) {
        self.now_s += 1;
        let (next, outputs) = tick(self.state.clone(), Input::Clock(self.now_s));
        self.state = next;
        self.emit(&outputs);
    }

    /// Snapshot the whole consensus state through the sandbox on a `Quiesce` drain (§10.2): the
    /// same typed quiesce→`StateManifest` path the trainer guest uses. `CoordinatorState` is small
    /// in-memory CBOR with no device residency, so the snapshot is synchronous — one
    /// consensus-canonical section staged, then the manifest submitted. A standby/restart
    /// re-instantiates from the accepted snapshot via [`Coordinator::migrate`] and continues the
    /// same logical timeline bit-identically (its `now_s`/round ring are restored verbatim).
    ///
    /// Returns the `da_run` outcome: `OUTCOME_QUIESCE_READY` once the host accepts the manifest, or
    /// a nonzero module-defined code if serialization or the submission is refused (fail loud).
    fn snapshot_and_ready(&self) -> u32 {
        let Ok(state_bytes) = to_canonical_vec(&self.state) else {
            return MIGRATE_INCOMPATIBLE;
        };
        let section = OwnedSection {
            name: STATE_SECTION.to_string(),
            schema: STATE_SCHEMA,
            class: 0, // consensus-canonical (the published-decision-bearing state)
            bytes: state_bytes,
        };
        let _staging_id = daemon_vhc_sdk_v2::stage_state(&section.bytes);
        // `module` is zeroed — a module cannot hash its own bytes; the host verifies the section
        // by content hash before staging (the §10.2 discipline the trainer follows too).
        let manifest = build_manifest(
            Hash([0u8; 32]),
            STATE_SCHEMA,
            std::slice::from_ref(&section),
        );
        let Ok(manifest_bytes) = to_canonical_vec(&manifest) else {
            return MIGRATE_INCOMPATIBLE;
        };
        if daemon_vhc_sdk_v2::snapshot_state(&manifest_bytes) != 0 {
            return MIGRATE_INCOMPATIBLE;
        }
        OUTCOME_QUIESCE_READY
    }

    /// Publish every `RoundOpen`/`RoundRecord` the coordinator produced, in order (§6.2). Notes and
    /// rejects are advisory and not published (a live embedder may surface them as metrics).
    fn emit(&self, outputs: &[Output]) {
        for o in outputs {
            if let Output::Publish(msg) = o {
                if let Ok(bytes) = to_canonical_vec(&**msg) {
                    daemon_vhc_sdk_v2::abi::publish(self.control_channel, &bytes);
                }
            }
        }
    }
}

impl V2Module for Coordinator {
    fn decl() -> ModuleDecl {
        ModuleDecl {
            name: "coordinator-quorum",
            version: env!("CARGO_PKG_VERSION"),
            // Phase-A closed subset only: next_event + publish + set_timer (no compute@, no data@).
            abi_minor: 0,
            // The control channel it publishes RoundOpen/RoundRecord on (§6.2).
            channels: vec![0],
            // Small host-accountable state (the round ring + roster); no device residency.
            host_state_bytes: 1 << 16,
            host_scratch_bytes: 1 << 16,
            device_state_bytes: 0,
            device_scratch_bytes: 0,
        }
    }

    fn init(config: &[u8], _grants: &[u8]) -> Result<Self, u32> {
        // A malformed config is a module-defined init refusal (§9.4 step 11, detail >= 16).
        let init: CoordinatorInit = from_canonical_slice(config).map_err(|_| 16u32)?;
        // Resume the synthetic clock from the restored state's clock (0 for a fresh run): a
        // standby coordinator reconstructed from the archive + journal (refactor §8/D2 failover)
        // continues the SAME logical timeline, so its decisions (e.g. `RoundOpen.deadline_unix_s`,
        // a function of `state.now_s`) are byte-identical to an uninterrupted run's.
        let now_s = init.state.now_s;
        Ok(Coordinator {
            state: init.state,
            now_s,
            tick_period_ms: init.tick_period_ms,
            control_channel: init.control_channel,
        })
    }

    /// The §10.2 consuming protocol (the standby/restart re-instantiation): read back the single
    /// `consensus` state-manifest section the old instance's `snapshot_and_ready` staged and
    /// replace this fresh instance's state with it. `da_init` already ran (with the genesis config,
    /// so `run` has a valid `proto_version` etc.); this overrides its state — and the resumed
    /// `now_s` — with the exported one, so the module continues the same logical timeline and its
    /// published decisions are byte-identical to an uninterrupted run's.
    fn migrate(&mut self, descriptor: &MigrationDescriptor, reader: &mut dyn SectionReader) -> u32 {
        let mut restored: Option<CoordinatorState> = None;
        for binding in &descriptor.sections {
            if binding.name != STATE_SECTION {
                return MIGRATE_INCOMPATIBLE;
            }
            let bytes = reader.read(binding.staging_id);
            match from_canonical_slice::<CoordinatorState>(&bytes) {
                Ok(s) => restored = Some(s),
                Err(_) => return MIGRATE_INCOMPATIBLE,
            }
        }
        let Some(state) = restored else {
            return MIGRATE_INCOMPATIBLE;
        };
        self.now_s = state.now_s;
        self.state = state;
        0
    }

    fn run(&mut self) -> u32 {
        let proto_version = self.state.config.proto_version;
        let mut buf: Vec<u8> = Vec::with_capacity(512);
        if self.tick_period_ms > 0 {
            daemon_vhc_sdk_v2::abi::set_timer(self.tick_period_ms);
        }
        loop {
            let ev = daemon_vhc_sdk_v2::abi::next_event(&mut buf);
            match ev.tag {
                t if t == EV_TAG_FRAME => {
                    let channel = ev.uint(1);
                    let sender = ev.bytes(3);
                    let payload = ev.bytes(4);
                    // D1's Authority vocabulary (the reconciled seam): a delivered Frame on the
                    // declared authoritative records channel was signature-verified above the
                    // pump — exactly the provenance `Authorized::from_authoritative_channel`
                    // encodes. Frames on any other channel carry no record authority and never
                    // reach the tick (ignoring a delivered event is module policy, §5.2).
                    if channel == u64::from(DEFAULT_RECORDS_CHANNEL) {
                        let authorized =
                            Authorized::from_authoritative_channel(DEFAULT_RECORDS_CHANNEL);
                        if let Ok(arr) = <[u8; 32]>::try_from(sender.as_slice()) {
                            if let Ok(msg) = from_canonical_slice::<SwarmMessage>(&payload) {
                                let (next, outputs) = tick_authenticated(
                                    self.state.clone(),
                                    PeerId(arr),
                                    proto_version,
                                    msg,
                                    authorized,
                                );
                                self.state = next;
                                self.emit(&outputs);
                            }
                        }
                    }
                    // Event-driven synthetic clock: one tick per delivered frame so the
                    // clock-only transition (WaitingForMembers -> Warmup) and any timeout run
                    // deterministically without a wall-clock timer (module docs).
                    self.advance_clock();
                }
                t if t == EV_TAG_TIMER => {
                    self.advance_clock();
                    if self.tick_period_ms > 0 {
                        daemon_vhc_sdk_v2::abi::set_timer(self.tick_period_ms);
                    }
                }
                // Terminal: a clean stop returns Ok promptly (no un-snapshotted durable state).
                t if t == EV_TAG_STOP => return 0,
                // Drain: snapshot the consensus state through the sandbox (§10.2) so a standby /
                // restart re-instantiates from the exported state, then return QuiesceReady.
                t if t == EV_TAG_QUIESCE => return self.snapshot_and_ready(),
                // Advisory / unknown-but-delivered events: ignore (only unknown TAGS fail closed,
                // which the SDK `next_event` decoder already enforces, §5.2).
                _ => {}
            }
        }
    }
}

daemon_vhc_sdk_v2::main!(Coordinator);
