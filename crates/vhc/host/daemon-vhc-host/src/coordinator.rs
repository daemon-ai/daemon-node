// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The coordinator half of a whole run (refactor §8/D2; decisions D3 the refusal and end-state combinations): configure
//! the production `coordinator_quorum.wasm` blob **from a frozen envelope**, run it under the real
//! major-2 event-loop driver, and route frames to/from it.
//!
//! This is PRODUCTION-SHAPED machinery (consensus never runs outside the sandboxed,
//! content-addressed module — even in verification): the worker's in-process self-driven join
//! drives its run's coordinator through this seat, and the testkit's whole-run harness
//! re-exports it (one copy, no fork). It supersedes the native-coordinator drive shells for
//! every whole-run lane. It is nonetheless **HARNESS-GATED**
//! (`#[cfg(any(test, feature = "harness"))]`): it decodes SDK round decisions, and no default
//! (production) host build may link a round schema (dep-check-enforced) — the shipped worker
//! reaches consensus through the role session's opaque-frame path instead.
//!
//! ## The configuration seat and the matrix refusals (decisions D3)
//!
//! A coordinator can only run under an envelope that **configures** it: its module hash must
//! be pinned in the role set and its `Authority` (launch: the `SingleKey` coordinator identity)
//! named in `[identities]` — both envelope-v2 (genesis) features. [`configure_coordinator`]
//! is that seat: it sniffs the frozen envelope bytes' schema major and
//!
//! - **schema 1 (envelope v1)** → the typed refusal [`CoordError::EnvelopeCannotConfigure`]
//!   — there is no coordinator role entry to pin a module hash, and no `Authority`/identities
//!   section to name the signer. This is the **v1-worker/envelope-v1 refusal and v2-worker/envelope-v1 refusal** negative (v1 or v2 workers make
//!   no difference: the coordinator is unconfigurable before any worker is considered).
//! - **schema 2 (genesis)** → derive the [`CoordinatorSpec`]: the coordinator role's pinned
//!   module hash, its **verbatim opaque config bytes** (the host never interprets them — they are
//!   byte-identically the guest's `da_init` input, architecture §5.1 seam rule), and the run's
//!   declared [`AuthorityConfig`] decoded from the opaque `authority` section. This is **cell
//!   8**'s configuration half.
//!
//! This derivation is harness-level infrastructure: the *production* admission seat for the
//! genesis join flow (threading `RoleGrants` through `admit`, certified keys) is D1's landed
//! work — this module deliberately duplicates none of it.
//!
//! ## The `Authority` seam (reconciled onto D1's contract — sitting 3)
//!
//! [`CoordinatorSpec::authority`] is D1's typed `AuthorityConfig`; every judgment goes through
//! `AuthorityConfig::authorize` — the network seat's frame check
//! ([`authorize_coordinator_frame`]), the archive's head check, and the guest-side
//! `tick_authenticated` token. The pre-D1 identity-comparison stubs are gone.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ciborium::value::Value;

use daemon_vhc_abi::CandidateDriver;
use daemon_vhc_proto::{
    from_canonical_slice, peek_schema, to_canonical_vec, FrozenGenesis, Hash, GENESIS_SCHEMA_MAJOR,
};
use daemon_vhc_sdk_consensus::coordinator::CoordinatorState;
use daemon_vhc_sdk_consensus::{AuthorityConfig, RecordSig};
use daemon_vhc_sdk_consensus::{SignedMessage, VhcMessage};

use crate::run::{
    start_run_migrating, DeliverVerdict, MemorySink, MigrationInput, Run, RunConfig, RunEnd,
    RunIdentity, SinkEntry, SnapshotCapture,
};
use crate::{select_driver, EngineConfig, Worker};

/// The state-manifest section the `coordinator-quorum` module snapshots its whole
/// `CoordinatorState` into on `Quiesce` (must match the guest's `STATE_SECTION`).
const COORDINATOR_STATE_SECTION: &str = "consensus";

/// Typed refusals from the coordinator configuration seat (decisions D3 the envelope-v1 refusal combinations negatives
/// + end-state well-formedness).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CoordError {
    /// **Cells 3/7 (decisions D3):** the envelope cannot configure a coordinator — a
    /// schema-major-1 envelope has no coordinator role entry (no module-hash pin) and no
    /// `Authority`/identities section. The typed refusal that pushes an author to a v2 envelope.
    #[error(
        "envelope schema major {0} cannot configure a coordinator: no coordinator role \
         entry to pin a module hash, no Authority/identities section to name the signer \
         (decisions D3 the envelope-v1 refusal combinations — author a genesis envelope v2)"
    )]
    EnvelopeCannotConfigure(u32),
    /// The frozen bytes carry no recognizable `[run].schema` major.
    #[error("frozen envelope bytes carry no recognizable schema major")]
    UnknownSchema,
    /// A genesis envelope without a coordinator-lane role (validation admits label-only checks;
    /// the coordinator seat needs the lane selector).
    #[error("genesis envelope has no role with lane `coordinator`")]
    NoCoordinatorRole,
    /// The coordinator role's module key is missing from the artifact map (envelope authoring).
    #[error("coordinator role module `{0}` absent from the genesis artifact map")]
    ModuleUnpinned(String),
    /// The genesis envelope's opaque `authority` section does not decode as D1's typed
    /// `AuthorityConfig` — the run declares no usable trust topology, so a coordinator's
    /// records could never be judged (architecture §4.2/§5.1).
    #[error("genesis authority section does not decode: {0}")]
    NoAuthority(String),
    /// The genesis bytes failed to decode/verify.
    #[error("genesis decode: {0}")]
    Genesis(String),
}

/// What a genesis envelope configures a coordinator with (the end-state configuration half).
#[derive(Debug, Clone)]
pub struct CoordinatorSpec {
    /// The coordinator role's pinned module blake3 (the artifact-map pin the blob must match).
    pub module_hash: Hash,
    /// The role's **verbatim** opaque config bytes — byte-identically the guest's `da_init`
    /// input; the host never decodes them (architecture §5.1 seam rule).
    pub config_bytes: Vec<u8>,
    /// The run's declared trust topology — D1's typed [`AuthorityConfig`], decoded from the
    /// genesis envelope's opaque `authority` section (the reconciled seam: judgments go through
    /// [`AuthorityConfig::authorize`], never an identity comparison).
    pub authority: AuthorityConfig,
    /// The run's cryptographic `RunId` (the genesis hash).
    pub run_id: Hash,
}

/// Configure a coordinator from frozen envelope bytes (see module docs). Schema-major
/// routing per `peek_schema`: v1 → the typed cells-3/7 refusal; v2 → the derived spec.
///
/// # Errors
/// A typed [`CoordError`]; the v1 case is the ratified matrix negative.
pub fn configure_coordinator(frozen: &FrozenGenesis) -> Result<CoordinatorSpec, CoordError> {
    let env = frozen
        .decode()
        .map_err(|e| CoordError::Genesis(e.to_string()))?;
    let (role_name, role) = env
        .roles
        .iter()
        .find(|(_, r)| r.lane == "coordinator")
        .ok_or(CoordError::NoCoordinatorRole)?;
    let module = env
        .artifacts
        .get(&role.module)
        .ok_or_else(|| CoordError::ModuleUnpinned(role.module.clone()))?;
    let authority = AuthorityConfig::decode(&env.authority)
        .map_err(|e| CoordError::NoAuthority(e.to_string()))?;
    let config_bytes = frozen
        .role_config_bytes(role_name)
        .map_err(|e| CoordError::Genesis(e.to_string()))?
        .expect("role name came from the decoded role set");
    Ok(CoordinatorSpec {
        module_hash: module.blake3,
        config_bytes,
        authority,
        run_id: *frozen.run_id(),
    })
}

/// The schema-major gate in front of [`configure_coordinator`]: raw frozen-envelope bytes of
/// EITHER schema. A schema-major-1 envelope is the typed cells-3/7 refusal — refused **before**
/// any worker-ABI consideration (the coordinator is unconfigurable regardless of the worker axis).
///
/// # Errors
/// [`CoordError::EnvelopeCannotConfigure`] for schema 1; [`CoordError::UnknownSchema`]
/// for unrecognizable bytes.
pub fn refuse_unconfigurable_envelope(bytes: &[u8]) -> Result<(), CoordError> {
    match peek_schema(bytes) {
        Some(GENESIS_SCHEMA_MAJOR) => Ok(()),
        Some(major) => Err(CoordError::EnvelopeCannotConfigure(major)),
        None => Err(CoordError::UnknownSchema),
    }
}

/// The network seat's record-authority judgment over one §12.1 signed frame — the reconciled D2
/// seam (formerly a sender-identity comparison): the frame envelope's canonical bytes are the
/// signed preimage and its `(sender, sig)` the presented [`RecordSig`], judged through
/// [`AuthorityConfig::authorize`]. Returns the authenticated sender + D1's `Authorized` token.
///
/// Layering note: a §12.1 transport frame carries exactly one host-oracle signature, so this seat
/// authorizes under `SingleKey`; a `ThresholdKeys` run's record quorum rides **record-level**
/// signature sets judged at the archive/head seam ([`daemon_vhc_sdk_consensus::Authority`]) — a
/// single transport signature correctly FAILS a threshold topology here (sub-quorum), which the
/// adversarial suite pins.
///
/// # Errors
/// The topology's typed `AuthError`, or a `Config` error for a malformed frame.
pub fn authorize_coordinator_frame(
    config: &AuthorityConfig,
    frame_bytes: &[u8],
) -> Result<([u8; 32], daemon_vhc_sdk_consensus::Authorized), daemon_vhc_sdk_consensus::AuthError> {
    let malformed = |reason: &str| daemon_vhc_sdk_consensus::AuthError::Config {
        reason: reason.to_string(),
    };
    let v: Value = ciborium::de::from_reader(frame_bytes)
        .map_err(|e| malformed(&format!("frame decode: {e}")))?;
    let Value::Array(parts) = &v else {
        return Err(malformed("frame is not [envelope, payload, sig]"));
    };
    let (Some(envelope), Some(Value::Bytes(sig))) = (parts.first(), parts.get(2)) else {
        return Err(malformed("frame missing envelope or signature"));
    };
    let sender = frame_sender(&v).map_err(|e| malformed(&e))?;
    let preimage =
        to_canonical_vec(envelope).map_err(|e| malformed(&format!("envelope encode: {e}")))?;
    let sig64: [u8; 64] = sig
        .as_slice()
        .try_into()
        .map_err(|_| daemon_vhc_sdk_consensus::AuthError::Malformed)?;
    let rs = RecordSig {
        signer: daemon_vhc_proto::PeerId(sender),
        sig: daemon_vhc_proto::Signature(sig64),
    };
    let token = config.authorize(&preimage, &[rs])?;
    Ok((sender, token))
}

/// A production `coordinator_quorum.wasm` blob running under the real major-2 driver, with the
/// frame routing a whole run needs: deliver authenticated worker messages in, poll decoded
/// coordinator decisions out. The harness plays the network seat (sign → verify above the pump →
/// deliver, per-sender dense seqs) exactly as the testkit barrier harness does for workers.
pub struct Coordinator {
    pump: crate::run::PumpHandle,
    run: Option<Run>,
    /// Keep the engine alive for the run's lifetime.
    _engine: Worker,
    sink: Arc<Mutex<MemorySink>>,
    /// Per-sender dense delivery seq (§12.2 discipline, channel 0).
    seqs: BTreeMap<[u8; 32], u64>,
    /// How many published frames have been consumed by [`Coordinator::next_message`].
    consumed: usize,
}

impl Coordinator {
    /// Start the coordinator blob under the event-loop driver: verify the blob against the spec's
    /// pin, select (must be major-2), and `start_run` with the spec's verbatim config bytes.
    ///
    /// `instance` is the role-instance incarnation (0 for a primary; a standby mints a fresh one,
    /// §8.1 never-reused). `key_seed` seeds the §12.1 frame signer — for the primary this is the
    /// key behind the envelope-named `SingleKey` identity ([`CoordinatorSpec::authority`]);
    /// a standby signing under a *different* key is exactly the signer-transfer gap the
    /// `Authority` contract owns (architecture §4.4 — D1/sitting 3).
    ///
    /// # Errors
    /// A `String` on selection/start failure (harness-level).
    pub fn start(
        wasm: &[u8],
        spec: &CoordinatorSpec,
        grants: Vec<u8>,
        instance: u64,
        key_seed: [u8; 32],
    ) -> Result<Self, String> {
        Self::start_inner(wasm, spec, grants, instance, key_seed, None)
    }

    /// Re-instantiate the coordinator module from a state-export (the standby / restart path,
    /// architecture §4.4; ABI §10.3 step 4). `capture` is the accepted snapshot a prior instance's
    /// [`Coordinator::quiesce_snapshot`] produced (its whole `CoordinatorState` as one
    /// consensus-canonical section): `da_init` runs with the spec's genesis config, then
    /// `da_migrate(descriptor)` restores that state through the sandbox — so the resumed module
    /// continues the same logical timeline and publishes byte-identical decisions. This is the
    /// re-seated substitute for reconstructing `CoordinatorState` by folding the pure native `tick`.
    ///
    /// # Errors
    /// A `String` on selection/start/migrate failure (harness-level).
    pub fn start_migrating(
        wasm: &[u8],
        spec: &CoordinatorSpec,
        grants: Vec<u8>,
        instance: u64,
        key_seed: [u8; 32],
        capture: SnapshotCapture,
    ) -> Result<Self, String> {
        Self::start_inner(
            wasm,
            spec,
            grants,
            instance,
            key_seed,
            Some(MigrationInput {
                capture,
                restore: true,
                migrate_fuel: None,
                carried_state: Vec::new(),
            }),
        )
    }

    fn start_inner(
        wasm: &[u8],
        spec: &CoordinatorSpec,
        grants: Vec<u8>,
        instance: u64,
        key_seed: [u8; 32],
        migration: Option<MigrationInput>,
    ) -> Result<Self, String> {
        let module_hash = *blake3::hash(wasm).as_bytes();
        if module_hash != spec.module_hash.0 {
            return Err(format!(
                "coordinator blob does not match the envelope pin: {} != {}",
                Hash(module_hash).to_hex(),
                spec.module_hash.to_hex()
            ));
        }
        let engine = Worker::new(EngineConfig::default()).map_err(|e| format!("engine: {e}"))?;
        let sel = select_driver(&engine, wasm, Some(&module_hash))
            .map_err(|e| format!("coordinator selection: {e}"))?;
        if sel.driver != CandidateDriver::V2 {
            return Err(format!(
                "coordinator-quorum must select the major-2 driver, got {:?}",
                sel.driver
            ));
        }
        let identity = RunIdentity {
            run_id: spec.run_id.0,
            epoch: 0,
            role: "coordinator".to_string(),
            instance,
            module: module_hash,
        };
        let sink = Arc::new(Mutex::new(MemorySink::new()));
        let run_cfg = RunConfig::new(identity, key_seed, spec.config_bytes.clone(), grants);
        let run = start_run_migrating(&engine, wasm, run_cfg, Box::new(sink.clone()), migration)
            .map_err(|e| format!("coordinator start_run: {e}"))?;
        Ok(Self {
            pump: run.pump.clone(),
            run: Some(run),
            _engine: engine,
            sink,
            seqs: BTreeMap::new(),
            consumed: 0,
        })
    }

    /// Deliver one worker control message as a host-verified authoritative frame: sign it with the
    /// sender's key (the wire evidence), verify above the pump, deliver with a per-sender dense
    /// seq — back-pressuring (never dropping) on `SpoolFull`/`SenderQuota` per §4.7.
    ///
    /// # Errors
    /// A `String` on sign/deliver failure or a persistent back-pressure timeout.
    pub fn deliver(
        &mut self,
        key: &daemon_vhc_proto::SigningKey,
        msg: &VhcMessage,
    ) -> Result<(), String> {
        let signed = SignedMessage::sign(key, daemon_vhc_proto::VHC_PROTO_VERSION, msg.clone())
            .map_err(|e| format!("sign: {e}"))?;
        self.deliver_signed(&signed)
    }

    /// Deliver a **pre-signed** frame verbatim, preserving its original `signer` (the §12.1 sender).
    /// Unlike [`Coordinator::deliver`] this does not re-sign — so a recorded `SignedMessage`
    /// (e.g. a journal-captured `Input::Message`) replays into the module with its authentic sender,
    /// which is exactly what reconstructing consensus state from recorded inputs requires (a
    /// re-signed frame would change the sender and break per-peer round accounting).
    ///
    /// # Errors
    /// A `String` on verify/encode/deliver failure or a persistent back-pressure timeout.
    pub fn deliver_signed(&mut self, signed: &SignedMessage) -> Result<(), String> {
        signed
            .verify()
            .map_err(|e| format!("frame REFUSED above the pump: {e}"))?;
        let sender = signed.signer.0;
        let payload =
            to_canonical_vec(&signed.payload).map_err(|e| format!("payload encode: {e}"))?;
        let evidence = to_canonical_vec(signed).map_err(|e| format!("evidence encode: {e}"))?;
        let seq = self.seqs.entry(sender).or_insert(0);
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match self
                .pump
                .deliver_frame(0, *seq, sender, payload.clone(), evidence.clone())
                .map_err(|e| format!("deliver: {e}"))?
            {
                DeliverVerdict::Accepted => break,
                DeliverVerdict::SpoolFull | DeliverVerdict::SenderQuota => {
                    if Instant::now() >= deadline {
                        return Err("coordinator spool never drained (back-pressure)".into());
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                other => return Err(format!("unexpected deliver verdict: {other:?}")),
            }
        }
        *seq += 1;
        Ok(())
    }

    /// Every decision published so far, decoded: `(channel, seq, signed frame bytes, message)`.
    #[must_use]
    pub fn published(&self) -> Vec<(u64, u64, Vec<u8>, VhcMessage)> {
        self.pump
            .published()
            .into_iter()
            .filter_map(|(ch, seq, frame)| {
                let v: Value = ciborium::de::from_reader(frame.as_slice()).ok()?;
                let Value::Array(parts) = v else { return None };
                let Value::Bytes(payload) = parts.get(1)? else {
                    return None;
                };
                let msg = from_canonical_slice::<VhcMessage>(payload).ok()?;
                Some((ch, seq, frame, msg))
            })
            .collect()
    }

    /// Pop the next not-yet-consumed decision, waiting (bounded) for the guest to produce one.
    ///
    /// # Errors
    /// A `String` timeout naming what was published so far.
    pub fn next_message(&mut self, timeout: Duration) -> Result<VhcMessage, String> {
        self.next_decision(timeout).map(|(_, _, msg)| msg)
    }

    /// Pop the next not-yet-consumed decision with its provenance: the §12.1 frame's `sender`
    /// (the coordinator's frame identity — the network seat judges it against the envelope-named
    /// `SingleKey` authority) and the complete original signed frame (the workers' tag-12
    /// evidence input).
    ///
    /// # Errors
    /// A `String` timeout, or a malformed frame.
    pub fn next_decision(
        &mut self,
        timeout: Duration,
    ) -> Result<([u8; 32], Vec<u8>, VhcMessage), String> {
        let deadline = Instant::now() + timeout;
        loop {
            let published = self.published();
            if let Some((_, _, frame, msg)) = published.get(self.consumed) {
                self.consumed += 1;
                let v: Value =
                    ciborium::de::from_reader(frame.as_slice()).map_err(|e| format!("{e}"))?;
                let sender = frame_sender(&v)?;
                return Ok((sender, frame.clone(), msg.clone()));
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "coordinator produced no further decision ({} consumed)",
                    self.consumed
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// The journal sink entries recorded so far (the §8 observations of the coordinator's run).
    #[must_use]
    pub fn sink_entries(&self) -> Vec<SinkEntry> {
        self.sink.lock().expect("sink").entries.clone()
    }

    /// The embedder's pump handle (hold/release, stop — the rig controls).
    #[must_use]
    pub fn pump(&self) -> crate::run::PumpHandle {
        self.pump.clone()
    }

    /// Stop the coordinator cleanly and join its guest thread.
    ///
    /// # Errors
    /// A `String` if the stop or the guest thread fails.
    pub fn stop(mut self) -> Result<RunEnd, String> {
        self.pump
            .stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
            .map_err(|e| format!("coordinator stop: {e}"))?;
        self.run
            .take()
            .expect("run present")
            .wait()
            .map_err(|e| format!("coordinator guest thread: {e}"))
    }

    /// Export the coordinator's consensus state through the sandbox (architecture §4.4; ABI §10.2):
    /// open a `Quiesce` drain, wait for the module to snapshot its `CoordinatorState` as a typed
    /// state-manifest and return `QuiesceReady`, then return the accepted [`SnapshotCapture`]. This
    /// consumes the instance (its `da_run` has returned). Feed the capture to
    /// [`Coordinator::start_migrating`] to re-instantiate a standby / restart from it.
    ///
    /// # Errors
    /// A `String` if the drain fails, the guest does not quiesce cleanly, or no snapshot was
    /// accepted in the drain.
    pub fn quiesce_snapshot(mut self, deadline_ms: u64) -> Result<SnapshotCapture, String> {
        self.pump
            .quiesce(daemon_vhc_abi::QUIESCE_REASON_UPGRADE, deadline_ms)
            .map_err(|e| format!("coordinator quiesce: {e}"))?;
        let end = self
            .run
            .take()
            .expect("run present")
            .wait()
            .map_err(|e| format!("coordinator guest thread: {e}"))?;
        match end {
            RunEnd::Outcome(code)
                if u64::from(code) == u64::from(daemon_vhc_abi::OUTCOME_QUIESCE_READY) => {}
            other => return Err(format!("coordinator did not quiesce cleanly: {other:?}")),
        }
        self.pump
            .snapshot_capture()
            .ok_or_else(|| "coordinator produced no accepted snapshot in the drain".to_string())
    }

    /// Kill the coordinator abruptly (the failover drill's "kill coordinator node"): stop the run
    /// with a fault reason and join the guest thread, returning however it ended. The pump and
    /// journal simply cease — no clean drain is negotiated.
    ///
    /// # Errors
    /// A `String` if the guest thread cannot be joined.
    pub fn kill(mut self) -> Result<RunEnd, String> {
        self.pump
            .stop(daemon_vhc_abi::STOP_REASON_FAULT)
            .map_err(|e| format!("coordinator kill: {e}"))?;
        self.run
            .take()
            .expect("run present")
            .wait()
            .map_err(|e| format!("coordinator guest thread: {e}"))
    }
}

/// Extract the §12.1 frame envelope's `sender` field from a decoded `[envelope, payload, sig]`
/// (public: whole-run harness seats key their per-sender delivery seqs on it).
pub fn frame_sender(frame: &Value) -> Result<[u8; 32], String> {
    let Value::Array(parts) = frame else {
        return Err("frame is not [envelope, payload, sig]".into());
    };
    let Some(Value::Map(env)) = parts.first() else {
        return Err("frame envelope is not a map".into());
    };
    let sender = env
        .iter()
        .find_map(|(k, v)| match (k, v) {
            (Value::Text(t), Value::Bytes(b)) if t == "sender" => Some(b.clone()),
            _ => None,
        })
        .ok_or("frame envelope missing sender")?;
    <[u8; 32]>::try_from(sender.as_slice()).map_err(|_| "sender is not 32 bytes".into())
}

/// Decode the exported [`CoordinatorState`] out of a [`SnapshotCapture`] the coordinator module
/// produced (the single `consensus` section is its whole state, canonical CBOR). Lets a drill
/// inspect the reconstructed state (e.g. its round height) without re-deriving it natively.
///
/// # Errors
/// A `String` if the capture carries no `consensus` section or the bytes do not decode.
pub fn coordinator_state_from_capture(
    capture: &SnapshotCapture,
) -> Result<CoordinatorState, String> {
    let bytes = capture
        .sections
        .iter()
        .find_map(|s| match s {
            daemon_vhc_proto::det_state::CkptDocSection::Inline(name, bytes)
                if name == COORDINATOR_STATE_SECTION =>
            {
                Some(bytes)
            }
            _ => None,
        })
        .ok_or_else(|| format!("capture carries no `{COORDINATOR_STATE_SECTION}` section"))?;
    from_canonical_slice::<CoordinatorState>(bytes)
        .map_err(|e| format!("coordinator state decode: {e}"))
}
