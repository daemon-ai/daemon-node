// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Fit-probe **input authoring** — the orchestrator half of the probe directory contract
//! ([`daemon_vhc_resource::probe`]).
//!
//! The worker binary runs the probe (the verdict names its sealed revision identity) but carries
//! no round vocabulary, so everything the drive consumes is authored HERE, where the ceremony
//! vocabulary already lives, and handed over as opaque bytes: the trainer's config, the round-0
//! open frame, the host-staged batch wrappers, and the completion condition as pure data.
//!
//! The one authoring input is the trainer config itself — geometry (steps, micro-batch, sequence
//! length, vocabulary) and the state contract's chunk size are read back out of the same bytes
//! the probe will run, so the inputs cannot disagree with the run they drive.

use std::path::Path;

use ciborium::value::Value;
use daemon_vhc_proto::{to_canonical_vec, Hash, PeerId, Seed, StateContract};
use daemon_vhc_resource::probe as contract;
use daemon_vhc_sdk_consensus::messages::{BatchWindow, RoundOpen, VhcMessage};

/// The compute@2 trainer's committed-container voice: the `[tag, round, …]` publish head that
/// marks a round committed (tag 3). Handed to the worker as data in the drive spec.
pub const TRAINER_COMMIT_TAG: u64 = 3;

/// The per-slice readback allowance the production TRAINER lane grants (the t2 gates' value).
pub const TRAINER_LANE_READBACK_BYTES: u64 = 64 << 20;
/// The live-buffer byte grant the production TRAINER lane carries.
pub const TRAINER_LANE_BUFFER_BYTES: u64 = 1 << 30;
/// The live-buffer handle grant the production TRAINER lane carries.
pub const TRAINER_LANE_BUFFER_HANDLES: u64 = 1024;
/// The compute-queue depth a real-geometry trainer runs under.
pub const TRAINER_LANE_COMPUTE_QUEUE_DEPTH: u64 = 1 << 20;

/// The geometry members the authoring reads back out of a trainer config.
struct TrainerGeometry {
    steps_per_round: u64,
    micro_batch: u64,
    seq_len: u64,
    vocab: u64,
    state_chunk_size: u64,
}

/// Author a complete probe directory for the compute@2 trainer at the geometry `config` pins.
///
/// `deadline_s` is the whole-drive wall (a ceremony-geometry CPU init alone is ~100 s; size it
/// for the lane being probed).
///
/// # Errors
/// A human-readable failure when the config does not carry the trainer's geometry members or the
/// directory cannot be written.
// Plain local-fs writes to the **orchestrator-chosen** probe directory (gate/dev authoring — the
// caller derives it under its own scratch root); never an attacker-influenced path, so
// `ContainedRoot` containment adds nothing here. Same posture as the session harness observer.
#[allow(clippy::disallowed_methods)]
pub fn write_trainer_probe_dir(
    dir: &Path,
    module: &[u8],
    config: &[u8],
    requirements: &daemon_vhc_proto::RoleExecutionRequirements,
    deadline_s: u64,
) -> Result<(), String> {
    let geometry = trainer_geometry(config)?;
    let write = |name: &str, bytes: &[u8]| {
        std::fs::write(dir.join(name), bytes).map_err(|e| format!("write {name}: {e}"))
    };
    std::fs::create_dir_all(dir.join(contract::STAGE_DIR))
        .map_err(|e| format!("create {}: {e}", contract::STAGE_DIR))?;
    write(contract::MODULE_FILE, module)?;
    write(contract::CONFIG_FILE, config)?;
    write(
        contract::REQUIREMENTS_FILE,
        &to_canonical_vec(requirements).map_err(|e| format!("encode requirements: {e}"))?,
    )?;

    // The round-0 open, sized to consume every staged batch.
    let open = to_canonical_vec(&VhcMessage::RoundOpen(RoundOpen {
        round: 0,
        seed: Seed([0; 32]),
        roster_digest: Hash([0; 32]),
        batch: BatchWindow {
            start: 0,
            end: geometry.micro_batch * geometry.steps_per_round,
        },
        deadline_unix_s: 0,
    }))
    .map_err(|e| format!("encode the round open: {e}"))?;
    write(contract::OPEN_FRAME_FILE, &open)?;

    // One host-staged kind-0 batch wrapper per step, deterministic varied tokens (the t2 gates'
    // mixer, so no two wrappers are byte-identical and the pump's dedup never coalesces them).
    for step in 0..geometry.steps_per_round {
        let wrapper = batch_wrapper(&geometry, 0, step);
        std::fs::write(
            dir.join(contract::STAGE_DIR)
                .join(format!("{step:06}.cbor")),
            wrapper,
        )
        .map_err(|e| format!("write staged batch {step}: {e}"))?;
    }

    let drive = contract::FitProbeDrive {
        role: "trainer".to_string(),
        commit_tag: TRAINER_COMMIT_TAG,
        commit_round: 0,
        deadline_s,
        state_chunk_size: geometry.state_chunk_size,
        compute_queue_depth: Some(TRAINER_LANE_COMPUTE_QUEUE_DEPTH),
        max_readback_bytes_per_slice: Some(TRAINER_LANE_READBACK_BYTES),
        max_live_buffer_bytes: Some(TRAINER_LANE_BUFFER_BYTES),
        max_live_buffer_handles: Some(TRAINER_LANE_BUFFER_HANDLES),
    };
    write(
        contract::DRIVE_FILE,
        &to_canonical_vec(&drive).map_err(|e| format!("encode the drive: {e}"))?,
    )?;
    Ok(())
}

/// A stand-in roster member for probe-config authoring: the probe drives host staging, where the
/// roster is join wiring that never enters the plan (the ceremony authoring's own argument for
/// assessing against a stand-in member).
#[must_use]
pub fn probe_peer() -> PeerId {
    PeerId([0x3B; 32])
}

/// Read the trainer's drive geometry back out of its own config bytes.
fn trainer_geometry(config: &[u8]) -> Result<TrainerGeometry, String> {
    let value: Value = ciborium::de::from_reader(config)
        .map_err(|e| format!("the probe config does not decode as CBOR: {e}"))?;
    let Value::Map(fields) = &value else {
        return Err("the probe config is not a map".into());
    };
    let field = |name: &str| {
        fields
            .iter()
            .find(|(k, _)| matches!(k, Value::Text(t) if t == name))
            .map(|(_, v)| v)
    };
    let uint = |v: &Value, what: &str| {
        v.as_integer()
            .and_then(|n| u64::try_from(i128::from(n)).ok())
            .ok_or_else(|| format!("config member `{what}` is not an unsigned integer"))
    };
    let steps_per_round = uint(
        field("steps_per_round").ok_or("config carries no `steps_per_round`")?,
        "steps_per_round",
    )?;
    let micro_batch = uint(
        field("micro_batch").ok_or("config carries no `micro_batch`")?,
        "micro_batch",
    )?;
    let Some(Value::Map(model)) = field("model") else {
        return Err("config carries no `model` map".into());
    };
    let model_field = |name: &str| {
        model
            .iter()
            .find(|(k, _)| matches!(k, Value::Text(t) if t == name))
            .map(|(_, v)| v)
    };
    let seq_len = uint(
        model_field("seq_len").ok_or("model carries no `seq_len`")?,
        "model.seq_len",
    )?;
    let vocab = uint(
        model_field("vocab").ok_or("model carries no `vocab`")?,
        "model.vocab",
    )?;
    let state_chunk_size = match field("state") {
        Some(v) => {
            let contract: StateContract = v
                .clone()
                .deserialized()
                .map_err(|e| format!("config `state` is not a state contract: {e}"))?;
            contract.chunk_size
        }
        None => 0,
    };
    Ok(TrainerGeometry {
        steps_per_round,
        micro_batch,
        seq_len,
        vocab,
        state_chunk_size,
    })
}

/// One host-staged batch, in the guest's documented absent-`live` wrapper shape
/// `[0, round, step, sequences, seq_len, tokens_le]`.
fn batch_wrapper(geometry: &TrainerGeometry, round: u64, step: u64) -> Vec<u8> {
    let n = geometry.micro_batch * geometry.seq_len;
    let mut le = Vec::with_capacity(usize::try_from(n * 4).expect("token bytes fit usize"));
    for i in 0..n {
        let x = i + 1_000 * step + 100_000 * round + 1;
        let token = u32::try_from(x.wrapping_mul(2_654_435_761) % geometry.vocab)
            .expect("a token id fits u32");
        le.extend_from_slice(&token.to_le_bytes());
    }
    to_canonical_vec(&Value::Array(vec![
        Value::from(0u8),
        Value::from(round),
        Value::from(step),
        Value::from(geometry.micro_batch),
        Value::from(geometry.seq_len),
        Value::Bytes(le),
    ]))
    .expect("batch wrapper")
}
