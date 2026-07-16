// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The coordinator's resolved run configuration (spec §6.1/§6.2).
//!
//! [`RunConfig`] is the coordination-consumed projection of the frozen run envelope (§4.3 seam rule:
//! `[run]`/`[data]`/`[phases]`/`[requirements].capabilities` only — never `[experiment.config]`),
//! plus the coordinator-only knobs the envelope does not carry ([`CoordinatorParams`]). It is part of
//! [`crate::CoordinatorState`] and therefore canonical-CBOR-serializable (the replay foundation,
//! PROTO-20).

use crate::assignment::WITNESS_TARGET_DEFAULT;
use daemon_vhc_proto::canonical::{from_canonical_slice, to_canonical_vec};
use daemon_vhc_proto::envelope::{Envelope, GlobalBatch, StopCondition};
use daemon_vhc_proto::{
    blake3_hash, CapabilitySet, GenesisEnvelope, Hash, PeerId, SwarmProtoError, SwarmProtoVersion,
};
use serde::{Deserialize, Serialize};

use crate::coordinator::CoordinatorError;

/// Default K record-absences before a peer is dropped (§6.4 daemon Delta; TDD PROTO-7).
pub const K_ABSENCES_DEFAULT: u32 = 3;

/// Coordinator-only run parameters that the frozen envelope does not carry (ledger-P2 note).
///
/// Supplied at run creation (Wave-3 authoring), never read from `[experiment.config]` at runtime, so
/// the seam rule (§4.3) is preserved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoordinatorParams {
    /// Tokens per sequence — converts the `[data].global_batch` (sequences/round) into tokens for a
    /// `[data].stop = { tokens }` termination (§6.1). `1` means "count sequences as tokens".
    pub seq_len: u64,
    /// Target witness-committee size (§6.3). `0` means "every peer witnesses".
    pub witness_target: u32,
    /// Deliberate batch overlap in basis points (0–10000; §6.3), 0 = exact partition.
    pub overlap_bps: u32,
    /// K record-absences before a peer is dropped (§6.4).
    pub k_absences: u32,
    /// Verifier-committee sampling percent (§12) — `0` keeps the seam a no-op (TDD PROTO-15).
    pub verification_percent: u32,
    /// Principals (node identities) authorized to pause/resume (§11.1; TDD PROTO-14).
    pub authorized: Vec<PeerId>,
}

impl Default for CoordinatorParams {
    fn default() -> Self {
        Self {
            seq_len: 1,
            witness_target: WITNESS_TARGET_DEFAULT,
            overlap_bps: 0,
            k_absences: K_ABSENCES_DEFAULT,
            verification_percent: 0,
            authorized: Vec::new(),
        }
    }
}

/// The resolved, coordination-consumed run configuration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunConfig {
    /// Run identity (`[run].run_id`).
    pub run_id: String,
    /// The swarm proto version this run is pinned to (exact-match join gate, §16).
    pub proto_version: SwarmProtoVersion,
    /// blake3 hash of the frozen envelope (§6.1) — the admission envelope-hash anchor.
    pub envelope_hash: Hash,
    /// The run's required capability set (`[requirements].capabilities`, §6.5).
    pub required_capabilities: CapabilitySet,
    /// `min_peers` floor to leave `WaitingForMembers` (§6.2).
    pub min_peers: u32,
    /// `max_peers` roster ceiling.
    pub max_peers: u32,
    /// Warmup timeout (seconds).
    pub warmup_s: u64,
    /// Max training time per round (seconds).
    pub round_train_max_s: u64,
    /// Witness grace window (seconds).
    pub round_witness_s: u64,
    /// Cooldown duration (seconds).
    pub cooldown_s: u64,
    /// Rounds per epoch (roster-stable span, §6.2).
    pub epoch_rounds: u64,
    /// Fetch-recovery budget before a stalled peer must leave (§6.4).
    pub stall_rounds_max: u32,
    /// Sequences-per-round schedule (`[data].global_batch`, §6.1).
    pub global_batch: GlobalBatch,
    /// Termination condition (`[data].stop`, §6.2).
    pub stop: StopCondition,
    /// Inner steps per round (H) — carried for peers, not consumed by `tick` (§6.1).
    pub steps_per_round: u32,
    /// Tokens per sequence (coordinator-only, [`CoordinatorParams`]).
    pub seq_len: u64,
    /// Target witness-committee size (coordinator-only).
    pub witness_target: u32,
    /// Deliberate batch overlap in basis points (coordinator-only).
    pub overlap_bps: u32,
    /// K record-absences drop threshold (coordinator-only).
    pub k_absences: u32,
    /// Verifier-committee sampling percent (coordinator-only).
    pub verification_percent: u32,
    /// Principals authorized to pause/resume (coordinator-only).
    pub authorized: Vec<PeerId>,
}

/// The coordinator role's **opaque module config** as it lives in an envelope-v2 genesis
/// (architecture §5.1; refactor §8/D0): the `[data]` schedule and `[phases]` policy that **left the
/// v1 envelope** at D0 now live here, in the coordinator [`RoleEntry::config`](daemon_vhc_proto::RoleEntry).
/// The host never interprets it (the seam rule); the coordinator module (or, transitionally, the
/// native-coordinator adapter, [`RunConfig::from_genesis`]) does.
///
/// This is the schema the **transitional native-coordinator adapter** (mixed-fleet cell 6) decodes
/// — the adapter exists only through D1 and retires at D2, when the wasm `coordinator-quorum` guest
/// reads the same config in-guest and `daemon-vhc-coordinator` dissolves (decisions D3 cell 6;
/// refactor §8 D2/§11).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoordinatorRoleConfig {
    /// Sequences-per-round schedule (was `[data].global_batch`).
    pub global_batch: GlobalBatch,
    /// Termination condition (was `[data].stop`).
    pub stop: StopCondition,
    /// Inner steps per round, H (was `[data].steps_per_round`).
    pub steps_per_round: u32,
    /// Warmup timeout, seconds (was `[phases].warmup`).
    pub warmup: u32,
    /// Max training time per round, seconds (was `[phases].round_train_max`).
    pub round_train_max: u32,
    /// Witness grace window, seconds (was `[phases].round_witness`).
    pub round_witness: u32,
    /// Cooldown duration, seconds (was `[phases].cooldown`).
    pub cooldown: u32,
    /// Rounds per epoch (was `[phases].epoch_rounds`).
    pub epoch_rounds: u32,
    /// Fetch-recovery budget before a stalled peer leaves (was `[phases].stall_rounds_max`).
    pub stall_rounds_max: u32,
}

impl RunConfig {
    /// Project a resolved [`Envelope`] + coordinator params into a [`RunConfig`].
    ///
    /// The `envelope_hash` is recomputed from the envelope's canonical CBOR (blake3), byte-identical
    /// to [`daemon_vhc_proto::FrozenEnvelope::hash`]. Fails if the envelope is invalid (§6.1) or a
    /// capability token is malformed.
    pub fn from_envelope(
        env: &Envelope,
        params: CoordinatorParams,
    ) -> Result<Self, CoordinatorError> {
        env.validate()?;
        let bytes = to_canonical_vec(env)?;
        let envelope_hash = blake3_hash(&bytes);
        let required_capabilities =
            CapabilitySet::from_tokens(env.requirements.capabilities.iter())?;
        Ok(Self {
            run_id: env.run.run_id.clone(),
            proto_version: daemon_vhc_proto::SWARM_PROTO_VERSION,
            envelope_hash,
            required_capabilities,
            min_peers: env.run.min_peers,
            max_peers: env.run.max_peers,
            warmup_s: u64::from(env.phases.warmup),
            round_train_max_s: u64::from(env.phases.round_train_max),
            round_witness_s: u64::from(env.phases.round_witness),
            cooldown_s: u64::from(env.phases.cooldown),
            epoch_rounds: u64::from(env.phases.epoch_rounds),
            stall_rounds_max: env.phases.stall_rounds_max,
            global_batch: env.data.global_batch,
            stop: env.data.stop,
            steps_per_round: env.data.steps_per_round,
            seq_len: params.seq_len,
            witness_target: params.witness_target,
            overlap_bps: params.overlap_bps,
            k_absences: params.k_absences,
            verification_percent: params.verification_percent,
            authorized: params.authorized,
        })
    }

    /// **The transitional native-coordinator envelope-v2 adapter (mixed-fleet cell 6).** Project an
    /// envelope-v2 [`GenesisEnvelope`] + coordinator params into a [`RunConfig`] so the native `tick`
    /// can drive a v2 run in the D0→D1 window, before the wasm `coordinator-quorum` guest exists.
    ///
    /// Reads the role set (the `coordinator_role` entry), decodes that role's opaque
    /// [`CoordinatorRoleConfig`] (the `[data]`/`[phases]` policy that left the envelope at D0), and
    /// anchors on the **genesis hash** (the cryptographic `RunId`, byte-identical to
    /// [`daemon_vhc_proto::FrozenGenesis::run_id`]); `run_id` (the string field) carries the
    /// human/registry `RunLabel`. `required_capabilities` is empty here: envelope-v2 admission is
    /// the worker's claim-funnel + role grants (architecture §3.5), not the v1 capability subset.
    ///
    /// This adapter exists **only through D1** and is retired at D2 (decisions D3 cell 6): when the
    /// wasm coordinator supersedes it, cell 8 (wasm coordinator × v2 workers) takes over and this
    /// crate dissolves into `sdk-consensus` + `coordinator-quorum` (refactor §8 D2/§11).
    ///
    /// # Errors
    /// [`CoordinatorError::Proto`] if the envelope is invalid, the `coordinator_role` is absent, or
    /// its opaque config does not decode as a [`CoordinatorRoleConfig`].
    pub fn from_genesis(
        env: &GenesisEnvelope,
        coordinator_role: &str,
        params: CoordinatorParams,
    ) -> Result<Self, CoordinatorError> {
        env.validate()?;
        let bytes = to_canonical_vec(env)?;
        // The genesis hash IS the cryptographic RunId (architecture §5.1; ABI §8.1) — the same value
        // `FrozenGenesis::run_id` exposes, recomputed here over the canonical bytes.
        let envelope_hash = blake3_hash(&bytes);
        let role = env.roles.get(coordinator_role).ok_or_else(|| {
            SwarmProtoError::Validation(format!(
                "genesis envelope has no `{coordinator_role}` role to adapt (cell-6 adapter needs \
                 the coordinator role's config)"
            ))
        })?;
        // Decode the coordinator role's opaque config (the seam rule: the host reads it only here,
        // in the transitional adapter). Canonical-CBOR round-trip keeps the decode deterministic.
        let cfg_bytes = to_canonical_vec(&role.config)?;
        let cfg: CoordinatorRoleConfig = from_canonical_slice(&cfg_bytes).map_err(|e| {
            SwarmProtoError::Validation(format!(
                "coordinator role `{coordinator_role}` config is not a CoordinatorRoleConfig \
                 (the [data]/[phases] policy that left the v1 envelope at D0): {e}"
            ))
        })?;
        Ok(Self {
            run_id: env.run.run_label.clone(),
            proto_version: daemon_vhc_proto::SWARM_PROTO_VERSION,
            envelope_hash,
            required_capabilities: CapabilitySet::new(),
            min_peers: env.run.min_peers,
            max_peers: env.run.max_peers,
            warmup_s: u64::from(cfg.warmup),
            round_train_max_s: u64::from(cfg.round_train_max),
            round_witness_s: u64::from(cfg.round_witness),
            cooldown_s: u64::from(cfg.cooldown),
            epoch_rounds: u64::from(cfg.epoch_rounds),
            stall_rounds_max: cfg.stall_rounds_max,
            global_batch: cfg.global_batch,
            stop: cfg.stop,
            steps_per_round: cfg.steps_per_round,
            seq_len: params.seq_len,
            witness_target: params.witness_target,
            overlap_bps: params.overlap_bps,
            k_absences: params.k_absences,
            verification_percent: params.verification_percent,
            authorized: params.authorized,
        })
    }
}

#[cfg(test)]
mod genesis_adapter_tests {
    use super::*;
    use daemon_vhc_proto::envelope::{Access, DeviceMinimums};
    use daemon_vhc_proto::genesis::{
        Identities, RoleEntry, RunSectionV2, TransportSelection, GENESIS_SCHEMA_MAJOR,
    };
    use daemon_vhc_proto::RoleGrants;
    use daemon_vhc_proto::SigningKey;
    use std::collections::BTreeMap;

    fn coord_role_config() -> CoordinatorRoleConfig {
        CoordinatorRoleConfig {
            global_batch: GlobalBatch {
                start: 8,
                end: 8,
                ramp_rounds: 0,
            },
            stop: StopCondition::Rounds(4),
            steps_per_round: 2,
            warmup: 3,
            round_train_max: 30,
            round_witness: 5,
            cooldown: 2,
            epoch_rounds: 10,
            stall_rounds_max: 2,
        }
    }

    fn genesis_with_coordinator(cfg: &CoordinatorRoleConfig) -> GenesisEnvelope {
        let cfg_value: ciborium::value::Value = {
            let bytes = to_canonical_vec(cfg).unwrap();
            from_canonical_slice(&bytes).unwrap()
        };
        let mut artifacts = BTreeMap::new();
        artifacts.insert(
            "coord-mod".to_string(),
            daemon_vhc_proto::genesis::SnapshotArtifact {
                url: "r2://mods/coord.wasm".into(),
                blake3: Hash([2; 32]),
                size: Some(2048),
            },
        );
        artifacts.insert(
            "worker-mod".to_string(),
            daemon_vhc_proto::genesis::SnapshotArtifact {
                url: "r2://mods/worker.wasm".into(),
                blake3: Hash([1; 32]),
                size: Some(4096),
            },
        );
        let mut roles = BTreeMap::new();
        roles.insert(
            "coordinator".to_string(),
            RoleEntry {
                lane: "coordinator".into(),
                module: "coord-mod".into(),
                abi: "vhc@2".into(),
                config: cfg_value,
                grants: RoleGrants::default(),
                device_min: DeviceMinimums::default(),
            },
        );
        roles.insert(
            "worker".to_string(),
            RoleEntry {
                lane: "trainer".into(),
                module: "worker-mod".into(),
                abi: "vhc@2".into(),
                config: ciborium::value::Value::Map(vec![]),
                grants: RoleGrants::default(),
                device_min: DeviceMinimums::default(),
            },
        );
        GenesisEnvelope {
            run: RunSectionV2 {
                schema: GENESIS_SCHEMA_MAJOR,
                run_label: "cell6-run".into(),
                min_peers: 1,
                max_peers: 16,
                access: Access::Org,
            },
            roles,
            artifacts,
            authority: ciborium::value::Value::Map(vec![]),
            transport: TransportSelection::default(),
            identities: Identities::default(),
        }
    }

    #[test]
    fn from_genesis_projects_the_coordinator_role_config_and_anchors_on_the_genesis_hash() {
        let cfg = coord_role_config();
        let env = genesis_with_coordinator(&cfg);
        let rc = RunConfig::from_genesis(&env, "coordinator", CoordinatorParams::default())
            .expect("adapt genesis");

        // The [data]/[phases] policy came from the coordinator role's opaque config.
        assert_eq!(rc.global_batch, cfg.global_batch);
        assert_eq!(rc.stop, cfg.stop);
        assert_eq!(rc.steps_per_round, cfg.steps_per_round);
        assert_eq!(rc.warmup_s, u64::from(cfg.warmup));
        assert_eq!(rc.epoch_rounds, u64::from(cfg.epoch_rounds));
        assert_eq!(rc.stall_rounds_max, cfg.stall_rounds_max);
        assert_eq!(rc.min_peers, 1);
        assert_eq!(rc.run_id, "cell6-run"); // the RunLabel

        // The envelope_hash the adapter anchors on IS the genesis hash / cryptographic RunId.
        let frozen = env.freeze(&SigningKey::from_bytes(&[7; 32])).unwrap();
        assert_eq!(&rc.envelope_hash, frozen.run_id());
    }

    #[test]
    fn from_genesis_refuses_a_missing_coordinator_role() {
        let env = genesis_with_coordinator(&coord_role_config());
        assert!(
            RunConfig::from_genesis(&env, "no-such-role", CoordinatorParams::default()).is_err()
        );
    }

    #[test]
    fn from_genesis_refuses_a_role_whose_config_is_not_a_coordinator_config() {
        let mut env = genesis_with_coordinator(&coord_role_config());
        // Point the adapter at the worker role, whose opaque config is an empty map (not phases).
        assert!(RunConfig::from_genesis(&env, "worker", CoordinatorParams::default()).is_err());
        // A garbage coordinator config is likewise refused.
        env.roles.get_mut("coordinator").unwrap().config =
            ciborium::value::Value::from("not a config");
        assert!(
            RunConfig::from_genesis(&env, "coordinator", CoordinatorParams::default()).is_err()
        );
    }
}
