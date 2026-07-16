// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Tighten-only grants derivation (D0 deliverable 2; ABI §2.6 derivation; architecture §3.5).
//!
//! The genesis envelope's per-role [`crate::genesis::RoleGrants`] is one contributor to the ABI
//! §2.6 *admitted grants document*:
//!
//! ```text
//! admitted = lane profile ceilings  ∩  envelope role grant list  ∩  owner standing policy
//!          ∩  module manifest requests          (tightest value per bound)
//! ```
//!
//! This module is the **pure, wasm-clean core** of that intersection for the numeric quotas the
//! Phase-B `V2RunConfig` seam consumes (`payload_depth`/`advisory_depth`/`gossip_depth`, the
//! authoritative spool bounds `spool_frames`/`per_sender_quota`, `max_frame_bytes`, buffer and
//! completion ceilings, and the granted-artifact set). It takes the **lane ceilings as plain
//! numbers** ([`LaneCeilings`]) so the host passes its `ParticipationLane` profile in without this
//! crate depending on any host type — proto stays algorithm-free wire mechanism.
//!
//! **Tighten-only is enforced, not assumed** (architecture §3.5, §5.1): an envelope grant may only
//! *narrow* a lane ceiling. A role that requests **more** than its lane allows — a larger frame,
//! a deeper spool, a world/custom-op the lane does not offer, an artifact outside the run's map —
//! is a [`GrantsError`] (the host surfaces it as `GrantsExceedLane`, ABI §1.5). An absent envelope
//! bound inherits the lane ceiling; a `0` lane ceiling means "the lane imposes no ceiling on this
//! bound" (the envelope/manifest value stands).

use std::collections::BTreeSet;

use crate::bytes::Hash;
use crate::genesis::RoleGrants;

/// A lane profile's ceilings as pure numbers (the host's `ParticipationLane` §9.6 fields, passed
/// in so this crate needs no host dependency). A `0` ceiling means **"no lane ceiling on this
/// bound"** — the envelope/manifest value is used as-is (raw bytes / counts, never MB/Mbps).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LaneCeilings {
    /// Ceiling on any channel's `max_frame_bytes` (ABI §9.6 `channel_ceilings.max_frame_bytes`).
    pub max_frame_bytes: u64,
    /// Ceiling on any authoritative channel's `spool_frames`.
    pub spool_frames: u64,
    /// Ceiling on any authoritative channel's `per_sender_quota`.
    pub per_sender_quota: u64,
    /// Ceiling on any authoritative channel's `replay_window` (carried through for completeness).
    pub replay_window: u64,
    /// Ceiling on `rate_per_min` for any channel.
    pub rate_per_min: u64,
    /// Ceiling on advisory `Timer`-class queue depth (`0` = no lane ceiling).
    pub advisory_depth: u64,
    /// Ceiling on advisory `PayloadReady`-class queue depth (`0` = no lane ceiling).
    pub payload_depth: u64,
    /// Ceiling on advisory gossip-class queue depth (`0` = no lane ceiling).
    pub gossip_depth: u64,
    /// Ceiling on `buffer-req.max_live_handles`.
    pub max_live_handles: u64,
    /// Ceiling on `buffer-req.max_live_bytes`.
    pub max_live_bytes: u64,
    /// Ceiling on `buffer-req.max_readback_bytes`.
    pub max_readback_bytes: u64,
    /// Ceiling on the async-completion `max_outstanding` grant.
    pub max_outstanding_ops: u64,
    /// Ceiling on the compute@2 command-queue depth grant (C1, ABI §15 — "a queue-depth grant
    /// bounds outstanding device work"; D0∩C1 union: the ninth tightened quota).
    pub compute_queue_depth: u64,
    /// The worlds a role admitted under this lane MAY be granted (ABI §9.6 `worlds`).
    pub worlds: BTreeSet<String>,
    /// The custom ops this lane offers (ABI §9.6 `custom_ops`).
    pub custom_ops: BTreeSet<String>,
}

/// The admitted numeric quotas — the tightened values the host copies into `V2RunConfig` (host
/// `daemon-vhc-host::v2::V2RunConfig`). Every field is the tightest of (lane, envelope, manifest).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AdmittedQuotas {
    /// Admitted per-frame byte ceiling (`publish`).
    pub max_frame_bytes: u64,
    /// Admitted authoritative spool depth.
    pub spool_frames: u64,
    /// Admitted authoritative per-sender outstanding quota.
    pub per_sender_quota: u64,
    /// Admitted advisory `Timer` queue depth.
    pub advisory_depth: u64,
    /// Admitted advisory `PayloadReady` queue depth.
    pub payload_depth: u64,
    /// Admitted advisory gossip queue depth.
    pub gossip_depth: u64,
    /// Admitted live-buffer handle ceiling.
    pub max_live_handles: u64,
    /// Admitted live-buffer byte ceiling.
    pub max_live_bytes: u64,
    /// Admitted per-slice readback byte ceiling.
    pub max_readback_bytes: u64,
    /// Admitted concurrent-operation ceiling.
    pub max_outstanding_ops: u64,
    /// Admitted compute@2 command-queue depth (C1's `V2RunConfig.compute_queue_depth`; tightened
    /// exactly like the other quotas — D0∩C1 union).
    pub compute_queue_depth: u64,
    /// The admitted artifact set (a subset of the run's artifact map, intersected with the role
    /// grant) — the module's `data.fetch` allow-list.
    pub granted_artifacts: BTreeSet<Hash>,
}

/// A tighten-only violation — the host surfaces it as the ABI §1.5 `GrantsExceedLane` refusal.
/// Hand-rolled (no `thiserror`) to keep the crate `wasm32`-clean, matching
/// [`crate::error::SwarmProtoError`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GrantsError {
    /// A numeric bound the envelope requested exceeds the lane ceiling.
    BoundExceedsLane {
        /// The bound name.
        bound: String,
        /// The envelope-requested value.
        requested: u64,
        /// The lane ceiling.
        ceiling: u64,
    },
    /// The envelope requests a world the lane does not offer.
    WorldNotInLane(String),
    /// The envelope requests a custom op the lane does not offer.
    CustomOpNotInLane(String),
    /// The envelope grants an artifact absent from the run's artifact map.
    ArtifactNotInMap(String),
}

impl core::fmt::Display for GrantsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BoundExceedsLane {
                bound,
                requested,
                ceiling,
            } => write!(
                f,
                "grant `{bound}` request {requested} exceeds lane ceiling {ceiling}"
            ),
            Self::WorldNotInLane(w) => {
                write!(f, "role requests world `{w}` not offered by the lane")
            }
            Self::CustomOpNotInLane(op) => {
                write!(f, "role requests custom op `{op}` not offered by the lane")
            }
            Self::ArtifactNotInMap(h) => {
                write!(
                    f,
                    "role grants artifact {h} absent from the run artifact map"
                )
            }
        }
    }
}

impl std::error::Error for GrantsError {}

/// Take the tightest of an envelope-requested value against a lane ceiling, enforcing tighten-only.
///
/// - lane ceiling `0` ⇒ "no lane ceiling": the envelope value stands (or, if the envelope did not
///   request this bound, `0` = "unbounded by this grant", per ABI §2.3);
/// - a present envelope request **above** the ceiling is a [`GrantsError::BoundExceedsLane`];
/// - otherwise the admitted value is `min(request, ceiling)`, with an absent request inheriting
///   the ceiling.
fn tighten(bound: &str, request: Option<u64>, ceiling: u64) -> Result<u64, GrantsError> {
    match (request, ceiling) {
        (Some(r), 0) => Ok(r),
        (None, c) => Ok(c),
        (Some(r), c) if r > c => Err(GrantsError::BoundExceedsLane {
            bound: bound.to_string(),
            requested: r,
            ceiling: c,
        }),
        (Some(r), c) => Ok(r.min(c)),
    }
}

impl RoleGrants {
    /// The largest `max_frame_bytes` any channel in this role declares (the run needs headroom for
    /// its widest channel). `None` when the role declares no channels.
    fn max_channel_frame_bytes(&self) -> Option<u64> {
        self.channels.iter().map(|c| c.max_frame_bytes).max()
    }

    /// The largest authoritative-channel `spool_frames` this role declares.
    fn max_channel_spool(&self) -> Option<u64> {
        self.channels.iter().filter_map(|c| c.spool_frames).max()
    }

    /// The largest authoritative-channel `per_sender_quota` this role declares.
    fn max_channel_sender_quota(&self) -> Option<u64> {
        self.channels
            .iter()
            .filter_map(|c| c.per_sender_quota)
            .max()
    }

    /// The declared depth for an advisory event class (`"timer"`/`"payload-ready"`/`"gossip"`).
    fn event_depth(&self, class: &str) -> Option<u64> {
        self.events.classes.get(class).map(|c| c.depth)
    }
}

/// Derive the admitted numeric quotas for one role by intersecting its envelope grant list with
/// the lane ceilings (tighten-only) — the pure core of ABI §2.6's derivation for the `V2RunConfig`
/// quotas. `run_artifacts` is the set of blake3 hashes in the run's artifact map (the role's
/// granted artifacts must be a subset).
///
/// The owner-policy and module-manifest contributors intersect on top of this host-side (they are
/// tighter-or-equal by construction); this function fixes the lane∩envelope core the two later
/// tighten further.
///
/// # Errors
/// A [`GrantsError`] when any envelope grant exceeds the lane (the host raises `GrantsExceedLane`).
pub fn derive_admitted_quotas(
    role: &RoleGrants,
    lane: &LaneCeilings,
    run_artifacts: &BTreeSet<Hash>,
) -> Result<AdmittedQuotas, GrantsError> {
    // Worlds ⊆ lane worlds (only enforced when the lane enumerates its worlds).
    if !lane.worlds.is_empty() {
        for w in role.worlds.keys() {
            if !lane.worlds.contains(w) {
                return Err(GrantsError::WorldNotInLane(w.clone()));
            }
        }
    }
    // Custom ops ⊆ lane custom ops (always — a lane offering none refuses any request).
    for op in &role.custom_ops {
        if !lane.custom_ops.contains(op) {
            return Err(GrantsError::CustomOpNotInLane(op.clone()));
        }
    }
    // Granted artifacts ⊆ the run's artifact map.
    for h in &role.artifacts {
        if !run_artifacts.contains(h) {
            return Err(GrantsError::ArtifactNotInMap(h.to_hex()));
        }
    }
    // Per-channel bounds must each be ≤ the lane ceiling (tighten-only), enforced by taking the
    // widest declared value against the ceiling.
    let max_frame_bytes = tighten(
        "max_frame_bytes",
        role.max_channel_frame_bytes(),
        lane.max_frame_bytes,
    )?;
    let spool_frames = tighten("spool_frames", role.max_channel_spool(), lane.spool_frames)?;
    let per_sender_quota = tighten(
        "per_sender_quota",
        role.max_channel_sender_quota(),
        lane.per_sender_quota,
    )?;
    let advisory_depth = tighten(
        "advisory_depth",
        role.event_depth("timer"),
        lane.advisory_depth,
    )?;
    let payload_depth = tighten(
        "payload_depth",
        role.event_depth("payload-ready"),
        lane.payload_depth,
    )?;
    let gossip_depth = tighten(
        "gossip_depth",
        role.event_depth("gossip"),
        lane.gossip_depth,
    )?;
    let max_live_handles = tighten(
        "max_live_handles",
        non_zero(role.buffers.max_live_handles),
        lane.max_live_handles,
    )?;
    let max_live_bytes = tighten(
        "max_live_bytes",
        non_zero(role.buffers.max_live_bytes),
        lane.max_live_bytes,
    )?;
    let max_readback_bytes = tighten(
        "max_readback_bytes",
        non_zero(role.buffers.max_readback_bytes),
        lane.max_readback_bytes,
    )?;
    let max_outstanding_ops = tighten(
        "max_outstanding_ops",
        non_zero(role.max_outstanding_ops),
        lane.max_outstanding_ops,
    )?;
    let compute_queue_depth = tighten(
        "compute_queue_depth",
        non_zero(role.compute_queue_depth),
        lane.compute_queue_depth,
    )?;

    Ok(AdmittedQuotas {
        max_frame_bytes,
        spool_frames,
        per_sender_quota,
        advisory_depth,
        payload_depth,
        gossip_depth,
        max_live_handles,
        max_live_bytes,
        max_readback_bytes,
        max_outstanding_ops,
        compute_queue_depth,
        granted_artifacts: role.artifacts.clone(),
    })
}

/// Treat a `0` envelope grant as "unspecified" (absent) — the ABI §2.3 "absent = unbounded by
/// this grant" convention, so a `0` inherits the lane ceiling rather than tightening to zero.
fn non_zero(v: u64) -> Option<u64> {
    (v != 0).then_some(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genesis::{BufferReq, ChannelDecl, EventCap, EventCaps};

    fn hash(n: u8) -> Hash {
        Hash([n; 32])
    }

    fn lane() -> LaneCeilings {
        LaneCeilings {
            max_frame_bytes: 1 << 20,
            spool_frames: 256,
            per_sender_quota: 64,
            replay_window: 1024,
            rate_per_min: 6000,
            advisory_depth: 128,
            payload_depth: 128,
            gossip_depth: 128,
            max_live_handles: 256,
            max_live_bytes: 1 << 26,
            max_readback_bytes: 1 << 20,
            max_outstanding_ops: 64,
            compute_queue_depth: 512,
            worlds: ["vhc", "net", "sys", "data"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            custom_ops: BTreeSet::new(),
        }
    }

    fn auth_channel(frame: u64, spool: u64, quota: u64) -> ChannelDecl {
        ChannelDecl {
            id: 0,
            name: "control".into(),
            class: 0,
            direction: 2,
            max_frame_bytes: frame,
            rate_per_min: 600,
            spool_frames: Some(spool),
            replay_window: Some(512),
            per_sender_quota: Some(quota),
        }
    }

    #[test]
    fn absent_grants_inherit_the_lane_ceiling() {
        let admitted =
            derive_admitted_quotas(&RoleGrants::default(), &lane(), &BTreeSet::new()).unwrap();
        assert_eq!(admitted.max_frame_bytes, 1 << 20);
        assert_eq!(admitted.spool_frames, 256);
        assert_eq!(admitted.advisory_depth, 128);
        assert_eq!(admitted.max_outstanding_ops, 64);
    }

    #[test]
    fn envelope_tightens_but_never_loosens() {
        let mut role = RoleGrants {
            channels: vec![auth_channel(4096, 32, 8)],
            events: EventCaps {
                classes: [(
                    "timer".to_string(),
                    EventCap {
                        depth: 16,
                        coalesce: 1,
                    },
                )]
                .into_iter()
                .collect(),
            },
            buffers: BufferReq {
                max_live_handles: 16,
                max_live_bytes: 1 << 10,
                max_readback_bytes: 4096,
            },
            max_outstanding_ops: 8,
            ..Default::default()
        };
        role.artifacts.insert(hash(1));
        let run_artifacts: BTreeSet<Hash> = [hash(1)].into_iter().collect();
        let a = derive_admitted_quotas(&role, &lane(), &run_artifacts).unwrap();
        assert_eq!(a.max_frame_bytes, 4096, "tightened below the lane ceiling");
        assert_eq!(a.spool_frames, 32);
        assert_eq!(a.per_sender_quota, 8);
        assert_eq!(a.advisory_depth, 16);
        assert_eq!(a.max_live_handles, 16);
        assert_eq!(a.granted_artifacts, run_artifacts);
    }

    #[test]
    fn exceeding_the_lane_is_refused() {
        let role = RoleGrants {
            channels: vec![auth_channel(1 << 24, 32, 8)], // frame > lane 1<<20
            ..Default::default()
        };
        let err = derive_admitted_quotas(&role, &lane(), &BTreeSet::new()).unwrap_err();
        assert!(matches!(
            err,
            GrantsError::BoundExceedsLane { ref bound, .. } if bound == "max_frame_bytes"
        ));
    }

    #[test]
    fn world_and_custom_op_and_artifact_membership_enforced() {
        let mut role = RoleGrants::default();
        role.worlds
            .insert("compute".to_string(), crate::genesis::WorldGrant::default());
        assert!(matches!(
            derive_admitted_quotas(&role, &lane(), &BTreeSet::new()),
            Err(GrantsError::WorldNotInLane(_))
        ));

        let mut role = RoleGrants::default();
        role.custom_ops.push("flash_attn@1".to_string());
        assert!(matches!(
            derive_admitted_quotas(&role, &lane(), &BTreeSet::new()),
            Err(GrantsError::CustomOpNotInLane(_))
        ));

        let mut role = RoleGrants::default();
        role.artifacts.insert(hash(9));
        assert!(matches!(
            derive_admitted_quotas(&role, &lane(), &BTreeSet::new()),
            Err(GrantsError::ArtifactNotInMap(_))
        ));
    }
}
