// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **owner-scoped resource arbiter** (decisions D6; refactor §9 Phase E) — the aggregate
//! admission/preemption authority over the N sandbox instances a node supervises, across all
//! runs on the host.
//!
//! Shape, exactly as designed in Phase 0.5 (decisions D6):
//!
//! - **Typed ledgers, not a scalar sum**: one **device-memory ledger per accelerator** (memory is
//!   not fungible across devices) plus **host-wide ledgers** for RAM, disk/cache, network
//!   up/down, and duty cycle, all under one [`OwnerBudget`]. Quantities are raw `u64` bytes /
//!   bits-per-second (ABI §9.6 units rule); duty is percent of the owner's duty grant.
//! - **One accelerator per role-instance** (the interim binding rule): an instance's `device`
//!   tier charges exactly one device's ledger; its workspace is a device-local cost on the same
//!   ledger. Multi-device role-instances are out of scope (deferred past Phase E).
//! - **All three claim tiers are disjointly summed and reserved** ([`ClaimTiers`]):
//!   `hard_accountable` (charged — breach is the module's typed trap), `declared_peak`
//!   (reserved headroom), and `workspace` (reserved as host-computed overhead — the host MAY
//!   substitute its measured estimate, and the **larger** of declared vs measured is reserved,
//!   ABI §9.6). Charging only the hard tier would overcommit the device.
//! - **Admission is a single atomic check-and-reserve** ([`OwnerArbiter::admit`]) against
//!   *remaining* budget under one lock: two concurrent admissions can never both pass against
//!   the same remaining budget. A reservation is committed before the instance is created and
//!   released only on **observed teardown** ([`OwnerArbiter::release`]).
//! - **Preemption releases before the replacement is admitted**
//!   ([`OwnerArbiter::plan_preemption`] picks strictly-lower-priority victims; it never releases
//!   them — the caller throttles/pauses the victims and calls `release` only when their teardown
//!   is *observed*, then admits the replacement; admitting optimistically against memory a
//!   victim has not surrendered would double-commit the device).
//! - **Owner priority is node-side state** (the policy store; never the envelope).
//! - **Crash reconciliation** ([`OwnerArbiter::reconcile`]): on restart the ledger converges to
//!   the set of genuinely-live incarnations — reservations with no live incarnation are
//!   reclaimed, live incarnations with no reservation are re-charged from their recorded claim.
//!
//! The keys are the frozen execution identity's demux tuple (ABI §8.1, decisions D1):
//! [`RoleInstanceId`] `{ run_id, epoch, role, instance }`, where `instance` is the never-reused,
//! node-durable, monotonic u64 incarnation id ([`crate::VhcStore::mint_incarnation`]).

use std::collections::BTreeMap;
use std::sync::Mutex;

use daemon_vhc_session::config::OwnerBudgetConfig;
use daemon_vhc_session::protocol::Hardware;

/// A device-ledger key: the probe/owner-config accelerator identifier (e.g. `"gpu:0"`).
pub type DeviceId = String;

/// The frozen execution-identity demux key (ABI §8.1; decisions D1): `module_hash` is omitted
/// per invariant D1-EPOCH (every module transition advances the epoch).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RoleInstanceId {
    /// The run identity — the 32-byte genesis-envelope hash (`RunId`); a v1-era run without one
    /// uses `blake3(RunLabel)` as its node-local stand-in (documented at the call site).
    pub run_id: [u8; 32],
    /// The transition-chain epoch.
    pub epoch: u64,
    /// The envelope-level role label (opaque to the host beyond lane selection).
    pub role: String,
    /// The never-reused durable u64 role-instance incarnation id.
    pub instance: u64,
}

/// One claim tier's split charge (ABI §9.6 `tier-bytes`): device-local vs host-side bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TierBytes {
    /// Bytes charged to the bound accelerator's device-memory ledger.
    pub device: u64,
    /// Bytes charged to the host-RAM ledger.
    pub host: u64,
}

impl TierBytes {
    /// Element-wise max — the ABI §9.6 workspace rule (larger of declared vs host-measured).
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self {
            device: self.device.max(other.device),
            host: self.host.max(other.host),
        }
    }
}

/// The three disjoint claim tiers (ABI §9.1/§9.6; decisions D6 point 3). A v1-era instance maps
/// its autotune VRAM verdict onto `hard_accountable` with zero peak/workspace.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClaimTiers {
    /// Charged (hard cap; breach is the module's typed `BudgetMemory`/`BudgetHandles` trap).
    pub hard_accountable: TierBytes,
    /// Reserved headroom (the declared expected high-water mark).
    pub declared_peak: TierBytes,
    /// Reserved host-computed overhead (kernel temporaries, allocator slack, compiler caches).
    pub workspace: TierBytes,
}

impl ClaimTiers {
    /// Apply the host's measured workspace estimate: the **larger** of declared vs measured is
    /// what the ledger reserves (ABI §9.6 mapping).
    #[must_use]
    pub fn with_measured_workspace(mut self, measured: TierBytes) -> Self {
        self.workspace = self.workspace.max(measured);
        self
    }

    /// The instance's total device-ledger footprint: the three tiers disjointly summed.
    #[must_use]
    pub fn device_total(&self) -> u64 {
        self.hard_accountable
            .device
            .saturating_add(self.declared_peak.device)
            .saturating_add(self.workspace.device)
    }

    /// The instance's total host-RAM-ledger footprint.
    #[must_use]
    pub fn host_total(&self) -> u64 {
        self.hard_accountable
            .host
            .saturating_add(self.declared_peak.host)
            .saturating_add(self.workspace.host)
    }
}

/// Everything one role-instance charges against the owner's ledgers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceCharge {
    /// The one accelerator this role-instance is bound to for its lifetime (D6 point 2).
    pub device: DeviceId,
    /// The three-tier memory claim.
    pub tiers: ClaimTiers,
    /// Disk/cache bytes.
    pub disk_bytes: u64,
    /// Network uplink, bits per second.
    pub net_up_bps: u64,
    /// Network downlink, bits per second.
    pub net_down_bps: u64,
    /// Duty-cycle percent (`JoinPolicy.duty_cycle_pct` — D6 point 4).
    pub duty_pct: u8,
}

impl InstanceCharge {
    /// A device-memory-only charge (the common v1 autotune shape): `bytes` hard-accountable on
    /// `device`, `duty_pct` duty, nothing else.
    #[must_use]
    pub fn device_memory(device: impl Into<DeviceId>, bytes: u64, duty_pct: u8) -> Self {
        Self {
            device: device.into(),
            tiers: ClaimTiers {
                hard_accountable: TierBytes {
                    device: bytes,
                    host: 0,
                },
                ..ClaimTiers::default()
            },
            disk_bytes: 0,
            net_up_bps: 0,
            net_down_bps: 0,
            duty_pct,
        }
    }
}

/// The owner's standing aggregate grants — node-global, cross-run, cross-role (D6 point 1).
/// The per-run `JoinPolicy` remains a tightening overlay; it can only narrow these, never
/// exceed them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnerBudget {
    /// One device-memory ledger per accelerator, keyed by device id, in bytes.
    pub device_memory: BTreeMap<DeviceId, u64>,
    /// Host RAM, bytes.
    pub host_ram: u64,
    /// Disk/cache, bytes.
    pub disk: u64,
    /// Network uplink, bits per second.
    pub net_up_bps: u64,
    /// Network downlink, bits per second.
    pub net_down_bps: u64,
    /// The duty ledger, in percent (100 = the owner grants one full accelerator-duty).
    pub duty_pct: u32,
    /// Max concurrently-admitted role-instances.
    pub max_instances: u32,
}

impl OwnerBudget {
    /// A permissive budget (everything unbounded) — the explicit opt-out (`[vhc.owner_budget]
    /// unbounded = true`) and the default in tests; the arbiter still enforces key uniqueness and
    /// teardown ordering.
    #[must_use]
    pub fn unbounded() -> Self {
        Self {
            device_memory: BTreeMap::from([(String::from("gpu:0"), u64::MAX)]),
            host_ram: u64::MAX,
            disk: u64::MAX,
            net_up_bps: u64::MAX,
            net_down_bps: u64::MAX,
            duty_pct: u32::MAX,
            max_instances: u32::MAX,
        }
    }

    /// The conservative finite ceiling on concurrently-admitted role-instances applied when the
    /// owner does not configure `max_instances` (a finite bound — never `u32::MAX`).
    pub const DEFAULT_MAX_INSTANCES: u32 = 4;

    /// Map an `[vhc.owner_budget]` config + an optional hardware probe into the standing owner
    /// ledgers (decisions D6). `cfg.unbounded` short-circuits to [`Self::unbounded`]; otherwise
    /// every ledger left at its zero/empty default is derived **conservatively and finitely** —
    /// from the probe where one is available, else a documented fixed floor — so an enabled node
    /// with no configured budget still bounds admission rather than granting everything. The exact
    /// per-field derivation is documented on [`OwnerBudgetConfig`]; `data_cache_gb` is the
    /// `[vhc]` cache bound the disk ledger defaults to.
    #[must_use]
    pub fn from_config(cfg: &OwnerBudgetConfig, hw: Option<&Hardware>, data_cache_gb: u32) -> Self {
        const MIB: u64 = 1 << 20;
        // Conservative finite floors when neither config nor probe supplies a value.
        const FLOOR_DEVICE_MB: u64 = 4096; // 4 GiB
        const FLOOR_HOST_MB: u64 = 8192; // 8 GiB
        const FLOOR_NET_KBPS: u64 = 1_000_000; // 1 Gbit/s finite ceiling

        if cfg.unbounded {
            return Self::unbounded();
        }

        // `cfg` wins when set (> 0); else the probed value when present (> 0); else the floor.
        let pick = |configured: u64, probed: Option<u64>, floor: u64| -> u64 {
            if configured > 0 {
                configured
            } else {
                probed.filter(|v| *v > 0).unwrap_or(floor)
            }
        };

        // Device ledgers: explicit config wins; else a single `gpu:0` ledger sized to the probed
        // dedicated VRAM (v2.0 fleets are single-accelerator-per-member, spec §10 FT-4); else the
        // conservative floor.
        let device_memory: BTreeMap<DeviceId, u64> = if cfg.device_memory_mb.is_empty() {
            let vram_mb = hw
                .map(|h| h.vram_mb)
                .filter(|v| *v > 0)
                .unwrap_or(FLOOR_DEVICE_MB);
            BTreeMap::from([(String::from("gpu:0"), vram_mb.saturating_mul(MIB))])
        } else {
            cfg.device_memory_mb
                .iter()
                .map(|(dev, mb)| (dev.clone(), mb.saturating_mul(MIB)))
                .collect()
        };

        Self {
            device_memory,
            host_ram: pick(cfg.host_ram_mb, hw.map(|h| h.ram_mb), FLOOR_HOST_MB)
                .saturating_mul(MIB),
            disk: pick(
                cfg.disk_mb,
                Some(u64::from(data_cache_gb).saturating_mul(1024)),
                0,
            )
            .saturating_mul(MIB),
            net_up_bps: pick(cfg.net_up_kbps, hw.map(|h| h.up_kbps), FLOOR_NET_KBPS)
                .saturating_mul(1000),
            net_down_bps: pick(cfg.net_down_kbps, hw.map(|h| h.down_kbps), FLOOR_NET_KBPS)
                .saturating_mul(1000),
            duty_pct: if cfg.duty_pct > 0 { cfg.duty_pct } else { 100 },
            max_instances: if cfg.max_instances > 0 {
                cfg.max_instances
            } else {
                Self::DEFAULT_MAX_INSTANCES
            },
        }
    }
}

/// A typed admission refusal (the funnel's last stage — D6 point 5; supreme, owner can always
/// refuse). Carries the observed-vs-remaining numbers so the refusal names the offending value.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdmitRefusal {
    /// The charge names a device the owner has no ledger for.
    UnknownDevice {
        /// The unknown device id.
        device: DeviceId,
    },
    /// The bound device's memory ledger cannot fit the three-tier sum.
    DeviceMemoryExhausted {
        /// The bound device.
        device: DeviceId,
        /// The requested three-tier device total, bytes.
        requested: u64,
        /// The ledger's remaining bytes.
        remaining: u64,
    },
    /// The host-RAM ledger cannot fit the three-tier host sum.
    HostRamExhausted {
        /// Requested bytes.
        requested: u64,
        /// Remaining bytes.
        remaining: u64,
    },
    /// The disk/cache ledger is exhausted.
    DiskExhausted {
        /// Requested bytes.
        requested: u64,
        /// Remaining bytes.
        remaining: u64,
    },
    /// The network-uplink ledger is exhausted.
    NetUpExhausted {
        /// Requested bits per second.
        requested: u64,
        /// Remaining bits per second.
        remaining: u64,
    },
    /// The network-downlink ledger is exhausted.
    NetDownExhausted {
        /// Requested bits per second.
        requested: u64,
        /// Remaining bits per second.
        remaining: u64,
    },
    /// The duty ledger is exhausted.
    DutyExhausted {
        /// Requested duty percent.
        requested: u32,
        /// Remaining duty percent.
        remaining: u32,
    },
    /// The owner's instance ceiling is reached.
    MaxInstances {
        /// The configured ceiling.
        max: u32,
    },
    /// The execution identity is already reserved (an incarnation id is never reused, so this is
    /// always a caller bug or a replayed admission).
    DuplicateInstance,
}

impl std::fmt::Display for AdmitRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownDevice { device } => write!(f, "no owner ledger for device `{device}`"),
            Self::DeviceMemoryExhausted {
                device,
                requested,
                remaining,
            } => write!(
                f,
                "device `{device}` memory exhausted: requested {requested} B, remaining \
                 {remaining} B"
            ),
            Self::HostRamExhausted {
                requested,
                remaining,
            } => write!(
                f,
                "host RAM exhausted: requested {requested} B, remaining {remaining} B"
            ),
            Self::DiskExhausted {
                requested,
                remaining,
            } => write!(
                f,
                "disk exhausted: requested {requested} B, remaining {remaining} B"
            ),
            Self::NetUpExhausted {
                requested,
                remaining,
            } => write!(
                f,
                "uplink exhausted: requested {requested} bps, remaining {remaining} bps"
            ),
            Self::NetDownExhausted {
                requested,
                remaining,
            } => write!(
                f,
                "downlink exhausted: requested {requested} bps, remaining {remaining} bps"
            ),
            Self::DutyExhausted {
                requested,
                remaining,
            } => write!(
                f,
                "duty cycle exhausted: requested {requested}%, remaining {remaining}%"
            ),
            Self::MaxInstances { max } => write!(f, "owner instance ceiling reached ({max})"),
            Self::DuplicateInstance => write!(f, "execution identity already reserved"),
        }
    }
}

impl std::error::Error for AdmitRefusal {}

/// One committed reservation.
#[derive(Debug, Clone)]
struct Reserved {
    charge: InstanceCharge,
    priority: u8,
}

/// A point-in-time view of remaining budget (observability / tests).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetSnapshot {
    /// Remaining bytes per device ledger.
    pub device_memory: BTreeMap<DeviceId, u64>,
    /// Remaining host RAM, bytes.
    pub host_ram: u64,
    /// Remaining disk, bytes.
    pub disk: u64,
    /// Remaining uplink, bps.
    pub net_up_bps: u64,
    /// Remaining downlink, bps.
    pub net_down_bps: u64,
    /// Remaining duty percent.
    pub duty_pct: u32,
    /// Live reservations.
    pub instances: u32,
}

struct Ledgers {
    budget: OwnerBudget,
    entries: BTreeMap<RoleInstanceId, Reserved>,
}

impl Ledgers {
    /// Remaining budget with `exclude` treated as already released (the preemption planner's
    /// hypothetical view; `exclude` empty = the real remaining).
    fn remaining_excluding(&self, exclude: &[&RoleInstanceId]) -> BudgetSnapshot {
        let mut snap = BudgetSnapshot {
            device_memory: self.budget.device_memory.clone(),
            host_ram: self.budget.host_ram,
            disk: self.budget.disk,
            net_up_bps: self.budget.net_up_bps,
            net_down_bps: self.budget.net_down_bps,
            duty_pct: self.budget.duty_pct,
            instances: 0,
        };
        for (id, r) in &self.entries {
            if exclude.contains(&id) {
                continue;
            }
            snap.instances += 1;
            if let Some(d) = snap.device_memory.get_mut(&r.charge.device) {
                *d = d.saturating_sub(r.charge.tiers.device_total());
            }
            snap.host_ram = snap.host_ram.saturating_sub(r.charge.tiers.host_total());
            snap.disk = snap.disk.saturating_sub(r.charge.disk_bytes);
            snap.net_up_bps = snap.net_up_bps.saturating_sub(r.charge.net_up_bps);
            snap.net_down_bps = snap.net_down_bps.saturating_sub(r.charge.net_down_bps);
            snap.duty_pct = snap.duty_pct.saturating_sub(u32::from(r.charge.duty_pct));
        }
        snap
    }

    /// The admission predicate against a remaining-budget view — the D6 point-5 math.
    fn check(
        snap: &BudgetSnapshot,
        budget: &OwnerBudget,
        c: &InstanceCharge,
    ) -> Result<(), AdmitRefusal> {
        if snap.instances >= budget.max_instances {
            return Err(AdmitRefusal::MaxInstances {
                max: budget.max_instances,
            });
        }
        let Some(dev_remaining) = snap.device_memory.get(&c.device) else {
            return Err(AdmitRefusal::UnknownDevice {
                device: c.device.clone(),
            });
        };
        let dev_need = c.tiers.device_total();
        if dev_need > *dev_remaining {
            return Err(AdmitRefusal::DeviceMemoryExhausted {
                device: c.device.clone(),
                requested: dev_need,
                remaining: *dev_remaining,
            });
        }
        let host_need = c.tiers.host_total();
        if host_need > snap.host_ram {
            return Err(AdmitRefusal::HostRamExhausted {
                requested: host_need,
                remaining: snap.host_ram,
            });
        }
        if c.disk_bytes > snap.disk {
            return Err(AdmitRefusal::DiskExhausted {
                requested: c.disk_bytes,
                remaining: snap.disk,
            });
        }
        if c.net_up_bps > snap.net_up_bps {
            return Err(AdmitRefusal::NetUpExhausted {
                requested: c.net_up_bps,
                remaining: snap.net_up_bps,
            });
        }
        if c.net_down_bps > snap.net_down_bps {
            return Err(AdmitRefusal::NetDownExhausted {
                requested: c.net_down_bps,
                remaining: snap.net_down_bps,
            });
        }
        if u32::from(c.duty_pct) > snap.duty_pct {
            return Err(AdmitRefusal::DutyExhausted {
                requested: u32::from(c.duty_pct),
                remaining: snap.duty_pct,
            });
        }
        Ok(())
    }
}

/// The owner arbiter: [`OwnerBudget`] + the live reservation set behind one lock, so
/// check-and-reserve is intrinsically atomic (D6 point 5).
pub struct OwnerArbiter {
    inner: Mutex<Ledgers>,
}

impl OwnerArbiter {
    /// An arbiter over `budget`.
    #[must_use]
    pub fn new(budget: OwnerBudget) -> Self {
        Self {
            inner: Mutex::new(Ledgers {
                budget,
                entries: BTreeMap::new(),
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Ledgers> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// **Atomic check-and-reserve** (D6 point 5): admit `id` with `charge` at `priority` iff
    /// every ledger it touches has room in its *remaining* budget; the reservation is committed
    /// before this returns. Refusals are typed and name the offending value.
    ///
    /// # Errors
    /// The first exhausted ledger, in the fixed check order: instance ceiling → device memory →
    /// host RAM → disk → uplink → downlink → duty.
    pub fn admit(
        &self,
        id: RoleInstanceId,
        charge: InstanceCharge,
        priority: u8,
    ) -> Result<(), AdmitRefusal> {
        let mut inner = self.lock();
        if inner.entries.contains_key(&id) {
            return Err(AdmitRefusal::DuplicateInstance);
        }
        let snap = inner.remaining_excluding(&[]);
        Ledgers::check(&snap, &inner.budget, &charge)?;
        inner.entries.insert(id, Reserved { charge, priority });
        Ok(())
    }

    /// Release `id`'s reservation — called only on **observed teardown** (the child has actually
    /// surrendered its allocations, D6 point 6; never optimistically). Returns whether a
    /// reservation existed.
    pub fn release(&self, id: &RoleInstanceId) -> bool {
        self.lock().entries.remove(id).is_some()
    }

    /// Plan a preemption for an `incoming` charge that currently does not fit: choose the
    /// minimal set of **strictly lower-priority** victims (lowest priority first; ties broken by
    /// key order for determinism) whose release would make the incoming admission pass.
    ///
    /// Returns `None` when no such set exists (the incoming admission stays refused — equal or
    /// higher-priority instances are never preempted). Victims are **not** released here: the
    /// caller pauses them (`Command::Throttle{paused}` — preemption-as-churn) and calls
    /// [`release`](Self::release) per victim on *observed teardown*, after which the replacement
    /// is admitted through the normal [`admit`](Self::admit) — never before (D6 point 6).
    #[must_use]
    pub fn plan_preemption(
        &self,
        incoming: &InstanceCharge,
        incoming_priority: u8,
    ) -> Option<Vec<RoleInstanceId>> {
        let inner = self.lock();
        // Candidates: strictly lower priority, lowest first, deterministic within a priority.
        let mut candidates: Vec<(&RoleInstanceId, &Reserved)> = inner
            .entries
            .iter()
            .filter(|(_, r)| r.priority < incoming_priority)
            .collect();
        candidates.sort_by(|a, b| (a.1.priority, a.0).cmp(&(b.1.priority, b.0)));

        let mut victims: Vec<&RoleInstanceId> = Vec::new();
        for (id, _) in &candidates {
            let snap = inner.remaining_excluding(&victims);
            if Ledgers::check(&snap, &inner.budget, incoming).is_ok() {
                break;
            }
            victims.push(id);
        }
        let snap = inner.remaining_excluding(&victims);
        if Ledgers::check(&snap, &inner.budget, incoming).is_ok() {
            Some(victims.into_iter().cloned().collect())
        } else {
            None
        }
    }

    /// **Crash reconciliation** (D6 point 7): converge the ledger to the set of genuinely-live
    /// incarnations. `live` is the ground truth — the journal's tag-13 instantiation records ∩
    /// the supervisor's restart sweep — each with its recorded claim + priority. Reservations
    /// with no live incarnation are **reclaimed**; live incarnations with no reservation are
    /// **re-charged**. Returns `(reclaimed, recharged)` counts.
    pub fn reconcile(&self, live: &[(RoleInstanceId, InstanceCharge, u8)]) -> (usize, usize) {
        let mut inner = self.lock();
        let live_keys: Vec<&RoleInstanceId> = live.iter().map(|(id, _, _)| id).collect();
        let stale: Vec<RoleInstanceId> = inner
            .entries
            .keys()
            .filter(|k| !live_keys.contains(k))
            .cloned()
            .collect();
        for k in &stale {
            inner.entries.remove(k);
        }
        let mut recharged = 0;
        for (id, charge, priority) in live {
            if !inner.entries.contains_key(id) {
                // Ground truth: the incarnation IS running, so its footprint exists whether or
                // not the ledger fits — re-charge unconditionally (the ledger reflects reality;
                // pressure resolution is the throttle lever's job, not reconciliation's).
                inner.entries.insert(
                    id.clone(),
                    Reserved {
                        charge: charge.clone(),
                        priority: *priority,
                    },
                );
                recharged += 1;
            }
        }
        (stale.len(), recharged)
    }

    /// The current remaining-budget snapshot.
    #[must_use]
    pub fn remaining(&self) -> BudgetSnapshot {
        self.lock().remaining_excluding(&[])
    }

    /// The owner's device-ledger ids, in deterministic (id) order — the placement domain.
    #[must_use]
    pub fn devices(&self) -> Vec<DeviceId> {
        self.lock().budget.device_memory.keys().cloned().collect()
    }

    /// The live reservation count.
    #[must_use]
    pub fn instances(&self) -> usize {
        self.lock().entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(tag: u8, role: &str, instance: u64) -> RoleInstanceId {
        RoleInstanceId {
            run_id: [tag; 32],
            epoch: 0,
            role: role.to_string(),
            instance,
        }
    }

    fn budget_2dev() -> OwnerBudget {
        OwnerBudget {
            device_memory: BTreeMap::from([
                ("gpu:0".to_string(), 10_000),
                ("gpu:1".to_string(), 4_000),
            ]),
            host_ram: 100_000,
            disk: 50_000,
            net_up_bps: 1_000,
            net_down_bps: 1_000,
            duty_pct: 100,
            max_instances: 8,
        }
    }

    /// D6 point 5: check-and-reserve is atomic and against *remaining* budget; the third
    /// instance that would jointly exceed the device is refused typed with the exact numbers.
    #[test]
    fn admission_is_against_remaining_budget_with_typed_refusals() {
        let arb = OwnerArbiter::new(budget_2dev());
        arb.admit(
            id(1, "trainer", 1),
            InstanceCharge::device_memory("gpu:0", 6_000, 40),
            100,
        )
        .unwrap();
        arb.admit(
            id(2, "verifier", 2),
            InstanceCharge::device_memory("gpu:0", 3_000, 40),
            100,
        )
        .unwrap();
        // 9000/10000 committed on gpu:0 — a 2000-byte charge no longer fits, and the refusal
        // carries requested vs remaining ("two runs each within cap can jointly exceed").
        let refusal = arb
            .admit(
                id(3, "trainer", 3),
                InstanceCharge::device_memory("gpu:0", 2_000, 10),
                100,
            )
            .unwrap_err();
        assert_eq!(
            refusal,
            AdmitRefusal::DeviceMemoryExhausted {
                device: "gpu:0".into(),
                requested: 2_000,
                remaining: 1_000,
            }
        );
        // Memory is NOT fungible across accelerators: the same charge fits gpu:1's own ledger.
        arb.admit(
            id(3, "trainer", 3),
            InstanceCharge::device_memory("gpu:1", 2_000, 10),
            100,
        )
        .unwrap();
        // A device the owner granted nothing on is a typed refusal, not a silent pass.
        assert_eq!(
            arb.admit(
                id(4, "trainer", 4),
                InstanceCharge::device_memory("gpu:9", 1, 0),
                100,
            )
            .unwrap_err(),
            AdmitRefusal::UnknownDevice {
                device: "gpu:9".into()
            }
        );
    }

    /// D6 point 3: the three tiers are disjointly summed (charging only hard_accountable would
    /// overcommit), and the workspace reservation takes max(declared, host-measured) (ABI §9.6).
    #[test]
    fn three_tiers_disjointly_summed_and_workspace_max_rule() {
        let tiers = ClaimTiers {
            hard_accountable: TierBytes {
                device: 1_000,
                host: 100,
            },
            declared_peak: TierBytes {
                device: 500,
                host: 50,
            },
            workspace: TierBytes {
                device: 200,
                host: 10,
            },
        }
        .with_measured_workspace(TierBytes {
            device: 800, // host measured MORE than declared → the larger is reserved
            host: 5,     // less than declared → declared stands
        });
        assert_eq!(
            tiers.workspace,
            TierBytes {
                device: 800,
                host: 10
            }
        );
        assert_eq!(tiers.device_total(), 1_000 + 500 + 800);
        assert_eq!(tiers.host_total(), 100 + 50 + 10);

        // A budget sized for the hard tier alone must REFUSE the full three-tier sum.
        let arb = OwnerArbiter::new(OwnerBudget {
            device_memory: BTreeMap::from([("gpu:0".to_string(), 1_500)]),
            ..budget_2dev()
        });
        let charge = InstanceCharge {
            device: "gpu:0".into(),
            tiers,
            disk_bytes: 0,
            net_up_bps: 0,
            net_down_bps: 0,
            duty_pct: 0,
        };
        assert!(matches!(
            arb.admit(id(1, "trainer", 1), charge, 100).unwrap_err(),
            AdmitRefusal::DeviceMemoryExhausted {
                requested: 2_300,
                ..
            }
        ));
    }

    /// The host-wide ledgers (RAM / disk / net / duty) and the instance ceiling are enforced
    /// independently of device memory.
    #[test]
    fn host_wide_ledgers_and_instance_ceiling() {
        let arb = OwnerArbiter::new(OwnerBudget {
            max_instances: 2,
            ..budget_2dev()
        });
        let base = |duty: u8| InstanceCharge {
            device: "gpu:0".into(),
            tiers: ClaimTiers::default(),
            disk_bytes: 30_000,
            net_up_bps: 600,
            net_down_bps: 100,
            duty_pct: duty,
        };
        arb.admit(id(1, "trainer", 1), base(60), 100).unwrap();
        // Disk: 30k remaining of 50k.
        assert!(matches!(
            arb.admit(id(2, "verifier", 2), base(10), 100).unwrap_err(),
            AdmitRefusal::DiskExhausted {
                requested: 30_000,
                remaining: 20_000
            }
        ));
        let mut small = base(60);
        small.disk_bytes = 0;
        // Uplink: 400 remaining of 1000.
        assert!(matches!(
            arb.admit(id(2, "verifier", 2), small.clone(), 100)
                .unwrap_err(),
            AdmitRefusal::NetUpExhausted {
                requested: 600,
                remaining: 400
            }
        ));
        small.net_up_bps = 0;
        // Duty: 40 remaining of 100.
        assert!(matches!(
            arb.admit(id(2, "verifier", 2), small.clone(), 100)
                .unwrap_err(),
            AdmitRefusal::DutyExhausted {
                requested: 60,
                remaining: 40
            }
        ));
        small.duty_pct = 40;
        arb.admit(id(2, "verifier", 2), small.clone(), 100).unwrap();
        // Ceiling: 2 of 2 live.
        small.duty_pct = 0;
        assert!(matches!(
            arb.admit(id(3, "trainer", 3), small, 100).unwrap_err(),
            AdmitRefusal::MaxInstances { max: 2 }
        ));
    }

    /// An incarnation id is never reused; a duplicate reservation is a typed refusal.
    #[test]
    fn duplicate_execution_identity_is_refused() {
        let arb = OwnerArbiter::new(budget_2dev());
        let charge = InstanceCharge::device_memory("gpu:0", 1, 0);
        arb.admit(id(1, "trainer", 1), charge.clone(), 100).unwrap();
        assert_eq!(
            arb.admit(id(1, "trainer", 1), charge, 100).unwrap_err(),
            AdmitRefusal::DuplicateInstance
        );
    }

    /// D6 point 6 — the strict preemption ordering: the planner names strictly-lower-priority
    /// victims only; the replacement's admission FAILS until the victim's release is observed
    /// (never optimistically against memory the victim has not surrendered), and passes after.
    #[test]
    fn preemption_releases_before_replacement_is_admitted() {
        let arb = OwnerArbiter::new(budget_2dev());
        // Low-priority tenant filling gpu:0.
        arb.admit(
            id(1, "trainer", 1),
            InstanceCharge::device_memory("gpu:0", 9_000, 20),
            10,
        )
        .unwrap();
        let incoming = InstanceCharge::device_memory("gpu:0", 8_000, 20);

        // An EQUAL-priority incoming cannot preempt (no victims); it stays refused.
        assert_eq!(arb.plan_preemption(&incoming, 10), None);
        // A higher-priority incoming plans the low-priority victim.
        let victims = arb.plan_preemption(&incoming, 200).unwrap();
        assert_eq!(victims, vec![id(1, "trainer", 1)]);

        // ORDERING: before the victim's observed teardown, admission still fails.
        assert!(matches!(
            arb.admit(id(2, "trainer", 2), incoming.clone(), 200)
                .unwrap_err(),
            AdmitRefusal::DeviceMemoryExhausted { .. }
        ));
        // Observed teardown → release → the replacement admits against the freed ledger.
        assert!(arb.release(&id(1, "trainer", 1)));
        arb.admit(id(2, "trainer", 2), incoming, 200).unwrap();
    }

    /// The planner picks the minimal victim prefix, lowest priority first, deterministic; and
    /// returns None when even all lower-priority releases would not fit the incoming charge.
    #[test]
    fn preemption_planner_is_minimal_and_priority_ordered() {
        let arb = OwnerArbiter::new(budget_2dev());
        arb.admit(
            id(1, "a", 1),
            InstanceCharge::device_memory("gpu:0", 4_000, 0),
            30,
        )
        .unwrap();
        arb.admit(
            id(2, "b", 2),
            InstanceCharge::device_memory("gpu:0", 3_000, 0),
            20,
        )
        .unwrap();
        arb.admit(
            id(3, "c", 3),
            InstanceCharge::device_memory("gpu:0", 2_000, 0),
            10,
        )
        .unwrap();
        // 1000 remaining. 5500 incoming at priority 25: releasing p10 (2000) then p20 (3000)
        // frees 6000 — the p30 tenant outranks the incoming and is untouched.
        let incoming = InstanceCharge::device_memory("gpu:0", 5_500, 0);
        let victims = arb.plan_preemption(&incoming, 25).unwrap();
        assert_eq!(victims, vec![id(3, "c", 3), id(2, "b", 2)]);
        // 9500 incoming can never fit even with both lower-priority victims gone (p30 holds
        // 4000 of 10000) → no plan; the refusal stands.
        let too_big = InstanceCharge::device_memory("gpu:0", 9_500, 0);
        assert_eq!(arb.plan_preemption(&too_big, 25), None);
    }

    /// D6 point 7 — crash reconciliation: a reserved entry with no live incarnation is
    /// reclaimed; a live incarnation with no reservation is re-charged from its recorded claim;
    /// the ledger converges to the genuinely-running set.
    #[test]
    fn crash_reconciliation_converges_to_ground_truth() {
        let arb = OwnerArbiter::new(budget_2dev());
        arb.admit(
            id(1, "trainer", 1),
            InstanceCharge::device_memory("gpu:0", 6_000, 10),
            100,
        )
        .unwrap();
        arb.admit(
            id(2, "verifier", 2),
            InstanceCharge::device_memory("gpu:0", 2_000, 10),
            100,
        )
        .unwrap();

        // Ground truth after a crash: instance 1 died (tag-13 record but no live child);
        // instance 3 is alive but was never re-reserved (reservation lost with the process).
        let live = vec![
            (
                id(2, "verifier", 2),
                InstanceCharge::device_memory("gpu:0", 2_000, 10),
                100u8,
            ),
            (
                id(3, "trainer", 3),
                InstanceCharge::device_memory("gpu:0", 1_500, 10),
                100u8,
            ),
        ];
        let (reclaimed, recharged) = arb.reconcile(&live);
        assert_eq!((reclaimed, recharged), (1, 1));
        assert_eq!(arb.instances(), 2);
        let snap = arb.remaining();
        assert_eq!(snap.device_memory["gpu:0"], 10_000 - 2_000 - 1_500);
    }

    /// Atomicity under contention: many threads race for the last slot; exactly one wins.
    #[test]
    fn concurrent_admissions_never_double_commit() {
        use std::sync::Arc;
        let arb = Arc::new(OwnerArbiter::new(OwnerBudget {
            device_memory: BTreeMap::from([("gpu:0".to_string(), 1_000)]),
            ..budget_2dev()
        }));
        let mut handles = Vec::new();
        for t in 0..16u64 {
            let arb = arb.clone();
            handles.push(std::thread::spawn(move || {
                arb.admit(
                    id(9, "trainer", t),
                    InstanceCharge::device_memory("gpu:0", 800, 0),
                    100,
                )
                .is_ok()
            }));
        }
        let wins: usize = handles
            .into_iter()
            .map(|h| usize::from(h.join().unwrap()))
            .sum();
        assert_eq!(
            wins, 1,
            "two admissions can never both pass against the same remaining budget"
        );
    }
}
