// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **Device Capability Report** — the participating node's statement of what it has
//! (`docs/specs/vhc-architecture-spec.md` §9.6 `[RC-4]`(3); §9.7 `[PC-3]`).
//!
//! A statement of **supply**, never of demand. The measurement behind it already existed and is
//! already per-platform; what did not exist was an assembled, digested *artifact* — so nothing could
//! be cited, compared across incarnations, or recorded in evidence.
//!
//! ## The budget is stable, not instantaneous
//!
//! [`DeviceMemoryBudget`] describes the **owner-allocated stable budget** for this run, not
//! whatever the device happens to have free at the moment of the probe. Instantaneous free memory
//! is the wrong quantity twice over: it moves under a co-tenant between the probe and the
//! allocation, and a run admitted against a transient high-water mark will be refused its own
//! claim later for reasons no operator can reconstruct. Volatile occupancy pressure is the
//! governor's business, and it is handled without mutating this report.
//!
//! ## A probe that cannot measure does not return zero
//!
//! `[PC-3]`: every probe returns a measured value or a **typed unavailability**. A zero resource
//! reading is an admission refusal wearing a measurement's clothes — it refuses the machine rather
//! than reporting the defect, and it sends whoever investigates to the wrong place.

use std::collections::BTreeSet;

use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Hash};
use serde::{Deserialize, Serialize};

use crate::revision::{BackendClass, Maybe, Unavailable};

/// The report schema this build authors and accepts.
pub const DEVICE_CAPABILITY_REPORT_SCHEMA: u32 = 1;

/// Where a device-memory budget figure came from.
///
/// There is deliberately **no fallback variant**. A per-backend capability inventory that fell back
/// to the per-buffer ceiling when its real source was unavailable was observed reporting a
/// two-gigabyte device supply on a card with a thirty-gigabyte budget, in the same report whose
/// top-level figure was correct. That path is not inherited here, and making it unrepresentable is
/// what prevents inheriting it: a source that cannot be named cannot be silently substituted, and
/// the honest alternative — a typed unavailability — is what [`DeviceCapabilityReport`] carries
/// instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceMemorySource {
    /// The platform stated a **stable** budget for this process, and it was used as stated.
    ///
    /// Reserved for a platform figure that is a property of the device and the OS build rather than of
    /// the moment. A budget that moves with co-tenant pressure is not this: it is pressure, it belongs
    /// to the governor, and no arm currently qualifies — Linux's heap budget and the Windows local
    /// budget are both documented-dynamic, and both are recorded beside the report instead.
    PlatformStatedBudget,
    /// The validated static derivation for a unified device: the dedicated carve-out plus most of the
    /// shared pool, clamped by physical RAM and by the heap the backend advertises.
    ///
    /// The obvious reading is wrong on this class of device — the driver's carve-out on a unified part
    /// is not a statement of what the device may use, and reading it as one under-reports the supply by
    /// most of the machine's memory.
    UnifiedStaticDerivation,
    /// A discrete device's dedicated memory, which for a discrete card *is* the budget.
    DedicatedDeviceMemory,
}

/// How much of this device is **usable supply**, as measured on the node that has it.
///
/// Supply, not policy. The two used to share one field with one source enum that could name either a
/// platform query or an owner's allocation, which made the figure's meaning depend on a sibling —
/// admission could not compare against both because there was only ever one number, and whichever
/// authority wrote it last erased the other.
///
/// So this carries measured supply only, and its source enum names only host derivations: there is no
/// variant a human number could arrive through. An owner's wish to use less of their machine is a
/// separate, optional [`OwnerDeviceCap`], and admission compares against each independently.
///
/// Deriving it is the platform adapter's job and it must be **conservative**. Where no trustworthy
/// derivation exists the honest answer is that the backend cannot be admitted on that platform — see
/// [`DeviceCapabilityReport::usable_device_supply_bytes`]. Asking an operator for their card's number
/// is not an alternative: it inverts the responsibility, since the node is the thing that can measure
/// and the human is the thing that cannot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMemorySupply {
    /// Usable device memory in bytes, conservatively derived.
    pub usable_bytes: u64,
    /// Which **derivation** produced it. No variant admits a human-supplied figure.
    ///
    /// A derivation, not a platform. The source used to be picked from the platform before the
    /// derivation branched, so a report read the same whether the platform had stated the figure or our
    /// arithmetic had hit a clamp — one name for two different provenances, which is the defect class
    /// that also put one lane's adapter identity in another lane's record. The platform is a separate
    /// member and the clamps below say which bound actually bit.
    pub source: DeviceMemorySource,
    /// Whose facts the derivation read.
    pub platform: SupplyPlatform,
    /// Whether the physical-RAM clamp is what bound the figure.
    ///
    /// Recorded because a clamp that bites is the difference between a derivation and a limit: a
    /// variable-graphics-memory part whose figure came out at exactly physical RAM was clamped, not
    /// measured, and a reader comparing two boxes should be able to see that without re-deriving.
    pub ram_clamp_bound: bool,
    /// Whether the backend's advertised device heap is what bound the figure.
    pub device_heap_clamp_bound: bool,
}

/// The platform whose facts a supply derivation is reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupplyPlatform {
    /// Linux.
    Linux,
    /// macOS.
    Macos,
    /// Windows.
    Windows,
    /// The CUDA driver, whose device-total query is authoritative regardless of OS.
    Cuda,
}

/// The host facts a supply derivation reads. Every one of them is measured on this node.
///
/// A flat struct of platform facts rather than a platform-specific type, so the derivation is a pure
/// function that can be tested against each platform's real numbers without that platform. The probes
/// that produce these already exist and have existed since before the refactor; what was missing was
/// their connection to the resource model, which is what this closes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostDeviceFacts {
    /// Which platform's facts these are.
    pub platform: SupplyPlatform,
    /// Whether the device shares host DRAM. On a unified device the dedicated figure is a carve-out
    /// rather than the budget, which is the misreading that produced a 4 GiB supply on a 32 GiB box.
    pub unified: bool,
    /// Dedicated device memory in bytes, `0` when unknown. A true lower bound, never the budget on a
    /// unified device.
    pub dedicated_bytes: u64,
    /// The shared / spillover pool the device can page into, in bytes. `0` for a classic discrete card.
    pub shared_pool_bytes: u64,
    /// Physical host RAM in bytes, `0` when unknown. The ceiling on a unified device: a driver may
    /// report tens of gigabytes of "video memory" that is really this same RAM.
    pub host_ram_bytes: u64,
    /// The platform's own budget query, where this platform has one: the DXGI local budget, Metal's
    /// recommended maximum working-set size, or a Vulkan memory-budget report.
    ///
    /// Preferred over any static derivation when present, because it is the platform stating what this
    /// process may use rather than us inferring it from totals.
    ///
    /// **A figure that moves with co-tenant pressure does not belong here.** Some platforms expose their
    /// budget as a live quantity that shrinks while another process holds memory; that is precisely the
    /// volatile pressure the governor owns, and the report is the *stable* statement admission compares
    /// against and cites by digest. A live figure here would give two probes of one idle machine two
    /// digests, and no artifact could cite the report it was produced beside.
    pub platform_budget_bytes: Option<u64>,
    /// What the backend itself advertises as the size of the device heap it allocates from, where the
    /// backend says. `None` when it does not.
    ///
    /// This bounds the static derivation, and it is the reason the static derivation is a *fallback*
    /// rather than an alternative. Adding most of a unified machine's spillover pool to the dedicated
    /// carve-out can exceed what the backend will ever hand out of its device heap: the same DRAM is
    /// presented to the allocator as a device heap of a fixed fraction, and a supply figure above that
    /// promises memory no allocation can reach. Admitting against it would put the refusal at the first
    /// allocation instead of at admission, which is where a supply figure exists to put it.
    pub advertised_device_heap_bytes: Option<u64>,
}

/// The fraction of the shared pool a static derivation will claim, in percent.
///
/// Ninety rather than a hundred, carried over from the derivation this restores: the shared pool is not
/// exclusively the device's, and claiming all of it produces a supply figure the machine cannot honour
/// once anything else on the box wants memory.
const SHARED_POOL_CLAIMABLE_PERCENT: u64 = 90;

/// Derive conservative usable device supply from host facts, or report that this platform has none.
///
/// The authority model in one function: the node measures, and where it cannot measure trustworthily it
/// says so. There is no parameter through which a human figure could enter, because the operator is the
/// one party in this system that cannot measure the device — asking them was a responsibility inversion,
/// and their legitimate wish to lend less is [`OwnerDeviceCap`], applied afterwards.
///
/// The order of preference is the point:
///
/// 1. **The platform's own budget query**, where it has one. The platform stating what this process may
///    use beats us inferring it from totals.
/// 2. **The validated static derivation** for a unified device: the dedicated carve-out plus most of the
///    shared pool, clamped by physical RAM **and** by the heap the backend advertises. The RAM clamp
///    matters because a variable-graphics-memory part can report tens of gigabytes of "video memory"
///    that is the same RAM counted twice; the heap clamp matters because the backend hands out device
///    memory from a heap of its own choosing, and a supply figure above that heap promises memory no
///    allocation can reach. Both are conservative by construction, and a fallback that is not
///    conservative is not a fallback — it is a different, unvalidated answer.
/// 3. **The dedicated figure** for a discrete device, which for a discrete card *is* the budget.
/// 4. **Nothing**, typed, which fails closed.
#[must_use]
pub fn derive_device_supply(facts: &HostDeviceFacts) -> Maybe<DeviceMemorySupply> {
    let ram_ceiling = (facts.unified && facts.host_ram_bytes > 0).then_some(facts.host_ram_bytes);
    let heap_ceiling = facts.advertised_device_heap_bytes.filter(|h| *h > 0);
    // Which ceiling bit is recorded, not just applied: a figure that came out at exactly a ceiling was
    // clamped rather than derived, and a reader comparing two boxes should not have to re-derive to
    // learn that.
    let clamp = |bytes: u64| {
        let mut out = bytes;
        let mut ram_bound = false;
        let mut heap_bound = false;
        if let Some(ram) = ram_ceiling {
            if out > ram {
                out = ram;
                ram_bound = true;
            }
        }
        if let Some(heap) = heap_ceiling {
            if out > heap {
                out = heap;
                heap_bound = true;
                // A RAM clamp that was itself clamped away did not bind the answer.
                ram_bound = false;
            }
        }
        (out, ram_bound, heap_bound)
    };
    let supply = |usable_bytes: u64, source, (ram_clamp_bound, device_heap_clamp_bound)| {
        Maybe::Available(DeviceMemorySupply {
            usable_bytes,
            source,
            platform: facts.platform,
            ram_clamp_bound,
            device_heap_clamp_bound,
        })
    };

    if let Some(budget) = facts.platform_budget_bytes.filter(|b| *b > 0) {
        let (usable, ram_bound, heap_bound) = clamp(budget);
        return supply(
            usable,
            DeviceMemorySource::PlatformStatedBudget,
            (ram_bound, heap_bound),
        );
    }

    if facts.unified {
        let claimable = facts
            .shared_pool_bytes
            .saturating_mul(SHARED_POOL_CLAIMABLE_PERCENT)
            / 100;
        let (derived, ram_bound, heap_bound) =
            clamp(facts.dedicated_bytes.saturating_add(claimable));
        if derived > 0 {
            return supply(
                derived,
                DeviceMemorySource::UnifiedStaticDerivation,
                (ram_bound, heap_bound),
            );
        }
        return Maybe::Unavailable(Unavailable::ProbeFailed);
    }

    if facts.dedicated_bytes > 0 {
        let (usable, ram_bound, heap_bound) = clamp(facts.dedicated_bytes);
        return supply(
            usable,
            DeviceMemorySource::DedicatedDeviceMemory,
            (ram_bound, heap_bound),
        );
    }

    // Nothing trustworthy. `ProbeFailed` rather than "not exposed by the platform": every platform
    // this runs on *has* a way to report a budget, so an absence here is a defect to fix on this node
    // and not a property of the platform.
    Maybe::Unavailable(Unavailable::ProbeFailed)
}

/// An optional owner policy: use no more than this much of the device, whatever the supply is.
///
/// Node policy, deliberately not part of the capability report. An owner may always choose to lend
/// less of their machine than it has, and that is a wish rather than a measurement — keeping it out of
/// the report is what stops a preference from being read later as a hardware fact.
///
/// Absent means the owner has expressed no preference, which is **not** a cap of zero and not a cap of
/// infinity: admission simply has one comparison to make instead of two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnerDeviceCap {
    /// The most of this device the owner will lend, in bytes.
    pub max_bytes: u64,
}

/// The measured link capacity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkCapacity {
    /// Measured uplink, bits/s.
    pub uplink_bps: Maybe<u64>,
    /// Measured downlink, bits/s.
    pub downlink_bps: Maybe<u64>,
}

/// The digest of a backend implementation revision record, as a capability report cites it.
///
/// The report and the record are read together at admission — one says what the machine has, the other
/// says what is running on it — so the report cites the record it was produced beside rather than
/// restating any of it. Citing by digest is what makes "produced beside" checkable later.
///
/// # Errors
/// [`CapabilityError::Invalid`] when the record does not encode canonically.
pub fn revision_record_digest(
    record: &crate::revision::BackendImplementationRevision,
) -> Result<Hash, CapabilityError> {
    let bytes = to_canonical_vec(record)
        .map_err(|e| CapabilityError::Invalid(format!("revision record encoding: {e}")))?;
    Ok(blake3_hash(&bytes))
}

/// How a per-allocation ceiling was measured.
///
/// A method is **required** to accompany the number, which is why the report carries this type rather
/// than a bare `u64`. Every ceiling a platform *states* — a framework constant, a driver's advertised
/// buffer limit — is reachable from a probe without allocating anything, and promoting one of those into
/// a field whose contract says "measured" is the substitution that once reported a two-gigabyte device
/// supply on a card with thirty. Carrying the method makes the promotion unrepresentable: there is no
/// way to put a figure here without saying how it was obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllocationProbeMethod {
    /// Bounded bisection over **real device allocations**.
    ///
    /// Each candidate size is actually allocated and immediately released; the largest size the driver
    /// accepted is the ceiling. Bounded in both directions on purpose: `start_bytes` caps how much of a
    /// live machine the probe will ask for, and `attempts` caps how long it can take, so the measurement
    /// is conservative and non-disruptive rather than exhaustive.
    BoundedBisection {
        /// The largest size the search was permitted to attempt.
        start_bytes: u64,
        /// The smallest size the search would consider.
        floor_bytes: u64,
        /// How many allocations were attempted.
        attempts: u32,
    },
}

/// The largest single allocation the driver was **measured** to accept, and how that was established.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasuredAllocationCeiling {
    /// The largest single allocation the driver accepted.
    pub accepted_bytes: u64,
    /// The smallest allocation it refused, when the search found one.
    ///
    /// `None` means the search never saw a refusal — the ceiling is at least the largest size the probe
    /// was permitted to attempt, and the real limit may be higher. That is a bound the probe declined to
    /// exceed, not a limit of the device, and a reader should be able to tell the two apart.
    pub refused_bytes: Option<u64>,
    /// How it was measured.
    pub method: AllocationProbeMethod,
}

/// Whether this device's memory and the host's are one physical pool or two.
///
/// Carried in the report because the report states two memory figures, and on a unified device those
/// two figures are **the same DRAM described twice**. A reader that adds them, or that spends against
/// one without counting the other, has over-committed the machine while every individual comparison
/// passed — the failure then arrives at an allocation rather than at admission, attributed to whichever
/// side happened to ask last.
///
/// A device the platform reports as unified is not a special case to tolerate; it is the fleet's own
/// hardware. Making the topology explicit is what lets [`admit_node_memory_bytes`] make the joint
/// comparison instead of leaving each caller to remember that it should.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPoolTopology {
    /// Device memory and host memory are separate physical pools, independently available.
    Separate,
    /// One physical pool serves both. The device and host figures are not additive supply.
    Unified,
}

/// What one participating node reports about one device.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCapabilityReport {
    /// Schema version.
    pub schema: u32,
    /// The backend class this device would serve.
    pub backend_class: BackendClass,
    /// The adapter's reported name, for a human reading a refusal.
    pub adapter_name: String,
    /// Usable device memory as **measured supply**, or a typed absence when this platform has no
    /// trustworthy derivation.
    ///
    /// Absent fails closed: a device whose usable supply cannot be derived cannot be admitted, because
    /// every comparison admission makes about device memory is against this figure. Substituting
    /// anything — a per-buffer ceiling, a driver carve-out, an operator's estimate — would put a number
    /// of unknown meaning where a measurement belongs, which is how a two-gigabyte reading once stood
    /// in for a thirty-gigabyte budget in a report whose other figures were correct.
    pub device_supply: Maybe<DeviceMemorySupply>,
    /// Whether the device supply and [`Self::host_memory_bytes`] describe one physical pool or two.
    pub memory_pool: MemoryPoolTopology,
    /// The largest single physical allocation the driver was **measured** to accept. Typed
    /// unavailability when it has not been probed — the reported ceiling is not evidence of what
    /// the driver enforces, and the two are carried separately in the profile for the same reason.
    pub measured_max_allocation: Maybe<MeasuredAllocationCeiling>,
    /// Usable host memory, bytes.
    pub host_memory_bytes: Maybe<u64>,
    /// Usable free disk, bytes.
    pub disk_bytes: Maybe<u64>,
    /// The operation families this device supports.
    pub supported_operation_families: BTreeSet<String>,
    /// The dtype spellings it supports.
    pub supported_dtypes: BTreeSet<String>,
    /// The measured link capacity.
    pub link: LinkCapacity,
    /// The digest of the backend implementation revision record this report was produced beside.
    /// The two are read together at admission: a report says what the machine has, the revision
    /// record says what is running on it, and a profile has to match both.
    pub implementation_revision_digest: Hash,
    /// The digest of the certified Backend Execution Profile that applies to this device, once one
    /// has been resolved.
    pub applicable_profile_digest: Maybe<Hash>,
}

/// Why a capability report is not usable.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CapabilityError {
    /// The report is not a well-formed schema-1 document.
    #[error("device capability report is invalid: {0}")]
    Invalid(String),
    /// A probe returned zero rather than failing loud.
    #[error(
        "device capability report states {quantity} is zero; a probe that cannot measure must \
         return a typed unavailability, because a zero reading refuses the machine instead of \
         reporting the defect"
    )]
    ZeroInsteadOfUnavailable {
        /// Which quantity.
        quantity: &'static str,
    },
    /// A required measurement is absent, so the property depending on it cannot be evaluated.
    #[error("device capability report has no measured {quantity}: {detail}")]
    Unmeasured {
        /// Which quantity.
        quantity: &'static str,
        /// Why it matters.
        detail: &'static str,
    },
}

impl DeviceCapabilityReport {
    /// The report's canonical CBOR bytes.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, CapabilityError> {
        to_canonical_vec(self).map_err(|e| CapabilityError::Invalid(format!("encoding: {e}")))
    }

    /// blake3 of the canonical bytes — the capability-report digest the admitted tuple and the
    /// composition evidence record carry.
    pub fn report_digest(&self) -> Result<Hash, CapabilityError> {
        Ok(blake3_hash(&self.to_canonical_bytes()?))
    }

    /// Validate the report.
    pub fn validate(&self) -> Result<(), CapabilityError> {
        if self.schema != DEVICE_CAPABILITY_REPORT_SCHEMA {
            return Err(CapabilityError::Invalid(format!(
                "unknown report schema {} (this build understands \
                 {DEVICE_CAPABILITY_REPORT_SCHEMA})",
                self.schema
            )));
        }
        if self.adapter_name.trim().is_empty() {
            return Err(CapabilityError::Invalid(
                "a report must name the adapter it describes".into(),
            ));
        }
        // A zero is never a measurement. Absence is expressed by `Maybe`, and these three are the
        // quantities a failing probe historically reported as zero.
        if self
            .device_supply
            .value()
            .is_some_and(|s| s.usable_bytes == 0)
        {
            return Err(CapabilityError::ZeroInsteadOfUnavailable {
                quantity: "usable device supply",
            });
        }
        for (quantity, value) in [
            ("host memory", &self.host_memory_bytes),
            ("free disk", &self.disk_bytes),
        ] {
            if value.value().is_some_and(|v| *v == 0) {
                return Err(CapabilityError::ZeroInsteadOfUnavailable { quantity });
            }
        }
        if self
            .measured_max_allocation
            .value()
            .is_some_and(|c| c.accepted_bytes == 0)
        {
            return Err(CapabilityError::ZeroInsteadOfUnavailable {
                quantity: "the measured per-allocation ceiling",
            });
        }
        // A per-allocation ceiling above the whole budget is not a ceiling; it is the compile-time
        // constant one platform reports in place of one.
        if let (Some(ceiling), Some(supply)) = (
            self.measured_max_allocation.value(),
            self.device_supply.value(),
        ) {
            if ceiling.accepted_bytes > supply.usable_bytes {
                return Err(CapabilityError::Invalid(format!(
                    "the measured per-allocation ceiling ({} bytes) exceeds the whole \
                     usable device supply ({} bytes), so it is not a ceiling this device can honour",
                    ceiling.accepted_bytes, supply.usable_bytes
                )));
            }
        }
        // A refusal the search never saw cannot be below the size that was accepted.
        if let Some(ceiling) = self.measured_max_allocation.value() {
            if ceiling
                .refused_bytes
                .is_some_and(|refused| refused <= ceiling.accepted_bytes)
            {
                return Err(CapabilityError::Invalid(
                    "the allocation probe records a refusal at or below the size it accepted, which \
                     cannot both be true of one search"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    /// The per-allocation ceiling admission may compare a composed claim against.
    ///
    /// Measured or nothing. The reported figure is not evidence of what the driver enforces, and
    /// substituting it here would re-create the inventory fallback this artifact exists to avoid.
    pub fn max_allocation_bytes(&self) -> Result<u64, CapabilityError> {
        self.measured_max_allocation
            .value()
            .map(|ceiling| ceiling.accepted_bytes)
            .ok_or(CapabilityError::Unmeasured {
                quantity: "per-allocation ceiling",
                detail: "a composed claim's maximum individual allocation cannot be validated \
                         against an unmeasured limit",
            })
    }

    /// Usable device supply in bytes, or a typed refusal when this platform has no trustworthy
    /// derivation.
    ///
    /// **Fails closed.** A backend on a platform whose usable supply cannot be derived is not
    /// admissible there, and that is the whole answer: the alternative is to compare a claim against a
    /// figure of unknown meaning, which admits whatever that figure happens to be.
    pub fn usable_device_supply_bytes(&self) -> Result<u64, CapabilityError> {
        self.device_supply
            .value()
            .map(|supply| supply.usable_bytes)
            .ok_or(CapabilityError::Unmeasured {
                quantity: "usable device supply",
                detail: "no trustworthy derivation of usable device memory exists for this \
                         platform, so this backend cannot be admitted here; a human-supplied figure \
                         is not an alternative, because the node is what can measure and the \
                         operator is what cannot",
            })
    }
}

/// Admit a composed device figure against supply and, where set, the owner's cap.
///
/// **Two independent comparisons**, not a combined one. They answer different questions — the device
/// physically cannot serve more than its supply, and the owner has chosen to lend no more than their
/// cap — and collapsing them to a single `min` would lose which of the two refused. An operator whose
/// cap is the binding constraint should be told that, not told their hardware is too small.
///
/// A cap above the supply is not an error and not a correction: the owner is free to permit more than
/// this particular device happens to have, and the supply comparison refuses on its own if it must.
///
/// # Errors
/// [`CapabilityError`] naming which comparison refused, and by how much.
pub fn admit_device_bytes(
    claimed_bytes: u64,
    report: &DeviceCapabilityReport,
    owner_cap: Option<OwnerDeviceCap>,
) -> Result<(), DeviceAdmissionRefusal> {
    let supply = report.usable_device_supply_bytes().map_err(|_| {
        DeviceAdmissionRefusal::NoTrustworthySupply {
            adapter: report.adapter_name.clone(),
        }
    })?;
    if claimed_bytes > supply {
        return Err(DeviceAdmissionRefusal::ExceedsSupply {
            claimed_bytes,
            supply_bytes: supply,
            adapter: report.adapter_name.clone(),
        });
    }
    if let Some(cap) = owner_cap {
        if claimed_bytes > cap.max_bytes {
            return Err(DeviceAdmissionRefusal::ExceedsOwnerCap {
                claimed_bytes,
                cap_bytes: cap.max_bytes,
                supply_bytes: supply,
            });
        }
    }
    Ok(())
}

/// Admit a role's device **and** host figures against one node's report, respecting pool topology.
///
/// On a device with [`MemoryPoolTopology::Separate`] memory this is exactly the device comparison plus
/// the host one: two pools, two independent budgets, and nothing links them.
///
/// On a [`MemoryPoolTopology::Unified`] device the two figures are one DRAM pool described twice, so a
/// third comparison applies and it is the binding one: **the sum** must fit in the smaller of the
/// derived device supply and the measured host memory. Without it, a role claiming most of the supply
/// on the device side and most of host memory on the linear-memory side passes both comparisons and
/// over-commits the machine — the historical shape of this error, which the pre-refactor derivation
/// guarded with a joint-pool check that was lost with the rest of the arithmetic.
///
/// # Errors
/// [`DeviceAdmissionRefusal`] naming which comparison refused. The joint refusal is distinct from the
/// supply refusal on purpose: "your role fits the device but not beside its own host footprint" sends
/// an operator somewhere different than "your role does not fit this device".
pub fn admit_node_memory_bytes(
    device_claimed_bytes: u64,
    host_claimed_bytes: u64,
    report: &DeviceCapabilityReport,
    owner_cap: Option<OwnerDeviceCap>,
) -> Result<(), DeviceAdmissionRefusal> {
    admit_device_bytes(device_claimed_bytes, report, owner_cap)?;
    if report.memory_pool == MemoryPoolTopology::Separate {
        return Ok(());
    }
    // The pool is whichever of the two figures is smaller: the device side cannot exceed the supply
    // derivation, and neither side can exceed the physical RAM they share.
    let supply = report.usable_device_supply_bytes().map_err(|_| {
        DeviceAdmissionRefusal::NoTrustworthySupply {
            adapter: report.adapter_name.clone(),
        }
    })?;
    let pool = match report.host_memory_bytes.value() {
        Some(host) => supply.min(*host),
        // Host memory unmeasured: the device derivation is the only bound in hand, and using it alone
        // is conservative here because the derivation is already clamped by physical RAM.
        None => supply,
    };
    let joint = device_claimed_bytes.saturating_add(host_claimed_bytes);
    if joint > pool {
        return Err(DeviceAdmissionRefusal::ExceedsUnifiedPool {
            device_bytes: device_claimed_bytes,
            host_bytes: host_claimed_bytes,
            pool_bytes: pool,
            adapter: report.adapter_name.clone(),
        });
    }
    Ok(())
}

/// Which of the two device-memory comparisons refused.
///
/// Typed rather than a message, because the two are acted on differently: a supply refusal means this
/// device cannot host the role and a smaller configuration might, while a cap refusal means the owner
/// has chosen not to lend memory the device demonstrably has. Telling an operator their hardware is too
/// small when their own policy is the binding constraint sends them to buy a card they already own.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum DeviceAdmissionRefusal {
    /// This platform has no trustworthy derivation of usable device supply.
    #[error(
        "no trustworthy derivation of usable device memory exists for `{adapter}` on this \
         platform, so this backend cannot be admitted here; this fails closed rather than asking \
         an operator for a figure, because the node is what can measure the device"
    )]
    NoTrustworthySupply {
        /// The adapter that could not be derived for.
        adapter: String,
    },
    /// The claim exceeds what this node measured as usable supply.
    #[error(
        "the composed claim needs {claimed_bytes} bytes of device memory, above the \
         {supply_bytes} bytes this node measured as usable supply on `{adapter}`"
    )]
    ExceedsSupply {
        /// What the claim needs.
        claimed_bytes: u64,
        /// What the node measured.
        supply_bytes: u64,
        /// The adapter.
        adapter: String,
    },
    /// The claim exceeds the owner's cap, though the device has the memory.
    #[error(
        "the composed claim needs {claimed_bytes} bytes of device memory, above the {cap_bytes} \
         bytes the owner of this node has chosen to lend; the device itself has {supply_bytes} \
         bytes of usable supply, so this is a policy refusal and not a hardware one"
    )]
    ExceedsOwnerCap {
        /// What the claim needs.
        claimed_bytes: u64,
        /// The owner's cap.
        cap_bytes: u64,
        /// What the device actually has, so the refusal cannot be mistaken for a hardware limit.
        supply_bytes: u64,
    },
    /// Device and host figures each fit, but not together in the one pool they share.
    #[error(
        "on `{adapter}` the device memory and the host memory are one physical pool, and the role's \
         {device_bytes} device bytes beside its {host_bytes} host bytes exceed the {pool_bytes} \
         bytes that pool holds; each figure fits on its own, which is why this is checked jointly"
    )]
    ExceedsUnifiedPool {
        /// The device side of the claim.
        device_bytes: u64,
        /// The host side of the claim.
        host_bytes: u64,
        /// The shared pool the two compete for.
        pool_bytes: u64,
        /// The adapter.
        adapter: String,
    },
}

impl DeviceAdmissionRefusal {
    /// What was asked for, as the refusing comparison counted it.
    ///
    /// The joint refusal counts both sides, because a caller reporting "required N against available M"
    /// would otherwise print the device figure beside a pool bound that the host figure is what
    /// exceeded — a pair of numbers that do not explain each other.
    #[must_use]
    pub fn claimed_bytes(&self) -> u64 {
        match self {
            Self::NoTrustworthySupply { .. } => 0,
            Self::ExceedsSupply { claimed_bytes, .. }
            | Self::ExceedsOwnerCap { claimed_bytes, .. } => *claimed_bytes,
            Self::ExceedsUnifiedPool {
                device_bytes,
                host_bytes,
                ..
            } => device_bytes.saturating_add(*host_bytes),
        }
    }

    /// The limit that bound, for a caller reporting `required` against `available`.
    #[must_use]
    pub fn binding_limit_bytes(&self) -> u64 {
        match self {
            Self::NoTrustworthySupply { .. } => 0,
            Self::ExceedsSupply { supply_bytes, .. } => *supply_bytes,
            Self::ExceedsOwnerCap { cap_bytes, .. } => *cap_bytes,
            Self::ExceedsUnifiedPool { pool_bytes, .. } => *pool_bytes,
        }
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    /// A measured per-allocation ceiling for a fixture, as a bounded bisection would have found it.
    pub(crate) fn measured_ceiling(accepted_bytes: u64) -> MeasuredAllocationCeiling {
        MeasuredAllocationCeiling {
            accepted_bytes,
            refused_bytes: Some(accepted_bytes.saturating_mul(2)),
            method: AllocationProbeMethod::BoundedBisection {
                start_bytes: accepted_bytes.saturating_mul(2),
                floor_bytes: 1 << 20,
                attempts: 4,
            },
        }
    }

    /// A derived supply figure for a fixture: the static derivation, with nothing clamped.
    pub(crate) fn derived_supply(usable_bytes: u64) -> DeviceMemorySupply {
        DeviceMemorySupply {
            usable_bytes,
            source: DeviceMemorySource::UnifiedStaticDerivation,
            platform: SupplyPlatform::Linux,
            ram_clamp_bound: false,
            device_heap_clamp_bound: false,
        }
    }

    /// A complete, valid report.
    pub(crate) fn report(class: BackendClass) -> DeviceCapabilityReport {
        DeviceCapabilityReport {
            schema: DEVICE_CAPABILITY_REPORT_SCHEMA,
            backend_class: class,
            adapter_name: "Radeon 8060S Graphics (RADV GFX1151)".into(),
            device_supply: Maybe::Available(derived_supply(32_952_745_984)),
            memory_pool: MemoryPoolTopology::Unified,
            measured_max_allocation: Maybe::Available(measured_ceiling(4 << 30)),
            host_memory_bytes: Maybe::Available(64 << 30),
            disk_bytes: Maybe::Available(247 << 30),
            supported_operation_families: ["gemm".to_string()].into_iter().collect(),
            supported_dtypes: ["f32".to_string(), "bool1".to_string()]
                .into_iter()
                .collect(),
            link: LinkCapacity {
                uplink_bps: Maybe::Available(500_000_000),
                downlink_bps: Maybe::Available(1_000_000_000),
            },
            implementation_revision_digest: Hash([5u8; 32]),
            applicable_profile_digest: Maybe::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{derived_supply, measured_ceiling, report};
    use super::*;
    use crate::revision::Unavailable;

    #[test]
    fn a_complete_report_validates_and_digests_deterministically() {
        let r = report(BackendClass::Vulkan);
        r.validate().expect("valid");
        assert_eq!(
            r.report_digest().unwrap(),
            report(BackendClass::Vulkan).report_digest().unwrap()
        );
        assert_ne!(
            r.report_digest().unwrap(),
            report(BackendClass::Metal).report_digest().unwrap()
        );
    }

    /// **The Strix regression, fixed.** The amdgpu carve-out is a wrong derivation on a unified part,
    /// not a budget.
    ///
    /// This box reports 4 GiB of dedicated VRAM beside ~28 GiB of GTT on 32 GiB of physical RAM. Read
    /// as a budget, the carve-out under-reports usable supply by most of the machine — and a run
    /// admitted against 4 GiB on a 32 GiB box is refused work it could have done. The derivation this
    /// restores adds most of the shared pool and clamps by physical RAM.
    #[test]
    fn a_unified_device_derives_supply_from_the_shared_pool_not_the_carve_out() {
        let strix = HostDeviceFacts {
            platform: SupplyPlatform::Linux,
            unified: true,
            dedicated_bytes: 4 * 1024 * 1024 * 1024,
            shared_pool_bytes: 28 * 1024 * 1024 * 1024,
            host_ram_bytes: 32 * 1024 * 1024 * 1024,
            platform_budget_bytes: None,
            advertised_device_heap_bytes: None,
        };
        let supply = derive_device_supply(&strix)
            .value()
            .copied()
            .expect("a unified device with a shared pool derives supply");

        assert_eq!(supply.source, DeviceMemorySource::UnifiedStaticDerivation);
        // 4 GiB + 90% of 28 GiB = 29.2 GiB, under the 32 GiB physical ceiling. Multiply before
        // dividing, so the percentage does not lose a remainder to integer truncation.
        assert_eq!(
            supply.usable_bytes,
            4 * 1024 * 1024 * 1024 + (28 * 1024 * 1024 * 1024_u64 * 90 / 100)
        );
        assert!(
            supply.usable_bytes > 4 * 1024 * 1024 * 1024 * 7,
            "the derived supply is most of the machine, not the carve-out"
        );
        assert!(supply.usable_bytes < strix.host_ram_bytes);
    }

    /// The backend's own heap bounds the static fallback, so the fallback stays a fallback.
    ///
    /// This is the fleet's Strix box: a 4 GiB carve-out beside ~117 GiB of spillover pool on ~121 GiB
    /// of RAM, where the backend presents a device heap of ~81 GiB. The unclamped arithmetic yields
    /// ~109 GiB — under physical RAM, so the RAM clamp does not catch it, and above every allocation
    /// the backend will serve out of that heap. A supply figure the allocator cannot honour moves the
    /// refusal from admission to the first allocation, which is the one place it must never be.
    #[test]
    fn the_advertised_device_heap_bounds_the_static_fallback() {
        let mib = 1024 * 1024_u64;
        let strix = HostDeviceFacts {
            platform: SupplyPlatform::Linux,
            unified: true,
            dedicated_bytes: 4096 * mib,
            shared_pool_bytes: 120_000 * mib,
            host_ram_bytes: 124_096 * mib,
            platform_budget_bytes: None,
            advertised_device_heap_bytes: Some(82_730 * mib),
        };
        let clamped = derive_device_supply(&strix).value().copied().unwrap();
        assert_eq!(clamped.usable_bytes, 82_730 * mib);

        let unclamped = derive_device_supply(&HostDeviceFacts {
            advertised_device_heap_bytes: None,
            ..strix
        })
        .value()
        .copied()
        .unwrap();
        assert!(
            unclamped.usable_bytes > clamped.usable_bytes,
            "without the heap clamp the fallback exceeds what the backend hands out"
        );
        assert!(
            unclamped.usable_bytes < strix.host_ram_bytes,
            "and physical RAM does not catch it, which is why the heap clamp is not redundant"
        );
    }

    /// The platform's own budget still wins over both clamps' inputs — it is a statement, not a total.
    #[test]
    fn the_platform_budget_outranks_the_advertised_heap_derivation() {
        let mib = 1024 * 1024_u64;
        let facts = HostDeviceFacts {
            platform: SupplyPlatform::Linux,
            unified: true,
            dedicated_bytes: 4096 * mib,
            shared_pool_bytes: 120_000 * mib,
            host_ram_bytes: 124_096 * mib,
            platform_budget_bytes: Some(60_000 * mib),
            advertised_device_heap_bytes: Some(82_730 * mib),
        };
        let supply = derive_device_supply(&facts).value().copied().unwrap();
        assert_eq!(supply.usable_bytes, 60_000 * mib);
    }

    /// One physical pool is compared jointly, and the joint refusal is its own attribution.
    ///
    /// Each figure fitting on its own is exactly the state in which the machine gets over-committed:
    /// both comparisons pass and the sum does not fit.
    #[test]
    fn a_unified_pool_is_compared_jointly_and_refuses_with_its_own_attribution() {
        let r = report(BackendClass::Vulkan);
        let supply = r.usable_device_supply_bytes().unwrap();
        let host = *r.host_memory_bytes.value().unwrap();
        let pool = supply.min(host);

        // Each side alone is admissible; together they are not.
        admit_device_bytes(pool - 1024, &r, None).expect("the device side fits on its own");
        let refusal = admit_node_memory_bytes(pool - 1024, 4096, &r, None).unwrap_err();
        assert!(matches!(
            refusal,
            DeviceAdmissionRefusal::ExceedsUnifiedPool { .. }
        ));
        assert!(refusal.to_string().contains("one physical pool"));

        // Fitting jointly admits.
        admit_node_memory_bytes(pool / 2, pool / 4, &r, None).expect("a joint fit admits");

        // A separate-pool device makes no joint comparison: two pools are two budgets.
        let mut discrete = report(BackendClass::Cuda);
        discrete.memory_pool = MemoryPoolTopology::Separate;
        admit_node_memory_bytes(supply, host, &discrete, None)
            .expect("separate pools are independently available");
    }

    /// Physical RAM is the ceiling on a unified device, so a driver reporting inflated "video memory"
    /// cannot produce a supply larger than the machine.
    ///
    /// Variable-graphics-memory parts do exactly this: they present tens of gigabytes of unified RAM as
    /// dedicated video memory, which is the same RAM counted twice.
    #[test]
    fn physical_ram_caps_an_inflated_unified_derivation() {
        let inflated = HostDeviceFacts {
            platform: SupplyPlatform::Windows,
            unified: true,
            dedicated_bytes: 48 * 1024 * 1024 * 1024,
            shared_pool_bytes: 48 * 1024 * 1024 * 1024,
            host_ram_bytes: 16 * 1024 * 1024 * 1024,
            platform_budget_bytes: None,
            advertised_device_heap_bytes: None,
        };
        let supply = derive_device_supply(&inflated).value().copied().unwrap();
        assert_eq!(
            supply.usable_bytes, inflated.host_ram_bytes,
            "a unified device cannot supply more than the machine physically has"
        );
    }

    /// A platform-stated **stable** budget wins over any static derivation, and says so as itself.
    ///
    /// The slot is kept for a platform figure that is a property of the device and the OS build. No arm
    /// currently populates it — the Linux heap budget and the Windows local budget are both
    /// documented-dynamic and print beside the report as pressure — but the preference is the rule, and
    /// a report that used it must be able to say it did.
    #[test]
    fn a_platform_stated_budget_is_preferred_and_names_itself() {
        let facts = HostDeviceFacts {
            platform: SupplyPlatform::Macos,
            unified: true,
            dedicated_bytes: 8 * 1024 * 1024 * 1024,
            shared_pool_bytes: 24 * 1024 * 1024 * 1024,
            host_ram_bytes: 32 * 1024 * 1024 * 1024,
            platform_budget_bytes: Some(21 * 1024 * 1024 * 1024),
            advertised_device_heap_bytes: None,
        };
        let supply = derive_device_supply(&facts).value().copied().unwrap();
        assert_eq!(supply.usable_bytes, 21 * 1024 * 1024 * 1024);
        assert_eq!(supply.source, DeviceMemorySource::PlatformStatedBudget);
        assert_eq!(supply.platform, SupplyPlatform::Macos);
        assert!(!supply.ram_clamp_bound && !supply.device_heap_clamp_bound);
    }

    /// **The report names the derivation, not the platform.**
    ///
    /// Two boxes on different platforms reaching the figure the same way say the same thing about how
    /// they reached it, and one box whose arithmetic hit a clamp does not read as though its platform
    /// had stated the number. The source used to be chosen from the platform before the derivation
    /// branched, so a Windows report said "local budget" while deriving statically from totals — the
    /// same defect class as one lane's record carrying another lane's adapter.
    #[test]
    fn the_derivation_is_named_and_the_binding_clamp_is_recorded() {
        let gib = 1024 * 1024 * 1024_u64;
        let unified_of = |platform| HostDeviceFacts {
            platform,
            unified: true,
            dedicated_bytes: 4 * gib,
            shared_pool_bytes: 28 * gib,
            host_ram_bytes: 32 * gib,
            platform_budget_bytes: None,
            advertised_device_heap_bytes: None,
        };
        for platform in [
            SupplyPlatform::Linux,
            SupplyPlatform::Windows,
            SupplyPlatform::Macos,
        ] {
            let supply = derive_device_supply(&unified_of(platform))
                .value()
                .copied()
                .unwrap();
            assert_eq!(
                supply.source,
                DeviceMemorySource::UnifiedStaticDerivation,
                "the same arithmetic is named the same way on every platform"
            );
            assert_eq!(
                supply.platform, platform,
                "the platform is a separate member"
            );
            assert!(
                !supply.ram_clamp_bound && !supply.device_heap_clamp_bound,
                "nothing bound here: 4 + 90% of 28 is under the 32 GiB machine"
            );
        }

        // The heap clamp binding is recorded, and it supersedes the RAM clamp it clamped away.
        let heap_bound = derive_device_supply(&HostDeviceFacts {
            advertised_device_heap_bytes: Some(8 * gib),
            ..unified_of(SupplyPlatform::Linux)
        })
        .value()
        .copied()
        .unwrap();
        assert_eq!(heap_bound.usable_bytes, 8 * gib);
        assert!(heap_bound.device_heap_clamp_bound && !heap_bound.ram_clamp_bound);

        // The RAM clamp binding is recorded on its own.
        let ram_bound = derive_device_supply(&HostDeviceFacts {
            shared_pool_bytes: 64 * gib,
            ..unified_of(SupplyPlatform::Windows)
        })
        .value()
        .copied()
        .unwrap();
        assert_eq!(ram_bound.usable_bytes, 32 * gib);
        assert!(ram_bound.ram_clamp_bound && !ram_bound.device_heap_clamp_bound);
    }

    /// The macOS shape: the working set bounds the derivation as a heap, and there is no carve-out.
    ///
    /// This is the arm the addendum corrected. `recommendedMaxWorkingSetSize` is a fixed fraction of
    /// physical RAM — a property of the device and the OS build — so it belongs where the Vulkan device
    /// heap belongs, bounding the static derivation, rather than in the slot that means "the platform
    /// told us what we may use right now". The figure is the same; what it claims about itself is not.
    #[test]
    fn the_macos_working_set_bounds_the_derivation_as_a_heap() {
        let gib = 1024 * 1024 * 1024_u64;
        // An M-series shape: no dedicated carve-out, the machine's DRAM as the shared pool, and the
        // working set at two thirds of it.
        let apple_silicon = HostDeviceFacts {
            platform: SupplyPlatform::Macos,
            unified: true,
            dedicated_bytes: 0,
            shared_pool_bytes: 32 * gib,
            host_ram_bytes: 32 * gib,
            platform_budget_bytes: None,
            advertised_device_heap_bytes: Some(32 * gib * 2 / 3),
        };
        let supply = derive_device_supply(&apple_silicon)
            .value()
            .copied()
            .unwrap();
        assert_eq!(
            supply.usable_bytes,
            32 * gib * 2 / 3,
            "the working set bounds it, so the answer is the working set"
        );
        assert_eq!(supply.source, DeviceMemorySource::UnifiedStaticDerivation);
        assert!(supply.device_heap_clamp_bound, "and the clamp is recorded");
        assert!(
            supply.usable_bytes < 32 * gib * 9 / 10,
            "which is more conservative than claiming most of the machine, as dropping it would"
        );
    }

    /// A discrete card's dedicated memory IS its budget, and host RAM does not cap it — a card with
    /// more memory than the box is a real configuration.
    #[test]
    fn a_discrete_device_supplies_its_dedicated_memory_uncapped_by_host_ram() {
        let discrete = HostDeviceFacts {
            platform: SupplyPlatform::Cuda,
            unified: false,
            dedicated_bytes: 24 * 1024 * 1024 * 1024,
            shared_pool_bytes: 0,
            host_ram_bytes: 16 * 1024 * 1024 * 1024,
            platform_budget_bytes: None,
            advertised_device_heap_bytes: None,
        };
        let supply = derive_device_supply(&discrete).value().copied().unwrap();
        assert_eq!(supply.usable_bytes, 24 * 1024 * 1024 * 1024);
        assert_eq!(supply.source, DeviceMemorySource::DedicatedDeviceMemory);
    }

    /// **No trustworthy derivation fails closed**, and never asks a human for a number.
    ///
    /// A platform that supplied nothing yields a typed absence, the report refuses to answer, and the
    /// backend is not admissible there. The alternative — an operator's estimate — inverts the
    /// responsibility: the node is the thing that can measure and the operator is the thing that cannot.
    #[test]
    fn a_platform_with_no_derivation_fails_closed_rather_than_asking_anyone() {
        let nothing = HostDeviceFacts {
            platform: SupplyPlatform::Linux,
            unified: false,
            dedicated_bytes: 0,
            shared_pool_bytes: 0,
            host_ram_bytes: 0,
            platform_budget_bytes: None,
            advertised_device_heap_bytes: None,
        };
        assert!(derive_device_supply(&nothing).value().is_none());

        let mut r = report(BackendClass::Vulkan);
        r.device_supply = derive_device_supply(&nothing);
        let err = r
            .usable_device_supply_bytes()
            .expect_err("an underivable platform must not admit");
        assert!(matches!(err, CapabilityError::Unmeasured { .. }));

        let refusal = admit_device_bytes(1024, &r, None).unwrap_err();
        assert!(matches!(
            refusal,
            DeviceAdmissionRefusal::NoTrustworthySupply { .. }
        ));
    }

    /// An owner cap only tightens; it can never be the source of the mandatory hardware figure.
    ///
    /// A cap above the supply is not an error and not a correction — an owner may permit more than this
    /// particular device happens to have, and the supply comparison still refuses on its own terms.
    #[test]
    fn an_owner_cap_only_tightens_and_never_supplies_the_hardware_figure() {
        let r = report(BackendClass::Vulkan);
        let supply = r.usable_device_supply_bytes().unwrap();

        // A generous cap changes nothing: supply still binds.
        admit_device_bytes(
            supply,
            &r,
            Some(OwnerDeviceCap {
                max_bytes: u64::MAX,
            }),
        )
        .expect("a cap above supply does not raise supply");
        assert!(matches!(
            admit_device_bytes(
                supply + 1,
                &r,
                Some(OwnerDeviceCap {
                    max_bytes: u64::MAX
                })
            )
            .unwrap_err(),
            DeviceAdmissionRefusal::ExceedsSupply { .. }
        ));

        // A tightening cap binds, and says so as policy rather than as hardware.
        let refusal =
            admit_device_bytes(supply, &r, Some(OwnerDeviceCap { max_bytes: 4096 })).unwrap_err();
        assert!(matches!(
            refusal,
            DeviceAdmissionRefusal::ExceedsOwnerCap { .. }
        ));
        assert!(refusal.to_string().contains("policy refusal"));
    }
    #[test]
    fn every_supply_source_is_a_host_derivation_and_none_admits_a_human_figure() {
        // Every variant names a measurement this node can take. There is deliberately no variant an
        // operator-supplied number could arrive through: the node is what can measure the device, and
        // asking the operator instead inverted the responsibility.
        for source in [
            DeviceMemorySource::PlatformStatedBudget,
            DeviceMemorySource::UnifiedStaticDerivation,
            DeviceMemorySource::DedicatedDeviceMemory,
        ] {
            let mut candidate = report(BackendClass::Dx12);
            candidate.device_supply = Maybe::Available(DeviceMemorySupply {
                source,
                ..derived_supply(32_952_745_984)
            });
            candidate
                .validate()
                .expect("every source is a host-derived measurement");
        }
    }

    /// The shape of the observed defect: a ceiling substituted for the budget would make the
    /// per-allocation limit equal the whole supply, which this refuses on its own terms.
    #[test]
    fn a_ceiling_above_the_whole_budget_is_refused() {
        let mut r = report(BackendClass::Dx12);
        r.device_supply = Maybe::Available(DeviceMemorySupply {
            source: DeviceMemorySource::PlatformStatedBudget,
            ..derived_supply(2047 << 20)
        });
        r.measured_max_allocation = Maybe::Available(measured_ceiling(4 << 30));
        assert!(r
            .validate()
            .unwrap_err()
            .to_string()
            .contains("exceeds the whole usable device supply"));
    }

    /// A probe that cannot measure fails loud. Zero is not a measurement.
    #[test]
    fn a_zero_reading_is_refused_and_absence_is_typed() {
        let mut r = report(BackendClass::Vulkan);
        r.device_supply = Maybe::Available(derived_supply(0));
        assert!(matches!(
            r.validate().unwrap_err(),
            CapabilityError::ZeroInsteadOfUnavailable { .. }
        ));

        // The M4 box's older worker reported zero free disk on a filesystem with hundreds of
        // gigabytes free. Zero is refused; "the probe failed" is expressible and is the honest form.
        let mut r = report(BackendClass::Metal);
        r.disk_bytes = Maybe::Available(0);
        assert!(matches!(
            r.validate().unwrap_err(),
            CapabilityError::ZeroInsteadOfUnavailable {
                quantity: "free disk"
            }
        ));

        let mut r = report(BackendClass::Metal);
        r.disk_bytes = Maybe::Unavailable(Unavailable::ProbeFailed);
        r.validate().expect("a typed unavailability is legitimate");
    }

    #[test]
    fn an_unmeasured_ceiling_cannot_validate_a_composed_claim() {
        let mut r = report(BackendClass::Vulkan);
        r.measured_max_allocation = Maybe::default();
        r.validate().expect("absence is legitimate in the report");
        assert!(matches!(
            r.max_allocation_bytes().unwrap_err(),
            CapabilityError::Unmeasured { .. }
        ));
        assert_eq!(
            report(BackendClass::Vulkan).max_allocation_bytes().unwrap(),
            4 << 30
        );
    }

    #[test]
    fn a_report_must_name_its_adapter() {
        let mut r = report(BackendClass::Vulkan);
        r.adapter_name = "  ".into();
        assert!(r
            .validate()
            .unwrap_err()
            .to_string()
            .contains("name the adapter"));
    }
}
