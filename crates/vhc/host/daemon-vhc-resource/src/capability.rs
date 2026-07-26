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
    /// The Windows local-budget query for this adapter.
    DxgiLocalBudget,
    /// The Metal recommended working-set size.
    MetalRecommendedWorkingSet,
    /// A conservative derivation for a Linux unified-memory device, from the platform's own
    /// memory-budget facts.
    ///
    /// Named separately because the obvious reading is wrong on this class of device: the driver's
    /// carve-out on a unified-memory part is not a statement of what the device may use, and reading
    /// it as one under-reports the supply by most of the machine's memory.
    LinuxUnifiedMemoryBudget,
    /// The CUDA driver's device-total query.
    CudaDeviceTotal,
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
    /// Which host derivation produced it. No variant admits a human-supplied figure.
    pub source: DeviceMemorySource,
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
    pub platform_budget_bytes: Option<u64>,
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
///    shared pool, clamped by physical RAM. The clamp matters — a variable-graphics-memory part can
///    report tens of gigabytes of "video memory" that is the same RAM counted twice.
/// 3. **The dedicated figure** for a discrete device, which for a discrete card *is* the budget.
/// 4. **Nothing**, typed, which fails closed.
#[must_use]
pub fn derive_device_supply(facts: &HostDeviceFacts) -> Maybe<DeviceMemorySupply> {
    let source = match facts.platform {
        SupplyPlatform::Windows => DeviceMemorySource::DxgiLocalBudget,
        SupplyPlatform::Macos => DeviceMemorySource::MetalRecommendedWorkingSet,
        SupplyPlatform::Linux => DeviceMemorySource::LinuxUnifiedMemoryBudget,
        SupplyPlatform::Cuda => DeviceMemorySource::CudaDeviceTotal,
    };
    let ram_clamp = |bytes: u64| {
        // Only a unified device competes with the host for one pool; clamping a discrete card by host
        // RAM would under-report a card with more memory than the box.
        if facts.unified && facts.host_ram_bytes > 0 {
            bytes.min(facts.host_ram_bytes)
        } else {
            bytes
        }
    };

    if let Some(budget) = facts.platform_budget_bytes.filter(|b| *b > 0) {
        return Maybe::Available(DeviceMemorySupply {
            usable_bytes: ram_clamp(budget),
            source,
        });
    }

    if facts.unified {
        let claimable = facts
            .shared_pool_bytes
            .saturating_mul(SHARED_POOL_CLAIMABLE_PERCENT)
            / 100;
        let derived = ram_clamp(facts.dedicated_bytes.saturating_add(claimable));
        if derived > 0 {
            return Maybe::Available(DeviceMemorySupply {
                usable_bytes: derived,
                source,
            });
        }
        return Maybe::Unavailable(Unavailable::ProbeFailed);
    }

    if facts.dedicated_bytes > 0 {
        return Maybe::Available(DeviceMemorySupply {
            usable_bytes: facts.dedicated_bytes,
            source,
        });
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
    /// The largest single physical allocation the driver was **measured** to accept. Typed
    /// unavailability when it has not been probed — the reported ceiling is not evidence of what
    /// the driver enforces, and the two are carried separately in the profile for the same reason.
    pub measured_max_allocation_bytes: Maybe<u64>,
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
            .measured_max_allocation_bytes
            .value()
            .is_some_and(|v| *v == 0)
        {
            return Err(CapabilityError::ZeroInsteadOfUnavailable {
                quantity: "the measured per-allocation ceiling",
            });
        }
        // A per-allocation ceiling above the whole budget is not a ceiling; it is the compile-time
        // constant one platform reports in place of one.
        if let (Some(ceiling), Some(supply)) = (
            self.measured_max_allocation_bytes.value(),
            self.device_supply.value(),
        ) {
            if *ceiling > supply.usable_bytes {
                return Err(CapabilityError::Invalid(format!(
                    "the measured per-allocation ceiling ({ceiling} bytes) exceeds the whole \
                     usable device supply ({} bytes), so it is not a ceiling this device can honour",
                    supply.usable_bytes
                )));
            }
        }
        Ok(())
    }

    /// The per-allocation ceiling admission may compare a composed claim against.
    ///
    /// Measured or nothing. The reported figure is not evidence of what the driver enforces, and
    /// substituting it here would re-create the inventory fallback this artifact exists to avoid.
    pub fn max_allocation_bytes(&self) -> Result<u64, CapabilityError> {
        self.measured_max_allocation_bytes
            .value()
            .copied()
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
}

impl DeviceAdmissionRefusal {
    /// The limit that bound, for a caller reporting `required` against `available`.
    #[must_use]
    pub fn binding_limit_bytes(&self) -> u64 {
        match self {
            Self::NoTrustworthySupply { .. } => 0,
            Self::ExceedsSupply { supply_bytes, .. } => *supply_bytes,
            Self::ExceedsOwnerCap { cap_bytes, .. } => *cap_bytes,
        }
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    /// A complete, valid report.
    pub(crate) fn report(class: BackendClass) -> DeviceCapabilityReport {
        DeviceCapabilityReport {
            schema: DEVICE_CAPABILITY_REPORT_SCHEMA,
            backend_class: class,
            adapter_name: "Radeon 8060S Graphics (RADV GFX1151)".into(),
            device_supply: Maybe::Available(DeviceMemorySupply {
                usable_bytes: 32_952_745_984,
                source: DeviceMemorySource::LinuxUnifiedMemoryBudget,
            }),
            measured_max_allocation_bytes: Maybe::Available(4 << 30),
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
    use super::fixtures::report;
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
        };
        let supply = derive_device_supply(&strix)
            .value()
            .copied()
            .expect("a unified device with a shared pool derives supply");

        assert_eq!(supply.source, DeviceMemorySource::LinuxUnifiedMemoryBudget);
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
        };
        let supply = derive_device_supply(&inflated).value().copied().unwrap();
        assert_eq!(
            supply.usable_bytes, inflated.host_ram_bytes,
            "a unified device cannot supply more than the machine physically has"
        );
    }

    /// The platform's own budget query wins over any static derivation.
    ///
    /// The platform stating what this process may use beats us inferring it from totals — that is what
    /// makes the Windows and macOS paths authoritative rather than approximate.
    #[test]
    fn a_platform_budget_query_is_preferred_over_the_static_derivation() {
        let facts = HostDeviceFacts {
            platform: SupplyPlatform::Macos,
            unified: true,
            dedicated_bytes: 8 * 1024 * 1024 * 1024,
            shared_pool_bytes: 24 * 1024 * 1024 * 1024,
            host_ram_bytes: 32 * 1024 * 1024 * 1024,
            platform_budget_bytes: Some(21 * 1024 * 1024 * 1024),
        };
        let supply = derive_device_supply(&facts).value().copied().unwrap();
        assert_eq!(supply.usable_bytes, 21 * 1024 * 1024 * 1024);
        assert_eq!(
            supply.source,
            DeviceMemorySource::MetalRecommendedWorkingSet
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
        };
        let supply = derive_device_supply(&discrete).value().copied().unwrap();
        assert_eq!(supply.usable_bytes, 24 * 1024 * 1024 * 1024);
        assert_eq!(supply.source, DeviceMemorySource::CudaDeviceTotal);
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
            DeviceMemorySource::DxgiLocalBudget,
            DeviceMemorySource::MetalRecommendedWorkingSet,
            DeviceMemorySource::LinuxUnifiedMemoryBudget,
            DeviceMemorySource::CudaDeviceTotal,
        ] {
            let mut candidate = report(BackendClass::Dx12);
            candidate.device_supply = Maybe::Available(DeviceMemorySupply {
                usable_bytes: 32_952_745_984,
                source,
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
            usable_bytes: 2047 << 20,
            source: DeviceMemorySource::DxgiLocalBudget,
        });
        r.measured_max_allocation_bytes = Maybe::Available(4 << 30);
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
        r.device_supply = Maybe::Available(DeviceMemorySupply {
            usable_bytes: 0,
            source: DeviceMemorySource::LinuxUnifiedMemoryBudget,
        });
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
        r.measured_max_allocation_bytes = Maybe::default();
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
