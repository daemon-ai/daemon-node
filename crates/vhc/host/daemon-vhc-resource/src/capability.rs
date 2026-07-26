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

use crate::revision::{BackendClass, Maybe};

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
    /// The owner's explicit allocation for this run. Authoritative when present: the owner may
    /// always allocate less than the device physically has.
    OwnerAllocatedBudget,
    /// A platform query for the device's own memory budget.
    PlatformDeviceMemoryQuery,
}

/// The owner-allocated stable device-memory budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMemoryBudget {
    /// The stable budget in bytes.
    pub stable_bytes: u64,
    /// Which authority produced it.
    pub source: DeviceMemorySource,
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
    /// The stable device-memory budget.
    pub device_memory: DeviceMemoryBudget,
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
        if self.device_memory.stable_bytes == 0 {
            return Err(CapabilityError::ZeroInsteadOfUnavailable {
                quantity: "the device-memory budget",
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
        if let Some(ceiling) = self.measured_max_allocation_bytes.value() {
            if *ceiling > self.device_memory.stable_bytes {
                return Err(CapabilityError::Invalid(format!(
                    "the measured per-allocation ceiling ({ceiling} bytes) exceeds the whole \
                     device-memory budget ({} bytes), so it is not a ceiling this device can honour",
                    self.device_memory.stable_bytes
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
            device_memory: DeviceMemoryBudget {
                stable_bytes: 32_952_745_984,
                source: DeviceMemorySource::OwnerAllocatedBudget,
            },
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

    /// The budget is the owner's stable allocation, and its source is named. There is no variant
    /// for falling back to the per-buffer ceiling, so the inventory path that reported a
    /// two-gigabyte supply on a thirty-gigabyte card cannot be expressed here at all.
    #[test]
    fn the_budget_names_its_source_and_no_ceiling_fallback_is_representable() {
        let r = report(BackendClass::Dx12);
        assert_eq!(
            r.device_memory.source,
            DeviceMemorySource::OwnerAllocatedBudget
        );
        // The two admissible sources are both device-memory authorities. A per-buffer ceiling is
        // not among them, and the enum is closed.
        for source in [
            DeviceMemorySource::OwnerAllocatedBudget,
            DeviceMemorySource::PlatformDeviceMemoryQuery,
        ] {
            let mut candidate = report(BackendClass::Dx12);
            candidate.device_memory.source = source;
            candidate
                .validate()
                .expect("both sources are device-memory authorities");
        }
    }

    /// The shape of the observed defect: a ceiling substituted for the budget would make the
    /// per-allocation limit equal the whole supply, which this refuses on its own terms.
    #[test]
    fn a_ceiling_above_the_whole_budget_is_refused() {
        let mut r = report(BackendClass::Dx12);
        r.device_memory.stable_bytes = 2047 << 20;
        r.measured_max_allocation_bytes = Maybe::Available(4 << 30);
        assert!(r
            .validate()
            .unwrap_err()
            .to_string()
            .contains("exceeds the whole device-memory budget"));
    }

    /// A probe that cannot measure fails loud. Zero is not a measurement.
    #[test]
    fn a_zero_reading_is_refused_and_absence_is_typed() {
        let mut r = report(BackendClass::Vulkan);
        r.device_memory.stable_bytes = 0;
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
