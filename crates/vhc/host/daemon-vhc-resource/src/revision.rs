// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **backend implementation revision record** — the machine-readable statement a host makes
//! about which backend implementation it is actually running
//! (`docs/specs/vhc-architecture-spec.md` §9.7 `[PC-10]`(1), §9.6 `[RC-4]` revision binding).
//!
//! The revision this replaces was *reported in prose*: a `Debug`-formatted adapter line in the
//! log, carrying real information that no code path could act on. Nothing parsed it, nothing
//! carried it into the admitted tuple, and nothing could compare it to the revision range a
//! Backend Execution Profile names — so `[RC-4]`'s revision binding was unenforceable and
//! `[PC-10]`(1)'s "implementation identity is reported correctly" was untestable, because there
//! was no report, only a print. This record is the structure that used to be a print.
//!
//! ## Unavailability is typed, never zero and never empty
//!
//! Platforms differ in what they will tell you, and the difference has to survive into the record.
//! The compute framework supplies **no driver revision at all** on Metal; as an empty string that
//! is indistinguishable from a driver whose revision *is* the empty string, and a profile range
//! check cannot tell "absent" from "matches nothing". Hence [`Maybe`]: an unavailable value carries
//! the *reason* it is unavailable, and the OS build — which is required, not optional — is the only
//! implementation-revision signal that exists on that backend.
//!
//! The same discipline applies to a probe that fails: a probe MUST NOT return zero. A zero
//! resource reading is an admission refusal wearing a measurement's clothes, and it refuses the
//! machine rather than reporting the defect.

use serde::{Deserialize, Serialize};

/// Why a platform did not supply a value. Recording the reason is what makes a later divergence
/// attributable to the probe rather than to the profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unavailable {
    /// The operating system or vendor stack does not expose it.
    NotExposedByPlatform,
    /// The compute framework does not surface what the platform exposes.
    NotExposedByFramework,
    /// The probe ran and failed. Distinct from "not exposed": this is a defect to fix, not a
    /// property of the platform.
    ProbeFailed,
    /// Reading it needs a privilege this process does not hold.
    RequiresPrivilege,
}

/// A value a platform may not supply.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Maybe<T> {
    /// The platform supplied it.
    Available(T),
    /// It is absent, with the reason.
    Unavailable(Unavailable),
}

impl<T> Default for Maybe<T> {
    /// An absent value defaults to "the platform does not expose it" — never to a zero or an empty
    /// string, which is the whole reason this type exists.
    fn default() -> Self {
        Self::Unavailable(Unavailable::NotExposedByPlatform)
    }
}

impl<T> Maybe<T> {
    /// The value, if the platform supplied one.
    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Available(v) => Some(v),
            Self::Unavailable(_) => None,
        }
    }

    /// Whether the platform supplied a value.
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available(_))
    }
}

/// The closed backend-class set the admitted tuple already uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendClass {
    /// Vulkan.
    Vulkan,
    /// Metal.
    Metal,
    /// Direct3D 12.
    Dx12,
    /// CUDA.
    Cuda,
    /// A CPU backend.
    Cpu,
}

impl BackendClass {
    /// The stable slug, matching the envelope's allowed-backend-class spelling.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
            Self::Dx12 => "dx12",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }

    /// Whether a role admitted for this class runs on an accelerator.
    pub fn is_device(self) -> bool {
        !matches!(self, Self::Cpu)
    }
}

/// The adapter's reported device class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterDeviceType {
    /// A discrete accelerator.
    DiscreteGpu,
    /// An integrated accelerator sharing host memory.
    IntegratedGpu,
    /// A virtualized accelerator.
    VirtualGpu,
    /// A CPU adapter.
    Cpu,
    /// Anything else the platform reports.
    Other,
}

/// Stable hardware identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterIdentity {
    /// PCI vendor id.
    pub vendor_id: Maybe<u32>,
    /// PCI device id.
    pub device_id: Maybe<u32>,
    /// PCI bus address.
    pub pci_bus_id: Maybe<String>,
    /// Adapter UUID.
    pub uuid: Maybe<String>,
}

/// The adapter this record describes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Adapter {
    /// The adapter's reported name.
    pub name: String,
    /// Its reported device class.
    pub device_type: AdapterDeviceType,
    /// Whether the platform flags this adapter as a software rasterizer.
    ///
    /// Required, not optional, and it is a refusal rather than a note: a device-lane role admitted
    /// against a software rasterizer that reports a device backend class is precisely the silent
    /// CPU fallback `[PC-6]` forbids. A loader that enumerates a software adapter beside a real one
    /// makes this reachable without any operator intending it.
    pub is_software_rasterizer: bool,
    /// Stable hardware identity.
    pub identity: AdapterIdentity,
}

/// The guest-facing compute-framework surface and its runtime.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeFramework {
    /// Framework name.
    pub name: String,
    /// Framework revision.
    pub revision: String,
    /// Runtime name beneath it.
    pub runtime_name: String,
    /// Runtime revision.
    pub runtime_revision: String,
}

/// Which graphics API the implementation resolved to, and whether an operator chose it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiSelectionSource {
    /// The platform's own default.
    PlatformDefault,
    /// An operator override. It changes the implementation actually in use, so it changes the
    /// record — and therefore which profiles are compatible.
    OperatorOverride,
}

/// The host-side execution of the framework API for a device class. Changing this never re-pins a
/// guest and always requires a compatible certified profile.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendImplementation {
    /// Implementation name.
    pub name: String,
    /// Implementation revision.
    pub revision: String,
    /// The resolved graphics API, where the implementation selects one.
    pub graphics_api_selected: Maybe<String>,
    /// Whether the resolution was the platform default or an operator override.
    pub graphics_api_selection_source: ApiSelectionSource,
}

/// Pooling, retention and release behavior beneath the backend implementation. Separately
/// versioned because a profile's pooling and retention terms are calibrated against exactly this
/// and nothing else.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocatorImplementation {
    /// Allocator name.
    pub name: String,
    /// Allocator revision.
    pub revision: String,
    /// The allocation mode in force, where the allocator exposes one.
    pub allocation_mode: Maybe<String>,
    /// Whether this build can report allocator statistics.
    ///
    /// A profile whose pooling terms were calibrated from allocator statistics MUST NOT be accepted
    /// by a binary that cannot produce them: its conformance evidence is unreproducible on that
    /// binary, which makes the profile's own certification unverifiable there.
    pub statistics_available: bool,
}

/// The operating system, required rather than optional.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatingSystem {
    /// OS family.
    pub family: OsFamily,
    /// OS version.
    pub version: String,
    /// OS build. On a backend whose framework supplies no driver revision, this is the
    /// implementation-revision signal a profile range has to constrain instead.
    pub build: Maybe<String>,
    /// Kernel revision, where the platform has a separable one.
    pub kernel: Maybe<String>,
}

/// The OS families the host supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OsFamily {
    /// Linux.
    Linux,
    /// macOS.
    Macos,
    /// Windows.
    Windows,
}

/// A driver's several revision numberings. They do not order against each other, so all are
/// recorded and a profile range names **which** numbering it constrains.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DriverRevision {
    /// Driver name.
    pub name: Maybe<String>,
    /// Free-form driver info.
    pub info: Maybe<String>,
    /// The driver revision as text.
    pub version_text: Maybe<String>,
    /// The driver revision as the platform's raw integer.
    pub version_raw: Maybe<u32>,
    /// The vendor's own release numbering, where it differs from the OS/display-driver numbering.
    /// The same driver can carry two unrelated numbers, and a range that names one says nothing
    /// about the other.
    pub vendor_release: Maybe<String>,
    /// Video BIOS revision.
    pub vbios: Maybe<String>,
}

/// The vendor stack the implementation calls.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DriverApi {
    /// The platform API in use.
    pub api: PlatformApi,
    /// Its version.
    pub api_version: Maybe<String>,
    /// The driver's revision numberings.
    pub driver: DriverRevision,
    /// The kernel driver, where the platform has a separable one.
    pub kernel_driver: Maybe<String>,
    /// The operating system. Required.
    pub os: OperatingSystem,
}

impl DriverApi {
    /// The **implementation-revision signal** a profile range must constrain on this platform.
    ///
    /// A driver revision where one exists; otherwise the OS build, which is the documented
    /// fallback for a backend whose framework supplies no driver revision at all. If neither
    /// exists the platform has told us nothing comparable, and that is reported as such rather
    /// than papered over with an empty string — a profile cannot be range-checked against a value
    /// nobody supplied, and pretending otherwise would admit any profile at all.
    pub fn revision_signal(&self) -> RevisionSignal {
        if let Some(text) = self.driver.version_text.value() {
            return RevisionSignal::DriverVersion(text.clone());
        }
        if let Some(release) = self.driver.vendor_release.value() {
            return RevisionSignal::VendorRelease(release.clone());
        }
        if let Some(build) = self.os.build.value() {
            return RevisionSignal::OsBuild {
                family: self.os.family,
                version: self.os.version.clone(),
                build: build.clone(),
            };
        }
        RevisionSignal::None
    }
}

/// Which numbering a revision comparison is actually keyed on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RevisionSignal {
    /// The driver's own version text.
    DriverVersion(String),
    /// The vendor's release numbering, where the driver version text is absent.
    VendorRelease(String),
    /// The OS build, the documented fallback where the framework supplies no driver revision.
    OsBuild {
        /// OS family.
        family: OsFamily,
        /// OS version.
        version: String,
        /// OS build.
        build: String,
    },
    /// Nothing comparable was supplied. A profile range cannot be evaluated against this, and the
    /// correct response is a typed refusal rather than an assumed match.
    None,
}

/// The platform APIs a backend implementation may call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlatformApi {
    /// Vulkan.
    Vulkan,
    /// Metal.
    Metal,
    /// Direct3D 12.
    D3d12,
    /// CUDA.
    Cuda,
    /// No device API (a CPU backend).
    None,
}

/// The binary that produced a record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedBinaryIdentity {
    /// The binary's blake3.
    pub blake3: [u8; 32],
    /// Its size in bytes.
    pub size_bytes: u64,
}

/// How a record came to exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProducedBy {
    /// The running binary's own probe path. **The only value admissible as admission evidence.**
    WorkerProbePath,
    /// Extracted externally, for a calibration campaign or a conformance cross-check. A record the
    /// running binary did not produce about itself proves nothing about what it is running, so it
    /// MUST NOT be accepted at admission.
    ExternalExtraction,
}

/// The machine-readable statement a host makes about which backend implementation it is running.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendImplementationRevision {
    /// The backend class the role would execute on.
    pub backend_class: BackendClass,
    /// The adapter.
    pub adapter: Adapter,
    /// The guest-facing compute framework.
    pub compute_framework: ComputeFramework,
    /// The host-side implementation of that framework for this device class.
    pub backend_implementation: BackendImplementation,
    /// The allocator beneath it.
    pub allocator: AllocatorImplementation,
    /// The vendor stack it calls.
    pub driver_api: DriverApi,
    /// The binary that produced this record.
    pub sealed_binary: SealedBinaryIdentity,
    /// How the record came to exist.
    pub produced_by: ProducedBy,
}

impl BackendImplementationRevision {
    /// Whether this record may be used as admission evidence.
    ///
    /// Two independent refusals live here, and both are refusals rather than warnings. A record the
    /// running binary did not produce about itself proves nothing about what it is running. And a
    /// software rasterizer presenting a device backend class is the silent CPU fallback `[PC-6]`
    /// forbids — the loader enumerating it is enough to make that reachable, so the check cannot be
    /// left to operator discipline.
    pub fn admissible(&self) -> Result<(), RevisionRefusal> {
        if self.produced_by != ProducedBy::WorkerProbePath {
            return Err(RevisionRefusal::NotSelfProduced);
        }
        if self.adapter.is_software_rasterizer && self.backend_class.is_device() {
            return Err(RevisionRefusal::SoftwareRasterizerOnDeviceLane {
                adapter: self.adapter.name.clone(),
                class: self.backend_class,
            });
        }
        Ok(())
    }
}

/// Why a revision record cannot be used at admission.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RevisionRefusal {
    /// The record was not produced by the running binary's own probe path.
    #[error(
        "backend implementation revision was not produced by the running binary's own probe path; \
         a record the binary did not make about itself proves nothing about what it is running"
    )]
    NotSelfProduced,
    /// A software rasterizer was resolved for a device lane.
    #[error(
        "adapter `{adapter}` is a software rasterizer but reports backend class `{}`; a device \
         lane must fail typed rather than execute on a CPU rasterizer",
        class.slug()
    )]
    SoftwareRasterizerOnDeviceLane {
        /// The adapter's name.
        adapter: String,
        /// The class it reported.
        class: BackendClass,
    },
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    fn os(family: OsFamily, build: Maybe<String>) -> OperatingSystem {
        OperatingSystem {
            family,
            version: "1.0".into(),
            build,
            kernel: Maybe::Unavailable(Unavailable::NotExposedByPlatform),
        }
    }

    /// A complete, admissible revision record for one backend class.
    pub(crate) fn revision(class: BackendClass) -> BackendImplementationRevision {
        BackendImplementationRevision {
            backend_class: class,
            adapter: Adapter {
                name: "test adapter".into(),
                device_type: AdapterDeviceType::IntegratedGpu,
                is_software_rasterizer: false,
                identity: AdapterIdentity {
                    vendor_id: Maybe::Available(4098),
                    device_id: Maybe::Available(5510),
                    pci_bus_id: Maybe::Available("0000:c4:00.0".into()),
                    uuid: Maybe::Unavailable(Unavailable::NotExposedByPlatform),
                },
            },
            compute_framework: ComputeFramework {
                name: "framework".into(),
                revision: "0.21.0".into(),
                runtime_name: "runtime".into(),
                runtime_revision: "0.10.0".into(),
            },
            backend_implementation: BackendImplementation {
                name: "implementation".into(),
                revision: "0.10.0".into(),
                graphics_api_selected: Maybe::Available(class.slug().into()),
                graphics_api_selection_source: ApiSelectionSource::PlatformDefault,
            },
            allocator: AllocatorImplementation {
                name: "allocator".into(),
                revision: "0.10.0".into(),
                allocation_mode: Maybe::Available("Auto".into()),
                statistics_available: true,
            },
            driver_api: DriverApi {
                api: PlatformApi::Vulkan,
                api_version: Maybe::Available("1.3".into()),
                driver: DriverRevision {
                    version_text: Maybe::Available("25.1.0".into()),
                    ..Default::default()
                },
                kernel_driver: Maybe::Available("amdgpu".into()),
                os: os(OsFamily::Linux, Maybe::Available("6.19.7".into())),
            },
            sealed_binary: SealedBinaryIdentity {
                blake3: [1u8; 32],
                size_bytes: 44_323_328,
            },
            produced_by: ProducedBy::WorkerProbePath,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::revision;
    use super::*;

    #[test]
    fn a_driver_version_is_the_revision_signal_when_one_exists() {
        let rev = revision(BackendClass::Vulkan);
        assert_eq!(
            rev.driver_api.revision_signal(),
            RevisionSignal::DriverVersion("25.1.0".into())
        );
    }

    /// A backend whose framework supplies no driver revision falls back to the OS build — the only
    /// implementation-revision signal that exists there. Absence is typed, so it is
    /// distinguishable from a driver whose revision genuinely is the empty string.
    #[test]
    fn a_backend_with_no_driver_revision_falls_back_to_the_os_build() {
        let mut rev = revision(BackendClass::Metal);
        rev.driver_api.api = PlatformApi::Metal;
        rev.driver_api.driver = DriverRevision::default();
        assert!(!rev.driver_api.driver.version_text.is_available());
        assert_eq!(
            rev.driver_api.revision_signal(),
            RevisionSignal::OsBuild {
                family: OsFamily::Linux,
                version: "1.0".into(),
                build: "6.19.7".into(),
            }
        );
    }

    /// With neither a driver revision nor an OS build there is nothing comparable, and that is
    /// reported rather than assumed to match. A range evaluated against nothing would admit
    /// everything.
    #[test]
    fn no_comparable_signal_is_reported_rather_than_assumed() {
        let mut rev = revision(BackendClass::Metal);
        rev.driver_api.driver = DriverRevision::default();
        rev.driver_api.os.build = Maybe::Unavailable(Unavailable::NotExposedByPlatform);
        assert_eq!(rev.driver_api.revision_signal(), RevisionSignal::None);
    }

    #[test]
    fn only_a_self_produced_record_is_admission_evidence() {
        let mut rev = revision(BackendClass::Vulkan);
        rev.admissible()
            .expect("self-produced record is admissible");
        rev.produced_by = ProducedBy::ExternalExtraction;
        assert_eq!(rev.admissible(), Err(RevisionRefusal::NotSelfProduced));
    }

    /// The loader on at least one fleet box enumerates a software rasterizer beside the real
    /// adapter, so this is reachable without anyone intending it.
    #[test]
    fn a_software_rasterizer_cannot_serve_a_device_lane() {
        let mut rev = revision(BackendClass::Vulkan);
        rev.adapter.is_software_rasterizer = true;
        assert!(matches!(
            rev.admissible(),
            Err(RevisionRefusal::SoftwareRasterizerOnDeviceLane { .. })
        ));

        // On an explicit CPU lane a software adapter is what was asked for.
        let mut cpu = revision(BackendClass::Cpu);
        cpu.adapter.is_software_rasterizer = true;
        cpu.admissible().expect("a cpu lane may use a cpu adapter");
    }

    #[test]
    fn unavailability_carries_its_reason() {
        let probe_failed: Maybe<u32> = Maybe::Unavailable(Unavailable::ProbeFailed);
        assert!(!probe_failed.is_available());
        assert_eq!(probe_failed.value(), None);
        // A failed probe is a defect to fix, not a property of the platform, and the two are
        // distinguishable.
        assert_ne!(
            probe_failed,
            Maybe::Unavailable(Unavailable::NotExposedByPlatform)
        );
    }
}
