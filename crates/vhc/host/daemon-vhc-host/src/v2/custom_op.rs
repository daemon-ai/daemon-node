// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The host-side custom-op registry (Phase C; architecture §3.2, refactor §7).
//!
//! Fused kernels (flash-attention variants, fused optimizer steps) cannot cross the wasm boundary
//! as code, so they are registered **host-side under versioned names**. This registry is the
//! **named-op admission/dispatch layer**: it advertises which versioned custom ops the host
//! implements (seeded from the shared ABI vocabulary [`daemon_vhc_abi::HOST_CUSTOM_OPS`],
//! `flash_attn@1` the first entry), and admission of a module whose `manifest.custom_ops`
//! (ABI §2.3) names an op the host lacks fails **cleanly** with a typed
//! [`AbiRefusalCode::CustomOpUnsupported`] refusal — never a trap (architecture §3.2 "admission
//! fails cleanly on hosts that lack them"; §5.2).
//!
//! ## The RESERVED `compute@2` seam (coordination with C1)
//!
//! This layer fills the `OperationIr::Custom` variant that the `compute@2` IR wire leaves RESERVED
//! (ABI §15). The two halves are coordinated **by vocabulary, not shared code**: C1 owns the
//! `burn_ir::OperationIr` wire and **refuses** the `Custom` IR variant until it is specified; this
//! registry owns the **named** admission (does the host advertise `flash_attn@1`?) and is where a
//! future `compute@2` `Custom{name}` op resolves its handler by name. Keeping the two apart lets
//! the Phase-C track land serially: the named-op vocabulary here, the IR wire there.

use daemon_vhc_abi::{AbiRefusal, AbiRefusalCode, HOST_CUSTOM_OPS};

/// A host-side registry of the versioned named custom ops (fused kernels) this host can serve.
///
/// The default registry is seeded from the shared ABI vocabulary ([`HOST_CUSTOM_OPS`]) so the
/// host's *advertised* set and the contract crate stay in lockstep; [`Self::register`] lets tests
/// (and future host builds) add a fusion the host implements.
#[derive(Debug, Clone)]
pub struct CustomOpRegistry {
    ops: std::collections::BTreeSet<String>,
}

impl Default for CustomOpRegistry {
    fn default() -> Self {
        Self {
            ops: HOST_CUSTOM_OPS.iter().map(|s| (*s).to_string()).collect(),
        }
    }
}

impl CustomOpRegistry {
    /// The default host registry (seeded from [`HOST_CUSTOM_OPS`] — `flash_attn@1` today).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the versioned op `name` is registered (advertised and dispatchable).
    #[must_use]
    pub fn supports(&self, name: &str) -> bool {
        self.ops.contains(name)
    }

    /// The advertised versioned names, sorted (the `advertised` set of `required ⊆ granted ⊆
    /// advertised`, architecture §3.2).
    #[must_use]
    pub fn advertised(&self) -> Vec<&str> {
        self.ops.iter().map(String::as_str).collect()
    }

    /// Register a versioned custom op this host implements (additive; permanent, version-suffixed).
    pub fn register(&mut self, name: impl Into<String>) {
        self.ops.insert(name.into());
    }

    /// Admit a module's required custom ops: every entry MUST be advertised, else a typed
    /// [`AbiRefusalCode::CustomOpUnsupported`] naming the offending op and the advertised set
    /// (§1.5 observed-vs-supported discipline). This is the named-op admission the funnel runs at
    /// stage 4 (ABI §9.4 step 6, alongside the world/channel checks).
    ///
    /// # Errors
    /// [`AbiRefusalCode::CustomOpUnsupported`] for the first required op absent from the registry.
    pub fn admit(&self, required: &[String]) -> Result<(), AbiRefusal> {
        for name in required {
            if !self.supports(name) {
                return Err(AbiRefusal::new(
                    AbiRefusalCode::CustomOpUnsupported,
                    format!(
                        "module requires custom op `{name}`, absent from the host custom-op \
                         registry {:?}",
                        self.advertised()
                    ),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_advertises_flash_attn_v1() {
        let reg = CustomOpRegistry::new();
        assert!(reg.supports(daemon_vhc_abi::CUSTOM_OP_FLASH_ATTN_V1));
        assert!(reg.supports("flash_attn@1"));
        // Versioning is part of the name — a different version is a distinct, unregistered op.
        assert!(!reg.supports("flash_attn@2"));
        assert!(!reg.supports("flash_attn"));
    }

    #[test]
    fn admit_passes_empty_and_registered_refuses_absent() {
        let reg = CustomOpRegistry::new();
        assert!(reg.admit(&[]).is_ok(), "no custom ops required → admitted");
        assert!(reg.admit(&["flash_attn@1".to_string()]).is_ok());
        let err = reg.admit(&["fused_moe@1".to_string()]).unwrap_err();
        assert_eq!(err.code, AbiRefusalCode::CustomOpUnsupported);
        assert!(err.detail.contains("fused_moe@1"));
    }

    #[test]
    fn register_adds_a_fusion() {
        let mut reg = CustomOpRegistry::new();
        assert!(!reg.supports("fused_adamw@1"));
        reg.register("fused_adamw@1");
        assert!(reg.supports("fused_adamw@1"));
        assert!(reg.admit(&["fused_adamw@1".to_string()]).is_ok());
    }
}
