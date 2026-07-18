// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `[vhc]` node-config section (spec §10.6).
//!
//! [`VhcConfig`] is the typed projection of the figment `[vhc]` table the node layers
//! (defaults ← TOML ← env ← CLI). It is defined **here** (lane R) rather than in the node's main
//! config crate — that crate is outside lane R's file set, so the struct + its extraction test land
//! in `daemon-vhc-session` and the node wiring (embedding it in `NodeConfig`) is post-MVP node work.
//!
//! The struct is `serde`-only (no figment on the default participant build); the extraction test
//! exercises the figment layering as a dev-dependency, proving the `[vhc]` keys deserialize
//! additively with the spec §10.6 defaults.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::protocol::PolicyMode;

/// Operator posture for run-supplied experiment modules (spec §10.6, §12).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModuleTrust {
    /// Any author-signed module (the permissioned-org default).
    #[default]
    Signed,
    /// Only `daemon-vhc-host`'s preset experiments.
    FirstParty,
}

/// The default participation policy for newly-joined runs (`[vhc].default_policy`, §10.5).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct VhcPolicyConfig {
    /// Availability mode.
    pub mode: PolicyMode,
    /// VRAM cap in MiB (`0` = uncapped).
    pub vram_cap_mb: u32,
    /// Duty-cycle percentage (`0..=100`).
    pub duty_cycle_pct: u8,
    /// Optional cron schedule (for [`PolicyMode::Scheduled`]).
    pub schedule: Option<String>,
}

impl Default for VhcPolicyConfig {
    fn default() -> Self {
        // Spec §10.6: `default_policy = { mode = "idle", vram_cap_mb = 0, duty_cycle_pct = 100 }`.
        Self {
            mode: PolicyMode::Idle,
            vram_cap_mb: 0,
            duty_cycle_pct: 100,
            schedule: None,
        }
    }
}

/// The `vhc:*` credential a registry request carries (`[vhc.registry].auth`, §11.1). Mirrors
/// `daemon_vhc_net::ws_client::WsAuth` / `RegistryClient`'s auth modes — never hardcoded.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistryAuthConfig {
    /// No auth headers (a bare dev target).
    #[default]
    None,
    /// `Authorization: Bearer <token>` (the gateway `vhc:*` API-key path).
    Bearer {
        /// The bearer token.
        token: String,
    },
    /// The internal identity headers (the direct-to-`apps/vhc` dev path).
    Internal {
        /// `x-daemon-org-id`.
        org_id: String,
        /// `x-daemon-actor`.
        actor: String,
    },
}

/// The coordinator-registry discovery surface (`[vhc.registry]`; A3 — the A1-noted boot follow-on).
///
/// When `base` is non-empty, `bins/daemon` constructs a `RegistryClient`-backed `EgressRunDiscovery`
/// at boot, so `vhc_join` discovers the run, fetches + blake3-verifies the frozen envelope, and
/// runs the worker's real §6.5 `AssessRun` before `JoinRun`. Empty (the default) keeps
/// `discovery: None` — the probe-based fallback against the allowlist. **Deploy-swappable by
/// config only**: the same node targets wrangler-dev (`http://127.0.0.1:8795/api/v1/vhc`) or the
/// real workers.dev deployment (e.g. `https://daemon-vhc-dev.<acct>.workers.dev/api/v1/vhc`)
/// without a code change.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RegistryConfig {
    /// The registry base URL (`""` = no registry → no discovery seam).
    pub base: String,
    /// The `vhc:*` credential for registry + presign requests.
    pub auth: RegistryAuthConfig,
}

/// The iroh transport knobs (`[vhc].iroh`, §7.1). Gossip is mandatory, so unreachable relays make
/// the node vhc-ineligible (§6.5); this MVP surface carries only the relay selector.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct IrohConfig {
    /// `"default"` for the built-in relays, or explicit relay URLs.
    pub relays: String,
}

impl Default for IrohConfig {
    fn default() -> Self {
        Self {
            relays: "default".to_string(),
        }
    }
}

/// The owner's aggregate resource grants for vhc participation (`[vhc.owner_budget]`,
/// decisions D6) — the standing per-device + host-wide ledgers every admitted role-instance is
/// charged against. Maps into the node's `daemon_vhc_node::OwnerBudget`.
///
/// **Default posture: conservative and FINITE, not unbounded.** When `[vhc]` is enabled and no
/// explicit budget is configured, the node derives finite ledgers from the worker's hardware
/// probe rather than granting everything (which would let the arbiter admit without limit). A
/// field left at its zero/empty default is derived; a non-zero field overrides its ledger
/// explicitly. The derivation the node applies (`OwnerBudget::from_config`):
///
/// - **device memory** — `device_memory_mb` if set; else a single `gpu:0` ledger sized to the
///   probed dedicated VRAM (v2.0 fleets are single-accelerator-per-member); else, with no probe,
///   a conservative 4 GiB floor.
/// - **host RAM** — `host_ram_mb` if set; else the probed host RAM; else an 8 GiB floor.
/// - **disk** — `disk_mb` if set; else the `[vhc].data_cache_gb` cache bound.
/// - **uplink / downlink** — `net_{up,down}_kbps` if set; else the probed link rates; else a 1
///   Gbit/s finite ceiling.
/// - **duty** — `duty_pct` if set; else 100 (one full accelerator-duty).
/// - **instances** — `max_instances` if set; else a conservative finite default
///   (`OwnerBudget::DEFAULT_MAX_INSTANCES`).
///
/// `unbounded = true` is the explicit opt-out (grant everything — the pre-budget permissive
/// posture), the only route back to unbounded ledgers; kept for single-tenant boxes and tests.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct OwnerBudgetConfig {
    /// Explicit opt-out: grant everything (the pre-budget permissive posture). Overrides all other
    /// fields.
    pub unbounded: bool,
    /// Per-accelerator VRAM ledger in MiB, keyed by device id (e.g. `"gpu:0"`). Empty → derive
    /// from the probed dedicated VRAM.
    pub device_memory_mb: BTreeMap<String, u64>,
    /// Host-RAM ledger in MiB. `0` → derive from the probed host RAM.
    pub host_ram_mb: u64,
    /// Disk/cache ledger in MiB. `0` → derive from `[vhc].data_cache_gb`.
    pub disk_mb: u64,
    /// Uplink ledger in kbit/s. `0` → derive from the probed uplink.
    pub net_up_kbps: u64,
    /// Downlink ledger in kbit/s. `0` → derive from the probed downlink.
    pub net_down_kbps: u64,
    /// Duty-cycle ledger in percent (100 = one full accelerator-duty). `0` → 100.
    pub duty_pct: u32,
    /// Max concurrently-admitted role-instances. `0` → a conservative finite default.
    pub max_instances: u32,
}

/// Live module-upgrade bounds (`[vhc.upgrade]`, ABI §4.4/§9.6/§10.3) — the node clamps every
/// `SwitchModule` it issues to these before touching the worker.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct UpgradeConfig {
    /// Ceiling on the quiesce drain deadline the node passes with a `SwitchModule` (§4.4/§9.6): the
    /// requested deadline is clamped to this before sending, so a guest can never be granted an
    /// unbounded drain (the host wall-clock-enforces it → `QuiesceDeadlineExceeded`). Milliseconds.
    pub quiesce_deadline_max_ms: u64,
    /// Max rollback-and-retry cycles the LOCAL upgrade transaction runs before it leaves the run
    /// (ABI §10.3 step 7; the mid-migration crash drill recovers within this budget).
    pub max_retries: u32,
}

impl Default for UpgradeConfig {
    fn default() -> Self {
        Self {
            quiesce_deadline_max_ms: 30_000,
            max_retries: 1,
        }
    }
}

/// The `[vhc]` config section (spec §10.6).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct VhcConfig {
    /// Master switch (default off; the feature-gated worker must also be installed).
    pub enabled: bool,
    /// Path to the `daemon-vhc-worker` binary (resolved like the `daemon-infer` worker).
    pub worker_path: String,
    /// Data/artifact cache budget in GiB (the artifact LRU bound, §8, RUN-4).
    pub data_cache_gb: u32,
    /// Default participation policy for joined runs.
    pub default_policy: VhcPolicyConfig,
    /// Module-trust posture.
    pub module_trust: ModuleTrust,
    /// Allowlisted coordinator endpoints (discovery + join, §11.1).
    pub coordinator_allowlist: Vec<String>,
    /// The coordinator-registry discovery surface (A3; additive — defaults to "no registry").
    pub registry: RegistryConfig,
    /// iroh transport knobs.
    pub iroh: IrohConfig,
    /// The owner's aggregate resource grants (decisions D6). Additive; the default is derived
    /// conservatively + finitely from the hardware probe when `enabled` (never unbounded — see
    /// [`OwnerBudgetConfig`]).
    pub owner_budget: OwnerBudgetConfig,
    /// Live module-upgrade (`SwitchModule`) bounds (ABI §10.3). Additive; sensible finite defaults.
    pub upgrade: UpgradeConfig,
}

impl Default for VhcConfig {
    fn default() -> Self {
        // Mirrors the spec §10.6 TOML defaults verbatim.
        Self {
            enabled: false,
            // The real training worker binary (`crates/vhc/bins/daemon-vhc-worker`). The pre-A2
            // initial `daemon-vhc` scaffold that this defaulted to only printed a version line and
            // exited, so a stock `[vhc] enabled` node crash-looped its supervisor on spawn.
            worker_path: "daemon-vhc-worker".to_string(),
            data_cache_gb: 50,
            default_policy: VhcPolicyConfig::default(),
            module_trust: ModuleTrust::Signed,
            coordinator_allowlist: vec!["https://api.daemon.ai/api/v1/vhc".to_string()],
            registry: RegistryConfig::default(),
            iroh: IrohConfig::default(),
            owner_budget: OwnerBudgetConfig::default(),
            upgrade: UpgradeConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use figment::providers::{Format, Toml};
    use figment::Figment;

    #[test]
    fn defaults_match_spec() {
        let cfg = VhcConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.worker_path, "daemon-vhc-worker");
        assert_eq!(cfg.data_cache_gb, 50);
        assert_eq!(cfg.default_policy.mode, PolicyMode::Idle);
        assert_eq!(cfg.default_policy.duty_cycle_pct, 100);
        assert_eq!(cfg.module_trust, ModuleTrust::Signed);
        assert_eq!(cfg.iroh.relays, "default");
    }

    #[test]
    fn figment_extracts_vhc_section_additively() {
        // A node config TOML with a partial `[vhc]` table: the supplied keys win, the omitted keys
        // fall back to the §10.6 defaults (additive layering — the seam rule).
        let toml = r#"
            [other]
            unrelated = true

            [vhc]
            enabled = true
            module_trust = "first_party"
            coordinator_allowlist = ["https://coord.local/vhc"]

            [vhc.default_policy]
            mode = "scheduled"
            duty_cycle_pct = 40
            schedule = "0 2 * * *"
        "#;
        let cfg: VhcConfig = Figment::new()
            .merge(Toml::string(toml))
            .extract_inner("vhc")
            .expect("extract [vhc]");

        assert!(cfg.enabled);
        assert_eq!(cfg.module_trust, ModuleTrust::FirstParty);
        assert_eq!(cfg.coordinator_allowlist, vec!["https://coord.local/vhc"]);
        assert_eq!(cfg.default_policy.mode, PolicyMode::Scheduled);
        assert_eq!(cfg.default_policy.duty_cycle_pct, 40);
        assert_eq!(cfg.default_policy.schedule.as_deref(), Some("0 2 * * *"));
        // Omitted keys keep their defaults.
        assert_eq!(cfg.worker_path, "daemon-vhc-worker");
        assert_eq!(cfg.data_cache_gb, 50);
        assert_eq!(cfg.default_policy.vram_cap_mb, 0);
        assert_eq!(cfg.iroh.relays, "default");
    }

    #[test]
    fn owner_budget_section_extracts_additively_and_defaults_finite_derivable() {
        // Default: no owner budget configured — every ledger is left for the node to derive
        // conservatively + finitely from the probe (decisions D6); `unbounded` is off.
        let cfg = VhcConfig::default();
        assert!(!cfg.owner_budget.unbounded);
        assert!(cfg.owner_budget.device_memory_mb.is_empty());
        assert_eq!(cfg.owner_budget.host_ram_mb, 0);
        assert_eq!(cfg.owner_budget.duty_pct, 0);
        assert_eq!(cfg.owner_budget.max_instances, 0);

        // An explicit finite owner budget (incl. the per-device ledger map) extracts additively.
        let toml = r#"
            [vhc]
            enabled = true

            [vhc.owner_budget]
            host_ram_mb = 16384
            duty_pct = 50
            max_instances = 2

            [vhc.owner_budget.device_memory_mb]
            "gpu:0" = 12000
        "#;
        let cfg: VhcConfig = Figment::new()
            .merge(Toml::string(toml))
            .extract_inner("vhc")
            .expect("extract [vhc]");
        assert!(!cfg.owner_budget.unbounded);
        assert_eq!(cfg.owner_budget.host_ram_mb, 16_384);
        assert_eq!(cfg.owner_budget.duty_pct, 50);
        assert_eq!(cfg.owner_budget.max_instances, 2);
        assert_eq!(
            cfg.owner_budget.device_memory_mb.get("gpu:0").copied(),
            Some(12_000)
        );

        // The explicit opt-out extracts too.
        let toml = r#"
            [vhc.owner_budget]
            unbounded = true
        "#;
        let cfg: VhcConfig = Figment::new()
            .merge(Toml::string(toml))
            .extract_inner("vhc")
            .expect("extract [vhc]");
        assert!(cfg.owner_budget.unbounded);
    }

    #[test]
    fn registry_section_extracts_additively() {
        // The A3 `[vhc.registry]` table: base + auth extract; omitted → the "no registry" default
        // (discovery stays None at boot). Both auth modes deserialize.
        let toml = r#"
            [vhc]
            enabled = true

            [vhc.registry]
            base = "http://127.0.0.1:8795/api/v1/vhc"

            [vhc.registry.auth.internal]
            org_id = "org_live"
            actor = "key:live"
        "#;
        let cfg: VhcConfig = Figment::new()
            .merge(Toml::string(toml))
            .extract_inner("vhc")
            .expect("extract [vhc]");
        assert_eq!(cfg.registry.base, "http://127.0.0.1:8795/api/v1/vhc");
        assert_eq!(
            cfg.registry.auth,
            RegistryAuthConfig::Internal {
                org_id: "org_live".into(),
                actor: "key:live".into()
            }
        );

        // Default: no registry configured.
        let cfg = VhcConfig::default();
        assert!(cfg.registry.base.is_empty());
        assert_eq!(cfg.registry.auth, RegistryAuthConfig::None);

        // Bearer mode also extracts.
        let toml = r#"
            base = "https://daemon-vhc-dev.example.workers.dev/api/v1/vhc"
            [auth.bearer]
            token = "sk-test"
        "#;
        let reg: RegistryConfig = Figment::new()
            .merge(Toml::string(toml))
            .extract()
            .expect("extract registry");
        assert_eq!(
            reg.auth,
            RegistryAuthConfig::Bearer {
                token: "sk-test".into()
            }
        );
    }

    #[test]
    // `figment::Jail::expect_with` requires a `Result<_, figment::Error>` return; that error is
    // large, but it is the harness's fixed signature (test-only).
    #[allow(clippy::result_large_err)]
    fn figment_env_overrides_a_key() {
        figment::Jail::expect_with(|jail| {
            jail.set_env("DAEMON_VHC_DATA_CACHE_GB", "128");
            let cfg: VhcConfig = Figment::new()
                .merge(figment::providers::Env::prefixed("DAEMON_VHC_"))
                .extract()
                .expect("extract from env");
            assert_eq!(cfg.data_cache_gb, 128);
            Ok(())
        });
    }
}
