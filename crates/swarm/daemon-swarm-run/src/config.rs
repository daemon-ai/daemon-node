// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `[swarm]` node-config section (spec §10.6).
//!
//! [`SwarmConfig`] is the typed projection of the figment `[swarm]` table the node layers
//! (defaults ← TOML ← env ← CLI). It is defined **here** (lane R) rather than in the node's main
//! config crate — that crate is outside lane R's file set, so the struct + its extraction test land
//! in `daemon-swarm-run` and the node wiring (embedding it in `NodeConfig`) is post-MVP node work.
//!
//! The struct is `serde`-only (no figment on the default participant build); the extraction test
//! exercises the figment layering as a dev-dependency, proving the `[swarm]` keys deserialize
//! additively with the spec §10.6 defaults.

use serde::{Deserialize, Serialize};

use crate::protocol::PolicyMode;

/// Operator posture for run-supplied experiment modules (spec §10.6, §12).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModuleTrust {
    /// Any author-signed module (the permissioned-org default).
    #[default]
    Signed,
    /// Only `daemon-train`'s preset experiments.
    FirstParty,
}

/// The default participation policy for newly-joined runs (`[swarm].default_policy`, §10.5).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SwarmPolicyConfig {
    /// Availability mode.
    pub mode: PolicyMode,
    /// VRAM cap in MiB (`0` = uncapped).
    pub vram_cap_mb: u32,
    /// Duty-cycle percentage (`0..=100`).
    pub duty_cycle_pct: u8,
    /// Optional cron schedule (for [`PolicyMode::Scheduled`]).
    pub schedule: Option<String>,
}

impl Default for SwarmPolicyConfig {
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

/// The `swarm:*` credential a registry request carries (`[swarm.registry].auth`, §11.1). Mirrors
/// `daemon_swarm_net::ws_client::WsAuth` / `RegistryClient`'s auth modes — never hardcoded.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistryAuthConfig {
    /// No auth headers (a bare dev target).
    #[default]
    None,
    /// `Authorization: Bearer <token>` (the gateway `swarm:*` API-key path).
    Bearer {
        /// The bearer token.
        token: String,
    },
    /// The internal identity headers (the direct-to-`apps/swarm` dev path).
    Internal {
        /// `x-daemon-org-id`.
        org_id: String,
        /// `x-daemon-actor`.
        actor: String,
    },
}

/// The coordinator-registry discovery surface (`[swarm.registry]`; A3 — the A1-noted boot follow-on).
///
/// When `base` is non-empty, `bins/daemon` constructs a `RegistryClient`-backed `EgressRunDiscovery`
/// at boot, so `swarm_join` discovers the run, fetches + blake3-verifies the frozen envelope, and
/// runs the worker's real §6.5 `AssessRun` before `JoinRun`. Empty (the default) keeps
/// `discovery: None` — the W1 probe-based fallback against the allowlist. **Deploy-swappable by
/// config only**: the same node targets wrangler-dev (`http://127.0.0.1:8795/api/v1/swarm`) or the
/// real workers.dev deployment (e.g. `https://daemon-swarm-dev.<acct>.workers.dev/api/v1/swarm`)
/// without a code change.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RegistryConfig {
    /// The registry base URL (`""` = no registry → no discovery seam).
    pub base: String,
    /// The `swarm:*` credential for registry + presign requests.
    pub auth: RegistryAuthConfig,
}

/// The iroh transport knobs (`[swarm].iroh`, §7.1). Gossip is mandatory, so unreachable relays make
/// the node swarm-ineligible (§6.5); this MVP surface carries only the relay selector.
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

/// The `[swarm]` config section (spec §10.6).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SwarmConfig {
    /// Master switch (default off; the feature-gated worker must also be installed).
    pub enabled: bool,
    /// Path to the `daemon-train` worker binary (resolved like the `daemon-infer` worker).
    pub worker_path: String,
    /// Data/artifact cache budget in GiB (the artifact LRU bound, §8, RUN-4).
    pub data_cache_gb: u32,
    /// Default participation policy for joined runs.
    pub default_policy: SwarmPolicyConfig,
    /// Module-trust posture.
    pub module_trust: ModuleTrust,
    /// Allowlisted coordinator endpoints (discovery + join, §11.1).
    pub coordinator_allowlist: Vec<String>,
    /// The coordinator-registry discovery surface (A3; additive — defaults to "no registry").
    pub registry: RegistryConfig,
    /// iroh transport knobs.
    pub iroh: IrohConfig,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        // Mirrors the spec §10.6 TOML defaults verbatim.
        Self {
            enabled: false,
            worker_path: "daemon-train".to_string(),
            data_cache_gb: 50,
            default_policy: SwarmPolicyConfig::default(),
            module_trust: ModuleTrust::Signed,
            coordinator_allowlist: vec!["https://api.daemon.ai/api/v1/swarm".to_string()],
            registry: RegistryConfig::default(),
            iroh: IrohConfig::default(),
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
        let cfg = SwarmConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.worker_path, "daemon-train");
        assert_eq!(cfg.data_cache_gb, 50);
        assert_eq!(cfg.default_policy.mode, PolicyMode::Idle);
        assert_eq!(cfg.default_policy.duty_cycle_pct, 100);
        assert_eq!(cfg.module_trust, ModuleTrust::Signed);
        assert_eq!(cfg.iroh.relays, "default");
    }

    #[test]
    fn figment_extracts_swarm_section_additively() {
        // A node config TOML with a partial `[swarm]` table: the supplied keys win, the omitted keys
        // fall back to the §10.6 defaults (additive layering — the seam rule).
        let toml = r#"
            [other]
            unrelated = true

            [swarm]
            enabled = true
            module_trust = "first_party"
            coordinator_allowlist = ["https://coord.local/swarm"]

            [swarm.default_policy]
            mode = "scheduled"
            duty_cycle_pct = 40
            schedule = "0 2 * * *"
        "#;
        let cfg: SwarmConfig = Figment::new()
            .merge(Toml::string(toml))
            .extract_inner("swarm")
            .expect("extract [swarm]");

        assert!(cfg.enabled);
        assert_eq!(cfg.module_trust, ModuleTrust::FirstParty);
        assert_eq!(cfg.coordinator_allowlist, vec!["https://coord.local/swarm"]);
        assert_eq!(cfg.default_policy.mode, PolicyMode::Scheduled);
        assert_eq!(cfg.default_policy.duty_cycle_pct, 40);
        assert_eq!(cfg.default_policy.schedule.as_deref(), Some("0 2 * * *"));
        // Omitted keys keep their defaults.
        assert_eq!(cfg.worker_path, "daemon-train");
        assert_eq!(cfg.data_cache_gb, 50);
        assert_eq!(cfg.default_policy.vram_cap_mb, 0);
        assert_eq!(cfg.iroh.relays, "default");
    }

    #[test]
    fn registry_section_extracts_additively() {
        // The A3 `[swarm.registry]` table: base + auth extract; omitted → the "no registry" default
        // (discovery stays None at boot). Both auth modes deserialize.
        let toml = r#"
            [swarm]
            enabled = true

            [swarm.registry]
            base = "http://127.0.0.1:8795/api/v1/swarm"

            [swarm.registry.auth.internal]
            org_id = "org_live"
            actor = "key:live"
        "#;
        let cfg: SwarmConfig = Figment::new()
            .merge(Toml::string(toml))
            .extract_inner("swarm")
            .expect("extract [swarm]");
        assert_eq!(cfg.registry.base, "http://127.0.0.1:8795/api/v1/swarm");
        assert_eq!(
            cfg.registry.auth,
            RegistryAuthConfig::Internal {
                org_id: "org_live".into(),
                actor: "key:live".into()
            }
        );

        // Default: no registry configured.
        let cfg = SwarmConfig::default();
        assert!(cfg.registry.base.is_empty());
        assert_eq!(cfg.registry.auth, RegistryAuthConfig::None);

        // Bearer mode also extracts.
        let toml = r#"
            base = "https://daemon-swarm-dev.example.workers.dev/api/v1/swarm"
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
            jail.set_env("DAEMON_SWARM_DATA_CACHE_GB", "128");
            let cfg: SwarmConfig = Figment::new()
                .merge(figment::providers::Env::prefixed("DAEMON_SWARM_"))
                .extract()
                .expect("extract from env");
            assert_eq!(cfg.data_cache_gb, 128);
            Ok(())
        });
    }
}
