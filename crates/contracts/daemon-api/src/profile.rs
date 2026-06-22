//! Profile and config contract types: the serializable bundle a GUI creates/edits to configure an
//! agent, plus the runtime config surface.
//!
//! A [`ProfileSpec`] is the full configuration bundle for an agent (the analogue of a hermes
//! `HERMES_HOME` `config.yaml`): which provider + model it talks to, its persona, the tools it may
//! use, its budget/engine tunables, its context/memory backends, and the credential it acquires
//! capabilities from. The host resolves a `ProfileSpec` into an engine-construction `EngineProfile`
//! at session open (`daemon-host`), so a GUI can create/select/edit a profile without restarting
//! the node.
//!
//! These are *contract* types: serializable primitives only (no `daemon-core` types), so the
//! surface never drags the engine's concrete construction types into the wire protocol.

use daemon_common::{SkillBundle, WireVersion};
use serde::{Deserialize, Serialize};

/// Which model provider implementation a profile binds to. Mirrors the host's internal
/// `ProviderKind`, kept as a contract enum so the wire surface does not depend on the binary's
/// config crate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderSelector {
    /// The deterministic in-tree provider (no network/keys).
    #[default]
    Mock,
    /// Any networked provider served by `genai` — the adapter (OpenAI, Anthropic, Gemini, Groq,
    /// DeepSeek, xAI, OpenRouter, Cohere, …) is inferred from the (optionally namespaced) model
    /// name; there is no daemon-side provider registry. Legacy provider names persisted before this
    /// collapse (`openai`, `anthropic`, and the short-lived per-family variants) deserialize here
    /// via serde aliases.
    #[serde(
        rename = "genai",
        alias = "openai",
        alias = "anthropic",
        alias = "gemini",
        alias = "groq",
        alias = "deep_seek",
        alias = "xai",
        alias = "open_router",
        alias = "cohere"
    )]
    GenAi,
    /// A local llama.cpp model via the supervised `daemon-infer` worker (on-disk GGUF, listed from
    /// the `ModelManager` catalog).
    LlamaCpp,
    /// A local mistral.rs model via the supervised `daemon-infer` worker (on-disk, listed from the
    /// `ModelManager` catalog).
    MistralRs,
}

impl ProviderSelector {
    /// Whether this selector is a local-inference engine (llama.cpp / mistral.rs).
    pub fn is_local(self) -> bool {
        matches!(self, ProviderSelector::LlamaCpp | ProviderSelector::MistralRs)
    }
}

/// Which default context engine (§10) a profile wires into its engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextEngineSel {
    /// The native LCM port (`daemon-context-lcm`) — the default.
    #[default]
    Lcm,
    /// The in-core budgeted (drop-oldest) context engine.
    Budgeted,
}

/// Which default memory provider (§11) a profile wires into its engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryProviderSel {
    /// The native Mnemosyne port (`daemon-mnemosyne`) — the default.
    #[default]
    Mnemosyne,
    /// The in-core file-backed memory over a frozen snapshot.
    File,
    /// No memory provider.
    None,
}

/// The subset of engine tunables (§20) a profile can override. `None` fields fall back to the
/// node/engine default at resolution time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineTunables {
    /// Per-turn model retry attempts.
    pub model_retry_attempts: Option<u8>,
    /// Soft context-token budget hint for compaction.
    pub context_budget_tokens: Option<u32>,
    /// Per-turn ReAct round cap.
    pub max_iterations: Option<u32>,
    /// Per-tool result-byte cap.
    pub tool_result_budget: Option<usize>,
}

/// An optional budget ceiling carried on a profile (token / wall-clock).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BudgetSpec {
    /// Optional token ceiling (`None` = unbounded).
    pub tokens: Option<u64>,
    /// Optional wall-clock ceiling in milliseconds (`None` = unbounded).
    pub wall_ms: Option<u64>,
}

/// The full agent configuration bundle a GUI creates/edits and a session binds to.
///
/// One profile is the unit a GUI manages: it names a provider + model, the persona system prompt,
/// the tool allowlist, the engine budget/tunables, the context/memory backends, and the credential
/// it acquires from. The host resolves it into an `EngineProfile` per session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileSpec {
    /// The profile's unique id/name (also its on-disk key and its credential profile by default).
    pub id: String,
    /// Which provider implementation this profile binds to.
    #[serde(default)]
    pub provider: ProviderSelector,
    /// The model name (cloud model id, or local GGUF path / HF id).
    #[serde(default)]
    pub model: String,
    /// Optional provider API base-URL override (`None` = the provider default endpoint).
    #[serde(default)]
    pub base_url: Option<String>,
    /// The persona / system prompt this profile's engine runs under.
    #[serde(default)]
    pub system_prompt: String,
    /// The tools this profile's engine may use. `None` = the full node toolset; `Some(list)` =
    /// only those tool names (an allowlist).
    #[serde(default)]
    pub tool_allowlist: Option<Vec<String>>,
    /// The profile's budget ceiling.
    #[serde(default)]
    pub budget: BudgetSpec,
    /// Engine tunable overrides.
    #[serde(default)]
    pub tunables: EngineTunables,
    /// The default context engine (§10).
    #[serde(default)]
    pub context_engine: ContextEngineSel,
    /// The default memory provider (§11).
    #[serde(default)]
    pub memory_provider: MemoryProviderSel,
    /// The credential reference (profile/key) this engine acquires capabilities from. `None`
    /// defaults to the profile `id`.
    #[serde(default)]
    pub credential_ref: Option<String>,
    /// An optional fallback credential profile the engine fails over to when the primary credential
    /// profile is exhausted (the `Recovery::Fallback` hop). Composes a cross-credential failover
    /// chain on top of the per-profile multi-key pool. `None` = no fallback.
    #[serde(default)]
    pub fallback_credential_ref: Option<String>,
}

impl ProfileSpec {
    /// A minimal profile over a provider + model, with empty persona and full toolset.
    pub fn new(id: impl Into<String>, provider: ProviderSelector, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            provider,
            model: model.into(),
            base_url: None,
            system_prompt: String::new(),
            tool_allowlist: None,
            budget: BudgetSpec::default(),
            tunables: EngineTunables::default(),
            context_engine: ContextEngineSel::default(),
            memory_provider: MemoryProviderSel::default(),
            credential_ref: None,
            fallback_credential_ref: None,
        }
    }

    /// The credential profile this spec acquires from (explicit `credential_ref`, else the id).
    pub fn credential_profile(&self) -> &str {
        self.credential_ref.as_deref().unwrap_or(&self.id)
    }

    /// The fallback credential profile, if configured (the `Recovery::Fallback` target).
    pub fn fallback_credential_profile(&self) -> Option<&str> {
        self.fallback_credential_ref.as_deref()
    }
}

/// A portable, self-contained profile distribution: the unit you export from one node and import on
/// another. It carries the [`ProfileSpec`] plus the profile's local skill bundles; `credential_ref`
/// is **kept** (it is a name, not a secret — the importer registers the key via `CredentialSet`).
/// Secrets never live in a profile, so nothing is stripped.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Distribution {
    /// The wire version this distribution was produced under (import validates compatibility).
    pub wire_version: WireVersion,
    /// The profile configuration bundle.
    pub profile: ProfileSpec,
    /// The profile's local (non-binary-bundled) skills, reconstituted on import.
    #[serde(default)]
    pub skills: Vec<SkillBundle>,
    /// The profile's head revision sequence at export time, for provenance display (`None` if the
    /// profile had no recorded history yet).
    #[serde(default)]
    pub head_seq: Option<u64>,
    /// Optional free-form origin label (who/where it came from).
    #[serde(default)]
    pub source: Option<String>,
}

/// A redacted view of a profile for listing (no secrets live in a profile, but this is the shape a
/// GUI list renders).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileInfo {
    /// The profile id/name.
    pub id: String,
    /// The provider it binds to.
    pub provider: ProviderSelector,
    /// The model name.
    pub model: String,
    /// Whether this profile is the active default.
    pub is_active: bool,
}

impl ProfileInfo {
    /// Build a listing view from a spec, marking active state.
    pub fn from_spec(spec: &ProfileSpec, is_active: bool) -> Self {
        Self {
            id: spec.id.clone(),
            provider: spec.provider,
            model: spec.model.clone(),
            is_active,
        }
    }
}

/// A partial update to a profile's runtime-settable config: the dynamically tunable surface a GUI
/// drives (`DAEMON_MODEL`, `DAEMON_MODEL_PROVIDER`, base URL, persona, credential). Every field is
/// optional; `None` leaves the existing value unchanged.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConfigPatch {
    /// Re-bind the provider implementation.
    pub provider: Option<ProviderSelector>,
    /// Set the model name.
    pub model: Option<String>,
    /// Set the provider API base URL (empty string clears the override back to the default).
    pub base_url: Option<String>,
    /// Set the persona / system prompt.
    pub system_prompt: Option<String>,
    /// Set the credential reference (empty string clears it back to the profile id).
    pub credential_ref: Option<String>,
    /// Override engine tunables (merged field-wise).
    pub tunables: Option<EngineTunables>,
}

impl ConfigPatch {
    /// Apply this patch onto a spec in place, returning whether anything changed.
    pub fn apply(&self, spec: &mut ProfileSpec) -> bool {
        let mut changed = false;
        if let Some(p) = self.provider {
            spec.provider = p;
            changed = true;
        }
        if let Some(m) = &self.model {
            spec.model = m.clone();
            changed = true;
        }
        if let Some(b) = &self.base_url {
            spec.base_url = if b.is_empty() { None } else { Some(b.clone()) };
            changed = true;
        }
        if let Some(s) = &self.system_prompt {
            spec.system_prompt = s.clone();
            changed = true;
        }
        if let Some(c) = &self.credential_ref {
            spec.credential_ref = if c.is_empty() { None } else { Some(c.clone()) };
            changed = true;
        }
        if let Some(t) = self.tunables {
            spec.tunables = t;
            changed = true;
        }
        changed
    }
}

/// One settable config field, as the `ConfigSchema` advertises it to a GUI building a settings form.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigField {
    /// The field key (matches a `ConfigPatch` field / `DAEMON_*` knob).
    pub key: String,
    /// A short value-kind hint (`"string"`, `"enum"`, `"u32"`, ...).
    pub kind: String,
    /// Human-readable description.
    pub description: String,
    /// For `enum` kinds, the permitted values.
    #[serde(default)]
    pub options: Vec<String>,
}

/// A discoverable model entry: what a GUI's model picker renders. Merges cloud-provider catalog
/// entries (well-known models incl. `claude-opus-4-8`) and locally-installed models.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDescriptor {
    /// The model id sent to the provider (or the local catalog id).
    pub id: String,
    /// The provider this model is served by.
    pub provider: ProviderSelector,
    /// The context window in tokens, when known (the denominator for a context-fill HUD).
    pub context_length: Option<u32>,
    /// Input price in micro-USD per million tokens, when known (e.g. $3.00 => 3_000_000).
    pub input_price_micros_per_mtok: Option<u64>,
    /// Output price in micro-USD per million tokens, when known.
    pub output_price_micros_per_mtok: Option<u64>,
    /// Whether this is a locally-installed model (vs a cloud model).
    pub local: bool,
}

impl ModelDescriptor {
    /// A cloud model entry.
    pub fn cloud(
        id: impl Into<String>,
        provider: ProviderSelector,
        context_length: Option<u32>,
    ) -> Self {
        Self {
            id: id.into(),
            provider,
            context_length,
            input_price_micros_per_mtok: None,
            output_price_micros_per_mtok: None,
            local: false,
        }
    }

    /// Attach per-million-token pricing (USD), stored as micro-USD.
    pub fn with_pricing(mut self, input_usd: f64, output_usd: f64) -> Self {
        self.input_price_micros_per_mtok = Some((input_usd * 1_000_000.0) as u64);
        self.output_price_micros_per_mtok = Some((output_usd * 1_000_000.0) as u64);
        self
    }

    /// The built-in genai-model catalog: a curated static fallback for the GUI picker (used when no
    /// provider key is set for live `all_model_names` discovery) and the pricing/context overlay
    /// keyed by model id (genai supplies neither price nor context window). Every entry is the
    /// `GenAi` provider; the genai adapter is inferred from the model id, so ids that do not
    /// self-identify by prefix (Groq, OpenRouter) are namespaced (`groq::…`, `open_router::…`).
    /// Context windows are the published maxima; pricing is the public list price (USD per Mtok).
    pub fn builtin_cloud_catalog() -> Vec<ModelDescriptor> {
        use ProviderSelector::GenAi;
        vec![
            ModelDescriptor::cloud("claude-opus-4-8", GenAi, Some(200_000))
                .with_pricing(15.0, 75.0),
            ModelDescriptor::cloud("claude-sonnet-4-5", GenAi, Some(200_000))
                .with_pricing(3.0, 15.0),
            ModelDescriptor::cloud("claude-3-5-sonnet-latest", GenAi, Some(200_000))
                .with_pricing(3.0, 15.0),
            ModelDescriptor::cloud("claude-3-5-haiku-latest", GenAi, Some(200_000))
                .with_pricing(0.80, 4.0),
            ModelDescriptor::cloud("gpt-4o", GenAi, Some(128_000)).with_pricing(2.5, 10.0),
            ModelDescriptor::cloud("gpt-4o-mini", GenAi, Some(128_000)).with_pricing(0.15, 0.60),
            ModelDescriptor::cloud("o3", GenAi, Some(200_000)).with_pricing(2.0, 8.0),
            ModelDescriptor::cloud("gemini-2.5-pro", GenAi, Some(1_048_576))
                .with_pricing(1.25, 10.0),
            ModelDescriptor::cloud("gemini-2.5-flash", GenAi, Some(1_048_576))
                .with_pricing(0.30, 2.5),
            ModelDescriptor::cloud("deepseek-chat", GenAi, Some(128_000))
                .with_pricing(0.27, 1.10),
            ModelDescriptor::cloud("deepseek-reasoner", GenAi, Some(128_000))
                .with_pricing(0.55, 2.19),
            ModelDescriptor::cloud("grok-4", GenAi, Some(256_000)).with_pricing(3.0, 15.0),
            // Groq models need an explicit namespace for adapter inference (genai v0.6.0+).
            ModelDescriptor::cloud("groq::llama-3.3-70b-versatile", GenAi, Some(131_072))
                .with_pricing(0.59, 0.79),
            ModelDescriptor::cloud("command-r-plus", GenAi, Some(128_000)).with_pricing(2.5, 10.0),
            // OpenRouter is a namespaced gateway: a representative default entry.
            ModelDescriptor::cloud("open_router::openai/gpt-4o", GenAi, Some(128_000))
                .with_pricing(2.5, 10.0),
        ]
    }

    /// The published context window for a well-known cloud model id, if any (the denominator a
    /// provider reports as `Capabilities.max_context`).
    pub fn known_context_length(id: &str) -> Option<u32> {
        Self::builtin_cloud_catalog()
            .into_iter()
            .find(|m| m.id == id)
            .and_then(|m| m.context_length)
    }
}

/// A redacted view of a stored credential (the shape a GUI's "API keys" list renders). The secret
/// itself is never returned on a read — only whether one is present and a short masked hint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialInfo {
    /// The profile / credential-ref the secret is keyed by.
    pub profile: String,
    /// Whether a secret is stored for this profile.
    pub present: bool,
    /// A masked hint (e.g. the last four characters), never the full secret.
    pub hint: String,
}

impl CredentialInfo {
    /// Build a redacted view from a profile and its (optional) secret.
    pub fn redacted(profile: impl Into<String>, secret: Option<&str>) -> Self {
        let profile = profile.into();
        match secret {
            Some(s) if !s.is_empty() => {
                let tail: String = s.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
                Self {
                    profile,
                    present: true,
                    hint: format!("…{tail}"),
                }
            }
            _ => Self {
                profile,
                present: false,
                hint: String::new(),
            },
        }
    }
}

/// The settable-config schema a GUI renders as a settings form. Static for v1.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigSchema {
    /// The settable fields.
    pub fields: Vec<ConfigField>,
}

impl ConfigSchema {
    /// The built-in schema describing the dynamically-settable profile config surface.
    pub fn builtin() -> Self {
        let field = |key: &str, kind: &str, description: &str, options: &[&str]| ConfigField {
            key: key.to_string(),
            kind: kind.to_string(),
            description: description.to_string(),
            options: options.iter().map(|s| s.to_string()).collect(),
        };
        Self {
            fields: vec![
                field(
                    "provider",
                    "enum",
                    "Model provider implementation",
                    &["mock", "genai", "llama_cpp", "mistral_rs"],
                ),
                field("model", "string", "Model name / id (e.g. claude-opus-4-8)", &[]),
                field("base_url", "string", "Provider API base URL override (empty = default)", &[]),
                field("system_prompt", "string", "Persona / system prompt", &[]),
                field("credential_ref", "string", "Credential profile this engine acquires from", &[]),
                field("context_budget_tokens", "u32", "Soft context-token budget hint", &[]),
                field("max_iterations", "u32", "Per-turn ReAct round cap", &[]),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_locality_and_serde_form() {
        assert!(ProviderSelector::LlamaCpp.is_local());
        assert!(ProviderSelector::MistralRs.is_local());
        assert!(!ProviderSelector::GenAi.is_local());
        assert!(!ProviderSelector::Mock.is_local());
        // The networked selector serializes to the stable "genai" wire id.
        assert_eq!(
            serde_json::to_string(&ProviderSelector::GenAi).unwrap(),
            "\"genai\""
        );
    }

    #[test]
    fn legacy_provider_names_migrate_to_genai() {
        // Profiles persisted before the collapse used per-provider names; they all deserialize to
        // the single genai-backed selector (the adapter is then inferred from the model id).
        for legacy in [
            "openai",
            "anthropic",
            "gemini",
            "groq",
            "deep_seek",
            "xai",
            "open_router",
            "cohere",
            "genai",
        ] {
            let sel: ProviderSelector =
                serde_json::from_str(&format!("\"{legacy}\"")).unwrap();
            assert_eq!(sel, ProviderSelector::GenAi, "{legacy} should map to GenAi");
        }
        // The local engines keep their own ids.
        assert_eq!(
            serde_json::from_str::<ProviderSelector>("\"llama_cpp\"").unwrap(),
            ProviderSelector::LlamaCpp
        );
    }

    #[test]
    fn catalog_is_genai_with_known_context() {
        let catalog = ModelDescriptor::builtin_cloud_catalog();
        assert!(catalog.iter().all(|m| m.provider == ProviderSelector::GenAi));
        // opus is still discoverable with its published context window (the overlay).
        assert_eq!(
            ModelDescriptor::known_context_length("claude-opus-4-8"),
            Some(200_000)
        );
        // Groq/OpenRouter ids are namespaced so genai can infer the adapter.
        assert!(catalog
            .iter()
            .any(|m| m.id.starts_with("groq::") || m.id.starts_with("open_router::")));
    }

    #[test]
    fn fallback_credential_ref_accessor() {
        let mut spec = ProfileSpec::new("primary", ProviderSelector::GenAi, "claude-opus-4-8");
        assert_eq!(spec.fallback_credential_profile(), None);
        spec.fallback_credential_ref = Some("backup".into());
        assert_eq!(spec.fallback_credential_profile(), Some("backup"));
    }
}
