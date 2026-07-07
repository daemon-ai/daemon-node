// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Profile contract types: the serializable bundle a GUI creates/edits to configure an agent.
//!
//! A [`ProfileSpec`] is the full configuration bundle for an agent (the analogue of a hermes
//! `HERMES_HOME` `config.yaml`): which provider + model it talks to, its persona, the tools it may
//! use, its budget/engine tunables, its context/memory backends, and the credential it acquires
//! capabilities from. The host resolves a `ProfileSpec` into an engine-construction `EngineProfile`
//! at session open (`daemon-host`), so a GUI can create/select/edit a profile without restarting
//! the node. There is no separate runtime-config surface: a profile is edited in full via
//! `ProfileUpdate`, and a live session is adjusted via a `SessionOverlay`.
//!
//! These are *contract* types: serializable primitives only (no `daemon-core` types), so the
//! surface never drags the engine's concrete construction types into the wire protocol.

use daemon_common::{SkillBundle, WireVersion};
// Relocated to `daemon-protocol` so the wire types that cannot depend on `daemon-api` can carry
// them without a contract-crate cycle: `EngineSelector` (wire v29, the fleet tree's `UnitNode`);
// `ProviderSelector` + `ForeignBackend` (Phase 1, the inline sub-agent spec on the delegation
// payload). Re-exported here because `ProfileSpec` is their primary carrier.
pub use daemon_protocol::{EngineSelector, ForeignBackend, ProviderSelector};
use serde::{Deserialize, Serialize};

/// Which default context engine (§10) a profile wires into its engine.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BudgetSpec {
    /// Optional token ceiling (`None` = unbounded).
    pub tokens: Option<u64>,
    /// Optional wall-clock ceiling in milliseconds (`None` = unbounded).
    pub wall_ms: Option<u64>,
}

/// A transport-instance account bound to this profile (event-io spec §5.9.4 / Matrix spec §6.2).
///
/// This is the **account → profile** binding declared as profile data (not a route-table column):
/// the host derives the routing registry's `instance_profiles` map (precedence step 2 — the
/// account's default agent for all its scopes) from every profile's `bound_accounts`. It is
/// transport-agnostic — any chat/transport family reuses the same shape.
///
/// `transport_instance` is the instance-qualified [`TransportId`](daemon_common) string (e.g.
/// `matrix/@bot:hs.org`); `credential_ref` names where the opaque account session blob lives in the
/// `CredentialStore` (the system of record). Routing consumes only `transport_instance`;
/// `credential_ref` is metadata a live transport (M2/M3) reads to restore the account's client. No
/// secret ever lives here — `credential_ref` is a name, not the blob.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundAccount {
    /// The instance-qualified transport id this account speaks as (e.g. `matrix/@bot:hs.org`).
    pub transport_instance: String,
    /// The credential ref naming the account's stored session blob (a name, never the secret).
    pub credential_ref: String,
}

impl BoundAccount {
    /// A binding of `transport_instance` to its stored `credential_ref`.
    pub fn new(transport_instance: impl Into<String>, credential_ref: impl Into<String>) -> Self {
        Self {
            transport_instance: transport_instance.into(),
            credential_ref: credential_ref.into(),
        }
    }
}

/// The full agent configuration bundle a GUI creates/edits and a session binds to.
///
/// One profile is the unit a GUI manages: it names a provider + model, the persona system prompt,
/// the tool allowlist, the engine budget/tunables, the context/memory backends, and the credential
/// it acquires from. The host resolves it into an `EngineProfile` per session.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
    /// Transport-instance accounts bound to this profile (§5.9.4): the host derives the routing
    /// registry's `instance_profiles` baseline (account → this profile) from these. Empty by default.
    #[serde(default)]
    pub bound_accounts: Vec<BoundAccount>,
    /// Which execution engine this profile's sessions run on (wire v23; generalized v29): the
    /// native `daemon-core` engine (default), or a foreign agent referenced from the node's
    /// catalog by name only.
    #[serde(default)]
    pub engine: EngineSelector,
    /// For a `Foreign` engine: how that agent sources its model backend (its own, model-steered via
    /// ACP; or routed through the node gateway to a node provider). Ignored for `Core` (whose
    /// provider/model live in the top-level fields). Defaults to `AgentNative { model: None }` so a
    /// pre-`foreign_backend` encoding decodes to today's behavior (safe migration).
    #[serde(default)]
    pub foreign_backend: ForeignBackend,
}

impl ProfileSpec {
    /// A minimal profile over a provider + model, with empty persona and full toolset.
    pub fn new(
        id: impl Into<String>,
        provider: ProviderSelector,
        model: impl Into<String>,
    ) -> Self {
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
            bound_accounts: Vec::new(),
            engine: EngineSelector::Core,
            foreign_backend: ForeignBackend::default(),
        }
    }

    /// Materialize a full [`ProfileSpec`] from an ad-hoc [`InlineProfileSpec`](daemon_protocol::InlineProfileSpec)
    /// under a synthetic `id` — the Phase 1 inline-sub-agent seam. The inline spec carries only the
    /// security-relevant subset (persona + tool allowlist + engine + foreign backend + model); every
    /// other field takes the `ProfileSpec::new` default (provider `Mock`, no credential/budget), so a
    /// Core inline sub-agent resolves through the ordinary `resolve_effective` path and a Foreign one
    /// routes to the foreign incarnation. It is never persisted in the profile store (the child binds
    /// `bound_profile = None`); the id is only its transient engine identity.
    pub fn from_inline(id: impl Into<String>, inline: &daemon_protocol::InlineProfileSpec) -> Self {
        Self {
            system_prompt: inline.system_prompt.clone(),
            tool_allowlist: inline.tool_allowlist.clone(),
            model: inline.model.clone(),
            engine: inline.engine.clone(),
            foreign_backend: inline.foreign_backend.clone(),
            ..Self::new(id, ProviderSelector::default(), inline.model.clone())
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

    /// Declare the transport-instance accounts bound to this profile (§5.9.4 account → profile).
    pub fn with_bound_accounts(mut self, accounts: Vec<BoundAccount>) -> Self {
        self.bound_accounts = accounts;
        self
    }
}

/// A portable, self-contained profile distribution: the unit you export from one node and import on
/// another. It carries the [`ProfileSpec`] plus the profile's local skill bundles; `credential_ref`
/// is **kept** (it is a name, not a secret — the importer registers the key via `CredentialSet`).
/// Secrets never live in a profile, so nothing is stripped.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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

/// One row of a profile's curator listing ([`crate::ProfileApi::curator_list`]): a discovered or
/// archived skill with its usage + lifecycle record. The `usage` defaults (all-zero, `Active`) for a
/// skill that has no `.usage.json` entry yet (e.g. a freshly-seeded bundled skill).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CuratorEntry {
    /// The skill (bundle) name.
    pub name: String,
    /// The category path segment, if any.
    pub category: Option<String>,
    /// Whether this is a binary-bundled skill (protected from auto-curation).
    pub is_bundled: bool,
    /// The per-skill usage + lifecycle record (counts, state, pinned, provenance).
    pub usage: daemon_common::SkillUsage,
}

/// One lifecycle change a curator run applied ([`crate::ProfileApi::curator_run`]).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CuratorChange {
    /// The skill (bundle) name.
    pub name: String,
    /// The state it moved from.
    pub from: daemon_common::SkillState,
    /// The state it moved to.
    pub to: daemon_common::SkillState,
}

/// A redacted view of a profile for listing (no secrets live in a profile, but this is the shape a
/// GUI list renders).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
    /// The transport-instance accounts bound to this profile (§5.9.4). Names only, never secrets.
    #[serde(default)]
    pub bound_accounts: Vec<BoundAccount>,
}

impl ProfileInfo {
    /// Build a listing view from a spec, marking active state.
    pub fn from_spec(spec: &ProfileSpec, is_active: bool) -> Self {
        Self {
            id: spec.id.clone(),
            provider: spec.provider,
            model: spec.model.clone(),
            is_active,
            bound_accounts: spec.bound_accounts.clone(),
        }
    }
}

/// How a [`SessionOverlay`] overrides the bound profile's `tool_allowlist`. A tri-state so the
/// overlay can distinguish "leave the profile's allowlist alone" from "override to the full node
/// toolset" from "override to this explicit list".
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolsOverride {
    /// Inherit the bound profile's `tool_allowlist` (no session-level override).
    #[default]
    Inherit,
    /// Override to the full node toolset (the profile's allowlist is ignored for this session).
    FullToolset,
    /// Override to exactly these tool names (a session-level allowlist).
    Allowlist(Vec<String>),
}

/// A per-session override layered on top of the session's bound [`ProfileSpec`] at engine
/// construction. This is the single per-session adjustment surface (it subsumes the older
/// per-session model switch and edit-approval mode): the profile is the durable base, the overlay
/// is the live tweak. It is persisted as host-level session metadata, so it is **restored on
/// rehydration** rather than lost on restart. Every field is optional / inherit; unset fields fall
/// through to the bound profile.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionOverlay {
    /// Override the model id (`None` = inherit the profile's model).
    pub model: Option<String>,
    /// Override the provider implementation (`None` = inherit the profile's provider).
    pub provider: Option<ProviderSelector>,
    /// Override the tool allowlist (see [`ToolsOverride`]).
    pub tool_allowlist: ToolsOverride,
    /// Override the edit-approval mode (`None` = inherit the profile/engine default).
    pub approval_mode: Option<crate::ApprovalMode>,
    /// How the session's workspace root is chosen (`None` = the node default = isolated
    /// per-session sandbox). `Some(Bound(path))` roots the session's engine + filesystem surface
    /// at an operator-specified directory, in place (the "work on my repo" case).
    pub workspace: Option<daemon_common::WorkspaceBinding>,
}

impl SessionOverlay {
    /// Whether this overlay is a pure no-op (every field inherits the profile).
    pub fn is_empty(&self) -> bool {
        self.model.is_none()
            && self.provider.is_none()
            && matches!(self.tool_allowlist, ToolsOverride::Inherit)
            && self.approval_mode.is_none()
            && self.workspace.is_none()
    }

    /// Whether this overlay *widens* the session's security posture — a security-relevant change that
    /// requires an operator-tier capability to apply (Cluster E policy partition). Two widenings:
    /// an autonomy-widening `approval_mode` ([`ApprovalMode::widens_autonomy`]), or
    /// [`ToolsOverride::FullToolset`] (override to the full node toolset). Every other overlay field —
    /// model/provider/workspace switches, an explicit `Allowlist` (a restriction), `Inherit`, and the
    /// non-widening `Ask`/`Deny` approval modes — is not a widening and stays owner-allowed.
    pub fn widens_security_posture(&self) -> bool {
        self.approval_mode
            .is_some_and(crate::ApprovalMode::widens_autonomy)
            || matches!(self.tool_allowlist, ToolsOverride::FullToolset)
    }

    /// Apply the model/provider/tool-allowlist overrides onto a profile spec in place. The
    /// `approval_mode` is applied to the engine separately (it is not a `ProfileSpec` field).
    pub fn apply_to(&self, spec: &mut ProfileSpec) {
        if let Some(model) = &self.model {
            spec.model = model.clone();
        }
        if let Some(provider) = self.provider {
            spec.provider = provider;
        }
        match &self.tool_allowlist {
            ToolsOverride::Inherit => {}
            ToolsOverride::FullToolset => spec.tool_allowlist = None,
            ToolsOverride::Allowlist(list) => spec.tool_allowlist = Some(list.clone()),
        }
    }
}

/// A discoverable model entry: what a GUI's model picker renders. Merges cloud-provider catalog
/// entries (well-known models incl. `claude-opus-4-8`) and locally-installed models.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDescriptor {
    /// The model id sent to the provider (or the local catalog id). This is directly usable as
    /// `ProfileSpec.model` with NO client-side rewriting: Daemon Cloud ids stay `author/slug`; genai
    /// vendor ids stay whatever genai returns, namespaced only when required for adapter inference.
    pub id: String,
    /// The provider this model is served by.
    pub provider: ProviderSelector,
    /// An optional nicer label for the picker (genai returns bare ids); `None` => render `id`.
    #[serde(default)]
    pub display_name: Option<String>,
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
            display_name: None,
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
            ModelDescriptor::cloud("deepseek-chat", GenAi, Some(128_000)).with_pricing(0.27, 1.10),
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

/// The picker "kind" bucket a [`ProviderDescriptor`] groups by. Distinct from [`ProviderSelector`]
/// (the wire binding a profile persists): `DaemonCloud` is grouped separately in the UI but maps to
/// `ProviderSelector::DaemonApi`; every genai cloud vendor maps to `ProviderSelector::GenAi`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKindWire {
    /// A local inference engine (llama.cpp / mistral.rs).
    Local,
    /// A third-party cloud vendor served through genai (Anthropic, OpenAI, Gemini, …).
    Cloud,
    /// Daemon Cloud — the OpenRouter-clone gateway (`ProviderSelector::DaemonApi`).
    DaemonCloud,
}

/// The interactive sign-in a provider advertises (wire v30, CON-15). The node states the auth
/// family and render label; the client calls `auth_begin { family, params: {} }` and the node fills
/// every flow detail. Clients key nothing off vendor ids.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderSignIn {
    /// The `auth_begin.family` to drive interactive sign-in for this provider.
    pub family: String,
    /// The node-decided button/render label (e.g. "Sign in with OpenRouter").
    pub label: String,
}

/// One discoverable provider row for the GUI/TUI picker (returned by `ProviderCatalog`). Carries
/// everything the client needs to render the provider list and, on selection, drive `ProviderModels`
/// and persist a working profile — without hardcoding any endpoint or provider list.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderDescriptor {
    /// Stable provider id (e.g. `"anthropic"`, `"openai"`, `"llama_cpp"`, `"daemon_cloud"`).
    pub id: String,
    /// Human label for the picker (genai-sourced where possible; not hand-relabeled beyond case).
    pub display_name: String,
    /// The picker grouping bucket.
    pub kind: ProviderKindWire,
    /// The wire selector a persisted `ProfileSpec` must use for this provider.
    pub wire_selector: ProviderSelector,
    /// Whether LISTING this provider's models needs a key: `false` for Daemon Cloud (keyless list)
    /// and local engines; `true` for genai cloud vendors. (A turn always uses the stored profile
    /// credential regardless.)
    pub requires_key: bool,
    /// Whether `ProviderModels` can enumerate this provider (local providers still support it — they
    /// return the installed models; the client appends its own "Discover More").
    pub supports_model_discovery: bool,
    /// The gateway/base URL the client should persist for this provider, so it never hardcodes one.
    /// Daemon Cloud carries `https://api.daemon.ai/api/v1/`; `None` for genai vendors + local.
    pub default_base_url: Option<String>,
    /// The interactive sign-in this provider supports (wire v30, CON-15). `Some` for a provider
    /// with an interactive login (the OpenRouter genai row advertises `family
    /// "provider/openrouter"`); `None` for key-in-a-field-only and local providers.
    #[serde(default)]
    pub sign_in: Option<ProviderSignIn>,
}

/// A redacted view of a stored credential (the shape a GUI's "API keys" list renders). The secret
/// itself is never returned on a read — only whether one is present and a short masked hint.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
                let tail: String = s
                    .chars()
                    .rev()
                    .take(4)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_overlay_inherits_the_profile() {
        let overlay = SessionOverlay::default();
        assert!(overlay.is_empty());
        let mut spec = ProfileSpec::new("p", ProviderSelector::GenAi, "base-model");
        spec.tool_allowlist = Some(vec!["fs".to_string()]);
        let before = spec.clone();
        overlay.apply_to(&mut spec);
        // An all-inherit overlay is a pure no-op: every field falls through to the profile.
        assert_eq!(spec, before);
    }

    #[test]
    fn overlay_overrides_model_provider_and_tools() {
        let overlay = SessionOverlay {
            model: Some("override-model".to_string()),
            provider: Some(ProviderSelector::Mock),
            tool_allowlist: ToolsOverride::Allowlist(vec!["fs".to_string()]),
            approval_mode: Some(crate::ApprovalMode::AutoAllow),
            workspace: None,
        };
        assert!(!overlay.is_empty());
        let mut spec = ProfileSpec::new("p", ProviderSelector::GenAi, "base-model");
        overlay.apply_to(&mut spec);
        assert_eq!(spec.model, "override-model");
        assert_eq!(spec.provider, ProviderSelector::Mock);
        assert_eq!(spec.tool_allowlist, Some(vec!["fs".to_string()]));
    }

    #[test]
    fn approval_mode_widening_classification() {
        // Cluster E: only the autonomy-*widening* modes are operator-gated on an overlay; the safe
        // directions (default `Ask`, strictest `Deny`) are not widenings and stay owner-allowed.
        assert!(crate::ApprovalMode::AcceptEdits.widens_autonomy());
        assert!(crate::ApprovalMode::AutoAllow.widens_autonomy());
        assert!(!crate::ApprovalMode::Ask.widens_autonomy());
        assert!(!crate::ApprovalMode::Deny.widens_autonomy());
    }

    #[test]
    fn overlay_widens_on_full_toolset_or_autonomy_widening() {
        // FullToolset widens the tool surface.
        let full = SessionOverlay {
            tool_allowlist: ToolsOverride::FullToolset,
            ..SessionOverlay::default()
        };
        assert!(full.widens_security_posture());
        // A widening approval mode widens autonomy.
        let yolo = SessionOverlay {
            approval_mode: Some(crate::ApprovalMode::AutoAllow),
            ..SessionOverlay::default()
        };
        assert!(yolo.widens_security_posture());
        let accept = SessionOverlay {
            approval_mode: Some(crate::ApprovalMode::AcceptEdits),
            ..SessionOverlay::default()
        };
        assert!(accept.widens_security_posture());
        // Non-widening: an explicit allowlist, a narrowing/neutral approval mode, and a bare
        // model/provider/workspace switch are all owner-allowed (not security-widening).
        let allowlist = SessionOverlay {
            tool_allowlist: ToolsOverride::Allowlist(vec!["fs".into()]),
            approval_mode: Some(crate::ApprovalMode::Deny),
            model: Some("m".into()),
            provider: Some(ProviderSelector::Mock),
            ..SessionOverlay::default()
        };
        assert!(!allowlist.widens_security_posture());
        assert!(!SessionOverlay::default().widens_security_posture());
        let ask = SessionOverlay {
            approval_mode: Some(crate::ApprovalMode::Ask),
            ..SessionOverlay::default()
        };
        assert!(!ask.widens_security_posture());
    }

    #[test]
    fn tools_override_full_toolset_clears_the_allowlist() {
        // `FullToolset` overrides a profile that pinned an allowlist back to the full node toolset.
        let overlay = SessionOverlay {
            tool_allowlist: ToolsOverride::FullToolset,
            ..SessionOverlay::default()
        };
        let mut spec = ProfileSpec::new("p", ProviderSelector::GenAi, "m");
        spec.tool_allowlist = Some(vec!["fs".to_string()]);
        overlay.apply_to(&mut spec);
        assert_eq!(spec.tool_allowlist, None);
    }

    #[test]
    fn overlay_cbor_round_trips() {
        let overlay = SessionOverlay {
            model: Some("m".to_string()),
            provider: Some(ProviderSelector::GenAi),
            tool_allowlist: ToolsOverride::Allowlist(vec!["a".into(), "b".into()]),
            approval_mode: Some(crate::ApprovalMode::Deny),
            workspace: None,
        };
        let mut buf = Vec::new();
        ciborium::into_writer(&overlay, &mut buf).unwrap();
        let back: SessionOverlay = ciborium::from_reader(&buf[..]).unwrap();
        assert_eq!(overlay, back);
    }

    #[test]
    fn selector_locality_and_serde_form() {
        assert!(ProviderSelector::LlamaCpp.is_local());
        assert!(ProviderSelector::MistralRs.is_local());
        assert!(!ProviderSelector::GenAi.is_local());
        assert!(!ProviderSelector::Mock.is_local());
        // The daemon-api gateway is networked, not local.
        assert!(!ProviderSelector::DaemonApi.is_local());
        // The networked selector serializes to the stable "genai" wire id.
        assert_eq!(
            serde_json::to_string(&ProviderSelector::GenAi).unwrap(),
            "\"genai\""
        );
    }

    #[test]
    fn daemon_api_selector_wire_id_round_trips() {
        // The daemon-api gateway selector serializes to the stable snake_case "daemon_api" wire id
        // and deserializes back (serde JSON + CBOR/ciborium — the two on-wire encodings).
        assert_eq!(
            serde_json::to_string(&ProviderSelector::DaemonApi).unwrap(),
            "\"daemon_api\""
        );
        assert_eq!(
            serde_json::from_str::<ProviderSelector>("\"daemon_api\"").unwrap(),
            ProviderSelector::DaemonApi
        );
        let mut buf = Vec::new();
        ciborium::into_writer(&ProviderSelector::DaemonApi, &mut buf).unwrap();
        let back: ProviderSelector = ciborium::from_reader(&buf[..]).unwrap();
        assert_eq!(back, ProviderSelector::DaemonApi);

        // And it round-trips inside a full ProfileSpec (the shape a GUI creates/edits).
        let spec = ProfileSpec {
            base_url: Some("https://api.daemon.ai/api/v1/".to_string()),
            ..ProfileSpec::new(
                "daemon",
                ProviderSelector::DaemonApi,
                "anthropic/claude-sonnet-4-5",
            )
        };
        let mut buf = Vec::new();
        ciborium::into_writer(&spec, &mut buf).unwrap();
        let back: ProfileSpec = ciborium::from_reader(&buf[..]).unwrap();
        assert_eq!(back, spec);
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
            let sel: ProviderSelector = serde_json::from_str(&format!("\"{legacy}\"")).unwrap();
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
        assert!(catalog
            .iter()
            .all(|m| m.provider == ProviderSelector::GenAi));
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

    #[test]
    fn engine_selector_wire_shapes_round_trip() {
        // The unit variant serializes as the bare "Core" string; the foreign arm as the nested
        // {"Foreign": {"agent": ...}} map — mirroring the CDDL `engine-selector` union exactly.
        assert_eq!(
            serde_json::to_string(&EngineSelector::Core).unwrap(),
            "\"Core\""
        );
        assert_eq!(
            serde_json::to_string(&EngineSelector::Foreign {
                agent: "gemini".into()
            })
            .unwrap(),
            "{\"Foreign\":{\"agent\":\"gemini\"}}"
        );
        // And a full ProfileSpec carrying the foreign selector round-trips through CBOR (the
        // on-wire encoding); the recipe never travels — only the catalog NAME does.
        let spec = ProfileSpec {
            engine: EngineSelector::Foreign {
                agent: "goose".into(),
            },
            ..ProfileSpec::new("foreign", ProviderSelector::Mock, "")
        };
        let mut buf = Vec::new();
        ciborium::into_writer(&spec, &mut buf).unwrap();
        let back: ProfileSpec = ciborium::from_reader(&buf[..]).unwrap();
        assert_eq!(back, spec);
    }

    #[test]
    fn foreign_backend_default_and_wire_shapes() {
        // The default arm is the safe-migration behavior: the agent's own backend, unsteered.
        assert_eq!(
            ForeignBackend::default(),
            ForeignBackend::AgentNative { model: None }
        );
        // Externally-tagged struct variants — the shape the CDDL `foreign-backend` union mirrors.
        assert_eq!(
            serde_json::to_string(&ForeignBackend::AgentNative {
                model: Some("mock-model-b".into())
            })
            .unwrap(),
            "{\"AgentNative\":{\"model\":\"mock-model-b\"}}"
        );
        assert_eq!(
            serde_json::to_string(&ForeignBackend::NodeProvider {
                provider: ProviderSelector::GenAi,
                model: "gpt-4o".into(),
                credential_ref: None,
            })
            .unwrap(),
            "{\"NodeProvider\":{\"provider\":\"genai\",\"model\":\"gpt-4o\",\"credential_ref\":null}}"
        );
        // Round-trips through CBOR (the on-wire encoding) inside a full ProfileSpec.
        let spec = ProfileSpec {
            engine: EngineSelector::Foreign {
                agent: "codex".into(),
            },
            foreign_backend: ForeignBackend::NodeProvider {
                provider: ProviderSelector::GenAi,
                model: "gpt-4o".into(),
                credential_ref: Some("openai".into()),
            },
            ..ProfileSpec::new("routed", ProviderSelector::Mock, "")
        };
        let mut buf = Vec::new();
        ciborium::into_writer(&spec, &mut buf).unwrap();
        let back: ProfileSpec = ciborium::from_reader(&buf[..]).unwrap();
        assert_eq!(back, spec);
    }

    #[test]
    fn pre_foreign_backend_encodings_decode_as_agent_native() {
        // A profile persisted before `foreign_backend` existed omits the key; it must decode with
        // the default `AgentNative { model: None }` (the additive-compat contract for the optional
        // CDDL field), preserving today's "agent's own backend" behavior.
        let legacy = serde_json::json!({
            "id": "old-foreign",
            "provider": "mock",
            "model": "",
            "engine": { "Foreign": { "agent": "codex" } }
        });
        let spec: ProfileSpec = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            spec.foreign_backend,
            ForeignBackend::AgentNative { model: None }
        );
    }

    #[test]
    fn pre_engine_profile_encodings_decode_as_core() {
        // A profile persisted/sent before the engine selector existed has no "engine" key; it must
        // decode with the Core default (the additive-compat contract for the optional CDDL field).
        let legacy = serde_json::json!({
            "id": "old",
            "provider": "genai",
            "model": "claude-opus-4-8"
        });
        let spec: ProfileSpec = serde_json::from_value(legacy).unwrap();
        assert_eq!(spec.engine, EngineSelector::Core);
    }
}
