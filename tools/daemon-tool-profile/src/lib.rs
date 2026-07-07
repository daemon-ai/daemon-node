// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-tool-profile` — the agent veneer over the shared profile surface (layout §4: tool surface).
//!
//! Exposes profile authoring to the engine as a single `daemon_core::Tool` (`profile_manage`) so an
//! orchestrator-capable agent can compose reusable agent profiles from existing building blocks
//! (registered foreign agents + providers/models), persist them, and later spawn from them via
//! `orchestrate spawn { source: Profile(id) }`. It is a thin handle over the shared
//! [`ProfileOps`](daemon_host::ProfileOps) — the **same** validation (`validate_engine` +
//! `validate_inference`) + store + revision path the operator `profile_create` op uses (one engine,
//! not two) — mirroring how the `cron` tool wraps `CronOps` and `skill_manage` wraps `SkillStore`.
//!
//! PROVENANCE + NAMESPACE: every write records `Author::Agent("profile_manage")` on the revision log
//! (exactly as `SkillStore.default_author` records agent-authored skill edits), and a created
//! profile is namespaced `agent/{session}/{name}` — so agent-authored profiles are self-evident and
//! never clobber operator profiles.
//!
//! SUBTREE SCOPING: `edit`/`delete` are authorized only when the target profile was authored within
//! the caller's OWN subtree — the target's authoring session is parsed from its `agent/{session}/…`
//! id and gated by the shared [`owns_subtree`](daemon_store::owns_subtree) check (reflexive for the
//! caller's own profiles). An agent can never touch an ancestor's, a sibling's, or an operator's
//! profile.

#![forbid(unsafe_code)]

use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::{EngineSelector, ForeignBackend, ProfileSpec, ProviderSelector};
use daemon_common::{Author, SessionId};
use daemon_core::{Tool, ToolCall, ToolOutcome, TurnCx};
use daemon_host::ProfileOps;
use daemon_store::SessionStore;
use serde::Deserialize;

/// The canonical tool name (used for tool-allowlist gating).
pub const PROFILE_TOOL_NAME: &str = "profile_manage";

/// Max accepted profile-name length (bytes). A name keys the `agent/{session}/{name}` id + on-disk
/// file, so it is bounded and sanitized (no separators / control chars).
const MAX_NAME_LEN: usize = 128;

/// The agent's handle onto the node's shared [`ProfileOps`], plus the durable session graph the
/// subtree-authorization check walks.
pub struct ProfileManageTool {
    ops: Arc<ProfileOps>,
    store: Arc<dyn SessionStore>,
}

/// The `profile_manage` tool-call arguments (a `ProfileSpec` shape + the action verb). Optional spec
/// fields are applied on top of a new (create) or the existing (edit) profile; `id` is derived by the
/// tool on create (never taken from args) and names the target on edit/delete.
#[derive(Debug, Default, Deserialize)]
struct ManageArgs {
    action: String,
    /// The short profile name (create): keys the `agent/{session}/{name}` id.
    #[serde(default)]
    name: Option<String>,
    /// The full profile id (edit/delete): `agent/{session}/{name}`.
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    provider: Option<ProviderSelector>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    tool_allowlist: Option<Vec<String>>,
    #[serde(default)]
    engine: Option<EngineSelector>,
    #[serde(default)]
    foreign_backend: Option<ForeignBackend>,
    #[serde(default)]
    credential_ref: Option<String>,
}

/// Compose the namespaced id for an agent-authored profile: `agent/{session}/{name}`. The authoring
/// session is embedded so the profile is self-evident and the subtree check can recover it later.
fn agent_profile_id(session: &SessionId, name: &str) -> String {
    format!("agent/{}/{}", session.as_str(), name)
}

/// Recover the authoring session id from an `agent/{session}/{name}` profile id. `None` for an
/// operator profile (no `agent/` prefix) or a malformed id — either denies management.
fn parse_authoring_session(id: &str) -> Option<SessionId> {
    let rest = id.strip_prefix("agent/")?;
    // The session id may itself embed lineage (`s1/c2`), so the NAME is the final segment and the
    // session is everything before it.
    let (session, name) = rest.rsplit_once('/')?;
    if session.is_empty() || name.is_empty() {
        return None;
    }
    Some(SessionId::new(session))
}

/// Validate a create name: non-empty, bounded, and free of path separators / control characters (so
/// it keys a well-formed id + on-disk file and cannot escape the `agent/{session}/` namespace).
fn check_name(name: &str) -> Result<(), String> {
    let n = name.trim();
    if n.is_empty() {
        return Err("`name` is empty".into());
    }
    if n.len() > MAX_NAME_LEN {
        return Err(format!("`name` exceeds {MAX_NAME_LEN} bytes"));
    }
    if n.contains('/') || n.contains('\\') {
        return Err("`name` must not contain a path separator".into());
    }
    if n.chars().any(|c| c.is_control()) {
        return Err("`name` must not contain control characters".into());
    }
    Ok(())
}

/// Apply the provided optional spec fields onto `spec` (create builds from a fresh spec; edit applies
/// on top of the fetched one, so an omitted field is left unchanged).
fn apply_fields(spec: &mut ProfileSpec, args: &ManageArgs) {
    if let Some(provider) = args.provider {
        spec.provider = provider;
    }
    if let Some(model) = &args.model {
        spec.model = model.clone();
    }
    if let Some(base_url) = &args.base_url {
        spec.base_url = Some(base_url.clone());
    }
    if let Some(system_prompt) = &args.system_prompt {
        spec.system_prompt = system_prompt.clone();
    }
    if let Some(tool_allowlist) = &args.tool_allowlist {
        spec.tool_allowlist = Some(tool_allowlist.clone());
    }
    if let Some(engine) = &args.engine {
        spec.engine = engine.clone();
    }
    if let Some(foreign_backend) = &args.foreign_backend {
        spec.foreign_backend = foreign_backend.clone();
    }
    if let Some(credential_ref) = &args.credential_ref {
        spec.credential_ref = Some(credential_ref.clone());
    }
}

impl ProfileManageTool {
    /// A `profile_manage` tool over the node's shared [`ProfileOps`] + the durable session `store`
    /// (the subtree-authorization graph).
    pub fn new(ops: Arc<ProfileOps>, store: Arc<dyn SessionStore>) -> Self {
        Self { ops, store }
    }

    /// The revision author every agent write records (mirrors `SkillStore.default_author`).
    fn author() -> Author {
        Author::Agent(PROFILE_TOOL_NAME.to_string())
    }

    /// Whether `caller` may manage the profile `id`: reflexively for its OWN profiles (authoring
    /// session == caller) and for any profile authored within the caller's subtree (the shared
    /// `owns_subtree` check). An operator profile / malformed id / ancestor / sibling is denied.
    async fn caller_owns(&self, caller: &SessionId, id: &str) -> bool {
        let Some(authoring) = parse_authoring_session(id) else {
            return false;
        };
        caller.as_str() == authoring.as_str()
            || daemon_store::owns_subtree(self.store.as_ref(), caller, &authoring).await
    }

    /// The action dispatch, factored off [`Tool::run`] so it is unit-testable without a `TurnCx`:
    /// `caller` is the calling session (`cx.session_id`).
    async fn dispatch(&self, caller: &SessionId, args: ManageArgs) -> Result<String, String> {
        match args.action.as_str() {
            "create" => {
                let name = args
                    .name
                    .as_deref()
                    .ok_or("profile_manage create: `name` is required")?;
                check_name(name)?;
                let id = agent_profile_id(caller, name.trim());
                let mut spec = ProfileSpec::new(&id, ProviderSelector::default(), String::new());
                apply_fields(&mut spec, &args);
                self.ops
                    .create(spec, Self::author())
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(format!("created profile `{id}`"))
            }
            "edit" => {
                let id = args
                    .id
                    .as_deref()
                    .ok_or("profile_manage edit: `id` is required")?;
                if !self.caller_owns(caller, id).await {
                    return Err(format!(
                        "profile_manage edit denied: `{id}` was not authored within this session's subtree"
                    ));
                }
                let mut spec = self
                    .ops
                    .get(id)
                    .map_err(|e| e.to_string())?
                    .ok_or_else(|| format!("profile_manage edit: unknown profile `{id}`"))?;
                apply_fields(&mut spec, &args);
                self.ops
                    .update(spec, Self::author())
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(format!("edited profile `{id}`"))
            }
            "delete" => {
                let id = args
                    .id
                    .as_deref()
                    .ok_or("profile_manage delete: `id` is required")?;
                if !self.caller_owns(caller, id).await {
                    return Err(format!(
                        "profile_manage delete denied: `{id}` was not authored within this session's subtree"
                    ));
                }
                if self.ops.get(id).map_err(|e| e.to_string())?.is_none() {
                    return Err(format!("profile_manage delete: unknown profile `{id}`"));
                }
                self.ops.delete(id).map_err(|e| e.to_string())?;
                Ok(format!("deleted profile `{id}`"))
            }
            other => Err(format!("profile_manage: unknown action `{other}`")),
        }
    }
}

#[async_trait]
impl Tool for ProfileManageTool {
    fn name(&self) -> &str {
        PROFILE_TOOL_NAME
    }

    fn schema(&self) -> &str {
        r#"{"type":"object","required":["action"],"properties":{"action":{"type":"string","enum":["create","edit","delete"],"description":"author a reusable profile from existing building blocks"},"name":{"type":"string","description":"create: the short profile name (keys the agent/{session}/{name} id)"},"id":{"type":"string","description":"edit/delete: the full profile id (agent/{session}/{name}); only profiles authored within this session's subtree may be managed"},"provider":{"type":"string","description":"the model provider selector (Core engine)"},"model":{"type":"string","description":"the model id (Core engine)"},"base_url":{"type":"string","description":"optional provider API base-URL override"},"system_prompt":{"type":"string","description":"the profile's persona / system prompt"},"tool_allowlist":{"type":"array","items":{"type":"string"},"description":"the tools this profile's engine may use (an allowlist; omit for the full node toolset)"},"engine":{"description":"\"Core\" (default) or {\"Foreign\":{\"agent\":\"name\"}} referencing a registered agent"},"foreign_backend":{"description":"for a Foreign engine: how it sources its model backend (AgentNative or NodeProvider)"},"credential_ref":{"type":"string","description":"the credential profile this engine acquires from (defaults to the id)"}}}"#
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let args: ManageArgs = match serde_json::from_str(&call.args) {
            Ok(a) => a,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("profile_manage: invalid arguments: {e}"),
                )
            }
        };
        match self.dispatch(&cx.session_id, args).await {
            Ok(msg) => ToolOutcome::text(call.call_id.clone(), true, msg),
            Err(e) => ToolOutcome::text(call.call_id.clone(), false, e),
        }
    }
}

#[cfg(test)]
mod tests;
