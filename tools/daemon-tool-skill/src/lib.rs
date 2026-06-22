//! `daemon-tool-skill` — the model-facing skills tools (`daemon_core::Tool`) over a
//! [`daemon_skills::SkillStore`]: the progressive-disclosure level-2/3 reads and the CRUD surface.
//!
//! Three tools mirror hermes' skills toolset:
//! - `skills_list` — the compact roster (name + description + category) for ad-hoc discovery beyond
//!   the always-on index;
//! - `skill_view(name, file_path?)` — load a skill's full `SKILL.md` body, or a linked support file;
//! - `skill_manage(action, …)` — `create`/`edit`/`patch`/`delete`/`write_file`/`remove_file` on the
//!   local skills dir.
//!
//! All three share one `Arc<SkillStore>`; writes invalidate the store's memoized index (so the next
//! turn's stable-tier block reflects the change while staying byte-stable in between). Skill bodies
//! are locally-authored procedural memory, so reads are returned **trusted** (not §12-fenced).

#![forbid(unsafe_code)]

use std::sync::Arc;

use async_trait::async_trait;
use daemon_core::{Tool, ToolCall, ToolConcurrency, ToolOutcome, TurnCx};
use daemon_protocol::ToolDetail;
use daemon_skills::SkillStore;
use serde::Deserialize;

/// The canonical skill-tool names (used for tool-allowlist gating + the index-injection check).
pub const SKILL_TOOL_NAMES: [&str; 3] = ["skills_list", "skill_view", "skill_manage"];

/// All three skills tools over a shared store, ready to register on a [`daemon_core::ToolRegistry`].
pub fn skill_tools(store: Arc<SkillStore>) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(SkillsListTool::new(store.clone())),
        Arc::new(SkillViewTool::new(store.clone())),
        Arc::new(SkillManageTool::new(store)),
    ]
}

// ---------------------------------------------------------------------------
// skills_list
// ---------------------------------------------------------------------------

const SKILLS_LIST_SCHEMA: &str = r#"{"type":"object","properties":{}}"#;

/// `skills_list`: the compact roster of all discovered skills.
pub struct SkillsListTool {
    store: Arc<SkillStore>,
}

impl SkillsListTool {
    /// A list tool over `store`.
    pub fn new(store: Arc<SkillStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for SkillsListTool {
    fn name(&self) -> &str {
        "skills_list"
    }
    fn schema(&self) -> &str {
        SKILLS_LIST_SCHEMA
    }
    async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> ToolOutcome {
        let items = self.store.list();
        let mut text = if items.is_empty() {
            "No skills available.".to_string()
        } else {
            let mut s = format!("{} skill(s):\n", items.len());
            for it in &items {
                let cat = it.category.as_deref().unwrap_or("general");
                s.push_str(&format!("- {} [{}]: {}\n", it.name, cat, it.description));
            }
            s
        };
        text.truncate(text.trim_end().len());
        let body = serde_json::to_vec(
            &items
                .iter()
                .map(|i| {
                    serde_json::json!({
                        "name": i.name,
                        "description": i.description,
                        "category": i.category,
                    })
                })
                .collect::<Vec<_>>(),
        )
        .unwrap_or_default();
        ToolOutcome::text(call.call_id.clone(), true, text).with_detail(ToolDetail {
            kind: "skills_list".to_string(),
            body,
        })
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }
}

// ---------------------------------------------------------------------------
// skill_view
// ---------------------------------------------------------------------------

const SKILL_VIEW_SCHEMA: &str = r#"{
  "type": "object",
  "required": ["name"],
  "properties": {
    "name": {"type": "string", "description": "The skill (bundle) name to load."},
    "file_path": {"type": "string", "description": "Optional support file to load instead of SKILL.md (e.g. references/api.md)."}
  }
}"#;

/// `skill_view`: load a skill's full body (or a linked support file) on demand.
pub struct SkillViewTool {
    store: Arc<SkillStore>,
}

impl SkillViewTool {
    /// A view tool over `store`.
    pub fn new(store: Arc<SkillStore>) -> Self {
        Self { store }
    }
}

#[derive(Deserialize)]
struct ViewArgs {
    name: String,
    #[serde(default)]
    file_path: Option<String>,
}

#[async_trait]
impl Tool for SkillViewTool {
    fn name(&self) -> &str {
        "skill_view"
    }
    fn schema(&self) -> &str {
        SKILL_VIEW_SCHEMA
    }
    async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> ToolOutcome {
        let args: ViewArgs = match serde_json::from_str(&call.args) {
            Ok(a) => a,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("skill_view: invalid arguments: {e}"),
                )
            }
        };
        match self.store.view(&args.name, args.file_path.as_deref()) {
            Ok(body) => ToolOutcome::text(call.call_id.clone(), true, body),
            Err(e) => ToolOutcome::text(call.call_id.clone(), false, format!("skill_view: {e}")),
        }
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Parallel
    }
}

// ---------------------------------------------------------------------------
// skill_manage
// ---------------------------------------------------------------------------

const SKILL_MANAGE_SCHEMA: &str = r#"{
  "type": "object",
  "required": ["action", "name"],
  "properties": {
    "action": {"type": "string", "enum": ["create", "edit", "patch", "delete", "write_file", "remove_file"]},
    "name": {"type": "string", "description": "The skill (bundle) name."},
    "content": {"type": "string", "description": "Full SKILL.md content (create/edit)."},
    "category": {"type": "string", "description": "Category for a new skill (create)."},
    "old_string": {"type": "string", "description": "Text to find (patch)."},
    "new_string": {"type": "string", "description": "Replacement text (patch)."},
    "file_path": {"type": "string", "description": "Support file path (patch/write_file/remove_file)."},
    "file_content": {"type": "string", "description": "Support file contents (write_file)."},
    "replace_all": {"type": "boolean", "description": "Replace all matches (patch); default false."}
  }
}"#;

/// `skill_manage`: CRUD on the local skills dir (create/edit/patch/delete/write_file/remove_file).
pub struct SkillManageTool {
    store: Arc<SkillStore>,
}

impl SkillManageTool {
    /// A manage tool over `store`.
    pub fn new(store: Arc<SkillStore>) -> Self {
        Self { store }
    }
}

#[derive(Deserialize)]
struct ManageArgs {
    action: String,
    name: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    old_string: Option<String>,
    #[serde(default)]
    new_string: Option<String>,
    #[serde(default)]
    file_path: Option<String>,
    #[serde(default)]
    file_content: Option<String>,
    #[serde(default)]
    replace_all: bool,
}

impl SkillManageTool {
    fn dispatch(&self, a: &ManageArgs) -> Result<String, String> {
        let missing = |f: &str| format!("skill_manage {}: `{f}` is required", a.action);
        match a.action.as_str() {
            "create" => {
                let content = a.content.as_deref().ok_or_else(|| missing("content"))?;
                let path = self
                    .store
                    .create(&a.name, content, a.category.as_deref())
                    .map_err(|e| e.to_string())?;
                Ok(format!("created skill `{}` at {}", a.name, path.display()))
            }
            "edit" => {
                let content = a.content.as_deref().ok_or_else(|| missing("content"))?;
                self.store.edit(&a.name, content).map_err(|e| e.to_string())?;
                Ok(format!("edited skill `{}`", a.name))
            }
            "patch" => {
                let old = a.old_string.as_deref().ok_or_else(|| missing("old_string"))?;
                let new = a.new_string.as_deref().ok_or_else(|| missing("new_string"))?;
                self.store
                    .patch(&a.name, old, new, a.file_path.as_deref(), a.replace_all)
                    .map_err(|e| e.to_string())?;
                Ok(format!("patched skill `{}`", a.name))
            }
            "delete" => {
                self.store.delete(&a.name).map_err(|e| e.to_string())?;
                Ok(format!("deleted skill `{}`", a.name))
            }
            "write_file" => {
                let path = a.file_path.as_deref().ok_or_else(|| missing("file_path"))?;
                let content = a.file_content.as_deref().ok_or_else(|| missing("file_content"))?;
                self.store
                    .write_file(&a.name, path, content)
                    .map_err(|e| e.to_string())?;
                Ok(format!("wrote `{}` to skill `{}`", path, a.name))
            }
            "remove_file" => {
                let path = a.file_path.as_deref().ok_or_else(|| missing("file_path"))?;
                self.store
                    .remove_file(&a.name, path)
                    .map_err(|e| e.to_string())?;
                Ok(format!("removed `{}` from skill `{}`", path, a.name))
            }
            other => Err(format!("skill_manage: unknown action `{other}`")),
        }
    }
}

#[async_trait]
impl Tool for SkillManageTool {
    fn name(&self) -> &str {
        "skill_manage"
    }
    fn schema(&self) -> &str {
        SKILL_MANAGE_SCHEMA
    }
    async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> ToolOutcome {
        let args: ManageArgs = match serde_json::from_str(&call.args) {
            Ok(a) => a,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("skill_manage: invalid arguments: {e}"),
                )
            }
        };
        match self.dispatch(&args) {
            Ok(msg) => ToolOutcome::text(call.call_id.clone(), true, msg),
            Err(e) => ToolOutcome::text(call.call_id.clone(), false, e),
        }
    }
}

#[cfg(test)]
mod tests;
