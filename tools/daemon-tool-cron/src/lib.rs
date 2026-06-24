//! `daemon-tool-cron` — the agent veneer over the cron backing (I15; layout §4: tool surface).
//!
//! Exposes scheduling to the engine as a single `daemon_core::Tool` so the agent can create, list,
//! update, pause/resume, run-now, and remove its own scheduled jobs. It is a thin handle over the
//! shared [`CronOps`](daemon_host::CronOps) — the exact same validation + store path the operator
//! `cron_*` control ops use, so there is one job engine, not two.
//!
//! SAFETY (mirrors Hermes): the tool **refuses every action inside a cron-fired session**
//! (`cron_{id}_{ts}`), so a scheduled run cannot schedule more work (runaway self-replication). The
//! node also gates the tool out of the cron-run tool registry; this in-tool guard is defense in
//! depth that holds regardless of which profile resolves the run. Inputs are guarded: the prompt is
//! size-capped and control-char-scanned, and a `no_agent` `script` path must be workspace-relative
//! (no absolute paths, no `..` escape).

#![forbid(unsafe_code)]

use async_trait::async_trait;
use daemon_api::{CatchUpPolicy, CronSpec, OverlapPolicy};
use daemon_core::{Tool, ToolCall, ToolOutcome, ToolResult, TurnCx};
use daemon_host::CronOps;
use std::sync::Arc;

/// Max accepted prompt length (bytes) for a created/updated job (matches the cron context cap).
const MAX_PROMPT_BYTES: usize = 16_384;

/// The agent's handle onto the node's shared cron operations.
pub struct CronTool {
    ops: Arc<CronOps>,
}

impl CronTool {
    /// A cron tool over the node's shared [`CronOps`].
    pub fn new(ops: Arc<CronOps>) -> Self {
        Self { ops }
    }

    fn ok(call: &ToolCall, content: String) -> ToolOutcome {
        ToolOutcome {
            result: ToolResult {
                call_id: call.call_id.clone(),
                ok: true,
                content,
            },
            effects: Vec::new(),
            detail: None,
            untrusted: false,
        }
    }

    fn err(call: &ToolCall, content: String) -> ToolOutcome {
        ToolOutcome {
            result: ToolResult {
                call_id: call.call_id.clone(),
                ok: false,
                content,
            },
            effects: Vec::new(),
            detail: None,
            untrusted: false,
        }
    }
}

/// Whether a session id is a cron-fired run (`cron_{id}_{ts}`). The cron tool refuses inside one.
pub fn is_cron_session(session_id: &str) -> bool {
    session_id.starts_with("cron_")
}

/// Validate a `no_agent` script path: workspace-relative only (no absolute path, no `..` escape, no
/// leading separator). Returns the reason on rejection.
fn check_script(script: &str) -> Result<(), String> {
    let s = script.trim();
    if s.is_empty() {
        return Err("script path is empty".into());
    }
    if s.starts_with('/') || s.starts_with('\\') {
        return Err("script path must be workspace-relative (no leading separator)".into());
    }
    if s.split(['/', '\\']).any(|seg| seg == "..") {
        return Err("script path must not contain `..`".into());
    }
    Ok(())
}

/// Light prompt guard: reject control characters (NUL etc.) and over-long prompts (prompt-injection
/// surface reduction; the heavy lifting is the run-time tool/approval constraint on the cron run).
fn check_prompt(prompt: &str) -> Result<(), String> {
    if prompt.len() > MAX_PROMPT_BYTES {
        return Err(format!("prompt exceeds {MAX_PROMPT_BYTES} bytes"));
    }
    if prompt.chars().any(|c| c.is_control() && c != '\n' && c != '\t' && c != '\r') {
        return Err("prompt contains control characters".into());
    }
    Ok(())
}

/// Parse the `overlap` policy string (default [`OverlapPolicy::Skip`]).
fn parse_overlap(s: &str) -> Result<OverlapPolicy, String> {
    match s.trim().to_lowercase().as_str() {
        "" | "skip" => Ok(OverlapPolicy::Skip),
        "allow" => Ok(OverlapPolicy::Allow),
        "queue" => Ok(OverlapPolicy::Queue),
        other => Err(format!("unknown overlap policy {other:?} (skip|allow|queue)")),
    }
}

/// Parse the `catch_up` policy string (default [`CatchUpPolicy::Grace`]).
fn parse_catch_up(s: &str) -> Result<CatchUpPolicy, String> {
    match s.trim().to_lowercase().as_str() {
        "" | "grace" => Ok(CatchUpPolicy::Grace),
        "skip" => Ok(CatchUpPolicy::Skip),
        "always" => Ok(CatchUpPolicy::Always),
        other => Err(format!("unknown catch_up policy {other:?} (grace|skip|always)")),
    }
}

/// Build a [`CronSpec`] from the JSON arg map (create/update). `name` + `schedule` are required.
fn spec_from(map: &serde_json::Map<String, serde_json::Value>) -> Result<CronSpec, String> {
    let name = map
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .ok_or("missing `name`")?
        .to_owned();
    let schedule = map
        .get("schedule")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .ok_or("missing `schedule`")?
        .to_owned();
    let prompt = map
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_owned();
    check_prompt(&prompt)?;
    let str_field = |k: &str| {
        map.get(k)
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .filter(|s| !s.is_empty())
    };
    let str_list = |k: &str| {
        map.get(k).and_then(|v| v.as_array()).map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(str::to_owned))
                .collect::<Vec<_>>()
        })
    };
    let script = str_field("script");
    if let Some(s) = &script {
        check_script(s)?;
    }
    let no_agent = map.get("no_agent").and_then(|v| v.as_bool()).unwrap_or(false);
    let overlap = match map.get("overlap").and_then(|v| v.as_str()) {
        Some(s) => parse_overlap(s)?,
        None => OverlapPolicy::default(),
    };
    let catch_up = match map.get("catch_up").and_then(|v| v.as_str()) {
        Some(s) => parse_catch_up(s)?,
        None => CatchUpPolicy::default(),
    };
    Ok(CronSpec {
        name,
        schedule,
        target: str_field("target"),
        payload: prompt.into_bytes(),
        enabled: map.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true),
        timezone: str_field("timezone"),
        repeat: map.get("repeat").and_then(|v| v.as_u64()).map(|n| n as u32),
        jitter_secs: map.get("jitter_secs").and_then(|v| v.as_u64()).map(|n| n as u32),
        overlap,
        catch_up,
        script,
        no_agent,
        context_from: str_list("context_from").unwrap_or_default(),
        deliver: str_field("deliver"),
        enabled_toolsets: str_list("enabled_toolsets"),
        workdir: str_field("workdir"),
        model: str_field("model"),
        provider: str_field("provider"),
        skills: str_list("skills").unwrap_or_default(),
        // The creating origin is stamped from the calling session at create time (the tool path),
        // not taken from args — see `run`'s `create` arm.
        origin: None,
    })
}

#[async_trait]
impl Tool for CronTool {
    fn name(&self) -> &str {
        "cron"
    }

    fn schema(&self) -> &str {
        r#"{"type":"object","properties":{"action":{"type":"string","enum":["create","list","update","pause","resume","run","remove"],"description":"the operation; defaults to list"},"id":{"type":"string","description":"the job id (required for update/pause/resume/run/remove)"},"name":{"type":"string","description":"a human name for the job (required for create/update)"},"schedule":{"type":"string","description":"cron expr (e.g. \"0 9 * * *\"), @every <dur>, or an ISO timestamp (required for create/update)"},"prompt":{"type":"string","description":"the agent instruction the fired run executes (a cron run has no chat memory, so be self-contained)"},"timezone":{"type":"string","description":"IANA timezone for cron evaluation (e.g. \"Europe/Berlin\")"},"repeat":{"type":"integer","description":"auto-delete after this many fires; omit for unlimited"},"jitter_secs":{"type":"integer","description":"random 0..=N second spread applied to each fire"},"enabled":{"type":"boolean","description":"create armed (true, default) or paused (false)"},"overlap":{"type":"string","enum":["skip","allow","queue"],"description":"behavior when a fire overlaps a still-running run (default skip)"},"catch_up":{"type":"string","enum":["grace","skip","always"],"description":"missed-fire catch-up policy when overdue (default grace)"},"target":{"type":"string","description":"the profile/agent the run is bound to"},"deliver":{"type":"string","description":"delivery routing: \"origin\", \"all\", \"<transport>:<chat>\", or omit for store-only"},"script":{"type":"string","description":"a node-scripts-relative path to run before/instead of the agent (no absolute paths, no ..)"},"no_agent":{"type":"boolean","description":"run script only, no LLM turn (requires script)"},"context_from":{"type":"array","items":{"type":"string"},"description":"job ids whose latest output is injected into this job's prompt (output chaining)"},"enabled_toolsets":{"type":"array","items":{"type":"string"},"description":"restrict the run's toolset to these tool names"},"skills":{"type":"array","items":{"type":"string"},"description":"skill names to preload (their content is injected ahead of the prompt) before the run"},"workdir":{"type":"string","description":"absolute working directory the run is bound to"},"model":{"type":"string","description":"per-job model override"},"provider":{"type":"string","description":"per-job provider override"}},"required":["action"]}"#
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        // SAFETY: a scheduled run may not manage cron (no runaway self-scheduling).
        if is_cron_session(cx.session_id.as_str()) {
            return Self::err(
                call,
                "the cron tool is unavailable inside a scheduled run".into(),
            );
        }
        let map = match serde_json::from_str::<serde_json::Value>(&call.args) {
            Ok(serde_json::Value::Object(map)) => map,
            _ => return Self::err(call, "cron tool args must be a JSON object".into()),
        };
        let action = map.get("action").and_then(|v| v.as_str()).unwrap_or("list");
        let id_arg = || {
            map.get("id")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .filter(|s| !s.is_empty())
        };
        match action {
            "create" => {
                let mut spec = match spec_from(&map) {
                    Ok(s) => s,
                    Err(e) => return Self::err(call, format!("invalid cron spec: {e}")),
                };
                // Stamp the creating origin (wire v17) so `deliver = "origin"` can route a run's
                // result back to the chat this job was created from. Resolved from the calling
                // session's routing pin; `None` (store-only) for an unpinned/deterministic session.
                spec.origin = self.ops.origin_for_session(&cx.session_id).await;
                match self.ops.create(spec).await {
                    Ok(id) => Self::ok(call, format!("created cron job {id}")),
                    Err(e) => Self::err(call, format!("create failed: {e}")),
                }
            }
            "list" => {
                let jobs = self.ops.list().await;
                let mut lines = vec![format!("{} cron job(s):", jobs.len())];
                for j in jobs {
                    lines.push(format!(
                        "- {} \"{}\" [{}]{} next={}",
                        j.id,
                        j.spec.name,
                        j.spec.schedule,
                        if j.paused { " (paused)" } else { "" },
                        j.next_fire_unix
                            .map(|t| t.to_string())
                            .unwrap_or_else(|| "-".into()),
                    ));
                }
                Self::ok(call, lines.join("\n"))
            }
            "update" => {
                let Some(id) = id_arg() else {
                    return Self::err(call, "update requires `id`".into());
                };
                let spec = match spec_from(&map) {
                    Ok(s) => s,
                    Err(e) => return Self::err(call, format!("invalid cron spec: {e}")),
                };
                match self.ops.update(id.clone(), spec).await {
                    Ok(()) => Self::ok(call, format!("updated cron job {id}")),
                    Err(e) => Self::err(call, format!("update failed: {e}")),
                }
            }
            "pause" | "resume" => {
                let Some(id) = id_arg() else {
                    return Self::err(call, format!("{action} requires `id`"));
                };
                let paused = action == "pause";
                match self.ops.pause(id.clone(), paused).await {
                    Ok(()) => Self::ok(call, format!("{action}d cron job {id}")),
                    Err(e) => Self::err(call, format!("{action} failed: {e}")),
                }
            }
            "run" => {
                let Some(id) = id_arg() else {
                    return Self::err(call, "run requires `id`".into());
                };
                match self.ops.trigger(id.clone()).await {
                    Ok(()) => Self::ok(call, format!("triggered cron job {id}")),
                    Err(e) => Self::err(call, format!("run failed: {e}")),
                }
            }
            "remove" => {
                let Some(id) = id_arg() else {
                    return Self::err(call, "remove requires `id`".into());
                };
                match self.ops.delete(id.clone()).await {
                    Ok(()) => Self::ok(call, format!("removed cron job {id}")),
                    Err(e) => Self::err(call, format!("remove failed: {e}")),
                }
            }
            other => Self::err(call, format!("unknown cron action: {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_store::InMemoryStore;

    fn obj(json: &str) -> serde_json::Map<String, serde_json::Value> {
        match serde_json::from_str::<serde_json::Value>(json).unwrap() {
            serde_json::Value::Object(m) => m,
            _ => unreachable!("expected a JSON object"),
        }
    }

    #[test]
    fn cron_session_detection() {
        assert!(is_cron_session("cron_abc_123"));
        assert!(!is_cron_session("s1"));
        assert!(!is_cron_session("s1/c1"));
    }

    #[test]
    fn script_sandbox_rejects_escapes() {
        assert!(check_script("scripts/ok.sh").is_ok());
        assert!(check_script("/etc/passwd").is_err());
        assert!(check_script("\\abs").is_err());
        assert!(check_script("../../etc/passwd").is_err());
        assert!(check_script("a/../../b").is_err());
        assert!(check_script("   ").is_err());
    }

    #[test]
    fn prompt_guard_rejects_control_and_oversize() {
        assert!(check_prompt("a normal\nmulti-line\tprompt").is_ok());
        assert!(check_prompt("bad\0null").is_err());
        assert!(check_prompt(&"x".repeat(MAX_PROMPT_BYTES + 1)).is_err());
    }

    #[test]
    fn spec_requires_name_and_schedule() {
        assert!(spec_from(&obj(r#"{"name":"x"}"#)).is_err());
        assert!(spec_from(&obj(r#"{"schedule":"0 9 * * *"}"#)).is_err());
        let spec =
            spec_from(&obj(r#"{"name":"x","schedule":"0 9 * * *","prompt":"hi"}"#)).unwrap();
        assert_eq!(spec.name, "x");
        assert_eq!(spec.payload, b"hi");
    }

    #[test]
    fn spec_maps_full_arg_set() {
        let spec = spec_from(&obj(
            r#"{"name":"digest","schedule":"@every 1h","prompt":"go","timezone":"UTC",
                "repeat":3,"jitter_secs":30,"no_agent":true,"script":"scripts/run.sh",
                "context_from":["job-a"],"deliver":"origin","enabled_toolsets":["fs"],
                "overlap":"queue","catch_up":"always","target":"opus","workdir":"/srv/p",
                "model":"gpt-5","provider":"genai","skills":["briefing","calendar"]}"#,
        ))
        .unwrap();
        assert_eq!(spec.timezone.as_deref(), Some("UTC"));
        assert_eq!(spec.repeat, Some(3));
        assert_eq!(spec.jitter_secs, Some(30));
        assert!(spec.no_agent);
        assert_eq!(spec.script.as_deref(), Some("scripts/run.sh"));
        assert_eq!(spec.context_from, vec!["job-a".to_owned()]);
        assert_eq!(spec.enabled_toolsets, Some(vec!["fs".to_owned()]));
        assert_eq!(spec.overlap, OverlapPolicy::Queue);
        assert_eq!(spec.catch_up, CatchUpPolicy::Always);
        assert_eq!(spec.target.as_deref(), Some("opus"));
        assert_eq!(spec.workdir.as_deref(), Some("/srv/p"));
        assert_eq!(spec.model.as_deref(), Some("gpt-5"));
        assert_eq!(spec.provider.as_deref(), Some("genai"));
        assert_eq!(spec.skills, vec!["briefing".to_owned(), "calendar".to_owned()]);
    }

    #[test]
    fn schema_is_valid_json_and_advertises_full_surface() {
        let tool = CronTool::new(Arc::new(CronOps::new(Arc::new(
            daemon_store::InMemoryStore::new(),
        ))));
        let schema: serde_json::Value =
            serde_json::from_str(tool.schema()).expect("schema is valid JSON");
        let props = schema["properties"].as_object().expect("properties object");
        // Every CronSpec-backed field the agent can set is discoverable in the schema.
        for field in [
            "action", "id", "name", "schedule", "prompt", "timezone", "repeat", "jitter_secs",
            "enabled", "overlap", "catch_up", "target", "deliver", "script", "no_agent",
            "context_from", "enabled_toolsets", "skills", "workdir", "model", "provider",
        ] {
            assert!(props.contains_key(field), "schema must advertise `{field}`");
        }
    }

    #[test]
    fn spec_defaults_policies_and_rejects_bad_ones() {
        let spec = spec_from(&obj(r#"{"name":"x","schedule":"0 9 * * *"}"#)).unwrap();
        assert_eq!(spec.overlap, OverlapPolicy::Skip);
        assert_eq!(spec.catch_up, CatchUpPolicy::Grace);
        assert!(spec.skills.is_empty());
        assert!(spec_from(&obj(
            r#"{"name":"x","schedule":"0 9 * * *","overlap":"nonsense"}"#
        ))
        .is_err());
        assert!(spec_from(&obj(
            r#"{"name":"x","schedule":"0 9 * * *","catch_up":"nonsense"}"#
        ))
        .is_err());
    }

    #[test]
    fn spec_rejects_escaping_script() {
        assert!(spec_from(&obj(
            r#"{"name":"x","schedule":"0 9 * * *","script":"/etc/passwd"}"#
        ))
        .is_err());
    }

    #[tokio::test]
    async fn ops_create_and_list_round_trip() {
        let ops = CronOps::new(Arc::new(InMemoryStore::new()));
        let spec = spec_from(&obj(
            r#"{"name":"digest","schedule":"0 9 * * *","prompt":"go"}"#,
        ))
        .unwrap();
        let id = ops.create(spec).await.unwrap();
        let jobs = ops.list().await;
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].id, id);
        assert_eq!(jobs[0].spec.name, "digest");
    }
}
