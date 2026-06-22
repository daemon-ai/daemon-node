//! `daemon-skills` — the filesystem skills subsystem (a Rust port of hermes' skills).
//!
//! A *skill* is a directory bundle under `<profile_home>/skills/<category>/<name>/` whose `SKILL.md`
//! carries a YAML frontmatter (`name`, `description`, `version`, `platforms`, `metadata.hermes.*`)
//! plus a markdown body, optionally beside `references/`, `templates/`, `scripts/`, `assets/`. The
//! subsystem implements **progressive disclosure** (hermes `prompt_builder.build_skills_system_prompt`):
//!
//! 1. a compact, category-grouped *index* (name + ≤60-char description) lives in the stable system
//!    prompt tier — cheap and cache-stable;
//! 2. the full body loads on demand via `skill_view(name)`;
//! 3. linked files load via `skill_view(name, file_path)`.
//!
//! It also exposes CRUD (`create`/`edit`/`patch`/`delete`/`write_file`/`remove_file`) writing to the
//! local skills dir, and a one-shot bundled→user seed. The model-facing tool surface lives in the
//! separate `daemon-tool-skill` crate; this crate is pure discovery/parse/index/CRUD logic.
//!
//! Cache discipline (the prompt-caching invariant): the rendered index is memoized and invalidated
//! only on a write through this store, so the system prompt stays byte-stable across a conversation.

#![forbid(unsafe_code)]

use daemon_common::{Author, RevisionKind, RevisionLog, SkillBundle};
use daemon_core::StablePromptSource;
use include_dir::{include_dir, Dir};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// The curated, tool-agnostic skills daemon ships with, embedded into the binary at compile time and
/// materialized into the profile's skills dir on first run (see [`SkillStore::seed_bundled`]) — the
/// Rust analogue of hermes' wheel-data bundle synced by `tools/skills_sync.py`. Deliberately a small,
/// portable subset (software-development methodology + a couple of writing/design skills); the bulk of
/// hermes' 73 bundled skills are integrations against tools daemon does not (yet) host. See
/// `bundled/README` provenance in `daemon-gui-readiness-roadmap.md`.
pub static BUNDLED: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/bundled");

/// The index uses a short description; longer descriptions are truncated to this many chars (hermes
/// `extract_skill_description`'s ≤60).
const INDEX_DESCRIPTION_MAX: usize = 60;

/// The support-file subdirectories a skill bundle may hold (the only paths `write_file`/`remove_file`
/// will touch besides `SKILL.md`), mirroring hermes' bundle layout.
const SUPPORT_DIRS: [&str; 4] = ["references", "templates", "scripts", "assets"];

/// Errors surfaced by the skills store.
#[derive(Debug, Error)]
pub enum SkillError {
    /// The named skill does not exist under the store root.
    #[error("skill not found: {0}")]
    NotFound(String),
    /// A skill with that name already exists (on `create`).
    #[error("skill already exists: {0}")]
    Exists(String),
    /// The request named an invalid skill name or file path (path traversal, reserved dir, …).
    #[error("invalid skill request: {0}")]
    Invalid(String),
    /// A `SKILL.md` is missing required frontmatter (`name` + `description`) or a body.
    #[error("malformed skill: {0}")]
    Malformed(String),
    /// A `patch` did not find its `old_string` (so nothing was changed).
    #[error("patch target not found in {0}")]
    PatchMiss(String),
    /// An underlying filesystem error.
    #[error("skills io: {0}")]
    Io(String),
}

impl From<std::io::Error> for SkillError {
    fn from(e: std::io::Error) -> Self {
        SkillError::Io(e.to_string())
    }
}

/// The `metadata.hermes.*` conditional-visibility hints (offer-time tool/toolset gating + curation
/// tags). Parsed best-effort; unknown keys are ignored.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct HermesMeta {
    /// Curation/search tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Sibling skills a reader may also want.
    #[serde(default)]
    pub related_skills: Vec<String>,
    /// The skill is shown only when one of these toolsets is present.
    #[serde(default)]
    pub requires_toolsets: Vec<String>,
    /// The skill is shown as a fallback only when none of these toolsets are present.
    #[serde(default)]
    pub fallback_for_toolsets: Vec<String>,
    /// The skill is shown only when one of these tools is present.
    #[serde(default)]
    pub requires_tools: Vec<String>,
    /// The skill is shown as a fallback only when none of these tools are present.
    #[serde(default)]
    pub fallback_for_tools: Vec<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RawMetadata {
    #[serde(default)]
    hermes: HermesMeta,
}

/// The parsed `SKILL.md` frontmatter (the subset the index + tools need).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SkillFrontmatter {
    /// The skill's frontmatter name (required; ≤64 chars in hermes).
    #[serde(default)]
    pub name: String,
    /// One-line description (required; the index truncates to ≤60 chars).
    #[serde(default)]
    pub description: String,
    /// Optional semantic version.
    #[serde(default)]
    pub version: Option<String>,
    /// Optional license id.
    #[serde(default)]
    pub license: Option<String>,
    /// Optional OS gate (`macos`, `linux`, `windows`); empty means all platforms.
    #[serde(default)]
    pub platforms: Vec<String>,
    /// Hermes-namespaced conditional-visibility metadata.
    #[serde(default)]
    metadata: RawMetadata,
}

impl SkillFrontmatter {
    /// The `metadata.hermes.*` hints.
    pub fn hermes(&self) -> &HermesMeta {
        &self.metadata.hermes
    }
}

/// One discovered skill bundle: its on-disk identity (directory name + category), parsed frontmatter,
/// and bundle directory. The body is read lazily (only `skill_view` needs it).
#[derive(Debug, Clone)]
pub struct SkillEntry {
    /// The bundle directory name (the canonical skill name used by tools).
    pub name: String,
    /// The category path segment under the skills root (`None` for a top-level skill).
    pub category: Option<String>,
    /// Parsed frontmatter.
    pub frontmatter: SkillFrontmatter,
    /// The bundle directory (`<root>/<category>/<name>`).
    pub dir: PathBuf,
}

impl SkillEntry {
    /// The path to this bundle's `SKILL.md`.
    pub fn skill_md(&self) -> PathBuf {
        self.dir.join("SKILL.md")
    }

    /// The short, index-facing description (truncated to [`INDEX_DESCRIPTION_MAX`]).
    pub fn short_description(&self) -> String {
        truncate(&self.frontmatter.description, INDEX_DESCRIPTION_MAX)
    }
}

/// A compact `skills_list` row.
#[derive(Debug, Clone)]
pub struct SkillListItem {
    /// The bundle name.
    pub name: String,
    /// One-line description (untruncated).
    pub description: String,
    /// The category, if any.
    pub category: Option<String>,
}

/// The filesystem skills store rooted at `<profile_home>/skills/`. Discovery is recomputed lazily and
/// the rendered index is memoized until a write invalidates it.
pub struct SkillStore {
    root: PathBuf,
    /// Memoized rendered index (the prompt-caching invariant); cleared on any write.
    index_cache: Mutex<Option<String>>,
    /// Optional append-only revision history; when set, every write records a [`SkillBundle`]
    /// snapshot so skills are versioned (incl. the agent's own background-review edits).
    revisions: Option<Arc<dyn RevisionLog>>,
    /// The author attributed to writes through the model-facing tool surface (the agent). Operator
    /// writes (import / revert from the NodeApi) pass their author explicitly.
    default_author: Author,
}

impl SkillStore {
    /// A store over the skills directory `root` (created on demand by writes; reads tolerate absence).
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            index_cache: Mutex::new(None),
            revisions: None,
            default_author: Author::Agent("skill_manage".to_string()),
        }
    }

    /// Attach an append-only [`RevisionLog`] so every write records a versioned skill snapshot.
    pub fn with_revisions(mut self, revisions: Arc<dyn RevisionLog>) -> Self {
        self.revisions = Some(revisions);
        self
    }

    /// The skills root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Discover every skill bundle under the root (recursively finding `SKILL.md`), skipping
    /// sidecar/hidden dirs and bundles whose frontmatter cannot be parsed. Results are sorted by
    /// `(category, name)` so the index is deterministic.
    pub fn discover(&self) -> Vec<SkillEntry> {
        let mut out = Vec::new();
        walk_skill_files(&self.root, &mut out);
        let mut entries: Vec<SkillEntry> = out
            .into_iter()
            .filter_map(|md| self.load_entry(&md))
            .collect();
        entries.sort_by(|a, b| {
            (a.category.as_deref().unwrap_or(""), a.name.as_str())
                .cmp(&(b.category.as_deref().unwrap_or(""), b.name.as_str()))
        });
        entries
    }

    /// A compact `skills_list` view (name + description + category) of all discovered skills.
    pub fn list(&self) -> Vec<SkillListItem> {
        self.discover()
            .into_iter()
            .map(|e| SkillListItem {
                name: e.name.clone(),
                description: e.frontmatter.description.clone(),
                category: e.category.clone(),
            })
            .collect()
    }

    /// The progressive-disclosure index (the stable-tier system-prompt block). Memoized; the empty
    /// string when there are no skills (the caller then injects nothing). Mirrors hermes'
    /// `build_skills_system_prompt` shape.
    pub fn render_index(&self) -> String {
        if let Some(cached) = self.index_cache.lock().unwrap().clone() {
            return cached;
        }
        let rendered = self.render_index_uncached();
        *self.index_cache.lock().unwrap() = Some(rendered.clone());
        rendered
    }

    fn render_index_uncached(&self) -> String {
        let entries = self.discover();
        if entries.is_empty() {
            return String::new();
        }
        let mut s = String::new();
        s.push_str("## Skills (mandatory)\n");
        s.push_str(
            "Before replying, scan the skills below. If one matches the task, load its full \
             instructions with skill_view(name) before proceeding.\n\n",
        );
        s.push_str("<available_skills>\n");
        let mut current: Option<String> = None;
        for e in &entries {
            let cat = e.category.clone().unwrap_or_else(|| "general".to_string());
            if current.as_deref() != Some(cat.as_str()) {
                s.push_str(&format!("  {cat}:\n"));
                current = Some(cat);
            }
            s.push_str(&format!("    - {}: {}\n", e.name, e.short_description()));
        }
        s.push_str("</available_skills>\n");
        s
    }

    /// Invalidate the memoized index (called after every write through this store).
    pub fn invalidate(&self) {
        *self.index_cache.lock().unwrap() = None;
    }

    /// The full `SKILL.md` body of `name`, or a linked support file's contents when `file_path` is
    /// given (loaded on demand — the progressive-disclosure level-2/3 read).
    pub fn view(&self, name: &str, file_path: Option<&str>) -> Result<String, SkillError> {
        let entry = self.find(name)?;
        match file_path {
            None => Ok(fs::read_to_string(entry.skill_md())?),
            Some(rel) => {
                let path = self.safe_join(&entry.dir, rel)?;
                Ok(fs::read_to_string(path)?)
            }
        }
    }

    /// Create a new skill bundle `<root>/[category/]name/SKILL.md` from a full `SKILL.md` `content`.
    /// Fails if the skill already exists or the content lacks frontmatter `name`+`description`/body.
    pub fn create(
        &self,
        name: &str,
        content: &str,
        category: Option<&str>,
    ) -> Result<PathBuf, SkillError> {
        validate_name(name)?;
        validate_skill_md(content)?;
        if self.find(name).is_ok() {
            return Err(SkillError::Exists(name.to_string()));
        }
        let dir = match category {
            Some(cat) => {
                validate_segment(cat)?;
                self.root.join(cat).join(name)
            }
            None => self.root.join(name),
        };
        fs::create_dir_all(&dir)?;
        let md = dir.join("SKILL.md");
        fs::write(&md, content)?;
        self.invalidate();
        self.record(name, self.default_author.clone(), "create");
        Ok(md)
    }

    /// Replace an existing skill's `SKILL.md` wholesale.
    pub fn edit(&self, name: &str, content: &str) -> Result<PathBuf, SkillError> {
        validate_skill_md(content)?;
        let entry = self.find(name)?;
        fs::write(entry.skill_md(), content)?;
        self.invalidate();
        self.record(name, self.default_author.clone(), "edit");
        Ok(entry.skill_md())
    }

    /// Find/replace a substring in a skill's `SKILL.md` (or a support file when `file_path` is set).
    /// Replaces the first occurrence, or all occurrences when `replace_all`. Errors if `old` is
    /// absent (so a no-op patch is never silently accepted).
    pub fn patch(
        &self,
        name: &str,
        old: &str,
        new: &str,
        file_path: Option<&str>,
        replace_all: bool,
    ) -> Result<PathBuf, SkillError> {
        let entry = self.find(name)?;
        let target = match file_path {
            None => entry.skill_md(),
            Some(rel) => self.safe_join(&entry.dir, rel)?,
        };
        let body = fs::read_to_string(&target)?;
        if !body.contains(old) {
            return Err(SkillError::PatchMiss(display_path(&target)));
        }
        let patched = if replace_all {
            body.replace(old, new)
        } else {
            body.replacen(old, new, 1)
        };
        fs::write(&target, patched)?;
        self.invalidate();
        self.record(name, self.default_author.clone(), "patch");
        Ok(target)
    }

    /// Delete a skill bundle (its whole directory). Path-guarded to the store root.
    pub fn delete(&self, name: &str) -> Result<(), SkillError> {
        let entry = self.find(name)?;
        // Defense in depth: never recurse-remove anything outside the store root.
        let canon_root = self.root.canonicalize().unwrap_or_else(|_| self.root.clone());
        let canon_dir = entry
            .dir
            .canonicalize()
            .unwrap_or_else(|_| entry.dir.clone());
        if !canon_dir.starts_with(&canon_root) || canon_dir == canon_root {
            return Err(SkillError::Invalid(format!(
                "refusing to delete outside skills root: {}",
                display_path(&entry.dir)
            )));
        }
        fs::remove_dir_all(&entry.dir)?;
        self.invalidate();
        self.record(name, self.default_author.clone(), "delete");
        Ok(())
    }

    /// Write a support file under a skill bundle (only `SKILL.md` or `references/`,`templates/`,
    /// `scripts/`,`assets/` paths; other paths are rejected).
    pub fn write_file(
        &self,
        name: &str,
        file_path: &str,
        content: &str,
    ) -> Result<PathBuf, SkillError> {
        let entry = self.find(name)?;
        let path = self.safe_support_path(&entry.dir, file_path)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, content)?;
        self.invalidate();
        self.record(name, self.default_author.clone(), "write_file");
        Ok(path)
    }

    /// Remove a support file from a skill bundle (same path guard as [`Self::write_file`]).
    pub fn remove_file(&self, name: &str, file_path: &str) -> Result<(), SkillError> {
        let entry = self.find(name)?;
        let path = self.safe_support_path(&entry.dir, file_path)?;
        fs::remove_file(&path)?;
        self.invalidate();
        self.record(name, self.default_author.clone(), "remove_file");
        Ok(())
    }

    // -- versioning + distribution ------------------------------------------------------------

    /// Export one skill bundle as a portable [`SkillBundle`] (its `SKILL.md` + support files). Used
    /// as the revision snapshot blob and the distribution payload.
    pub fn export_bundle(&self, name: &str) -> Result<SkillBundle, SkillError> {
        let entry = self.find(name)?;
        let mut files = BTreeMap::new();
        collect_bundle_files(&entry.dir, &entry.dir, &mut files)?;
        Ok(SkillBundle {
            name: entry.name,
            category: entry.category,
            files,
        })
    }

    /// Every locally-authored (non-bundled) skill as a portable [`SkillBundle`] — the set a profile
    /// distribution carries (bundled skills are reconstituted from the binary on import, never shipped).
    pub fn export_local(&self) -> Result<Vec<SkillBundle>, SkillError> {
        let bundled = bundled_names();
        let mut out = Vec::new();
        for entry in self.discover() {
            if bundled.contains(&entry.name) {
                continue;
            }
            out.push(self.export_bundle(&entry.name)?);
        }
        Ok(out)
    }

    /// Write a [`SkillBundle`] to disk (overwriting any existing bundle of that name), recording a
    /// revision under `author`/`reason`. Used by import and revert; path-guarded to the store root.
    pub fn import_bundle(
        &self,
        bundle: &SkillBundle,
        author: Author,
        reason: &str,
    ) -> Result<(), SkillError> {
        validate_name(&bundle.name)?;
        let dir = match &bundle.category {
            Some(cat) => {
                validate_segment(cat)?;
                self.root.join(cat).join(&bundle.name)
            }
            None => self.root.join(&bundle.name),
        };
        // Replace wholesale so a revert/import is the exact recorded snapshot (drops stray files).
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        fs::create_dir_all(&dir)?;
        for (rel, content) in &bundle.files {
            let path = self.safe_support_path(&dir, rel)?;
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, content)?;
        }
        // A bundle is malformed without a parseable SKILL.md; reject rather than persist a broken skill.
        if let Some(md) = bundle.files.get("SKILL.md") {
            validate_skill_md(md)?;
        } else {
            return Err(SkillError::Malformed(format!(
                "bundle `{}` has no SKILL.md",
                bundle.name
            )));
        }
        self.invalidate();
        self.record(&bundle.name, author, reason);
        Ok(())
    }

    /// Whether `name` is one of the binary-bundled skills (read-only: not versioned/reverted/exported).
    pub fn is_bundled(&self, name: &str) -> bool {
        bundled_names().contains(name)
    }

    /// Record a revision of `name`'s current on-disk bundle (a tombstone when it was just deleted).
    /// Best-effort: a revision-log hiccup never fails the underlying skill write.
    fn record(&self, name: &str, author: Author, reason: &str) {
        let Some(log) = &self.revisions else {
            return;
        };
        let bundle = self.export_bundle(name).unwrap_or_else(|_| SkillBundle {
            name: name.to_string(),
            category: None,
            files: BTreeMap::new(),
        });
        let mut blob = Vec::new();
        if ciborium::into_writer(&bundle, &mut blob).is_ok() {
            let _ = log.append(RevisionKind::Skill, name, &blob, author, reason);
        }
    }

    /// One-shot bundled→user seed: copy any bundle present under `bundled_root` but absent under this
    /// store's root (by name), so a fresh profile starts with the shipped skills without clobbering
    /// user edits. Returns the names seeded.
    pub fn seed_from(&self, bundled_root: impl AsRef<Path>) -> Result<Vec<String>, SkillError> {
        let bundled = SkillStore::new(bundled_root.as_ref().to_path_buf());
        let existing: std::collections::HashSet<String> =
            self.discover().into_iter().map(|e| e.name).collect();
        let mut seeded = Vec::new();
        for entry in bundled.discover() {
            if existing.contains(&entry.name) {
                continue;
            }
            let dest = match &entry.category {
                Some(cat) => self.root.join(cat).join(&entry.name),
                None => self.root.join(&entry.name),
            };
            copy_dir_all(&entry.dir, &dest)?;
            seeded.push(entry.name);
        }
        if !seeded.is_empty() {
            self.invalidate();
        }
        Ok(seeded)
    }

    /// Seed the profile's skills dir from the compiled-in [`BUNDLED`] tree, skipping any skill whose
    /// name already exists on disk (so user edits/deletions are never clobbered — hermes' sync
    /// stance). Returns the names newly written. Idempotent: a no-op once seeded.
    pub fn seed_bundled(&self) -> Result<Vec<String>, SkillError> {
        let existing: std::collections::HashSet<String> =
            self.discover().into_iter().map(|e| e.name).collect();
        let mut seeded = Vec::new();
        for skill_dir in embedded_skill_dirs(&BUNDLED) {
            // The bundle name is the directory holding `SKILL.md` (e.g. `software-development/plan`).
            let Some(name) = skill_dir
                .path()
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
            else {
                continue;
            };
            if existing.contains(&name) {
                continue;
            }
            write_embedded_dir(skill_dir, &self.root)?;
            seeded.push(name);
        }
        if !seeded.is_empty() {
            self.invalidate();
        }
        Ok(seeded)
    }

    /// Resolve a skill by bundle name (the first match in discovery order).
    pub fn find(&self, name: &str) -> Result<SkillEntry, SkillError> {
        self.discover()
            .into_iter()
            .find(|e| e.name == name)
            .ok_or_else(|| SkillError::NotFound(name.to_string()))
    }

    fn load_entry(&self, skill_md: &Path) -> Option<SkillEntry> {
        let dir = skill_md.parent()?.to_path_buf();
        let name = dir.file_name()?.to_string_lossy().into_owned();
        let body = fs::read_to_string(skill_md).ok()?;
        let frontmatter = parse_frontmatter(&body).ok()?;
        if frontmatter.name.is_empty() && frontmatter.description.is_empty() {
            return None;
        }
        // Category = the first path segment under the root, unless the bundle sits directly under the
        // root (then that segment *is* the bundle name → no category).
        let category = dir
            .strip_prefix(&self.root)
            .ok()
            .and_then(|rel| rel.components().next())
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .filter(|seg| seg != &name);
        Some(SkillEntry {
            name,
            category,
            frontmatter,
            dir,
        })
    }

    /// Join a relative path under a bundle dir, rejecting traversal/absolute paths.
    fn safe_join(&self, dir: &Path, rel: &str) -> Result<PathBuf, SkillError> {
        let rel_path = Path::new(rel);
        if rel_path.is_absolute()
            || rel_path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(SkillError::Invalid(format!("unsafe file path: {rel}")));
        }
        Ok(dir.join(rel_path))
    }

    /// Like [`Self::safe_join`] but additionally restricts to `SKILL.md` or a known support dir.
    fn safe_support_path(&self, dir: &Path, rel: &str) -> Result<PathBuf, SkillError> {
        let path = self.safe_join(dir, rel)?;
        let allowed = rel == "SKILL.md"
            || SUPPORT_DIRS
                .iter()
                .any(|d| rel.starts_with(&format!("{d}/")));
        if !allowed {
            return Err(SkillError::Invalid(format!(
                "file path must be SKILL.md or under {SUPPORT_DIRS:?}: {rel}"
            )));
        }
        Ok(path)
    }
}

/// The skills *index* as a [`StablePromptSource`] (§10): emits the progressive-disclosure index into
/// the stable system-prompt tier, but **only when enabled** — hermes injects the index only if the
/// agent actually has the skills tools (`skills_list`/`skill_view`/`skill_manage`), so the binary
/// constructs this with `enabled = skills-tools-present`. The index is cache-stable (memoized in the
/// store, invalidated only on a write), preserving prompt caching; full bodies load via `skill_view`.
pub struct SkillsPromptSource {
    store: Arc<SkillStore>,
    enabled: bool,
}

impl SkillsPromptSource {
    /// A source over `store`, enabled (emits the index) by default.
    pub fn new(store: Arc<SkillStore>) -> Self {
        Self {
            store,
            enabled: true,
        }
    }

    /// Gate emission on whether the engine has the skills tools (hermes' `valid_tool_names` check).
    pub fn enabled(mut self, yes: bool) -> Self {
        self.enabled = yes;
        self
    }
}

impl StablePromptSource for SkillsPromptSource {
    fn block(&self) -> Option<String> {
        if !self.enabled {
            return None;
        }
        let index = self.store.render_index();
        (!index.is_empty()).then_some(index)
    }
}

/// Parse the leading `---`-delimited YAML frontmatter of a `SKILL.md` body.
pub fn parse_frontmatter(content: &str) -> Result<SkillFrontmatter, SkillError> {
    let trimmed = content.trim_start_matches('\u{feff}');
    let rest = trimmed
        .strip_prefix("---")
        .ok_or_else(|| SkillError::Malformed("missing frontmatter delimiter".into()))?;
    // The frontmatter ends at the next line that is exactly `---`.
    let end = rest
        .find("\n---")
        .ok_or_else(|| SkillError::Malformed("unterminated frontmatter".into()))?;
    let yaml = rest[..end].trim_start_matches(['\r', '\n']);
    serde_yaml::from_str::<SkillFrontmatter>(yaml)
        .map_err(|e| SkillError::Malformed(format!("frontmatter yaml: {e}")))
}

/// Validate a full `SKILL.md` document: parseable frontmatter with `name`+`description`, and a
/// non-empty body after the closing delimiter.
fn validate_skill_md(content: &str) -> Result<(), SkillError> {
    let fm = parse_frontmatter(content)?;
    if fm.name.trim().is_empty() {
        return Err(SkillError::Malformed("frontmatter `name` is required".into()));
    }
    if fm.description.trim().is_empty() {
        return Err(SkillError::Malformed(
            "frontmatter `description` is required".into(),
        ));
    }
    // Body present after the second `---`.
    let after = content
        .trim_start_matches('\u{feff}')
        .strip_prefix("---")
        .and_then(|r| r.find("\n---").map(|i| &r[i + 4..]))
        .unwrap_or("");
    if after.trim().is_empty() {
        return Err(SkillError::Malformed(
            "skill body after frontmatter is required".into(),
        ));
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<(), SkillError> {
    validate_segment(name)
}

fn validate_segment(seg: &str) -> Result<(), SkillError> {
    if seg.is_empty()
        || seg.len() > 64
        || seg.contains('/')
        || seg.contains('\\')
        || seg.contains("..")
        || seg.starts_with('.')
    {
        return Err(SkillError::Invalid(format!("invalid name/segment: {seg}")));
    }
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn display_path(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Whether a directory name is a sidecar/hidden entry to skip during discovery (hermes excludes
/// `.git`, `.archive`, `.hub`, venvs, …).
fn is_excluded_dir(name: &str) -> bool {
    name.starts_with('.')
        || matches!(
            name,
            "node_modules" | "venv" | ".venv" | "__pycache__" | "target"
        )
}

/// Recursively collect `SKILL.md` paths under `root`, skipping excluded dirs.
fn walk_skill_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(read) = fs::read_dir(root) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        let Ok( file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if is_excluded_dir(&name) {
                continue;
            }
            // A bundle dir contains SKILL.md; record it and do not descend into the bundle's support
            // dirs, but keep walking for nested categories.
            let md = path.join("SKILL.md");
            if md.is_file() {
                out.push(md);
            }
            walk_skill_files(&path, out);
        }
    }
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<(), SkillError> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)?.flatten() {
        let path = entry.path();
        let dest = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_all(&path, &dest)?;
        } else {
            fs::copy(&path, &dest)?;
        }
    }
    Ok(())
}

/// The set of binary-bundled skill names (the dir holding each `SKILL.md` in [`BUNDLED`]). Bundled
/// skills are read-only for versioning/distribution: never reverted, never shipped in a distribution
/// (the importer reconstitutes them from its own binary).
pub fn bundled_names() -> std::collections::HashSet<String> {
    embedded_skill_dirs(&BUNDLED)
        .iter()
        .filter_map(|d| {
            d.path()
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
        })
        .collect()
}

/// Recursively collect a bundle's text files (relative to `base`), restricted to `SKILL.md` and the
/// recognized support dirs — the same surface `write_file` permits, so an import is always writable.
fn collect_bundle_files(
    base: &Path,
    dir: &Path,
    out: &mut BTreeMap<String, String>,
) -> Result<(), SkillError> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_bundle_files(base, &path, out)?;
            continue;
        }
        let Ok(rel) = path.strip_prefix(base) else {
            continue;
        };
        let rel = rel.to_string_lossy().replace('\\', "/");
        let allowed = rel == "SKILL.md"
            || SUPPORT_DIRS.iter().any(|d| rel.starts_with(&format!("{d}/")));
        if !allowed {
            continue;
        }
        match fs::read_to_string(&path) {
            Ok(content) => {
                out.insert(rel, content);
            }
            // Skip non-UTF8 (binary) assets: a bundle is text-only (markdown + support docs).
            Err(_) => continue,
        }
    }
    Ok(())
}

/// Recursively collect every embedded directory that directly contains a `SKILL.md` — i.e. the skill
/// bundle dirs in the compiled-in [`BUNDLED`] tree.
fn embedded_skill_dirs<'a>(dir: &'a Dir<'a>) -> Vec<&'a Dir<'a>> {
    let mut out = Vec::new();
    if dir
        .files()
        .any(|f| f.path().file_name().is_some_and(|n| n == "SKILL.md"))
    {
        out.push(dir);
    }
    for sub in dir.dirs() {
        out.extend(embedded_skill_dirs(sub));
    }
    out
}

/// Write every embedded file under `dir` (recursively) to `dest_root`, preserving the embedded
/// relative path (which already encodes `<category>/<name>/…`).
fn write_embedded_dir(dir: &Dir<'_>, dest_root: &Path) -> Result<(), SkillError> {
    let mut stack = vec![dir];
    while let Some(d) = stack.pop() {
        for file in d.files() {
            let dest = dest_root.join(file.path());
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&dest, file.contents())?;
        }
        stack.extend(d.dirs());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
