// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `FileSkillUsageLog` — the file-backed [`SkillUsageLog`] (a profile's `.usage.json` sidecar).
//!
//! Co-located with the profile's skills dir (`<profile_home>/skills/.usage.json`) so the usage +
//! lifecycle record travels with the skill library it describes (profile distribution, per-agent
//! curation). The whole map is held in memory behind a mutex and rewritten on each mutation; writes
//! are best-effort (a usage-bump failure never fails a turn). This mirrors `FileRevisionLog`'s
//! file-backed stance for [`daemon_common::RevisionLog`], but keyed by skill name within one profile.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use daemon_common::{SkillCreator, SkillState, SkillUsage, SkillUsageLog};

/// The sidecar filename under a profile's skills dir.
const USAGE_FILE: &str = ".usage.json";

/// Wall-clock milliseconds since the Unix epoch (saturating; monotonic enough for staleness).
pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A file-backed [`SkillUsageLog`] persisting to `<skills_root>/.usage.json`.
pub struct FileSkillUsageLog {
    path: PathBuf,
    map: Mutex<BTreeMap<String, SkillUsage>>,
}

impl FileSkillUsageLog {
    /// Open (or lazily create) the usage sidecar under the skills `root`. A missing or corrupt file
    /// starts an empty log (best-effort — usage is non-authoritative telemetry).
    pub fn open(root: impl AsRef<Path>) -> Self {
        let path = root.as_ref().join(USAGE_FILE);
        let map = std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<BTreeMap<String, SkillUsage>>(&bytes).ok())
            .unwrap_or_default();
        Self {
            path,
            map: Mutex::new(map),
        }
    }

    /// Persist the current map (best-effort): create the parent dir, then write the JSON.
    fn flush(&self, map: &BTreeMap<String, SkillUsage>) {
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(bytes) = serde_json::to_vec_pretty(map) {
            let _ = std::fs::write(&self.path, bytes);
        }
    }

    /// Apply `f` to the (possibly newly-inserted) entry for `name`, then persist.
    fn mutate(&self, name: &str, f: impl FnOnce(&mut SkillUsage)) {
        let mut map = self.map.lock().unwrap();
        let entry = map.entry(name.to_string()).or_insert_with(|| SkillUsage {
            created_at_ms: now_ms(),
            ..SkillUsage::default()
        });
        f(entry);
        self.flush(&map);
    }
}

impl SkillUsageLog for FileSkillUsageLog {
    fn record_create(&self, name: &str, creator: SkillCreator) {
        let now = now_ms();
        let mut map = self.map.lock().unwrap();
        match map.get_mut(name) {
            // Re-create of a known skill keeps its earliest created_at but refreshes provenance.
            Some(entry) => {
                entry.created_by = creator;
                entry.state = SkillState::Active;
            }
            None => {
                map.insert(
                    name.to_string(),
                    SkillUsage {
                        created_by: creator,
                        created_at_ms: now,
                        ..SkillUsage::default()
                    },
                );
            }
        }
        self.flush(&map);
    }

    fn record_view(&self, name: &str) {
        let now = now_ms();
        self.mutate(name, |e| {
            e.view_count += 1;
            e.use_count += 1;
            e.last_viewed_ms = Some(now);
            e.last_used_ms = Some(now);
            // A view reactivates a stale skill (it is back in use); archived stays archived until
            // an explicit restore.
            if e.state == SkillState::Stale {
                e.state = SkillState::Active;
            }
        });
    }

    fn record_patch(&self, name: &str) {
        let now = now_ms();
        self.mutate(name, |e| {
            e.patch_count += 1;
            e.last_patched_ms = Some(now);
            if e.state == SkillState::Stale {
                e.state = SkillState::Active;
            }
        });
    }

    fn forget(&self, name: &str) {
        let mut map = self.map.lock().unwrap();
        if map.remove(name).is_some() {
            self.flush(&map);
        }
    }

    fn set_state(&self, name: &str, state: SkillState) {
        self.mutate(name, |e| e.state = state);
    }

    fn set_pinned(&self, name: &str, pinned: bool) {
        self.mutate(name, |e| e.pinned = pinned);
    }

    fn get(&self, name: &str) -> Option<SkillUsage> {
        self.map.lock().unwrap().get(name).cloned()
    }

    fn all(&self) -> BTreeMap<String, SkillUsage> {
        self.map.lock().unwrap().clone()
    }
}
