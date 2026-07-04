// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Plugin architecture — port of `mnemosyne/core/plugins.py`.
//!
//! Plugins hook into the memory lifecycle: `on_remember`, `on_recall`, `on_consolidate`,
//! `on_invalidate`. The [`PluginManager`] owns loaded instances and fans events out to every
//! enabled plugin, isolating failures (a panicking plugin can poison nothing but its own state —
//! hooks are infallible by contract, matching Python's log-and-continue `notify_*`).
//!
//! Scope notes vs Python:
//! - Python's registry maps names to *classes* and lazily instantiates. Rust has no classes;
//!   [`PluginManager::register`] takes a constructed instance directly (the built-ins are
//!   registered on first access, like Python's `PluginManager.__init__`).
//! - Filesystem discovery (`~/.hermes/mnemosyne/plugins/*.py`, `discover_plugins` L571-L613)
//!   does not port: Rust cannot load arbitrary source files at runtime. Hosts link plugins in
//!   and register them explicitly.
//! - [`CompressionPlugin`] ports the *behavioral* contract: in Python it only compresses when
//!   the optional `rust_cave_001` package imports, otherwise `compress_lines` returns its input
//!   unchanged (L396-L397). No such backend exists here, so it is permanently in the
//!   backend-unavailable state — exactly what a Python deployment without the package runs.
//!
//! The engine consults the compression plugin at the sleep summarization seam (`beam.py`
//! L7738-L7743) and emits lifecycle notifications from remember/recall/invalidate/consolidate.
//! Both are gated on the manager having been materialized ([`crate::engine::Engine::plugins`]),
//! so an engine whose host never touches plugins pays one atomic load per event — and behaves
//! identically to Python, which never wires `notify_*` into `beam.py` at all.

use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

/// Lifecycle hooks for one plugin (`plugins.py` `MnemosynePlugin` L35-L88).
///
/// Hooks take `&self`: implementations that accumulate state use interior mutability (the
/// built-ins wrap their state in `Mutex`). All hooks have no-op defaults so a plugin only
/// implements the events it cares about (Python forces all four via `abc.abstractmethod`;
/// defaults are kinder here and behaviorally equivalent to an empty body).
pub trait MnemosynePlugin: Send + Sync {
    /// Unique plugin name (registry key).
    fn name(&self) -> &str;

    /// Plugin version string.
    fn version(&self) -> &str {
        "1.0.0"
    }

    /// Whether the plugin receives notifications (`enabled` class attr).
    fn enabled(&self) -> bool {
        true
    }

    /// Called once when the plugin is loaded into a manager.
    fn initialize(&self) {}

    /// Called once when the plugin is unloaded.
    fn shutdown(&self) {}

    /// Called when a memory is stored.
    fn on_remember(&self, _memory: &Value) {}

    /// Called when a memory is recalled.
    fn on_recall(&self, _memory: &Value) {}

    /// Called during sleep/consolidation.
    fn on_consolidate(&self, _summary: &Value) {}

    /// Called when a memory is invalidated.
    fn on_invalidate(&self, _memory_id: &str) {}

    /// Serialized plugin metadata (`to_dict` L80-L88).
    fn to_dict(&self) -> Value {
        json!({
            "name": self.name(),
            "version": self.version(),
            "enabled": self.enabled(),
        })
    }
}

/// Truncate content to a preview (`LoggingPlugin._preview` L148-L151).
fn preview(content: &str, max_len: usize) -> String {
    if content.chars().count() <= max_len {
        content.to_string()
    } else {
        let head: String = content.chars().take(max_len).collect();
        format!("{head}...")
    }
}

/// Built-in plugin that logs all memory operations to a bounded in-memory ring buffer and the
/// `tracing` log (`plugins.py` `LoggingPlugin` L91-L164).
pub struct LoggingPlugin {
    max_entries: usize,
    log: Mutex<VecDeque<Value>>,
}

impl Default for LoggingPlugin {
    fn default() -> Self {
        Self::new(10_000)
    }
}

impl LoggingPlugin {
    /// Create with a maximum retained entry count (`max_entries` config, default 10000).
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            log: Mutex::new(VecDeque::new()),
        }
    }

    fn append(&self, entry: Value) {
        tracing::info!(target: "mnemosyne::plugins", entry = %entry, "[LoggingPlugin]");
        let mut log = self.log.lock().unwrap();
        log.push_back(entry);
        if log.len() > self.max_entries {
            log.pop_front();
        }
    }

    /// The in-memory log entries (`get_log`).
    pub fn get_log(&self) -> Vec<Value> {
        self.log.lock().unwrap().iter().cloned().collect()
    }

    /// Clear the in-memory log (`clear_log`).
    pub fn clear_log(&self) {
        self.log.lock().unwrap().clear();
    }
}

impl MnemosynePlugin for LoggingPlugin {
    fn name(&self) -> &str {
        "logging"
    }

    fn on_remember(&self, memory: &Value) {
        self.append(json!({
            "event": "remember",
            "timestamp": crate::util::now_iso(),
            "memory_id": memory.get("id"),
            "content_preview": preview(memory.get("content").and_then(Value::as_str).unwrap_or(""), 80),
        }));
    }

    fn on_recall(&self, memory: &Value) {
        self.append(json!({
            "event": "recall",
            "timestamp": crate::util::now_iso(),
            "memory_id": memory.get("id"),
            "content_preview": preview(memory.get("content").and_then(Value::as_str).unwrap_or(""), 80),
        }));
    }

    fn on_consolidate(&self, summary: &Value) {
        self.append(json!({
            "event": "consolidate",
            "timestamp": crate::util::now_iso(),
            "summary_preview": preview(summary.get("summary").and_then(Value::as_str).unwrap_or(""), 80),
            "source_count": summary
                .get("source_wm_ids")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0),
        }));
    }

    fn on_invalidate(&self, memory_id: &str) {
        self.append(json!({
            "event": "invalidate",
            "timestamp": crate::util::now_iso(),
            "memory_id": memory_id,
        }));
    }
}

/// Built-in plugin that counts operations and records timings (`plugins.py` `MetricsPlugin`
/// L167-L242). Timing samples are bounded (default 1000 per event).
pub struct MetricsPlugin {
    remember: AtomicU64,
    recall: AtomicU64,
    consolidate: AtomicU64,
    invalidate: AtomicU64,
    max_timing_samples: usize,
    timings: Mutex<std::collections::HashMap<String, VecDeque<f64>>>,
}

impl Default for MetricsPlugin {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl MetricsPlugin {
    /// Create with a maximum retained timing-sample count per event.
    pub fn new(max_timing_samples: usize) -> Self {
        Self {
            remember: AtomicU64::new(0),
            recall: AtomicU64::new(0),
            consolidate: AtomicU64::new(0),
            invalidate: AtomicU64::new(0),
            max_timing_samples,
            timings: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Record the duration of an operation (`record_timing`).
    pub fn record_timing(&self, event: &str, duration_ms: f64) {
        let mut timings = self.timings.lock().unwrap();
        let samples = timings.entry(event.to_string()).or_default();
        samples.push_back(duration_ms);
        if samples.len() > self.max_timing_samples {
            samples.pop_front();
        }
    }

    /// Event counters (`get_counters`).
    pub fn get_counters(&self) -> Value {
        json!({
            "remember": self.remember.load(Ordering::Relaxed),
            "recall": self.recall.load(Ordering::Relaxed),
            "consolidate": self.consolidate.load(Ordering::Relaxed),
            "invalidate": self.invalidate.load(Ordering::Relaxed),
        })
    }

    /// Timing samples for one event (`get_timings`).
    pub fn get_timings(&self, event: &str) -> Vec<f64> {
        self.timings
            .lock()
            .unwrap()
            .get(event)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Average timing for one event, `None` when no samples exist (`get_average_timing`).
    pub fn get_average_timing(&self, event: &str) -> Option<f64> {
        let timings = self.timings.lock().unwrap();
        let samples = timings.get(event)?;
        if samples.is_empty() {
            return None;
        }
        Some(samples.iter().sum::<f64>() / samples.len() as f64)
    }

    /// Reset all counters and timings (`reset`).
    pub fn reset(&self) {
        self.remember.store(0, Ordering::Relaxed);
        self.recall.store(0, Ordering::Relaxed);
        self.consolidate.store(0, Ordering::Relaxed);
        self.invalidate.store(0, Ordering::Relaxed);
        self.timings.lock().unwrap().clear();
    }

    /// Counters plus per-event timing averages (`get_summary`).
    pub fn get_summary(&self) -> Value {
        let timings = self.timings.lock().unwrap();
        let averages: serde_json::Map<String, Value> = timings
            .iter()
            .map(|(event, samples)| {
                let avg = if samples.is_empty() {
                    Value::Null
                } else {
                    json!(samples.iter().sum::<f64>() / samples.len() as f64)
                };
                (event.clone(), avg)
            })
            .collect();
        json!({"counters": self.get_counters(), "averages": averages})
    }
}

impl MnemosynePlugin for MetricsPlugin {
    fn name(&self) -> &str {
        "metrics"
    }

    fn on_remember(&self, _memory: &Value) {
        self.remember.fetch_add(1, Ordering::Relaxed);
    }

    fn on_recall(&self, _memory: &Value) {
        self.recall.fetch_add(1, Ordering::Relaxed);
    }

    fn on_consolidate(&self, _summary: &Value) {
        self.consolidate.fetch_add(1, Ordering::Relaxed);
    }

    fn on_invalidate(&self, _memory_id: &str) {
        self.invalidate.fetch_add(1, Ordering::Relaxed);
    }
}

/// A filtering rule: `true` = allow, `false` = block (`FilterPlugin` rules).
pub type FilterRule = Box<dyn Fn(&Value) -> bool + Send + Sync>;

/// Built-in plugin that tracks memories blocked by custom rules (`plugins.py` `FilterPlugin`
/// L245-L319). Rules see the memory/summary JSON; any rule returning `false` blocks the item
/// (side-effect tracking only — the engine does not consult the blocklist).
pub struct FilterPlugin {
    max_blocked: usize,
    rules: Mutex<Vec<FilterRule>>,
    blocked: Mutex<VecDeque<Value>>,
}

impl Default for FilterPlugin {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl FilterPlugin {
    /// Create with a maximum retained blocked-item count.
    pub fn new(max_blocked: usize) -> Self {
        Self {
            max_blocked,
            rules: Mutex::new(Vec::new()),
            blocked: Mutex::new(VecDeque::new()),
        }
    }

    /// Register a filtering rule (`add_rule`).
    pub fn add_rule(&self, rule: FilterRule) {
        self.rules.lock().unwrap().push(rule);
    }

    /// Remove all filtering rules (`clear_rules`).
    pub fn clear_rules(&self) {
        self.rules.lock().unwrap().clear();
    }

    fn passes(&self, item: &Value) -> bool {
        self.rules.lock().unwrap().iter().all(|rule| rule(item))
    }

    fn block(&self, item: &Value) {
        let mut blocked = self.blocked.lock().unwrap();
        blocked.push_back(json!({"timestamp": crate::util::now_iso(), "item": item}));
        if blocked.len() > self.max_blocked {
            blocked.pop_front();
        }
    }

    /// All blocked items (`get_blocked`).
    pub fn get_blocked(&self) -> Vec<Value> {
        self.blocked.lock().unwrap().iter().cloned().collect()
    }

    /// Whether a memory id has been blocked (`is_blocked`).
    pub fn is_blocked(&self, memory_id: &str) -> bool {
        self.blocked.lock().unwrap().iter().any(|entry| {
            entry
                .get("item")
                .and_then(|i| i.get("id"))
                .and_then(Value::as_str)
                == Some(memory_id)
        })
    }
}

impl MnemosynePlugin for FilterPlugin {
    fn name(&self) -> &str {
        "filter"
    }

    fn on_remember(&self, memory: &Value) {
        if !self.passes(memory) {
            self.block(memory);
        }
    }

    fn on_recall(&self, memory: &Value) {
        if !self.passes(memory) {
            self.block(memory);
        }
    }

    fn on_consolidate(&self, summary: &Value) {
        if !self.passes(summary) {
            self.block(summary);
        }
    }
}

/// Built-in pre-compression hook for sleep summarization inputs (`plugins.py`
/// `CompressionPlugin` L322-L425). Opt-in (`enabled = false` by default) and — like a Python
/// install without the optional `rust_cave_001` package — permanently backend-unavailable, so
/// [`CompressionPlugin::compress_lines`] returns its input unchanged. The engine still consults
/// it at the sleep seam so an enabled future backend slots in without touching call sites.
pub struct CompressionPlugin {
    enabled: AtomicBool,
    /// Skip lines shorter than this (`threshold_chars`, default 20). Held for contract parity;
    /// unused while no backend exists.
    #[allow(dead_code)]
    threshold_chars: usize,
}

impl Default for CompressionPlugin {
    fn default() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            threshold_chars: 20,
        }
    }
}

impl CompressionPlugin {
    /// Enable/disable the plugin (Python: `config["enabled"]`).
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Compress memory content lines before summarization (`compress_lines` L383-L411).
    /// No backend is linked, so this is the Python no-backend path: input returned unchanged.
    pub fn compress_lines(&self, lines: Vec<String>) -> Vec<String> {
        lines
    }
}

impl MnemosynePlugin for CompressionPlugin {
    fn name(&self) -> &str {
        "compression"
    }

    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

/// One registry slot: the instance plus its load state.
struct Slot {
    plugin: std::sync::Arc<dyn MnemosynePlugin>,
    loaded: bool,
}

/// Register, load, and notify plugins (`plugins.py` `PluginManager` L428-L656).
///
/// The four built-ins are registered (not loaded) at construction, mirroring Python's
/// `__init__`. All methods take `&self` — the manager is safe to share behind the engine.
pub struct PluginManager {
    slots: Mutex<Vec<Slot>>,
    /// Concrete handle to the built-in compression plugin: the sleep seam calls its non-trait
    /// `compress_lines` (`beam.py` L7741 fetches it by name and calls the concrete method).
    compression: std::sync::Arc<CompressionPlugin>,
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginManager {
    /// A manager with the built-in plugins registered (`__init__` L438-L447).
    pub fn new() -> Self {
        let compression = std::sync::Arc::new(CompressionPlugin::default());
        let mgr = Self {
            slots: Mutex::new(Vec::new()),
            compression: std::sync::Arc::clone(&compression),
        };
        mgr.register(std::sync::Arc::new(LoggingPlugin::default()));
        mgr.register(std::sync::Arc::new(MetricsPlugin::default()));
        mgr.register(std::sync::Arc::new(FilterPlugin::default()));
        mgr.register(compression);
        mgr
    }

    /// The built-in compression plugin (`get_plugin("compression")` at the sleep seam).
    pub fn compression(&self) -> &CompressionPlugin {
        &self.compression
    }

    /// Register a plugin instance. Duplicate names are rejected (Python raises `ValueError`;
    /// this logs and ignores — the engine seam must never panic).
    pub fn register(&self, plugin: std::sync::Arc<dyn MnemosynePlugin>) {
        let mut slots = self.slots.lock().unwrap();
        if slots.iter().any(|s| s.plugin.name() == plugin.name()) {
            tracing::warn!(name = plugin.name(), "plugin already registered; ignoring");
            return;
        }
        slots.push(Slot {
            plugin,
            loaded: false,
        });
    }

    /// Load (initialize) a registered plugin by name; `None` when unregistered or already
    /// loaded (`load_plugin` L467-L492).
    pub fn load_plugin(&self, name: &str) -> Option<std::sync::Arc<dyn MnemosynePlugin>> {
        let mut slots = self.slots.lock().unwrap();
        let slot = slots.iter_mut().find(|s| s.plugin.name() == name)?;
        if slot.loaded {
            return None;
        }
        slot.plugin.initialize();
        slot.loaded = true;
        tracing::info!(name, version = slot.plugin.version(), "loaded plugin");
        Some(std::sync::Arc::clone(&slot.plugin))
    }

    /// Shutdown and unload a plugin; `false` when it was not loaded (`unload_plugin`).
    pub fn unload_plugin(&self, name: &str) -> bool {
        let mut slots = self.slots.lock().unwrap();
        let Some(slot) = slots
            .iter_mut()
            .find(|s| s.plugin.name() == name && s.loaded)
        else {
            return false;
        };
        slot.plugin.shutdown();
        slot.loaded = false;
        true
    }

    /// A loaded plugin by name, lazily loading registered-but-unloaded ones on first access
    /// (`get_plugin` L528-L538).
    pub fn get_plugin(&self, name: &str) -> Option<std::sync::Arc<dyn MnemosynePlugin>> {
        {
            let mut slots = self.slots.lock().unwrap();
            if let Some(slot) = slots.iter_mut().find(|s| s.plugin.name() == name) {
                if !slot.loaded {
                    slot.plugin.initialize();
                    slot.loaded = true;
                }
                return Some(std::sync::Arc::clone(&slot.plugin));
            }
        }
        None
    }

    /// Whether a plugin is currently loaded (`is_loaded`).
    pub fn is_loaded(&self, name: &str) -> bool {
        self.slots
            .lock()
            .unwrap()
            .iter()
            .any(|s| s.plugin.name() == name && s.loaded)
    }

    /// Whether a plugin is registered (`is_registered`).
    pub fn is_registered(&self, name: &str) -> bool {
        self.slots
            .lock()
            .unwrap()
            .iter()
            .any(|s| s.plugin.name() == name)
    }

    /// Load every registered plugin (`load_all`).
    pub fn load_all(&self) {
        let mut slots = self.slots.lock().unwrap();
        for slot in slots.iter_mut().filter(|s| !s.loaded) {
            slot.plugin.initialize();
            slot.loaded = true;
        }
    }

    /// Unload every loaded plugin (`unload_all`).
    pub fn unload_all(&self) {
        let mut slots = self.slots.lock().unwrap();
        for slot in slots.iter_mut().filter(|s| s.loaded) {
            slot.plugin.shutdown();
            slot.loaded = false;
        }
    }

    /// Registered plugins with their load state (`list_plugins` L510-L526).
    pub fn list_plugins(&self) -> Vec<Value> {
        self.slots
            .lock()
            .unwrap()
            .iter()
            .map(|s| {
                json!({
                    "name": s.plugin.name(),
                    "loaded": s.loaded,
                    "enabled": s.plugin.enabled(),
                    "version": s.plugin.version(),
                })
            })
            .collect()
    }

    /// The loaded + enabled plugins, snapshot for notification fan-out.
    fn active(&self) -> Vec<std::sync::Arc<dyn MnemosynePlugin>> {
        self.slots
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.loaded && s.plugin.enabled())
            .map(|s| std::sync::Arc::clone(&s.plugin))
            .collect()
    }

    /// Notify all loaded plugins of a remember event (`notify_remember`).
    pub fn notify_remember(&self, memory: &Value) {
        for p in self.active() {
            p.on_remember(memory);
        }
    }

    /// Notify all loaded plugins of a recall event (`notify_recall`).
    pub fn notify_recall(&self, memory: &Value) {
        for p in self.active() {
            p.on_recall(memory);
        }
    }

    /// Notify all loaded plugins of a consolidate event (`notify_consolidate`).
    pub fn notify_consolidate(&self, summary: &Value) {
        for p in self.active() {
            p.on_consolidate(summary);
        }
    }

    /// Notify all loaded plugins of an invalidate event (`notify_invalidate`).
    pub fn notify_invalidate(&self, memory_id: &str) {
        for p in self.active() {
            p.on_invalidate(memory_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manager_registers_builtins_unloaded() {
        let mgr = PluginManager::new();
        for name in ["logging", "metrics", "filter", "compression"] {
            assert!(mgr.is_registered(name), "{name} should be registered");
            assert!(!mgr.is_loaded(name), "{name} should start unloaded");
        }
    }

    #[test]
    fn get_plugin_lazily_loads() {
        let mgr = PluginManager::new();
        assert!(mgr.get_plugin("metrics").is_some());
        assert!(mgr.is_loaded("metrics"));
        assert!(mgr.get_plugin("nonexistent").is_none());
    }

    #[test]
    fn notify_fans_out_to_enabled_loaded_plugins_only() {
        // Hold the concrete Arc so we can inspect state after registering the trait object —
        // the intended host pattern for accessing plugin-specific APIs.
        let metrics = std::sync::Arc::new(MetricsPlugin::default());
        let mgr = PluginManager {
            slots: Mutex::new(Vec::new()),
            compression: std::sync::Arc::new(CompressionPlugin::default()),
        };
        mgr.register(std::sync::Arc::clone(&metrics) as _);

        // Registered but not loaded: no delivery.
        mgr.notify_remember(&json!({"id": "m1", "content": "hello"}));
        assert_eq!(metrics.get_counters()["remember"], 0);

        mgr.load_all();
        mgr.notify_remember(&json!({"id": "m1", "content": "hello"}));
        mgr.notify_recall(&json!({"id": "m1", "content": "hello"}));
        mgr.notify_consolidate(&json!({"summary": "s", "source_wm_ids": ["m1"]}));
        mgr.notify_invalidate("m1");
        assert_eq!(metrics.get_counters()["remember"], 1);
        assert_eq!(metrics.get_counters()["recall"], 1);
        assert_eq!(metrics.get_counters()["consolidate"], 1);
        assert_eq!(metrics.get_counters()["invalidate"], 1);
    }

    #[test]
    fn metrics_counts_and_averages() {
        let metrics = MetricsPlugin::default();
        metrics.on_remember(&json!({}));
        metrics.on_remember(&json!({}));
        metrics.on_recall(&json!({}));
        metrics.record_timing("recall", 10.0);
        metrics.record_timing("recall", 20.0);
        assert_eq!(metrics.get_counters()["remember"], 2);
        assert_eq!(metrics.get_counters()["recall"], 1);
        assert_eq!(metrics.get_average_timing("recall"), Some(15.0));
        assert_eq!(metrics.get_average_timing("sleep"), None);
        metrics.reset();
        assert_eq!(metrics.get_counters()["remember"], 0);
        assert!(metrics.get_timings("recall").is_empty());
    }

    #[test]
    fn logging_ring_buffer_bounds_and_previews() {
        let logging = LoggingPlugin::new(2);
        logging.on_remember(&json!({"id": "m1", "content": "x".repeat(100)}));
        logging.on_recall(&json!({"id": "m2", "content": "short"}));
        logging.on_invalidate("m3");
        let log = logging.get_log();
        assert_eq!(log.len(), 2, "oldest entry evicted at max_entries=2");
        assert_eq!(log[0]["event"], "recall");
        assert_eq!(log[1]["event"], "invalidate");
        logging.on_remember(&json!({"id": "m4", "content": "y".repeat(100)}));
        let log = logging.get_log();
        let previewed = log[1]["content_preview"].as_str().unwrap();
        assert_eq!(previewed.chars().count(), 83, "80 chars + ellipsis");
        logging.clear_log();
        assert!(logging.get_log().is_empty());
    }

    #[test]
    fn filter_blocks_and_tracks() {
        let filter = FilterPlugin::default();
        filter.add_rule(Box::new(|m: &Value| {
            m.get("content")
                .and_then(Value::as_str)
                .map(|c| !c.contains("secret"))
                .unwrap_or(true)
        }));
        filter.on_remember(&json!({"id": "ok", "content": "fine"}));
        filter.on_remember(&json!({"id": "bad", "content": "a secret thing"}));
        assert!(!filter.is_blocked("ok"));
        assert!(filter.is_blocked("bad"));
        assert_eq!(filter.get_blocked().len(), 1);
    }

    #[test]
    fn compression_disabled_by_default_and_identity() {
        let c = CompressionPlugin::default();
        assert!(!c.enabled());
        c.set_enabled(true);
        assert!(c.enabled());
        let lines = vec!["a".to_string(), "b".repeat(50)];
        assert_eq!(c.compress_lines(lines.clone()), lines);
    }

    #[test]
    fn unload_stops_notifications() {
        let mgr = PluginManager::new();
        mgr.load_all();
        mgr.notify_remember(&json!({"id": "m1", "content": "one"}));
        assert!(mgr.unload_plugin("metrics"));
        mgr.notify_remember(&json!({"id": "m2", "content": "two"}));
        let metrics = mgr.get_plugin("metrics").unwrap(); // re-loads lazily
        assert_eq!(metrics.to_dict()["name"], "metrics");
        assert!(!mgr.unload_plugin("never-registered"));
    }
}
