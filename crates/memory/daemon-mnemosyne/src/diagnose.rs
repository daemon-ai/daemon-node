// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! PII-safe diagnostics — port of `mnemosyne/diagnose.py`.
//!
//! Runs a full health scan, appends a JSONL log under `<data_dir>/logs/` (Python:
//! `~/.hermes/mnemosyne/logs/diagnose_*.jsonl`; the Rust node keeps logs bank-adjacent because
//! hosts own path policy), and returns a summary with key findings. Never includes memory
//! content, queries, or secret values — env vars are reported presence-only.
//!
//! The Python check catalog maps onto the Rust build like so:
//! - **`deps` / pip checks** (`fastembed`, `sqlite_vec`, `numpy`, `huggingface_hub`,
//!   `ctransformers`): Rust has no runtime-installable dependencies. The moral equivalents are
//!   injection/compile facts: is a host [`Embedder`] wired (vs the deterministic hash fallback),
//!   is the `vec-ext` feature compiled (parity tests only per the §7 storage decision), is an
//!   LLM [`Extractor`] injected (optional, like `ctransformers`).
//! - **`vec_working` coverage/repair**: remapped to the stores the Rust engine actually reads —
//!   `memory_embeddings` (f32 JSON) + `episodic_memory.binary_vector` (MIB), via
//!   [`Engine::vector_coverage`] / [`Engine::repair_vector_coverage`]. The tool argument keeps
//!   Python's wire name (`repair_vec_working`) so existing clients work unchanged.
//! - **`auto_fix`**: ported shape-compatibly ({fixed, failed, skipped, ran}), but nothing is
//!   installable at runtime — every fixable finding lands in `skipped` with its host-side
//!   remediation. pip subprocesses have no place in a daemon node.

use crate::embeddings::Embedder;
use crate::engine::Engine;
use crate::extract::Extractor;
use serde_json::{json, Value};
use std::io::Write;

/// Options for [`run_diagnostics`] (`diagnose.py` `run_diagnostics` kwargs; the tool arg keeps
/// the Python wire name `repair_vec_working`).
#[derive(Clone, Copy, Debug, Default)]
pub struct DiagnoseOptions {
    /// Idempotently backfill the deterministic vector gap (episodic MIB binaries) after the scan.
    pub repair_vec_working: bool,
    /// With `repair_vec_working`, report what would be repaired without writing.
    pub dry_run: bool,
}

/// Env var presence indicator, never the value (`_safe_env` L59-L62).
fn safe_env(name: &str) -> &'static str {
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => "set",
        _ => "unset",
    }
}

/// One diagnostic entry (`run_diagnostics` `log()` closure shape).
fn entry(category: &str, check: &str, status: impl ToString, detail: &str) -> Value {
    let mut e = json!({
        "ts": crate::util::now_iso(),
        "category": category,
        "check": check,
        "status": status.to_string(),
        "detail": detail,
    });
    if detail.is_empty() {
        e["detail"] = json!("");
    }
    e
}

/// Run the full diagnostic scan and write the PII-safe JSONL log; returns the summary
/// (`run_diagnostics` L65-L291). In-memory banks skip the log write (`log_path: null`).
pub fn run_diagnostics(
    engine: &Engine,
    embedder: &Embedder,
    extractor: &Extractor,
    opts: DiagnoseOptions,
) -> Value {
    let mut entries: Vec<Value> = Vec::new();
    let mut log = |cat: &str, check: &str, status: &dyn ToString, detail: &str| {
        entries.push(entry(cat, check, status.to_string(), detail));
    };

    // ── Environment (Python: python_version/platform/executable) ──
    log("env", "crate_version", &env!("CARGO_PKG_VERSION"), "");
    log(
        "env",
        "platform",
        &format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        "",
    );

    // ── Package ──
    log(
        "package",
        "mnemosyne_version",
        &env!("CARGO_PKG_VERSION"),
        "",
    );

    // ── Capability checks (Python's pip-dependency section, remapped to injection/compile facts) ──
    if embedder.available() {
        let detail = embedder
            .model()
            .map(|m| format!("model={m}"))
            .unwrap_or_default();
        log("deps", "embedder", &"OK", &detail);
    } else {
        log(
            "deps",
            "embedder",
            &"MISSING",
            "no embedding provider injected; deterministic hash fallback active",
        );
    }
    if cfg!(feature = "vec-ext") {
        log(
            "deps",
            "sqlite_vec",
            &"OK",
            "vec-ext compiled (parity tests only; runtime path is f32-BLOB + scalar cosine)",
        );
    } else {
        log(
            "deps",
            "sqlite_vec",
            &"OPTIONAL",
            "vec-ext feature not compiled; the §7 runtime path needs no extension",
        );
    }
    if extractor.available() {
        log("deps", "llm_extractor", &"OK", "");
    } else {
        log(
            "deps",
            "llm_extractor",
            &"OPTIONAL",
            "no LLM provider injected; regex extraction baseline active",
        );
    }

    // ── Core components (Python: embeddings_available / sqlite_vec_available) ──
    log(
        "core",
        "embeddings_available",
        if embedder.available() { &"YES" } else { &"NO" },
        "",
    );
    if let Some(m) = embedder.model() {
        log("core", "embeddings_model", &m, "");
    }

    // ── Database state (counts and config only, never content) ──
    match engine.stats() {
        Ok(stats) => {
            log("db", "working_total", &stats.working, "");
            log("db", "episodic_total", &stats.episodic, "");
        }
        Err(e) => log("db", "stats", &"ERROR", &e.to_string()),
    }
    match engine.diagnose() {
        Ok(d) => {
            log("db", "episodic_vectors", &d.embedded_episodic, "");
            log(
                "db",
                "episodic_vec_type",
                &if d.embedded_episodic > 0 {
                    "binary"
                } else {
                    "none"
                },
                "",
            );
        }
        Err(e) => log("db", "diagnose", &"ERROR", &e.to_string()),
    }
    log(
        "db",
        "db_path",
        &if engine.is_persistent() {
            engine.config().bank_db_path().display().to_string()
        } else {
            ":memory:".to_string()
        },
        "",
    );

    // ── Vector coverage (Python's vec_working block, remapped per §7) ──
    let coverage = if opts.repair_vec_working {
        match engine.repair_vector_coverage(opts.dry_run) {
            Ok(repair) => {
                log(
                    "db",
                    "vec_working_repair_status",
                    &repair["status"].as_str().unwrap_or("unknown"),
                    "",
                );
                log("db", "vec_working_repair_inserted", &repair["inserted"], "");
                Some(repair["after"].clone())
            }
            Err(e) => {
                log("db", "vec_working_repair", &"ERROR", &e.to_string());
                None
            }
        }
    } else {
        match engine.vector_coverage() {
            Ok(c) => Some(c),
            Err(e) => {
                log("db", "vec_working_coverage", &"ERROR", &e.to_string());
                None
            }
        }
    };
    if let Some(after) = &coverage {
        log(
            "db",
            "vec_working_status",
            &after["status"].as_str().unwrap_or("unknown"),
            "",
        );
        log("db", "vec_working_rows", &after["episodic_binary_rows"], "");
        log(
            "db",
            "vec_working_missing",
            &after["missing_episodic_binary"],
            "",
        );
        log(
            "db",
            "vec_working_orphans",
            &after["orphan_embedding_rows"],
            "",
        );
        log(
            "db",
            "working_embedding_rows",
            &after["working_embedding_rows"],
            "",
        );
    }

    // ── Env var presence (Python's list; values never logged) ──
    for var in [
        "MNEMOSYNE_DATA_DIR",
        "MNEMOSYNE_LLM_ENABLED",
        "MNEMOSYNE_LLM_BASE_URL",
        "MNEMOSYNE_VEC_TYPE",
        "MNEMOSYNE_WM_MAX_ITEMS",
        "HERMES_HOME",
    ] {
        log("env", var, &safe_env(var), "");
    }

    // ── Write the JSONL log (skipped for in-memory banks) ──
    let log_path = if engine.is_persistent() {
        write_jsonl(engine, &entries)
    } else {
        None
    };

    // ── Summary (`run_diagnostics` L225-L291) ──
    let non_failure = ["OK", "YES", "set", "OPTIONAL"];
    let failure = ["MISSING", "NO", "ERROR"];
    let status_of = |e: &Value| e["status"].as_str().unwrap_or("").to_string();
    let mut findings: Vec<Value> = Vec::new();
    let mut fixable: Vec<Value> = Vec::new();

    let embed_ok = embedder.available();
    let ep_vectors = entries
        .iter()
        .find(|e| e["check"] == "episodic_vectors")
        .and_then(|e| e["status"].as_str().and_then(|s| s.parse::<i64>().ok()))
        .unwrap_or(0);
    if !embed_ok {
        findings.push(json!(
            "no embedding provider injected - recall runs on the deterministic hash fallback; \
             wire a host EmbeddingProvider for semantic quality"
        ));
        fixable.push(json!("embedder"));
    }
    if embed_ok && ep_vectors == 0 {
        findings.push(json!(
            "embedder is available but episodic vectors=0 - memories may not have been \
             consolidated yet. Run: mnemosyne_sleep"
        ));
    }
    if ep_vectors > 0 {
        findings.push(json!(format!(
            "Semantic search is active with {ep_vectors} vectors in episodic memory \
             (backend: binary)"
        )));
    }
    if opts.repair_vec_working {
        if let Some(repair_status) = entries
            .iter()
            .find(|e| e["check"] == "vec_working_repair_status")
            .and_then(|e| e["status"].as_str())
        {
            let inserted = entries
                .iter()
                .find(|e| e["check"] == "vec_working_repair_inserted")
                .and_then(|e| e["status"].as_str())
                .unwrap_or("0")
                .to_string();
            let action = if opts.dry_run {
                "would insert"
            } else {
                "inserted"
            };
            findings.push(json!(format!(
                "vector repair {repair_status}: {action} {inserted} rows"
            )));
        }
    }
    if let Some(after) = &coverage {
        let status = after["status"].as_str().unwrap_or("unknown");
        let missing = after["missing_episodic_binary"].as_i64().unwrap_or(0);
        if status == "complete" {
            findings.push(json!(format!(
                "Vector coverage complete: binary rows={}, fallback embeddings={}",
                after["episodic_binary_rows"], after["working_embedding_rows"]
            )));
        } else if missing > 0 {
            findings.push(json!(format!(
                "{missing} episodic rows have embeddings but no MIB binary vector - run \
                 mnemosyne_diagnose with repair_vec_working=true"
            )));
        }
    }
    json!({
        "log_path": log_path,
        "checks_total": entries.len(),
        "checks_passed": entries.iter().filter(|e| non_failure.contains(&status_of(e).as_str())).count(),
        "checks_failed": entries.iter().filter(|e| failure.contains(&status_of(e).as_str())).count(),
        "key_findings": findings,
        "fixable": fixable,
        "entries": entries,
    })
}

/// Append the entries as one JSONL file under `<data_dir>/logs/` (`_log_path` L53-L56).
fn write_jsonl(engine: &Engine, entries: &[Value]) -> Option<String> {
    let dir = engine.config().data_dir.join("logs");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::debug!(error = %e, "diagnose log dir create failed");
        return None;
    }
    let ts = chrono::Local::now().format("%Y-%m-%d_%H%M%S");
    let path = dir.join(format!("diagnose_{ts}.jsonl"));
    let mut file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            tracing::debug!(error = %e, "diagnose log create failed");
            return None;
        }
    };
    for e in entries {
        if writeln!(file, "{e}").is_err() {
            return None;
        }
    }
    Some(path.display().to_string())
}

/// Shape-compatible port of `auto_fix` L294-L336. Rust has no pip: every fixable finding is
/// reported under `skipped` with its host-side remediation (injection or compile flag), never
/// executed. `dry_run` keeps Python's "WOULD ..." phrasing under `fixed` for CLI parity.
pub fn auto_fix(summary: &Value, dry_run: bool) -> Value {
    let mut fixed: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let remediations = [(
        "embedder",
        "inject a host EmbeddingProvider (Embedder::with_provider) - no runtime install exists",
    )];
    let fixable: Vec<String> = summary["fixable"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    for (key, remedy) in remediations {
        if !fixable.iter().any(|f| f == key) {
            continue;
        }
        if dry_run {
            fixed.push(format!("WOULD fix: {key} ({remedy})"));
        } else {
            skipped.push(format!("{key}: {remedy}"));
        }
    }
    json!({"fixed": fixed, "failed": [], "skipped": skipped, "ran": true})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MnemosyneConfig;

    fn engine() -> Engine {
        Engine::open_in_memory(MnemosyneConfig::default()).expect("engine")
    }

    #[test]
    fn scan_reports_counts_and_findings_without_logging_in_memory() {
        let e = engine();
        e.remember("Maya works at Acme", &Default::default())
            .expect("remember");
        let summary = run_diagnostics(
            &e,
            &Embedder::new(),
            &Extractor::new(),
            DiagnoseOptions::default(),
        );
        assert!(
            summary["log_path"].is_null(),
            "in-memory bank writes no log"
        );
        assert!(summary["checks_total"].as_u64().unwrap() > 10);
        // No embedder injected -> fixable finding present.
        assert!(summary["fixable"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f == "embedder"));
        let entries = summary["entries"].as_array().unwrap();
        let check = |name: &str| -> String {
            entries
                .iter()
                .find(|e| e["check"] == name)
                .map(|e| e["status"].as_str().unwrap_or_default().to_string())
                .unwrap_or_default()
        };
        assert_eq!(check("working_total"), "1");
        assert_eq!(check("db_path"), ":memory:");
        assert_eq!(check("embedder"), "MISSING");
        assert_eq!(check("llm_extractor"), "OPTIONAL");
        // Value-statuses count neither as passed nor failed (Python semantics).
        let passed = summary["checks_passed"].as_u64().unwrap();
        let failed = summary["checks_failed"].as_u64().unwrap();
        assert!(passed + failed < summary["checks_total"].as_u64().unwrap());
    }

    #[test]
    fn repair_backfills_missing_episodic_binaries() {
        let e = engine();
        // Manufacture the gap a legacy/interrupted consolidation leaves: an episodic row with a
        // stored f32 embedding but no MIB binary.
        e.remember("gap row", &Default::default())
            .expect("remember");
        {
            let stats = e.stats().expect("stats");
            assert_eq!(stats.working, 1);
        }
        e.insert_episodic_embedding_gap_for_test("ep-gap", "an episode missing its binary");

        let before = e.vector_coverage().expect("coverage");
        assert_eq!(before["missing_episodic_binary"], 1);
        assert_eq!(before["status"], "partial");

        // Dry run: reports, writes nothing.
        let dry = e.repair_vector_coverage(true).expect("dry");
        assert_eq!(dry["status"], "dry_run");
        assert_eq!(dry["would_insert"], 1);
        assert_eq!(
            e.vector_coverage().expect("coverage")["missing_episodic_binary"],
            1
        );

        // Real repair closes the gap idempotently.
        let fixed = e.repair_vector_coverage(false).expect("repair");
        assert_eq!(fixed["status"], "repaired");
        assert_eq!(fixed["inserted"], 1);
        assert_eq!(fixed["after"]["missing_episodic_binary"], 0);
        assert_eq!(fixed["after"]["status"], "complete");
        let again = e.repair_vector_coverage(false).expect("repair again");
        assert_eq!(again["inserted"], 0);
    }

    #[test]
    fn auto_fix_is_shape_compatible_and_never_executes() {
        let e = engine();
        let summary = run_diagnostics(
            &e,
            &Embedder::new(),
            &Extractor::new(),
            DiagnoseOptions::default(),
        );
        let dry = auto_fix(&summary, true);
        assert_eq!(dry["ran"], true);
        assert!(dry["fixed"][0]
            .as_str()
            .unwrap()
            .starts_with("WOULD fix: embedder"));
        let real = auto_fix(&summary, false);
        assert!(real["fixed"].as_array().unwrap().is_empty());
        assert!(real["skipped"][0]
            .as_str()
            .unwrap()
            .starts_with("embedder:"));
        assert!(real["failed"].as_array().unwrap().is_empty());
    }
}
