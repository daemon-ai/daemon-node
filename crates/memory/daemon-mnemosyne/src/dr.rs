// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Disaster recovery — port of `mnemosyne/dr/recovery.py`.
//!
//! Backup archives are gzipped SQL text dumps (`mnemosyne_backup_YYYYMMDD_HHMMSS.db.gz` plus a
//! `.json` metadata sidecar), the same wire format Python produces via `iterdump`. Restore
//! executes the archived SQL into a fresh database — the same semantics as Python's
//! `executescript` — so any archive Python's restore accepts, this restore accepts.
//!
//! Divergences from the Python module (all deliberate):
//! - **Paths are explicit.** Python defaults to `~/.mnemosyne/{data,backups}`; the daemon node
//!   never touches `$HOME` — hosts pass `db_path`/`backup_dir` (see [`default_backup_dir`]).
//! - **The dump is schema-aware where CPython's `iterdump` is broken.** `iterdump` emits FTS5
//!   shadow tables (CPython gh-90016), producing archives that fail their own restore on any
//!   FTS-bearing bank. This dump skips virtual-table shadows, orders triggers *after* data (so
//!   restore doesn't double-fire them), rebuilds the FTS indexes at the end, and embeds
//!   `PRAGMA user_version` so the rusqlite-migration ladder doesn't re-run on a restored bank.
//! - **Restore is atomic.** Python rebuilds in `:memory:` and overwrites the target in place;
//!   here the restored database is built as a sibling temp file, integrity-checked, and renamed
//!   over the target (WAL sidecars removed) — a crash mid-restore never leaves a torn bank. The
//!   pre-restore safety copy keeps Python's `.emergency_backup.db` name.
//! - **[`emergency_restore`] reports the real attempt count** (Python hardcodes `"attempts": 1`).

use crate::error::{Error, Result};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use rusqlite::Connection;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// The bank-adjacent backup directory (`get_default_paths` L18-L23, re-rooted at the host's
/// `data_dir` instead of `~/.mnemosyne`).
pub fn default_backup_dir(config: &crate::config::MnemosyneConfig) -> PathBuf {
    config.data_dir.join("backups")
}

/// First 16 hex chars of the SHA-256 of `path`'s bytes (`create_backup` L67-L68).
fn checksum16(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let digest = Sha256::digest(&bytes);
    Ok(format!("{digest:x}")[..16].to_string())
}

/// Quote an identifier for SQL text.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Encode one column value as a SQL literal (the `iterdump` row encoding).
fn sql_literal(v: rusqlite::types::ValueRef<'_>) -> String {
    use rusqlite::types::ValueRef;
    match v {
        ValueRef::Null => "NULL".to_string(),
        ValueRef::Integer(i) => i.to_string(),
        ValueRef::Real(f) => {
            // `{}` on f64 prints the shortest round-trip representation, but bare integral
            // floats ("1") would re-enter SQLite as INTEGER; keep them REAL.
            let s = f.to_string();
            if s.contains(['.', 'e', 'E']) || s.contains("inf") || s.contains("NaN") {
                s
            } else {
                format!("{s}.0")
            }
        }
        ValueRef::Text(t) => format!("'{}'", String::from_utf8_lossy(t).replace('\'', "''")),
        ValueRef::Blob(b) => {
            let mut hex = String::with_capacity(b.len() * 2 + 3);
            hex.push_str("X'");
            for byte in b {
                hex.push_str(&format!("{byte:02X}"));
            }
            hex.push('\'');
            hex
        }
    }
}

/// Produce the executescript-compatible SQL dump of an open connection: real tables + rows,
/// virtual-table DDL (shadows skipped), then indexes/triggers/views, then FTS
/// rebuild/repopulate, then the `user_version` stamp.
fn dump_sql(conn: &Connection) -> Result<String> {
    let mut out = String::from("BEGIN TRANSACTION;\n");

    let tables: Vec<(String, String)> = {
        let mut stmt = conn.prepare(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL \
             AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    let virtuals: Vec<String> = tables
        .iter()
        .filter(|(_, sql)| {
            sql.trim_start()
                .to_uppercase()
                .starts_with("CREATE VIRTUAL")
        })
        .map(|(name, _)| name.clone())
        .collect();
    // Shadow tables (FTS5 `x_data`/`x_idx`/..., vec0 `x_chunks`/...) are owned by their module:
    // restoring them alongside the virtual-table DDL is the CPython gh-90016 failure.
    let is_shadow = |name: &str| {
        virtuals
            .iter()
            .any(|v| name.len() > v.len() && name.starts_with(&format!("{v}_")))
    };

    for (name, sql) in &tables {
        if is_shadow(name) {
            continue;
        }
        out.push_str(sql);
        out.push_str(";\n");
        if virtuals.contains(name) {
            continue;
        }
        let quoted = quote_ident(name);
        let mut stmt = conn.prepare(&format!("SELECT * FROM {quoted}"))?;
        let ncols = stmt.column_count();
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let mut values = Vec::with_capacity(ncols);
            for i in 0..ncols {
                values.push(sql_literal(row.get_ref(i)?));
            }
            out.push_str(&format!(
                "INSERT INTO {quoted} VALUES({});\n",
                values.join(",")
            ));
        }
    }

    let others: Vec<String> = {
        let mut stmt = conn.prepare(
            "SELECT sql FROM sqlite_master WHERE type IN ('index','trigger','view') \
             AND sql IS NOT NULL ORDER BY type, name",
        )?;
        let rows = stmt.query_map([], |r| r.get(0))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    for sql in others {
        out.push_str(&sql);
        out.push_str(";\n");
    }

    // FTS indexes were bypassed above (data landed before the sync triggers existed): rebuild
    // external-content tables from their content source, repopulate the regular one.
    for v in &virtuals {
        match v.as_str() {
            "fts_episodes" | "fts_facts" => {
                out.push_str(&format!("INSERT INTO {v}({v}) VALUES('rebuild');\n"));
            }
            "fts_working" => out.push_str(
                "INSERT INTO fts_working(id, content) SELECT id, content FROM working_memory;\n",
            ),
            _ => {}
        }
    }

    let user_version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
    out.push_str(&format!("PRAGMA user_version = {user_version};\n"));
    out.push_str("COMMIT;\n");
    Ok(out)
}

/// Create a compressed backup of the bank (`create_backup` L26-L89). Returns the metadata dict
/// (backup/metadata paths, sizes, truncated checksums, timestamp).
pub fn create_backup(db_path: &Path, backup_dir: &Path) -> Result<Value> {
    if !db_path.exists() {
        return Err(Error::Invalid(format!(
            "database not found: {}",
            db_path.display()
        )));
    }
    std::fs::create_dir_all(backup_dir)?;

    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let backup_path = backup_dir.join(format!("mnemosyne_backup_{timestamp}.db.gz"));

    // Reading through a connection (not copying file bytes) is WAL-aware and yields a
    // consistent snapshot — the Python module's `sqlite3.backup()` rationale.
    let sql = {
        let conn =
            Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        dump_sql(&conn)?
    };
    let mut encoder = GzEncoder::new(
        std::fs::File::create(&backup_path)?,
        flate2::Compression::default(),
    );
    encoder.write_all(sql.as_bytes())?;
    encoder.finish()?;

    let metadata = json!({
        "timestamp": timestamp,
        "original_size": std::fs::metadata(db_path)?.len(),
        "backup_size": std::fs::metadata(&backup_path)?.len(),
        "db_checksum": checksum16(db_path)?,
        "backup_checksum": checksum16(&backup_path)?,
        "compressed": true,
    });
    let meta_path = backup_dir.join(format!("mnemosyne_backup_{timestamp}.db.gz.json"));
    std::fs::write(&meta_path, serde_json::to_string_pretty(&metadata)?)?;

    let mut result = metadata;
    result["backup_path"] = json!(backup_path.display().to_string());
    result["metadata_path"] = json!(meta_path.display().to_string());
    Ok(result)
}

/// Restore a bank from a compressed backup (`restore_backup` L92-L136): safety-copy the current
/// database, rebuild the archive SQL into a sibling temp file, integrity-check it, and atomically
/// rename it over the target.
pub fn restore_backup(backup_path: &Path, db_path: &Path) -> Result<Value> {
    if !backup_path.exists() {
        return Err(Error::Invalid(format!(
            "backup not found: {}",
            backup_path.display()
        )));
    }
    if db_path.exists() {
        let emergency = db_path.with_extension("emergency_backup.db");
        std::fs::copy(db_path, emergency)?;
    }
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut sql = String::new();
    GzDecoder::new(std::fs::File::open(backup_path)?).read_to_string(&mut sql)?;

    let tmp_path = db_path.with_extension("restore_tmp.db");
    let _ = std::fs::remove_file(&tmp_path);
    {
        // Auto-register sqlite-vec (vec-ext builds) so archives of banks that used vec0 tables
        // restore; without the feature those archives fail here, exactly like Python without
        // the sqlite_vec package.
        crate::store::register_vec_extension();
        let conn = Connection::open(&tmp_path)?;
        conn.execute_batch(&sql)?;
    }
    let is_valid = verify_integrity(&tmp_path);

    // Replace the target only once the rebuilt bank checks out; stale WAL sidecars of the old
    // database must not survive the swap.
    let _ = std::fs::remove_file(db_path);
    for suffix in ["-wal", "-shm"] {
        let _ = std::fs::remove_file(PathBuf::from(format!("{}{suffix}", db_path.display())));
    }
    std::fs::rename(&tmp_path, db_path)?;

    Ok(json!({
        "restored": true,
        "backup_used": backup_path.display().to_string(),
        "database_path": db_path.display().to_string(),
        "integrity_check": is_valid,
    }))
}

/// All backup archives under `backup_dir`, newest-first by name.
fn backup_files(backup_dir: &Path) -> Vec<PathBuf> {
    let Ok(read) = std::fs::read_dir(backup_dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("mnemosyne_backup_") && n.ends_with(".db.gz"))
        })
        .collect();
    files.sort();
    files.reverse();
    files
}

/// Restore from the most recent backup whose rebuild passes the integrity check, trying older
/// archives on failure (`emergency_restore` L139-L169).
pub fn emergency_restore(backup_dir: &Path, db_path: &Path) -> Result<Value> {
    let backups = backup_files(backup_dir);
    if backups.is_empty() {
        return Err(Error::Invalid(format!(
            "no backups found in {}",
            backup_dir.display()
        )));
    }
    for (i, backup) in backups.iter().enumerate() {
        match restore_backup(backup, db_path) {
            Ok(result) if result["integrity_check"] == json!(true) => {
                return Ok(json!({
                    "restored": true,
                    "backup_used": backup.display().to_string(),
                    "attempts": i + 1,
                }));
            }
            Ok(_) | Err(_) => continue,
        }
    }
    Err(Error::Invalid(
        "all backups failed integrity check".to_string(),
    ))
}

/// `PRAGMA integrity_check` == `ok` (`verify_integrity` L172-L199). `false` for a missing or
/// unopenable file.
pub fn verify_integrity(db_path: &Path) -> bool {
    if !db_path.exists() {
        return false;
    }
    let Ok(conn) = Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
    else {
        return false;
    };
    conn.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))
        .map(|s| s == "ok")
        .unwrap_or(false)
}

/// All backups with their metadata sidecars, newest first (`list_backups` L202-L229).
pub fn list_backups(backup_dir: &Path) -> Vec<Value> {
    backup_files(backup_dir)
        .into_iter()
        .map(|file| {
            let mut info = json!({
                "file": file.display().to_string(),
                "name": file.file_name().and_then(|n| n.to_str()).unwrap_or_default(),
                "size": std::fs::metadata(&file).map(|m| m.len()).unwrap_or(0),
                "modified": std::fs::metadata(&file)
                    .and_then(|m| m.modified())
                    .ok()
                    .map(|t| chrono::DateTime::<chrono::Local>::from(t).to_rfc3339())
                    .unwrap_or_default(),
            });
            let meta_path = PathBuf::from(format!("{}.json", file.display()));
            if let Ok(raw) = std::fs::read_to_string(meta_path) {
                if let Ok(meta) = serde_json::from_str::<Value>(&raw) {
                    info["metadata"] = meta;
                }
            }
            info
        })
        .collect()
}

/// Keep only the most recent `keep` backups, deleting older archives + sidecars
/// (`rotate_backups` L232-L263).
pub fn rotate_backups(backup_dir: &Path, keep: usize) -> Result<Value> {
    let mut backups = backup_files(backup_dir);
    backups.reverse(); // oldest first
    let total = backups.len();
    let n_delete = total.saturating_sub(keep);
    let mut deleted_files = Vec::new();
    for backup in backups.into_iter().take(n_delete) {
        std::fs::remove_file(&backup)?;
        let meta = PathBuf::from(format!("{}.json", backup.display()));
        let _ = std::fs::remove_file(meta);
        deleted_files.push(
            backup
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string(),
        );
    }
    Ok(json!({
        "total_backups": total,
        "kept": keep,
        "deleted": deleted_files.len(),
        "deleted_files": deleted_files,
    }))
}

/// Database + backup health summary (`health_check` L266-L295).
pub fn health_check(db_path: &Path, backup_dir: &Path) -> Value {
    let db_exists = db_path.exists();
    let db_valid = db_exists && verify_integrity(db_path);
    let backups = backup_files(backup_dir);
    json!({
        "database": {
            "exists": db_exists,
            "valid": db_valid,
            "path": db_path.display().to_string(),
            "message": if db_valid { "Database integrity verified" } else { "Database missing or corrupt" },
        },
        "backups": {
            "total": backups.len(),
            "latest": backups.first().map(|p| p.display().to_string()),
            "directory": backup_dir.display().to_string(),
        },
        "status": if db_valid { "healthy" } else { "unhealthy" },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MnemosyneConfig;
    use crate::engine::Engine;

    fn bank(dir: &Path) -> (Engine, PathBuf) {
        let config = MnemosyneConfig {
            data_dir: dir.to_path_buf(),
            ..Default::default()
        };
        let db_path = config.bank_db_path();
        (Engine::open(config).expect("engine"), db_path)
    }

    #[test]
    fn backup_and_restore_round_trip_preserves_rows_and_fts() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let backups = tmp.path().join("backups");
        let db_path;
        {
            let (engine, path) = bank(tmp.path());
            db_path = path;
            engine
                .remember("Maya moved to Lisbon in March", &Default::default())
                .expect("remember");
            engine
                .remember("the reactor design was approved", &Default::default())
                .expect("remember");
        }

        let result = create_backup(&db_path, &backups).expect("backup");
        assert_eq!(result["compressed"], true);
        let backup_path = PathBuf::from(result["backup_path"].as_str().unwrap());
        assert!(backup_path.exists());
        assert!(PathBuf::from(result["metadata_path"].as_str().unwrap()).exists());
        assert_eq!(result["db_checksum"].as_str().unwrap().len(), 16);

        // Lose the bank entirely, then restore.
        std::fs::remove_file(&db_path).unwrap();
        let restored = restore_backup(&backup_path, &db_path).expect("restore");
        assert_eq!(restored["restored"], true);
        assert_eq!(restored["integrity_check"], true);

        // The restored bank opens through the normal ladder (user_version preserved) and both
        // rows + the FTS index survived.
        let (engine, _) = bank(tmp.path());
        let stats = engine.stats().expect("stats");
        assert_eq!(stats.working, 2);
        let hits = engine.recall("Lisbon", 5).expect("recall");
        assert!(
            hits.iter().any(|h| h.content.contains("Lisbon")),
            "FTS must find the restored row: {hits:?}"
        );
    }

    #[test]
    fn restore_writes_emergency_copy_of_current_bank() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let backups = tmp.path().join("backups");
        let db_path;
        {
            let (engine, path) = bank(tmp.path());
            db_path = path;
            engine
                .remember("original row", &Default::default())
                .expect("remember");
        }
        let result = create_backup(&db_path, &backups).expect("backup");
        let backup_path = PathBuf::from(result["backup_path"].as_str().unwrap());
        restore_backup(&backup_path, &db_path).expect("restore");
        assert!(
            db_path.with_extension("emergency_backup.db").exists(),
            "pre-restore safety copy must exist"
        );
    }

    #[test]
    fn emergency_restore_skips_corrupt_newest_backup() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let backups = tmp.path().join("backups");
        let db_path;
        {
            let (engine, path) = bank(tmp.path());
            db_path = path;
            engine
                .remember("survivor row", &Default::default())
                .expect("remember");
        }
        let result = create_backup(&db_path, &backups).expect("backup");
        let good = PathBuf::from(result["backup_path"].as_str().unwrap());

        // A newer archive that is garbage (not gzip).
        let bad = backups.join("mnemosyne_backup_99991231_235959.db.gz");
        std::fs::write(&bad, b"not a gzip archive").unwrap();

        let out = emergency_restore(&backups, &db_path).expect("emergency restore");
        assert_eq!(out["restored"], true);
        assert_eq!(out["attempts"], 2, "newest failed, older succeeded");
        assert_eq!(
            out["backup_used"].as_str().unwrap(),
            good.display().to_string()
        );

        let (engine, _) = bank(tmp.path());
        assert_eq!(engine.stats().expect("stats").working, 1);
    }

    #[test]
    fn rotate_keeps_most_recent_and_lists_are_newest_first() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let backups = tmp.path().join("backups");
        std::fs::create_dir_all(&backups).unwrap();
        for i in 0..12 {
            let name = backups.join(format!("mnemosyne_backup_202601{:02}_000000.db.gz", i + 1));
            std::fs::write(&name, b"x").unwrap();
            std::fs::write(format!("{}.json", name.display()), "{}").unwrap();
        }
        let listed = list_backups(&backups);
        assert_eq!(listed.len(), 12);
        assert!(
            listed[0]["name"].as_str().unwrap().contains("20260112"),
            "newest first"
        );
        assert!(listed[0]["metadata"].is_object());

        let out = rotate_backups(&backups, 10).expect("rotate");
        assert_eq!(out["total_backups"], 12);
        assert_eq!(out["deleted"], 2);
        assert_eq!(
            out["deleted_files"][0].as_str().unwrap(),
            "mnemosyne_backup_20260101_000000.db.gz"
        );
        assert_eq!(list_backups(&backups).len(), 10);
        assert_eq!(
            std::fs::read_dir(&backups).unwrap().count(),
            20,
            "sidecars rotate with their archives"
        );
    }

    #[test]
    fn verify_and_health_report_corruption() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let backups = tmp.path().join("backups");
        let missing = tmp.path().join("nope.db");
        assert!(!verify_integrity(&missing));
        let health = health_check(&missing, &backups);
        assert_eq!(health["status"], "unhealthy");
        assert_eq!(health["database"]["exists"], false);

        let (engine, db_path) = bank(tmp.path());
        engine
            .remember("healthy row", &Default::default())
            .expect("remember");
        create_backup(&db_path, &backups).expect("backup");
        drop(engine);
        let health = health_check(&db_path, &backups);
        assert_eq!(health["status"], "healthy");
        assert_eq!(health["backups"]["total"], 1);
        assert!(health["backups"]["latest"].is_string());
    }

    /// The compatibility contract: an archive in Python's format (a plain `iterdump`-style SQL
    /// script of an old-shape Python bank, gzipped) restores through the same path, and the
    /// legacy reconcile ladder then upgrades the shape on open.
    #[test]
    fn python_style_dump_restores_and_reconciles() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let sql = "BEGIN TRANSACTION;\n\
                   CREATE TABLE working_memory (\n\
                       id TEXT PRIMARY KEY,\n\
                       content TEXT NOT NULL,\n\
                       source TEXT,\n\
                       timestamp TEXT,\n\
                       session_id TEXT DEFAULT 'default',\n\
                       importance REAL DEFAULT 0.5,\n\
                       metadata_json TEXT,\n\
                       veracity TEXT DEFAULT 'unknown',\n\
                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n\
                   );\n\
                   INSERT INTO working_memory VALUES('py1','a python era memory','user',\
                   '2024-05-01T00:00:00','default',0.5,NULL,'unknown','2024-05-01 00:00:00');\n\
                   COMMIT;\n";
        let archive = tmp.path().join("mnemosyne_backup_20240501_000000.db.gz");
        let mut enc = GzEncoder::new(
            std::fs::File::create(&archive).unwrap(),
            flate2::Compression::default(),
        );
        enc.write_all(sql.as_bytes()).unwrap();
        enc.finish().unwrap();

        let config = MnemosyneConfig {
            data_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let db_path = config.bank_db_path();
        let restored = restore_backup(&archive, &db_path).expect("restore python dump");
        assert_eq!(restored["integrity_check"], true);

        // Opening runs reconcile_columns + the ladder over the restored legacy shape.
        let engine = Engine::open(config).expect("open restored python bank");
        let row = engine.get("py1").expect("get").expect("row present");
        assert_eq!(row.content, "a python era memory");
    }
}
