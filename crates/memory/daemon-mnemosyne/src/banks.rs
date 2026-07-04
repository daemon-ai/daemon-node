// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Memory-bank isolation — port of `banks.py`.
//!
//! Named banks are fully isolated at the database level: each bank is a self-contained directory
//! with its own SQLite file under `<data_dir>/banks/<name>/mnemosyne.db`, while the `default`
//! bank uses the legacy path `<data_dir>/mnemosyne.db` (`banks.py` L123-L132 — the same
//! resolution as [`crate::MnemosyneConfig::bank_db_path`]). Unlike Python there is no
//! env-derived default root: the host injects `data_dir`, exactly as it does for the engine.

use crate::error::{Error, Result};
use std::path::{Path, PathBuf};

/// Maximum bank-name length (`banks.py` L184).
pub const MAX_BANK_NAME_LEN: usize = 64;

/// Statistics for one bank (`banks.py` `get_bank_stats` L160-L174).
#[derive(Clone, Debug)]
pub struct BankStats {
    /// The bank name as queried.
    pub name: String,
    /// Whether the bank's database file exists.
    pub exists: bool,
    /// The resolved database path.
    pub db_path: PathBuf,
    /// Database file size in bytes (0 when absent).
    pub db_size_bytes: u64,
}

/// Manage named memory banks under one data directory (`banks.py` `BankManager`).
pub struct BankManager {
    data_dir: PathBuf,
}

impl BankManager {
    /// A manager rooted at `data_dir` (the banks directory is created lazily by the operations
    /// that need it, so constructing a manager never touches the filesystem).
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
        }
    }

    fn banks_dir(&self) -> PathBuf {
        self.data_dir.join("banks")
    }

    /// The database path for a bank (`banks.py` `get_bank_db_path` L123-L132): `default` (or
    /// empty) maps to the legacy `<data_dir>/mnemosyne.db`, everything else to
    /// `<data_dir>/banks/<name>/mnemosyne.db`.
    pub fn bank_db_path(&self, name: &str) -> PathBuf {
        if name.is_empty() || name == "default" {
            self.data_dir.join("mnemosyne.db")
        } else {
            self.banks_dir().join(name).join("mnemosyne.db")
        }
    }

    /// Create a new bank directory + empty database file (`banks.py` `create_bank` L60-L83).
    /// Errors if the name is invalid or the bank already exists. Returns the database path.
    pub fn create_bank(&self, name: &str) -> Result<PathBuf> {
        validate_name(name)?;
        let bank_dir = self.banks_dir().join(name);
        if bank_dir.exists() {
            return Err(Error::Invalid(format!("Bank '{name}' already exists")));
        }
        std::fs::create_dir_all(&bank_dir)?;
        let db_path = bank_dir.join("mnemosyne.db");
        // Initialize by opening once (Python connects and closes; the full schema lands on the
        // first Store::open).
        let conn = rusqlite::Connection::open(&db_path)
            .map_err(|e| Error::Invalid(format!("create bank '{name}': {e}")))?;
        drop(conn);
        Ok(db_path)
    }

    /// Delete a bank and all its data (`banks.py` `delete_bank` L85-L105). Refuses to delete
    /// `default` unless `force`. Returns `false` if the bank didn't exist.
    pub fn delete_bank(&self, name: &str, force: bool) -> Result<bool> {
        if name == "default" && !force {
            return Err(Error::Invalid(
                "Cannot delete 'default' bank without force".into(),
            ));
        }
        let bank_dir = self.banks_dir().join(name);
        if !bank_dir.exists() {
            return Ok(false);
        }
        std::fs::remove_dir_all(&bank_dir)?;
        Ok(true)
    }

    /// All existing bank names, sorted, with `default` always present (`banks.py` `list_banks`
    /// L107-L115).
    pub fn list_banks(&self) -> Result<Vec<String>> {
        let mut banks: Vec<String> = Vec::new();
        let dir = self.banks_dir();
        if dir.exists() {
            for entry in std::fs::read_dir(&dir)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    banks.push(entry.file_name().to_string_lossy().into_owned());
                }
            }
        }
        if !banks.iter().any(|b| b == "default") {
            banks.push("default".to_string());
        }
        banks.sort();
        Ok(banks)
    }

    /// Whether a bank exists (`banks.py` `bank_exists` L117-L121). `default` always exists.
    pub fn bank_exists(&self, name: &str) -> bool {
        name == "default" || self.banks_dir().join(name).is_dir()
    }

    /// Rename a bank (`banks.py` `rename_bank` L134-L158). `default` cannot be renamed; the new
    /// name must validate and be free. Returns the new database path.
    pub fn rename_bank(&self, old_name: &str, new_name: &str) -> Result<PathBuf> {
        if old_name == "default" {
            return Err(Error::Invalid("Cannot rename 'default' bank".into()));
        }
        validate_name(new_name)?;
        let old_dir = self.banks_dir().join(old_name);
        let new_dir = self.banks_dir().join(new_name);
        if !old_dir.exists() {
            return Err(Error::Invalid(format!("Bank '{old_name}' does not exist")));
        }
        if new_dir.exists() {
            return Err(Error::Invalid(format!("Bank '{new_name}' already exists")));
        }
        std::fs::rename(&old_dir, &new_dir)?;
        Ok(new_dir.join("mnemosyne.db"))
    }

    /// Existence + path + size for a bank (`banks.py` `get_bank_stats` L160-L174).
    pub fn bank_stats(&self, name: &str) -> BankStats {
        let db_path = self.bank_db_path(name);
        let size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
        BankStats {
            name: name.to_string(),
            exists: db_path.exists(),
            db_path,
            db_size_bytes: size,
        }
    }

    /// The manager's data-dir root (for callers composing paths).
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}

/// Validate a bank name (`banks.py` `_validate_name` L176-L185): non-empty, alphanumeric plus
/// `-_`, at most [`MAX_BANK_NAME_LEN`] characters. `default` is always valid.
pub fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::Invalid("Bank name cannot be empty".into()));
    }
    if name == "default" {
        return Ok(());
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || "-_".contains(c))
    {
        return Err(Error::Invalid(format!(
            "Invalid bank name '{name}'. Use alphanumeric, hyphens, underscores only."
        )));
    }
    if name.chars().count() > MAX_BANK_NAME_LEN {
        return Err(Error::Invalid(format!(
            "Bank name '{name}' exceeds {MAX_BANK_NAME_LEN} characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager() -> (BankManager, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "mnemosyne-banks-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        (BankManager::new(&dir), dir)
    }

    #[test]
    fn create_list_rename_delete_round_trip() {
        let (m, dir) = manager();
        assert_eq!(m.list_banks().unwrap(), vec!["default"]);

        let db = m.create_bank("work").unwrap();
        assert!(db.ends_with("banks/work/mnemosyne.db"));
        assert!(db.exists());
        assert!(m.bank_exists("work"));
        assert!(m.create_bank("work").is_err(), "duplicate create rejected");
        assert_eq!(m.list_banks().unwrap(), vec!["default", "work"]);

        let renamed = m.rename_bank("work", "personal").unwrap();
        assert!(renamed.ends_with("banks/personal/mnemosyne.db"));
        assert!(!m.bank_exists("work"));
        assert!(m.bank_exists("personal"));

        let stats = m.bank_stats("personal");
        assert!(stats.exists);

        assert!(m.delete_bank("personal", false).unwrap());
        assert!(!m.bank_exists("personal"));
        assert!(!m.delete_bank("personal", false).unwrap(), "already gone");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_bank_is_protected_and_always_listed() {
        let (m, dir) = manager();
        assert!(m.bank_exists("default"));
        assert!(
            m.delete_bank("default", false).is_err(),
            "default needs force"
        );
        assert!(m.rename_bank("default", "other").is_err());
        assert_eq!(
            m.bank_db_path("default"),
            m.data_dir().join("mnemosyne.db"),
            "default uses the legacy path"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn name_validation_matches_python() {
        assert!(validate_name("default").is_ok());
        assert!(validate_name("work-2024_a").is_ok());
        assert!(validate_name("").is_err());
        assert!(validate_name("has space").is_err());
        assert!(validate_name("dots.bad").is_err());
        assert!(validate_name(&"x".repeat(65)).is_err());
        assert!(validate_name(&"x".repeat(64)).is_ok());
    }
}
