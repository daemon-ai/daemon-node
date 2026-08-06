// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **central disk custodian** (pre-C2 Phase 6): one custodian per filesystem root, atomic
//! capacity reservation for every durable VHC write path, global + per-scope quotas, a
//! reserved-free-space floor, an emergency reserve for the recovery-critical stream, and derived
//! pressure states.
//!
//! # The invariant this crate enforces
//!
//! VHC consumes only its assigned quota, always preserves a host free-space reserve, and a
//! durable write either FITS (its bytes were reserved before they touched disk) or refuses
//! TYPED — never a raw `ENOSPC` discovered mid-write. The refusal maps onto the typed storage
//! taxonomy ([`CustodyRefusal::to_io`] yields `StorageFull`/`QuotaExceeded` error kinds, which
//! the journaling seam classifies `HostStorageExhausted` → `FailedStorage` → the node's storage
//! gate), so a refused reservation rides the SAME retry-without-budget path a device `ENOSPC`
//! does — the custodian just refuses *before* the disk is actually full.
//!
//! # Scope of authority (what is enforced where)
//!
//! - **The free-space floor is cross-process truth**: every reservation re-probes the OS free
//!   count, so concurrent writers (the node process + its worker children) all back off the same
//!   physical floor even though each holds its own ledger.
//! - **Quotas are per-process accounting**: the ledger seeds from a one-time walk of the root at
//!   open and then tracks this process's committed writes. The worker (which owns every
//!   high-volume write path: journal, spill, payload cache) is therefore the quota authority for
//!   its run; the node's custodian is the resume-authorization + reporting surface.
//! - **Policy lives with the caller**: this crate sizes nothing. The node derives the config
//!   from `[vhc.storage]` and hands it to workers over the environment
//!   ([`CustodyConfig::from_env`]).
//!
//! # Write classes
//!
//! [`WriteClass::Critical`] is the emergency-reserve carve-out (plan: "sealing the active
//! journal and completing an in-flight checkpoint always succeed during pressure handling"): the
//! journal's record stream — the run's recovery input, including the terminal/seal records and
//! the checkpoint anchor — may draw the root down to the HOST reserve, past the emergency
//! margin, and is quota-exempt (refusing the recovery stream to honor a quota would trade a
//! bounded overrun for a forked run). Everything else ([`WriteClass::Normal`]: spill, payload
//! cache, archive heads) refuses at `reserve + emergency`, leaving the margin for the critical
//! stream to finish.

// Sanctioned raw-fs home (module docs): the custodian is the capacity authority for host-owned
// durable roots — it walks and probes the filesystem directly, on caller threads that are not
// async. Paths come from the node/worker configuration, never from attacker-influenced input.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Env var carrying the governed filesystem root (node → worker), the same path-reference
/// delivery as the identity store and the run-state root. When set, every durable store whose
/// root lives UNDER this path attaches to the root's custodian at construction
/// ([`ambient_for`]) — the stores need no plumbing, and a process without the variable (unit
/// tests, non-VHC embedders) is simply uncustodied.
pub const CUSTODY_ROOT_ENV: &str = "DAEMON_VHC_CUSTODY_ROOT";
/// Env var carrying the host free-space floor in MiB (node → worker). Absent = default floor.
pub const DISK_RESERVE_MB_ENV: &str = "DAEMON_VHC_DISK_RESERVE_MB";
/// Env var carrying the global VHC quota in MiB (node → worker). Absent/0 = unbounded.
pub const DISK_QUOTA_MB_ENV: &str = "DAEMON_VHC_DISK_QUOTA_MB";
/// Env var carrying the per-scope (per-run) quota in MiB (node → worker). Absent/0 = unbounded.
pub const DISK_RUN_QUOTA_MB_ENV: &str = "DAEMON_VHC_DISK_RUN_QUOTA_MB";
/// Env var carrying the emergency (critical-stream) reserve in MiB (node → worker).
pub const DISK_EMERGENCY_MB_ENV: &str = "DAEMON_VHC_DISK_EMERGENCY_MB";
/// Env var carrying the archive-then-prune recovery horizon in SEGMENTS (node → worker): the
/// newest N archived sealed segments stay local; `0` disables local pruning entirely. Absent =
/// [`DEFAULT_PRUNE_HORIZON_SEGMENTS`].
pub const DISK_PRUNE_HORIZON_ENV: &str = "DAEMON_VHC_DISK_PRUNE_HORIZON_SEGMENTS";

/// The default archive-then-prune recovery horizon (segments): deep enough that a coordinator
/// reconstruction / replay usually starts from warm local files, shallow enough that a churning
/// run's journal footprint stays bounded by `horizon × segment size` per chain.
pub const DEFAULT_PRUNE_HORIZON_SEGMENTS: u64 = 4;

/// The node-provided prune horizon from the environment (see [`DISK_PRUNE_HORIZON_ENV`]).
#[must_use]
pub fn prune_horizon_from_env() -> u64 {
    std::env::var(DISK_PRUNE_HORIZON_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_PRUNE_HORIZON_SEGMENTS)
}

const MIB: u64 = 1024 * 1024;

/// The custodian's sizing policy — derived by the node from `[vhc.storage]`, carried to workers
/// over the environment. All byte counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CustodyConfig {
    /// The global VHC byte quota on this root (`None` = unbounded — the floor still holds).
    pub quota_bytes: Option<u64>,
    /// The uniform per-scope (per-run) byte quota (`None` = unbounded).
    pub scope_quota_bytes: Option<u64>,
    /// The host free-space floor: no VHC write may take the root's free space below this.
    pub reserve_bytes: u64,
    /// The emergency margin above the floor, reachable only by [`WriteClass::Critical`] writes
    /// (the journal's recovery stream), so sealing and the in-flight checkpoint anchor always
    /// have room during pressure handling.
    pub emergency_bytes: u64,
}

impl Default for CustodyConfig {
    fn default() -> Self {
        Self {
            quota_bytes: None,
            scope_quota_bytes: None,
            reserve_bytes: 2_048 * MIB,
            emergency_bytes: 256 * MIB,
        }
    }
}

impl CustodyConfig {
    /// Read the node-provided sizing from the environment (the worker's construction path).
    /// Absent variables keep the defaults; `0` quota = unbounded.
    #[must_use]
    pub fn from_env() -> Self {
        let mb =
            |name: &str| -> Option<u64> { std::env::var(name).ok()?.trim().parse::<u64>().ok() };
        let mut cfg = Self::default();
        if let Some(v) = mb(DISK_RESERVE_MB_ENV) {
            cfg.reserve_bytes = v.saturating_mul(MIB);
        }
        if let Some(v) = mb(DISK_EMERGENCY_MB_ENV) {
            cfg.emergency_bytes = v.saturating_mul(MIB);
        }
        cfg.quota_bytes = mb(DISK_QUOTA_MB_ENV).filter(|v| *v > 0).map(|v| v * MIB);
        cfg.scope_quota_bytes = mb(DISK_RUN_QUOTA_MB_ENV)
            .filter(|v| *v > 0)
            .map(|v| v * MIB);
        cfg
    }
}

/// What kind of durable write a reservation covers (see the module docs' write-class contract).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteClass {
    /// Spill / payload cache / archive heads — refuses at `reserve + emergency`, quota-checked.
    Normal,
    /// The journal's recovery-critical record stream — may draw down to the host reserve,
    /// quota-exempt (a bounded overrun beats a forked run).
    Critical,
}

/// The derived pressure state (plan: warn → refuse new work → seal/publish → reclaim). The
/// custodian derives the first two; sealing/publication/reclaim are the session's and node's
/// reactions to them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Pressure {
    /// Capacity is comfortable.
    Nominal,
    /// Approaching the floor or the quota — reclaim should start; new work still admits.
    Warn,
    /// A [`WriteClass::Normal`] reservation would refuse — new work must not be admitted.
    RefuseNew,
}

/// A typed reservation refusal — the write must NOT proceed.
#[derive(Debug, thiserror::Error)]
pub enum CustodyRefusal {
    /// The global or per-scope quota would be exceeded.
    #[error("vhc disk quota exceeded ({scope}: {requested} B over the {quota} B quota)")]
    QuotaExceeded {
        /// The charged scope.
        scope: String,
        /// The refused byte count.
        requested: u64,
        /// The quota that refused.
        quota: u64,
    },
    /// The write would take the root's free space below the protected floor.
    #[error("host free-space floor: {requested} B refused with {free} B free (floor {floor} B)")]
    FloorBreached {
        /// The refused byte count.
        requested: u64,
        /// The probed free bytes.
        free: u64,
        /// The floor the class must respect.
        floor: u64,
    },
}

impl CustodyRefusal {
    /// Map onto the typed storage taxonomy's exhaustion kinds, so callers surfacing this as an
    /// `io::Error` classify `HostStorageExhausted` (retry-without-budget), never `BadModule`.
    #[must_use]
    pub fn to_io(&self) -> std::io::Error {
        let kind = match self {
            Self::QuotaExceeded { .. } => std::io::ErrorKind::QuotaExceeded,
            Self::FloorBreached { .. } => std::io::ErrorKind::StorageFull,
        };
        std::io::Error::new(kind, self.to_string())
    }
}

/// A serializable usage snapshot (the CLI's `vhc disk` payload).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CustodyUsage {
    /// The custodian's root.
    pub root: String,
    /// Probed free bytes on the root's filesystem.
    pub free_bytes: u64,
    /// Committed bytes in this process's ledger (seeded by the open-time walk).
    pub used_bytes: u64,
    /// Reserved-but-uncommitted bytes.
    pub pending_bytes: u64,
    /// The configured global quota (`0` = unbounded).
    pub quota_bytes: u64,
    /// The configured host floor.
    pub reserve_bytes: u64,
    /// The configured emergency margin.
    pub emergency_bytes: u64,
    /// The derived pressure state.
    pub pressure: Pressure,
    /// Committed bytes per scope (run-state directory key → bytes).
    pub scopes: Vec<(String, u64)>,
}

/// The per-root custodian. Construct through [`DiskCustodian::for_root`] (the per-process
/// singleton registry — "one custodian per filesystem root") or [`DiskCustodian::open`] directly
/// in tests.
pub struct DiskCustodian {
    root: PathBuf,
    cfg: CustodyConfig,
    used_total: AtomicU64,
    pending: AtomicU64,
    scopes: Mutex<BTreeMap<String, u64>>,
}

/// The per-process custodian registry, keyed by the canonicalized root (one VHC root per
/// filesystem in every deployment shape; two roots on one filesystem would each see the same
/// OS floor, so the floor invariant still holds).
fn registry() -> &'static Mutex<BTreeMap<PathBuf, Arc<DiskCustodian>>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<PathBuf, Arc<DiskCustodian>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// The **ambient attachment seam**: the custodian + charge scope for a durable store rooted at
/// `path`, derived from the environment ([`CUSTODY_ROOT_ENV`] + the sizing variables). `None`
/// when no custody root is exported or `path` does not live under it (an uncustodied store —
/// unit tests, non-VHC embedders, an out-of-root payload override).
///
/// The scope is the first path component under the custody root — the run-state directory key
/// (`blake3(run label)` hex), which is exactly how the open-time walk seeds the ledger. Every
/// plane of one run (journal, spill, payload, archive heads) therefore charges one scope.
#[must_use]
pub fn ambient_for(path: &Path) -> Option<(Arc<DiskCustodian>, String)> {
    let root = PathBuf::from(std::env::var_os(CUSTODY_ROOT_ENV)?);
    scoped_for(&root, path)
}

/// [`ambient_for`] with an explicit custody root (the node's in-process path, and tests).
#[must_use]
pub fn scoped_for(root: &Path, path: &Path) -> Option<(Arc<DiskCustodian>, String)> {
    // Best-effort normalization: the node exports both the custody root and the store roots
    // from the same configured path, so a textual prefix match holds; canonicalization just
    // absorbs symlinked parents when both sides resolve.
    let canon = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let (root_c, path_c) = (canon(root), canon(path));
    let rel = path_c.strip_prefix(&root_c).ok()?;
    let scope = rel
        .components()
        .next()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())?;
    let custodian = DiskCustodian::for_root(&root_c, CustodyConfig::from_env()).ok()?;
    Some((custodian, scope))
}

impl DiskCustodian {
    /// Open a custodian over `root` (created if missing), seeding the ledger with a one-time
    /// walk of the existing contents (grouped by top-level entry = the run-state directory key).
    ///
    /// # Errors
    /// [`std::io::Error`] if the root cannot be created or walked.
    pub fn open(root: impl AsRef<Path>, cfg: CustodyConfig) -> std::io::Result<Arc<Self>> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        let mut scopes = BTreeMap::new();
        let mut total = 0u64;
        for dent in std::fs::read_dir(&root)? {
            let dent = dent?;
            let bytes = walk_bytes(&dent.path());
            total = total.saturating_add(bytes);
            scopes.insert(dent.file_name().to_string_lossy().into_owned(), bytes);
        }
        Ok(Arc::new(Self {
            root,
            cfg,
            used_total: AtomicU64::new(total),
            pending: AtomicU64::new(0),
            scopes: Mutex::new(scopes),
        }))
    }

    /// The per-process singleton for `root`: the first opener's config wins; later callers get
    /// the same custodian (one capacity ledger per root, however many planes write through it).
    ///
    /// # Errors
    /// See [`DiskCustodian::open`].
    pub fn for_root(root: impl AsRef<Path>, cfg: CustodyConfig) -> std::io::Result<Arc<Self>> {
        let key = root
            .as_ref()
            .canonicalize()
            .unwrap_or_else(|_| root.as_ref().to_path_buf());
        let mut reg = registry().lock().expect("custody registry lock");
        if let Some(existing) = reg.get(&key) {
            return Ok(Arc::clone(existing));
        }
        let custodian = Self::open(&key, cfg)?;
        reg.insert(key, Arc::clone(&custodian));
        Ok(custodian)
    }

    /// The custodian's root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Atomically reserve `bytes` for a durable write in `scope`. The returned guard must be
    /// [`Reservation::commit`]ed after the bytes are durably on disk; dropping it uncommitted
    /// releases the reservation (the failed-write path).
    ///
    /// # Errors
    /// A typed [`CustodyRefusal`]; the write must not proceed (surface it via
    /// [`CustodyRefusal::to_io`] so the storage taxonomy classifies it exhaustion-retryable).
    pub fn reserve(
        self: &Arc<Self>,
        scope: &str,
        bytes: u64,
        class: WriteClass,
    ) -> Result<Reservation, CustodyRefusal> {
        let pending = self.pending.load(Ordering::Relaxed);
        if class == WriteClass::Normal {
            if let Some(quota) = self.cfg.quota_bytes {
                let used = self.used_total.load(Ordering::Relaxed);
                if used.saturating_add(pending).saturating_add(bytes) > quota {
                    return Err(CustodyRefusal::QuotaExceeded {
                        scope: scope.to_string(),
                        requested: bytes,
                        quota,
                    });
                }
            }
            if let Some(quota) = self.cfg.scope_quota_bytes {
                let scoped = self
                    .scopes
                    .lock()
                    .expect("custody scope lock")
                    .get(scope)
                    .copied()
                    .unwrap_or(0);
                if scoped.saturating_add(bytes) > quota {
                    return Err(CustodyRefusal::QuotaExceeded {
                        scope: scope.to_string(),
                        requested: bytes,
                        quota,
                    });
                }
            }
        }
        let floor = match class {
            WriteClass::Normal => self
                .cfg
                .reserve_bytes
                .saturating_add(self.cfg.emergency_bytes),
            WriteClass::Critical => self.cfg.reserve_bytes,
        };
        // The floor is judged against LIVE free space minus this process's outstanding
        // reservations — cross-process truthful for the floor itself (every process probes the
        // same filesystem), per-process for in-flight intent.
        let free = free_bytes(&self.root).saturating_sub(pending);
        if free.saturating_sub(bytes) < floor {
            return Err(CustodyRefusal::FloorBreached {
                requested: bytes,
                free,
                floor,
            });
        }
        self.pending.fetch_add(bytes, Ordering::Relaxed);
        Ok(Reservation {
            custodian: Arc::clone(self),
            scope: scope.to_string(),
            bytes,
            committed: false,
        })
    }

    /// Release `bytes` of committed charge for `scope` (a prune / wipe reclaimed them).
    pub fn discharge(&self, scope: &str, bytes: u64) {
        let mut scopes = self.scopes.lock().expect("custody scope lock");
        if let Some(entry) = scopes.get_mut(scope) {
            *entry = entry.saturating_sub(bytes);
        }
        let _ = self
            .used_total
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(bytes))
            });
    }

    /// Drop a scope from the ledger entirely (its directory was reclaimed).
    pub fn forget_scope(&self, scope: &str) {
        let mut scopes = self.scopes.lock().expect("custody scope lock");
        if let Some(bytes) = scopes.remove(scope) {
            let _ = self
                .used_total
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    Some(v.saturating_sub(bytes))
                });
        }
    }

    /// The derived pressure state (see [`Pressure`]).
    #[must_use]
    pub fn pressure(&self) -> Pressure {
        let free = free_bytes(&self.root).saturating_sub(self.pending.load(Ordering::Relaxed));
        let refuse_floor = self
            .cfg
            .reserve_bytes
            .saturating_add(self.cfg.emergency_bytes);
        if free <= refuse_floor || self.quota_exhausted() {
            return Pressure::RefuseNew;
        }
        if free < refuse_floor.saturating_mul(2) || self.quota_warns() {
            return Pressure::Warn;
        }
        Pressure::Nominal
    }

    /// Whether a [`WriteClass::Normal`] reservation of `bytes` would currently succeed — the
    /// node's resume-authorization question for a storage-gated run (replaces the Phase-1
    /// bare free-space check).
    #[must_use]
    pub fn can_admit(self: &Arc<Self>, scope: &str, bytes: u64) -> bool {
        // A dry-run reservation: acquire and immediately release.
        self.reserve(scope, bytes, WriteClass::Normal).is_ok()
    }

    /// A usage snapshot for reporting (`daemon-cli vhc disk`).
    #[must_use]
    pub fn usage(&self) -> CustodyUsage {
        let scopes = self
            .scopes
            .lock()
            .expect("custody scope lock")
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        CustodyUsage {
            root: self.root.display().to_string(),
            free_bytes: free_bytes(&self.root),
            used_bytes: self.used_total.load(Ordering::Relaxed),
            pending_bytes: self.pending.load(Ordering::Relaxed),
            quota_bytes: self.cfg.quota_bytes.unwrap_or(0),
            reserve_bytes: self.cfg.reserve_bytes,
            emergency_bytes: self.cfg.emergency_bytes,
            pressure: self.pressure(),
            scopes,
        }
    }

    fn quota_exhausted(&self) -> bool {
        self.cfg.quota_bytes.is_some_and(|q| {
            self.used_total
                .load(Ordering::Relaxed)
                .saturating_add(self.pending.load(Ordering::Relaxed))
                >= q
        })
    }

    fn quota_warns(&self) -> bool {
        self.cfg.quota_bytes.is_some_and(|q| {
            let used = self
                .used_total
                .load(Ordering::Relaxed)
                .saturating_add(self.pending.load(Ordering::Relaxed));
            // Warn at ≥ 7/8 of the quota.
            used >= q.saturating_sub(q / 8)
        })
    }
}

/// An in-flight capacity reservation (RAII): commit after the durable write succeeds, drop to
/// release on failure.
pub struct Reservation {
    custodian: Arc<DiskCustodian>,
    scope: String,
    bytes: u64,
    committed: bool,
}

impl std::fmt::Debug for Reservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reservation")
            .field("scope", &self.scope)
            .field("bytes", &self.bytes)
            .field("committed", &self.committed)
            .finish_non_exhaustive()
    }
}

impl Reservation {
    /// The bytes were durably written: move the reservation into the committed ledger.
    pub fn commit(mut self) {
        self.committed = true;
        self.custodian
            .pending
            .fetch_sub(self.bytes, Ordering::Relaxed);
        self.custodian
            .used_total
            .fetch_add(self.bytes, Ordering::Relaxed);
        let mut scopes = self.custodian.scopes.lock().expect("custody scope lock");
        *scopes.entry(self.scope.clone()).or_insert(0) += self.bytes;
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if !self.committed {
            self.custodian
                .pending
                .fetch_sub(self.bytes, Ordering::Relaxed);
        }
    }
}

/// Total bytes of regular files under `path` (0 for a file's own metadata errors — the walk is
/// advisory seeding, not a gate).
fn walk_bytes(path: &Path) -> u64 {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(_) => return 0,
    };
    if meta.is_file() {
        return meta.len();
    }
    if !meta.is_dir() {
        return 0; // symlinks and specials never count (and are never followed).
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|dent| walk_bytes(&dent.path()))
        .fold(0u64, u64::saturating_add)
}

/// Free bytes (available to an unprivileged caller) on `path`'s filesystem. `0` on probe failure
/// — fail-closed: a root whose capacity cannot be probed admits nothing above the floor.
#[must_use]
pub fn free_bytes(path: &Path) -> u64 {
    #[cfg(unix)]
    {
        unix_free_bytes(path)
    }
    #[cfg(windows)]
    {
        win_free_bytes(path)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = path;
        0
    }
}

#[cfg(unix)]
fn unix_free_bytes(path: &Path) -> u64 {
    use std::os::unix::ffi::OsStrExt;
    let mut cpath: Vec<u8> = path.as_os_str().as_bytes().to_vec();
    cpath.push(0);
    // SAFETY: `cpath` is a NUL-terminated C string; `stat` is a valid, zero-initialized
    // `statvfs` out-pointer of the exact type the libc binding expects.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(cpath.as_ptr().cast::<libc::c_char>(), &mut stat) };
    if rc != 0 {
        return 0;
    }
    #[allow(clippy::unnecessary_cast)] // the field widths differ across unix targets
    {
        (stat.f_frsize as u64).saturating_mul(stat.f_bavail as u64)
    }
}

#[cfg(windows)]
fn win_free_bytes(path: &Path) -> u64 {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetDiskFreeSpaceExW(
            directory: *const u16,
            free_to_caller: *mut u64,
            total: *mut u64,
            total_free: *mut u64,
        ) -> i32;
    }
    let wide: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let mut free: u64 = 0;
    // SAFETY: `wide` is a NUL-terminated wide string; `free` is a valid u64 out-pointer; the
    // unused out-params are null (documented as optional).
    let rc = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut free,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if rc == 0 {
        0
    } else {
        free
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_root(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "daemon-vhc-custody-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        p
    }

    /// A config whose floor can never trip on the test host (the quota is the subject).
    fn unfloored(quota: Option<u64>, scope_quota: Option<u64>) -> CustodyConfig {
        CustodyConfig {
            quota_bytes: quota,
            scope_quota_bytes: scope_quota,
            reserve_bytes: 0,
            emergency_bytes: 0,
        }
    }

    #[test]
    fn reservation_commit_and_release_move_the_ledger() {
        let root = tmp_root("ledger");
        let c = DiskCustodian::open(&root, unfloored(None, None)).unwrap();
        let r = c.reserve("run-a", 100, WriteClass::Normal).unwrap();
        assert_eq!(c.usage().pending_bytes, 100);
        r.commit();
        assert_eq!(c.usage().pending_bytes, 0);
        assert_eq!(c.usage().used_bytes, 100);
        // A dropped (failed-write) reservation releases without charging.
        drop(c.reserve("run-a", 50, WriteClass::Normal).unwrap());
        assert_eq!(c.usage().pending_bytes, 0);
        assert_eq!(c.usage().used_bytes, 100);
        c.discharge("run-a", 60);
        assert_eq!(c.usage().used_bytes, 40);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_global_quota_refuses_normal_but_never_critical() {
        let root = tmp_root("quota");
        let c = DiskCustodian::open(&root, unfloored(Some(1_000), None)).unwrap();
        c.reserve("run-a", 900, WriteClass::Normal)
            .unwrap()
            .commit();
        let refusal = c.reserve("run-a", 200, WriteClass::Normal).unwrap_err();
        assert!(matches!(refusal, CustodyRefusal::QuotaExceeded { .. }));
        assert_eq!(refusal.to_io().kind(), std::io::ErrorKind::QuotaExceeded);
        // The recovery-critical stream is quota-exempt by contract.
        c.reserve("run-a", 200, WriteClass::Critical)
            .unwrap()
            .commit();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_scope_quota_is_per_run() {
        let root = tmp_root("scope-quota");
        let c = DiskCustodian::open(&root, unfloored(None, Some(500))).unwrap();
        c.reserve("run-a", 500, WriteClass::Normal)
            .unwrap()
            .commit();
        assert!(c.reserve("run-a", 1, WriteClass::Normal).is_err());
        // Another run's ledger is untouched.
        c.reserve("run-b", 500, WriteClass::Normal)
            .unwrap()
            .commit();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_floor_holds_and_critical_reaches_the_emergency_margin() {
        let root = tmp_root("floor");
        std::fs::create_dir_all(&root).unwrap();
        let free_now = free_bytes(&root);
        assert!(free_now > 0, "test host must probe free space");
        // A floor sized so free space sits INSIDE the emergency margin: Normal refuses,
        // Critical (floor excludes the margin) admits.
        let cfg = CustodyConfig {
            quota_bytes: None,
            scope_quota_bytes: None,
            reserve_bytes: free_now.saturating_sub(10 * MIB),
            emergency_bytes: 20 * MIB,
        };
        let c = DiskCustodian::open(&root, cfg).unwrap();
        let refusal = c.reserve("run-a", MIB, WriteClass::Normal).unwrap_err();
        assert!(matches!(refusal, CustodyRefusal::FloorBreached { .. }));
        assert_eq!(refusal.to_io().kind(), std::io::ErrorKind::StorageFull);
        assert_eq!(c.pressure(), Pressure::RefuseNew);
        c.reserve("run-a", MIB, WriteClass::Critical)
            .unwrap()
            .commit();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_open_walk_seeds_the_ledger_per_scope() {
        let root = tmp_root("seed");
        std::fs::create_dir_all(root.join("scope-1/journal")).unwrap();
        std::fs::write(root.join("scope-1/journal/seg"), vec![0u8; 300]).unwrap();
        std::fs::write(root.join("scope-1/top"), vec![0u8; 100]).unwrap();
        std::fs::create_dir_all(root.join("scope-2")).unwrap();
        std::fs::write(root.join("scope-2/x"), vec![0u8; 50]).unwrap();
        let c = DiskCustodian::open(&root, unfloored(None, None)).unwrap();
        let usage = c.usage();
        assert_eq!(usage.used_bytes, 450);
        assert!(usage.scopes.contains(&("scope-1".to_string(), 400)));
        assert!(usage.scopes.contains(&("scope-2".to_string(), 50)));
        c.forget_scope("scope-1");
        assert_eq!(c.usage().used_bytes, 50);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn for_root_returns_one_custodian_per_root() {
        let root = tmp_root("singleton");
        let a = DiskCustodian::for_root(&root, unfloored(None, None)).unwrap();
        let b = DiskCustodian::for_root(&root, unfloored(Some(1), None)).unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "the first opener's custodian is shared"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn scoped_for_derives_the_run_scope_and_shares_the_root_custodian() {
        let root = tmp_root("scoped");
        std::fs::create_dir_all(root.join("runhash-1/trainer-0/journal")).unwrap();
        let (c1, scope1) =
            scoped_for(&root, &root.join("runhash-1/trainer-0/journal")).expect("under the root");
        assert_eq!(scope1, "runhash-1");
        let (c2, scope2) = scoped_for(&root, &root.join("runhash-1/payload")).expect("payload");
        assert_eq!(scope2, "runhash-1");
        assert!(Arc::ptr_eq(&c1, &c2), "one custodian per root");
        // A path outside the root is uncustodied.
        assert!(scoped_for(&root, &std::env::temp_dir()).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn config_from_env_parses_and_zero_means_unbounded() {
        // Serialized via a lock-free convention: unique var values per test process are fine —
        // this is the only test touching these vars.
        std::env::set_var(DISK_RESERVE_MB_ENV, "10");
        std::env::set_var(DISK_QUOTA_MB_ENV, "0");
        std::env::set_var(DISK_RUN_QUOTA_MB_ENV, "5");
        std::env::set_var(DISK_EMERGENCY_MB_ENV, "1");
        let cfg = CustodyConfig::from_env();
        assert_eq!(cfg.reserve_bytes, 10 * MIB);
        assert_eq!(cfg.quota_bytes, None);
        assert_eq!(cfg.scope_quota_bytes, Some(5 * MIB));
        assert_eq!(cfg.emergency_bytes, MIB);
        std::env::remove_var(DISK_RESERVE_MB_ENV);
        std::env::remove_var(DISK_QUOTA_MB_ENV);
        std::env::remove_var(DISK_RUN_QUOTA_MB_ENV);
        std::env::remove_var(DISK_EMERGENCY_MB_ENV);
    }
}
