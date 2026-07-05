// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The injected time source: every TTL / LRU / watch-rate-limit decision reads through [`Clock`],
//! so tests drive a [`FakeClock`] deterministically (no sleeps, no wall-clock flakiness).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The registry's time source. Unix seconds feed uptime/TTL bookkeeping; monotonic milliseconds
/// feed the watch rate-limit windows.
pub trait Clock: Send + Sync {
    /// Wall-clock seconds since the Unix epoch.
    fn now_unix(&self) -> u64;
    /// Monotonic milliseconds since an arbitrary origin (never goes backwards).
    fn now_ms(&self) -> u64;
}

/// The production clock: `SystemTime` for unix seconds, `Instant` for monotonic milliseconds.
pub struct RealClock {
    origin: Instant,
}

impl RealClock {
    /// A real clock with its monotonic origin at construction.
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for RealClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for RealClock {
    fn now_unix(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    fn now_ms(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

/// A hand-advanced clock for deterministic TTL / LRU / watch tests.
pub struct FakeClock {
    unix: AtomicU64,
    ms: AtomicU64,
}

impl FakeClock {
    /// A fake clock starting at `unix` seconds / 0 monotonic ms.
    pub fn at(unix: u64) -> Self {
        Self {
            unix: AtomicU64::new(unix),
            ms: AtomicU64::new(0),
        }
    }

    /// Advance both the unix and monotonic reading by `secs`.
    pub fn advance_secs(&self, secs: u64) {
        self.unix.fetch_add(secs, Ordering::SeqCst);
        self.ms.fetch_add(secs * 1000, Ordering::SeqCst);
    }

    /// Advance both readings by `ms` milliseconds (unix seconds move by whole seconds only).
    pub fn advance_ms(&self, ms: u64) {
        self.ms.fetch_add(ms, Ordering::SeqCst);
        self.unix.fetch_add(ms / 1000, Ordering::SeqCst);
    }
}

impl Clock for FakeClock {
    fn now_unix(&self) -> u64 {
        self.unix.load(Ordering::SeqCst)
    }

    fn now_ms(&self) -> u64 {
        self.ms.load(Ordering::SeqCst)
    }
}
