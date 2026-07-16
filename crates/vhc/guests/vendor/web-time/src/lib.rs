// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! JS-free `web-time` shim (see Cargo.toml): the upstream API is exactly `std::time`'s
//! (`Instant`, `SystemTime`, `Duration`, `SystemTimeError`, `TryFromFloatSecsError`,
//! `UNIX_EPOCH`), re-mapped to the browser on wasm — here mapped straight back to `std::time`.
//! On wasm32-unknown-unknown, `Instant::now()`/`SystemTime::now()` panic **if called**; a
//! conforming daemon-vhc guest never calls them (its time surface is `sys@2::now`).

pub use std::time::*;
