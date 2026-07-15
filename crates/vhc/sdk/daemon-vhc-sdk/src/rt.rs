// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `da_*` export runtime glue (wasm guest only): the guest allocator the host requests buffers
//! through (`da_alloc`/`da_free`, ABI §4) plus the CBOR marshalling for `da_manifest`/`da_defaults`
//! and `da_build` config unmarshalling. Called by the code [`crate::experiment!`] generates.
//!
//! This is the crate's only `unsafe`: the raw pointer arithmetic across the ABI boundary. It is
//! compiled solely for `wasm32-unknown-unknown` (never `sim`/native).

use crate::Config;
use std::alloc::{alloc, dealloc, Layout};

fn layout(size: u32, align: u32) -> Layout {
    Layout::from_size_align(size as usize, (align as usize).max(1)).expect("valid da_alloc layout")
}

/// The host requests a guest buffer through this (ABI §4). Returns a linear-memory offset.
#[must_use]
pub fn da_alloc(size: u32, align: u32) -> u32 {
    if size == 0 {
        return 0;
    }
    // SAFETY: layout is non-zero-sized and validity-checked; the host pairs this with `da_free`.
    let ptr = unsafe { alloc(layout(size, align)) };
    ptr as u32
}

/// Paired release for a `da_alloc` buffer / a `(ptr,len)` return the host has finished reading.
pub fn da_free(ptr: u32, size: u32, align: u32) {
    if ptr == 0 || size == 0 {
        return;
    }
    // SAFETY: `ptr`/`size`/`align` match a prior `da_alloc` (host obligation, ABI §4).
    unsafe { dealloc(ptr as *mut u8, layout(size, align)) };
}

/// Copy `bytes` into a freshly-allocated guest buffer and return `(ptr << 32) | len` — the packed
/// `(u, u)` return form for `da_manifest`/`da_defaults` (see [`crate::experiment!`]). The host reads
/// it, copies out, then calls [`da_free`] with `(ptr, len, 1)`.
#[must_use]
pub fn emit_cbor(bytes: &[u8]) -> u64 {
    let len = bytes.len();
    if len == 0 {
        return 0;
    }
    let ptr = da_alloc(len as u32, 1);
    // SAFETY: `ptr` is a fresh `len`-byte allocation; `bytes` is `len` bytes; regions don't overlap.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, len) };
    ((ptr as u64) << 32) | len as u64
}

/// Read a `(ptr, len)` config span the host wrote and wrap it as a [`Config`].
#[must_use]
pub fn config_from_raw(ptr: u32, len: u32) -> Config {
    if len == 0 {
        return Config::from_bytes(Vec::new());
    }
    // SAFETY: the host guarantees `[ptr, ptr+len)` is an in-bounds span it just wrote (ABI §3.1).
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
    Config::from_bytes(slice.to_vec())
}
