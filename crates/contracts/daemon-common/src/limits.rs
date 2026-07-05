// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Shared ingress size bounds for the length-framed transports.
//!
//! Every length-framed carrier (the Unix-socket / named-pipe / TLS mux in `daemon-host`, the
//! cross-node `remote` transport in `daemon-transport`) reads a 4-byte big-endian length prefix and
//! then allocates a buffer of that size. Without a bound, a hostile or corrupt prefix forces a
//! multi-gigabyte allocation *before* any decode or authentication — a trivial pre-auth DoS. This
//! module carries the single shared cap those transports enforce before allocating.

/// The maximum accepted size (in bytes) of one length-framed wire frame, rejected *before* the
/// receive buffer is allocated.
///
/// **Value rationale (640 MiB).** The largest legitimate single frame is a `BlobPut` /
/// `FsWriteFromBlob` payload, bounded server-side by `MAX_BLOB_SIZE` = 256 MiB (see
/// `daemon-host`'s blob store). Rust `Vec<u8>` serializes as a CBOR **array of ints** (one to two
/// bytes per element — the workspace uses no `serde_bytes`), so a 256 MiB blob is up to ~512 MiB on
/// the wire, plus the `WireC2S::Call` / `ApiRequest::BlobPut` envelope. 640 MiB (2 × 256 MiB + 128
/// MiB headroom) accepts every in-spec frame while cutting the pre-decode allocation ceiling from
/// the u32 maximum (~4 GiB) to ~640 MiB. This is a coarse Phase-1 bound; the Phase-4 ingress
/// governor is where a tighter *pre-auth* cap and *per-transport* caps belong (the `remote`
/// control frames, for instance, are all tiny and could take a far smaller cap).
pub const MAX_FRAME_BYTES: usize = 640 * 1024 * 1024;
