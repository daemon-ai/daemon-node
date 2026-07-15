// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-abi` — the ABI surface both sides of the wasm boundary must agree on.
//!
//! The frozen `tabi@1` import vocabulary ([`TABI_IMPORTS`]) plus the ABI version constants
//! ([`DA_ABI_MAJOR`] / [`DA_ABI_MINOR`] / [`DA_ABI_VERSION`]). This is a `contracts/` crate: it is
//! dual-compiled for native *and* `wasm32-unknown-unknown`, links neither `sdk/*` nor `host/*`, and
//! carries no third-party dependencies.
//!
//! Extracted verbatim from the guest SDK (`daemon-vhc-sdk`) so the host runtime can assert its
//! `Linker` / phase-legality table against the frozen surface *without dev-depending on the SDK*
//! (the host->SDK dev-dep wart). The SDK re-exports these items, so guest-side and existing
//! consumers see an unchanged path. Growth is additive only (ABI §9) — append to [`TABI_IMPORTS`],
//! never reorder or remove.

#![forbid(unsafe_code)]

/// The tensor-ABI major version this contract is pinned to.
pub const DA_ABI_MAJOR: u32 = 1;
/// The tensor-ABI minor version this contract is pinned to.
pub const DA_ABI_MINOR: u32 = 0;

/// The tensor-ABI version, packed as `(major << 16) | minor`.
///
/// The guest advertises it via the `da_abi` export so the host can reject an incompatible module
/// before instantiation (swarm-tensor-abi-spec.md §4).
pub const DA_ABI_VERSION: u32 = (DA_ABI_MAJOR << 16) | DA_ABI_MINOR;

// ================================================================================================
// The journal (ABI companion §8): the normative record grammar + its framing constants.
//
// The crash-safe segmented journal makes policy-determinism operational (ABI §8, architecture
// §3.6). Its record format is a versioned ABI artifact that lives *here* (ABI §8.3: "it lives
// verbatim in `daemon-vhc-abi`"), because both the substrate that writes it (`daemon-vhc-observe`)
// and any conformance validator must agree on one authoritative grammar. This crate stays
// dependency-free / dual-compiled for wasm32, so it holds the grammar text + the framing constants
// only; the Rust record types + canonical CBOR codec live host-side in
// `daemon-vhc-observe::journal` and validate their output against [`JOURNAL_CDDL`].
// ================================================================================================

/// The normative, machine-valid, complete journal record grammar (ABI §8.2/§8.3/§8.5).
///
/// This is the authoritative CDDL artifact: it MUST validate as-is under `cddl-cat`, and tier-1 CI
/// validates every record of every conformance-run journal against it (ABI §13). Root rules:
/// `journal-record` (the §8.3 tagged union), `segment-header-body` (§8.2), and `sidecar-header`
/// (§8.5). Growth is additive by minor — tags 18–63 are reserved; tags are permanent.
pub const JOURNAL_CDDL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/journal.cddl"));

/// The on-disk magic identifying a VHC journal segment file (ABI §8.2 header).
pub const JOURNAL_SEGMENT_MAGIC: &[u8; 8] = b"DVHCJRN2";

/// The on-disk magic identifying a VHC encrypted sidecar file (ABI §8.5 file layout).
pub const JOURNAL_SIDECAR_MAGIC: &[u8; 8] = b"DVHCSC01";

/// The journal format version carried in every segment header and the run-header record (ABI §8.2
/// `format_version`, §8.3 tag-0 `format`).
pub const JOURNAL_FORMAT_VERSION: u32 = 1;

/// The inline/sidecar threshold for `read_back` values (ABI §8.5): a value whose plaintext exceeds
/// this many bytes is stored as an encrypted content-addressed sidecar rather than inline in the
/// record. An ABI constant so writer and replay agree on the boundary.
pub const READBACK_INLINE_MAX: usize = 4096;

/// The numeric tag of every journal record variant (ABI §8.3). Tags are permanent and additive by
/// minor; the reserved range is 18–63. Kept as an explicit list so the substrate and any conformance
/// tool share one source of truth for "the complete record set".
pub const JOURNAL_RECORD_TAGS: &[u8] = &[
    0,  // run-header
    1,  // event
    2,  // read-back
    3,  // clock
    4,  // publish
    5,  // timer-arm
    6,  // timer-cancel
    7,  // drop
    8,  // throttle
    9,  // terminal
    10, // snapshot
    11, // init
    12, // signed-frame
    13, // instantiation
    14, // completion (reserved, Phase B)
    15, // device-profile (reserved, Phase B)
    16, // condition
    17, // seal
];

/// The complete `tabi@1` import vocabulary the guest SDK binds (the extern block in the SDK's
/// `abi.rs`), in registration order: the Merge-1 frozen 50-import subset followed by the Wave-2
/// additions.
///
/// This is the **frozen surface**: the host `Linker` (`daemon-vhc-host`) and the phase-legality
/// table must agree with it name-for-name (asserted by `daemon-vhc-host/tests/abi_surface.rs`).
/// Growth is additive only (ABI §9) — append here, never reorder or remove.
pub const TABI_IMPORTS: &[&str] = &[
    // --- Merge-1 frozen subset (50) ---
    "param@1",
    "persistent@1",
    "det_persistent@1",
    "drop@1",
    "param_round_base@1",
    "backward@1",
    "grad@1",
    "zero_grads@1",
    "assign@1",
    "zeros@1",
    "ones@1",
    "full@1",
    "add@1",
    "sub@1",
    "mul@1",
    "mul_s@1",
    "matmul@1",
    "relu@1",
    "cross_entropy@1",
    "scalar@1",
    "metric@1",
    "log@1",
    "abi_minor@1",
    "adamw_step@1",
    "batch_tokens@1",
    "batch_size@1",
    "batch_seq_len@1",
    "upd_new@1",
    "upd_push_bytes@1",
    "upd_push_tensor@1",
    "upd_sections@1",
    "upd_kind@1",
    "upd_bytes_len@1",
    "upd_read_bytes@1",
    "upd_tensor@1",
    "det_zeros@1",
    "det_sum@1",
    "det_scale@1",
    "det_l2norm@1",
    "det_sign@1",
    "det_add@1",
    "det_sub@1",
    "det_mul@1",
    "det_absmax_unpack@1",
    "det_chunk_scatter_add@1",
    "det_chunk_scatter@1",
    "det_assign@1",
    "det_param@1",
    "det_reset_param_to_base@1",
    "det_axpy_param@1",
    // --- Wave-2 additions (16) ---
    "embedding@1",
    "rmsnorm@1",
    "softmax@1",
    "silu@1",
    "rope@1",
    "flash_attn@1",
    "reshape@1",
    "transpose@1",
    "slice@1",
    "topk_chunk@1",
    "chunk_scatter@1",
    "absmax_pack@1",
    "absmax_unpack@1",
    "dct2@1",
    "idct2@1",
    "det_idct2@1",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_packs_major_minor() {
        assert_eq!(DA_ABI_VERSION >> 16, 1);
        assert_eq!(DA_ABI_VERSION & 0xffff, 0);
    }

    #[test]
    fn tabi_imports_are_unique_and_complete() {
        // 50 Merge-1 frozen imports + 16 Wave-2 additions = the frozen v1 vocabulary.
        assert_eq!(TABI_IMPORTS.len(), 66);
        let mut names: Vec<&str> = TABI_IMPORTS.to_vec();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "tabi import names must be unique");
        // Every name carries an explicit @version (additive growth is by version, ABI §9).
        assert!(TABI_IMPORTS.iter().all(|n| n.contains('@')));
    }
}
