// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `meta` mode output — the `MetaReport` (ABI §6.4, HOST-8).
//!
//! A meta run covers the full lifecycle once and yields the resource estimates admission compares
//! against probed VRAM/RAM (architecture §6.5): the parameter/persistent layout, the byte footprints
//! (params in storage dtype + fp32 masters + grads), the payload/ingest estimates, per-entry-point
//! host-op-call counts, the two-point ingest-per-peer fit, and the static op set actually exercised.
//!
//! This build derives the report from a real execute-mode pass on the CPU backend (the numbers are
//! exact for what ran, not a shape-only symbolic propagation). A fully allocation-free shape-only
//! interpreter (and the fuel measurement + `value_dependent` scalar-branch detection it enables) is
//! a later refinement — see the E2 ledger's Merge-2 watch list.

use std::collections::BTreeMap;

/// The canonical-CBOR meta report (ABI §6.4). Field names/shape match the CDDL in the ABI spec.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetaReport {
    /// The module's `da_abi()` (`(major << 16) | minor`).
    pub abi: u32,
    /// `[name, dims, dtype]` per param, in canonical (registration) order.
    pub params: Vec<(String, Vec<u32>, u32)>,
    /// `[name, dims, dtype, class]` per native persistent.
    pub persistent: Vec<(String, Vec<u32>, u32, u32)>,
    /// `[name, dims, class]` per det persistent.
    pub det_persistent: Vec<(String, Vec<u32>, u32)>,
    /// Parameter storage bytes (declared dtype).
    pub param_bytes: u64,
    /// fp32 canonical master bytes.
    pub master_bytes: u64,
    /// fp32 gradient-accumulator bytes.
    pub grad_bytes: u64,
    /// Peak live-activation estimate (coarse in this build; see module docs).
    pub act_bytes_est: u64,
    /// Payload bytes from the meta `da_make_update` section sizes.
    pub payload_bytes_est: u64,
    /// Peak staged + working-set estimate under the streaming discipline (§5.9).
    pub ingest_bytes_est: u64,
    /// CPU-side estimate: masters + round base + staged payloads (feeds `[requirements].ram_gb_min`).
    pub host_ram_bytes_est: u64,
    /// Per-entry-point host-op-call counts (the `da_ingest_updates` entry is the count=1 base).
    pub op_calls: BTreeMap<String, u64>,
    /// Linear per-peer ingest op-call slope from the two-point (1, 2) measurement.
    pub ingest_op_calls_per_peer: u64,
    /// The set of `tabi@1` imports actually charged during the pass (static import proxy).
    pub ops_used: Vec<String>,
    /// Whether any `scalar@1` result drove control flow (always `false` in this build — the
    /// shape-only detector is a later refinement; requirements fall back to author bounds, §2.4).
    pub value_dependent: bool,
}

impl MetaReport {
    /// Encode to canonical CBOR (the `daemon-cli swarm module check` wire form, ABI §6.4).
    #[must_use]
    pub fn to_cbor(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        ciborium::into_writer(self, &mut bytes).expect("MetaReport is always CBOR-serializable");
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_report_cbor_roundtrips() {
        let mut op_calls = BTreeMap::new();
        op_calls.insert("da_step".to_string(), 12);
        let r = MetaReport {
            abi: 1 << 16,
            params: vec![("w".into(), vec![4, 4], 0)],
            persistent: vec![("m".into(), vec![4, 4], 0, 0)],
            det_persistent: vec![("mom".into(), vec![4, 4], 1)],
            param_bytes: 64,
            master_bytes: 64,
            grad_bytes: 64,
            act_bytes_est: 64,
            payload_bytes_est: 16,
            ingest_bytes_est: 160,
            host_ram_bytes_est: 192,
            op_calls,
            ingest_op_calls_per_peer: 3,
            ops_used: vec!["matmul@1".into()],
            value_dependent: false,
        };
        let back: MetaReport = ciborium::from_reader(r.to_cbor().as_slice()).unwrap();
        assert_eq!(back.params, r.params);
        assert_eq!(back.op_calls["da_step"], 12);
        assert_eq!(back.ingest_op_calls_per_peer, 3);
    }
}
