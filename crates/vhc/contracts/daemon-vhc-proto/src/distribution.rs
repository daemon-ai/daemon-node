// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! On-plane carriage of the §12.3 **distribution records** — certificates and revocations
//! travelling over the control plane beside (never inside) the §12.1 signed frames.
//!
//! The ABI companion fixes the record shapes and their domain tags (`daemon-vhc/cert/2`,
//! `daemon-vhc/revocation/2`) and says they "propagate on the control plane best-effort"; this
//! module fixes HOW a receiver tells a distribution record from a frame on one shared byte
//! plane, without decoding either speculatively:
//!
//! - a §12.1 **frame** is a top-level canonical-CBOR **array** `[envelope, payload, sig]`;
//! - a **distribution record** is a top-level single-entry **map** whose key names the record
//!   kind (`"cert"` / `"revocation"`) and whose value is the §12.3 record verbatim.
//!
//! The two top-level shapes are disjoint, so classification is structural: a receiver attempts
//! the [`DistributionRecord`] decode first (cheap; fails immediately on an array) and hands
//! everything else to the frame attach. Both directions stay MECHANISM ([OWN-1]): no round
//! vocabulary, no schema-crate edge — a host can carry, verify, and cache these records without
//! ever giving payload bytes meaning.
//!
//! Trust is the receiver's job, exactly as with frames: an ingested certificate counts only
//! after its chain verifies to a genesis-trusted base identity, and a revocation only through
//! the replay-protected ledger — an unverified record must never advance any trust state
//! (certificate floors included).

use serde::{Deserialize, Serialize};

use crate::canonical::{from_canonical_slice, to_canonical_vec};
use crate::cert::RunKeyCertificate;
use crate::error::VhcProtoError;
use crate::revocation::RunKeyRevocation;

/// One §12.3 record on the control plane: a certificate or a revocation, externally tagged by
/// kind. Additive: a future record kind is a new map key, refused typed by old receivers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributionRecord {
    /// A run-key certificate (`{ body, base_identity, sig }`, domain `daemon-vhc/cert/2`).
    #[serde(rename = "cert")]
    Cert(RunKeyCertificate),
    /// A run-key revocation (`{ body, base_identity, sig }`, domain `daemon-vhc/revocation/2`).
    #[serde(rename = "revocation")]
    Revocation(RunKeyRevocation),
}

impl DistributionRecord {
    /// Encode to the canonical-CBOR bytes published on the control plane.
    ///
    /// # Errors
    /// Canonical-CBOR encode failure.
    pub fn to_bytes(&self) -> Result<Vec<u8>, VhcProtoError> {
        to_canonical_vec(self)
    }

    /// Decode control-plane bytes as a distribution record. A §12.1 frame (a top-level array)
    /// fails this decode structurally — the caller's classification signal, not an error to
    /// report.
    ///
    /// # Errors
    /// The bytes are not a distribution record.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VhcProtoError> {
        from_canonical_slice(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes::Hash;
    use crate::cert::CertScope;
    use crate::sign::{peer_id, SigningKey};

    fn cert() -> RunKeyCertificate {
        let base = SigningKey::from_bytes(&[7; 32]);
        let run_key = peer_id(&SigningKey::from_bytes(&[9; 32]));
        RunKeyCertificate::issue(
            &base,
            CertScope {
                run_id: Hash([1; 32]),
                epoch: 0,
                role: "trainer".into(),
                instance: 2,
                module_hash: Hash([3; 32]),
            },
            run_key,
        )
        .expect("issue")
    }

    #[test]
    fn records_round_trip_and_stay_verifiable() {
        let record = DistributionRecord::Cert(cert());
        let bytes = record.to_bytes().expect("encode");
        let back = DistributionRecord::from_bytes(&bytes).expect("decode");
        assert_eq!(back, record);
        let DistributionRecord::Cert(c) = back else {
            panic!("kind preserved");
        };
        c.verify_chain().expect("carried record still verifies");
    }

    #[test]
    fn a_frame_shaped_array_is_not_a_distribution_record() {
        // The §12.1 wire form is a top-level array — the structural disambiguator.
        let frame = ciborium::value::Value::Array(vec![
            ciborium::value::Value::Map(vec![]),
            ciborium::value::Value::Bytes(b"payload".to_vec()),
            ciborium::value::Value::Bytes(vec![0; 64]),
        ]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&frame, &mut bytes).expect("encode");
        assert!(DistributionRecord::from_bytes(&bytes).is_err());
    }

    #[test]
    fn a_distribution_record_is_a_single_entry_map() {
        // Pin the wire shape the disambiguation rule documents: map, one entry, kind key.
        let bytes = DistributionRecord::Cert(cert()).to_bytes().expect("encode");
        let v: ciborium::value::Value = ciborium::de::from_reader(&bytes[..]).expect("cbor");
        let ciborium::value::Value::Map(entries) = v else {
            panic!("top-level map");
        };
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].0,
            ciborium::value::Value::Text("cert".to_string())
        );
    }
}
