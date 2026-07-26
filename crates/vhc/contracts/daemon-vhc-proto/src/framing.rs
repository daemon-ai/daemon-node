// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **bounded outer framing header** for a frozen genesis envelope
//! (`docs/specs/vhc-architecture-spec.md` §5.4 `[DI-6]`).
//!
//! Routing on the schema major is only genuinely pre-decode and resource-bounded if the reader can
//! obtain the major **without deserializing the payload**. The previous mechanism could not: the
//! schema peek deserializes the entire CBOR document into a value before it inspects the `schema`
//! member, so a refusal of an untrusted, arbitrarily large document had already paid the cost the
//! refusal was meant to avoid. A reader that must deserialize a document in order to discover that
//! it must refuse it has spent exactly what the refusal was meant to save.
//!
//! This header is therefore a fixed-width binary prefix, checkable in constant time and constant
//! memory:
//!
//! ```text
//! offset  size  field
//!      0     8  magic
//!      8     4  schema major        (little-endian u32)
//!     12     4  required features   (little-endian u32 bitfield)
//!     16     8  payload length      (little-endian u64, <= GENESIS_PAYLOAD_BYTES_MAX)
//!     24    32  payload digest      (blake3 of the payload bytes)
//! ```
//!
//! A host reads 56 bytes, refuses an unknown major, refuses a required feature it does not
//! implement, refuses an over-ceiling length, and only then commits memory and parse time to the
//! payload — whose digest it verifies before decoding.
//!
//! ## Two-directional compatibility is proven, not assumed
//!
//! The bump has to fail closed in **both** directions, or the property is assumed rather than
//! demonstrated: a reader of the current generation must refuse next-generation bytes, and a
//! next-generation reader must refuse previous-generation bytes. Neither direction implies the
//! other, and both have negative tests beside this module.

use crate::bytes::Hash;
use crate::error::VhcProtoError;
use crate::hash::blake3_hash;

/// The framing magic. Its final byte is a framing-format generation, distinct from the envelope
/// schema major: a change to the *header layout itself* moves this, a change to the payload schema
/// moves the major.
pub const GENESIS_FRAME_MAGIC: [u8; 8] = *b"vhc-gen\x01";

/// The exact size of the framing header, in bytes. Fixed by construction so a refusal is bounded.
pub const GENESIS_FRAME_HEADER_LEN: usize = 56;

/// The ceiling on a framed genesis payload. An envelope is a bounded description of a run, not a
/// data channel; a length above this is refused from the header alone, before a byte of payload is
/// read.
pub const GENESIS_PAYLOAD_BYTES_MAX: u64 = 4 * 1024 * 1024;

/// The reader features this build implements. A framing header whose `required_features` carries a
/// bit outside this mask names something this reader cannot evaluate, and is refused rather than
/// skipped — a requirement a reader silently ignores is not a requirement.
pub const GENESIS_SUPPORTED_FEATURES: u32 = GENESIS_FEATURE_EXECUTION_REQUIREMENTS;

/// Feature bit: the role entries carry the execution-requirement structure — the canonical Logical
/// Resource Plan, allowed backend classes, profile-certification requirements, the
/// hardware-independent minima and the selection scope with its frozen grant or equivalence
/// contract.
pub const GENESIS_FEATURE_EXECUTION_REQUIREMENTS: u32 = 1 << 0;

/// The parsed framing header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenesisFrameHeader {
    /// The envelope schema major the payload is written under.
    pub schema_major: u32,
    /// The bitfield of reader features the payload requires.
    pub required_features: u32,
    /// The payload length in bytes.
    pub payload_len: u64,
    /// blake3 of the payload bytes.
    pub payload_digest: Hash,
}

impl GenesisFrameHeader {
    /// Build a header over `payload`.
    ///
    /// # Errors
    /// [`VhcProtoError::Validation`] if the payload is above [`GENESIS_PAYLOAD_BYTES_MAX`].
    pub fn for_payload(
        schema_major: u32,
        required_features: u32,
        payload: &[u8],
    ) -> Result<Self, VhcProtoError> {
        let payload_len = payload.len() as u64;
        if payload_len > GENESIS_PAYLOAD_BYTES_MAX {
            return Err(VhcProtoError::Validation(format!(
                "genesis payload is {payload_len} bytes, above the framing ceiling \
                 {GENESIS_PAYLOAD_BYTES_MAX}"
            )));
        }
        Ok(Self {
            schema_major,
            required_features,
            payload_len,
            payload_digest: blake3_hash(payload),
        })
    }

    /// The header's fixed-width encoding.
    #[must_use]
    pub fn encode(&self) -> [u8; GENESIS_FRAME_HEADER_LEN] {
        let mut out = [0u8; GENESIS_FRAME_HEADER_LEN];
        out[0..8].copy_from_slice(&GENESIS_FRAME_MAGIC);
        out[8..12].copy_from_slice(&self.schema_major.to_le_bytes());
        out[12..16].copy_from_slice(&self.required_features.to_le_bytes());
        out[16..24].copy_from_slice(&self.payload_len.to_le_bytes());
        out[24..56].copy_from_slice(&self.payload_digest.0);
        out
    }

    /// Read the header from the front of `bytes` **without** touching the payload. This is the
    /// bounded pre-payload read: it consumes a fixed 56 bytes and allocates nothing.
    ///
    /// # Errors
    /// [`VhcProtoError::Validation`] if the input is short or the magic does not match.
    pub fn peek(bytes: &[u8]) -> Result<Self, VhcProtoError> {
        let head: &[u8; GENESIS_FRAME_HEADER_LEN] = bytes
            .get(..GENESIS_FRAME_HEADER_LEN)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| {
                VhcProtoError::Validation(format!(
                    "framed genesis is shorter than its {GENESIS_FRAME_HEADER_LEN}-byte header"
                ))
            })?;
        if head[0..8] != GENESIS_FRAME_MAGIC {
            return Err(VhcProtoError::Validation(
                "framed genesis does not carry the framing magic".into(),
            ));
        }
        Ok(Self {
            schema_major: u32::from_le_bytes(head[8..12].try_into().expect("4 bytes")),
            required_features: u32::from_le_bytes(head[12..16].try_into().expect("4 bytes")),
            payload_len: u64::from_le_bytes(head[16..24].try_into().expect("8 bytes")),
            payload_digest: Hash(head[24..56].try_into().expect("32 bytes")),
        })
    }

    /// Whether this reader may proceed to the payload: the major is the one it implements, every
    /// required feature is one it implements, and the length is within the ceiling.
    ///
    /// # Errors
    /// [`VhcProtoError::Validation`] naming which of the three refused.
    pub fn accept(&self, implemented_major: u32) -> Result<(), VhcProtoError> {
        if self.schema_major != implemented_major {
            return Err(VhcProtoError::Validation(format!(
                "genesis schema major {} is not the major this build implements \
                 ({implemented_major})",
                self.schema_major
            )));
        }
        let unknown = self.required_features & !GENESIS_SUPPORTED_FEATURES;
        if unknown != 0 {
            return Err(VhcProtoError::Validation(format!(
                "genesis requires reader features {unknown:#010x} this build does not implement"
            )));
        }
        if self.payload_len > GENESIS_PAYLOAD_BYTES_MAX {
            return Err(VhcProtoError::Validation(format!(
                "genesis payload length {} is above the framing ceiling \
                 {GENESIS_PAYLOAD_BYTES_MAX}",
                self.payload_len
            )));
        }
        Ok(())
    }
}

/// Prepend a framing header to `payload`.
///
/// # Errors
/// [`VhcProtoError::Validation`] if the payload is above the framing ceiling.
pub fn frame(
    schema_major: u32,
    required_features: u32,
    payload: &[u8],
) -> Result<Vec<u8>, VhcProtoError> {
    let header = GenesisFrameHeader::for_payload(schema_major, required_features, payload)?;
    let mut out = Vec::with_capacity(GENESIS_FRAME_HEADER_LEN + payload.len());
    out.extend_from_slice(&header.encode());
    out.extend_from_slice(payload);
    Ok(out)
}

/// Check the framing and return the payload slice, refusing **before** any payload decode.
///
/// The order is the whole point: peek the fixed header, refuse on major / required features /
/// length, then confirm the payload is exactly as long as the header claims and hashes to the
/// digest the header carries. Only a caller past all of that has a payload worth decoding.
///
/// # Errors
/// [`VhcProtoError::Validation`] naming the first check that refused.
pub fn unframe(bytes: &[u8], implemented_major: u32) -> Result<&[u8], VhcProtoError> {
    let header = GenesisFrameHeader::peek(bytes)?;
    header.accept(implemented_major)?;
    let payload = bytes
        .get(GENESIS_FRAME_HEADER_LEN..)
        .ok_or_else(|| VhcProtoError::Validation("framed genesis has no payload".into()))?;
    if payload.len() as u64 != header.payload_len {
        return Err(VhcProtoError::Validation(format!(
            "framed genesis carries {} payload bytes; its header declares {}",
            payload.len(),
            header.payload_len
        )));
    }
    if blake3_hash(payload) != header.payload_digest {
        return Err(VhcProtoError::Validation(
            "framed genesis payload does not match the digest its header declares".into(),
        ));
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genesis::GENESIS_SCHEMA_MAJOR;

    const NEXT_GENERATION: u32 = GENESIS_SCHEMA_MAJOR + 1;
    const PREVIOUS_GENERATION: u32 = GENESIS_SCHEMA_MAJOR - 1;

    #[test]
    fn a_framed_payload_round_trips() {
        let payload = b"a canonical genesis payload".to_vec();
        let framed = frame(
            GENESIS_SCHEMA_MAJOR,
            GENESIS_FEATURE_EXECUTION_REQUIREMENTS,
            &payload,
        )
        .unwrap();
        assert_eq!(framed.len(), GENESIS_FRAME_HEADER_LEN + payload.len());
        assert_eq!(unframe(&framed, GENESIS_SCHEMA_MAJOR).unwrap(), payload);

        let header = GenesisFrameHeader::peek(&framed).unwrap();
        assert_eq!(header.schema_major, GENESIS_SCHEMA_MAJOR);
        assert_eq!(header.payload_len, payload.len() as u64);
        assert_eq!(header.payload_digest, blake3_hash(&payload));
    }

    /// Direction one: this generation's reader refuses next-generation bytes.
    #[test]
    fn this_reader_refuses_next_generation_bytes() {
        let framed = frame(NEXT_GENERATION, 0, b"next").unwrap();
        let err = unframe(&framed, GENESIS_SCHEMA_MAJOR).unwrap_err();
        assert!(err.to_string().contains("is not the major this build"));
    }

    /// Direction two: a next-generation reader refuses previous-generation bytes. Neither
    /// direction implies the other, so both are asserted.
    #[test]
    fn a_next_generation_reader_refuses_previous_generation_bytes() {
        let framed = frame(PREVIOUS_GENERATION, 0, b"previous").unwrap();
        let err = unframe(&framed, GENESIS_SCHEMA_MAJOR).unwrap_err();
        assert!(err.to_string().contains("is not the major this build"));

        // And symmetrically, a reader one generation ahead refuses today's bytes.
        let today = frame(GENESIS_SCHEMA_MAJOR, 0, b"today").unwrap();
        assert!(unframe(&today, NEXT_GENERATION).is_err());
    }

    /// A required feature this reader does not implement is refused, not skipped. An optional
    /// member a reader may ignore is indistinguishable from not expressing the requirement at all.
    #[test]
    fn an_unimplemented_required_feature_fails_closed() {
        let framed = frame(GENESIS_SCHEMA_MAJOR, 1 << 31, b"payload").unwrap();
        let err = unframe(&framed, GENESIS_SCHEMA_MAJOR).unwrap_err();
        assert!(err.to_string().contains("reader features"));
    }

    /// The refusal is bounded: the header is read and rejected from a fixed 56 bytes, so an
    /// enormous declared payload never becomes an allocation.
    #[test]
    fn an_over_ceiling_length_is_refused_from_the_header_alone() {
        let mut header = GenesisFrameHeader::for_payload(GENESIS_SCHEMA_MAJOR, 0, b"x").unwrap();
        header.payload_len = GENESIS_PAYLOAD_BYTES_MAX + 1;
        let encoded = header.encode();
        assert_eq!(encoded.len(), GENESIS_FRAME_HEADER_LEN);

        let peeked = GenesisFrameHeader::peek(&encoded).unwrap();
        let err = peeked.accept(GENESIS_SCHEMA_MAJOR).unwrap_err();
        assert!(err.to_string().contains("above the framing ceiling"));
    }

    #[test]
    fn a_truncated_or_unmagicked_frame_is_refused() {
        assert!(GenesisFrameHeader::peek(b"short").is_err());
        let mut framed = frame(GENESIS_SCHEMA_MAJOR, 0, b"payload").unwrap();
        framed[0] ^= 0xff;
        assert!(GenesisFrameHeader::peek(&framed)
            .unwrap_err()
            .to_string()
            .contains("framing magic"));
    }

    #[test]
    fn a_tampered_payload_is_caught_by_the_header_digest() {
        let mut framed = frame(GENESIS_SCHEMA_MAJOR, 0, b"payload").unwrap();
        let last = framed.len() - 1;
        framed[last] ^= 0xff;
        assert!(unframe(&framed, GENESIS_SCHEMA_MAJOR)
            .unwrap_err()
            .to_string()
            .contains("does not match the digest"));

        framed.pop();
        assert!(unframe(&framed, GENESIS_SCHEMA_MAJOR)
            .unwrap_err()
            .to_string()
            .contains("payload bytes"));
    }
}
