// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The async completion-result codec (ABI §7.5) — Phase B (track B1).
//!
//! Any capability call that cannot complete immediately returns an `OpId` (handle kind 10) and
//! completes through an `Event::Completion(op, result)` (event tag 6, ABI §3.3/§4.6). This module is
//! the host-side model + canonical-CBOR codec of the `result` — the [`CompletionResult`] — whose
//! wire shape is **fixed now** by ABI §7.5 (before the protocol is linked) so journals and SDKs stay
//! stable. The numeric assignments live in [`daemon_vhc_abi`] ([`COMPLETION_RESULT_OK`],
//! [`COMP_ERR_CANCELLED`], …) and the normative grammar is
//! [`daemon_vhc_abi::COMPLETION_RESULT_CDDL`]; this is the Rust codec that produces bytes validating
//! against it.
//!
//! Two consumers share it: the event codec ([`super::event`]) nests a completion-result as the
//! third element of a `completion-ev` frame (`[6, op, completion-result]`), and the journal (§8.3
//! tag 14 `completion-rec.result`) stores the standalone [`CompletionResult::encode`] bytes verbatim
//! (opaque `bstr` to the journal grammar). [`CompletionResult::to_value`]/[`from_value`] are the
//! shared primitives; `encode`/`decode` wrap them through the same RFC 8949 §4.2 canonical writer.

use ciborium::value::Value;

use daemon_vhc_abi::{
    comp_err_slug, COMPLETION_RESULT_ERR, COMPLETION_RESULT_OK, COMP_ERR_CANCELLED,
};
use daemon_vhc_proto::{from_canonical_slice, to_canonical_vec};

/// The success payload of a completion (ABI §7.5 `success-payload`): exactly one of an opaque
/// handle, a 32-byte content hash, or unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuccessPayload {
    /// A kind-tagged handle (buffer/tensor/stream — §7.2), e.g. from `payload_get`/`stream_read`.
    Handle(u64),
    /// A 32-byte content hash — the `payload_put` commitment.
    Hash([u8; 32]),
    /// Unit success — `stream_write`, a publish-ack, or a `cancel` whose target had no result.
    Unit,
}

/// The failure payload of a completion (ABI §7.5 `comp-error`): a numeric `code` (the
/// [`COMP_ERR_CANCELLED`]… assignments) plus an optional human-readable `detail`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompError {
    /// The `comp-error` code (0 = Cancelled … 7 = GrantExhausted; 8..=63 reserved).
    pub code: u64,
    /// An optional human-readable detail.
    pub detail: Option<String>,
}

impl CompError {
    /// A cancelled-operation error (`vhc@2::cancel`'s completion, ABI §7.5).
    #[must_use]
    pub fn cancelled() -> Self {
        Self {
            code: COMP_ERR_CANCELLED,
            detail: None,
        }
    }

    /// The stable slug of this error's code, or `None` if the code is reserved/unknown (ABI §7.5).
    #[must_use]
    pub fn slug(&self) -> Option<&'static str> {
        comp_err_slug(self.code)
    }
}

/// A completion result (ABI §7.5 `completion-result`): success with a payload, or a typed failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionResult {
    /// `[0, success-payload]` — the operation succeeded.
    Ok(SuccessPayload),
    /// `[1, comp-error]` — the operation failed with a typed code.
    Err(CompError),
}

/// Completion-result codec failures (host-side: event/journal decode + tests).
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CompletionCodecError {
    /// The bytes/value are not a well-formed `completion-result` (ABI §7.5).
    #[error("malformed completion-result: {0}")]
    Malformed(String),
}

impl CompletionResult {
    /// A cancellation completion (`Err(Cancelled)`), the completion `vhc@2::cancel` reports (§7.5).
    #[must_use]
    pub fn cancelled() -> Self {
        Self::Err(CompError::cancelled())
    }

    /// Build the CBOR value tree of this result (`[0, payload]` / `[1, comp-error]`, ABI §7.5), for
    /// nesting inside a `completion-ev` event frame.
    #[must_use]
    pub fn to_value(&self) -> Value {
        match self {
            Self::Ok(payload) => {
                let p = match payload {
                    SuccessPayload::Handle(h) => Value::from(*h),
                    SuccessPayload::Hash(h) => Value::Bytes(h.to_vec()),
                    SuccessPayload::Unit => Value::Null,
                };
                Value::Array(vec![Value::from(COMPLETION_RESULT_OK), p])
            }
            Self::Err(e) => {
                let mut m = vec![(Value::from("code"), Value::from(e.code))];
                if let Some(detail) = &e.detail {
                    m.push((Value::from("detail"), Value::from(detail.as_str())));
                }
                Value::Array(vec![Value::from(COMPLETION_RESULT_ERR), Value::Map(m)])
            }
        }
    }

    /// Decode a completion-result from its CBOR value tree (the inverse of [`to_value`], ABI §7.5).
    /// Fails closed on an unknown variant discriminant, a mis-typed success payload, or a
    /// `comp-error` missing its `code` (ABI §5.2/§7.5).
    ///
    /// # Errors
    ///
    /// [`CompletionCodecError::Malformed`] for any shape outside the §7.5 grammar.
    pub fn from_value(value: &Value) -> Result<Self, CompletionCodecError> {
        let Value::Array(items) = value else {
            return Err(CompletionCodecError::Malformed(
                "completion-result is not a CBOR array".into(),
            ));
        };
        let variant = as_u64(items.first(), "variant")?;
        match variant {
            v if v == COMPLETION_RESULT_OK => {
                let payload = match items.get(1) {
                    Some(Value::Integer(i)) => {
                        SuccessPayload::Handle(u64::try_from(i128::from(*i)).map_err(|_| {
                            CompletionCodecError::Malformed("handle out of u64 range".into())
                        })?)
                    }
                    Some(Value::Bytes(b)) => {
                        let h: [u8; 32] = b.as_slice().try_into().map_err(|_| {
                            CompletionCodecError::Malformed(
                                "success hash payload is not 32 bytes".into(),
                            )
                        })?;
                        SuccessPayload::Hash(h)
                    }
                    Some(Value::Null) => SuccessPayload::Unit,
                    _ => {
                        return Err(CompletionCodecError::Malformed(
                            "success payload is not a handle, 32-byte hash, or null".into(),
                        ))
                    }
                };
                Ok(Self::Ok(payload))
            }
            v if v == COMPLETION_RESULT_ERR => {
                let Some(Value::Map(entries)) = items.get(1) else {
                    return Err(CompletionCodecError::Malformed(
                        "failure variant does not carry a comp-error map".into(),
                    ));
                };
                let code = map_u64(entries, "code").ok_or_else(|| {
                    CompletionCodecError::Malformed("comp-error missing `code`".into())
                })?;
                let detail = match map_get(entries, "detail") {
                    None => None,
                    Some(Value::Text(s)) => Some(s.clone()),
                    Some(_) => {
                        return Err(CompletionCodecError::Malformed(
                            "comp-error `detail` is not a text string".into(),
                        ))
                    }
                };
                Ok(Self::Err(CompError { code, detail }))
            }
            other => Err(CompletionCodecError::Malformed(format!(
                "unknown completion-result variant discriminant {other} (fail closed)"
            ))),
        }
    }

    /// Encode this result to its standalone canonical-CBOR bytes (ABI §7.5) — the exact bytes stored
    /// in a journal tag-14 `completion-rec.result` (§8.3).
    ///
    /// # Errors
    ///
    /// [`CompletionCodecError::Malformed`] only on a canonical-encoder failure (the fixed value
    /// shapes here cannot produce one in practice).
    pub fn encode(&self) -> Result<Vec<u8>, CompletionCodecError> {
        to_canonical_vec(&self.to_value())
            .map_err(|e| CompletionCodecError::Malformed(format!("encode: {e}")))
    }

    /// Decode a result from standalone canonical-CBOR bytes (the inverse of [`encode`], ABI §7.5).
    ///
    /// # Errors
    ///
    /// [`CompletionCodecError::Malformed`] on malformed CBOR or any shape outside the §7.5 grammar.
    pub fn decode(bytes: &[u8]) -> Result<Self, CompletionCodecError> {
        let value: Value = from_canonical_slice(bytes)
            .map_err(|e| CompletionCodecError::Malformed(format!("decode: {e}")))?;
        Self::from_value(&value)
    }
}

// -- small typed accessors (shared shape with the event codec) -----------------------------------

fn as_u64(v: Option<&Value>, field: &str) -> Result<u64, CompletionCodecError> {
    match v {
        Some(Value::Integer(i)) => u64::try_from(i128::from(*i))
            .map_err(|_| CompletionCodecError::Malformed(format!("`{field}` out of u64 range"))),
        _ => Err(CompletionCodecError::Malformed(format!(
            "`{field}` missing or not an unsigned integer"
        ))),
    }
}

fn map_get<'a>(entries: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
    entries.iter().find_map(|(k, v)| match k {
        Value::Text(s) if s == key => Some(v),
        _ => None,
    })
}

fn map_u64(entries: &[(Value, Value)], key: &str) -> Option<u64> {
    match map_get(entries, key) {
        Some(Value::Integer(i)) => u64::try_from(i128::from(*i)).ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_abi::{COMP_ERR_HASH_MISMATCH, HANDLE_KIND_BUFFER};

    fn samples() -> Vec<CompletionResult> {
        vec![
            CompletionResult::Ok(SuccessPayload::Handle(daemon_vhc_abi::pack_handle(
                HANDLE_KIND_BUFFER,
                1,
                1,
            ))),
            CompletionResult::Ok(SuccessPayload::Hash([9u8; 32])),
            CompletionResult::Ok(SuccessPayload::Unit),
            CompletionResult::cancelled(),
            CompletionResult::Err(CompError {
                code: COMP_ERR_HASH_MISMATCH,
                detail: Some("expected != actual".into()),
            }),
        ]
    }

    #[test]
    fn every_variant_round_trips_through_bytes_and_value() {
        for r in samples() {
            let bytes = r.encode().unwrap();
            assert_eq!(CompletionResult::decode(&bytes).unwrap(), r);
            assert_eq!(CompletionResult::from_value(&r.to_value()).unwrap(), r);
        }
    }

    #[test]
    fn encoding_is_canonical_and_deterministic() {
        for r in samples() {
            assert_eq!(r.encode().unwrap(), r.encode().unwrap());
        }
    }

    #[test]
    fn cancel_completion_is_cancelled_error() {
        // vhc@2::cancel's completion reports Cancelled (ABI §7.5).
        let r = CompletionResult::cancelled();
        assert_eq!(
            r,
            CompletionResult::Err(CompError {
                code: COMP_ERR_CANCELLED,
                detail: None
            })
        );
        if let CompletionResult::Err(e) = &r {
            assert_eq!(e.slug(), Some("Cancelled"));
        }
    }

    #[test]
    fn malformed_results_fail_closed() {
        // Unknown variant discriminant.
        let unknown = Value::Array(vec![Value::from(9u64), Value::Null]);
        assert!(CompletionResult::from_value(&unknown).is_err());
        // Success payload of the wrong type (a text string is none of handle/hash/null).
        let bad_ok = Value::Array(vec![
            Value::from(COMPLETION_RESULT_OK),
            Value::from("not-a-payload"),
        ]);
        assert!(CompletionResult::from_value(&bad_ok).is_err());
        // A 31-byte "hash" is not a valid success hash payload.
        let bad_hash = Value::Array(vec![
            Value::from(COMPLETION_RESULT_OK),
            Value::Bytes(vec![0u8; 31]),
        ]);
        assert!(CompletionResult::from_value(&bad_hash).is_err());
        // Failure without a comp-error map.
        let bad_err = Value::Array(vec![Value::from(COMPLETION_RESULT_ERR), Value::from(0u64)]);
        assert!(CompletionResult::from_value(&bad_err).is_err());
    }
}
