// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The major-2 event-frame codec (ABI §4.2 vocabulary, §5.1 encoding, §5.2 versioning).
//!
//! An event frame is **canonical CBOR**: a definite-length array whose first element is the
//! integer event tag ([`daemon_vhc_abi`]'s `EV_TAG_*`), followed by tag-specific positional
//! fields. Canonicality matters normatively (ABI §5.3): for a given logical event the host MUST
//! produce a byte-identical frame every time — the frame bytes are what `next_event` writes into
//! the guest buffer AND what the journal records verbatim (§8.3 tag 1 `frame`), so replay equality
//! and any digest over the delivered stream are defined over these bytes. Encoding rides
//! [`daemon_vhc_proto::to_canonical_vec`] — the same RFC 8949 §4.2 deterministic profile as every
//! other consensus-critical byte in the tree.
//!
//! This host-side codec covers the Phase A closed subset (`{Frame, PayloadReady, Timer, Budget,
//! Stop, Quiesce}`, ABI §4.2) **plus the Phase B `Completion` variant** (tag 6, ABI §4.6/§7.5 —
//! track B1's async completion protocol): every non-immediate capability call returns an `OpId` and
//! completes through `Event::Completion(op, result)`. The reserved `Fence` tag (5) is still **not a
//! variant** of [`EventV2`], so a host without `compute@2` cannot deliver it by construction (§4.6);
//! it arrives with Phase C. Decoding accepts and ignores trailing fields beyond those known
//! (additive minors, §5.2) and fails closed on an unknown tag — the decoder here serves the
//! journal/replay verifier and tests; the guest-side fail-closed trap (§5.2) is the SDK's
//! obligation.
//!
//! Whether the host *delivers* a `Completion` is governed by minor negotiation (a minor-0 host with
//! no completion sources never produces one); representability here is what lets the completion
//! protocol, journal (§8.3 tag 14), and replay verifier all speak one shape.

use ciborium::value::Value;

use daemon_vhc_abi::{
    EV_TAG_BUDGET, EV_TAG_COMPLETION, EV_TAG_FRAME, EV_TAG_PAYLOAD_READY, EV_TAG_QUIESCE,
    EV_TAG_STOP, EV_TAG_TIMER,
};
use daemon_vhc_proto::to_canonical_vec;

use super::completion::CompletionResult;

/// Metadata beside a staged payload announcement (ABI §4.2 `payload-meta`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadMeta {
    /// Staged byte size.
    pub size: u64,
    /// Staged kind (`STAGED_KIND_*`: 0 = bytes, 1 = bridge batch, 2 = bridge update container).
    pub kind: u64,
    /// The channel whose frame referenced this payload, if any.
    pub channel: Option<u32>,
}

/// The throttle sub-report of a `Budget` event (ABI §4.2 `budget-report.throttle`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThrottleReport {
    /// Whether the instance is paused.
    pub paused: bool,
    /// Duty cycle, `0..=100`.
    pub duty_pct: u64,
    /// VRAM cap in raw bytes; `0` = uncapped (ABI §9.6 units rule).
    pub vram_cap_bytes: u64,
}

/// A host-initiated budget/pressure/throttle notification (ABI §4.2 `budget-report`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetReport {
    /// Remaining-fuel class: 0 = ample, 1 = low, 2 = critical.
    pub fuel: u64,
    /// Memory-pressure class: 0 = none, 1 = elevated, 2 = critical.
    pub mem: u64,
    /// Current throttle posture.
    pub throttle: ThrottleReport,
}

/// A major-2 event: the Phase-A closed subset (ABI §4.2) plus the Phase-B `Completion` variant
/// (tag 6, §4.6/§7.5 — track B1).
///
/// The reserved `Fence` (tag 5) variant is deliberately absent: a host without `compute@2` MUST NOT
/// deliver it (§4.6), and leaving it unrepresentable makes that a compile-time property of the pump
/// (it arrives with Phase C). `Completion` **is** representable now: track B1's completion protocol
/// generalizes every non-immediate capability call to an `OpId` + `Event::Completion(op, result)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventV2 {
    /// Tag 0 — a verified signed control frame (§4.3). `payload` is opaque module-authored bytes.
    Frame {
        /// The declared channel the frame arrived on (§6.2).
        channel: u32,
        /// The sender's durable per-stream sequence number (§12.2).
        seq: u64,
        /// The signing identity (ed25519 public key).
        sender: [u8; 32],
        /// Opaque module-authored payload bytes; the host never interprets them.
        payload: Vec<u8>,
    },
    /// Tag 1 — content-addressed bytes staged under `staging_id` (§4.3).
    PayloadReady {
        /// The token the guest passes to `read_back` (§6.4).
        staging_id: u64,
        /// blake3 of the staged bytes.
        hash: [u8; 32],
        /// Size/kind/channel metadata.
        meta: PayloadMeta,
    },
    /// Tag 2 — a one-shot logical-clock timer elapsed (§6.3).
    Timer {
        /// The ID `set_timer` returned.
        timer_id: u64,
        /// Logical delivery time in ms (§6.5).
        fired_at: u64,
    },
    /// Tag 3 — a budget/pressure/throttle notification (§4.3).
    Budget {
        /// The fully-defined report body.
        report: BudgetReport,
    },
    /// Tag 6 — the async result of a non-immediate capability call (§4.6/§7.5, track B1).
    Completion {
        /// The `OpId` the originating capability call returned (handle kind 10, §7.2).
        op: u64,
        /// The typed success/failure result (§7.5).
        result: CompletionResult,
    },
    /// Tag 4 — terminal; after delivery every import traps `PhaseViolation` (§4.4).
    Stop {
        /// `STOP_REASON_*` (0 = RunComplete, 1 = LeaveRequested, 2 = Fault, 3 = OwnerPolicy).
        reason: u64,
    },
    /// Tag 7 — opens a bounded drain to `QuiesceReady` (§4.4).
    Quiesce {
        /// `QUIESCE_REASON_*` (0 = Upgrade, 1 = Throttle).
        reason: u64,
        /// The effective drain deadline (logical ms; `min(owner setting, lane maximum)`, §4.4).
        deadline_ms: u64,
    },
}

/// Event-frame codec failures (host-side: journal/replay verification and tests).
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EventCodecError {
    /// The frame is not the required shape (not an array, missing/mis-typed fields, bad lengths).
    #[error("malformed event frame: {0}")]
    Malformed(String),
    /// The leading tag is not a Phase-A deliverable tag (reserved, or from a future minor). Fails
    /// closed per ABI §5.2.
    #[error("unknown event tag {0} (fail closed, ABI §5.2)")]
    UnknownTag(u64),
}

/// Encode an event as its canonical frame bytes (ABI §5.1/§5.3).
///
/// # Errors
///
/// [`EventCodecError::Malformed`] only on a canonical-encoder failure (which the fixed value
/// shapes below cannot produce in practice).
pub fn encode_event_frame(event: &EventV2) -> Result<Vec<u8>, EventCodecError> {
    let tree = match event {
        EventV2::Frame {
            channel,
            seq,
            sender,
            payload,
        } => Value::Array(vec![
            Value::from(EV_TAG_FRAME),
            Value::from(*channel),
            Value::from(*seq),
            Value::Bytes(sender.to_vec()),
            Value::Bytes(payload.clone()),
        ]),
        EventV2::PayloadReady {
            staging_id,
            hash,
            meta,
        } => {
            let mut m = vec![
                (Value::from("size"), Value::from(meta.size)),
                (Value::from("kind"), Value::from(meta.kind)),
            ];
            if let Some(ch) = meta.channel {
                m.push((Value::from("channel"), Value::from(ch)));
            }
            Value::Array(vec![
                Value::from(EV_TAG_PAYLOAD_READY),
                Value::from(*staging_id),
                Value::Bytes(hash.to_vec()),
                Value::Map(m),
            ])
        }
        EventV2::Timer { timer_id, fired_at } => Value::Array(vec![
            Value::from(EV_TAG_TIMER),
            Value::from(*timer_id),
            Value::from(*fired_at),
        ]),
        EventV2::Budget { report } => Value::Array(vec![
            Value::from(EV_TAG_BUDGET),
            Value::Map(vec![
                (Value::from("fuel"), Value::from(report.fuel)),
                (Value::from("mem"), Value::from(report.mem)),
                (
                    Value::from("throttle"),
                    Value::Map(vec![
                        (Value::from("paused"), Value::Bool(report.throttle.paused)),
                        (
                            Value::from("duty_pct"),
                            Value::from(report.throttle.duty_pct),
                        ),
                        (
                            Value::from("vram_cap_bytes"),
                            Value::from(report.throttle.vram_cap_bytes),
                        ),
                    ]),
                ),
            ]),
        ]),
        EventV2::Completion { op, result } => Value::Array(vec![
            Value::from(EV_TAG_COMPLETION),
            Value::from(*op),
            result.to_value(),
        ]),
        EventV2::Stop { reason } => {
            Value::Array(vec![Value::from(EV_TAG_STOP), Value::from(*reason)])
        }
        EventV2::Quiesce {
            reason,
            deadline_ms,
        } => Value::Array(vec![
            Value::from(EV_TAG_QUIESCE),
            Value::from(*reason),
            Value::from(*deadline_ms),
        ]),
    };
    to_canonical_vec(&tree).map_err(|e| EventCodecError::Malformed(format!("encode: {e}")))
}

/// Decode a canonical event frame, tolerating trailing fields beyond those known (additive
/// minors, ABI §5.2) and failing closed on an unknown tag.
///
/// # Errors
///
/// [`EventCodecError::UnknownTag`] for a tag outside the Phase-A deliverable subset (including
/// the reserved `Fence`/`Completion` tags); [`EventCodecError::Malformed`] for anything that is
/// not a well-formed frame of the tagged shape.
pub fn decode_event_frame(bytes: &[u8]) -> Result<EventV2, EventCodecError> {
    let tree: Value = ciborium::de::from_reader(bytes)
        .map_err(|e| EventCodecError::Malformed(format!("decode: {e}")))?;
    let Value::Array(items) = tree else {
        return Err(EventCodecError::Malformed(
            "event frame is not a CBOR array".into(),
        ));
    };
    let tag = as_u64(items.first(), "tag")?;
    match tag {
        t if t == EV_TAG_FRAME => Ok(EventV2::Frame {
            channel: as_u32(items.get(1), "channel")?,
            seq: as_u64(items.get(2), "seq")?,
            sender: as_hash32(items.get(3), "sender")?,
            payload: as_bytes(items.get(4), "payload")?,
        }),
        t if t == EV_TAG_PAYLOAD_READY => Ok(EventV2::PayloadReady {
            staging_id: as_u64(items.get(1), "staging_id")?,
            hash: as_hash32(items.get(2), "hash")?,
            meta: decode_meta(items.get(3))?,
        }),
        t if t == EV_TAG_TIMER => Ok(EventV2::Timer {
            timer_id: as_u64(items.get(1), "timer_id")?,
            fired_at: as_u64(items.get(2), "fired_at")?,
        }),
        t if t == EV_TAG_BUDGET => Ok(EventV2::Budget {
            report: decode_budget(items.get(1))?,
        }),
        t if t == EV_TAG_COMPLETION => Ok(EventV2::Completion {
            op: as_u64(items.get(1), "op")?,
            result: CompletionResult::from_value(items.get(2).ok_or_else(|| {
                EventCodecError::Malformed("completion frame missing `result`".into())
            })?)
            .map_err(|e| EventCodecError::Malformed(e.to_string()))?,
        }),
        t if t == EV_TAG_STOP => Ok(EventV2::Stop {
            reason: as_u64(items.get(1), "reason")?,
        }),
        t if t == EV_TAG_QUIESCE => Ok(EventV2::Quiesce {
            reason: as_u64(items.get(1), "reason")?,
            deadline_ms: as_u64(items.get(2), "deadline_ms")?,
        }),
        other => Err(EventCodecError::UnknownTag(other)),
    }
}

fn decode_meta(v: Option<&Value>) -> Result<PayloadMeta, EventCodecError> {
    let entries = as_map(v, "meta")?;
    Ok(PayloadMeta {
        size: map_u64(entries, "size")?,
        kind: map_u64(entries, "kind")?,
        channel: map_u64_opt(entries, "channel")?.map(|c| c as u32),
    })
}

fn decode_budget(v: Option<&Value>) -> Result<BudgetReport, EventCodecError> {
    let entries = as_map(v, "report")?;
    let throttle_v = map_get(entries, "throttle")
        .ok_or_else(|| EventCodecError::Malformed("report missing `throttle`".into()))?;
    let throttle = as_map(Some(throttle_v), "throttle")?;
    let paused = match map_get(throttle, "paused") {
        Some(Value::Bool(b)) => *b,
        _ => {
            return Err(EventCodecError::Malformed(
                "throttle.paused is not a bool".into(),
            ))
        }
    };
    Ok(BudgetReport {
        fuel: map_u64(entries, "fuel")?,
        mem: map_u64(entries, "mem")?,
        throttle: ThrottleReport {
            paused,
            duty_pct: map_u64(throttle, "duty_pct")?,
            vram_cap_bytes: map_u64(throttle, "vram_cap_bytes")?,
        },
    })
}

// -- small typed accessors (positional fields + string-keyed maps) -------------------------------

fn as_u64(v: Option<&Value>, field: &str) -> Result<u64, EventCodecError> {
    match v {
        Some(Value::Integer(i)) => u64::try_from(i128::from(*i))
            .map_err(|_| EventCodecError::Malformed(format!("`{field}` out of u64 range"))),
        _ => Err(EventCodecError::Malformed(format!(
            "`{field}` missing or not an unsigned integer"
        ))),
    }
}

fn as_u32(v: Option<&Value>, field: &str) -> Result<u32, EventCodecError> {
    u32::try_from(as_u64(v, field)?)
        .map_err(|_| EventCodecError::Malformed(format!("`{field}` out of u32 range")))
}

fn as_bytes(v: Option<&Value>, field: &str) -> Result<Vec<u8>, EventCodecError> {
    match v {
        Some(Value::Bytes(b)) => Ok(b.clone()),
        _ => Err(EventCodecError::Malformed(format!(
            "`{field}` missing or not a byte string"
        ))),
    }
}

fn as_hash32(v: Option<&Value>, field: &str) -> Result<[u8; 32], EventCodecError> {
    let b = as_bytes(v, field)?;
    b.try_into()
        .map_err(|_| EventCodecError::Malformed(format!("`{field}` is not 32 bytes")))
}

fn as_map<'a>(v: Option<&'a Value>, field: &str) -> Result<&'a [(Value, Value)], EventCodecError> {
    match v {
        Some(Value::Map(entries)) => Ok(entries),
        _ => Err(EventCodecError::Malformed(format!(
            "`{field}` missing or not a map"
        ))),
    }
}

fn map_get<'a>(entries: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
    entries.iter().find_map(|(k, v)| match k {
        Value::Text(s) if s == key => Some(v),
        _ => None,
    })
}

fn map_u64(entries: &[(Value, Value)], key: &str) -> Result<u64, EventCodecError> {
    as_u64(map_get(entries, key), key)
}

fn map_u64_opt(entries: &[(Value, Value)], key: &str) -> Result<Option<u64>, EventCodecError> {
    match map_get(entries, key) {
        None => Ok(None),
        some => as_u64(some, key).map(Some),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::completion::{CompError, SuccessPayload};
    use daemon_vhc_abi::{
        COMP_ERR_TIMEOUT, EV_TAG_COMPLETION, EV_TAG_FENCE, HANDLE_KIND_BUFFER,
        QUIESCE_REASON_UPGRADE, STOP_REASON_RUN_COMPLETE,
    };

    fn samples() -> Vec<EventV2> {
        vec![
            EventV2::Frame {
                channel: 0,
                seq: 42,
                sender: [7u8; 32],
                payload: b"opaque-module-bytes".to_vec(),
            },
            EventV2::PayloadReady {
                staging_id: 9,
                hash: [3u8; 32],
                meta: PayloadMeta {
                    size: 4096,
                    kind: 0,
                    channel: Some(0),
                },
            },
            EventV2::PayloadReady {
                staging_id: 10,
                hash: [4u8; 32],
                meta: PayloadMeta {
                    size: 1,
                    kind: 2,
                    channel: None,
                },
            },
            EventV2::Timer {
                timer_id: 1,
                fired_at: 1500,
            },
            EventV2::Budget {
                report: BudgetReport {
                    fuel: 1,
                    mem: 0,
                    throttle: ThrottleReport {
                        paused: false,
                        duty_pct: 100,
                        vram_cap_bytes: 0,
                    },
                },
            },
            EventV2::Completion {
                op: daemon_vhc_abi::pack_handle(daemon_vhc_abi::HANDLE_KIND_OP_ID, 1, 7),
                result: CompletionResult::Ok(SuccessPayload::Handle(daemon_vhc_abi::pack_handle(
                    HANDLE_KIND_BUFFER,
                    1,
                    3,
                ))),
            },
            EventV2::Completion {
                op: daemon_vhc_abi::pack_handle(daemon_vhc_abi::HANDLE_KIND_OP_ID, 1, 8),
                result: CompletionResult::Err(CompError {
                    code: COMP_ERR_TIMEOUT,
                    detail: Some("fetch deadline".into()),
                }),
            },
            EventV2::Stop {
                reason: STOP_REASON_RUN_COMPLETE,
            },
            EventV2::Quiesce {
                reason: QUIESCE_REASON_UPGRADE,
                deadline_ms: 30_000,
            },
        ]
    }

    #[test]
    fn every_phase_a_event_round_trips() {
        for ev in samples() {
            let bytes = encode_event_frame(&ev).unwrap();
            let back = decode_event_frame(&bytes).unwrap();
            assert_eq!(back, ev);
        }
    }

    #[test]
    fn encoding_is_canonical_and_deterministic() {
        // ABI §5.3: byte-identical frames for the same logical event, every time.
        for ev in samples() {
            assert_eq!(
                encode_event_frame(&ev).unwrap(),
                encode_event_frame(&ev).unwrap()
            );
        }
        // Positional array with the tag as the first element (§5.1): a small frame's head is a
        // definite-length array head (major 4) and its first item is the integer tag.
        let bytes = encode_event_frame(&EventV2::Stop { reason: 0 }).unwrap();
        assert_eq!(bytes[0] >> 5, 4, "definite-length array head");
        assert_eq!(bytes[1], 0x04, "leading integer event tag 4 (Stop)");
    }

    #[test]
    fn fence_and_future_tags_fail_closed_but_completion_decodes() {
        // ABI §4.6/§5.2: `Fence` (tag 5) is still reserved (Phase C) and a future-minor tag is
        // unknown — both fail closed. `Completion` (tag 6) is now representable (track B1).
        for tag in [EV_TAG_FENCE, 63u64] {
            let bytes =
                to_canonical_vec(&Value::Array(vec![Value::from(tag), Value::from(1u64)])).unwrap();
            assert_eq!(
                decode_event_frame(&bytes).unwrap_err(),
                EventCodecError::UnknownTag(tag)
            );
        }
        // A well-formed completion frame decodes to the `Completion` variant.
        let completion = EventV2::Completion {
            op: 5,
            result: CompletionResult::cancelled(),
        };
        let bytes = encode_event_frame(&completion).unwrap();
        assert_eq!(decode_event_frame(&bytes).unwrap(), completion);
    }

    #[test]
    fn completion_frame_is_tag6_positional() {
        // ABI §4.2 `completion-ev = [6, op, completion-result]`.
        let bytes = encode_event_frame(&EventV2::Completion {
            op: 1,
            result: CompletionResult::Ok(SuccessPayload::Unit),
        })
        .unwrap();
        assert_eq!(bytes[0] >> 5, 4, "definite-length array head");
        assert_eq!(bytes[1], 0x06, "leading integer event tag 6 (Completion)");
    }

    #[test]
    fn malformed_completion_result_is_a_malformed_frame() {
        // A completion frame whose nested result is not a valid completion-result fails closed as a
        // malformed frame (the completion codec error is surfaced through the event codec).
        let bad = Value::Array(vec![
            Value::from(EV_TAG_COMPLETION),
            Value::from(1u64),
            Value::Array(vec![Value::from(9u64), Value::Null]), // unknown result discriminant
        ]);
        let bytes = to_canonical_vec(&bad).unwrap();
        assert!(matches!(
            decode_event_frame(&bytes),
            Err(EventCodecError::Malformed(_))
        ));
    }

    #[test]
    fn trailing_fields_are_ignored_additively() {
        // ABI §5.2: fields are append-only within a tag; a decoder accepts and ignores trailing
        // fields beyond those it knows.
        let future_timer = Value::Array(vec![
            Value::from(EV_TAG_TIMER),
            Value::from(5u64),
            Value::from(777u64),
            Value::from("a-future-minor-field"),
        ]);
        let bytes = to_canonical_vec(&future_timer).unwrap();
        assert_eq!(
            decode_event_frame(&bytes).unwrap(),
            EventV2::Timer {
                timer_id: 5,
                fired_at: 777
            }
        );
    }

    #[test]
    fn malformed_frames_are_typed_errors() {
        // Not an array.
        let not_array = to_canonical_vec(&Value::from(3u64)).unwrap();
        assert!(matches!(
            decode_event_frame(&not_array),
            Err(EventCodecError::Malformed(_))
        ));
        // Wrong sender width.
        let bad_sender = Value::Array(vec![
            Value::from(EV_TAG_FRAME),
            Value::from(0u64),
            Value::from(1u64),
            Value::Bytes(vec![0u8; 31]),
            Value::Bytes(Vec::new()),
        ]);
        let bytes = to_canonical_vec(&bad_sender).unwrap();
        assert!(matches!(
            decode_event_frame(&bytes),
            Err(EventCodecError::Malformed(_))
        ));
        // Truncated Quiesce (missing deadline).
        let short = Value::Array(vec![Value::from(EV_TAG_QUIESCE), Value::from(0u64)]);
        let bytes = to_canonical_vec(&short).unwrap();
        assert!(matches!(
            decode_event_frame(&bytes),
            Err(EventCodecError::Malformed(_))
        ));
    }
}
