// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The journal record types + canonical CBOR codec (ABI companion §8.3).
//!
//! The normative record grammar is the one tagged-union CDDL artifact that lives in
//! [`daemon_vhc_abi::JOURNAL_CDDL`]; these are the host-side Rust types that encode/decode to it. A
//! record on disk is the CBOR array `[tag, ord, body]` (ABI §8.2 `record-CBOR`): `tag` is the
//! permanent numeric variant tag (§8.3), `ord` the per-journal monotone record ordinal, and `body` a
//! canonical-CBOR map with explicit string keys. Encoding runs the value tree through the same
//! RFC 8949 §4.2 deterministic writer the wire contract uses ([`daemon_vhc_proto::to_canonical_vec`]),
//! so two encodings of one record are byte-identical.
//!
//! The Rust types live here rather than in `daemon-vhc-abi` because the abi crate is dependency-free
//! and dual-compiled for `wasm32` (it holds the grammar, not a serde encoder); journal encoding is a
//! host-side concern (guests never encode journal records). The `journal_grammar` test validates a
//! sample of every tag against `JOURNAL_CDDL`, closing the serde↔grammar loop.

use ciborium::value::Value;
use serde::{Deserialize, Serialize};

use daemon_vhc_proto::{from_canonical_slice, to_canonical_vec, Hash};

use super::JournalError;

/// A per-journal monotone record ordinal (ABI §8.3 `ord`).
pub type Ord = u64;

/// The frozen execution-identity five-tuple (ABI §8.1): `(run_id, epoch, role, instance,
/// module_hash)`. `instance` is the **never-reused, node-durable, monotonic** role-instance
/// incarnation id (ABI §8.1 erratum / decisions D1) — not a reusable supervision slot. Carried in
/// every segment header (§8.2), the run-header record (§8.3 tag 0), and every sidecar header (§8.5).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecIdentity {
    /// The 32-byte genesis/frozen-envelope hash (the cryptographic `RunId`).
    pub run_id: Hash,
    /// The transition-chain position.
    pub epoch: u64,
    /// The envelope-level role label (opaque to the host beyond lane selection).
    pub role: String,
    /// The never-reused monotonic role-instance incarnation id.
    pub instance: u64,
    /// The pinned module blob hash.
    pub module: Hash,
}

/// A content-addressed sidecar reference (ABI §8.3 `sidecar-ref`, §8.5).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SidecarRef {
    /// blake3 of the plaintext (the content address).
    pub hash: Hash,
    /// Plaintext size in bytes.
    pub size: u64,
    /// The segment ordinal the referencing record lives in.
    pub seg: u64,
}

/// An archive reference for a durably-archived signed frame (ABI §8.3 `evidence-ref`, §8.6).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    /// blake3 of the archived frame.
    pub hash: Hash,
    /// The archive locator.
    pub locator: String,
}

/// Trap detail for a terminal fault (ABI §8.3 `trap-info`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrapInfo {
    /// The trap code (§7.6 taxonomy).
    pub code: String,
    /// The import that trapped.
    pub import: String,
    /// The call context.
    pub context: String,
    /// A human-readable detail.
    pub detail: String,
}

/// The dropped-item identity of a drop record (ABI §8.3 `drop-id`). Every field is optional; which
/// are present depends on the drop `class`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DropId {
    /// A payload hash (payload-ready drops).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub hash: Option<Hash>,
    /// A timer id (timer drops).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub timer_id: Option<u64>,
    /// A channel (gossip / payload drops).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub channel: Option<u64>,
    /// A sender identity.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sender: Option<Hash>,
    /// A sequence number.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub seq: Option<u64>,
}

// ---- per-tag body structs (the §8.3 map bodies; explicit string keys, numeric enums) ------------

/// tag 0 — run-header (ABI §8.3). The admitted manifest/config/grants/claim/channels/device are the
/// verbatim canonical-CBOR bytes of the admitted value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunHeader {
    /// Execution identity's run id.
    pub run_id: Hash,
    /// Execution identity's epoch.
    pub epoch: u64,
    /// Execution identity's role.
    pub role: String,
    /// Execution identity's instance incarnation.
    pub instance: u64,
    /// Execution identity's module hash.
    pub module: Hash,
    /// Negotiated `(major << 16) | minor` ABI version.
    pub abi: u64,
    /// Negotiated per-world minors.
    pub worlds: std::collections::BTreeMap<String, u64>,
    /// Whether the `tabi@1` bridge is linked.
    pub bridge: bool,
    /// Admitted manifest bytes.
    #[serde(with = "serde_bytes")]
    pub manifest: Vec<u8>,
    /// Admitted config bytes.
    #[serde(with = "serde_bytes")]
    pub config: Vec<u8>,
    /// Admitted grants bytes.
    #[serde(with = "serde_bytes")]
    pub grants: Vec<u8>,
    /// Admitted claim bytes.
    #[serde(with = "serde_bytes")]
    pub claim: Vec<u8>,
    /// Admitted channels bytes.
    #[serde(with = "serde_bytes")]
    pub channels: Vec<u8>,
    /// Admitted device profile bytes.
    #[serde(with = "serde_bytes")]
    pub device: Vec<u8>,
    /// Journal format version (1).
    pub format: u64,
}

/// tag 1 — event (ABI §8.3): a delivered event frame verbatim with its logical delivery time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventRec {
    /// Logical delivery time (§6.5), sampled before delivery.
    pub at: u64,
    /// The exact frame bytes returned by `next_event`.
    #[serde(with = "serde_bytes")]
    pub frame: Vec<u8>,
}

/// tag 2 — read-back (ABI §8.3): a nondeterministic import result, inline or via an encrypted sidecar
/// (the group choice `("value" // "sidecar")`, resolved by exactly one of the two fields present).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadBackRec {
    /// The read-back source.
    pub src: u64,
    /// The read-back kind.
    pub kind: u64,
    /// The read-back status.
    pub status: u64,
    /// Inline plaintext value (iff `<= READBACK_INLINE_MAX`).
    #[serde(with = "serde_bytes", skip_serializing_if = "Option::is_none", default)]
    pub value: Option<Vec<u8>>,
    /// An encrypted content-addressed sidecar (iff oversize).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sidecar: Option<SidecarRef>,
}

/// tag 3 — clock (ABI §8.3): a `now()` reading (clocks are not messages, but must be captured).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockRec {
    /// The logical `now` value read.
    pub now: u64,
}

/// tag 4 — publish (ABI §8.3): the durable seq advance + the complete signed outbound wire frame.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishRec {
    /// The channel published on.
    pub channel: u64,
    /// The durable, channel-scoped sequence number (§12.2).
    pub seq: u64,
    /// blake3 of the guest payload.
    pub hash: Hash,
    /// The complete signed wire frame (§8.6, §12).
    #[serde(with = "serde_bytes")]
    pub frame: Vec<u8>,
}

/// tag 5 — timer-arm (ABI §8.3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimerArmRec {
    /// Timer id.
    pub id: u64,
    /// Requested delay.
    pub delay: u64,
    /// Logical time the timer was armed.
    pub armed_at: u64,
}

/// tag 6 — timer-cancel (ABI §8.3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimerCancelRec {
    /// Timer id.
    pub id: u64,
    /// Cancel status (0 = suppressed-before-fire, 1 = already-fired).
    pub status: u64,
}

/// tag 7 — drop (ABI §8.3): an advisory drop/coalesce.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DropRec {
    /// Drop class (0 payload-ready, 1 timer, 2 gossip, 3 budget).
    pub class: u64,
    /// Coalesce rule (0 dedup-hash, 1 latest-wins, 2 drop-oldest).
    pub rule: u64,
    /// The dropped item's identity.
    pub dropped: DropId,
}

/// tag 8 — throttle (ABI §8.3): a throttle change.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThrottleRec {
    /// Whether the run is paused.
    pub paused: bool,
    /// Duty-cycle percentage.
    pub duty_pct: u64,
    /// VRAM cap in bytes.
    pub vram_cap_bytes: u64,
}

/// tag 9 — terminal (ABI §8.3): the terminal fact (outcome / trap / forced interruption).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalRec {
    /// Terminal kind (0 outcome, 1 trap, 2 forced interruption).
    pub kind: u64,
    /// Present iff `kind = 0`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub outcome: Option<u64>,
    /// Present iff `kind = 1` or `2`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub trap: Option<TrapInfo>,
}

/// tag 10 — snapshot (ABI §8.3): verbatim accepted state-manifest bytes (§10.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotRec {
    /// The accepted state-manifest bytes.
    #[serde(with = "serde_bytes")]
    pub manifest: Vec<u8>,
}

/// tag 11 — init (ABI §8.3): the `da_init` call + its config/grants hash-pin.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitRec {
    /// blake3 of the tag-0 config bytes.
    pub config_hash: Hash,
    /// blake3 of the tag-0 grants bytes.
    pub grants_hash: Hash,
    /// The `da_init` return status (§9.4 step 11).
    pub status: u64,
}

/// tag 12 — signed-frame (ABI §8.3, §8.6): the original signed wire frame behind an authoritative
/// event, inline (Phase A) or as an archive evidence reference (Phase D). Exactly one present.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedFrameRec {
    /// The channel.
    pub channel: u64,
    /// The sequence number.
    pub seq: u64,
    /// The frame sender identity.
    pub sender: Hash,
    /// The inline original signed wire frame (Phase A — no archive exists).
    #[serde(with = "serde_bytes", skip_serializing_if = "Option::is_none", default)]
    pub frame: Option<Vec<u8>>,
    /// An archive reference once durably archived (Phase D).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub evidence: Option<EvidenceRef>,
}

/// tag 13 — instantiation (ABI §8.3, §7.1): every (re-)instantiation, journaled before any guest code.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstantiationRec {
    /// The generation seed of §7.1 (the per-run instantiation counter).
    pub counter: u64,
    /// Reason (0 initial, 1 trap-restart, 2 upgrade-activation).
    pub reason: u64,
    /// Logical time of instantiation (the `da_init` `now()` value, §6.5).
    pub at: u64,
}

/// tag 14 — completion (ABI §8.3): reserved (Phase B).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionRec {
    /// The op id.
    pub op: u64,
    /// The completion result bytes.
    #[serde(with = "serde_bytes")]
    pub result: Vec<u8>,
}

/// tag 15 — device-profile (ABI §8.3): reserved (Phase B).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceProfileRec {
    /// The device profile bytes.
    #[serde(with = "serde_bytes")]
    pub profile: Vec<u8>,
}

/// tag 16 — condition (ABI §8.3, §6.7): a run condition (`SpoolExhausted`, etc).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionRec {
    /// The condition code.
    pub code: String,
    /// A human-readable detail.
    pub detail: String,
}

/// tag 17 — seal (ABI §8.3, §8.2): a cleanly-rolled segment's self-excluding hash + record count.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealRec {
    /// blake3 of this segment EXCLUDING this seal record's own framing (§8.2).
    pub segment_blake3: Hash,
    /// The number of records in the segment (excluding the seal).
    pub records: u64,
}

/// A journal record body — one variant per §8.3 tag. The numeric tag is [`Body::tag`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Body {
    /// tag 0.
    RunHeader(RunHeader),
    /// tag 1.
    Event(EventRec),
    /// tag 2.
    ReadBack(ReadBackRec),
    /// tag 3.
    Clock(ClockRec),
    /// tag 4.
    Publish(PublishRec),
    /// tag 5.
    TimerArm(TimerArmRec),
    /// tag 6.
    TimerCancel(TimerCancelRec),
    /// tag 7.
    Drop(DropRec),
    /// tag 8.
    Throttle(ThrottleRec),
    /// tag 9.
    Terminal(TerminalRec),
    /// tag 10.
    Snapshot(SnapshotRec),
    /// tag 11.
    Init(InitRec),
    /// tag 12.
    SignedFrame(SignedFrameRec),
    /// tag 13.
    Instantiation(InstantiationRec),
    /// tag 14.
    Completion(CompletionRec),
    /// tag 15.
    DeviceProfile(DeviceProfileRec),
    /// tag 16.
    Condition(ConditionRec),
    /// tag 17.
    Seal(SealRec),
}

impl Body {
    /// The permanent numeric §8.3 tag of this record.
    #[must_use]
    pub fn tag(&self) -> u8 {
        match self {
            Body::RunHeader(_) => 0,
            Body::Event(_) => 1,
            Body::ReadBack(_) => 2,
            Body::Clock(_) => 3,
            Body::Publish(_) => 4,
            Body::TimerArm(_) => 5,
            Body::TimerCancel(_) => 6,
            Body::Drop(_) => 7,
            Body::Throttle(_) => 8,
            Body::Terminal(_) => 9,
            Body::Snapshot(_) => 10,
            Body::Init(_) => 11,
            Body::SignedFrame(_) => 12,
            Body::Instantiation(_) => 13,
            Body::Completion(_) => 14,
            Body::DeviceProfile(_) => 15,
            Body::Condition(_) => 16,
            Body::Seal(_) => 17,
        }
    }

    fn body_value(&self) -> Result<Value, JournalError> {
        let v = match self {
            Body::RunHeader(x) => Value::serialized(x),
            Body::Event(x) => Value::serialized(x),
            Body::ReadBack(x) => Value::serialized(x),
            Body::Clock(x) => Value::serialized(x),
            Body::Publish(x) => Value::serialized(x),
            Body::TimerArm(x) => Value::serialized(x),
            Body::TimerCancel(x) => Value::serialized(x),
            Body::Drop(x) => Value::serialized(x),
            Body::Throttle(x) => Value::serialized(x),
            Body::Terminal(x) => Value::serialized(x),
            Body::Snapshot(x) => Value::serialized(x),
            Body::Init(x) => Value::serialized(x),
            Body::SignedFrame(x) => Value::serialized(x),
            Body::Instantiation(x) => Value::serialized(x),
            Body::Completion(x) => Value::serialized(x),
            Body::DeviceProfile(x) => Value::serialized(x),
            Body::Condition(x) => Value::serialized(x),
            Body::Seal(x) => Value::serialized(x),
        };
        v.map_err(|e| JournalError::Codec(format!("serialize body: {e}")))
    }

    fn from_tag_value(tag: u8, body: &Value) -> Result<Self, JournalError> {
        fn de<T: serde::de::DeserializeOwned>(v: &Value) -> Result<T, JournalError> {
            v.deserialized()
                .map_err(|e| JournalError::Codec(format!("decode body: {e}")))
        }
        Ok(match tag {
            0 => Body::RunHeader(de(body)?),
            1 => Body::Event(de(body)?),
            2 => Body::ReadBack(de(body)?),
            3 => Body::Clock(de(body)?),
            4 => Body::Publish(de(body)?),
            5 => Body::TimerArm(de(body)?),
            6 => Body::TimerCancel(de(body)?),
            7 => Body::Drop(de(body)?),
            8 => Body::Throttle(de(body)?),
            9 => Body::Terminal(de(body)?),
            10 => Body::Snapshot(de(body)?),
            11 => Body::Init(de(body)?),
            12 => Body::SignedFrame(de(body)?),
            13 => Body::Instantiation(de(body)?),
            14 => Body::Completion(de(body)?),
            15 => Body::DeviceProfile(de(body)?),
            16 => Body::Condition(de(body)?),
            17 => Body::Seal(de(body)?),
            other => {
                return Err(JournalError::UnknownTag(other));
            }
        })
    }
}

/// A journal record: the CBOR array `[tag, ord, body]` (ABI §8.2 `record-CBOR`, §8.3 grammar).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    /// The per-journal monotone record ordinal.
    pub ord: Ord,
    /// The record body (the tag is `body.tag()`).
    pub body: Body,
}

impl Record {
    /// Construct a record from an ordinal + body.
    #[must_use]
    pub fn new(ord: Ord, body: Body) -> Self {
        Self { ord, body }
    }

    /// The record's numeric §8.3 tag.
    #[must_use]
    pub fn tag(&self) -> u8 {
        self.body.tag()
    }

    /// Encode to canonical CBOR (`[tag, ord, body]`, RFC 8949 §4.2 deterministic).
    ///
    /// # Errors
    /// [`JournalError::Codec`] if the body cannot be serialized.
    pub fn to_canonical(&self) -> Result<Vec<u8>, JournalError> {
        let array = Value::Array(vec![
            Value::Integer(u64::from(self.tag()).into()),
            Value::Integer(self.ord.into()),
            self.body.body_value()?,
        ]);
        to_canonical_vec(&array).map_err(|e| JournalError::Codec(format!("encode record: {e}")))
    }

    /// Decode a record from canonical CBOR produced by [`Record::to_canonical`].
    ///
    /// # Errors
    /// [`JournalError::Codec`] on malformed CBOR; [`JournalError::UnknownTag`] on an unassigned tag.
    pub fn from_canonical(bytes: &[u8]) -> Result<Self, JournalError> {
        let value: Value = from_canonical_slice(bytes)
            .map_err(|e| JournalError::Codec(format!("decode record: {e}")))?;
        let items = match &value {
            Value::Array(items) if items.len() == 3 => items,
            _ => {
                return Err(JournalError::Codec(
                    "record is not a 3-element array [tag, ord, body]".into(),
                ))
            }
        };
        let tag = int_u64(&items[0])?;
        let ord = int_u64(&items[1])?;
        let tag = u8::try_from(tag).map_err(|_| JournalError::UnknownTag(u8::MAX))?;
        let body = Body::from_tag_value(tag, &items[2])?;
        Ok(Record { ord, body })
    }
}

fn int_u64(v: &Value) -> Result<u64, JournalError> {
    match v {
        Value::Integer(i) => {
            let n: i128 = (*i).into();
            u64::try_from(n).map_err(|_| JournalError::Codec("integer out of u64 range".into()))
        }
        _ => Err(JournalError::Codec("expected an integer".into())),
    }
}
