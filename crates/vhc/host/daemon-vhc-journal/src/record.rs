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
    /// Keep-reserved (always `false`): the retired `tabi@1` bridge flag — the field stays so
    /// the record grammar is unchanged and pre-existing journals stay parseable.
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
    /// Admitted claim bytes — the **legacy** variant's physical figure, as the module declared it.
    ///
    /// Absent on a certification-minor run, where the module declares no physical figure at all and
    /// the members below carry the composed answer instead. The grammar forbids carrying both: a
    /// record holding a declared claim beside a composed one would leave a reader to guess which
    /// figure the run was actually admitted on.
    #[serde(with = "serde_bytes", skip_serializing_if = "Option::is_none", default)]
    pub claim: Option<Vec<u8>>,
    /// Tag-0 certification member: the module's Logical Resource Plan, verbatim.
    ///
    /// The plan is recorded rather than only its digest because a replay has to be able to re-derive
    /// the composition, and a digest names bytes a verifier may not hold.
    #[serde(with = "serde_bytes", skip_serializing_if = "Option::is_none", default)]
    pub resource_plan: Option<Vec<u8>>,
    /// blake3 of [`Self::resource_plan`], so a reader can check the bytes it was given.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub resource_plan_hash: Option<Hash>,
    /// Tag-0 certification member: the composed Physical Estimate for this role instance.
    #[serde(with = "serde_bytes", skip_serializing_if = "Option::is_none", default)]
    pub physical_estimate: Option<Vec<u8>>,
    /// blake3 of [`Self::physical_estimate`].
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub physical_estimate_hash: Option<Hash>,
    /// Tag-0 certification member: the node/device aggregate the instance was admitted within.
    ///
    /// Distinct from the per-instance claim on purpose: a role colocated with another shares device
    /// resources, and the aggregate is what the node actually reserved. A record carrying only the
    /// per-instance figure cannot explain why two colocated roles fit.
    #[serde(with = "serde_bytes", skip_serializing_if = "Option::is_none", default)]
    pub aggregate_estimate: Option<Vec<u8>>,
    /// blake3 of [`Self::aggregate_estimate`].
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub aggregate_estimate_hash: Option<Hash>,
    /// Tag-0 certification member: the Execution Grant this instance ran under.
    #[serde(with = "serde_bytes", skip_serializing_if = "Option::is_none", default)]
    pub execution_grant: Option<Vec<u8>>,
    /// blake3 of [`Self::execution_grant`].
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub execution_grant_hash: Option<Hash>,
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

/// tag 18 — the Execution Grant application result (ABI §8.3 at the certification minor).
///
/// Written exactly once after `da_apply_execution_grant` **returns** — status zero or nonzero — and
/// before the tag-11 init record. On a trap it is absent and exactly one terminal trap carrying the
/// grant-application context occupies that branch instead; the grammar admits one or the other,
/// never both and never neither, and a replay reproduces whichever branch was taken.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionGrantRec {
    /// blake3 of the tag-0 `execution_grant` bytes.
    pub execution_grant_hash: Hash,
    /// The `da_apply_execution_grant` return status (`0` = accepted).
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
    /// tag 0. Boxed: the certification variant carries the plan, the composed estimate, the aggregate
    /// and the grant, which makes this much the largest body — unboxed, every record of every other
    /// tag would be sized for it.
    RunHeader(Box<RunHeader>),
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
    /// tag 18.
    ExecutionGrant(ExecutionGrantRec),
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
            Body::ExecutionGrant(_) => daemon_vhc_abi::JOURNAL_TAG_EXECUTION_GRANT,
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
            Body::ExecutionGrant(x) => Value::serialized(x),
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
            // The grant-application result. Writable but not readable until now, which means a
            // certification run could journal a record no replay could decode — and replay is the whole
            // point of journaling it. Encode and decode are two halves of one contract; a tag added to
            // one of them is not added.
            18 => Body::ExecutionGrant(de(body)?),
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

impl RunHeader {
    /// Whether this header records a certification-minor run.
    ///
    /// Determined from the members present rather than from the ABI number, because the members are
    /// what a reader has: a verifier holding these bytes must be able to tell which variant it is
    /// looking at without first agreeing with the writer about what a minor implies.
    #[must_use]
    pub fn is_certification_variant(&self) -> bool {
        self.resource_plan.is_some() || self.physical_estimate.is_some()
    }

    /// Check the tag-0 variant grammar: exactly one of the two shapes, never both and never neither.
    ///
    /// The exclusivity is the point. A record carrying a module-declared claim beside a host-composed
    /// one leaves a reader to guess which figure the run was admitted on, and the two disagreeing is
    /// precisely the case where guessing matters. A record carrying neither claims an admission with
    /// no resource basis at all.
    ///
    /// # Errors
    /// A description of which rule the header breaks.
    pub fn check_variant(&self) -> Result<(), String> {
        if self.is_certification_variant() {
            if self.claim.is_some() {
                return Err(
                    "a certification-variant run header carries a module-declared claim as well as \
                     a composed one; a reader cannot tell which figure the run was admitted on"
                        .into(),
                );
            }
            for (name, present) in [
                ("resource_plan", self.resource_plan.is_some()),
                ("resource_plan_hash", self.resource_plan_hash.is_some()),
                ("physical_estimate", self.physical_estimate.is_some()),
                (
                    "physical_estimate_hash",
                    self.physical_estimate_hash.is_some(),
                ),
            ] {
                if !present {
                    return Err(format!(
                        "a certification-variant run header is missing `{name}`; a partial \
                         composition record cannot be re-derived by a verifier"
                    ));
                }
            }
            // Each digest must name the bytes beside it, or the record is self-inconsistent and the
            // bytes a verifier re-derives from would not be the ones the writer meant.
            for (name, bytes, digest) in [
                (
                    "resource_plan",
                    self.resource_plan.as_ref(),
                    self.resource_plan_hash,
                ),
                (
                    "physical_estimate",
                    self.physical_estimate.as_ref(),
                    self.physical_estimate_hash,
                ),
                (
                    "aggregate_estimate",
                    self.aggregate_estimate.as_ref(),
                    self.aggregate_estimate_hash,
                ),
                (
                    "execution_grant",
                    self.execution_grant.as_ref(),
                    self.execution_grant_hash,
                ),
            ] {
                if let (Some(bytes), Some(digest)) = (bytes, digest) {
                    if daemon_vhc_proto::hash::blake3_hash(bytes) != digest {
                        return Err(format!(
                            "`{name}_hash` does not name the `{name}` bytes beside it"
                        ));
                    }
                } else if bytes.is_some() != digest.is_some() {
                    return Err(format!(
                        "`{name}` and `{name}_hash` must be present or absent together"
                    ));
                }
            }
            Ok(())
        } else if self.claim.is_some() {
            Ok(())
        } else {
            Err(
                "a run header carries neither a declared claim nor a composed one, so it records \
                 an admission with no resource basis"
                    .into(),
            )
        }
    }
}

#[cfg(test)]
mod run_header_variant_tests {
    use super::*;

    fn legacy() -> RunHeader {
        RunHeader {
            run_id: Hash([1; 32]),
            epoch: 0,
            role: "trainer".into(),
            instance: 0,
            module: Hash([2; 32]),
            abi: u64::from(daemon_vhc_abi::DA_ABI_MAJOR_V2) << 16,
            worlds: std::collections::BTreeMap::new(),
            bridge: false,
            manifest: Vec::new(),
            config: Vec::new(),
            grants: Vec::new(),
            claim: Some(vec![0xA1]),
            channels: Vec::new(),
            device: Vec::new(),
            resource_plan: None,
            resource_plan_hash: None,
            physical_estimate: None,
            physical_estimate_hash: None,
            aggregate_estimate: None,
            aggregate_estimate_hash: None,
            execution_grant: None,
            execution_grant_hash: None,
            format: 1,
        }
    }

    fn certification() -> RunHeader {
        let plan = vec![0xB1, 0xB2];
        let claim = vec![0xC1];
        let mut h = legacy();
        h.claim = None;
        h.resource_plan_hash = Some(daemon_vhc_proto::hash::blake3_hash(&plan));
        h.resource_plan = Some(plan);
        h.physical_estimate_hash = Some(daemon_vhc_proto::hash::blake3_hash(&claim));
        h.physical_estimate = Some(claim);
        h
    }

    /// Both variants are well-formed on their own.
    #[test]
    fn each_variant_is_well_formed_by_itself() {
        legacy().check_variant().expect("the legacy variant");
        certification()
            .check_variant()
            .expect("the certification variant");
        assert!(!legacy().is_certification_variant());
        assert!(certification().is_certification_variant());
    }

    /// A record carrying both a declared claim and a composed one is refused.
    ///
    /// This is the rule worth having: the two figures disagreeing is exactly when it matters which
    /// one the run was admitted on, and a record holding both leaves that to a reader's guess.
    #[test]
    fn a_header_carrying_both_a_declared_and_a_composed_claim_is_refused() {
        let mut both = certification();
        both.claim = Some(vec![0xA1]);
        let err = both
            .check_variant()
            .expect_err("both shapes must be refused");
        assert!(
            err.contains("which figure the run was admitted on"),
            "{err}"
        );
    }

    /// A record carrying neither records an admission with no resource basis.
    #[test]
    fn a_header_carrying_no_claim_at_all_is_refused() {
        let mut neither = legacy();
        neither.claim = None;
        assert!(neither.check_variant().is_err());
    }

    /// A partial composition record cannot be re-derived, so it is refused rather than half-read.
    #[test]
    fn a_partial_certification_record_is_refused() {
        let mut partial = certification();
        partial.physical_estimate = None;
        partial.physical_estimate_hash = None;
        let err = partial.check_variant().expect_err("a partial record");
        assert!(err.contains("physical_estimate"), "{err}");
    }

    /// A digest that does not name the bytes beside it is refused: a verifier re-deriving from those
    /// bytes would not be working from what the writer meant.
    #[test]
    fn a_digest_that_does_not_name_its_bytes_is_refused() {
        let mut wrong = certification();
        wrong.resource_plan_hash = Some(Hash([0xEE; 32]));
        let err = wrong.check_variant().expect_err("a mismatched digest");
        assert!(err.contains("resource_plan_hash"), "{err}");
    }

    /// The legacy variant's encoding is unchanged by the new members: they are skipped when absent,
    /// so journals written before this change still decode and still re-encode identically.
    #[test]
    fn the_legacy_variants_encoding_is_unchanged_by_the_new_members() {
        let encoded = daemon_vhc_proto::canonical::to_canonical_vec(&legacy()).expect("encodes");
        let text = format!("{encoded:?}");
        for absent in [
            "resource_plan",
            "physical_estimate",
            "aggregate_estimate",
            "execution_grant",
        ] {
            assert!(
                !String::from_utf8_lossy(&encoded).contains(absent),
                "`{absent}` must not appear in a legacy record's bytes ({text:.0})"
            );
        }
        let round: RunHeader =
            ciborium::from_reader(encoded.as_slice()).expect("a legacy record round-trips");
        assert_eq!(round, legacy());
    }
}
