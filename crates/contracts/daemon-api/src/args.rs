// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Parameter structs for the multi-argument interface methods.
//!
//! Each struct bundles the arguments of one trait method that would otherwise take 4+ scalars
//! (the "Excess Arguments" smell). They are deliberately shaped to mirror the corresponding
//! [`ApiRequest`] variant field-for-field (names, types, and serde attributes) so the C2 step -
//! lifting a struct onto the wire as `ApiRequest::Variant(Args)` - is a near-mechanical swap; a
//! serde newtype-variant wrapping such a struct encodes identically to today's struct-variant.
//!
//! These are NOT yet on the wire: `ApiRequest`/`ApiResponse` and the CDDL are untouched. The
//! structs derive exactly what `ApiRequest` derives so they are ready to become variant payloads.

use crate::*;
use serde::{Deserialize, Serialize};

/// Arguments for [`SessionApi::submit_as`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmitAsArgs {
    /// Target session.
    pub session: SessionId,
    /// Optional per-event attribution (`None` = host-local default).
    #[serde(default)]
    pub origin: Option<Origin>,
    /// The §17 command.
    pub command: AgentCommand,
    /// Optional explicit profile to bind on open (`None` = routing-config / default binding).
    #[serde(default)]
    pub profile: Option<ProfileRef>,
}

/// Arguments for [`SessionApi::record_meta`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordMetaArgs {
    /// The session whose merged log to append to.
    pub session: SessionId,
    /// The attribution for the meta event.
    pub origin: Origin,
    /// The renderer/router discriminator (e.g. `"presence"` / `"attach"`).
    pub kind: String,
    /// The opaque encoded body, decoded by the consumer per `kind`.
    #[serde(with = "serde_bytes")]
    pub body: Vec<u8>,
}

/// Arguments for [`ControlApi::conv_send`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvSendArgs {
    /// The owning transport.
    pub transport: TransportId,
    /// The conversation id.
    pub conv: String,
    /// The author (`None` = the account/operator).
    #[serde(default)]
    pub from: Option<Participant>,
    /// The message.
    pub message: UserMsg,
}

/// Arguments for [`ControlApi::conv_history`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvHistoryArgs {
    /// The owning transport.
    pub transport: TransportId,
    /// The conversation id.
    pub conv: String,
    /// Return entries with cursor strictly greater than this (`0` from the start). Ignored when
    /// `before_cursor` is present.
    #[serde(default)]
    pub after_cursor: u64,
    /// Backward window (rung 2, api vNEXT): when `Some(B)`, return the `max` newest entries with
    /// `cursor < B` (newest-anchored; pass a value past head — e.g. `u64::MAX` — for the latest
    /// window in one round-trip). Wins over `after_cursor` when present. Absent on the wire when
    /// `None` (never null).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_cursor: Option<u64>,
    /// Max entries (`0` = all).
    #[serde(default)]
    pub max: u32,
}

/// Arguments for [`ControlApi::member_invite`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemberInviteArgs {
    /// The owning transport.
    pub transport: TransportId,
    /// The conversation id.
    pub conv: String,
    /// Who to invite.
    pub who: Participant,
    /// An optional invite message.
    #[serde(default)]
    pub message: Option<String>,
}

/// Arguments for [`ControlApi::member_remove`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemberRemoveArgs {
    /// The owning transport.
    pub transport: TransportId,
    /// The conversation id.
    pub conv: String,
    /// Who to remove.
    pub who: Participant,
    /// An optional reason.
    #[serde(default)]
    pub reason: Option<String>,
}

/// Arguments for [`ControlApi::member_ban`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemberBanArgs {
    /// The owning transport.
    pub transport: TransportId,
    /// The conversation id.
    pub conv: String,
    /// Who to ban.
    pub who: Participant,
    /// An optional reason.
    #[serde(default)]
    pub reason: Option<String>,
}

/// Arguments for [`ControlApi::member_set_role`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemberSetRoleArgs {
    /// The owning transport.
    pub transport: TransportId,
    /// The conversation id.
    pub conv: String,
    /// Whose role to set.
    pub who: Participant,
    /// The new role.
    pub role: MemberRole,
}

/// Arguments for [`ControlApi::fs_write`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FsWriteArgs {
    /// The root (Workspace/Session only).
    pub root: FsRootId,
    /// Root-relative path.
    pub path: String,
    /// The bytes to write.
    #[serde(with = "serde_bytes")]
    pub bytes: Vec<u8>,
    /// The base etag for optimistic concurrency (`None` = create-or-overwrite).
    #[serde(default)]
    pub base_revision: Option<FsRevision>,
    /// Override the sensitive-path / `Deny` gate.
    #[serde(default)]
    pub force: bool,
}

/// Arguments for [`ControlApi::fs_watch_after`] (the `FsWatchPoll` wire variant).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FsWatchAfterArgs {
    /// The root.
    pub root: FsRootId,
    /// Root-relative directory being watched.
    pub dir: String,
    /// Drain changes after this cursor.
    pub after_seq: u64,
    /// Max events to drain.
    pub max: u32,
}

/// Arguments for [`ControlApi::fs_write_from_blob`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FsWriteFromBlobArgs {
    /// The target root (Workspace/Session only).
    pub root: FsRootId,
    /// Root-relative destination path.
    pub path: String,
    /// The blob to materialize.
    pub hash: ContentHash,
    /// The base etag for optimistic concurrency (`None` = create-or-overwrite).
    #[serde(default)]
    pub base_revision: Option<FsRevision>,
    /// Override the sensitive-path / `Deny` gate.
    #[serde(default)]
    pub force: bool,
}

/// Arguments for [`ModelApi::model_recommend`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRecommendArgs {
    /// The `org/name` repo id.
    pub repo: String,
    /// The git revision (`None` = `main`).
    pub revision: Option<String>,
    /// The engine the recommendation targets.
    pub engine: ModelEngine,
    /// An explicit memory budget in bytes (`None` = auto-detect VRAM/RAM).
    pub budget_bytes: Option<u64>,
}

/// Arguments for [`ModelApi::model_quantize`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelQuantizeArgs {
    /// The `org/name` repo id whose GGUF is quantized.
    pub repo: String,
    /// The git revision (`None` = `main`).
    pub revision: Option<String>,
    /// The target quant label (e.g. `Q4_K_M`).
    pub target_quant: String,
    /// The source GGUF file (`None` = the highest-precision one in the repo).
    pub source_file: Option<String>,
}
