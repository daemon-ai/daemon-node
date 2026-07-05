// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `vision_analyze` failure taxonomy: every way resolving, validating, or analyzing an image
//! can fail. Each variant maps onto a distinct, actionable operator/model message (see
//! [`crate::friendly_analysis`]), mirroring hermes' error-hint classification
//! (`tools/vision_tools.py`).

use std::time::Duration;

/// What went wrong resolving or analyzing an image.
#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    /// The URL (or a redirect hop) was rejected by the egress safety policy (SSRF guard).
    #[error("blocked url: {0}")]
    Ssrf(String),
    /// The image could not be downloaded (transport error, non-2xx status, redirect problems).
    #[error("unreachable: {0}")]
    Unreachable(String),
    /// The image exceeds a hard size cap (download bytes or base64 payload).
    #[error("too large: {0}")]
    TooLarge(String),
    /// The bytes are not a supported image format (magic-byte sniff failed).
    #[error("not an image: {0}")]
    NotImage(String),
    /// The `image_url` argument itself is malformed (empty, bad `data:` URL, ...).
    #[error("bad input: {0}")]
    BadInput(String),
    /// A local path could not be read through the execution environment (containment or I/O).
    #[error("workspace: {0}")]
    Workspace(String),
    /// The aux vision provider failed (includes "model lacks vision capability" rejections).
    #[error("provider: {0}")]
    Provider(String),
    /// The turn was cancelled cooperatively while the tool ran.
    #[error("cancelled")]
    Cancelled,
    /// The aux vision call did not answer within the configured deadline.
    #[error("timed out after {0:?}")]
    Timeout(Duration),
}
