// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Approval detail (wire v30, item 7): the optional `ApprovalInfo.detail` (a node-computed
//! `ToolDetail` with kind "fs.diff"), with pre-v30 back-compat (`detail` is serde-default).

use daemon_api::{from_cbor, to_cbor, ApprovalInfo};
use daemon_common::SessionId;
use daemon_protocol::ToolDetail;
use serde::Serialize;

#[test]
fn approval_info_with_detail_round_trips() {
    let info = ApprovalInfo {
        session: SessionId::new("s1"),
        request_id: "req-1".into(),
        prompt: "Apply edit to src/lib.rs".into(),
        path: Some("src/lib.rs".into()),
        fingerprint: None,
        detail: Some(ToolDetail::new(
            "fs.diff",
            br#"{"path":"src/lib.rs","diff":"@@ -1 +1 @@\n-old\n+new\n"}"#.to_vec(),
        )),
    };
    assert_eq!(info, from_cbor::<ApprovalInfo>(&to_cbor(&info)).unwrap());
    assert_eq!(info.detail.as_ref().unwrap().kind, "fs.diff");
}

#[test]
fn pre_v30_approval_info_decodes_with_none_detail() {
    #[derive(Serialize)]
    struct OldApprovalInfo {
        session: SessionId,
        request_id: String,
        prompt: String,
        path: Option<String>,
        fingerprint: Option<String>,
    }
    let old = OldApprovalInfo {
        session: SessionId::new("s1"),
        request_id: "req-1".into(),
        prompt: "run: rm -rf /tmp/x".into(),
        path: None,
        fingerprint: Some("ab12".into()),
    };
    let decoded = from_cbor::<ApprovalInfo>(&to_cbor(&old)).unwrap();
    assert!(decoded.detail.is_none());
}
