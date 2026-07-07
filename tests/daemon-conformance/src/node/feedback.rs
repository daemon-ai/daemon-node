// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! N1 (user feedback over OpenTelemetry): the host `feedback_submit` handler + the node-owned
//! telemetry consent surface. Covers server-side validation, consent provenance recorded on the
//! persisted record, and the accepted+queued ack.

use super::harness::*;
use daemon_api::{
    ApiError, FeedbackDiagnostics, FeedbackKind, FeedbackRating, FeedbackSubmitArgs, FeedbackTarget,
};

/// Assemble a node retaining its shared durable store, so a test can inspect the persisted
/// feedback record directly (consent provenance, mapped fields).
fn assemble_with_store() -> (
    Arc<NodeApiImpl>,
    daemon_host::SupervisorHandle,
    Arc<dyn SessionStore>,
) {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 0, [0x6f; 32], fast_host_config());
    (node, handle, store)
}

fn app_args() -> FeedbackSubmitArgs {
    FeedbackSubmitArgs {
        kind: FeedbackKind::App,
        target: None,
        rating: Some(FeedbackRating::Up),
        comment: Some("love it".into()),
        include_content: false,
        diagnostics: Some(FeedbackDiagnostics {
            app_version: Some("1.2.3".into()),
            os: Some("linux".into()),
        }),
        surface: "settings".into(),
    }
}

/// App feedback is accepted+queued, and — with the global telemetry toggle OFF — the persisted
/// record records `explicit-one-shot` consent provenance (explicit feedback is per-event consent).
#[tokio::test]
async fn app_feedback_queued_with_one_shot_consent_when_toggle_off() {
    let (node, _handle, store) = assemble_with_store();

    // Consent defaults OFF (opt-in).
    assert!(!node.telemetry_consent_get().await.unwrap());

    let ack = node.feedback_submit(app_args()).await.expect("accepted");
    assert!(ack.accepted && ack.queued, "explicit feedback is queued");

    let pending = store.feedback_pending(0).await;
    assert_eq!(pending.len(), 1);
    let rec = &pending[0];
    assert_eq!(rec.kind, "app");
    assert_eq!(rec.rating.as_deref(), Some("up"));
    assert_eq!(rec.comment.as_deref(), Some("love it"));
    assert_eq!(rec.surface, "settings");
    assert_eq!(rec.app_version.as_deref(), Some("1.2.3"));
    assert_eq!(rec.os.as_deref(), Some("linux"));
    assert_eq!(
        rec.consent, "explicit-one-shot",
        "queued even though the global toggle is off"
    );
    assert!(!rec.node_version.is_empty(), "node version is stamped");
    assert!(rec.id.starts_with("fb-"), "host-minted feedback id");
}

/// With the toggle ON, the persisted record records `opted-in` consent provenance. The set reply
/// echoes the new state.
#[tokio::test]
async fn consent_toggle_round_trips_and_marks_opted_in() {
    let (node, _handle, store) = assemble_with_store();

    assert!(
        node.telemetry_consent_set(true).await.unwrap(),
        "echoes new state"
    );
    assert!(node.telemetry_consent_get().await.unwrap(), "persisted");

    node.feedback_submit(app_args()).await.expect("accepted");
    let pending = store.feedback_pending(0).await;
    assert_eq!(pending[0].consent, "opted-in");

    // Flipping it back off round-trips.
    assert!(!node.telemetry_consent_set(false).await.unwrap());
    assert!(!node.telemetry_consent_get().await.unwrap());
}

/// Server-side validation is the enforcement point: response feedback needs a target + rating; app
/// feedback needs a comment or a rating; a comment over the cap is rejected; and a response target
/// must point at an existing session.
#[tokio::test]
async fn validation_rejects_malformed_submissions() {
    let (node, _handle, _store) = assemble_with_store();

    // Response kind without a target.
    let err = node
        .feedback_submit(FeedbackSubmitArgs {
            kind: FeedbackKind::Response,
            target: None,
            rating: Some(FeedbackRating::Down),
            comment: None,
            include_content: false,
            diagnostics: None,
            surface: "transcript".into(),
        })
        .await;
    assert!(
        matches!(err, Err(ApiError::Other(_))),
        "needs a target: {err:?}"
    );

    // Response kind with a target but no rating (target points at a nonexistent session, but the
    // rating check fires first here — either way it must be rejected).
    let err = node
        .feedback_submit(FeedbackSubmitArgs {
            kind: FeedbackKind::Response,
            target: Some(FeedbackTarget {
                session: "s-missing".into(),
                cursor: 1,
                trace: None,
            }),
            rating: None,
            comment: Some("no rating".into()),
            include_content: false,
            diagnostics: None,
            surface: "transcript".into(),
        })
        .await;
    assert!(err.is_err(), "response feedback needs a rating: {err:?}");

    // Response kind targeting a session that does not exist.
    let err = node
        .feedback_submit(FeedbackSubmitArgs {
            kind: FeedbackKind::Response,
            target: Some(FeedbackTarget {
                session: "s-missing".into(),
                cursor: 1,
                trace: None,
            }),
            rating: Some(FeedbackRating::Up),
            comment: None,
            include_content: false,
            diagnostics: None,
            surface: "transcript".into(),
        })
        .await;
    assert!(
        matches!(err, Err(ApiError::UnknownSession(_))),
        "unknown session: {err:?}"
    );

    // App kind with neither a comment nor a rating.
    let err = node
        .feedback_submit(FeedbackSubmitArgs {
            kind: FeedbackKind::App,
            target: None,
            rating: None,
            comment: None,
            include_content: false,
            diagnostics: None,
            surface: "settings".into(),
        })
        .await;
    assert!(
        matches!(err, Err(ApiError::Other(_))),
        "needs comment or rating: {err:?}"
    );

    // A comment over the byte cap is rejected.
    let err = node
        .feedback_submit(FeedbackSubmitArgs {
            kind: FeedbackKind::App,
            target: None,
            rating: Some(FeedbackRating::Up),
            comment: Some("x".repeat(daemon_api::FEEDBACK_COMMENT_MAX + 1)),
            include_content: false,
            diagnostics: None,
            surface: "settings".into(),
        })
        .await;
    assert!(
        matches!(err, Err(ApiError::Other(_))),
        "comment too long: {err:?}"
    );
}
