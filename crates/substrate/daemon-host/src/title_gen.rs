// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Background session-title generation (hermes `agent/title_generator.py` parity).
//!
//! After a session's first exchange completes, the live event pump fires one best-effort auxiliary
//! model call (`task = "title_generation"`, temperature 0.3, max_tokens 500) over the first user
//! message + first assistant reply, cleans the result (strip quotes / a `title:` prefix, cap at 80
//! chars), and persists it via the session meta — replacing the truncation-seeded roster title.
//! It runs off the turn path (spawned), never blocks a reply, and a failure leaves the seed intact.

use daemon_core::{Provider, Request, RequestMsg, RequestParams};
use std::time::Duration;

/// The generation instruction (hermes `_TITLE_PROMPT`, verbatim).
const TITLE_PROMPT: &str = "Generate a short, descriptive title (3-7 words) for a conversation \
     that starts with the following exchange. The title should capture the main topic or intent. \
     Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes.";

/// How much of each side of the exchange rides in the request (hermes truncates both to 500).
const SNIPPET_CHARS: usize = 500;
/// The overall deadline on the auxiliary call (hermes `timeout = 30.0`).
const TITLE_TIMEOUT: Duration = Duration::from_secs(30);
/// The persisted title length cap (hermes: `title[:77] + "..."` past 80).
const TITLE_MAX_CHARS: usize = 80;

/// Generate a session title from the first exchange via the auxiliary provider. Returns `None` on
/// any failure/timeout/empty result (the caller keeps the seeded title). Never panics.
pub(crate) async fn generate_title(
    aux: &dyn Provider,
    user_message: &str,
    assistant_response: &str,
) -> Option<String> {
    let user_snippet: String = user_message.chars().take(SNIPPET_CHARS).collect();
    let assistant_snippet: String = assistant_response.chars().take(SNIPPET_CHARS).collect();
    let req = Request {
        system: TITLE_PROMPT.to_string(),
        messages: vec![RequestMsg {
            role: "user".into(),
            content: format!("User: {user_snippet}\n\nAssistant: {assistant_snippet}"),
            ..RequestMsg::default()
        }],
        ..Request::default()
    }
    .with_params(RequestParams {
        temperature: Some(0.3),
        max_tokens: Some(500),
        ..RequestParams::default()
    })
    .with_task("title_generation");

    let output = tokio::time::timeout(TITLE_TIMEOUT, aux.chat(req))
        .await
        .ok()?
        .ok()?;
    clean_title(&output.text)
}

/// Clean a raw model title: trim, strip wrapping quotes, drop a leading `title:` prefix, and cap
/// the length (hermes cleanup rules, verbatim). `None` when nothing usable remains.
pub(crate) fn clean_title(raw: &str) -> Option<String> {
    let mut title = raw.trim().trim_matches(['"', '\'']).trim().to_string();
    if title.to_lowercase().starts_with("title:") {
        title = title[6..].trim().to_string();
    }
    if title.chars().count() > TITLE_MAX_CHARS {
        let head: String = title.chars().take(TITLE_MAX_CHARS - 3).collect();
        title = format!("{head}...");
    }
    (!title.is_empty()).then_some(title)
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_core::MockProvider;

    #[test]
    fn clean_strips_quotes_prefix_and_caps_length() {
        assert_eq!(clean_title("\"Build Fixes\""), Some("Build Fixes".into()));
        assert_eq!(clean_title("'Build Fixes'"), Some("Build Fixes".into()));
        assert_eq!(
            clean_title("Title: Debugging the parser"),
            Some("Debugging the parser".into())
        );
        assert_eq!(clean_title("   \n"), None);
        assert_eq!(clean_title("\"\""), None);
        let long = "x".repeat(200);
        let cleaned = clean_title(&long).unwrap();
        assert_eq!(cleaned.chars().count(), TITLE_MAX_CHARS);
        assert!(cleaned.ends_with("..."));
    }

    #[tokio::test]
    async fn generate_uses_aux_reply_and_cleans_it() {
        let aux = MockProvider::completing("\"Parser Pipeline Refactor\"");
        let title = generate_title(&aux, "please refactor the parser", "done, refactored").await;
        assert_eq!(title.as_deref(), Some("Parser Pipeline Refactor"));
    }

    #[tokio::test]
    async fn generate_swallows_provider_failure() {
        let aux = daemon_core::UnconfiguredProvider::new();
        assert!(generate_title(&aux, "hi", "hello").await.is_none());
    }
}
