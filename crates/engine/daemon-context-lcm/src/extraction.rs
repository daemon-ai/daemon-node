// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Pre-compaction extraction (`daemon-context-lcm-port-spec.md` §9.2).
//!
//! A best-effort side channel invoked in the compaction loop *before* summarization: it asks the aux
//! provider to distill durable decisions/commitments/outcomes/rules from the region about to be
//! compacted and appends the bullets to a daily markdown file (`<output>/YYYY-MM-DD.md`), so those
//! survive even if the DAG summary is later lost. It **never blocks compaction** — any error, a
//! `NOTHING_TO_EXTRACT` reply, or an empty reply is a silent no-op. Gated by `extraction_enabled`
//! (default off) and an available extraction directory.
//!
//! The call carries the hermes extraction params (`_call_extraction_llm`,
//! `LCM:extraction.py:50-55`): `temperature 0.2`, `max_tokens 2000`, task `"extraction"`, via the
//! per-call [`Request::params`]/[`Request::task`] surface.

use crate::escalation::strip_reasoning_blocks;
use daemon_core::{Provider, Request, RequestMsg, RequestParams};
use regex::Regex;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

/// The verbatim extraction prompt (`LCM:extraction.py:29-42`).
const EXTRACTION_PROMPT_HEAD: &str =
    "Extract decisions, commitments, outcomes, and rules from this conversation segment.\n\n\
Format as a flat list of bullet points. Each bullet should be self-contained and understandable\n\
without the surrounding conversation. Include:\n\
- Decisions made (what was chosen, and why if stated)\n\
- Commitments (who will do what)\n\
- Outcomes (what happened as a result of an action)\n\
- Rules or constraints discovered\n\n\
Skip: greetings, meta-discussion, reasoning that led nowhere, repeated information.\n\
If there is nothing worth extracting, respond with exactly: NOTHING_TO_EXTRACT\n\n\
CONTENT:\n";

/// The sentinel reply meaning "no durable content" (`LCM:extraction.py`).
const NOTHING_TO_EXTRACT: &str = "NOTHING_TO_EXTRACT";

/// What an extraction attempt did (for diagnostics/tests).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExtractionOutcome {
    /// Bullets were appended to the daily markdown file.
    Wrote,
    /// The aux provider returned `NOTHING_TO_EXTRACT` or an empty reply.
    NothingToExtract,
    /// Extraction was skipped (disabled, no directory, aux failure/timeout, or write error).
    Skipped,
}

/// Strip base64 media data-URIs to a `[Media attachment]` marker before serializing a chunk for
/// summarization/extraction (`sanitize_pre_compaction_content`, §9.2).
pub fn sanitize_pre_compaction_content(text: &str) -> String {
    media_uri_re()
        .replace_all(text, "[Media attachment]")
        .into_owned()
}

/// Run pre-compaction extraction over `content` and append any bullets to the daily file under
/// `dir`. Best-effort: returns [`ExtractionOutcome::Skipped`] on any failure and never propagates an
/// error (the caller treats extraction as non-blocking).
pub async fn run_extraction(
    aux: &dyn Provider,
    content: &str,
    dir: Option<&Path>,
    timeout: Duration,
    now_secs: f64,
) -> ExtractionOutcome {
    let Some(dir) = dir else {
        return ExtractionOutcome::Skipped;
    };
    let sanitized = sanitize_pre_compaction_content(content);
    if sanitized.trim().is_empty() {
        return ExtractionOutcome::NothingToExtract;
    }
    let prompt = format!("{EXTRACTION_PROMPT_HEAD}{sanitized}");
    let request = Request {
        system: String::new(),
        messages: vec![RequestMsg {
            role: "user".into(),
            content: prompt,
            ..Default::default()
        }],
        ..Default::default()
    }
    // The hermes extraction params (`LCM:extraction.py:50-55`).
    .with_params(RequestParams {
        temperature: Some(0.2),
        max_tokens: Some(2000),
        ..Default::default()
    })
    .with_task("extraction");
    let text = match tokio::time::timeout(timeout, aux.chat(request)).await {
        Ok(Ok(out)) => strip_reasoning_blocks(&out.text),
        Ok(Err(_)) | Err(_) => return ExtractionOutcome::Skipped,
    };
    let text = text.trim();
    if text.is_empty() || text == NOTHING_TO_EXTRACT {
        return ExtractionOutcome::NothingToExtract;
    }
    match append_daily(dir, text, now_secs) {
        Ok(()) => ExtractionOutcome::Wrote,
        Err(e) => {
            tracing::warn!(error = %e, "lcm: extraction append failed; skipping");
            ExtractionOutcome::Skipped
        }
    }
}

/// Append a timestamped section to `<dir>/YYYY-MM-DD.md`, creating the directory if needed.
fn append_daily(dir: &Path, bullets: &str, now_secs: f64) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    set_dir_mode(dir);
    let (y, m, d, hh, mm, ss) = civil_datetime(now_secs);
    let file = dir.join(format!("{y:04}-{m:02}-{d:02}.md"));
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(file)?;
    writeln!(f, "## {hh:02}:{mm:02}:{ss:02} UTC")?;
    writeln!(f, "{bullets}")?;
    writeln!(f)?;
    Ok(())
}

fn media_uri_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)data:[^\s;,]*;base64,[A-Za-z0-9+/=]{64,}").expect("media-uri regex")
    })
}

/// Convert unix seconds (UTC) to `(year, month, day, hour, min, sec)` (Howard Hinnant's
/// `civil_from_days`); avoids a `chrono` dependency for the daily-file name (also stamps the
/// `/lcm backup` snapshot filename).
pub(crate) fn civil_datetime(secs: f64) -> (i64, u32, u32, u32, u32, u32) {
    let secs = secs.max(0.0) as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    // days since 1970-01-01 -> civil date.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m as u32, d as u32, hh as u32, mm as u32, ss as u32)
}

#[cfg(unix)]
fn set_dir_mode(dir: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(dir, fs::Permissions::from_mode(0o700));
}

#[cfg(not(unix))]
fn set_dir_mode(_dir: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use daemon_core::provider::{Capabilities, Failure, ModelOutput, ToolCallFormat};

    struct FixedAux {
        reply: Option<String>,
    }

    #[async_trait]
    impl Provider for FixedAux {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_native_tools: false,
                supports_streaming: false,
                tool_call_format: ToolCallFormat::Native,
                max_context: Some(8192),
            }
        }
        async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
            match &self.reply {
                Some(t) => Ok(ModelOutput {
                    text: t.clone(),
                    ..Default::default()
                }),
                None => Err(Failure::Provider("aux down".into())),
            }
        }
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("lcm-extract-{}-{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn civil_date_matches_known_epoch() {
        // 2021-01-01 00:00:00 UTC = 1609459200.
        assert_eq!(civil_datetime(1_609_459_200.0), (2021, 1, 1, 0, 0, 0));
        // 1970-01-01 00:00:00 UTC.
        assert_eq!(civil_datetime(0.0), (1970, 1, 1, 0, 0, 0));
    }

    #[test]
    fn sanitize_strips_media_uris() {
        let s = format!("before data:image/png;base64,{} after", "QUJD".repeat(40));
        let out = sanitize_pre_compaction_content(&s);
        assert!(out.contains("[Media attachment]"));
        assert!(out.contains("before") && out.contains("after"));
        assert!(!out.contains("QUJDQUJD"));
    }

    #[tokio::test]
    async fn writes_bullets_to_daily_file() {
        let dir = tmp("write");
        let aux = FixedAux {
            reply: Some("- decided X\n- committed to Y".into()),
        };
        let outcome = run_extraction(
            &aux,
            "some segment",
            Some(dir.as_path()),
            Duration::from_secs(5),
            1_609_459_200.0,
        )
        .await;
        assert_eq!(outcome, ExtractionOutcome::Wrote);
        let body = fs::read_to_string(dir.join("2021-01-01.md")).unwrap();
        assert!(body.contains("decided X"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn nothing_to_extract_writes_no_file() {
        let dir = tmp("nothing");
        let aux = FixedAux {
            reply: Some(NOTHING_TO_EXTRACT.into()),
        };
        let outcome = run_extraction(
            &aux,
            "small talk",
            Some(dir.as_path()),
            Duration::from_secs(5),
            1_609_459_200.0,
        )
        .await;
        assert_eq!(outcome, ExtractionOutcome::NothingToExtract);
        assert!(!dir.join("2021-01-01.md").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn aux_failure_is_skipped_not_fatal() {
        let dir = tmp("fail");
        let aux = FixedAux { reply: None };
        let outcome = run_extraction(
            &aux,
            "segment",
            Some(dir.as_path()),
            Duration::from_secs(5),
            1_609_459_200.0,
        )
        .await;
        assert_eq!(outcome, ExtractionOutcome::Skipped);
        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn no_dir_is_skipped() {
        let aux = FixedAux {
            reply: Some("- x".into()),
        };
        let outcome = run_extraction(
            &aux,
            "segment",
            None,
            Duration::from_secs(5),
            1_609_459_200.0,
        )
        .await;
        assert_eq!(outcome, ExtractionOutcome::Skipped);
    }

    /// The extraction call carries the hermes params (`LCM:extraction.py:50-55`):
    /// temperature 0.2, max_tokens 2000, task "extraction".
    #[tokio::test]
    async fn extraction_request_carries_params_and_task() {
        struct CapturingAux {
            request: std::sync::Mutex<Option<Request>>,
        }
        #[async_trait]
        impl Provider for CapturingAux {
            fn capabilities(&self) -> Capabilities {
                Capabilities {
                    supports_native_tools: false,
                    supports_streaming: false,
                    tool_call_format: ToolCallFormat::Native,
                    max_context: Some(8192),
                }
            }
            async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
                *self.request.lock().unwrap() = Some(req);
                Ok(ModelOutput {
                    text: super::NOTHING_TO_EXTRACT.into(),
                    ..Default::default()
                })
            }
        }

        let aux = CapturingAux {
            request: std::sync::Mutex::new(None),
        };
        let dir = tmp("params");
        let _ = run_extraction(
            &aux,
            "a segment",
            Some(dir.as_path()),
            Duration::from_secs(5),
            1_609_459_200.0,
        )
        .await;
        let req = aux.request.lock().unwrap().take().expect("aux was called");
        assert_eq!(req.params.temperature, Some(0.2));
        assert_eq!(req.params.max_tokens, Some(2000));
        assert_eq!(req.task.as_deref(), Some("extraction"));
        let _ = fs::remove_dir_all(&dir);
    }
}
