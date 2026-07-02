// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `xtask` — repo automation (codegen, CI helpers).
//!
//! Subcommands:
//! - `gen-headers` — run `cbindgen` over both binding crates to (re)generate the committed C
//!   headers `bindings/daemon-core-ffi/include/daemon_core.h` (the L1 brain seam) and
//!   `bindings/daemon-ffi/include/daemon.h` (the L2 durable-host seam). The generated headers plus
//!   the published `daemon-api.cddl` are the complete non-Rust contract (daemon-ffi-spec §3.6).
//! - `cddl` — check the `daemon-api` mirror CDDL artifact covers the Rust wire enum variants.
//! - `api-fixtures` — write canonical CBOR request/response fixtures for non-Rust clients.
//! - `gen-zcbor` — generate the client zcbor C codec from a CDDL (the artifact `daemon-app` vendors).
//! - `verify-codec` — decode every CBOR fixture with the generated C codec, proving the CDDL/zcbor
//!   path stays byte-compatible with the serde/ciborium runtime wire format.

#![forbid(unsafe_code)]

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::Command;

/// `xtask` — repo automation (codegen, CI helpers).
#[derive(Parser)]
#[command(name = "xtask", about)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// (Re)generate the committed C headers for both binding crates via `cbindgen`.
    GenHeaders,
    /// Check the `daemon-api` mirror CDDL covers the Rust wire enum variants.
    Cddl,
    /// Write canonical CBOR request/response fixtures for non-Rust clients.
    ApiFixtures,
    /// Generate the client zcbor C codec from a CDDL.
    GenZcbor {
        /// The CDDL contract (defaults to the pinned `daemon-api.cddl`).
        #[arg(long)]
        cddl: Option<PathBuf>,
        /// The output directory (defaults to `target/zcbor-codec`).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Decode every CBOR fixture with the generated C codec (wire-compat gate).
    VerifyCodec,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Cmd::GenHeaders => gen_headers(),
        Cmd::Cddl => check_cddl(),
        Cmd::ApiFixtures => gen_api_fixtures(),
        Cmd::GenZcbor { cddl, out } => gen_zcbor(cddl, out),
        Cmd::VerifyCodec => verify_codec(),
    }
}

/// The workspace root (xtask's manifest dir is `<root>/xtask`).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives under the workspace root")
        .to_path_buf()
}

/// Generate the committed C headers for both binding crates via `cbindgen`.
fn gen_headers() -> anyhow::Result<()> {
    let root = workspace_root();
    // (crate name, crate dir relative to root, output header relative to the crate dir).
    let crates = [
        (
            "daemon-core-ffi",
            "bindings/daemon-core-ffi",
            "include/daemon_core.h",
        ),
        ("daemon-ffi", "bindings/daemon-ffi", "include/daemon.h"),
    ];
    for (name, dir, header) in crates {
        gen_one_header(&root, name, dir, header)?;
    }
    Ok(())
}

/// Run `cbindgen` over one binding crate, writing its committed header.
fn gen_one_header(root: &Path, name: &str, dir: &str, header: &str) -> anyhow::Result<()> {
    let crate_dir = root.join(dir);
    let config = crate_dir.join("cbindgen.toml");
    let out = crate_dir.join(header);
    std::fs::create_dir_all(out.parent().unwrap())?;

    let status = Command::new("cbindgen")
        .arg("--config")
        .arg(&config)
        .arg("--crate")
        .arg(name)
        .arg("--output")
        .arg(&out)
        .arg(&crate_dir)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to run cbindgen (is it on PATH?): {e}"))?;
    anyhow::ensure!(status.success(), "cbindgen exited with {status} for {name}");

    println!("generated {}", out.display());
    Ok(())
}

/// Check that the `daemon-api` CDDL mirror artifact exists and names every Rust request/response
/// variant. This is intentionally a syntactic parity gate: schema validation/codegen is handled by
/// downstream CDDL tooling, but adding a Rust wire variant without updating the published contract
/// must fail CI.
fn check_cddl() -> anyhow::Result<()> {
    let root = workspace_root();
    let path = root.join("crates/contracts/daemon-api/daemon-api.cddl");
    let text = read_to_string(&path)?;
    anyhow::ensure!(!text.trim().is_empty(), "{} is empty", path.display());
    for rule in [
        "api-request",
        "api-response",
        "wire_version",
        // wire v2: the merged live session event log shapes.
        "session-log-entry",
        "session-payload",
        "log-page-view",
        "direction",
        "disposition",
        "origin",
        // wire v2: outbound delivery targets + handover (§5.4).
        "delivery-target",
        "sink-kind",
        "route-addr",
    ] {
        anyhow::ensure!(
            text.contains(rule),
            "{} is missing the `{rule}` rule",
            path.display()
        );
    }
    // ApiRequest / ApiResponse live in the `wire` submodule of daemon-api.
    let rust = read_to_string(&root.join("crates/contracts/daemon-api/src/wire.rs"))?;
    assert_cddl_covers_enum(&text, &rust, "ApiRequest", "api-request")?;
    assert_cddl_covers_enum(&text, &rust, "ApiResponse", "api-response")?;
    println!("ok: {} defines the api mirror", path.display());
    Ok(())
}

fn read_to_string(path: &Path) -> anyhow::Result<String> {
    std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))
}

fn assert_cddl_covers_enum(
    cddl: &str,
    rust: &str,
    enum_name: &str,
    rule_name: &str,
) -> anyhow::Result<()> {
    let variants = rust_enum_variants(rust, enum_name)?;
    let missing: Vec<_> = variants
        .iter()
        .filter(|variant| !cddl_rule_mentions_variant(cddl, rule_name, variant))
        .cloned()
        .collect();
    anyhow::ensure!(
        missing.is_empty(),
        "{rule_name} is missing Rust {enum_name} variants: {}",
        missing.join(", ")
    );
    Ok(())
}

fn rust_enum_variants(rust: &str, enum_name: &str) -> anyhow::Result<Vec<String>> {
    let marker = format!("pub enum {enum_name}");
    let start = rust
        .find(&marker)
        .ok_or_else(|| anyhow::anyhow!("could not find `{marker}`"))?;
    let after_marker = &rust[start + marker.len()..];
    let open = after_marker
        .find('{')
        .ok_or_else(|| anyhow::anyhow!("could not find body for `{enum_name}`"))?;
    let body_start = start + marker.len() + open + 1;
    let mut depth = 1i32;
    let mut end = None;
    for (offset, ch) in rust[body_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(body_start + offset);
                    break;
                }
            }
            _ => {}
        }
    }
    let body_end = end.ok_or_else(|| anyhow::anyhow!("unterminated `{enum_name}` body"))?;
    let mut variants = Vec::new();
    let mut depth = 1i32;
    for line in rust[body_start..body_end].lines() {
        let trimmed = line.trim();
        if depth == 1
            && !trimmed.is_empty()
            && !trimmed.starts_with("///")
            && !trimmed.starts_with("#[")
            && !trimmed.starts_with("//")
            && !trimmed.starts_with('}')
        {
            let ident: String = trimmed
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !ident.is_empty() {
                variants.push(ident);
            }
        }
        for ch in line.chars() {
            match ch {
                '{' | '(' => depth += 1,
                '}' | ')' => depth -= 1,
                _ => {}
            }
        }
    }
    variants.sort();
    variants.dedup();
    Ok(variants)
}

/// A Rust enum variant is "covered" when the CDDL carries its externally-tagged wire key as a
/// quoted string `"Variant"`. In the unified CDDL each `api-request`/`api-response` arm is its own
/// named rule (e.g. `request-submit = { "Submit": ... }`, `request-health = "Health"`), so the key
/// lives in the arm rule rather than inline in the union block; searching the whole file is the
/// format-stable parity check. `rule_name` is kept for call-site clarity.
fn cddl_rule_mentions_variant(cddl: &str, rule_name: &str, variant: &str) -> bool {
    let _ = rule_name;
    cddl.contains(&format!("\"{variant}\""))
}

fn gen_api_fixtures() -> anyhow::Result<()> {
    use daemon_api::{
        ApiRequest, ApiResponse, CommandInvocation, CommandOutput, CredentialInfo, EventsPage,
        HealthReport, LogPageView, ModelDescriptor, NodeEvent, ProfileSpec, ProviderDescriptor,
        ProviderKindWire, ProviderSelector, ServiceHealth, SessionPage,
    };
    use daemon_common::{ProfileRef, ReqId, SessionId};
    use daemon_protocol::{AgentCommand, UserMsg};

    let root = workspace_root();
    let out = root.join("crates/contracts/daemon-api/fixtures/cbor");
    std::fs::create_dir_all(&out)?;

    write_cbor(&out, "request-health.cbor", &ApiRequest::Health)?;
    write_cbor(
        &out,
        "request-sessions-query.cbor",
        &ApiRequest::SessionsQuery {
            query: daemon_api::SessionQuery {
                scope: daemon_api::SessionScope::TopLevel,
                after: None,
                limit: 25,
                since_rev: None,
            },
        },
    )?;
    write_cbor(
        &out,
        "request-subscribe.cbor",
        &ApiRequest::Subscribe {
            session: SessionId::new("fixture-session"),
            after_seq: 0,
            max: 64,
        },
    )?;
    write_cbor(
        &out,
        "request-events-since.cbor",
        &ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: Some(1000),
        },
    )?;
    write_cbor(
        &out,
        "request-submit.cbor",
        &ApiRequest::Submit {
            session: SessionId::new("fixture-session"),
            command: AgentCommand::StartTurn {
                input: UserMsg::new("hello from daemon-app"),
                request_id: ReqId(1),
            },
            origin: None,
            profile: Some(ProfileRef::new("default")),
        },
    )?;
    write_cbor(
        &out,
        "request-session-create.cbor",
        &ApiRequest::SessionCreate {
            session: Some(SessionId::new("fixture-session")),
            profile: Some(ProfileRef::new("default")),
        },
    )?;
    write_cbor(
        &out,
        "response-session-created.cbor",
        &ApiResponse::SessionCreated {
            session: SessionId::new("fixture-session"),
        },
    )?;
    write_cbor(&out, "request-profile-list.cbor", &ApiRequest::ProfileList)?;
    write_cbor(
        &out,
        "request-model-current.cbor",
        &ApiRequest::ModelCurrent {
            profile: Some("default".into()),
        },
    )?;
    write_cbor(&out, "request-fs-roots.cbor", &ApiRequest::FsRoots)?;
    write_cbor(&out, "request-command-list.cbor", &ApiRequest::CommandList)?;
    write_cbor(
        &out,
        "request-command-invoke.cbor",
        &ApiRequest::CommandInvoke {
            invocation: CommandInvocation {
                name: "help".into(),
                ..Default::default()
            },
        },
    )?;
    // Onboarding (CON-4 / CON-6): credentials + model discovery/selection.
    write_cbor(
        &out,
        "request-credential-set.cbor",
        &ApiRequest::CredentialSet {
            profile: "default".into(),
            secret: "sk-fixture-secret".into(),
        },
    )?;
    write_cbor(
        &out,
        "request-credential-list.cbor",
        &ApiRequest::CredentialList,
    )?;
    write_cbor(
        &out,
        "request-credential-remove.cbor",
        &ApiRequest::CredentialRemove {
            profile: "default".into(),
        },
    )?;
    write_cbor(&out, "request-models.cbor", &ApiRequest::Models)?;
    write_cbor(
        &out,
        "request-set-session-model.cbor",
        &ApiRequest::SetSessionModel {
            session: SessionId::new("fixture-session"),
            model: "claude-opus-4-8".into(),
            provider: Some(ProviderSelector::GenAi),
        },
    )?;
    // Profiles CRUD (PRO-2/3/4): exercise the now-concrete profile-spec (the optional arms -
    // tool_allowlist Some, base_url/system_prompt set - and the nested budget/tunables maps).
    let mut fixture_spec = ProfileSpec::new("work", ProviderSelector::GenAi, "claude-opus-4-8");
    fixture_spec.system_prompt = "You are a helpful work assistant.".into();
    fixture_spec.tool_allowlist = Some(vec!["read".into(), "search".into()]);
    write_cbor(
        &out,
        "request-profile-create.cbor",
        &ApiRequest::ProfileCreate {
            spec: fixture_spec.clone(),
        },
    )?;
    write_cbor(
        &out,
        "request-profile-update.cbor",
        &ApiRequest::ProfileUpdate {
            spec: fixture_spec.clone(),
        },
    )?;
    write_cbor(
        &out,
        "request-profile-get.cbor",
        &ApiRequest::ProfileGet { id: "work".into() },
    )?;
    write_cbor(
        &out,
        "request-profile-clone.cbor",
        &ApiRequest::ProfileClone {
            source: "default".into(),
            new_id: "work".into(),
        },
    )?;
    write_cbor(
        &out,
        "response-profile.cbor",
        &ApiResponse::Profile(Some(fixture_spec)),
    )?;
    // The daemon-api gateway selector (wire `"daemon_api"`): a full profile-spec exercising the new
    // additive `provider-selector` value so `verify-codec` proves the generated zcbor C decoder
    // accepts it (OpenRouter-style `author/slug` model id + the pinned OpenAI-compatible base URL).
    let daemon_api_spec = ProfileSpec {
        base_url: Some("https://api.daemon.ai/api/v1/".into()),
        ..ProfileSpec::new(
            "daemon",
            ProviderSelector::DaemonApi,
            "anthropic/claude-sonnet-4-5",
        )
    };
    write_cbor(
        &out,
        "response-profile-daemon-api.cbor",
        &ApiResponse::Profile(Some(daemon_api_spec)),
    )?;

    let fixture_descriptor = ModelDescriptor {
        id: "claude-opus-4-8".into(),
        provider: ProviderSelector::GenAi,
        display_name: None,
        context_length: Some(200_000),
        input_price_micros_per_mtok: Some(15_000_000),
        output_price_micros_per_mtok: Some(75_000_000),
        local: false,
    };
    write_cbor(
        &out,
        "response-credentials.cbor",
        &ApiResponse::Credentials(vec![CredentialInfo {
            profile: "default".into(),
            present: true,
            hint: "\u{2026}cret".into(),
        }]),
    )?;
    write_cbor(
        &out,
        "response-models.cbor",
        &ApiResponse::Models(vec![fixture_descriptor.clone()]),
    )?;
    write_cbor(
        &out,
        "response-model-current.cbor",
        &ApiResponse::ModelCurrent(Some(fixture_descriptor)),
    )?;
    // Provider + model discovery (v22): the enumeration op, a credential-aware per-provider listing
    // (with a transient key), and their responses. The response descriptors exercise the additive
    // `provider-descriptor` shape and a `model-descriptor` carrying the optional `display_name`.
    write_cbor(
        &out,
        "request-provider-catalog.cbor",
        &ApiRequest::ProviderCatalog,
    )?;
    write_cbor(
        &out,
        "request-provider-models.cbor",
        &ApiRequest::ProviderModels {
            provider: "anthropic".into(),
            credential_ref: None,
            transient_key: Some("sk-fixture-transient".into()),
        },
    )?;
    write_cbor(
        &out,
        "response-provider-catalog.cbor",
        &ApiResponse::ProviderCatalog(vec![ProviderDescriptor {
            id: "daemon_cloud".into(),
            display_name: "Daemon Cloud".into(),
            kind: ProviderKindWire::DaemonCloud,
            wire_selector: ProviderSelector::DaemonApi,
            // Daemon Cloud needs a key to run turns (lists keyless — see the host-spec semantics).
            requires_key: true,
            supports_model_discovery: true,
            default_base_url: Some("https://api.daemon.ai/api/v1/".into()),
        }]),
    )?;
    write_cbor(
        &out,
        "response-provider-models.cbor",
        &ApiResponse::ProviderModels(vec![ModelDescriptor {
            id: "anthropic/claude-sonnet-4-5".into(),
            provider: ProviderSelector::DaemonApi,
            display_name: Some("Claude Sonnet 4.5".into()),
            context_length: Some(200_000),
            input_price_micros_per_mtok: Some(3_000_000),
            output_price_micros_per_mtok: Some(15_000_000),
            local: false,
        }]),
    )?;
    write_cbor(&out, "response-ok.cbor", &ApiResponse::Ok)?;
    write_cbor(
        &out,
        "response-session-page.cbor",
        &ApiResponse::SessionPage(SessionPage {
            sessions: Vec::new(),
            next_cursor: None,
            rev: 0,
            removed: Vec::new(),
        }),
    )?;
    write_cbor(
        &out,
        "response-log-page.cbor",
        &ApiResponse::LogPage(LogPageView {
            entries: Vec::new(),
            next_seq: 0,
            head_seq: 0,
            epoch: 0,
        }),
    )?;
    write_cbor(
        &out,
        "response-events-page.cbor",
        &ApiResponse::EventsPage(EventsPage {
            events: vec![
                NodeEvent::RosterChanged { rev: 7 },
                NodeEvent::ApprovalPending {
                    session: SessionId::new("fixture-session"),
                    request_id: "req-1".into(),
                },
            ],
            next_cursor: 12,
            head_cursor: 12,
        }),
    )?;
    write_cbor(
        &out,
        "response-fs-roots.cbor",
        &ApiResponse::FsRoots(Vec::new()),
    )?;
    write_cbor(
        &out,
        "response-commands.cbor",
        &ApiResponse::Commands(Vec::new()),
    )?;
    write_cbor(
        &out,
        "response-command-output.cbor",
        &ApiResponse::CommandOutput(CommandOutput::default()),
    )?;
    write_cbor(
        &out,
        "response-health.cbor",
        &ApiResponse::Health(HealthReport {
            all_ok: true,
            services: vec![ServiceHealth {
                name: "fixture".into(),
                ok: true,
                restarts: 0,
                detail: None,
            }],
        }),
    )?;
    // Local model track (Phase 2): exercise the model arrays + ModelRef/ModelSource through the
    // regenerated (cap-bumped to 64) C codec. The quant is the chosen GGUF file carried as
    // ModelRef::Hf{ file: Some(...) }; ModelId is content-derived (no quant in it).
    {
        use daemon_common::{
            DownloadState, DownloadStatus, InstalledModel, ModelEngine, ModelFile, ModelId,
            ModelRef, ModelSource, QuantCandidate, QuantRecommendation, SearchHit, SearchPage,
            SearchQuery, SearchSort,
        };
        let repo = "bartowski/SmolLM2-135M-Instruct-GGUF";
        let gguf = "SmolLM2-135M-Instruct-Q4_K_M.gguf";
        let hf_ref = || {
            ModelRef::new(
                ModelEngine::Llama,
                ModelSource::Hf {
                    repo: repo.into(),
                    file: Some(gguf.into()),
                    revision: "main".into(),
                },
            )
        };
        write_cbor(
            &out,
            "request-model-search.cbor",
            &ApiRequest::ModelSearch {
                query: SearchQuery {
                    text: "SmolLM2".into(),
                    engine: ModelEngine::Llama,
                    sort: SearchSort::Trending,
                    page: 0,
                    limit: 25,
                },
            },
        )?;
        write_cbor(
            &out,
            "request-model-files.cbor",
            &ApiRequest::ModelFiles {
                repo: repo.into(),
                revision: None,
                engine: ModelEngine::Llama,
            },
        )?;
        write_cbor(
            &out,
            "request-model-download.cbor",
            &ApiRequest::ModelDownload { model: hf_ref() },
        )?;
        write_cbor(
            &out,
            "request-model-downloads.cbor",
            &ApiRequest::ModelDownloads,
        )?;
        write_cbor(
            &out,
            "request-model-catalog.cbor",
            &ApiRequest::ModelCatalog,
        )?;
        write_cbor(
            &out,
            "request-model-recommend.cbor",
            &ApiRequest::ModelRecommend(daemon_api::ModelRecommendArgs {
                repo: repo.into(),
                revision: None,
                engine: ModelEngine::Llama,
                budget_bytes: Some(6 * 1024 * 1024 * 1024),
            }),
        )?;
        // Responses exercise the bumped array caps: search-page.results, [model-file],
        // [download-status], [installed-model], plus the nested quant candidate list.
        write_cbor(
            &out,
            "response-model-search.cbor",
            &ApiResponse::ModelSearch(SearchPage {
                page: 0,
                results: vec![SearchHit {
                    repo: repo.into(),
                    author: Some("bartowski".into()),
                    downloads: 12_345,
                    likes: 42,
                    num_parameters: Some(135_000_000),
                    pipeline_tag: Some("text-generation".into()),
                    last_modified: Some("2025-01-01T00:00:00Z".into()),
                    gated: false,
                    private: false,
                }],
                has_more: false,
            }),
        )?;
        write_cbor(
            &out,
            "response-model-files.cbor",
            &ApiResponse::ModelFiles(vec![
                ModelFile {
                    path: gguf.into(),
                    size_bytes: 92_000_000,
                    quant: Some("Q4_K_M".into()),
                    is_split: false,
                    is_first_shard: false,
                },
                ModelFile {
                    path: "SmolLM2-135M-Instruct-Q8_0.gguf".into(),
                    size_bytes: 145_000_000,
                    quant: Some("Q8_0".into()),
                    is_split: false,
                    is_first_shard: false,
                },
            ]),
        )?;
        write_cbor(
            &out,
            "response-model-downloads.cbor",
            &ApiResponse::ModelDownloads(vec![DownloadStatus {
                id: daemon_common::DownloadId(1),
                model: hf_ref(),
                state: DownloadState::Downloading,
                downloaded_bytes: 46_000_000,
                total_bytes: 92_000_000,
                files_done: 0,
                files_total: 1,
                error: None,
            }]),
        )?;
        write_cbor(
            &out,
            "response-model-catalog.cbor",
            &ApiResponse::ModelCatalog(vec![InstalledModel {
                id: ModelId::new("smollm2-135m-q4km"),
                model: hf_ref(),
                display_name: "SmolLM2-135M-Instruct".into(),
                local_path: "/cache/models/SmolLM2-135M-Instruct-Q4_K_M.gguf".into(),
                size_bytes: 92_000_000,
                quant: Some("Q4_K_M".into()),
                installed_at_ms: 1_700_000_000_000,
                arch: Some("llama".into()),
                context_length: Some(8192),
                file_type: Some("Q4_K_M".into()),
            }]),
        )?;
        write_cbor(
            &out,
            "response-model-recommend.cbor",
            &ApiResponse::ModelRecommend(QuantRecommendation {
                engine: ModelEngine::Llama,
                repo: repo.into(),
                file: Some(gguf.into()),
                quant: "Q4_K_M".into(),
                size_bytes: Some(92_000_000),
                budget_bytes: 6 * 1024 * 1024 * 1024,
                fits: true,
                reason: "best quality that fits the detected ~6 GiB budget".into(),
                candidates: vec![
                    QuantCandidate {
                        quant: "Q8_0".into(),
                        file: Some("SmolLM2-135M-Instruct-Q8_0.gguf".into()),
                        size_bytes: Some(145_000_000),
                        fits: true,
                    },
                    QuantCandidate {
                        quant: "Q4_K_M".into(),
                        file: Some(gguf.into()),
                        size_bytes: Some(92_000_000),
                        fits: true,
                    },
                ],
            }),
        )?;
        write_cbor(
            &out,
            "response-model-download-started.cbor",
            &ApiResponse::ModelDownloadStarted(daemon_common::DownloadId(1)),
        )?;
    }
    // Multiplexed/server-streaming envelope (wire L0): prove the Rust serde shapes match the
    // wire-c2s / wire-s2c CDDL rules. The client hand-codes this envelope, so these fixtures are the
    // schema gate that keeps both sides in agreement.
    {
        use daemon_api::{WireC2S, WireS2C, WIRE_FEATURE_MUX, WIRE_FEATURE_STREAM, WIRE_VERSION};
        let features = vec![
            WIRE_FEATURE_MUX.to_string(),
            WIRE_FEATURE_STREAM.to_string(),
        ];
        write_cbor(
            &out,
            "wire-c2s-hello.cbor",
            &WireC2S::Hello {
                wire_version: WIRE_VERSION,
                features: features.clone(),
            },
        )?;
        write_cbor(
            &out,
            "wire-c2s-call.cbor",
            &WireC2S::Call {
                id: 1,
                req: ApiRequest::Subscribe {
                    session: SessionId::new("fixture-session"),
                    after_seq: 0,
                    max: 64,
                },
            },
        )?;
        write_cbor(
            &out,
            "wire-c2s-open.cbor",
            &WireC2S::Open {
                id: 2,
                req: ApiRequest::Subscribe {
                    session: SessionId::new("fixture-session"),
                    after_seq: 0,
                    max: 64,
                },
            },
        )?;
        write_cbor(&out, "wire-c2s-cancel.cbor", &WireC2S::Cancel { id: 1 })?;
        write_cbor(
            &out,
            "wire-s2c-hello.cbor",
            &WireS2C::Hello {
                wire_version: WIRE_VERSION,
                features,
                auth_mechanisms: Vec::new(),
            },
        )?;
        write_cbor(
            &out,
            "wire-s2c-reply.cbor",
            &WireS2C::Reply {
                id: 1,
                res: ApiResponse::Ok,
            },
        )?;
        write_cbor(
            &out,
            "wire-s2c-item.cbor",
            &WireS2C::Item {
                id: 1,
                res: ApiResponse::LogPage(LogPageView {
                    entries: Vec::new(),
                    next_seq: 0,
                    head_seq: 0,
                    epoch: 0,
                }),
            },
        )?;
        write_cbor(
            &out,
            "wire-s2c-end.cbor",
            &WireS2C::End { id: 1, error: None },
        )?;
        write_cbor(
            &out,
            "wire-s2c-reset.cbor",
            &WireS2C::Reset {
                id: 1,
                epoch: 0,
                head_seq: 0,
            },
        )?;
    }

    // ----- access control (Auth 5) -----
    write_cbor(
        &out,
        "request-user-create.cbor",
        &ApiRequest::UserCreate {
            username: "alice".into(),
            password: "correct horse".into(),
            roles: vec!["user".into()],
        },
    )?;
    write_cbor(&out, "request-user-list.cbor", &ApiRequest::UserList)?;
    write_cbor(&out, "request-who-am-i.cbor", &ApiRequest::WhoAmI)?;
    write_cbor(
        &out,
        "request-session-revoke.cbor",
        &ApiRequest::SessionRevoke {
            user_id: "u1".into(),
        },
    )?;
    write_cbor(
        &out,
        "request-resource-grant-create.cbor",
        &ApiRequest::ResourceGrantCreate {
            user_id: "u1".into(),
            resource_kind: "session".into(),
            resource_id: "s1".into(),
            capability: "session_read".into(),
        },
    )?;
    write_cbor(
        &out,
        "response-access-user.cbor",
        &ApiResponse::AccessUser(daemon_api::AccessUser {
            user_id: "u1".into(),
            username: "alice".into(),
            disabled: false,
            created_at: 0,
            roles: vec!["user".into()],
        }),
    )?;
    write_cbor(
        &out,
        "response-access-users.cbor",
        &ApiResponse::AccessUsers(Vec::new()),
    )?;
    write_cbor(
        &out,
        "response-access-roles.cbor",
        &ApiResponse::AccessRoles(vec![daemon_api::RoleInfo {
            role: "admin".into(),
            capabilities: vec!["access_admin".into()],
        }]),
    )?;
    write_cbor(
        &out,
        "response-who-am-i.cbor",
        &ApiResponse::WhoAmI(daemon_api::PrincipalView {
            user_id: "u1".into(),
            username: "alice".into(),
            roles: vec!["admin".into()],
            capabilities: vec!["access_admin".into()],
        }),
    )?;

    println!("generated CBOR fixtures in {}", out.display());
    Ok(())
}

fn write_cbor<T: serde::Serialize>(dir: &Path, name: &str, value: &T) -> anyhow::Result<()> {
    let bytes = daemon_api::to_cbor(value);
    std::fs::write(dir.join(name), bytes)?;
    Ok(())
}

/// Base name passed to the codegen script; the generated entry types are `api_request`/`api_response`.
const ZCBOR_BASENAME: &str = "daemon_api_client";

fn codegen_script(root: &Path) -> PathBuf {
    root.join("crates/contracts/daemon-api/zcbor-codegen.sh")
}

/// The single authoritative CDDL. It is authored in zcbor dialect (quoted map keys, named union
/// arms, labeled tuples, `any` for opaque fields, plus a few `-t` rule-name disambiguators) so the
/// one file both documents the full wire contract and generates the client C codec. `verify-codec`
/// proves the generated decoder accepts real ciborium fixtures; the `daemon-api` cddl-cat
/// conformance tests prove the schema matches the serde wire format.
fn default_cddl(root: &Path) -> PathBuf {
    root.join("crates/contracts/daemon-api/daemon-api.cddl")
}

/// Run the canonical codegen script. `extra` forwards flags such as `--copy-sources`.
fn run_codegen(root: &Path, cddl: &Path, out: &Path, extra: &[&str]) -> anyhow::Result<()> {
    let status = Command::new("bash")
        .arg(codegen_script(root))
        .arg(cddl)
        .arg(out)
        .args(extra)
        .status()
        .map_err(|e| {
            anyhow::anyhow!(
                "running zcbor-codegen.sh (is zcbor on PATH / in the flake shell?): {e}"
            )
        })?;
    anyhow::ensure!(status.success(), "zcbor codegen failed with {status}");
    Ok(())
}

/// `gen-zcbor [--cddl <path>] [--out <dir>]` — (re)generate the client CBOR codec.
///
/// A thin dev wrapper over `zcbor-codegen.sh`. daemon-node owns generation because the CDDL is
/// authoritative here and zcbor lives in this flake; the output is the committed artifact
/// `daemon-app` vendors (no Python/zcbor in the Qt build). The superproject's pure
/// `packages.daemon-zcbor-codec` derivation invokes the same script.
fn gen_zcbor(cddl: Option<PathBuf>, out: Option<PathBuf>) -> anyhow::Result<()> {
    let root = workspace_root();
    let cddl = cddl.unwrap_or_else(|| default_cddl(&root));
    let out = out.unwrap_or_else(|| root.join("target/zcbor-codec"));
    run_codegen(&root, &cddl, &out, &[])?;
    println!(
        "generated zcbor codec from {} in {}",
        cddl.display(),
        out.display()
    );
    Ok(())
}

/// The verify-codec harness: decode every ciborium-produced fixture with the zcbor-generated decoder.
/// A `response-*` filename is decoded as `api_response`, anything else as `api_request`; success
/// means the generated decoder accepted the bytes (ZCBOR_SUCCESS) and consumed all of them.
const VERIFY_CODEC_C: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "daemon_api_client_decode.h"

static unsigned char buf[1u << 20];

int main(int argc, char **argv) {
    int failures = 0;
    for (int i = 1; i < argc; i++) {
        const char *path = argv[i];
        FILE *f = fopen(path, "rb");
        if (!f) {
            fprintf(stderr, "FAIL %s: cannot open\n", path);
            failures++;
            continue;
        }
        size_t n = fread(buf, 1, sizeof buf, f);
        fclose(f);

        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;

        size_t consumed = 0;
        int ret;
        if (strncmp(base, "response", 8) == 0) {
            struct api_response_r *r = calloc(1, sizeof *r);
            ret = cbor_decode_api_response(buf, n, r, &consumed);
            free(r);
        } else {
            struct api_request_r *r = calloc(1, sizeof *r);
            ret = cbor_decode_api_request(buf, n, r, &consumed);
            free(r);
        }

        if (ret != 0) {
            fprintf(stderr, "FAIL %s: zcbor decode error %d\n", base, ret);
            failures++;
        } else if (consumed != n) {
            fprintf(stderr, "FAIL %s: decoded %zu of %zu bytes\n", base, consumed, n);
            failures++;
        } else {
            fprintf(stderr, "ok   %s (%zu bytes)\n", base, n);
        }
    }

    if (failures) {
        fprintf(stderr, "%d fixture(s) failed to decode\n", failures);
        return 1;
    }
    fprintf(stderr, "all fixtures decoded with the generated zcbor codec\n");
    return 0;
}
"#;

/// `verify-codec` — prove the generated C codec accepts real ciborium wire bytes.
///
/// Closes the loop the syntactic `cddl` gate cannot: generate the codec from the CDDL, compile its
/// decoder with the zcbor runtime, then decode every `fixtures/cbor/*.cbor` (each emitted by
/// `api-fixtures` through ciborium — the runtime truth) and assert success + full consumption. Any
/// drift between the serde wire format and the CDDL/zcbor path fails here.
fn verify_codec() -> anyhow::Result<()> {
    let root = workspace_root();

    let fixtures_dir = root.join("crates/contracts/daemon-api/fixtures/cbor");
    if !fixtures_dir.exists() {
        gen_api_fixtures()?;
    }
    let mut fixtures: Vec<PathBuf> = std::fs::read_dir(&fixtures_dir)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", fixtures_dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().map(|ext| ext == "cbor").unwrap_or(false))
        // The multiplexed-envelope fixtures (`wire-c2s-*` / `wire-s2c-*`) are NOT `api-request` /
        // `api-response`, and the vendored C codec is deliberately scoped to those two entry types
        // (the client hand-codes the tiny envelope). Their schema is covered by the cddl-cat
        // conformance test against `wire-c2s` / `wire-s2c`, not this generated-decoder harness.
        .filter(|path| {
            !path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("wire-"))
                .unwrap_or(false)
        })
        .collect();
    fixtures.sort();
    anyhow::ensure!(
        !fixtures.is_empty(),
        "no CBOR fixtures in {}",
        fixtures_dir.display()
    );

    // Decode every committed fixture with the generated codec (an independent C cross-check of the
    // serde wire bytes). Per-variant coverage is no longer asserted here: the unified CDDL now spans
    // the full surface (~150 variants), and the comprehensive "Rust output always matches the CDDL"
    // gate is the cddl-cat round-trip + proptest conformance in the `daemon-api` crate. This harness
    // proves the zcbor-generated decoder agrees with ciborium on the fixtures that exist.
    let work = std::env::temp_dir().join(format!("daemon-verify-codec-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    let codec = work.join("codec");
    std::fs::create_dir_all(&codec)?;
    // `--copy-sources` drops the zcbor C runtime flat alongside the generated codec.
    run_codegen(&root, &default_cddl(&root), &codec, &["--copy-sources"])?;

    let harness_c = work.join("verify_codec.c");
    std::fs::write(&harness_c, VERIFY_CODEC_C)?;
    let bin = work.join("verify-codec");

    let status = Command::new("cc")
        .arg(&harness_c)
        .arg(codec.join(format!("{ZCBOR_BASENAME}_decode.c")))
        .arg(codec.join("zcbor_decode.c"))
        .arg(codec.join("zcbor_common.c"))
        .arg(format!("-I{}", codec.display()))
        .arg("-o")
        .arg(&bin)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to run cc (is it in the flake shell?): {e}"))?;
    anyhow::ensure!(
        status.success(),
        "compiling the verify harness failed with {status}"
    );

    let status = Command::new(&bin).args(&fixtures).status()?;
    anyhow::ensure!(
        status.success(),
        "codec verification failed: a fixture did not decode with the generated codec"
    );
    let _ = std::fs::remove_dir_all(&work);
    println!(
        "verified {} fixtures decode with the generated zcbor codec",
        fixtures.len()
    );
    Ok(())
}
