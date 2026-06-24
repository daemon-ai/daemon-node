//! `xtask` — repo automation (codegen, CI helpers).
//!
//! Subcommands:
//! - `gen-headers` — run `cbindgen` over both binding crates to (re)generate the committed C
//!   headers `bindings/daemon-core-ffi/include/daemon_core.h` (the L1 brain seam) and
//!   `bindings/daemon-ffi/include/daemon.h` (the L2 durable-host seam). The generated headers plus
//!   the published `daemon-api.cddl` are the complete non-Rust contract (daemon-ffi-spec §3.6).
//! - `cddl` — check the `daemon-api` mirror CDDL artifact covers the Rust wire enum variants.
//! - `api-fixtures` — write canonical CBOR request/response fixtures for non-Rust clients.
//! - `zcbor-spike` — run zcbor codegen over a representative CDDL subset.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> anyhow::Result<()> {
    let sub = std::env::args().nth(1).unwrap_or_default();
    match sub.as_str() {
        "gen-headers" => gen_headers(),
        "cddl" => check_cddl(),
        "api-fixtures" => gen_api_fixtures(),
        "zcbor-spike" => run_zcbor_spike(),
        other => {
            eprintln!("usage: xtask <gen-headers|cddl|api-fixtures|zcbor-spike>");
            anyhow::bail!("unknown xtask subcommand: {other:?}");
        }
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
    let rust = read_to_string(&root.join("crates/contracts/daemon-api/src/lib.rs"))?;
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

fn cddl_rule_mentions_variant(cddl: &str, rule_name: &str, variant: &str) -> bool {
    let Some(start) = cddl.find(&format!("{rule_name} =")) else {
        return false;
    };
    let tail = &cddl[start..];
    let end = tail
        .find("\n\n")
        .map(|offset| start + offset)
        .unwrap_or(cddl.len());
    let rule = &cddl[start..end];
    rule.contains(&format!("\"{variant}\""))
}

fn gen_api_fixtures() -> anyhow::Result<()> {
    use daemon_api::{ApiRequest, ApiResponse, HealthReport, ServiceHealth};
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
    write_cbor(&out, "request-profile-list.cbor", &ApiRequest::ProfileList)?;
    write_cbor(
        &out,
        "request-model-current.cbor",
        &ApiRequest::ModelCurrent {
            profile: Some("default".into()),
        },
    )?;
    write_cbor(&out, "request-fs-roots.cbor", &ApiRequest::FsRoots)?;
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
    println!("generated CBOR fixtures in {}", out.display());
    Ok(())
}

fn write_cbor<T: serde::Serialize>(dir: &Path, name: &str, value: &T) -> anyhow::Result<()> {
    let bytes = daemon_api::to_cbor(value);
    std::fs::write(dir.join(name), bytes)?;
    Ok(())
}

fn run_zcbor_spike() -> anyhow::Result<()> {
    let root = workspace_root();
    let schema = root.join("crates/contracts/daemon-api/zcbor-smoke.cddl");
    let out = root.join("target/zcbor-spike");
    std::fs::create_dir_all(out.join("src"))?;
    std::fs::create_dir_all(out.join("include"))?;
    let status = Command::new("zcbor")
        .arg("code")
        .arg("--cddl")
        .arg(&schema)
        .arg("--entry-types")
        .arg("api-request")
        .arg("api-response")
        .arg("--decode")
        .arg("--encode")
        .arg("--default-max-qty")
        .arg("16")
        .arg("--output-c")
        .arg(out.join("src/daemon_api_smoke.c"))
        .arg("--output-h")
        .arg(out.join("include/daemon_api_smoke.h"))
        .arg("--output-h-types")
        .arg(out.join("include/daemon_api_smoke_types.h"))
        .status()
        .map_err(|e| {
            anyhow::anyhow!("failed to run zcbor (is it in the daemon flake shell?): {e}")
        })?;
    anyhow::ensure!(status.success(), "zcbor codegen failed with {status}");
    println!("generated zcbor smoke codec in {}", out.display());
    Ok(())
}
