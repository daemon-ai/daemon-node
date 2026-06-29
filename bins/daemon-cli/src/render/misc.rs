// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The exhaustive tail of the render chain: the ACP catalog, provider/tool listings, config dump,
//! and the generic `{:?}` fallback for variants without a first-class CLI rendering (e.g. the
//! filesystem surface). Because this arm is total, the compiler still proves every `ApiResponse`
//! variant is handled across the render chain.

use daemon_api::ApiResponse;

pub(super) fn render_rest(resp: ApiResponse) {
    match resp {
        ApiResponse::AcpCatalog(entries) => {
            println!("acp agents: {}", entries.len());
            for e in entries {
                println!(
                    "  - {} [{:?}] installed={} version={:?}",
                    e.name, e.source, e.installed, e.version
                );
            }
        }
        ApiResponse::Providers(providers) => {
            for p in providers {
                println!("  - {} available={}", p.name, p.available);
            }
        }
        ApiResponse::Tools(tools) => {
            for t in tools {
                println!("  - {}", t.name);
            }
        }
        ApiResponse::Config(c) => println!("config ({}):\n{}", c.format, c.body),
        // Filesystem-surface responses (daemon-fs-surface-spec.md) and any other variant: the CLI
        // has no first-class fs command yet, so render the debug form generically.
        other => println!("{other:?}"),
    }
}
