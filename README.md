# daemon

A BEAM-like substrate for durably running and supervising agents: a `daemon-core` engine, a durable
activation/supervision host, and an orchestration layer, organized as a Cargo workspace.

- **Workspace layout & rationale:** [`docs/daemon-workspace-layout.md`](docs/daemon-workspace-layout.md)
- **Authoritative specs:** [`docs/specs/`](docs/specs/)
- **Research / how we got here:** design inputs are kept outside this workspace in the companion
  `daemon-hermes` archive; specs in this repo are the normative source.
- **The engine spec lives with its crate:** [`crates/engine/daemon-core/docs/`](crates/engine/daemon-core/docs/)

## Layout

```text
crates/
  contracts/      daemon-common · daemon-protocol · daemon-supervision · daemon-api
  engine/         daemon-core · daemon-context-lcm
  memory/         daemon-mnemosyne
  substrate/      daemon-store · daemon-activation · daemon-provision · daemon-credentials
                  daemon-telemetry · daemon-host · daemon-transport · daemon-schedule
  providers/      daemon-providers · daemon-infer · daemon-models
  adapters/       daemon-acp · daemon-http · daemon-mcp-client · daemon-delivery
                  daemon-ingest · daemon-matrix
  coprocessor/    daemon-metta · daemon-metta-client · daemon-pytool · daemon-pytool-client
  orchestration/  daemon-orchestration
  node/           daemon-node
  skills/         daemon-skills
tools/         shell · fs · tkx (stub) · orchestrate · cron · web · browser · metta
               todo · clarify · skill
bindings/      daemon-core-ffi · daemon-ffi
bins/          daemon · daemon-cli
tests/         daemon-conformance
xtask/
```

## Build

```bash
cargo check --workspace
```

The default workspace gate checks the Rust surfaces that do not require heavyweight local-inference or
MeTTa engine features. Optional worker engines remain feature-gated in their own crates.

## Status

The workspace contains the durable host, reference engine, NodeApi/CLI surfaces, FFI bindings, chat and
HTTP adapters, local/remote provider wiring, memory/LCM layers, and conformance tests. Some long-tail
surfaces are intentionally still stubs or partial ports, notably `daemon-tool-tkx`, parts of
Mnemosyne, and deferred remote-host transport. Profiles pin `panic = "unwind"` because
catch-unwind-based supervision is void under `panic = "abort"`.
