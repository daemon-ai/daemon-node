# daemon

A BEAM-like substrate for durably running and supervising agents: a `daemon-core` engine, a durable
activation/supervision host, and an orchestration layer, organized as a Cargo workspace.

- **Workspace layout & rationale:** [`docs/daemon-workspace-layout.md`](docs/daemon-workspace-layout.md)
- **Authoritative specs:** [`docs/specs/`](docs/specs/)
- **Research / how we got here:** [`docs/research/`](docs/research/) (substrate evaluation, source audit,
  reference extraction) and [`docs/research/hermes/`](docs/research/hermes/) (legacy-system analysis)
- **The engine spec lives with its crate:** [`crates/engine/daemon-core/docs/`](crates/engine/daemon-core/docs/)

## Layout

```text
crates/
  contracts/   daemon-common · daemon-protocol · daemon-supervision
  engine/      daemon-core
  substrate/   daemon-store · daemon-activation · daemon-provision · daemon-credentials
               daemon-telemetry · daemon-host · daemon-transport
  orchestration/ daemon-orchestration
tools/         daemon-tool-shell · daemon-tool-fs · daemon-tool-tkx · daemon-tool-orchestrate
bins/          daemon · daemon-cli
tests/         daemon-conformance · daemon-stub-engine
xtask/
```

## Build

```bash
cargo check --workspace
```

Crates are currently stubs. The build-first milestone is the durable activation core
(`daemon-store` + `daemon-activation`), proven against `daemon-conformance` with `daemon-stub-engine`;
see the build-order mapping in the layout doc.

## Status

Scaffold only. Profiles pin `panic = "unwind"` because catch-unwind-based supervision is void under
`panic = "abort"`.
