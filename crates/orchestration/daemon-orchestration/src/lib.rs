//! `daemon-orchestration` — the fleet runtime (not an engine).
//!
//! The machinery *between* the brain and the wire for an orchestrator node (layout §4): the engine
//! decides *what/when* to delegate (policy) and calls in through `daemon-tool-orchestrate`; this
//! crate owns the *how* — child placement, the management-protocol client loop, the child-event
//! fan-in, and the upward answer/escalation chain. It is a runtime/state library, deliberately
//! **not** a second engine: it deals only in `daemon_supervision::ManagedUnit` + the durable
//! [`daemon_store::SessionStore`], so the brain stays `daemon-core` and a deterministic-policy
//! driver could reuse the same runtime without a conversation engine at all.
//!
//! The phase-4 worker [`FleetRuntime::process_jobs_once`] is the real replacement for the substrate's
//! placeholder worker: it drains a parent's durable delegation job, spawns and drives the child as a
//! managed unit, folds the child's `ManageEvent`s into fleet state, answers/escalates the child's
//! `ManageRequest`s, and records the child's outcome as the parent's `JobCompletion` — which wakes
//! the parent as a `BackgroundCompletion` (synthesis §3.1; layout §4).
//!
//! Child construction is the injected [`ChildSpawner`] seam (engine-backed in tests / `bins/daemon`),
//! so the runtime never depends on `daemon-core`. See `docs/research/daemon-orchestration-synthesis.md`.

#![forbid(unsafe_code)]

pub mod orchestrator;
pub mod policy;
pub mod registry;
pub mod runtime;
pub mod spawner;

pub use orchestrator::{OrchestratorSpawner, OrchestratorUnit};
pub use policy::{AnswerPolicy, Decision, DefaultAnswerPolicy};
pub use registry::{ChildRecord, ChildStatus};
pub use runtime::{FleetRuntime, OrchestrationError};
pub use spawner::ChildSpawner;
