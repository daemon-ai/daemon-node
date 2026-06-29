// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-conformance` — the substrate + translation conformance harness.
//!
//! The executable acceptance gate for the build-first milestones: the seven substrate acceptance
//! tests from [`rust-substrate-evaluation.md`](../../../docs/specs/rust-substrate-evaluation.md) §6,
//! run against the in-memory [`daemon_store::InMemoryStore`] driven through [`daemon_activation`].
//! From phase 3 the engine under test is the *real* `daemon-core`, driven via the host's
//! [`CoreEngineFactory`](daemon_host::CoreEngineFactory) (the stub engine is retired): the substrate
//! invariants are now proven against the real engine's deterministic delegate→suspend→resume cycle.
//!
//! Coverage map (acceptance test -> lifecycle §4 invariant):
//! 1 churn/baseline (#8), 2 crash-after-every-boundary (#2/#3/#7), 3 idempotency (#2/#3),
//! 4 dual-node fencing (#5/#6), 5 empty-mailbox kill (#1/#7), 6 ownership-transfer (#5/#6),
//! 7 lost-wake recovery (#1/#7).
//!
//! `mod supervision` (phase 2): the resident-service supervisor (restart/backoff/meltdown) and the
//! running `daemon_host::Host` driving sessions to completion under churn and service crashes.
//! `mod translation` (phase 3 gate): the §17 ⇄ management protocol round-trip — the host presents
//! the real engine as a `ManagedUnit` and the supervision §4 mapping table is exercised end to end.
//! `mod orchestration` (phase 4 gate): one engine delegates to a child via the `daemon-orchestration`
//! fleet runtime + `daemon-tool-orchestrate` veneer — events fan in, the child's completion wakes the
//! parent, and a child request is answered/escalated (layout §7 phase-4 gate).

#![forbid(unsafe_code)]

#[cfg(test)]
mod acceptance;
#[cfg(test)]
mod approval;
#[cfg(test)]
mod background_spawn;
#[cfg(test)]
mod credentials;
#[cfg(test)]
mod harness;
#[cfg(test)]
mod journal;
#[cfg(test)]
mod node;
#[cfg(test)]
mod orchestration;
#[cfg(test)]
mod store_backends;
#[cfg(test)]
mod supervision;
#[cfg(test)]
mod tool_provider;
#[cfg(test)]
mod translation;
#[cfg(test)]
mod web_tools;
