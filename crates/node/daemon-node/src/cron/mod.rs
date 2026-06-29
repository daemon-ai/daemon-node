// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The resident cron scheduler (I15) composition glue:
//!
//! - [`worker`]: the [`CronWorker`] struct + lifecycle (reconcile/fire) + `CronScheduler`/`CronFiring`.
//! - [`seed`]: seed-prompt assembly, run capture, `no_agent` scripts, overlay projection.
//! - [`schedule`]: spec decoding + `daemon-schedule` arithmetic (catch-up gate, next-fire advance).

pub mod schedule;
pub mod seed;
pub mod worker;

pub use worker::{CronSkillLoader, CronWorker};
