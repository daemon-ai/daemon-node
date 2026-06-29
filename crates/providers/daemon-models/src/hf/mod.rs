// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The Hugging Face Hub read surface: a thin `reqwest` client ([`client`]) plus the two read
//! operations the GUI's two-step flow needs — repo [`search`] (step 1) and per-repo file listing
//! ([`files`], step 2). Acquisition (the actual byte transfer) is `hf-hub`'s job (see
//! [`crate::acquire`]); this module is the discovery half.

pub mod client;
pub mod files;
pub mod search;

pub use client::HfClient;
