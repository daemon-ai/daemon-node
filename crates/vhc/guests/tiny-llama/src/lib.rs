// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `tiny-llama` — the reference guest experiment module.
//!
//! A `cdylib` compiled to `wasm32-unknown-unknown` and instantiated by the `daemon-train` host in
//! the wasm sandbox (tensor-ABI spec §5.1). The model itself — a genuinely tiny LLaMA-family decoder
//! (embedding → N×(rmsnorm → RoPE attention → rmsnorm → SwiGLU) → tied logits, cross-entropy loss,
//! AdamW inner, wired to a comm profile by config) — is the SDK's first-party preset
//! [`daemon_train_sdk::models::TinyLlama`] (architecture §10.5), so the wasm guest and the SDK's sim
//! tests exercise the identical code through the two backends (ABI §10.4). This module is the
//! one-line `experiment!` binding that emits the `da_*` exports.

use daemon_train_sdk::models::TinyLlama;
use daemon_train_sdk::Experiment;

daemon_train_sdk::experiment!(TinyLlama);
