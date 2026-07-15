// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `daemon-train` worker binary.
//!
//! Wave-0 scaffold: prints a version line so the supervisor client (`daemon-train-client`) has a
//! real binary to spawn. The length-framed CBOR stdio protocol (swarm-training-spec.md §10.2) and
//! the host runtime land with lane **E**; the node process never links this binary's engine.

#![forbid(unsafe_code)]

fn main() {
    // stdout will become the length-framed cut transport once the protocol lands; for now emit a
    // single version line (diagnostics belong on stderr, like the other coprocessor workers).
    println!(
        "daemon-train {} (tensor-abi {}.{})",
        env!("CARGO_PKG_VERSION"),
        daemon_train::TENSOR_ABI_VERSION >> 16,
        daemon_train::TENSOR_ABI_VERSION & 0xffff,
    );
}
