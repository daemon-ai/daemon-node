// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Hardware probe for the quant recommender.
//!
//! Reports the memory budget a model must fit into: system RAM (via `sysinfo`, cross-platform) and,
//! best-effort, dedicated GPU VRAM. VRAM is read by shelling to `nvidia-smi` when it is on `PATH`
//! (so we link no GPU SDK and stay dependency-light); on machines without it — including Apple
//! Metal's unified memory — VRAM is `None` and the recommender falls back to the RAM budget.

use std::process::Command;

/// Detected memory the recommender fits a model against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HardwareProbe {
    /// Total dedicated GPU VRAM in bytes, when detectable (NVIDIA via `nvidia-smi`).
    pub vram_bytes: Option<u64>,
    /// Total system RAM in bytes.
    pub ram_bytes: u64,
}

impl HardwareProbe {
    /// Probe the host: RAM via `sysinfo`, VRAM best-effort via `nvidia-smi`.
    pub fn detect() -> Self {
        Self {
            vram_bytes: detect_vram_bytes(),
            ram_bytes: detect_ram_bytes(),
        }
    }

    /// The memory budget a model should fit into: dedicated VRAM when present (a GPU offload
    /// targets VRAM), otherwise system RAM.
    pub fn budget_bytes(&self) -> u64 {
        self.vram_bytes.unwrap_or(self.ram_bytes)
    }

    /// Whether the probe found a discrete GPU (VRAM was detected).
    pub fn has_gpu(&self) -> bool {
        self.vram_bytes.is_some()
    }
}

/// Total system RAM in bytes.
fn detect_ram_bytes() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.total_memory()
}

/// Best-effort dedicated VRAM in bytes: the largest single GPU reported by `nvidia-smi`. A model
/// load targets one device, so we budget against the biggest GPU rather than the sum.
fn detect_vram_bytes() -> Option<u64> {
    // Spawns the fixed `nvidia-smi` probe (argv-only, no shell) for VRAM sizing. Not agent-reachable.
    #[allow(clippy::disallowed_methods)]
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nobytes"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let max_mib = text
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter_map(|n| n.parse::<u64>().ok())
        .max()?;
    (max_mib > 0).then(|| max_mib * 1024 * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_prefers_vram_then_ram() {
        let with_gpu = HardwareProbe {
            vram_bytes: Some(24 * 1024 * 1024 * 1024),
            ram_bytes: 64 * 1024 * 1024 * 1024,
        };
        assert_eq!(with_gpu.budget_bytes(), 24 * 1024 * 1024 * 1024);
        assert!(with_gpu.has_gpu());

        let cpu_only = HardwareProbe {
            vram_bytes: None,
            ram_bytes: 32 * 1024 * 1024 * 1024,
        };
        assert_eq!(cpu_only.budget_bytes(), 32 * 1024 * 1024 * 1024);
        assert!(!cpu_only.has_gpu());
    }

    #[test]
    fn detect_reports_some_ram() {
        // The probe must always find some RAM on a real host.
        assert!(HardwareProbe::detect().ram_bytes > 0);
    }
}
