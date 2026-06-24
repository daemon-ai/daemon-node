//! SHMR (Self-Harmonizing Memory Reasoning) — port of `shmr.py` (P3, opt-in background pass).
//!
//! Greedy cosine clustering (>= 0.70) + belief convergence (<= 3 iters to harmony >= 0.60) writing
//! `harmonic_beliefs` / `memory_resonance_log`. NOTE: Python never wires this into `sleep()` despite
//! its docstring (`shmr.py` L356); the port keeps it an explicit opt-in pass. Runtime clustering is
//! not wired yet.

/// Cluster similarity threshold (`shmr.py` L30).
pub const SIMILARITY_THRESHOLD: f64 = 0.70;
/// Harmony acceptance threshold (`shmr.py` L31).
pub const HARMONY_THRESHOLD: f64 = 0.60;

/// Whether a cosine similarity score belongs in an SHMR candidate cluster.
pub fn is_cluster_candidate(similarity: f64) -> bool {
    similarity >= SIMILARITY_THRESHOLD
}

/// Whether a convergence score is high enough to accept as harmonic.
pub fn accepts_harmony(harmony: f64) -> bool {
    harmony >= HARMONY_THRESHOLD
}
