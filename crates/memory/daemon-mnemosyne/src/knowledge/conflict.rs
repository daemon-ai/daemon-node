//! LLM conflict detection — port of `llm_conflict_detector.py`.
//!
//! Opt-in tier-2 validator (`MNEMOSYNE_LLM_CONFLICT_DETECTION`) atop the embedding heuristic in
//! sleep/consolidation; in the Rust port the LLM call routes through the daemon-core `Provider`.
//! Scaffold.
