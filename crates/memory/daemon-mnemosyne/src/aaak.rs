//! AAAK lossless shorthand — port of `aaak.py` (the dependency-free sleep-summary fallback).
//!
//! Scaffold: the category/phrase/structural replacement maps (`aaak.py` L11-L90) and the `encode`
//! pipeline (L125-L152) are TODO; this stub returns the input unchanged so the engine compiles.

/// Encode text into AAAK shorthand (`aaak.py` `encode` L125-L152). Scaffold: identity.
pub fn encode(text: &str) -> String {
    // TODO: apply CATEGORY_MAP -> PHRASE_MAP (longest-first) -> STRUCTURAL_REPLACEMENTS.
    text.to_string()
}
