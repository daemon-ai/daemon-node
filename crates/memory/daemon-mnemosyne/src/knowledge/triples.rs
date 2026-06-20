//! Temporal TripleStore — port of `triples.py`.
//!
//! Single-current-truth SPO chains: `add(supersede=true)` stamps prior open rows' `valid_until` to
//! the new `valid_from` then inserts; `query(as_of)` selects `valid_from <= as_of AND (valid_until
//! IS NULL OR valid_until > as_of)` (`triples.py` L163-L227). Scaffold: schema lives in
//! [`crate::store::schema`]; the `TripleStore` operations are TODO.

/// A temporal triple row (`triples` table, `triples.py` L91-L108).
#[derive(Clone, Debug)]
pub struct Triple {
    /// Subject.
    pub subject: String,
    /// Predicate.
    pub predicate: String,
    /// Object.
    pub object: String,
    /// Start of validity (ISO date).
    pub valid_from: String,
    /// End of validity (ISO date), `None` while open/current.
    pub valid_until: Option<String>,
    /// Source tag.
    pub source: String,
    /// Confidence `[0, 1]`.
    pub confidence: f64,
}
