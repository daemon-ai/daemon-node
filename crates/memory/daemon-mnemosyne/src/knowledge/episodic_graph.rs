//! EpisodicGraph — port of `episodic_graph.py`.
//!
//! Rule-based gists, regex SPO facts, and `graph_edges` (types `rel`/`ctx`/`syn`/`related_to`/
//! `references`); `find_related_memories(depth)` BFS (`episodic_graph.py` L113-L484). Proactive
//! linking is engine-side (`beam.py` `_proactively_link` L3358). Scaffold.

/// A graph edge (`graph_edges` table, `episodic_graph.py` L146-L155).
#[derive(Clone, Debug)]
pub struct GraphEdge {
    /// Source node id.
    pub source: String,
    /// Target node id.
    pub target: String,
    /// Edge type (`ctx`, `related_to`, `references`, ...).
    pub edge_type: String,
    /// Edge weight.
    pub weight: f64,
}
