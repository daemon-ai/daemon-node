//! Natural-language temporal parsing — port of `temporal_parser.py`.
//!
//! `parse_nl_date` first-match priority (ISO -> relative -> weekday -> week/month/year -> intervals
//! -> vague) and `extract_temporal` (`temporal_parser.py` L106-L390). Scaffold.

/// The result of temporal extraction (`temporal_parser.py` `extract_temporal` L385-L389).
#[derive(Clone, Debug, Default)]
pub struct Temporal {
    /// Resolved ISO date, if any.
    pub event_date: Option<String>,
    /// Precision: `day | week | month | year | relative | unknown`.
    pub event_date_precision: String,
    /// Extracted temporal tags.
    pub temporal_tags: Vec<String>,
}

/// Extract temporal signals from text (`temporal_parser.py` L357-L390). Scaffold: none.
pub fn extract_temporal(_text: &str) -> Temporal {
    Temporal {
        event_date_precision: "unknown".to_string(),
        ..Default::default()
    }
}
