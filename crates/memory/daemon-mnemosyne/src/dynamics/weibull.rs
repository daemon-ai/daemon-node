//! Per-type Weibull survival decay — port of `weibull.py`.
//!
//! `boost(t) = exp(-(age_hours / eta)^k)` (`weibull.py` L150-L154), with per-type `{k, eta}` params
//! (`weibull.py` L28-L59, `eta` in hours). Recall blends this as `score*0.7 + wb*0.3`
//! (`beam.py` L6272). Unknown types map to `general`.

/// Default half-life in hours when no per-type param applies (`weibull.py` L63).
pub const DEFAULT_HALFLIFE_HOURS: f64 = 168.0;

/// Per-type Weibull params `(memory_type, k, eta_hours)` — verbatim from `weibull.py` L28-L59.
pub const WEIBULL_PARAMS: &[(&str, f64, f64)] = &[
    ("profile", 0.3, 8760.0),
    ("preference", 0.4, 4380.0),
    ("relationship", 0.35, 8760.0),
    ("learning", 0.7, 1440.0),
    ("fact", 0.8, 720.0),
    ("entity", 0.5, 4380.0),
    ("setup", 0.6, 2160.0),
    ("pattern", 0.6, 1680.0),
    ("context", 0.85, 360.0),
    ("observation", 0.9, 480.0),
    ("artifact", 0.75, 2160.0),
    ("project", 0.85, 1080.0),
    ("goal", 0.9, 720.0),
    ("decision", 1.0, 336.0),
    ("commitment", 1.0, 240.0),
    ("event", 1.2, 168.0),
    ("instruction", 0.9, 480.0),
    ("error", 1.1, 336.0),
    ("issue", 1.1, 336.0),
    ("request", 1.5, 72.0),
    ("general", 1.0, 168.0),
];

/// The `(k, eta)` params for a memory type, falling back to `general`.
fn params_for(memory_type: &str) -> (f64, f64) {
    WEIBULL_PARAMS
        .iter()
        .find(|(t, _, _)| *t == memory_type)
        .map(|(_, k, eta)| (*k, *eta))
        .unwrap_or((1.0, DEFAULT_HALFLIFE_HOURS))
}

/// Weibull decay factor for an age in hours (`weibull.py` `weibull_decay_factor` L157-L183).
pub fn weibull_decay_factor(age_hours: f64, memory_type: &str) -> f64 {
    if age_hours <= 0.0 {
        return 1.0; // future / now -> no decay
    }
    let (k, eta) = params_for(memory_type);
    (-(age_hours / eta).powf(k)).exp()
}

/// Temporal boost from a memory's age (`weibull.py` `weibull_boost` L66-L154). Unlike
/// [`weibull_decay_factor`], an **unknown** age (unparseable / missing timestamp) returns `0.0`
/// (the row gets no temporal credit); a `Some(age)` defers to the survival function (future ages
/// clamp to `1.0`). This is the form the enhanced-recall blend `score*0.7 + wb*0.3` uses
/// (`beam.py` L6272).
pub fn weibull_boost(age_hours: Option<f64>, memory_type: &str) -> f64 {
    match age_hours {
        Some(age) => weibull_decay_factor(age, memory_type),
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_monotonic_and_bounded() {
        let fresh = weibull_decay_factor(1.0, "fact");
        let old = weibull_decay_factor(10_000.0, "fact");
        assert!(fresh <= 1.0 && fresh > old);
        assert!(old >= 0.0);
    }

    #[test]
    fn unknown_type_uses_general() {
        let a = weibull_decay_factor(500.0, "does-not-exist");
        let b = weibull_decay_factor(500.0, "general");
        assert!((a - b).abs() < 1e-12);
    }

    #[test]
    fn future_is_one() {
        assert_eq!(weibull_decay_factor(0.0, "event"), 1.0);
    }

    #[test]
    fn boost_none_age_is_zero() {
        assert_eq!(weibull_boost(None, "fact"), 0.0);
        // A known age defers to the survival function.
        assert_eq!(weibull_boost(Some(0.0), "event"), 1.0);
        assert!(
            (weibull_boost(Some(720.0), "fact") - weibull_decay_factor(720.0, "fact")).abs()
                < 1e-12
        );
    }
}
