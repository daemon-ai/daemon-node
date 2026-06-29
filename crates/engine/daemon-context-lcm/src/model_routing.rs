// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Model-routing shim (`daemon-context-lcm-port-spec.md` §12.4).
//!
//! Python's `model_routing.py` parses `summary_model` / `expansion_model` override strings into a
//! `{provider, model}` pair and applies them as per-call LLM kwargs. daemon-core has **no per-call
//! model/provider kwargs** (§7.4) — the provider *is* the model — so this is a thin **config →
//! aux-profile selection** shim: [`parse_lcm_model_override`] documents the parse rules, and the
//! routing *effect* is which registered aux `Provider` the engine resolves at construction. The
//! minimal port wires a single aux provider, so the parsed route is currently informational
//! (surfaced by `lcm_status`/diagnostics); multi-provider resolution from a registry is out of scope.

/// Registry providers that an unprefixed `provider/model` override may name. Anything else is treated
/// as a model-only override (the leading token is part of the model id).
const PROVIDER_ALLOWLIST: &[&str] = &[
    "openai",
    "anthropic",
    "gemini",
    "google",
    "groq",
    "ollama",
    "openrouter",
    "mistral",
    "cohere",
    "xai",
    "deepseek",
    "together",
];

/// A parsed model override (`parse_lcm_model_override`, §12.4).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ModelRoute {
    /// The original override string.
    pub raw: String,
    /// An allowlisted registry provider, when the override selected one.
    pub provider: Option<String>,
    /// A named custom provider (the `custom:<name>` prefix).
    pub custom_provider: Option<String>,
    /// The model id (always set for a non-empty override).
    pub model: Option<String>,
}

/// Parse a `summary_model`/`expansion_model` override. Rules (`model_routing.py:61-88`):
/// - empty → `None`.
/// - `custom:<name>[/<model>]` → a named custom provider (+ optional model).
/// - `<provider>/<model>` where `<provider>` is allowlisted → that registry provider + model.
/// - otherwise → a model-only override (the whole string is the model id).
pub fn parse_lcm_model_override(spec: &str) -> Option<ModelRoute> {
    let spec = spec.trim();
    if spec.is_empty() {
        return None;
    }
    if let Some(rest) = spec.strip_prefix("custom:") {
        let (name, model) = split_once_slash(rest);
        return Some(ModelRoute {
            raw: spec.to_string(),
            provider: None,
            custom_provider: Some(name.to_string()),
            model: model.map(|m| m.to_string()),
        });
    }
    match split_once_slash(spec) {
        (left, Some(right)) if PROVIDER_ALLOWLIST.contains(&left.to_ascii_lowercase().as_str()) => {
            Some(ModelRoute {
                raw: spec.to_string(),
                provider: Some(left.to_string()),
                custom_provider: None,
                model: Some(right.to_string()),
            })
        }
        _ => Some(ModelRoute {
            raw: spec.to_string(),
            provider: None,
            custom_provider: None,
            model: Some(spec.to_string()),
        }),
    }
}

/// Split on the first `/`, returning `(head, Some(tail))` or `(whole, None)`.
fn split_once_slash(s: &str) -> (&str, Option<&str>) {
    match s.split_once('/') {
        Some((a, b)) => (a, Some(b)),
        None => (s, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_none() {
        assert!(parse_lcm_model_override("").is_none());
        assert!(parse_lcm_model_override("   ").is_none());
    }

    #[test]
    fn custom_prefix_routes_to_named_provider() {
        let r = parse_lcm_model_override("custom:myprov/some-model").unwrap();
        assert_eq!(r.custom_provider.as_deref(), Some("myprov"));
        assert_eq!(r.model.as_deref(), Some("some-model"));
        assert!(r.provider.is_none());

        let bare = parse_lcm_model_override("custom:myprov").unwrap();
        assert_eq!(bare.custom_provider.as_deref(), Some("myprov"));
        assert!(bare.model.is_none());
    }

    #[test]
    fn allowlisted_provider_slash_model() {
        let r = parse_lcm_model_override("anthropic/claude-haiku").unwrap();
        assert_eq!(r.provider.as_deref(), Some("anthropic"));
        assert_eq!(r.model.as_deref(), Some("claude-haiku"));
    }

    #[test]
    fn unknown_leading_token_is_model_only() {
        // A slug like a HF org/model is NOT an allowlisted provider -> model-only.
        let r = parse_lcm_model_override("myorg/my-model").unwrap();
        assert!(r.provider.is_none());
        assert_eq!(r.model.as_deref(), Some("myorg/my-model"));

        let plain = parse_lcm_model_override("gpt-4o-mini").unwrap();
        assert!(plain.provider.is_none());
        assert_eq!(plain.model.as_deref(), Some("gpt-4o-mini"));
    }
}
