// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The versioned credential-store envelope (credential plan Phase 3).
//!
//! A stored credential blob is either a **bare key** (every legacy secret, every pasted key,
//! every minted provider key — stored verbatim, no wrapping) or a **versioned envelope**: a JSON
//! object carrying the unambiguous magic discriminator [`CREDENTIAL_ENVELOPE_MAGIC`]. Discrimination
//! is BY THE MAGIC ONLY — "parses as JSON" never reclassifies a blob, so a pasted key that happens
//! to be JSON stays a bare key.
//!
//! The envelope carries token MATERIAL plus the trusted descriptor identity
//! (`provider_id`/`method_id`) — never request policy: no header overrides, no URLs. A trusted
//! projector on the provider side (the descriptor table) decides bearer formatting and permitted
//! headers from the method id, so a tampered store cannot redirect a secret to an attacker header
//! or endpoint (see the store threat-model note in the plan).

use serde::{Deserialize, Serialize};

/// The magic key whose presence (as a top-level JSON object member) marks a stored blob as an
/// envelope. Its value is the envelope format version.
pub const CREDENTIAL_ENVELOPE_MAGIC: &str = "daemon_credential";

/// The envelope format version this build writes (and the highest it reads).
pub const CREDENTIAL_ENVELOPE_VERSION: u32 = 1;

/// An OAuth token set: the refreshable credential material one interactive sign-in minted,
/// plus the TRUSTED identity of the descriptor that minted it. Deliberately no request policy
/// (headers/URLs) — the projector owns that, keyed on `method_id`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OAuthTokenSet {
    /// The provider this credential authenticates against (the canonical vendor id, e.g.
    /// `"huggingface"`) — trusted because the NODE wrote it at mint time, not the client.
    pub provider_id: String,
    /// The auth-method (descriptor) identity that minted this set (e.g. `"provider/huggingface"`).
    /// The projector resolves formatting/permitted headers from this id via its own curated table.
    pub method_id: String,
    /// The bearer material presented on requests.
    pub access_token: String,
    /// The refresh token, when the grant issued one (`None` = not refreshable).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    /// Unix seconds when `access_token` expires (`None` = the vendor stated no expiry).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
}

/// One decoded credential-store blob.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialEnvelope {
    /// A bare secret string (API key / token) — the legacy and pasted-key shape, stored verbatim.
    Key(String),
    /// A versioned OAuth token set (see [`OAuthTokenSet`]).
    OAuthTokenSet(OAuthTokenSet),
}

/// Why a magic-marked blob could not be decoded. A bare key never errors (any string without the
/// magic IS a key); an error here means the store holds an envelope this build cannot honor —
/// callers must fail the auth cleanly, NEVER forward the raw blob as a bearer.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EnvelopeError {
    /// The magic is present but the version is newer than this build reads.
    #[error(
        "credential envelope version {0} is newer than supported ({CREDENTIAL_ENVELOPE_VERSION})"
    )]
    UnsupportedVersion(u64),
    /// The magic is present but the envelope body does not decode.
    #[error("credential envelope is malformed: {0}")]
    Malformed(String),
}

/// The serialized v1 envelope: flattened token set + the magic/version and kind tag.
#[derive(Serialize, Deserialize)]
struct EnvelopeV1 {
    #[serde(rename = "daemon_credential")]
    version: u32,
    kind: String,
    #[serde(flatten)]
    token_set: OAuthTokenSet,
}

impl CredentialEnvelope {
    /// Decode a stored blob. Any string NOT carrying the magic (including non-JSON, and JSON
    /// without the magic member) is a bare [`Key`](Self::Key) — verbatim, untrimmed. A blob WITH
    /// the magic must decode as a supported envelope or the caller gets an error to fail cleanly
    /// on (never a silent fall-back to bearer-forwarding the JSON).
    pub fn parse(blob: &str) -> Result<Self, EnvelopeError> {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(blob) else {
            return Ok(Self::Key(blob.to_string()));
        };
        let Some(version) = value.get(CREDENTIAL_ENVELOPE_MAGIC) else {
            return Ok(Self::Key(blob.to_string()));
        };
        let version = version
            .as_u64()
            .ok_or_else(|| EnvelopeError::Malformed("non-integer version".into()))?;
        if version > u64::from(CREDENTIAL_ENVELOPE_VERSION) {
            return Err(EnvelopeError::UnsupportedVersion(version));
        }
        let envelope: EnvelopeV1 =
            serde_json::from_value(value).map_err(|e| EnvelopeError::Malformed(e.to_string()))?;
        match envelope.kind.as_str() {
            "oauth_token_set" => Ok(Self::OAuthTokenSet(envelope.token_set)),
            other => Err(EnvelopeError::Malformed(format!(
                "unknown envelope kind {other:?}"
            ))),
        }
    }

    /// Encode for storage: a bare key stays the verbatim string; a token set becomes the
    /// magic-marked v1 JSON envelope.
    #[must_use]
    pub fn encode(&self) -> String {
        match self {
            Self::Key(k) => k.clone(),
            Self::OAuthTokenSet(ts) => serde_json::to_string(&EnvelopeV1 {
                version: CREDENTIAL_ENVELOPE_VERSION,
                kind: "oauth_token_set".into(),
                token_set: ts.clone(),
            })
            .expect("envelope serialization is infallible (string/number fields only)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> OAuthTokenSet {
        OAuthTokenSet {
            provider_id: "huggingface".into(),
            method_id: "provider/huggingface".into(),
            access_token: "hf_access".into(),
            refresh_token: Some("hf_refresh".into()),
            expires_at: Some(1_900_000_000),
        }
    }

    #[test]
    fn bare_strings_are_keys_even_when_json() {
        assert_eq!(
            CredentialEnvelope::parse("sk-plain").unwrap(),
            CredentialEnvelope::Key("sk-plain".into())
        );
        // JSON without the magic is still a bare key — no "JSON means OAuth" guessing.
        let jsonish = r#"{"access_token":"x","kind":"oauth_token_set"}"#;
        assert_eq!(
            CredentialEnvelope::parse(jsonish).unwrap(),
            CredentialEnvelope::Key(jsonish.into())
        );
    }

    #[test]
    fn token_set_round_trips_through_the_v1_envelope() {
        let env = CredentialEnvelope::OAuthTokenSet(sample());
        let encoded = env.encode();
        assert!(encoded.contains(r#""daemon_credential":1"#), "{encoded}");
        assert_eq!(CredentialEnvelope::parse(&encoded).unwrap(), env);
    }

    #[test]
    fn optional_fields_are_omitted_and_default() {
        let ts = OAuthTokenSet {
            refresh_token: None,
            expires_at: None,
            ..sample()
        };
        let encoded = CredentialEnvelope::OAuthTokenSet(ts.clone()).encode();
        assert!(!encoded.contains("refresh_token"), "{encoded}");
        assert_eq!(
            CredentialEnvelope::parse(&encoded).unwrap(),
            CredentialEnvelope::OAuthTokenSet(ts)
        );
    }

    #[test]
    fn magic_marked_garbage_fails_closed_not_open() {
        // A future version must not be misread (fail — refresh/re-auth, don't guess).
        let future = r#"{"daemon_credential":99,"kind":"oauth_token_set","provider_id":"p","method_id":"m","access_token":"a"}"#;
        assert_eq!(
            CredentialEnvelope::parse(future),
            Err(EnvelopeError::UnsupportedVersion(99))
        );
        // Magic present but the body is not a decodable envelope: an error, NEVER a bare key
        // (forwarding the JSON as a bearer would leak the blob to the vendor).
        let broken = r#"{"daemon_credential":1,"kind":"oauth_token_set"}"#;
        assert!(matches!(
            CredentialEnvelope::parse(broken),
            Err(EnvelopeError::Malformed(_))
        ));
        let unknown_kind = r#"{"daemon_credential":1,"kind":"exotic","provider_id":"p","method_id":"m","access_token":"a"}"#;
        assert!(matches!(
            CredentialEnvelope::parse(unknown_kind),
            Err(EnvelopeError::Malformed(_))
        ));
    }
}
