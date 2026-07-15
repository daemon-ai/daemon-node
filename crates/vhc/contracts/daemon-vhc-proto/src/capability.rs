// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Capability sets and subset admission (spec §6.5, §16; TDD PROTO-12).
//!
//! Every host primitive carries a `name@version` tag (spec §5.2). A run's envelope pins its
//! **required** set (the module's static import list, §6.1); a peer advertises the set its worker
//! supports (§10.2). The join predicate is **subset inclusion**: `required ⊆ advertised`. Growth is
//! additive (new ops); a breaking change is a new major, so admission compares exact `name@version`
//! tokens. The envelope's capability list is a *pre-screen* only — a peer re-derives the true set
//! from the module bytes at assess (§6.5); this type is the shared representation for both.

use std::collections::BTreeSet;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::SwarmProtoError;

/// A single `name@version` capability (e.g. `tensor-abi@1`, `adamw_step@1`).
///
/// Serializes as the `name@version` string token (matching the envelope's capability list).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Capability {
    /// The op / ABI name.
    pub name: String,
    /// The major version.
    pub version: u32,
}

impl Capability {
    /// Construct from parts.
    #[must_use]
    pub fn new(name: impl Into<String>, version: u32) -> Self {
        Self {
            name: name.into(),
            version,
        }
    }

    /// Parse a `name@version` token.
    pub fn parse(token: &str) -> Result<Self, SwarmProtoError> {
        let (name, version) = token.rsplit_once('@').ok_or_else(|| {
            SwarmProtoError::Capability(format!("`{token}` is not a name@version token"))
        })?;
        if name.is_empty() {
            return Err(SwarmProtoError::Capability(format!(
                "`{token}` has an empty capability name"
            )));
        }
        let version = version.parse::<u32>().map_err(|_| {
            SwarmProtoError::Capability(format!("`{token}` has a non-numeric version"))
        })?;
        Ok(Self {
            name: name.to_string(),
            version,
        })
    }

    /// Render back to the `name@version` token.
    #[must_use]
    pub fn token(&self) -> String {
        format!("{}@{}", self.name, self.version)
    }
}

impl Serialize for Capability {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.token())
    }
}

impl<'de> Deserialize<'de> for Capability {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct TokenVisitor;
        impl Visitor<'_> for TokenVisitor {
            type Value = Capability;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a name@version capability token")
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Capability, E> {
                Capability::parse(v).map_err(|e| E::custom(e.to_string()))
            }
        }
        deserializer.deserialize_str(TokenVisitor)
    }
}

/// A typed, de-duplicated set of capabilities.
///
/// Serializes as a CBOR array of `name@version` tokens (sorted, since it is a `BTreeSet`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CapabilitySet(BTreeSet<Capability>);

impl CapabilitySet {
    /// An empty set.
    #[must_use]
    pub fn new() -> Self {
        Self(BTreeSet::new())
    }

    /// Build from `name@version` tokens (e.g. the envelope's `capabilities` list).
    pub fn from_tokens<I, S>(tokens: I) -> Result<Self, SwarmProtoError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut set = BTreeSet::new();
        for token in tokens {
            set.insert(Capability::parse(token.as_ref())?);
        }
        Ok(Self(set))
    }

    /// Insert a capability.
    pub fn insert(&mut self, cap: Capability) -> bool {
        self.0.insert(cap)
    }

    /// Whether the exact `name@version` is present.
    #[must_use]
    pub fn contains(&self, cap: &Capability) -> bool {
        self.0.contains(cap)
    }

    /// The capabilities in `required` that are **not** in `self` (self = advertised).
    #[must_use]
    pub fn missing(&self, required: &CapabilitySet) -> Vec<Capability> {
        required.0.difference(&self.0).cloned().collect()
    }

    /// Admission check: `required ⊆ self` (self = advertised). Errors listing the missing ops if not.
    pub fn admits(&self, required: &CapabilitySet) -> Result<(), SwarmProtoError> {
        let missing = self.missing(required);
        if missing.is_empty() {
            Ok(())
        } else {
            let tokens: Vec<String> = missing.iter().map(Capability::token).collect();
            Err(SwarmProtoError::Capability(format!(
                "advertised set is missing required capabilities: {}",
                tokens.join(", ")
            )))
        }
    }

    /// Iterate the capabilities (sorted).
    pub fn iter(&self) -> impl Iterator<Item = &Capability> {
        self.0.iter()
    }

    /// The number of capabilities.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl FromIterator<Capability> for CapabilitySet {
    fn from_iter<I: IntoIterator<Item = Capability>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_roundtrip() {
        let c = Capability::parse("tensor-abi@1").unwrap();
        assert_eq!(c, Capability::new("tensor-abi", 1));
        assert_eq!(c.token(), "tensor-abi@1");
    }

    #[test]
    fn parse_rejects_malformed() {
        assert!(Capability::parse("noversion").is_err());
        assert!(Capability::parse("@1").is_err());
        assert!(Capability::parse("op@notanumber").is_err());
    }

    #[test]
    fn name_with_at_uses_last_separator() {
        // rsplit keeps `a@b` names working if that ever arises; version is the final segment.
        let c = Capability::parse("group@op@2").unwrap();
        assert_eq!(c.name, "group@op");
        assert_eq!(c.version, 2);
    }
}
