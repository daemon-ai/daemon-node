// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! A permissive boolean serde adapter for layered configuration.
//!
//! `figment` auto-types environment values (so `DAEMON_X__ENABLE=1` arrives as an integer, `=on`
//! as a string), which plain `bool` deserialization rejects. Applying
//! `#[serde(with = "daemon_common::flex_bool")]` to a `bool` field accepts `true`/`false`, `1`/`0`,
//! `yes`/`no`, `on`/`off` (case-insensitive) across the TOML, env, and CLI layers, while still
//! serializing as a native boolean.

use serde::{de, Deserializer, Serializer};
use std::fmt;

/// Serialize a `bool` natively.
pub fn serialize<S: Serializer>(v: &bool, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_bool(*v)
}

/// Deserialize a permissive boolean (see the module docs for the accepted forms).
pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
    struct V;
    impl de::Visitor<'_> for V {
        type Value = bool;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a boolean (true/false, 1/0, yes/no, on/off)")
        }

        fn visit_bool<E>(self, v: bool) -> Result<bool, E> {
            Ok(v)
        }

        fn visit_u64<E: de::Error>(self, v: u64) -> Result<bool, E> {
            int_bool(i128::from(v))
        }

        fn visit_i64<E: de::Error>(self, v: i64) -> Result<bool, E> {
            int_bool(i128::from(v))
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<bool, E> {
            match v.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" | "t" | "y" => Ok(true),
                "0" | "false" | "no" | "off" | "f" | "n" | "" => Ok(false),
                other => Err(E::custom(format!("invalid boolean {other:?}"))),
            }
        }
    }
    d.deserialize_any(V)
}

fn int_bool<E: de::Error>(v: i128) -> Result<bool, E> {
    match v {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(E::custom(format!("invalid boolean {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use serde::de::value::{BoolDeserializer, Error, StrDeserializer, U64Deserializer};

    fn from_bool(v: bool) -> bool {
        super::deserialize(BoolDeserializer::<Error>::new(v)).expect("bool")
    }
    fn from_u64(v: u64) -> bool {
        super::deserialize(U64Deserializer::<Error>::new(v)).expect("u64")
    }
    fn from_str(v: &str) -> bool {
        super::deserialize(StrDeserializer::<Error>::new(v)).expect("str")
    }

    #[test]
    fn accepts_bool_int_and_string_forms() {
        assert!(from_bool(true));
        assert!(!from_bool(false));
        assert!(from_u64(1));
        assert!(!from_u64(0));
        assert!(from_str("yes"));
        assert!(from_str("on"));
        assert!(!from_str("off"));
        assert!(from_str("TRUE"));
    }

    #[test]
    fn rejects_nonsense() {
        assert!(super::deserialize(StrDeserializer::<Error>::new("maybe")).is_err());
        assert!(super::deserialize(U64Deserializer::<Error>::new(2)).is_err());
    }
}
