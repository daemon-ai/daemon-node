// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Fixed-width byte newtypes used across the wire contract.
//!
//! Each serializes as a CBOR **byte string** (`bstr`) via `serialize_bytes`, so the CDDL declares
//! them as `bstr` and the on-wire form is compact (a raw array of `uint` would be both larger and a
//! worse CDDL). Length is enforced Rust-side on deserialize; `cddl-cat` does not check `.size`, so
//! the newtype is the length authority (see `daemon-swarm.cddl`).

use core::fmt;

use serde::de::{self, Visitor};
use serde::{Deserializer, Serializer};

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use core::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

macro_rules! byte_newtype {
    ($name:ident, $len:literal, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub [u8; $len]);

        impl $name {
            /// The fixed byte length of this identifier.
            pub const LEN: usize = $len;

            /// Wrap a fixed-length byte array.
            #[must_use]
            pub const fn new(bytes: [u8; $len]) -> Self {
                Self(bytes)
            }

            /// Borrow the underlying bytes.
            #[must_use]
            pub const fn as_bytes(&self) -> &[u8; $len] {
                &self.0
            }

            /// Lowercase hex rendering.
            #[must_use]
            pub fn to_hex(&self) -> String {
                to_hex(&self.0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self.to_hex())
            }
        }

        impl From<[u8; $len]> for $name {
            fn from(bytes: [u8; $len]) -> Self {
                Self(bytes)
            }
        }

        impl serde::Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_bytes(&self.0)
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct ByteVisitor;
                impl<'de> Visitor<'de> for ByteVisitor {
                    type Value = $name;

                    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        write!(f, concat!("a ", stringify!($len), "-byte string"))
                    }

                    fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<$name, E> {
                        let arr: [u8; $len] = v
                            .try_into()
                            .map_err(|_| E::invalid_length(v.len(), &self))?;
                        Ok($name(arr))
                    }

                    fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<$name, E> {
                        self.visit_bytes(&v)
                    }

                    fn visit_seq<A>(self, mut seq: A) -> Result<$name, A::Error>
                    where
                        A: de::SeqAccess<'de>,
                    {
                        // Tolerate an array-of-uint encoding for robustness (non-canonical inputs).
                        let mut arr = [0u8; $len];
                        for (i, slot) in arr.iter_mut().enumerate() {
                            *slot = seq
                                .next_element()?
                                .ok_or_else(|| de::Error::invalid_length(i, &self))?;
                        }
                        if seq.next_element::<u8>()?.is_some() {
                            return Err(de::Error::invalid_length($len + 1, &self));
                        }
                        Ok($name(arr))
                    }
                }
                deserializer.deserialize_bytes(ByteVisitor)
            }
        }
    };
}

byte_newtype!(
    PeerId,
    32,
    "A node's ed25519 public-key identity (spec §7.2). Sets and records order by these bytes."
);
byte_newtype!(
    Hash,
    32,
    "A blake3 content hash of an artifact / payload / checkpoint (spec §7.3)."
);
byte_newtype!(
    Root,
    32,
    "A blake3 merkle root committing to a set (spec §6.4)."
);
byte_newtype!(
    Seed,
    32,
    "A round seed driving deterministic assignment + the digest schedule (spec §6.3, §5.6)."
);
byte_newtype!(
    Signature,
    64,
    "An ed25519 signature over canonical CBOR (spec §7.3)."
);
