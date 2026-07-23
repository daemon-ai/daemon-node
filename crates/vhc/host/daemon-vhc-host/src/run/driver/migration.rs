// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The §10.2 migration wire helpers: the host-side state-manifest section decode (CBOR-value
//! level — the host never links the SDK's typed manifest) and the migration-descriptor
//! encoding handed to `da_migrate` (manifest verbatim + restore bindings in manifest order).

/// One restore binding the host hands `da_migrate` (ABI §10.2, [SF-6]): an **inline** section
/// staged host-side (read via `read_back(kind = 3)`) or a **by-reference** family the new
/// instance registers ([SF-R2]) and streams in `da_run`. Mirrors the SDK's `MigrationSection`
/// untagged forms; built at the CBOR-value level (the host never links the SDK — dependency wall).
pub(crate) enum RestoreBinding {
    /// `{name, staging_id}` — an inline staged section.
    Inline { name: String, staging_id: u64 },
    /// `{name, family}` — a by-reference already-sealed family.
    ByRef {
        name: String,
        family: daemon_vhc_proto::det_state::FamilyRef,
    },
}

/// Build the §10.2 migration-descriptor bytes: the old module's accepted manifest **verbatim**
/// (decoded and re-embedded as a CBOR value — the bytes were journaled verbatim as tag 10; the
/// descriptor is a fresh encoding whose `manifest` field decodes to the identical value) plus the
/// restore bindings in manifest order. Built at the CBOR-value level for the same dependency-wall
/// reason as [`decode_manifest_sections`]; each by-ref family is re-encoded from its canonical
/// form so the field decodes to the SDK's `FamilyRef` verbatim.
pub(crate) fn build_migration_descriptor(
    manifest: &[u8],
    bindings: &[RestoreBinding],
) -> Result<Vec<u8>, String> {
    use ciborium::value::Value;
    let manifest_value: Value = ciborium::de::from_reader(manifest).map_err(|e| e.to_string())?;
    let sections = bindings
        .iter()
        .map(|b| -> Result<Value, String> {
            Ok(match b {
                RestoreBinding::Inline { name, staging_id } => Value::Map(vec![
                    (Value::Text("name".into()), Value::Text(name.clone())),
                    (
                        Value::Text("staging_id".into()),
                        Value::Integer((*staging_id).into()),
                    ),
                ]),
                RestoreBinding::ByRef { name, family } => {
                    // Re-encode the FamilyRef to a CBOR value so the descriptor's `family` field
                    // decodes to the SDK `FamilyRef` shape verbatim.
                    let family_bytes =
                        daemon_vhc_proto::to_canonical_vec(family).map_err(|e| e.to_string())?;
                    let family_value: Value = ciborium::de::from_reader(family_bytes.as_slice())
                        .map_err(|e| e.to_string())?;
                    Value::Map(vec![
                        (Value::Text("name".into()), Value::Text(name.clone())),
                        (Value::Text("family".into()), family_value),
                    ])
                }
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let descriptor = Value::Map(vec![
        (Value::Text("manifest".into()), manifest_value),
        (Value::Text("sections".into()), Value::Array(sections)),
    ]);
    let mut out = Vec::new();
    ciborium::into_writer(&descriptor, &mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// One decoded `state-section-decl` (ABI §10.2) — the fields the HOST verifies (name for the
/// descriptor binding, hash + size for the staged-consistency check). Decoded at the CBOR-value
/// level: the host never links the SDK's typed manifest, and the bytes stay verbatim.
pub(crate) struct SectionDeclWire {
    pub(crate) name: String,
    pub(crate) hash: [u8; 32],
    pub(crate) size: u64,
}

/// Decode the §10.2 `state-manifest`'s `sections` array from its verbatim CBOR bytes.
pub(crate) fn decode_manifest_sections(manifest: &[u8]) -> Result<Vec<SectionDeclWire>, String> {
    use ciborium::value::Value;
    let v: Value = ciborium::de::from_reader(manifest).map_err(|e| e.to_string())?;
    let Value::Map(entries) = v else {
        return Err("state-manifest is not a map".into());
    };
    let sections = entries
        .iter()
        .find_map(|(k, val)| match k {
            Value::Text(t) if t == "sections" => Some(val),
            _ => None,
        })
        .ok_or("state-manifest has no `sections`")?;
    let Value::Array(items) = sections else {
        return Err("`sections` is not an array".into());
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let Value::Map(fields) = item else {
            return Err("a section decl is not a map".into());
        };
        let field = |name: &str| {
            fields.iter().find_map(|(k, val)| match k {
                Value::Text(t) if t == name => Some(val),
                _ => None,
            })
        };
        let name = match field("name") {
            Some(Value::Text(t)) => t.clone(),
            _ => return Err("section decl missing `name`".into()),
        };
        let hash: [u8; 32] = match field("hash") {
            Some(Value::Bytes(b)) => b
                .as_slice()
                .try_into()
                .map_err(|_| "section `hash` is not 32 bytes".to_string())?,
            _ => return Err("section decl missing `hash`".into()),
        };
        let size = match field("size") {
            Some(Value::Integer(i)) => u64::try_from(i128::from(*i))
                .map_err(|_| "section `size` out of range".to_string())?,
            _ => return Err("section decl missing `size`".into()),
        };
        out.push(SectionDeclWire { name, hash, size });
    }
    Ok(out)
}
