// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **Execution Grant** — the host's selected logical configuration, delivered to the guest
//! (`docs/specs/vhc-architecture-spec.md` §9.6 `[RC-11]`, `[RC-12]`).
//!
//! Some algorithm parameters are legitimately free within bounds and the right value depends on
//! the machine. That does not license the guest to inspect the machine. The guest states the
//! bounded choice set in its [`crate::resource_plan::LogicalResourcePlan`]; the host composes over
//! that set with a certified Backend Execution Profile, selects an admissible configuration under
//! the declared scope, and hands back **logical values only** — selected micro-batch, token
//! window, concurrency. No backend identity, no device memory figure, no adapter or driver data,
//! no profile content.
//!
//! ## The grant is not an input to its own derivation
//!
//! The canonical configuration and the immutable Capability Grants are the complete inputs from
//! which a guest derives, and later reproduces, its parametric plan. The Execution Grant is a
//! separately encoded, separately hashed artifact produced only *after* that plan has been
//! composed and a choice selected. It MUST NOT be inserted into, mutate, or otherwise affect any
//! byte from which the plan is derived — which is why it travels through its own ABI export
//! rather than through the Capability Grants document.
//!
//! ## What is observable, stated honestly
//!
//! Because the host selected the value using physical facts, the value may reveal something about
//! capacity. Promising the guest can infer nothing would be untestable. The enforceable property
//! is narrower and real: the guest receives no physical identifier, measurement or profile
//! content, and can branch only on the deliberately declassified logical choice. Two hosts that
//! produce the same grant are indistinguishable to the guest through this interface.

use std::collections::BTreeMap;

use ciborium::value::Value;

use crate::bytes::Hash;
use crate::canonical::to_canonical_vec;
use crate::hash::blake3_hash;
use crate::resource_plan::{
    Binding, DimensionValue, Domain, LogicalResourcePlan, PlanRefusal, SelectionScope,
};

/// The schema this build authors and accepts.
pub const EXECUTION_GRANT_SCHEMA: u64 = 1;

/// The hard ceiling on a grant's canonical encoding, in bytes (`[RC-12]`).
pub const EXECUTION_GRANT_BYTES_MAX: usize = 65_536;

/// One selected logical value. The grammar admits four scalar shapes for forward compatibility;
/// validation against a plan narrows them to what that plan's dimension domains actually declare.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GrantValue {
    /// An unsigned selection — a micro-batch, a token window, a concurrency level.
    Uint(u64),
    /// A signed selection.
    Int(i64),
    /// A boolean selection.
    Bool(bool),
    /// A spelling from a closed set.
    Text(String),
}

/// The host's selected logical configuration for one role instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionGrant {
    /// blake3 of the canonical Logical Resource Plan this grant was selected against. A grant is
    /// meaningless without the plan whose choice set it resolves.
    pub logical_resource_plan_hash: Hash,
    /// The scope under which the value was selected.
    pub scope: SelectionScope,
    /// Every dimension the plan declares, exactly once, and no other key.
    pub values: BTreeMap<String, GrantValue>,
}

impl ExecutionGrant {
    /// The grant that selects each dimension's **smallest admissible value**.
    ///
    /// A named authoring policy, not a default. Selecting a configuration is a run-authoring
    /// decision — nothing about a plan says which point in its parametric freedom a run should
    /// occupy — so this exists to be *chosen*, and its name says what it chooses.
    ///
    /// It is the floor of the plan's cost, soundly: every operator in the plan's expression grammar
    /// is monotone non-decreasing in its dimension arguments, so no other admissible selection costs
    /// less. That makes it the right choice for a harness whose subject is some other rule, and the
    /// wrong one for a run that means to train at a real batch size.
    ///
    /// An enumerated dimension takes its first spelling, which is well-defined because a plan's
    /// enum domains are validated as sorted and unique.
    ///
    /// # Errors
    /// [`PlanRefusal`] when the plan does not validate, when its digest cannot be computed, or when
    /// a dimension declares an empty enum domain (nothing is admissible, so there is no minimum).
    pub fn selecting_domain_minimum(plan: &LogicalResourcePlan) -> Result<Self, PlanRefusal> {
        plan.validate()?;
        let mut values = BTreeMap::new();
        for dimension in &plan.dimensions {
            let selected = match &dimension.domain {
                Domain::UintRange { lo, .. } => GrantValue::Uint(*lo),
                Domain::Enum(spellings) => {
                    let first = spellings.first().ok_or_else(|| {
                        PlanRefusal::Invalid(format!(
                            "dimension `{}` declares an empty enum domain, so it has no smallest \
                             admissible value",
                            dimension.name
                        ))
                    })?;
                    GrantValue::Text(first.clone())
                }
            };
            values.insert(dimension.name.clone(), selected);
        }
        Ok(Self {
            logical_resource_plan_hash: plan.plan_hash()?,
            scope: plan.selection_scope,
            values,
        })
    }

    /// The grant's canonical CBOR bytes — the exact bytes frozen in a signed role entry for
    /// [`SelectionScope::UniformRun`], and the exact bytes every participant consumes.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, PlanRefusal> {
        let value = Value::Map(vec![
            (
                Value::Text("logical_resource_plan_hash".into()),
                Value::Bytes(self.logical_resource_plan_hash.0.to_vec()),
            ),
            (
                Value::Text("schema".into()),
                Value::Integer(EXECUTION_GRANT_SCHEMA.into()),
            ),
            (
                Value::Text("scope".into()),
                Value::Text(self.scope.spelling().into()),
            ),
            (
                Value::Text("values".into()),
                Value::Map(
                    self.values
                        .iter()
                        .map(|(k, v)| {
                            (
                                Value::Text(k.clone()),
                                match v {
                                    GrantValue::Uint(n) => Value::Integer((*n).into()),
                                    GrantValue::Int(n) => Value::Integer((*n).into()),
                                    GrantValue::Bool(b) => Value::Bool(*b),
                                    GrantValue::Text(s) => Value::Text(s.clone()),
                                },
                            )
                        })
                        .collect(),
                ),
            ),
        ]);
        let bytes = to_canonical_vec(&value)
            .map_err(|e| PlanRefusal::Invalid(format!("grant encoding: {e}")))?;
        if bytes.len() > EXECUTION_GRANT_BYTES_MAX {
            return Err(PlanRefusal::ExceedsPolicy(format!(
                "{} grant bytes exceeds {EXECUTION_GRANT_BYTES_MAX}",
                bytes.len()
            )));
        }
        Ok(bytes)
    }

    /// blake3 of the grant's complete canonical bytes — the digest the admitted tuple records.
    pub fn grant_hash(&self) -> Result<Hash, PlanRefusal> {
        Ok(blake3_hash(&self.to_canonical_bytes()?))
    }

    /// Decode a grant from canonical bytes, enforcing the byte ceiling, the closed member set and
    /// canonicality. Floating-point values, duplicate keys, unknown top-level members and
    /// non-canonical encodings are refused.
    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, PlanRefusal> {
        if bytes.len() > EXECUTION_GRANT_BYTES_MAX {
            return Err(PlanRefusal::ExceedsPolicy(format!(
                "{} grant bytes exceeds {EXECUTION_GRANT_BYTES_MAX}",
                bytes.len()
            )));
        }
        let value: Value = ciborium::de::from_reader(bytes)
            .map_err(|e| PlanRefusal::Invalid(format!("grant is not well-formed CBOR: {e}")))?;
        let Value::Map(entries) = &value else {
            return Err(PlanRefusal::Invalid("execution-grant must be a map".into()));
        };
        let mut schema = None;
        let mut plan_hash = None;
        let mut scope = None;
        let mut values = None;
        for (k, v) in entries {
            let Value::Text(key) = k else {
                return Err(PlanRefusal::Invalid(
                    "execution-grant has a non-text member key".into(),
                ));
            };
            let slot = match key.as_str() {
                "schema" => {
                    schema = Some(v);
                    continue;
                }
                "logical_resource_plan_hash" => &mut plan_hash,
                "scope" => &mut scope,
                "values" => &mut values,
                other => {
                    return Err(PlanRefusal::Invalid(format!(
                        "execution-grant has the unknown member `{other}`"
                    )))
                }
            };
            if slot.is_some() {
                return Err(PlanRefusal::Invalid(format!(
                    "execution-grant has the duplicate member `{key}`"
                )));
            }
            *slot = Some(v);
        }

        let schema = schema.ok_or_else(|| PlanRefusal::Invalid("grant has no schema".into()))?;
        let Value::Integer(schema_int) = schema else {
            return Err(PlanRefusal::Invalid(
                "grant schema must be an integer".into(),
            ));
        };
        if i128::from(*schema_int) != i128::from(EXECUTION_GRANT_SCHEMA) {
            return Err(PlanRefusal::Invalid(format!(
                "unknown execution-grant schema (this build understands {EXECUTION_GRANT_SCHEMA})"
            )));
        }

        let Some(Value::Bytes(hash_bytes)) = plan_hash else {
            return Err(PlanRefusal::Invalid(
                "grant logical_resource_plan_hash must be a 32-byte string".into(),
            ));
        };
        let logical_resource_plan_hash =
            Hash(<[u8; 32]>::try_from(hash_bytes.as_slice()).map_err(|_| {
                PlanRefusal::Invalid("grant logical_resource_plan_hash must be 32 bytes".into())
            })?);

        let Some(Value::Text(scope_text)) = scope else {
            return Err(PlanRefusal::Invalid("grant scope must be text".into()));
        };
        let scope = SelectionScope::parse_spelling(scope_text)
            .ok_or_else(|| PlanRefusal::Invalid("unknown grant scope".into()))?;

        let Some(Value::Map(value_entries)) = values else {
            return Err(PlanRefusal::Invalid("grant values must be a map".into()));
        };
        let mut selected = BTreeMap::new();
        for (k, v) in value_entries {
            let Value::Text(key) = k else {
                return Err(PlanRefusal::Invalid(
                    "grant values has a non-text key".into(),
                ));
            };
            let parsed = match v {
                Value::Integer(n) => {
                    let raw = i128::from(*n);
                    if let Ok(u) = u64::try_from(raw) {
                        GrantValue::Uint(u)
                    } else {
                        GrantValue::Int(i64::try_from(raw).map_err(|_| {
                            PlanRefusal::Invalid(format!(
                                "grant value for `{key}` is outside the 64-bit integer range"
                            ))
                        })?)
                    }
                }
                Value::Bool(b) => GrantValue::Bool(*b),
                Value::Text(s) => GrantValue::Text(s.clone()),
                Value::Float(_) => {
                    return Err(PlanRefusal::Invalid(format!(
                        "grant value for `{key}` is floating point; the grammar admits integers, \
                         booleans and text only"
                    )))
                }
                _ => {
                    return Err(PlanRefusal::Invalid(format!(
                        "grant value for `{key}` is not an admissible scalar"
                    )))
                }
            };
            if selected.insert(key.clone(), parsed).is_some() {
                return Err(PlanRefusal::Invalid(format!(
                    "grant values has the duplicate key `{key}`"
                )));
            }
        }

        let grant = Self {
            logical_resource_plan_hash,
            scope,
            values: selected,
        };
        if grant.to_canonical_bytes()? != bytes {
            return Err(PlanRefusal::Invalid(
                "grant bytes are not canonical CBOR for their own content".into(),
            ));
        }
        Ok(grant)
    }

    /// The grant's values as a plan [`Binding`], **without** the plan-side check.
    ///
    /// For a consumer that holds the frozen grant but not (yet) the plan — a participant reading
    /// the signed role entry before its own module has been assessed. The values are converted,
    /// never validated: composition re-checks the binding against the plan it composes, so a
    /// value outside its dimension's domain still refuses, just at the step that holds the plan.
    ///
    /// # Errors
    /// [`PlanRefusal::Invalid`] when a value has a shape no schema-1 dimension domain can take.
    pub fn values_binding(&self) -> Result<Binding, PlanRefusal> {
        let mut binding = Binding::new();
        for (key, value) in &self.values {
            let selected = match value {
                GrantValue::Uint(n) => DimensionValue::Uint(*n),
                GrantValue::Text(s) => DimensionValue::Enum(s.clone()),
                GrantValue::Int(_) | GrantValue::Bool(_) => {
                    return Err(PlanRefusal::Invalid(format!(
                        "grant value for `{key}` has no matching dimension domain; schema-1 plan \
                         dimensions are uint ranges and enums"
                    )))
                }
            };
            binding.insert(key.clone(), selected);
        }
        Ok(binding)
    }

    /// Check the grant against the plan it names: the digest matches, the scope agrees with the
    /// plan's declared scope, every declared dimension is selected exactly once and no other key
    /// is present, and every value satisfies its dimension's declared type and bounds. Returns
    /// the [`Binding`] the planner and the footprint arithmetic consume.
    pub fn bind_to(&self, plan: &LogicalResourcePlan) -> Result<Binding, PlanRefusal> {
        let plan_hash = plan.plan_hash()?;
        if plan_hash != self.logical_resource_plan_hash {
            return Err(PlanRefusal::Invalid(
                "grant names a different Logical Resource Plan than the one supplied".into(),
            ));
        }
        // The envelope may narrow the plan's scope, never broaden it: a plan permitting
        // per-participant selection may be run uniform, but a uniform-run plan may not be
        // selected per participant.
        if plan.selection_scope == SelectionScope::UniformRun
            && self.scope == SelectionScope::PerParticipant
        {
            return Err(PlanRefusal::Invalid(
                "grant broadens the plan's uniform-run selection scope to per-participant \
                 ([RC-11])"
                    .into(),
            ));
        }

        let binding = self.values_binding()?;
        plan.check_binding(&binding)?;
        Ok(binding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource_plan::{Dimension, Domain, Expr, Lifetime, Retention, TensorDecl};

    fn plan() -> LogicalResourcePlan {
        LogicalResourcePlan {
            selection_scope: SelectionScope::UniformRun,
            equivalence_contract_hash: None,
            dimensions: vec![
                Dimension {
                    name: "micro_batch".into(),
                    domain: Domain::UintRange { lo: 1, hi: 8 },
                },
                Dimension {
                    name: "precision".into(),
                    domain: Domain::Enum(vec!["full".into(), "half".into()]),
                },
            ],
            tensors: vec![TensorDecl {
                name: "params".into(),
                shape: vec![Expr::Const(1024)],
                dtype: crate::resource_plan::Dtype::F32,
                layout: vec![],
                lifetime: Lifetime::Persistent(Retention::Run),
            }],
            operations: vec![],
            transfers: vec![],
            linear_memory: vec![],
            transient_live_sets: vec![],
            linear_fragmentation_headroom: Expr::Const(0),
        }
    }

    fn grant_for(plan: &LogicalResourcePlan, micro_batch: u64) -> ExecutionGrant {
        ExecutionGrant {
            logical_resource_plan_hash: plan.plan_hash().unwrap(),
            scope: SelectionScope::UniformRun,
            values: BTreeMap::from([
                ("micro_batch".to_string(), GrantValue::Uint(micro_batch)),
                (
                    "precision".to_string(),
                    GrantValue::Text("half".to_string()),
                ),
            ]),
        }
    }

    #[test]
    fn a_grant_round_trips_canonically_and_binds_to_its_plan() {
        let plan = plan();
        plan.validate().unwrap();
        let grant = grant_for(&plan, 4);
        let bytes = grant.to_canonical_bytes().unwrap();
        let decoded = ExecutionGrant::decode_canonical(&bytes).unwrap();
        assert_eq!(decoded, grant);
        assert_eq!(decoded.grant_hash().unwrap(), grant.grant_hash().unwrap());
        let binding = decoded.bind_to(&plan).unwrap();
        assert_eq!(binding.len(), 2);
        assert!(plan.footprint(&binding).is_ok());
    }

    #[test]
    fn a_grant_naming_a_different_plan_is_refused() {
        let plan = plan();
        let mut grant = grant_for(&plan, 4);
        grant.logical_resource_plan_hash = Hash([0xAB; 32]);
        assert!(grant
            .bind_to(&plan)
            .unwrap_err()
            .detail()
            .contains("different Logical Resource Plan"));
    }

    #[test]
    fn a_value_outside_its_declared_domain_is_refused() {
        let plan = plan();
        assert!(grant_for(&plan, 99)
            .bind_to(&plan)
            .unwrap_err()
            .detail()
            .contains("outside its declared range"));
    }

    #[test]
    fn every_declared_dimension_is_selected_exactly_once_and_no_other_key() {
        let plan = plan();
        let mut grant = grant_for(&plan, 4);
        grant.values.remove("precision");
        assert!(grant.bind_to(&plan).is_err());

        let mut grant = grant_for(&plan, 4);
        grant
            .values
            .insert("stowaway".to_string(), GrantValue::Uint(1));
        assert!(grant.bind_to(&plan).is_err());
    }

    #[test]
    fn a_grant_must_not_broaden_the_plans_selection_scope() {
        let plan = plan();
        let mut grant = grant_for(&plan, 4);
        grant.scope = SelectionScope::PerParticipant;
        assert!(grant
            .bind_to(&plan)
            .unwrap_err()
            .detail()
            .contains("broadens"));
    }

    #[test]
    fn floating_point_and_unknown_members_are_refused() {
        let plan = plan();
        let grant = grant_for(&plan, 4);
        let bytes = grant.to_canonical_bytes().unwrap();

        let Value::Map(mut entries) =
            ciborium::de::from_reader::<Value, _>(bytes.as_slice()).unwrap()
        else {
            unreachable!()
        };
        entries.push((Value::Text("extra".into()), Value::Integer(1.into())));
        let tampered = to_canonical_vec(&Value::Map(entries)).unwrap();
        assert!(ExecutionGrant::decode_canonical(&tampered)
            .unwrap_err()
            .detail()
            .contains("unknown member"));

        let float = to_canonical_vec(&Value::Map(vec![
            (
                Value::Text("schema".into()),
                Value::Integer(EXECUTION_GRANT_SCHEMA.into()),
            ),
            (
                Value::Text("logical_resource_plan_hash".into()),
                Value::Bytes(vec![0u8; 32]),
            ),
            (
                Value::Text("scope".into()),
                Value::Text("uniform-run".into()),
            ),
            (
                Value::Text("values".into()),
                Value::Map(vec![(Value::Text("micro_batch".into()), Value::Float(2.0))]),
            ),
        ]))
        .unwrap();
        assert!(ExecutionGrant::decode_canonical(&float)
            .unwrap_err()
            .detail()
            .contains("floating point"));
    }

    #[test]
    fn the_byte_ceiling_is_enforced_on_decode() {
        let oversized = vec![0u8; EXECUTION_GRANT_BYTES_MAX + 1];
        assert_eq!(
            ExecutionGrant::decode_canonical(&oversized)
                .unwrap_err()
                .slug(),
            "LogicalResourcePlanExceedsPolicy"
        );
    }
}
