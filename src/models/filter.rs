//! Payload filter DSL, modeled after Qdrant's filters.
//!
//! The wire format ([`Filter`]) is validated once into a typed form
//! ([`ValidFilter`]) that every vector store backend can translate or evaluate.
//!
//! ```json
//! {
//!   "must":     [{ "key": "category", "match": { "value": "footwear" } }],
//!   "should":   [{ "key": "brand", "match": { "any": ["acme", "globex"] } }],
//!   "must_not": [{ "key": "price", "range": { "gt": 200 } }],
//!   "min_should_count": 1
//! }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use utoipa::ToSchema;

use crate::error::AppError;

const MAX_CONDITIONS: usize = 64;
const MAX_KEY_LEN: usize = 256;

/// Payload filter. Conditions are combined as `must AND (should) AND NOT must_not`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Filter {
    /// All of these conditions must match (AND).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub must: Vec<Condition>,
    /// At least `min_should_count` (default 1) of these conditions must match (OR).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub should: Vec<Condition>,
    /// None of these conditions may match (NOT).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub must_not: Vec<Condition>,
    /// Minimum number of `should` conditions that must match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_should_count: Option<u64>,
}

/// A condition on one payload field. Exactly one of `match` or `range` must be set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Condition {
    /// Payload field; use dots for nested fields (e.g. `attributes.color`).
    #[schema(example = "category")]
    pub key: String,
    #[serde(rename = "match", default, skip_serializing_if = "Option::is_none")]
    pub r#match: Option<MatchCondition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<RangeCondition>,
}

/// Exact-value match. Exactly one of `value`, `any` or `except` must be set.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct MatchCondition {
    /// Field equals this string, integer or boolean.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = json!("footwear"))]
    pub value: Option<Value>,
    /// Field equals any of these strings or integers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub any: Option<Vec<Value>>,
    /// Field equals none of these strings or integers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub except: Option<Vec<Value>>,
}

/// Numeric range. At least one bound must be set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct RangeCondition {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gt: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gte: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lt: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lte: Option<f64>,
}

impl RangeCondition {
    pub fn contains(&self, x: f64) -> bool {
        self.gt.is_none_or(|b| x > b)
            && self.gte.is_none_or(|b| x >= b)
            && self.lt.is_none_or(|b| x < b)
            && self.lte.is_none_or(|b| x <= b)
    }
}

// ─── Validated form ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    Keyword(String),
    Integer(i64),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueSet {
    Keywords(Vec<String>),
    Integers(Vec<i64>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    Equals(Scalar),
    AnyOf(ValueSet),
    NoneOf(ValueSet),
    Range(RangeCondition),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldCondition {
    pub key: String,
    pub predicate: Predicate,
}

/// A filter whose conditions have been type-checked.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ValidFilter {
    pub must: Vec<FieldCondition>,
    pub should: Vec<FieldCondition>,
    pub must_not: Vec<FieldCondition>,
    /// When set, `should` is evaluated as "at least N must match".
    pub min_should: Option<u64>,
}

impl Filter {
    pub fn validate(&self) -> Result<ValidFilter, AppError> {
        let total = self.must.len() + self.should.len() + self.must_not.len();
        if total > MAX_CONDITIONS {
            return Err(AppError::bad_request(format!(
                "filter has {total} conditions; the maximum is {MAX_CONDITIONS}"
            )));
        }
        if let Some(n) = self.min_should_count
            && (n == 0 || n as usize > self.should.len())
        {
            return Err(AppError::bad_request(format!(
                "min_should_count must be between 1 and the number of `should` conditions ({})",
                self.should.len()
            )));
        }
        let convert = |conds: &[Condition]| {
            conds
                .iter()
                .map(Condition::validate)
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(ValidFilter {
            must: convert(&self.must)?,
            should: convert(&self.should)?,
            must_not: convert(&self.must_not)?,
            min_should: self.min_should_count,
        })
    }
}

impl Condition {
    fn validate(&self) -> Result<FieldCondition, AppError> {
        let key = self.key.trim();
        if key.is_empty() || key.len() > MAX_KEY_LEN {
            return Err(AppError::bad_request(
                "filter condition key must be 1-256 characters",
            ));
        }
        let predicate = match (&self.r#match, &self.range) {
            (Some(m), None) => m.validate(key)?,
            (None, Some(r)) => {
                let bounds = [r.gt, r.gte, r.lt, r.lte];
                if bounds.iter().all(Option::is_none) {
                    return Err(AppError::bad_request(format!(
                        "range condition on '{key}' needs at least one bound"
                    )));
                }
                if bounds.iter().flatten().any(|b| !b.is_finite()) {
                    return Err(AppError::bad_request(format!(
                        "range bounds on '{key}' must be finite numbers"
                    )));
                }
                Predicate::Range(*r)
            }
            _ => {
                return Err(AppError::bad_request(format!(
                    "condition on '{key}' must have exactly one of `match` or `range`"
                )));
            }
        };
        Ok(FieldCondition {
            key: key.to_string(),
            predicate,
        })
    }
}

impl MatchCondition {
    fn validate(&self, key: &str) -> Result<Predicate, AppError> {
        match (&self.value, &self.any, &self.except) {
            (Some(v), None, None) => Ok(Predicate::Equals(scalar(key, v)?)),
            (None, Some(vs), None) => Ok(Predicate::AnyOf(value_set(key, vs)?)),
            (None, None, Some(vs)) => Ok(Predicate::NoneOf(value_set(key, vs)?)),
            _ => Err(AppError::bad_request(format!(
                "match on '{key}' must have exactly one of `value`, `any` or `except`"
            ))),
        }
    }
}

fn scalar(key: &str, v: &Value) -> Result<Scalar, AppError> {
    match v {
        Value::String(s) => Ok(Scalar::Keyword(s.clone())),
        Value::Bool(b) => Ok(Scalar::Bool(*b)),
        Value::Number(n) => n.as_i64().map(Scalar::Integer).ok_or_else(|| {
            AppError::bad_request(format!(
                "match on '{key}' only supports integers; use a `range` condition for decimal numbers"
            ))
        }),
        _ => Err(AppError::bad_request(format!(
            "match value on '{key}' must be a string, integer or boolean"
        ))),
    }
}

fn value_set(key: &str, values: &[Value]) -> Result<ValueSet, AppError> {
    if values.is_empty() {
        return Err(AppError::bad_request(format!(
            "match list on '{key}' must not be empty"
        )));
    }
    if let Some(strings) = values
        .iter()
        .map(|v| v.as_str().map(str::to_string))
        .collect::<Option<Vec<_>>>()
    {
        return Ok(ValueSet::Keywords(strings));
    }
    if let Some(ints) = values.iter().map(Value::as_i64).collect::<Option<Vec<_>>>() {
        return Ok(ValueSet::Integers(ints));
    }
    Err(AppError::bad_request(format!(
        "match list on '{key}' must contain only strings or only integers"
    )))
}

// ─── In-process evaluation (used by the memory store) ───────────────────────

impl ValidFilter {
    pub fn is_empty(&self) -> bool {
        self.must.is_empty() && self.should.is_empty() && self.must_not.is_empty()
    }

    pub fn matches(&self, payload: &Map<String, Value>) -> bool {
        if !self.must.iter().all(|c| c.matches(payload)) {
            return false;
        }
        if self.must_not.iter().any(|c| c.matches(payload)) {
            return false;
        }
        if self.should.is_empty() {
            return true;
        }
        let hits = self.should.iter().filter(|c| c.matches(payload)).count() as u64;
        hits >= self.min_should.unwrap_or(1)
    }
}

impl FieldCondition {
    pub fn matches(&self, payload: &Map<String, Value>) -> bool {
        let values = lookup(payload, &self.key);
        match &self.predicate {
            Predicate::Equals(s) => values.iter().any(|v| scalar_eq(s, v)),
            Predicate::AnyOf(set) => values.iter().any(|v| set_contains(set, v)),
            Predicate::NoneOf(set) => {
                !values.is_empty() && values.iter().all(|v| !set_contains(set, v))
            }
            Predicate::Range(r) => values
                .iter()
                .filter_map(|v| v.as_f64())
                .any(|x| r.contains(x)),
        }
    }
}

/// Resolves a dotted path, flattening arrays (Qdrant semantics: an array field
/// matches when any of its elements matches).
fn lookup<'a>(payload: &'a Map<String, Value>, key: &str) -> Vec<&'a Value> {
    let mut current: Vec<&Value> = Vec::new();
    for (i, part) in key.split('.').enumerate() {
        let next: Vec<&Value> = if i == 0 {
            payload.get(part).into_iter().collect()
        } else {
            current
                .iter()
                .filter_map(|v| v.as_object().and_then(|o| o.get(part)))
                .collect()
        };
        current = next
            .into_iter()
            .flat_map(|v| match v {
                Value::Array(items) => items.iter().collect::<Vec<_>>(),
                other => vec![other],
            })
            .collect();
    }
    current
}

fn scalar_eq(s: &Scalar, v: &Value) -> bool {
    match (s, v) {
        (Scalar::Keyword(a), Value::String(b)) => a == b,
        (Scalar::Integer(a), Value::Number(b)) => b.as_i64() == Some(*a),
        (Scalar::Bool(a), Value::Bool(b)) => a == b,
        _ => false,
    }
}

fn set_contains(set: &ValueSet, v: &Value) -> bool {
    match (set, v) {
        (ValueSet::Keywords(ks), Value::String(s)) => ks.iter().any(|k| k == s),
        (ValueSet::Integers(is), Value::Number(n)) => n.as_i64().is_some_and(|x| is.contains(&x)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn filter(v: Value) -> Result<ValidFilter, AppError> {
        serde_json::from_value::<Filter>(v)
            .map_err(|e| AppError::bad_request(e.to_string()))?
            .validate()
    }

    fn payload(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn parses_and_validates_full_filter() {
        let f = filter(json!({
            "must": [{"key": "category", "match": {"value": "footwear"}}],
            "should": [
                {"key": "price", "range": {"gte": 10.0, "lte": 100.0}},
                {"key": "tags", "match": {"any": ["sale", "new"]}}
            ],
            "must_not": [{"key": "in_stock", "match": {"value": false}}],
            "min_should_count": 1
        }))
        .unwrap();
        assert_eq!(
            f.must[0].predicate,
            Predicate::Equals(Scalar::Keyword("footwear".into()))
        );
        assert_eq!(f.should.len(), 2);
        assert_eq!(f.min_should, Some(1));
    }

    #[test]
    fn rejects_invalid_conditions() {
        let bad = [
            json!({"must": [{"key": "x"}]}),
            json!({"must": [{"key": "x", "match": {"value": 1}, "range": {"gt": 1}}]}),
            json!({"must": [{"key": "x", "match": {"value": 1.5}}]}),
            json!({"must": [{"key": "x", "match": {"value": {"a": 1}}}]}),
            json!({"must": [{"key": "x", "match": {"any": []}}]}),
            json!({"must": [{"key": "x", "match": {"any": ["a", 1]}}]}),
            json!({"must": [{"key": "x", "range": {}}]}),
            json!({"must": [{"key": "", "match": {"value": "a"}}]}),
            json!({"should": [{"key": "x", "match": {"value": "a"}}], "min_should_count": 2}),
            json!({"must": [{"key": "x", "match": {"value": "a"}}], "unknown": true}),
        ];
        for b in bad {
            assert!(filter(b.clone()).is_err(), "{b}");
        }
    }

    #[test]
    fn evaluates_against_payloads() {
        let f = filter(json!({
            "must": [{"key": "category", "match": {"value": "footwear"}}],
            "must_not": [{"key": "price", "range": {"gt": 150}}]
        }))
        .unwrap();
        assert!(f.matches(&payload(json!({"category": "footwear", "price": 129.99}))));
        assert!(!f.matches(&payload(json!({"category": "footwear", "price": 189.99}))));
        assert!(!f.matches(&payload(json!({"category": "electronics", "price": 10}))));
        assert!(!f.matches(&payload(json!({"price": 10}))));
    }

    #[test]
    fn should_and_min_should_semantics() {
        let any_one = filter(json!({"should": [
            {"key": "a", "match": {"value": 1}},
            {"key": "b", "match": {"value": 2}}
        ]}))
        .unwrap();
        assert!(any_one.matches(&payload(json!({"a": 1}))));
        assert!(!any_one.matches(&payload(json!({"a": 3}))));

        let both = filter(json!({"should": [
            {"key": "a", "match": {"value": 1}},
            {"key": "b", "match": {"value": 2}}
        ], "min_should_count": 2}))
        .unwrap();
        assert!(!both.matches(&payload(json!({"a": 1}))));
        assert!(both.matches(&payload(json!({"a": 1, "b": 2}))));
    }

    #[test]
    fn nested_keys_and_arrays() {
        let f = filter(json!({"must": [
            {"key": "attrs.color", "match": {"any": ["red", "blue"]}},
            {"key": "tags", "match": {"value": "sale"}}
        ]}))
        .unwrap();
        assert!(f.matches(&payload(
            json!({"attrs": {"color": "blue"}, "tags": ["new", "sale"]})
        )));
        assert!(!f.matches(&payload(
            json!({"attrs": {"color": "green"}, "tags": ["sale"]})
        )));

        let except =
            filter(json!({"must": [{"key": "brand", "match": {"except": ["acme"]}}]})).unwrap();
        assert!(except.matches(&payload(json!({"brand": "globex"}))));
        assert!(!except.matches(&payload(json!({"brand": "acme"}))));
    }
}
