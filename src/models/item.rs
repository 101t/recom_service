//! Items (the things being recommended) and user interactions with them.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use utoipa::ToSchema;

/// An item ready to be embedded and indexed.
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub id: String,
    pub text: String,
    pub payload: Map<String, Value>,
}

/// Body of `POST /api/v1/datasets/{id}/items`.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct UpsertItemsRequest {
    /// Items as flat JSON objects, using the same columns as file uploads.
    #[schema(value_type = Vec<Object>, example = json!([{"id": "42", "name": "Trail shoes", "description": "Grippy shoes for muddy trails", "category": "footwear", "price": 99.5}]))]
    pub items: Vec<Map<String, Value>>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct UpsertItemsResponse {
    pub dataset_id: String,
    /// Items embedded and written to the index.
    pub upserted: u64,
    /// Rows skipped because they had no id or no text.
    pub skipped: u64,
}

/// An indexed item.
#[derive(Debug, Serialize, ToSchema)]
pub struct ItemResponse {
    pub id: String,
    #[schema(value_type = Object)]
    pub payload: Map<String, Value>,
}

/// A user/item interaction (view, click, purchase, rating, ...).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Interaction {
    #[serde(deserialize_with = "string_or_number")]
    #[schema(example = "user-1")]
    pub user_id: String,
    #[serde(deserialize_with = "string_or_number")]
    #[schema(example = "42")]
    pub item_id: String,
    /// Strength of the signal (default 1.0). Negative values express dislike.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 1.0)]
    pub weight: Option<f32>,
    /// Free-form event type, kept for analytics (e.g. `view`, `purchase`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
    /// Prediction that led to this interaction, for attribution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prediction_id: Option<String>,
}

impl Interaction {
    pub fn weight(&self) -> f32 {
        self.weight.unwrap_or(1.0)
    }
}

/// Body of `POST /api/v1/datasets/{id}/interactions`.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct RecordInteractionsRequest {
    pub interactions: Vec<Interaction>,
}

#[derive(Debug, Default, Serialize, ToSchema)]
pub struct RecordInteractionsResponse {
    pub dataset_id: String,
    /// Interactions applied to user profiles.
    pub accepted: u64,
    /// Users whose profile was created or updated.
    pub users_updated: u64,
    /// Item ids that are not in the dataset (their interactions were stored but ignored).
    pub unknown_items: Vec<String>,
}

/// Accepts ids sent as JSON strings or integers.
pub fn string_or_number<'de, D: Deserializer<'de>>(de: D) -> Result<String, D::Error> {
    match Value::deserialize(de)? {
        Value::String(s) => Ok(s),
        Value::Number(n) => Ok(n.to_string()),
        other => Err(serde::de::Error::custom(format!(
            "expected a string or number id, got {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn interaction_ids_accept_numbers() {
        let i: Interaction = serde_json::from_value(json!({"user_id": 7, "item_id": "a"})).unwrap();
        assert_eq!(i.user_id, "7");
        assert_eq!(i.weight(), 1.0);
        assert!(
            serde_json::from_value::<Interaction>(json!({"user_id": null, "item_id": "a"}))
                .is_err()
        );
        assert!(
            serde_json::from_value::<Interaction>(json!({"user_id": "u", "item_id": "a", "x": 1}))
                .is_err()
        );
    }
}
