//! Recommendation ("predict") request and response.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use utoipa::{IntoParams, ToSchema};

use super::filter::Filter;

/// How multiple positive/negative examples are combined.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum RecommendStrategy {
    /// Search around the average of the examples (fast, good default).
    #[default]
    AverageVector,
    /// Score each candidate by its best match among the examples (handles diverse examples).
    BestScore,
    /// Sum of similarities to positives minus similarities to negatives.
    SumScores,
}

/// Request body of `POST /api/v1/datasets/{id}/recommend`.
///
/// Combine any of `query`, `item_ids` and `user_id`; at least one must be given.
#[derive(Debug, Clone, Default, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct RecommendRequest {
    /// Free text in any language supported by the embedding model.
    #[schema(example = "comfortable shoes for running")]
    pub query: Option<String>,
    /// Recommend items similar to these ("more like this").
    #[serde(default)]
    pub item_ids: Vec<String>,
    /// Steer results away from these items.
    #[serde(default)]
    pub negative_item_ids: Vec<String>,
    /// Personalize for this user, based on their recorded interactions.
    pub user_id: Option<String>,
    pub filter: Option<Filter>,
    /// Number of results (default 10).
    #[schema(example = 5)]
    pub limit: Option<usize>,
    /// Number of results to skip, for pagination.
    pub offset: Option<usize>,
    /// Only return results scoring at least this much.
    pub score_threshold: Option<f32>,
    /// Trade relevance for variety: 0 = most relevant, 1 = most diverse (MMR re-ranking).
    pub diversity: Option<f32>,
    pub strategy: Option<RecommendStrategy>,
    /// Exclude items the user already interacted with (default true).
    pub exclude_seen: Option<bool>,
    /// Include item payloads in the response (default true).
    pub with_payload: Option<bool>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct Recommendation {
    /// Item id.
    pub id: String,
    /// Similarity score (higher is better).
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<Object>)]
    pub payload: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct RecommendResponse {
    /// Id of this prediction; send it back with interactions for attribution.
    pub prediction_id: String,
    pub dataset_id: String,
    pub recommendations: Vec<Recommendation>,
    pub latency_ms: u64,
}

/// A logged prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct PredictionLog {
    pub prediction_id: String,
    pub timestamp: DateTime<Utc>,
    pub dataset_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub item_ids: Vec<String>,
    pub result_ids: Vec<String>,
    pub scores: Vec<f32>,
    pub latency_ms: u64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct PredictionLogList {
    pub logs: Vec<PredictionLog>,
    pub total: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    #[default]
    Jsonl,
    Csv,
    Parquet,
}

/// Query parameters for listing and exporting prediction logs.
#[derive(Debug, Clone, Default, Deserialize, IntoParams)]
#[into_params(parameter_in = Query)]
#[serde(deny_unknown_fields)]
pub struct PredictionQuery {
    /// Only this day (`YYYY-MM-DD`, UTC).
    pub date: Option<String>,
    /// Only this dataset.
    pub dataset_id: Option<String>,
    /// Maximum number of logs (default 50 for listing, 100000 for export).
    pub limit: Option<usize>,
    /// Export format: `jsonl` (default), `csv` or `parquet`.
    pub format: Option<ExportFormat>,
}
