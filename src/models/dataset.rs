//! Dataset metadata and the requests that create or configure datasets.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::{IntoParams, ToSchema};

/// Lifecycle of a dataset's index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DatasetStatus {
    /// Created, no data indexed yet.
    Pending,
    /// A training job (indexing items or building user profiles) is running.
    Training,
    /// Index is up to date.
    Ready,
    /// The last training job failed; see `error`. Previously indexed data stays queryable.
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum TrainingStage {
    Items,
    Users,
}

/// Progress of the running training job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct TrainingProgress {
    pub stage: TrainingStage,
    pub processed: u64,
    pub total: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct DatasetStats {
    /// Items currently indexed.
    pub items: u64,
    /// Users with a preference profile.
    pub users: u64,
    /// Interactions recorded.
    pub interactions: u64,
    /// Files and item batches stored for this dataset.
    pub uploads: u64,
}

/// A catalog of items (plus optional user interactions) that recommendations are served from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct Dataset {
    #[schema(example = "products")]
    pub id: String,
    #[schema(example = "Product catalog")]
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Column holding the item id. When absent, ids are derived from row content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "id")]
    pub id_column: Option<String>,
    /// Columns concatenated into the text that gets embedded. Empty until the first upload
    /// when auto-detection is used.
    #[serde(default)]
    #[schema(example = json!(["name", "description"]))]
    pub text_columns: Vec<String>,
    pub status: DatasetStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<TrainingProgress>,
    #[serde(default)]
    pub stats: DatasetStats,
    /// Union of all columns seen in uploads.
    #[serde(default)]
    pub columns: Vec<String>,
    /// Embedding model the dataset was last trained with.
    pub embedding_model: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trained_at: Option<DateTime<Utc>>,
    /// Last progress update of a running job (used to detect interrupted jobs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heartbeat_at: Option<DateTime<Utc>>,
}

/// Body of `POST /api/v1/datasets`.
#[derive(Debug, Clone, Default, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CreateDatasetRequest {
    /// Dataset id (`[A-Za-z0-9_-]`, max 64). Generated when omitted.
    #[schema(example = "products")]
    pub id: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    /// Column holding the item id (default: auto-detect `id`, `item_id`, `sku`, ...).
    pub id_column: Option<String>,
    /// Columns to embed (default: auto-detect `title`, `name`, `description`, `text`, ...).
    pub text_columns: Option<Vec<String>>,
}

/// Query parameters accepted by upload endpoints. Dataset options only apply when
/// the upload creates the dataset.
#[derive(Debug, Clone, Default, Deserialize, IntoParams)]
#[into_params(parameter_in = Query)]
#[serde(deny_unknown_fields)]
pub struct UploadParams {
    /// Block until training finishes and return the final dataset (default: false).
    pub wait: Option<bool>,
    /// Name for a dataset created by this upload.
    pub name: Option<String>,
    /// Item id column for a dataset created by this upload.
    pub id_column: Option<String>,
    /// Comma-separated columns to embed, for a dataset created by this upload.
    pub text_columns: Option<String>,
}

impl UploadParams {
    pub fn as_create_request(&self, id: &str) -> CreateDatasetRequest {
        CreateDatasetRequest {
            id: Some(id.to_string()),
            name: self.name.clone(),
            description: None,
            id_column: self.id_column.clone(),
            text_columns: self.text_columns.as_ref().map(|cols| {
                cols.split(',')
                    .map(|c| c.trim().to_string())
                    .filter(|c| !c.is_empty())
                    .collect()
            }),
        }
    }
}

/// Query parameters of `POST /api/v1/datasets/{id}/train`.
#[derive(Debug, Clone, Default, Deserialize, IntoParams)]
#[into_params(parameter_in = Query)]
#[serde(deny_unknown_fields)]
pub struct TrainParams {
    /// Block until training finishes (default: false).
    pub wait: Option<bool>,
    /// Drop the dataset's vectors before rebuilding (default: false, which keeps
    /// serving the old index while the new one is written).
    pub reset: Option<bool>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct DatasetList {
    pub datasets: Vec<Dataset>,
}

/// Response of upload and train endpoints.
#[derive(Debug, Serialize, ToSchema)]
pub struct IngestResponse {
    /// The dataset after the job was accepted, or after it finished when `wait=true`.
    pub dataset: Dataset,
    /// Rows read from the uploaded file (0 for retraining).
    pub rows: u64,
    /// Identifier of the stored upload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upload_id: Option<String>,
}
