//! Vector storage backends.
//!
//! [`VectorStore`] is the only interface the services use. Two implementations ship
//! with the service:
//!
//! * [`qdrant::QdrantStore`] — the default, production backend.
//! * [`memory::MemoryStore`] — in-process brute force search for tests and local development.
//!
//! Items of all datasets share one collection and user profiles share another;
//! datasets are isolated by a tenant-indexed `_dataset` payload field (Qdrant's
//! recommended multitenancy layout), so thousands of datasets stay cheap.

pub mod memory;
pub mod mmr;
pub mod qdrant;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::{Map, Value};
use utoipa::ToSchema;

use crate::error::AppError;
use crate::models::{RecommendStrategy, ValidFilter};

/// Payload keys reserved by the service.
pub const DATASET_KEY: &str = "_dataset";
pub const ITEM_ID_KEY: &str = "_item_id";
pub const USER_ID_KEY: &str = "_user_id";

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Unavailable(String),
    #[error("{0}")]
    Backend(String),
}

impl From<StoreError> for AppError {
    fn from(err: StoreError) -> Self {
        match err {
            StoreError::NotFound(m) => AppError::NotFound(m),
            StoreError::InvalidInput(m) => AppError::BadRequest(m),
            StoreError::Unavailable(m) => AppError::Unavailable(m),
            StoreError::Backend(m) => AppError::VectorStore(m),
        }
    }
}

pub type StoreResult<T> = Result<T, StoreError>;

/// An item to index.
#[derive(Debug, Clone)]
pub struct ItemPoint {
    pub item_id: String,
    pub vector: Vec<f32>,
    pub payload: Map<String, Value>,
}

/// An item read back from the store. Reserved keys are stripped from `payload`.
#[derive(Debug, Clone)]
pub struct StoredItem {
    pub item_id: String,
    pub payload: Map<String, Value>,
    pub vector: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct ScoredItem {
    pub item_id: String,
    pub score: f32,
    pub payload: Option<Map<String, Value>>,
}

/// A positive or negative example for a recommendation query.
#[derive(Debug, Clone)]
pub enum Example {
    /// An indexed item of the same dataset.
    Item(String),
    /// A raw vector (e.g. an embedded text query or a user profile).
    Vector(Vec<f32>),
}

#[derive(Debug, Clone)]
pub struct ItemQuery {
    pub positive: Vec<Example>,
    pub negative: Vec<Example>,
    pub strategy: RecommendStrategy,
    pub filter: Option<ValidFilter>,
    /// Never return these items (example items are always excluded too).
    pub exclude_item_ids: Vec<String>,
    pub limit: usize,
    pub offset: usize,
    pub score_threshold: Option<f32>,
    /// MMR diversity in `[0, 1]`.
    pub diversity: Option<f32>,
    pub with_payload: bool,
}

impl ItemQuery {
    /// Ids that must not appear in results: explicit exclusions plus example items.
    /// Sorted and deduplicated.
    pub fn excluded_ids(&self) -> Vec<String> {
        let mut ids = self.exclude_item_ids.clone();
        for example in self.positive.iter().chain(&self.negative) {
            if let Example::Item(id) = example {
                ids.push(id.clone());
            }
        }
        ids.sort();
        ids.dedup();
        ids
    }

    /// True for the common single-signal case, served by a plain nearest-neighbour search.
    pub fn is_simple(&self) -> bool {
        self.positive.len() == 1 && self.negative.is_empty()
    }

    /// How many candidates to fetch before MMR re-ranking.
    pub fn mmr_candidates(&self) -> usize {
        ((self.offset + self.limit) * 4).clamp(50, 500)
    }
}

/// A user's preference profile: the weighted sum of the vectors of items they
/// interacted with. Recommending for a user is a nearest-neighbour search around it.
#[derive(Debug, Clone, PartialEq)]
pub struct UserProfile {
    pub user_id: String,
    pub vector: Vec<f32>,
    pub interactions: u64,
    /// Most recent distinct items, oldest first.
    pub recent_items: Vec<String>,
    pub updated_at: DateTime<Utc>,
}

impl UserProfile {
    pub fn new(user_id: impl Into<String>, dimension: usize) -> Self {
        Self {
            user_id: user_id.into(),
            vector: vec![0.0; dimension],
            interactions: 0,
            recent_items: Vec::new(),
            updated_at: Utc::now(),
        }
    }

    /// True when the profile carries no usable direction (no or cancelling interactions).
    pub fn is_empty(&self) -> bool {
        norm(&self.vector) < 1e-6
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct CollectionStats {
    pub name: String,
    pub points: u64,
    pub status: String,
}

#[async_trait]
pub trait VectorStore: Send + Sync + 'static {
    /// Backend name, e.g. `qdrant`.
    fn kind(&self) -> &'static str;

    /// Creates (or validates) the collections for vectors of `dimension`.
    async fn initialize(&self, dimension: usize) -> StoreResult<()>;

    /// Fails when the backend is not reachable.
    async fn health(&self) -> StoreResult<()>;

    async fn upsert_items(&self, dataset_id: &str, items: Vec<ItemPoint>) -> StoreResult<()>;

    async fn delete_items(&self, dataset_id: &str, item_ids: &[String]) -> StoreResult<()>;

    /// Returns the items that exist, in no particular order.
    async fn get_items(
        &self,
        dataset_id: &str,
        item_ids: &[String],
        with_vectors: bool,
    ) -> StoreResult<Vec<StoredItem>>;

    async fn count_items(&self, dataset_id: &str) -> StoreResult<u64>;

    async fn query_items(&self, dataset_id: &str, query: ItemQuery)
    -> StoreResult<Vec<ScoredItem>>;

    async fn upsert_users(&self, dataset_id: &str, profiles: Vec<UserProfile>) -> StoreResult<()>;

    /// Returns the profiles that exist, in no particular order.
    async fn get_users(
        &self,
        dataset_id: &str,
        user_ids: &[String],
    ) -> StoreResult<Vec<UserProfile>>;

    async fn count_users(&self, dataset_id: &str) -> StoreResult<u64>;

    /// Removes every item and user profile of a dataset.
    async fn delete_dataset(&self, dataset_id: &str) -> StoreResult<()>;

    async fn stats(&self) -> StoreResult<Vec<CollectionStats>>;
}

/// Removes reserved keys from a stored payload.
pub fn strip_reserved(mut payload: Map<String, Value>) -> Map<String, Value> {
    payload.remove(DATASET_KEY);
    payload.remove(ITEM_ID_KEY);
    payload
}

pub fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub fn normalized(v: &[f32]) -> Vec<f32> {
    let n = norm(v);
    if n > 0.0 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
