//! Domain types shared by the API, services and storage layers.

pub mod dataset;
pub mod filter;
pub mod item;
pub mod recommend;

pub use dataset::{
    CreateDatasetRequest, Dataset, DatasetList, DatasetStats, DatasetStatus, IngestResponse,
    TrainParams, TrainingProgress, TrainingStage, UploadParams,
};
pub use filter::{Filter, ValidFilter};
pub use item::{
    Interaction, Item, ItemResponse, RecordInteractionsRequest, RecordInteractionsResponse,
    UpsertItemsRequest, UpsertItemsResponse,
};
pub use recommend::{
    ExportFormat, PredictionLog, PredictionLogList, PredictionQuery, RecommendRequest,
    RecommendResponse, RecommendStrategy, Recommendation,
};
