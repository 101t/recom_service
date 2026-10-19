//! Shared application state: the wired-up components every handler uses.

use std::sync::Arc;

use anyhow::Context;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::config::{Settings, VectorStoreKind};
use crate::embedding::{self, Embedder};
use crate::services::{DatasetService, PredictionLogger, Recommender};
use crate::storage::DatasetRepository;
use crate::vector_store::VectorStore;
use crate::vector_store::memory::MemoryStore;
use crate::vector_store::qdrant::QdrantStore;

#[derive(Clone)]
pub struct AppState {
    pub settings: Arc<Settings>,
    pub embedder: Arc<dyn Embedder>,
    pub store: Arc<dyn VectorStore>,
    pub datasets: Arc<DatasetService>,
    pub recommender: Arc<Recommender>,
    pub predictions: Arc<PredictionLogger>,
    pub started_at: DateTime<Utc>,
}

impl AppState {
    /// Builds every component from configuration: loads the embedding model,
    /// connects to the vector store and prepares its collections.
    pub async fn initialize(settings: Settings) -> anyhow::Result<Self> {
        tokio::fs::create_dir_all(&settings.data_dir)
            .await
            .with_context(|| format!("cannot create DATA_DIR {}", settings.data_dir.display()))?;

        let embedder = embedding::build(&settings.embedding).await?;
        let store: Arc<dyn VectorStore> = match settings.vector_store.kind {
            VectorStoreKind::Qdrant => {
                Arc::new(QdrantStore::connect(&settings.vector_store.qdrant).await?)
            }
            VectorStoreKind::Memory => {
                tracing::warn!(
                    "using the in-memory vector store: data is lost on restart (development only)"
                );
                Arc::new(MemoryStore::new())
            }
        };
        store.initialize(embedder.dimension()).await?;
        Self::with_components(Arc::new(settings), embedder, store).await
    }

    /// Assembles the state from already-built components (also used by tests).
    pub async fn with_components(
        settings: Arc<Settings>,
        embedder: Arc<dyn Embedder>,
        store: Arc<dyn VectorStore>,
    ) -> anyhow::Result<Self> {
        let instance = instance_id();
        let repo = Arc::new(DatasetRepository::new(&settings.data_dir, &instance));
        repo.init().await?;
        let predictions = Arc::new(PredictionLogger::start(
            &settings.data_dir,
            &instance,
            settings.prediction_log_enabled,
        ));
        let datasets = Arc::new(DatasetService::new(
            Arc::clone(&repo),
            Arc::clone(&store),
            Arc::clone(&embedder),
            settings.training.clone(),
        ));
        datasets.recover_interrupted_jobs().await?;
        let recommender = Arc::new(Recommender::new(
            repo,
            Arc::clone(&store),
            Arc::clone(&embedder),
            Arc::clone(&predictions),
            settings.recommend.clone(),
        ));
        Ok(Self {
            settings,
            embedder,
            store,
            datasets,
            recommender,
            predictions,
            started_at: Utc::now(),
        })
    }

    /// Flushes buffered data before the process exits.
    pub async fn shutdown(&self) {
        self.predictions.shutdown().await;
    }
}

/// Identifies this replica in file names (container hostname when available).
fn instance_id() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::fs::read_to_string("/etc/hostname").ok())
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| Uuid::new_v4().simple().to_string()[..8].to_string())
}
