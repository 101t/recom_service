//! Dataset lifecycle and background training jobs.
//!
//! Item ingestion lives in `indexing.rs`, user profiles in `profiles.rs`; they
//! extend [`DatasetService`] with further `impl` blocks.

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::Utc;
use polars::prelude::DataFrame;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::config::TrainingSettings;
use crate::embedding::Embedder;
use crate::error::{AppError, AppResult};
use crate::ids::validate_dataset_id;
use crate::models::{
    CreateDatasetRequest, Dataset, DatasetStats, DatasetStatus, TrainingProgress, TrainingStage,
};
use crate::storage::DatasetRepository;
use crate::storage::tabular::{self, FileFormat};
use crate::vector_store::VectorStore;

/// Number of lock stripes serializing concurrent updates of the same user profile.
const USER_LOCK_STRIPES: usize = 64;
const MAX_NAME_LEN: usize = 200;
const INTERRUPTED: &str = "training was interrupted (the service restarted); \
                           call POST /api/v1/datasets/{id}/train to rebuild the index";

/// A file received by an upload endpoint, stored in a temporary location.
#[derive(Debug)]
pub struct Upload {
    pub path: PathBuf,
    pub filename: String,
    pub format: FileFormat,
}

impl Upload {
    /// Parses the file and removes the temporary copy.
    pub(crate) async fn read(self) -> AppResult<DataFrame> {
        let path = self.path.clone();
        let format = self.format;
        let result = tokio::task::spawn_blocking(move || tabular::read_file(&path, format)).await;
        self.discard().await;
        result?
    }

    /// Removes the temporary copy without reading it.
    pub(crate) async fn discard(&self) {
        if let Err(e) = tokio::fs::remove_file(&self.path).await {
            tracing::warn!(file = %self.path.display(), error = %e, "could not remove temporary upload");
        }
    }
}

/// A job accepted for background processing.
pub struct Accepted {
    pub dataset: Dataset,
    pub rows: u64,
    pub upload_id: Option<String>,
    pub job: JoinHandle<()>,
}

pub struct DatasetService {
    pub(crate) repo: Arc<DatasetRepository>,
    pub(crate) store: Arc<dyn VectorStore>,
    pub(crate) embedder: Arc<dyn Embedder>,
    pub(crate) settings: TrainingSettings,
    job_slots: Arc<Semaphore>,
    job_locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    pub(crate) user_stripes: Vec<tokio::sync::Mutex<()>>,
    /// Runtime that runs background jobs (the multi-threaded main runtime, not
    /// the single-threaded HTTP workers).
    runtime: tokio::runtime::Handle,
}

/// Progress of a running job, persisted to the dataset at most once per second.
pub(crate) struct Progress {
    pub stage: TrainingStage,
    pub processed: u64,
    pub total: u64,
    last_report: Instant,
}

impl Progress {
    pub fn new(stage: TrainingStage, total: u64) -> Self {
        Self {
            stage,
            processed: 0,
            total,
            last_report: Instant::now(),
        }
    }
}

impl DatasetService {
    /// Must be called from within a Tokio runtime; background jobs run on it.
    pub fn new(
        repo: Arc<DatasetRepository>,
        store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedder>,
        settings: TrainingSettings,
    ) -> Self {
        Self {
            repo,
            store,
            embedder,
            job_slots: Arc::new(Semaphore::new(settings.max_concurrent_jobs)),
            settings,
            job_locks: Mutex::default(),
            user_stripes: (0..USER_LOCK_STRIPES)
                .map(|_| tokio::sync::Mutex::new(()))
                .collect(),
            runtime: tokio::runtime::Handle::current(),
        }
    }

    pub async fn create(&self, req: CreateDatasetRequest) -> AppResult<Dataset> {
        let id = req
            .id
            .unwrap_or_else(|| Uuid::new_v4().simple().to_string());
        validate_dataset_id(&id)?;
        let name = req
            .name
            .map(|n| n.trim().to_string())
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| id.clone());
        if name.chars().count() > MAX_NAME_LEN {
            return Err(AppError::bad_request(format!(
                "name must be at most {MAX_NAME_LEN} characters"
            )));
        }
        let id_column = req
            .id_column
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty());
        let text_columns: Vec<String> = req
            .text_columns
            .unwrap_or_default()
            .into_iter()
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .collect();
        let now = Utc::now();
        let dataset = Dataset {
            id,
            name,
            description: req.description.filter(|d| !d.trim().is_empty()),
            id_column,
            text_columns,
            status: DatasetStatus::Pending,
            error: None,
            progress: None,
            stats: DatasetStats::default(),
            columns: Vec::new(),
            embedding_model: self.embedder.model().to_string(),
            created_at: now,
            updated_at: now,
            trained_at: None,
            heartbeat_at: None,
        };
        self.repo.create(&dataset).await?;
        tracing::info!(dataset = %dataset.id, "dataset created");
        Ok(dataset)
    }

    /// Returns the dataset, creating it with `req` when it does not exist yet.
    pub async fn get_or_create(&self, id: &str, req: CreateDatasetRequest) -> AppResult<Dataset> {
        match self.repo.get(id).await {
            Ok(dataset) => Ok(dataset),
            Err(AppError::NotFound(_)) => {
                match self
                    .create(CreateDatasetRequest {
                        id: Some(id.to_string()),
                        ..req
                    })
                    .await
                {
                    Err(AppError::Conflict(_)) => self.repo.get(id).await,
                    other => other,
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Returns the dataset if it exists.
    pub(crate) async fn find(&self, id: &str) -> AppResult<Option<Dataset>> {
        match self.repo.get(id).await {
            Ok(dataset) => Ok(Some(dataset)),
            Err(AppError::NotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub async fn get(&self, id: &str) -> AppResult<Dataset> {
        self.repo.get(id).await
    }

    /// Directory for in-flight uploads (same volume as the dataset files).
    pub async fn uploads_dir(&self) -> AppResult<PathBuf> {
        self.repo.uploads_dir().await
    }

    pub async fn list(&self) -> AppResult<Vec<Dataset>> {
        self.repo.list().await
    }

    /// Deletes the dataset's vectors, profiles and stored data.
    pub async fn delete(&self, id: &str) -> AppResult<()> {
        self.repo.get(id).await?;
        self.store.delete_dataset(id).await?;
        self.repo.delete(id).await?;
        tracing::info!(dataset = %id, "dataset deleted");
        Ok(())
    }

    /// Marks datasets whose training job died with the previous process as failed.
    pub async fn recover_interrupted_jobs(&self) -> AppResult<()> {
        let stale = chrono::Duration::from_std(self.settings.stale_job_after)
            .unwrap_or(chrono::Duration::minutes(10));
        for dataset in self.repo.list().await? {
            let is_stale = dataset.heartbeat_at.is_none_or(|t| Utc::now() - t > stale);
            if dataset.status == DatasetStatus::Training && is_stale {
                tracing::warn!(dataset = %dataset.id, "marking interrupted training job as failed");
                self.repo
                    .update(&dataset.id, |d| {
                        d.status = DatasetStatus::Failed;
                        d.error = Some(INTERRUPTED.replace("{id}", &d.id));
                        d.progress = None;
                        d.heartbeat_at = None;
                    })
                    .await?;
            }
        }
        Ok(())
    }

    /// Marks the dataset as training (visible immediately to pollers).
    pub(crate) async fn mark_training(
        &self,
        id: &str,
        progress: Option<TrainingProgress>,
    ) -> AppResult<Dataset> {
        self.repo
            .update(id, |d| {
                d.status = DatasetStatus::Training;
                d.error = None;
                d.progress = progress;
                d.heartbeat_at = Some(Utc::now());
            })
            .await
    }

    /// Persists job progress (throttled to once per second unless `force`).
    pub(crate) async fn report(
        &self,
        id: &str,
        progress: &mut Progress,
        force: bool,
    ) -> AppResult<()> {
        if !force && progress.last_report.elapsed() < Duration::from_secs(1) {
            return Ok(());
        }
        progress.last_report = Instant::now();
        let snapshot = TrainingProgress {
            stage: progress.stage,
            processed: progress.processed,
            total: progress.total,
        };
        self.repo
            .update(id, |d| {
                d.progress = Some(snapshot);
                d.heartbeat_at = Some(Utc::now());
            })
            .await
            .map(|_| ())
    }

    fn job_lock(&self, id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.job_locks.lock().unwrap_or_else(|e| e.into_inner());
        locks.entry(id.to_string()).or_default().clone()
    }

    /// Runs `job` in the background. Jobs of one dataset run one at a time and at
    /// most `TRAINING_MAX_CONCURRENT_JOBS` run in total. The dataset's status,
    /// counters and error are updated when the job ends.
    pub(crate) fn spawn_job<F, Fut>(self: &Arc<Self>, id: &str, job: F) -> JoinHandle<()>
    where
        F: FnOnce(Arc<Self>) -> Fut + Send + 'static,
        Fut: Future<Output = AppResult<()>> + Send + 'static,
    {
        let this = Arc::clone(self);
        let id = id.to_string();
        self.runtime.spawn(async move {
            let lock = this.job_lock(&id);
            let _serial = lock.lock().await;
            let _slot = Arc::clone(&this.job_slots)
                .acquire_owned()
                .await
                .expect("job semaphore is never closed");
            let started = Instant::now();
            let result = match this
                .repo
                .update(&id, |d| d.heartbeat_at = Some(Utc::now()))
                .await
            {
                Ok(_) => job(Arc::clone(&this)).await,
                Err(e) => Err(e),
            };
            this.finish_job(&id, result, started).await;
        })
    }

    async fn finish_job(&self, id: &str, result: AppResult<()>, started: Instant) {
        let elapsed_ms = started.elapsed().as_millis() as u64;
        if !self.repo.exists(id).await.unwrap_or(false) {
            // The dataset was deleted while the job ran: drop anything it wrote.
            tracing::info!(dataset = %id, "dataset deleted during training; cleaning up");
            if let Err(e) = self.store.delete_dataset(id).await {
                tracing::warn!(dataset = %id, error = %e, "cleanup after deletion failed");
            }
            return;
        }
        let update = match result {
            Ok(()) => {
                let items = self.store.count_items(id).await.ok();
                let users = self.store.count_users(id).await.ok();
                let model = self.embedder.model().to_string();
                tracing::info!(dataset = %id, elapsed_ms, items, users, "training finished");
                self.repo
                    .update(id, |d| {
                        d.status = DatasetStatus::Ready;
                        d.error = None;
                        d.progress = None;
                        d.heartbeat_at = None;
                        d.trained_at = Some(Utc::now());
                        d.embedding_model = model;
                        if let Some(items) = items {
                            d.stats.items = items;
                        }
                        if let Some(users) = users {
                            d.stats.users = users;
                        }
                    })
                    .await
            }
            Err(err) => {
                tracing::error!(dataset = %id, elapsed_ms, error = %err, "training failed");
                self.repo
                    .update(id, |d| {
                        d.status = DatasetStatus::Failed;
                        d.error = Some(err.to_string());
                        d.progress = None;
                        d.heartbeat_at = None;
                    })
                    .await
            }
        };
        if let Err(e) = update {
            tracing::error!(dataset = %id, error = %e, "could not record job result");
        }
    }
}
