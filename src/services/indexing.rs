//! Item ingestion: file uploads, JSON upserts, deletions and full retraining.

use std::sync::Arc;

use chrono::Utc;
use polars::prelude::{DataFrame, ParquetReader, SerReader};
use serde_json::{Map, Value};
use tokio::task::JoinHandle;

use super::datasets::{Accepted, DatasetService, Progress, Upload};
use crate::error::{AppError, AppResult};
use crate::ids::{validate_dataset_id, validate_entity_id};
use crate::models::{
    CreateDatasetRequest, Dataset, DatasetStatus, Item, ItemResponse, TrainingProgress,
    TrainingStage, UploadParams, UpsertItemsResponse,
};
use crate::storage::ItemPart;
use crate::storage::tabular::{self, ColumnProfile, ItemSchema};
use crate::vector_store::{ItemPoint, StoreResult};

/// Rows converted to items at once when indexing a table (bounds memory use).
const SLICE_ROWS: usize = 10_000;
/// Maximum items accepted by one JSON upsert request.
pub const MAX_ITEMS_PER_REQUEST: usize = 10_000;

impl DatasetService {
    /// Validates and stores an uploaded items file, then indexes it in the
    /// background. Creates the dataset (with `params`) when it does not exist.
    pub async fn upload_items(
        self: &Arc<Self>,
        id: &str,
        upload: Upload,
        params: &UploadParams,
    ) -> AppResult<Accepted> {
        if let Err(e) = validate_dataset_id(id) {
            upload.discard().await;
            return Err(e);
        }
        let df = upload.read().await?;
        let create = params.as_create_request(id);
        let (dataset, schema) = self
            .resolve_schema(id, &tabular::profile_frame(&df), create, true)
            .await?;
        let rows = df.height() as u64;
        let columns: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|c| c.to_string())
            .collect();
        let upload_id = self
            .repo
            .write_items_upload(&dataset.id, df.clone())
            .await?;

        let dataset = self
            .repo
            .update(id, |d| {
                apply_schema(d, &schema, &columns);
                d.stats.uploads += 1;
                d.status = DatasetStatus::Training;
                d.error = None;
                d.heartbeat_at = Some(Utc::now());
                d.progress = Some(TrainingProgress {
                    stage: TrainingStage::Items,
                    processed: 0,
                    total: rows,
                });
            })
            .await?;
        tracing::info!(dataset = %id, rows, %upload_id, "items upload accepted");

        let dataset_id = id.to_string();
        let job = self.spawn_job(id, move |this| async move {
            let mut progress = Progress::new(TrainingStage::Items, rows);
            this.index_frame(&dataset_id, &schema, df, &mut progress)
                .await?;
            this.report(&dataset_id, &mut progress, true).await
        });
        Ok(Accepted {
            dataset,
            rows,
            upload_id: Some(upload_id),
            job,
        })
    }

    /// Embeds and indexes items synchronously. Creates the dataset when needed.
    pub async fn upsert_items(
        &self,
        id: &str,
        rows: Vec<Map<String, Value>>,
    ) -> AppResult<UpsertItemsResponse> {
        validate_dataset_id(id)?;
        if rows.is_empty() {
            return Err(AppError::bad_request(
                "`items` must contain at least one item",
            ));
        }
        if rows.len() > MAX_ITEMS_PER_REQUEST {
            return Err(AppError::bad_request(format!(
                "at most {MAX_ITEMS_PER_REQUEST} items per request; use the upload endpoint for bulk loads"
            )));
        }
        let profile = tabular::profile_rows(&rows);
        let (_, schema) = self
            .resolve_schema(id, &profile, CreateDatasetRequest::default(), false)
            .await?;
        let columns: Vec<String> = profile.into_iter().map(|(c, _)| c).collect();
        let (items, skipped) = tabular::build_items(rows.clone(), &schema);
        if items.is_empty() {
            return Err(AppError::bad_request(format!(
                "no valid items: each item needs an id ({}) and text ({})",
                schema
                    .id_column
                    .as_deref()
                    .map_or("derived from content".to_string(), |c| format!("`{c}`")),
                schema.text_columns.join(", ")
            )));
        }

        // Log first (like a WAL) so a retrain always includes acknowledged writes.
        self.repo.write_items_batch(id, &rows).await?;
        let upserted = self.index_items(id, items, None).await?;
        let count = self.store.count_items(id).await?;
        self.repo
            .update(id, |d| {
                apply_schema(d, &schema, &columns);
                d.stats.uploads += 1;
                d.stats.items = count;
                if d.status == DatasetStatus::Pending {
                    d.status = DatasetStatus::Ready;
                    d.trained_at = Some(Utc::now());
                }
            })
            .await?;
        Ok(UpsertItemsResponse {
            dataset_id: id.to_string(),
            upserted,
            skipped,
        })
    }

    pub async fn get_item(&self, id: &str, item_id: &str) -> AppResult<ItemResponse> {
        self.repo.get(id).await?;
        validate_entity_id("item", item_id)?;
        self.store
            .get_items(id, &[item_id.to_string()], false)
            .await?
            .into_iter()
            .next()
            .map(|item| ItemResponse {
                id: item.item_id,
                payload: item.payload,
            })
            .ok_or_else(|| {
                AppError::not_found(format!("item '{item_id}' not found in dataset '{id}'"))
            })
    }

    pub async fn delete_item(&self, id: &str, item_id: &str) -> AppResult<()> {
        self.get_item(id, item_id).await?;
        let ids = [item_id.to_string()];
        self.repo.write_item_deletes(id, &ids).await?;
        self.store.delete_items(id, &ids).await?;
        let count = self.store.count_items(id).await?;
        self.repo.update(id, |d| d.stats.items = count).await?;
        Ok(())
    }

    /// Rebuilds the dataset's index from its stored uploads and interactions,
    /// e.g. after switching embedding models.
    pub async fn train(self: &Arc<Self>, id: &str, reset: bool) -> AppResult<Accepted> {
        self.repo.get(id).await?;
        let dataset = self.mark_training(id, None).await?;
        let dataset_id = id.to_string();
        let job = self.spawn_job(id, move |this| async move {
            if reset {
                this.store.delete_dataset(&dataset_id).await?;
            }
            this.replay_items(&dataset_id).await?;
            this.rebuild_profiles(&dataset_id).await
        });
        Ok(Accepted {
            dataset,
            rows: 0,
            upload_id: None,
            job,
        })
    }

    /// Resolves the item schema for incoming data: the dataset's configuration
    /// when it exists, otherwise `create` plus auto-detection. A dataset is only
    /// created once the data is known to be usable.
    async fn resolve_schema(
        &self,
        id: &str,
        columns: &[ColumnProfile],
        create: CreateDatasetRequest,
        strict: bool,
    ) -> AppResult<(Dataset, ItemSchema)> {
        let existing = self.find(id).await?;
        let (id_column, text_columns) = match &existing {
            Some(d) => (d.id_column.clone(), d.text_columns.clone()),
            None => (
                create.id_column.clone(),
                create.text_columns.clone().unwrap_or_default(),
            ),
        };
        let schema =
            tabular::resolve_item_schema(columns, id_column.as_deref(), &text_columns, strict)?;
        let dataset = match existing {
            Some(d) => d,
            None => self.get_or_create(id, create).await?,
        };
        Ok((dataset, schema))
    }

    async fn replay_items(&self, id: &str) -> AppResult<()> {
        let dataset = self.repo.get(id).await?;
        let parts = self.repo.item_parts(id).await?;
        if parts.is_empty() {
            return Ok(());
        }
        let schema = ItemSchema {
            id_column: dataset.id_column.clone(),
            text_columns: dataset.text_columns.clone(),
        };
        let counted = parts.clone();
        let total =
            tokio::task::spawn_blocking(move || counted.iter().map(part_rows).sum::<u64>()).await?;
        let mut progress = Progress::new(TrainingStage::Items, total);
        self.report(id, &mut progress, true).await?;

        for part in parts {
            match part {
                ItemPart::Upload(path) => {
                    let df =
                        tokio::task::spawn_blocking(move || tabular::read_parquet(&path)).await??;
                    self.index_frame(id, &schema, df, &mut progress).await?;
                }
                ItemPart::Batch(path) => {
                    let rows =
                        tokio::task::spawn_blocking(move || tabular::read_jsonl(&path)).await??;
                    let (items, skipped) = tabular::build_items(rows, &schema);
                    progress.processed += skipped;
                    self.index_items(id, items, Some(&mut progress)).await?;
                }
                ItemPart::Deletes(path) => {
                    let ids: Vec<String> = serde_json::from_slice(&tokio::fs::read(&path).await?)?;
                    self.store.delete_items(id, &ids).await?;
                }
            }
        }
        self.report(id, &mut progress, true).await
    }

    /// Indexes a table slice by slice.
    pub(crate) async fn index_frame(
        &self,
        id: &str,
        schema: &ItemSchema,
        df: DataFrame,
        progress: &mut Progress,
    ) -> AppResult<u64> {
        let height = df.height();
        let mut indexed = 0;
        let mut offset = 0;
        while offset < height {
            let len = SLICE_ROWS.min(height - offset);
            let mut slice = df.slice(offset as i64, len);
            let slice_schema = schema.clone();
            let (items, skipped) =
                tokio::task::spawn_blocking(move || -> AppResult<(Vec<Item>, u64)> {
                    Ok(tabular::build_items(
                        tabular::rows(&mut slice)?,
                        &slice_schema,
                    ))
                })
                .await??;
            progress.processed += skipped;
            indexed += self.index_items(id, items, Some(progress)).await?;
            offset += len;
        }
        Ok(indexed)
    }

    /// Embeds items in batches of `TRAINING_BATCH_SIZE` and writes them to the
    /// store. Embedding of batch N+1 overlaps with the upsert of batch N.
    pub(crate) async fn index_items(
        &self,
        id: &str,
        items: Vec<Item>,
        mut progress: Option<&mut Progress>,
    ) -> AppResult<u64> {
        let mut pending: Option<JoinHandle<StoreResult<()>>> = None;
        let mut indexed = 0u64;
        let mut items = items.into_iter();
        loop {
            let chunk: Vec<Item> = items.by_ref().take(self.settings.batch_size).collect();
            if chunk.is_empty() {
                break;
            }
            let texts: Vec<String> = chunk.iter().map(|item| item.text.clone()).collect();
            let vectors = self.embedder.embed_documents(texts).await?;
            if vectors.len() != chunk.len() {
                return Err(AppError::Embedding(format!(
                    "embedder returned {} vectors for {} texts",
                    vectors.len(),
                    chunk.len()
                )));
            }
            let points: Vec<ItemPoint> = chunk
                .into_iter()
                .zip(vectors)
                .map(|(item, vector)| ItemPoint {
                    item_id: item.id,
                    vector,
                    payload: item.payload,
                })
                .collect();
            let n = points.len() as u64;

            if let Some(previous) = pending.take() {
                previous.await??;
            }
            let store = Arc::clone(&self.store);
            let dataset_id = id.to_string();
            pending = Some(tokio::spawn(async move {
                store.upsert_items(&dataset_id, points).await
            }));

            indexed += n;
            if let Some(p) = progress.as_deref_mut() {
                p.processed += n;
                self.report(id, p, false).await?;
            }
        }
        if let Some(last) = pending {
            last.await??;
        }
        Ok(indexed)
    }
}

/// Records the detected schema on first use and the union of seen columns.
fn apply_schema(d: &mut Dataset, schema: &ItemSchema, columns: &[String]) {
    if d.id_column.is_none() {
        d.id_column = schema.id_column.clone();
    }
    if d.text_columns.is_empty() {
        d.text_columns = schema.text_columns.clone();
    }
    for column in columns {
        if !d.columns.contains(column) {
            d.columns.push(column.clone());
        }
    }
}

/// Number of rows in a stored part (cheap: Parquet metadata or line count).
fn part_rows(part: &ItemPart) -> u64 {
    match part {
        ItemPart::Upload(path) => std::fs::File::open(path)
            .ok()
            .and_then(|f| ParquetReader::new(f).num_rows().ok())
            .unwrap_or(0) as u64,
        ItemPart::Batch(path) => std::fs::read(path)
            .map(|bytes| {
                bytes
                    .split(|b| *b == b'\n')
                    .filter(|l| !l.is_empty())
                    .count() as u64
            })
            .unwrap_or(0),
        ItemPart::Deletes(_) => 0,
    }
}
