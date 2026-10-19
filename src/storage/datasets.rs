//! File-system persistence for dataset metadata and the raw data each dataset
//! was built from.
//!
//! ```text
//! {DATA_DIR}/datasets/{dataset_id}/
//!   dataset.json                                  metadata (atomic writes)
//!   items/{ts}-{rand}.parquet                     uploaded files
//!   items/{ts}-{rand}.jsonl                       items sent through the JSON API
//!   items/{ts}-{rand}.deletes.json                deleted item ids
//!   interactions/{date}-{instance}.jsonl          interactions sent through the API (appended)
//!   interactions/upload-{ts}-{rand}.jsonl         uploaded interaction files
//! ```
//!
//! Item parts are named with a zero-padded microsecond timestamp, so replaying
//! them in file-name order reproduces the dataset (see `POST /datasets/{id}/train`).
//! The directory can live on a shared volume when running several replicas.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::Utc;
use polars::prelude::DataFrame;
use serde_json::{Map, Value};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::error::{AppError, AppResult};
use crate::ids::validate_dataset_id;
use crate::models::{Dataset, Interaction};
use crate::storage::tabular;

const METADATA_FILE: &str = "dataset.json";

/// One stored write to a dataset's items, in replay order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ItemPart {
    Upload(PathBuf),
    Batch(PathBuf),
    Deletes(PathBuf),
}

pub struct DatasetRepository {
    root: PathBuf,
    instance: String,
    locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl DatasetRepository {
    /// `instance` distinguishes append-only files written by different replicas.
    pub fn new(data_dir: &Path, instance: impl Into<String>) -> Self {
        let instance: String = instance.into();
        let instance = instance
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        Self {
            root: data_dir.join("datasets"),
            instance,
            locks: Mutex::default(),
        }
    }

    pub async fn init(&self) -> AppResult<()> {
        tokio::fs::create_dir_all(&self.root).await?;
        Ok(())
    }

    fn dir(&self, id: &str) -> AppResult<PathBuf> {
        validate_dataset_id(id)?;
        Ok(self.root.join(id))
    }

    /// Per-dataset lock serializing read-modify-write cycles of the metadata.
    fn lock(&self, id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.locks.lock().unwrap_or_else(|e| e.into_inner());
        locks.entry(id.to_string()).or_default().clone()
    }

    pub async fn create(&self, dataset: &Dataset) -> AppResult<()> {
        let dir = self.dir(&dataset.id)?;
        tokio::fs::create_dir_all(&self.root).await?;
        match tokio::fs::create_dir(&dir).await {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(AppError::Conflict(format!(
                    "dataset '{}' already exists",
                    dataset.id
                )));
            }
            Err(e) => return Err(e.into()),
        }
        self.write_metadata(&dir, dataset).await
    }

    pub async fn get(&self, id: &str) -> AppResult<Dataset> {
        let path = self.dir(id)?.join(METADATA_FILE);
        match tokio::fs::read(&path).await {
            Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(AppError::not_found(format!("dataset '{id}' not found")))
            }
            Err(e) => Err(e.into()),
        }
    }

    pub async fn exists(&self, id: &str) -> AppResult<bool> {
        Ok(tokio::fs::try_exists(self.dir(id)?.join(METADATA_FILE)).await?)
    }

    pub async fn list(&self) -> AppResult<Vec<Dataset>> {
        let mut out = Vec::new();
        let mut entries = match tokio::fs::read_dir(&self.root).await {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(e.into()),
        };
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path().join(METADATA_FILE);
            match tokio::fs::read(&path).await {
                Ok(bytes) => match serde_json::from_slice::<Dataset>(&bytes) {
                    Ok(dataset) => out.push(dataset),
                    Err(e) => {
                        tracing::warn!(file = %path.display(), error = %e, "skipping unreadable dataset metadata")
                    }
                },
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
        out.sort_by(|a, b| {
            a.created_at
                .cmp(&b.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(out)
    }

    /// Applies `change` to the stored metadata under the dataset's lock and
    /// returns the updated dataset.
    pub async fn update(&self, id: &str, change: impl FnOnce(&mut Dataset)) -> AppResult<Dataset> {
        let lock = self.lock(id);
        let _guard = lock.lock().await;
        let mut dataset = self.get(id).await?;
        change(&mut dataset);
        dataset.updated_at = Utc::now();
        self.write_metadata(&self.dir(id)?, &dataset).await?;
        Ok(dataset)
    }

    pub async fn delete(&self, id: &str) -> AppResult<()> {
        let dir = self.dir(id)?;
        match tokio::fs::remove_dir_all(&dir).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(AppError::not_found(format!("dataset '{id}' not found")))
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn write_metadata(&self, dir: &Path, dataset: &Dataset) -> AppResult<()> {
        let tmp = dir.join(format!(".{METADATA_FILE}.{}", Uuid::new_v4().simple()));
        tokio::fs::write(&tmp, serde_json::to_vec_pretty(dataset)?).await?;
        tokio::fs::rename(&tmp, dir.join(METADATA_FILE)).await?;
        Ok(())
    }

    /// Directory for temporary uploads (on the same volume, so moves are cheap).
    pub async fn uploads_dir(&self) -> AppResult<PathBuf> {
        let dir = self.root.join(".uploads");
        tokio::fs::create_dir_all(&dir).await?;
        Ok(dir)
    }

    // ─── Item parts ──────────────────────────────────────────────────────────

    async fn new_item_part(&self, id: &str, suffix: &str) -> AppResult<(String, PathBuf)> {
        let dir = self.dir(id)?.join("items");
        tokio::fs::create_dir_all(&dir).await?;
        let name = format!(
            "{:020}-{}",
            Utc::now().timestamp_micros(),
            &Uuid::new_v4().simple().to_string()[..8]
        );
        let path = dir.join(format!("{name}.{suffix}"));
        Ok((name, path))
    }

    /// Stores an uploaded table as Parquet. Returns the upload id.
    pub async fn write_items_upload(&self, id: &str, mut df: DataFrame) -> AppResult<String> {
        let (name, path) = self.new_item_part(id, "parquet").await?;
        tokio::task::spawn_blocking(move || tabular::write_parquet(&mut df, &path)).await??;
        Ok(name)
    }

    /// Stores items received through the JSON API. Returns the batch id.
    pub async fn write_items_batch(
        &self,
        id: &str,
        rows: &[Map<String, Value>],
    ) -> AppResult<String> {
        let (name, path) = self.new_item_part(id, "jsonl").await?;
        let mut body = Vec::new();
        for row in rows {
            serde_json::to_writer(&mut body, row)?;
            body.push(b'\n');
        }
        tokio::fs::write(&path, body).await?;
        Ok(name)
    }

    pub async fn write_item_deletes(&self, id: &str, item_ids: &[String]) -> AppResult<()> {
        let (_, path) = self.new_item_part(id, "deletes.json").await?;
        tokio::fs::write(&path, serde_json::to_vec(item_ids)?).await?;
        Ok(())
    }

    /// All item parts in replay (chronological) order.
    pub async fn item_parts(&self, id: &str) -> AppResult<Vec<ItemPart>> {
        let mut parts: Vec<(String, ItemPart)> = list_files(&self.dir(id)?.join("items"))
            .await?
            .into_iter()
            .filter_map(|path| {
                let name = path.file_name()?.to_str()?.to_string();
                let part = if name.ends_with(".deletes.json") {
                    ItemPart::Deletes(path)
                } else if name.ends_with(".parquet") {
                    ItemPart::Upload(path)
                } else if name.ends_with(".jsonl") {
                    ItemPart::Batch(path)
                } else {
                    return None;
                };
                Some((name, part))
            })
            .collect();
        parts.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(parts.into_iter().map(|(_, part)| part).collect())
    }

    // ─── Interactions ────────────────────────────────────────────────────────

    /// Appends interactions to today's log of this instance.
    pub async fn append_interactions(
        &self,
        id: &str,
        interactions: &[Interaction],
    ) -> AppResult<()> {
        let dir = self.dir(id)?.join("interactions");
        tokio::fs::create_dir_all(&dir).await?;
        let path = dir.join(format!(
            "{}-{}.jsonl",
            Utc::now().format("%Y-%m-%d"),
            self.instance
        ));
        let body = to_jsonl(interactions)?;
        let lock = self.lock(&format!("{id}/interactions"));
        let _guard = lock.lock().await;
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;
        file.write_all(&body).await?;
        file.flush().await?;
        Ok(())
    }

    /// Stores an uploaded interactions file (already parsed). Returns the upload id.
    pub async fn write_interactions_upload(
        &self,
        id: &str,
        interactions: &[Interaction],
    ) -> AppResult<String> {
        let dir = self.dir(id)?.join("interactions");
        tokio::fs::create_dir_all(&dir).await?;
        let name = format!(
            "upload-{:020}-{}",
            Utc::now().timestamp_micros(),
            &Uuid::new_v4().simple().to_string()[..8]
        );
        tokio::fs::write(dir.join(format!("{name}.jsonl")), to_jsonl(interactions)?).await?;
        Ok(name)
    }

    /// Paths of all interaction logs of a dataset.
    pub async fn interaction_files(&self, id: &str) -> AppResult<Vec<PathBuf>> {
        let mut files: Vec<PathBuf> = list_files(&self.dir(id)?.join("interactions"))
            .await?
            .into_iter()
            .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
            .collect();
        files.sort();
        Ok(files)
    }
}

fn to_jsonl(interactions: &[Interaction]) -> AppResult<Vec<u8>> {
    let mut body = Vec::new();
    for interaction in interactions {
        serde_json::to_writer(&mut body, interaction)?;
        body.push(b'\n');
    }
    Ok(body)
}

/// Reads an interactions log, skipping malformed lines. Blocking.
pub fn read_interactions(path: &Path) -> AppResult<Vec<Interaction>> {
    Ok(tabular::read_jsonl(path)?
        .into_iter()
        .filter_map(|row| serde_json::from_value(Value::Object(row)).ok())
        .collect())
}

async fn list_files(dir: &Path) -> AppResult<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut entries = match tokio::fs::read_dir(dir).await {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e.into()),
    };
    while let Some(entry) = entries.next_entry().await? {
        if entry.file_type().await?.is_file() {
            out.push(entry.path());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{DatasetStats, DatasetStatus};
    use serde_json::json;

    fn dataset(id: &str) -> Dataset {
        let now = Utc::now();
        Dataset {
            id: id.into(),
            name: id.into(),
            description: None,
            id_column: None,
            text_columns: vec![],
            status: DatasetStatus::Pending,
            error: None,
            progress: None,
            stats: DatasetStats::default(),
            columns: vec![],
            embedding_model: "hash-384".into(),
            created_at: now,
            updated_at: now,
            trained_at: None,
            heartbeat_at: None,
        }
    }

    #[tokio::test]
    async fn crud_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = DatasetRepository::new(tmp.path(), "test");
        repo.create(&dataset("a")).await.unwrap();
        repo.create(&dataset("b")).await.unwrap();
        assert!(matches!(
            repo.create(&dataset("a")).await,
            Err(AppError::Conflict(_))
        ));

        let updated = repo
            .update("a", |d| d.status = DatasetStatus::Ready)
            .await
            .unwrap();
        assert_eq!(updated.status, DatasetStatus::Ready);
        assert_eq!(repo.get("a").await.unwrap().status, DatasetStatus::Ready);
        assert_eq!(repo.list().await.unwrap().len(), 2);

        repo.delete("a").await.unwrap();
        assert!(matches!(repo.get("a").await, Err(AppError::NotFound(_))));
        assert!(matches!(repo.delete("a").await, Err(AppError::NotFound(_))));
    }

    #[tokio::test]
    async fn rejects_path_traversal() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = DatasetRepository::new(tmp.path(), "test");
        assert!(matches!(
            repo.get("../etc").await,
            Err(AppError::BadRequest(_))
        ));
        assert!(matches!(
            repo.create(&dataset("../x")).await,
            Err(AppError::BadRequest(_))
        ));
    }

    #[tokio::test]
    async fn item_parts_are_replayed_in_order() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = DatasetRepository::new(tmp.path(), "test");
        repo.create(&dataset("a")).await.unwrap();
        repo.write_items_batch("a", &[json!({"id": 1}).as_object().unwrap().clone()])
            .await
            .unwrap();
        repo.write_item_deletes("a", &["1".into()]).await.unwrap();
        let parts = repo.item_parts("a").await.unwrap();
        assert!(matches!(parts[0], ItemPart::Batch(_)));
        assert!(matches!(parts[1], ItemPart::Deletes(_)));
    }

    #[tokio::test]
    async fn appends_and_reads_interactions() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = DatasetRepository::new(tmp.path(), "host/1");
        repo.create(&dataset("a")).await.unwrap();
        let i = |u: &str| Interaction {
            user_id: u.into(),
            item_id: "x".into(),
            weight: None,
            event_type: None,
            timestamp: None,
            prediction_id: None,
        };
        repo.append_interactions("a", &[i("u1")]).await.unwrap();
        repo.append_interactions("a", &[i("u2"), i("u3")])
            .await
            .unwrap();
        repo.write_interactions_upload("a", &[i("u4")])
            .await
            .unwrap();
        let files = repo.interaction_files("a").await.unwrap();
        assert_eq!(files.len(), 2);
        let total: usize = files
            .iter()
            .map(|f| read_interactions(f).unwrap().len())
            .sum();
        assert_eq!(total, 4);
        assert!(files.iter().any(|f| f.to_string_lossy().contains("host_1")));
    }
}
