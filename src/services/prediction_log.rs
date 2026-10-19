//! Asynchronous prediction logging.
//!
//! Requests hand log entries to a bounded channel (never blocking the request);
//! a background task appends them in batches to
//! `{DATA_DIR}/predictions/{YYYY-MM-DD}/{instance}.jsonl`. Each replica writes its
//! own file, so a shared volume needs no cross-process locking.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use chrono::{NaiveDate, Utc};
use polars::prelude::*;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::error::{AppError, AppResult};
use crate::models::{ExportFormat, PredictionLog, PredictionQuery};
use crate::storage::tabular;

const CHANNEL_CAPACITY: usize = 10_000;
const MAX_BATCH: usize = 1_024;

pub struct PredictionLogger {
    dir: PathBuf,
    sender: Mutex<Option<mpsc::Sender<PredictionLog>>>,
    writer: tokio::sync::Mutex<Option<JoinHandle<()>>>,
    dropped: AtomicU64,
}

/// An export ready to be sent as a file download.
pub struct Export {
    pub body: Vec<u8>,
    pub content_type: &'static str,
    pub filename: String,
}

impl PredictionLogger {
    /// Starts the background writer (when `enabled`). Must run inside a Tokio runtime.
    pub fn start(data_dir: &Path, instance: &str, enabled: bool) -> Self {
        let dir = data_dir.join("predictions");
        let (sender, writer) = if enabled {
            let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
            (
                Some(tx),
                Some(tokio::spawn(write_loop(
                    rx,
                    dir.clone(),
                    instance.to_string(),
                ))),
            )
        } else {
            (None, None)
        };
        Self {
            dir,
            sender: Mutex::new(sender),
            writer: tokio::sync::Mutex::new(writer),
            dropped: AtomicU64::new(0),
        }
    }

    /// Queues a log entry. Drops it (and counts the drop) when the queue is full.
    pub fn record(&self, log: PredictionLog) {
        let sender = self.sender.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(tx) = sender.as_ref()
            && tx.try_send(log).is_err()
        {
            let dropped = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
            if dropped.is_power_of_two() {
                tracing::warn!(dropped, "prediction log queue full; dropping entries");
            }
        }
    }

    /// Flushes pending entries and stops the writer.
    pub async fn shutdown(&self) {
        drop(self.sender.lock().unwrap_or_else(|e| e.into_inner()).take());
        let writer = self.writer.lock().await.take();
        if let Some(handle) = writer
            && tokio::time::timeout(Duration::from_secs(10), handle)
                .await
                .is_err()
        {
            tracing::warn!("timed out flushing prediction logs");
        }
    }

    /// Most recent logs first.
    pub async fn list(
        &self,
        query: &PredictionQuery,
        limit: usize,
    ) -> AppResult<Vec<PredictionLog>> {
        if let Some(date) = &query.date {
            NaiveDate::parse_from_str(date, "%Y-%m-%d")
                .map_err(|_| AppError::bad_request("date must be formatted as YYYY-MM-DD"))?;
        }
        let dir = self.dir.clone();
        let query = query.clone();
        tokio::task::spawn_blocking(move || read_logs(&dir, &query, limit)).await?
    }

    pub async fn export(&self, query: &PredictionQuery) -> AppResult<Export> {
        let format = query.format.unwrap_or_default();
        let logs = self
            .list(query, query.limit.unwrap_or(100_000).min(1_000_000))
            .await?;
        let stamp = Utc::now().format("%Y%m%dT%H%M%SZ");
        tokio::task::spawn_blocking(move || -> AppResult<Export> {
            Ok(match format {
                ExportFormat::Jsonl => {
                    let mut body = Vec::new();
                    for log in &logs {
                        serde_json::to_writer(&mut body, log)?;
                        body.push(b'\n');
                    }
                    Export {
                        body,
                        content_type: "application/x-ndjson",
                        filename: format!("predictions-{stamp}.jsonl"),
                    }
                }
                ExportFormat::Csv => {
                    let mut body = Vec::new();
                    CsvWriter::new(&mut body)
                        .finish(&mut logs_frame(&logs)?)
                        .map_err(|e| AppError::internal(format!("csv export failed: {e}")))?;
                    Export {
                        body,
                        content_type: "text/csv",
                        filename: format!("predictions-{stamp}.csv"),
                    }
                }
                ExportFormat::Parquet => {
                    let mut body = Vec::new();
                    ParquetWriter::new(&mut body)
                        .finish(&mut logs_frame(&logs)?)
                        .map_err(|e| AppError::internal(format!("parquet export failed: {e}")))?;
                    Export {
                        body,
                        content_type: "application/vnd.apache.parquet",
                        filename: format!("predictions-{stamp}.parquet"),
                    }
                }
            })
        })
        .await?
    }
}

async fn write_loop(mut rx: mpsc::Receiver<PredictionLog>, dir: PathBuf, instance: String) {
    let mut batch = Vec::with_capacity(MAX_BATCH);
    while let Some(first) = rx.recv().await {
        batch.push(first);
        while batch.len() < MAX_BATCH {
            match rx.try_recv() {
                Ok(log) => batch.push(log),
                Err(_) => break,
            }
        }
        if let Err(e) = append(&dir, &instance, &batch).await {
            tracing::warn!(error = %e, count = batch.len(), "failed to write prediction logs");
        }
        batch.clear();
    }
}

async fn append(dir: &Path, instance: &str, batch: &[PredictionLog]) -> std::io::Result<()> {
    let mut by_date: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for log in batch {
        let buf = by_date
            .entry(log.timestamp.format("%Y-%m-%d").to_string())
            .or_default();
        serde_json::to_writer(&mut *buf, log).map_err(std::io::Error::other)?;
        buf.push(b'\n');
    }
    for (date, bytes) in by_date {
        let day = dir.join(date);
        tokio::fs::create_dir_all(&day).await?;
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(day.join(format!("{instance}.jsonl")))
            .await?;
        file.write_all(&bytes).await?;
        file.flush().await?;
    }
    Ok(())
}

fn read_logs(dir: &Path, query: &PredictionQuery, limit: usize) -> AppResult<Vec<PredictionLog>> {
    let mut dates: Vec<String> = match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .filter(|e| e.path().is_dir())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|name| NaiveDate::parse_from_str(name, "%Y-%m-%d").is_ok())
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    dates.sort_unstable_by(|a, b| b.cmp(a));
    if let Some(date) = &query.date {
        dates.retain(|d| d == date);
    }

    let mut out = Vec::new();
    for date in dates {
        let mut day: Vec<PredictionLog> = Vec::new();
        for entry in std::fs::read_dir(dir.join(&date))?.filter_map(Result::ok) {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "jsonl") {
                continue;
            }
            day.extend(
                tabular::read_jsonl(&path)?
                    .into_iter()
                    .filter_map(|row| {
                        serde_json::from_value::<PredictionLog>(serde_json::Value::Object(row)).ok()
                    })
                    .filter(|log| {
                        query
                            .dataset_id
                            .as_ref()
                            .is_none_or(|d| &log.dataset_id == d)
                    }),
            );
        }
        day.sort_by_key(|log| std::cmp::Reverse(log.timestamp));
        out.extend(day);
        if out.len() >= limit {
            out.truncate(limit);
            break;
        }
    }
    Ok(out)
}

fn logs_frame(logs: &[PredictionLog]) -> AppResult<DataFrame> {
    let join = |values: &[String]| values.join(",");
    let frame = DataFrame::new_infer_height(vec![
        Column::new(
            "prediction_id".into(),
            logs.iter()
                .map(|l| l.prediction_id.as_str())
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "timestamp".into(),
            logs.iter()
                .map(|l| l.timestamp.to_rfc3339())
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "dataset_id".into(),
            logs.iter()
                .map(|l| l.dataset_id.as_str())
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "query".into(),
            logs.iter().map(|l| l.query.as_deref()).collect::<Vec<_>>(),
        ),
        Column::new(
            "user_id".into(),
            logs.iter()
                .map(|l| l.user_id.as_deref())
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "item_ids".into(),
            logs.iter().map(|l| join(&l.item_ids)).collect::<Vec<_>>(),
        ),
        Column::new(
            "result_ids".into(),
            logs.iter().map(|l| join(&l.result_ids)).collect::<Vec<_>>(),
        ),
        Column::new(
            "scores".into(),
            logs.iter()
                .map(|l| {
                    l.scores
                        .iter()
                        .map(|s| format!("{s:.6}"))
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "result_count".into(),
            logs.iter()
                .map(|l| l.result_ids.len() as u32)
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "latency_ms".into(),
            logs.iter().map(|l| l.latency_ms).collect::<Vec<_>>(),
        ),
    ])
    .map_err(|e| AppError::internal(format!("could not build export table: {e}")))?;
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log(id: &str, dataset: &str, secs: i64) -> PredictionLog {
        PredictionLog {
            prediction_id: id.into(),
            timestamp: chrono::DateTime::from_timestamp(1_760_000_000 + secs, 0).unwrap(),
            dataset_id: dataset.into(),
            query: Some("q".into()),
            user_id: None,
            item_ids: vec![],
            result_ids: vec!["a".into(), "b".into()],
            scores: vec![0.9, 0.8],
            latency_ms: 3,
        }
    }

    #[tokio::test]
    async fn writes_lists_and_exports() {
        let tmp = tempfile::tempdir().unwrap();
        let logger = PredictionLogger::start(tmp.path(), "test", true);
        logger.record(log("p1", "a", 0));
        logger.record(log("p2", "b", 10));
        logger.record(log("p3", "a", 20));
        logger.shutdown().await;

        let all = logger.list(&PredictionQuery::default(), 10).await.unwrap();
        assert_eq!(
            all.iter()
                .map(|l| l.prediction_id.as_str())
                .collect::<Vec<_>>(),
            vec!["p3", "p2", "p1"]
        );

        let only_a = PredictionQuery {
            dataset_id: Some("a".into()),
            ..Default::default()
        };
        assert_eq!(logger.list(&only_a, 10).await.unwrap().len(), 2);
        assert_eq!(
            logger
                .list(&PredictionQuery::default(), 1)
                .await
                .unwrap()
                .len(),
            1
        );

        for format in [
            ExportFormat::Jsonl,
            ExportFormat::Csv,
            ExportFormat::Parquet,
        ] {
            let export = logger
                .export(&PredictionQuery {
                    format: Some(format),
                    ..Default::default()
                })
                .await
                .unwrap();
            assert!(!export.body.is_empty(), "{format:?}");
        }
        let csv = logger
            .export(&PredictionQuery {
                format: Some(ExportFormat::Csv),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(
            String::from_utf8(csv.body)
                .unwrap()
                .starts_with("prediction_id,timestamp")
        );
    }

    #[tokio::test]
    async fn disabled_logger_records_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let logger = PredictionLogger::start(tmp.path(), "test", false);
        logger.record(log("p1", "a", 0));
        logger.shutdown().await;
        assert!(
            logger
                .list(&PredictionQuery::default(), 10)
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            logger
                .list(
                    &PredictionQuery {
                        date: Some("bad".into()),
                        ..Default::default()
                    },
                    1
                )
                .await
                .is_err()
        );
    }
}
