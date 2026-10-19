//! Multipart file uploads, streamed to disk with a size limit.

use actix_multipart::Multipart;
use actix_web::HttpResponse;
use actix_web::http::header::LOCATION;
use futures_util::TryStreamExt;
use tokio::io::AsyncWriteExt;
use utoipa::{IntoParams, ToSchema};
use uuid::Uuid;

use crate::error::{AppError, AppResult};
use crate::models::{DatasetStatus, IngestResponse};
use crate::services::{Accepted, Upload};
use crate::state::AppState;
use crate::storage::tabular::{FileFormat, SUPPORTED_FORMATS};

/// `multipart/form-data` body with a single file field.
#[derive(ToSchema)]
#[allow(dead_code)]
pub struct UploadForm {
    /// CSV, TSV, JSON, NDJSON or Parquet file.
    #[schema(value_type = String, format = Binary)]
    file: Vec<u8>,
}

/// Query parameters of endpoints that only support waiting.
#[derive(Debug, Default, serde::Deserialize, IntoParams)]
#[into_params(parameter_in = Query)]
#[serde(deny_unknown_fields)]
pub struct WaitParams {
    /// Block until processing finishes (default: false).
    pub wait: Option<bool>,
}

/// Streams the first file field of the request to a temporary file.
pub async fn receive_file(mut payload: Multipart, state: &AppState) -> AppResult<Upload> {
    let max = state.settings.server.max_upload_bytes;
    let dir = state.datasets.uploads_dir().await?;
    while let Some(mut field) = payload
        .try_next()
        .await
        .map_err(|e| AppError::bad_request(format!("invalid multipart body: {e}")))?
    {
        let Some(filename) = field
            .content_disposition()
            .and_then(|cd| cd.get_filename())
            .map(str::to_string)
        else {
            continue;
        };
        let format = FileFormat::from_filename(&filename).ok_or_else(|| {
            AppError::UnsupportedMediaType(format!(
                "unsupported file type '{filename}'; supported: {SUPPORTED_FORMATS}"
            ))
        })?;
        let path = dir.join(format!(
            "{}.{}",
            Uuid::new_v4().simple(),
            format.extension()
        ));
        let mut file = tokio::fs::File::create(&path).await?;
        let mut written = 0usize;
        let result: AppResult<()> = async {
            while let Some(chunk) = field
                .try_next()
                .await
                .map_err(|e| AppError::bad_request(format!("upload interrupted: {e}")))?
            {
                written += chunk.len();
                if written > max {
                    return Err(AppError::PayloadTooLarge(format!(
                        "file exceeds the {} MB limit (MAX_UPLOAD_MB)",
                        max / (1024 * 1024)
                    )));
                }
                file.write_all(&chunk).await?;
            }
            file.flush().await?;
            if written == 0 {
                return Err(AppError::bad_request("the uploaded file is empty"));
            }
            Ok(())
        }
        .await;
        if let Err(e) = result {
            drop(file);
            let _ = tokio::fs::remove_file(&path).await;
            return Err(e);
        }
        tracing::debug!(%filename, bytes = written, "upload received");
        return Ok(Upload {
            path,
            filename,
            format,
        });
    }
    Err(AppError::bad_request(
        "no file found: send it as multipart/form-data in a field named `file`",
    ))
}

/// 202 with the accepted job, or (with `wait=true`) 200 once it finished.
pub async fn job_response(
    state: &AppState,
    accepted: Accepted,
    wait: bool,
) -> AppResult<HttpResponse> {
    let Accepted {
        dataset,
        rows,
        upload_id,
        job,
    } = accepted;
    let location = format!("/api/v1/datasets/{}", dataset.id);
    if !wait {
        return Ok(HttpResponse::Accepted()
            .insert_header((LOCATION, location))
            .json(IngestResponse {
                dataset,
                rows,
                upload_id,
            }));
    }
    job.await?;
    let dataset = state.datasets.get(&dataset.id).await?;
    if dataset.status == DatasetStatus::Failed {
        return Err(AppError::internal(format!(
            "training failed: {}",
            dataset.error.as_deref().unwrap_or("unknown error")
        )));
    }
    Ok(HttpResponse::Ok().json(IngestResponse {
        dataset,
        rows,
        upload_id,
    }))
}
