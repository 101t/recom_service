use actix_multipart::Multipart;
use actix_web::{post, web, HttpResponse, get, delete};
use futures_util::{stream::StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use utoipa::{IntoParams, ToSchema};
use uuid::Uuid;

use crate::AppState;

#[derive(Debug, Serialize, ToSchema)]
pub struct UploadResponse {
    /// Operation status
    pub status: String,
    /// Success message
    pub message: String,
    /// Generated dataset ID
    pub dataset_id: String,
    /// Job ID for tracking background processing (if async)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct DatasetListResponse {
    /// List of datasets
    pub datasets: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct DatasetInfoResponse {
    /// Dataset information
    pub dataset: serde_json::Value,
}

/// Upload query parameters
#[derive(Debug, Deserialize, IntoParams)]
pub struct UploadParams {
    /// Process in background for large files (default: false)
    pub background: Option<bool>,
    /// Existing dataset ID to append to
    pub dataset_id: Option<String>,
    /// Name of the text column to embed
    pub text_column: Option<String>,
}

/// Upload a dataset file (CSV, JSON, or Parquet)
///
/// For large datasets, set `?background=true` to process in the background.
/// The response will include a `job_id` you can use to check progress via
/// `GET /api/v1/datasets/jobs/{job_id}`.
#[utoipa::path(
    post,
    path = "/api/v1/datasets/upload",
    tag = "Datasets",
    params(UploadParams),
    responses(
        (status = 200, description = "File uploaded and processed synchronously", body = UploadResponse),
        (status = 202, description = "File accepted for background processing", body = UploadResponse),
        (status = 400, description = "Invalid file type"),
        (status = 500, description = "Internal server error")
    )
)]
#[post("/datasets/upload")]
pub async fn upload_dataset(
    mut payload: Multipart,
    query: web::Query<UploadParams>,
    app_state: web::Data<AppState>,
) -> HttpResponse {
    let background = query.background.unwrap_or(false);
    let existing_dataset_id = query.dataset_id.clone();
    let text_column = query.text_column.clone();

    // Iterate over multipart stream
    while let Ok(Some(mut field)) = payload.try_next().await {
        let filename = field
            .content_disposition()
            .and_then(|cd| cd.get_filename())
            .unwrap_or("unknown")
            .to_string();

        if !is_valid_file_type(&filename) {
            return HttpResponse::BadRequest().json(
                json!({"error": "Invalid file type. Supported types: CSV, JSON, Parquet"}),
            );
        }
        
        // Save uploaded file to disk
        let file_id = Uuid::new_v4();
        let filepath = format!("/tmp/{}_{}", file_id, filename);
        
        let mut f = match File::create(&filepath).await {
            Ok(f) => f,
            Err(e) => {
                return HttpResponse::InternalServerError().json(json!({
                    "error": format!("Failed to create file: {}", e)
                }));
            }
        };

        while let Some(chunk) = field.next().await {
            let data = match chunk {
                Ok(d) => d,
                Err(e) => {
                    return HttpResponse::InternalServerError().json(json!({
                        "error": format!("Failed to read upload chunk: {}", e)
                    }));
                }
            };
            if let Err(e) = f.write_all(&data).await {
                return HttpResponse::InternalServerError().json(json!({
                    "error": format!("Failed to write to file: {}", e)
                }));
            }
        }

        // ─── Background processing ─────────────────────────────────
        if background {
            let dataset_id = existing_dataset_id
                .clone()
                .unwrap_or_else(|| Uuid::new_v4().to_string());
            let job_id = Uuid::new_v4().to_string();

            // Register the job
            {
                let mut jobs = app_state.jobs.lock().await;
                jobs.insert(
                    job_id.clone(),
                    crate::JobStatus {
                        job_id: job_id.clone(),
                        dataset_id: dataset_id.clone(),
                        status: "processing".to_string(),
                        message: "Dataset upload started — processing in background".to_string(),
                        vectors_count: None,
                    },
                );
            }

            let state = app_state.clone();
            let jid = job_id.clone();
            let dsid = dataset_id.clone();
            let tc = text_column.clone();
            let fp = filepath.clone();
            tokio::spawn(async move {
                process_and_embed(&state, &fp, Some(dsid.clone()), tc, &jid).await;
            });

            return HttpResponse::Accepted().json(json!({
                "status": "accepted",
                "message": "File accepted for background processing",
                "dataset_id": dataset_id,
                "job_id": job_id
            }));
        }

        // ─── Synchronous processing ───────────────────────────────
        let processed = match app_state
            .dataset_handler
            .process_file(&filepath, existing_dataset_id.clone(), text_column.clone())
            .await
        {
            Ok(data) => data,
            Err(e) => {
                return HttpResponse::InternalServerError().json(json!({
                    "error": format!("Failed to process file: {}", e)
                }));
            }
        };
        
        // Embed and store
        if !processed.texts.is_empty() {
            if let Err(e) = app_state.qdrant_client.ensure_collection(384).await {
                log::warn!("Collection setup warning: {}", e);
            }
            
            match app_state.embedder.batch_embed(processed.texts.clone()).await {
                Ok(vectors) => {
                    if let Err(e) = app_state
                        .qdrant_client
                        .upsert_dataset_vectors(
                            &processed.dataset_id,
                            processed.texts.clone(),
                            vectors,
                        )
                        .await
                    {
                        log::error!("Failed to store vectors in Qdrant: {}", e);
                        return HttpResponse::InternalServerError().json(json!({
                            "error": format!("Failed to store vectors: {}", e)
                        }));
                    }
                }
                Err(e) => {
                    log::error!("Failed to generate embeddings: {}", e);
                    return HttpResponse::InternalServerError().json(json!({
                        "error": format!("Failed to generate embeddings: {}", e)
                    }));
                }
            }
        }
        
        // Cleanup temp file (best-effort)
        let _ = tokio::fs::remove_file(&filepath).await;

        return HttpResponse::Ok().json(json!({
            "status": "success",
            "message": "File uploaded, processed, and vectors stored",
            "dataset_id": processed.dataset_id,
            "vectors_count": processed.texts.len()
        }));
    }
    
    HttpResponse::BadRequest().json(json!({
        "error": "No valid file found in request"
    }))
}

/// Background processing helper
async fn process_and_embed(
    state: &web::Data<AppState>,
    filepath: &str,
    dataset_id: Option<String>,
    text_column: Option<String>,
    job_id: &str,
) {
    let result = state
        .dataset_handler
        .process_file(filepath, dataset_id, text_column)
        .await;

    match result {
        Ok(processed) => {
            if processed.texts.is_empty() {
                let mut jobs = state.jobs.lock().await;
                if let Some(job) = jobs.get_mut(job_id) {
                    job.status = "completed".to_string();
                    job.message = "No texts found to embed".to_string();
                    job.vectors_count = Some(0);
                }
                return;
            }

            if let Err(e) = state.qdrant_client.ensure_collection(384).await {
                log::warn!("Collection setup warning: {}", e);
            }

            match state.embedder.batch_embed(processed.texts.clone()).await {
                Ok(vectors) => {
                    let count = vectors.len();
                    match state
                        .qdrant_client
                        .upsert_dataset_vectors(
                            &processed.dataset_id,
                            processed.texts.clone(),
                            vectors,
                        )
                        .await
                    {
                        Ok(_) => {
                            let mut jobs = state.jobs.lock().await;
                            if let Some(job) = jobs.get_mut(job_id) {
                                job.status = "completed".to_string();
                                job.message = format!(
                                    "Processed {} rows, stored {} vectors",
                                    processed.row_count, count
                                );
                                job.vectors_count = Some(count);
                            }
                        }
                        Err(e) => {
                            log::error!("Background upsert failed: {}", e);
                            let mut jobs = state.jobs.lock().await;
                            if let Some(job) = jobs.get_mut(job_id) {
                                job.status = "failed".to_string();
                                job.message = format!("Failed to store vectors: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    log::error!("Background embedding failed: {}", e);
                    let mut jobs = state.jobs.lock().await;
                    if let Some(job) = jobs.get_mut(job_id) {
                        job.status = "failed".to_string();
                        job.message = format!("Failed to generate embeddings: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            log::error!("Background file processing failed: {}", e);
            let mut jobs = state.jobs.lock().await;
            if let Some(job) = jobs.get_mut(job_id) {
                job.status = "failed".to_string();
                job.message = format!("Failed to process file: {}", e);
            }
        }
    }

    // Cleanup temp file (best-effort)
    let _ = tokio::fs::remove_file(filepath).await;
}

/// Check the status of a background upload job
#[utoipa::path(
    get,
    path = "/api/v1/datasets/jobs/{job_id}",
    tag = "Datasets",
    params(
        ("job_id" = String, Path, description = "Background job ID")
    ),
    responses(
        (status = 200, description = "Job status", body = crate::JobStatus),
        (status = 404, description = "Job not found")
    )
)]
#[get("/datasets/jobs/{job_id}")]
pub async fn get_job_status(
    path: web::Path<String>,
    app_state: web::Data<AppState>,
) -> HttpResponse {
    let job_id = path.into_inner();
    let jobs = app_state.jobs.lock().await;
    
    match jobs.get(&job_id) {
        Some(status) => HttpResponse::Ok().json(status),
        None => HttpResponse::NotFound().json(json!({
            "error": format!("Job {} not found", job_id)
        })),
    }
}

/// List all available datasets
#[utoipa::path(
    get,
    path = "/api/v1/datasets",
    tag = "Datasets",
    responses(
        (status = 200, description = "List of datasets", body = DatasetListResponse),
        (status = 500, description = "Internal server error")
    )
)]
#[get("/datasets")]
pub async fn list_datasets(app_state: web::Data<AppState>) -> HttpResponse {
    match app_state.dataset_handler.list_datasets().await {
        Ok(datasets) => HttpResponse::Ok().json(json!({ "datasets": datasets })),
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to list datasets: {}", e)
        })),
    }
}

/// Get information about a specific dataset
#[utoipa::path(
    get,
    path = "/api/v1/datasets/{dataset_id}",
    tag = "Datasets",
    params(
        ("dataset_id" = String, Path, description = "Dataset unique identifier")
    ),
    responses(
        (status = 200, description = "Dataset information", body = DatasetInfoResponse),
        (status = 500, description = "Internal server error")
    )
)]
#[get("/datasets/{dataset_id}")]
pub async fn get_dataset_info(
    path: web::Path<String>,
    app_state: web::Data<AppState>,
) -> HttpResponse {
    let dataset_id = path.into_inner();
    
    match app_state.dataset_handler.get_dataset_info(&dataset_id).await {
        Ok(info) => HttpResponse::Ok().json(json!({ "dataset": info })),
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to get dataset info: {}", e)
        })),
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct DeleteResponse {
    /// Operation status
    pub status: String,
    /// Message describing the result
    pub message: String,
}

/// Delete a dataset and its vectors
#[utoipa::path(
    delete,
    path = "/api/v1/datasets/{dataset_id}",
    tag = "Datasets",
    params(
        ("dataset_id" = String, Path, description = "Dataset unique identifier to delete")
    ),
    responses(
        (status = 200, description = "Dataset deleted successfully", body = DeleteResponse),
        (status = 500, description = "Internal server error")
    )
)]
#[delete("/datasets/{dataset_id}")]
pub async fn delete_dataset(
    path: web::Path<String>,
    app_state: web::Data<AppState>,
) -> HttpResponse {
    let dataset_id = path.into_inner();
    
    // Delete from Qdrant
    if let Err(e) = app_state.qdrant_client.delete_by_dataset_id(&dataset_id).await {
        log::error!("Failed to delete vectors from Qdrant: {}", e);
        return HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to delete vectors: {}", e)
        }));
    }
    
    // Delete dataset files
    if let Err(e) = app_state.dataset_handler.delete_dataset(&dataset_id).await {
        log::warn!("Dataset file deletion warning: {}", e);
    }
    
    HttpResponse::Ok().json(json!({
        "status": "success",
        "message": format!("Dataset {} deleted successfully", dataset_id)
    }))
}

fn is_valid_file_type(filename: &str) -> bool {
    let ext = filename.split('.').last().unwrap_or("").to_lowercase();
    matches!(ext.as_str(), "csv" | "json" | "parquet")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_file_type_csv() {
        assert!(is_valid_file_type("data.csv"));
        assert!(is_valid_file_type("data.CSV"));
        assert!(is_valid_file_type("my_file.data.csv"));
    }

    #[test]
    fn test_is_valid_file_type_json() {
        assert!(is_valid_file_type("data.json"));
        assert!(is_valid_file_type("data.JSON"));
        assert!(is_valid_file_type("config.data.json"));
    }

    #[test]
    fn test_is_valid_file_type_parquet() {
        assert!(is_valid_file_type("data.parquet"));
        assert!(is_valid_file_type("data.PARQUET"));
        assert!(is_valid_file_type("table.data.parquet"));
    }

    #[test]
    fn test_is_valid_file_type_invalid() {
        assert!(!is_valid_file_type("data.txt"));
        assert!(!is_valid_file_type("data.xlsx"));
        assert!(!is_valid_file_type("data.xml"));
        assert!(!is_valid_file_type("noextension"));
        assert!(!is_valid_file_type(""));
    }

    #[test]
    fn test_upload_response_serialization() {
        let response = UploadResponse {
            status: "success".to_string(),
            message: "File uploaded".to_string(),
            dataset_id: "test-uuid-123".to_string(),
            job_id: None,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("success"));
        assert!(json.contains("File uploaded"));
        assert!(json.contains("test-uuid-123"));
        assert!(!json.contains("job_id"));
    }

    #[test]
    fn test_upload_response_with_job_id() {
        let response = UploadResponse {
            status: "accepted".to_string(),
            message: "Processing in background".to_string(),
            dataset_id: "ds-1".to_string(),
            job_id: Some("job-abc".to_string()),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("accepted"));
        assert!(json.contains("job-abc"));
    }

    #[test]
    fn test_dataset_list_response_serialization() {
        let response = DatasetListResponse {
            datasets: vec![
                serde_json::json!({"id": "dataset1"}),
                serde_json::json!({"id": "dataset2"}),
            ],
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("dataset1"));
        assert!(json.contains("dataset2"));
    }

    #[test]
    fn test_dataset_info_response_serialization() {
        let response = DatasetInfoResponse {
            dataset: serde_json::json!({
                "id": "test-dataset",
                "row_count": 100
            }),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("test-dataset"));
        assert!(json.contains("100"));
    }

    #[test]
    fn test_upload_params_defaults() {
        let json = "{}";
        let params: UploadParams = serde_json::from_str(json).unwrap();
        assert!(params.background.is_none());
        assert!(params.dataset_id.is_none());
        assert!(params.text_column.is_none());
    }
}