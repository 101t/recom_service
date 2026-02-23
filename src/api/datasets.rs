use actix_multipart::Multipart;
use actix_web::{post, web, HttpResponse, get, delete};
use futures_util::{stream::StreamExt, TryStreamExt};
use serde::Serialize;
use serde_json::json;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use utoipa::ToSchema;
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

/// Upload a dataset file (CSV, JSON, or Parquet)
#[utoipa::path(
    post,
    path = "/api/v1/datasets/upload",
    tag = "Datasets",
    responses(
        (status = 200, description = "File uploaded successfully", body = UploadResponse),
        (status = 400, description = "Invalid file type"),
        (status = 500, description = "Internal server error")
    )
)]
#[post("/datasets/upload")]
pub async fn upload_dataset(
    mut payload: Multipart,
    app_state: web::Data<AppState>,
) -> HttpResponse {
    // Iterate over multipart stream
    while let Ok(Some(mut field)) = payload.try_next().await {
        let filename = field
            .content_disposition()
            .and_then(|cd| cd.get_filename())
            .unwrap_or("unknown");

        if !is_valid_file_type(filename) {
            return HttpResponse::BadRequest().json(
                json!({"error": "Invalid file type. Supported types: CSV, JSON, Parquet"}),
            );
        }
        
        // Create file path
        let file_id = Uuid::new_v4();
        let filepath = format!("/tmp/{}_{}", file_id, filename);
        
        let mut f = File::create(&filepath).await.map_err(|e| {
            HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to create file: {}", e)
            }))
        }).unwrap();

        while let Some(chunk) = field.next().await {
            let data = chunk.unwrap();
            if let Err(e) = f.write_all(&data).await {
                return HttpResponse::InternalServerError().json(json!({
                    "error": format!("Failed to write to file: {}", e)
                }));
            }
        }

        
        // Process the uploaded file (None = new dataset, None = auto-detect text column)
        let processed = match app_state.dataset_handler.process_file(&filepath, None, None).await {
            Ok(data) => data,
            Err(e) => {
                return HttpResponse::InternalServerError().json(json!({
                    "error": format!("Failed to process file: {}", e)
                }));
            }
        };
        
        // Embed the texts and store in Qdrant
        if !processed.texts.is_empty() {
            // Ensure collection exists (384 is typical embedding dimension)
            if let Err(e) = app_state.qdrant_client.ensure_collection(384).await {
                log::warn!("Collection setup warning: {}", e);
            }
            
            // Generate embeddings for all texts
            match app_state.embedder.batch_embed(processed.texts.clone()).await {
                Ok(vectors) => {
                    // Store in Qdrant
                    if let Err(e) = app_state.qdrant_client
                        .upsert_dataset_vectors(&processed.dataset_id, processed.texts.clone(), vectors)
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
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("success"));
        assert!(json.contains("File uploaded"));
        assert!(json.contains("test-uuid-123"));
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
}