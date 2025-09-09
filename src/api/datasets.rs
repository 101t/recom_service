use actix_multipart::Multipart;
use actix_web::{post, web, HttpResponse, get};
use futures_util::{stream::StreamExt, TryStreamExt};
use serde_json::json;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::AppState;

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

        
        // Process the uploaded file
        match app_state.delta_handler.process_file(&filepath).await {
            Ok(dataset_id) => {
                return HttpResponse::Ok().json(json!({
                    "status": "success",
                    "message": "File uploaded and processed successfully",
                    "dataset_id": dataset_id
                }));
            }
            Err(e) => {
                return HttpResponse::InternalServerError().json(json!({
                    "error": format!("Failed to process file: {}", e)
                }));
            }
        }
    }
    
    HttpResponse::BadRequest().json(json!({
        "error": "No valid file found in request"
    }))
}

#[get("/datasets")]
pub async fn list_datasets(app_state: web::Data<AppState>) -> HttpResponse {
    match app_state.delta_handler.list_datasets().await {
        Ok(datasets) => HttpResponse::Ok().json(json!({ "datasets": datasets })),
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to list datasets: {}", e)
        })),
    }
}

#[get("/datasets/{dataset_id}")]
pub async fn get_dataset_info(
    path: web::Path<String>,
    app_state: web::Data<AppState>,
) -> HttpResponse {
    let dataset_id = path.into_inner();
    
    match app_state.delta_handler.get_dataset_info(&dataset_id).await {
        Ok(info) => HttpResponse::Ok().json(json!({ "dataset": info })),
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to get dataset info: {}", e)
        })),
    }
}

fn is_valid_file_type(filename: &str) -> bool {
    let ext = filename.split('.').last().unwrap_or("").to_lowercase();
    matches!(ext.as_str(), "csv" | "json" | "parquet")
}