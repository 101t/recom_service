use actix_web::{get, web, HttpResponse};
use serde::Deserialize;
use serde_json::json;
use utoipa::IntoParams;

use crate::AppState;

#[derive(Debug, Deserialize, IntoParams)]
pub struct PredictionLogQuery {
    /// Filter by date (YYYY-MM-DD)
    pub date: Option<String>,
    /// Maximum number of logs to return (default: 50)
    pub limit: Option<usize>,
}

/// List prediction logs
#[utoipa::path(
    get,
    path = "/api/v1/predictions",
    tag = "Predictions",
    params(PredictionLogQuery),
    responses(
        (status = 200, description = "Prediction logs retrieved",
         body = crate::data::prediction_logger::PredictionLogListResponse),
        (status = 500, description = "Internal server error")
    )
)]
#[get("/predictions")]
pub async fn list_prediction_logs(
    query: web::Query<PredictionLogQuery>,
    app_state: web::Data<AppState>,
) -> HttpResponse {
    let limit = query.limit.unwrap_or(50);
    let date_filter = query.date.as_deref();

    match app_state
        .prediction_logger
        .list_logs(date_filter, limit)
        .await
    {
        Ok(logs) => {
            let total = logs.len();
            HttpResponse::Ok().json(json!({
                "logs": logs,
                "total": total
            }))
        }
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to list prediction logs: {}", e)
        })),
    }
}

/// Export prediction logs to Parquet
#[utoipa::path(
    get,
    path = "/api/v1/predictions/export",
    tag = "Predictions",
    responses(
        (status = 200, description = "Export completed successfully"),
        (status = 500, description = "Failed to export")
    )
)]
#[get("/predictions/export")]
pub async fn export_prediction_logs(app_state: web::Data<AppState>) -> HttpResponse {
    let output_path = format!("/tmp/prediction_export_{}.parquet", uuid::Uuid::new_v4());

    match app_state
        .prediction_logger
        .export_to_parquet(&output_path)
        .await
    {
        Ok(path) => HttpResponse::Ok().json(json!({
            "status": "success",
            "message": "Prediction logs exported",
            "file_path": path
        })),
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to export prediction logs: {}", e)
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_log_query_defaults() {
        let json = "{}";
        let query: PredictionLogQuery = serde_json::from_str(json).unwrap();
        assert!(query.date.is_none());
        assert!(query.limit.is_none());
        assert_eq!(query.limit.unwrap_or(50), 50);
    }

    #[test]
    fn test_prediction_log_query_with_values() {
        let json = r#"{"date": "2025-06-15", "limit": 100}"#;
        let query: PredictionLogQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.date.as_deref(), Some("2025-06-15"));
        assert_eq!(query.limit, Some(100));
    }
}
