//! Prediction log endpoints.

use actix_web::http::header::{ContentDisposition, DispositionParam, DispositionType};
use actix_web::{HttpResponse, get, web};

use crate::error::{AppError, AppResult, ErrorBody};
use crate::models::{PredictionLogList, PredictionQuery};
use crate::state::AppState;

const DEFAULT_LIST_LIMIT: usize = 50;
const MAX_LIST_LIMIT: usize = 1_000;

/// List logged predictions, most recent first.
#[utoipa::path(
    tag = "Predictions",
    params(PredictionQuery),
    responses((status = 200, body = PredictionLogList), (status = 400, body = ErrorBody))
)]
#[get("/predictions")]
pub async fn list_predictions(
    state: web::Data<AppState>,
    query: web::Query<PredictionQuery>,
) -> AppResult<web::Json<PredictionLogList>> {
    let limit = query.limit.unwrap_or(DEFAULT_LIST_LIMIT);
    if limit == 0 || limit > MAX_LIST_LIMIT {
        return Err(AppError::bad_request(format!(
            "limit must be between 1 and {MAX_LIST_LIMIT}"
        )));
    }
    let logs = state.predictions.list(&query, limit).await?;
    Ok(web::Json(PredictionLogList {
        total: logs.len(),
        logs,
    }))
}

/// Download logged predictions as JSONL, CSV or Parquet (for offline evaluation).
#[utoipa::path(
    tag = "Predictions",
    params(PredictionQuery),
    responses(
        (status = 200, description = "File download", content_type = "application/octet-stream"),
        (status = 400, body = ErrorBody)
    )
)]
#[get("/predictions/export")]
pub async fn export_predictions(
    state: web::Data<AppState>,
    query: web::Query<PredictionQuery>,
) -> AppResult<HttpResponse> {
    let export = state.predictions.export(&query).await?;
    Ok(HttpResponse::Ok()
        .content_type(export.content_type)
        .insert_header(ContentDisposition {
            disposition: DispositionType::Attachment,
            parameters: vec![DispositionParam::Filename(export.filename)],
        })
        .body(export.body))
}
