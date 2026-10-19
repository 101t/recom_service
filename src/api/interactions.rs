//! Interaction endpoints (personalization signals).

use actix_multipart::Multipart;
use actix_web::{HttpResponse, post, web};

use super::upload::{UploadForm, WaitParams, job_response, receive_file};
use crate::error::{AppResult, ErrorBody};
use crate::models::{IngestResponse, RecordInteractionsRequest, RecordInteractionsResponse};
use crate::state::AppState;

/// Record user interactions (views, clicks, purchases, ratings, ...).
///
/// Profiles are updated immediately, so the next personalized recommendation
/// already reflects them. Up to 10 000 interactions per request.
#[utoipa::path(
    tag = "Interactions",
    params(("dataset_id" = String, Path, description = "Dataset id")),
    request_body = RecordInteractionsRequest,
    responses(
        (status = 200, body = RecordInteractionsResponse),
        (status = 400, body = ErrorBody),
        (status = 404, body = ErrorBody)
    )
)]
#[post("/datasets/{dataset_id}/interactions")]
pub async fn record_interactions(
    state: web::Data<AppState>,
    path: web::Path<String>,
    body: web::Json<RecordInteractionsRequest>,
) -> AppResult<web::Json<RecordInteractionsResponse>> {
    Ok(web::Json(
        state
            .datasets
            .record_interactions(&path, body.into_inner().interactions)
            .await?,
    ))
}

/// Upload historical interactions (CSV, TSV, JSON, NDJSON or Parquet).
///
/// Needs a user column (`user_id`, `user`, `customer_id`, ...) and an item column
/// (`item_id`, `product_id`, `sku`, ...); `weight`/`rating`, `event_type` and
/// `timestamp` are optional. Processed in the background unless `wait=true`.
#[utoipa::path(
    tag = "Interactions",
    params(("dataset_id" = String, Path, description = "Dataset id"), WaitParams),
    request_body(content = UploadForm, content_type = "multipart/form-data"),
    responses(
        (status = 202, description = "Upload accepted", body = IngestResponse),
        (status = 200, description = "Upload applied (`wait=true`)", body = IngestResponse),
        (status = 400, body = ErrorBody),
        (status = 404, body = ErrorBody)
    )
)]
#[post("/datasets/{dataset_id}/interactions/upload")]
pub async fn upload_interactions(
    state: web::Data<AppState>,
    path: web::Path<String>,
    params: web::Query<WaitParams>,
    payload: Multipart,
) -> AppResult<HttpResponse> {
    let upload = receive_file(payload, &state).await?;
    let accepted = state.datasets.upload_interactions(&path, upload).await?;
    job_response(&state, accepted, params.wait.unwrap_or(false)).await
}
