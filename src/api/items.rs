//! Item endpoints.

use actix_multipart::Multipart;
use actix_web::{HttpResponse, delete, get, post, web};

use super::upload::{UploadForm, job_response, receive_file};
use crate::error::{AppResult, ErrorBody};
use crate::models::{
    IngestResponse, ItemResponse, UploadParams, UpsertItemsRequest, UpsertItemsResponse,
};
use crate::state::AppState;

/// Add or update items (synchronous; up to 10 000 per request).
///
/// Items are JSON objects with the same columns as file uploads. Creates the
/// dataset when it does not exist.
#[utoipa::path(
    tag = "Items",
    params(("dataset_id" = String, Path, description = "Dataset id")),
    request_body = UpsertItemsRequest,
    responses((status = 200, body = UpsertItemsResponse), (status = 400, body = ErrorBody))
)]
#[post("/datasets/{dataset_id}/items")]
pub async fn upsert_items(
    state: web::Data<AppState>,
    path: web::Path<String>,
    body: web::Json<UpsertItemsRequest>,
) -> AppResult<web::Json<UpsertItemsResponse>> {
    Ok(web::Json(
        state
            .datasets
            .upsert_items(&path, body.into_inner().items)
            .await?,
    ))
}

/// Upload a catalog file (CSV, TSV, JSON, NDJSON or Parquet) and index it ("train").
///
/// Processing runs in the background; poll `GET /api/v1/datasets/{dataset_id}` or
/// pass `wait=true`. Creates the dataset when it does not exist.
#[utoipa::path(
    tag = "Items",
    params(("dataset_id" = String, Path, description = "Dataset id"), UploadParams),
    request_body(content = UploadForm, content_type = "multipart/form-data"),
    responses(
        (status = 202, description = "Upload accepted", body = IngestResponse),
        (status = 200, description = "Upload indexed (`wait=true`)", body = IngestResponse),
        (status = 400, body = ErrorBody),
        (status = 413, body = ErrorBody),
        (status = 415, body = ErrorBody)
    )
)]
#[post("/datasets/{dataset_id}/items/upload")]
pub async fn upload_items(
    state: web::Data<AppState>,
    path: web::Path<String>,
    params: web::Query<UploadParams>,
    payload: Multipart,
) -> AppResult<HttpResponse> {
    let upload = receive_file(payload, &state).await?;
    let accepted = state.datasets.upload_items(&path, upload, &params).await?;
    job_response(&state, accepted, params.wait.unwrap_or(false)).await
}

/// Get an indexed item.
#[utoipa::path(
    tag = "Items",
    params(
        ("dataset_id" = String, Path, description = "Dataset id"),
        ("item_id" = String, Path, description = "Item id")
    ),
    responses((status = 200, body = ItemResponse), (status = 404, body = ErrorBody))
)]
#[get("/datasets/{dataset_id}/items/{item_id}")]
pub async fn get_item(
    state: web::Data<AppState>,
    path: web::Path<(String, String)>,
) -> AppResult<web::Json<ItemResponse>> {
    let (dataset_id, item_id) = path.into_inner();
    Ok(web::Json(
        state.datasets.get_item(&dataset_id, &item_id).await?,
    ))
}

/// Remove an item from the dataset.
#[utoipa::path(
    tag = "Items",
    params(
        ("dataset_id" = String, Path, description = "Dataset id"),
        ("item_id" = String, Path, description = "Item id")
    ),
    responses((status = 204, description = "Deleted"), (status = 404, body = ErrorBody))
)]
#[delete("/datasets/{dataset_id}/items/{item_id}")]
pub async fn delete_item(
    state: web::Data<AppState>,
    path: web::Path<(String, String)>,
) -> AppResult<HttpResponse> {
    let (dataset_id, item_id) = path.into_inner();
    state.datasets.delete_item(&dataset_id, &item_id).await?;
    Ok(HttpResponse::NoContent().finish())
}
