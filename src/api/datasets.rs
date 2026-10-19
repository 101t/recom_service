//! Dataset endpoints.

use actix_web::http::header::LOCATION;
use actix_web::{HttpResponse, delete, get, post, web};

use super::upload::job_response;
use crate::error::{AppResult, ErrorBody};
use crate::models::{CreateDatasetRequest, Dataset, DatasetList, IngestResponse, TrainParams};
use crate::state::AppState;

/// Create an empty dataset with an explicit configuration.
///
/// Optional: uploading items to a dataset that does not exist creates it with
/// auto-detected columns.
#[utoipa::path(
    tag = "Datasets",
    request_body = CreateDatasetRequest,
    responses(
        (status = 201, description = "Dataset created", body = Dataset),
        (status = 400, body = ErrorBody),
        (status = 409, description = "Dataset already exists", body = ErrorBody)
    )
)]
#[post("/datasets")]
pub async fn create_dataset(
    state: web::Data<AppState>,
    body: web::Json<CreateDatasetRequest>,
) -> AppResult<HttpResponse> {
    let dataset = state.datasets.create(body.into_inner()).await?;
    Ok(HttpResponse::Created()
        .insert_header((LOCATION, format!("/api/v1/datasets/{}", dataset.id)))
        .json(dataset))
}

/// List datasets.
#[utoipa::path(tag = "Datasets", responses((status = 200, body = DatasetList)))]
#[get("/datasets")]
pub async fn list_datasets(state: web::Data<AppState>) -> AppResult<web::Json<DatasetList>> {
    Ok(web::Json(DatasetList {
        datasets: state.datasets.list().await?,
    }))
}

/// Get a dataset, including training status and progress.
#[utoipa::path(
    tag = "Datasets",
    params(("dataset_id" = String, Path, description = "Dataset id")),
    responses((status = 200, body = Dataset), (status = 404, body = ErrorBody))
)]
#[get("/datasets/{dataset_id}")]
pub async fn get_dataset(
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> AppResult<web::Json<Dataset>> {
    Ok(web::Json(state.datasets.get(&path).await?))
}

/// Delete a dataset with all its vectors, user profiles and stored data.
#[utoipa::path(
    tag = "Datasets",
    params(("dataset_id" = String, Path, description = "Dataset id")),
    responses((status = 204, description = "Deleted"), (status = 404, body = ErrorBody))
)]
#[delete("/datasets/{dataset_id}")]
pub async fn delete_dataset(
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> AppResult<HttpResponse> {
    state.datasets.delete(&path).await?;
    Ok(HttpResponse::NoContent().finish())
}

/// Retrain: rebuild the dataset's index from its stored uploads and interactions.
///
/// Use after changing the embedding model, or to compact user profiles.
#[utoipa::path(
    tag = "Datasets",
    params(("dataset_id" = String, Path, description = "Dataset id"), TrainParams),
    responses(
        (status = 202, description = "Training started", body = IngestResponse),
        (status = 200, description = "Training finished (`wait=true`)", body = IngestResponse),
        (status = 404, body = ErrorBody)
    )
)]
#[post("/datasets/{dataset_id}/train")]
pub async fn train_dataset(
    state: web::Data<AppState>,
    path: web::Path<String>,
    params: web::Query<TrainParams>,
) -> AppResult<HttpResponse> {
    let accepted = state
        .datasets
        .train(&path, params.reset.unwrap_or(false))
        .await?;
    job_response(&state, accepted, params.wait.unwrap_or(false)).await
}
