//! Recommendation ("predict") endpoint.

use actix_web::{post, web};

use crate::error::{AppResult, ErrorBody};
use crate::models::{RecommendRequest, RecommendResponse};
use crate::state::AppState;

/// Get recommendations.
///
/// Combine any of:
/// * `query` — free text in any language the model supports;
/// * `item_ids` / `negative_item_ids` — "more like these, less like those";
/// * `user_id` — personalize from recorded interactions (already seen items are excluded).
///
/// Narrow results with a Qdrant-style `filter`, page with `limit`/`offset`, and
/// trade relevance for variety with `diversity`.
#[utoipa::path(
    tag = "Recommendations",
    params(("dataset_id" = String, Path, description = "Dataset id")),
    request_body = RecommendRequest,
    responses(
        (status = 200, body = RecommendResponse),
        (status = 400, body = ErrorBody),
        (status = 404, description = "Unknown dataset, item or user", body = ErrorBody)
    )
)]
#[post("/datasets/{dataset_id}/recommend")]
pub async fn recommend(
    state: web::Data<AppState>,
    path: web::Path<String>,
    body: web::Json<RecommendRequest>,
) -> AppResult<web::Json<RecommendResponse>> {
    Ok(web::Json(
        state
            .recommender
            .recommend(&path, body.into_inner())
            .await?,
    ))
}
