use actix_web::{post, web, HttpResponse};
use serde::{Deserialize, Serialize};
use serde_json::json;
use utoipa::ToSchema;

use crate::{embedding::Embedder, AppState};

#[derive(Debug, Deserialize, ToSchema)]
pub struct RecommendationRequest {
    /// Search query text
    pub query: String,
    /// Optional dataset ID to search within
    pub dataset_id: Option<String>,
    /// Maximum number of results (default: 10)
    pub limit: Option<u32>,
    /// Optional filters for the search
    pub filters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct Recommendation {
    /// Unique identifier
    pub id: String,
    /// Similarity score
    pub score: f32,
    /// Additional payload data
    pub payload: serde_json::Value,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RecommendationResponse {
    /// List of recommendations
    pub recommendations: Vec<Recommendation>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    /// Error message
    pub error: String,
}

/// Get recommendations based on a query
#[utoipa::path(
    post,
    path = "/api/v1/recommend",
    tag = "Recommendations",
    request_body = RecommendationRequest,
    responses(
        (status = 200, description = "Recommendations retrieved successfully", body = RecommendationResponse),
        (status = 500, description = "Internal server error", body = ErrorResponse)
    )
)]
#[post("/recommend")]
pub async fn get_recommendations(
    req: web::Json<RecommendationRequest>,
    app_state: web::Data<AppState>,
) -> HttpResponse {
    // Generate embedding for the query
    let embedding = match (*app_state.embedder).embed(&req.query).await {
        Ok(emb) => emb,
        Err(e) => {
            return HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to generate embedding: {}", e)
            }))
        }
    };
    
    // Get recommendations from Qdrant
    let limit = req.limit.unwrap_or(10);
    let filters = req.filters.clone();
    let dataset_id = req.dataset_id.clone();
    
    match app_state.qdrant_client.search(embedding, limit, filters, dataset_id).await {
        Ok(results) => {
            let recommendations: Vec<Recommendation> = results
                .into_iter()
                .map(|point| Recommendation {
                    id: point.id,
                    score: point.score,
                    payload: point.payload,
                })
                .collect();
                
            HttpResponse::Ok().json(json!({ "recommendations": recommendations }))
        }
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to get recommendations: {}", e)
        })),
    }
}