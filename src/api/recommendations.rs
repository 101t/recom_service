use actix_web::{post, web, HttpResponse};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{embedding::Embedder, AppState};

#[derive(Debug, Deserialize)]
pub struct RecommendationRequest {
    pub query: String,
    pub dataset_id: Option<String>,
    pub limit: Option<u32>,
    pub filters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct Recommendation {
    pub id: String,
    pub score: f32,
    pub payload: serde_json::Value,
}

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