use actix_web::{post, web, HttpResponse};
use serde::{Deserialize, Serialize};
use serde_json::json;
use utoipa::ToSchema;

use crate::AppState;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommendation_request_deserialization() {
        let json = r#"{
            "query": "find similar products",
            "dataset_id": "dataset-123",
            "limit": 5
        }"#;
        let request: RecommendationRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.query, "find similar products");
        assert_eq!(request.dataset_id, Some("dataset-123".to_string()));
        assert_eq!(request.limit, Some(5));
        assert!(request.filters.is_none());
    }

    #[test]
    fn test_recommendation_request_minimal() {
        let json = r#"{"query": "test query"}"#;
        let request: RecommendationRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.query, "test query");
        assert!(request.dataset_id.is_none());
        assert!(request.limit.is_none());
        assert!(request.filters.is_none());
    }

    #[test]
    fn test_recommendation_request_with_filters() {
        let json = r#"{
            "query": "search",
            "filters": {"category": "electronics", "brand": "apple"}
        }"#;
        let request: RecommendationRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.query, "search");
        let filters = request.filters.unwrap();
        assert_eq!(filters["category"], "electronics");
        assert_eq!(filters["brand"], "apple");
    }

    #[test]
    fn test_recommendation_serialization() {
        let recommendation = Recommendation {
            id: "item-123".to_string(),
            score: 0.95,
            payload: json!({"name": "Test Product", "price": 99.99}),
        };
        let json = serde_json::to_string(&recommendation).unwrap();
        assert!(json.contains("item-123"));
        assert!(json.contains("0.95"));
        assert!(json.contains("Test Product"));
    }

    #[test]
    fn test_recommendation_response_serialization() {
        let response = RecommendationResponse {
            recommendations: vec![
                Recommendation {
                    id: "1".to_string(),
                    score: 0.9,
                    payload: json!({"name": "Product A"}),
                },
                Recommendation {
                    id: "2".to_string(),
                    score: 0.8,
                    payload: json!({"name": "Product B"}),
                },
            ],
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("Product A"));
        assert!(json.contains("Product B"));
        assert!(json.contains("0.9"));
        assert!(json.contains("0.8"));
    }

    #[test]
    fn test_error_response_serialization() {
        let error = ErrorResponse {
            error: "Something went wrong".to_string(),
        };
        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("Something went wrong"));
    }

    #[test]
    fn test_default_limit() {
        let json = r#"{"query": "test"}"#;
        let request: RecommendationRequest = serde_json::from_str(json).unwrap();
        let limit = request.limit.unwrap_or(10);
        assert_eq!(limit, 10);
    }
}