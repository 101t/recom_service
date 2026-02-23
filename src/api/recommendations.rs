use actix_web::{post, web, HttpResponse};
use serde::{Deserialize, Serialize};
use serde_json::json;
use utoipa::ToSchema;

use crate::AppState;
use crate::qdrant::client::QdrantFilter;

#[derive(Debug, Deserialize, ToSchema)]
pub struct RecommendationRequest {
    /// Search query text
    pub query: String,
    /// Optional dataset ID to search within
    pub dataset_id: Option<String>,
    /// Maximum number of results (default: 10)
    pub limit: Option<u32>,
    /// Optional simple key-value filters (legacy, backward-compatible)
    pub filters: Option<serde_json::Value>,
    /// Full Qdrant-style filter with must / must_not / should / min_should_count
    pub filter: Option<QdrantFilter>,
    /// Minimum score threshold — only return results above this score
    pub score_threshold: Option<f32>,
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
    /// Query latency in milliseconds
    pub latency_ms: u64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    /// Error message
    pub error: String,
}

/// Get recommendations based on a query
///
/// Supports full Qdrant-style filtering:
/// - **must**: all conditions must match (AND)
/// - **must_not**: none of these conditions may match (NOT)
/// - **should**: at least one condition should match (OR)
/// - **min_should_count**: minimum number of should-conditions that must match
///
/// Each condition can be a **match** (exact keyword/integer/boolean) or a **range** (gt/gte/lt/lte).
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
    let start = std::time::Instant::now();

    // Generate embedding for the query
    let embedding = match (*app_state.embedder).embed(&req.query).await {
        Ok(emb) => emb,
        Err(e) => {
            return HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to generate embedding: {}", e)
            }))
        }
    };
    
    let limit = req.limit.unwrap_or(10);
    let dataset_id = req.dataset_id.clone();
    let score_threshold = req.score_threshold;

    // Determine which filter to use: new-style `filter` takes precedence
    let qdrant_filter = if req.filter.is_some() {
        req.filter.clone()
    } else if let Some(ref legacy) = req.filters {
        // Convert legacy flat filter → QdrantFilter with must conditions
        if let serde_json::Value::Object(map) = legacy {
            let must: Vec<crate::qdrant::client::FilterCondition> = map
                .iter()
                .map(|(k, v)| crate::qdrant::client::FilterCondition {
                    key: k.clone(),
                    r#match: Some(crate::qdrant::client::MatchCondition { value: v.clone() }),
                    range: None,
                })
                .collect();
            Some(QdrantFilter {
                must: Some(must),
                must_not: None,
                should: None,
                min_should_count: None,
            })
        } else {
            None
        }
    } else {
        None
    };

    match app_state
        .qdrant_client
        .search_with_filter(embedding, limit, qdrant_filter, dataset_id.clone(), score_threshold)
        .await
    {
        Ok(results) => {
            let recommendations: Vec<Recommendation> = results
                .into_iter()
                .map(|point| Recommendation {
                    id: point.id,
                    score: point.score,
                    payload: point.payload,
                })
                .collect();

            let latency_ms = start.elapsed().as_millis() as u64;

            // Log prediction asynchronously (fire-and-forget)
            let result_ids: Vec<String> = recommendations.iter().map(|r| r.id.clone()).collect();
            let scores: Vec<f32> = recommendations.iter().map(|r| r.score).collect();
            let logger = app_state.prediction_logger.clone();
            let query = req.query.clone();
            let ds_id = dataset_id.clone();
            tokio::spawn(async move {
                if let Err(e) = logger
                    .log_prediction(&query, ds_id.as_deref(), result_ids, scores, latency_ms)
                    .await
                {
                    log::warn!("Failed to log prediction: {}", e);
                }
            });
                
            HttpResponse::Ok().json(json!({
                "recommendations": recommendations,
                "latency_ms": latency_ms
            }))
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
            latency_ms: 42,
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