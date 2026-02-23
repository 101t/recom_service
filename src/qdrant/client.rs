use anyhow::Result;
use qdrant_client;
use qdrant_client::qdrant::{Query, QueryPointsBuilder, DeletePointsBuilder};
use qdrant_client::qdrant::{
    PointStruct, UpsertPointsBuilder, Distance,
    CreateCollectionBuilder, VectorParamsBuilder, VectorInput,
    point_id::PointIdOptions,
};
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::string::{String, ToString};
use utoipa::ToSchema;
use uuid::Uuid;


pub struct QdrantClient {
    client: Qdrant,
    collection_name: String,
}

/// Info about a Qdrant collection, returned by the management API
#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct CollectionDetails {
    /// Collection name
    pub name: String,
    /// Number of points (vectors) in the collection
    pub points_count: u64,
    /// Number of indexed vectors
    pub vectors_count: u64,
    /// Current collection status
    pub status: String,
}

/// Request to create a new collection
#[derive(Debug, Deserialize, ToSchema)]
pub struct CreateCollectionRequest {
    /// Collection name
    pub name: String,
    /// Vector dimension size
    pub vector_size: u64,
    /// Distance metric: "dot", "cosine", or "euclid" (default: "cosine")
    pub distance: Option<String>,
}

impl QdrantClient {
    pub async fn new() -> Self {
        let host = std::env::var("QDRANT_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port = std::env::var("QDRANT_PORT").unwrap_or_else(|_| "6336".to_string());
        let api_key = std::env::var("QDRANT_API_KEY").ok();
        let use_https = std::env::var("QDRANT_HTTPS")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(false);
        
        let scheme = if use_https { "https" } else { "http" };
        let url = format!("{}://{}:{}", scheme, host, port);
        
        let mut builder = Qdrant::from_url(&url).skip_compatibility_check();
        
        if let Some(key) = api_key {
            builder = builder.api_key(key);
        }
        
        let client = builder.build().expect("Failed to create Qdrant client");
            
        Self {
            client,
            collection_name: "recommendations".to_string(),
        }
    }

    /// Get the default collection name
    pub fn default_collection(&self) -> &str {
        &self.collection_name
    }
    
    pub async fn create_collection_with_name(
        &self,
        name: &str,
        vector_size: u64,
        distance: Distance,
    ) -> Result<()> {
        self.client
            .create_collection(
                CreateCollectionBuilder::new(name)
                    .vectors_config(VectorParamsBuilder::new(vector_size, distance)),
            )
            .await?;
        Ok(())
    }

    pub async fn create_collection(&self, vector_size: u64) -> Result<()> {
        self.create_collection_with_name(&self.collection_name, vector_size, Distance::Dot)
            .await
    }

    pub async fn delete_collection(&self, name: &str) -> Result<()> {
        self.client
            .delete_collection(name)
            .await?;
        Ok(())
    }

    pub async fn list_collections(&self) -> Result<Vec<String>> {
        let response = self.client.list_collections().await?;
        let names: Vec<String> = response
            .collections
            .into_iter()
            .map(|c| c.name)
            .collect();
        Ok(names)
    }

    pub async fn get_collection_info(&self, name: &str) -> Result<CollectionDetails> {
        let info = self.client.collection_info(name).await?;
        let result = &info.result;
        
        let status_str = match result.as_ref().map(|r| r.status) {
            Some(1) => "green",
            Some(2) => "yellow",
            Some(3) => "red",
            _ => "unknown",
        };

        let points = result.as_ref().and_then(|r| r.points_count).unwrap_or(0);
        let vectors = result.as_ref().and_then(|r| r.indexed_vectors_count).unwrap_or(0);

        Ok(CollectionDetails {
            name: name.to_string(),
            points_count: points,
            vectors_count: vectors,
            status: status_str.to_string(),
        })
    }
    
    pub async fn upsert_points(&self, points: Vec<PointStruct>) -> Result<()> {
        self.client.upsert_points(
            UpsertPointsBuilder::new(self.collection_name.clone(), points).wait(true),
        ).await?;
        Ok(())
    }
    
    /// Upsert vectors with dataset_id and text content as payload
    pub async fn upsert_dataset_vectors(
        &self,
        dataset_id: &str,
        texts: Vec<String>,
        vectors: Vec<Vec<f32>>,
    ) -> Result<()> {
        let points: Vec<PointStruct> = texts
            .into_iter()
            .zip(vectors.into_iter())
            .map(|(text, vector)| {
                let point_id = Uuid::new_v4().to_string();
                let mut payload = HashMap::new();
                payload.insert("dataset_id".to_string(), JsonValue::String(dataset_id.to_string()));
                payload.insert("text".to_string(), JsonValue::String(text));
                
                PointStruct::new(point_id, vector, payload)
            })
            .collect();
        
        if points.is_empty() {
            return Ok(());
        }
        
        self.client.upsert_points(
            UpsertPointsBuilder::new(self.collection_name.clone(), points).wait(true),
        ).await?;
        
        Ok(())
    }
    
    /// Delete all points belonging to a dataset
    pub async fn delete_by_dataset_id(&self, dataset_id: &str) -> Result<()> {
        let filter = qdrant_client::qdrant::Filter {
            must: vec![
                qdrant_client::qdrant::Condition {
                    condition_one_of: Some(
                        qdrant_client::qdrant::condition::ConditionOneOf::Field(
                            qdrant_client::qdrant::FieldCondition {
                                key: "dataset_id".to_string(),
                                r#match: Some(
                                    qdrant_client::qdrant::Match {
                                        match_value: Some(
                                            qdrant_client::qdrant::r#match::MatchValue::Keyword(
                                                dataset_id.to_string()
                                            )
                                        )
                                    }
                                ),
                                ..Default::default()
                            }
                        )
                    )
                }
            ],
            ..Default::default()
        };
        
        self.client.delete_points(
            DeletePointsBuilder::new(self.collection_name.clone())
                .points(filter)
                .wait(true),
        ).await?;
        
        Ok(())
    }
    
    /// Check if collection exists, create if not
    pub async fn ensure_collection(&self, vector_size: u64) -> Result<()> {
        match self.client.collection_exists(&self.collection_name).await {
            Ok(exists) => {
                if !exists {
                    self.create_collection(vector_size).await?;
                    log::info!("Created collection: {}", self.collection_name);
                }
            }
            Err(_) => {
                // Try to create if check fails
                if let Err(e) = self.create_collection(vector_size).await {
                    log::warn!("Collection may already exist: {}", e);
                }
            }
        }
        Ok(())
    }
    
    /// Search with full Qdrant-style filter support (must, must_not, should, min_should)
    pub async fn search_with_filter(
        &self,
        vector: Vec<f32>,
        limit: u32,
        filter: Option<QdrantFilter>,
        dataset_id: Option<String>,
        score_threshold: Option<f32>,
    ) -> Result<Vec<SearchResult>> {
        let mut must_conditions = Vec::new();
        let mut must_not_conditions = Vec::new();
        let mut should_conditions = Vec::new();
        let mut min_should: Option<qdrant_client::qdrant::MinShould> = None;
        
        // Add dataset filter if provided
        if let Some(dataset_id) = dataset_id {
            must_conditions.push(make_keyword_condition("dataset_id", &dataset_id));
        }
        
        // Parse full filter object
        if let Some(f) = filter {
            if let Some(musts) = f.must {
                for fc in musts {
                    if let Some(cond) = filter_condition_to_qdrant(&fc) {
                        must_conditions.push(cond);
                    }
                }
            }
            if let Some(must_nots) = f.must_not {
                for fc in must_nots {
                    if let Some(cond) = filter_condition_to_qdrant(&fc) {
                        must_not_conditions.push(cond);
                    }
                }
            }
            if let Some(shoulds) = f.should {
                for fc in shoulds {
                    if let Some(cond) = filter_condition_to_qdrant(&fc) {
                        should_conditions.push(cond);
                    }
                }
            }
            if let Some(count) = f.min_should_count {
                min_should = Some(qdrant_client::qdrant::MinShould {
                    conditions: should_conditions.clone(),
                    min_count: count,
                });
            }
        }
        
        let filter = if must_conditions.is_empty()
            && must_not_conditions.is_empty()
            && should_conditions.is_empty()
            && min_should.is_none()
        {
            None
        } else {
            Some(qdrant_client::qdrant::Filter {
                must: must_conditions,
                must_not: must_not_conditions,
                should: should_conditions,
                min_should: min_should.map(|ms| ms),
            })
        };
        
        let mut query_builder = QueryPointsBuilder::new(self.collection_name.clone())
            .query(Query::new_nearest(VectorInput::from(vector)))
            .limit(u64::from(limit))
            .with_payload(true);
        
        if let Some(f) = filter {
            query_builder = query_builder.filter(f);
        }
        if let Some(threshold) = score_threshold {
            query_builder = query_builder.score_threshold(threshold);
        }
        
        let search_result = self.client.query(query_builder).await?;
         
        let results: Vec<SearchResult> = search_result
            .result
            .into_iter()
            .map(|point| SearchResult {
                id: match point.id {
                    Some(id) => match id.point_id_options {
                        Some(PointIdOptions::Uuid(id)) => id,
                        Some(PointIdOptions::Num(id)) => id.to_string(),
                        None => "unknown".to_string(),
                    },
                    None => "unknown".to_string(),
                },
                score: point.score,
                payload: serde_json::to_value(point.payload).unwrap_or(JsonValue::Null),
            })
            .collect();
            
        Ok(results)
    }

    /// Legacy search with simple flat filters (backward compatible)
    pub async fn search(
        &self,
        vector: Vec<f32>,
        limit: u32,
        filter: Option<JsonValue>,
        dataset_id: Option<String>,
    ) -> Result<Vec<SearchResult>> {
        // Convert legacy flat filter to QdrantFilter
        let qdrant_filter = filter.map(|fv| {
            let mut must_conditions = Vec::new();
            if let JsonValue::Object(map) = fv {
                for (key, value) in map {
                    must_conditions.push(FilterCondition {
                        key,
                        r#match: Some(MatchCondition { value }),
                        range: None,
                    });
                }
            }
            QdrantFilter {
                must: Some(must_conditions),
                must_not: None,
                should: None,
                min_should_count: None,
            }
        });

        self.search_with_filter(vector, limit, qdrant_filter, dataset_id, None).await
    }
}

// ─── Qdrant filter model (mirrors Qdrant's filter DSL) ─────────────────

/// Full Qdrant-style filter with must / must_not / should / min_should
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct QdrantFilter {
    /// All conditions must match (AND)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub must: Option<Vec<FilterCondition>>,
    /// None of these conditions must match (NOT)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub must_not: Option<Vec<FilterCondition>>,
    /// At least one should match (OR) — or use with min_should_count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub should: Option<Vec<FilterCondition>>,
    /// Minimum number of should-conditions that must match
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_should_count: Option<u64>,
}

/// A single filter condition on a payload field
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct FilterCondition {
    /// Payload field key
    pub key: String,
    /// Match exact value (keyword or integer)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#match: Option<MatchCondition>,
    /// Range filter (gt, gte, lt, lte)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<RangeCondition>,
}

/// Exact match on a value (string → keyword, integer → integer)
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct MatchCondition {
    /// The value to match (string or number)
    pub value: JsonValue,
}

/// Numeric range filter
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RangeCondition {
    /// Greater than
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt: Option<f64>,
    /// Greater than or equal
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gte: Option<f64>,
    /// Less than
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lt: Option<f64>,
    /// Less than or equal
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lte: Option<f64>,
}

pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub payload: JsonValue,
}

// ─── helpers ────────────────────────────────────────────────────────────

fn make_keyword_condition(key: &str, value: &str) -> qdrant_client::qdrant::Condition {
    qdrant_client::qdrant::Condition {
        condition_one_of: Some(
            qdrant_client::qdrant::condition::ConditionOneOf::Field(
                qdrant_client::qdrant::FieldCondition {
                    key: key.to_string(),
                    r#match: Some(qdrant_client::qdrant::Match {
                        match_value: Some(
                            qdrant_client::qdrant::r#match::MatchValue::Keyword(value.to_string()),
                        ),
                    }),
                    ..Default::default()
                },
            ),
        ),
    }
}

fn filter_condition_to_qdrant(fc: &FilterCondition) -> Option<qdrant_client::qdrant::Condition> {
    if let Some(m) = &fc.r#match {
        let match_value = match &m.value {
            JsonValue::String(s) => {
                qdrant_client::qdrant::r#match::MatchValue::Keyword(s.clone())
            }
            JsonValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    qdrant_client::qdrant::r#match::MatchValue::Integer(i)
                } else {
                    return None;
                }
            }
            JsonValue::Bool(b) => {
                qdrant_client::qdrant::r#match::MatchValue::Boolean(*b)
            }
            _ => return None,
        };
        Some(qdrant_client::qdrant::Condition {
            condition_one_of: Some(
                qdrant_client::qdrant::condition::ConditionOneOf::Field(
                    qdrant_client::qdrant::FieldCondition {
                        key: fc.key.clone(),
                        r#match: Some(qdrant_client::qdrant::Match {
                            match_value: Some(match_value),
                        }),
                        ..Default::default()
                    },
                ),
            ),
        })
    } else if let Some(r) = &fc.range {
        Some(qdrant_client::qdrant::Condition {
            condition_one_of: Some(
                qdrant_client::qdrant::condition::ConditionOneOf::Field(
                    qdrant_client::qdrant::FieldCondition {
                        key: fc.key.clone(),
                        range: Some(qdrant_client::qdrant::Range {
                            gt: r.gt,
                            gte: r.gte,
                            lt: r.lt,
                            lte: r.lte,
                        }),
                        ..Default::default()
                    },
                ),
            ),
        })
    } else {
        None
    }
}

/// Parse a Distance string into qdrant Distance enum
pub fn parse_distance(s: &str) -> Distance {
    match s.to_lowercase().as_str() {
        "dot" => Distance::Dot,
        "euclid" | "euclidean" | "l2" => Distance::Euclid,
        "manhattan" => Distance::Manhattan,
        _ => Distance::Cosine,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_search_result_creation() {
        let result = SearchResult {
            id: "test-id-123".to_string(),
            score: 0.95,
            payload: json!({"name": "Test Item", "category": "electronics"}),
        };
        assert_eq!(result.id, "test-id-123");
        assert!((result.score - 0.95).abs() < f32::EPSILON);
        assert_eq!(result.payload["name"], "Test Item");
        assert_eq!(result.payload["category"], "electronics");
    }

    #[test]
    fn test_search_result_with_empty_payload() {
        let result = SearchResult {
            id: "empty-payload".to_string(),
            score: 0.5,
            payload: json!({}),
        };
        assert_eq!(result.id, "empty-payload");
        assert!(result.payload.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_search_result_with_null_payload() {
        let result = SearchResult {
            id: "null-payload".to_string(),
            score: 0.0,
            payload: JsonValue::Null,
        };
        assert!(result.payload.is_null());
    }

    #[test]
    fn test_search_result_with_complex_payload() {
        let result = SearchResult {
            id: "complex".to_string(),
            score: 0.88,
            payload: json!({
                "title": "Product",
                "price": 99.99,
                "tags": ["electronics", "sale"],
                "metadata": {
                    "created_at": "2024-01-01",
                    "updated_at": "2024-06-01"
                }
            }),
        };
        assert_eq!(result.payload["title"], "Product");
        assert_eq!(result.payload["tags"][0], "electronics");
        assert_eq!(result.payload["metadata"]["created_at"], "2024-01-01");
    }

    #[test]
    fn test_search_result_score_boundaries() {
        let min_result = SearchResult {
            id: "min".to_string(),
            score: 0.0,
            payload: json!({}),
        };
        assert_eq!(min_result.score, 0.0);

        let max_result = SearchResult {
            id: "max".to_string(),
            score: 1.0,
            payload: json!({}),
        };
        assert_eq!(max_result.score, 1.0);

        let neg_result = SearchResult {
            id: "neg".to_string(),
            score: -0.5,
            payload: json!({}),
        };
        assert!(neg_result.score < 0.0);
    }

    #[test]
    fn test_parse_distance() {
        assert!(matches!(parse_distance("dot"), Distance::Dot));
        assert!(matches!(parse_distance("cosine"), Distance::Cosine));
        assert!(matches!(parse_distance("Cosine"), Distance::Cosine));
        assert!(matches!(parse_distance("euclid"), Distance::Euclid));
        assert!(matches!(parse_distance("euclidean"), Distance::Euclid));
        assert!(matches!(parse_distance("l2"), Distance::Euclid));
        assert!(matches!(parse_distance("manhattan"), Distance::Manhattan));
        assert!(matches!(parse_distance("unknown"), Distance::Cosine)); // default
    }

    #[test]
    fn test_qdrant_filter_serialization() {
        let filter = QdrantFilter {
            must: Some(vec![FilterCondition {
                key: "category".to_string(),
                r#match: Some(MatchCondition {
                    value: json!("electronics"),
                }),
                range: None,
            }]),
            must_not: Some(vec![FilterCondition {
                key: "status".to_string(),
                r#match: Some(MatchCondition {
                    value: json!("deleted"),
                }),
                range: None,
            }]),
            should: None,
            min_should_count: None,
        };

        let json_str = serde_json::to_string(&filter).unwrap();
        assert!(json_str.contains("must"));
        assert!(json_str.contains("must_not"));
        assert!(json_str.contains("electronics"));
        assert!(json_str.contains("deleted"));
    }

    #[test]
    fn test_qdrant_filter_deserialization() {
        let json = r#"{
            "must": [{"key": "color", "match": {"value": "red"}}],
            "must_not": [{"key": "brand", "match": {"value": "unknown"}}],
            "should": [
                {"key": "price", "range": {"gte": 10.0, "lte": 100.0}},
                {"key": "category", "match": {"value": "shoes"}}
            ],
            "min_should_count": 1
        }"#;
        let filter: QdrantFilter = serde_json::from_str(json).unwrap();
        assert_eq!(filter.must.as_ref().unwrap().len(), 1);
        assert_eq!(filter.must_not.as_ref().unwrap().len(), 1);
        assert_eq!(filter.should.as_ref().unwrap().len(), 2);
        assert_eq!(filter.min_should_count, Some(1));
    }

    #[test]
    fn test_filter_condition_with_range() {
        let json = r#"{"key": "price", "range": {"gte": 10.0, "lte": 99.99}}"#;
        let fc: FilterCondition = serde_json::from_str(json).unwrap();
        assert_eq!(fc.key, "price");
        assert!(fc.r#match.is_none());
        let range = fc.range.unwrap();
        assert_eq!(range.gte, Some(10.0));
        assert_eq!(range.lte, Some(99.99));
        assert!(range.gt.is_none());
        assert!(range.lt.is_none());
    }

    #[test]
    fn test_filter_condition_to_qdrant_keyword() {
        let fc = FilterCondition {
            key: "color".to_string(),
            r#match: Some(MatchCondition { value: json!("red") }),
            range: None,
        };
        let cond = filter_condition_to_qdrant(&fc);
        assert!(cond.is_some());
    }

    #[test]
    fn test_filter_condition_to_qdrant_integer() {
        let fc = FilterCondition {
            key: "count".to_string(),
            r#match: Some(MatchCondition { value: json!(42) }),
            range: None,
        };
        let cond = filter_condition_to_qdrant(&fc);
        assert!(cond.is_some());
    }

    #[test]
    fn test_filter_condition_to_qdrant_boolean() {
        let fc = FilterCondition {
            key: "active".to_string(),
            r#match: Some(MatchCondition { value: json!(true) }),
            range: None,
        };
        let cond = filter_condition_to_qdrant(&fc);
        assert!(cond.is_some());
    }

    #[test]
    fn test_filter_condition_to_qdrant_range() {
        let fc = FilterCondition {
            key: "price".to_string(),
            r#match: None,
            range: Some(RangeCondition {
                gt: Some(0.0),
                gte: None,
                lt: Some(100.0),
                lte: None,
            }),
        };
        let cond = filter_condition_to_qdrant(&fc);
        assert!(cond.is_some());
    }

    #[test]
    fn test_filter_condition_to_qdrant_empty() {
        let fc = FilterCondition {
            key: "empty".to_string(),
            r#match: None,
            range: None,
        };
        let cond = filter_condition_to_qdrant(&fc);
        assert!(cond.is_none());
    }

    #[test]
    fn test_create_collection_request_deserialization() {
        let json = r#"{"name": "my_collection", "vector_size": 768, "distance": "cosine"}"#;
        let req: CreateCollectionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "my_collection");
        assert_eq!(req.vector_size, 768);
        assert_eq!(req.distance.as_deref(), Some("cosine"));
    }

    #[test]
    fn test_collection_details_serialization() {
        let details = CollectionDetails {
            name: "test".to_string(),
            points_count: 1000,
            vectors_count: 1000,
            status: "green".to_string(),
        };
        let json = serde_json::to_string(&details).unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("1000"));
        assert!(json.contains("green"));
    }
}