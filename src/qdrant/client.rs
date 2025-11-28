use anyhow::Result;
use qdrant_client;
use qdrant_client::qdrant::{Query, QueryPointsBuilder};
use qdrant_client::qdrant::{
    PointStruct, SearchPoints, UpsertPointsBuilder, Distance,
    WithPayloadSelector, with_payload_selector::SelectorOptions,
    CreateCollectionBuilder, VectorParamsBuilder, VectorInput,
    point_id::PointIdOptions
};
use qdrant_client::Qdrant;
use serde_json::Value as JsonValue;
use std::string::{String, ToString};


pub struct QdrantClient {
    client: Qdrant,
    collection_name: String,
}

impl QdrantClient {
    pub async fn new() -> Self {
        let host = std::env::var("QDRANT_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port = std::env::var("QDRANT_GRPC_PORT").unwrap_or_else(|_| "6337".to_string());
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
    
    pub async fn create_collection(&self, vector_size: u64) -> Result<()> {
        self.client.create_collection(
            CreateCollectionBuilder::new(self.collection_name.clone())
                .vectors_config(VectorParamsBuilder::new(vector_size, Distance::Dot)),
        )
        .await?;
        
        Ok(())
    }
    
    pub async fn upsert_points(&self, points: Vec<PointStruct>) -> Result<()> {
        self.client.upsert_points(
            UpsertPointsBuilder::new(self.collection_name.clone(), points).wait(true),
        ).await?;
        Ok(())
    }
    
    pub async fn search(
        &self,
        vector: Vec<f32>,
        limit: u32,
        filter: Option<JsonValue>,
        dataset_id: Option<String>,
    ) -> Result<Vec<SearchResult>> {
        let mut must_conditions = Vec::new();
        
        // Add dataset filter if provided
        if let Some(dataset_id) = dataset_id {
            must_conditions.push(
                qdrant_client::qdrant::Condition {
                    condition_one_of: Some(
                        qdrant_client::qdrant::condition::ConditionOneOf::Field(
                            qdrant_client::qdrant::FieldCondition {
                                key: "dataset_id".to_string(),
                                r#match: Some(
                                    qdrant_client::qdrant::Match {
                                        match_value: Some(
                                            qdrant_client::qdrant::r#match::MatchValue::Keyword(
                                                dataset_id
                                            )
                                        )
                                    }
                                ),
                                ..Default::default()
                            }
                        )
                    )
                }
            );
        }
        
        // Add custom filters if provided
        if let Some(filter_value) = filter {
            if let JsonValue::Object(filter_map) = filter_value {
                for (key, value) in filter_map {
                    if let JsonValue::String(str_value) = value {
                        must_conditions.push(
                            qdrant_client::qdrant::Condition {
                                condition_one_of: Some(
                                    qdrant_client::qdrant::condition::ConditionOneOf::Field(
                                        qdrant_client::qdrant::FieldCondition {
                                            key: key.clone(),
                                            r#match: Some(
                                                qdrant_client::qdrant::Match {
                                                    match_value: Some(
                                                        qdrant_client::qdrant::r#match::MatchValue::Keyword(
                                                            str_value
                                                        )
                                                    )
                                                }
                                            ),
                                            ..Default::default()
                                        }
                                    )
                                )
                            }
                        );
                    }
                }
            }
        }
        
        let filter = if must_conditions.is_empty() {
            None
        } else {
            Some(qdrant_client::qdrant::Filter {
                must: must_conditions,
                ..Default::default()
            })
        };
        
        let payload_selector = Some(WithPayloadSelector {
            selector_options: Some(SelectorOptions::Enable(true)),
        });
        
        let search_points = SearchPoints {
            collection_name: self.collection_name.clone(),
            vector,
            filter,
            limit: u64::from(limit),
            with_payload: payload_selector,
            ..Default::default()
        };
        
        let search_result = self.client.query(
            QueryPointsBuilder::new(self.collection_name.clone())
            .query(Query::new_nearest(VectorInput::from(search_points.vector))),
        ).await?;
         
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
}

pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub payload: JsonValue,
}