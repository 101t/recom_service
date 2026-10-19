//! Qdrant backend (gRPC, `qdrant-client` 1.19).
//!
//! Layout:
//! * `{prefix}_items` — one point per item; id = UUIDv5(dataset, item_id).
//! * `{prefix}_users` — one point per user profile; id = UUIDv5(dataset, user_id).
//!
//! Both use cosine distance and a tenant keyword index on `_dataset`, so a query
//! only touches the segments of its own dataset.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use qdrant_client::qdrant::{
    self, CollectionStatus, Condition, CountPointsBuilder, CreateCollectionBuilder,
    CreateFieldIndexCollectionBuilder, DeletePointsBuilder, Distance, FieldType, Filter,
    GetPointsBuilder, KeywordIndexParamsBuilder, MinShould, Mmr, PointId, PointStruct,
    QuantizationType, Query, QueryPointsBuilder, RecommendInputBuilder, RepeatedIntegers,
    RepeatedStrings, ScalarQuantizationBuilder, UpsertPointsBuilder, VectorInput,
    VectorParamsBuilder, r#match::MatchValue, point_id::PointIdOptions, vector_output,
    vectors_config,
};
use qdrant_client::{Payload, Qdrant, QdrantError};
use serde_json::{Map, Value, json};

use super::{
    CollectionStats, DATASET_KEY, Example, ITEM_ID_KEY, ItemPoint, ItemQuery, ScoredItem,
    StoreError, StoreResult, StoredItem, USER_ID_KEY, UserProfile, VectorStore, mmr, norm,
    strip_reserved,
};
use crate::config::{PayloadIndexKind, QdrantSettings, Quantization};
use crate::ids::{item_point_id, user_point_id};
use crate::models::RecommendStrategy;
use crate::models::filter::{FieldCondition, Predicate, Scalar, ValidFilter, ValueSet};

const NORM_KEY: &str = "_norm";

pub struct QdrantStore {
    client: Qdrant,
    items: String,
    users: String,
    settings: QdrantSettings,
}

impl QdrantStore {
    /// Connects and waits (up to `QDRANT_STARTUP_TIMEOUT_SECS`) for Qdrant to answer.
    pub async fn connect(settings: &QdrantSettings) -> anyhow::Result<Self> {
        let client = Qdrant::from_url(&settings.url)
            .api_key(settings.api_key.clone())
            .timeout(settings.timeout)
            .connect_timeout(Duration::from_secs(5))
            .keep_alive_while_idle()
            .skip_compatibility_check()
            .build()?;

        let deadline = Instant::now() + settings.startup_timeout;
        loop {
            match client.health_check().await {
                Ok(reply) => {
                    tracing::info!(url = %settings.url, version = %reply.version, "connected to Qdrant");
                    break;
                }
                Err(err) if Instant::now() < deadline => {
                    tracing::warn!(url = %settings.url, error = %err, "Qdrant not reachable yet, retrying");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
                Err(err) => anyhow::bail!("cannot reach Qdrant at {}: {err}", settings.url),
            }
        }

        Ok(Self {
            client,
            items: format!("{}_items", settings.collection_prefix),
            users: format!("{}_users", settings.collection_prefix),
            settings: settings.clone(),
        })
    }

    async fn ensure_collection(&self, name: &str, dimension: usize) -> StoreResult<()> {
        if self
            .client
            .collection_exists(name)
            .await
            .map_err(map_err("check collection"))?
        {
            return self.verify_collection(name, dimension).await;
        }

        let mut builder = CreateCollectionBuilder::new(name)
            .vectors_config(VectorParamsBuilder::new(dimension as u64, Distance::Cosine))
            .on_disk_payload(self.settings.on_disk_payload);
        if self.settings.quantization == Quantization::Int8 {
            builder = builder.quantization_config(
                ScalarQuantizationBuilder::default()
                    .r#type(QuantizationType::Int8 as i32)
                    .quantile(0.99)
                    .always_ram(true),
            );
        }
        if let Some(shards) = self.settings.shard_number {
            builder = builder.shard_number(shards);
        }
        if let Some(replicas) = self.settings.replication_factor {
            builder = builder.replication_factor(replicas);
        }

        match self.client.create_collection(builder).await {
            Ok(_) => {
                tracing::info!(collection = name, dimension, "created Qdrant collection");
                Ok(())
            }
            // Another replica may have created it concurrently.
            Err(err) if self.client.collection_exists(name).await.unwrap_or(false) => {
                tracing::debug!(collection = name, error = %err, "collection created concurrently");
                self.verify_collection(name, dimension).await
            }
            Err(err) => Err(map_err("create collection")(err)),
        }
    }

    async fn verify_collection(&self, name: &str, dimension: usize) -> StoreResult<()> {
        let info = self
            .client
            .collection_info(name)
            .await
            .map_err(map_err("collection info"))?;
        let config = info
            .result
            .and_then(|r| r.config)
            .and_then(|c| c.params)
            .and_then(|p| p.vectors_config)
            .and_then(|v| v.config);
        match config {
            Some(vectors_config::Config::Params(params)) if params.size as usize == dimension => {
                if params.distance != Distance::Cosine as i32 {
                    tracing::warn!(
                        collection = name,
                        "collection does not use cosine distance; scores may be off"
                    );
                }
                Ok(())
            }
            Some(vectors_config::Config::Params(params)) => Err(StoreError::InvalidInput(format!(
                "Qdrant collection '{name}' stores {}-dimensional vectors but the embedding model produces {dimension}. \
                 After switching models, set a new QDRANT_COLLECTION_PREFIX and retrain your datasets \
                 (POST /api/v1/datasets/{{id}}/train)",
                params.size
            ))),
            _ => Err(StoreError::InvalidInput(format!(
                "Qdrant collection '{name}' uses named vectors and was not created by this service; \
                 choose another QDRANT_COLLECTION_PREFIX"
            ))),
        }
    }

    async fn ensure_index(&self, collection: &str, field: &str, kind: FieldType, tenant: bool) {
        let mut builder =
            CreateFieldIndexCollectionBuilder::new(collection, field, kind).wait(true);
        if tenant {
            builder =
                builder.field_index_params(KeywordIndexParamsBuilder::default().is_tenant(true));
        }
        if let Err(err) = self.client.create_field_index(builder).await {
            tracing::warn!(collection, field, error = %err, "could not create payload index");
        }
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    fn kind(&self) -> &'static str {
        "qdrant"
    }

    async fn initialize(&self, dimension: usize) -> StoreResult<()> {
        self.ensure_collection(&self.items, dimension).await?;
        self.ensure_collection(&self.users, dimension).await?;
        self.ensure_index(&self.items, DATASET_KEY, FieldType::Keyword, true)
            .await;
        self.ensure_index(&self.items, ITEM_ID_KEY, FieldType::Keyword, false)
            .await;
        self.ensure_index(&self.users, DATASET_KEY, FieldType::Keyword, true)
            .await;
        self.ensure_index(&self.users, USER_ID_KEY, FieldType::Keyword, false)
            .await;
        for index in &self.settings.payload_indexes {
            let kind = match index.kind {
                PayloadIndexKind::Keyword => FieldType::Keyword,
                PayloadIndexKind::Integer => FieldType::Integer,
                PayloadIndexKind::Float => FieldType::Float,
                PayloadIndexKind::Bool => FieldType::Bool,
                PayloadIndexKind::Datetime => FieldType::Datetime,
                PayloadIndexKind::Text => FieldType::Text,
            };
            self.ensure_index(&self.items, &index.field, kind, false)
                .await;
        }
        Ok(())
    }

    async fn health(&self) -> StoreResult<()> {
        self.client
            .health_check()
            .await
            .map(|_| ())
            .map_err(map_err("health check"))
    }

    async fn upsert_items(&self, dataset_id: &str, items: Vec<ItemPoint>) -> StoreResult<()> {
        if items.is_empty() {
            return Ok(());
        }
        let points: Vec<PointStruct> = items
            .into_iter()
            .map(|item| {
                let mut payload = item.payload;
                payload.insert(DATASET_KEY.into(), Value::String(dataset_id.into()));
                payload.insert(ITEM_ID_KEY.into(), Value::String(item.item_id.clone()));
                PointStruct::new(
                    item_pid(dataset_id, &item.item_id),
                    item.vector,
                    Payload::from(payload),
                )
            })
            .collect();
        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.items, points).wait(true))
            .await
            .map(|_| ())
            .map_err(map_err("upsert items"))
    }

    async fn delete_items(&self, dataset_id: &str, item_ids: &[String]) -> StoreResult<()> {
        if item_ids.is_empty() {
            return Ok(());
        }
        let ids: Vec<PointId> = item_ids.iter().map(|id| item_pid(dataset_id, id)).collect();
        self.client
            .delete_points(DeletePointsBuilder::new(&self.items).points(ids).wait(true))
            .await
            .map(|_| ())
            .map_err(map_err("delete items"))
    }

    async fn get_items(
        &self,
        dataset_id: &str,
        item_ids: &[String],
        with_vectors: bool,
    ) -> StoreResult<Vec<StoredItem>> {
        if item_ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<PointId> = item_ids.iter().map(|id| item_pid(dataset_id, id)).collect();
        let response = self
            .client
            .get_points(
                GetPointsBuilder::new(&self.items, ids)
                    .with_payload(true)
                    .with_vectors(with_vectors),
            )
            .await
            .map_err(map_err("get items"))?;
        Ok(response
            .result
            .into_iter()
            .map(|point| {
                let payload = json_payload(point.payload);
                StoredItem {
                    item_id: payload_item_id(&payload, point.id.as_ref()),
                    vector: if with_vectors {
                        dense(point.vectors)
                    } else {
                        None
                    },
                    payload: strip_reserved(payload),
                }
            })
            .collect())
    }

    async fn count_items(&self, dataset_id: &str) -> StoreResult<u64> {
        count(&self.client, &self.items, dataset_id).await
    }

    async fn query_items(
        &self,
        dataset_id: &str,
        query: ItemQuery,
    ) -> StoreResult<Vec<ScoredItem>> {
        let excluded: Vec<PointId> = query
            .excluded_ids()
            .iter()
            .map(|id| item_pid(dataset_id, id))
            .collect();
        let filter = build_filter(dataset_id, query.filter.as_ref(), excluded);
        let to_input = |example: &Example| match example {
            Example::Item(id) => VectorInput::new_id(item_pid(dataset_id, id)),
            Example::Vector(v) => VectorInput::new_dense(v.clone()),
        };
        let positive: Vec<VectorInput> = query.positive.iter().map(to_input).collect();
        let negative: Vec<VectorInput> = query.negative.iter().map(to_input).collect();

        // Qdrant applies MMR natively for nearest-neighbour queries; multi-example
        // queries fetch extra candidates (with vectors) and are re-ranked here.
        let local_mmr = query.diversity.is_some() && !query.is_simple();
        let search = if query.is_simple() {
            let input = positive
                .into_iter()
                .next()
                .expect("simple query has one positive");
            match query.diversity {
                Some(diversity) => Query::new_nearest_with_mmr(
                    input,
                    Mmr {
                        diversity: Some(diversity),
                        candidates_limit: Some(query.mmr_candidates() as u32),
                    },
                ),
                None => Query::new_nearest(input),
            }
        } else {
            let strategy = match query.strategy {
                RecommendStrategy::AverageVector => qdrant::RecommendStrategy::AverageVector,
                RecommendStrategy::BestScore => qdrant::RecommendStrategy::BestScore,
                RecommendStrategy::SumScores => qdrant::RecommendStrategy::SumScores,
            };
            Query::new_recommend(
                RecommendInputBuilder::default()
                    .positive(positive)
                    .negative(negative)
                    .strategy(strategy),
            )
        };

        let (limit, offset) = if local_mmr {
            (query.mmr_candidates(), 0)
        } else {
            (query.limit, query.offset)
        };
        let mut request = QueryPointsBuilder::new(&self.items)
            .query(search)
            .filter(filter)
            .limit(limit as u64)
            .offset(offset as u64)
            .with_payload(true)
            .with_vectors(local_mmr);
        if let Some(threshold) = query.score_threshold {
            request = request.score_threshold(threshold);
        }
        let response = self
            .client
            .query(request)
            .await
            .map_err(map_err("query items"))?;

        let scored = response.result.into_iter().map(|point| {
            let payload = json_payload(point.payload);
            let item_id = payload_item_id(&payload, point.id.as_ref());
            let payload = query.with_payload.then(|| strip_reserved(payload));
            (
                ScoredItem {
                    item_id,
                    score: point.score,
                    payload,
                },
                dense(point.vectors),
            )
        });

        Ok(match query.diversity.filter(|_| local_mmr) {
            Some(diversity) => {
                let candidates = scored
                    .map(|(item, vector)| mmr::Candidate {
                        score: item.score,
                        vector: vector.unwrap_or_default(),
                        value: item,
                    })
                    .collect();
                mmr::rerank(candidates, query.offset + query.limit, diversity)
                    .into_iter()
                    .skip(query.offset)
                    .map(|(item, _)| item)
                    .collect()
            }
            None => scored.map(|(item, _)| item).collect(),
        })
    }

    async fn upsert_users(&self, dataset_id: &str, profiles: Vec<UserProfile>) -> StoreResult<()> {
        if profiles.is_empty() {
            return Ok(());
        }
        let points: Vec<PointStruct> = profiles
            .into_iter()
            .map(|profile| {
                // Cosine collections normalize stored vectors, so the magnitude of the
                // preference sum is kept in the payload to allow incremental updates.
                let magnitude = norm(&profile.vector);
                let vector = if profile.is_empty() {
                    let mut unit = vec![0.0; profile.vector.len()];
                    unit[0] = 1.0;
                    unit
                } else {
                    profile.vector.clone()
                };
                let payload = json!({
                    DATASET_KEY: dataset_id,
                    USER_ID_KEY: profile.user_id,
                    NORM_KEY: if profile.is_empty() { 0.0 } else { f64::from(magnitude) },
                    "interactions": profile.interactions,
                    "recent_items": profile.recent_items,
                    "updated_at": profile.updated_at.to_rfc3339(),
                });
                PointStruct::new(
                    user_pid(dataset_id, &profile.user_id),
                    vector,
                    Payload::try_from(payload).expect("payload is a JSON object"),
                )
            })
            .collect();
        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.users, points).wait(true))
            .await
            .map(|_| ())
            .map_err(map_err("upsert users"))
    }

    async fn get_users(
        &self,
        dataset_id: &str,
        user_ids: &[String],
    ) -> StoreResult<Vec<UserProfile>> {
        if user_ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<PointId> = user_ids.iter().map(|id| user_pid(dataset_id, id)).collect();
        let response = self
            .client
            .get_points(
                GetPointsBuilder::new(&self.users, ids)
                    .with_payload(true)
                    .with_vectors(true),
            )
            .await
            .map_err(map_err("get users"))?;
        Ok(response
            .result
            .into_iter()
            .filter_map(|point| {
                let payload = json_payload(point.payload);
                let magnitude = payload.get(NORM_KEY).and_then(Value::as_f64).unwrap_or(0.0) as f32;
                let stored = dense(point.vectors)?;
                Some(UserProfile {
                    user_id: payload.get(USER_ID_KEY)?.as_str()?.to_string(),
                    vector: stored.iter().map(|x| x * magnitude).collect(),
                    interactions: payload
                        .get("interactions")
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                    recent_items: payload
                        .get("recent_items")
                        .and_then(Value::as_array)
                        .map(|items| {
                            items
                                .iter()
                                .filter_map(|v| v.as_str().map(str::to_string))
                                .collect()
                        })
                        .unwrap_or_default(),
                    updated_at: payload
                        .get("updated_at")
                        .and_then(Value::as_str)
                        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                        .map(|t| t.with_timezone(&Utc))
                        .unwrap_or_else(Utc::now),
                })
            })
            .collect())
    }

    async fn count_users(&self, dataset_id: &str) -> StoreResult<u64> {
        count(&self.client, &self.users, dataset_id).await
    }

    async fn delete_dataset(&self, dataset_id: &str) -> StoreResult<()> {
        for collection in [&self.items, &self.users] {
            self.client
                .delete_points(
                    DeletePointsBuilder::new(collection.as_str())
                        .points(Filter::must([tenant(dataset_id)]))
                        .wait(true),
                )
                .await
                .map_err(map_err("delete dataset"))?;
        }
        Ok(())
    }

    async fn stats(&self) -> StoreResult<Vec<CollectionStats>> {
        let mut stats = Vec::new();
        for collection in [&self.items, &self.users] {
            let info = self
                .client
                .collection_info(collection.as_str())
                .await
                .map_err(map_err("collection info"))?;
            let result = info.result.unwrap_or_default();
            let status = match CollectionStatus::try_from(result.status) {
                Ok(CollectionStatus::Green) => "green",
                Ok(CollectionStatus::Yellow) => "yellow",
                Ok(CollectionStatus::Red) => "red",
                Ok(CollectionStatus::Grey) => "grey",
                _ => "unknown",
            };
            stats.push(CollectionStats {
                name: collection.clone(),
                points: result.points_count.unwrap_or(0),
                status: status.into(),
            });
        }
        Ok(stats)
    }
}

async fn count(client: &Qdrant, collection: &str, dataset_id: &str) -> StoreResult<u64> {
    let response = client
        .count(
            CountPointsBuilder::new(collection)
                .filter(Filter::must([tenant(dataset_id)]))
                .exact(true),
        )
        .await
        .map_err(map_err("count points"))?;
    Ok(response.result.map_or(0, |r| r.count))
}

fn item_pid(dataset_id: &str, item_id: &str) -> PointId {
    PointId::from(item_point_id(dataset_id, item_id).to_string())
}

fn user_pid(dataset_id: &str, user_id: &str) -> PointId {
    PointId::from(user_point_id(dataset_id, user_id).to_string())
}

fn tenant(dataset_id: &str) -> Condition {
    // Built explicitly: `From<String>` would turn values with spaces into full-text matches.
    Condition::matches(DATASET_KEY, MatchValue::Keyword(dataset_id.to_string()))
}

fn build_filter(dataset_id: &str, filter: Option<&ValidFilter>, excluded: Vec<PointId>) -> Filter {
    let mut out = Filter::must([tenant(dataset_id)]);
    if let Some(filter) = filter {
        out.must.extend(filter.must.iter().map(condition));
        out.must_not.extend(filter.must_not.iter().map(condition));
        let should: Vec<Condition> = filter.should.iter().map(condition).collect();
        match filter.min_should {
            Some(min_count) => {
                out.min_should = Some(MinShould {
                    conditions: should,
                    min_count,
                })
            }
            None => out.should = should,
        }
    }
    if !excluded.is_empty() {
        out.must_not.push(Condition::has_id(excluded));
    }
    out
}

fn condition(c: &FieldCondition) -> Condition {
    let key = c.key.as_str();
    let match_value = match &c.predicate {
        Predicate::Range(r) => {
            return Condition::range(
                key,
                qdrant::Range {
                    gt: r.gt,
                    gte: r.gte,
                    lt: r.lt,
                    lte: r.lte,
                },
            );
        }
        Predicate::Equals(Scalar::Keyword(s)) => MatchValue::Keyword(s.clone()),
        Predicate::Equals(Scalar::Integer(i)) => MatchValue::Integer(*i),
        Predicate::Equals(Scalar::Bool(b)) => MatchValue::Boolean(*b),
        Predicate::AnyOf(ValueSet::Keywords(strings)) => MatchValue::Keywords(RepeatedStrings {
            strings: strings.clone(),
        }),
        Predicate::AnyOf(ValueSet::Integers(integers)) => MatchValue::Integers(RepeatedIntegers {
            integers: integers.clone(),
        }),
        Predicate::NoneOf(ValueSet::Keywords(strings)) => {
            MatchValue::ExceptKeywords(RepeatedStrings {
                strings: strings.clone(),
            })
        }
        Predicate::NoneOf(ValueSet::Integers(integers)) => {
            MatchValue::ExceptIntegers(RepeatedIntegers {
                integers: integers.clone(),
            })
        }
    };
    Condition::matches(key, match_value)
}

fn json_payload(payload: HashMap<String, qdrant::Value>) -> Map<String, Value> {
    payload
        .into_iter()
        .map(|(k, v)| (k, Value::from(v)))
        .collect()
}

fn payload_item_id(payload: &Map<String, Value>, id: Option<&PointId>) -> String {
    payload
        .get(ITEM_ID_KEY)
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| match id.and_then(|id| id.point_id_options.clone()) {
            Some(PointIdOptions::Uuid(uuid)) => uuid,
            Some(PointIdOptions::Num(num)) => num.to_string(),
            None => String::new(),
        })
}

fn dense(vectors: Option<qdrant::VectorsOutput>) -> Option<Vec<f32>> {
    match vectors?.get_vector()? {
        vector_output::Vector::Dense(dense) => Some(dense.data),
        _ => None,
    }
}

fn map_err(context: &'static str) -> impl Fn(QdrantError) -> StoreError {
    move |err| {
        let message = format!("{context}: {err}");
        match &err {
            QdrantError::ResponseError { status } => match status.code() {
                tonic::Code::NotFound => {
                    StoreError::NotFound(format!("{context}: {}", status.message()))
                }
                tonic::Code::InvalidArgument => {
                    StoreError::InvalidInput(format!("{context}: {}", status.message()))
                }
                tonic::Code::Unavailable
                | tonic::Code::DeadlineExceeded
                | tonic::Code::Cancelled => StoreError::Unavailable(message),
                _ => StoreError::Backend(message),
            },
            QdrantError::ResourceExhaustedError { .. } => StoreError::Unavailable(message),
            _ => StoreError::Backend(message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Filter as ApiFilter;

    #[test]
    fn filter_always_scopes_to_tenant_and_exclusions() {
        let api: ApiFilter = serde_json::from_value(json!({
            "must": [{"key": "category", "match": {"value": "footwear"}}],
            "should": [{"key": "a", "match": {"any": [1, 2]}}, {"key": "b", "range": {"lt": 5.0}}],
            "must_not": [{"key": "brand", "match": {"except": ["x"]}}],
            "min_should_count": 1
        }))
        .unwrap();
        let f = build_filter(
            "shop",
            Some(&api.validate().unwrap()),
            vec![item_pid("shop", "1")],
        );
        assert_eq!(f.must.len(), 2, "tenant + category");
        assert_eq!(f.must_not.len(), 2, "brand + excluded ids");
        assert!(f.should.is_empty());
        assert_eq!(f.min_should.as_ref().unwrap().conditions.len(), 2);
    }

    #[test]
    fn tenant_condition_is_keyword_even_with_spaces() {
        let c = Condition::matches(DATASET_KEY, MatchValue::Keyword("a b".into()));
        let field = match c.condition_one_of.unwrap() {
            qdrant::condition::ConditionOneOf::Field(f) => f,
            _ => unreachable!(),
        };
        assert!(matches!(
            field.r#match.unwrap().match_value,
            Some(MatchValue::Keyword(_))
        ));
    }
}
