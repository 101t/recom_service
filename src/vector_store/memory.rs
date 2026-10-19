//! In-memory vector store (exact brute-force search).
//!
//! Mirrors the Qdrant backend's semantics closely enough for tests and local
//! development (`VECTOR_STORE=memory`). Data is lost on restart and not shared
//! between instances, so never use it in production.

use std::collections::HashMap;
use std::sync::RwLock;

use async_trait::async_trait;

use super::{
    CollectionStats, Example, ItemPoint, ItemQuery, ScoredItem, StoreError, StoreResult,
    StoredItem, UserProfile, VectorStore, dot, mmr, normalized,
};
use crate::models::RecommendStrategy;

#[derive(Default)]
pub struct MemoryStore {
    tenants: RwLock<HashMap<String, Tenant>>,
}

#[derive(Default)]
struct Tenant {
    items: HashMap<String, StoredItem>,
    users: HashMap<String, UserProfile>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, Tenant>> {
        self.tenants.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, HashMap<String, Tenant>> {
        self.tenants.write().unwrap_or_else(|e| e.into_inner())
    }
}

fn resolve(tenant: &Tenant, examples: &[Example]) -> StoreResult<Vec<Vec<f32>>> {
    examples
        .iter()
        .map(|example| match example {
            Example::Vector(v) => Ok(normalized(v)),
            Example::Item(id) => tenant
                .items
                .get(id)
                .and_then(|item| item.vector.clone())
                .ok_or_else(|| StoreError::NotFound(format!("item '{id}' not found"))),
        })
        .collect()
}

fn mean(vectors: &[Vec<f32>]) -> Vec<f32> {
    let mut out = vec![0.0; vectors.first().map_or(0, Vec::len)];
    for v in vectors {
        out.iter_mut().zip(v).for_each(|(o, x)| *o += x);
    }
    out.iter_mut()
        .for_each(|o| *o /= vectors.len().max(1) as f32);
    out
}

/// Scoring that mirrors Qdrant's recommendation strategies.
struct Scorer {
    target: Option<Vec<f32>>,
    positive: Vec<Vec<f32>>,
    negative: Vec<Vec<f32>>,
    strategy: RecommendStrategy,
}

impl Scorer {
    fn new(positive: Vec<Vec<f32>>, negative: Vec<Vec<f32>>, strategy: RecommendStrategy) -> Self {
        let simple = positive.len() == 1 && negative.is_empty();
        let target = (simple || strategy == RecommendStrategy::AverageVector).then(|| {
            let avg_pos = mean(&positive);
            if negative.is_empty() {
                normalized(&avg_pos)
            } else {
                let avg_neg = mean(&negative);
                normalized(
                    &avg_pos
                        .iter()
                        .zip(&avg_neg)
                        .map(|(p, n)| p + (p - n))
                        .collect::<Vec<_>>(),
                )
            }
        });
        Self {
            target,
            positive,
            negative,
            strategy,
        }
    }

    fn score(&self, v: &[f32]) -> f32 {
        if let Some(target) = &self.target {
            return dot(target, v);
        }
        let sims = |examples: &[Vec<f32>]| examples.iter().map(|e| dot(e, v)).collect::<Vec<_>>();
        match self.strategy {
            RecommendStrategy::SumScores => {
                sims(&self.positive).iter().sum::<f32>() - sims(&self.negative).iter().sum::<f32>()
            }
            _ => {
                let best_pos = sims(&self.positive).into_iter().fold(f32::MIN, f32::max);
                if self.negative.is_empty() {
                    return best_pos;
                }
                let best_neg = sims(&self.negative).into_iter().fold(f32::MIN, f32::max);
                if best_pos > best_neg {
                    best_pos
                } else {
                    -(best_neg * best_neg)
                }
            }
        }
    }
}

#[async_trait]
impl VectorStore for MemoryStore {
    fn kind(&self) -> &'static str {
        "memory"
    }

    async fn initialize(&self, _dimension: usize) -> StoreResult<()> {
        Ok(())
    }

    async fn health(&self) -> StoreResult<()> {
        Ok(())
    }

    async fn upsert_items(&self, dataset_id: &str, items: Vec<ItemPoint>) -> StoreResult<()> {
        let mut tenants = self.write();
        let tenant = tenants.entry(dataset_id.to_string()).or_default();
        for item in items {
            tenant.items.insert(
                item.item_id.clone(),
                StoredItem {
                    item_id: item.item_id,
                    payload: item.payload,
                    vector: Some(normalized(&item.vector)),
                },
            );
        }
        Ok(())
    }

    async fn delete_items(&self, dataset_id: &str, item_ids: &[String]) -> StoreResult<()> {
        if let Some(tenant) = self.write().get_mut(dataset_id) {
            for id in item_ids {
                tenant.items.remove(id);
            }
        }
        Ok(())
    }

    async fn get_items(
        &self,
        dataset_id: &str,
        item_ids: &[String],
        with_vectors: bool,
    ) -> StoreResult<Vec<StoredItem>> {
        let tenants = self.read();
        let Some(tenant) = tenants.get(dataset_id) else {
            return Ok(Vec::new());
        };
        Ok(item_ids
            .iter()
            .filter_map(|id| tenant.items.get(id))
            .map(|item| StoredItem {
                item_id: item.item_id.clone(),
                payload: item.payload.clone(),
                vector: if with_vectors {
                    item.vector.clone()
                } else {
                    None
                },
            })
            .collect())
    }

    async fn count_items(&self, dataset_id: &str) -> StoreResult<u64> {
        Ok(self
            .read()
            .get(dataset_id)
            .map_or(0, |t| t.items.len() as u64))
    }

    async fn query_items(
        &self,
        dataset_id: &str,
        query: ItemQuery,
    ) -> StoreResult<Vec<ScoredItem>> {
        let tenants = self.read();
        let Some(tenant) = tenants.get(dataset_id) else {
            return Ok(Vec::new());
        };
        let scorer = Scorer::new(
            resolve(tenant, &query.positive)?,
            resolve(tenant, &query.negative)?,
            query.strategy,
        );
        let excluded = query.excluded_ids();

        let mut scored: Vec<(&StoredItem, f32)> = tenant
            .items
            .values()
            .filter(|item| excluded.binary_search(&item.item_id).is_err())
            .filter(|item| {
                query
                    .filter
                    .as_ref()
                    .is_none_or(|f| f.matches(&item.payload))
            })
            .filter_map(|item| item.vector.as_ref().map(|v| (item, scorer.score(v))))
            .filter(|(_, score)| query.score_threshold.is_none_or(|t| *score >= t))
            .collect();
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.item_id.cmp(&b.0.item_id))
        });

        let ranked: Vec<(&StoredItem, f32)> = match query.diversity {
            Some(diversity) => {
                let candidates = scored
                    .into_iter()
                    .take(query.mmr_candidates())
                    .map(|(item, score)| mmr::Candidate {
                        value: item,
                        score,
                        vector: item.vector.clone().unwrap_or_default(),
                    })
                    .collect();
                mmr::rerank(candidates, query.offset + query.limit, diversity)
            }
            None => scored,
        };

        Ok(ranked
            .into_iter()
            .skip(query.offset)
            .take(query.limit)
            .map(|(item, score)| ScoredItem {
                item_id: item.item_id.clone(),
                score,
                payload: query.with_payload.then(|| item.payload.clone()),
            })
            .collect())
    }

    async fn upsert_users(&self, dataset_id: &str, profiles: Vec<UserProfile>) -> StoreResult<()> {
        let mut tenants = self.write();
        let tenant = tenants.entry(dataset_id.to_string()).or_default();
        for profile in profiles {
            tenant.users.insert(profile.user_id.clone(), profile);
        }
        Ok(())
    }

    async fn get_users(
        &self,
        dataset_id: &str,
        user_ids: &[String],
    ) -> StoreResult<Vec<UserProfile>> {
        let tenants = self.read();
        let Some(tenant) = tenants.get(dataset_id) else {
            return Ok(Vec::new());
        };
        Ok(user_ids
            .iter()
            .filter_map(|id| tenant.users.get(id).cloned())
            .collect())
    }

    async fn count_users(&self, dataset_id: &str) -> StoreResult<u64> {
        Ok(self
            .read()
            .get(dataset_id)
            .map_or(0, |t| t.users.len() as u64))
    }

    async fn delete_dataset(&self, dataset_id: &str) -> StoreResult<()> {
        self.write().remove(dataset_id);
        Ok(())
    }

    async fn stats(&self) -> StoreResult<Vec<CollectionStats>> {
        let tenants = self.read();
        Ok(vec![
            CollectionStats {
                name: "items".into(),
                points: tenants.values().map(|t| t.items.len() as u64).sum(),
                status: "green".into(),
            },
            CollectionStats {
                name: "users".into(),
                points: tenants.values().map(|t| t.users.len() as u64).sum(),
                status: "green".into(),
            },
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Filter;
    use serde_json::json;

    fn point(id: &str, vector: Vec<f32>, category: &str) -> ItemPoint {
        ItemPoint {
            item_id: id.into(),
            vector,
            payload: json!({"category": category}).as_object().unwrap().clone(),
        }
    }

    fn query(positive: Vec<Example>) -> ItemQuery {
        ItemQuery {
            positive,
            negative: vec![],
            strategy: RecommendStrategy::AverageVector,
            filter: None,
            exclude_item_ids: vec![],
            limit: 10,
            offset: 0,
            score_threshold: None,
            diversity: None,
            with_payload: true,
        }
    }

    async fn store() -> MemoryStore {
        let s = MemoryStore::new();
        s.upsert_items(
            "ds",
            vec![
                point("a", vec![1.0, 0.0, 0.0], "x"),
                point("b", vec![0.9, 0.1, 0.0], "x"),
                point("c", vec![0.0, 1.0, 0.0], "y"),
                point("d", vec![0.0, 0.0, 1.0], "y"),
            ],
        )
        .await
        .unwrap();
        s
    }

    #[tokio::test]
    async fn nearest_by_vector_and_by_item() {
        let s = store().await;
        let res = s
            .query_items("ds", query(vec![Example::Vector(vec![1.0, 0.0, 0.0])]))
            .await
            .unwrap();
        assert_eq!(res[0].item_id, "a");

        let res = s
            .query_items("ds", query(vec![Example::Item("a".into())]))
            .await
            .unwrap();
        assert_eq!(res[0].item_id, "b", "example item itself is excluded");
        assert!(res.iter().all(|r| r.item_id != "a"));
    }

    #[tokio::test]
    async fn filters_threshold_and_pagination() {
        let s = store().await;
        let mut q = query(vec![Example::Vector(vec![1.0, 0.0, 0.0])]);
        q.filter = Some(
            serde_json::from_value::<Filter>(
                json!({"must": [{"key": "category", "match": {"value": "y"}}]}),
            )
            .unwrap()
            .validate()
            .unwrap(),
        );
        let res = s.query_items("ds", q.clone()).await.unwrap();
        assert!(
            res.iter()
                .all(|r| r.payload.as_ref().unwrap()["category"] == "y")
        );

        let mut q = query(vec![Example::Vector(vec![1.0, 0.0, 0.0])]);
        q.score_threshold = Some(0.5);
        assert_eq!(s.query_items("ds", q).await.unwrap().len(), 2);

        let mut q = query(vec![Example::Vector(vec![1.0, 0.0, 0.0])]);
        q.offset = 1;
        q.limit = 1;
        assert_eq!(s.query_items("ds", q).await.unwrap()[0].item_id, "b");
    }

    #[tokio::test]
    async fn negative_examples_push_results_away() {
        let s = store().await;
        let mut q = query(vec![Example::Item("b".into())]);
        q.negative = vec![Example::Item("c".into())];
        let res = s.query_items("ds", q).await.unwrap();
        assert_eq!(res[0].item_id, "a");
        assert_eq!(res.last().unwrap().item_id, "d");
    }

    #[tokio::test]
    async fn unknown_example_item_is_not_found() {
        let s = store().await;
        let err = s
            .query_items("ds", query(vec![Example::Item("zzz".into())]))
            .await
            .unwrap_err();
        assert!(matches!(err, StoreError::NotFound(_)));
    }

    #[tokio::test]
    async fn datasets_are_isolated_and_deletable() {
        let s = store().await;
        assert!(
            s.query_items("other", query(vec![Example::Vector(vec![1.0, 0.0, 0.0])]))
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(s.count_items("ds").await.unwrap(), 4);
        s.delete_dataset("ds").await.unwrap();
        assert_eq!(s.count_items("ds").await.unwrap(), 0);
    }
}
