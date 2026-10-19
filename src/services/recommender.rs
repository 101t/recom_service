//! Serving recommendations ("predict").

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use uuid::Uuid;

use super::prediction_log::PredictionLogger;
use crate::config::RecommendSettings;
use crate::embedding::Embedder;
use crate::error::{AppError, AppResult};
use crate::ids::{validate_dataset_id, validate_entity_id};
use crate::models::{PredictionLog, RecommendRequest, RecommendResponse, Recommendation};
use crate::storage::DatasetRepository;
use crate::vector_store::{Example, ItemQuery, VectorStore, normalized};

const MAX_EXAMPLES: usize = 100;
const MAX_QUERY_CHARS: usize = 8_192;

pub struct Recommender {
    repo: Arc<DatasetRepository>,
    store: Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedder>,
    predictions: Arc<PredictionLogger>,
    settings: RecommendSettings,
}

impl Recommender {
    pub fn new(
        repo: Arc<DatasetRepository>,
        store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedder>,
        predictions: Arc<PredictionLogger>,
        settings: RecommendSettings,
    ) -> Self {
        Self {
            repo,
            store,
            embedder,
            predictions,
            settings,
        }
    }

    /// Combines the request's signals (text query, example items, user profile)
    /// into one vector query and logs the prediction.
    pub async fn recommend(
        &self,
        dataset_id: &str,
        req: RecommendRequest,
    ) -> AppResult<RecommendResponse> {
        let started = Instant::now();
        validate_dataset_id(dataset_id)?;
        self.repo.get(dataset_id).await?;

        let limit = req.limit.unwrap_or(self.settings.default_limit);
        if limit == 0 || limit > self.settings.max_limit {
            return Err(AppError::bad_request(format!(
                "limit must be between 1 and {}",
                self.settings.max_limit
            )));
        }
        let offset = req.offset.unwrap_or(0);
        if offset > self.settings.max_offset {
            return Err(AppError::bad_request(format!(
                "offset must be at most {}",
                self.settings.max_offset
            )));
        }
        if req.diversity.is_some_and(|d| !(0.0..=1.0).contains(&d)) {
            return Err(AppError::bad_request("diversity must be between 0 and 1"));
        }
        if req.score_threshold.is_some_and(|t| !t.is_finite()) {
            return Err(AppError::bad_request(
                "score_threshold must be a finite number",
            ));
        }
        let filter = req.filter.as_ref().map(|f| f.validate()).transpose()?;

        let mut positive = Vec::new();
        let mut negative = Vec::new();
        let mut exclude = Vec::new();

        self.check_items_exist(dataset_id, &req).await?;
        positive.extend(req.item_ids.iter().cloned().map(Example::Item));
        negative.extend(req.negative_item_ids.iter().cloned().map(Example::Item));

        if let Some(user_id) = &req.user_id {
            validate_entity_id("user", user_id)?;
            let profile = self
                .store
                .get_users(dataset_id, std::slice::from_ref(user_id))
                .await?
                .into_iter()
                .next();
            if let Some(profile) = profile {
                if req.exclude_seen.unwrap_or(true) {
                    exclude.extend(profile.recent_items.iter().cloned());
                }
                if !profile.is_empty() {
                    positive.push(Example::Vector(normalized(&profile.vector)));
                }
            }
        }

        if let Some(query) = req
            .query
            .as_deref()
            .map(str::trim)
            .filter(|q| !q.is_empty())
        {
            if query.chars().count() > MAX_QUERY_CHARS {
                return Err(AppError::bad_request(format!(
                    "query must be at most {MAX_QUERY_CHARS} characters"
                )));
            }
            let vector = self.embedder.embed_query(query).await?;
            positive.push(Example::Vector(normalized(&vector)));
        }

        if positive.is_empty() {
            return Err(match &req.user_id {
                Some(user_id) => AppError::not_found(format!(
                    "no positive interactions recorded for user '{user_id}' in dataset '{dataset_id}'; \
                     add a `query` or `item_ids` as a fallback"
                )),
                None => AppError::bad_request(
                    "provide at least one of `query`, `item_ids` or `user_id`",
                ),
            });
        }

        let results = self
            .store
            .query_items(
                dataset_id,
                ItemQuery {
                    positive,
                    negative,
                    strategy: req.strategy.unwrap_or_default(),
                    filter,
                    exclude_item_ids: exclude,
                    limit,
                    offset,
                    score_threshold: req.score_threshold,
                    diversity: req.diversity,
                    with_payload: req.with_payload.unwrap_or(true),
                },
            )
            .await?;

        let recommendations: Vec<Recommendation> = results
            .into_iter()
            .map(|r| Recommendation {
                id: r.item_id,
                score: r.score,
                payload: r.payload,
            })
            .collect();
        let latency_ms = started.elapsed().as_millis() as u64;
        let prediction_id = Uuid::new_v4().to_string();

        self.predictions.record(PredictionLog {
            prediction_id: prediction_id.clone(),
            timestamp: Utc::now(),
            dataset_id: dataset_id.to_string(),
            query: req.query,
            user_id: req.user_id,
            item_ids: req.item_ids,
            result_ids: recommendations.iter().map(|r| r.id.clone()).collect(),
            scores: recommendations.iter().map(|r| r.score).collect(),
            latency_ms,
        });

        Ok(RecommendResponse {
            prediction_id,
            dataset_id: dataset_id.to_string(),
            recommendations,
            latency_ms,
        })
    }

    /// Fails with 404 listing example items that are not in the dataset.
    async fn check_items_exist(&self, dataset_id: &str, req: &RecommendRequest) -> AppResult<()> {
        let mut ids: Vec<String> = req
            .item_ids
            .iter()
            .chain(&req.negative_item_ids)
            .cloned()
            .collect();
        if ids.is_empty() {
            return Ok(());
        }
        if ids.len() > MAX_EXAMPLES {
            return Err(AppError::bad_request(format!(
                "at most {MAX_EXAMPLES} example items are allowed"
            )));
        }
        for id in &ids {
            validate_entity_id("item", id)?;
        }
        ids.sort();
        ids.dedup();
        let found: HashSet<String> = self
            .store
            .get_items(dataset_id, &ids, false)
            .await?
            .into_iter()
            .map(|i| i.item_id)
            .collect();
        let missing: Vec<&str> = ids
            .iter()
            .filter(|id| !found.contains(*id))
            .map(String::as_str)
            .collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(AppError::not_found(format!(
                "item(s) not found in dataset '{dataset_id}': {}",
                missing.join(", ")
            )))
        }
    }
}
