//! User preference profiles learned from interactions.
//!
//! A profile is the weighted sum of the (normalized) vectors of the items a user
//! interacted with. Because a sum is order-independent, profiles can be updated
//! incrementally per event and rebuilt from the interaction log at any time with
//! the same result.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use chrono::Utc;
use polars::prelude::DataFrame;

use super::datasets::{Accepted, DatasetService, Progress, Upload};
use crate::error::{AppError, AppResult};
use crate::ids::{validate_dataset_id, validate_entity_id};
use crate::models::{
    DatasetStatus, Interaction, RecordInteractionsResponse, TrainingProgress, TrainingStage,
};
use crate::storage::datasets::read_interactions;
use crate::storage::tabular;
use crate::vector_store::{UserProfile, normalized};

/// Maximum interactions accepted by one JSON request.
pub const MAX_INTERACTIONS_PER_REQUEST: usize = 10_000;
const USERS_PER_CHUNK: usize = 512;
const ITEMS_PER_FETCH: usize = 1_000;
const MAX_UNKNOWN_REPORTED: usize = 100;
const ROWS_PER_SLICE: usize = 50_000;

#[derive(Debug, Default)]
pub(crate) struct ApplyOutcome {
    pub accepted: u64,
    pub users_updated: u64,
    pub new_users: u64,
    pub unknown_items: BTreeSet<String>,
}

impl DatasetService {
    /// Stores interactions and updates the affected user profiles synchronously.
    pub async fn record_interactions(
        &self,
        id: &str,
        mut interactions: Vec<Interaction>,
    ) -> AppResult<RecordInteractionsResponse> {
        validate_dataset_id(id)?;
        if interactions.is_empty() {
            return Err(AppError::bad_request(
                "`interactions` must contain at least one interaction",
            ));
        }
        if interactions.len() > MAX_INTERACTIONS_PER_REQUEST {
            return Err(AppError::bad_request(format!(
                "at most {MAX_INTERACTIONS_PER_REQUEST} interactions per request; use the upload endpoint for bulk loads"
            )));
        }
        validate_interactions(&interactions)?;
        self.repo.get(id).await?;

        let now = Utc::now();
        for interaction in &mut interactions {
            interaction.timestamp.get_or_insert(now);
        }
        self.repo.append_interactions(id, &interactions).await?;
        let total = interactions.len() as u64;
        let outcome = self
            .apply_interactions(id, interactions, false, None)
            .await?;
        self.repo
            .update(id, |d| {
                d.stats.interactions += total;
                d.stats.users += outcome.new_users;
            })
            .await?;

        Ok(RecordInteractionsResponse {
            dataset_id: id.to_string(),
            accepted: outcome.accepted,
            users_updated: outcome.users_updated,
            unknown_items: outcome
                .unknown_items
                .into_iter()
                .take(MAX_UNKNOWN_REPORTED)
                .collect(),
        })
    }

    /// Stores an uploaded interactions file and applies it in the background.
    pub async fn upload_interactions(
        self: &Arc<Self>,
        id: &str,
        upload: Upload,
    ) -> AppResult<Accepted> {
        let exists = match validate_dataset_id(id) {
            Ok(()) => self.repo.get(id).await.map(|_| ()),
            Err(e) => Err(e),
        };
        if let Err(e) = exists {
            upload.discard().await;
            return Err(e);
        }
        let df = upload.read().await?;
        let (interactions, skipped) =
            tokio::task::spawn_blocking(move || interactions_from_frame(df)).await??;
        if interactions.is_empty() {
            return Err(AppError::bad_request(
                "the file contains no valid interactions",
            ));
        }
        validate_interactions(&interactions)?;
        let upload_id = self
            .repo
            .write_interactions_upload(id, &interactions)
            .await?;
        let accepted = interactions.len() as u64;

        let dataset = self
            .repo
            .update(id, |d| {
                d.stats.interactions += accepted;
                d.stats.uploads += 1;
                d.status = DatasetStatus::Training;
                d.error = None;
                d.heartbeat_at = Some(Utc::now());
                d.progress = Some(TrainingProgress {
                    stage: TrainingStage::Users,
                    processed: 0,
                    total: accepted,
                });
            })
            .await?;
        tracing::info!(dataset = %id, accepted, skipped, %upload_id, "interactions upload accepted");

        let dataset_id = id.to_string();
        let job = self.spawn_job(id, move |this| async move {
            let mut progress = Progress::new(TrainingStage::Users, accepted);
            this.apply_interactions(&dataset_id, interactions, false, Some(&mut progress))
                .await?;
            this.report(&dataset_id, &mut progress, true).await
        });
        Ok(Accepted {
            dataset,
            rows: accepted + skipped,
            upload_id: Some(upload_id),
            job,
        })
    }

    /// Recomputes every profile of the dataset from its interaction log.
    pub(crate) async fn rebuild_profiles(&self, id: &str) -> AppResult<()> {
        let files = self.repo.interaction_files(id).await?;
        if files.is_empty() {
            return Ok(());
        }
        let interactions = tokio::task::spawn_blocking(move || -> AppResult<Vec<Interaction>> {
            let mut all = Vec::new();
            for file in &files {
                all.extend(read_interactions(file)?);
            }
            Ok(all)
        })
        .await??;
        let mut progress = Progress::new(TrainingStage::Users, interactions.len() as u64);
        self.report(id, &mut progress, true).await?;
        self.apply_interactions(id, interactions, true, Some(&mut progress))
            .await?;
        self.report(id, &mut progress, true).await
    }

    /// Adds interactions to user profiles. With `fresh`, existing profiles are
    /// ignored and rebuilt from these interactions only.
    pub(crate) async fn apply_interactions(
        &self,
        id: &str,
        interactions: Vec<Interaction>,
        fresh: bool,
        mut progress: Option<&mut Progress>,
    ) -> AppResult<ApplyOutcome> {
        let dimension = self.embedder.dimension();
        let max_recent = self.settings.profile_max_items;

        // Group by user, keeping first-seen order.
        let mut order: Vec<String> = Vec::new();
        let mut groups: HashMap<String, Vec<Interaction>> = HashMap::new();
        for interaction in interactions {
            match groups.get_mut(&interaction.user_id) {
                Some(group) => group.push(interaction),
                None => {
                    order.push(interaction.user_id.clone());
                    groups.insert(interaction.user_id.clone(), vec![interaction]);
                }
            }
        }

        let mut outcome = ApplyOutcome::default();
        for users in order.chunks(USERS_PER_CHUNK) {
            let item_ids: Vec<String> = users
                .iter()
                .flat_map(|u| groups[u].iter().map(|i| i.item_id.clone()))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            let vectors = self.item_vectors(id, &item_ids).await?;

            let guards = self.lock_users(id, users).await;
            let mut existing: HashMap<String, UserProfile> = if fresh {
                HashMap::new()
            } else {
                self.store
                    .get_users(id, users)
                    .await?
                    .into_iter()
                    .map(|p| (p.user_id.clone(), p))
                    .collect()
            };

            let mut updated = Vec::new();
            let mut events_in_chunk = 0u64;
            for user in users {
                let mut events = groups.remove(user).unwrap_or_default();
                events_in_chunk += events.len() as u64;
                events.sort_by_key(|e| e.timestamp);
                let (mut profile, is_new) = match existing.remove(user) {
                    Some(profile) => (profile, false),
                    None => (UserProfile::new(user.clone(), dimension), true),
                };
                if profile.vector.len() != dimension {
                    // The embedding model changed since the profile was built.
                    profile.vector = vec![0.0; dimension];
                }
                let mut changed = false;
                for event in events {
                    let Some(vector) = vectors.get(&event.item_id) else {
                        outcome.unknown_items.insert(event.item_id);
                        continue;
                    };
                    let weight = event.weight();
                    profile
                        .vector
                        .iter_mut()
                        .zip(vector)
                        .for_each(|(p, x)| *p += weight * x);
                    profile.interactions += 1;
                    profile.recent_items.retain(|seen| seen != &event.item_id);
                    profile.recent_items.push(event.item_id);
                    outcome.accepted += 1;
                    changed = true;
                }
                if profile.recent_items.len() > max_recent {
                    let excess = profile.recent_items.len() - max_recent;
                    profile.recent_items.drain(..excess);
                }
                if changed {
                    profile.updated_at = Utc::now();
                    outcome.new_users += u64::from(is_new);
                    updated.push(profile);
                }
            }
            outcome.users_updated += updated.len() as u64;
            self.store.upsert_users(id, updated).await?;
            drop(guards);

            if let Some(p) = progress.as_deref_mut() {
                p.processed += events_in_chunk;
                self.report(id, p, false).await?;
            }
        }
        Ok(outcome)
    }

    async fn item_vectors(
        &self,
        id: &str,
        item_ids: &[String],
    ) -> AppResult<HashMap<String, Vec<f32>>> {
        let mut out = HashMap::with_capacity(item_ids.len());
        for batch in item_ids.chunks(ITEMS_PER_FETCH) {
            for item in self.store.get_items(id, batch, true).await? {
                if let Some(vector) = item.vector {
                    out.insert(item.item_id, normalized(&vector));
                }
            }
        }
        Ok(out)
    }

    /// Locks the stripes of `users` in ascending order (deadlock-free), so
    /// concurrent updates of the same profile cannot lose writes on this instance.
    async fn lock_users(&self, id: &str, users: &[String]) -> Vec<tokio::sync::MutexGuard<'_, ()>> {
        let mut stripes: Vec<usize> = users
            .iter()
            .map(|user| {
                let mut hasher = DefaultHasher::new();
                (id, user).hash(&mut hasher);
                (hasher.finish() % self.user_stripes.len() as u64) as usize
            })
            .collect();
        stripes.sort_unstable();
        stripes.dedup();
        let mut guards = Vec::with_capacity(stripes.len());
        for stripe in stripes {
            guards.push(self.user_stripes[stripe].lock().await);
        }
        guards
    }
}

fn validate_interactions(interactions: &[Interaction]) -> AppResult<()> {
    for interaction in interactions {
        validate_entity_id("user", &interaction.user_id)?;
        validate_entity_id("item", &interaction.item_id)?;
        if interaction.weight.is_some_and(|w| !w.is_finite()) {
            return Err(AppError::bad_request(
                "interaction weight must be a finite number",
            ));
        }
    }
    Ok(())
}

fn interactions_from_frame(df: DataFrame) -> AppResult<(Vec<Interaction>, u64)> {
    let mut out = Vec::new();
    let mut skipped = 0;
    let height = df.height();
    let mut offset = 0;
    while offset < height {
        let len = ROWS_PER_SLICE.min(height - offset);
        let mut slice = df.slice(offset as i64, len);
        let (mut parsed, s) = tabular::parse_interactions(tabular::rows(&mut slice)?)?;
        out.append(&mut parsed);
        skipped += s;
        offset += len;
    }
    Ok((out, skipped))
}
