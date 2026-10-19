//! Liveness, readiness and service information.

use actix_web::{HttpResponse, get, web};
use serde::Serialize;
use utoipa::ToSchema;

use crate::embedding::{ModelInfo, is_multilingual, supported_models};
use crate::error::{AppResult, ErrorBody};
use crate::state::AppState;
use crate::vector_store::CollectionStats;

#[derive(Serialize, ToSchema)]
pub struct HealthResponse {
    #[schema(example = "ok")]
    pub status: String,
    #[schema(example = "0.3.0")]
    pub version: String,
}

/// Liveness probe: the process is up.
#[utoipa::path(tag = "Health", security(()), responses((status = 200, body = HealthResponse)))]
#[get("/health")]
pub async fn health() -> web::Json<HealthResponse> {
    web::Json(HealthResponse {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
    })
}

/// Readiness probe: the vector store is reachable.
#[utoipa::path(
    tag = "Health",
    security(()),
    responses((status = 200, body = HealthResponse), (status = 503, body = ErrorBody))
)]
#[get("/ready")]
pub async fn ready(state: web::Data<AppState>) -> AppResult<HttpResponse> {
    state.store.health().await?;
    Ok(HttpResponse::Ok().json(HealthResponse {
        status: "ready".into(),
        version: env!("CARGO_PKG_VERSION").into(),
    }))
}

#[derive(Serialize, ToSchema)]
pub struct EmbeddingInfo {
    #[schema(example = "fastembed")]
    pub provider: String,
    #[schema(example = "intfloat/multilingual-e5-small")]
    pub model: String,
    #[schema(example = 384)]
    pub dimension: usize,
    pub multilingual: bool,
}

#[derive(Serialize, ToSchema)]
pub struct VectorStoreInfo {
    #[schema(example = "qdrant")]
    pub kind: String,
    pub collections: Vec<CollectionStats>,
}

#[derive(Serialize, ToSchema)]
pub struct ServiceInfo {
    pub name: String,
    pub version: String,
    pub embedding: EmbeddingInfo,
    pub vector_store: VectorStoreInfo,
    pub uptime_secs: i64,
}

/// Service configuration summary: embedding model and vector store statistics.
#[utoipa::path(tag = "Health", responses((status = 200, body = ServiceInfo)))]
#[get("/info")]
pub async fn info(state: web::Data<AppState>) -> AppResult<web::Json<ServiceInfo>> {
    let collections = state.store.stats().await?;
    Ok(web::Json(ServiceInfo {
        name: env!("CARGO_PKG_NAME").into(),
        version: env!("CARGO_PKG_VERSION").into(),
        embedding: EmbeddingInfo {
            provider: state.embedder.provider().into(),
            model: state.embedder.model().into(),
            dimension: state.embedder.dimension(),
            multilingual: is_multilingual(state.embedder.model()),
        },
        vector_store: VectorStoreInfo {
            kind: state.store.kind().into(),
            collections,
        },
        uptime_secs: (chrono::Utc::now() - state.started_at).num_seconds(),
    }))
}

#[derive(Serialize, ToSchema)]
pub struct ModelList {
    /// Model currently in use.
    pub current: String,
    /// Models available to the built-in `fastembed` provider.
    pub models: Vec<ModelInfo>,
}

/// Embedding models that can be selected with `EMBEDDING_MODEL`.
#[utoipa::path(tag = "Health", responses((status = 200, body = ModelList)))]
#[get("/models")]
pub async fn models(state: web::Data<AppState>) -> web::Json<ModelList> {
    web::Json(ModelList {
        current: state.embedder.model().into(),
        models: supported_models(),
    })
}
