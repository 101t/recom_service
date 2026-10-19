//! HTTP API (`/api/v1`) and its OpenAPI description.
//!
//! Handlers are thin: they extract input, call a service and shape the response.
//! Routes registered here are collected into the OpenAPI document automatically
//! (via `utoipa-actix-web`), so adding an endpoint is a single `.service(...)` call.

pub mod auth;
pub mod datasets;
pub mod health;
pub mod interactions;
pub mod items;
pub mod predictions;
pub mod recommend;
mod upload;

use actix_web::middleware::from_fn;
use utoipa::openapi::security::{ApiKey, ApiKeyValue, HttpAuthScheme, HttpBuilder, SecurityScheme};
use utoipa::{Modify, OpenApi};
use utoipa_actix_web::scope;
use utoipa_actix_web::service_config::ServiceConfig;

use crate::error::{ErrorBody, ErrorDetail};

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Recommendation Service",
        description = "Multilingual vector recommendations backed by Qdrant.\n\n\
            1. Upload a catalog (`POST /api/v1/datasets/{dataset_id}/items/upload`) or send items as JSON.\n\
            2. Optionally record user interactions to personalize results.\n\
            3. Ask for recommendations by text, by example items and/or by user.",
        license(name = "MIT", identifier = "MIT")
    ),
    components(schemas(ErrorBody, ErrorDetail)),
    modifiers(&SecuritySchemes),
    security(("api_key" = []), ("bearer" = [])),
    tags(
        (name = "Health", description = "Liveness, readiness and service information"),
        (name = "Datasets", description = "Create, inspect, retrain and delete datasets"),
        (name = "Items", description = "Add, upload, read and delete catalog items"),
        (name = "Interactions", description = "User events that personalize recommendations"),
        (name = "Recommendations", description = "Get recommendations (predict)"),
        (name = "Predictions", description = "Logged predictions and exports"),
    )
)]
pub struct ApiDoc;

struct SecuritySchemes;

impl Modify for SecuritySchemes {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        let components = openapi.components.get_or_insert_with(Default::default);
        components.add_security_scheme(
            "api_key",
            SecurityScheme::ApiKey(ApiKey::Header(ApiKeyValue::new("X-API-Key"))),
        );
        components.add_security_scheme(
            "bearer",
            SecurityScheme::Http(HttpBuilder::new().scheme(HttpAuthScheme::Bearer).build()),
        );
    }
}

/// Registers every route. `/health` and `/ready` stay unauthenticated.
pub fn routes(cfg: &mut ServiceConfig) {
    cfg.service(health::health).service(health::ready).service(
        scope::scope("/api/v1")
            .wrap(from_fn(auth::require_api_key))
            .service(health::info)
            .service(health::models)
            .service(datasets::create_dataset)
            .service(datasets::list_datasets)
            .service(datasets::get_dataset)
            .service(datasets::delete_dataset)
            .service(datasets::train_dataset)
            .service(items::upsert_items)
            .service(items::upload_items)
            .service(items::get_item)
            .service(items::delete_item)
            .service(interactions::record_interactions)
            .service(interactions::upload_interactions)
            .service(recommend::recommend)
            .service(predictions::list_predictions)
            .service(predictions::export_predictions),
    );
}
