mod api;
mod data;
mod embedding;
mod qdrant;

use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use data::dataset_handler::DatasetHandler;
use data::prediction_logger::PredictionLogger;
use embedding::local_embedder::LocalEmbedder;
use embedding::voyage_embedder::VoyageEmbedder;
use embedding::Embedder;
use qdrant::client::QdrantClient;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

use api::{
    health::{self, HealthResponse},
    recommendations::{self, RecommendationRequest, Recommendation, RecommendationResponse, ErrorResponse},
    datasets::{self, UploadResponse, DatasetListResponse, DatasetInfoResponse, DeleteResponse},
    collections,
    predictions,
};
use data::prediction_logger::{PredictionLog, PredictionLogListResponse};
use qdrant::client::{
    CollectionDetails, CreateCollectionRequest, QdrantFilter, FilterCondition,
    MatchCondition, RangeCondition,
};

/// Status of a background dataset upload job
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct JobStatus {
    /// Unique job identifier
    pub job_id: String,
    /// Dataset identifier
    pub dataset_id: String,
    /// Current status: pending | processing | completed | failed
    pub status: String,
    /// Human-readable message
    pub message: String,
    /// Number of vectors upserted (populated on completion)
    pub vectors_count: Option<usize>,
}

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Recommendation Service API",
        description = "Vector-based recommendation service with Parquet dataset storage and Qdrant vector database",
        version = "0.2.0",
        contact(name = "API Support"),
        license(name = "MIT")
    ),
    paths(
        health::health_check,
        recommendations::get_recommendations,
        datasets::upload_dataset,
        datasets::list_datasets,
        datasets::get_dataset_info,
        datasets::delete_dataset,
        datasets::get_job_status,
        predictions::list_prediction_logs,
        predictions::export_prediction_logs,
        collections::list_collections,
        collections::create_collection,
        collections::get_collection,
        collections::delete_collection,
    ),
    components(schemas(
        HealthResponse,
        RecommendationRequest,
        Recommendation,
        RecommendationResponse,
        ErrorResponse,
        UploadResponse,
        DatasetListResponse,
        DatasetInfoResponse,
        DeleteResponse,
        PredictionLog,
        PredictionLogListResponse,
        JobStatus,
        CollectionDetails,
        CreateCollectionRequest,
        QdrantFilter,
        FilterCondition,
        MatchCondition,
        RangeCondition,
    )),
    tags(
        (name = "Health", description = "Health check endpoints"),
        (name = "Recommendations", description = "Recommendation endpoints"),
        (name = "Datasets", description = "Dataset management endpoints"),
        (name = "Predictions", description = "Prediction log endpoints"),
        (name = "Collections", description = "Qdrant collection management")
    )
)]
struct ApiDoc;

#[derive(Clone)]
pub struct AppState {
    pub qdrant_client: Arc<QdrantClient>,
    pub embedder: Arc<dyn Embedder>,
    pub dataset_handler: Arc<DatasetHandler>,
    pub prediction_logger: Arc<PredictionLogger>,
    pub jobs: Arc<Mutex<HashMap<String, JobStatus>>>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Load .env file before anything else
    dotenvy::dotenv().ok();
    env_logger::init();
    
    // Initialize components
    let qdrant_client = Arc::new(QdrantClient::new().await);
    
    // Select embedder based on environment variable
    // USE_LOCAL_EMBEDDER=true uses the local FastEmbed service (default)
    // USE_LOCAL_EMBEDDER=false uses Voyage AI (requires VOYAGE_API_KEY)
    let use_local = std::env::var("USE_LOCAL_EMBEDDER")
        .map(|v| v.to_lowercase() != "false")
        .unwrap_or(true);
    
    let embedder: Arc<dyn Embedder> = if use_local {
        log::info!("Using local FastEmbed service for embeddings");
        Arc::new(LocalEmbedder::new())
    } else {
        log::info!("Using Voyage AI for embeddings");
        Arc::new(VoyageEmbedder::new())
    };
    
    let dataset_handler = Arc::new(DatasetHandler::new());
    let prediction_logger = Arc::new(PredictionLogger::new());
    
    let app_state = AppState {
        qdrant_client,
        embedder,
        dataset_handler,
        prediction_logger,
        jobs: Arc::new(Mutex::new(HashMap::new())),
    };

    let openapi = ApiDoc::openapi();
    
    log::info!("Starting server at http://0.0.0.0:8000");
    log::info!("Swagger UI available at http://0.0.0.0:8000/docs/");
    
    // Start HTTP server
    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);
            
        App::new()
            .app_data(web::Data::new(app_state.clone()))
            .wrap(cors)
            .configure(api::config)
            .service(
                SwaggerUi::new("/docs/{_:.*}")
                    .url("/api-docs/openapi.json", openapi.clone())
            )
    })
    .bind("0.0.0.0:8000")?
    .run()
    .await
}