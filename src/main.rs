mod api;
mod data;
mod embedding;
mod qdrant;

use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use data::delta_handler::DeltaTableHandler;
use embedding::voyage_embedder::VoyageEmbedder;
use qdrant::client::QdrantClient;
use std::sync::Arc;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use api::{
    health::{self, HealthResponse},
    recommendations::{self, RecommendationRequest, Recommendation, RecommendationResponse, ErrorResponse},
    datasets::{self, UploadResponse, DatasetListResponse, DatasetInfoResponse},
};

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Recommendation Service API",
        description = "Vector-based recommendation service with Delta Lake storage",
        version = "0.1.0",
        contact(name = "API Support"),
        license(name = "MIT")
    ),
    paths(
        health::health_check,
        recommendations::get_recommendations,
        datasets::upload_dataset,
        datasets::list_datasets,
        datasets::get_dataset_info,
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
    )),
    tags(
        (name = "Health", description = "Health check endpoints"),
        (name = "Recommendations", description = "Recommendation endpoints"),
        (name = "Datasets", description = "Dataset management endpoints")
    )
)]
struct ApiDoc;

#[derive(Clone)]
pub struct AppState {
    pub qdrant_client: Arc<QdrantClient>,
    pub embedder: Arc<VoyageEmbedder>,
    pub delta_handler: Arc<DeltaTableHandler>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    
    // Initialize components
    let qdrant_client = Arc::new(QdrantClient::new().await);
    let embedder = Arc::new(VoyageEmbedder::new());
    let delta_handler = Arc::new(DeltaTableHandler::new().await);
    
    let app_state = AppState {
        qdrant_client,
        embedder,
        delta_handler,
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