mod api;
mod data;
mod embedding;
//mod models;
mod qdrant;

use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use data::delta_handler::DeltaTableHandler;
use embedding::voyage_embedder::VoyageEmbedder;
use qdrant::client::QdrantClient;
use std::sync::Arc;

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
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}