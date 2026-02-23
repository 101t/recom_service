pub mod collections;
pub mod datasets;
pub mod predictions;
pub mod recommendations;
pub mod health;

use actix_web::web;

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/v1")
            .service(health::health_check)
            .service(datasets::upload_dataset)
            .service(datasets::get_job_status)
            .service(recommendations::get_recommendations)
            .service(datasets::list_datasets)
            .service(datasets::get_dataset_info)
            .service(datasets::delete_dataset)
            .service(predictions::list_prediction_logs)
            .service(predictions::export_prediction_logs)
            .service(collections::list_collections)
            .service(collections::create_collection)
            .service(collections::get_collection)
            .service(collections::delete_collection),
    );
}