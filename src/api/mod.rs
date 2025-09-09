pub mod datasets;
pub mod recommendations;
pub mod health;

use actix_web::web;

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/v1")
            .service(health::health_check)
            .service(datasets::upload_dataset)
            .service(recommendations::get_recommendations)
            .service(datasets::list_datasets)
            .service(datasets::get_dataset_info),
    );
}