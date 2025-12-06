use actix_web::{get, HttpResponse};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    /// Service health status
    pub status: String,
    /// Service version
    pub version: String,
}

/// Health check endpoint
#[utoipa::path(
    get,
    path = "/api/v1/health",
    tag = "Health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse)
    )
)]
#[get("/health")]
pub async fn health_check() -> HttpResponse {
    HttpResponse::Ok().json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::App;

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "healthy".to_string(),
            version: "0.1.0".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("healthy"));
        assert!(json.contains("0.1.0"));
    }

    #[test]
    fn test_health_response_deserialization() {
        let json = r#"{"status":"healthy","version":"0.1.0"}"#;
        let response: HealthResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "healthy");
        assert_eq!(response.version, "0.1.0");
    }

    #[actix_web::test]
    async fn test_health_check_endpoint() {
        let app = actix_web::test::init_service(
            App::new().service(health_check)
        ).await;

        let req = actix_web::test::TestRequest::get().uri("/health").to_request();
        let resp = actix_web::test::call_service(&app, req).await;

        assert!(resp.status().is_success());

        let body: HealthResponse = actix_web::test::read_body_json(resp).await;
        assert_eq!(body.status, "healthy");
        assert!(!body.version.is_empty());
    }
}