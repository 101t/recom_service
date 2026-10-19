//! HTTP server assembly: routes, middleware, OpenAPI/Swagger UI and graceful shutdown.

use actix_cors::Cors;
use actix_web::body::MessageBody;
use actix_web::dev::{ServiceFactory, ServiceRequest, ServiceResponse};
use actix_web::middleware::Condition;
use actix_web::{App, HttpServer, web};
use actix_web_prom::{PrometheusMetrics, PrometheusMetricsBuilder};
use tracing_actix_web::TracingLogger;
use utoipa::OpenApi;
use utoipa_actix_web::AppExt;
use utoipa_swagger_ui::SwaggerUi;

use crate::api::{self, ApiDoc};
use crate::config::Settings;
use crate::error::AppError;
use crate::state::AppState;

/// Starts the service and blocks until it receives SIGINT/SIGTERM.
pub async fn run(settings: Settings) -> anyhow::Result<()> {
    let state = AppState::initialize(settings).await?;
    let metrics = metrics()?;
    let server = state.settings.server.clone();

    let app_state = state.clone();
    let mut http = HttpServer::new(move || build_app(app_state.clone(), metrics.clone()))
        .shutdown_timeout(server.shutdown_timeout_secs)
        .bind((server.host.as_str(), server.port))?;
    if let Some(workers) = server.workers {
        http = http.workers(workers);
    }
    tracing::info!(
        address = %format!("{}:{}", server.host, server.port),
        auth = server.api_key.is_some(),
        "recommendation service listening (Swagger UI at /docs/)"
    );
    http.run().await?;

    tracing::info!("shutting down");
    state.shutdown().await;
    Ok(())
}

/// Prometheus metrics (`/metrics`): request counts and latency histograms per route.
pub fn metrics() -> anyhow::Result<PrometheusMetrics> {
    PrometheusMetricsBuilder::new("recom")
        .endpoint("/metrics")
        .exclude("/health")
        .exclude("/ready")
        .mask_unmatched_patterns("UNKNOWN")
        .build()
        .map_err(|e| anyhow::anyhow!("cannot build metrics: {e}"))
}

/// Builds the application. Used by the server (once per worker) and by tests.
pub fn build_app(
    state: AppState,
    metrics: PrometheusMetrics,
) -> App<
    impl ServiceFactory<
        ServiceRequest,
        Config = (),
        Response = ServiceResponse<impl MessageBody>,
        Error = actix_web::Error,
        InitError = (),
    >,
> {
    let server = state.settings.server.clone();
    let (app, openapi) =
        App::new()
            .into_utoipa_app()
            .openapi(ApiDoc::openapi())
            .app_data(web::Data::new(state))
            .app_data(
                web::JsonConfig::default()
                    .limit(server.json_limit_bytes)
                    .error_handler(|err, _| {
                        AppError::bad_request(format!("invalid JSON body: {err}")).into()
                    }),
            )
            .app_data(web::QueryConfig::default().error_handler(|err, _| {
                AppError::bad_request(format!("invalid query parameters: {err}")).into()
            }))
            .app_data(web::PathConfig::default().error_handler(|err, _| {
                AppError::bad_request(format!("invalid path: {err}")).into()
            }))
            .configure(api::routes)
            .split_for_parts();

    let app = if server.swagger_ui {
        app.service(SwaggerUi::new("/docs/{_:.*}").url("/api-docs/openapi.json", openapi))
    } else {
        app
    };

    app.default_service(web::to(|| async {
        Err::<actix_web::HttpResponse, _>(AppError::not_found("no such route"))
    }))
    .wrap(Condition::new(server.metrics, metrics))
    .wrap(cors(&server.cors_allowed_origins))
    .wrap(TracingLogger::default())
}

fn cors(origins: &[String]) -> Cors {
    let cors = if origins.iter().any(|o| o == "*") {
        Cors::default().allow_any_origin()
    } else {
        origins
            .iter()
            .fold(Cors::default(), |cors, origin| cors.allowed_origin(origin))
    };
    cors.allow_any_method()
        .allow_any_header()
        .expose_any_header()
        .max_age(3600)
}
