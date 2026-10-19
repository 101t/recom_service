//! Application error type and its HTTP mapping.
//!
//! Every handler returns `Result<_, AppError>`; the [`ResponseError`] impl turns
//! errors into a consistent JSON body:
//!
//! ```json
//! { "error": { "code": "not_found", "message": "dataset 'shop' not found" } }
//! ```

use actix_web::http::StatusCode;
use actix_web::{HttpResponse, ResponseError};
use serde::Serialize;
use utoipa::ToSchema;

pub type AppResult<T> = Result<T, AppError>;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("{0}")]
    BadRequest(String),

    #[error("{0}")]
    Unauthorized(String),

    #[error("{0}")]
    NotFound(String),

    #[error("{0}")]
    Conflict(String),

    #[error("{0}")]
    PayloadTooLarge(String),

    #[error("{0}")]
    UnsupportedMediaType(String),

    #[error("embedding provider error: {0}")]
    Embedding(String),

    #[error("vector store error: {0}")]
    VectorStore(String),

    #[error("service unavailable: {0}")]
    Unavailable(String),

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

impl AppError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::BadRequest(msg.into())
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }

    pub fn internal(msg: impl std::fmt::Display) -> Self {
        Self::Internal(anyhow::anyhow!("{msg}"))
    }

    /// Stable, machine-readable error code returned to clients.
    pub fn code(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "bad_request",
            Self::Unauthorized(_) => "unauthorized",
            Self::NotFound(_) => "not_found",
            Self::Conflict(_) => "conflict",
            Self::PayloadTooLarge(_) => "payload_too_large",
            Self::UnsupportedMediaType(_) => "unsupported_media_type",
            Self::Embedding(_) => "embedding_error",
            Self::VectorStore(_) => "vector_store_error",
            Self::Unavailable(_) => "unavailable",
            Self::Internal(_) => "internal_error",
        }
    }
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        Self::Internal(err.into())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(err: serde_json::Error) -> Self {
        Self::Internal(err.into())
    }
}

impl From<polars::error::PolarsError> for AppError {
    fn from(err: polars::error::PolarsError) -> Self {
        Self::BadRequest(format!("could not process tabular data: {err}"))
    }
}

impl From<tokio::task::JoinError> for AppError {
    fn from(err: tokio::task::JoinError) -> Self {
        Self::Internal(anyhow::anyhow!("background task failed: {err}"))
    }
}

/// JSON error envelope returned by every endpoint.
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorDetail {
    /// Machine-readable error code (e.g. `not_found`, `bad_request`).
    #[schema(example = "not_found")]
    pub code: String,
    /// Human-readable description.
    #[schema(example = "dataset 'shop' not found")]
    pub message: String,
}

impl ResponseError for AppError {
    fn status_code(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::PayloadTooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            Self::UnsupportedMediaType(_) => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Self::Embedding(_) | Self::VectorStore(_) => StatusCode::BAD_GATEWAY,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_response(&self) -> HttpResponse {
        let status = self.status_code();
        if status.is_server_error() {
            tracing::error!(error = %self, code = self.code(), "request failed");
        }
        HttpResponse::build(status).json(ErrorBody {
            error: ErrorDetail {
                code: self.code().to_string(),
                message: self.to_string(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::body::to_bytes;

    #[test]
    fn maps_variants_to_status_codes() {
        assert_eq!(
            AppError::bad_request("x").status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            AppError::not_found("x").status_code(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            AppError::Conflict("x".into()).status_code(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            AppError::Embedding("x".into()).status_code(),
            StatusCode::BAD_GATEWAY
        );
        assert_eq!(
            AppError::internal("x").status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[actix_web::test]
    async fn renders_json_envelope() {
        let resp = AppError::not_found("dataset 'shop' not found").error_response();
        let body = to_bytes(resp.into_body()).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "not_found");
        assert_eq!(json["error"]["message"], "dataset 'shop' not found");
    }
}
