//! Optional API-key authentication for `/api/v1`.
//!
//! Enabled by setting `API_KEY`. Clients send it as `X-API-Key: <key>` or
//! `Authorization: Bearer <key>`.

use actix_web::body::{EitherBody, MessageBody};
use actix_web::dev::{ServiceRequest, ServiceResponse};
use actix_web::http::header::AUTHORIZATION;
use actix_web::middleware::Next;
use actix_web::{ResponseError, web};

use crate::error::AppError;
use crate::state::AppState;

pub async fn require_api_key<B: MessageBody>(
    req: ServiceRequest,
    next: Next<B>,
) -> Result<ServiceResponse<EitherBody<B>>, actix_web::Error> {
    let expected = req
        .app_data::<web::Data<AppState>>()
        .and_then(|state| state.settings.server.api_key.clone());
    if let Some(expected) = expected {
        let headers = req.headers();
        let provided = headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .or_else(|| {
                headers
                    .get(AUTHORIZATION)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| {
                        v.strip_prefix("Bearer ")
                            .or_else(|| v.strip_prefix("bearer "))
                    })
            });
        if !provided.is_some_and(|key| constant_time_eq(key.trim().as_bytes(), expected.as_bytes()))
        {
            let response =
                AppError::Unauthorized("missing or invalid API key".into()).error_response();
            return Ok(req.into_response(response).map_into_right_body());
        }
    }
    Ok(next.call(req).await?.map_into_left_body())
}

/// Compares secrets without leaking their content through timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::constant_time_eq;

    #[test]
    fn compares_keys() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secret2"));
    }
}
