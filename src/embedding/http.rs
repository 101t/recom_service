//! Shared HTTP plumbing for remote embedding APIs: JSON POST with retries.

use std::time::Duration;

use reqwest::StatusCode;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::{EmbedError, EmbedResult};

const MAX_BACKOFF: Duration = Duration::from_secs(8);
const MAX_RETRY_AFTER: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub(crate) struct JsonClient {
    client: reqwest::Client,
    api_key: Option<String>,
    max_retries: u32,
}

impl JsonClient {
    pub fn new(
        timeout: Duration,
        api_key: Option<String>,
        max_retries: u32,
    ) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent(concat!(
                "recommendation-service/",
                env!("CARGO_PKG_VERSION")
            ))
            .build()?;
        Ok(Self {
            client,
            api_key,
            max_retries,
        })
    }

    /// POSTs `body` as JSON, retrying timeouts, connection errors, 429 and 5xx responses
    /// with exponential backoff (honoring `Retry-After`).
    pub async fn post<B: Serialize + ?Sized, R: DeserializeOwned>(
        &self,
        url: &str,
        body: &B,
    ) -> EmbedResult<R> {
        let mut attempt = 0u32;
        loop {
            let mut request = self.client.post(url).json(body);
            if let Some(key) = &self.api_key {
                request = request.bearer_auth(key);
            }
            let (retryable, retry_after, error) = match request.send().await {
                Ok(resp) if resp.status().is_success() => {
                    return resp
                        .json::<R>()
                        .await
                        .map_err(|e| EmbedError::new(format!("invalid response from {url}: {e}")));
                }
                Ok(resp) => {
                    let status = resp.status();
                    let retry_after = resp
                        .headers()
                        .get(reqwest::header::RETRY_AFTER)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok())
                        .map(Duration::from_secs);
                    let body = resp.text().await.unwrap_or_default();
                    let retryable =
                        status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
                    (
                        retryable,
                        retry_after,
                        format!("{url} returned {status}: {}", truncate(&body, 500)),
                    )
                }
                Err(e) => (
                    e.is_timeout() || e.is_connect() || e.is_request(),
                    None,
                    format!("request to {url} failed: {e}"),
                ),
            };

            if !retryable || attempt >= self.max_retries {
                return Err(EmbedError::new(error));
            }
            let backoff = retry_after
                .map(|d| d.min(MAX_RETRY_AFTER))
                .unwrap_or_else(|| {
                    (Duration::from_millis(250) * 2u32.pow(attempt)).min(MAX_BACKOFF)
                });
            tracing::warn!(attempt = attempt + 1, ?backoff, %error, "embedding request failed, retrying");
            tokio::time::sleep(backoff).await;
            attempt += 1;
        }
    }
}

fn truncate(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}
