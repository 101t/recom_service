//! OpenAI-compatible `/embeddings` API.
//!
//! Works with OpenAI and Azure OpenAI as well as self-hosted servers exposing the
//! same API: Ollama, vLLM, Hugging Face TEI, LiteLLM, LM Studio, ...
//! Set `EMBEDDING_API_BASE` (default `https://api.openai.com/v1`).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::http::JsonClient;
use super::{EmbedError, EmbedResult, Embedder, Prefixes, check_dimensions};
use crate::config::EmbeddingSettings;

const DEFAULT_API_BASE: &str = "https://api.openai.com/v1";

pub struct OpenAiEmbedder {
    client: JsonClient,
    url: String,
    model: String,
    dimension: usize,
    /// Sent as `dimensions` (only when explicitly configured).
    requested_dimension: Option<usize>,
    batch_size: usize,
    prefixes: Prefixes,
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Deserialize)]
struct Response {
    data: Vec<Datum>,
}

#[derive(Deserialize)]
struct Datum {
    #[serde(default)]
    index: usize,
    embedding: Vec<f32>,
}

impl OpenAiEmbedder {
    /// Creates the client and, unless `EMBEDDING_DIMENSION` is set, probes the
    /// model once to learn its dimension (which also validates credentials).
    pub async fn connect(settings: &EmbeddingSettings) -> anyhow::Result<Self> {
        let base = settings
            .api_base
            .as_deref()
            .unwrap_or(DEFAULT_API_BASE)
            .trim_end_matches('/');
        let mut embedder = Self {
            client: JsonClient::new(
                settings.timeout,
                settings.api_key.clone(),
                settings.max_retries,
            )?,
            url: format!("{base}/embeddings"),
            model: settings.model.clone(),
            dimension: settings.dimension.unwrap_or(0),
            requested_dimension: settings.dimension,
            batch_size: settings.batch_size,
            prefixes: Prefixes::resolve(&settings.model, settings),
        };
        if embedder.dimension == 0 {
            let probe = embedder.call(&["dimension probe".to_string()]).await?;
            embedder.dimension = probe.first().map(Vec::len).unwrap_or(0);
            anyhow::ensure!(
                embedder.dimension > 0,
                "embedding API returned an empty vector"
            );
        }
        Ok(embedder)
    }

    async fn call(&self, inputs: &[String]) -> EmbedResult<Vec<Vec<f32>>> {
        let body = Request {
            model: &self.model,
            input: inputs,
            encoding_format: "float",
            dimensions: self.requested_dimension,
        };
        let mut response: Response = self.client.post(&self.url, &body).await?;
        if response.data.len() != inputs.len() {
            return Err(EmbedError::new(format!(
                "embedding API returned {} vectors for {} inputs",
                response.data.len(),
                inputs.len()
            )));
        }
        response.data.sort_by_key(|d| d.index);
        Ok(response.data.into_iter().map(|d| d.embedding).collect())
    }

    async fn embed_batched(&self, texts: Vec<String>) -> EmbedResult<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            out.extend(self.call(chunk).await?);
        }
        check_dimensions(&out, self.dimension)?;
        Ok(out)
    }
}

#[async_trait]
impl Embedder for OpenAiEmbedder {
    fn provider(&self) -> &'static str {
        "openai"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed_documents(&self, texts: Vec<String>) -> EmbedResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.embed_batched(self.prefixes.documents(texts)).await
    }

    async fn embed_query(&self, text: &str) -> EmbedResult<Vec<f32>> {
        let mut vectors = self.embed_batched(vec![self.prefixes.query(text)]).await?;
        vectors
            .pop()
            .ok_or_else(|| EmbedError::new("embedding API returned no vector"))
    }
}
