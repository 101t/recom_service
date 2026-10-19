//! [Voyage AI](https://docs.voyageai.com/reference/embeddings-api) embeddings.
//!
//! Voyage handles query/document asymmetry itself through `input_type`, so no
//! text prefixes are added unless configured explicitly.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::http::JsonClient;
use super::{EmbedError, EmbedResult, Embedder, check_dimensions};
use crate::config::EmbeddingSettings;

const DEFAULT_API_BASE: &str = "https://api.voyageai.com/v1";

pub struct VoyageEmbedder {
    client: JsonClient,
    url: String,
    model: String,
    dimension: usize,
    requested_dimension: Option<usize>,
    batch_size: usize,
    query_prefix: String,
    document_prefix: String,
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: &'a [String],
    input_type: &'static str,
    truncation: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimension: Option<usize>,
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

impl VoyageEmbedder {
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
            query_prefix: settings.query_prefix.clone().unwrap_or_default(),
            document_prefix: settings.document_prefix.clone().unwrap_or_default(),
        };
        if embedder.dimension == 0 {
            let probe = embedder
                .call(&["dimension probe".to_string()], "query")
                .await?;
            embedder.dimension = probe.first().map(Vec::len).unwrap_or(0);
            anyhow::ensure!(embedder.dimension > 0, "Voyage returned an empty vector");
        }
        Ok(embedder)
    }

    async fn call(
        &self,
        inputs: &[String],
        input_type: &'static str,
    ) -> EmbedResult<Vec<Vec<f32>>> {
        let body = Request {
            model: &self.model,
            input: inputs,
            input_type,
            truncation: true,
            output_dimension: self.requested_dimension,
        };
        let mut response: Response = self.client.post(&self.url, &body).await?;
        if response.data.len() != inputs.len() {
            return Err(EmbedError::new(format!(
                "Voyage returned {} vectors for {} inputs",
                response.data.len(),
                inputs.len()
            )));
        }
        response.data.sort_by_key(|d| d.index);
        Ok(response.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[async_trait]
impl Embedder for VoyageEmbedder {
    fn provider(&self) -> &'static str {
        "voyage"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed_documents(&self, texts: Vec<String>) -> EmbedResult<Vec<Vec<f32>>> {
        let texts: Vec<String> = texts
            .into_iter()
            .map(|t| format!("{}{t}", self.document_prefix))
            .collect();
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            out.extend(self.call(chunk, "document").await?);
        }
        check_dimensions(&out, self.dimension)?;
        Ok(out)
    }

    async fn embed_query(&self, text: &str) -> EmbedResult<Vec<f32>> {
        let mut out = self
            .call(&[format!("{}{text}", self.query_prefix)], "query")
            .await?;
        check_dimensions(&out, self.dimension)?;
        out.pop()
            .ok_or_else(|| EmbedError::new("Voyage returned no vector"))
    }
}
