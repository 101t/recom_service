//! Text embedding providers.
//!
//! Implement [`Embedder`] to add a provider, then wire it into [`build`]. The rest
//! of the service only depends on the trait.

#[cfg(feature = "fastembed")]
pub mod fastembed;
pub mod hash;
mod http;
pub mod openai;
pub mod voyage;

use std::sync::Arc;

use async_trait::async_trait;
use serde::Serialize;
use utoipa::ToSchema;

use crate::config::{EmbeddingProvider, EmbeddingSettings};
use crate::error::AppError;

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EmbedError(pub String);

impl EmbedError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl From<EmbedError> for AppError {
    fn from(err: EmbedError) -> Self {
        AppError::Embedding(err.0)
    }
}

pub type EmbedResult<T> = Result<T, EmbedError>;

/// Turns text into dense vectors.
///
/// Queries and documents are embedded separately because many retrieval models
/// (E5, BGE, Nomic, EmbeddingGemma, Voyage) expect different instructions or
/// prefixes for each side.
#[async_trait]
pub trait Embedder: Send + Sync + 'static {
    /// Provider name, e.g. `fastembed`.
    fn provider(&self) -> &'static str;
    /// Model identifier, e.g. `intfloat/multilingual-e5-small`.
    fn model(&self) -> &str;
    /// Dimension of the produced vectors.
    fn dimension(&self) -> usize;
    /// Embeds catalog texts. Implementations handle batching; output order matches input.
    async fn embed_documents(&self, texts: Vec<String>) -> EmbedResult<Vec<Vec<f32>>>;
    /// Embeds a search query.
    async fn embed_query(&self, text: &str) -> EmbedResult<Vec<f32>>;
}

/// Builds the embedder selected by configuration. May download model weights.
pub async fn build(settings: &EmbeddingSettings) -> anyhow::Result<Arc<dyn Embedder>> {
    let embedder: Arc<dyn Embedder> = match settings.provider {
        EmbeddingProvider::FastEmbed => build_fastembed(settings).await?,
        EmbeddingProvider::OpenAi => Arc::new(openai::OpenAiEmbedder::connect(settings).await?),
        EmbeddingProvider::Voyage => Arc::new(voyage::VoyageEmbedder::connect(settings).await?),
        EmbeddingProvider::Hash => Arc::new(hash::HashEmbedder::new(
            settings
                .dimension
                .unwrap_or(crate::config::DEFAULT_HASH_DIMENSION),
        )),
    };
    tracing::info!(
        provider = embedder.provider(),
        model = embedder.model(),
        dimension = embedder.dimension(),
        "embedder ready"
    );
    Ok(embedder)
}

#[cfg(feature = "fastembed")]
async fn build_fastembed(settings: &EmbeddingSettings) -> anyhow::Result<Arc<dyn Embedder>> {
    let settings = settings.clone();
    let embedder =
        tokio::task::spawn_blocking(move || fastembed::FastEmbedEmbedder::load(&settings))
            .await??;
    Ok(Arc::new(embedder))
}

#[cfg(not(feature = "fastembed"))]
async fn build_fastembed(_: &EmbeddingSettings) -> anyhow::Result<Arc<dyn Embedder>> {
    anyhow::bail!("this binary was built without the `fastembed` feature")
}

/// Query/document prefixes applied before embedding.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Prefixes {
    pub query: String,
    pub document: String,
}

impl Prefixes {
    /// The model's recommended prefixes, overridden by explicit settings.
    pub fn resolve(model: &str, settings: &EmbeddingSettings) -> Self {
        let (query, document) = recommended_prefixes(model);
        Self {
            query: settings
                .query_prefix
                .clone()
                .unwrap_or_else(|| query.to_string()),
            document: settings
                .document_prefix
                .clone()
                .unwrap_or_else(|| document.to_string()),
        }
    }

    pub fn query(&self, text: &str) -> String {
        format!("{}{}", self.query, text)
    }

    pub fn documents(&self, texts: Vec<String>) -> Vec<String> {
        if self.document.is_empty() {
            texts
        } else {
            texts
                .into_iter()
                .map(|t| format!("{}{t}", self.document))
                .collect()
        }
    }
}

/// Prefixes recommended by the authors of well-known retrieval models.
pub fn recommended_prefixes(model: &str) -> (&'static str, &'static str) {
    const BGE_EN_QUERY: &str = "Represent this sentence for searching relevant passages: ";
    let m = model.to_ascii_lowercase();
    if m.contains("multilingual-e5")
        || m.contains("/e5-")
        || m.starts_with("e5-")
        || m.contains("multilinguale5")
    {
        ("query: ", "passage: ")
    } else if m.contains("embeddinggemma") {
        ("task: search result | query: ", "title: none | text: ")
    } else if m.contains("nomic-embed-text") || m.contains("nomicembedtext") {
        ("search_query: ", "search_document: ")
    } else if m.contains("bge-") && m.contains("-zh") {
        ("为这个句子生成表示以用于检索相关文章：", "")
    } else if (m.contains("bge-") && m.contains("-en"))
        || m.contains("snowflake-arctic-embed")
        || m.contains("mxbai-embed-large")
    {
        (BGE_EN_QUERY, "")
    } else {
        ("", "")
    }
}

/// Whether a model is known to handle many languages.
pub fn is_multilingual(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    [
        "multilingual",
        "bge-m3",
        "embeddinggemma",
        "voyage-3",
        "voyage-multilingual",
        "text-embedding-3",
    ]
    .iter()
    .any(|k| m.contains(k))
}

/// A model that can be selected with `EMBEDDING_MODEL`.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ModelInfo {
    /// Value for `EMBEDDING_MODEL` (Hugging Face id).
    #[schema(example = "intfloat/multilingual-e5-small")]
    pub model: String,
    /// Alternative short name, also accepted by `EMBEDDING_MODEL`.
    #[schema(example = "MultilingualE5Small")]
    pub alias: String,
    pub dimension: usize,
    pub description: String,
    pub multilingual: bool,
}

/// Models available to the in-process provider.
pub fn supported_models() -> Vec<ModelInfo> {
    #[cfg(feature = "fastembed")]
    {
        fastembed::supported_models()
    }
    #[cfg(not(feature = "fastembed"))]
    {
        Vec::new()
    }
}

pub(crate) fn check_dimensions(vectors: &[Vec<f32>], expected: usize) -> EmbedResult<()> {
    match vectors.iter().find(|v| v.len() != expected) {
        Some(v) => Err(EmbedError::new(format!(
            "provider returned a {}-dimensional vector, expected {expected}",
            v.len()
        ))),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Settings;

    #[test]
    fn known_models_get_prefixes() {
        assert_eq!(
            recommended_prefixes("intfloat/multilingual-e5-small"),
            ("query: ", "passage: ")
        );
        assert_eq!(
            recommended_prefixes("Qdrant/multilingual-e5-large-onnx"),
            ("query: ", "passage: ")
        );
        assert_eq!(
            recommended_prefixes("nomic-ai/nomic-embed-text-v1.5").0,
            "search_query: "
        );
        assert!(
            recommended_prefixes("Xenova/bge-small-en-v1.5")
                .0
                .starts_with("Represent")
        );
        assert_eq!(recommended_prefixes("BAAI/bge-m3"), ("", ""));
        assert_eq!(recommended_prefixes("text-embedding-3-small"), ("", ""));
    }

    #[test]
    fn settings_override_prefixes() {
        let mut s = Settings::for_testing("/tmp").embedding;
        s.query_prefix = Some(String::new());
        s.document_prefix = Some("doc: ".into());
        let p = Prefixes::resolve("intfloat/multilingual-e5-small", &s);
        assert_eq!(p.query("hello"), "hello");
        assert_eq!(p.documents(vec!["a".into()]), vec!["doc: a"]);
    }

    #[test]
    fn detects_multilingual_models() {
        assert!(is_multilingual("intfloat/multilingual-e5-small"));
        assert!(is_multilingual("BAAI/bge-m3"));
        assert!(!is_multilingual("Xenova/bge-small-en-v1.5"));
    }
}
