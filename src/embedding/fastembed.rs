//! In-process embeddings with [fastembed](https://github.com/Anush008/fastembed-rs) (ONNX Runtime).
//!
//! Models are downloaded from Hugging Face on first use and cached in
//! `EMBEDDING_CACHE_DIR`. Inference is CPU-bound, so it runs on Tokio's blocking
//! pool; `EMBEDDING_POOL_SIZE` model instances allow concurrent requests.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use async_trait::async_trait;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use super::{EmbedError, EmbedResult, Embedder, ModelInfo, Prefixes, is_multilingual};
use crate::config::EmbeddingSettings;

pub struct FastEmbedEmbedder {
    inner: Arc<Inner>,
}

struct Inner {
    pool: Vec<Mutex<TextEmbedding>>,
    next: AtomicUsize,
    model: String,
    dimension: usize,
    batch_size: usize,
    prefixes: Prefixes,
}

impl FastEmbedEmbedder {
    /// Loads (and if needed downloads) the configured model. Blocking.
    pub fn load(settings: &EmbeddingSettings) -> anyhow::Result<Self> {
        let (model, info) = resolve_model(&settings.model)?;
        std::fs::create_dir_all(&settings.cache_dir).with_context(|| {
            format!(
                "cannot create model cache dir {}",
                settings.cache_dir.display()
            )
        })?;
        tracing::info!(
            model = %info.model_code,
            cache_dir = %settings.cache_dir.display(),
            instances = settings.pool_size,
            "loading embedding model (first start downloads it from Hugging Face)"
        );

        let pool = (0..settings.pool_size)
            .map(|_| {
                let mut options = TextInitOptions::new(model.clone())
                    .with_cache_dir(settings.cache_dir.clone())
                    .with_show_download_progress(false);
                if let Some(max_length) = settings.max_length {
                    options = options.with_max_length(max_length);
                }
                if let Some(threads) = settings.threads {
                    options = options.with_intra_threads(threads);
                }
                TextEmbedding::try_new(options).map(Mutex::new)
            })
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("failed to load embedding model {}", info.model_code))?;

        Ok(Self {
            inner: Arc::new(Inner {
                pool,
                next: AtomicUsize::new(0),
                prefixes: Prefixes::resolve(&info.model_code, settings),
                model: info.model_code.clone(),
                dimension: info.dim,
                batch_size: settings.batch_size,
            }),
        })
    }
}

impl Inner {
    fn embed_blocking(&self, texts: &[String]) -> EmbedResult<Vec<Vec<f32>>> {
        let slot = self.next.fetch_add(1, Ordering::Relaxed) % self.pool.len();
        let mut model = self.pool[slot]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        model
            .embed(texts, Some(self.batch_size))
            .map_err(|e| EmbedError::new(format!("fastembed inference failed: {e}")))
    }
}

#[async_trait]
impl Embedder for FastEmbedEmbedder {
    fn provider(&self) -> &'static str {
        "fastembed"
    }

    fn model(&self) -> &str {
        &self.inner.model
    }

    fn dimension(&self) -> usize {
        self.inner.dimension
    }

    async fn embed_documents(&self, texts: Vec<String>) -> EmbedResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let inner = Arc::clone(&self.inner);
        let texts = inner.prefixes.documents(texts);
        tokio::task::spawn_blocking(move || inner.embed_blocking(&texts))
            .await
            .map_err(|e| EmbedError::new(format!("embedding task failed: {e}")))?
    }

    async fn embed_query(&self, text: &str) -> EmbedResult<Vec<f32>> {
        let inner = Arc::clone(&self.inner);
        let text = inner.prefixes.query(text);
        let mut vectors = tokio::task::spawn_blocking(move || inner.embed_blocking(&[text]))
            .await
            .map_err(|e| EmbedError::new(format!("embedding task failed: {e}")))??;
        vectors
            .pop()
            .ok_or_else(|| EmbedError::new("fastembed returned no vector"))
    }
}

/// Finds a model by Hugging Face id (`intfloat/multilingual-e5-small`) or fastembed
/// alias (`MultilingualE5Small`), case-insensitively. Non-quantized variants win ties.
pub fn resolve_model(
    name: &str,
) -> anyhow::Result<(EmbeddingModel, fastembed::ModelInfo<EmbeddingModel>)> {
    let wanted = name.trim().to_ascii_lowercase();
    TextEmbedding::list_supported_models()
        .into_iter()
        .find(|m| format!("{:?}", m.model).to_ascii_lowercase() == wanted || m.model_code.to_ascii_lowercase() == wanted)
        .map(|info| (info.model.clone(), info))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "unknown fastembed model '{name}'. GET /api/v1/models lists the supported models; \
                 use EMBEDDING_PROVIDER=openai to serve other models through an OpenAI-compatible server"
            )
        })
}

pub fn supported_models() -> Vec<ModelInfo> {
    TextEmbedding::list_supported_models()
        .into_iter()
        .map(|m| ModelInfo {
            multilingual: is_multilingual(&m.model_code),
            alias: format!("{:?}", m.model),
            model: m.model_code,
            dimension: m.dim,
            description: m.description,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_models_by_code_or_alias() {
        let (by_code, info) = resolve_model("intfloat/multilingual-e5-small").unwrap();
        let (by_alias, _) = resolve_model("multilingualE5small").unwrap();
        assert_eq!(by_code, by_alias);
        assert_eq!(info.dim, 384);
        assert!(resolve_model("not-a-model").is_err());
    }

    #[test]
    fn lists_multilingual_models() {
        let models = supported_models();
        assert!(
            models
                .iter()
                .any(|m| m.model == "intfloat/multilingual-e5-small" && m.multilingual)
        );
        assert!(
            models
                .iter()
                .any(|m| m.model == "BAAI/bge-m3" && m.multilingual)
        );
    }
}
