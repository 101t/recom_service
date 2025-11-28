pub mod voyage_embedder;

use async_trait::async_trait;
use anyhow::Result;

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    // async fn batch_embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}