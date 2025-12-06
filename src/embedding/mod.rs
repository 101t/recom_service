pub mod local_embedder;
pub mod voyage_embedder;

use async_trait::async_trait;
use anyhow::Result;

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    // async fn batch_embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Mock embedder for testing purposes
    pub struct MockEmbedder {
        pub embedding_size: usize,
        pub should_fail: bool,
    }

    impl MockEmbedder {
        pub fn new(embedding_size: usize) -> Self {
            Self {
                embedding_size,
                should_fail: false,
            }
        }

        pub fn failing() -> Self {
            Self {
                embedding_size: 0,
                should_fail: true,
            }
        }
    }

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, text: &str) -> Result<Vec<f32>> {
            if self.should_fail {
                anyhow::bail!("Mock embedding failure");
            }
            // Generate deterministic embedding based on text length
            let seed = text.len() as f32;
            Ok((0..self.embedding_size)
                .map(|i| (seed + i as f32) / self.embedding_size as f32)
                .collect())
        }
    }

    #[tokio::test]
    async fn test_mock_embedder_success() {
        let embedder = MockEmbedder::new(128);
        let result = embedder.embed("test query").await;
        assert!(result.is_ok());
        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 128);
    }

    #[tokio::test]
    async fn test_mock_embedder_deterministic() {
        let embedder = MockEmbedder::new(64);
        let result1 = embedder.embed("same text").await.unwrap();
        let result2 = embedder.embed("same text").await.unwrap();
        assert_eq!(result1, result2);
    }

    #[tokio::test]
    async fn test_mock_embedder_different_inputs() {
        let embedder = MockEmbedder::new(64);
        let result1 = embedder.embed("short").await.unwrap();
        let result2 = embedder.embed("longer text here").await.unwrap();
        assert_ne!(result1, result2);
    }

    #[tokio::test]
    async fn test_mock_embedder_failure() {
        let embedder = MockEmbedder::failing();
        let result = embedder.embed("test").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Mock embedding failure"));
    }

    #[test]
    fn test_embedder_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockEmbedder>();
    }
}