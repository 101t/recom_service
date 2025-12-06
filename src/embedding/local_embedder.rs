use super::Embedder;
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Serialize)]
struct EmbedRequest {
    text: String,
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
    #[allow(dead_code)]
    dimension: usize,
}

/// Local embedder that calls the FastEmbed service
pub struct LocalEmbedder {
    client: Client,
    base_url: String,
}

impl LocalEmbedder {
    pub fn new() -> Self {
        let host = env::var("EMBEDDING_SERVICE_HOST")
            .unwrap_or_else(|_| "127.0.0.1".to_string());
        let port = env::var("EMBEDDING_SERVICE_PORT")
            .unwrap_or_else(|_| "8001".to_string());
        
        let base_url = format!("http://{}:{}", host, port);
        
        log::info!("LocalEmbedder configured to use: {}", base_url);
        
        Self {
            client: Client::new(),
            base_url,
        }
    }
    
    #[allow(dead_code)]
    pub fn with_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }
}

impl Default for LocalEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Embedder for LocalEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request = EmbedRequest {
            text: text.to_string(),
        };
        
        let response = self.client
            .post(format!("{}/api/v1/embed", self.base_url))
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Embedding service error ({}): {}", status, error_text);
        }
        
        let embed_response: EmbedResponse = response.json().await?;
        Ok(embed_response.embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_embedder_creation() {
        let embedder = LocalEmbedder::new();
        assert!(embedder.base_url.starts_with("http://"));
    }

    #[test]
    fn test_local_embedder_with_url() {
        let embedder = LocalEmbedder::with_url("http://localhost:9000");
        assert_eq!(embedder.base_url, "http://localhost:9000");
    }

    #[test]
    fn test_embed_request_serialization() {
        let request = EmbedRequest {
            text: "test query".to_string(),
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("test query"));
    }

    #[test]
    fn test_embed_response_deserialization() {
        let json = r#"{"embedding": [0.1, 0.2, 0.3], "dimension": 3}"#;
        let response: EmbedResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.embedding.len(), 3);
        assert_eq!(response.dimension, 3);
    }
}
