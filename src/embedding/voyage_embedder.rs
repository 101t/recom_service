use super::Embedder;
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use async_trait::async_trait;

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

pub struct VoyageEmbedder {
    client: Client,
    api_key: String,
    model: String,
}

impl VoyageEmbedder {
    pub fn new() -> Self {
        let api_key = env::var("VOYAGE_API_KEY")
            .expect("VOYAGE_API_KEY must be set");
            
        Self {
            client: Client::new(),
            api_key,
            model: "voyage-3-large".to_string(),
        }
    }
}

#[async_trait]
impl Embedder for VoyageEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request = EmbeddingRequest {
            model: self.model.clone(),
            input: vec![text.to_string()],
        };
        
        let response = self.client
            .post("https://api.voyageai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;
            
        if !response.status().is_success() {
            anyhow::bail!("API request failed: {}", response.status());
        }
        
        let embedding_response: EmbeddingResponse = response.json().await?;
        
        match embedding_response.data.first() {
            Some(data) => Ok(data.embedding.clone()),
            None => anyhow::bail!("No embedding data received"),
        }
    }
    
    async fn batch_embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let request = EmbeddingRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        };
        
        let response = self.client
            .post("https://api.voyageai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;
            
        if !response.status().is_success() {
            anyhow::bail!("API request failed: {}", response.status());
        }
        
        let embedding_response: EmbeddingResponse = response.json().await?;
        
        Ok(embedding_response.data
            .into_iter()
            .map(|data| data.embedding)
            .collect())
    }
}