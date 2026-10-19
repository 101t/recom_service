//! Deterministic feature-hashing embedder.
//!
//! Hashes words and character trigrams into a fixed-size vector. It captures
//! lexical overlap only (no semantics), needs no model download and is fully
//! reproducible, which makes it ideal for tests, CI and offline development.
//! Do not use it in production.

use async_trait::async_trait;

use super::{EmbedResult, Embedder};

pub struct HashEmbedder {
    dimension: usize,
    model: String,
}

impl HashEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            model: format!("hash-{dimension}"),
        }
    }

    pub fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dimension];
        let lower = text.to_lowercase();
        for word in lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
        {
            self.add(&mut v, word, 1.0);
            let chars: Vec<char> = format!("#{word}#").chars().collect();
            for gram in chars.windows(3) {
                self.add(&mut v, &gram.iter().collect::<String>(), 0.5);
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        } else {
            // Keep the vector valid for cosine similarity.
            v[0] = 1.0;
        }
        v
    }

    fn add(&self, v: &mut [f32], feature: &str, weight: f32) {
        let h = fnv1a(feature.as_bytes());
        let idx = (h % self.dimension as u64) as usize;
        let sign = if (h >> 63) == 0 { 1.0 } else { -1.0 };
        v[idx] += sign * weight;
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
    })
}

#[async_trait]
impl Embedder for HashEmbedder {
    fn provider(&self) -> &'static str {
        "hash"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed_documents(&self, texts: Vec<String>) -> EmbedResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_text(t)).collect())
    }

    async fn embed_query(&self, text: &str) -> EmbedResult<Vec<f32>> {
        Ok(self.embed_text(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[tokio::test]
    async fn is_deterministic_and_normalized() {
        let e = HashEmbedder::new(64);
        let a = e.embed_query("Running shoes").await.unwrap();
        let b = e.embed_query("running SHOES").await.unwrap();
        assert_eq!(a, b);
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-5);
        assert_eq!(e.embed_query("").await.unwrap()[0], 1.0);
    }

    #[tokio::test]
    async fn lexical_overlap_scores_higher() {
        let e = HashEmbedder::new(384);
        let q = e.embed_query("running shoes").await.unwrap();
        let docs = e
            .embed_documents(vec![
                "lightweight running shoes".into(),
                "noise cancelling headphones".into(),
            ])
            .await
            .unwrap();
        assert!(cosine(&q, &docs[0]) > cosine(&q, &docs[1]));
    }
}
