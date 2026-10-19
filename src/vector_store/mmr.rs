//! Maximal Marginal Relevance (MMR) re-ranking.
//!
//! Greedily picks the candidate maximizing
//! `(1 - diversity) * relevance - diversity * max_similarity_to_already_picked`,
//! trading a little relevance for less redundant result lists.

use super::{dot, normalized};

/// A candidate with its relevance score and vector.
pub struct Candidate<T> {
    pub value: T,
    pub score: f32,
    pub vector: Vec<f32>,
}

/// Returns up to `limit` candidates in MMR order. Scores are left untouched.
pub fn rerank<T>(candidates: Vec<Candidate<T>>, limit: usize, diversity: f32) -> Vec<(T, f32)> {
    let lambda = diversity.clamp(0.0, 1.0);
    let mut pool: Vec<(T, f32, Vec<f32>)> = candidates
        .into_iter()
        .map(|c| (c.value, c.score, normalized(&c.vector)))
        .collect();
    let mut picked_vectors: Vec<Vec<f32>> = Vec::with_capacity(limit);
    let mut out = Vec::with_capacity(limit.min(pool.len()));

    while out.len() < limit && !pool.is_empty() {
        let best = pool
            .iter()
            .enumerate()
            .map(|(i, (_, score, vector))| {
                let redundancy = picked_vectors
                    .iter()
                    .map(|p| dot(p, vector))
                    .fold(f32::MIN, f32::max);
                let redundancy = if picked_vectors.is_empty() {
                    0.0
                } else {
                    redundancy
                };
                (i, (1.0 - lambda) * score - lambda * redundancy)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
            .expect("pool is not empty");
        let (value, score, vector) = pool.swap_remove(best);
        picked_vectors.push(vector);
        out.push((value, score));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(id: &'static str, score: f32, vector: Vec<f32>) -> Candidate<&'static str> {
        Candidate {
            value: id,
            score,
            vector,
        }
    }

    #[test]
    fn zero_diversity_keeps_relevance_order() {
        let out = rerank(
            vec![
                c("a", 0.9, vec![1.0, 0.0]),
                c("b", 0.8, vec![1.0, 0.01]),
                c("c", 0.5, vec![0.0, 1.0]),
            ],
            3,
            0.0,
        );
        assert_eq!(
            out.iter().map(|x| x.0).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
    }

    #[test]
    fn high_diversity_skips_near_duplicates() {
        let out = rerank(
            vec![
                c("a", 0.9, vec![1.0, 0.0]),
                c("a-dup", 0.89, vec![1.0, 0.001]),
                c("other", 0.6, vec![0.0, 1.0]),
            ],
            2,
            0.7,
        );
        assert_eq!(
            out.iter().map(|x| x.0).collect::<Vec<_>>(),
            vec!["a", "other"]
        );
        assert_eq!(out[1].1, 0.6, "original scores are preserved");
    }

    #[test]
    fn respects_limit_and_empty_input() {
        assert!(rerank::<&str>(vec![], 5, 0.5).is_empty());
        assert_eq!(rerank(vec![c("a", 1.0, vec![1.0])], 0, 0.5).len(), 0);
    }
}
