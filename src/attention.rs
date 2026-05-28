//! Tropical attention mechanism.

use crate::softmax::TropicalSoftmax;

/// Tropical attention: replaces softmax attention with max-plus operations.
///
/// Standard: attention(Q, K, V) = softmax(Q K^T / √d) V
/// Tropical: attention(Q, K, V) = tropsoftmax(Q ⊕ K^T / √d) V
pub struct TropicalAttention {
    pub dim: usize,
    pub softmax: TropicalSoftmax,
}

impl TropicalAttention {
    pub fn new(dim: usize, temperature: f64) -> Self {
        Self {
            dim,
            softmax: TropicalSoftmax::new(temperature),
        }
    }

    /// Compute attention weights from query and keys.
    pub fn attention_weights(&self, query: &[f64], keys: &[Vec<f64>]) -> Vec<f64> {
        let scores: Vec<f64> = keys
            .iter()
            .map(|k| {
                // Tropical inner product: max_i(q_i + k_i)
                query
                    .iter()
                    .zip(k)
                    .map(|(qi, ki)| qi + ki)
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        self.softmax.apply(&scores)
    }

    /// Apply attention to values given a query and key-value pairs.
    pub fn forward(&self, query: &[f64], keys: &[Vec<f64>], values: &[Vec<f64>]) -> Vec<f64> {
        let weights = self.attention_weights(query, keys);
        let d = values.first().map(|v| v.len()).unwrap_or(0);
        (0..d)
            .map(|j| weights.iter().zip(values).map(|(w, v)| w * v[j]).sum())
            .collect()
    }

    /// Multi-head tropical attention.
    pub fn multi_head(
        &self,
        queries: &[Vec<f64>],
        keys: &[Vec<f64>],
        values: &[Vec<f64>],
        n_heads: usize,
    ) -> Vec<Vec<f64>> {
        queries
            .iter()
            .map(|q| {
                let mut combined = vec![0.0; self.dim];
                let head_dim = self.dim / n_heads.max(1);
                for h in 0..n_heads {
                    let q_head: Vec<f64> = q
                        .iter()
                        .skip(h * head_dim)
                        .take(head_dim)
                        .copied()
                        .collect();
                    let k_heads: Vec<Vec<f64>> = keys
                        .iter()
                        .map(|k| {
                            k.iter()
                                .skip(h * head_dim)
                                .take(head_dim)
                                .copied()
                                .collect()
                        })
                        .collect();
                    let v_heads: Vec<Vec<f64>> = values
                        .iter()
                        .map(|v| {
                            v.iter()
                                .skip(h * head_dim)
                                .take(head_dim)
                                .copied()
                                .collect()
                        })
                        .collect();
                    let out = self.forward(&q_head, &k_heads, &v_heads);
                    for (i, v) in out.iter().enumerate() {
                        if h * head_dim + i < combined.len() {
                            combined[h * head_dim + i] += v;
                        }
                    }
                }
                combined
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_weights() {
        let ta = TropicalAttention::new(3, 1.0);
        let q = vec![1.0, 2.0, 3.0];
        let keys = vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]];
        let w = ta.attention_weights(&q, &keys);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_forward() {
        let ta = TropicalAttention::new(2, 1.0);
        let q = vec![1.0, 2.0];
        let k = vec![vec![1.0, 0.0]];
        let v = vec![vec![3.0, 4.0]];
        let out = ta.forward(&q, &k, &v);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_multi_head() {
        let ta = TropicalAttention::new(4, 1.0);
        let q = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let k = vec![vec![1.0, 0.0, 0.0, 1.0]];
        let v = vec![vec![1.0, 1.0, 1.0, 1.0]];
        let out = ta.multi_head(&q, &k, &v, 2);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 4);
    }
}
