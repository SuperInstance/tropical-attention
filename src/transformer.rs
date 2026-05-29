//! Tropical transformer layer.

use crate::attention::TropicalAttention;

/// A tropical transformer layer: tropical attention + tropical feed-forward.
pub struct TropicalTransformerLayer {
    pub attention: TropicalAttention,
    pub ff_dim: usize,
}

impl TropicalTransformerLayer {
    pub fn new(dim: usize, ff_dim: usize, temperature: f64) -> Self {
        Self {
            attention: TropicalAttention::new(dim, temperature),
            ff_dim,
        }
    }

    /// Forward pass: attention + residual + tropical ReLU FFN + residual.
    pub fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let _n = input.len();
        let dim = input.first().map(|v| v.len()).unwrap_or(0);

        // Self-attention
        let attended = self.attention.multi_head(input, input, input, 1);

        // Residual connection
        let residual: Vec<Vec<f64>> = attended
            .iter()
            .zip(input)
            .map(|(a, x)| a.iter().zip(x).map(|(ai, xi)| ai + xi).collect())
            .collect();

        // Tropical feed-forward: max(Wx + b, 0) (tropical ReLU)
        residual
            .iter()
            .map(|v| (0..dim).map(|i| v[i].max(0.0)).collect())
            .collect()
    }

    /// Number of parameters (approximate).
    pub fn param_count(&self) -> usize {
        let d = self.attention.dim;
        d * d * 4 + d * self.ff_dim * 2 + self.ff_dim * d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_forward() {
        let layer = TropicalTransformerLayer::new(4, 8, 1.0);
        let input = vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.5, 1.0, 1.5, 2.0]];
        let out = layer.forward(&input);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn test_param_count() {
        let layer = TropicalTransformerLayer::new(4, 8, 1.0);
        assert!(layer.param_count() > 0);
    }

    #[test]
    fn test_residual_preserves_shape() {
        let layer = TropicalTransformerLayer::new(3, 6, 1.0);
        let input = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let out = layer.forward(&input);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 3);
    }
}
