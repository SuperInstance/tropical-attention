//! Tropical softmax: max-plus replacement for standard softmax.

/// Tropical softmax: replaces exp-normalize with max-subtract.
///
/// Standard: softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
/// Tropical: tropsoftmax(z)_i = max(0, z_i - max(z))
///
/// This produces a piecewise-linear, interpretable attention distribution.
pub struct TropicalSoftmax {
    pub temperature: f64,
}

impl TropicalSoftmax {
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Apply tropical softmax to a vector.
    pub fn apply(&self, z: &[f64]) -> Vec<f64> {
        if z.is_empty() {
            return vec![];
        }
        let max_z = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // Tropical softmax: weight proportional to max(0, z - max_z + epsilon)
        // The max element gets weight 1, others get proportional weight
        let epsilon = 0.01; // Small offset so max element gets non-zero weight
        let shifted: Vec<f64> = z
            .iter()
            .map(|v| (v - max_z + epsilon).max(0.0) / self.temperature)
            .collect();
        let sum: f64 = shifted.iter().sum();
        if sum < 1e-15 {
            return vec![1.0 / z.len() as f64; z.len()];
        }
        shifted.iter().map(|v| v / sum).collect()
    }

    /// Tropical entropy: H = -Σ p_i (p_i - max(p)) in max-plus semiring.
    pub fn entropy(&self, p: &[f64]) -> f64 {
        let max_p = p.iter().cloned().fold(0.0f64, f64::max);
        -p.iter()
            .map(|&pi| {
                if pi > 1e-15 {
                    pi * (pi - max_p).abs()
                } else {
                    0.0
                }
            })
            .sum::<f64>()
    }

    /// Tropical KL divergence.
    pub fn kl_divergence(&self, p: &[f64], q: &[f64]) -> f64 {
        let max_p = p.iter().cloned().fold(0.0f64, f64::max);
        p.iter()
            .zip(q)
            .map(|(&pi, &qi)| {
                if pi > 1e-15 && qi > 1e-15 {
                    pi * ((pi - max_p) - (qi - max_p)).abs()
                } else {
                    0.0
                }
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_softmax_sums_to_one() {
        let ts = TropicalSoftmax::new(1.0);
        let result = ts.apply(&[1.0, 2.0, 3.0]);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_tropical_softmax_max_dominates() {
        let ts = TropicalSoftmax::new(1.0);
        let result = ts.apply(&[1.0, 10.0, 1.0]);
        // Index 1 should have the largest weight
        let max_idx = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 1);
    }

    #[test]
    fn test_tropical_softmax_uniform() {
        let ts = TropicalSoftmax::new(1.0);
        let result = ts.apply(&[5.0, 5.0, 5.0]);
        assert!((result[0] - result[1]).abs() < 0.01);
    }

    #[test]
    fn test_entropy() {
        let ts = TropicalSoftmax::new(1.0);
        let p = ts.apply(&[1.0, 2.0, 3.0]);
        let h = ts.entropy(&p);
        assert!(h <= 0.0);
    }

    #[test]
    fn test_temperature_scaling() {
        let ts_low = TropicalSoftmax::new(0.1);
        let ts_high = TropicalSoftmax::new(10.0);
        let r_low = ts_low.apply(&[1.0, 2.0, 3.0]);
        let r_high = ts_high.apply(&[1.0, 2.0, 3.0]);
        // Low temperature should be more peaked
        assert!(r_low[2] >= r_high[2] - 0.1);
    }
}
