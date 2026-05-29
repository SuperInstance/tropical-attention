//! Newton polytope for tropical polynomial analysis.

/// Newton polytope: the convex hull of the exponent vectors of a tropical polynomial.
///
/// For a tropical polynomial f(x) = max_i (c_i + ⟨α_i, x⟩),
/// the Newton polytope is ConvHull({α_i}).
pub struct NewtonPolytope {
    pub vertices: Vec<Vec<f64>>,
    pub dimension: usize,
}

impl NewtonPolytope {
    /// Create from exponent vectors.
    pub fn new(vertices: Vec<Vec<f64>>) -> Self {
        let dim = vertices.first().map(|v| v.len()).unwrap_or(0);
        Self {
            vertices,
            dimension: dim,
        }
    }

    /// Number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Evaluate the tropical polynomial at a point.
    /// f(x) = max_i (c_i + ⟨α_i, x⟩) where c_i are coefficients, α_i are exponents.
    pub fn evaluate(&self, coefficients: &[f64], point: &[f64]) -> f64 {
        self.vertices
            .iter()
            .zip(coefficients)
            .map(|(v, c)| {
                let dot: f64 = v.iter().zip(point).map(|(vi, pi)| vi * pi).sum();
                c + dot
            })
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// The tropical variety (corner locus): set where the max is achieved by at least 2 terms.
    pub fn is_on_variety(&self, coefficients: &[f64], point: &[f64]) -> bool {
        let values: Vec<f64> = self
            .vertices
            .iter()
            .zip(coefficients)
            .map(|(v, c)| c + v.iter().zip(point).map(|(vi, pi)| vi * pi).sum::<f64>())
            .collect();
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let count = values
            .iter()
            .filter(|&&v| (v - max_val).abs() < 1e-10)
            .count();
        count >= 2
    }

    /// Volume of the polytope (simplified: bounding box volume).
    pub fn volume(&self) -> f64 {
        if self.vertices.is_empty() || self.dimension == 0 {
            return 0.0;
        }
        let mut vol = 1.0;
        for d in 0..self.dimension {
            let coords: Vec<f64> = self.vertices.iter().map(|v| v[d]).collect();
            let min = coords.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = coords.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            vol *= (max - min).max(0.0);
        }
        vol
    }

    /// Centroid of the polytope.
    pub fn centroid(&self) -> Vec<f64> {
        if self.vertices.is_empty() {
            return vec![];
        }
        let _n = self.vertices.len() as f64;
        (0..self.dimension)
            .map(|d| self.vertices.iter().map(|v| v[d]).sum::<f64>() / n)
            .collect()
    }

    /// Subdivide: split the polytope at a given point (for mixed subdivision).
    pub fn subdivide(&self, _lift: &[f64]) -> Vec<NewtonPolytope> {
        if self.vertices.len() <= self.dimension + 1 {
            return vec![self.clone()];
        }
        // Simplified: just return the whole polytope
        vec![self.clone()]
    }
}

impl Clone for NewtonPolytope {
    fn clone(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            dimension: self.dimension,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let np = NewtonPolytope::new(vec![vec![0.0], vec![1.0], vec![2.0]]);
        assert_eq!(np.n_vertices(), 3);
    }

    #[test]
    fn test_evaluate() {
        let np = NewtonPolytope::new(vec![vec![0.0], vec![1.0], vec![2.0]]);
        let val = np.evaluate(&[1.0, 2.0, 3.0], &[1.0]);
        // max(1+0, 2+1, 3+2) = 5
        assert!((val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_variety() {
        let np = NewtonPolytope::new(vec![vec![0.0], vec![1.0]]);
        // At x=0: max(1+0, 2+0) = 2 → only one term
        assert!(!np.is_on_variety(&[1.0, 2.0], &[0.0]));
    }

    #[test]
    fn test_volume() {
        let np = NewtonPolytope::new(vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ]);
        let v = np.volume();
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_centroid() {
        let np = NewtonPolytope::new(vec![vec![0.0], vec![2.0]]);
        let c = np.centroid();
        assert!((c[0] - 1.0).abs() < 1e-10);
    }
}
