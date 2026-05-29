use tropical_attention::*;

#[test]
fn test_tropical_attention_weights() {
    let ta = TropicalAttention::new(4, 1.0);
    let query = vec![1.0, 2.0, 3.0, 4.0];
    let keys = vec![vec![0.5, 1.0, 1.5, 2.0], vec![2.0, 1.0, 0.5, 0.25]];
    let weights = ta.attention_weights(&query, &keys);
    assert_eq!(weights.len(), 2);
    for w in &weights {
        assert!(*w >= 0.0);
    }
}

#[test]
fn test_tropical_attention_forward() {
    let ta = TropicalAttention::new(3, 1.0);
    let q = vec![1.0, 0.0, -1.0];
    let k = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let v = vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]];
    let out = ta.forward(&q, &k, &v);
    assert_eq!(out.len(), 3);
}

#[test]
fn test_tropical_softmax_single() {
    let ts = TropicalSoftmax::new(1.0);
    let weights = ts.apply(&[5.0]);
    assert_eq!(weights.len(), 1);
    assert!((weights[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_tropical_softmax_dominant() {
    let ts = TropicalSoftmax::new(1.0);
    let weights = ts.apply(&[100.0, -100.0]);
    assert!(weights[0] > weights[1]);
}

#[test]
fn test_tropical_softmax_empty() {
    let ts = TropicalSoftmax::new(1.0);
    assert_eq!(ts.apply(&[]), vec![]);
}

#[test]
fn test_tropical_entropy() {
    let ts = TropicalSoftmax::new(1.0);
    let p = ts.apply(&[1.0, 2.0, 3.0]);
    let h = ts.entropy(&p);
    assert!(h <= 0.0); // tropical entropy is non-positive
}

#[test]
fn test_newton_polytope_vertices() {
    let np = NewtonPolytope::new(vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]);
    assert_eq!(np.n_vertices(), 3);
    assert_eq!(np.dimension, 2);
}

#[test]
fn test_newton_polytope_evaluate() {
    let np = NewtonPolytope::new(vec![vec![0.0], vec![1.0]]);
    let val = np.evaluate(&[0.0, 1.0], &[2.0]);
    // max(0+0*2, 1+1*2) = max(0, 3) = 3
    assert!((val - 3.0).abs() < 1e-10);
}

#[test]
fn test_newton_polytope_centroid() {
    let np = NewtonPolytope::new(vec![vec![0.0, 0.0], vec![2.0, 0.0], vec![0.0, 2.0]]);
    let c = np.centroid();
    assert!((c[0] - 2.0/3.0).abs() < 1e-10);
    assert!((c[1] - 2.0/3.0).abs() < 1e-10);
}
