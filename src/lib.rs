#![allow(
    clippy::needless_range_loop,
    clippy::new_without_default,
    clippy::type_complexity,
    dead_code
)]
//! # Tropical Attention
//!
//! Tropical (max-plus) attention mechanisms for neural architectures.
//! Replaces softmax with tropical operations for piecewise-linear decision boundaries.

mod attention;
mod polytope;
mod softmax;
mod transformer;

pub use attention::TropicalAttention;
pub use polytope::NewtonPolytope;
pub use softmax::TropicalSoftmax;
pub use transformer::TropicalTransformerLayer;
