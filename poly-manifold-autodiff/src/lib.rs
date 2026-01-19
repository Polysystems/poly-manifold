pub mod dual;
pub mod gradient;
pub mod optimizer;

pub use dual::Dual;
pub use gradient::{numerical_gradient, riemannian_gradient};
pub use optimizer::{GradientDescent, RiemannianOptimizer};
