pub mod error;
pub mod manifold;
pub mod metric;
pub mod tangent;

pub use error::{ManifoldError, Result};
pub use manifold::Manifold;
pub use metric::RiemannianMetric;
pub use tangent::TangentVector;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
