use crate::{Result, TangentVector};
use nalgebra::{ComplexField, DMatrix, RealField};

pub trait RiemannianMetric {
    type Scalar: RealField;

    fn metric_tensor(&self, point: &[Self::Scalar]) -> Result<DMatrix<Self::Scalar>>;

    fn inner_product(
        &self,
        point: &[Self::Scalar],
        v1: &TangentVector<Self::Scalar>,
        v2: &TangentVector<Self::Scalar>,
    ) -> Result<Self::Scalar> {
        let g = self.metric_tensor(point)?;
        Ok(v1.components.dot(&(g * &v2.components)))
    }

    fn norm(
        &self,
        point: &[Self::Scalar],
        v: &TangentVector<Self::Scalar>,
    ) -> Result<Self::Scalar> {
        Ok(self.inner_product(point, v, v)?.sqrt())
    }
}

pub struct EuclideanMetric;

impl RiemannianMetric for EuclideanMetric {
    type Scalar = f64;

    fn metric_tensor(&self, point: &[Self::Scalar]) -> Result<DMatrix<Self::Scalar>> {
        let dim = point.len();
        Ok(DMatrix::identity(dim, dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DVector;

    #[test]
    fn test_euclidean_metric() {
        let metric = EuclideanMetric;
        let point = vec![0.0, 0.0, 0.0];
        let g = metric.metric_tensor(&point).unwrap();

        assert_eq!(g.nrows(), 3);
        assert_eq!(g.ncols(), 3);
        assert_relative_eq!(g[(0, 0)], 1.0);
        assert_relative_eq!(g[(1, 1)], 1.0);
        assert_relative_eq!(g[(2, 2)], 1.0);
    }

    #[test]
    fn test_euclidean_inner_product() {
        let metric = EuclideanMetric;
        let point = vec![0.0, 0.0];
        let v1 = TangentVector::new(DVector::from_vec(vec![1.0, 0.0]));
        let v2 = TangentVector::new(DVector::from_vec(vec![0.0, 1.0]));

        let inner = metric.inner_product(&point, &v1, &v2).unwrap();
        assert_relative_eq!(inner, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_euclidean_norm() {
        let metric = EuclideanMetric;
        let point = vec![0.0, 0.0];
        let v = TangentVector::new(DVector::from_vec(vec![3.0, 4.0]));

        let norm = metric.norm(&point, &v).unwrap();
        assert_relative_eq!(norm, 5.0, epsilon = 1e-10);
    }
}
