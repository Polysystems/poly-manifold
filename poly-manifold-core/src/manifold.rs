use crate::{Result, TangentVector};
use nalgebra::{ComplexField, RealField};

pub trait Manifold {
    type Scalar: RealField;

    fn dim(&self) -> usize;

    fn check_point(&self, point: &[Self::Scalar]) -> Result<()>;

    fn check_tangent_vector(
        &self,
        point: &[Self::Scalar],
        tangent: &TangentVector<Self::Scalar>,
    ) -> Result<()>;

    fn project_to_manifold(&self, point: &[Self::Scalar]) -> Result<Vec<Self::Scalar>>;

    fn project_to_tangent_space(
        &self,
        point: &[Self::Scalar],
        vector: &TangentVector<Self::Scalar>,
    ) -> Result<TangentVector<Self::Scalar>>;

    fn exp(
        &self,
        point: &[Self::Scalar],
        tangent: &TangentVector<Self::Scalar>,
    ) -> Result<Vec<Self::Scalar>>;

    fn log(
        &self,
        point: &[Self::Scalar],
        other: &[Self::Scalar],
    ) -> Result<TangentVector<Self::Scalar>>;

    fn inner_product(
        &self,
        point: &[Self::Scalar],
        v1: &TangentVector<Self::Scalar>,
        v2: &TangentVector<Self::Scalar>,
    ) -> Result<Self::Scalar>;

    fn norm(
        &self,
        point: &[Self::Scalar],
        v: &TangentVector<Self::Scalar>,
    ) -> Result<Self::Scalar> {
        Ok(self.inner_product(point, v, v)?.sqrt())
    }

    fn retraction(
        &self,
        point: &[Self::Scalar],
        tangent: &TangentVector<Self::Scalar>,
    ) -> Result<Vec<Self::Scalar>> {
        self.exp(point, tangent)
    }

    fn distance(&self, point1: &[Self::Scalar], point2: &[Self::Scalar]) -> Result<Self::Scalar> {
        let log_vec = self.log(point1, point2)?;
        self.norm(point1, &log_vec)
    }

    fn geodesic(
        &self,
        point: &[Self::Scalar],
        tangent: &TangentVector<Self::Scalar>,
        t: Self::Scalar,
    ) -> Result<Vec<Self::Scalar>> {
        let scaled_tangent = tangent.clone() * t;
        self.exp(point, &scaled_tangent)
    }

    fn parallel_transport(
        &self,
        point: &[Self::Scalar],
        tangent: &TangentVector<Self::Scalar>,
        direction: &TangentVector<Self::Scalar>,
    ) -> Result<TangentVector<Self::Scalar>> {
        let new_point = self.exp(point, direction)?;
        self.project_to_tangent_space(&new_point, tangent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    struct TestEuclideanManifold {
        dimension: usize,
    }

    impl Manifold for TestEuclideanManifold {
        type Scalar = f64;

        fn dim(&self) -> usize {
            self.dimension
        }

        fn check_point(&self, point: &[Self::Scalar]) -> Result<()> {
            if point.len() != self.dimension {
                return Err(crate::ManifoldError::DimensionMismatch {
                    expected: self.dimension,
                    got: point.len(),
                });
            }
            Ok(())
        }

        fn check_tangent_vector(
            &self,
            point: &[Self::Scalar],
            tangent: &TangentVector<Self::Scalar>,
        ) -> Result<()> {
            self.check_point(point)?;
            if tangent.dim() != self.dimension {
                return Err(crate::ManifoldError::DimensionMismatch {
                    expected: self.dimension,
                    got: tangent.dim(),
                });
            }
            Ok(())
        }

        fn project_to_manifold(&self, point: &[Self::Scalar]) -> Result<Vec<Self::Scalar>> {
            Ok(point.to_vec())
        }

        fn project_to_tangent_space(
            &self,
            _point: &[Self::Scalar],
            vector: &TangentVector<Self::Scalar>,
        ) -> Result<TangentVector<Self::Scalar>> {
            Ok(vector.clone())
        }

        fn exp(
            &self,
            point: &[Self::Scalar],
            tangent: &TangentVector<Self::Scalar>,
        ) -> Result<Vec<Self::Scalar>> {
            self.check_tangent_vector(point, tangent)?;
            let result: Vec<f64> = (0..self.dimension)
                .map(|i| point[i] + tangent.components[i])
                .collect();
            Ok(result)
        }

        fn log(
            &self,
            point: &[Self::Scalar],
            other: &[Self::Scalar],
        ) -> Result<TangentVector<Self::Scalar>> {
            self.check_point(point)?;
            self.check_point(other)?;
            let components: Vec<f64> = (0..self.dimension).map(|i| other[i] - point[i]).collect();
            Ok(TangentVector::new(DVector::from_vec(components)))
        }

        fn inner_product(
            &self,
            _point: &[Self::Scalar],
            v1: &TangentVector<Self::Scalar>,
            v2: &TangentVector<Self::Scalar>,
        ) -> Result<Self::Scalar> {
            Ok(v1.components.dot(&v2.components))
        }
    }

    #[test]
    fn test_manifold_dimension() {
        let manifold = TestEuclideanManifold { dimension: 3 };
        assert_eq!(manifold.dim(), 3);
    }

    #[test]
    fn test_check_point() {
        let manifold = TestEuclideanManifold { dimension: 3 };
        assert!(manifold.check_point(&[1.0, 2.0, 3.0]).is_ok());
        assert!(manifold.check_point(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_exp_log_inverse() {
        let manifold = TestEuclideanManifold { dimension: 2 };
        let point = vec![1.0, 2.0];
        let tangent = TangentVector::new(DVector::from_vec(vec![0.5, 0.5]));

        let new_point = manifold.exp(&point, &tangent).unwrap();
        let recovered_tangent = manifold.log(&point, &new_point).unwrap();

        use approx::assert_relative_eq;
        assert_relative_eq!(
            tangent.components[0],
            recovered_tangent.components[0],
            epsilon = 1e-10
        );
        assert_relative_eq!(
            tangent.components[1],
            recovered_tangent.components[1],
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_distance() {
        let manifold = TestEuclideanManifold { dimension: 2 };
        let point1 = vec![0.0, 0.0];
        let point2 = vec![3.0, 4.0];

        let dist = manifold.distance(&point1, &point2).unwrap();
        use approx::assert_relative_eq;
        assert_relative_eq!(dist, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_geodesic() {
        let manifold = TestEuclideanManifold { dimension: 2 };
        let point = vec![0.0, 0.0];
        let tangent = TangentVector::new(DVector::from_vec(vec![1.0, 0.0]));

        let mid_point = manifold.geodesic(&point, &tangent, 0.5).unwrap();
        use approx::assert_relative_eq;
        assert_relative_eq!(mid_point[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(mid_point[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inner_product() {
        let manifold = TestEuclideanManifold { dimension: 2 };
        let point = vec![0.0, 0.0];
        let v1 = TangentVector::new(DVector::from_vec(vec![1.0, 0.0]));
        let v2 = TangentVector::new(DVector::from_vec(vec![0.0, 1.0]));

        let inner = manifold.inner_product(&point, &v1, &v2).unwrap();
        use approx::assert_relative_eq;
        assert_relative_eq!(inner, 0.0, epsilon = 1e-10);
    }
}
