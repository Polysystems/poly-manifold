use poly_manifold_core::{Manifold, ManifoldError, Result, TangentVector};
use nalgebra::DVector;

pub struct Euclidean {
    pub dimension: usize,
}

impl Euclidean {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Manifold for Euclidean {
    type Scalar = f64;

    fn dim(&self) -> usize {
        self.dimension
    }

    fn check_point(&self, point: &[Self::Scalar]) -> Result<()> {
        if point.len() != self.dimension {
            return Err(ManifoldError::DimensionMismatch {
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
            return Err(ManifoldError::DimensionMismatch {
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
        let mut components = vec![0.0; self.dimension];
        for i in 0..self.dimension {
            components[i] = other[i] - point[i];
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_euclidean_dimension() {
        let euclidean = Euclidean::new(3);
        assert_eq!(euclidean.dim(), 3);
    }

    #[test]
    fn test_euclidean_exp_log() {
        let euclidean = Euclidean::new(2);
        let point = vec![1.0, 2.0];
        let tangent = TangentVector::new(DVector::from_vec(vec![0.5, 0.5]));

        let new_point = euclidean.exp(&point, &tangent).unwrap();
        assert_relative_eq!(new_point[0], 1.5);
        assert_relative_eq!(new_point[1], 2.5);

        let recovered_tangent = euclidean.log(&point, &new_point).unwrap();
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
    fn test_euclidean_distance() {
        let euclidean = Euclidean::new(2);
        let point1 = vec![0.0, 0.0];
        let point2 = vec![3.0, 4.0];

        let dist = euclidean.distance(&point1, &point2).unwrap();
        assert_relative_eq!(dist, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_euclidean_inner_product() {
        let euclidean = Euclidean::new(3);
        let point = vec![0.0, 0.0, 0.0];
        let v1 = TangentVector::new(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let v2 = TangentVector::new(DVector::from_vec(vec![4.0, 5.0, 6.0]));

        let inner = euclidean.inner_product(&point, &v1, &v2).unwrap();
        assert_relative_eq!(inner, 32.0, epsilon = 1e-10);
    }
}
