use poly_manifold_core::{Manifold, ManifoldError, Result, TangentVector};
use nalgebra::DVector;

pub struct Sphere {
    pub dimension: usize,
}

impl Sphere {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn embedding_dim(&self) -> usize {
        self.dimension + 1
    }
}

impl Manifold for Sphere {
    type Scalar = f64;

    fn dim(&self) -> usize {
        self.dimension
    }

    fn check_point(&self, point: &[Self::Scalar]) -> Result<()> {
        if point.len() != self.embedding_dim() {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.embedding_dim(),
                got: point.len(),
            });
        }

        let norm_sq: f64 = point.iter().map(|x| x * x).sum();
        if (norm_sq - 1.0).abs() > 1e-10 {
            return Err(ManifoldError::PointNotOnManifold {
                reason: format!("Point norm is {} instead of 1.0", norm_sq.sqrt()),
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
        if tangent.dim() != self.embedding_dim() {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.embedding_dim(),
                got: tangent.dim(),
            });
        }

        let dot_product: f64 = point
            .iter()
            .zip(tangent.components.iter())
            .map(|(p, t)| p * t)
            .sum();

        if dot_product.abs() > 1e-10 {
            return Err(ManifoldError::InvalidTangentVector {
                reason: format!(
                    "Tangent vector not orthogonal to point, dot product: {}",
                    dot_product
                ),
            });
        }

        Ok(())
    }

    fn project_to_manifold(&self, point: &[Self::Scalar]) -> Result<Vec<Self::Scalar>> {
        let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Err(ManifoldError::NumericalError(
                "Cannot project zero vector to sphere".to_string(),
            ));
        }
        Ok(point.iter().map(|x| x / norm).collect())
    }

    fn project_to_tangent_space(
        &self,
        point: &[Self::Scalar],
        vector: &TangentVector<Self::Scalar>,
    ) -> Result<TangentVector<Self::Scalar>> {
        self.check_point(point)?;

        let dot_product: f64 = point
            .iter()
            .zip(vector.components.iter())
            .map(|(p, v)| p * v)
            .sum();

        let mut projected = vector.components.clone();
        for i in 0..self.embedding_dim() {
            projected[i] -= dot_product * point[i];
        }

        Ok(TangentVector::new(projected))
    }

    fn exp(
        &self,
        point: &[Self::Scalar],
        tangent: &TangentVector<Self::Scalar>,
    ) -> Result<Vec<Self::Scalar>> {
        self.check_tangent_vector(point, tangent)?;

        let tangent_norm = tangent.norm();

        if tangent_norm < 1e-10 {
            return Ok(point.to_vec());
        }

        let mut result = vec![0.0; self.embedding_dim()];
        for i in 0..self.embedding_dim() {
            result[i] = point[i] * tangent_norm.cos()
                + tangent.components[i] * tangent_norm.sin() / tangent_norm;
        }

        Ok(result)
    }

    fn log(
        &self,
        point: &[Self::Scalar],
        other: &[Self::Scalar],
    ) -> Result<TangentVector<Self::Scalar>> {
        self.check_point(point)?;
        self.check_point(other)?;

        let dot_product: f64 = point.iter().zip(other.iter()).map(|(p, o)| p * o).sum();
        let dot_product = dot_product.clamp(-1.0, 1.0);

        let theta = dot_product.acos();

        if theta.abs() < 1e-10 {
            return Ok(TangentVector::new(DVector::zeros(self.embedding_dim())));
        }

        let sin_theta = theta.sin();
        if sin_theta.abs() < 1e-10 {
            return Err(ManifoldError::NumericalError(
                "Points are antipodal, logarithm map is not unique".to_string(),
            ));
        }

        let components: Vec<f64> = (0..self.embedding_dim())
            .map(|i| (other[i] - point[i] * dot_product) * theta / sin_theta)
            .collect();

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
    use std::f64::consts::PI;

    #[test]
    fn test_sphere_dimension() {
        let sphere = Sphere::new(2);
        assert_eq!(sphere.dim(), 2);
    }

    #[test]
    fn test_sphere_check_point() {
        let sphere = Sphere::new(2);
        assert!(sphere.check_point(&[1.0, 0.0, 0.0]).is_ok());
        assert!(sphere.check_point(&[0.0, 1.0, 0.0]).is_ok());
        assert!(sphere.check_point(&[0.5, 0.5, 0.5]).is_err());
    }

    #[test]
    fn test_sphere_project_to_manifold() {
        let sphere = Sphere::new(2);
        let point = vec![2.0, 0.0, 0.0];
        let projected = sphere.project_to_manifold(&point).unwrap();
        assert_relative_eq!(projected[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(projected[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(projected[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sphere_exp_log() {
        let sphere = Sphere::new(2);
        let point = vec![1.0, 0.0, 0.0];
        let tangent = TangentVector::new(DVector::from_vec(vec![0.0, 0.5, 0.0]));

        let new_point = sphere.exp(&point, &tangent).unwrap();
        assert!(sphere.check_point(&new_point).is_ok());

        let recovered_tangent = sphere.log(&point, &new_point).unwrap();
        assert_relative_eq!(
            tangent.components[0],
            recovered_tangent.components[0],
            epsilon = 1e-9
        );
        assert_relative_eq!(
            tangent.components[1],
            recovered_tangent.components[1],
            epsilon = 1e-9
        );
        assert_relative_eq!(
            tangent.components[2],
            recovered_tangent.components[2],
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_sphere_distance() {
        let sphere = Sphere::new(2);
        let point1 = vec![1.0, 0.0, 0.0];
        let point2 = vec![0.0, 1.0, 0.0];

        let dist = sphere.distance(&point1, &point2).unwrap();
        assert_relative_eq!(dist, PI / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_sphere_geodesic() {
        let sphere = Sphere::new(2);
        let point = vec![1.0, 0.0, 0.0];
        let tangent = TangentVector::new(DVector::from_vec(vec![0.0, PI / 2.0, 0.0]));

        let mid_point = sphere.geodesic(&point, &tangent, 0.5).unwrap();
        assert!(sphere.check_point(&mid_point).is_ok());

        let cos45 = (PI / 4.0).cos();
        let sin45 = (PI / 4.0).sin();
        assert_relative_eq!(mid_point[0], cos45, epsilon = 1e-9);
        assert_relative_eq!(mid_point[1], sin45, epsilon = 1e-9);
        assert_relative_eq!(mid_point[2], 0.0, epsilon = 1e-9);
    }
}
