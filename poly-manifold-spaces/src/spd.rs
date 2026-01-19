use poly_manifold_core::{Manifold, ManifoldError, Result, TangentVector};
use nalgebra::{DMatrix, DVector};

pub struct SPD {
    pub dimension: usize,
}

impl SPD {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn vec_to_matrix(&self, vec: &[f64]) -> DMatrix<f64> {
        assert_eq!(vec.len(), self.dimension * self.dimension);
        DMatrix::from_row_slice(self.dimension, self.dimension, vec)
    }

    fn matrix_to_vec(&self, mat: &DMatrix<f64>) -> Vec<f64> {
        mat.as_slice().to_vec()
    }

    fn is_symmetric(&self, mat: &DMatrix<f64>) -> bool {
        for i in 0..self.dimension {
            for j in (i + 1)..self.dimension {
                if (mat[(i, j)] - mat[(j, i)]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    fn is_positive_definite(&self, mat: &DMatrix<f64>) -> bool {
        mat.clone().cholesky().is_some()
    }
}

impl Manifold for SPD {
    type Scalar = f64;

    fn dim(&self) -> usize {
        self.dimension * (self.dimension + 1) / 2
    }

    fn check_point(&self, point: &[Self::Scalar]) -> Result<()> {
        if point.len() != self.dimension * self.dimension {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.dimension * self.dimension,
                got: point.len(),
            });
        }

        let mat = self.vec_to_matrix(point);

        if !self.is_symmetric(&mat) {
            return Err(ManifoldError::PointNotOnManifold {
                reason: "Matrix is not symmetric".to_string(),
            });
        }

        if !self.is_positive_definite(&mat) {
            return Err(ManifoldError::PointNotOnManifold {
                reason: "Matrix is not positive definite".to_string(),
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
        if tangent.dim() != self.dimension * self.dimension {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.dimension * self.dimension,
                got: tangent.dim(),
            });
        }

        let tangent_mat = self.vec_to_matrix(tangent.components.as_slice());
        if !self.is_symmetric(&tangent_mat) {
            return Err(ManifoldError::InvalidTangentVector {
                reason: "Tangent vector matrix is not symmetric".to_string(),
            });
        }

        Ok(())
    }

    fn project_to_manifold(&self, point: &[Self::Scalar]) -> Result<Vec<Self::Scalar>> {
        let mut mat = self.vec_to_matrix(point);

        mat = (mat.clone() + mat.transpose()) * 0.5;

        let epsilon = 1e-10;
        for i in 0..self.dimension {
            mat[(i, i)] = mat[(i, i)].max(epsilon);
        }

        Ok(self.matrix_to_vec(&mat))
    }

    fn project_to_tangent_space(
        &self,
        _point: &[Self::Scalar],
        vector: &TangentVector<Self::Scalar>,
    ) -> Result<TangentVector<Self::Scalar>> {
        let mat = self.vec_to_matrix(vector.components.as_slice());
        let symmetric = (mat.clone() + mat.transpose()) * 0.5;
        let projected_vec = DVector::from_vec(self.matrix_to_vec(&symmetric));
        Ok(TangentVector::new(projected_vec))
    }

    fn exp(
        &self,
        point: &[Self::Scalar],
        tangent: &TangentVector<Self::Scalar>,
    ) -> Result<Vec<Self::Scalar>> {
        self.check_tangent_vector(point, tangent)?;

        let p_mat = self.vec_to_matrix(point);
        let v_mat = self.vec_to_matrix(tangent.components.as_slice());

        let chol = p_mat.cholesky().ok_or_else(|| {
            ManifoldError::LinearAlgebraError("Cholesky decomposition failed".to_string())
        })?;
        let l = chol.l();

        let l_inv = l.clone().try_inverse().ok_or_else(|| {
            ManifoldError::LinearAlgebraError("Matrix inversion failed".to_string())
        })?;

        let w = &l_inv * &v_mat * l_inv.transpose();

        let w_exp = matrix_exponential(&w);

        let result = &l * w_exp * l.transpose();

        Ok(self.matrix_to_vec(&result))
    }

    fn log(
        &self,
        point: &[Self::Scalar],
        other: &[Self::Scalar],
    ) -> Result<TangentVector<Self::Scalar>> {
        self.check_point(point)?;
        self.check_point(other)?;

        let p_mat = self.vec_to_matrix(point);
        let q_mat = self.vec_to_matrix(other);

        let chol_p = p_mat.cholesky().ok_or_else(|| {
            ManifoldError::LinearAlgebraError("Cholesky decomposition failed for point".to_string())
        })?;
        let l_p = chol_p.l();

        let l_p_inv = l_p.clone().try_inverse().ok_or_else(|| {
            ManifoldError::LinearAlgebraError("Matrix inversion failed".to_string())
        })?;

        let w = &l_p_inv * &q_mat * l_p_inv.transpose();

        let w_log = matrix_logarithm(&w)?;

        let v = &l_p * w_log * l_p.transpose();

        Ok(TangentVector::new(DVector::from_vec(
            self.matrix_to_vec(&v),
        )))
    }

    fn inner_product(
        &self,
        point: &[Self::Scalar],
        v1: &TangentVector<Self::Scalar>,
        v2: &TangentVector<Self::Scalar>,
    ) -> Result<Self::Scalar> {
        let p_mat = self.vec_to_matrix(point);
        let p_inv = p_mat.clone().try_inverse().ok_or_else(|| {
            ManifoldError::LinearAlgebraError("Matrix inversion failed".to_string())
        })?;

        let v1_mat = self.vec_to_matrix(v1.components.as_slice());
        let v2_mat = self.vec_to_matrix(v2.components.as_slice());

        let tmp = &p_inv * &v1_mat * &p_inv * v2_mat;
        Ok(tmp.trace())
    }
}

fn matrix_exponential(mat: &DMatrix<f64>) -> DMatrix<f64> {
    let n = mat.nrows();
    let mut result = DMatrix::identity(n, n);
    let mut term = DMatrix::identity(n, n);

    for k in 1..20 {
        term = &term * mat / (k as f64);
        result += &term;

        if term.iter().all(|&x| x.abs() < 1e-12) {
            break;
        }
    }

    result
}

fn matrix_logarithm(mat: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let n = mat.nrows();
    let identity = DMatrix::identity(n, n);

    let a = mat - &identity;

    let mut result = DMatrix::zeros(n, n);
    let mut term = a.clone();

    for k in 1..50 {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        result += &term * (sign / k as f64);

        term = &term * &a;

        if term.iter().all(|&x| x.abs() < 1e-12) {
            break;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_spd_dimension() {
        let spd = SPD::new(3);
        assert_eq!(spd.dim(), 6);
    }

    #[test]
    fn test_spd_check_point() {
        let spd = SPD::new(2);
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        assert!(spd.check_point(&identity).is_ok());

        let not_symmetric = vec![1.0, 0.5, 0.3, 1.0];
        assert!(spd.check_point(&not_symmetric).is_err());
    }

    #[test]
    fn test_spd_project_to_manifold() {
        let spd = SPD::new(2);
        let not_symmetric = vec![1.0, 0.5, 0.3, 2.0];
        let projected = spd.project_to_manifold(&not_symmetric).unwrap();
        let projected_mat = spd.vec_to_matrix(&projected);
        assert!(spd.is_symmetric(&projected_mat));
    }

    #[test]
    fn test_spd_exp_log() {
        let spd = SPD::new(2);
        let point = vec![2.0, 0.0, 0.0, 2.0];
        let tangent = TangentVector::new(DVector::from_vec(vec![0.1, 0.0, 0.0, 0.1]));

        let new_point = spd.exp(&point, &tangent).unwrap();
        assert!(spd.check_point(&new_point).is_ok());

        let recovered_tangent = spd.log(&point, &new_point).unwrap();
        for i in 0..4 {
            assert_relative_eq!(
                tangent.components[i],
                recovered_tangent.components[i],
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn test_spd_inner_product() {
        let spd = SPD::new(2);
        let point = vec![1.0, 0.0, 0.0, 1.0];
        let v1 = TangentVector::new(DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]));
        let v2 = TangentVector::new(DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0]));

        let inner = spd.inner_product(&point, &v1, &v2).unwrap();
        assert_relative_eq!(inner, 0.0, epsilon = 1e-10);
    }
}
