use nalgebra::{DVector, RealField};

pub struct TangentVector<T: RealField> {
    pub components: DVector<T>,
}

impl<T: RealField> TangentVector<T> {
    pub fn new(components: DVector<T>) -> Self {
        Self { components }
    }

    pub fn dim(&self) -> usize {
        self.components.len()
    }

    pub fn zero(dim: usize) -> Self {
        Self {
            components: DVector::zeros(dim),
        }
    }

    pub fn norm_squared(&self) -> T {
        self.components.dot(&self.components)
    }

    pub fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }
}

impl<T: RealField> Clone for TangentVector<T> {
    fn clone(&self) -> Self {
        Self {
            components: self.components.clone(),
        }
    }
}

impl<T: RealField> std::ops::Add for TangentVector<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            components: self.components + other.components,
        }
    }
}

impl<T: RealField> std::ops::Sub for TangentVector<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            components: self.components - other.components,
        }
    }
}

impl<T: RealField> std::ops::Mul<T> for TangentVector<T> {
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self {
            components: self.components * scalar,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tangent_vector_creation() {
        let v = TangentVector::new(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        assert_eq!(v.dim(), 3);
    }

    #[test]
    fn test_tangent_vector_zero() {
        let v = TangentVector::<f64>::zero(5);
        assert_eq!(v.dim(), 5);
        assert_relative_eq!(v.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tangent_vector_norm() {
        let v = TangentVector::new(DVector::from_vec(vec![3.0, 4.0]));
        assert_relative_eq!(v.norm(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tangent_vector_addition() {
        let v1 = TangentVector::new(DVector::from_vec(vec![1.0, 2.0]));
        let v2 = TangentVector::new(DVector::from_vec(vec![3.0, 4.0]));
        let v3 = v1 + v2;
        assert_eq!(v3.components[0], 4.0);
        assert_eq!(v3.components[1], 6.0);
    }

    #[test]
    fn test_tangent_vector_scalar_multiplication() {
        let v = TangentVector::new(DVector::from_vec(vec![1.0, 2.0]));
        let v2 = v * 2.0;
        assert_eq!(v2.components[0], 2.0);
        assert_eq!(v2.components[1], 4.0);
    }
}
