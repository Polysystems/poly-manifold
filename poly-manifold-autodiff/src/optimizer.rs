use crate::gradient::numerical_gradient;
use poly_manifold_core::{Manifold, Result};

pub trait RiemannianOptimizer {
    fn minimize<M, F>(
        &self,
        manifold: &M,
        initial_point: &[f64],
        cost_function: F,
    ) -> Result<Vec<f64>>
    where
        M: Manifold<Scalar = f64>,
        F: Fn(&[f64]) -> f64;
}

pub struct GradientDescent {
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl GradientDescent {
    pub fn new(learning_rate: f64, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            learning_rate,
            max_iterations,
            tolerance,
        }
    }
}

impl RiemannianOptimizer for GradientDescent {
    fn minimize<M, F>(
        &self,
        manifold: &M,
        initial_point: &[f64],
        cost_function: F,
    ) -> Result<Vec<f64>>
    where
        M: Manifold<Scalar = f64>,
        F: Fn(&[f64]) -> f64,
    {
        manifold.check_point(initial_point)?;

        let mut point = initial_point.to_vec();
        let mut prev_cost = cost_function(&point);

        for _iter in 0..self.max_iterations {
            let gradient = numerical_gradient(manifold, &point, &cost_function, 1e-7)?;

            let descent_direction = gradient * (-self.learning_rate);

            point = manifold.exp(&point, &descent_direction)?;

            let current_cost = cost_function(&point);

            if (prev_cost - current_cost).abs() < self.tolerance {
                break;
            }

            prev_cost = current_cost;
        }

        Ok(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use poly_manifold_spaces::Euclidean;

    #[test]
    fn test_gradient_descent_quadratic() {
        let euclidean = Euclidean::new(2);
        let initial_point = vec![5.0, 5.0];

        let cost = |p: &[f64]| (p[0] - 1.0).powi(2) + (p[1] - 2.0).powi(2);

        let optimizer = GradientDescent::new(0.1, 1000, 1e-6);
        let result = optimizer
            .minimize(&euclidean, &initial_point, cost)
            .unwrap();

        assert_relative_eq!(result[0], 1.0, epsilon = 1e-2);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-2);
    }

    #[test]
    fn test_gradient_descent_rosenbrock() {
        let euclidean = Euclidean::new(2);
        let initial_point = vec![0.0, 0.0];

        let cost = |p: &[f64]| {
            let a = 1.0;
            let b = 100.0;
            (a - p[0]).powi(2) + b * (p[1] - p[0].powi(2)).powi(2)
        };

        let optimizer = GradientDescent::new(0.001, 10000, 1e-6);
        let result = optimizer
            .minimize(&euclidean, &initial_point, cost)
            .unwrap();

        assert_relative_eq!(result[0], 1.0, epsilon = 1e-1);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-1);
    }
}
