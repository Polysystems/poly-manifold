use poly_manifold_core::{Manifold, Result, TangentVector};
use nalgebra::DVector;

pub fn numerical_gradient<M, F>(
    manifold: &M,
    point: &[f64],
    cost_function: F,
    epsilon: f64,
) -> Result<TangentVector<f64>>
where
    M: Manifold<Scalar = f64>,
    F: Fn(&[f64]) -> f64,
{
    manifold.check_point(point)?;

    let n = point.len();
    let mut gradient = vec![0.0; n];

    let f0 = cost_function(point);

    for i in 0..n {
        let mut point_plus = point.to_vec();
        point_plus[i] += epsilon;

        let f_plus = cost_function(&point_plus);
        gradient[i] = (f_plus - f0) / epsilon;
    }

    let grad_tangent = TangentVector::new(DVector::from_vec(gradient));
    manifold.project_to_tangent_space(point, &grad_tangent)
}

pub fn riemannian_gradient<M, F>(
    manifold: &M,
    point: &[f64],
    euclidean_gradient: &TangentVector<f64>,
) -> Result<TangentVector<f64>>
where
    M: Manifold<Scalar = f64>,
    F: Fn(&[f64]) -> f64,
{
    manifold.project_to_tangent_space(point, euclidean_gradient)
}

pub fn finite_difference_gradient<M, F>(
    manifold: &M,
    point: &[f64],
    direction: &TangentVector<f64>,
    cost_function: F,
    epsilon: f64,
) -> Result<f64>
where
    M: Manifold<Scalar = f64>,
    F: Fn(&[f64]) -> f64,
{
    manifold.check_tangent_vector(point, direction)?;

    let scaled_direction = direction.clone() * epsilon;
    let point_plus = manifold.exp(point, &scaled_direction)?;

    let f0 = cost_function(point);
    let f_plus = cost_function(&point_plus);

    Ok((f_plus - f0) / epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use poly_manifold_spaces::Euclidean;

    #[test]
    fn test_numerical_gradient() {
        let euclidean = Euclidean::new(2);
        let point = vec![1.0, 2.0];

        let cost = |p: &[f64]| p[0] * p[0] + p[1] * p[1];

        let grad = numerical_gradient(&euclidean, &point, cost, 1e-7).unwrap();

        assert_relative_eq!(grad.components[0], 2.0, epsilon = 1e-5);
        assert_relative_eq!(grad.components[1], 4.0, epsilon = 1e-5);
    }

    #[test]
    fn test_finite_difference_gradient() {
        let euclidean = Euclidean::new(2);
        let point = vec![1.0, 2.0];
        let direction = TangentVector::new(DVector::from_vec(vec![1.0, 0.0]));

        let cost = |p: &[f64]| p[0] * p[0] + p[1] * p[1];

        let directional_derivative =
            finite_difference_gradient(&euclidean, &point, &direction, cost, 1e-7).unwrap();

        assert_relative_eq!(directional_derivative, 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_riemannian_gradient_euclidean() {
        let euclidean = Euclidean::new(3);
        let point = vec![0.0, 0.0, 0.0];
        let euclidean_grad = TangentVector::new(DVector::from_vec(vec![1.0, 2.0, 3.0]));

        let riemannian_grad: TangentVector<f64> =
            riemannian_gradient::<Euclidean, fn(&[f64]) -> f64>(
                &euclidean,
                &point,
                &euclidean_grad,
            )
            .unwrap();

        assert_relative_eq!(riemannian_grad.components[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(riemannian_grad.components[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(riemannian_grad.components[2], 3.0, epsilon = 1e-10);
    }
}
