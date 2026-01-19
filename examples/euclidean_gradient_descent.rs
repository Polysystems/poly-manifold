use poly_manifold_autodiff::{GradientDescent, RiemannianOptimizer};
use poly_manifold_core::Manifold;
use poly_manifold_spaces::Euclidean;

fn main() {
    println!("Euclidean Gradient Descent Example");
    println!("===================================\n");

    let euclidean = Euclidean::new(2);

    // Example 1: Minimize (x-1)^2 + (y-2)^2
    println!("Example 1: Quadratic function");
    println!("Minimize: f(x,y) = (x-1)^2 + (y-2)^2\n");

    let initial_point = vec![5.0, 5.0];
    let cost = |p: &[f64]| (p[0] - 1.0).powi(2) + (p[1] - 2.0).powi(2);

    println!(
        "Initial point: ({:.3}, {:.3})",
        initial_point[0], initial_point[1]
    );
    println!("Initial cost: {:.6}", cost(&initial_point));

    let optimizer = GradientDescent::new(0.1, 1000, 1e-6);
    let result = optimizer
        .minimize(&euclidean, &initial_point, cost)
        .unwrap();

    println!("\nOptimized point: ({:.3}, {:.3})", result[0], result[1]);
    println!("Final cost: {:.6}", cost(&result));
    println!("Expected minimum: (1.000, 2.000)\n");

    // Example 2: Rosenbrock function
    println!("Example 2: Rosenbrock function");
    println!("Minimize: f(x,y) = (1-x)^2 + 100(y-x^2)^2\n");

    let initial_point = vec![0.0, 0.0];
    let rosenbrock = |p: &[f64]| (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0].powi(2)).powi(2);

    println!(
        "Initial point: ({:.3}, {:.3})",
        initial_point[0], initial_point[1]
    );
    println!("Initial cost: {:.6}", rosenbrock(&initial_point));

    let optimizer = GradientDescent::new(0.001, 10000, 1e-6);
    let result = optimizer
        .minimize(&euclidean, &initial_point, rosenbrock)
        .unwrap();

    println!("\nOptimized point: ({:.3}, {:.3})", result[0], result[1]);
    println!("Final cost: {:.6}", rosenbrock(&result));
    println!("Expected minimum: (1.000, 1.000)");
}
