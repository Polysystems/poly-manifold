use poly_manifold_autodiff::{GradientDescent, RiemannianOptimizer};
use poly_manifold_core::Manifold;
use poly_manifold_spaces::Sphere;

fn main() {
    println!("Sphere Optimization Example");
    println!("============================\n");

    let sphere = Sphere::new(2);

    println!("Finding point on S^2 closest to target (0.5, 0.5, 0.5)");

    let initial_point = vec![1.0, 0.0, 0.0];
    let target = vec![0.5, 0.5, 0.5];

    let cost = |p: &[f64]| {
        (p[0] - target[0]).powi(2) + (p[1] - target[1]).powi(2) + (p[2] - target[2]).powi(2)
    };

    println!(
        "Initial point: ({:.3}, {:.3}, {:.3})",
        initial_point[0], initial_point[1], initial_point[2]
    );
    println!("Initial cost: {:.6}", cost(&initial_point));

    let optimizer = GradientDescent::new(0.1, 100, 1e-8);
    let result = optimizer.minimize(&sphere, &initial_point, cost).unwrap();

    println!(
        "\nOptimized point: ({:.3}, {:.3}, {:.3})",
        result[0], result[1], result[2]
    );
    println!("Final cost: {:.6}", cost(&result));

    let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
    println!("Point norm (should be 1.0): {:.6}", norm);

    let dist = sphere.distance(&initial_point, &result).unwrap();
    println!("Geodesic distance traveled: {:.6}", dist);
}
