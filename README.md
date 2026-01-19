# Manifold - Differentiable Manifolds for Geometric ML

A high-performance Rust library for differential geometry and optimization on Riemannian manifolds. Designed for geometric machine learning, physics simulations, and physics-informed neural networks.

## Features

- **Core Manifold Framework**: Abstract traits for working with smooth manifolds
- **Concrete Manifolds**: Pre-implemented common manifolds
  - Euclidean spaces (R^n)
  - Spheres (S^n)
  - Symmetric Positive Definite matrices (SPD)
- **Automatic Differentiation**: Dual numbers and numerical gradients
- **Riemannian Optimization**: Gradient descent on manifolds
- **Type-Safe**: Leverages Rust's type system for correctness

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
manifold-core = { path = "manifold/manifold-core" }
manifold-spaces = { path = "manifold/manifold-spaces" }
manifold-autodiff = { path = "manifold/manifold-autodiff" }
```

## Quick Start

### Working with Euclidean Space

```rust
use manifold_core::{Manifold, TangentVector};
use manifold_spaces::Euclidean;
use nalgebra::DVector;

let euclidean = Euclidean::new(3);
let point = vec![1.0, 2.0, 3.0];
let tangent = TangentVector::new(DVector::from_vec(vec![0.1, 0.2, 0.3]));

// Exponential map (geodesic flow)
let new_point = euclidean.exp(&point, &tangent).unwrap();

// Logarithmic map
let recovered_tangent = euclidean.log(&point, &new_point).unwrap();

// Distance computation
let dist = euclidean.distance(&point, &new_point).unwrap();
```

### Optimization on a Sphere

```rust
use manifold_core::Manifold;
use manifold_spaces::Sphere;
use manifold_autodiff::{GradientDescent, RiemannianOptimizer};

let sphere = Sphere::new(2); // S^2 embedded in R^3
let initial_point = vec![1.0, 0.0, 0.0];

// Minimize distance to target point
let target = vec![0.0, 1.0, 0.0];
let cost = |p: &[f64]| {
    let dx = p[0] - target[0];
    let dy = p[1] - target[1];
    let dz = p[2] - target[2];
    dx * dx + dy * dy + dz * dz
};

let optimizer = GradientDescent::new(0.1, 100, 1e-6);
let result = optimizer.minimize(&sphere, &initial_point, cost).unwrap();
```

### Automatic Differentiation

```rust
use manifold_autodiff::Dual;

let x = Dual::variable(2.0);
let y = (x * Dual::constant(3.0) + Dual::constant(1.0)).powi(2);

println!("f(2) = {}", y.value);        // 49.0
println!("f'(2) = {}", y.derivative);  // 42.0
```

### SPD Matrices

```rust
use manifold_core::{Manifold, TangentVector};
use manifold_spaces::SPD;
use nalgebra::DVector;

let spd = SPD::new(2);

// 2x2 identity matrix (flattened)
let identity = vec![1.0, 0.0, 0.0, 1.0];

// Tangent vector (symmetric matrix)
let tangent = TangentVector::new(DVector::from_vec(vec![0.1, 0.0, 0.0, 0.1]));

// Geodesic on SPD manifold
let new_point = spd.exp(&identity, &tangent).unwrap();
```

## Architecture

The library is organized into three main crates:

### manifold-core

Defines core traits and types:
- `Manifold` trait: Core operations (exp, log, distance, geodesic)
- `RiemannianMetric`: Metric tensor and inner products
- `TangentVector`: Tangent space vectors
- Error types and result handling

### manifold-spaces

Concrete manifold implementations:
- `Euclidean`: Standard Euclidean space R^n
- `Sphere`: n-sphere S^n embedded in R^(n+1)
- `SPD`: Symmetric positive definite matrices with affine-invariant metric

### manifold-autodiff

Automatic differentiation and optimization:
- `Dual`: Dual numbers for forward-mode AD
- Numerical gradient computation
- Riemannian gradient descent
- Optimization on manifolds

## Mathematical Background

### Manifolds

A manifold is a topological space that locally resembles Euclidean space. This library focuses on Riemannian manifolds, which have a metric tensor defining distances and angles.

Key operations:
- **Exponential map** (`exp`): Maps tangent vectors to points on the manifold
- **Logarithmic map** (`log`): Inverse of exp, maps points to tangent vectors
- **Distance**: Geodesic distance between points
- **Geodesic**: Shortest path between points

### Riemannian Optimization

Optimization on manifolds extends classical optimization to curved spaces:

1. Compute Euclidean gradient
2. Project to tangent space (Riemannian gradient)
3. Move along geodesic using exponential map
4. Repeat until convergence

## Examples

See the `examples/` directory for complete examples:
- `euclidean_optimization.rs`: Basic optimization in R^n
- `sphere_fitting.rs`: Fitting points on a sphere
- `spd_interpolation.rs`: Geodesic interpolation on SPD manifolds

## Testing

Run all tests:

```bash
cargo test --workspace
```

Run tests for a specific crate:

```bash
cd manifold-core && cargo test
cd manifold-spaces && cargo test
cd manifold-autodiff && cargo test
```

## Performance

The library is designed for performance:
- Zero-cost abstractions using Rust's trait system
- Uses `nalgebra` for efficient linear algebra
- Release builds with optimizations: `cargo build --release`

## Applications

- **Geometric Machine Learning**: Neural networks on manifolds
- **Computer Vision**: Rotation estimation, pose estimation
- **Robotics**: Motion planning on configuration manifolds
- **Signal Processing**: Covariance matrix processing
- **Physics Simulations**: Constrained mechanical systems

## Contributing

Contributions are welcome! Areas for improvement:
- Additional manifolds (Grassmannian, Stiefel, etc.)
- GPU acceleration
- More optimization algorithms
- Parallel transport
- Curvature computations

## License

MIT License

## References

1. Absil, P. A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.
2. Lee, J. M. (2018). *Introduction to Riemannian Manifolds*. Springer.
3. Pennec, X., Fillard, P., & Ayache, N. (2006). A Riemannian framework for tensor computing. *International Journal of Computer Vision*.
