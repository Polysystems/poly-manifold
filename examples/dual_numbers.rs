use poly_manifold_autodiff::Dual;

fn main() {
    println!("Automatic Differentiation with Dual Numbers");
    println!("============================================\n");

    // Example 1: f(x) = x^2
    println!("Example 1: f(x) = x^2");
    let x = Dual::variable(3.0);
    let f1 = x.powi(2);
    println!("f(3) = {}", f1.value);
    println!("f'(3) = {} (expected: 6)\n", f1.derivative);

    // Example 2: f(x) = sin(x)
    println!("Example 2: f(x) = sin(x)");
    let x = Dual::variable(0.0);
    let f2 = x.sin();
    println!("f(0) = {}", f2.value);
    println!("f'(0) = {} (expected: 1)\n", f2.derivative);

    // Example 3: f(x) = (2x + 1)^2
    println!("Example 3: f(x) = (2x + 1)^2");
    let x = Dual::variable(1.0);
    let f3 = (x * 2.0 + 1.0).powi(2);
    println!("f(1) = {}", f3.value);
    println!("f'(1) = {} (expected: 12)\n", f3.derivative);

    // Example 4: f(x) = e^(x^2)
    println!("Example 4: f(x) = e^(x^2)");
    let x = Dual::variable(2.0);
    let f4 = x.powi(2).exp();
    println!("f(2) = {}", f4.value);
    println!(
        "f'(2) = {} (expected: 4 * e^4 = {})\n",
        f4.derivative,
        4.0 * (4.0_f64).exp()
    );

    // Example 5: f(x) = sqrt(x)
    println!("Example 5: f(x) = sqrt(x)");
    let x = Dual::variable(4.0);
    let f5 = x.sqrt();
    println!("f(4) = {}", f5.value);
    println!("f'(4) = {} (expected: 0.25)\n", f5.derivative);

    // Example 6: Chain rule f(x) = ln(sin(x))
    println!("Example 6: f(x) = ln(sin(x))");
    let x = Dual::variable(std::f64::consts::PI / 2.0);
    let f6 = x.sin().ln();
    println!("f(π/2) = {}", f6.value);
    println!("f'(π/2) = {} (expected: 0)\n", f6.derivative);
}
