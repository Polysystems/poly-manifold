use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone, Copy)]
pub struct Dual {
    pub value: f64,
    pub derivative: f64,
}

impl Dual {
    pub fn constant(value: f64) -> Self {
        Self {
            value,
            derivative: 0.0,
        }
    }

    pub fn variable(value: f64) -> Self {
        Self {
            value,
            derivative: 1.0,
        }
    }

    pub fn sin(self) -> Self {
        Self {
            value: self.value.sin(),
            derivative: self.derivative * self.value.cos(),
        }
    }

    pub fn cos(self) -> Self {
        Self {
            value: self.value.cos(),
            derivative: -self.derivative * self.value.sin(),
        }
    }

    pub fn exp(self) -> Self {
        let exp_val = self.value.exp();
        Self {
            value: exp_val,
            derivative: self.derivative * exp_val,
        }
    }

    pub fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            derivative: self.derivative / self.value,
        }
    }

    pub fn sqrt(self) -> Self {
        let sqrt_val = self.value.sqrt();
        Self {
            value: sqrt_val,
            derivative: self.derivative / (2.0 * sqrt_val),
        }
    }

    pub fn powi(self, n: i32) -> Self {
        Self {
            value: self.value.powi(n),
            derivative: self.derivative * (n as f64) * self.value.powi(n - 1),
        }
    }

    pub fn powf(self, n: f64) -> Self {
        Self {
            value: self.value.powf(n),
            derivative: self.derivative * n * self.value.powf(n - 1.0),
        }
    }
}

impl Add for Dual {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            derivative: self.derivative + other.derivative,
        }
    }
}

impl Sub for Dual {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            derivative: self.derivative - other.derivative,
        }
    }
}

impl Mul for Dual {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            derivative: self.derivative * other.value + self.value * other.derivative,
        }
    }
}

impl Add<f64> for Dual {
    type Output = Self;

    fn add(self, other: f64) -> Self {
        Self {
            value: self.value + other,
            derivative: self.derivative,
        }
    }
}

impl Mul<f64> for Dual {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        Self {
            value: self.value * other,
            derivative: self.derivative * other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dual_constant() {
        let x = Dual::constant(5.0);
        assert_eq!(x.value, 5.0);
        assert_eq!(x.derivative, 0.0);
    }

    #[test]
    fn test_dual_variable() {
        let x = Dual::variable(3.0);
        assert_eq!(x.value, 3.0);
        assert_eq!(x.derivative, 1.0);
    }

    #[test]
    fn test_dual_addition() {
        let x = Dual::variable(2.0);
        let y = Dual::constant(3.0);
        let z = x + y;
        assert_eq!(z.value, 5.0);
        assert_eq!(z.derivative, 1.0);
    }

    #[test]
    fn test_dual_multiplication() {
        let x = Dual::variable(3.0);
        let y = Dual::constant(4.0);
        let z = x * y;
        assert_eq!(z.value, 12.0);
        assert_eq!(z.derivative, 4.0);
    }

    #[test]
    fn test_dual_power() {
        let x = Dual::variable(2.0);
        let y = x.powi(3);
        assert_eq!(y.value, 8.0);
        assert_eq!(y.derivative, 12.0);
    }

    #[test]
    fn test_dual_sin() {
        let x = Dual::variable(0.0);
        let y = x.sin();
        assert_relative_eq!(y.value, 0.0, epsilon = 1e-10);
        assert_relative_eq!(y.derivative, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dual_composite() {
        let x = Dual::variable(1.0);
        let y = (x * Dual::constant(2.0) + Dual::constant(1.0)).powi(2);
        assert_eq!(y.value, 9.0);
        assert_eq!(y.derivative, 12.0);
    }
}
