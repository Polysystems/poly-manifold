use thiserror::Error;

#[derive(Error, Debug)]
pub enum ManifoldError {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Point not on manifold: {reason}")]
    PointNotOnManifold { reason: String },

    #[error("Tangent vector not in tangent space: {reason}")]
    InvalidTangentVector { reason: String },

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceError { iterations: usize },

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Linear algebra error: {0}")]
    LinearAlgebraError(String),
}

pub type Result<T> = std::result::Result<T, ManifoldError>;
