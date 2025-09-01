use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum DynMatrixError {
    #[error("Dimension mismatch")]
    DimensionMismatch,
    #[error("Matrix is singular")]
    SingularMatrix,
    #[error("Index out of bounds")]
    IndexOutOfBounds,
    #[error("Invalid matrix shape")]
    InvalidShape,
    #[error("Matrix must be square")]
    MustBeSquare,
}

pub type DynMatrixResult<T> = Result<T, DynMatrixError>;
