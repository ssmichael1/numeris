use thiserror::Error;

#[derive(Error, Debug)]
pub enum DynArrayError {
    #[error("Index out of bounds")]
    OutOfBounds,
    #[error("Dimension mismatch")]
    DimMismatch,
    #[error("Shape mismatch")]
    ShapeMismatch,
}

pub type DynArrayResult<T> = Result<T, DynArrayError>;
