//! Cholesky rank-1 update and downdate.
//!
//! Delegates to the generic implementations in [`crate::linalg`].

use crate::linalg::{cholesky_rank1_update, cholesky_rank1_downdate};
use crate::matrix::vector::ColumnVector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::EstimateError;

/// In-place Cholesky rank-1 update (`sign = +1`) or downdate (`sign = -1`).
///
/// `l` must be lower-triangular on entry (upper triangle is ignored/untouched).
/// `v` is used as workspace and is modified in place.
///
/// O(N²), no allocation.
#[allow(dead_code)]
pub(crate) fn cholupdate<T: FloatScalar, const N: usize>(
    l: &mut Matrix<T, N, N>,
    v: &mut ColumnVector<T, N>,
    sign: T,
) -> Result<(), EstimateError> {
    let result = if sign >= T::zero() {
        cholesky_rank1_update(l, v.as_mut_slice())
    } else {
        cholesky_rank1_downdate(l, v.as_mut_slice())
    };
    result.map_err(|_| EstimateError::CholdowndateFailed)
}
