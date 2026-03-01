//! Cholesky rank-1 update and downdate.
//!
//! Given lower-triangular `L` where `P = L·Lᵀ`, compute in-place `L'`
//! such that `P' = L'·L'ᵀ = L·Lᵀ + sign·v·vᵀ`.

use crate::matrix::vector::ColumnVector;
use crate::traits::FloatScalar;
use crate::Matrix;

use super::EstimateError;

/// In-place Cholesky rank-1 update (`sign = +1`) or downdate (`sign = -1`).
///
/// `l` must be lower-triangular on entry (upper triangle is ignored/untouched).
/// `v` is used as workspace and is modified in place.
///
/// Algorithm (direct formulation, cf. LINPACK `dchud`/`dchdd`):
///
/// For j = 0..N:
///   1. `r = sqrt(L[j,j]² + sign·v[j]²)`  (fails for downdate if arg < 0)
///   2. `c = r / L[j,j]`, `s = v[j] / L[j,j]`
///   3. `L[j,j] = r`
///   4. For i = j+1..N:
///        `L[i,j] = (L[i,j] + sign·s·v[i]) / c`
///        `v[i]   = c·v[i] - s·L[i,j]_new`
///
/// O(N²), no allocation.
#[allow(dead_code)]
pub(crate) fn cholupdate<T: FloatScalar, const N: usize>(
    l: &mut Matrix<T, N, N>,
    v: &mut ColumnVector<T, N>,
    sign: T,
) -> Result<(), EstimateError> {
    for j in 0..N {
        let ljj = l[(j, j)];
        let vj = v[(j, 0)];
        let arg = ljj * ljj + sign * vj * vj;

        if arg <= T::zero() {
            return Err(EstimateError::CholdowndateFailed);
        }

        let r = arg.sqrt();
        let c = r / ljj;
        let s = vj / ljj;
        l[(j, j)] = r;

        for i in (j + 1)..N {
            l[(i, j)] = (l[(i, j)] + sign * s * v[(i, 0)]) / c;
            v[(i, 0)] = c * v[(i, 0)] - s * l[(i, j)];
        }
    }

    Ok(())
}
