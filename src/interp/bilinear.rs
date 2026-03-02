use crate::traits::FloatScalar;

use super::{InterpError, find_interval, validate_sorted};

/// Bilinear interpolant on a rectangular grid (fixed-size, stack-allocated).
///
/// Interpolates z = f(x, y) on an NX × NY grid by linearly interpolating
/// first in x, then in y. Out-of-bounds queries extrapolate using the
/// nearest boundary cell.
///
/// Internal storage is column-major: `zs[ix][iy]` holds the value at
/// `(xs[ix], ys[iy])`. The constructor accepts row-major input (each inner
/// array is a row at fixed y) and transposes internally, matching the
/// `Matrix::new()` convention.
///
/// # Example
///
/// ```
/// use numeris::interp::BilinearInterp;
///
/// // 3×2 grid: z = x + y
/// let xs = [0.0_f64, 1.0, 2.0];
/// let ys = [0.0, 1.0];
/// // Row-major input: zs_input[iy][ix]
/// let zs = [
///     [0.0, 1.0, 2.0],  // y = 0
///     [1.0, 2.0, 3.0],  // y = 1
/// ];
/// let interp = BilinearInterp::new(xs, ys, zs).unwrap();
/// assert!((interp.eval(0.5, 0.5) - 1.0).abs() < 1e-14);
/// ```
#[derive(Debug, Clone)]
pub struct BilinearInterp<T, const NX: usize, const NY: usize> {
    xs: [T; NX],
    ys: [T; NY],
    zs: [[T; NY]; NX], // column-major: zs[ix][iy]
}

impl<T: FloatScalar, const NX: usize, const NY: usize> BilinearInterp<T, NX, NY> {
    /// Construct a bilinear interpolant from sorted grid knots.
    ///
    /// `zs_rows[j][i]` is the value at `(xs[i], ys[j])` — row-major input.
    /// Each inner array is a "row" of z-values at a fixed y-coordinate.
    /// Internally transposed to column-major storage.
    ///
    /// Returns `InterpError::TooFewPoints` if `NX < 2` or `NY < 2`,
    /// `InterpError::NotSorted` if `xs` or `ys` is not strictly increasing.
    pub fn new(xs: [T; NX], ys: [T; NY], zs_rows: [[T; NX]; NY]) -> Result<Self, InterpError> {
        if NX < 2 || NY < 2 {
            return Err(InterpError::TooFewPoints);
        }
        validate_sorted(&xs)?;
        validate_sorted(&ys)?;
        // Transpose row-major input to column-major storage
        let mut zs = [[T::zero(); NY]; NX];
        let mut ix = 0;
        while ix < NX {
            let mut iy = 0;
            while iy < NY {
                zs[ix][iy] = zs_rows[iy][ix];
                iy += 1;
            }
            ix += 1;
        }
        Ok(Self { xs, ys, zs })
    }

    /// Evaluate the interpolant at `(x, y)`.
    pub fn eval(&self, x: T, y: T) -> T {
        let ix = find_interval(&self.xs, x);
        let iy = find_interval(&self.ys, y);
        let tx = (x - self.xs[ix]) / (self.xs[ix + 1] - self.xs[ix]);
        let ty = (y - self.ys[iy]) / (self.ys[iy + 1] - self.ys[iy]);
        let one = T::one();
        let z00 = self.zs[ix][iy];
        let z10 = self.zs[ix + 1][iy];
        let z01 = self.zs[ix][iy + 1];
        let z11 = self.zs[ix + 1][iy + 1];
        (one - tx) * (one - ty) * z00
            + tx * (one - ty) * z10
            + (one - tx) * ty * z01
            + tx * ty * z11
    }

    /// The grid x-knots.
    pub fn xs(&self) -> &[T; NX] {
        &self.xs
    }

    /// The grid y-knots.
    pub fn ys(&self) -> &[T; NY] {
        &self.ys
    }
}

// ---------- Dynamic variant ----------

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Bilinear interpolant on a rectangular grid (heap-allocated, runtime-sized).
///
/// Dynamic counterpart of [`BilinearInterp`]. Stores grid values in a flat
/// column-major `Vec` where `zs[ix * ny + iy]` holds the value at
/// `(xs[ix], ys[iy])`, matching the `DynMatrix` convention.
///
/// # Example
///
/// ```
/// use numeris::interp::DynBilinearInterp;
///
/// // 3×2 grid: z = x + y
/// let xs = vec![0.0_f64, 1.0, 2.0];
/// let ys = vec![0.0, 1.0];
/// // Row-major input: zs[iy][ix]
/// let zs = vec![
///     vec![0.0, 1.0, 2.0],  // y = 0
///     vec![1.0, 2.0, 3.0],  // y = 1
/// ];
/// let interp = DynBilinearInterp::new(xs, ys, zs).unwrap();
/// assert!((interp.eval(0.5, 0.5) - 1.0).abs() < 1e-14);
/// ```
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct DynBilinearInterp<T> {
    xs: Vec<T>,
    ys: Vec<T>,
    zs: Vec<T>, // column-major: zs[ix * ny + iy]
    ny: usize,
}

#[cfg(feature = "alloc")]
impl<T: FloatScalar> DynBilinearInterp<T> {
    /// Construct a bilinear interpolant from row-major grid data.
    ///
    /// `zs_rows[j][i]` is the value at `(xs[i], ys[j])`. Each inner Vec is a
    /// "row" of z-values at a fixed y-coordinate. Internally transposed to
    /// column-major storage.
    ///
    /// Returns `InterpError::TooFewPoints` if `xs.len() < 2` or `ys.len() < 2`,
    /// `InterpError::NotSorted` if `xs` or `ys` is not strictly increasing,
    /// `InterpError::LengthMismatch` if `zs_rows` dimensions don't match.
    pub fn new(
        xs: Vec<T>,
        ys: Vec<T>,
        zs_rows: Vec<Vec<T>>,
    ) -> Result<Self, InterpError> {
        let nx = xs.len();
        let ny = ys.len();
        if nx < 2 || ny < 2 {
            return Err(InterpError::TooFewPoints);
        }
        if zs_rows.len() != ny {
            return Err(InterpError::LengthMismatch);
        }
        for row in &zs_rows {
            if row.len() != nx {
                return Err(InterpError::LengthMismatch);
            }
        }
        validate_sorted(&xs)?;
        validate_sorted(&ys)?;

        // Transpose row-major input to column-major storage
        let mut zs = alloc::vec![T::zero(); nx * ny];
        for iy in 0..ny {
            for ix in 0..nx {
                zs[ix * ny + iy] = zs_rows[iy][ix];
            }
        }
        Ok(Self { xs, ys, zs, ny })
    }

    /// Construct from pre-flattened column-major data.
    ///
    /// `zs_col_major[ix * ny + iy]` is the value at `(xs[ix], ys[iy])`.
    pub fn from_slice(
        xs: Vec<T>,
        ys: Vec<T>,
        zs_col_major: Vec<T>,
    ) -> Result<Self, InterpError> {
        let nx = xs.len();
        let ny = ys.len();
        if nx < 2 || ny < 2 {
            return Err(InterpError::TooFewPoints);
        }
        if zs_col_major.len() != nx * ny {
            return Err(InterpError::LengthMismatch);
        }
        validate_sorted(&xs)?;
        validate_sorted(&ys)?;
        Ok(Self {
            xs,
            ys,
            zs: zs_col_major,
            ny,
        })
    }

    /// Evaluate the interpolant at `(x, y)`.
    pub fn eval(&self, x: T, y: T) -> T {
        let ix = find_interval(&self.xs, x);
        let iy = find_interval(&self.ys, y);
        let tx = (x - self.xs[ix]) / (self.xs[ix + 1] - self.xs[ix]);
        let ty = (y - self.ys[iy]) / (self.ys[iy + 1] - self.ys[iy]);
        let one = T::one();
        let z00 = self.zs[ix * self.ny + iy];
        let z10 = self.zs[(ix + 1) * self.ny + iy];
        let z01 = self.zs[ix * self.ny + (iy + 1)];
        let z11 = self.zs[(ix + 1) * self.ny + (iy + 1)];
        (one - tx) * (one - ty) * z00
            + tx * (one - ty) * z10
            + (one - tx) * ty * z01
            + tx * ty * z11
    }

    /// The grid x-knots.
    pub fn xs(&self) -> &[T] {
        &self.xs
    }

    /// The grid y-knots.
    pub fn ys(&self) -> &[T] {
        &self.ys
    }
}
