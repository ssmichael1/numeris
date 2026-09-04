//! Runtime-sized 2D FFT ([`DynFft2`], [`DynRealFft2`], and 2D shifts; requires
//! `alloc`).
//!
//! The 2D DFT is *separable*: an `rows × cols` transform factors into a batch of
//! 1D FFTs along each axis (transform every column, then every row — the order
//! does not matter). So no new kernel is needed; 2D is built entirely on the 1D
//! [`DynFft`], with Bluestein transparently covering non-power-of-two dimensions.
//!
//! # Layout
//!
//! Data is a column-major [`DynMatrix`]`<Complex<T>>`: element `(row, col)` lives
//! at `col*rows + row`, so each *column* is a contiguous slice. The column pass
//! therefore slices straight out of the backing buffer and hands it to the plan;
//! the row pass is strided (stride `rows`) and is handled by gathering each row
//! into a scratch buffer, transforming, and scattering it back.
//!
//! # Examples
//!
//! ```
//! # use numeris::fft::DynFft2;
//! # use numeris::{DynMatrix, Complex};
//! let mut plan = DynFft2::<f64>::new(4, 4);
//! let mut img = DynMatrix::from_fn(4, 4, |r, c| Complex::new((r + c) as f64, 0.0));
//! plan.forward(&mut img);
//! plan.inverse(&mut img);
//! assert!((img[(1, 2)].re - 3.0).abs() < 1e-10);
//! ```

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use num_complex::Complex;

use super::dynfft::DynFft;
use super::real::DynRealFft;
use crate::dynmatrix::DynMatrix;
use crate::traits::FloatScalar;

/// A cached 2D FFT plan for fixed runtime dimensions (requires `alloc`).
///
/// Build once with [`DynFft2::new`], then call [`forward`](DynFft2::forward) /
/// [`inverse`](DynFft2::inverse) repeatedly on `rows × cols`
/// [`DynMatrix`]`<Complex<T>>` buffers. Two 1D plans are held internally — one of
/// length `cols` for the row transforms and one of length `rows` for the column
/// transforms — so twiddles and scratch are computed once and reused. Any
/// dimensions are supported, including non-power-of-two (Bluestein handles those).
pub struct DynFft2<T: FloatScalar> {
    rows: usize,
    cols: usize,
    /// Length-`cols` plan for the (strided) row transforms.
    row_plan: DynFft<T>,
    /// Length-`rows` plan for the (contiguous) column transforms.
    col_plan: DynFft<T>,
    /// Length-`cols` gather/scatter buffer for the strided row pass.
    scratch: Vec<Complex<T>>,
}

impl<T: FloatScalar> DynFft2<T> {
    /// Build a plan for `rows × cols` transforms. Panics if either dimension is
    /// zero.
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "DynFft2 dimensions must be non-zero");
        Self {
            rows,
            cols,
            row_plan: DynFft::new(cols),
            col_plan: DynFft::new(rows),
            scratch: vec![Complex::new(T::zero(), T::zero()); cols],
        }
    }

    /// The row count this plan was built for.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The column count this plan was built for.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// In-place forward 2D FFT. Panics unless `data` is `rows × cols`.
    ///
    /// Uses the `exp(-2πi k n / N)` sign convention on each axis.
    pub fn forward(&mut self, data: &mut DynMatrix<Complex<T>>) {
        self.transform(data, false);
    }

    /// In-place inverse 2D FFT, normalized by `1/(rows*cols)`. Panics unless
    /// `data` is `rows × cols`.
    pub fn inverse(&mut self, data: &mut DynMatrix<Complex<T>>) {
        self.transform(data, true);
    }

    fn transform(&mut self, data: &mut DynMatrix<Complex<T>>, inverse: bool) {
        assert_eq!(
            data.nrows(),
            self.rows,
            "DynFft2: matrix row count must equal plan rows"
        );
        assert_eq!(
            data.ncols(),
            self.cols,
            "DynFft2: matrix column count must equal plan cols"
        );
        let (rows, cols) = (self.rows, self.cols);
        let buf = data.as_mut_slice();

        // Column pass — each column is a contiguous `rows`-length chunk.
        for col in buf.chunks_mut(rows) {
            if inverse {
                self.col_plan.inverse(col);
            } else {
                self.col_plan.forward(col);
            }
        }

        // Row pass — rows are strided (stride `rows`); gather → transform →
        // scatter through the length-`cols` scratch buffer.
        for r in 0..rows {
            for (c, slot) in self.scratch.iter_mut().enumerate() {
                *slot = buf[c * rows + r];
            }
            if inverse {
                self.row_plan.inverse(&mut self.scratch);
            } else {
                self.row_plan.forward(&mut self.scratch);
            }
            for c in 0..cols {
                buf[c * rows + r] = self.scratch[c];
            }
        }
    }
}

/// A cached real-input 2D FFT plan (requires `alloc`).
///
/// For a real `rows × cols` image, the transform along the (contiguous) column
/// axis is a real FFT — its spectrum is Hermitian, so only the `rows/2 + 1`
/// non-redundant bins are kept — and the transform along the row axis is a full
/// complex FFT. The forward result is therefore a `(rows/2 + 1) × cols` complex
/// matrix, at roughly half the cost and storage of a full complex 2D FFT. This
/// is the form image processing wants (see [`fft_convolve`](super::fft_convolve)
/// for the 1D analogue).
pub struct DynRealFft2<T: FloatScalar> {
    rows: usize,
    cols: usize,
    /// Length-`rows` real plan for the contiguous column axis.
    real_plan: DynRealFft<T>,
    /// Length-`cols` complex plan for the (strided) row axis.
    col_plan: DynFft<T>,
    /// Length-`cols` gather/scatter buffer for the strided row axis.
    row_scratch: Vec<Complex<T>>,
    /// `(rows/2 + 1) * cols` workspace so `inverse` never mutates its input.
    spec_scratch: Vec<Complex<T>>,
}

impl<T: FloatScalar> DynRealFft2<T> {
    /// Build a plan for real `rows × cols` images. Panics if either dimension is
    /// zero.
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(
            rows > 0 && cols > 0,
            "DynRealFft2 dimensions must be non-zero"
        );
        let half = rows / 2 + 1;
        Self {
            rows,
            cols,
            real_plan: DynRealFft::new(rows),
            col_plan: DynFft::new(cols),
            row_scratch: vec![Complex::new(T::zero(), T::zero()); cols],
            spec_scratch: vec![Complex::new(T::zero(), T::zero()); half * cols],
        }
    }

    /// The row count this plan was built for.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The column count this plan was built for.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of non-redundant spectral rows, `rows/2 + 1` — the row count of the
    /// complex output.
    #[inline]
    pub fn spectrum_rows(&self) -> usize {
        self.rows / 2 + 1
    }

    /// Forward real 2D FFT. `input` is `rows × cols` real; `output` receives the
    /// `(rows/2 + 1) × cols` complex half-spectrum.
    pub fn forward(&mut self, input: &DynMatrix<T>, output: &mut DynMatrix<Complex<T>>) {
        let half = self.rows / 2 + 1;
        assert_eq!(input.nrows(), self.rows, "DynRealFft2: input rows mismatch");
        assert_eq!(input.ncols(), self.cols, "DynRealFft2: input cols mismatch");
        assert_eq!(
            output.nrows(),
            half,
            "DynRealFft2: output rows must be rows/2 + 1"
        );
        assert_eq!(
            output.ncols(),
            self.cols,
            "DynRealFft2: output cols mismatch"
        );

        let cols = self.cols;
        let src = input.as_slice();
        let dst = output.as_mut_slice();

        // Column pass — real FFT of each contiguous input column into the
        // matching (shorter) contiguous output column.
        for (in_col, out_col) in src.chunks(self.rows).zip(dst.chunks_mut(half)) {
            self.real_plan.forward(in_col, out_col);
        }

        // Row pass — full complex FFT along the strided row axis of the
        // half-spectrum (stride `half`).
        for r in 0..half {
            for (c, slot) in self.row_scratch.iter_mut().enumerate() {
                *slot = dst[c * half + r];
            }
            self.col_plan.forward(&mut self.row_scratch);
            for c in 0..cols {
                dst[c * half + r] = self.row_scratch[c];
            }
        }
    }

    /// Inverse real 2D FFT. `input` is the `(rows/2 + 1) × cols` complex
    /// half-spectrum; `output` receives the `rows × cols` real image (normalized
    /// by `1/(rows*cols)`). `input` is not modified.
    pub fn inverse(&mut self, input: &DynMatrix<Complex<T>>, output: &mut DynMatrix<T>) {
        let half = self.rows / 2 + 1;
        assert_eq!(
            input.nrows(),
            half,
            "DynRealFft2: input rows must be rows/2 + 1"
        );
        assert_eq!(input.ncols(), self.cols, "DynRealFft2: input cols mismatch");
        assert_eq!(
            output.nrows(),
            self.rows,
            "DynRealFft2: output rows mismatch"
        );
        assert_eq!(
            output.ncols(),
            self.cols,
            "DynRealFft2: output cols mismatch"
        );

        let cols = self.cols;
        self.spec_scratch.copy_from_slice(input.as_slice());
        let spec = &mut self.spec_scratch;

        // Row pass — inverse complex FFT along the strided row axis (normalizes
        // by 1/cols).
        for r in 0..half {
            for (c, slot) in self.row_scratch.iter_mut().enumerate() {
                *slot = spec[c * half + r];
            }
            self.col_plan.inverse(&mut self.row_scratch);
            for c in 0..cols {
                spec[c * half + r] = self.row_scratch[c];
            }
        }

        // Column pass — inverse real FFT of each contiguous column (normalizes by
        // 1/rows), reconstructing the real image.
        let dst = output.as_mut_slice();
        for (in_col, out_col) in spec.chunks(half).zip(dst.chunks_mut(self.rows)) {
            self.real_plan.inverse(in_col, out_col);
        }
    }
}

/// Shift the zero-frequency component to the center of a 2D spectrum.
///
/// The 2D analogue of [`fftshift`](super::fftshift): swaps diagonal quadrants,
/// equivalent to a 1D `fftshift` along each axis. For odd dimensions the larger
/// half moves toward the center on that axis. In-place, no allocation.
///
/// Works on a column-major [`DynMatrix`] of any `Copy` element type by rotating
/// each contiguous column, then rotating the buffer by whole column-blocks (which
/// shifts the column axis).
pub fn fftshift2d<T: Copy>(data: &mut DynMatrix<T>) {
    let (rows, cols) = (data.nrows(), data.ncols());
    let buf = data.as_mut_slice();
    // Row-axis shift: rotate each contiguous column by rows/2.
    for col in buf.chunks_mut(rows) {
        col.rotate_right(rows / 2);
    }
    // Column-axis shift: whole columns are `rows`-length blocks, so shifting the
    // columns by cols/2 is a rotation of the flat buffer by (cols/2)*rows.
    buf.rotate_right((cols / 2) * rows);
}

/// Inverse of [`fftshift2d`]: move the center component back to `(0, 0)`.
///
/// The 2D analogue of [`ifftshift`](super::ifftshift). For even dimensions it is
/// identical to [`fftshift2d`]; for odd dimensions it is the true inverse.
/// In-place, no allocation.
pub fn ifftshift2d<T: Copy>(data: &mut DynMatrix<T>) {
    let (rows, cols) = (data.nrows(), data.ncols());
    let buf = data.as_mut_slice();
    buf.rotate_left((cols / 2) * rows);
    for col in buf.chunks_mut(rows) {
        col.rotate_left(rows / 2);
    }
}
