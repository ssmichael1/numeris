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
//! therefore slices straight out of the backing buffer and hands it to the plan.
//! The row axis is strided (stride `rows`), so the row pass runs on a
//! *transposed* copy: transpose into a `cols × rows` work buffer (a cache-blocked
//! copy), batch-transform its now-contiguous columns, and transpose back. Both
//! FFT passes then hit the contiguous SIMD path, and every column of either pass
//! is disjoint output — which is what lets the batches run in parallel.
//!
//! # Parallelism
//!
//! A single 1D FFT is never multithreaded (its stages are a serial chain), but
//! the 2D passes are batches of independent 1D transforms. Under the `rayon`
//! feature each pass fans out over columns above a work threshold, sharing the
//! read-only plan and giving each worker its own scratch
//! ([`DynFft::make_scratch`]); results are identical to the sequential path.
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

use super::dynfft::{DynFft, DynFftScratch};
use super::real::{DynRealFft, DynRealFftScratch};
use crate::dynmatrix::DynMatrix;
use crate::par::{self, MaybeSync};
use crate::traits::FloatScalar;

/// Work budget (in element·stage units, `len · log2(len)` per column) above
/// which a batch of 1D transforms is spread across the rayon pool. Same scale as
/// `imageproc`'s separable-convolution budget.
#[cfg(feature = "rayon")]
const FFT_PAR_WORK_BUDGET: usize = 500_000;
/// Floor on the column count for a parallel batch.
#[cfg(feature = "rayon")]
const FFT_PAR_MIN_COLS: usize = 8;
/// Work budget for the blocked transposes (elements copied per band).
const TRANSPOSE_PAR_WORK_BUDGET: usize = 250_000;
/// Row-band height of the blocked transpose.
const TRANSPOSE_TILE: usize = 32;

/// Approximate per-column cost of a length-`len` FFT: `len · (log2(len) + 1)`.
#[cfg(feature = "rayon")]
#[inline]
fn fft_col_work(len: usize) -> usize {
    len * (len.max(2).ilog2() as usize + 1)
}

/// Transform every contiguous `plan.len()`-length column of `buf` in place.
///
/// Under `rayon`, above the work threshold the columns run in parallel through
/// [`par::for_each_chunk_mut_init`], each worker allocating its own scratch;
/// otherwise (and always without `rayon`) they run sequentially through the
/// caller's `scratch`, allocating nothing.
fn batch_transform<T: FloatScalar + MaybeSync>(
    buf: &mut [Complex<T>],
    plan: &DynFft<T>,
    scratch: &mut DynFftScratch<T>,
    inverse: bool,
) {
    let len = plan.len();
    #[cfg(feature = "rayon")]
    {
        let threshold =
            par::work_col_threshold(fft_col_work(len), FFT_PAR_WORK_BUDGET, FFT_PAR_MIN_COLS);
        if buf.len() / len >= threshold {
            par::for_each_chunk_mut_init(
                buf,
                len,
                threshold,
                || plan.make_scratch(),
                |s, _, col| {
                    if inverse {
                        plan.inverse_with(col, s);
                    } else {
                        plan.forward_with(col, s);
                    }
                },
            );
            return;
        }
    }
    for col in buf.chunks_exact_mut(len) {
        if inverse {
            plan.inverse_with(col, scratch);
        } else {
            plan.forward_with(col, scratch);
        }
    }
}

/// Real forward FFT of every `rows`-length column of `src` into the matching
/// `rows/2 + 1`-length column of `dst` (same batching policy as
/// [`batch_transform`]).
fn batch_real_forward<T: FloatScalar + MaybeSync>(
    src: &[T],
    dst: &mut [Complex<T>],
    plan: &DynRealFft<T>,
    scratch: &mut DynRealFftScratch<T>,
) {
    let rows = plan.len();
    let half = rows / 2 + 1;
    #[cfg(feature = "rayon")]
    {
        let threshold =
            par::work_col_threshold(fft_col_work(rows), FFT_PAR_WORK_BUDGET, FFT_PAR_MIN_COLS);
        if dst.len() / half >= threshold {
            par::for_each_chunk_mut_init(
                dst,
                half,
                threshold,
                || plan.make_scratch(),
                |s, j, out_col| plan.forward_with(&src[j * rows..(j + 1) * rows], out_col, s),
            );
            return;
        }
    }
    for (in_col, out_col) in src.chunks_exact(rows).zip(dst.chunks_exact_mut(half)) {
        plan.forward_with(in_col, out_col, scratch);
    }
}

/// Real inverse FFT of every `rows/2 + 1`-length column of `spec` into the
/// matching `rows`-length column of `dst`.
fn batch_real_inverse<T: FloatScalar + MaybeSync>(
    spec: &[Complex<T>],
    dst: &mut [T],
    plan: &DynRealFft<T>,
    scratch: &mut DynRealFftScratch<T>,
) {
    let rows = plan.len();
    let half = rows / 2 + 1;
    #[cfg(feature = "rayon")]
    {
        let threshold =
            par::work_col_threshold(fft_col_work(rows), FFT_PAR_WORK_BUDGET, FFT_PAR_MIN_COLS);
        if dst.len() / rows >= threshold {
            par::for_each_chunk_mut_init(
                dst,
                rows,
                threshold,
                || plan.make_scratch(),
                |s, j, out_col| plan.inverse_with(&spec[j * half..(j + 1) * half], out_col, s),
            );
            return;
        }
    }
    for (in_col, out_col) in spec.chunks_exact(half).zip(dst.chunks_exact_mut(rows)) {
        plan.inverse_with(in_col, out_col, scratch);
    }
}

/// Out-of-place transpose of a column-major `rows × cols` buffer into a
/// column-major `cols × rows` buffer: `dst[(c, r)] = src[(r, c)]`, i.e.
/// `dst[r*cols + c] = src[c*rows + r]`.
///
/// Cache-blocked in `TRANSPOSE_TILE`-row bands (each band of `dst` is a
/// disjoint block of whole output columns, so the bands parallelize under
/// `rayon` above a work threshold): within a band every source column read is
/// a short contiguous run and the destination writes stay inside a
/// tile × tile working set.
fn transpose_into<T: Copy + MaybeSync>(src: &[T], rows: usize, cols: usize, dst: &mut [T]) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);
    if rows == 0 || cols == 0 {
        return;
    }
    let band = TRANSPOSE_TILE * cols;
    let threshold = par::work_col_threshold(band, TRANSPOSE_PAR_WORK_BUDGET, 4);
    par::for_each_chunk_mut(dst, band, threshold, |t, chunk| {
        let r0 = t * TRANSPOSE_TILE;
        let nr = chunk.len() / cols;
        let mut c0 = 0;
        while c0 < cols {
            let c1 = (c0 + TRANSPOSE_TILE).min(cols);
            for c in c0..c1 {
                let col = &src[c * rows + r0..c * rows + r0 + nr];
                for (dr, &v) in col.iter().enumerate() {
                    chunk[dr * cols + c] = v;
                }
            }
            c0 = c1;
        }
    });
}

/// A cached 2D FFT plan for fixed runtime dimensions (requires `alloc`).
///
/// Build once with [`DynFft2::new`], then call [`forward`](DynFft2::forward) /
/// [`inverse`](DynFft2::inverse) repeatedly on `rows × cols`
/// [`DynMatrix`]`<Complex<T>>` buffers. Two 1D plans are held internally — one of
/// length `cols` for the row transforms and one of length `rows` for the column
/// transforms — so twiddles and scratch are computed once and reused. Any
/// dimensions are supported, including non-power-of-two (Bluestein handles those).
///
/// The row pass runs on a transposed copy so both passes are contiguous (see
/// the module docs); under `rayon` both passes and the transposes parallelize.
pub struct DynFft2<T: FloatScalar> {
    rows: usize,
    cols: usize,
    /// Length-`cols` plan for the row transforms (run on the transposed buffer).
    row_plan: DynFft<T>,
    /// Length-`rows` plan for the (contiguous) column transforms.
    col_plan: DynFft<T>,
    row_scratch: DynFftScratch<T>,
    col_scratch: DynFftScratch<T>,
    /// `cols × rows` transposed work buffer for the row pass.
    tbuf: Vec<Complex<T>>,
}

impl<T: FloatScalar + MaybeSync> DynFft2<T> {
    /// Build a plan for `rows × cols` transforms. Panics if either dimension is
    /// zero.
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "DynFft2 dimensions must be non-zero");
        let row_plan = DynFft::new(cols);
        let col_plan = DynFft::new(rows);
        let row_scratch = row_plan.make_scratch();
        let col_scratch = col_plan.make_scratch();
        Self {
            rows,
            cols,
            row_plan,
            col_plan,
            row_scratch,
            col_scratch,
            tbuf: vec![Complex::new(T::zero(), T::zero()); rows * cols],
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
        batch_transform(buf, &self.col_plan, &mut self.col_scratch, inverse);

        // Row pass — transpose so rows become contiguous columns, transform
        // them, transpose back.
        transpose_into(buf, rows, cols, &mut self.tbuf);
        batch_transform(
            &mut self.tbuf,
            &self.row_plan,
            &mut self.row_scratch,
            inverse,
        );
        transpose_into(&self.tbuf, cols, rows, buf);
    }
}

/// A cached real-input 2D FFT plan (requires `alloc`).
///
/// For a real `rows × cols` image, the transform along the (contiguous) column
/// axis is a real FFT — its spectrum is Hermitian, so only the `rows/2 + 1`
/// non-redundant bins are kept — and the transform along the row axis is a full
/// complex FFT. The forward result is therefore a `(rows/2 + 1) × cols` complex
/// matrix, at roughly half the cost and storage of a full complex 2D FFT. This
/// is the form image processing wants (see [`fft_convolve2d`](super::fft_convolve2d)).
///
/// Same transpose-based row pass and `rayon` batching as [`DynFft2`].
pub struct DynRealFft2<T: FloatScalar> {
    rows: usize,
    cols: usize,
    /// Length-`rows` real plan for the contiguous column axis.
    real_plan: DynRealFft<T>,
    /// Length-`cols` complex plan for the row axis (run on the transposed buffer).
    col_plan: DynFft<T>,
    real_scratch: DynRealFftScratch<T>,
    col_scratch: DynFftScratch<T>,
    /// `cols × (rows/2 + 1)` transposed work buffer for the row pass.
    tbuf: Vec<Complex<T>>,
    /// `(rows/2 + 1) × cols` workspace for the inverse's column pass, so
    /// `inverse` never mutates its input.
    spec: Vec<Complex<T>>,
}

impl<T: FloatScalar + MaybeSync> DynRealFft2<T> {
    /// Build a plan for real `rows × cols` images. Panics if either dimension is
    /// zero.
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(
            rows > 0 && cols > 0,
            "DynRealFft2 dimensions must be non-zero"
        );
        let half = rows / 2 + 1;
        let real_plan = DynRealFft::new(rows);
        let col_plan = DynFft::new(cols);
        let real_scratch = real_plan.make_scratch();
        let col_scratch = col_plan.make_scratch();
        let zero = Complex::new(T::zero(), T::zero());
        Self {
            rows,
            cols,
            real_plan,
            col_plan,
            real_scratch,
            col_scratch,
            tbuf: vec![zero; half * cols],
            spec: vec![zero; half * cols],
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
        let dst = output.as_mut_slice();

        // Column pass — real FFT of each contiguous input column into the
        // matching (shorter) contiguous output column.
        batch_real_forward(
            input.as_slice(),
            dst,
            &self.real_plan,
            &mut self.real_scratch,
        );

        // Row pass — full complex FFT along the row axis of the half-spectrum,
        // via the transposed buffer.
        transpose_into(dst, half, cols, &mut self.tbuf);
        batch_transform(&mut self.tbuf, &self.col_plan, &mut self.col_scratch, false);
        transpose_into(&self.tbuf, cols, half, dst);
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

        // Row pass — inverse complex FFT along the row axis (normalizes by
        // 1/cols), transposing straight out of the (untouched) input and back
        // into the private spectrum workspace.
        transpose_into(input.as_slice(), half, cols, &mut self.tbuf);
        batch_transform(&mut self.tbuf, &self.col_plan, &mut self.col_scratch, true);
        transpose_into(&self.tbuf, cols, half, &mut self.spec);

        // Column pass — inverse real FFT of each contiguous column (normalizes by
        // 1/rows), reconstructing the real image.
        batch_real_inverse(
            &self.spec,
            output.as_mut_slice(),
            &self.real_plan,
            &mut self.real_scratch,
        );
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
