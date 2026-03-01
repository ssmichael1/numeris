//! AVX-accelerated f64 kernels for x86_64.
//!
//! AVX provides 256-bit registers → 4×f64 lanes.
//! Only compiled when `target_feature = "avx"` is enabled
//! (e.g. via `-C target-cpu=native` on Haswell+).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Dot product of two f64 slices using AVX.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let tail = chunks * 4;

    unsafe {
        let mut acc = _mm256_setzero_pd();

        for i in 0..chunks {
            let va = _mm256_loadu_pd(a.as_ptr().add(i * 4));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i * 4));
            acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
        }

        // Horizontal sum: [a, b, c, d] → a+b+c+d
        let hi128 = _mm256_extractf128_pd(acc, 1); // [c, d]
        let lo128 = _mm256_castpd256_pd128(acc); // [a, b]
        let sum128 = _mm_add_pd(hi128, lo128); // [a+c, b+d]
        let hi64 = _mm_unpackhi_pd(sum128, sum128); // [b+d, b+d]
        let result = _mm_add_sd(sum128, hi64); // [a+b+c+d, ...]
        let mut sum = _mm_cvtsd_f64(result);

        for i in tail..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Matrix multiply C += A * B using AVX with register-blocked micro-kernel.
///
/// Uses an MR×NR (8×4) register-blocked micro-kernel that accumulates the full
/// k-sum in 8 AVX registers before writing back to C, reducing memory traffic
/// from O(m·n·p) to O(m·p) stores. Technique inspired by nano-gemm
/// (Sarah Quinones, <https://github.com/sarah-quinones/nano-gemm>).
///
/// `a` is m×n, `b` is n×p, `c` is m×p (column-major flat slices).
/// Column-major indexing: element (row, col) is at `col * nrows + row`.
#[inline]
pub fn matmul(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, p: usize) {
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(b.len(), n * p);
    debug_assert_eq!(c.len(), m * p);

    const MR: usize = 8; // 2 __m256d registers × 4 f64 lanes
    const NR: usize = 4;

    let m_full = (m / MR) * MR;
    let p_full = (p / NR) * NR;

    // Interior: full MR×NR tiles, register-blocked
    for jb in 0..p_full / NR {
        let j0 = jb * NR;
        for ib in 0..m_full / MR {
            let i0 = ib * MR;
            unsafe { microkernel_8x4(a, b, c, m, n, i0, j0); }
        }
    }

    // Bottom edge: rows m_full..m, cols 0..p_full (scalar)
    for j in 0..p_full {
        for k in 0..n {
            let b_kj = b[j * n + k];
            let a_col = k * m;
            let c_col = j * m;
            for i in m_full..m {
                c[c_col + i] += a[a_col + i] * b_kj;
            }
        }
    }

    // Right edge: cols p_full..p, all rows (SIMD j-k-i on inner loop)
    let i_simd = m / 4;
    let i_tail = i_simd * 4;
    for j in p_full..p {
        for k in 0..n {
            let b_kj = b[j * n + k];
            let a_col = k * m;
            let c_col = j * m;
            unsafe {
                let vb = _mm256_set1_pd(b_kj);
                for i in 0..i_simd {
                    let offset = i * 4;
                    let vc = _mm256_loadu_pd(c.as_ptr().add(c_col + offset));
                    let va = _mm256_loadu_pd(a.as_ptr().add(a_col + offset));
                    let result = _mm256_add_pd(vc, _mm256_mul_pd(va, vb));
                    _mm256_storeu_pd(c.as_mut_ptr().add(c_col + offset), result);
                }
            }
            for i in i_tail..m {
                c[c_col + i] += a[a_col + i] * b_kj;
            }
        }
    }
}

/// Register-blocked 8×4 micro-kernel: accumulates C[i0..i0+8, j0..j0+4] in
/// 8 AVX registers across the full k-loop, writing C only once.
#[inline(always)]
unsafe fn microkernel_8x4(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, n: usize, i0: usize, j0: usize,
) {
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // 8 accumulator registers: 2 vectors × 4 columns
        let mut acc00 = _mm256_setzero_pd();
        let mut acc10 = _mm256_setzero_pd();
        let mut acc01 = _mm256_setzero_pd();
        let mut acc11 = _mm256_setzero_pd();
        let mut acc02 = _mm256_setzero_pd();
        let mut acc12 = _mm256_setzero_pd();
        let mut acc03 = _mm256_setzero_pd();
        let mut acc13 = _mm256_setzero_pd();

        for k in 0..n {
            let a_off = k * m + i0;
            let a0 = _mm256_loadu_pd(a_ptr.add(a_off));
            let a1 = _mm256_loadu_pd(a_ptr.add(a_off + 4));

            let b0 = _mm256_set1_pd(*b_ptr.add(j0 * n + k));
            acc00 = _mm256_add_pd(acc00, _mm256_mul_pd(a0, b0));
            acc10 = _mm256_add_pd(acc10, _mm256_mul_pd(a1, b0));

            let b1 = _mm256_set1_pd(*b_ptr.add((j0 + 1) * n + k));
            acc01 = _mm256_add_pd(acc01, _mm256_mul_pd(a0, b1));
            acc11 = _mm256_add_pd(acc11, _mm256_mul_pd(a1, b1));

            let b2 = _mm256_set1_pd(*b_ptr.add((j0 + 2) * n + k));
            acc02 = _mm256_add_pd(acc02, _mm256_mul_pd(a0, b2));
            acc12 = _mm256_add_pd(acc12, _mm256_mul_pd(a1, b2));

            let b3 = _mm256_set1_pd(*b_ptr.add((j0 + 3) * n + k));
            acc03 = _mm256_add_pd(acc03, _mm256_mul_pd(a0, b3));
            acc13 = _mm256_add_pd(acc13, _mm256_mul_pd(a1, b3));
        }

        // Write back: C += acc
        let c_ptr = c.as_mut_ptr();

        let off0 = j0 * m + i0;
        _mm256_storeu_pd(c_ptr.add(off0), _mm256_add_pd(_mm256_loadu_pd(c_ptr.add(off0)), acc00));
        _mm256_storeu_pd(c_ptr.add(off0 + 4), _mm256_add_pd(_mm256_loadu_pd(c_ptr.add(off0 + 4)), acc10));

        let off1 = (j0 + 1) * m + i0;
        _mm256_storeu_pd(c_ptr.add(off1), _mm256_add_pd(_mm256_loadu_pd(c_ptr.add(off1)), acc01));
        _mm256_storeu_pd(c_ptr.add(off1 + 4), _mm256_add_pd(_mm256_loadu_pd(c_ptr.add(off1 + 4)), acc11));

        let off2 = (j0 + 2) * m + i0;
        _mm256_storeu_pd(c_ptr.add(off2), _mm256_add_pd(_mm256_loadu_pd(c_ptr.add(off2)), acc02));
        _mm256_storeu_pd(c_ptr.add(off2 + 4), _mm256_add_pd(_mm256_loadu_pd(c_ptr.add(off2 + 4)), acc12));

        let off3 = (j0 + 3) * m + i0;
        _mm256_storeu_pd(c_ptr.add(off3), _mm256_add_pd(_mm256_loadu_pd(c_ptr.add(off3)), acc03));
        _mm256_storeu_pd(c_ptr.add(off3 + 4), _mm256_add_pd(_mm256_loadu_pd(c_ptr.add(off3 + 4)), acc13));
    }
}

/// Element-wise addition: out[i] = a[i] + b[i].
#[inline]
pub fn add_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;

    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm256_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
            _mm256_storeu_pd(out.as_mut_ptr().add(offset), _mm256_add_pd(va, vb));
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        out[i] = a[i] + b[i];
    }
}

/// Element-wise subtraction: out[i] = a[i] - b[i].
#[inline]
pub fn sub_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;

    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm256_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
            _mm256_storeu_pd(out.as_mut_ptr().add(offset), _mm256_sub_pd(va, vb));
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        out[i] = a[i] - b[i];
    }
}

/// Scalar multiplication: out[i] = a[i] * scalar.
#[inline]
pub fn scale_slices(a: &[f64], scalar: f64, out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;

    unsafe {
        let vs = _mm256_set1_pd(scalar);
        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm256_loadu_pd(a.as_ptr().add(offset));
            _mm256_storeu_pd(out.as_mut_ptr().add(offset), _mm256_mul_pd(va, vs));
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        out[i] = a[i] * scalar;
    }
}

/// AXPY: y[i] -= alpha * x[i].
#[inline]
pub fn axpy_neg(y: &mut [f64], alpha: f64, x: &[f64]) {
    debug_assert_eq!(y.len(), x.len());
    let n = y.len();
    let chunks = n / 4;

    unsafe {
        let va = _mm256_set1_pd(alpha);
        for i in 0..chunks {
            let offset = i * 4;
            let vy = _mm256_loadu_pd(y.as_ptr().add(offset));
            let vx = _mm256_loadu_pd(x.as_ptr().add(offset));
            let result = _mm256_sub_pd(vy, _mm256_mul_pd(va, vx));
            _mm256_storeu_pd(y.as_mut_ptr().add(offset), result);
        }
    }

    let tail = chunks * 4;
    for i in tail..n {
        y[i] -= alpha * x[i];
    }
}
