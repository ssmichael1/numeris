//! AVX-accelerated f32 kernels for x86_64.
//!
//! AVX provides 256-bit registers → 8×f32 lanes.
//! Only compiled when `target_feature = "avx"` is enabled
//! (e.g. via `-C target-cpu=native` on Haswell+).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Dot product of two f32 slices using AVX.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let tail = chunks * 8;

    unsafe {
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
        }

        // Horizontal sum: 8 lanes → 1
        let hi128 = _mm256_extractf128_ps(acc, 1);
        let lo128 = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(hi128, lo128); // 4 lanes
        let shuf = _mm_movehl_ps(sum128, sum128);
        let sums = _mm_add_ps(sum128, shuf); // 2 partial sums
        let shuf2 = _mm_shuffle_ps(sums, sums, 1);
        let total = _mm_add_ss(sums, shuf2);
        let mut sum = _mm_cvtss_f32(total);

        for i in tail..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Matrix multiply C += A * B using AVX with register-blocked micro-kernel.
///
/// Uses an MR×NR (16×4) register-blocked micro-kernel that accumulates the full
/// k-sum in 8 AVX registers before writing back to C, reducing memory traffic
/// from O(m·n·p) to O(m·p) stores. Technique inspired by nano-gemm
/// (Sarah Quinones, <https://github.com/sarah-quinones/nano-gemm>).
///
/// `a` is m×n, `b` is n×p, `c` is m×p (column-major flat slices).
/// Column-major indexing: element (row, col) is at `col * nrows + row`.
#[inline]
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, p: usize) {
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(b.len(), n * p);
    debug_assert_eq!(c.len(), m * p);

    const MR: usize = 16; // 2 __m256 registers × 8 f32 lanes
    const NR: usize = 4;

    let m_full = (m / MR) * MR;
    let p_full = (p / NR) * NR;

    // Interior: full MR×NR tiles, register-blocked
    for jb in 0..p_full / NR {
        let j0 = jb * NR;
        for ib in 0..m_full / MR {
            let i0 = ib * MR;
            unsafe { microkernel_16x4(a, b, c, m, n, i0, j0); }
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
    let i_simd = m / 8;
    let i_tail = i_simd * 8;
    for j in p_full..p {
        for k in 0..n {
            let b_kj = b[j * n + k];
            let a_col = k * m;
            let c_col = j * m;
            unsafe {
                let vb = _mm256_set1_ps(b_kj);
                for i in 0..i_simd {
                    let offset = i * 8;
                    let vc = _mm256_loadu_ps(c.as_ptr().add(c_col + offset));
                    let va = _mm256_loadu_ps(a.as_ptr().add(a_col + offset));
                    let result = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
                    _mm256_storeu_ps(c.as_mut_ptr().add(c_col + offset), result);
                }
            }
            for i in i_tail..m {
                c[c_col + i] += a[a_col + i] * b_kj;
            }
        }
    }
}

/// Register-blocked 16×4 micro-kernel: accumulates C[i0..i0+16, j0..j0+4] in
/// 8 AVX registers across the full k-loop, writing C only once.
#[inline(always)]
unsafe fn microkernel_16x4(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, i0: usize, j0: usize,
) {
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // 8 accumulator registers: 2 vectors × 4 columns
        let mut acc00 = _mm256_setzero_ps();
        let mut acc10 = _mm256_setzero_ps();
        let mut acc01 = _mm256_setzero_ps();
        let mut acc11 = _mm256_setzero_ps();
        let mut acc02 = _mm256_setzero_ps();
        let mut acc12 = _mm256_setzero_ps();
        let mut acc03 = _mm256_setzero_ps();
        let mut acc13 = _mm256_setzero_ps();

        for k in 0..n {
            let a_off = k * m + i0;
            let a0 = _mm256_loadu_ps(a_ptr.add(a_off));
            let a1 = _mm256_loadu_ps(a_ptr.add(a_off + 8));

            let b0 = _mm256_set1_ps(*b_ptr.add(j0 * n + k));
            acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b0));
            acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a1, b0));

            let b1 = _mm256_set1_ps(*b_ptr.add((j0 + 1) * n + k));
            acc01 = _mm256_add_ps(acc01, _mm256_mul_ps(a0, b1));
            acc11 = _mm256_add_ps(acc11, _mm256_mul_ps(a1, b1));

            let b2 = _mm256_set1_ps(*b_ptr.add((j0 + 2) * n + k));
            acc02 = _mm256_add_ps(acc02, _mm256_mul_ps(a0, b2));
            acc12 = _mm256_add_ps(acc12, _mm256_mul_ps(a1, b2));

            let b3 = _mm256_set1_ps(*b_ptr.add((j0 + 3) * n + k));
            acc03 = _mm256_add_ps(acc03, _mm256_mul_ps(a0, b3));
            acc13 = _mm256_add_ps(acc13, _mm256_mul_ps(a1, b3));
        }

        // Write back: C += acc
        let c_ptr = c.as_mut_ptr();

        let off0 = j0 * m + i0;
        _mm256_storeu_ps(c_ptr.add(off0), _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off0)), acc00));
        _mm256_storeu_ps(c_ptr.add(off0 + 8), _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off0 + 8)), acc10));

        let off1 = (j0 + 1) * m + i0;
        _mm256_storeu_ps(c_ptr.add(off1), _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off1)), acc01));
        _mm256_storeu_ps(c_ptr.add(off1 + 8), _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off1 + 8)), acc11));

        let off2 = (j0 + 2) * m + i0;
        _mm256_storeu_ps(c_ptr.add(off2), _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off2)), acc02));
        _mm256_storeu_ps(c_ptr.add(off2 + 8), _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off2 + 8)), acc12));

        let off3 = (j0 + 3) * m + i0;
        _mm256_storeu_ps(c_ptr.add(off3), _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off3)), acc03));
        _mm256_storeu_ps(c_ptr.add(off3 + 8), _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(off3 + 8)), acc13));
    }
}

/// Element-wise addition: out[i] = a[i] + b[i].
#[inline]
pub fn add_slices(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 8;

    unsafe {
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), _mm256_add_ps(va, vb));
        }
    }

    let tail = chunks * 8;
    for i in tail..n {
        out[i] = a[i] + b[i];
    }
}

/// Element-wise subtraction: out[i] = a[i] - b[i].
#[inline]
pub fn sub_slices(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 8;

    unsafe {
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), _mm256_sub_ps(va, vb));
        }
    }

    let tail = chunks * 8;
    for i in tail..n {
        out[i] = a[i] - b[i];
    }
}

/// Scalar multiplication: out[i] = a[i] * scalar.
#[inline]
pub fn scale_slices(a: &[f32], scalar: f32, out: &mut [f32]) {
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 8;

    unsafe {
        let vs = _mm256_set1_ps(scalar);
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), _mm256_mul_ps(va, vs));
        }
    }

    let tail = chunks * 8;
    for i in tail..n {
        out[i] = a[i] * scalar;
    }
}

/// AXPY: y[i] -= alpha * x[i].
#[inline]
pub fn axpy_neg(y: &mut [f32], alpha: f32, x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    let n = y.len();
    let chunks = n / 8;

    unsafe {
        let va = _mm256_set1_ps(alpha);
        for i in 0..chunks {
            let offset = i * 8;
            let vy = _mm256_loadu_ps(y.as_ptr().add(offset));
            let vx = _mm256_loadu_ps(x.as_ptr().add(offset));
            let result = _mm256_sub_ps(vy, _mm256_mul_ps(va, vx));
            _mm256_storeu_ps(y.as_mut_ptr().add(offset), result);
        }
    }

    let tail = chunks * 8;
    for i in tail..n {
        y[i] -= alpha * x[i];
    }
}
