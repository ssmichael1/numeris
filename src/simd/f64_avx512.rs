//! AVX-512-accelerated f64 kernels for x86_64.
//!
//! AVX-512F provides 512-bit registers → 8×f64 lanes.
//! Only compiled when `target_feature = "avx512f"` is enabled
//! (e.g. via `-C target-cpu=native` on Skylake-X+ / Zen 4+).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Dot product of two f64 slices using AVX-512.
///
/// Uses 4 independent accumulators (32 f64 per iteration) to hide
/// multiply-add latency.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 32; // 4 accumulators × 8 lanes

    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();

        let mut acc0 = _mm512_setzero_pd();
        let mut acc1 = _mm512_setzero_pd();
        let mut acc2 = _mm512_setzero_pd();
        let mut acc3 = _mm512_setzero_pd();

        for i in 0..chunks {
            let off = i * 32;
            acc0 = _mm512_add_pd(acc0, _mm512_mul_pd(_mm512_loadu_pd(ap.add(off)), _mm512_loadu_pd(bp.add(off))));
            acc1 = _mm512_add_pd(acc1, _mm512_mul_pd(_mm512_loadu_pd(ap.add(off + 8)), _mm512_loadu_pd(bp.add(off + 8))));
            acc2 = _mm512_add_pd(acc2, _mm512_mul_pd(_mm512_loadu_pd(ap.add(off + 16)), _mm512_loadu_pd(bp.add(off + 16))));
            acc3 = _mm512_add_pd(acc3, _mm512_mul_pd(_mm512_loadu_pd(ap.add(off + 24)), _mm512_loadu_pd(bp.add(off + 24))));
        }

        acc0 = _mm512_add_pd(acc0, acc1);
        acc2 = _mm512_add_pd(acc2, acc3);
        acc0 = _mm512_add_pd(acc0, acc2);
        let mut sum = _mm512_reduce_add_pd(acc0);

        // Remainder: up to 31 elements — handle octets then scalar
        let tail = chunks * 32;
        let remaining = n - tail;
        let rem_octs = remaining / 8;
        let mut acc_rem = _mm512_setzero_pd();
        for i in 0..rem_octs {
            let off = tail + i * 8;
            acc_rem = _mm512_add_pd(acc_rem, _mm512_mul_pd(_mm512_loadu_pd(ap.add(off)), _mm512_loadu_pd(bp.add(off))));
        }
        sum += _mm512_reduce_add_pd(acc_rem);

        let scalar_start = tail + rem_octs * 8;
        for i in scalar_start..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Matrix multiply C += A * B using AVX-512 with register-blocked micro-kernel.
///
/// Uses an MR×NR (16×4) register-blocked micro-kernel that accumulates the full
/// k-sum in 8 AVX-512 registers before writing back to C, reducing memory traffic
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

    const MR: usize = 16; // 2 AVX-512 registers × 8 f64 lanes
    const NR: usize = 4;
    const KC: usize = 256;

    let m_full = (m / MR) * MR;
    let p_full = (p / NR) * NR;

    let mut kb = 0;
    while kb < n {
        let k_end = (kb + KC).min(n);

        // Interior: full MR×NR tiles, register-blocked
        for jb in 0..p_full / NR {
            let j0 = jb * NR;
            for ib in 0..m_full / MR {
                let i0 = ib * MR;
                unsafe { microkernel_16x4(a, b, c, m, n, i0, j0, kb, k_end); }
            }
        }

        // Bottom edge: rows m_full..m, cols 0..p_full
        // Cascade through smaller SIMD widths: 8×4 → 4×4 → 2×4 → scalar
        let mut i0 = m_full;
        while i0 + 8 <= m {
            for jb in 0..p_full / NR {
                let j0 = jb * NR;
                unsafe { microkernel_8x4(a, b, c, m, n, i0, j0, kb, k_end); }
            }
            i0 += 8;
        }
        while i0 + 4 <= m {
            for jb in 0..p_full / NR {
                let j0 = jb * NR;
                unsafe { microkernel_4x4(a, b, c, m, n, i0, j0, kb, k_end); }
            }
            i0 += 4;
        }
        while i0 + 2 <= m {
            for jb in 0..p_full / NR {
                let j0 = jb * NR;
                unsafe { microkernel_2x4(a, b, c, m, n, i0, j0, kb, k_end); }
            }
            i0 += 2;
        }
        if i0 < m {
            for j in 0..p_full {
                for k in kb..k_end {
                    c[j * m + i0] += a[k * m + i0] * b[j * n + k];
                }
            }
        }

        // Right edge: cols p_full..p, all rows (SIMD j-k-i on inner loop)
        let i_simd = m / 8;
        let i_tail = i_simd * 8;
        for j in p_full..p {
            for k in kb..k_end {
                let b_kj = b[j * n + k];
                let a_col = k * m;
                let c_col = j * m;
                unsafe {
                    let vb = _mm512_set1_pd(b_kj);
                    for i in 0..i_simd {
                        let offset = i * 8;
                        let vc = _mm512_loadu_pd(c.as_ptr().add(c_col + offset));
                        let va = _mm512_loadu_pd(a.as_ptr().add(a_col + offset));
                        let result = _mm512_add_pd(vc, _mm512_mul_pd(va, vb));
                        _mm512_storeu_pd(c.as_mut_ptr().add(c_col + offset), result);
                    }
                }
                for i in i_tail..m {
                    c[c_col + i] += a[a_col + i] * b_kj;
                }
            }
        }

        kb += KC;
    }
}

/// Register-blocked 16×4 micro-kernel: accumulates C[i0..i0+16, j0..j0+4] in
/// 8 AVX-512 registers across the k-loop, writing C only once.
#[inline(always)]
unsafe fn microkernel_16x4(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, n: usize, i0: usize, j0: usize,
    k_start: usize, k_end: usize,
) {
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // 8 accumulator registers: 2 vectors × 4 columns
        let mut acc00 = _mm512_setzero_pd();
        let mut acc10 = _mm512_setzero_pd();
        let mut acc01 = _mm512_setzero_pd();
        let mut acc11 = _mm512_setzero_pd();
        let mut acc02 = _mm512_setzero_pd();
        let mut acc12 = _mm512_setzero_pd();
        let mut acc03 = _mm512_setzero_pd();
        let mut acc13 = _mm512_setzero_pd();

        for k in k_start..k_end {
            let a_off = k * m + i0;
            let a0 = _mm512_loadu_pd(a_ptr.add(a_off));
            let a1 = _mm512_loadu_pd(a_ptr.add(a_off + 8));

            let b0 = _mm512_set1_pd(*b_ptr.add(j0 * n + k));
            acc00 = _mm512_fmadd_pd(a0, b0, acc00);
            acc10 = _mm512_fmadd_pd(a1, b0, acc10);

            let b1 = _mm512_set1_pd(*b_ptr.add((j0 + 1) * n + k));
            acc01 = _mm512_fmadd_pd(a0, b1, acc01);
            acc11 = _mm512_fmadd_pd(a1, b1, acc11);

            let b2 = _mm512_set1_pd(*b_ptr.add((j0 + 2) * n + k));
            acc02 = _mm512_fmadd_pd(a0, b2, acc02);
            acc12 = _mm512_fmadd_pd(a1, b2, acc12);

            let b3 = _mm512_set1_pd(*b_ptr.add((j0 + 3) * n + k));
            acc03 = _mm512_fmadd_pd(a0, b3, acc03);
            acc13 = _mm512_fmadd_pd(a1, b3, acc13);
        }

        // Write back: C += acc
        let c_ptr = c.as_mut_ptr();

        let off0 = j0 * m + i0;
        _mm512_storeu_pd(c_ptr.add(off0), _mm512_add_pd(_mm512_loadu_pd(c_ptr.add(off0)), acc00));
        _mm512_storeu_pd(c_ptr.add(off0 + 8), _mm512_add_pd(_mm512_loadu_pd(c_ptr.add(off0 + 8)), acc10));

        let off1 = (j0 + 1) * m + i0;
        _mm512_storeu_pd(c_ptr.add(off1), _mm512_add_pd(_mm512_loadu_pd(c_ptr.add(off1)), acc01));
        _mm512_storeu_pd(c_ptr.add(off1 + 8), _mm512_add_pd(_mm512_loadu_pd(c_ptr.add(off1 + 8)), acc11));

        let off2 = (j0 + 2) * m + i0;
        _mm512_storeu_pd(c_ptr.add(off2), _mm512_add_pd(_mm512_loadu_pd(c_ptr.add(off2)), acc02));
        _mm512_storeu_pd(c_ptr.add(off2 + 8), _mm512_add_pd(_mm512_loadu_pd(c_ptr.add(off2 + 8)), acc12));

        let off3 = (j0 + 3) * m + i0;
        _mm512_storeu_pd(c_ptr.add(off3), _mm512_add_pd(_mm512_loadu_pd(c_ptr.add(off3)), acc03));
        _mm512_storeu_pd(c_ptr.add(off3 + 8), _mm512_add_pd(_mm512_loadu_pd(c_ptr.add(off3 + 8)), acc13));
    }
}

/// 8×4 mini-kernel using AVX-512 (1 __m512d per column).
#[inline(always)]
unsafe fn microkernel_8x4(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, n: usize, i0: usize, j0: usize,
    k_start: usize, k_end: usize,
) {
    unsafe {
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let mut a0 = _mm512_setzero_pd(); let mut a1 = _mm512_setzero_pd();
        let mut a2 = _mm512_setzero_pd(); let mut a3 = _mm512_setzero_pd();
        for k in k_start..k_end {
            let av = _mm512_loadu_pd(ap.add(k * m + i0));
            a0 = _mm512_add_pd(a0, _mm512_mul_pd(av, _mm512_set1_pd(*bp.add(j0 * n + k))));
            a1 = _mm512_add_pd(a1, _mm512_mul_pd(av, _mm512_set1_pd(*bp.add((j0+1) * n + k))));
            a2 = _mm512_add_pd(a2, _mm512_mul_pd(av, _mm512_set1_pd(*bp.add((j0+2) * n + k))));
            a3 = _mm512_add_pd(a3, _mm512_mul_pd(av, _mm512_set1_pd(*bp.add((j0+3) * n + k))));
        }
        let cp = c.as_mut_ptr();
        for (j, acc) in [(j0, a0), (j0+1, a1), (j0+2, a2), (j0+3, a3)] {
            let off = j * m + i0;
            _mm512_storeu_pd(cp.add(off), _mm512_add_pd(_mm512_loadu_pd(cp.add(off)), acc));
        }
    }
}

/// 4×4 mini-kernel using AVX-256 width (1 __m256d per column).
#[inline(always)]
unsafe fn microkernel_4x4(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, n: usize, i0: usize, j0: usize,
    k_start: usize, k_end: usize,
) {
    unsafe {
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let mut a0 = _mm256_setzero_pd(); let mut a1 = _mm256_setzero_pd();
        let mut a2 = _mm256_setzero_pd(); let mut a3 = _mm256_setzero_pd();
        for k in k_start..k_end {
            let av = _mm256_loadu_pd(ap.add(k * m + i0));
            a0 = _mm256_add_pd(a0, _mm256_mul_pd(av, _mm256_set1_pd(*bp.add(j0 * n + k))));
            a1 = _mm256_add_pd(a1, _mm256_mul_pd(av, _mm256_set1_pd(*bp.add((j0+1) * n + k))));
            a2 = _mm256_add_pd(a2, _mm256_mul_pd(av, _mm256_set1_pd(*bp.add((j0+2) * n + k))));
            a3 = _mm256_add_pd(a3, _mm256_mul_pd(av, _mm256_set1_pd(*bp.add((j0+3) * n + k))));
        }
        let cp = c.as_mut_ptr();
        for (j, acc) in [(j0, a0), (j0+1, a1), (j0+2, a2), (j0+3, a3)] {
            let off = j * m + i0;
            _mm256_storeu_pd(cp.add(off), _mm256_add_pd(_mm256_loadu_pd(cp.add(off)), acc));
        }
    }
}

/// 2×4 mini-kernel using SSE2 width (1 __m128d per column).
#[inline(always)]
unsafe fn microkernel_2x4(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, n: usize, i0: usize, j0: usize,
    k_start: usize, k_end: usize,
) {
    unsafe {
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let mut a0 = _mm_setzero_pd(); let mut a1 = _mm_setzero_pd();
        let mut a2 = _mm_setzero_pd(); let mut a3 = _mm_setzero_pd();
        for k in k_start..k_end {
            let av = _mm_loadu_pd(ap.add(k * m + i0));
            a0 = _mm_add_pd(a0, _mm_mul_pd(av, _mm_set1_pd(*bp.add(j0 * n + k))));
            a1 = _mm_add_pd(a1, _mm_mul_pd(av, _mm_set1_pd(*bp.add((j0+1) * n + k))));
            a2 = _mm_add_pd(a2, _mm_mul_pd(av, _mm_set1_pd(*bp.add((j0+2) * n + k))));
            a3 = _mm_add_pd(a3, _mm_mul_pd(av, _mm_set1_pd(*bp.add((j0+3) * n + k))));
        }
        let cp = c.as_mut_ptr();
        for (j, acc) in [(j0, a0), (j0+1, a1), (j0+2, a2), (j0+3, a3)] {
            let off = j * m + i0;
            _mm_storeu_pd(cp.add(off), _mm_add_pd(_mm_loadu_pd(cp.add(off)), acc));
        }
    }
}

/// Element-wise addition: out[i] = a[i] + b[i].
#[inline]
pub fn add_slices(a: &[f64], b: &[f64], out: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 8;

    unsafe {
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm512_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm512_loadu_pd(b.as_ptr().add(offset));
            _mm512_storeu_pd(out.as_mut_ptr().add(offset), _mm512_add_pd(va, vb));
        }
    }

    let tail = chunks * 8;
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
    let chunks = n / 8;

    unsafe {
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm512_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm512_loadu_pd(b.as_ptr().add(offset));
            _mm512_storeu_pd(out.as_mut_ptr().add(offset), _mm512_sub_pd(va, vb));
        }
    }

    let tail = chunks * 8;
    for i in tail..n {
        out[i] = a[i] - b[i];
    }
}

/// Scalar multiplication: out[i] = a[i] * scalar.
#[inline]
pub fn scale_slices(a: &[f64], scalar: f64, out: &mut [f64]) {
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 8;

    unsafe {
        let vs = _mm512_set1_pd(scalar);
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm512_loadu_pd(a.as_ptr().add(offset));
            _mm512_storeu_pd(out.as_mut_ptr().add(offset), _mm512_mul_pd(va, vs));
        }
    }

    let tail = chunks * 8;
    for i in tail..n {
        out[i] = a[i] * scalar;
    }
}

/// AXPY: y[i] -= alpha * x[i].
#[inline]
pub fn axpy_neg(y: &mut [f64], alpha: f64, x: &[f64]) {
    debug_assert_eq!(y.len(), x.len());
    let n = y.len();
    let chunks = n / 8;

    unsafe {
        let va = _mm512_set1_pd(alpha);
        for i in 0..chunks {
            let offset = i * 8;
            let vy = _mm512_loadu_pd(y.as_ptr().add(offset));
            let vx = _mm512_loadu_pd(x.as_ptr().add(offset));
            let result = _mm512_sub_pd(vy, _mm512_mul_pd(va, vx));
            _mm512_storeu_pd(y.as_mut_ptr().add(offset), result);
        }
    }

    let tail = chunks * 8;
    for i in tail..n {
        y[i] -= alpha * x[i];
    }
}
