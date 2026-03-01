//! AVX-512-accelerated f32 kernels for x86_64.
//!
//! AVX-512F provides 512-bit registers → 16×f32 lanes.
//! Only compiled when `target_feature = "avx512f"` is enabled
//! (e.g. via `-C target-cpu=native` on Skylake-X+ / Zen 4+).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Dot product of two f32 slices using AVX-512.
///
/// Uses 4 independent accumulators (64 f32 per iteration) to hide
/// multiply-add latency.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 64; // 4 accumulators × 16 lanes

    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();

        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        for i in 0..chunks {
            let off = i * 64;
            acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(_mm512_loadu_ps(ap.add(off)), _mm512_loadu_ps(bp.add(off))));
            acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(_mm512_loadu_ps(ap.add(off + 16)), _mm512_loadu_ps(bp.add(off + 16))));
            acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(_mm512_loadu_ps(ap.add(off + 32)), _mm512_loadu_ps(bp.add(off + 32))));
            acc3 = _mm512_add_ps(acc3, _mm512_mul_ps(_mm512_loadu_ps(ap.add(off + 48)), _mm512_loadu_ps(bp.add(off + 48))));
        }

        acc0 = _mm512_add_ps(acc0, acc1);
        acc2 = _mm512_add_ps(acc2, acc3);
        acc0 = _mm512_add_ps(acc0, acc2);
        let mut sum = _mm512_reduce_add_ps(acc0);

        // Remainder: up to 63 elements — handle 16-wide then scalar
        let tail = chunks * 64;
        let remaining = n - tail;
        let rem_sixteens = remaining / 16;
        let mut acc_rem = _mm512_setzero_ps();
        for i in 0..rem_sixteens {
            let off = tail + i * 16;
            acc_rem = _mm512_add_ps(acc_rem, _mm512_mul_ps(_mm512_loadu_ps(ap.add(off)), _mm512_loadu_ps(bp.add(off))));
        }
        sum += _mm512_reduce_add_ps(acc_rem);

        let scalar_start = tail + rem_sixteens * 16;
        for i in scalar_start..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Matrix multiply C += A * B using AVX-512 with register-blocked micro-kernel.
///
/// Uses an MR×NR (32×4) register-blocked micro-kernel that accumulates the full
/// k-sum in 8 AVX-512 registers before writing back to C, reducing memory traffic
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

    const MR: usize = 32; // 2 AVX-512 registers × 16 f32 lanes
    const NR: usize = 4;

    let m_full = (m / MR) * MR;
    let p_full = (p / NR) * NR;

    // Interior: full MR×NR tiles, register-blocked
    for jb in 0..p_full / NR {
        let j0 = jb * NR;
        for ib in 0..m_full / MR {
            let i0 = ib * MR;
            unsafe { microkernel_32x4(a, b, c, m, n, i0, j0); }
        }
    }

    // Bottom edge: cascade 16×4 → 8×4 → 4×4 → scalar
    let mut i0 = m_full;
    while i0 + 16 <= m {
        for jb in 0..p_full / NR {
            let j0 = jb * NR;
            unsafe { microkernel_16x4(a, b, c, m, n, i0, j0); }
        }
        i0 += 16;
    }
    while i0 + 8 <= m {
        for jb in 0..p_full / NR {
            let j0 = jb * NR;
            unsafe { microkernel_8x4(a, b, c, m, n, i0, j0); }
        }
        i0 += 8;
    }
    while i0 + 4 <= m {
        for jb in 0..p_full / NR {
            let j0 = jb * NR;
            unsafe { microkernel_4x4(a, b, c, m, n, i0, j0); }
        }
        i0 += 4;
    }
    if i0 < m {
        for j in 0..p_full {
            for k in 0..n {
                let b_kj = b[j * n + k];
                for i in i0..m {
                    c[j * m + i] += a[k * m + i] * b_kj;
                }
            }
        }
    }

    // Right edge: cols p_full..p, all rows (SIMD j-k-i on inner loop)
    let i_simd = m / 16;
    let i_tail = i_simd * 16;
    for j in p_full..p {
        for k in 0..n {
            let b_kj = b[j * n + k];
            let a_col = k * m;
            let c_col = j * m;
            unsafe {
                let vb = _mm512_set1_ps(b_kj);
                for i in 0..i_simd {
                    let offset = i * 16;
                    let vc = _mm512_loadu_ps(c.as_ptr().add(c_col + offset));
                    let va = _mm512_loadu_ps(a.as_ptr().add(a_col + offset));
                    let result = _mm512_add_ps(vc, _mm512_mul_ps(va, vb));
                    _mm512_storeu_ps(c.as_mut_ptr().add(c_col + offset), result);
                }
            }
            for i in i_tail..m {
                c[c_col + i] += a[a_col + i] * b_kj;
            }
        }
    }
}

/// Register-blocked 32×4 micro-kernel: accumulates C[i0..i0+32, j0..j0+4] in
/// 8 AVX-512 registers across the full k-loop, writing C only once.
#[inline(always)]
unsafe fn microkernel_32x4(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, i0: usize, j0: usize,
) {
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // 8 accumulator registers: 2 vectors × 4 columns
        let mut acc00 = _mm512_setzero_ps();
        let mut acc10 = _mm512_setzero_ps();
        let mut acc01 = _mm512_setzero_ps();
        let mut acc11 = _mm512_setzero_ps();
        let mut acc02 = _mm512_setzero_ps();
        let mut acc12 = _mm512_setzero_ps();
        let mut acc03 = _mm512_setzero_ps();
        let mut acc13 = _mm512_setzero_ps();

        for k in 0..n {
            let a_off = k * m + i0;
            let a0 = _mm512_loadu_ps(a_ptr.add(a_off));
            let a1 = _mm512_loadu_ps(a_ptr.add(a_off + 16));

            let b0 = _mm512_set1_ps(*b_ptr.add(j0 * n + k));
            acc00 = _mm512_fmadd_ps(a0, b0, acc00);
            acc10 = _mm512_fmadd_ps(a1, b0, acc10);

            let b1 = _mm512_set1_ps(*b_ptr.add((j0 + 1) * n + k));
            acc01 = _mm512_fmadd_ps(a0, b1, acc01);
            acc11 = _mm512_fmadd_ps(a1, b1, acc11);

            let b2 = _mm512_set1_ps(*b_ptr.add((j0 + 2) * n + k));
            acc02 = _mm512_fmadd_ps(a0, b2, acc02);
            acc12 = _mm512_fmadd_ps(a1, b2, acc12);

            let b3 = _mm512_set1_ps(*b_ptr.add((j0 + 3) * n + k));
            acc03 = _mm512_fmadd_ps(a0, b3, acc03);
            acc13 = _mm512_fmadd_ps(a1, b3, acc13);
        }

        // Write back: C += acc
        let c_ptr = c.as_mut_ptr();

        let off0 = j0 * m + i0;
        _mm512_storeu_ps(c_ptr.add(off0), _mm512_add_ps(_mm512_loadu_ps(c_ptr.add(off0)), acc00));
        _mm512_storeu_ps(c_ptr.add(off0 + 16), _mm512_add_ps(_mm512_loadu_ps(c_ptr.add(off0 + 16)), acc10));

        let off1 = (j0 + 1) * m + i0;
        _mm512_storeu_ps(c_ptr.add(off1), _mm512_add_ps(_mm512_loadu_ps(c_ptr.add(off1)), acc01));
        _mm512_storeu_ps(c_ptr.add(off1 + 16), _mm512_add_ps(_mm512_loadu_ps(c_ptr.add(off1 + 16)), acc11));

        let off2 = (j0 + 2) * m + i0;
        _mm512_storeu_ps(c_ptr.add(off2), _mm512_add_ps(_mm512_loadu_ps(c_ptr.add(off2)), acc02));
        _mm512_storeu_ps(c_ptr.add(off2 + 16), _mm512_add_ps(_mm512_loadu_ps(c_ptr.add(off2 + 16)), acc12));

        let off3 = (j0 + 3) * m + i0;
        _mm512_storeu_ps(c_ptr.add(off3), _mm512_add_ps(_mm512_loadu_ps(c_ptr.add(off3)), acc03));
        _mm512_storeu_ps(c_ptr.add(off3 + 16), _mm512_add_ps(_mm512_loadu_ps(c_ptr.add(off3 + 16)), acc13));
    }
}

/// 16×4 mini-kernel (1 __m512 per column, 16 f32 rows).
#[inline(always)]
unsafe fn microkernel_16x4(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, i0: usize, j0: usize,
) {
    unsafe {
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let mut a0 = _mm512_setzero_ps(); let mut a1 = _mm512_setzero_ps();
        let mut a2 = _mm512_setzero_ps(); let mut a3 = _mm512_setzero_ps();
        for k in 0..n {
            let av = _mm512_loadu_ps(ap.add(k * m + i0));
            a0 = _mm512_add_ps(a0, _mm512_mul_ps(av, _mm512_set1_ps(*bp.add(j0 * n + k))));
            a1 = _mm512_add_ps(a1, _mm512_mul_ps(av, _mm512_set1_ps(*bp.add((j0+1) * n + k))));
            a2 = _mm512_add_ps(a2, _mm512_mul_ps(av, _mm512_set1_ps(*bp.add((j0+2) * n + k))));
            a3 = _mm512_add_ps(a3, _mm512_mul_ps(av, _mm512_set1_ps(*bp.add((j0+3) * n + k))));
        }
        let cp = c.as_mut_ptr();
        for (j, acc) in [(j0, a0), (j0+1, a1), (j0+2, a2), (j0+3, a3)] {
            let off = j * m + i0;
            _mm512_storeu_ps(cp.add(off), _mm512_add_ps(_mm512_loadu_ps(cp.add(off)), acc));
        }
    }
}

/// 8×4 mini-kernel (1 __m256 per column).
#[inline(always)]
unsafe fn microkernel_8x4(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, i0: usize, j0: usize,
) {
    unsafe {
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let mut a0 = _mm256_setzero_ps(); let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps(); let mut a3 = _mm256_setzero_ps();
        for k in 0..n {
            let av = _mm256_loadu_ps(ap.add(k * m + i0));
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(av, _mm256_set1_ps(*bp.add(j0 * n + k))));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(av, _mm256_set1_ps(*bp.add((j0+1) * n + k))));
            a2 = _mm256_add_ps(a2, _mm256_mul_ps(av, _mm256_set1_ps(*bp.add((j0+2) * n + k))));
            a3 = _mm256_add_ps(a3, _mm256_mul_ps(av, _mm256_set1_ps(*bp.add((j0+3) * n + k))));
        }
        let cp = c.as_mut_ptr();
        for (j, acc) in [(j0, a0), (j0+1, a1), (j0+2, a2), (j0+3, a3)] {
            let off = j * m + i0;
            _mm256_storeu_ps(cp.add(off), _mm256_add_ps(_mm256_loadu_ps(cp.add(off)), acc));
        }
    }
}

/// 4×4 mini-kernel (1 __m128 per column).
#[inline(always)]
unsafe fn microkernel_4x4(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, i0: usize, j0: usize,
) {
    unsafe {
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let mut a0 = _mm_setzero_ps(); let mut a1 = _mm_setzero_ps();
        let mut a2 = _mm_setzero_ps(); let mut a3 = _mm_setzero_ps();
        for k in 0..n {
            let av = _mm_loadu_ps(ap.add(k * m + i0));
            a0 = _mm_add_ps(a0, _mm_mul_ps(av, _mm_set1_ps(*bp.add(j0 * n + k))));
            a1 = _mm_add_ps(a1, _mm_mul_ps(av, _mm_set1_ps(*bp.add((j0+1) * n + k))));
            a2 = _mm_add_ps(a2, _mm_mul_ps(av, _mm_set1_ps(*bp.add((j0+2) * n + k))));
            a3 = _mm_add_ps(a3, _mm_mul_ps(av, _mm_set1_ps(*bp.add((j0+3) * n + k))));
        }
        let cp = c.as_mut_ptr();
        for (j, acc) in [(j0, a0), (j0+1, a1), (j0+2, a2), (j0+3, a3)] {
            let off = j * m + i0;
            _mm_storeu_ps(cp.add(off), _mm_add_ps(_mm_loadu_ps(cp.add(off)), acc));
        }
    }
}

/// Element-wise addition: out[i] = a[i] + b[i].
#[inline]
pub fn add_slices(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 16;

    unsafe {
        for i in 0..chunks {
            let offset = i * 16;
            let va = _mm512_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
            _mm512_storeu_ps(out.as_mut_ptr().add(offset), _mm512_add_ps(va, vb));
        }
    }

    let tail = chunks * 16;
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
    let chunks = n / 16;

    unsafe {
        for i in 0..chunks {
            let offset = i * 16;
            let va = _mm512_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
            _mm512_storeu_ps(out.as_mut_ptr().add(offset), _mm512_sub_ps(va, vb));
        }
    }

    let tail = chunks * 16;
    for i in tail..n {
        out[i] = a[i] - b[i];
    }
}

/// Scalar multiplication: out[i] = a[i] * scalar.
#[inline]
pub fn scale_slices(a: &[f32], scalar: f32, out: &mut [f32]) {
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 16;

    unsafe {
        let vs = _mm512_set1_ps(scalar);
        for i in 0..chunks {
            let offset = i * 16;
            let va = _mm512_loadu_ps(a.as_ptr().add(offset));
            _mm512_storeu_ps(out.as_mut_ptr().add(offset), _mm512_mul_ps(va, vs));
        }
    }

    let tail = chunks * 16;
    for i in tail..n {
        out[i] = a[i] * scalar;
    }
}

/// AXPY: y[i] -= alpha * x[i].
#[inline]
pub fn axpy_neg(y: &mut [f32], alpha: f32, x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    let n = y.len();
    let chunks = n / 16;

    unsafe {
        let va = _mm512_set1_ps(alpha);
        for i in 0..chunks {
            let offset = i * 16;
            let vy = _mm512_loadu_ps(y.as_ptr().add(offset));
            let vx = _mm512_loadu_ps(x.as_ptr().add(offset));
            let result = _mm512_sub_ps(vy, _mm512_mul_ps(va, vx));
            _mm512_storeu_ps(y.as_mut_ptr().add(offset), result);
        }
    }

    let tail = chunks * 16;
    for i in tail..n {
        y[i] -= alpha * x[i];
    }
}
