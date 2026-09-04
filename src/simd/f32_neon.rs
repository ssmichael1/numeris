//! NEON-accelerated f32 kernels for aarch64.
//!
//! NEON provides 128-bit registers → 4×f32 lanes.

use core::arch::aarch64::*;

/// Dot product of two f32 slices using NEON.
///
/// Uses 4 independent accumulators (16 f32 per iteration) to hide
/// FMA latency.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // SAFETY: register broadcasts of zero; they touch no memory.
    let (mut acc0, mut acc1, mut acc2, mut acc3) = unsafe {
        (
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
        )
    };

    // 4 accumulators × 4 lanes = 16 elements per iteration.
    let mut ai = a.chunks_exact(16);
    let mut bi = b.chunks_exact(16);
    for (ac, bc) in (&mut ai).zip(&mut bi) {
        // SAFETY: `chunks_exact(16)` yields chunks of exactly 16 `f32`, so the
        // four 4-lane loads at offsets 0, 4, 8 and 12 cover each chunk exactly.
        unsafe {
            let (ap, bp) = (ac.as_ptr(), bc.as_ptr());
            acc0 = vfmaq_f32(acc0, vld1q_f32(ap), vld1q_f32(bp));
            acc1 = vfmaq_f32(acc1, vld1q_f32(ap.add(4)), vld1q_f32(bp.add(4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(ap.add(8)), vld1q_f32(bp.add(8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(ap.add(12)), vld1q_f32(bp.add(12)));
        }
    }

    // SAFETY: register arithmetic only — no memory is touched.
    let mut sum = unsafe {
        let s01 = vaddq_f32(acc0, acc1);
        let s23 = vaddq_f32(acc2, acc3);
        vaddvq_f32(vaddq_f32(s01, s23))
    };

    // Remainder: up to 15 elements — 4-wide vectors first, then scalar.
    // SAFETY: register broadcast of zero.
    let mut acc_rem = unsafe { vdupq_n_f32(0.0) };
    let mut ar = ai.remainder().chunks_exact(4);
    let mut br = bi.remainder().chunks_exact(4);
    for (ac, bc) in (&mut ar).zip(&mut br) {
        // SAFETY: each chunk is exactly 4 `f32` — one vector load each.
        unsafe {
            acc_rem = vfmaq_f32(acc_rem, vld1q_f32(ac.as_ptr()), vld1q_f32(bc.as_ptr()));
        }
    }
    // SAFETY: register arithmetic only.
    sum += unsafe { vaddvq_f32(acc_rem) };

    for (&x, &y) in ar.remainder().iter().zip(br.remainder()) {
        sum += x * y;
    }
    sum
}

/// Matrix multiply C += A * B using NEON with register-blocked micro-kernel.
///
/// Uses an MR×NR (8×4) register-blocked micro-kernel with k-blocking (KC=256)
/// to keep the A panel and B micro-panel in L2 cache. Technique inspired by
/// nano-gemm (Sarah Quinones, <https://github.com/sarah-quinones/nano-gemm>).
///
/// `a` is m×n, `b` is n×p, `c` is m×p (column-major flat slices).
/// Column-major indexing: element (row, col) is at `col * nrows + row`.
///
/// # Panics
///
/// Panics unless `a.len() == m·n`, `b.len() == n·p` and `c.len() == m·p`.
/// The microkernels' `# Safety` bounds contracts assume these dimensions, so
/// they are checked in release builds, not just under `debug_assertions`.
#[inline]
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, p: usize) {
    assert_eq!(a.len(), m * n, "matmul: a.len() != m*n");
    assert_eq!(b.len(), n * p, "matmul: b.len() != n*p");
    assert_eq!(c.len(), m * p, "matmul: c.len() != m*p");

    const MR: usize = 8; // 2 NEON registers × 4 f32 lanes
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
                // SAFETY: `i0 + 8 <= m_full <= m` and `j0 + 4 <= p_full <= p` by the
                // tile loops' construction, and `kb <= k_end <= n` — exactly the
                // microkernel's `# Safety` bounds contract.
                unsafe {
                    microkernel_8x4(a, b, c, m, n, i0, j0, kb, k_end);
                }
            }
        }

        // Bottom edge: rows m_full..m, cols 0..p_full
        let mut i0 = m_full;
        while i0 + 4 <= m {
            for jb in 0..p_full / NR {
                let j0 = jb * NR;
                // SAFETY: the loop condition guarantees `i0 + 4 <= m`, the `jb` loop
                // gives `j0 + 4 <= p_full <= p`, and `kb <= k_end <= n` — exactly the
                // microkernel's `# Safety` bounds contract.
                unsafe {
                    microkernel_4x4(a, b, c, m, n, i0, j0, kb, k_end);
                }
            }
            i0 += 4;
        }
        // Scalar tail for remaining rows (0-3)
        if i0 < m {
            for j in 0..p_full {
                for k in kb..k_end {
                    let b_kj = b[j * n + k];
                    let a_col = k * m;
                    let c_col = j * m;
                    for i in i0..m {
                        c[c_col + i] += a[a_col + i] * b_kj;
                    }
                }
            }
        }

        // Right edge: cols p_full..p, all rows (SIMD j-k-i on inner loop)
        let i_simd = m / 4;
        let i_tail = i_simd * 4;
        for j in p_full..p {
            for k in kb..k_end {
                let b_kj = b[j * n + k];
                let a_col = k * m;
                let c_col = j * m;
                // SAFETY: the broadcast touches no memory. Each iteration loads and
                // stores one 4-lane vector at `offset = i·4` with `i < i_simd = m / 4`,
                // so `offset + 4 <= m`: every access stays inside column `k` of `a`
                // (`a_col = k·m`, `k < n`) and column `j` of `c` (`c_col = j·m`, `j < p`).
                unsafe {
                    let vb = vdupq_n_f32(b_kj);
                    for i in 0..i_simd {
                        let offset = i * 4;
                        let vc = vld1q_f32(c.as_ptr().add(c_col + offset));
                        let va = vld1q_f32(a.as_ptr().add(a_col + offset));
                        vst1q_f32(c.as_mut_ptr().add(c_col + offset), vfmaq_f32(vc, va, vb));
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

/// Register-blocked 8×4 micro-kernel: accumulates C[i0..i0+8, j0..j0+4] in
/// 8 NEON registers across a k-block, writing C only once per block.
///
/// # Safety
///
/// With `a` an `m×n`, `b` an `n×p` and `c` an `m×p` column-major matrix — so
/// `a.len() == m * n`, `b.len() == n * p` and `c.len() == m * p` — the caller
/// must guarantee that the tile and k-range lie inside them:
///
/// - `i0 + 8 <= m`, so the tile's 8 rows are within `a`'s and `c`'s columns;
/// - `j0 + 4 <= p`, so the tile's 4 columns are within `b` and `c`;
/// - `k_start <= k_end <= n`, so every `k` indexes a real column of `a` / row of `b`.
///
/// Every load and store below is then in bounds. NEON is unconditionally available on
/// `aarch64`, which the module's `#[cfg(target_arch = "aarch64")]` gate guarantees.
#[inline(always)]
unsafe fn microkernel_8x4(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    i0: usize,
    j0: usize,
    k_start: usize,
    k_end: usize,
) {
    // SAFETY: the caller upholds the `# Safety` contract above, which puts
    // every pointer offset below in bounds of `a`, `b` and `c`; the
    // broadcasts and vector arithmetic touch no memory.
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // 8 accumulator registers: 2 vectors × 4 columns
        let mut acc00 = vdupq_n_f32(0.0);
        let mut acc10 = vdupq_n_f32(0.0);
        let mut acc01 = vdupq_n_f32(0.0);
        let mut acc11 = vdupq_n_f32(0.0);
        let mut acc02 = vdupq_n_f32(0.0);
        let mut acc12 = vdupq_n_f32(0.0);
        let mut acc03 = vdupq_n_f32(0.0);
        let mut acc13 = vdupq_n_f32(0.0);

        for k in k_start..k_end {
            let a_off = k * m + i0;
            let a0 = vld1q_f32(a_ptr.add(a_off));
            let a1 = vld1q_f32(a_ptr.add(a_off + 4));

            let b0 = vdupq_n_f32(*b_ptr.add(j0 * n + k));
            acc00 = vfmaq_f32(acc00, a0, b0);
            acc10 = vfmaq_f32(acc10, a1, b0);

            let b1 = vdupq_n_f32(*b_ptr.add((j0 + 1) * n + k));
            acc01 = vfmaq_f32(acc01, a0, b1);
            acc11 = vfmaq_f32(acc11, a1, b1);

            let b2 = vdupq_n_f32(*b_ptr.add((j0 + 2) * n + k));
            acc02 = vfmaq_f32(acc02, a0, b2);
            acc12 = vfmaq_f32(acc12, a1, b2);

            let b3 = vdupq_n_f32(*b_ptr.add((j0 + 3) * n + k));
            acc03 = vfmaq_f32(acc03, a0, b3);
            acc13 = vfmaq_f32(acc13, a1, b3);
        }

        // Write back: C += acc
        let c_ptr = c.as_mut_ptr();

        let off0 = j0 * m + i0;
        vst1q_f32(
            c_ptr.add(off0),
            vaddq_f32(vld1q_f32(c_ptr.add(off0)), acc00),
        );
        vst1q_f32(
            c_ptr.add(off0 + 4),
            vaddq_f32(vld1q_f32(c_ptr.add(off0 + 4)), acc10),
        );

        let off1 = (j0 + 1) * m + i0;
        vst1q_f32(
            c_ptr.add(off1),
            vaddq_f32(vld1q_f32(c_ptr.add(off1)), acc01),
        );
        vst1q_f32(
            c_ptr.add(off1 + 4),
            vaddq_f32(vld1q_f32(c_ptr.add(off1 + 4)), acc11),
        );

        let off2 = (j0 + 2) * m + i0;
        vst1q_f32(
            c_ptr.add(off2),
            vaddq_f32(vld1q_f32(c_ptr.add(off2)), acc02),
        );
        vst1q_f32(
            c_ptr.add(off2 + 4),
            vaddq_f32(vld1q_f32(c_ptr.add(off2 + 4)), acc12),
        );

        let off3 = (j0 + 3) * m + i0;
        vst1q_f32(
            c_ptr.add(off3),
            vaddq_f32(vld1q_f32(c_ptr.add(off3)), acc03),
        );
        vst1q_f32(
            c_ptr.add(off3 + 4),
            vaddq_f32(vld1q_f32(c_ptr.add(off3 + 4)), acc13),
        );
    }
}

/// Register-blocked 4×4 mini-kernel for bottom-edge rows (1 NEON f32 register per col).
///
/// # Safety
///
/// With `a` an `m×n`, `b` an `n×p` and `c` an `m×p` column-major matrix — so
/// `a.len() == m * n`, `b.len() == n * p` and `c.len() == m * p` — the caller
/// must guarantee that the tile and k-range lie inside them:
///
/// - `i0 + 4 <= m`, so the tile's 4 rows are within `a`'s and `c`'s columns;
/// - `j0 + 4 <= p`, so the tile's 4 columns are within `b` and `c`;
/// - `k_start <= k_end <= n`, so every `k` indexes a real column of `a` / row of `b`.
///
/// Every load and store below is then in bounds. NEON is unconditionally available on
/// `aarch64`, which the module's `#[cfg(target_arch = "aarch64")]` gate guarantees.
#[inline(always)]
unsafe fn microkernel_4x4(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    i0: usize,
    j0: usize,
    k_start: usize,
    k_end: usize,
) {
    // SAFETY: the caller upholds the `# Safety` contract above, which puts
    // every pointer offset below in bounds of `a`, `b` and `c`; the
    // broadcasts and vector arithmetic touch no memory.
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        for k in k_start..k_end {
            let a0 = vld1q_f32(a_ptr.add(k * m + i0));

            acc0 = vfmaq_f32(acc0, a0, vdupq_n_f32(*b_ptr.add(j0 * n + k)));
            acc1 = vfmaq_f32(acc1, a0, vdupq_n_f32(*b_ptr.add((j0 + 1) * n + k)));
            acc2 = vfmaq_f32(acc2, a0, vdupq_n_f32(*b_ptr.add((j0 + 2) * n + k)));
            acc3 = vfmaq_f32(acc3, a0, vdupq_n_f32(*b_ptr.add((j0 + 3) * n + k)));
        }

        let c_ptr = c.as_mut_ptr();
        let off0 = j0 * m + i0;
        vst1q_f32(c_ptr.add(off0), vaddq_f32(vld1q_f32(c_ptr.add(off0)), acc0));
        let off1 = (j0 + 1) * m + i0;
        vst1q_f32(c_ptr.add(off1), vaddq_f32(vld1q_f32(c_ptr.add(off1)), acc1));
        let off2 = (j0 + 2) * m + i0;
        vst1q_f32(c_ptr.add(off2), vaddq_f32(vld1q_f32(c_ptr.add(off2)), acc2));
        let off3 = (j0 + 3) * m + i0;
        vst1q_f32(c_ptr.add(off3), vaddq_f32(vld1q_f32(c_ptr.add(off3)), acc3));
    }
}

// Element-wise add/sub/scale and AXPY kernels are generated from the shared
// macros in `super` (identical across ISAs bar width + intrinsic names).
simd_elementwise_kernels!(
    f32,
    4,
    vld1q_f32,
    vst1q_f32,
    vaddq_f32,
    vsubq_f32,
    vmulq_f32,
    vdupq_n_f32
);
simd_fft_butterfly_kernel!(f32, 4, vld1q_f32, vst1q_f32, vaddq_f32, vsubq_f32, vmulq_f32);
simd_fft_butterfly4_kernel!(f32, 4, vld1q_f32, vst1q_f32, vaddq_f32, vsubq_f32, vmulq_f32);
simd_axpy_kernels_fma!(
    f32,
    4,
    vld1q_f32,
    vst1q_f32,
    vfmaq_f32,
    vfmsq_f32,
    vdupq_n_f32
);
simd_conv1d_kernel_fma!(f32, 4, vld1q_f32, vst1q_f32, vfmaq_f32, vdupq_n_f32);
