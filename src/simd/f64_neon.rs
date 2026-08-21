//! NEON-accelerated f64 kernels for aarch64.
//!
//! NEON provides 128-bit registers → 2×f64 lanes.

use core::arch::aarch64::*;

/// Dot product of two f64 slices using NEON.
///
/// Uses 4 independent accumulators (8 f64 per iteration) to hide
/// FMA latency (~4 cycles on Apple Silicon).
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    // SAFETY: register broadcasts of zero; they touch no memory.
    let (mut acc0, mut acc1, mut acc2, mut acc3) = unsafe {
        (
            vdupq_n_f64(0.0),
            vdupq_n_f64(0.0),
            vdupq_n_f64(0.0),
            vdupq_n_f64(0.0),
        )
    };

    // 4 accumulators × 2 lanes = 8 elements per iteration.
    let mut ai = a.chunks_exact(8);
    let mut bi = b.chunks_exact(8);
    for (ac, bc) in (&mut ai).zip(&mut bi) {
        // SAFETY: `chunks_exact(8)` yields chunks of exactly 8 `f64`, so the
        // four 2-lane loads at offsets 0, 2, 4 and 6 cover each chunk exactly.
        unsafe {
            let (ap, bp) = (ac.as_ptr(), bc.as_ptr());
            acc0 = vfmaq_f64(acc0, vld1q_f64(ap), vld1q_f64(bp));
            acc1 = vfmaq_f64(acc1, vld1q_f64(ap.add(2)), vld1q_f64(bp.add(2)));
            acc2 = vfmaq_f64(acc2, vld1q_f64(ap.add(4)), vld1q_f64(bp.add(4)));
            acc3 = vfmaq_f64(acc3, vld1q_f64(ap.add(6)), vld1q_f64(bp.add(6)));
        }
    }

    // SAFETY: register arithmetic only — no memory is touched.
    let mut sum = unsafe {
        let s01 = vaddq_f64(acc0, acc1);
        let s23 = vaddq_f64(acc2, acc3);
        vaddvq_f64(vaddq_f64(s01, s23))
    };

    // Remainder: up to 7 elements — 2-wide vectors first, then scalar.
    // SAFETY: register broadcast of zero.
    let mut acc_rem = unsafe { vdupq_n_f64(0.0) };
    let mut ar = ai.remainder().chunks_exact(2);
    let mut br = bi.remainder().chunks_exact(2);
    for (ac, bc) in (&mut ar).zip(&mut br) {
        // SAFETY: each chunk is exactly 2 `f64` — one vector load each.
        unsafe {
            acc_rem = vfmaq_f64(acc_rem, vld1q_f64(ac.as_ptr()), vld1q_f64(bc.as_ptr()));
        }
    }
    // SAFETY: register arithmetic only.
    sum += unsafe { vaddvq_f64(acc_rem) };

    for (&x, &y) in ar.remainder().iter().zip(br.remainder()) {
        sum += x * y;
    }
    sum
}

/// Matrix multiply C += A * B using NEON with register-blocked micro-kernel.
///
/// Uses an MR×NR (8×4) register-blocked micro-kernel with k-blocking (KC=256).
/// For large matrices (n > 32), A and B panels are packed into contiguous
/// stack buffers to eliminate TLB misses and maximize cache line utilization.
/// The 8×4 tile uses 16 accumulator registers (4 NEON f64 vectors × 4 columns),
/// maximizing register utilization on aarch64's 32-register file. Technique
/// inspired by nano-gemm (Sarah Quinones, <https://github.com/sarah-quinones/nano-gemm>).
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
pub fn matmul(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, p: usize) {
    assert_eq!(a.len(), m * n, "matmul: a.len() != m*n");
    assert_eq!(b.len(), n * p, "matmul: b.len() != n*p");
    assert_eq!(c.len(), m * p, "matmul: c.len() != m*p");

    const MR: usize = 8; // 4 NEON registers × 2 f64 lanes
    const NR: usize = 4;
    const KC: usize = 256;

    // For large matrices, use panel packing for cache efficiency
    if n > 64 {
        matmul_packed(a, b, c, m, n, p);
        return;
    }

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
        while i0 + 2 <= m {
            for jb in 0..p_full / NR {
                let j0 = jb * NR;
                // SAFETY: the loop condition guarantees `i0 + 2 <= m`, the `jb` loop
                // gives `j0 + 4 <= p_full <= p`, and `kb <= k_end <= n` — exactly the
                // microkernel's `# Safety` bounds contract.
                unsafe {
                    microkernel_2x4(a, b, c, m, n, i0, j0, kb, k_end);
                }
            }
            i0 += 2;
        }

        // Scalar tail: any single remaining row
        if i0 < m {
            for j in 0..p_full {
                for k in kb..k_end {
                    c[j * m + i0] += a[k * m + i0] * b[j * n + k];
                }
            }
        }

        // Right edge: cols p_full..p, all rows (SIMD j-k-i on inner loop)
        let i_simd = m / 2;
        let i_tail = i_simd * 2;
        for j in p_full..p {
            for k in kb..k_end {
                let b_kj = b[j * n + k];
                let a_col = k * m;
                let c_col = j * m;
                // SAFETY: the broadcast touches no memory. Each iteration loads and
                // stores one 2-lane vector at `offset = i·2` with `i < i_simd = m / 2`,
                // so `offset + 2 <= m`: every access stays inside column `k` of `a`
                // (`a_col = k·m`, `k < n`) and column `j` of `c` (`c_col = j·m`, `j < p`).
                unsafe {
                    let vb = vdupq_n_f64(b_kj);
                    for i in 0..i_simd {
                        let offset = i * 2;
                        let vc = vld1q_f64(c.as_ptr().add(c_col + offset));
                        let va = vld1q_f64(a.as_ptr().add(a_col + offset));
                        vst1q_f64(c.as_mut_ptr().add(c_col + offset), vfmaq_f64(vc, va, vb));
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

/// Panel-packed matmul for large matrices.
///
/// Packs B into NR-wide contiguous panels for sequential memory access.
/// B panel is packed once per (k-block, j-strip) and reused across all i-strips.
/// A is read directly from column-major storage (already contiguous within columns).
#[inline(never)]
fn matmul_packed(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, p: usize) {
    const MR: usize = 8;
    const NR: usize = 4;
    const KC: usize = 256;

    // Stack buffer for packed B panel: KC × NR = 256 × 4 = 1024 doubles = 8 KB
    let mut b_pack = [0.0f64; KC * NR];

    let m_full = (m / MR) * MR;
    let p_full = (p / NR) * NR;

    let mut kb = 0;
    while kb < n {
        let k_end = (kb + KC).min(n);
        let k_len = k_end - kb;

        // Process full NR-wide column blocks
        for jb in 0..p_full / NR {
            let j0 = jb * NR;

            // Pack B panel once: B[kb..k_end, j0..j0+NR] → b_pack[kk*NR + jj]
            pack_b(b, &mut b_pack, j0, kb, k_len, n);

            // Full MR-tall row blocks: unpacked A, packed B
            for ib in 0..m_full / MR {
                let i0 = ib * MR;
                // SAFETY: `i0 + 8 <= m_full <= m` and `j0 + 4 <= p_full <= p` by the
                // tile loops' construction, `kb + k_len == k_end <= n`, and `b_pack`
                // holds the `k_len × 4` panel `pack_b` wrote for this `(j0, kb)` —
                // exactly the microkernel's `# Safety` bounds contract.
                unsafe {
                    microkernel_8x4_bpacked(a, &b_pack, c, m, i0, j0, kb, k_len);
                }
            }

            // Bottom edge rows
            let mut i0 = m_full;
            while i0 + 4 <= m {
                // SAFETY: the loop condition guarantees `i0 + 4 <= m`, the `jb` loop
                // gives `j0 + 4 <= p_full <= p`, and `kb <= k_end <= n` — exactly the
                // microkernel's `# Safety` bounds contract.
                unsafe {
                    microkernel_4x4(a, b, c, m, n, i0, j0, kb, k_end);
                }
                i0 += 4;
            }
            while i0 + 2 <= m {
                // SAFETY: the loop condition guarantees `i0 + 2 <= m`, the `jb` loop
                // gives `j0 + 4 <= p_full <= p`, and `kb <= k_end <= n` — exactly the
                // microkernel's `# Safety` bounds contract.
                unsafe {
                    microkernel_2x4(a, b, c, m, n, i0, j0, kb, k_end);
                }
                i0 += 2;
            }
            if i0 < m {
                for jj in 0..NR {
                    for k in kb..k_end {
                        c[(j0 + jj) * m + i0] += a[k * m + i0] * b[(j0 + jj) * n + k];
                    }
                }
            }
        }

        // Right edge: cols p_full..p (unpacked fallback)
        let i_simd = m / 2;
        let i_tail = i_simd * 2;
        for j in p_full..p {
            for k in kb..k_end {
                let b_kj = b[j * n + k];
                let a_col = k * m;
                let c_col = j * m;
                // SAFETY: the broadcast touches no memory. Each iteration loads and
                // stores one 2-lane vector at `offset = i·2` with `i < i_simd = m / 2`,
                // so `offset + 2 <= m`: every access stays inside column `k` of `a`
                // (`a_col = k·m`, `k < n`) and column `j` of `c` (`c_col = j·m`, `j < p`).
                unsafe {
                    let vb = vdupq_n_f64(b_kj);
                    for i in 0..i_simd {
                        let offset = i * 2;
                        let vc = vld1q_f64(c.as_ptr().add(c_col + offset));
                        let va = vld1q_f64(a.as_ptr().add(a_col + offset));
                        vst1q_f64(c.as_mut_ptr().add(c_col + offset), vfmaq_f64(vc, va, vb));
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

/// Pack B panel: B[kb..kb+k_len, j0..j0+NR] → b_pack[kk * NR + jj]
/// Sequential layout makes micro-kernel B loads contiguous.
#[inline(always)]
fn pack_b(b: &[f64], b_pack: &mut [f64], j0: usize, kb: usize, k_len: usize, n: usize) {
    for kk in 0..k_len {
        let k = kb + kk;
        let dst = kk * 4; // NR = 4
        b_pack[dst] = b[j0 * n + k];
        b_pack[dst + 1] = b[(j0 + 1) * n + k];
        b_pack[dst + 2] = b[(j0 + 2) * n + k];
        b_pack[dst + 3] = b[(j0 + 3) * n + k];
    }
}

/// 8×4 micro-kernel with packed B, unpacked A.
/// B is read from contiguous b_pack, A from original column-major storage.
///
/// # Safety
///
/// With `a` an `m×n` column-major matrix (`a.len() == m * n`) and `c` an `m×p`
/// column-major matrix (`c.len() == m * p`), the caller must guarantee:
///
/// - `i0 + 8 <= m`, so the tile's 8 rows are within `a`'s and `c`'s columns;
/// - `j0 + 4 <= p`, so the tile's 4 columns are within `c`;
/// - `kb + k_len <= n`, so every `k` indexes a real column of `a`;
/// - `b_pack.len() >= k_len * 4`, holding the panel `pack_b` wrote for this
///   `(j0, kb, k_len)` — the kernel reads it sequentially, not by `(k, j)`.
///
/// Every load and store below is then in bounds. NEON is unconditionally available on
/// `aarch64`, which the module's `#[cfg(target_arch = "aarch64")]` gate guarantees.
#[inline(always)]
unsafe fn microkernel_8x4_bpacked(
    a: &[f64],
    b_pack: &[f64],
    c: &mut [f64],
    m: usize,
    i0: usize,
    j0: usize,
    kb: usize,
    k_len: usize,
) {
    // SAFETY: the caller upholds the `# Safety` contract above, which puts
    // every pointer offset below in bounds of `a`, `b_pack` and `c`; the
    // broadcasts and vector arithmetic touch no memory.
    unsafe {
        let ap = a.as_ptr();
        let bp = b_pack.as_ptr();

        let mut acc00 = vdupq_n_f64(0.0);
        let mut acc10 = vdupq_n_f64(0.0);
        let mut acc20 = vdupq_n_f64(0.0);
        let mut acc30 = vdupq_n_f64(0.0);
        let mut acc01 = vdupq_n_f64(0.0);
        let mut acc11 = vdupq_n_f64(0.0);
        let mut acc21 = vdupq_n_f64(0.0);
        let mut acc31 = vdupq_n_f64(0.0);
        let mut acc02 = vdupq_n_f64(0.0);
        let mut acc12 = vdupq_n_f64(0.0);
        let mut acc22 = vdupq_n_f64(0.0);
        let mut acc32 = vdupq_n_f64(0.0);
        let mut acc03 = vdupq_n_f64(0.0);
        let mut acc13 = vdupq_n_f64(0.0);
        let mut acc23 = vdupq_n_f64(0.0);
        let mut acc33 = vdupq_n_f64(0.0);

        for kk in 0..k_len {
            let a_off = (kb + kk) * m + i0;
            let a0 = vld1q_f64(ap.add(a_off));
            let a1 = vld1q_f64(ap.add(a_off + 2));
            let a2 = vld1q_f64(ap.add(a_off + 4));
            let a3 = vld1q_f64(ap.add(a_off + 6));

            let b_off = kk * 4; // NR = 4, contiguous in b_pack
            let b0 = vdupq_n_f64(*bp.add(b_off));
            acc00 = vfmaq_f64(acc00, a0, b0);
            acc10 = vfmaq_f64(acc10, a1, b0);
            acc20 = vfmaq_f64(acc20, a2, b0);
            acc30 = vfmaq_f64(acc30, a3, b0);

            let b1 = vdupq_n_f64(*bp.add(b_off + 1));
            acc01 = vfmaq_f64(acc01, a0, b1);
            acc11 = vfmaq_f64(acc11, a1, b1);
            acc21 = vfmaq_f64(acc21, a2, b1);
            acc31 = vfmaq_f64(acc31, a3, b1);

            let b2 = vdupq_n_f64(*bp.add(b_off + 2));
            acc02 = vfmaq_f64(acc02, a0, b2);
            acc12 = vfmaq_f64(acc12, a1, b2);
            acc22 = vfmaq_f64(acc22, a2, b2);
            acc32 = vfmaq_f64(acc32, a3, b2);

            let b3 = vdupq_n_f64(*bp.add(b_off + 3));
            acc03 = vfmaq_f64(acc03, a0, b3);
            acc13 = vfmaq_f64(acc13, a1, b3);
            acc23 = vfmaq_f64(acc23, a2, b3);
            acc33 = vfmaq_f64(acc33, a3, b3);
        }

        // Write back: C += acc
        let c_ptr = c.as_mut_ptr();

        let off0 = j0 * m + i0;
        vst1q_f64(
            c_ptr.add(off0),
            vaddq_f64(vld1q_f64(c_ptr.add(off0)), acc00),
        );
        vst1q_f64(
            c_ptr.add(off0 + 2),
            vaddq_f64(vld1q_f64(c_ptr.add(off0 + 2)), acc10),
        );
        vst1q_f64(
            c_ptr.add(off0 + 4),
            vaddq_f64(vld1q_f64(c_ptr.add(off0 + 4)), acc20),
        );
        vst1q_f64(
            c_ptr.add(off0 + 6),
            vaddq_f64(vld1q_f64(c_ptr.add(off0 + 6)), acc30),
        );

        let off1 = (j0 + 1) * m + i0;
        vst1q_f64(
            c_ptr.add(off1),
            vaddq_f64(vld1q_f64(c_ptr.add(off1)), acc01),
        );
        vst1q_f64(
            c_ptr.add(off1 + 2),
            vaddq_f64(vld1q_f64(c_ptr.add(off1 + 2)), acc11),
        );
        vst1q_f64(
            c_ptr.add(off1 + 4),
            vaddq_f64(vld1q_f64(c_ptr.add(off1 + 4)), acc21),
        );
        vst1q_f64(
            c_ptr.add(off1 + 6),
            vaddq_f64(vld1q_f64(c_ptr.add(off1 + 6)), acc31),
        );

        let off2 = (j0 + 2) * m + i0;
        vst1q_f64(
            c_ptr.add(off2),
            vaddq_f64(vld1q_f64(c_ptr.add(off2)), acc02),
        );
        vst1q_f64(
            c_ptr.add(off2 + 2),
            vaddq_f64(vld1q_f64(c_ptr.add(off2 + 2)), acc12),
        );
        vst1q_f64(
            c_ptr.add(off2 + 4),
            vaddq_f64(vld1q_f64(c_ptr.add(off2 + 4)), acc22),
        );
        vst1q_f64(
            c_ptr.add(off2 + 6),
            vaddq_f64(vld1q_f64(c_ptr.add(off2 + 6)), acc32),
        );

        let off3 = (j0 + 3) * m + i0;
        vst1q_f64(
            c_ptr.add(off3),
            vaddq_f64(vld1q_f64(c_ptr.add(off3)), acc03),
        );
        vst1q_f64(
            c_ptr.add(off3 + 2),
            vaddq_f64(vld1q_f64(c_ptr.add(off3 + 2)), acc13),
        );
        vst1q_f64(
            c_ptr.add(off3 + 4),
            vaddq_f64(vld1q_f64(c_ptr.add(off3 + 4)), acc23),
        );
        vst1q_f64(
            c_ptr.add(off3 + 6),
            vaddq_f64(vld1q_f64(c_ptr.add(off3 + 6)), acc33),
        );
    }
}

/// Register-blocked 8×4 micro-kernel: accumulates C[i0..i0+8, j0..j0+4] in
/// 16 NEON registers across a k-block, writing C only once per block.
/// Uses 4 NEON f64 vectors (8 elements) × 4 columns = 16 accumulators.
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
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
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

        // 16 accumulator registers: 4 vectors × 4 columns
        let mut acc00 = vdupq_n_f64(0.0);
        let mut acc10 = vdupq_n_f64(0.0);
        let mut acc20 = vdupq_n_f64(0.0);
        let mut acc30 = vdupq_n_f64(0.0);
        let mut acc01 = vdupq_n_f64(0.0);
        let mut acc11 = vdupq_n_f64(0.0);
        let mut acc21 = vdupq_n_f64(0.0);
        let mut acc31 = vdupq_n_f64(0.0);
        let mut acc02 = vdupq_n_f64(0.0);
        let mut acc12 = vdupq_n_f64(0.0);
        let mut acc22 = vdupq_n_f64(0.0);
        let mut acc32 = vdupq_n_f64(0.0);
        let mut acc03 = vdupq_n_f64(0.0);
        let mut acc13 = vdupq_n_f64(0.0);
        let mut acc23 = vdupq_n_f64(0.0);
        let mut acc33 = vdupq_n_f64(0.0);

        for k in k_start..k_end {
            let a_off = k * m + i0;
            let a0 = vld1q_f64(a_ptr.add(a_off));
            let a1 = vld1q_f64(a_ptr.add(a_off + 2));
            let a2 = vld1q_f64(a_ptr.add(a_off + 4));
            let a3 = vld1q_f64(a_ptr.add(a_off + 6));

            let b0 = vdupq_n_f64(*b_ptr.add(j0 * n + k));
            acc00 = vfmaq_f64(acc00, a0, b0);
            acc10 = vfmaq_f64(acc10, a1, b0);
            acc20 = vfmaq_f64(acc20, a2, b0);
            acc30 = vfmaq_f64(acc30, a3, b0);

            let b1 = vdupq_n_f64(*b_ptr.add((j0 + 1) * n + k));
            acc01 = vfmaq_f64(acc01, a0, b1);
            acc11 = vfmaq_f64(acc11, a1, b1);
            acc21 = vfmaq_f64(acc21, a2, b1);
            acc31 = vfmaq_f64(acc31, a3, b1);

            let b2 = vdupq_n_f64(*b_ptr.add((j0 + 2) * n + k));
            acc02 = vfmaq_f64(acc02, a0, b2);
            acc12 = vfmaq_f64(acc12, a1, b2);
            acc22 = vfmaq_f64(acc22, a2, b2);
            acc32 = vfmaq_f64(acc32, a3, b2);

            let b3 = vdupq_n_f64(*b_ptr.add((j0 + 3) * n + k));
            acc03 = vfmaq_f64(acc03, a0, b3);
            acc13 = vfmaq_f64(acc13, a1, b3);
            acc23 = vfmaq_f64(acc23, a2, b3);
            acc33 = vfmaq_f64(acc33, a3, b3);
        }

        // Write back: C += acc
        let c_ptr = c.as_mut_ptr();

        let off0 = j0 * m + i0;
        vst1q_f64(
            c_ptr.add(off0),
            vaddq_f64(vld1q_f64(c_ptr.add(off0)), acc00),
        );
        vst1q_f64(
            c_ptr.add(off0 + 2),
            vaddq_f64(vld1q_f64(c_ptr.add(off0 + 2)), acc10),
        );
        vst1q_f64(
            c_ptr.add(off0 + 4),
            vaddq_f64(vld1q_f64(c_ptr.add(off0 + 4)), acc20),
        );
        vst1q_f64(
            c_ptr.add(off0 + 6),
            vaddq_f64(vld1q_f64(c_ptr.add(off0 + 6)), acc30),
        );

        let off1 = (j0 + 1) * m + i0;
        vst1q_f64(
            c_ptr.add(off1),
            vaddq_f64(vld1q_f64(c_ptr.add(off1)), acc01),
        );
        vst1q_f64(
            c_ptr.add(off1 + 2),
            vaddq_f64(vld1q_f64(c_ptr.add(off1 + 2)), acc11),
        );
        vst1q_f64(
            c_ptr.add(off1 + 4),
            vaddq_f64(vld1q_f64(c_ptr.add(off1 + 4)), acc21),
        );
        vst1q_f64(
            c_ptr.add(off1 + 6),
            vaddq_f64(vld1q_f64(c_ptr.add(off1 + 6)), acc31),
        );

        let off2 = (j0 + 2) * m + i0;
        vst1q_f64(
            c_ptr.add(off2),
            vaddq_f64(vld1q_f64(c_ptr.add(off2)), acc02),
        );
        vst1q_f64(
            c_ptr.add(off2 + 2),
            vaddq_f64(vld1q_f64(c_ptr.add(off2 + 2)), acc12),
        );
        vst1q_f64(
            c_ptr.add(off2 + 4),
            vaddq_f64(vld1q_f64(c_ptr.add(off2 + 4)), acc22),
        );
        vst1q_f64(
            c_ptr.add(off2 + 6),
            vaddq_f64(vld1q_f64(c_ptr.add(off2 + 6)), acc32),
        );

        let off3 = (j0 + 3) * m + i0;
        vst1q_f64(
            c_ptr.add(off3),
            vaddq_f64(vld1q_f64(c_ptr.add(off3)), acc03),
        );
        vst1q_f64(
            c_ptr.add(off3 + 2),
            vaddq_f64(vld1q_f64(c_ptr.add(off3 + 2)), acc13),
        );
        vst1q_f64(
            c_ptr.add(off3 + 4),
            vaddq_f64(vld1q_f64(c_ptr.add(off3 + 4)), acc23),
        );
        vst1q_f64(
            c_ptr.add(off3 + 6),
            vaddq_f64(vld1q_f64(c_ptr.add(off3 + 6)), acc33),
        );
    }
}

/// Register-blocked 4×4 micro-kernel: accumulates C[i0..i0+4, j0..j0+4] in
/// 8 NEON registers across a k-block, writing C only once per block.
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
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
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
        let mut acc00 = vdupq_n_f64(0.0);
        let mut acc10 = vdupq_n_f64(0.0);
        let mut acc01 = vdupq_n_f64(0.0);
        let mut acc11 = vdupq_n_f64(0.0);
        let mut acc02 = vdupq_n_f64(0.0);
        let mut acc12 = vdupq_n_f64(0.0);
        let mut acc03 = vdupq_n_f64(0.0);
        let mut acc13 = vdupq_n_f64(0.0);

        for k in k_start..k_end {
            let a_off = k * m + i0;
            let a0 = vld1q_f64(a_ptr.add(a_off));
            let a1 = vld1q_f64(a_ptr.add(a_off + 2));

            let b0 = vdupq_n_f64(*b_ptr.add(j0 * n + k));
            acc00 = vfmaq_f64(acc00, a0, b0);
            acc10 = vfmaq_f64(acc10, a1, b0);

            let b1 = vdupq_n_f64(*b_ptr.add((j0 + 1) * n + k));
            acc01 = vfmaq_f64(acc01, a0, b1);
            acc11 = vfmaq_f64(acc11, a1, b1);

            let b2 = vdupq_n_f64(*b_ptr.add((j0 + 2) * n + k));
            acc02 = vfmaq_f64(acc02, a0, b2);
            acc12 = vfmaq_f64(acc12, a1, b2);

            let b3 = vdupq_n_f64(*b_ptr.add((j0 + 3) * n + k));
            acc03 = vfmaq_f64(acc03, a0, b3);
            acc13 = vfmaq_f64(acc13, a1, b3);
        }

        // Write back: C += acc
        let c_ptr = c.as_mut_ptr();

        let off0 = j0 * m + i0;
        vst1q_f64(
            c_ptr.add(off0),
            vaddq_f64(vld1q_f64(c_ptr.add(off0)), acc00),
        );
        vst1q_f64(
            c_ptr.add(off0 + 2),
            vaddq_f64(vld1q_f64(c_ptr.add(off0 + 2)), acc10),
        );

        let off1 = (j0 + 1) * m + i0;
        vst1q_f64(
            c_ptr.add(off1),
            vaddq_f64(vld1q_f64(c_ptr.add(off1)), acc01),
        );
        vst1q_f64(
            c_ptr.add(off1 + 2),
            vaddq_f64(vld1q_f64(c_ptr.add(off1 + 2)), acc11),
        );

        let off2 = (j0 + 2) * m + i0;
        vst1q_f64(
            c_ptr.add(off2),
            vaddq_f64(vld1q_f64(c_ptr.add(off2)), acc02),
        );
        vst1q_f64(
            c_ptr.add(off2 + 2),
            vaddq_f64(vld1q_f64(c_ptr.add(off2 + 2)), acc12),
        );

        let off3 = (j0 + 3) * m + i0;
        vst1q_f64(
            c_ptr.add(off3),
            vaddq_f64(vld1q_f64(c_ptr.add(off3)), acc03),
        );
        vst1q_f64(
            c_ptr.add(off3 + 2),
            vaddq_f64(vld1q_f64(c_ptr.add(off3 + 2)), acc13),
        );
    }
}

/// Register-blocked 2×4 mini-kernel for bottom-edge rows: accumulates
/// C[i0..i0+2, j0..j0+4] in 4 NEON registers across a k-block.
///
/// # Safety
///
/// With `a` an `m×n`, `b` an `n×p` and `c` an `m×p` column-major matrix — so
/// `a.len() == m * n`, `b.len() == n * p` and `c.len() == m * p` — the caller
/// must guarantee that the tile and k-range lie inside them:
///
/// - `i0 + 2 <= m`, so the tile's 2 rows are within `a`'s and `c`'s columns;
/// - `j0 + 4 <= p`, so the tile's 4 columns are within `b` and `c`;
/// - `k_start <= k_end <= n`, so every `k` indexes a real column of `a` / row of `b`.
///
/// Every load and store below is then in bounds. NEON is unconditionally available on
/// `aarch64`, which the module's `#[cfg(target_arch = "aarch64")]` gate guarantees.
#[inline(always)]
unsafe fn microkernel_2x4(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
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

        // 4 accumulator registers: 1 vector (2 f64) × 4 columns
        let mut acc0 = vdupq_n_f64(0.0);
        let mut acc1 = vdupq_n_f64(0.0);
        let mut acc2 = vdupq_n_f64(0.0);
        let mut acc3 = vdupq_n_f64(0.0);

        for k in k_start..k_end {
            let a0 = vld1q_f64(a_ptr.add(k * m + i0));

            acc0 = vfmaq_f64(acc0, a0, vdupq_n_f64(*b_ptr.add(j0 * n + k)));
            acc1 = vfmaq_f64(acc1, a0, vdupq_n_f64(*b_ptr.add((j0 + 1) * n + k)));
            acc2 = vfmaq_f64(acc2, a0, vdupq_n_f64(*b_ptr.add((j0 + 2) * n + k)));
            acc3 = vfmaq_f64(acc3, a0, vdupq_n_f64(*b_ptr.add((j0 + 3) * n + k)));
        }

        let c_ptr = c.as_mut_ptr();
        let off0 = j0 * m + i0;
        vst1q_f64(c_ptr.add(off0), vaddq_f64(vld1q_f64(c_ptr.add(off0)), acc0));
        let off1 = (j0 + 1) * m + i0;
        vst1q_f64(c_ptr.add(off1), vaddq_f64(vld1q_f64(c_ptr.add(off1)), acc1));
        let off2 = (j0 + 2) * m + i0;
        vst1q_f64(c_ptr.add(off2), vaddq_f64(vld1q_f64(c_ptr.add(off2)), acc2));
        let off3 = (j0 + 3) * m + i0;
        vst1q_f64(c_ptr.add(off3), vaddq_f64(vld1q_f64(c_ptr.add(off3)), acc3));
    }
}

// Element-wise add/sub/scale and AXPY kernels are generated from the shared
// macros in `super` (identical across ISAs bar width + intrinsic names).
simd_elementwise_kernels!(
    f64,
    2,
    vld1q_f64,
    vst1q_f64,
    vaddq_f64,
    vsubq_f64,
    vmulq_f64,
    vdupq_n_f64
);
simd_axpy_kernels_fma!(
    f64,
    2,
    vld1q_f64,
    vst1q_f64,
    vfmaq_f64,
    vfmsq_f64,
    vdupq_n_f64
);
simd_conv1d_kernel_fma!(f64, 2, vld1q_f64, vst1q_f64, vfmaq_f64, vdupq_n_f64);
