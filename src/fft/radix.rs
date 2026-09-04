//! Iterative radix-2 Cooley–Tukey butterfly passes (size-generic core).
//!
//! The core loops operate on a runtime-length `&mut [Complex<T>]` so a single
//! code path serves every fixed size `N` — the const-generic entry points in
//! [`super::fixed`] only add the compile-time precondition checks, avoiding a
//! full monomorphized transform per `N` (which would bloat `.text` on embedded).

use num_complex::Complex;

use super::cast;
use crate::traits::FloatScalar;

/// In-place bit-reversal permutation of a power-of-two-length buffer.
pub(crate) fn bit_reverse<T: Copy>(buf: &mut [Complex<T>]) {
    let n = buf.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
}

/// Radix-2 transform using a precomputed twiddle table.
///
/// `table` holds `w[j] = exp(-2πi j / n)` for `j = 0..n/2`. `inverse` conjugates
/// each twiddle and scales the result by `1/n`.
pub(crate) fn radix2_table<T: FloatScalar>(
    buf: &mut [Complex<T>],
    table: &[Complex<T>],
    inverse: bool,
) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());
    debug_assert!(n <= 1 || table.len() >= n / 2);
    bit_reverse(buf);

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = n / len; // index stride into the length-n/2 table
        let mut base = 0;
        while base < n {
            for k in 0..half {
                let mut tw = table[k * step];
                if inverse {
                    tw = tw.conj();
                }
                let u = buf[base + k];
                let v = buf[base + k + half] * tw;
                buf[base + k] = u + v;
                buf[base + k + half] = u - v;
            }
            base += len;
        }
        len <<= 1;
    }

    if inverse {
        scale_inverse(buf);
    }
}

/// Radix-2 transform generating stage twiddles inline (O(1) extra storage).
///
/// Each stage computes its principal root once via `sin`/`cos`, then advances
/// the per-butterfly twiddle by complex multiplication. Lower persistent memory
/// than [`radix2_table`] (no table held between calls), at the cost of `sin`/
/// `cos` per stage and slightly more accumulated rounding for large `n`.
pub(crate) fn radix2_inline<T: FloatScalar>(buf: &mut [Complex<T>], inverse: bool) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());
    bit_reverse(buf);

    let two_pi = cast::<T>(core::f64::consts::TAU);
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        // Principal root for this stage: exp(∓2πi / len) (− forward, + inverse).
        let theta = {
            let m = two_pi / cast::<T>(len as f64);
            if inverse {
                m
            } else {
                -m
            }
        };
        let wlen = Complex::new(theta.cos(), theta.sin());
        let mut base = 0;
        while base < n {
            let mut w = Complex::new(T::one(), T::zero());
            for k in 0..half {
                let u = buf[base + k];
                let v = buf[base + k + half] * w;
                buf[base + k] = u + v;
                buf[base + k + half] = u - v;
                w = w * wlen;
            }
            base += len;
        }
        len <<= 1;
    }

    if inverse {
        scale_inverse(buf);
    }
}

/// Normalize an inverse transform by `1/n`.
fn scale_inverse<T: FloatScalar>(buf: &mut [Complex<T>]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    let inv_n = T::one() / cast::<T>(n as f64);
    for x in buf.iter_mut() {
        x.re = x.re * inv_n;
        x.im = x.im * inv_n;
    }
}
