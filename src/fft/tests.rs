//! Tests for the fixed-size FFT: correctness vs a naive DFT, inverse
//! round-trips, Parseval's theorem, and table/inline path agreement.

extern crate alloc;
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use super::DynFft;
#[cfg(feature = "alloc")]
use super::DynRealFft;
use super::{fft, fft_inplace, fftshift, ifft, ifft_inplace, ifftshift, irfft, rfft, TwiddleTable};
#[cfg(feature = "alloc")]
use super::{fft_convolve, fft_correlate};
#[cfg(feature = "alloc")]
use super::{fftshift2d, ifftshift2d, DynFft2, DynRealFft2};
#[cfg(feature = "alloc")]
use crate::DynMatrix;
use num_complex::Complex;

/// Naive O(N²) DFT reference. `sign = -1.0` forward, `+1.0` inverse (unnormalized).
fn naive_dft(x: &[Complex<f64>], sign: f64) -> Vec<Complex<f64>> {
    let n = x.len();
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let mut acc = Complex::new(0.0, 0.0);
        for (j, &xj) in x.iter().enumerate() {
            let theta = sign * core::f64::consts::TAU * (k as f64) * (j as f64) / (n as f64);
            acc += xj * Complex::new(theta.cos(), theta.sin());
        }
        out.push(acc);
    }
    out
}

/// Deterministic pseudo-random-ish complex input (no rng dependency).
fn sample_input(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| {
            let re = (i as f64 * 0.7).sin() + 0.3 * (i as f64 * 1.9).cos();
            let im = (i as f64 * 1.3).cos() - 0.5 * (i as f64 * 0.4).sin();
            Complex::new(re, im)
        })
        .collect()
}

fn max_abs_diff(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

macro_rules! check_len {
    ($n:literal) => {{
        let input = sample_input($n);
        let reference = naive_dft(&input, -1.0);

        // inline path vs naive DFT
        let mut buf: [Complex<f64>; $n] = input.clone().try_into().unwrap();
        fft_inplace(&mut buf);
        assert!(
            max_abs_diff(&buf, &reference) < 1e-10,
            "fft_inplace mismatch at N={}",
            $n
        );

        // table path vs naive DFT
        let tw = TwiddleTable::<f64, $n>::new();
        let mut buf2: [Complex<f64>; $n] = input.clone().try_into().unwrap();
        fft(&mut buf2, &tw);
        assert!(
            max_abs_diff(&buf2, &reference) < 1e-10,
            "fft (table) mismatch at N={}",
            $n
        );

        // table and inline paths agree closely
        assert!(
            max_abs_diff(&buf, &buf2) < 1e-12,
            "table vs inline at N={}",
            $n
        );

        // inverse round-trip (inline)
        ifft_inplace(&mut buf);
        assert!(
            max_abs_diff(&buf, &input) < 1e-10,
            "ifft_inplace round-trip at N={}",
            $n
        );

        // inverse round-trip (table)
        ifft(&mut buf2, &tw);
        assert!(
            max_abs_diff(&buf2, &input) < 1e-10,
            "ifft (table) round-trip at N={}",
            $n
        );
    }};
}

#[test]
fn matches_naive_dft_and_round_trips() {
    check_len!(2);
    check_len!(4);
    check_len!(8);
    check_len!(16);
    check_len!(32);
    check_len!(64);
    check_len!(256);
}

#[test]
fn dc_bin_is_sum() {
    let input = sample_input(8);
    let sum: Complex<f64> = input.iter().sum();
    let mut buf: [Complex<f64>; 8] = input.try_into().unwrap();
    fft_inplace(&mut buf);
    assert!((buf[0] - sum).norm() < 1e-12);
}

#[test]
fn length_one_is_identity() {
    let mut buf = [Complex::new(3.0, -2.0)];
    fft_inplace(&mut buf);
    assert_eq!(buf[0], Complex::new(3.0, -2.0));
    ifft_inplace(&mut buf);
    assert_eq!(buf[0], Complex::new(3.0, -2.0));
}

#[test]
fn parseval_energy_conservation() {
    // sum |x|^2 == (1/N) sum |X|^2
    let input = sample_input(64);
    let time_energy: f64 = input.iter().map(|z| z.norm_sqr()).sum();
    let mut buf: [Complex<f64>; 64] = input.try_into().unwrap();
    fft_inplace(&mut buf);
    let freq_energy: f64 = buf.iter().map(|z| z.norm_sqr()).sum::<f64>() / 64.0;
    assert!((time_energy - freq_energy).abs() < 1e-9);
}

#[test]
fn single_frequency_is_a_spike() {
    // x[n] = exp(2πi * 3 n / N) -> single nonzero bin at k = 3.
    const N: usize = 16;
    let mut buf: [Complex<f64>; N] = core::array::from_fn(|n| {
        let theta = core::f64::consts::TAU * 3.0 * (n as f64) / (N as f64);
        Complex::new(theta.cos(), theta.sin())
    });
    fft_inplace(&mut buf);
    for (k, z) in buf.iter().enumerate() {
        if k == 3 {
            assert!((z.norm() - N as f64).abs() < 1e-9);
        } else {
            assert!(z.norm() < 1e-9, "bin {k} should be ~0");
        }
    }
}

#[cfg(feature = "alloc")]
#[test]
fn dynfft_matches_naive_dft_all_lengths() {
    // Power-of-two (radix path), composites, and primes (Bluestein path).
    for &n in &[
        1usize, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17, 31, 64, 100, 1000, 1009,
    ] {
        let input = sample_input(n);
        let reference = naive_dft(&input, -1.0);

        let mut plan = DynFft::<f64>::new(n);
        let mut buf = input.clone();
        plan.forward(&mut buf);
        assert!(
            max_abs_diff(&buf, &reference) < 1e-9,
            "DynFft forward mismatch at n={n}"
        );

        plan.inverse(&mut buf);
        assert!(
            max_abs_diff(&buf, &input) < 1e-9,
            "DynFft round-trip mismatch at n={n}"
        );
    }
}

#[cfg(feature = "alloc")]
#[test]
fn dynfft_simd_path_matches_scalar_reference() {
    // DynFft power-of-two routes through the deinterleaved SIMD butterfly; it
    // must match the interleaved scalar radix core bit-close. Sizes include
    // small stages (half < SIMD width -> exercises the scalar tail) and larger
    // ones (full vector blocks).
    use super::radix::radix2_inline;
    for &n in &[2usize, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
        let input = sample_input(n);

        let mut reference = input.clone();
        radix2_inline(&mut reference, false);

        let mut plan = DynFft::<f64>::new(n);
        let mut simd = input;
        plan.forward(&mut simd);

        assert!(
            max_abs_diff(&reference, &simd) < 1e-11,
            "SIMD butterfly disagrees with scalar reference at n={n}"
        );
    }
}

#[cfg(feature = "alloc")]
#[test]
fn dynfft_f32_simd_correct_vs_high_precision() {
    // f32 SIMD path vs the true spectrum (naive DFT in f64 of the same input).
    // Tolerance is relative to the peak magnitude, sized for f32 accumulation.
    for &n in &[4usize, 8, 16, 64, 256, 1024] {
        let input: Vec<Complex<f32>> = (0..n)
            .map(|i| Complex::new((i as f32 * 0.3).sin(), (i as f32 * 0.9).cos()))
            .collect();
        let promoted: Vec<Complex<f64>> = input
            .iter()
            .map(|z| Complex::new(z.re as f64, z.im as f64))
            .collect();
        let truth = naive_dft(&promoted, -1.0);
        let peak = truth.iter().map(|z| z.norm()).fold(0.0, f64::max).max(1.0);

        let mut plan = DynFft::<f32>::new(n);
        let mut simd = input;
        plan.forward(&mut simd);

        let err = simd
            .iter()
            .zip(&truth)
            .map(|(a, b)| ((a.re as f64 - b.re).powi(2) + (a.im as f64 - b.im).powi(2)).sqrt())
            .fold(0.0, f64::max);
        assert!(err / peak < 1e-5, "f32 SIMD FFT vs truth at n={n}: {err}");
    }
}

#[cfg(feature = "alloc")]
#[test]
fn dynfft_matches_fixed_tier() {
    // The runtime power-of-two path must agree with the fixed-size transform.
    let input = sample_input(64);
    let mut a: [Complex<f64>; 64] = input.clone().try_into().unwrap();
    fft_inplace(&mut a);

    let mut plan = DynFft::<f64>::new(64);
    let mut b = input;
    plan.forward(&mut b);

    assert!(max_abs_diff(&a, &b) < 1e-12);
}

#[cfg(feature = "alloc")]
#[test]
fn dynfft_f32_prime_round_trip() {
    let input: Vec<Complex<f32>> = (0..13)
        .map(|i| Complex::new((i as f32 * 0.5).sin(), (i as f32 * 0.2).cos()))
        .collect();
    let mut plan = DynFft::<f32>::new(13);
    let mut buf = input.clone();
    plan.forward(&mut buf);
    plan.inverse(&mut buf);
    let err = buf
        .iter()
        .zip(&input)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(err < 1e-4, "DynFft f32 prime round-trip error {err}");
}

#[cfg(feature = "alloc")]
#[test]
#[should_panic]
fn dynfft_zero_length_panics() {
    let _ = DynFft::<f64>::new(0);
}

fn real_sample(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (i as f64 * 0.6).sin() + 0.4 * (i as f64 * 1.7).cos() - 0.2 * i as f64)
        .collect()
}

#[test]
fn rfft_matches_complex_fft() {
    macro_rules! check {
        ($n:literal) => {{
            let real = real_sample($n);
            let complexified: Vec<Complex<f64>> =
                real.iter().map(|&x| Complex::new(x, 0.0)).collect();
            let reference = naive_dft(&complexified, -1.0);

            let input: [f64; $n] = real.clone().try_into().unwrap();
            let mut out = [Complex::new(0.0, 0.0); $n / 2 + 1];
            rfft(&input, &mut out);
            // The N/2+1 returned bins must match the full complex FFT.
            for k in 0..=$n / 2 {
                assert!(
                    (out[k] - reference[k]).norm() < 1e-10,
                    "rfft bin {} mismatch at N={}",
                    k,
                    $n
                );
            }

            // Inverse recovers the original real signal.
            let mut recovered = [0.0f64; $n];
            irfft(&out, &mut recovered);
            for i in 0..$n {
                assert!(
                    (recovered[i] - real[i]).abs() < 1e-10,
                    "irfft sample {} mismatch at N={}",
                    i,
                    $n
                );
            }
        }};
    }
    check!(2);
    check!(4);
    check!(8);
    check!(16);
    check!(64);
    check!(256);
}

#[cfg(feature = "alloc")]
#[test]
fn dyn_real_fft_even_and_odd() {
    for &n in &[2usize, 3, 4, 5, 6, 8, 9, 12, 15, 16, 100, 101] {
        let real = real_sample(n);
        let complexified: Vec<Complex<f64>> = real.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let reference = naive_dft(&complexified, -1.0);

        let mut plan = DynRealFft::<f64>::new(n);
        let mut spec = alloc::vec![Complex::new(0.0, 0.0); n / 2 + 1];
        plan.forward(&real, &mut spec);
        for k in 0..=n / 2 {
            assert!(
                (spec[k] - reference[k]).norm() < 1e-9,
                "DynRealFft bin {k} mismatch at n={n}"
            );
        }

        let mut recovered = alloc::vec![0.0f64; n];
        plan.inverse(&spec, &mut recovered);
        for i in 0..n {
            assert!(
                (recovered[i] - real[i]).abs() < 1e-9,
                "DynRealFft sample {i} mismatch at n={n}"
            );
        }
    }
}

#[test]
fn fftshift_round_trip_and_values() {
    // Even length: fftshift == ifftshift, swaps halves.
    let mut v = [0, 1, 2, 3];
    fftshift(&mut v);
    assert_eq!(v, [2, 3, 0, 1]);
    ifftshift(&mut v);
    assert_eq!(v, [0, 1, 2, 3]);

    // Odd length: ifftshift truly inverts fftshift.
    let mut w = [0, 1, 2, 3, 4];
    fftshift(&mut w);
    assert_eq!(w, [3, 4, 0, 1, 2]);
    ifftshift(&mut w);
    assert_eq!(w, [0, 1, 2, 3, 4]);
}

#[cfg(feature = "alloc")]
#[test]
fn fft_convolve_matches_direct() {
    let a = [1.0f64, 2.0, 3.0, 4.0, 5.0];
    let b = [0.5, -1.0, 0.25];

    // Direct linear convolution reference.
    let mut expected = alloc::vec![0.0f64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            expected[i + j] += ai * bj;
        }
    }

    let got = fft_convolve(&a, &b);
    assert_eq!(got.len(), expected.len());
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-10, "convolve {g} vs {e}");
    }
}

#[cfg(feature = "alloc")]
#[test]
fn fft_correlate_matches_reversed_convolution() {
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [1.0, 0.5, -0.5];

    // correlate(a, b) == convolve(a, reverse(b)).
    let brev: alloc::vec::Vec<f64> = b.iter().rev().copied().collect();
    let mut expected = alloc::vec![0.0f64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in brev.iter().enumerate() {
            expected[i + j] += ai * bj;
        }
    }

    let got = fft_correlate(&a, &b);
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-10, "correlate {g} vs {e}");
    }
}

#[cfg(feature = "alloc")]
#[test]
fn fft_convolve_empty() {
    let empty: [f64; 0] = [];
    assert!(fft_convolve(&empty, &[1.0, 2.0]).is_empty());
    assert!(fft_convolve(&[1.0, 2.0], &empty).is_empty());
}

#[test]
fn f32_round_trip() {
    let input: [Complex<f32>; 32] =
        core::array::from_fn(|i| Complex::new((i as f32 * 0.5).sin(), (i as f32 * 0.2).cos()));
    let mut buf = input;
    fft_inplace(&mut buf);
    ifft_inplace(&mut buf);
    let err = buf
        .iter()
        .zip(&input)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(err < 1e-4, "f32 round-trip error {err}");
}

// ─── 2D FFT ────────────────────────────────────────────────────────────

/// Naive O((rows*cols)²) 2D DFT reference on a column-major matrix.
#[cfg(feature = "alloc")]
fn naive_dft2(x: &DynMatrix<Complex<f64>>, sign: f64) -> DynMatrix<Complex<f64>> {
    let (rows, cols) = (x.nrows(), x.ncols());
    let tau = core::f64::consts::TAU;
    DynMatrix::from_fn(rows, cols, |kr, kc| {
        let mut acc = Complex::new(0.0, 0.0);
        for r in 0..rows {
            for c in 0..cols {
                let ang = sign
                    * tau
                    * (kr as f64 * r as f64 / rows as f64 + kc as f64 * c as f64 / cols as f64);
                acc += x[(r, c)] * Complex::new(ang.cos(), ang.sin());
            }
        }
        acc
    })
}

#[cfg(feature = "alloc")]
fn max_err2(a: &DynMatrix<Complex<f64>>, b: &DynMatrix<Complex<f64>>) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

#[cfg(feature = "alloc")]
#[test]
fn dynfft2_matches_naive_and_round_trips() {
    // Mix of power-of-two and non-power-of-two (Bluestein) dimensions.
    for &(rows, cols) in &[(4, 4), (8, 4), (3, 5), (6, 7), (1, 6), (5, 1)] {
        let input = DynMatrix::from_fn(rows, cols, |r, c| {
            Complex::new(
                (r as f64 * 0.7 + c as f64).sin(),
                (r as f64 - c as f64 * 0.3).cos(),
            )
        });
        let reference = naive_dft2(&input, -1.0);

        let mut plan = DynFft2::<f64>::new(rows, cols);
        let mut buf = input.clone();
        plan.forward(&mut buf);
        assert!(
            max_err2(&buf, &reference) < 1e-10,
            "DynFft2 forward mismatch at {rows}x{cols}"
        );

        plan.inverse(&mut buf);
        assert!(
            max_err2(&buf, &input) < 1e-10,
            "DynFft2 round-trip mismatch at {rows}x{cols}"
        );
    }
}

#[cfg(feature = "alloc")]
#[test]
fn dynrealfft2_matches_complex_and_round_trips() {
    for &(rows, cols) in &[(4, 4), (8, 6), (6, 5), (5, 4)] {
        let real = DynMatrix::from_fn(rows, cols, |r, c| {
            (r as f64 * 0.9).sin() + (c as f64 * 0.4).cos() + 0.1 * (r * c) as f64
        });
        // Reference: full complex 2D FFT of the real-valued input.
        let complexified = DynMatrix::from_fn(rows, cols, |r, c| Complex::new(real[(r, c)], 0.0));
        let full = naive_dft2(&complexified, -1.0);

        let mut plan = DynRealFft2::<f64>::new(rows, cols);
        let half = rows / 2 + 1;
        let mut spec = DynMatrix::zeros(half, cols);
        plan.forward(&real, &mut spec);

        // The half-spectrum must equal the top rows/2+1 rows of the full FFT.
        for kr in 0..half {
            for kc in 0..cols {
                assert!(
                    (spec[(kr, kc)] - full[(kr, kc)]).norm() < 1e-10,
                    "rfft2 bin ({kr},{kc}) mismatch at {rows}x{cols}"
                );
            }
        }

        let mut recon = DynMatrix::zeros(rows, cols);
        plan.inverse(&spec, &mut recon);
        let err = recon
            .as_slice()
            .iter()
            .zip(real.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-10, "rfft2 round-trip error {err} at {rows}x{cols}");
    }
}

#[cfg(feature = "alloc")]
#[test]
fn fftshift2d_swaps_quadrants_and_round_trips() {
    // Even dimensions: fftshift2d swaps diagonal quadrants; ifftshift2d undoes it.
    let m = DynMatrix::from_fn(4, 4, |r, c| (r * 10 + c) as f64);
    let mut s = m.clone();
    fftshift2d(&mut s);
    // DC at (0,0) lands at the center (2,2) for a 4x4.
    assert_eq!(s[(2, 2)], m[(0, 0)]);
    ifftshift2d(&mut s);
    assert_eq!(s.as_slice(), m.as_slice());

    // Odd dimension: ifftshift2d is the true inverse of fftshift2d.
    let mo = DynMatrix::from_fn(5, 3, |r, c| (r * 10 + c) as f64);
    let mut so = mo.clone();
    fftshift2d(&mut so);
    ifftshift2d(&mut so);
    assert_eq!(so.as_slice(), mo.as_slice());
}

#[cfg(feature = "alloc")]
#[test]
#[should_panic]
fn dynfft2_zero_dim_panics() {
    let _ = DynFft2::<f64>::new(0, 4);
}

// ─── Scratch-external API ──────────────────────────────────────────────

/// `forward_with` / `inverse_with` through a caller-owned scratch must give
/// exactly the same result as the internal-scratch methods, on both the
/// power-of-two and Bluestein paths.
#[cfg(feature = "alloc")]
#[test]
fn dynfft_with_scratch_matches_internal() {
    for &n in &[1usize, 2, 4, 8, 64, 6, 13, 100] {
        let input = sample_input(n);
        let mut plan = DynFft::<f64>::new(n);
        let mut a = input.clone();
        plan.forward(&mut a);

        let shared = DynFft::<f64>::new(n);
        let mut scratch = shared.make_scratch();
        assert_eq!(scratch.len(), n);
        let mut b = input.clone();
        shared.forward_with(&mut b, &mut scratch);
        assert_eq!(a, b, "forward_with differs from forward at n={n}");

        plan.inverse(&mut a);
        shared.inverse_with(&mut b, &mut scratch);
        assert_eq!(a, b, "inverse_with differs from inverse at n={n}");
        assert!(max_abs_diff(&b, &input) < 1e-10, "round trip at n={n}");
    }
}

#[cfg(feature = "alloc")]
#[test]
#[should_panic(expected = "different transform length")]
fn dynfft_scratch_length_mismatch_panics() {
    let plan = DynFft::<f64>::new(8);
    let other = DynFft::<f64>::new(16);
    let mut wrong = other.make_scratch();
    let mut buf = sample_input(8);
    plan.forward_with(&mut buf, &mut wrong);
}

#[cfg(feature = "alloc")]
#[test]
fn dynrealfft_with_scratch_matches_internal() {
    for &n in &[2usize, 8, 7, 10, 64] {
        let x = real_sample(n);
        let mut plan = DynRealFft::<f64>::new(n);
        let mut spec_a = alloc::vec![Complex::new(0.0, 0.0); n / 2 + 1];
        plan.forward(&x, &mut spec_a);

        let shared = DynRealFft::<f64>::new(n);
        let mut scratch = shared.make_scratch();
        assert_eq!(scratch.len(), n);
        let mut spec_b = alloc::vec![Complex::new(0.0, 0.0); n / 2 + 1];
        shared.forward_with(&x, &mut spec_b, &mut scratch);
        assert_eq!(spec_a, spec_b, "real forward_with differs at n={n}");

        let mut out_a = alloc::vec![0.0; n];
        let mut out_b = alloc::vec![0.0; n];
        plan.inverse(&spec_a, &mut out_a);
        shared.inverse_with(&spec_b, &mut out_b, &mut scratch);
        assert_eq!(out_a, out_b, "real inverse_with differs at n={n}");
        for (o, &xi) in out_b.iter().zip(&x) {
            assert!((o - xi).abs() < 1e-10, "real round trip at n={n}");
        }
    }
}

/// The half-size inverse (`irfft` / even-length `DynRealFft::inverse`) must
/// reproduce the full-length inverse of the Hermitian-filled spectrum.
#[test]
fn irfft_half_size_matches_full_inverse() {
    const N: usize = 64;
    let x: [f64; N] = core::array::from_fn(|i| (i as f64 * 0.37).sin() + 0.2 * i as f64);
    let mut spec = [Complex::new(0.0, 0.0); N / 2 + 1];
    rfft(&x, &mut spec);

    // Perturb the spectrum so the inverse is not a plain round trip.
    for (k, s) in spec.iter_mut().enumerate() {
        *s *= Complex::new(1.0 + 0.01 * k as f64, 0.0);
    }
    spec[0].im = 0.0;
    spec[N / 2].im = 0.0;

    // Reference: full Hermitian spectrum through the complex inverse.
    let mut full = [Complex::new(0.0, 0.0); N];
    for (k, slot) in full.iter_mut().enumerate() {
        *slot = if k <= N / 2 {
            spec[k]
        } else {
            spec[N - k].conj()
        };
    }
    ifft_inplace(&mut full);

    let mut out = [0.0; N];
    irfft(&spec, &mut out);
    for (o, f) in out.iter().zip(&full) {
        assert!((o - f.re).abs() < 1e-12, "irfft {o} vs {}", f.re);
    }
}

// ─── 2D: transposed row pass and batching at sizes past the SIMD widths ─

/// Reference 2D transform: 1D `DynFft` on every column, then a gather /
/// scatter 1D `DynFft` on every row (the original, un-transposed algorithm).
#[cfg(feature = "alloc")]
fn reference_fft2(x: &DynMatrix<Complex<f64>>, inverse: bool) -> DynMatrix<Complex<f64>> {
    let (rows, cols) = (x.nrows(), x.ncols());
    let mut out = x.clone();
    let mut col_plan = DynFft::<f64>::new(rows);
    for col in out.as_mut_slice().chunks_mut(rows) {
        if inverse {
            col_plan.inverse(col);
        } else {
            col_plan.forward(col);
        }
    }
    let mut row_plan = DynFft::<f64>::new(cols);
    let mut row = alloc::vec![Complex::new(0.0, 0.0); cols];
    for r in 0..rows {
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = out[(r, c)];
        }
        if inverse {
            row_plan.inverse(&mut row);
        } else {
            row_plan.forward(&mut row);
        }
        for (c, &v) in row.iter().enumerate() {
            out[(r, c)] = v;
        }
    }
    out
}

/// Sizes that are not multiples of the transpose tile, exceed the SIMD lane
/// widths, and (with `rayon`) exceed the parallel work threshold — so the
/// blocked transpose remainders and the parallel batch path are exercised.
#[cfg(feature = "alloc")]
#[test]
fn dynfft2_large_matches_reference() {
    for &(rows, cols) in &[
        (37, 50),
        (256, 700),
        (300, 400),
        (1024, 700),
        (1, 257),
        (129, 1),
    ] {
        let input = DynMatrix::from_fn(rows, cols, |r, c| {
            Complex::new(
                ((r * 7 + c * 13) % 251) as f64 / 251.0,
                ((r + 3 * c) % 17) as f64 / 17.0,
            )
        });
        let expected = reference_fft2(&input, false);
        let mut plan = DynFft2::<f64>::new(rows, cols);
        let mut buf = input.clone();
        plan.forward(&mut buf);
        let err = max_err2(&buf, &expected);
        assert!(err < 1e-9, "DynFft2 forward error {err} at {rows}x{cols}");

        plan.inverse(&mut buf);
        let err = max_err2(&buf, &input);
        assert!(
            err < 1e-9,
            "DynFft2 round-trip error {err} at {rows}x{cols}"
        );
    }
}

#[cfg(feature = "alloc")]
#[test]
fn dynrealfft2_large_matches_complex() {
    for &(rows, cols) in &[(37, 50), (256, 700), (301, 400), (1024, 700), (2, 257)] {
        let real = DynMatrix::from_fn(rows, cols, |r, c| ((r * 7 + c * 13) % 251) as f64 / 251.0);
        let complexified = DynMatrix::from_fn(rows, cols, |r, c| Complex::new(real[(r, c)], 0.0));
        let full = reference_fft2(&complexified, false);

        let mut plan = DynRealFft2::<f64>::new(rows, cols);
        let half = rows / 2 + 1;
        let mut spec = DynMatrix::zeros(half, cols);
        plan.forward(&real, &mut spec);
        let mut err = 0.0f64;
        for kr in 0..half {
            for kc in 0..cols {
                err = err.max((spec[(kr, kc)] - full[(kr, kc)]).norm());
            }
        }
        assert!(err < 1e-9, "rfft2 forward error {err} at {rows}x{cols}");

        let spec_copy = spec.clone();
        let mut recon = DynMatrix::zeros(rows, cols);
        plan.inverse(&spec, &mut recon);
        assert_eq!(
            spec.as_slice(),
            spec_copy.as_slice(),
            "inverse must not mutate its input"
        );
        let err = recon
            .as_slice()
            .iter()
            .zip(real.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-9, "rfft2 round-trip error {err} at {rows}x{cols}");
    }
}

// ─── 2D convolution ────────────────────────────────────────────────────

#[cfg(feature = "alloc")]
fn direct_convolve2d(a: &DynMatrix<f64>, b: &DynMatrix<f64>) -> DynMatrix<f64> {
    let (ra, ca, rb, cb) = (a.nrows(), a.ncols(), b.nrows(), b.ncols());
    let mut out = DynMatrix::zeros(ra + rb - 1, ca + cb - 1);
    for i in 0..ra {
        for j in 0..ca {
            for p in 0..rb {
                for q in 0..cb {
                    out[(i + p, j + q)] += a[(i, j)] * b[(p, q)];
                }
            }
        }
    }
    out
}

#[cfg(feature = "alloc")]
#[test]
fn fft_convolve2d_matches_direct() {
    use super::{fft_convolve2d, fft_correlate2d};
    for &((ra, ca), (rb, cb)) in &[
        ((5, 7), (3, 2)),
        ((1, 9), (4, 1)),
        ((16, 16), (5, 5)),
        ((6, 3), (6, 3)),
    ] {
        let a = DynMatrix::from_fn(ra, ca, |r, c| ((r * 3 + c * 5) % 7) as f64 - 3.0);
        let b = DynMatrix::from_fn(rb, cb, |r, c| ((r + 2 * c) % 5) as f64 * 0.5 - 1.0);

        let expected = direct_convolve2d(&a, &b);
        let got = fft_convolve2d(&a, &b);
        assert_eq!((got.nrows(), got.ncols()), (ra + rb - 1, ca + cb - 1));
        for (g, e) in got.as_slice().iter().zip(expected.as_slice()) {
            assert!(
                (g - e).abs() < 1e-10,
                "convolve2d {g} vs {e} at {ra}x{ca} * {rb}x{cb}"
            );
        }

        // correlate(a, b) == convolve(a, flip(b)).
        let bflip = DynMatrix::from_fn(rb, cb, |r, c| b[(rb - 1 - r, cb - 1 - c)]);
        let expected = direct_convolve2d(&a, &bflip);
        let got = fft_correlate2d(&a, &b);
        for (g, e) in got.as_slice().iter().zip(expected.as_slice()) {
            assert!(
                (g - e).abs() < 1e-10,
                "correlate2d {g} vs {e} at {ra}x{ca} * {rb}x{cb}"
            );
        }
    }
}

#[cfg(feature = "alloc")]
#[test]
fn fft_convolve2d_empty() {
    use super::fft_convolve2d;
    let a = DynMatrix::<f64>::zeros(3, 3);
    let empty = DynMatrix::<f64>::zeros(0, 3);
    let out = fft_convolve2d(&a, &empty);
    assert_eq!((out.nrows(), out.ncols()), (0, 0));
}
