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
