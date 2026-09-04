//! FFT throughput: `DynFft` power-of-two and Bluestein lengths, real-input
//! transforms, 2D transforms, and FFT convolution.
//!
//! Parallel vs. sequential (2D only — a single 1D transform never uses rayon)
//! is selected by the bench crate's `par` feature. Compare with Criterion
//! baselines:
//!
//! ```text
//! cargo bench -p numeris-bench --bench fft -- --save-baseline par
//! cargo bench -p numeris-bench --bench fft --no-default-features -- --baseline par
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numeris::fft::{fft_convolve, fft_inplace, DynFft, DynFft2, DynRealFft, DynRealFft2};
use numeris::{Complex, DynMatrix};

fn signal(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| Complex::new(((i * 7) % 13) as f64 - 6.0, ((i * 5) % 11) as f64 - 5.0))
        .collect()
}

fn signal_f32(n: usize) -> Vec<Complex<f32>> {
    signal(n)
        .into_iter()
        .map(|z| Complex::new(z.re as f32, z.im as f32))
        .collect()
}

fn bench_dynfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynfft_forward");
    for n in [256usize, 1024, 4096, 65536] {
        group.throughput(Throughput::Elements(n as u64));
        let mut plan = DynFft::<f64>::new(n);
        let src = signal(n);
        let mut buf = src.clone();
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |b, _| {
            b.iter(|| {
                buf.copy_from_slice(&src);
                plan.forward(&mut buf);
                std::hint::black_box(&buf);
            });
        });
        let mut plan32 = DynFft::<f32>::new(n);
        let src32 = signal_f32(n);
        let mut buf32 = src32.clone();
        group.bench_with_input(BenchmarkId::new("f32", n), &n, |b, _| {
            b.iter(|| {
                buf32.copy_from_slice(&src32);
                plan32.forward(&mut buf32);
                std::hint::black_box(&buf32);
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("dynfft_inverse");
    for n in [1024usize, 65536] {
        group.throughput(Throughput::Elements(n as u64));
        let mut plan = DynFft::<f64>::new(n);
        let src = signal(n);
        let mut buf = src.clone();
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |b, _| {
            b.iter(|| {
                buf.copy_from_slice(&src);
                plan.inverse(&mut buf);
                std::hint::black_box(&buf);
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("dynfft_bluestein");
    for n in [1000usize, 1009, 4093] {
        group.throughput(Throughput::Elements(n as u64));
        let mut plan = DynFft::<f64>::new(n);
        let src = signal(n);
        let mut buf = src.clone();
        group.bench_with_input(BenchmarkId::new("f64", n), &n, |b, _| {
            b.iter(|| {
                buf.copy_from_slice(&src);
                plan.forward(&mut buf);
                std::hint::black_box(&buf);
            });
        });
    }
    group.finish();
}

fn bench_fixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_inplace_fixed");
    let src = signal(1024);
    let mut buf: [Complex<f64>; 1024] = src.clone().try_into().unwrap();
    group.throughput(Throughput::Elements(1024));
    group.bench_function("f64/1024", |b| {
        b.iter(|| {
            buf.copy_from_slice(&src);
            fft_inplace(&mut buf);
            std::hint::black_box(&buf);
        });
    });
    group.finish();
}

fn bench_real(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynrealfft");
    for n in [4096usize, 65536] {
        group.throughput(Throughput::Elements(n as u64));
        let mut plan = DynRealFft::<f64>::new(n);
        let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
        let mut spec = vec![Complex::new(0.0, 0.0); n / 2 + 1];
        group.bench_with_input(BenchmarkId::new("forward", n), &n, |b, _| {
            b.iter(|| {
                plan.forward(&x, &mut spec);
                std::hint::black_box(&spec);
            });
        });
        plan.forward(&x, &mut spec);
        let mut out = vec![0.0; n];
        group.bench_with_input(BenchmarkId::new("inverse", n), &n, |b, _| {
            b.iter(|| {
                plan.inverse(&spec, &mut out);
                std::hint::black_box(&out);
            });
        });
    }
    group.finish();
}

fn bench_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynfft2");
    for n in [256usize, 512, 1024] {
        group.throughput(Throughput::Elements((n * n) as u64));
        let mut plan = DynFft2::<f64>::new(n, n);
        let src = DynMatrix::from_fn(n, n, |r, c| {
            Complex::new(((r * 7 + c * 13) % 251) as f64, ((r + c) % 17) as f64)
        });
        let mut img = src.clone();
        group.bench_with_input(BenchmarkId::new("forward", n), &n, |b, _| {
            b.iter(|| {
                img.as_mut_slice().copy_from_slice(src.as_slice());
                plan.forward(&mut img);
                std::hint::black_box(&img);
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("dynrealfft2");
    for n in [512usize, 1024] {
        group.throughput(Throughput::Elements((n * n) as u64));
        let mut plan = DynRealFft2::<f64>::new(n, n);
        let img = DynMatrix::from_fn(n, n, |r, c| ((r * 7 + c * 13) % 251) as f64);
        let mut spec = DynMatrix::zeros(n / 2 + 1, n);
        group.bench_with_input(BenchmarkId::new("forward", n), &n, |b, _| {
            b.iter(|| {
                plan.forward(&img, &mut spec);
                std::hint::black_box(&spec);
            });
        });
        plan.forward(&img, &mut spec);
        let mut out = DynMatrix::zeros(n, n);
        group.bench_with_input(BenchmarkId::new("inverse", n), &n, |b, _| {
            b.iter(|| {
                plan.inverse(&spec, &mut out);
                std::hint::black_box(&out);
            });
        });
    }
    group.finish();
}

fn bench_convolve(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_convolve");
    for (na, nb) in [(4096usize, 512usize), (10000, 1000)] {
        group.throughput(Throughput::Elements((na + nb) as u64));
        let a: Vec<f64> = (0..na).map(|i| ((i * 7) % 13) as f64).collect();
        let b: Vec<f64> = (0..nb).map(|i| ((i * 5) % 11) as f64).collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{na}x{nb}")),
            &(na, nb),
            |bench, _| {
                bench.iter(|| std::hint::black_box(fft_convolve(&a, &b)));
            },
        );
    }
    group.finish();
}

/// Head-to-head against rustfft (pure Rust, within ~1.5× of FFTW), complex
/// `f64` forward transforms at power-of-two sizes.
fn bench_vs_rustfft(c: &mut Criterion) {
    use rustfft::FftPlanner;
    let mut group = c.benchmark_group("fft_vs_rustfft");
    for n in [1024usize, 4096, 65536] {
        group.throughput(Throughput::Elements(n as u64));
        let src = signal(n);
        let mut plan = DynFft::<f64>::new(n);
        let mut buf = src.clone();
        group.bench_with_input(BenchmarkId::new("numeris", n), &n, |b, _| {
            b.iter(|| {
                buf.copy_from_slice(&src);
                plan.forward(&mut buf);
                std::hint::black_box(&buf);
            });
        });
        let mut planner = FftPlanner::<f64>::new();
        let rfft = planner.plan_fft_forward(n);
        let rsrc: Vec<rustfft::num_complex::Complex<f64>> = src
            .iter()
            .map(|z| rustfft::num_complex::Complex::new(z.re, z.im))
            .collect();
        let mut rbuf = rsrc.clone();
        group.bench_with_input(BenchmarkId::new("rustfft", n), &n, |b, _| {
            b.iter(|| {
                rbuf.copy_from_slice(&rsrc);
                rfft.process(&mut rbuf);
                std::hint::black_box(&rbuf);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dynfft,
    bench_fixed,
    bench_real,
    bench_2d,
    bench_convolve,
    bench_vs_rustfft
);
criterion_main!(benches);
