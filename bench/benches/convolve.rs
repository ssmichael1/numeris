//! Separable convolution (`gaussian_blur`) throughput across image sizes.
//!
//! Parallel vs. sequential is selected by the bench crate's `par` feature
//! (on by default → numeris `rayon` on). Compare with Criterion baselines:
//!
//! ```text
//! cargo bench -p numeris-bench --bench convolve -- --save-baseline par
//! cargo bench -p numeris-bench --bench convolve --no-default-features -- --baseline par
//! ```
//!
//! The parallel path fans out over output columns above numeris's internal
//! `CONV_PAR_MIN_COLS` (64), so sizes below that run sequentially regardless of
//! the feature; the larger sizes show the speedup.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numeris::imageproc::{gaussian_blur, gaussian_blur_cols, gaussian_blur_into, BorderMode};
use numeris::DynMatrix;

fn image(n: usize) -> DynMatrix<f64> {
    // Deterministic, non-uniform content so the AXPY work is real.
    DynMatrix::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 251) as f64)
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_blur");
    for n in [32usize, 128, 512] {
        let img = image(n);
        group.throughput(Throughput::Elements((n * n) as u64));
        // sigma=1 → 7-tap kernel (small-kernel regime), sigma=2 → 13-tap.
        for sigma in [1.0_f64, 2.0] {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{n}/sigma{sigma}")),
                &img,
                |b, img| {
                    b.iter(|| std::hint::black_box(gaussian_blur(img, sigma, BorderMode::Reflect)));
                },
            );
        }
    }
    group.finish();
}

/// The three ways of running one large blur: allocating, into reused
/// buffers, and streamed through the per-column callback (band 64 / 128).
/// 2048² f32 with σ = 1.5 (11 taps) is the star-tracker matched-filter case
/// that motivated the `_into` / `_cols` variants: at this size the fresh
/// allocations plus first-touch page faults are a large share of the
/// allocating call, especially in the parallel build.
fn bench_large_f32(c: &mut Criterion) {
    let n = 2048usize;
    let sigma = 1.5_f32;
    let img = DynMatrix::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 251) as f32);
    let mut group = c.benchmark_group("gaussian_blur_2048_f32");
    group.throughput(Throughput::Elements((n * n) as u64));
    group.bench_function("alloc", |b| {
        b.iter(|| std::hint::black_box(gaussian_blur(&img, sigma, BorderMode::Reflect)));
    });
    let mut tmp = DynMatrix::<f32>::zeros(n, n);
    let mut dst = DynMatrix::<f32>::zeros(n, n);
    group.bench_function("into", |b| {
        b.iter(|| {
            gaussian_blur_into(&img, sigma, BorderMode::Reflect, &mut tmp, &mut dst);
            std::hint::black_box(&dst);
        });
    });
    for band in [64usize, 128] {
        group.bench_with_input(BenchmarkId::new("cols", band), &band, |b, &band| {
            b.iter(|| {
                gaussian_blur_cols(&img, sigma, BorderMode::Reflect, band, |_j, col| {
                    std::hint::black_box(col[0]);
                });
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench, bench_large_f32);
criterion_main!(benches);
