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
use numeris::imageproc::{gaussian_blur, BorderMode};
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
        group.bench_with_input(BenchmarkId::from_parameter(n), &img, |b, img| {
            b.iter(|| std::hint::black_box(gaussian_blur(img, 2.0_f64, BorderMode::Reflect)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
