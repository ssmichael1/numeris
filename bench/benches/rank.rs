//! Median / rank filter throughput across image sizes.
//!
//! Parallel vs. sequential is selected by the bench crate's `par` feature
//! (default on → numeris `rayon` on). Compare with Criterion baselines:
//!
//! ```text
//! cargo bench -p numeris-bench --bench rank -- --save-baseline par
//! cargo bench -p numeris-bench --bench rank --no-default-features -- --baseline par
//! ```
//!
//! Quickselect per pixel is the most expensive per-pixel work in imageproc, so
//! the parallel win shows at smaller sizes than the cheap separable blur.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numeris::imageproc::{median_filter, BorderMode};
use numeris::DynMatrix;

fn image(n: usize) -> DynMatrix<f64> {
    DynMatrix::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 251) as f64)
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("median_filter");
    for n in [64usize, 128, 256] {
        let img = image(n);
        group.throughput(Throughput::Elements((n * n) as u64));
        for radius in [1usize, 2] {
            group.bench_with_input(
                BenchmarkId::new(format!("r{radius}"), n),
                &img,
                |b, img| {
                    b.iter(|| {
                        std::hint::black_box(median_filter(img, radius, BorderMode::Reflect))
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
