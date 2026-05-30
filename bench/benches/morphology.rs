//! Morphology (Van Herk dilation) throughput across image sizes.
//!
//! Parallel vs. sequential is selected by the bench crate's `par` feature
//! (default on → numeris `rayon` on). Compare with Criterion baselines:
//!
//! ```text
//! cargo bench -p numeris-bench --bench morphology -- --save-baseline par
//! cargo bench -p numeris-bench --bench morphology --no-default-features -- --baseline par
//! ```
//!
//! Van Herk is O(1) per pixel, so dilation is cheap per pixel; the parallel
//! implementation runs both passes and both transposes over output columns.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numeris::imageproc::{dilate, BorderMode};
use numeris::DynMatrix;

fn image(n: usize) -> DynMatrix<f64> {
    DynMatrix::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 251) as f64)
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("dilate");
    for n in [128usize, 512, 1024] {
        let img = image(n);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &img, |b, img| {
            b.iter(|| std::hint::black_box(dilate(img, 3, BorderMode::Reflect)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
