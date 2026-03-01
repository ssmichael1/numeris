use criterion::{criterion_group, criterion_main, Criterion};

// ---------------------------------------------------------------------------
// Helpers: build symmetric positive-definite matrices for Cholesky benchmarks
// ---------------------------------------------------------------------------

fn numeris_spd_4() -> numeris::Matrix4<f64> {
    let a = numeris::Matrix4::from_fn(|i, j| ((i + 1) * (j + 1)) as f64 + if i == j { 10.0 } else { 0.0 });
    let at = a.transpose();
    a * at
}

fn nalgebra_spd_4() -> nalgebra::Matrix4<f64> {
    let a = nalgebra::Matrix4::from_fn(|i, j| ((i + 1) * (j + 1)) as f64 + if i == j { 10.0 } else { 0.0 });
    a * a.transpose()
}

fn numeris_spd_6() -> numeris::Matrix6<f64> {
    let a = numeris::Matrix6::from_fn(|i, j| ((i + 1) * (j + 1)) as f64 + if i == j { 10.0 } else { 0.0 });
    let at = a.transpose();
    a * at
}

fn nalgebra_spd_6() -> nalgebra::Matrix6<f64> {
    let a = nalgebra::Matrix6::from_fn(|i, j| ((i + 1) * (j + 1)) as f64 + if i == j { 10.0 } else { 0.0 });
    a * a.transpose()
}

fn faer_spd(n: usize) -> faer::Mat<f64> {
    let a = faer::Mat::from_fn(n, n, |i, j| ((i + 1) * (j + 1)) as f64 + if i == j { 10.0 } else { 0.0 });
    &a * a.transpose()
}

// ---------------------------------------------------------------------------
// Matrix multiply
// ---------------------------------------------------------------------------

fn matmul_4x4(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul_4x4");

    g.bench_function("numeris", |b| {
        let a = numeris::Matrix4::from_fn(|i, j| (i * 4 + j + 1) as f64);
        let m = numeris::Matrix4::from_fn(|i, j| (i + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a) * std::hint::black_box(&m))
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra::Matrix4::from_fn(|i, j| (i * 4 + j + 1) as f64);
        let m = nalgebra::Matrix4::from_fn(|i, j| (i + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a) * std::hint::black_box(&m))
    });

    g.bench_function("faer", |b| {
        let a = faer::Mat::from_fn(4, 4, |i, j| (i * 4 + j + 1) as f64);
        let m = faer::Mat::from_fn(4, 4, |i, j| (i + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a) * std::hint::black_box(&m))
    });

    g.finish();
}

fn matmul_6x6(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul_6x6");

    g.bench_function("numeris", |b| {
        let a = numeris::Matrix6::from_fn(|i, j| (i * 6 + j + 1) as f64);
        let m = numeris::Matrix6::from_fn(|i, j| (i + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a) * std::hint::black_box(&m))
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra::Matrix6::from_fn(|i, j| (i * 6 + j + 1) as f64);
        let m = nalgebra::Matrix6::from_fn(|i, j| (i + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a) * std::hint::black_box(&m))
    });

    g.bench_function("faer", |b| {
        let a = faer::Mat::from_fn(6, 6, |i, j| (i * 6 + j + 1) as f64);
        let m = faer::Mat::from_fn(6, 6, |i, j| (i + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a) * std::hint::black_box(&m))
    });

    g.finish();
}

fn matmul_dyn_50(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul_dyn_50x50");

    g.bench_function("numeris", |b| {
        let a = numeris::DynMatrix::from_fn(50, 50, |i, j| (i * 50 + j + 1) as f64);
        let m = numeris::DynMatrix::from_fn(50, 50, |i, j| (i + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a) * std::hint::black_box(&m))
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra::DMatrix::from_fn(50, 50, |i, j| (i * 50 + j + 1) as f64);
        let m = nalgebra::DMatrix::from_fn(50, 50, |i, j| (i + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a) * std::hint::black_box(&m))
    });

    g.bench_function("faer", |b| {
        let a = faer::Mat::from_fn(50, 50, |i, j| (i * 50 + j + 1) as f64);
        let m = faer::Mat::from_fn(50, 50, |i, j| (i + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a) * std::hint::black_box(&m))
    });

    g.finish();
}

fn matmul_dyn_200(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul_dyn_200x200");

    g.bench_function("numeris", |b| {
        let a = numeris::DynMatrix::from_fn(200, 200, |i, j| (i * 200 + j + 1) as f64);
        let m = numeris::DynMatrix::from_fn(200, 200, |i, j| (i + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a) * std::hint::black_box(&m))
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra::DMatrix::from_fn(200, 200, |i, j| (i * 200 + j + 1) as f64);
        let m = nalgebra::DMatrix::from_fn(200, 200, |i, j| (i + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a) * std::hint::black_box(&m))
    });

    g.bench_function("faer", |b| {
        let a = faer::Mat::from_fn(200, 200, |i, j| (i * 200 + j + 1) as f64);
        let m = faer::Mat::from_fn(200, 200, |i, j| (i + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a) * std::hint::black_box(&m))
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Dot product
// ---------------------------------------------------------------------------

fn dot_100(c: &mut Criterion) {
    let mut g = c.benchmark_group("dot_100");

    g.bench_function("numeris", |b| {
        let a = numeris::DynVector::from_vec((0..100).map(|i| i as f64).collect());
        let v = numeris::DynVector::from_vec((0..100).map(|i| (i * 2) as f64).collect());
        b.iter(|| std::hint::black_box(&a).dot(std::hint::black_box(&v)))
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra::DVector::from_fn(100, |i, _| i as f64);
        let v = nalgebra::DVector::from_fn(100, |i, _| (i * 2) as f64);
        b.iter(|| std::hint::black_box(&a).dot(std::hint::black_box(&v)))
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// LU decomposition
// ---------------------------------------------------------------------------

fn lu_4x4(c: &mut Criterion) {
    let mut g = c.benchmark_group("lu_4x4");

    g.bench_function("numeris", |b| {
        let a = numeris::Matrix4::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64 + if i == j { 40.0 } else { 0.0 });
        b.iter(|| std::hint::black_box(&a).lu())
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra::Matrix4::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64 + if i == j { 40.0 } else { 0.0 });
        b.iter(|| std::hint::black_box(&a).lu())
    });

    g.bench_function("faer", |b| {
        let a = faer::Mat::from_fn(4, 4, |i, j| ((i + 1) * 10 + j + 1) as f64 + if i == j { 40.0 } else { 0.0 });
        b.iter(|| std::hint::black_box(&a).partial_piv_lu())
    });

    g.finish();
}

fn lu_6x6(c: &mut Criterion) {
    let mut g = c.benchmark_group("lu_6x6");

    g.bench_function("numeris", |b| {
        let a = numeris::Matrix6::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64 + if i == j { 60.0 } else { 0.0 });
        b.iter(|| std::hint::black_box(&a).lu())
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra::Matrix6::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64 + if i == j { 60.0 } else { 0.0 });
        b.iter(|| std::hint::black_box(&a).lu())
    });

    g.bench_function("faer", |b| {
        let a = faer::Mat::from_fn(6, 6, |i, j| ((i + 1) * 10 + j + 1) as f64 + if i == j { 60.0 } else { 0.0 });
        b.iter(|| std::hint::black_box(&a).partial_piv_lu())
    });

    g.finish();
}

fn lu_dyn_50(c: &mut Criterion) {
    let mut g = c.benchmark_group("lu_dyn_50x50");

    g.bench_function("numeris", |b| {
        let a = numeris::DynMatrix::from_fn(50, 50, |i, j| ((i + 1) * 10 + j + 1) as f64 + if i == j { 500.0 } else { 0.0 });
        b.iter(|| std::hint::black_box(&a).lu())
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra::DMatrix::from_fn(50, 50, |i, j| ((i + 1) * 10 + j + 1) as f64 + if i == j { 500.0 } else { 0.0 });
        b.iter(|| std::hint::black_box(&a).clone().lu())
    });

    g.bench_function("faer", |b| {
        let a = faer::Mat::from_fn(50, 50, |i, j| ((i + 1) * 10 + j + 1) as f64 + if i == j { 500.0 } else { 0.0 });
        b.iter(|| std::hint::black_box(&a).partial_piv_lu())
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Cholesky decomposition
// ---------------------------------------------------------------------------

fn cholesky_4x4(c: &mut Criterion) {
    let mut g = c.benchmark_group("cholesky_4x4");

    g.bench_function("numeris", |b| {
        let a = numeris_spd_4();
        b.iter(|| std::hint::black_box(&a).cholesky())
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra_spd_4();
        b.iter(|| std::hint::black_box(&a).cholesky())
    });

    g.bench_function("faer", |b| {
        let a = faer_spd(4);
        b.iter(|| std::hint::black_box(&a).llt(faer::Side::Lower))
    });

    g.finish();
}

fn cholesky_6x6(c: &mut Criterion) {
    let mut g = c.benchmark_group("cholesky_6x6");

    g.bench_function("numeris", |b| {
        let a = numeris_spd_6();
        b.iter(|| std::hint::black_box(&a).cholesky())
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra_spd_6();
        b.iter(|| std::hint::black_box(&a).cholesky())
    });

    g.bench_function("faer", |b| {
        let a = faer_spd(6);
        b.iter(|| std::hint::black_box(&a).llt(faer::Side::Lower))
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// QR decomposition
// ---------------------------------------------------------------------------

fn qr_4x4(c: &mut Criterion) {
    let mut g = c.benchmark_group("qr_4x4");

    g.bench_function("numeris", |b| {
        let a = numeris::Matrix4::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a).qr())
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra::Matrix4::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a).qr())
    });

    g.bench_function("faer", |b| {
        let a = faer::Mat::from_fn(4, 4, |i, j| ((i + 1) * 10 + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a).qr())
    });

    g.finish();
}

fn qr_6x6(c: &mut Criterion) {
    let mut g = c.benchmark_group("qr_6x6");

    g.bench_function("numeris", |b| {
        let a = numeris::Matrix6::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a).qr())
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra::Matrix6::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a).qr())
    });

    g.bench_function("faer", |b| {
        let a = faer::Mat::from_fn(6, 6, |i, j| ((i + 1) * 10 + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a).qr())
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// SVD
// ---------------------------------------------------------------------------

fn svd_4x4(c: &mut Criterion) {
    let mut g = c.benchmark_group("svd_4x4");

    g.bench_function("numeris", |b| {
        let a = numeris::Matrix4::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a).svd())
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra::Matrix4::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a).svd(true, true))
    });

    g.bench_function("faer", |b| {
        let a = faer::Mat::from_fn(4, 4, |i, j| ((i + 1) * 10 + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a).svd())
    });

    g.finish();
}

fn svd_6x6(c: &mut Criterion) {
    let mut g = c.benchmark_group("svd_6x6");

    g.bench_function("numeris", |b| {
        let a = numeris::Matrix6::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a).svd())
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra::Matrix6::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a).svd(true, true))
    });

    g.bench_function("faer", |b| {
        let a = faer::Mat::from_fn(6, 6, |i, j| ((i + 1) * 10 + j + 1) as f64);
        b.iter(|| std::hint::black_box(&a).svd())
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Inverse
// ---------------------------------------------------------------------------

fn inverse_4x4(c: &mut Criterion) {
    let mut g = c.benchmark_group("inverse_4x4");

    g.bench_function("numeris", |b| {
        let a = numeris::Matrix4::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64 + if i == j { 40.0 } else { 0.0 });
        b.iter(|| std::hint::black_box(&a).inverse())
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra::Matrix4::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64 + if i == j { 40.0 } else { 0.0 });
        b.iter(|| std::hint::black_box(&a).try_inverse())
    });

    g.finish();
}

fn inverse_6x6(c: &mut Criterion) {
    let mut g = c.benchmark_group("inverse_6x6");

    g.bench_function("numeris", |b| {
        let a = numeris::Matrix6::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64 + if i == j { 60.0 } else { 0.0 });
        b.iter(|| std::hint::black_box(&a).inverse())
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra::Matrix6::from_fn(|i, j| ((i + 1) * 10 + j + 1) as f64 + if i == j { 60.0 } else { 0.0 });
        b.iter(|| std::hint::black_box(&a).try_inverse())
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Symmetric eigendecomposition
// ---------------------------------------------------------------------------

fn eigen_symmetric_4x4(c: &mut Criterion) {
    let mut g = c.benchmark_group("eigen_symmetric_4x4");

    g.bench_function("numeris", |b| {
        let a = numeris_spd_4();
        b.iter(|| std::hint::black_box(&a).eig_symmetric())
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra_spd_4();
        b.iter(|| std::hint::black_box(&a).symmetric_eigen())
    });

    g.bench_function("faer", |b| {
        let a = faer_spd(4);
        b.iter(|| std::hint::black_box(&a).self_adjoint_eigen(faer::Side::Lower))
    });

    g.finish();
}

fn eigen_symmetric_6x6(c: &mut Criterion) {
    let mut g = c.benchmark_group("eigen_symmetric_6x6");

    g.bench_function("numeris", |b| {
        let a = numeris_spd_6();
        b.iter(|| std::hint::black_box(&a).eig_symmetric())
    });

    g.bench_function("nalgebra", |b| {
        let a = nalgebra_spd_6();
        b.iter(|| std::hint::black_box(&a).symmetric_eigen())
    });

    g.bench_function("faer", |b| {
        let a = faer_spd(6);
        b.iter(|| std::hint::black_box(&a).self_adjoint_eigen(faer::Side::Lower))
    });

    g.finish();
}

// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    matmul_4x4,
    matmul_6x6,
    matmul_dyn_50,
    matmul_dyn_200,
    dot_100,
    lu_4x4,
    lu_6x6,
    lu_dyn_50,
    cholesky_4x4,
    cholesky_6x6,
    qr_4x4,
    qr_6x6,
    svd_4x4,
    svd_6x6,
    inverse_4x4,
    inverse_6x6,
    eigen_symmetric_4x4,
    eigen_symmetric_6x6,
);
criterion_main!(benches);
