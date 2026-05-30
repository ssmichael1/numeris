//! Finite-difference Jacobian: parallel (`rayon`) vs. sequential.
//!
//! The bench crate's `par` feature (default on) enables numeris's `rayon`, so
//! `numeris::optim::finite_difference_jacobian_dyn` runs in parallel above its
//! internal column threshold. This file also keeps a faithful sequential
//! re-implementation of the exact same algorithm as an in-binary baseline, so a
//! single `cargo bench` run shows seq vs. par side by side. Both call an
//! identical synthetic `f`, so the only difference measured is the column
//! fan-out. (The convolve bench instead compares via the feature toggle.)
//!
//! Run: `cargo bench -p numeris-bench --bench fd_jacobian`
//!
//! The crossover is governed by two knobs: `n` (number of columns / fan-out) and
//! the per-evaluation cost of `f`. Parallelism wins once `n * cost(f)` exceeds
//! thread-dispatch overhead — large `n` and/or an expensive `f`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(feature = "par")]
use numeris::optim::finite_difference_jacobian_dyn_par;
use numeris::DynVector;

/// Synthetic residual `f: R^n -> R^n` with a tunable per-element cost.
///
/// `work` extra transcendental ops per output element simulate an expensive
/// evaluation (an ODE step, a measurement model, etc.) — the regime where
/// parallelizing the columns actually pays.
fn make_f(work: usize) -> impl Fn(&DynVector<f64>) -> DynVector<f64> + Sync + Send {
    move |x: &DynVector<f64>| {
        let n = x.len();
        let mut out = DynVector::zeros(n);
        for i in 0..n {
            let mut v = x[i] * x[i];
            for _ in 0..work {
                v = (v + 1.0).sqrt().ln_1p();
            }
            out[i] = v;
        }
        out
    }
}

/// Faithful sequential re-implementation of `finite_difference_jacobian_dyn`,
/// used as the baseline (matches the algorithm in `src/optim/dyn_optim.rs`).
fn fd_jacobian_seq<F: Fn(&DynVector<f64>) -> DynVector<f64>>(
    f: F,
    x: &DynVector<f64>,
) -> Vec<f64> {
    let sqrt_eps = f64::EPSILON.sqrt();
    let f0 = f(x);
    let m = f0.len();
    let n = x.len();
    let mut jac = vec![0.0; m * n];
    for j in 0..n {
        let h = sqrt_eps * x[j].abs().max(1.0);
        let mut xp = x.clone();
        xp[j] += h;
        let fp = f(&xp);
        for i in 0..m {
            jac[j * m + i] = (fp[i] - f0[i]) / h;
        }
    }
    jac
}

fn bench(c: &mut Criterion) {
    // (dimension, per-eval work) pairs spanning the crossover.
    let cases = [(4usize, 0usize), (16, 4), (64, 4), (256, 8)];

    let mut group = c.benchmark_group("fd_jacobian");
    for (n, work) in cases {
        let x = DynVector::from_slice(&(0..n).map(|k| 0.5 + k as f64).collect::<Vec<_>>());
        group.throughput(Throughput::Elements(n as u64));
        let label = format!("n{n}_w{work}");

        group.bench_with_input(BenchmarkId::new("seq", &label), &x, |b, x| {
            let f = make_f(work);
            b.iter(|| std::hint::black_box(fd_jacobian_seq(&f, x)));
        });

        // The parallel variant only exists when numeris's `rayon` feature is on
        // (selected via this bench crate's `par` feature).
        #[cfg(feature = "par")]
        group.bench_with_input(BenchmarkId::new("par", &label), &x, |b, x| {
            let f = make_f(work);
            b.iter(|| std::hint::black_box(finite_difference_jacobian_dyn_par(&f, x)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
