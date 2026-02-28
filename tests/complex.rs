#![cfg(feature = "complex")]

use num_complex::Complex;
use numeris::{Matrix, Vector};

type C = Complex<f64>;

fn c(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

const TOL: f64 = 1e-10;

fn assert_complex_near(a: C, b: C, tol: f64, msg: &str) {
    assert!(
        (a.re - b.re).abs() < tol && (a.im - b.im).abs() < tol,
        "{}: {:?} vs {:?}",
        msg,
        a,
        b
    );
}

// ── LU tests ─────────────────────────────────────────────────────────

#[test]
fn complex_lu_solve() {
    // A * x = b with complex entries
    let a = Matrix::new([
        [c(2.0, 1.0), c(1.0, -1.0)],
        [c(1.0, 0.0), c(3.0, 2.0)],
    ]);
    let b = Vector::from_array([c(5.0, 3.0), c(7.0, 4.0)]);

    let x = a.solve(&b).unwrap();

    // Verify A*x == b
    for i in 0..2 {
        let mut sum = C::default();
        for j in 0..2 {
            sum = sum + a[(i, j)] * x[j];
        }
        assert_complex_near(sum, b[i], TOL, &format!("row {}", i));
    }
}

#[test]
fn complex_lu_det() {
    // 2x2 complex determinant: ad - bc
    let a = Matrix::new([
        [c(1.0, 1.0), c(2.0, 0.0)],
        [c(0.0, 1.0), c(1.0, -1.0)],
    ]);
    let det = a.det();
    // (1+i)(1-i) - (2)(i) = 1-i^2 - 2i = 1+1-2i = 2-2i
    assert_complex_near(det, c(2.0, -2.0), TOL, "det");
}

#[test]
fn complex_lu_inverse() {
    let a = Matrix::new([
        [c(2.0, 1.0), c(1.0, -1.0)],
        [c(0.0, 1.0), c(3.0, 0.0)],
    ]);
    let a_inv = a.inverse().unwrap();
    let id = a * a_inv;

    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { c(1.0, 0.0) } else { c(0.0, 0.0) };
            assert_complex_near(id[(i, j)], expected, TOL, &format!("id[{},{}]", i, j));
        }
    }
}

// ── Cholesky tests ───────────────────────────────────────────────────

#[test]
fn complex_cholesky_hermitian() {
    // Hermitian positive-definite: A = [[4, 2+i], [2-i, 5]]
    let a = Matrix::new([
        [c(4.0, 0.0), c(2.0, 1.0)],
        [c(2.0, -1.0), c(5.0, 0.0)],
    ]);
    let chol = a.cholesky().unwrap();
    let l = chol.l_full();

    // Verify L * L^H == A
    // L^H = conjugate transpose
    let lh = Matrix::new([
        [l[(0, 0)].conj(), l[(1, 0)].conj()],
        [l[(0, 1)].conj(), l[(1, 1)].conj()],
    ]);
    let reconstructed = l * lh;
    for i in 0..2 {
        for j in 0..2 {
            assert_complex_near(
                reconstructed[(i, j)],
                a[(i, j)],
                TOL,
                &format!("L*L^H[{},{}]", i, j),
            );
        }
    }
}

#[test]
fn complex_cholesky_solve() {
    // Hermitian positive-definite
    let a = Matrix::new([
        [c(4.0, 0.0), c(2.0, 1.0)],
        [c(2.0, -1.0), c(5.0, 0.0)],
    ]);
    let b = Vector::from_array([c(8.0, 3.0), c(7.0, -1.0)]);

    let chol = a.cholesky().unwrap();
    let x = chol.solve(&b);

    // Verify A*x == b
    for i in 0..2 {
        let mut sum = C::default();
        for j in 0..2 {
            sum = sum + a[(i, j)] * x[j];
        }
        assert_complex_near(sum, b[i], TOL, &format!("row {}", i));
    }
}

// ── QR tests ─────────────────────────────────────────────────────────

#[test]
fn complex_qr_factorization() {
    let a = Matrix::new([
        [c(1.0, 1.0), c(2.0, 0.0)],
        [c(0.0, 1.0), c(1.0, -1.0)],
    ]);
    let qr = a.qr().unwrap();
    let q = qr.q();
    let r = qr.r();

    // Verify Q*R == A
    let qr_prod = q * r;
    for i in 0..2 {
        for j in 0..2 {
            assert_complex_near(qr_prod[(i, j)], a[(i, j)], TOL, &format!("QR[{},{}]", i, j));
        }
    }

    // Verify Q^H * Q == I (unitary)
    let qh = Matrix::new([
        [q[(0, 0)].conj(), q[(1, 0)].conj()],
        [q[(0, 1)].conj(), q[(1, 1)].conj()],
    ]);
    let qhq = qh * q;
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { c(1.0, 0.0) } else { c(0.0, 0.0) };
            assert_complex_near(qhq[(i, j)], expected, TOL, &format!("Q^HQ[{},{}]", i, j));
        }
    }
}

#[test]
fn complex_qr_solve() {
    let a = Matrix::new([
        [c(2.0, 1.0), c(1.0, -1.0)],
        [c(1.0, 0.0), c(3.0, 2.0)],
    ]);
    let b = Vector::from_array([c(5.0, 3.0), c(7.0, 4.0)]);

    let x = a.solve_qr(&b).unwrap();

    // Verify A*x == b
    for i in 0..2 {
        let mut sum = C::default();
        for j in 0..2 {
            sum = sum + a[(i, j)] * x[j];
        }
        assert_complex_near(sum, b[i], TOL, &format!("row {}", i));
    }
}

// ── Norm tests ───────────────────────────────────────────────────────

#[test]
fn complex_vector_norm() {
    // |[3+4i, 0]| = sqrt(|3+4i|^2) = sqrt(25) = 5
    let v = Vector::from_array([c(3.0, 4.0), c(0.0, 0.0)]);
    assert!((v.norm() - 5.0).abs() < TOL);
}

#[test]
fn complex_vector_norm_l1() {
    // ||[3+4i, 1+0i]||_1 = |3+4i| + |1| = 5 + 1 = 6
    let v = Vector::from_array([c(3.0, 4.0), c(1.0, 0.0)]);
    assert!((v.norm_l1() - 6.0).abs() < TOL);
}

#[test]
fn complex_frobenius_norm() {
    let m = Matrix::new([[c(3.0, 4.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]]);
    // sqrt(|3+4i|^2 + 0 + 0 + 1) = sqrt(25 + 1) = sqrt(26)
    assert!((m.frobenius_norm() - 26.0_f64.sqrt()).abs() < TOL);
}

#[test]
fn complex_vector_normalize() {
    let v = Vector::from_array([c(3.0, 4.0), c(0.0, 0.0)]);
    let u = v.normalize();
    assert!((u.norm() - 1.0).abs() < TOL);
    assert_complex_near(u[0], c(0.6, 0.8), TOL, "u[0]");
    assert_complex_near(u[1], c(0.0, 0.0), TOL, "u[1]");
}

#[test]
fn complex_3x3_lu_solve() {
    let a = Matrix::new([
        [c(1.0, 0.0), c(0.0, 1.0), c(2.0, 0.0)],
        [c(0.0, -1.0), c(3.0, 0.0), c(1.0, 1.0)],
        [c(2.0, 0.0), c(1.0, -1.0), c(4.0, 0.0)],
    ]);
    let b = Vector::from_array([c(3.0, 1.0), c(4.0, -1.0), c(7.0, 0.0)]);

    let x = a.solve(&b).unwrap();

    // Verify A*x == b
    for i in 0..3 {
        let mut sum = C::default();
        for j in 0..3 {
            sum = sum + a[(i, j)] * x[j];
        }
        assert_complex_near(sum, b[i], TOL, &format!("row {}", i));
    }
}
