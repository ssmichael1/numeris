use crate::Matrix;
use crate::traits::FloatScalar;
use super::lu::LuDecomposition;
use super::LinalgError;

/// Padé [13,13] coefficients (integer values, cast to T at use site).
const PADE_COEFF: [u64; 14] = [
    64764752532480000,
    32382376266240000,
    7771770303897600,
    1187353796428800,
    129060195264000,
    10559470521600,
    670442572800,
    33522128640,
    1323241920,
    40840800,
    960960,
    16380,
    182,
    1,
];

/// Threshold for the [13,13] Padé approximant (Higham 2005, Table 10.2).
const THETA_13: f64 = 5.371920351148152;

/// Matrix exponential via scaling-and-squaring with [13,13] Padé approximation.
///
/// Computes e^A for a square matrix A using the algorithm from
/// Higham (2005), "The Scaling and Squaring Method for the Matrix Exponential Revisited".
///
/// Returns `LinalgError::Singular` if the internal LU solve fails
/// (should not happen for well-conditioned inputs).
pub fn expm<T: FloatScalar, const N: usize>(
    a: &Matrix<T, N, N>,
) -> Result<Matrix<T, N, N>, LinalgError> {
    let norm = a.norm_one();
    let theta = T::from(THETA_13).unwrap();

    // Find smallest s >= 0 such that ||A||_1 / 2^s <= theta_13
    let mut s: u32 = 0;
    if norm > theta {
        // s = ceil(log2(norm / theta))
        let ratio = norm / theta;
        let s_real = ratio.log2().ceil();
        // Clamp to prevent overflow in 1u64 << s (max 63) and extreme squaring
        s = if s_real > T::from(63).unwrap() {
            63
        } else {
            s_real.to_u32().unwrap_or(1)
        };
        if s == 0 && ratio > T::one() {
            s = 1;
        }
    }

    // Scale: A_scaled = A / 2^s
    let scale = if s < 63 {
        T::from(1u64 << s).unwrap()
    } else {
        T::from(2.0).unwrap().powi(s as i32)
    };
    let a_scaled = *a * (T::one() / scale);

    // Compute powers: A^2, A^4, A^6
    let a2 = a_scaled * a_scaled;
    let a4 = a2 * a2;
    let a6 = a4 * a2;

    // Convert Padé coefficients to T
    let b: [T; 14] = core::array::from_fn(|i| T::from(PADE_COEFF[i]).unwrap());

    let ident = Matrix::<T, N, N>::eye();

    // Inner portion for U:
    //   u_inner = A6*(b13*A6 + b11*A4 + b9*A2) + b7*A6 + b5*A4 + b3*A2 + b1*I
    let u_inner = a6 * (a6 * b[13] + a4 * b[11] + a2 * b[9])
        + a6 * b[7]
        + a4 * b[5]
        + a2 * b[3]
        + ident * b[1];

    // U = A * u_inner
    let u = a_scaled * u_inner;

    // V = A6*(b12*A6 + b10*A4 + b8*A2) + b6*A6 + b4*A4 + b2*A2 + b0*I
    let v = a6 * (a6 * b[12] + a4 * b[10] + a2 * b[8])
        + a6 * b[6]
        + a4 * b[4]
        + a2 * b[2]
        + ident * b[0];

    // p13 = U + V,  q13 = V - U
    let p = u + v;
    let q = v - u;

    // Solve q * result = p  =>  result = q^{-1} * p
    let lu = LuDecomposition::new(&q)?;
    let mut result = lu.inverse() * p;

    // Square s times: result = result^(2^s)
    for _ in 0..s {
        result = result * result;
    }

    Ok(result)
}

impl<T: FloatScalar, const N: usize> Matrix<T, N, N> {
    /// Matrix exponential via scaling-and-squaring with [13,13] Padé approximation.
    ///
    /// Computes e^A for this square matrix.
    pub fn expm(&self) -> Result<Self, LinalgError> {
        expm(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn mat_approx_eq<const N: usize>(
        a: &Matrix<f64, N, N>,
        b: &Matrix<f64, N, N>,
        tol: f64,
    ) -> bool {
        for i in 0..N {
            for j in 0..N {
                if !approx_eq(a[(i, j)], b[(i, j)], tol) {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn expm_zero() {
        // e^0 = I
        let zero = Matrix::<f64, 3, 3>::zeros();
        let result = expm(&zero).unwrap();
        let eye = Matrix::<f64, 3, 3>::eye();
        assert!(mat_approx_eq(&result, &eye, 1e-12));
    }

    #[test]
    fn expm_identity() {
        // e^I = e * I
        let eye = Matrix::<f64, 3, 3>::eye();
        let result = expm(&eye).unwrap();
        let e = core::f64::consts::E;
        let expected = Matrix::new([[e, 0.0, 0.0], [0.0, e, 0.0], [0.0, 0.0, e]]);
        assert!(mat_approx_eq(&result, &expected, 1e-10));
    }

    #[test]
    fn expm_diagonal() {
        // e^diag(a, b) = diag(e^a, e^b)
        let a = 2.0_f64;
        let b = -1.0_f64;
        let m = Matrix::new([[a, 0.0], [0.0, b]]);
        let result = expm(&m).unwrap();
        let expected = Matrix::new([[a.exp(), 0.0], [0.0, b.exp()]]);
        assert!(mat_approx_eq(&result, &expected, 1e-10));
    }

    #[test]
    fn expm_nilpotent() {
        // A = [[0,1],[0,0]] is nilpotent (A^2 = 0)
        // e^A = I + A = [[1,1],[0,1]]
        let a = Matrix::new([[0.0_f64, 1.0], [0.0, 0.0]]);
        let result = expm(&a).unwrap();
        let expected = Matrix::new([[1.0, 1.0], [0.0, 1.0]]);
        assert!(mat_approx_eq(&result, &expected, 1e-12));
    }

    #[test]
    fn expm_antisymmetric() {
        // A = [[0, -theta], [theta, 0]]
        // e^A = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
        let theta = 0.7_f64;
        let a = Matrix::new([[0.0, -theta], [theta, 0.0]]);
        let result = expm(&a).unwrap();
        let expected = Matrix::new([
            [theta.cos(), -theta.sin()],
            [theta.sin(), theta.cos()],
        ]);
        assert!(mat_approx_eq(&result, &expected, 1e-12));
    }

    #[test]
    fn expm_known_3x3() {
        // A = [[1, 1, 0], [0, 0, 2], [0, 0, -1]]
        // Upper triangular, eigenvalues 1, 0, -1
        // Verified against scipy.linalg.expm
        let a = Matrix::new([
            [1.0_f64, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -1.0],
        ]);
        let result = expm(&a).unwrap();

        // Expected values (from scipy):
        // e^A[0,0] = e^1
        // e^A[1,1] = e^0 = 1
        // e^A[2,2] = e^{-1}
        // e^A[0,1] = e^1 - 1  (integral of e^{1-t}*1 dt from 0 to 1 = e - 1, times e^0...
        //            actually for upper triangular: (e^1 - e^0)/(1-0) = e - 1)
        // e^A[1,2] = 2*(e^0 - e^{-1})/(0-(-1)) = 2*(1 - 1/e) = 2 - 2/e
        // e^A[0,2] requires more care
        let e = core::f64::consts::E;

        // Check diagonal entries
        assert!(approx_eq(result[(0, 0)], e, 1e-10));
        assert!(approx_eq(result[(1, 1)], 1.0, 1e-10));
        assert!(approx_eq(result[(2, 2)], 1.0 / e, 1e-10));

        // Check off-diagonal (0,1): (e^1 - e^0)/(1 - 0) = e - 1
        assert!(approx_eq(result[(0, 1)], e - 1.0, 1e-10));

        // Check off-diagonal (1,2): 2*(e^0 - e^{-1})/(0 - (-1)) = 2*(1 - 1/e)
        assert!(approx_eq(result[(1, 2)], 2.0 * (1.0 - 1.0 / e), 1e-10));

        // Check lower triangle is zero
        assert!(approx_eq(result[(1, 0)], 0.0, 1e-10));
        assert!(approx_eq(result[(2, 0)], 0.0, 1e-10));
        assert!(approx_eq(result[(2, 1)], 0.0, 1e-10));
    }

    #[test]
    fn expm_scaling() {
        // Verify e^((t1+t2)*A) = e^(t1*A) * e^(t2*A)
        let a = Matrix::new([
            [0.1_f64, 0.2, -0.1],
            [-0.3, 0.0, 0.15],
            [0.05, -0.1, 0.2],
        ]);

        let t1 = 1.5_f64;
        let t2 = 2.3_f64;

        let exp_t1 = expm(&(a * t1)).unwrap();
        let exp_t2 = expm(&(a * t2)).unwrap();
        let exp_sum = expm(&(a * (t1 + t2))).unwrap();

        let product = exp_t1 * exp_t2;
        assert!(mat_approx_eq(&exp_sum, &product, 1e-10));
    }

    #[test]
    fn expm_f32() {
        // Verify f32 support
        let a = Matrix::new([[0.0_f32, -1.0], [1.0, 0.0]]);
        let result = expm(&a).unwrap();

        // e^A should be rotation by 1 radian
        let c = 1.0_f32.cos();
        let s = 1.0_f32.sin();
        let expected = Matrix::new([[c, -s], [s, c]]);

        for i in 0..2 {
            for j in 0..2 {
                assert!((result[(i, j)] - expected[(i, j)]).abs() < 1e-4);
            }
        }
    }
}
