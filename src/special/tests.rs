#[cfg(test)]
mod tests {
    use super::super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "approx_eq failed: {a} vs {b}, diff = {}, tol = {tol}",
            (a - b).abs()
        );
    }

    fn approx_eq_f32(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() < tol,
            "approx_eq_f32 failed: {a} vs {b}, diff = {}, tol = {tol}",
            (a - b).abs()
        );
    }

    // =====================================================================
    // gamma
    // =====================================================================

    #[test]
    fn gamma_positive_integers() {
        // Γ(n) = (n-1)!
        approx_eq(gamma(1.0_f64), 1.0, 1e-14);
        approx_eq(gamma(2.0), 1.0, 1e-14);
        approx_eq(gamma(3.0), 2.0, 1e-14);
        approx_eq(gamma(4.0), 6.0, 1e-13);
        approx_eq(gamma(5.0), 24.0, 1e-12);
        approx_eq(gamma(6.0), 120.0, 1e-10);
        approx_eq(gamma(10.0), 362880.0, 1e-6);
    }

    #[test]
    fn gamma_half_integers() {
        let sqrt_pi = core::f64::consts::PI.sqrt();
        approx_eq(gamma(0.5), sqrt_pi, 1e-14);
        // Γ(1.5) = √π/2
        approx_eq(gamma(1.5), sqrt_pi / 2.0, 1e-14);
        // Γ(2.5) = 3√π/4
        approx_eq(gamma(2.5), 3.0 * sqrt_pi / 4.0, 1e-13);
    }

    #[test]
    fn gamma_negative_values() {
        // Γ(-0.5) = -2√π
        let sqrt_pi = core::f64::consts::PI.sqrt();
        approx_eq(gamma(-0.5), -2.0 * sqrt_pi, 1e-13);
        // Γ(-1.5) = 4√π/3
        approx_eq(gamma(-1.5), 4.0 * sqrt_pi / 3.0, 1e-13);
    }

    #[test]
    fn gamma_poles() {
        assert!(gamma(0.0_f64).is_infinite());
        assert!(gamma(-1.0_f64).is_infinite());
        assert!(gamma(-2.0_f64).is_infinite());
    }

    #[test]
    fn gamma_large_values() {
        // Γ(20) = 19! = 121645100408832000
        approx_eq(gamma(20.0_f64), 121645100408832000.0, 1e2);
    }

    #[test]
    fn gamma_nan() {
        assert!(gamma(f64::NAN).is_nan());
    }

    #[test]
    fn gamma_recurrence_identity() {
        // x·Γ(x) = Γ(x+1) for various x
        for &x in &[0.3, 1.7, 3.14, 5.5] {
            let lhs = x * gamma(x);
            let rhs = gamma(x + 1.0);
            approx_eq(lhs, rhs, 1e-12);
        }
    }

    #[test]
    fn gamma_f32() {
        approx_eq_f32(gamma(5.0_f32), 24.0, 1e-4);
        approx_eq_f32(gamma(0.5_f32), core::f32::consts::PI.sqrt(), 1e-5);
    }

    // =====================================================================
    // lgamma
    // =====================================================================

    #[test]
    fn lgamma_positive_integers() {
        approx_eq(lgamma(1.0_f64), 0.0, 1e-14);
        approx_eq(lgamma(2.0), 0.0, 1e-14);
        approx_eq(lgamma(3.0), 2.0_f64.ln(), 1e-14);
        approx_eq(lgamma(4.0), 6.0_f64.ln(), 1e-13);
    }

    #[test]
    fn lgamma_half() {
        // ln Γ(0.5) = ln √π = 0.5·ln(π)
        let expected = 0.5 * core::f64::consts::PI.ln();
        approx_eq(lgamma(0.5_f64), expected, 1e-14);
    }

    #[test]
    fn lgamma_large_no_overflow() {
        // ln Γ(100) ≈ 359.1342053695754 (Stirling: (n-0.5)ln(n) - n + 0.5ln(2π))
        let val = lgamma(100.0_f64);
        assert!(val.is_finite());
        approx_eq(val, 359.1342053695754, 1e-8);
    }

    #[test]
    fn lgamma_negative() {
        // ln|Γ(-0.5)| = ln(2√π) = ln(2) + 0.5·ln(π)
        let val = lgamma(-0.5_f64);
        let expected = (2.0 * core::f64::consts::PI.sqrt()).ln();
        approx_eq(val, expected, 1e-13);
    }

    #[test]
    fn lgamma_poles() {
        assert!(lgamma(0.0_f64).is_infinite());
        assert!(lgamma(-1.0_f64).is_infinite());
    }

    #[test]
    fn lgamma_stirling_comparison() {
        // For large x, lgamma ≈ Stirling: (x-0.5)·ln(x) - x + 0.5·ln(2π)
        let x = 50.0_f64;
        let stirling = (x - 0.5) * x.ln() - x + 0.5 * core::f64::consts::TAU.ln();
        let diff = (lgamma(x) - stirling).abs();
        assert!(diff < 0.002, "Stirling approximation error too large: {diff}");
    }

    #[test]
    fn lgamma_gamma_consistency() {
        // For moderate positive x: exp(lgamma(x)) ≈ gamma(x)
        for &x in &[1.5_f64, 2.7, 5.0, 10.0] {
            let from_lgamma = lgamma(x).exp();
            let from_gamma = gamma(x);
            let rel_err = ((from_lgamma - from_gamma) / from_gamma).abs();
            assert!(rel_err < 1e-12, "gamma/lgamma inconsistency at x={x}: rel_err={rel_err}");
        }
    }

    #[test]
    fn lgamma_f32() {
        approx_eq_f32(lgamma(1.0_f32), 0.0, 1e-5);
        let val = lgamma(100.0_f32);
        assert!(val.is_finite());
    }

    // =====================================================================
    // digamma
    // =====================================================================

    #[test]
    fn digamma_positive_integers() {
        let euler = 0.5772156649015329_f64;
        // ψ(1) = -γ
        approx_eq(digamma(1.0_f64), -euler, 1e-12);
        // ψ(2) = 1 - γ
        approx_eq(digamma(2.0), 1.0 - euler, 1e-12);
        // ψ(3) = 1 + 1/2 - γ
        approx_eq(digamma(3.0), 1.5 - euler, 1e-12);
    }

    #[test]
    fn digamma_half() {
        // ψ(1/2) = -γ - 2ln(2)
        let euler = 0.5772156649015329_f64;
        let expected = -euler - 2.0 * 2.0_f64.ln();
        approx_eq(digamma(0.5_f64), expected, 1e-13);
    }

    #[test]
    fn digamma_large() {
        // ψ(100) ≈ ln(100) - 1/200 for large x
        let val = digamma(100.0_f64);
        let approx_val = 100.0_f64.ln() - 0.005;
        assert!((val - approx_val).abs() < 0.001);
    }

    #[test]
    fn digamma_negative() {
        // ψ(-0.5) via reflection: ψ(-0.5) = ψ(1.5) + π/tan(-π/2)
        // ψ(1.5) = ψ(0.5) + 1/0.5 = ψ(0.5) + 2
        // Direct computation should match known value
        let val = digamma(-0.5_f64);
        // ψ(-0.5) ≈ 0.03648997397857652 (from reference tables)
        approx_eq(val, 0.03648997397857652, 1e-10);
    }

    #[test]
    fn digamma_poles() {
        assert!(digamma(0.0_f64).is_nan());
        assert!(digamma(-1.0_f64).is_nan());
        assert!(digamma(-2.0_f64).is_nan());
    }

    #[test]
    fn digamma_recurrence_identity() {
        // ψ(x+1) = ψ(x) + 1/x
        for &x in &[0.3, 1.7, 5.5, 10.0] {
            let lhs = digamma(x + 1.0);
            let rhs = digamma(x) + 1.0 / x;
            approx_eq(lhs, rhs, 1e-12);
        }
    }

    #[test]
    fn digamma_numerical_derivative_of_lgamma() {
        // ψ(x) ≈ (lgamma(x+h) - lgamma(x-h)) / (2h) for small h
        let h = 1e-6_f64;
        for &x in &[1.5, 3.0, 7.0] {
            let numerical = (lgamma(x + h) - lgamma(x - h)) / (2.0 * h);
            let analytical = digamma(x);
            approx_eq(numerical, analytical, 1e-6);
        }
    }

    #[test]
    fn digamma_f32() {
        let euler_f32 = 0.5772157_f32;
        approx_eq_f32(digamma(1.0_f32), -euler_f32, 1e-4);
    }

    // =====================================================================
    // beta / lbeta
    // =====================================================================

    #[test]
    fn beta_known_values() {
        // B(1,1) = 1
        approx_eq(beta(1.0_f64, 1.0), 1.0, 1e-14);
        // B(2,3) = Γ(2)Γ(3)/Γ(5) = 1·2/24 = 1/12
        approx_eq(beta(2.0, 3.0), 1.0 / 12.0, 1e-14);
        // B(0.5, 0.5) = π
        approx_eq(beta(0.5, 0.5), core::f64::consts::PI, 1e-12);
    }

    #[test]
    fn beta_symmetry() {
        for &(a, b) in &[(2.0, 3.0), (0.5, 1.5), (5.0, 7.0)] {
            approx_eq(beta(a, b), beta(b, a), 1e-14);
        }
    }

    #[test]
    fn beta_gamma_relation() {
        // B(a,b) = Γ(a)·Γ(b)/Γ(a+b)
        let a = 3.5_f64;
        let b = 2.5;
        let from_gamma = gamma(a) * gamma(b) / gamma(a + b);
        approx_eq(beta(a, b), from_gamma, 1e-12);
    }

    #[test]
    fn lbeta_known() {
        approx_eq(lbeta(1.0_f64, 1.0), 0.0, 1e-14);
    }

    #[test]
    fn lbeta_large() {
        // For large args, lbeta avoids overflow
        let val = lbeta(100.0_f64, 200.0);
        assert!(val.is_finite());
        // Verify: exp(lbeta(a,b)) = beta(a,b) when finite
        // For large args, beta overflows so just check lbeta is reasonable
        assert!(val < 0.0); // B(100,200) < 1 so lbeta < 0
    }

    #[test]
    fn beta_f32() {
        approx_eq_f32(beta(2.0_f32, 3.0), 1.0 / 12.0, 1e-5);
    }

    // =====================================================================
    // gamma_inc / gamma_inc_upper
    // =====================================================================

    #[test]
    fn gamma_inc_trivial() {
        // P(a, 0) = 0 for any a > 0
        approx_eq(gamma_inc(1.0_f64, 0.0).unwrap(), 0.0, 1e-15);
        approx_eq(gamma_inc(5.0, 0.0).unwrap(), 0.0, 1e-15);
    }

    #[test]
    fn gamma_inc_exponential() {
        // P(1, x) = 1 - e^{-x}
        for &x in &[0.5_f64, 1.0, 2.0, 5.0] {
            let expected = 1.0 - (-x).exp();
            approx_eq(gamma_inc(1.0_f64, x).unwrap(), expected, 1e-13);
        }
    }

    #[test]
    fn gamma_inc_erf_cross_check() {
        // P(0.5, x²) = erf(x) for x > 0
        for &x in &[0.5, 1.0, 2.0] {
            let from_inc = gamma_inc(0.5_f64, x * x).unwrap();
            let from_erf = erf(x);
            approx_eq(from_inc, from_erf, 1e-12);
        }
    }

    #[test]
    fn gamma_inc_small_x() {
        // P(3, 0.1) — series region
        let val = gamma_inc(3.0_f64, 0.1).unwrap();
        // Reference: approximately 0.0000152 (very small)
        assert!(val > 0.0 && val < 0.01);
    }

    #[test]
    fn gamma_inc_large_x() {
        // P(2, 20) ≈ 1 (well into CF region)
        let val = gamma_inc(2.0_f64, 20.0).unwrap();
        approx_eq(val, 1.0, 1e-7);
    }

    #[test]
    fn gamma_inc_complement() {
        // P(a,x) + Q(a,x) = 1
        for &(a, x) in &[(2.0, 1.0), (3.0, 5.0), (0.5, 2.0), (10.0, 7.0)] {
            let p = gamma_inc(a, x).unwrap();
            let q = gamma_inc_upper(a, x).unwrap();
            approx_eq(p + q, 1.0, 1e-13);
        }
    }

    #[test]
    fn gamma_inc_monotonicity() {
        // P(a, x) is monotonically increasing in x for fixed a
        let a = 3.0_f64;
        let mut prev = 0.0;
        for x in [0.5, 1.0, 2.0, 5.0, 10.0] {
            let val = gamma_inc(a, x).unwrap();
            assert!(val > prev, "P({a},{x}) = {val} should be > P({a},prev) = {prev}");
            prev = val;
        }
    }

    #[test]
    fn gamma_inc_domain_errors() {
        assert_eq!(gamma_inc(0.0_f64, 1.0), Err(SpecialError::DomainError));
        assert_eq!(gamma_inc(-1.0_f64, 1.0), Err(SpecialError::DomainError));
        assert_eq!(gamma_inc(1.0_f64, -1.0), Err(SpecialError::DomainError));
    }

    #[test]
    fn gamma_inc_f32() {
        let p = gamma_inc(1.0_f32, 1.0).unwrap();
        let expected: f32 = 1.0 - (-1.0_f32).exp();
        approx_eq_f32(p, expected, 1e-5);
    }

    #[test]
    fn gamma_inc_upper_trivial() {
        // Q(a, 0) = 1
        approx_eq(gamma_inc_upper(2.0_f64, 0.0).unwrap(), 1.0, 1e-15);
    }

    // =====================================================================
    // erf / erfc
    // =====================================================================

    #[test]
    fn erf_zero() {
        approx_eq(erf(0.0_f64), 0.0, 1e-16);
    }

    #[test]
    fn erf_small() {
        // erf(0.1) ≈ 0.1124629160182849
        approx_eq(erf(0.1_f64), 0.1124629160182849, 1e-13);
    }

    #[test]
    fn erf_one() {
        approx_eq(erf(1.0_f64), 0.8427007929497149, 1e-13);
    }

    #[test]
    fn erf_two() {
        approx_eq(erf(2.0_f64), 0.9953222650189527, 1e-13);
    }

    #[test]
    fn erf_large() {
        // erf(6) ≈ 1.0 (to machine precision)
        approx_eq(erf(6.0_f64), 1.0, 1e-15);
    }

    #[test]
    fn erf_negative_symmetry() {
        // erf(-x) = -erf(x)
        for &x in &[0.3, 1.0, 2.5, 5.0] {
            approx_eq(erf(-x), -erf(x), 1e-15);
        }
    }

    #[test]
    fn erfc_small_x() {
        // erfc(0) = 1
        approx_eq(erfc(0.0_f64), 1.0, 1e-16);
        // erfc(0.5) ≈ 0.4795001221869535
        approx_eq(erfc(0.5_f64), 0.4795001221869535, 1e-13);
    }

    #[test]
    fn erfc_large_x() {
        // erfc(6) ≈ 0 (to machine precision)
        let val = erfc(6.0_f64);
        assert!(val >= 0.0 && val < 1e-15);
    }

    #[test]
    fn erf_plus_erfc_identity() {
        // erf(x) + erfc(x) = 1
        for &x in &[-2.0, -0.5, 0.0, 0.3, 1.0, 3.0, 5.0] {
            approx_eq(erf(x) + erfc(x), 1.0, 1e-14);
        }
    }

    #[test]
    fn erf_gamma_inc_cross_check() {
        // erf(x) = P(0.5, x²) for x > 0
        for &x in &[0.5, 1.0, 2.0, 3.0] {
            let from_erf = erf(x);
            let from_inc = gamma_inc(0.5_f64, x * x).unwrap();
            approx_eq(from_erf, from_inc, 1e-12);
        }
    }

    #[test]
    fn erf_f32() {
        approx_eq_f32(erf(0.0_f32), 0.0, 1e-7);
        approx_eq_f32(erf(1.0_f32), 0.8427008, 1e-5);
    }

    // =====================================================================
    // Cross-function consistency
    // =====================================================================

    #[test]
    fn cross_gamma_lgamma_consistency() {
        for &x in &[0.5_f64, 1.5, 3.7, 8.0, 15.0] {
            let from_lgamma = lgamma(x).exp();
            let from_gamma = gamma(x);
            let rel_err = ((from_lgamma - from_gamma) / from_gamma).abs();
            assert!(rel_err < 1e-12, "inconsistent at x={x}: rel_err={rel_err}");
        }
    }

    #[test]
    fn cross_beta_lgamma_consistency() {
        let a = 4.0_f64;
        let b = 3.0;
        let from_lbeta = lbeta(a, b).exp();
        let from_beta = beta(a, b);
        approx_eq(from_lbeta, from_beta, 1e-13);
    }

    #[test]
    fn cross_erf_gamma_inc_consistency() {
        // erf(x) = P(0.5, x²) — tested more extensively in erf_gamma_inc_cross_check
        // Additional test at x = 1.5
        let x = 1.5_f64;
        approx_eq(erf(x), gamma_inc(0.5, x * x).unwrap(), 1e-12);
    }

    #[test]
    fn cross_digamma_lgamma_numerical_derivative() {
        // ψ(x) ≈ d/dx lgamma(x) via finite differences
        let h = 1e-7_f64;
        for &x in &[0.5, 2.0, 10.0] {
            let numerical = (lgamma(x + h) - lgamma(x - h)) / (2.0 * h);
            let analytical = digamma(x);
            approx_eq(numerical, analytical, 1e-5);
        }
    }
}
