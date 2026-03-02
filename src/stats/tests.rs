use super::*;

// ======================== Normal ========================

#[test]
fn normal_pdf_standard() {
    let n = Normal::new(0.0_f64, 1.0).unwrap();
    let expected = 1.0 / (2.0 * core::f64::consts::PI).sqrt();
    assert!((n.pdf(0.0) - expected).abs() < 1e-14);
}

#[test]
fn normal_cdf_standard() {
    let n = Normal::new(0.0_f64, 1.0).unwrap();
    assert!((n.cdf(0.0) - 0.5).abs() < 1e-14);
    // Φ(1) ≈ 0.8413
    assert!((n.cdf(1.0) - 0.8413447460685429).abs() < 1e-10);
    // Φ(-1) ≈ 0.1587
    assert!((n.cdf(-1.0) - 0.15865525393145702).abs() < 1e-10);
}

#[test]
fn normal_quantile() {
    let n = Normal::new(0.0_f64, 1.0).unwrap();
    assert!((n.quantile(0.5)).abs() < 1e-8);
    assert!((n.quantile(0.975) - 1.959964).abs() < 1e-4);
    assert!((n.quantile(0.025) + 1.959964).abs() < 1e-4);
}

#[test]
fn normal_mean_variance() {
    let n = Normal::new(3.0_f64, 2.0).unwrap();
    assert!((n.mean() - 3.0).abs() < 1e-14);
    assert!((n.variance() - 4.0).abs() < 1e-14);
}

#[test]
fn normal_ln_pdf() {
    let n = Normal::new(0.0_f64, 1.0).unwrap();
    assert!((n.ln_pdf(0.0) - n.pdf(0.0).ln()).abs() < 1e-14);
    assert!((n.ln_pdf(3.0) - n.pdf(3.0).ln()).abs() < 1e-13);
}

#[test]
fn normal_invalid() {
    assert_eq!(Normal::new(0.0_f64, 0.0).unwrap_err(), StatsError::InvalidParameter);
    assert_eq!(Normal::new(0.0_f64, -1.0).unwrap_err(), StatsError::InvalidParameter);
}

#[test]
fn normal_f32() {
    let n = Normal::new(0.0_f32, 1.0).unwrap();
    assert!((n.cdf(0.0) - 0.5).abs() < 1e-5);
    assert!((n.mean()).abs() < 1e-7);
}

// ======================== Uniform ========================

#[test]
fn uniform_pdf_cdf() {
    let u = Uniform::new(0.0_f64, 1.0).unwrap();
    assert!((u.pdf(0.5) - 1.0).abs() < 1e-14);
    assert!((u.pdf(-0.1)).abs() < 1e-14);
    assert!((u.pdf(1.1)).abs() < 1e-14);
    assert!((u.cdf(0.5) - 0.5).abs() < 1e-14);
    assert!((u.cdf(-0.1)).abs() < 1e-14);
    assert!((u.cdf(1.1) - 1.0).abs() < 1e-14);
}

#[test]
fn uniform_quantile() {
    let u = Uniform::new(2.0_f64, 5.0).unwrap();
    assert!((u.quantile(0.0) - 2.0).abs() < 1e-14);
    assert!((u.quantile(1.0) - 5.0).abs() < 1e-14);
    assert!((u.quantile(0.5) - 3.5).abs() < 1e-14);
}

#[test]
fn uniform_mean_variance() {
    let u = Uniform::new(0.0_f64, 12.0).unwrap();
    assert!((u.mean() - 6.0).abs() < 1e-14);
    assert!((u.variance() - 12.0).abs() < 1e-14);
}

#[test]
fn uniform_invalid() {
    assert_eq!(Uniform::new(1.0_f64, 1.0).unwrap_err(), StatsError::InvalidParameter);
    assert_eq!(Uniform::new(2.0_f64, 1.0).unwrap_err(), StatsError::InvalidParameter);
}

// ======================== Exponential ========================

#[test]
fn exponential_pdf_cdf() {
    let e = Exponential::new(1.0_f64).unwrap();
    assert!((e.pdf(0.0) - 1.0).abs() < 1e-14);
    assert!((e.pdf(1.0) - (-1.0_f64).exp()).abs() < 1e-14);
    assert!((e.cdf(0.0)).abs() < 1e-14);
    assert!((e.cdf(1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-14);
}

#[test]
fn exponential_quantile() {
    let e = Exponential::new(2.0_f64).unwrap();
    assert!((e.quantile(0.0)).abs() < 1e-14);
    let q = e.quantile(0.5);
    assert!((e.cdf(q) - 0.5).abs() < 1e-12);
}

#[test]
fn exponential_mean_variance() {
    let e = Exponential::new(0.5_f64).unwrap();
    assert!((e.mean() - 2.0).abs() < 1e-14);
    assert!((e.variance() - 4.0).abs() < 1e-14);
}

#[test]
fn exponential_invalid() {
    assert_eq!(Exponential::new(0.0_f64).unwrap_err(), StatsError::InvalidParameter);
    assert_eq!(Exponential::new(-1.0_f64).unwrap_err(), StatsError::InvalidParameter);
}

// ======================== Gamma ========================

#[test]
fn gamma_pdf_at_mode() {
    // Gamma(2, 1): mode at x = (α-1)/β = 1
    let g = Gamma::new(2.0_f64, 1.0).unwrap();
    // pdf(1) = 1^1 * exp(-1) / Γ(2) = 1/e
    assert!((g.pdf(1.0) - (-1.0_f64).exp()).abs() < 1e-14);
}

#[test]
fn gamma_cdf() {
    // Gamma(1, 1) = Exponential(1)
    let g = Gamma::new(1.0_f64, 1.0).unwrap();
    assert!((g.cdf(1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
}

#[test]
fn gamma_quantile_roundtrip() {
    let g = Gamma::new(3.0_f64, 2.0).unwrap();
    for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
        let x = g.quantile(p);
        assert!((g.cdf(x) - p).abs() < 1e-8, "p={p}: cdf(quantile(p))={}", g.cdf(x));
    }
}

#[test]
fn gamma_mean_variance() {
    let g = Gamma::new(5.0_f64, 2.0).unwrap();
    assert!((g.mean() - 2.5).abs() < 1e-14);
    assert!((g.variance() - 1.25).abs() < 1e-14);
}

#[test]
fn gamma_invalid() {
    assert_eq!(Gamma::new(0.0_f64, 1.0).unwrap_err(), StatsError::InvalidParameter);
    assert_eq!(Gamma::new(1.0_f64, 0.0).unwrap_err(), StatsError::InvalidParameter);
}

// ======================== Beta ========================

#[test]
fn beta_uniform_case() {
    // Beta(1, 1) = Uniform(0, 1)
    let b = Beta::new(1.0_f64, 1.0).unwrap();
    assert!((b.pdf(0.5) - 1.0).abs() < 1e-14);
    assert!((b.cdf(0.5) - 0.5).abs() < 1e-12);
}

#[test]
fn beta_symmetric() {
    // Beta(2, 2): symmetric around 0.5
    let b = Beta::new(2.0_f64, 2.0).unwrap();
    assert!((b.cdf(0.5) - 0.5).abs() < 1e-12);
    assert!((b.mean() - 0.5).abs() < 1e-14);
}

#[test]
fn beta_quantile_roundtrip() {
    let b = Beta::new(2.0_f64, 5.0).unwrap();
    for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
        let x = b.quantile(p);
        assert!((b.cdf(x) - p).abs() < 1e-8, "p={p}: cdf(quantile(p))={}", b.cdf(x));
    }
}

#[test]
fn beta_mean_variance() {
    let b = Beta::new(2.0_f64, 3.0).unwrap();
    assert!((b.mean() - 0.4).abs() < 1e-14);
    assert!((b.variance() - 0.04).abs() < 1e-14); // 2*3 / (25*6) = 6/150 = 0.04
}

#[test]
fn beta_invalid() {
    assert_eq!(Beta::new(0.0_f64, 1.0).unwrap_err(), StatsError::InvalidParameter);
    assert_eq!(Beta::new(1.0_f64, 0.0).unwrap_err(), StatsError::InvalidParameter);
}

// ======================== ChiSquared ========================

#[test]
fn chi_squared_cdf() {
    // χ²(2) CDF at x has closed form: 1 - exp(-x/2)
    let chi2 = ChiSquared::new(2.0_f64).unwrap();
    assert!((chi2.cdf(2.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
}

#[test]
fn chi_squared_quantile_roundtrip() {
    let chi2 = ChiSquared::new(5.0_f64).unwrap();
    for &p in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.95] {
        let x = chi2.quantile(p);
        assert!(
            (chi2.cdf(x) - p).abs() < 1e-8,
            "p={p}: cdf(quantile(p))={}",
            chi2.cdf(x)
        );
    }
}

#[test]
fn chi_squared_mean_variance() {
    let chi2 = ChiSquared::new(10.0_f64).unwrap();
    assert!((chi2.mean() - 10.0).abs() < 1e-14);
    assert!((chi2.variance() - 20.0).abs() < 1e-14);
}

#[test]
fn chi_squared_invalid() {
    assert_eq!(ChiSquared::new(0.0_f64).unwrap_err(), StatsError::InvalidParameter);
}

// ======================== StudentT ========================

#[test]
fn student_t_symmetric() {
    let t = StudentT::new(5.0_f64).unwrap();
    assert!((t.cdf(0.0) - 0.5).abs() < 1e-12);
    // Symmetry: cdf(-x) = 1 - cdf(x)
    assert!((t.cdf(-2.0) + t.cdf(2.0) - 1.0).abs() < 1e-12);
}

#[test]
fn student_t_pdf_symmetric() {
    let t = StudentT::new(5.0_f64).unwrap();
    assert!((t.pdf(-1.5) - t.pdf(1.5)).abs() < 1e-14);
}

#[test]
fn student_t_quantile_roundtrip() {
    let t = StudentT::new(10.0_f64).unwrap();
    for &p in &[0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975] {
        let x = t.quantile(p);
        assert!(
            (t.cdf(x) - p).abs() < 1e-7,
            "p={p}: cdf(quantile(p))={}",
            t.cdf(x)
        );
    }
}

#[test]
fn student_t_large_df_approaches_normal() {
    let t = StudentT::new(1000.0_f64).unwrap();
    let n = Normal::new(0.0, 1.0).unwrap();
    // CDF should be very close to normal for large df
    for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
        assert!(
            (t.cdf(x) - n.cdf(x)).abs() < 1e-3,
            "x={x}: t={}, n={}",
            t.cdf(x),
            n.cdf(x)
        );
    }
}

#[test]
fn student_t_mean_variance() {
    let t = StudentT::new(5.0_f64).unwrap();
    assert!((t.mean()).abs() < 1e-14);
    assert!((t.variance() - 5.0 / 3.0).abs() < 1e-14);
}

#[test]
fn student_t_invalid() {
    assert_eq!(StudentT::new(0.0_f64).unwrap_err(), StatsError::InvalidParameter);
}

// ======================== Bernoulli ========================

#[test]
fn bernoulli_pmf() {
    let b = Bernoulli::new(0.3_f64).unwrap();
    assert!((b.pmf(0) - 0.7).abs() < 1e-14);
    assert!((b.pmf(1) - 0.3).abs() < 1e-14);
    assert!((b.pmf(2)).abs() < 1e-14);
}

#[test]
fn bernoulli_cdf() {
    let b = Bernoulli::new(0.4_f64).unwrap();
    assert!((b.cdf(0) - 0.6).abs() < 1e-14);
    assert!((b.cdf(1) - 1.0).abs() < 1e-14);
}

#[test]
fn bernoulli_mean_variance() {
    let b = Bernoulli::new(0.25_f64).unwrap();
    assert!((b.mean() - 0.25).abs() < 1e-14);
    assert!((b.variance() - 0.1875).abs() < 1e-14);
}

#[test]
fn bernoulli_invalid() {
    assert_eq!(Bernoulli::new(-0.1_f64).unwrap_err(), StatsError::InvalidParameter);
    assert_eq!(Bernoulli::new(1.1_f64).unwrap_err(), StatsError::InvalidParameter);
}

#[test]
fn bernoulli_edge_cases() {
    let b0 = Bernoulli::new(0.0_f64).unwrap();
    assert!((b0.pmf(0) - 1.0).abs() < 1e-14);
    assert!((b0.pmf(1)).abs() < 1e-14);
    let b1 = Bernoulli::new(1.0_f64).unwrap();
    assert!((b1.pmf(0)).abs() < 1e-14);
    assert!((b1.pmf(1) - 1.0).abs() < 1e-14);
}

// ======================== Binomial ========================

#[test]
fn binomial_pmf() {
    // B(3, 0.5): P(X=0) = 1/8, P(X=1) = 3/8, P(X=2) = 3/8, P(X=3) = 1/8
    let b = Binomial::new(3, 0.5_f64).unwrap();
    assert!((b.pmf(0) - 0.125).abs() < 1e-12);
    assert!((b.pmf(1) - 0.375).abs() < 1e-12);
    assert!((b.pmf(2) - 0.375).abs() < 1e-12);
    assert!((b.pmf(3) - 0.125).abs() < 1e-12);
    assert!((b.pmf(4)).abs() < 1e-14);
}

#[test]
fn binomial_cdf() {
    let b = Binomial::new(3, 0.5_f64).unwrap();
    assert!((b.cdf(0) - 0.125).abs() < 1e-10);
    assert!((b.cdf(1) - 0.5).abs() < 1e-10);
    assert!((b.cdf(2) - 0.875).abs() < 1e-10);
    assert!((b.cdf(3) - 1.0).abs() < 1e-14);
}

#[test]
fn binomial_cdf_via_pmf_sum() {
    // Verify CDF matches sum of PMFs
    let b = Binomial::new(10, 0.3_f64).unwrap();
    for k in 0..=10 {
        let cdf = b.cdf(k);
        let pmf_sum: f64 = (0..=k).map(|j| b.pmf(j)).sum();
        assert!(
            (cdf - pmf_sum).abs() < 1e-10,
            "k={k}: cdf={cdf}, pmf_sum={pmf_sum}"
        );
    }
}

#[test]
fn binomial_mean_variance() {
    let b = Binomial::new(20, 0.3_f64).unwrap();
    assert!((b.mean() - 6.0).abs() < 1e-14);
    assert!((b.variance() - 4.2).abs() < 1e-14);
}

#[test]
fn binomial_invalid() {
    assert_eq!(Binomial::new(10, -0.1_f64).unwrap_err(), StatsError::InvalidParameter);
    assert_eq!(Binomial::new(10, 1.1_f64).unwrap_err(), StatsError::InvalidParameter);
}

// ======================== Poisson ========================

#[test]
fn poisson_pmf() {
    let p = Poisson::new(1.0_f64).unwrap();
    // P(0) = e^{-1}
    assert!((p.pmf(0) - (-1.0_f64).exp()).abs() < 1e-14);
    // P(1) = e^{-1}
    assert!((p.pmf(1) - (-1.0_f64).exp()).abs() < 1e-14);
    // P(2) = e^{-1}/2
    assert!((p.pmf(2) - (-1.0_f64).exp() / 2.0).abs() < 1e-14);
}

#[test]
fn poisson_cdf() {
    let p = Poisson::new(3.0_f64).unwrap();
    // Verify CDF matches sum of PMFs
    for k in 0..15 {
        let cdf = p.cdf(k);
        let pmf_sum: f64 = (0..=k).map(|j| p.pmf(j)).sum();
        assert!(
            (cdf - pmf_sum).abs() < 1e-10,
            "k={k}: cdf={cdf}, pmf_sum={pmf_sum}"
        );
    }
}

#[test]
fn poisson_mean_variance() {
    let p = Poisson::new(5.5_f64).unwrap();
    assert!((p.mean() - 5.5).abs() < 1e-14);
    assert!((p.variance() - 5.5).abs() < 1e-14);
}

#[test]
fn poisson_invalid() {
    assert_eq!(Poisson::new(0.0_f64).unwrap_err(), StatsError::InvalidParameter);
    assert_eq!(Poisson::new(-1.0_f64).unwrap_err(), StatsError::InvalidParameter);
}

// ======================== Cross-distribution ========================

#[test]
fn gamma_exponential_equivalence() {
    // Gamma(1, λ) = Exponential(λ)
    let g = Gamma::new(1.0_f64, 2.0).unwrap();
    let e = Exponential::new(2.0).unwrap();
    for &x in &[0.0, 0.5, 1.0, 2.0, 5.0] {
        assert!(
            (g.pdf(x) - e.pdf(x)).abs() < 1e-12,
            "pdf at {x}: {} vs {}",
            g.pdf(x),
            e.pdf(x)
        );
        assert!(
            (g.cdf(x) - e.cdf(x)).abs() < 1e-12,
            "cdf at {x}: {} vs {}",
            g.cdf(x),
            e.cdf(x)
        );
    }
}

#[test]
fn chi_squared_gamma_equivalence() {
    // χ²(k) = Gamma(k/2, 1/2)
    let chi2 = ChiSquared::new(6.0_f64).unwrap();
    let g = Gamma::new(3.0, 0.5).unwrap();
    for &x in &[1.0, 3.0, 5.0, 10.0] {
        assert!(
            (chi2.pdf(x) - g.pdf(x)).abs() < 1e-12,
            "pdf at {x}: {} vs {}",
            chi2.pdf(x),
            g.pdf(x)
        );
        assert!(
            (chi2.cdf(x) - g.cdf(x)).abs() < 1e-12,
            "cdf at {x}: {} vs {}",
            chi2.cdf(x),
            g.cdf(x)
        );
    }
}

#[test]
fn error_display() {
    use core::fmt::Write;
    let mut s = alloc::string::String::new();
    write!(s, "{}", StatsError::InvalidParameter).unwrap();
    assert!(s.contains("parameter"));
}
