use super::*;

// ======================== RNG ========================

#[test]
fn rng_reproducible() {
    let mut a = Rng::new(42);
    let mut b = Rng::new(42);
    for _ in 0..100 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn rng_different_seeds() {
    let mut a = Rng::new(1);
    let mut b = Rng::new(2);
    // Extremely unlikely to match
    assert_ne!(a.next_u64(), b.next_u64());
}

#[test]
fn rng_f64_range() {
    let mut rng = Rng::new(123);
    for _ in 0..10000 {
        let x = rng.next_f64();
        assert!(x >= 0.0 && x < 1.0);
    }
}

#[test]
fn rng_f32_range() {
    let mut rng = Rng::new(456);
    for _ in 0..10000 {
        let x = rng.next_f32();
        assert!(x >= 0.0 && x < 1.0);
    }
}

#[test]
fn rng_normal_mean_variance() {
    let mut rng = Rng::new(789);
    let n = 50000;
    let mut sum = 0.0_f64;
    let mut sum2 = 0.0_f64;
    for _ in 0..n {
        let x = rng.next_normal_f64();
        sum += x;
        sum2 += x * x;
    }
    let mean = sum / n as f64;
    let var = sum2 / n as f64 - mean * mean;
    assert!(mean.abs() < 0.03, "normal mean={mean}");
    assert!((var - 1.0).abs() < 0.05, "normal var={var}");
}

// ======================== Sampling: continuous ========================

#[test]
fn sample_normal() {
    let dist = Normal::new(5.0_f64, 2.0).unwrap();
    let mut rng = Rng::new(100);
    let n = 20000;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for _ in 0..n {
        let x = dist.sample(&mut rng);
        sum += x;
        sum2 += x * x;
    }
    let mean = sum / n as f64;
    let var = sum2 / n as f64 - mean * mean;
    assert!((mean - 5.0).abs() < 0.1, "normal mean={mean}");
    assert!((var - 4.0).abs() < 0.2, "normal var={var}");
}

#[test]
fn sample_normal_f32() {
    let dist = Normal::new(0.0_f32, 1.0).unwrap();
    let mut rng = Rng::new(101);
    let n = 10000;
    let mut sum = 0.0_f32;
    for _ in 0..n {
        let x = dist.sample(&mut rng);
        sum += x;
    }
    let mean = sum / n as f32;
    assert!(mean.abs() < 0.1, "f32 normal mean={mean}");
}

#[test]
fn sample_uniform() {
    let dist = Uniform::new(2.0_f64, 5.0).unwrap();
    let mut rng = Rng::new(200);
    let n = 20000;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for _ in 0..n {
        let x = dist.sample(&mut rng);
        assert!(x >= 2.0 && x < 5.0, "uniform out of range: {x}");
        sum += x;
        sum2 += x * x;
    }
    let mean = sum / n as f64;
    let var = sum2 / n as f64 - mean * mean;
    assert!((mean - 3.5).abs() < 0.1, "uniform mean={mean}");
    assert!((var - 0.75).abs() < 0.1, "uniform var={var}");
}

#[test]
fn sample_exponential() {
    let dist = Exponential::new(2.0_f64).unwrap();
    let mut rng = Rng::new(300);
    let n = 20000;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for _ in 0..n {
        let x = dist.sample(&mut rng);
        assert!(x >= 0.0, "exp negative: {x}");
        sum += x;
        sum2 += x * x;
    }
    let mean = sum / n as f64;
    let var = sum2 / n as f64 - mean * mean;
    assert!((mean - 0.5).abs() < 0.03, "exp mean={mean}");
    assert!((var - 0.25).abs() < 0.03, "exp var={var}");
}

#[test]
fn sample_gamma() {
    let dist = Gamma::new(3.0_f64, 2.0).unwrap();
    let mut rng = Rng::new(400);
    let n = 20000;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for _ in 0..n {
        let x = dist.sample(&mut rng);
        assert!(x > 0.0, "gamma non-positive: {x}");
        sum += x;
        sum2 += x * x;
    }
    let mean = sum / n as f64;
    let var = sum2 / n as f64 - mean * mean;
    // Gamma(3,2): mean=1.5, var=0.75
    assert!((mean - 1.5).abs() < 0.05, "gamma mean={mean}");
    assert!((var - 0.75).abs() < 0.1, "gamma var={var}");
}

#[test]
fn sample_gamma_small_shape() {
    // shape < 1: uses the Gamma(alpha+1) * U^(1/alpha) trick
    let dist = Gamma::new(0.5_f64, 1.0).unwrap();
    let mut rng = Rng::new(401);
    let n = 20000;
    let mut sum = 0.0;
    for _ in 0..n {
        let x = dist.sample(&mut rng);
        assert!(x > 0.0);
        sum += x;
    }
    let mean = sum / n as f64;
    assert!((mean - 0.5).abs() < 0.05, "gamma(0.5,1) mean={mean}");
}

#[test]
fn sample_beta() {
    let dist = Beta::new(2.0_f64, 5.0).unwrap();
    let mut rng = Rng::new(500);
    let n = 20000;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for _ in 0..n {
        let x = dist.sample(&mut rng);
        assert!(x > 0.0 && x < 1.0, "beta out of [0,1]: {x}");
        sum += x;
        sum2 += x * x;
    }
    let mean = sum / n as f64;
    let var = sum2 / n as f64 - mean * mean;
    // Beta(2,5): mean=2/7≈0.2857, var=10/392≈0.02551
    let expected_mean = 2.0 / 7.0;
    let expected_var = 2.0 * 5.0 / (49.0 * 8.0);
    assert!((mean - expected_mean).abs() < 0.02, "beta mean={mean}");
    assert!((var - expected_var).abs() < 0.01, "beta var={var}");
}

#[test]
fn sample_chi_squared() {
    let dist = ChiSquared::new(5.0_f64).unwrap();
    let mut rng = Rng::new(600);
    let n = 20000;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for _ in 0..n {
        let x = dist.sample(&mut rng);
        assert!(x > 0.0);
        sum += x;
        sum2 += x * x;
    }
    let mean = sum / n as f64;
    let var = sum2 / n as f64 - mean * mean;
    assert!((mean - 5.0).abs() < 0.2, "chi2 mean={mean}");
    assert!((var - 10.0).abs() < 0.6, "chi2 var={var}");
}

#[test]
fn sample_student_t() {
    let dist = StudentT::new(10.0_f64).unwrap();
    let mut rng = Rng::new(700);
    let n = 20000;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for _ in 0..n {
        let x = dist.sample(&mut rng);
        sum += x;
        sum2 += x * x;
    }
    let mean = sum / n as f64;
    let var = sum2 / n as f64 - mean * mean;
    // t(10): mean=0, var=10/8=1.25
    assert!(mean.abs() < 0.1, "t mean={mean}");
    assert!((var - 1.25).abs() < 0.2, "t var={var}");
}

// ======================== Sampling: discrete ========================

#[test]
fn sample_bernoulli() {
    let dist = Bernoulli::new(0.3_f64).unwrap();
    let mut rng = Rng::new(800);
    let n = 20000;
    let mut ones = 0u64;
    for _ in 0..n {
        let x = dist.sample(&mut rng);
        assert!(x == 0 || x == 1);
        ones += x;
    }
    let mean = ones as f64 / n as f64;
    assert!((mean - 0.3).abs() < 0.03, "bernoulli mean={mean}");
}

#[test]
fn sample_binomial_small() {
    let dist = Binomial::new(10, 0.4_f64).unwrap();
    let mut rng = Rng::new(900);
    let n = 20000;
    let mut sum = 0u64;
    let mut sum2 = 0u64;
    for _ in 0..n {
        let x = dist.sample(&mut rng);
        assert!(x <= 10);
        sum += x;
        sum2 += x * x;
    }
    let mean = sum as f64 / n as f64;
    let var = sum2 as f64 / n as f64 - mean * mean;
    // B(10, 0.4): mean=4, var=2.4
    assert!((mean - 4.0).abs() < 0.15, "binomial mean={mean}");
    assert!((var - 2.4).abs() < 0.3, "binomial var={var}");
}

#[test]
fn sample_binomial_large() {
    // Large n triggers normal approximation path
    let dist = Binomial::new(100, 0.5_f64).unwrap();
    let mut rng = Rng::new(901);
    let n = 10000;
    let mut sum = 0u64;
    for _ in 0..n {
        let x = dist.sample(&mut rng);
        assert!(x <= 100);
        sum += x;
    }
    let mean = sum as f64 / n as f64;
    assert!((mean - 50.0).abs() < 1.0, "binomial(100,0.5) mean={mean}");
}

#[test]
fn sample_poisson_small() {
    let dist = Poisson::new(3.0_f64).unwrap();
    let mut rng = Rng::new(1000);
    let n = 20000;
    let mut sum = 0u64;
    let mut sum2 = 0u64;
    for _ in 0..n {
        let x = dist.sample(&mut rng);
        sum += x;
        sum2 += x * x;
    }
    let mean = sum as f64 / n as f64;
    let var = sum2 as f64 / n as f64 - mean * mean;
    assert!((mean - 3.0).abs() < 0.1, "poisson mean={mean}");
    assert!((var - 3.0).abs() < 0.2, "poisson var={var}");
}

#[test]
fn sample_poisson_large() {
    // lambda >= 30 triggers normal approximation
    let dist = Poisson::new(50.0_f64).unwrap();
    let mut rng = Rng::new(1001);
    let n = 10000;
    let mut sum = 0u64;
    for _ in 0..n {
        sum += dist.sample(&mut rng);
    }
    let mean = sum as f64 / n as f64;
    assert!((mean - 50.0).abs() < 1.5, "poisson(50) mean={mean}");
}

// ======================== sample_array ========================

#[test]
fn sample_array_normal() {
    let dist = Normal::new(0.0_f64, 1.0).unwrap();
    let mut rng = Rng::new(1100);
    let arr: [f64; 5] = dist.sample_array(&mut rng);
    // Just check we get 5 finite values
    for &x in &arr {
        assert!(x.is_finite());
    }
}

#[test]
fn sample_array_bernoulli() {
    let dist = Bernoulli::new(0.5_f64).unwrap();
    let mut rng = Rng::new(1101);
    let arr: [u64; 10] = dist.sample_array(&mut rng);
    for &x in &arr {
        assert!(x == 0 || x == 1);
    }
}

#[test]
fn sample_array_zero_length() {
    let dist = Normal::new(0.0_f64, 1.0).unwrap();
    let mut rng = Rng::new(1102);
    let arr: [f64; 0] = dist.sample_array(&mut rng);
    assert_eq!(arr.len(), 0);
}

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

// ======================== Discrete quantile ========================

#[test]
fn bernoulli_quantile() {
    let b = Bernoulli::new(0.3_f64).unwrap();
    assert_eq!(b.quantile(0.0), 0);
    assert_eq!(b.quantile(0.5), 0);   // CDF(0) = 0.7 >= 0.5
    assert_eq!(b.quantile(0.7), 0);   // CDF(0) = 0.7 >= 0.7
    assert_eq!(b.quantile(0.71), 1);  // CDF(0) = 0.7 < 0.71
    assert_eq!(b.quantile(1.0), 1);
}

#[test]
fn bernoulli_quantile_degenerate() {
    // Bernoulli(0): always returns 0, so CDF(0) = 1 for any p
    let b0 = Bernoulli::new(0.0_f64).unwrap();
    assert_eq!(b0.quantile(0.5), 0);
    assert_eq!(b0.quantile(1.0), 0);
    // Bernoulli(1): CDF(0) = 0, so quantile > 0 for any p > 0
    let b1 = Bernoulli::new(1.0_f64).unwrap();
    assert_eq!(b1.quantile(0.0), 0);
    assert_eq!(b1.quantile(0.5), 1);
    assert_eq!(b1.quantile(1.0), 1);
}

#[test]
fn binomial_quantile() {
    let b = Binomial::new(10, 0.5_f64).unwrap();
    assert_eq!(b.quantile(0.0), 0);
    assert_eq!(b.quantile(1.0), 10);
    // Verify minimality: CDF(k) >= p and CDF(k-1) < p
    for &p in &[0.1, 0.25, 0.5, 0.75, 0.9_f64] {
        let k = b.quantile(p);
        assert!(b.cdf(k) >= p, "p={p}: CDF({k})={} < p", b.cdf(k));
        if k > 0 {
            assert!(b.cdf(k - 1) < p, "p={p}: CDF({}) >= p (not minimal)", k - 1);
        }
    }
}

#[test]
fn binomial_quantile_skewed() {
    // B(20, 0.1): heavy right-skew
    let b = Binomial::new(20, 0.1_f64).unwrap();
    for &p in &[0.05, 0.25, 0.5, 0.75, 0.95_f64] {
        let k = b.quantile(p);
        assert!(b.cdf(k) >= p);
        if k > 0 {
            assert!(b.cdf(k - 1) < p);
        }
    }
}

#[test]
fn poisson_quantile() {
    let dist = Poisson::new(3.0_f64).unwrap();
    assert_eq!(dist.quantile(0.0), 0);
    for &p in &[0.1, 0.25, 0.5, 0.75, 0.9_f64] {
        let k = dist.quantile(p);
        assert!(dist.cdf(k) >= p, "p={p}: CDF({k})={} < p", dist.cdf(k));
        if k > 0 {
            assert!(dist.cdf(k - 1) < p, "p={p}: CDF({}) >= p (not minimal)", k - 1);
        }
    }
}

#[test]
fn poisson_quantile_large_lambda() {
    // Ensure the normal approximation starting point is accurate for large lambda
    let dist = Poisson::new(100.0_f64).unwrap();
    for &p in &[0.1, 0.5, 0.9_f64] {
        let k = dist.quantile(p);
        assert!(dist.cdf(k) >= p);
        if k > 0 {
            assert!(dist.cdf(k - 1) < p);
        }
    }
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
