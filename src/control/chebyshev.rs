use crate::traits::FloatScalar;

use super::biquad::{
    bilinear_hp_pair, bilinear_hp_real, bilinear_lp_pair, bilinear_lp_real, BiquadCascade,
};
use super::{validate_design_params, ControlError};

/// Design a Chebyshev Type I lowpass filter as a cascade of `N` biquad sections.
///
/// Chebyshev Type I filters have equiripple behavior in the passband and
/// monotonic rolloff in the stopband. The `ripple_db` parameter controls
/// the maximum passband deviation.
///
/// `N` must equal `ceil(order / 2)`.
///
/// # Arguments
///
/// * `order` — filter order (1, 2, 3, …)
/// * `ripple_db` — maximum passband ripple in dB (must be > 0)
/// * `cutoff` — cutoff frequency in Hz (passband edge, must be in `(0, sample_rate/2)`)
/// * `sample_rate` — sampling rate in Hz
///
/// # Example
///
/// ```
/// use numeris::control::{chebyshev1_lowpass, BiquadCascade};
///
/// // 4th-order, 1 dB ripple, 1 kHz cutoff, 8 kHz sample rate
/// let mut lpf: BiquadCascade<f64, 2> = chebyshev1_lowpass(4, 1.0, 1000.0, 8000.0).unwrap();
/// let y = lpf.tick(1.0);
/// ```
pub fn chebyshev1_lowpass<T: FloatScalar, const N: usize>(
    order: usize,
    ripple_db: T,
    cutoff: T,
    sample_rate: T,
) -> Result<BiquadCascade<T, N>, ControlError> {
    validate_design_params::<T, N>(order, cutoff, sample_rate)?;
    if ripple_db <= T::zero() || !ripple_db.is_finite() {
        return Err(ControlError::InvalidRipple);
    }

    let two = T::one() + T::one();
    let ten = T::from(10.0).unwrap();
    let pi = T::from(core::f64::consts::PI).unwrap();

    // Pre-warp
    let wa = two * sample_rate * (pi * cutoff / sample_rate).tan();
    let c = two * sample_rate;

    let n = order;
    let nf = T::from(n).unwrap();

    // ε = √(10^(Rp/10) − 1)
    let epsilon = (ten.powf(ripple_db / ten) - T::one()).sqrt();
    // a = asinh(1/ε) / n
    let a = (T::one() / epsilon).asinh() / nf;

    let sinh_a = a.sinh();
    let cosh_a = a.cosh();

    let mut sections = [super::biquad::Biquad::passthrough(); N];
    let mut idx = 0;

    let num_pairs = n / 2;
    for k in 0..num_pairs {
        let kf = T::from(k).unwrap();
        // φ_k = π·(2k + 1) / (2n)
        let phi = pi * (two * kf + T::one()) / (two * nf);
        let sigma = -wa * sinh_a * phi.sin();
        let omega = wa * cosh_a * phi.cos();
        sections[idx] = bilinear_lp_pair(sigma, omega, wa, c);
        idx += 1;
    }

    // Odd-order: real pole at σ = −ωa·sinh(a)
    if n % 2 == 1 {
        let sigma = -wa * sinh_a;
        // For Chebyshev odd-order, the real pole magnitude is ωa·sinh(a),
        // so we use that as the "analog cutoff" for the first-order section.
        let wa_real = wa * sinh_a;
        sections[idx] = bilinear_lp_real(sigma, wa_real, c);
    }

    // Normalize DC gain.
    // Odd-order: DC gain should be 1 (the passband peak reaches 0 dB).
    // Even-order: DC gain should be 1/√(1+ε²) = 10^(-Rp/20).
    // The bilinear helpers assume Butterworth-like numerator (wa²), so the raw
    // cascade DC gain is wrong for Chebyshev. Fix by measuring and rescaling.
    let target_dc = if n % 2 == 0 {
        T::one() / (T::one() + epsilon * epsilon).sqrt()
    } else {
        T::one()
    };
    let mut current_dc = T::one();
    for s in sections.iter() {
        let (b, a) = s.coefficients();
        let num_dc = b[0] + b[1] + b[2];
        let den_dc = a[0] + a[1] + a[2];
        current_dc = current_dc * num_dc / den_dc;
    }
    if current_dc.abs() > T::epsilon() {
        let scale = target_dc / current_dc;
        let (b, a) = sections[0].coefficients();
        sections[0] = super::biquad::Biquad::new(
            [b[0] * scale, b[1] * scale, b[2] * scale],
            a,
        );
    }

    Ok(BiquadCascade { sections })
}

/// Design a Chebyshev Type I highpass filter as a cascade of `N` biquad sections.
///
/// `N` must equal `ceil(order / 2)`.
///
/// # Arguments
///
/// * `order` — filter order (1, 2, 3, …)
/// * `ripple_db` — maximum passband ripple in dB (must be > 0)
/// * `cutoff` — cutoff frequency in Hz (must be in `(0, sample_rate/2)`)
/// * `sample_rate` — sampling rate in Hz
///
/// # Example
///
/// ```
/// use numeris::control::{chebyshev1_highpass, BiquadCascade};
///
/// let mut hpf: BiquadCascade<f64, 2> = chebyshev1_highpass(4, 1.0, 1000.0, 8000.0).unwrap();
/// let y = hpf.tick(1.0);
/// ```
pub fn chebyshev1_highpass<T: FloatScalar, const N: usize>(
    order: usize,
    ripple_db: T,
    cutoff: T,
    sample_rate: T,
) -> Result<BiquadCascade<T, N>, ControlError> {
    validate_design_params::<T, N>(order, cutoff, sample_rate)?;
    if ripple_db <= T::zero() || !ripple_db.is_finite() {
        return Err(ControlError::InvalidRipple);
    }

    let two = T::one() + T::one();
    let ten = T::from(10.0).unwrap();
    let pi = T::from(core::f64::consts::PI).unwrap();

    let wa = two * sample_rate * (pi * cutoff / sample_rate).tan();
    let c = two * sample_rate;

    let n = order;
    let nf = T::from(n).unwrap();

    let epsilon = (ten.powf(ripple_db / ten) - T::one()).sqrt();
    let a = (T::one() / epsilon).asinh() / nf;

    let sinh_a = a.sinh();
    let cosh_a = a.cosh();

    let mut sections = [super::biquad::Biquad::passthrough(); N];
    let mut idx = 0;

    // Compute analog LP prototype poles, then invert for HP: p_hp = ωa²/p_lp.
    // For conjugate pair p = σ + jω:
    //   p_hp = ωa²·(σ - jω) / (σ² + ω²)
    //   → σ_hp = ωa²·σ / (σ² + ω²), ω_hp = -ωa²·ω / (σ² + ω²)
    let wa2 = wa * wa;

    let num_pairs = n / 2;
    for k in 0..num_pairs {
        let kf = T::from(k).unwrap();
        let phi = pi * (two * kf + T::one()) / (two * nf);
        let sigma_lp = -wa * sinh_a * phi.sin();
        let omega_lp = wa * cosh_a * phi.cos();
        let mag2 = sigma_lp * sigma_lp + omega_lp * omega_lp;
        let sigma_hp = wa2 * sigma_lp / mag2;
        let omega_hp = -wa2 * omega_lp / mag2;
        // The HP analog transfer function for this section has zeros at s=0 (numerator = s²)
        // and poles at σ_hp ± j·ω_hp. Use the HP bilinear helper.
        sections[idx] = bilinear_hp_pair(sigma_hp, omega_hp, wa, c);
        idx += 1;
    }

    // Odd-order: real LP pole at σ = −ωa·sinh(a) → HP pole at ωa²/σ = −ωa/sinh(a)
    if n % 2 == 1 {
        let sigma_lp = -wa * sinh_a;
        let sigma_hp = wa2 / sigma_lp; // = -wa / sinh(a)
        sections[idx] = bilinear_hp_real(sigma_hp, wa, c);
    }

    // Normalize passband gain.
    // Odd-order: Nyquist gain should be 1 (passband peak at 0 dB).
    // Even-order: Nyquist gain should be 1/√(1+ε²).
    let target_nyq = if n % 2 == 0 {
        T::one() / (T::one() + epsilon * epsilon).sqrt()
    } else {
        T::one()
    };
    // Evaluate at z = -1 (Nyquist)
    let mut current_nyq = T::one();
    for s in sections.iter() {
        let (b, a) = s.coefficients();
        let num_nyq = b[0] - b[1] + b[2];
        let den_nyq = a[0] - a[1] + a[2];
        current_nyq = current_nyq * num_nyq / den_nyq;
    }
    if current_nyq.abs() > T::epsilon() {
        let scale = target_nyq / current_nyq.abs();
        let (b, a) = sections[0].coefficients();
        sections[0] = super::biquad::Biquad::new(
            [b[0] * scale, b[1] * scale, b[2] * scale],
            a,
        );
    }

    Ok(BiquadCascade { sections })
}
