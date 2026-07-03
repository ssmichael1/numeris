use crate::traits::FloatScalar;

use super::biquad::{
    assemble_cascade, bilinear_hp_pair, bilinear_hp_real, bilinear_lp_pair, bilinear_lp_real,
    cascade_gain_at, prewarp, scale_first_section_gain, BiquadCascade,
};
use super::{validate_design_params, ControlError};

/// Shared Chebyshev Type I prototype parameters `(ε, sinh a, cosh a)`, where
/// ε = √(10^(Rp/10) − 1) and a = asinh(1/ε) / n.
fn chebyshev_params<T: FloatScalar>(order: usize, ripple_db: T) -> (T, T, T) {
    let ten = T::from(10.0).unwrap();
    let nf = T::from(order).unwrap();
    let epsilon = (ten.powf(ripple_db / ten) - T::one()).sqrt();
    let a = (T::one() / epsilon).asinh() / nf;
    (epsilon, a.sinh(), a.cosh())
}

/// Analog LP prototype conjugate-pole pair `(σ, ω)` of the `k`-th Chebyshev
/// section, scaled to the pre-warped cutoff `wa`. φ_k = π·(2k + 1) / (2n).
fn chebyshev_lp_pole<T: FloatScalar>(
    order: usize,
    k: usize,
    wa: T,
    sinh_a: T,
    cosh_a: T,
) -> (T, T) {
    let two = T::one() + T::one();
    let pi = T::from(core::f64::consts::PI).unwrap();
    let nf = T::from(order).unwrap();
    let kf = T::from(k).unwrap();
    let phi = pi * (two * kf + T::one()) / (two * nf);
    (-wa * sinh_a * phi.sin(), wa * cosh_a * phi.cos())
}

/// Passband gain target: 1 for odd orders, 1/√(1+ε²) for even orders (where the
/// equiripple response starts at its lower edge rather than 0 dB).
fn passband_target<T: FloatScalar>(order: usize, epsilon: T) -> T {
    if order % 2 == 0 {
        T::one() / (T::one() + epsilon * epsilon).sqrt()
    } else {
        T::one()
    }
}

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

    let (wa, c) = prewarp(cutoff, sample_rate);
    let (epsilon, sinh_a, cosh_a) = chebyshev_params(order, ripple_db);

    let mut cascade = assemble_cascade::<T, N>(
        order,
        |k| {
            let (sigma, omega) = chebyshev_lp_pole(order, k, wa, sinh_a, cosh_a);
            bilinear_lp_pair(sigma, omega, wa, c)
        },
        || {
            // Odd-order real pole at σ = −ωa·sinh(a); its magnitude ωa·sinh(a)
            // is the analog cutoff of the first-order section.
            bilinear_lp_real(-wa * sinh_a, wa * sinh_a, c)
        },
    );

    // The bilinear helpers assume a Butterworth-like numerator (wa²), so the raw
    // cascade DC gain is wrong for Chebyshev. Rescale to the passband target.
    let current_dc = cascade_gain_at(&cascade.sections, T::one());
    if current_dc.abs() > T::epsilon() {
        let scale = passband_target(order, epsilon) / current_dc;
        scale_first_section_gain(&mut cascade.sections, scale);
    }

    Ok(cascade)
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

    let (wa, c) = prewarp(cutoff, sample_rate);
    let (epsilon, sinh_a, cosh_a) = chebyshev_params(order, ripple_db);
    let wa2 = wa * wa;

    let mut cascade = assemble_cascade::<T, N>(
        order,
        |k| {
            // Compute the analog LP prototype pole, then invert for HP:
            // p_hp = ωa²/p_lp, so for p = σ + jω:
            //   σ_hp = ωa²·σ / (σ² + ω²), ω_hp = −ωa²·ω / (σ² + ω²).
            let (sigma_lp, omega_lp) = chebyshev_lp_pole(order, k, wa, sinh_a, cosh_a);
            let mag2 = sigma_lp * sigma_lp + omega_lp * omega_lp;
            bilinear_hp_pair(wa2 * sigma_lp / mag2, -wa2 * omega_lp / mag2, wa, c)
        },
        || {
            // Real LP pole σ = −ωa·sinh(a) → HP pole ωa²/σ = −ωa/sinh(a).
            bilinear_hp_real(wa2 / (-wa * sinh_a), wa, c)
        },
    );

    // Rescale the Nyquist (passband) gain to the target — see the LP note.
    let current_nyq = cascade_gain_at(&cascade.sections, -T::one());
    if current_nyq.abs() > T::epsilon() {
        let scale = passband_target(order, epsilon) / current_nyq.abs();
        scale_first_section_gain(&mut cascade.sections, scale);
    }

    Ok(cascade)
}
