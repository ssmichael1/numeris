use crate::traits::FloatScalar;

use super::biquad::{
    bilinear_hp_pair, bilinear_hp_real, bilinear_lp_pair, bilinear_lp_real, BiquadCascade,
};
use super::{validate_design_params, ControlError};

/// Design a Butterworth lowpass filter as a cascade of `N` biquad sections.
///
/// `N` must equal `ceil(order / 2)`. The filter order determines the rolloff
/// steepness (−20·order dB/decade beyond the cutoff).
///
/// # Arguments
///
/// * `order` — filter order (1, 2, 3, …)
/// * `cutoff` — cutoff frequency in Hz (must be in `(0, sample_rate/2)`)
/// * `sample_rate` — sampling rate in Hz
///
/// # Example
///
/// ```
/// use numeris::control::{butterworth_lowpass, BiquadCascade};
///
/// // 4th-order → 2 sections
/// let mut lpf: BiquadCascade<f64, 2> = butterworth_lowpass(4, 1000.0, 8000.0).unwrap();
/// let y = lpf.tick(1.0);
///
/// // 5th-order → 3 sections (last is first-order degenerate)
/// let mut lpf: BiquadCascade<f64, 3> = butterworth_lowpass(5, 1000.0, 8000.0).unwrap();
/// ```
pub fn butterworth_lowpass<T: FloatScalar, const N: usize>(
    order: usize,
    cutoff: T,
    sample_rate: T,
) -> Result<BiquadCascade<T, N>, ControlError> {
    validate_design_params::<T, N>(order, cutoff, sample_rate)?;

    let two = T::one() + T::one();
    let pi = T::from(core::f64::consts::PI).unwrap();

    // Pre-warp
    let wa = two * sample_rate * (pi * cutoff / sample_rate).tan();
    let c = two * sample_rate;

    let n = order;
    let nf = T::from(n).unwrap();
    let mut sections = [super::biquad::Biquad::passthrough(); N];
    let mut idx = 0;

    // Conjugate pairs: k = 0, 1, …, (n/2 - 1)
    let num_pairs = n / 2;
    for k in 0..num_pairs {
        let kf = T::from(k).unwrap();
        // θ_k = π·(2k + n + 1) / (2n)
        let theta = pi * (two * kf + nf + T::one()) / (two * nf);
        let sigma = wa * theta.cos();
        let omega = wa * theta.sin();
        sections[idx] = bilinear_lp_pair(sigma, omega, wa, c);
        idx += 1;
    }

    // Odd-order: real pole at σ = −ωa (θ = π for k = (n-1)/2)
    if n % 2 == 1 {
        let sigma = -wa;
        sections[idx] = bilinear_lp_real(sigma, wa, c);
    }

    Ok(BiquadCascade { sections })
}

/// Design a Butterworth highpass filter as a cascade of `N` biquad sections.
///
/// `N` must equal `ceil(order / 2)`.
///
/// # Arguments
///
/// * `order` — filter order (1, 2, 3, …)
/// * `cutoff` — cutoff frequency in Hz (must be in `(0, sample_rate/2)`)
/// * `sample_rate` — sampling rate in Hz
///
/// # Example
///
/// ```
/// use numeris::control::{butterworth_highpass, BiquadCascade};
///
/// let mut hpf: BiquadCascade<f64, 2> = butterworth_highpass(4, 1000.0, 8000.0).unwrap();
/// let y = hpf.tick(1.0);
/// ```
pub fn butterworth_highpass<T: FloatScalar, const N: usize>(
    order: usize,
    cutoff: T,
    sample_rate: T,
) -> Result<BiquadCascade<T, N>, ControlError> {
    validate_design_params::<T, N>(order, cutoff, sample_rate)?;

    let two = T::one() + T::one();
    let pi = T::from(core::f64::consts::PI).unwrap();

    let wa = two * sample_rate * (pi * cutoff / sample_rate).tan();
    let c = two * sample_rate;

    let n = order;
    let nf = T::from(n).unwrap();
    let mut sections = [super::biquad::Biquad::passthrough(); N];
    let mut idx = 0;

    let num_pairs = n / 2;
    for k in 0..num_pairs {
        let kf = T::from(k).unwrap();
        let theta = pi * (two * kf + nf + T::one()) / (two * nf);
        let sigma = wa * theta.cos();
        let omega = wa * theta.sin();
        sections[idx] = bilinear_hp_pair(sigma, omega, wa, c);
        idx += 1;
    }

    if n % 2 == 1 {
        let sigma = -wa;
        sections[idx] = bilinear_hp_real(sigma, wa, c);
    }

    Ok(BiquadCascade { sections })
}
