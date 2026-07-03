use crate::traits::FloatScalar;

use super::biquad::{
    assemble_cascade, bilinear_hp_pair, bilinear_hp_real, bilinear_lp_pair, bilinear_lp_real,
    prewarp, BiquadCascade,
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
    let (wa, c) = prewarp(cutoff, sample_rate);
    Ok(assemble_cascade(
        order,
        |k| {
            let (sigma, omega) = butterworth_pole(order, k, wa);
            bilinear_lp_pair(sigma, omega, wa, c)
        },
        // Odd-order: real pole at σ = −ωa (θ = π).
        || bilinear_lp_real(-wa, wa, c),
    ))
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
    let (wa, c) = prewarp(cutoff, sample_rate);
    Ok(assemble_cascade(
        order,
        |k| {
            let (sigma, omega) = butterworth_pole(order, k, wa);
            bilinear_hp_pair(sigma, omega, wa, c)
        },
        || bilinear_hp_real(-wa, wa, c),
    ))
}

/// Analog conjugate-pole pair `(σ, ω)` of the `k`-th Butterworth section,
/// scaled to the pre-warped cutoff `wa`. θ_k = π·(2k + n + 1) / (2n).
fn butterworth_pole<T: FloatScalar>(order: usize, k: usize, wa: T) -> (T, T) {
    let two = T::one() + T::one();
    let pi = T::from(core::f64::consts::PI).unwrap();
    let nf = T::from(order).unwrap();
    let kf = T::from(k).unwrap();
    let theta = pi * (two * kf + nf + T::one()) / (two * nf);
    (wa * theta.cos(), wa * theta.sin())
}
