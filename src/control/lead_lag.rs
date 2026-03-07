use crate::traits::FloatScalar;
use super::biquad::Biquad;
use super::ControlError;

/// Design a lead compensator as a discrete-time biquad.
///
/// A lead compensator adds phase lead to improve stability margins.
/// The continuous-time transfer function is:
///
/// ```text
/// C(s) = (K/α) · (Ts + 1) / (αTs + 1)
/// ```
///
/// where `α = (1 - sin(φ)) / (1 + sin(φ))` and `T = 1 / (ωm · √α)`.
///
/// # Parameters
///
/// - `phase_lead`: desired maximum phase lead in radians, must be in `(0, π/2)`
/// - `center_freq`: frequency in Hz where maximum phase lead occurs
/// - `gain`: overall DC gain of the compensator (typically 1.0)
/// - `sample_rate`: sampling frequency in Hz
///
/// # Example
///
/// ```
/// use numeris::control::{lead_compensator, Biquad};
///
/// // 45° phase lead at 10 Hz, unity gain, 1 kHz sample rate
/// let comp = lead_compensator(
///     std::f64::consts::FRAC_PI_4, 10.0, 1.0, 1000.0,
/// ).unwrap();
/// ```
pub fn lead_compensator<T: FloatScalar>(
    phase_lead: T,
    center_freq: T,
    gain: T,
    sample_rate: T,
) -> Result<Biquad<T>, ControlError> {
    let zero = T::zero();
    let one = T::one();
    let two = one + one;
    let pi = T::from(core::f64::consts::PI).unwrap();
    let half_pi = pi / two;

    if phase_lead <= zero || phase_lead >= half_pi {
        return Err(ControlError::InvalidFrequency);
    }
    if center_freq <= zero || !center_freq.is_finite() {
        return Err(ControlError::InvalidFrequency);
    }
    if sample_rate <= zero || !sample_rate.is_finite() {
        return Err(ControlError::InvalidFrequency);
    }
    let nyquist = sample_rate / two;
    if center_freq >= nyquist {
        return Err(ControlError::InvalidFrequency);
    }

    // α = (1 - sin(φ)) / (1 + sin(φ))
    let sin_phi = phase_lead.sin();
    let alpha = (one - sin_phi) / (one + sin_phi);

    // T = 1 / (ωm · √α), ωm = 2π·f
    let omega_m = two * pi * center_freq;
    let t_const = one / (omega_m * alpha.sqrt());

    // Analog: C(s) = gain/α · (Ts + 1) / (αTs + 1)
    // Zero at s = -1/T, pole at s = -1/(αT)
    let z_freq = one / t_const;            // zero: -z_freq
    let p_freq = one / (alpha * t_const);  // pole: -p_freq
    let k_analog = gain / alpha;

    // Bilinear transform: s = c·(z-1)/(z+1), c = 2·fs
    let c = two * sample_rate;

    // H(z) = K · (c + z_freq) / (c + p_freq) · (1 + (c - z_freq)/(c + z_freq) · z⁻¹)
    //       / (1 + (c - p_freq)/(c + p_freq) · z⁻¹)
    let b0 = k_analog * (c + z_freq) / (c + p_freq);
    let b1 = k_analog * (-c + z_freq) / (c + p_freq);
    let a1 = (-c + p_freq) / (c + p_freq);

    Ok(Biquad::new(
        [b0, b1, T::zero()],
        [T::one(), a1, T::zero()],
    ))
}

/// Design a lag compensator as a discrete-time biquad.
///
/// A lag compensator boosts low-frequency gain for improved steady-state
/// accuracy without significantly affecting phase margin.
/// The continuous-time transfer function is:
///
/// ```text
/// C(s) = β · (s + ω/β) / (s + ω)
/// ```
///
/// where `β` is the DC gain boost ratio and `ω = 2π·f_corner`.
///
/// # Parameters
///
/// - `dc_boost`: ratio of DC gain to high-frequency gain, must be `> 1`
/// - `corner_freq`: upper corner frequency in Hz (above this, gain ≈ 1)
/// - `sample_rate`: sampling frequency in Hz
///
/// # Example
///
/// ```
/// use numeris::control::{lag_compensator, Biquad};
///
/// // 10x DC boost with corner at 1 Hz, 1 kHz sample rate
/// let comp = lag_compensator(10.0, 1.0, 1000.0).unwrap();
/// ```
pub fn lag_compensator<T: FloatScalar>(
    dc_boost: T,
    corner_freq: T,
    sample_rate: T,
) -> Result<Biquad<T>, ControlError> {
    let zero = T::zero();
    let one = T::one();
    let two = one + one;
    let pi = T::from(core::f64::consts::PI).unwrap();

    if dc_boost <= one {
        return Err(ControlError::InvalidFrequency);
    }
    if corner_freq <= zero || !corner_freq.is_finite() {
        return Err(ControlError::InvalidFrequency);
    }
    if sample_rate <= zero || !sample_rate.is_finite() {
        return Err(ControlError::InvalidFrequency);
    }
    let nyquist = sample_rate / two;
    if corner_freq >= nyquist {
        return Err(ControlError::InvalidFrequency);
    }

    // Textbook lag: C(s) = β · (τs + 1) / (βτs + 1)
    //   = (s + 1/τ) / (s + 1/(βτ))    [β cancels with τ/(βτ) ratio]
    // C(0) = (1/τ) / (1/(βτ)) = β (DC boost), C(∞) = 1 (unity at HF)
    // Zero at s = -1/τ = -ω, pole at s = -1/(βτ) = -ω/β
    let omega = two * pi * corner_freq;
    let z_freq = omega;                    // zero: -ω
    let p_freq = omega / dc_boost;         // pole: -ω/β
    let k_analog = one;

    // Bilinear transform: s = c·(z-1)/(z+1), c = 2·fs
    let c = two * sample_rate;

    let b0 = k_analog * (c + z_freq) / (c + p_freq);
    let b1 = k_analog * (-c + z_freq) / (c + p_freq);
    let a1 = (-c + p_freq) / (c + p_freq);

    Ok(Biquad::new(
        [b0, b1, T::zero()],
        [T::one(), a1, T::zero()],
    ))
}
