//! Digital control: IIR filters (Butterworth, Chebyshev Type I), PID controller.
//!
//! Provides biquad (second-order section) cascade filters designed via the
//! bilinear transform, and a discrete-time PID controller with anti-windup
//! and derivative filtering. All code is no-std compatible with no `complex`
//! feature dependency — pole computation uses real arithmetic only.
//!
//! # Examples
//!
//! ```
//! use numeris::control::{butterworth_lowpass, BiquadCascade};
//!
//! // 4th-order Butterworth lowpass at 1 kHz, 8 kHz sample rate
//! let mut lpf: BiquadCascade<f64, 2> = butterworth_lowpass(4, 1000.0, 8000.0).unwrap();
//! let y = lpf.tick(1.0); // filter one sample
//! ```
//!
//! ```
//! use numeris::control::Pid;
//!
//! // PID controller at 100 Hz with output clamping
//! let mut pid = Pid::new(2.0_f64, 0.5, 0.1, 0.01)
//!     .with_output_limits(-10.0, 10.0);
//! let u = pid.tick(1.0, 0.0); // setpoint=1, measurement=0
//! ```

mod biquad;
mod butterworth;
mod chebyshev;
mod pid;

#[cfg(test)]
mod tests;

pub use biquad::{Biquad, BiquadCascade};
pub use butterworth::{butterworth_highpass, butterworth_lowpass};
pub use chebyshev::{chebyshev1_highpass, chebyshev1_lowpass};
pub use pid::Pid;

use crate::traits::FloatScalar;

/// Errors from filter design functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlError {
    /// Filter order is zero, or `N` (number of sections) does not equal `ceil(order / 2)`.
    InvalidOrder,
    /// Cutoff frequency is not in the range `(0, sample_rate / 2)`.
    InvalidFrequency,
    /// Passband ripple is not positive (Chebyshev).
    InvalidRipple,
}

impl core::fmt::Display for ControlError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ControlError::InvalidOrder => write!(f, "invalid filter order or section count"),
            ControlError::InvalidFrequency => {
                write!(f, "cutoff frequency must be in (0, sample_rate/2)")
            }
            ControlError::InvalidRipple => write!(f, "passband ripple must be positive"),
        }
    }
}

/// Validate common filter design parameters.
///
/// Returns `ControlError` if any parameter is out of range.
pub(super) fn validate_design_params<T: FloatScalar, const N: usize>(
    order: usize,
    cutoff: T,
    sample_rate: T,
) -> Result<(), ControlError> {
    if order == 0 {
        return Err(ControlError::InvalidOrder);
    }
    // N must equal ceil(order / 2)
    let expected_sections = (order + 1) / 2;
    if N != expected_sections {
        return Err(ControlError::InvalidOrder);
    }
    let zero = T::zero();
    let two = T::one() + T::one();
    let nyquist = sample_rate / two;
    if cutoff <= zero || cutoff >= nyquist || !cutoff.is_finite() {
        return Err(ControlError::InvalidFrequency);
    }
    if !sample_rate.is_finite() || sample_rate <= zero {
        return Err(ControlError::InvalidFrequency);
    }
    Ok(())
}
