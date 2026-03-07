use crate::traits::FloatScalar;
use super::ControlError;

/// PID gains computed by a tuning rule.
///
/// # Example
///
/// ```
/// use numeris::control::{FopdtModel, PidGains};
///
/// let model = FopdtModel::new(2.0_f64, 1.0, 0.3).unwrap();
/// let gains = model.ziegler_nichols();
/// assert!(gains.kp > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PidGains<T> {
    /// Proportional gain.
    pub kp: T,
    /// Integral gain (= Kp / Ti).
    pub ki: T,
    /// Derivative gain (= Kp · Td).
    pub kd: T,
}

/// First-Order Plus Dead Time (FOPDT) process model.
///
/// Models a process as:
///
/// ```text
/// G(s) = K · e^(-Ls) / (τs + 1)
/// ```
///
/// where `K` is the static gain, `τ` is the time constant, and `L` is the
/// dead time (transport delay).
///
/// # Example
///
/// ```
/// use numeris::control::FopdtModel;
///
/// let model = FopdtModel::new(2.5_f64, 3.0, 0.5).unwrap();
/// assert_eq!(model.gain(), 2.5);
/// assert_eq!(model.tau(), 3.0);
/// assert_eq!(model.delay(), 0.5);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FopdtModel<T> {
    k: T,
    tau: T,
    l: T,
}

impl<T: FloatScalar> FopdtModel<T> {
    /// Create a new FOPDT model.
    ///
    /// # Parameters
    ///
    /// - `gain`: static process gain `K` (non-zero, finite)
    /// - `tau`: time constant `τ` (positive, finite)
    /// - `delay`: dead time `L` (non-negative, finite)
    ///
    /// # Errors
    ///
    /// Returns `ControlError::InvalidFrequency` if parameters are out of range.
    pub fn new(gain: T, tau: T, delay: T) -> Result<Self, ControlError> {
        let zero = T::zero();
        if gain == zero || !gain.is_finite() {
            return Err(ControlError::InvalidFrequency);
        }
        if tau <= zero || !tau.is_finite() {
            return Err(ControlError::InvalidFrequency);
        }
        if delay < zero || !delay.is_finite() {
            return Err(ControlError::InvalidFrequency);
        }
        Ok(Self { k: gain, tau, l: delay })
    }

    /// Static process gain `K`.
    pub fn gain(&self) -> T { self.k }

    /// Time constant `τ`.
    pub fn tau(&self) -> T { self.tau }

    /// Dead time (transport delay) `L`.
    pub fn delay(&self) -> T { self.l }

    /// Ziegler-Nichols open-loop tuning (reaction curve method).
    ///
    /// Based on the process reaction curve (step response). Gives aggressive
    /// tuning with ~25% overshoot.
    ///
    /// # Panics
    ///
    /// Panics if `L == 0` (Ziegler-Nichols requires non-zero dead time).
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::control::FopdtModel;
    ///
    /// let model = FopdtModel::new(1.0_f64, 1.0, 0.2).unwrap();
    /// let gains = model.ziegler_nichols();
    /// // Kp = 1.2·τ/(K·L), Ti = 2L, Td = L/2
    /// assert!((gains.kp - 6.0).abs() < 1e-10);
    /// ```
    pub fn ziegler_nichols(&self) -> PidGains<T> {
        let two = T::one() + T::one();
        assert!(self.l > T::zero(), "Ziegler-Nichols requires non-zero dead time");

        let r = self.tau / (self.k * self.l);
        let kp = T::from(1.2).unwrap() * r;
        let ti = two * self.l;
        let td = self.l / two;

        PidGains {
            kp,
            ki: kp / ti,
            kd: kp * td,
        }
    }

    /// Cohen-Coon tuning rule.
    ///
    /// Better than Ziegler-Nichols for processes with large dead time
    /// (L/τ > 0.5). Targets quarter-decay ratio.
    ///
    /// # Panics
    ///
    /// Panics if `L == 0`.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::control::FopdtModel;
    ///
    /// let model = FopdtModel::new(1.0_f64, 1.0, 0.5).unwrap();
    /// let gains = model.cohen_coon();
    /// assert!(gains.kp > 0.0);
    /// assert!(gains.ki > 0.0);
    /// assert!(gains.kd > 0.0);
    /// ```
    pub fn cohen_coon(&self) -> PidGains<T> {
        assert!(self.l > T::zero(), "Cohen-Coon requires non-zero dead time");

        let three = T::from(3.0).unwrap();
        let four = T::from(4.0).unwrap();

        let r = self.l / self.tau;
        let base = self.tau / (self.k * self.l);

        // Kp = (τ/(K·L)) · (4/3 + L/(4τ))
        let kp = base * (four / three + r / four);

        // Ti = L · (32 + 6·L/τ) / (13 + 8·L/τ)
        let ti = self.l * (T::from(32.0).unwrap() + T::from(6.0).unwrap() * r)
            / (T::from(13.0).unwrap() + T::from(8.0).unwrap() * r);

        // Td = L · 4 / (11 + 2·L/τ)
        let td = self.l * four / (T::from(11.0).unwrap() + T::from(2.0).unwrap() * r);

        PidGains {
            kp,
            ki: kp / ti,
            kd: kp * td,
        }
    }

    /// SIMC (Skogestad Internal Model Control) tuning rule.
    ///
    /// Provides a good balance between performance and robustness.
    /// The `tau_c` parameter controls the closed-loop time constant:
    /// larger values give more conservative (robust) tuning.
    ///
    /// A common starting point is `tau_c = L` (aggressive) or `tau_c = τ`
    /// (conservative).
    ///
    /// # Parameters
    ///
    /// - `tau_c`: desired closed-loop time constant (positive)
    ///
    /// # Panics
    ///
    /// Panics if `tau_c <= 0`.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::control::FopdtModel;
    ///
    /// let model = FopdtModel::new(1.0_f64, 2.0, 0.3).unwrap();
    /// let gains = model.simc(0.3); // tau_c = L (aggressive)
    /// assert!(gains.kp > 0.0);
    /// ```
    pub fn simc(&self, tau_c: T) -> PidGains<T> {
        assert!(tau_c > T::zero(), "closed-loop time constant must be positive");

        let two = T::one() + T::one();

        // Kp = τ / (K · (tau_c + L))
        let kp = self.tau / (self.k * (tau_c + self.l));

        // Ti = min(τ, 4·(tau_c + L))
        let four = two + two;
        let ti_candidate = four * (tau_c + self.l);
        let ti = if self.tau < ti_candidate { self.tau } else { ti_candidate };

        // Td = 0 for first-order process (PID only adds D for second-order)
        // But we include it as: Td = L/2 when L > 0, for dead-time compensation
        let td = self.l / two;

        PidGains {
            kp,
            ki: kp / ti,
            kd: kp * td,
        }
    }
}

/// Ziegler-Nichols ultimate gain (closed-loop) tuning.
///
/// Based on the ultimate gain `ku` and ultimate period `tu` found by
/// increasing proportional gain until sustained oscillation.
///
/// # Parameters
///
/// - `ku`: ultimate gain (gain at sustained oscillation, positive)
/// - `tu`: ultimate period in seconds (positive)
///
/// # Errors
///
/// Returns `ControlError::InvalidFrequency` if parameters are non-positive.
///
/// # Example
///
/// ```
/// use numeris::control::ziegler_nichols_ultimate;
///
/// let gains = ziegler_nichols_ultimate(10.0_f64, 0.5).unwrap();
/// // Kp = 0.6·Ku, Ti = Tu/2, Td = Tu/8
/// assert!((gains.kp - 6.0).abs() < 1e-10);
/// ```
pub fn ziegler_nichols_ultimate<T: FloatScalar>(
    ku: T,
    tu: T,
) -> Result<PidGains<T>, ControlError> {
    let zero = T::zero();
    if ku <= zero || !ku.is_finite() {
        return Err(ControlError::InvalidFrequency);
    }
    if tu <= zero || !tu.is_finite() {
        return Err(ControlError::InvalidFrequency);
    }

    let two = T::one() + T::one();
    let eight = T::from(8.0).unwrap();

    let kp = T::from(0.6).unwrap() * ku;
    let ti = tu / two;
    let td = tu / eight;

    Ok(PidGains {
        kp,
        ki: kp / ti,
        kd: kp * td,
    })
}
