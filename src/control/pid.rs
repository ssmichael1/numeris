use crate::traits::FloatScalar;

/// Discrete-time PID controller with anti-windup and derivative filtering.
///
/// Uses derivative-on-measurement to avoid derivative kick on setpoint changes,
/// trapezoidal integration for the integral term, and optional first-order
/// low-pass filtering on the derivative term.
///
/// Anti-windup is provided via back-calculation: when output saturates, the
/// integrator is corrected to prevent excessive windup.
///
/// # Example
///
/// ```
/// use numeris::control::Pid;
///
/// // PID controller running at 100 Hz
/// let mut pid = Pid::new(1.0_f64, 0.5, 0.1, 0.01)
///     .with_output_limits(-10.0, 10.0)
///     .with_derivative_filter(0.01);
///
/// // Simulate a few ticks
/// let setpoint = 5.0;
/// let measurement = 0.0;
/// let output = pid.tick(setpoint, measurement);
/// assert!(output > 0.0); // positive correction to reach setpoint
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Pid<T> {
    // Configuration
    kp: T,
    ki: T,
    kd: T,
    dt: T,
    output_min: T,
    output_max: T,
    tau_d: T,
    kb: T,

    // State
    integral: T,
    prev_error: T,
    prev_measurement: T,
    prev_derivative: T,
    initialized: bool,
}

impl<T: FloatScalar> Pid<T> {
    /// Create a new PID controller with the given gains and time step.
    ///
    /// # Panics
    ///
    /// Panics if `dt <= 0` or `dt` is not finite.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::control::Pid;
    ///
    /// let pid = Pid::new(1.0_f64, 0.1, 0.05, 0.01);
    /// assert_eq!(pid.gains(), (1.0, 0.1, 0.05));
    /// ```
    pub fn new(kp: T, ki: T, kd: T, dt: T) -> Self {
        assert!(dt > T::zero() && dt.is_finite(), "dt must be positive and finite");

        // Default back-calculation gain: ki/kp if kp != 0, else ki
        let kb = if kp != T::zero() { ki / kp } else { ki };

        Self {
            kp,
            ki,
            kd,
            dt,
            output_min: T::neg_infinity(),
            output_max: T::infinity(),
            tau_d: T::zero(),
            kb,
            integral: T::zero(),
            prev_error: T::zero(),
            prev_measurement: T::zero(),
            prev_derivative: T::zero(),
            initialized: false,
        }
    }

    /// Set output clamping limits. Returns `self` for chaining.
    ///
    /// # Panics
    ///
    /// Panics if `min >= max`.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::control::Pid;
    ///
    /// let pid = Pid::new(1.0_f64, 0.0, 0.0, 0.01)
    ///     .with_output_limits(-5.0, 5.0);
    /// ```
    pub fn with_output_limits(mut self, min: T, max: T) -> Self {
        assert!(min < max, "output_min must be less than output_max");
        self.output_min = min;
        self.output_max = max;
        self
    }

    /// Set derivative low-pass filter time constant. Returns `self` for chaining.
    ///
    /// When `tau > 0`, the derivative term is filtered through a first-order IIR
    /// with `alpha = dt / (tau + dt)`, smoothing out noise on the measurement signal.
    /// When `tau == 0` (the default), no filtering is applied.
    ///
    /// # Panics
    ///
    /// Panics if `tau < 0`.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::control::Pid;
    ///
    /// let pid = Pid::new(1.0_f64, 0.0, 0.5, 0.01)
    ///     .with_derivative_filter(0.02);
    /// ```
    pub fn with_derivative_filter(mut self, tau: T) -> Self {
        assert!(!(tau < T::zero()), "derivative filter time constant must be non-negative");
        self.tau_d = tau;
        self
    }

    /// Set the anti-windup back-calculation gain. Returns `self` for chaining.
    ///
    /// By default, `kb = ki / kp` (or `ki` if `kp == 0`). Set to zero to disable
    /// anti-windup correction entirely.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::control::Pid;
    ///
    /// let pid = Pid::new(1.0_f64, 0.5, 0.0, 0.01)
    ///     .with_output_limits(-1.0, 1.0)
    ///     .with_back_calculation_gain(2.0);
    /// ```
    pub fn with_back_calculation_gain(mut self, kb: T) -> Self {
        self.kb = kb;
        self
    }

    /// Process one time step and return the control output.
    ///
    /// `setpoint` is the desired value; `measurement` is the current process value.
    ///
    /// On the first tick after construction or `reset()`, the derivative and
    /// trapezoidal integration terms are zeroed to avoid startup transients.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::control::Pid;
    ///
    /// let mut pid = Pid::new(2.0_f64, 0.0, 0.0, 0.01);
    /// let u = pid.tick(10.0, 3.0); // error = 7, output = 2 * 7 = 14
    /// assert!((u - 14.0).abs() < 1e-12);
    /// ```
    #[inline]
    pub fn tick(&mut self, setpoint: T, measurement: T) -> T {
        let error = setpoint - measurement;
        let two = T::one() + T::one();

        // Proportional
        let p_term = self.kp * error;

        // Integral (trapezoidal rule, skip on first tick)
        if self.initialized {
            self.integral = self.integral + self.ki * (error + self.prev_error) * self.dt / two;
        }

        // Derivative on measurement (negated to avoid derivative kick)
        let d_raw = if self.initialized {
            -(measurement - self.prev_measurement) / self.dt
        } else {
            T::zero()
        };

        // Optional first-order low-pass filter on derivative
        let d_filtered = if self.tau_d > T::zero() && self.initialized {
            let alpha = self.dt / (self.tau_d + self.dt);
            self.prev_derivative + alpha * (d_raw - self.prev_derivative)
        } else {
            d_raw
        };

        let d_term = self.kd * d_filtered;

        // Total output (unclamped)
        let u_unclamped = p_term + self.integral + d_term;

        // Clamp
        let u_clamped = if u_unclamped > self.output_max {
            self.output_max
        } else if u_unclamped < self.output_min {
            self.output_min
        } else {
            u_unclamped
        };

        // Anti-windup back-calculation
        if self.initialized {
            self.integral =
                self.integral + self.kb * (u_clamped - u_unclamped) * self.dt;
        }

        // Update state
        self.prev_error = error;
        self.prev_measurement = measurement;
        self.prev_derivative = d_filtered;
        self.initialized = true;

        u_clamped
    }

    /// Reset all internal state (integral, derivative, initialization flag).
    ///
    /// Configuration (gains, limits, filter constants) is preserved.
    ///
    /// # Example
    ///
    /// ```
    /// use numeris::control::Pid;
    ///
    /// let mut pid = Pid::new(1.0_f64, 1.0, 0.0, 0.01);
    /// pid.tick(1.0, 0.0);
    /// pid.reset();
    /// assert_eq!(pid.integral(), 0.0);
    /// ```
    pub fn reset(&mut self) {
        self.integral = T::zero();
        self.prev_error = T::zero();
        self.prev_measurement = T::zero();
        self.prev_derivative = T::zero();
        self.initialized = false;
    }

    /// Return the current `(kp, ki, kd)` gains.
    pub fn gains(&self) -> (T, T, T) {
        (self.kp, self.ki, self.kd)
    }

    /// Update the PID gains at runtime.
    ///
    /// Does not reset internal state — the integrator and derivative filter
    /// continue from their current values.
    pub fn set_gains(&mut self, kp: T, ki: T, kd: T) {
        self.kp = kp;
        self.ki = ki;
        self.kd = kd;
    }

    /// Return the current integrator value.
    pub fn integral(&self) -> T {
        self.integral
    }

    /// Manually set the integrator value.
    ///
    /// Useful for bumpless transfer when switching between manual and automatic
    /// control modes.
    pub fn set_integral(&mut self, value: T) {
        self.integral = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;
    fn assert_near(a: f64, b: f64, tol: f64, msg: &str) {
        assert!(
            (a - b).abs() < tol,
            "{}: {} vs {} (diff {})",
            msg,
            a,
            b,
            (a - b).abs()
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Construction and configuration
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_new() {
        let pid = Pid::new(1.0, 2.0, 3.0, 0.01);
        assert_eq!(pid.gains(), (1.0, 2.0, 3.0));
        assert_eq!(pid.integral(), 0.0);
    }

    #[test]
    #[should_panic]
    fn test_zero_dt_panics() {
        Pid::new(1.0, 0.0, 0.0, 0.0);
    }

    #[test]
    #[should_panic]
    fn test_negative_dt_panics() {
        Pid::new(1.0, 0.0, 0.0, -0.01);
    }

    #[test]
    #[should_panic]
    fn test_invalid_limits_panics() {
        Pid::new(1.0, 0.0, 0.0, 0.01).with_output_limits(5.0, 5.0);
    }

    #[test]
    #[should_panic]
    fn test_invalid_limits_reversed_panics() {
        Pid::new(1.0, 0.0, 0.0, 0.01).with_output_limits(5.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_negative_tau_panics() {
        Pid::new(1.0, 0.0, 0.0, 0.01).with_derivative_filter(-0.01);
    }

    #[test]
    fn test_builder_chaining() {
        let pid = Pid::new(1.0, 0.5, 0.1, 0.01)
            .with_output_limits(-10.0, 10.0)
            .with_derivative_filter(0.02)
            .with_back_calculation_gain(1.5);
        assert_eq!(pid.gains(), (1.0, 0.5, 0.1));
    }

    // ═══════════════════════════════════════════════════════════════
    // P-only
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_p_only() {
        let mut pid = Pid::new(2.5, 0.0, 0.0, 0.01);
        // First tick: output = kp * error
        let u = pid.tick(10.0, 3.0);
        assert_near(u, 2.5 * 7.0, TOL, "P-only output");
    }

    #[test]
    fn test_p_only_negative_error() {
        let mut pid = Pid::new(3.0, 0.0, 0.0, 0.01);
        let u = pid.tick(1.0, 5.0);
        assert_near(u, 3.0 * (-4.0), TOL, "P-only negative error");
    }

    #[test]
    fn test_p_only_zero_error() {
        let mut pid = Pid::new(3.0, 0.0, 0.0, 0.01);
        let u = pid.tick(5.0, 5.0);
        assert_near(u, 0.0, TOL, "P-only zero error");
    }

    // ═══════════════════════════════════════════════════════════════
    // I-only
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_i_only_trapezoidal() {
        let dt = 0.01;
        let ki = 2.0;
        let mut pid = Pid::new(0.0, ki, 0.0, dt);

        // First tick: integral not accumulated (no previous error)
        let u1 = pid.tick(10.0, 0.0); // error = 10
        assert_near(u1, 0.0, TOL, "I-only first tick (no integration)");

        // Second tick: trapezoidal rule with error=10, prev_error=10
        // integral += ki * (10 + 10) * dt / 2 = 2 * 20 * 0.01 / 2 = 0.2
        let u2 = pid.tick(10.0, 0.0);
        assert_near(u2, 0.2, TOL, "I-only second tick");
        assert_near(pid.integral(), 0.2, TOL, "integral value after 2 ticks");

        // Third tick: same error, integral accumulates further
        // integral += 2 * (10 + 10) * 0.01 / 2 = 0.2, total = 0.4
        let u3 = pid.tick(10.0, 0.0);
        assert_near(u3, 0.4, TOL, "I-only third tick");
    }

    #[test]
    fn test_i_only_varying_error() {
        let dt = 0.1;
        let ki = 1.0;
        let mut pid = Pid::new(0.0, ki, 0.0, dt);

        pid.tick(10.0, 0.0); // error=10, no integration
        // error=5, integral += 1.0 * (5 + 10) * 0.1 / 2 = 0.75
        let u = pid.tick(5.0, 0.0);
        assert_near(u, 0.75, TOL, "I-only varying error trapezoidal");
    }

    // ═══════════════════════════════════════════════════════════════
    // D-only
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_d_only_step_on_measurement() {
        let dt = 0.01;
        let kd = 0.5;
        let mut pid = Pid::new(0.0, 0.0, kd, dt);

        // First tick: d_raw = 0 (not initialized)
        let u1 = pid.tick(0.0, 0.0);
        assert_near(u1, 0.0, TOL, "D first tick zero");

        // Step measurement from 0 to 1: d_raw = -(1 - 0)/dt = -100
        // d_term = kd * d_raw = 0.5 * (-100) = -50
        let u2 = pid.tick(0.0, 1.0);
        assert_near(u2, -50.0, TOL, "D step on measurement");

        // Constant measurement: d_raw = -(1 - 1)/dt = 0
        let u3 = pid.tick(0.0, 1.0);
        assert_near(u3, 0.0, TOL, "D constant measurement");
    }

    #[test]
    fn test_d_no_derivative_kick_on_setpoint_change() {
        let dt = 0.01;
        let kd = 1.0;
        let mut pid = Pid::new(0.0, 0.0, kd, dt);

        pid.tick(0.0, 5.0); // initialize
        // Setpoint changes from 0 to 100, measurement stays at 5
        // Derivative on measurement: d_raw = -(5 - 5)/dt = 0
        let u = pid.tick(100.0, 5.0);
        assert_near(u, 0.0, TOL, "no derivative kick on setpoint change");
    }

    #[test]
    fn test_d_filter_smoothing() {
        let dt = 0.01;
        let kd = 1.0;
        let tau = 0.05; // filter time constant
        let mut pid = Pid::new(0.0, 0.0, kd, dt).with_derivative_filter(tau);

        pid.tick(0.0, 0.0); // initialize
        // Step measurement from 0 to 1
        // d_raw = -(1 - 0)/0.01 = -100
        // alpha = 0.01 / (0.05 + 0.01) = 1/6
        // d_filtered = 0 + (1/6) * (-100 - 0) = -100/6
        let u = pid.tick(0.0, 1.0);
        let expected = kd * (-100.0 / 6.0);
        assert_near(u, expected, TOL, "D filtered step (attenuated)");

        // Without filter, response would be kd * (-100) = -100
        // Filter significantly reduces the impulse
        assert!(u.abs() < 100.0, "filter reduces derivative impulse");
    }

    // ═══════════════════════════════════════════════════════════════
    // Full PID — closed-loop step response
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_pid_first_order_plant_convergence() {
        // First-order plant: dy/dt = -y + u, discretized at dt
        let dt = 0.001;
        let mut pid = Pid::new(5.0, 10.0, 0.1, dt);
        let setpoint = 1.0;
        let mut y = 0.0;

        for _ in 0..10_000 {
            let u = pid.tick(setpoint, y);
            // Euler discretization of dy/dt = -y + u
            y = y + dt * (-y + u);
        }

        assert_near(y, setpoint, 1e-4, "PID converges to setpoint");
    }

    #[test]
    fn test_pid_step_response_integrator_eliminates_offset() {
        // P-only would leave steady-state error; I eliminates it
        let dt = 0.001;
        let mut pid = Pid::new(1.0, 5.0, 0.0, dt);
        let setpoint = 1.0;
        let mut y = 0.0;

        for _ in 0..20_000 {
            let u = pid.tick(setpoint, y);
            y = y + dt * (-y + u);
        }

        assert_near(y, setpoint, 1e-4, "PI eliminates steady-state error");
    }

    // ═══════════════════════════════════════════════════════════════
    // Clamping
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_output_clamped_upper() {
        let mut pid = Pid::new(100.0, 0.0, 0.0, 0.01).with_output_limits(-5.0, 5.0);
        let u = pid.tick(10.0, 0.0); // kp * error = 1000, clamped to 5
        assert_near(u, 5.0, TOL, "clamped to upper limit");
    }

    #[test]
    fn test_output_clamped_lower() {
        let mut pid = Pid::new(100.0, 0.0, 0.0, 0.01).with_output_limits(-5.0, 5.0);
        let u = pid.tick(0.0, 10.0); // kp * error = -1000, clamped to -5
        assert_near(u, -5.0, TOL, "clamped to lower limit");
    }

    #[test]
    fn test_output_within_range_not_clamped() {
        let mut pid = Pid::new(1.0, 0.0, 0.0, 0.01).with_output_limits(-100.0, 100.0);
        let u = pid.tick(5.0, 3.0); // kp * error = 2
        assert_near(u, 2.0, TOL, "within range, not clamped");
    }

    // ═══════════════════════════════════════════════════════════════
    // Anti-windup
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_anti_windup_faster_recovery() {
        // Compare two PI controllers: one with anti-windup, one without
        let dt: f64 = 0.01;
        let kp = 1.0;
        let ki = 10.0;
        let limit = 1.0;
        let setpoint = 10.0; // large setpoint to cause saturation

        // Controller with anti-windup (default kb = ki/kp)
        let mut pid_aw = Pid::new(kp, ki, 0.0, dt).with_output_limits(-limit, limit);

        // Controller without anti-windup (kb = 0)
        let mut pid_no_aw = Pid::new(kp, ki, 0.0, dt)
            .with_output_limits(-limit, limit)
            .with_back_calculation_gain(0.0);

        // Phase 1: drive both into saturation
        for _ in 0..100 {
            pid_aw.tick(setpoint, 0.0);
            pid_no_aw.tick(setpoint, 0.0);
        }

        // The no-anti-windup controller should have a much larger integral
        assert!(
            pid_no_aw.integral().abs() > pid_aw.integral().abs(),
            "without anti-windup, integral winds up more: {} vs {}",
            pid_no_aw.integral(),
            pid_aw.integral()
        );

        // Phase 2: switch setpoint to 0, measure recovery
        let mut y_aw = 0.0;
        let mut y_no_aw = 0.0;
        let mut recovery_aw = usize::MAX;
        let mut recovery_no_aw = usize::MAX;

        for i in 0..1000 {
            let u_aw = pid_aw.tick(0.0, y_aw);
            let u_no_aw = pid_no_aw.tick(0.0, y_no_aw);
            y_aw = y_aw + dt * (-y_aw + u_aw);
            y_no_aw = y_no_aw + dt * (-y_no_aw + u_no_aw);

            if recovery_aw == usize::MAX && y_aw.abs() < 0.1 {
                recovery_aw = i;
            }
            if recovery_no_aw == usize::MAX && y_no_aw.abs() < 0.1 {
                recovery_no_aw = i;
            }
        }

        // Anti-windup should recover faster (or at least not slower)
        assert!(
            recovery_aw <= recovery_no_aw,
            "anti-windup should recover faster: {} vs {}",
            recovery_aw,
            recovery_no_aw
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Reset
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_reset_clears_state() {
        let mut pid = Pid::new(1.0, 1.0, 1.0, 0.01);
        pid.tick(10.0, 0.0);
        pid.tick(10.0, 1.0);
        assert!(pid.integral() != 0.0);

        pid.reset();
        assert_eq!(pid.integral(), 0.0);
        assert_eq!(pid.gains(), (1.0, 1.0, 1.0)); // config preserved
    }

    #[test]
    fn test_reset_preserves_config() {
        let mut pid = Pid::new(2.0, 3.0, 4.0, 0.005)
            .with_output_limits(-10.0, 10.0)
            .with_derivative_filter(0.02);

        pid.tick(5.0, 0.0);
        pid.reset();

        // After reset, first tick should behave like a fresh controller
        let u = pid.tick(5.0, 0.0);
        // P-only on first tick (no integral, no derivative)
        assert_near(u, 2.0 * 5.0, TOL, "after reset, first tick is P-only");
    }

    // ═══════════════════════════════════════════════════════════════
    // Runtime gain changes
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_set_gains() {
        let mut pid = Pid::new(1.0, 0.0, 0.0, 0.01);
        pid.set_gains(5.0, 2.0, 1.0);
        assert_eq!(pid.gains(), (5.0, 2.0, 1.0));

        // Verify new gains take effect
        let u = pid.tick(3.0, 0.0);
        assert_near(u, 5.0 * 3.0, TOL, "new kp takes effect");
    }

    #[test]
    fn test_set_integral() {
        let mut pid = Pid::new(1.0, 1.0, 0.0, 0.01);
        pid.set_integral(5.0);
        assert_eq!(pid.integral(), 5.0);

        // First tick: P + pre-set integral (no trapezoidal contribution on first tick)
        let u = pid.tick(1.0, 0.0); // p_term = 1.0, integral = 5.0
        assert_near(u, 6.0, TOL, "set_integral adds to output");
    }

    // ═══════════════════════════════════════════════════════════════
    // f32 support
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_f32() {
        let mut pid = Pid::new(1.0_f32, 0.5, 0.1, 0.01)
            .with_output_limits(-10.0, 10.0)
            .with_derivative_filter(0.02);

        let u = pid.tick(5.0_f32, 0.0);
        assert!((u - 5.0_f32).abs() < 1e-5, "f32 P-only output");

        // Run a few more ticks to exercise integral and derivative
        for _ in 0..10 {
            pid.tick(5.0, u);
        }
        // Just verify no panics or NaNs
        assert!(pid.integral().is_finite(), "f32 integral is finite");
    }
}
