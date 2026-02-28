use crate::traits::FloatScalar;
use crate::matrix::vector::Vector;
use super::{OdeError, Solution};

/// Settings for adaptive step-size control.
pub struct AdaptiveSettings<T> {
    /// Absolute error tolerance (default: 1e-8).
    pub abs_tol: T,
    /// Relative error tolerance (default: 1e-8).
    pub rel_tol: T,
    /// Minimum step-size decrease factor (default: 0.2).
    pub min_factor: T,
    /// Maximum step-size increase factor (default: 10.0).
    pub max_factor: T,
    /// Safety factor for step-size controller (default: 0.9).
    pub safety: T,
    /// Minimum allowed step size (default: 1e-6).
    pub min_step: T,
    /// Maximum number of steps before returning [`OdeError::MaxStepsExceeded`]
    /// (default: 100_000).
    pub max_steps: usize,
    /// Whether to store dense output for interpolation (default: false).
    /// Only meaningful with the `std` feature.
    pub dense_output: bool,
}

impl Default for AdaptiveSettings<f64> {
    fn default() -> Self {
        Self {
            abs_tol: 1e-8,
            rel_tol: 1e-8,
            min_factor: 0.2,
            max_factor: 10.0,
            safety: 0.9,
            min_step: 1e-6,
            max_steps: 100_000,
            dense_output: false,
        }
    }
}

impl Default for AdaptiveSettings<f32> {
    fn default() -> Self {
        Self {
            abs_tol: 1e-6,
            rel_tol: 1e-6,
            min_factor: 0.2,
            max_factor: 10.0,
            safety: 0.9,
            min_step: 1e-4,
            max_steps: 100_000,
            dense_output: false,
        }
    }
}

/// Trait for adaptive Runge-Kutta solvers.
///
/// Each solver is a zero-size struct that implements this trait with
/// const Butcher tableau coefficients. `STAGES` is the number of stages
/// and `NI` is the number of interpolation polynomial terms.
///
/// The Butcher coefficients are stored as `f64` even when integrating
/// with `f32` — they are compile-time constants cast via
/// `T::from(coeff).unwrap()` at use sites.
///
/// Step-size control uses a PID controller based on:
///
/// > G. Söderlind and L. Wang, "Adaptive time-stepping and computational
/// > stability," *J. Comput. Appl. Math.*, vol. 185, no. 2, pp. 225–243,
/// > 2006. <https://doi.org/10.1016/j.cam.2005.03.008>
pub trait RKAdaptive<const STAGES: usize, const NI: usize> {
    /// Butcher A matrix (lower triangular).
    const A: [[f64; STAGES]; STAGES];
    /// Weights for the higher-order solution.
    const B: [f64; STAGES];
    /// Error weights: `B[i] - Bhat[i]` for step error estimation.
    const BERR: [f64; STAGES];
    /// Nodes (abscissae).
    const C: [f64; STAGES];
    /// Interpolation coefficients (STAGES × NI).
    const BI: [[f64; NI]; STAGES];
    /// Order of the higher-order method.
    const ORDER: usize;
    /// First Same As Last optimization.
    const FSAL: bool;

    /// Integrate from `t0` to `tf` with initial state `y0`.
    fn integrate<T: FloatScalar, const S: usize>(
        t0: T,
        tf: T,
        y0: &Vector<T, S>,
        mut f: impl FnMut(T, &Vector<T, S>) -> Vector<T, S>,
        settings: &AdaptiveSettings<T>,
    ) -> Result<Solution<T, S>, OdeError> {
        let mut nevals: usize = 0;
        let mut naccept: usize = 0;
        let mut nreject: usize = 0;
        let mut t = t0;
        let mut y = *y0;

        let one = T::one();
        let zero = T::zero();
        let tdir = if tf > t0 { one } else { -one };

        // PID controller state: two previous error norms (Söderlind & Wang 2006)
        let mut enorm_prev = T::from(1.0e-4).unwrap();
        let mut enorm_prev2 = T::from(1.0e-4).unwrap();

        // Initial step-size guess (adapted from OrdinaryDiffEq.jl)
        let mut h = {
            let sci = y0.abs() * settings.rel_tol + settings.abs_tol;
            let d0 = y0.element_div(&sci).scaled_norm();
            let ydot0 = f(t0, y0);
            let d1 = ydot0.element_div(&sci).scaled_norm();
            let h0 = T::from(0.01).unwrap() * d0 / d1 * tdir;
            let y1 = *y0 + ydot0 * h0;
            let ydot1 = f(t0 + h0, &y1);
            let d2 = (ydot1 - ydot0).element_div(&sci).scaled_norm() / h0;
            nevals += 2;

            let dmax = if d1 > d2 { d1 } else { d2 };
            let order_t = T::from(Self::ORDER).unwrap();
            let h1 = if dmax < T::from(1e-15).unwrap() {
                let h0_abs = h0.abs();
                let floor = T::from(1e-6).unwrap();
                if h0_abs * T::from(1e-3).unwrap() > floor {
                    h0_abs * T::from(1e-3).unwrap()
                } else {
                    floor
                }
            } else {
                T::from(10.0).unwrap()
                    .powf(-(T::from(2.0).unwrap() + dmax.log10()) / order_t)
            };

            let h0_100 = T::from(100.0).unwrap() * h0.abs();
            let h1_abs = h1.abs();
            (if h0_100 < h1_abs { h0_100 } else { h1_abs }) * tdir
        };

        // For FSAL methods, cache the last k evaluation
        let mut k_last: Option<Vector<T, S>> = None;

        #[cfg(feature = "std")]
        let mut dense_store = if settings.dense_output {
            Some(super::DenseOutput {
                t: Vec::new(),
                h: Vec::new(),
                stages: Vec::new(),
                y: Vec::new(),
            })
        } else {
            None
        };

        loop {
            // Clamp step to not overshoot end
            if (tdir > zero && (t + h) >= tf) || (tdir < zero && (t + h) <= tf) {
                h = tf - t;
            }

            // Compute k-stages on the stack
            let mut karr = [Vector::<T, S>::zeros(); STAGES];

            if Self::FSAL && k_last.is_some() {
                karr[0] = k_last.take().unwrap();
            } else {
                karr[0] = f(t, &y);
                nevals += 1;
            }

            for k in 1..STAGES {
                let mut ysum = y;
                for j in 0..k {
                    let a_kj = T::from(Self::A[k][j]).unwrap();
                    if a_kj != zero {
                        ysum = ysum + karr[j] * a_kj * h;
                    }
                }
                karr[k] = f(t + T::from(Self::C[k]).unwrap() * h, &ysum);
                nevals += 1;
            }

            // Higher-order solution
            let mut ynp1 = y / h;
            for (idx, ki) in karr.iter().enumerate() {
                let b_idx = T::from(Self::B[idx]).unwrap();
                if b_idx != zero {
                    ynp1 = ynp1 + *ki * b_idx;
                }
            }
            ynp1 = ynp1 * h;

            // Error estimate
            let mut yerr = Vector::<T, S>::zeros();
            for (idx, ki) in karr.iter().enumerate() {
                let berr_abs = Self::BERR[idx].abs();
                if berr_abs > 1.0e-20 {
                    let berr_t = T::from(Self::BERR[idx]).unwrap();
                    yerr = yerr + *ki * berr_t;
                }
            }
            yerr = yerr * h;

            // Normalized error
            let enorm = {
                let ymax = y.abs().element_max(&ynp1.abs()) * settings.rel_tol + settings.abs_tol;
                yerr.element_div(&ymax).scaled_norm()
            };

            if !enorm.is_finite() {
                return Err(OdeError::StepNotFinite);
            }

            // PID step-size controller (Söderlind & Wang 2006, §4)
            //
            // The step-size ratio is: h_{n+1}/h_n = 1/q, where
            //   q = (e_n)^β₁ · (e_{n-1})^β₂ · (e_{n-2})^β₃ / safety
            //
            // Coefficients from Söderlind's H211b controller:
            //   β₁ = 1/p + α/p,  β₂ = -α/p,  β₃ ≈ 0  (with α = 1/4)
            // which for the full PID becomes:
            //   β₁ = 0.7/p,  β₂ = -0.4/p,  β₃ = 0
            // We extend to PID with a small β₃ for additional smoothing.
            let order_f = T::from(Self::ORDER).unwrap();
            let beta1 = T::from(0.7).unwrap() / order_f;
            let beta2 = T::from(0.4).unwrap() / order_f;
            let beta3 = T::from(0.1).unwrap() / order_f;
            let q = {
                let raw = enorm.powf(beta1)
                    / enorm_prev.powf(beta2)
                    * enorm_prev2.powf(beta3)
                    / settings.safety;
                let lo = one / settings.max_factor;
                let hi = one / settings.min_factor;
                if raw < lo { lo } else if raw > hi { hi } else { raw }
            };

            if enorm < one || h.abs() <= settings.min_step {
                // Accept step
                #[cfg(feature = "std")]
                if let Some(ref mut ds) = dense_store {
                    ds.t.push(t);
                    ds.h.push(h);
                    ds.stages.push(karr.to_vec());
                    ds.y.push(y);
                }

                if Self::FSAL {
                    k_last = Some(karr[STAGES - 1]);
                }

                let floor = T::from(1.0e-4).unwrap();
                enorm_prev2 = enorm_prev;
                enorm_prev = if enorm > floor { enorm } else { floor };
                t = t + h;
                y = ynp1;
                h = h / q;

                naccept += 1;
                if (tdir > zero && t >= tf) || (tdir < zero && t <= tf) {
                    break;
                }
            } else {
                // Reject step — use a more conservative factor
                if Self::FSAL {
                    k_last = None;
                }
                nreject += 1;
                let hi = one / settings.min_factor;
                let reject_q = enorm.powf(beta1) / settings.safety;
                let denom = if reject_q < hi { reject_q } else { hi };
                h = h / denom;
            }

            if naccept + nreject >= settings.max_steps {
                return Err(OdeError::MaxStepsExceeded);
            }
        }

        Ok(Solution {
            t,
            y,
            evals: nevals,
            accepted: naccept,
            rejected: nreject,
            #[cfg(feature = "std")]
            dense: dense_store,
        })
    }

    /// Interpolate the dense output at a given point.
    ///
    /// Requires `std` feature and `dense_output = true` in settings.
    #[cfg(feature = "std")]
    fn interpolate<T: FloatScalar, const S: usize>(
        t_interp: T,
        sol: &Solution<T, S>,
    ) -> Result<Vector<T, S>, OdeError> {
        let dense = sol.dense.as_ref().ok_or(OdeError::NoDenseOutput)?;
        if dense.t.is_empty() {
            return Err(OdeError::NoDenseOutput);
        }

        let forward = sol.t > dense.t[0];

        // Bounds check
        let (lo, hi) = if forward {
            (dense.t[0], sol.t)
        } else {
            (sol.t, dense.t[0])
        };
        if t_interp < lo || t_interp > hi {
            return Err(OdeError::InterpOutOfBounds);
        }

        // Find the step containing t_interp
        let idx = if forward {
            let mut i = dense.t.iter().position(|&x| x >= t_interp).unwrap_or(dense.t.len());
            i = i.saturating_sub(1);
            i
        } else {
            let mut i = dense.t.iter().position(|&x| x <= t_interp).unwrap_or(dense.t.len());
            i = i.saturating_sub(1);
            i
        };

        let h = dense.h[idx];
        let t_frac = (t_interp - dense.t[idx]) / h;

        // Compute interpolant coefficients bi[i] = sum_j(BI[i][j] * t^(j+1))
        // Equation (6) of Verner 2010
        let mut bi = [T::zero(); STAGES];
        for i in 0..STAGES {
            let mut tj = T::one();
            let mut sum = T::zero();
            for j in 0..NI {
                tj = tj * t_frac;
                sum = sum + T::from(Self::BI[i][j]).unwrap() * tj;
            }
            bi[i] = sum;
        }

        // Equation (5): y_interp = (y/h + sum(k[i] * bi[i])) * h
        let mut result = dense.y[idx] / h;
        for (i, ki) in dense.stages[idx].iter().enumerate() {
            if bi[i] != T::zero() {
                result = result + *ki * bi[i];
            }
        }
        result = result * h;

        Ok(result)
    }
}
