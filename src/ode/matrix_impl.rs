use super::types::ODEState;
use crate::prelude::Matrix;

impl<const ROWS: usize, const COLS: usize> ODEState for Matrix<ROWS, COLS, f64> {
    fn ode_abs(&self) -> Self {
        self.map(|x| x.abs())
    }

    fn ode_elem_div(&self, other: &Self) -> Self {
        self.zip_map(other, |a, b| a / b)
    }

    fn ode_scalar_add(&self, s: f64) -> Self {
        self.map(|x| x + s)
    }

    fn ode_scaled_norm(&self) -> f64 {
        (self
            .data
            .iter()
            .flat_map(|row| row.iter())
            .map(|x| x * x)
            .sum::<f64>()
            / (ROWS * COLS) as f64)
            .sqrt()
    }

    fn ode_nelem(&self) -> usize {
        ROWS * COLS
    }

    fn ode_zero() -> Self {
        Self::zeros()
    }

    fn ode_elem_max(&self, other: &Self) -> Self {
        self.zip_map(other, |a, b| a.max(b))
    }
}
