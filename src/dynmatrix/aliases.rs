//! Pre-defined type aliases for common `DynMatrix` and `DynVector` element types.

use super::{DynMatrix, DynVector};

// ── DynMatrix scalar aliases ────────────────────────────────────────

/// Dynamic matrix with `f32` elements.
pub type DynMatrixf32 = DynMatrix<f32>;
/// Dynamic matrix with `f64` elements.
pub type DynMatrixf64 = DynMatrix<f64>;
/// Dynamic matrix with `i32` elements.
pub type DynMatrixi32 = DynMatrix<i32>;
/// Dynamic matrix with `i64` elements.
pub type DynMatrixi64 = DynMatrix<i64>;
/// Dynamic matrix with `u32` elements.
pub type DynMatrixu32 = DynMatrix<u32>;
/// Dynamic matrix with `u64` elements.
pub type DynMatrixu64 = DynMatrix<u64>;

// ── DynVector scalar aliases ────────────────────────────────────────

/// Dynamic vector with `f32` elements.
pub type DynVectorf32 = DynVector<f32>;
/// Dynamic vector with `f64` elements.
pub type DynVectorf64 = DynVector<f64>;
/// Dynamic vector with `i32` elements.
pub type DynVectori32 = DynVector<i32>;
/// Dynamic vector with `i64` elements.
pub type DynVectori64 = DynVector<i64>;
/// Dynamic vector with `u32` elements.
pub type DynVectoru32 = DynVector<u32>;
/// Dynamic vector with `u64` elements.
pub type DynVectoru64 = DynVector<u64>;

// ── Complex aliases (behind `complex` feature) ──────────────────────

/// Dynamic matrix with `Complex<f32>` elements.
#[cfg(feature = "complex")]
pub type DynMatrixz32 = DynMatrix<num_complex::Complex<f32>>;
/// Dynamic matrix with `Complex<f64>` elements.
#[cfg(feature = "complex")]
pub type DynMatrixz64 = DynMatrix<num_complex::Complex<f64>>;

/// Dynamic vector with `Complex<f32>` elements.
#[cfg(feature = "complex")]
pub type DynVectorz32 = DynVector<num_complex::Complex<f32>>;
/// Dynamic vector with `Complex<f64>` elements.
#[cfg(feature = "complex")]
pub type DynVectorz64 = DynVector<num_complex::Complex<f64>>;
