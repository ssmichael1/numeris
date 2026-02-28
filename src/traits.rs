use core::fmt::Debug;
use num_traits::{Float, Num, One, Zero};

#[cfg(feature = "complex")]
use num_complex::Complex;

/// Trait for types that can be used as matrix elements.
///
/// Blanket-implemented for all types satisfying the bounds.
/// Covers `f32`, `f64`, and all integer types.
pub trait Scalar: Copy + PartialEq + Debug + Zero + One + Num {}

impl<T: Copy + PartialEq + Debug + Zero + One + Num> Scalar for T {}

/// Trait for floating-point matrix elements.
///
/// Required by operations that need `sqrt`, `sin`, `abs`, etc.
/// (decompositions, norms, trigonometric functions).
/// Implies `LinalgScalar<Real = Self>` since real floats are their own real type.
pub trait FloatScalar: Scalar + Float + LinalgScalar<Real = Self> {}

impl<T: Scalar + Float + LinalgScalar<Real = T>> FloatScalar for T {}

/// Trait for matrix elements that support linear algebra operations.
///
/// Covers both real floats (`f32`, `f64`) and complex numbers (`Complex<f32>`,
/// `Complex<f64>`). Use this instead of `FloatScalar` in decompositions and norms.
///
/// `FloatScalar` remains for inherently-real operations (quaternions, ordered comparisons).
pub trait LinalgScalar: Scalar {
    /// The real component type (`Self` for reals, `T` for `Complex<T>`).
    type Real: FloatScalar;

    /// Absolute value / modulus: `|z|` for complex, `.abs()` for real.
    fn modulus(self) -> Self::Real;

    /// Complex conjugate (identity for reals).
    fn conj(self) -> Self;

    /// Real part.
    fn re(self) -> Self::Real;

    /// Square root.
    fn lsqrt(self) -> Self;

    /// Natural logarithm.
    fn lln(self) -> Self;

    /// Machine epsilon of the underlying real type.
    fn lepsilon() -> Self::Real;

    /// Promote a real value into `Self`.
    fn from_real(r: Self::Real) -> Self;
}

/// Concrete impls for real floats — trivial delegation.
macro_rules! impl_linalg_scalar_real {
    ($($t:ty),*) => {
        $(
            impl LinalgScalar for $t {
                type Real = $t;

                #[inline] fn modulus(self) -> $t { Float::abs(self) }
                #[inline] fn conj(self) -> $t { self }
                #[inline] fn re(self) -> $t { self }
                #[inline] fn lsqrt(self) -> $t { Float::sqrt(self) }
                #[inline] fn lln(self) -> $t { Float::ln(self) }
                #[inline] fn lepsilon() -> $t { <$t as Float>::epsilon() }
                #[inline] fn from_real(r: $t) -> $t { r }
            }
        )*
    };
}

impl_linalg_scalar_real!(f32, f64);

#[cfg(feature = "complex")]
impl<T: FloatScalar> LinalgScalar for Complex<T> {
    type Real = T;

    #[inline]
    fn modulus(self) -> T {
        self.norm()
    }

    #[inline]
    fn conj(self) -> Self {
        Complex::conj(&self)
    }

    #[inline]
    fn re(self) -> T {
        self.re
    }

    #[inline]
    fn lsqrt(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn lln(self) -> Self {
        self.ln()
    }

    #[inline]
    fn lepsilon() -> T {
        T::epsilon()
    }

    #[inline]
    fn from_real(r: T) -> Self {
        Complex::new(r, T::zero())
    }
}

/// Read-only access to a matrix-like type.
///
/// This trait allows algorithms to operate generically over
/// both fixed-size `Matrix` and future `DynMatrix` types.
pub trait MatrixRef<T> {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn get(&self, row: usize, col: usize) -> &T;
}

/// Mutable access to a matrix-like type.
///
/// Extends `MatrixRef` with mutable element access, enabling
/// in-place algorithms (Cholesky, LU, etc.) to work generically.
pub trait MatrixMut<T>: MatrixRef<T> {
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T;
}
